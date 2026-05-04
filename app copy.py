"""
Plateforme de simulation pharmacocinétique (PK/PD)
==================================================

Modèle 1 ou 2 compartiments — Voies orale / Bolus IV / Perfusion IV.
Profil patient dynamique (Cockcroft-Gault) + confrontation à des données réelles.

Architecture
------------
1. Domaine PK (calculs purs, type-hinted, sans dépendance Streamlit)
   - cockcroft_gault()      : clairance rénale (mL/min)
   - adjust_ke_renal()      : Ke ajusté selon ClCr
   - simulate_pk()          : simulateur ODE générique
   - compute_metrics()      : Cmax, Tmax, AUC, Cmin/Cmax/Css à l'état d'équilibre
   - fit_quality()          : RMSE et R² (vs données cliniques)

2. Interface Streamlit (sidebar + 4 onglets)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.integrate import odeint, trapezoid


# =============================================================================
# 1. DOMAINE PK — fonctions et structures pures
# =============================================================================

@dataclass(frozen=True)
class PatientProfile:
    sex: str          # "Homme" | "Femme"
    age: int          # années
    weight: float     # kg
    scr_umol_l: float # créatininémie en µmol/L


@dataclass(frozen=True)
class PKParams:
    dose: float       # mg
    F: float          # biodisponibilité (oral) ; ignoré pour IV
    vc: float         # volume central / Vd (L)
    ka: float         # constante d'absorption (h⁻¹)
    ke: float         # constante d'élimination (h⁻¹)
    k12: float = 0.0  # central → périphérique (h⁻¹) — 2-cmt seulement
    k21: float = 0.0  # périphérique → central (h⁻¹) — 2-cmt seulement


@dataclass(frozen=True)
class AdminConfig:
    route: str                    # "Orale" | "Bolus IV" | "Perfusion IV"
    n_compartments: int           # 1 | 2
    infusion_duration: float = 0  # h, uniquement si route == "Perfusion IV"


@dataclass(frozen=True)
class Regimen:
    mode: str                    # "Dose Unique" | "Doses Répétées"
    tau: Optional[float] = None  # h
    n_doses: int = 1

    @property
    def dose_times(self) -> list[float]:
        if self.mode == "Dose Unique":
            return [0.0]
        return [i * float(self.tau) for i in range(self.n_doses)]

    @property
    def total_treatment_time(self) -> float:
        if self.mode == "Dose Unique":
            return 0.0
        return (self.n_doses - 1) * float(self.tau)


@dataclass(frozen=True)
class TherapeuticWindow:
    cme: float
    toxicity: float


# -----------------------------------------------------------------------------
# Pharmacologie clinique
# -----------------------------------------------------------------------------
def cockcroft_gault(patient: PatientProfile) -> float:
    """Clairance de la créatinine (mL/min) selon la formule de Cockcroft-Gault.

    Convertit la créatininémie µmol/L → mg/dL (÷ 88.4) puis applique :
        ClCr = ((140 - âge) × poids) / (72 × Scr_mg/dL)
    Coefficient × 0.85 pour les femmes.
    """
    if patient.scr_umol_l <= 0 or patient.weight <= 0 or patient.age <= 0:
        raise ValueError("Données patient invalides (âge, poids et créatininémie doivent être > 0).")

    scr_mg_dl = patient.scr_umol_l / 88.4
    clcr = ((140 - patient.age) * patient.weight) / (72.0 * scr_mg_dl)
    if patient.sex == "Femme":
        clcr *= 0.85
    return max(0.0, clcr)


def adjust_ke_renal(
    ke_base: float,
    clcr: float,
    fraction_renal: float,
    clcr_normal: float = 120.0,
) -> float:
    """Ajuste la constante d'élimination en fonction de la clairance rénale.

        Ke = Ke_non_renal + α × ClCr
    où :
        Ke_non_renal   = Ke_base × (1 - fraction_renal)
        α              = (Ke_base × fraction_renal) / ClCr_normal
    """
    ke_non_renal = ke_base * (1.0 - fraction_renal)
    alpha = (ke_base * fraction_renal) / clcr_normal
    return max(1e-6, ke_non_renal + alpha * clcr)


# -----------------------------------------------------------------------------
# Simulateur PK générique (ODE)
# -----------------------------------------------------------------------------
def simulate_pk(
    t_grid: np.ndarray,
    params: PKParams,
    admin: AdminConfig,
    regimen: Regimen,
) -> np.ndarray:
    """Simulateur PK universel.

    Résout les EDO suivantes selon la configuration :

        Compartiment d'absorption (oral seulement) :
            dA_a/dt = -ka · A_a

        Compartiment central :
            dA_c/dt = entrée(t) - (ke [+ k12]) · A_c  [+ k21 · A_p]

        Compartiment périphérique (2-cmt) :
            dA_p/dt = k12 · A_c - k21 · A_p

    Les doses successives sont gérées par superposition exacte : on intègre
    par segments entre événements (administration / changement de débit) et on
    applique l'événement (impulsion ou changement de débit) au breakpoint.

    Renvoie la concentration plasmatique centrale (mg/L) sur t_grid.
    """
    if params.vc <= 0:
        raise ValueError("Le volume central Vc doit être strictement positif.")
    if admin.n_compartments == 2 and (params.k12 <= 0 or params.k21 <= 0):
        raise ValueError("k12 et k21 doivent être > 0 pour un modèle 2 compartiments.")
    if admin.route == "Perfusion IV" and admin.infusion_duration <= 0:
        raise ValueError("La durée de perfusion doit être strictement positive.")

    has_depot = admin.route == "Orale"
    n_states = (1 if has_depot else 0) + admin.n_compartments

    if has_depot:
        idx_depot, idx_c = 0, 1
        idx_p = 2 if admin.n_compartments == 2 else None
    else:
        idx_depot = None
        idx_c = 0
        idx_p = 1 if admin.n_compartments == 2 else None

    ka, ke, k12, k21 = params.ka, params.ke, params.k12, params.k21

    def rhs(y: np.ndarray, _t: float, R: float) -> list[float]:
        dy = [0.0] * n_states
        if has_depot:
            dy[idx_depot] = -ka * y[idx_depot]
            input_to_central = ka * y[idx_depot]
        else:
            input_to_central = R

        Ac = y[idx_c]
        if admin.n_compartments == 2:
            Ap = y[idx_p]
            dy[idx_c] = input_to_central - (ke + k12) * Ac + k21 * Ap
            dy[idx_p] = k12 * Ac - k21 * Ap
        else:
            dy[idx_c] = input_to_central - ke * Ac
        return dy

    effective_dose = params.F * params.dose if has_depot else params.dose

    events: list[tuple[float, str, float]] = []
    for t_d in regimen.dose_times:
        if admin.route == "Orale":
            events.append((t_d, "depot", effective_dose))
        elif admin.route == "Bolus IV":
            events.append((t_d, "central", effective_dose))
        elif admin.route == "Perfusion IV":
            R_inf = effective_dose / admin.infusion_duration
            events.append((t_d, "rate", R_inf))
            events.append((t_d + admin.infusion_duration, "rate", 0.0))
    events.sort(key=lambda e: e[0])

    t_grid_list = [float(x) for x in t_grid.tolist()]
    breakpoints = sorted(set([0.0] + [e[0] for e in events] + t_grid_list))

    state = np.zeros(n_states)
    R_current = 0.0
    snapshots: dict[float, np.ndarray] = {}
    ev_idx = 0

    while ev_idx < len(events) and events[ev_idx][0] <= 1e-12:
        _, ev_type, ev_val = events[ev_idx]
        if ev_type == "depot":
            state[idx_depot] += ev_val
        elif ev_type == "central":
            state[idx_c] += ev_val
        elif ev_type == "rate":
            R_current = ev_val
        ev_idx += 1
    snapshots[0.0] = state.copy()

    for i in range(len(breakpoints) - 1):
        t0, t1 = breakpoints[i], breakpoints[i + 1]
        if t1 - t0 > 1e-12:
            sol = odeint(rhs, state, [t0, t1], args=(R_current,), rtol=1e-8, atol=1e-10, mxstep=5000)
            state = sol[-1].copy()

        while ev_idx < len(events) and abs(events[ev_idx][0] - t1) < 1e-9:
            _, ev_type, ev_val = events[ev_idx]
            if ev_type == "depot":
                state[idx_depot] += ev_val
            elif ev_type == "central":
                state[idx_c] += ev_val
            elif ev_type == "rate":
                R_current = ev_val
            ev_idx += 1

        snapshots[t1] = state.copy()

    concentrations = np.array(
        [snapshots[tg][idx_c] / params.vc for tg in t_grid_list]
    )
    return np.maximum(concentrations, 0.0)


# -----------------------------------------------------------------------------
# Métriques PK
# -----------------------------------------------------------------------------
@dataclass
class PKMetrics:
    cmax: float
    tmax: float
    auc: float
    t_half_a: float
    t_half_e: float
    cmax_ss: Optional[float] = None
    cmin_ss: Optional[float] = None
    css: Optional[float] = None


def compute_metrics(
    t: np.ndarray,
    C: np.ndarray,
    params: PKParams,
    admin: AdminConfig,
    regimen: Regimen,
) -> PKMetrics:
    """Calcule l'ensemble des métriques pharmacocinétiques."""
    cmax = float(np.max(C))
    tmax = float(t[int(np.argmax(C))])
    auc = float(trapezoid(C, t))
    t_half_e = float(np.log(2) / params.ke) if params.ke > 0 else float("inf")
    t_half_a = float(np.log(2) / params.ka) if params.ka > 0 else float("inf")

    cmax_ss = cmin_ss = css = None
    if regimen.mode == "Doses Répétées" and regimen.tau is not None:
        t_last = (regimen.n_doses - 1) * regimen.tau
        mask = (t >= t_last) & (t <= t_last + regimen.tau)
        if mask.any():
            cmax_ss = float(np.max(C[mask]))
            cmin_ss = float(C[mask][-1])

        F = params.F if admin.route == "Orale" else 1.0
        css = (F * params.dose) / (params.vc * params.ke * regimen.tau)

    return PKMetrics(
        cmax=cmax,
        tmax=tmax,
        auc=auc,
        t_half_a=t_half_a,
        t_half_e=t_half_e,
        cmax_ss=cmax_ss,
        cmin_ss=cmin_ss,
        css=css,
    )


# -----------------------------------------------------------------------------
# Confrontation aux données réelles
# -----------------------------------------------------------------------------
@dataclass
class FitQuality:
    rmse: float
    r2: float
    n_points: int
    mae: float


def fit_quality(t_real: np.ndarray, c_real: np.ndarray, t_sim: np.ndarray, c_sim: np.ndarray) -> FitQuality:
    """Erreur entre concentrations simulées et observées (interpolation linéaire)."""
    if len(t_real) == 0:
        raise ValueError("Aucun point de données réel fourni.")
    c_pred = np.interp(t_real, t_sim, c_sim)
    residuals = c_real - c_pred
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((c_real - np.mean(c_real)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return FitQuality(rmse=rmse, r2=r2, n_points=int(len(t_real)), mae=mae)


def parse_real_data(uploaded_file) -> pd.DataFrame:
    """Lit un CSV utilisateur — colonnes 'Temps' et 'Concentration_Reelle'."""
    df = pd.read_csv(uploaded_file)
    df.columns = [c.strip() for c in df.columns]
    aliases = {
        "Temps": ["Temps", "temps", "Time", "time", "t", "T"],
        "Concentration_Reelle": [
            "Concentration_Reelle", "Concentration_Réelle",
            "Concentration", "concentration", "C", "Cobs", "Concentration (mg/L)",
        ],
    }
    rename = {}
    for canonical, candidates in aliases.items():
        match = next((c for c in candidates if c in df.columns), None)
        if match is None:
            raise KeyError(
                f"Colonne '{canonical}' introuvable. "
                f"Colonnes attendues : Temps, Concentration_Reelle. "
                f"Trouvées : {list(df.columns)}"
            )
        rename[match] = canonical
    df = df.rename(columns=rename)[["Temps", "Concentration_Reelle"]]
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    if df.empty:
        raise ValueError("Le fichier ne contient aucune ligne numérique exploitable.")
    return df.sort_values("Temps").reset_index(drop=True)


# =============================================================================
# 2. INTERFACE STREAMLIT
# =============================================================================

st.set_page_config(page_title="Plateforme PK/PD", layout="wide", page_icon="💊")
st.title("Plateforme de simulation pharmacocinétique (PK/PD)")
st.caption("Modèles 1 et 2 compartiments · Profil patient dynamique · Confrontation aux données cliniques")

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("👤 Profil patient")
    sex = st.radio("Sexe", ["Homme", "Femme"], horizontal=True)
    age = st.number_input("Âge (années)", min_value=1, max_value=110, value=50, step=1)
    weight = st.number_input("Poids (kg)", min_value=20.0, max_value=200.0, value=70.0, step=0.5)
    scr = st.number_input("Créatininémie (µmol/L)", min_value=10.0, max_value=600.0, value=88.0, step=1.0)

    st.markdown("---")
    st.header("💊 Paramètres PK")
    n_compartments = st.selectbox("Type de modèle", [1, 2], format_func=lambda n: f"{n} compartiment{'s' if n == 2 else ''}")
    route = st.selectbox("Voie d'administration", ["Orale", "Bolus IV", "Perfusion IV"])

    dose = st.slider("Dose (mg)", 10, 1000, 100, step=10)
    vc = st.slider("Volume central Vc (L)", 1.0, 100.0, 20.0, step=0.5)

    if route == "Orale":
        ka = st.slider("Ka — absorption (h⁻¹)", 0.1, 5.0, 1.0, step=0.05)
        F = st.slider("Biodisponibilité F", 0.05, 1.0, 0.7, step=0.05)
    else:
        ka = 0.0
        F = 1.0

    ke_base = st.slider("Ke base (h⁻¹) — pour ClCr = 120 mL/min", 0.01, 2.0, 0.2, step=0.01)
    fraction_renal = st.slider(
        "Fraction d'élimination rénale",
        0.0, 1.0, 0.7, step=0.05,
        help="Part de l'élimination qui dépend de la fonction rénale (le reste est non-rénal)",
    )

    if n_compartments == 2:
        k12 = st.slider("k12 — central → tissus (h⁻¹)", 0.01, 5.0, 0.5, step=0.05)
        k21 = st.slider("k21 — tissus → central (h⁻¹)", 0.01, 5.0, 0.3, step=0.05)
    else:
        k12 = k21 = 0.0

    if route == "Perfusion IV":
        infusion_duration = st.slider("Durée de perfusion (h)", 0.1, 24.0, 1.0, step=0.1)
    else:
        infusion_duration = 0.0

    st.markdown("---")
    st.header("📅 Régime")
    mode = st.selectbox("Régime de doses", ["Dose Unique", "Doses Répétées"])
    if mode == "Doses Répétées":
        tau = st.slider("Intervalle τ (h)", 1, 48, 8, step=1)
        n_doses = st.slider("Nombre total de doses", 2, 30, 5, step=1)
    else:
        tau = None
        n_doses = 1

    st.markdown("---")
    st.header("🎯 Fenêtre thérapeutique")
    cme = st.slider("CME (mg/L)", 0.0, 50.0, 2.0, step=0.1)
    toxicity = st.slider("Seuil de Toxicité (mg/L)", 0.0, 100.0, 10.0, step=0.1)

# -----------------------------------------------------------------------------
# Validation utilisateur (try/except + st.error)
# -----------------------------------------------------------------------------
try:
    patient = PatientProfile(sex=sex, age=int(age), weight=float(weight), scr_umol_l=float(scr))
    clcr = cockcroft_gault(patient)
    ke_eff = adjust_ke_renal(ke_base, clcr, fraction_renal)

    if route == "Perfusion IV" and mode == "Doses Répétées" and tau is not None and infusion_duration >= tau:
        raise ValueError(
            f"La durée de perfusion ({infusion_duration} h) doit être inférieure à l'intervalle τ ({tau} h)."
        )
    if cme >= toxicity:
        raise ValueError("La CME doit être strictement inférieure au Seuil de Toxicité.")
    if route == "Orale" and abs(ka - ke_eff) < 1e-6:
        raise ValueError("Ka et Ke effectif sont trop proches — modifiez Ka ou la fonction rénale.")

    params = PKParams(dose=dose, F=F, vc=vc, ka=ka, ke=ke_eff, k12=k12, k21=k21)
    admin = AdminConfig(route=route, n_compartments=n_compartments, infusion_duration=infusion_duration)
    regimen = Regimen(mode=mode, tau=tau, n_doses=n_doses)
    window = TherapeuticWindow(cme=cme, toxicity=toxicity)

    if mode == "Dose Unique":
        t_end = 48.0
    else:
        t_end = float((n_doses - 1) * tau + 48)
    t = np.linspace(0.0, t_end, max(1200, int(t_end * 25)))

    C = simulate_pk(t, params, admin, regimen)
    metrics = compute_metrics(t, C, params, admin, regimen)

except (ValueError, KeyError) as exc:
    st.error(f"⛔ {exc}")
    st.stop()
except Exception as exc:  # noqa: BLE001
    st.error(f"⛔ Erreur de simulation : {exc}")
    st.stop()

# -----------------------------------------------------------------------------
# Données réelles importées (session_state pour partager entre onglets)
# -----------------------------------------------------------------------------
if "real_data" not in st.session_state:
    st.session_state["real_data"] = None


def build_pk_figure(
    t: np.ndarray,
    C: np.ndarray,
    metrics: PKMetrics,
    window: TherapeuticWindow,
    regimen: Regimen,
    real_data: Optional[pd.DataFrame] = None,
) -> go.Figure:
    """Graphique Plotly interactif — fenêtre thérapeutique + courbe + données réelles."""
    fig = go.Figure()

    fig.add_hrect(
        y0=window.cme, y1=window.toxicity,
        fillcolor="rgba(34, 197, 94, 0.15)", line_width=0,
        annotation_text="Fenêtre thérapeutique",
        annotation_position="top left",
    )
    fig.add_hline(y=window.cme, line=dict(color="#16A34A", dash="dot", width=1.4),
                  annotation_text=f"CME = {window.cme:.1f}", annotation_position="bottom right")
    fig.add_hline(y=window.toxicity, line=dict(color="#DC2626", dash="dot", width=1.4),
                  annotation_text=f"Toxicité = {window.toxicity:.1f}", annotation_position="top right")

    fig.add_trace(go.Scatter(
        x=t, y=C, mode="lines", name="C(t) théorique",
        line=dict(color="#2563EB", width=2.6),
        fill="tozeroy", fillcolor="rgba(37, 99, 235, 0.08)",
        hovertemplate="t = %{x:.2f} h<br>C = %{y:.3f} mg/L<extra></extra>",
    ))

    if metrics.cmax_ss is not None:
        fig.add_hline(y=metrics.cmax_ss, line=dict(color="#7C3AED", dash="dash", width=1.2),
                      annotation_text=f"Cmax,ss = {metrics.cmax_ss:.2f}", annotation_position="top left")
        fig.add_hline(y=metrics.cmin_ss, line=dict(color="#F97316", dash="dashdot", width=1.2),
                      annotation_text=f"Cmin,ss = {metrics.cmin_ss:.2f}", annotation_position="bottom left")
    else:
        fig.add_vline(x=metrics.tmax, line=dict(color="#F59E0B", dash="dash", width=1.2),
                      annotation_text=f"Tmax = {metrics.tmax:.2f} h")
        fig.add_hline(y=metrics.cmax, line=dict(color="#7C3AED", dash="dash", width=1.2),
                      annotation_text=f"Cmax = {metrics.cmax:.2f}")

    if regimen.mode == "Doses Répétées":
        for t_d in regimen.dose_times:
            fig.add_vline(x=t_d, line=dict(color="rgba(148, 163, 184, 0.6)", dash="dot", width=1))

    if real_data is not None and not real_data.empty:
        fig.add_trace(go.Scatter(
            x=real_data["Temps"], y=real_data["Concentration_Reelle"],
            mode="markers", name="Données cliniques",
            marker=dict(color="#111827", size=9, symbol="circle", line=dict(color="white", width=1)),
            hovertemplate="t = %{x:.2f} h<br>C_obs = %{y:.3f} mg/L<extra></extra>",
        ))

    fig.update_layout(
        xaxis_title="Temps (h)",
        yaxis_title="Concentration plasmatique (mg/L)",
        hovermode="x unified",
        template="plotly_white",
        height=560,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=40, t=60, b=50),
    )
    return fig


# -----------------------------------------------------------------------------
# Bandeau d'en-tête (paramètres physiologiques effectifs)
# -----------------------------------------------------------------------------
with st.container(border=True):
    info_cols = st.columns(4)
    info_cols[0].metric("ClCr (Cockcroft-Gault)", f"{clcr:.1f} mL/min")
    info_cols[1].metric("Ke effectif", f"{ke_eff:.3f} h⁻¹",
                        delta=f"{(ke_eff - ke_base):+.3f} vs base", delta_color="off")
    info_cols[2].metric("Voie · Modèle", f"{route} · {n_compartments}-cmt")
    info_cols[3].metric(
        "Régime",
        f"{n_doses}× {dose} mg" if mode == "Doses Répétées" else f"1× {dose} mg",
        delta=f"τ = {tau} h" if mode == "Doses Répétées" else "Dose unique",
        delta_color="off",
    )

# -----------------------------------------------------------------------------
# Onglets
# -----------------------------------------------------------------------------
tab_sim, tab_clin, tab_fit, tab_export = st.tabs(
    ["📈 Simulation", "🩺 Analyse Clinique", "📊 Fit Data", "📥 Export"]
)

# === Onglet 1 : Simulation =================================================
with tab_sim:
    if mode == "Dose Unique":
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Cmax", f"{metrics.cmax:.2f} mg/L")
        c2.metric("Tmax", f"{metrics.tmax:.2f} h")
        c3.metric("AUC totale", f"{metrics.auc:.1f} mg·h/L")
        c4.metric("t½ élimination", f"{metrics.t_half_e:.2f} h")
        c5.metric("t½ absorption", f"{metrics.t_half_a:.2f} h" if route == "Orale" else "—")
    else:
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Cmax,ss", f"{metrics.cmax_ss:.2f} mg/L")
        c2.metric("Cmin,ss", f"{metrics.cmin_ss:.2f} mg/L")
        c3.metric("Css moyen", f"{metrics.css:.2f} mg/L")
        c4.metric("AUC totale", f"{metrics.auc:.1f} mg·h/L")
        c5.metric("t½ élimination", f"{metrics.t_half_e:.2f} h")
        c6.metric("t½ absorption", f"{metrics.t_half_a:.2f} h" if route == "Orale" else "—")

    fig = build_pk_figure(t, C, metrics, window, regimen, st.session_state.get("real_data"))
    st.plotly_chart(fig, use_container_width=True)

# === Onglet 2 : Analyse Clinique ===========================================
with tab_clin:
    st.subheader("Résumé du modèle")

    if admin.n_compartments == 1 and route == "Orale":
        st.latex(
            r"C(t) = \sum_{i=0}^{N-1} \frac{F \cdot D \cdot K_a}{V_d (K_a - K_e)}"
            r"\left(e^{-K_e(t-i\tau)} - e^{-K_a(t-i\tau)}\right) \mathbf{1}_{t \geq i\tau}"
        )
    elif admin.n_compartments == 1 and route == "Bolus IV":
        st.latex(r"C(t) = \sum_{i} \frac{D}{V_d} \, e^{-K_e (t - i\tau)} \mathbf{1}_{t \geq i\tau}")
    elif admin.n_compartments == 1 and route == "Perfusion IV":
        st.latex(
            r"C(t) = \frac{R_0}{V_d K_e}\left(1 - e^{-K_e t}\right) \quad (0 \leq t \leq T_{inf})"
        )
        st.latex(
            r"C(t) = \frac{R_0}{V_d K_e}\left(1 - e^{-K_e T_{inf}}\right) e^{-K_e (t - T_{inf})} \quad (t > T_{inf})"
        )
    else:
        st.latex(
            r"\begin{cases}"
            r"\dot A_c = \text{entrée}(t) - (K_e + k_{12}) A_c + k_{21} A_p \\"
            r"\dot A_p = k_{12} A_c - k_{21} A_p"
            r"\end{cases}"
        )
        st.caption("Modèle 2 compartiments — résolu numériquement (LSODA / odeint).")

    st.markdown("---")
    st.subheader("Profil patient")
    pcols = st.columns(4)
    pcols[0].markdown(f"**Sexe :** {sex}")
    pcols[1].markdown(f"**Âge :** {age} ans")
    pcols[2].markdown(f"**Poids :** {weight} kg")
    pcols[3].markdown(f"**Scr :** {scr} µmol/L")
    st.info(
        f"**Cockcroft-Gault** : ClCr = **{clcr:.1f} mL/min**  ·  "
        f"Ke base = {ke_base:.3f} h⁻¹ → **Ke effectif = {ke_eff:.3f} h⁻¹** "
        f"(fraction rénale : {fraction_renal:.0%})"
    )

    st.markdown("---")
    if mode == "Doses Répétées":
        st.subheader("Concentrations à l'état d'équilibre")
        st.latex(r"C_{ss} = \frac{F \cdot D}{V_d \cdot K_e \cdot \tau}")
        m1, m2, m3 = st.columns(3)
        m1.metric("C_max,ss", f"{metrics.cmax_ss:.3f} mg/L",
                  delta="↑ Toxique" if metrics.cmax_ss > toxicity else "✓ OK",
                  delta_color="inverse" if metrics.cmax_ss > toxicity else "normal")
        m2.metric("C_min,ss", f"{metrics.cmin_ss:.3f} mg/L",
                  delta="↓ Sous CME" if metrics.cmin_ss < cme else "✓ OK",
                  delta_color="inverse" if metrics.cmin_ss < cme else "normal")
        m3.metric("C_ss (moyen)", f"{metrics.css:.3f} mg/L")

        st.markdown("---")
        st.subheader("Bilan clinique")
        problems: list[str] = []
        if metrics.cmin_ss < cme:
            problems.append(
                f"⚠️ **Sous-exposition** : Cmin,ss = **{metrics.cmin_ss:.2f} mg/L** "
                f"< CME ({cme:.1f} mg/L). Risque de perte d'efficacité."
            )
        if metrics.cmax_ss > toxicity:
            problems.append(
                f"⚠️ **Risque toxique** : Cmax,ss = **{metrics.cmax_ss:.2f} mg/L** "
                f"> Toxicité ({toxicity:.1f} mg/L)."
            )
        if not problems:
            st.success(
                f"✅ **Traitement équilibré** : la concentration reste "
                f"dans la fenêtre thérapeutique [{cme:.1f} – {toxicity:.1f}] mg/L "
                f"à l'état d'équilibre."
            )
        else:
            for p in problems:
                st.error(p)
    else:
        st.info("Activez **Doses Répétées** pour analyser l'état d'équilibre.")

# === Onglet 3 : Fit Data ===================================================
with tab_fit:
    st.subheader("Confrontation au modèle théorique")
    st.markdown(
        "Importez un fichier CSV avec les colonnes **`Temps`** (h) et "
        "**`Concentration_Reelle`** (mg/L). Les points seront superposés à la "
        "courbe simulée et la qualité d'ajustement sera évaluée (RMSE & R²)."
    )

    uploaded = st.file_uploader("Charger un fichier CSV de mesures cliniques", type=["csv"])

    if uploaded is not None:
        try:
            real_df = parse_real_data(uploaded)
            st.session_state["real_data"] = real_df
        except (ValueError, KeyError, pd.errors.ParserError) as exc:
            st.error(f"⛔ Lecture du CSV impossible : {exc}")
            st.session_state["real_data"] = None
        except Exception as exc:  # noqa: BLE001
            st.error(f"⛔ Erreur inattendue : {exc}")
            st.session_state["real_data"] = None

    if st.button("🗑️ Réinitialiser les données réelles"):
        st.session_state["real_data"] = None
        st.rerun()

    real_df = st.session_state.get("real_data")
    if real_df is not None and not real_df.empty:
        try:
            quality = fit_quality(
                real_df["Temps"].to_numpy(),
                real_df["Concentration_Reelle"].to_numpy(),
                t, C,
            )
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Points observés", quality.n_points)
            mc2.metric("RMSE", f"{quality.rmse:.3f} mg/L")
            mc3.metric("MAE", f"{quality.mae:.3f} mg/L")
            mc4.metric("R²", f"{quality.r2:.3f}")

            if quality.r2 >= 0.90:
                st.success("✅ Excellent ajustement (R² ≥ 0.90).")
            elif quality.r2 >= 0.70:
                st.info("ℹ️ Ajustement correct (0.70 ≤ R² < 0.90).")
            else:
                st.warning("⚠️ Ajustement médiocre (R² < 0.70) — revoir les paramètres PK.")

            st.markdown("**Données importées**")
            st.dataframe(real_df, use_container_width=True, height=240)

            st.markdown("**Graphique de confrontation**")
            st.plotly_chart(
                build_pk_figure(t, C, metrics, window, regimen, real_df),
                use_container_width=True,
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"⛔ Calcul de la qualité d'ajustement impossible : {exc}")
    else:
        st.info("Aucune donnée réelle chargée pour l'instant.")
        with st.expander("📄 Exemple de format CSV attendu"):
            st.code(
                "Temps,Concentration_Reelle\n"
                "0.5,0.85\n"
                "1.0,1.92\n"
                "2.0,3.10\n"
                "4.0,3.85\n"
                "8.0,2.74\n"
                "12.0,1.65\n"
                "24.0,0.42\n",
                language="csv",
            )

# === Onglet 4 : Export =====================================================
with tab_export:
    st.subheader("Téléchargement des données simulées")
    df_export = pd.DataFrame({
        "Temps_h": np.round(t, 4),
        "Concentration_mgL": np.round(C, 6),
    })
    st.dataframe(df_export, use_container_width=True, height=320)

    csv_bytes = df_export.to_csv(index=False).encode("utf-8")
    fname = (
        f"pk_{route.replace(' ', '_')}_{n_compartments}cmt_"
        f"{'multi' if mode == 'Doses Répétées' else 'single'}.csv"
    )
    st.download_button(
        label="⬇️ Télécharger la simulation (CSV)",
        data=csv_bytes,
        file_name=fname,
        mime="text/csv",
    )

    st.markdown("---")
    st.caption(
        f"Simulation — {route} · {n_compartments} cmt · "
        f"Dose {dose} mg · Vc {vc} L · Ka {ka:.2f} h⁻¹ · Ke_eff {ke_eff:.3f} h⁻¹ · "
        f"F {F:.2f} · ClCr {clcr:.1f} mL/min"
    )

st.markdown("---")
st.caption("Plateforme PK/PD · Streamlit + Plotly + scipy.integrate.odeint")
