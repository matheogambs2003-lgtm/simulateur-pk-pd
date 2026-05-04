"""
Plateforme de simulation pharmacocinétique (PK/PD) — édition industrielle
==========================================================================

Modèles 1 ou 2 compartiments · Voies orale / Bolus IV / Perfusion IV.
Profil patient dynamique (Cockcroft-Gault + CKD-EPI · IBW Lorentz).
Simulation de population Monte Carlo · Confrontation aux données cliniques.

Architecture
------------
1. Domaine PK — fonctions pures, type-hintées, sans import Streamlit
   - ideal_body_weight()      : Poids Idéal Corporel (Lorentz)
   - cockcroft_gault()        : ClCr mL/min (IBW automatique si IMC > 25)
   - ckd_epi()                : eGFR mL/min/1.73 m² (KDIGO 2009)
   - adjust_ke_renal()        : Ke ajusté à la fonction rénale
   - simulate_pk()            : simulateur ODE générique (odeint/LSODA)
   - compute_metrics()        : PKMetrics complet
   - monte_carlo_simulate()   : 100 courbes, retourne (p5, p50, p95)
   - fit_quality()            : RMSE, MAE, R²
   - parse_real_data()        : lecture CSV robuste

2. Interface Streamlit — sidebar (4 expanders) + 4 onglets
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.integrate import odeint, trapezoid


# =============================================================================
# 1. DOMAINE PK — structures
# =============================================================================

@dataclass(frozen=True)
class PatientProfile:
    sex: str           # "Homme" | "Femme"
    age: int           # années
    weight: float      # kg (poids réel)
    height_cm: float   # cm
    scr_umol_l: float  # créatininémie µmol/L


@dataclass(frozen=True)
class PKParams:
    dose: float        # mg
    F: float           # biodisponibilité
    vc: float          # volume central (L)
    ka: float          # absorption (h⁻¹)
    ke: float          # élimination (h⁻¹)
    k12: float = 0.0
    k21: float = 0.0


@dataclass(frozen=True)
class AdminConfig:
    route: str              # "Orale" | "Bolus IV" | "Perfusion IV"
    n_compartments: int     # 1 | 2
    infusion_duration: float = 0.0


@dataclass(frozen=True)
class Regimen:
    mode: str                   # "Dose Unique" | "Doses Répétées"
    tau: Optional[float] = None
    n_doses: int = 1

    @property
    def dose_times(self) -> list[float]:
        if self.mode == "Dose Unique":
            return [0.0]
        return [i * float(self.tau) for i in range(self.n_doses)]


@dataclass(frozen=True)
class TherapeuticWindow:
    cme: float
    toxicity: float


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


@dataclass
class FitQuality:
    rmse: float
    r2: float
    n_points: int
    mae: float


# =============================================================================
# 2. DOMAINE PK — fonctions pures
# =============================================================================

def ideal_body_weight(sex: str, height_cm: float) -> float:
    """Poids Idéal Corporel selon la formule de Lorentz (kg).

    Hommes : IBW = height_cm - 100 - (height_cm - 150) / 4
    Femmes : IBW = height_cm - 100 - (height_cm - 150) / 2
    Plancher à 20 kg pour éviter les valeurs aberrantes (patients très petits).
    """
    if sex == "Homme":
        ibw = height_cm - 100.0 - (height_cm - 150.0) / 4.0
    else:
        ibw = height_cm - 100.0 - (height_cm - 150.0) / 2.0
    return max(ibw, 20.0)


def _bmi(weight_kg: float, height_cm: float) -> float:
    height_m = height_cm / 100.0
    return weight_kg / (height_m ** 2) if height_m > 0 else 0.0


def cockcroft_gault(patient: PatientProfile, use_ibw_if_obese: bool = True) -> float:
    """Clairance de la créatinine (mL/min) — Cockcroft & Gault (1976).

    Si use_ibw_if_obese=True et IMC > 25, le poids réel est remplacé par
    le Poids Idéal Corporel (Lorentz) pour éviter la surestimation chez le
    patient obèse ou en surpoids.

        ClCr = ((140 - âge) × poids_utilisé) / (72 × Scr_mg/dL)
    × 0.85 pour les femmes.
    """
    if patient.scr_umol_l <= 0 or patient.weight <= 0 or patient.age <= 0:
        raise ValueError("Données patient invalides (âge, poids et Scr doivent être > 0).")

    weight_used = patient.weight
    bmi = _bmi(patient.weight, patient.height_cm)
    if use_ibw_if_obese and bmi > 25.0:
        weight_used = ideal_body_weight(patient.sex, patient.height_cm)

    scr_mg_dl = patient.scr_umol_l / 88.4
    clcr = ((140 - patient.age) * weight_used) / (72.0 * scr_mg_dl)
    if patient.sex == "Femme":
        clcr *= 0.85
    return max(0.0, clcr)


def ckd_epi(patient: PatientProfile) -> float:
    """eGFR (mL/min/1.73 m²) — équation CKD-EPI 2009 (Levey et al., Ann Intern Med).

    Formule sans variable race (recommandation NKF/ASN 2021) :
        Femmes  Scr ≤ 0.7 mg/dL : 144 × (Scr/0.7)^(-0.329) × 0.993^âge
        Femmes  Scr > 0.7 mg/dL : 144 × (Scr/0.7)^(-1.209) × 0.993^âge
        Hommes  Scr ≤ 0.9 mg/dL : 141 × (Scr/0.9)^(-0.411) × 0.993^âge
        Hommes  Scr > 0.9 mg/dL : 141 × (Scr/0.9)^(-1.209) × 0.993^âge
    """
    if patient.scr_umol_l <= 0 or patient.age <= 0:
        raise ValueError("Données patient invalides pour CKD-EPI.")

    scr = patient.scr_umol_l / 88.4  # µmol/L → mg/dL

    if patient.sex == "Femme":
        kappa, alpha, coeff = 0.7, -0.329, 144.0
    else:
        kappa, alpha, coeff = 0.9, -0.411, 141.0

    ratio = scr / kappa
    exponent = alpha if scr <= kappa else -1.209
    egfr = coeff * (ratio ** exponent) * (0.993 ** patient.age)
    return max(0.0, egfr)


def adjust_ke_renal(
    ke_base: float,
    clcr: float,
    fraction_renal: float,
    clcr_normal: float = 120.0,
) -> float:
    """Ke effectif = Ke_non_rénal + α × ClCr.

    α = (Ke_base × fraction_rénale) / ClCr_normale
    """
    ke_non_renal = ke_base * (1.0 - fraction_renal)
    alpha = (ke_base * fraction_renal) / max(clcr_normal, 1e-6)
    return max(1e-6, ke_non_renal + alpha * clcr)


# -----------------------------------------------------------------------------
# Simulateur ODE générique
# -----------------------------------------------------------------------------

def simulate_pk(
    t_grid: np.ndarray,
    params: PKParams,
    admin: AdminConfig,
    regimen: Regimen,
) -> np.ndarray:
    """Simulateur PK universel — EDO résolue par odeint (LSODA).

    Les doses successives sont gérées par superposition exacte :
    intégration par segments entre breakpoints d'événements.
    """
    if params.vc <= 0:
        raise ValueError("Le volume central Vc doit être > 0.")
    if admin.n_compartments == 2 and (params.k12 <= 0 or params.k21 <= 0):
        raise ValueError("k12 et k21 doivent être > 0 (modèle 2 compartiments).")
    if admin.route == "Perfusion IV" and admin.infusion_duration <= 0:
        raise ValueError("La durée de perfusion doit être > 0.")

    has_depot = admin.route == "Orale"
    n_states = (1 if has_depot else 0) + admin.n_compartments

    idx_depot = 0 if has_depot else None
    idx_c = 1 if has_depot else 0
    idx_p = (2 if has_depot else 1) if admin.n_compartments == 2 else None

    ka, ke, k12, k21 = params.ka, params.ke, params.k12, params.k21

    def rhs(y: np.ndarray, _t: float, R: float) -> list[float]:
        dy = [0.0] * n_states
        if has_depot:
            dy[idx_depot] = -ka * y[idx_depot]
            inp = ka * y[idx_depot]
        else:
            inp = R
        Ac = y[idx_c]
        if admin.n_compartments == 2:
            Ap = y[idx_p]
            dy[idx_c] = inp - (ke + k12) * Ac + k21 * Ap
            dy[idx_p] = k12 * Ac - k21 * Ap
        else:
            dy[idx_c] = inp - ke * Ac
        return dy

    eff_dose = params.F * params.dose if has_depot else params.dose

    events: list[tuple[float, str, float]] = []
    for t_d in regimen.dose_times:
        if admin.route == "Orale":
            events.append((t_d, "depot", eff_dose))
        elif admin.route == "Bolus IV":
            events.append((t_d, "central", eff_dose))
        elif admin.route == "Perfusion IV":
            R_inf = eff_dose / admin.infusion_duration
            events.append((t_d, "rate", R_inf))
            events.append((t_d + admin.infusion_duration, "rate", 0.0))
    events.sort(key=lambda e: e[0])

    t_list = [float(x) for x in t_grid.tolist()]
    breakpoints = sorted(set([0.0] + [e[0] for e in events] + t_list))

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
            sol = odeint(rhs, state, [t0, t1], args=(R_current,),
                         rtol=1e-8, atol=1e-10, mxstep=5000)
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

    C = np.array([snapshots[tg][idx_c] / params.vc for tg in t_list])
    return np.maximum(C, 0.0)


def compute_metrics(
    t: np.ndarray,
    C: np.ndarray,
    params: PKParams,
    admin: AdminConfig,
    regimen: Regimen,
) -> PKMetrics:
    """Calcule Cmax, Tmax, AUC, t½, et concentrations à l'état d'équilibre."""
    cmax = float(np.max(C))
    tmax = float(t[int(np.argmax(C))])
    auc  = float(trapezoid(C, t))
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
        cmax=cmax, tmax=tmax, auc=auc,
        t_half_a=t_half_a, t_half_e=t_half_e,
        cmax_ss=cmax_ss, cmin_ss=cmin_ss, css=css,
    )


# -----------------------------------------------------------------------------
# Monte Carlo — variabilité inter-individuelle
# -----------------------------------------------------------------------------

def monte_carlo_simulate(
    t_grid: np.ndarray,
    params: PKParams,
    admin: AdminConfig,
    regimen: Regimen,
    cv: float,
    n_sims: int = 100,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulation de population Monte Carlo (N=100 individus).

    Modèle d'erreur log-normal sur Ke et Vc :
        P_i = P_pop × exp(η_i),  η_i ~ N(0, ω²)
        ω = sqrt(log(1 + CV²))   (paramétrage log-normal exact)

    Retourne (p5, p50, p95) en mg/L à chaque point de t_grid.
    """
    rng = np.random.default_rng(seed)
    omega = float(np.sqrt(np.log(1.0 + cv ** 2)))

    curves: list[np.ndarray] = []
    for _ in range(n_sims):
        ke_i = max(params.ke * np.exp(rng.normal(0.0, omega)), 1e-6)
        vc_i = max(params.vc * np.exp(rng.normal(0.0, omega)), 0.1)
        # évite ka ≈ ke pour la voie orale
        if admin.route == "Orale" and abs(params.ka - ke_i) < 1e-5:
            ke_i += 1e-4

        p_i = PKParams(
            dose=params.dose, F=params.F, vc=vc_i,
            ka=params.ka, ke=ke_i, k12=params.k12, k21=params.k21,
        )
        try:
            curves.append(simulate_pk(t_grid, p_i, admin, regimen))
        except Exception:  # noqa: BLE001
            pass

    if not curves:
        raise ValueError("Toutes les simulations Monte Carlo ont échoué.")

    arr = np.vstack(curves)
    return (
        np.percentile(arr, 5,  axis=0),
        np.percentile(arr, 50, axis=0),
        np.percentile(arr, 95, axis=0),
    )


# Version cachée pour éviter de recalculer à chaque interaction UI
@st.cache_data(show_spinner="Simulation de population (Monte Carlo)…")
def _cached_mc(
    t_end: float, n_pts: int,
    dose: float, F: float, vc: float, ka: float, ke: float, k12: float, k21: float,
    route: str, n_compartments: int, infusion_duration: float,
    mode: str, tau_val: float, n_doses: int,
    cv: float, n_sims: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_grid  = np.linspace(0.0, t_end, n_pts)
    params  = PKParams(dose=dose, F=F, vc=vc, ka=ka, ke=ke, k12=k12, k21=k21)
    admin   = AdminConfig(route=route, n_compartments=n_compartments, infusion_duration=infusion_duration)
    regimen = Regimen(mode=mode, tau=tau_val if tau_val > 0 else None, n_doses=n_doses)
    return monte_carlo_simulate(t_grid, params, admin, regimen, cv, n_sims)


# -----------------------------------------------------------------------------
# Confrontation aux données réelles
# -----------------------------------------------------------------------------

def fit_quality(
    t_real: np.ndarray, c_real: np.ndarray,
    t_sim: np.ndarray, c_sim: np.ndarray,
) -> FitQuality:
    """RMSE, MAE et R² entre données observées et courbe théorique interpolée."""
    if len(t_real) == 0:
        raise ValueError("Aucun point de données réel fourni.")
    c_pred = np.interp(t_real, t_sim, c_sim)
    residuals = c_real - c_pred
    rmse   = float(np.sqrt(np.mean(residuals ** 2)))
    mae    = float(np.mean(np.abs(residuals)))
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((c_real - np.mean(c_real)) ** 2))
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return FitQuality(rmse=rmse, r2=r2, n_points=int(len(t_real)), mae=mae)


def parse_real_data(uploaded_file) -> pd.DataFrame:
    """Lit un CSV clinique avec colonnes Temps et Concentration_Reelle."""
    df = pd.read_csv(uploaded_file)
    df.columns = [c.strip() for c in df.columns]
    aliases = {
        "Temps": ["Temps", "temps", "Time", "time", "t", "T"],
        "Concentration_Reelle": [
            "Concentration_Reelle", "Concentration_Réelle",
            "Concentration", "concentration", "C", "Cobs", "Concentration (mg/L)",
        ],
    }
    rename: dict[str, str] = {}
    for canonical, candidates in aliases.items():
        match = next((c for c in candidates if c in df.columns), None)
        if match is None:
            raise KeyError(
                f"Colonne '{canonical}' introuvable. Colonnes présentes : {list(df.columns)}"
            )
        rename[match] = canonical
    df = df.rename(columns=rename)[["Temps", "Concentration_Reelle"]]
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    if df.empty:
        raise ValueError("Le fichier ne contient aucune ligne numérique exploitable.")
    return df.sort_values("Temps").reset_index(drop=True)


# =============================================================================
# 3. INTERFACE STREAMLIT
# =============================================================================

st.set_page_config(page_title="Plateforme PK/PD", layout="wide", page_icon="💊")
st.title("Plateforme de simulation pharmacocinétique (PK/PD)")
st.caption(
    "Modèles 1 & 2 compartiments · Profil patient dynamique (CG / CKD-EPI · IBW) · "
    "Monte Carlo · Confrontation aux données cliniques"
)

# =============================================================================
# SIDEBAR — 4 expanders + Monte Carlo
# =============================================================================
with st.sidebar:

    # ── 1. Profil patient ────────────────────────────────────────────────────
    with st.expander("👤 Profil Patient", expanded=True):
        sex    = st.radio("Sexe", ["Homme", "Femme"], horizontal=True)
        age    = st.number_input("Âge (années)", 1, 110, 50, step=1)
        weight = st.number_input("Poids réel (kg)", 20.0, 200.0, 70.0, step=0.5)
        height = st.number_input("Taille (cm)", 100.0, 220.0, 170.0, step=1.0)
        scr    = st.number_input("Créatininémie (µmol/L)", 10.0, 600.0, 88.0, step=1.0)
        renal_formula = st.selectbox(
            "Calcul de la fonction rénale",
            ["Cockcroft-Gault", "CKD-EPI"],
            help=(
                "Cockcroft-Gault : ClCr (mL/min) — utilise le Poids Idéal (Lorentz) si IMC > 25.\n"
                "CKD-EPI : eGFR (mL/min/1.73 m²) — équation KDIGO 2009, sans variable race."
            ),
        )

    # ── 2. Paramètres PK ─────────────────────────────────────────────────────
    with st.expander("💊 Paramètres PK", expanded=True):
        n_compartments = st.selectbox(
            "Type de modèle", [1, 2],
            format_func=lambda n: f"{n} compartiment{'s' if n == 2 else ''}",
        )
        route = st.selectbox("Voie d'administration", ["Orale", "Bolus IV", "Perfusion IV"])
        dose  = st.slider("Dose (mg)", 10, 1000, 100, step=10)
        vc    = st.slider("Volume central Vc (L)", 1.0, 100.0, 20.0, step=0.5)

        if route == "Orale":
            ka = st.slider("Ka — absorption (h⁻¹)", 0.1, 5.0, 1.0, step=0.05)
            F  = st.slider("Biodisponibilité F", 0.05, 1.0, 0.90, step=0.05)
        else:
            ka, F = 0.0, 1.0

        ke_base = st.slider(
            "Ke base (h⁻¹) — référence ClCr 120 mL/min", 0.01, 2.0, 0.20, step=0.01,
        )
        fraction_renal = st.slider(
            "Fraction d'élimination rénale", 0.0, 1.0, 0.70, step=0.05,
            help="Proportion de Ke qui dépend de la fonction rénale.",
        )

        if n_compartments == 2:
            k12 = st.slider("k12 — central → tissus (h⁻¹)", 0.01, 5.0, 0.50, step=0.05)
            k21 = st.slider("k21 — tissus → central (h⁻¹)", 0.01, 5.0, 0.30, step=0.05)
        else:
            k12 = k21 = 0.0

        if route == "Perfusion IV":
            infusion_duration = st.slider("Durée de perfusion (h)", 0.1, 24.0, 1.0, step=0.1)
        else:
            infusion_duration = 0.0

    # ── 3. Régime d'administration ────────────────────────────────────────────
    with st.expander("💉 Régime d'administration", expanded=False):
        mode = st.selectbox("Régime de doses", ["Dose Unique", "Doses Répétées"])
        if mode == "Doses Répétées":
            tau     = st.slider("Intervalle τ (h)", 1, 48, 8, step=1)
            n_doses = st.slider("Nombre total de doses", 2, 30, 5, step=1)
        else:
            tau = None
            n_doses = 1

    # ── 4. Fenêtre thérapeutique ──────────────────────────────────────────────
    with st.expander("🎯 Fenêtre Thérapeutique", expanded=False):
        cme      = st.slider("CME (mg/L)", 0.0, 50.0, 2.0, step=0.1)
        toxicity = st.slider("Seuil de Toxicité (mg/L)", 0.0, 100.0, 10.0, step=0.1)

    # ── Monte Carlo ───────────────────────────────────────────────────────────
    st.markdown("---")
    mc_enabled = st.toggle(
        "Activer la variabilité inter-individuelle (Monte Carlo)",
        value=False,
        help="Génère 100 profils PK individuels avec distribution log-normale sur Ke et Vc.",
    )
    if mc_enabled:
        mc_cv = st.slider("Coefficient de Variation — CV (%)", 5, 30, 15, step=1) / 100.0
    else:
        mc_cv = 0.15


# =============================================================================
# CALCULS PRINCIPAUX
# =============================================================================
try:
    patient = PatientProfile(
        sex=sex, age=int(age), weight=float(weight),
        height_cm=float(height), scr_umol_l=float(scr),
    )
    bmi = _bmi(patient.weight, patient.height_cm)

    if renal_formula == "Cockcroft-Gault":
        clcr = cockcroft_gault(patient, use_ibw_if_obese=True)
        ibw  = ideal_body_weight(sex, float(height))
        clcr_label = f"ClCr CG = {clcr:.1f} mL/min"
        weight_note = f" (IBW = {ibw:.1f} kg utilisé, IMC = {bmi:.1f})" if bmi > 25 else f" (poids réel, IMC = {bmi:.1f})"
    else:
        clcr = ckd_epi(patient)
        clcr_label = f"eGFR CKD-EPI = {clcr:.1f} mL/min/1.73m²"
        weight_note = f" (IMC = {bmi:.1f})"

    ke_eff = adjust_ke_renal(ke_base, clcr, fraction_renal)

    if route == "Perfusion IV" and mode == "Doses Répétées" and tau is not None and infusion_duration >= tau:
        raise ValueError(f"Durée de perfusion ({infusion_duration} h) ≥ τ ({tau} h).")
    if cme >= toxicity:
        raise ValueError("La CME doit être < au Seuil de Toxicité.")
    if route == "Orale" and abs(ka - ke_eff) < 1e-6:
        raise ValueError("Ka et Ke effectif sont trop proches — différenciez-les.")

    params  = PKParams(dose=dose, F=F, vc=vc, ka=ka, ke=ke_eff, k12=k12, k21=k21)
    admin   = AdminConfig(route=route, n_compartments=n_compartments, infusion_duration=infusion_duration)
    regimen = Regimen(mode=mode, tau=tau, n_doses=n_doses)
    window  = TherapeuticWindow(cme=cme, toxicity=toxicity)

    t_end = 48.0 if mode == "Dose Unique" else float((n_doses - 1) * tau + 48)
    n_pts = max(1200, int(t_end * 25))
    t = np.linspace(0.0, t_end, n_pts)
    C = simulate_pk(t, params, admin, regimen)
    metrics = compute_metrics(t, C, params, admin, regimen)

except (ValueError, KeyError) as exc:
    st.error(f"⛔ {exc}")
    st.stop()
except Exception as exc:  # noqa: BLE001
    st.error(f"⛔ Erreur de simulation : {exc}")
    st.stop()

# Monte Carlo (mis en cache — ne recalcule que si les paramètres changent)
mc_band: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]] = None
if mc_enabled:
    try:
        mc_band = _cached_mc(
            t_end=t_end, n_pts=n_pts,
            dose=dose, F=F, vc=vc, ka=ka, ke=ke_eff, k12=k12, k21=k21,
            route=route, n_compartments=n_compartments, infusion_duration=infusion_duration,
            mode=mode, tau_val=float(tau) if tau is not None else 0.0, n_doses=n_doses,
            cv=mc_cv, n_sims=100,
        )
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Monte Carlo échoué : {exc}")

# Données réelles (persistées entre onglets via session_state)
if "real_data" not in st.session_state:
    st.session_state["real_data"] = None

real_df: Optional[pd.DataFrame] = st.session_state.get("real_data")


# =============================================================================
# GRAPHIQUE PLOTLY — fonction de construction
# =============================================================================

def build_pk_figure(
    t: np.ndarray,
    C: np.ndarray,
    metrics: PKMetrics,
    window: TherapeuticWindow,
    regimen: Regimen,
    real_data: Optional[pd.DataFrame] = None,
    log_y: bool = False,
    mc_band: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
) -> go.Figure:
    """Graphique Plotly interactif — linéaire ou semi-logarithmique."""
    # Pour l'échelle log, on plancher C à 1e-5 pour éviter log(0)
    C_plot = np.maximum(C, 1e-5) if log_y else C

    fig = go.Figure()

    # Fenêtre thérapeutique
    fig.add_hrect(
        y0=window.cme, y1=window.toxicity,
        fillcolor="rgba(34,197,94,0.13)", line_width=0,
        annotation_text="Fenêtre thérapeutique", annotation_position="top left",
    )
    fig.add_hline(y=window.cme, line=dict(color="#16A34A", dash="dot", width=1.4),
                  annotation_text=f"CME = {window.cme:.1f}", annotation_position="bottom right")
    fig.add_hline(y=window.toxicity, line=dict(color="#DC2626", dash="dot", width=1.4),
                  annotation_text=f"Toxicité = {window.toxicity:.1f}", annotation_position="top right")

    # ── Ruban Monte Carlo (IC 90 %) ────────────────────────────────────────
    if mc_band is not None:
        p5, p50, p95 = mc_band
        p5_plot  = np.maximum(p5,  1e-5) if log_y else p5
        p50_plot = np.maximum(p50, 1e-5) if log_y else p50
        p95_plot = np.maximum(p95, 1e-5) if log_y else p95

        # Bord inférieur (p5) — invisible, sert d'ancrage pour fill
        fig.add_trace(go.Scatter(
            x=t, y=p5_plot, mode="lines", name="IC 90% — borne basse",
            line=dict(color="rgba(147,197,253,0)"),
            showlegend=False, hoverinfo="skip",
        ))
        # Bord supérieur (p95) + fill vers p5
        fig.add_trace(go.Scatter(
            x=t, y=p95_plot, mode="lines", name="IC 90 % (5e–95e percentile)",
            fill="tonexty", fillcolor="rgba(147,197,253,0.35)",
            line=dict(color="rgba(147,197,253,0)"),
            hovertemplate="p95 = %{y:.3f} mg/L<extra></extra>",
        ))
        # Médiane
        fig.add_trace(go.Scatter(
            x=t, y=p50_plot, mode="lines", name="Médiane MC",
            line=dict(color="#1D4ED8", width=2.2, dash="dot"),
            hovertemplate="t=%{x:.2f} h · p50=%{y:.3f} mg/L<extra></extra>",
        ))

    # ── Courbe déterministe ────────────────────────────────────────────────
    fill_mode = "none" if (log_y or mc_band is not None) else "tozeroy"
    fig.add_trace(go.Scatter(
        x=t, y=C_plot, mode="lines",
        name="C(t) médian" if mc_band is not None else "C(t) théorique",
        line=dict(color="#2563EB", width=2.6),
        fill=fill_mode, fillcolor="rgba(37,99,235,0.07)",
        hovertemplate="t = %{x:.2f} h<br>C = %{y:.3f} mg/L<extra></extra>",
    ))

    # ── Repères Cmax / Cmin à l'état d'équilibre ──────────────────────────
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

    # ── Marqueurs d'administration ─────────────────────────────────────────
    if regimen.mode == "Doses Répétées":
        for t_d in regimen.dose_times:
            fig.add_vline(x=t_d, line=dict(color="rgba(148,163,184,0.55)", dash="dot", width=1))

    # ── Données cliniques observées ────────────────────────────────────────
    if real_data is not None and not real_data.empty:
        fig.add_trace(go.Scatter(
            x=real_data["Temps"], y=real_data["Concentration_Reelle"],
            mode="markers", name="Données cliniques observées",
            marker=dict(color="#111827", size=9, symbol="circle",
                        line=dict(color="white", width=1)),
            hovertemplate="t = %{x:.2f} h<br>C_obs = %{y:.3f} mg/L<extra></extra>",
        ))

    fig.update_layout(
        xaxis_title="Temps (h)",
        yaxis_title="Concentration plasmatique (mg/L)"
                    + (" — échelle logarithmique" if log_y else ""),
        yaxis_type="log" if log_y else "linear",
        hovermode="x unified",
        template="plotly_white",
        height=570,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=65, r=40, t=70, b=55),
    )
    return fig


# =============================================================================
# BANDEAU D'EN-TÊTE
# =============================================================================
with st.container(border=True):
    h1, h2, h3, h4, h5 = st.columns(5)
    h1.metric(renal_formula, clcr_label.split("=")[1].strip())
    h2.metric("Ke effectif", f"{ke_eff:.3f} h⁻¹",
              delta=f"{(ke_eff - ke_base):+.3f} vs base", delta_color="off")
    h3.metric("Voie · Modèle", f"{route} · {n_compartments}-cmt")
    h4.metric(
        "Régime",
        f"{n_doses}× {dose} mg" if mode == "Doses Répétées" else f"1× {dose} mg",
        delta=f"τ = {tau} h" if mode == "Doses Répétées" else "Dose unique",
        delta_color="off",
    )
    h5.metric("Monte Carlo", f"CV = {mc_cv*100:.0f} %" if mc_enabled else "Désactivé",
              delta="100 simulations" if mc_enabled else "Courbe unique", delta_color="off")

# =============================================================================
# ONGLETS
# =============================================================================
tab_sim, tab_clin, tab_fit, tab_export = st.tabs(
    ["📈 Simulation", "🩺 Analyse Clinique", "📊 Fit Data", "📥 Export"]
)

# ─── Onglet 1 — Simulation ───────────────────────────────────────────────────
with tab_sim:

    # Métriques
    if mode == "Dose Unique":
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Cmax",          f"{metrics.cmax:.2f} mg/L")
        c2.metric("Tmax",          f"{metrics.tmax:.2f} h")
        c3.metric("AUC totale",    f"{metrics.auc:.1f} mg·h/L")
        c4.metric("t½ élim.",      f"{metrics.t_half_e:.2f} h")
        c5.metric("t½ abs.",       f"{metrics.t_half_a:.2f} h" if route == "Orale" else "—")
    else:
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Cmax,ss",       f"{metrics.cmax_ss:.2f} mg/L")
        c2.metric("Cmin,ss",       f"{metrics.cmin_ss:.2f} mg/L")
        c3.metric("Css moyen",     f"{metrics.css:.2f} mg/L")
        c4.metric("AUC totale",    f"{metrics.auc:.1f} mg·h/L")
        c5.metric("t½ élim.",      f"{metrics.t_half_e:.2f} h")
        c6.metric("t½ abs.",       f"{metrics.t_half_a:.2f} h" if route == "Orale" else "—")

    # Toggle échelle logarithmique
    log_y = st.toggle(
        "Afficher l'axe Y en échelle logarithmique",
        value=False,
        help=(
            "L'échelle semi-logarithmique est standard en PK pour identifier "
            "la linéarité de la phase d'élimination (pente = −Ke / ln 10). "
            "Une petite constante (1×10⁻⁵) est ajoutée pour éviter log(0)."
        ),
    )

    st.plotly_chart(
        build_pk_figure(t, C, metrics, window, regimen, real_df, log_y=log_y, mc_band=mc_band),
        use_container_width=True,
    )

    if mc_band is not None:
        st.caption(
            f"Le ruban bleu représente l'**IC 90 %** (5e–95e percentile) "
            f"de 100 profils simulés avec CV = {mc_cv*100:.0f} % sur Ke et Vc "
            f"(distribution log-normale, ω = {np.sqrt(np.log(1+mc_cv**2)):.3f})."
        )

# ─── Onglet 2 — Analyse Clinique ──────────────────────────────────────────────
with tab_clin:
    st.subheader("Résumé mathématique du modèle")

    if n_compartments == 1 and route == "Orale":
        st.latex(
            r"C(t) = \sum_{i=0}^{N-1} \frac{F \cdot D \cdot K_a}{V_c (K_a - K_e)}"
            r"\left(e^{-K_e(t-i\tau)} - e^{-K_a(t-i\tau)}\right)\mathbf{1}_{t \geq i\tau}"
        )
    elif n_compartments == 1 and route == "Bolus IV":
        st.latex(r"C(t) = \sum_{i} \frac{D}{V_c}\,e^{-K_e(t-i\tau)}\,\mathbf{1}_{t \geq i\tau}")
    elif n_compartments == 1 and route == "Perfusion IV":
        st.latex(r"C(t)=\frac{R_0}{V_c K_e}\left(1-e^{-K_e t}\right),\;0\leq t\leq T_{inf}")
        st.latex(r"C(t)=\frac{R_0}{V_c K_e}\left(1-e^{-K_e T_{inf}}\right)e^{-K_e(t-T_{inf})},\;t>T_{inf}")
    else:
        st.latex(
            r"\begin{cases}"
            r"\dot{A}_c = \text{entrée}(t)-(K_e+k_{12})A_c+k_{21}A_p\\"
            r"\dot{A}_p = k_{12}A_c-k_{21}A_p"
            r"\end{cases}"
        )
        st.caption("Résolu numériquement via scipy.integrate.odeint (LSODA).")

    st.markdown("---")
    st.subheader("Profil patient & fonction rénale")
    pc = st.columns(5)
    pc[0].markdown(f"**Sexe** : {sex}")
    pc[1].markdown(f"**Âge** : {age} ans")
    pc[2].markdown(f"**Poids** : {weight} kg")
    pc[3].markdown(f"**Taille** : {height} cm")
    pc[4].markdown(f"**IMC** : {bmi:.1f} kg/m²")

    if renal_formula == "Cockcroft-Gault":
        ibw_val = ideal_body_weight(sex, float(height))
        st.info(
            f"**Cockcroft-Gault** : ClCr = **{clcr:.1f} mL/min**{weight_note}  ·  "
            f"Ke base = {ke_base:.3f} h⁻¹ → **Ke effectif = {ke_eff:.3f} h⁻¹** "
            f"(fraction rénale : {fraction_renal:.0%})"
        )
    else:
        st.info(
            f"**CKD-EPI 2009** : eGFR = **{clcr:.1f} mL/min/1.73 m²**  ·  "
            f"Ke base = {ke_base:.3f} h⁻¹ → **Ke effectif = {ke_eff:.3f} h⁻¹** "
            f"(fraction rénale : {fraction_renal:.0%})"
        )

    if renal_formula == "Cockcroft-Gault":
        st.latex(
            r"\text{ClCr} = \frac{(140 - \text{âge}) \times \text{poids}}"
            r"{72 \times \text{Scr}_{\text{mg/dL}}} \times [0.85\text{ si femme}]"
        )
        if bmi > 25:
            st.latex(
                r"\text{IBW}_{\text{Lorentz}} = \text{taille} - 100 "
                r"- \frac{\text{taille}-150}{4 \text{ (H) ou } 2 \text{ (F)}}"
            )
    else:
        st.latex(
            r"\text{eGFR} = A \times \left(\frac{\text{Scr}}{\kappa}\right)^{\alpha}"
            r"\times 0.993^{\text{âge}} \quad (A,\kappa,\alpha \text{ dépendent du sexe})"
        )

    st.markdown("---")
    if mode == "Doses Répétées":
        st.subheader("Concentrations à l'état d'équilibre")
        st.latex(r"C_{ss} = \frac{F \cdot D}{V_c \cdot K_e \cdot \tau}")
        m1, m2, m3 = st.columns(3)
        m1.metric("Cmax,ss", f"{metrics.cmax_ss:.3f} mg/L",
                  delta="↑ Toxique" if metrics.cmax_ss > toxicity else "✓ OK",
                  delta_color="inverse" if metrics.cmax_ss > toxicity else "normal")
        m2.metric("Cmin,ss", f"{metrics.cmin_ss:.3f} mg/L",
                  delta="↓ Sous CME" if metrics.cmin_ss < cme else "✓ OK",
                  delta_color="inverse" if metrics.cmin_ss < cme else "normal")
        m3.metric("Css moyen", f"{metrics.css:.3f} mg/L")

        st.markdown("---")
        st.subheader("Bilan clinique")
        problems = []
        if metrics.cmin_ss < cme:
            problems.append(
                f"⚠️ **Sous-exposition** : Cmin,ss = **{metrics.cmin_ss:.2f} mg/L** < CME ({cme:.1f} mg/L)."
            )
        if metrics.cmax_ss > toxicity:
            problems.append(
                f"⚠️ **Risque toxique** : Cmax,ss = **{metrics.cmax_ss:.2f} mg/L** > Toxicité ({toxicity:.1f} mg/L)."
            )
        if not problems:
            st.success(
                f"✅ **Traitement équilibré** : Cmax,ss ({metrics.cmax_ss:.2f}) et "
                f"Cmin,ss ({metrics.cmin_ss:.2f}) sont dans la fenêtre [{cme:.1f} – {toxicity:.1f}] mg/L."
            )
        else:
            for p in problems:
                st.error(p)
    else:
        st.info("Activez **Doses Répétées** pour l'analyse de l'état d'équilibre.")

# ─── Onglet 3 — Fit Data ──────────────────────────────────────────────────────
with tab_fit:
    st.subheader("Confrontation modèle théorique ↔ données cliniques")
    st.markdown(
        "Importez un CSV avec les colonnes **`Temps`** (h) et **`Concentration_Reelle`** (mg/L)."
    )

    uploaded = st.file_uploader("Charger un fichier CSV de mesures cliniques", type=["csv"])
    if uploaded is not None:
        try:
            st.session_state["real_data"] = parse_real_data(uploaded)
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
                real_df["Temps"].to_numpy(), real_df["Concentration_Reelle"].to_numpy(), t, C,
            )
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Points observés", quality.n_points)
            mc2.metric("RMSE",            f"{quality.rmse:.3f} mg/L")
            mc3.metric("MAE",             f"{quality.mae:.3f} mg/L")
            mc4.metric("R²",              f"{quality.r2:.3f}")

            if quality.r2 >= 0.90:
                st.success("✅ Excellent ajustement (R² ≥ 0.90).")
            elif quality.r2 >= 0.70:
                st.info("ℹ️ Ajustement correct (0.70 ≤ R² < 0.90).")
            else:
                st.warning("⚠️ Ajustement médiocre (R² < 0.70) — revoir les paramètres PK.")

            st.dataframe(real_df, use_container_width=True, height=220)
            st.plotly_chart(
                build_pk_figure(t, C, metrics, window, regimen, real_df, mc_band=mc_band),
                use_container_width=True,
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"⛔ Calcul d'ajustement impossible : {exc}")
    else:
        st.info("Aucune donnée clinique chargée.")
        with st.expander("📄 Exemple de format CSV attendu"):
            st.code(
                "Temps,Concentration_Reelle\n"
                "0.5,0.85\n1.0,1.92\n2.0,3.10\n4.0,3.85\n"
                "8.0,2.74\n12.0,1.65\n24.0,0.42\n",
                language="csv",
            )

# ─── Onglet 4 — Export ────────────────────────────────────────────────────────
with tab_export:
    st.subheader("Téléchargement des données simulées")

    df_export = pd.DataFrame({
        "Temps_h":                np.round(t, 4),
        "Concentration_mgL":      np.round(C, 6),
    })
    if mc_band is not None:
        p5, p50, p95 = mc_band
        df_export["MC_p5_mgL"]  = np.round(p5,  6)
        df_export["MC_p50_mgL"] = np.round(p50, 6)
        df_export["MC_p95_mgL"] = np.round(p95, 6)

    st.dataframe(df_export, use_container_width=True, height=320)

    fname = (
        f"pk_{route.replace(' ','_')}_{n_compartments}cmt_"
        f"{'multi' if mode=='Doses Répétées' else 'single'}"
        f"{'_MC' if mc_band is not None else ''}.csv"
    )
    st.download_button(
        label="⬇️ Télécharger la simulation (CSV)",
        data=df_export.to_csv(index=False).encode("utf-8"),
        file_name=fname, mime="text/csv",
    )

    st.markdown("---")
    st.caption(
        f"{route} · {n_compartments}-cmt · Dose {dose} mg · Vc {vc} L · "
        f"Ka {ka:.2f} h⁻¹ · Ke_eff {ke_eff:.3f} h⁻¹ · F {F:.2f} · "
        f"{clcr_label}{weight_note}"
    )

st.markdown("---")
st.caption("Plateforme PK/PD · Streamlit · Plotly · scipy.integrate.odeint")
