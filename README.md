# 💊 Plateforme de Simulation Pharmacocinétique Avancée (PK/PD)

Ce projet est une application web interactive développée en **Python** avec **Streamlit**, **Plotly** et **SciPy**. Elle permet de simuler des profils de concentration plasmatique pour des modèles à 1 et 2 compartiments, d'adapter les posologies selon la physiologie du patient (fonction rénale), et d'effectuer des simulations de population (Monte Carlo).

## 🚀 Lien de l'application en ligne
👉 **[Clique ici pour tester l'application en direct] https://simulateur-pk-pd-dti4c4aumhjziywvmeuwue.streamlit.app/**

## 🔬 Fonctionnalités principales
* **Modélisation avancée :** Modèles à 1 ou 2 compartiments, voies Orale, Bolus IV et Perfusion.
* **Profil Patient Dynamique :** Estimation de la clairance de la créatinine (Cockcroft-Gault et CKD-EPI) avec correction du poids.
* **Analyse de Population :** Simulation Monte Carlo avec intervalle de confiance à 90%.
* **Confrontation au réel :** Import de données réelles (CSV) et calcul de la qualité d'ajustement ($R^2$, RMSE).

## 🛠️ Installation en local
Pour lancer le projet sur votre machine :
1. Clonez ce dépôt.
2. Installez les dépendances : `pip install -r requirements.txt`
3. Lancez l'application : `streamlit run app.py`

---
*Projet développé par Mathéo (Double cursus Pharmacie / Génie Chimique).*