import streamlit as st
import pandas as pd
import plotly.express as px
import os
from pathlib import Path

# -----------------------------------------------------------------------------
# 📌 Configuration générale
# -----------------------------------------------------------------------------
BASE_HINT = r"C:\Users\aulac\OneDrive\Documents\Trading\PyCharmProject\MLStrategy\data_preprocessing\results_optim"

st.set_page_config(page_title="SHAP Feature Explorer", layout="wide")
st.title("📂 Analyse SHAP – Sélection manuelle + filtres dynamiques")

st.markdown(f"ℹ️ **Conseil :** sélectionnez un fichier dans : `{BASE_HINT}`")

# -----------------------------------------------------------------------------
# 📁 Sélection du dossier d'export (facultatif)
# -----------------------------------------------------------------------------
custom_output_dir = st.text_input(
    "📁 Dossier d'exportation personnalisé (laisser vide pour exporter dans le même dossier)",
    "",
)

# -----------------------------------------------------------------------------
# 📥 Téléversement du fichier CSV de métriques SHAP
# -----------------------------------------------------------------------------
uploaded_file = st.file_uploader("👉 Choisissez un fichier CSV SHAP à analyser :", type=["csv"])

if uploaded_file:
    # -------------------------------------------------------------------------
    # 📊 Lecture du fichier
    # -------------------------------------------------------------------------
    metrics_df = pd.read_csv(uploaded_file, sep=";", index_col=0)
    st.success(f"✅ Fichier chargé : {uploaded_file.name}")
    st.write(f"Nombre de features chargées : {metrics_df.shape[0]}")

    # -------------------------------------------------------------------------
    # 🔧 Pré‑calculs pour les métriques dépendantes des folds
    # -------------------------------------------------------------------------
    shap_fold_cols = [col for col in metrics_df.columns if col.startswith("fold_")]
    nb_folds = len(shap_fold_cols)

    # 🏷️ Rank par fold afin de recalculer freq_topN à la volée (pas besoin de la
    # colonne freq_topN présente dans certaines exports)
    # --------------------------------------------------------
    @st.cache_data(hash_funcs={pd.DataFrame: lambda _: None})
    def _compute_rank_df(df: pd.DataFrame) -> pd.DataFrame:
        # Classement décroissant (plus le SHAP est grand, plus le rang est petit)
        return df.rank(axis=0, method="min", ascending=False)

    rank_df = _compute_rank_df(metrics_df[shap_fold_cols])

    # -------------------------------------------------------------------------
    # 🎛️ Filtres de base dans la sidebar
    # -------------------------------------------------------------------------
    st.sidebar.header("🎛️ Filtres SHAP – Vue simplifiée")

    min_mean = st.sidebar.slider("Seuil minimal de mean_SHAP", 0.0, 0.05, 0.005, step=0.001)
    max_cv = st.sidebar.slider("Seuil maximal de cv_SHAP", 0.0, 1.0, 0.30, step=0.01)
    top_k = st.sidebar.number_input("Limiter à Top‑K features (0 = pas de limite)", 0, 300, 0, step=1)

    use_freq_topN_filter = st.sidebar.checkbox("Filtrer par fréquence dans le Top‑N (simple)")
    if use_freq_topN_filter:
        simple_topN = st.sidebar.slider("N pour freq_topN (Top‑N simple)", 1, 50, 20)
        # Recalcul simple de freq_topN
        freq_topN_simple = (rank_df <= simple_topN).sum(axis=1)
        metrics_df["freq_topN"] = freq_topN_simple  # 🔄 overwrite / crée
        min_freq_topN_simple = st.sidebar.slider("Fréquence minimale dans le Top‑N", 0, nb_folds, 3)
    else:
        min_freq_topN_simple = None

    # -------------------------------------------------------------------------
    # ⚙️ Paramètres avancés – Règles parallèles (Bloc A et Bloc B)
    # -------------------------------------------------------------------------
    st.sidebar.subheader("⚙️ Paramètres avancés (Bloc A & Bloc B)")

    # 🔹 Paramètres pour le calcul de freq_topN avancé
    adv_topN = st.sidebar.slider("N (Top‑N par fold) pour freq_topN avancé", 1, 50, 20)
    # Recalcul avancé de freq_topN (peut être différent du simple)
    metrics_df["freq_topN_adv"] = (rank_df <= adv_topN).sum(axis=1)

    # 🔹 Seuils Bloc A
    min_freq_topN_A = st.sidebar.slider("Bloc A : Fréquence minimale dans le Top‑N", 0, nb_folds, 2)
    max_cv_A = st.sidebar.slider("Bloc A : Seuil maximal de cv_SHAP", 0.0, 1.0, 0.5, step=0.01)

    # 🔹 Seuils Bloc B
    min_mean_B = st.sidebar.slider("Bloc B : Seuil minimal de mean_SHAP", 0.0, 0.05, 0.002, step=0.001)
    min_freq_nonzero_B = st.sidebar.slider("Bloc B : Fréquence minimale non‑zéro", 0, nb_folds, 2)

    # -------------------------------------------------------------------------
    # 🧮 Calcul des métriques dérivées (freq_nonzero)
    # -------------------------------------------------------------------------
    if "freq_nonzero" not in metrics_df.columns:
        metrics_df["freq_nonzero"] = (metrics_df[shap_fold_cols] != 0).sum(axis=1)

    # -------------------------------------------------------------------------
    # 🗂️ Application des filtres de base (kept)
    # -------------------------------------------------------------------------
    metrics_df["kept"] = (
        (metrics_df["mean_SHAP"] >= min_mean) & (metrics_df["cv_SHAP"] <= max_cv)
    )
    if min_freq_topN_simple is not None:
        metrics_df["kept"] &= (metrics_df["freq_topN"] >= min_freq_topN_simple)

    # -------------------------------------------------------------------------
    # 🔍 Règles parallèles avancées (Bloc A & Bloc B)
    # -------------------------------------------------------------------------
    mask_A = (
        (metrics_df["freq_topN_adv"] >= min_freq_topN_A)
        & (metrics_df["cv_SHAP"] <= max_cv_A)
    )
    mask_B = (
        (metrics_df["mean_SHAP"] >= min_mean_B)
        & (metrics_df["freq_nonzero"] >= min_freq_nonzero_B)
    )

    combined_mask = mask_A | mask_B
    nb_total_combined = int(combined_mask.sum())
    features_combined = metrics_df.index[combined_mask].tolist()

    # -------------------------------------------------------------------------
    # 🖨️ Logs console
    # -------------------------------------------------------------------------
    print(
        f"🔍 Bloc A (TopN ≥ {min_freq_topN_A} & cv_SHAP ≤ {max_cv_A}) : {mask_A.sum()} features"
    )
    print(
        f"🔍 Bloc B (mean_SHAP ≥ {min_mean_B} & freq_nonzero ≥ {min_freq_nonzero_B}) : {mask_B.sum()} features"
    )
    print(f"✅ Total combiné (A ∪ B) : {nb_total_combined} features")
    for feat in features_combined:
        print(f"  - {feat}")

    # -------------------------------------------------------------------------
    # 🖼️ Interface Streamlit – Récapitulatif Bloc A / Bloc B
    # -------------------------------------------------------------------------
    st.markdown("### 🔎 Aperçu des règles de filtrage avancées (parallèle)")
    summary_text = (
        f"Bloc A (TopN ≥ {min_freq_topN_A} & cv_SHAP ≤ {max_cv_A}) : {mask_A.sum()} features\n"
        f"Bloc B (mean_SHAP ≥ {min_mean_B} & freq_nonzero ≥ {min_freq_nonzero_B}) : {mask_B.sum()} features\n"
        f"Total combiné retenu par A ou B : {nb_total_combined} features"
    )
    st.code(summary_text, language="markdown")

    st.text_area(
        label="🧠 Features retenues (règle A ∪ B)",
        value="\n".join(features_combined),
        height=200,
    )

    # -------------------------------------------------------------------------
    # 📋 Affichage des features retenues par le filtrage principal
    # -------------------------------------------------------------------------
    filtered = metrics_df[metrics_df["kept"]].copy()
    if top_k > 0:
        filtered = filtered.sort_values("mean_SHAP", ascending=False).head(top_k)

    st.subheader(f"📊 {len(filtered)} features retenues après filtrage de base")
    st.dataframe(filtered.sort_values("mean_SHAP", ascending=False))

    fig = px.scatter(
        metrics_df,
        x="mean_SHAP",
        y="cv_SHAP",
        color="kept",
        hover_name=metrics_df.index,
        size="freq_topN_adv",
        title="SHAP importance vs Stabilité",
    )
    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------------------------------
    # 💾 Export CSV
    # -------------------------------------------------------------------------
    if st.button("💾 Exporter les features filtrées"):
        export_name = f"filtered_{uploaded_file.name}"

        # Log features filtrées
        print(f"✅ {len(filtered)} features retenues (filtrage standard):")
        for f in filtered.index.tolist():
            print(f"  - {f}")

        if custom_output_dir and os.path.isdir(custom_output_dir):
            export_path = os.path.join(custom_output_dir, export_name)
            try:
                filtered.to_csv(export_path, sep=";")
                st.success(f"✅ Exporté avec succès dans : `{export_path}`")
            except Exception as e:
                st.error(f"❌ Erreur lors de l'export : {str(e)}")
                st.download_button(
                    label="📥 Télécharger le fichier CSV filtré",
                    data=filtered.to_csv(sep=";").encode("utf-8"),
                    file_name=export_name,
                    mime="text/csv",
                )
        else:
            st.download_button(
                label="📥 Télécharger le fichier CSV filtré",
                data=filtered.to_csv(sep=";").encode("utf-8"),
                file_name=export_name,
                mime="text/csv",
            )
            if custom_output_dir:
                st.warning(f"⚠️ Le dossier `{custom_output_dir}` est invalide.")
else:
    st.info(f"Veuillez téléverser un fichier CSV à partir de `{BASE_HINT}` ou ailleurs.")
