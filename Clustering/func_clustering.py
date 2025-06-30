# ────────────────────────────────────
# 0)  IMPORTS & CONFIG
# ────────────────────────────────────
import os
import logging
logger = logging.getLogger(__name__)
os.environ["OMP_NUM_THREADS"] = "2"
import os
import platform
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
from sklearn.cluster import KMeans
from typing import Dict, Any
import logging

from Tools.func_features_preprocessing import calculate_session_metrics_enhanced,print_enhanced_summary_statistics_with_sessions
import datetime

from func_standard import setup_plotting_style
from Tools.func_features_preprocessing import validate_dataframe_structure
clustering_with_K =1
GROUPE_SESSION_1 = [0]
GROUPE_SESSION_2 = [0,1,2,3,4,5,6]
def get_feature_columns():
    """
    Retourne la liste des colonnes de features actives.
    """
    feature_columns = [
        #'volume_p50',
        'volMeanPerTick_p25',
        'volMeanPerTick_p75',

        #'atr_p50',
        # 'duration_p50',
        # 'extreme_ratio',
        # 'vol_above_p50',
       # 'volume_spread',
        #'volume_above_spread',
        # 'atr_spread',
        # 'duration_spread',
       # 'event'
    ]
    return feature_columns
def relabel_by_metric(labels: np.ndarray, metric_values: pd.Series):
    """
    Trie les clusters par valeur moyenne CROISSANTE d'une métrique (ex: volume_p50_g1)
    et renvoie (new_labels, mapping{old→new}).

    - new label 0  = le cluster au volume moyen le plus FAIBLE   (« CANDLE_MEAN_LOW »)
    - new label K-1= le cluster au volume moyen le plus FORT     (« CANDLE_MEAN_HIGH »)
    """
    order = (metric_values.groupby(labels).mean()      # moyenne par cluster
                              .sort_values(ascending=True)
                              .index.to_list())
    mapping = {old: new for new, old in enumerate(order)}
    return np.vectorize(mapping.get)(labels), mapping

# ────────────────────────────────────
# 3)  STANDARDISATION G1 et G2
# ────────────────────────────────────
from sklearn.preprocessing import StandardScaler

def scale_features(df, feature_cols):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])
    return X_scaled, scaler


# ─── Stratégies dynamiques adaptatives
def get_adaptive_strategy(src_level, dst_level, clustering_with_K,level_names):
    src_idx = level_names.index(src_level)
    dst_idx = level_names.index(dst_level)

    if src_idx == dst_idx:
        if src_idx == 0:
            return "Maintenir paramètres conservateurs"
        elif src_idx == clustering_with_K - 1:
            return "Maintenir paramètres agressifs"
        else:
            return f"Maintenir stratégie niveau {src_level}"
    elif dst_idx > src_idx:
        diff = dst_idx - src_idx
        if diff == 1:
            return "Escalade progressive, ajuster position size"
        else:
            return "Escalade forte, préparer stratégie agressive"
    else:  # dst_idx < src_idx
        diff = src_idx - dst_idx
        if diff == 1:
            return "Décélération progressive, considérer prise profits"
        else:
            return "Retour au CANDLE_MEAN_LOW, stratégie mean-reversion"



def calculate_cluster_bounds(df, cluster_col, feature_cols, percentiles=[5, 25, 50, 75, 95]):
    """
    Calcule les bornes (percentiles) de chaque cluster pour les features données
    """
    bounds_data = {}

    for cluster_id in sorted(df[cluster_col].unique()):
        cluster_data = df[df[cluster_col] == cluster_id]
        bounds_data[cluster_id] = {}

        for feature in feature_cols:
            if feature in df.columns:
                feature_data = cluster_data[feature]
                bounds_data[cluster_id][feature] = {
                    f'P{p}': feature_data.quantile(p / 100) for p in percentiles
                }
                # Ajout min/max pour référence
                bounds_data[cluster_id][feature]['Min'] = feature_data.min()
                bounds_data[cluster_id][feature]['Max'] = feature_data.max()
                bounds_data[cluster_id][feature]['Count'] = len(feature_data)

    return bounds_data


def create_bounds_summary(bounds_g1,bounds_g2,feature_columns_g1,feature_columns_g2,clustering_with_K):
    """Crée un tableau récapitulatif des bornes principales"""
    summary_data = []

    for group, bounds, features in [('G1', bounds_g1, feature_columns_g1), ('G2', bounds_g2, feature_columns_g2)]:
        for cluster_id in range(clustering_with_K):
            if cluster_id in bounds:
                cluster_label = {0: 'CANDLE_MEAN_LOW', 1: 'VOL_CADL_MED' if clustering_with_K >= 3 else 'CANDLE_MEAN_HIGH',
                                 2: 'CANDLE_MEAN_HIGH' if clustering_with_K >= 3 else None}.get(cluster_id,
                                                                                            f'CLUSTER_{cluster_id}')

                row = {
                    'Groupe': group,
                    'Cluster': f"{cluster_id} ({cluster_label})",
                    'Count': bounds[cluster_id][features[0]]['Count']  # Utilise la première feature pour le count
                }

                for feature in features:
                    if feature != 'event':
                        base_name = feature.replace('_g1', '').replace('_g2', '')
                        row[f'{base_name}_P25'] = bounds[cluster_id][feature]['P25']
                        row[f'{base_name}_P50'] = bounds[cluster_id][feature]['P50']
                        row[f'{base_name}_P75'] = bounds[cluster_id][feature]['P75']

                summary_data.append(row)

    return pd.DataFrame(summary_data)

def calculate_consecutive_regime(regime_series):
    """Calcule le nombre de sessions consécutives dans le même régime"""
    consecutive = []
    current_count = 1

    for i in range(len(regime_series)):
        if i == 0:
            consecutive.append(1)
        elif regime_series.iloc[i] == regime_series.iloc[i - 1]:
            current_count += 1
            consecutive.append(current_count)
        else:
            current_count = 1
            consecutive.append(current_count)

    return consecutive

def calculate_risk_level(regime_g1, prediction_g2, confidence):
    """
    Calcule le niveau de risque de la session
    """
    g1_risk = {'CANDLE_MEAN_LOW': 1, 'VOL_CADL_MED': 2, 'CANDLE_MEAN_HIGH': 3}.get(regime_g1, 2)
    g2_risk = {'CANDLE_MEAN_LOW': 1, 'VOL_CADL_MED': 2, 'CANDLE_MEAN_HIGH': 3}.get(prediction_g2, 2)

    # Risque moyen pondéré par la confiance
    avg_risk = (g1_risk + g2_risk) / 2
    confidence_factor = confidence * 0.5 + 0.5  # Entre 0.5 et 1.0

    risk_score = avg_risk * confidence_factor

    if risk_score < 1.5:
        return 'LOW', risk_score
    elif risk_score < 2.5:
        return 'MEDIUM', risk_score
    else:
        return 'HIGH', risk_score



def generate_trading_signal(regime_g1, prediction_g2, probability, volume_momentum):
    """
    Génère un signal de trading basé sur la transition prédite
    """
    # Conversion en labels si nécessaire
    g1_label = {0: 'CANDLE_MEAN_LOW', 1: 'VOL_CADL_MED', 2: 'CANDLE_MEAN_HIGH'}.get(regime_g1, regime_g1)
    g2_label = {0: 'CANDLE_MEAN_LOW', 1: 'VOL_CADL_MED', 2: 'CANDLE_MEAN_HIGH'}.get(prediction_g2, prediction_g2)

    # Logique de signaux
    if g1_label == 'CANDLE_MEAN_LOW' and g2_label in ['VOL_CADL_MED', 'CANDLE_MEAN_HIGH'] and probability > 0.6:
        return 'BUY_SIGNAL', 'Escalade prédite avec forte probabilité'
    elif g1_label == 'CANDLE_MEAN_HIGH' and g2_label == 'CANDLE_MEAN_LOW' and probability > 0.6:
        return 'SELL_SIGNAL', 'Retour au CANDLE_MEAN_LOW prédit'
    elif g1_label == g2_label and probability > 0.7:
        return 'HOLD_SIGNAL', 'Persistance prédite'
    elif probability < 0.4:
        return 'UNCERTAIN', 'Prédiction peu fiable'
    else:
        return 'NEUTRAL', 'Pas de signal clair'

def predict_g2_from_g1(regime_g1, transition_matrix):
    if regime_g1 in transition_matrix.index:
        col = transition_matrix.loc[regime_g1]
        return col.idxmax(), col.max()
    else:
        # Proba uniforme en cas d'absence
        k = transition_matrix.shape[1]
        return regime_g1, 1 / k



def analyze_correlations_clustering(df, feature_list, group_name="Features", suffix="", save_plots=False):
    """
    Analyse les corrélations Pearson et Spearman pour un groupe de features

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame contenant les données
    feature_list : list
        Liste des noms de features à analyser
    group_name : str
        Nom du groupe pour l'affichage (ex: "Groupe 1", "Groupe 2")
    suffix : str
        Suffixe des features (ex: "_g1", "_g2")
    save_plots : bool
        Si True, sauvegarde les graphiques

    Returns:
    --------
    dict : Résultats de l'analyse (matrices de corrélation, recommandations)
    """

    print(f"\n📊 ANALYSE DES CORRÉLATIONS - {group_name.upper()}")
    print("=" * 60)

    # ────────────────────────────────────────────────────────────────
    # 1) PRÉPARATION DES FEATURES
    # ────────────────────────────────────────────────────────────────

    # Ajouter le suffixe aux features sauf pour 'event'
    features_with_suffix = []
    for feature in feature_list:
        if feature == 'event':
            # 'event' reste sans suffixe (colonne commune G1/G2)
            features_with_suffix.append('event')
        elif suffix:
            # Autres features prennent le suffixe
            features_with_suffix.append(f"{feature}{suffix}")
        else:
            features_with_suffix.append(feature)

    print(f"\n🔍 FEATURES À ANALYSER ({len(features_with_suffix)}):")
    for i, feature in enumerate(features_with_suffix, 1):
        special_note = " (colonne commune G1/G2)" if feature == 'event' else ""
        print(f"   {i:2d}. {feature}{special_note}")

    # Vérifier que toutes les features existent
    missing_features = [f for f in features_with_suffix if f not in df.columns]
    if missing_features:
        print(f"\n⚠️  FEATURES MANQUANTES:")
        for f in missing_features:
            print(f"   • {f}")
        # Garder seulement les features existantes
        features_with_suffix = [f for f in features_with_suffix if f in df.columns]
        print(f"\n✅ FEATURES DISPONIBLES ({len(features_with_suffix)}):")
        for f in features_with_suffix:
            print(f"   • {f}")

    # ────────────────────────────────────────────────────────────────
    # 2) GESTION DES FEATURES CATÉGORIELLES
    # ────────────────────────────────────────────────────────────────

    df_work = df.copy()
    encoded_features = {}

    print(f"\n🔢 VÉRIFICATION DES TYPES DE DONNÉES:")

    for feature in features_with_suffix:
        if feature in df_work.columns:
            dtype = df_work[feature].dtype
            print(f"   {feature}: {dtype}")

            # Encoder les features catégorielles
            if dtype == 'object':
                print(f"     → Encodage nécessaire (catégoriel)")
                le = LabelEncoder()

                # Gérer les valeurs manquantes
                non_null_mask = df_work[feature].notna()
                if non_null_mask.sum() > 0:
                    encoded_col = f"{feature}_encoded"
                    df_work[encoded_col] = np.nan
                    df_work.loc[non_null_mask, encoded_col] = le.fit_transform(df_work.loc[non_null_mask, feature])

                    # Stocker le mapping
                    encoded_features[feature] = {
                        'encoded_name': encoded_col,
                        'mapping': dict(zip(le.classes_, le.transform(le.classes_)))
                    }

                    print(f"     → Mapping:")
                    for original, encoded in encoded_features[feature]['mapping'].items():
                        print(f"       {original} → {encoded}")
                else:
                    print(f"     → Toutes les valeurs sont NaN, feature ignorée")

    # Remplacer les features catégorielles par leur version encodée
    final_features = []
    for feature in features_with_suffix:
        if feature in encoded_features:
            final_features.append(encoded_features[feature]['encoded_name'])
        elif feature in df_work.columns:
            final_features.append(feature)

    print(f"\n📊 Features finales pour analyse: {len(final_features)}")

    # ────────────────────────────────────────────────────────────────
    # 3) CALCUL DES CORRÉLATIONS
    # ────────────────────────────────────────────────────────────────

    # Sous-ensemble des données
    df_corr = df_work[final_features].copy()

    # Statistiques des données manquantes
    initial_rows = len(df_corr)
    df_corr_clean = df_corr.dropna()
    final_rows = len(df_corr_clean)

    print(f"\n📈 PRÉPARATION DES DONNÉES:")
    print(f"   Lignes initiales: {initial_rows}")
    print(f"   Lignes après suppression NaN: {final_rows}")
    print(f"   Données utilisées: {final_rows / initial_rows:.1%}")

    if final_rows < 10:
        print("⚠️  ATTENTION: Très peu de données disponibles pour l'analyse")
        return None

    # Calcul des matrices de corrélation
    corr_pearson = df_corr_clean.corr(method='pearson')
    corr_spearman = df_corr_clean.corr(method='spearman')

    print(f"✅ Corrélations calculées")

    # ────────────────────────────────────────────────────────────────
    # 4) AFFICHAGE DES MATRICES
    # ────────────────────────────────────────────────────────────────

    print(f"\n📊 MATRICE DE CORRÉLATION PEARSON - {group_name}:")
    print("=" * 80)
    print(corr_pearson.round(3))

    print(f"\n📊 MATRICE DE CORRÉLATION SPEARMAN - {group_name}:")
    print("=" * 80)
    print(corr_spearman.round(3))

    # ────────────────────────────────────────────────────────────────
    # 5) ANALYSE DES CORRÉLATIONS SIGNIFICATIVES
    # ────────────────────────────────────────────────────────────────

    def extract_correlations(corr_matrix, method_name):
        """Extrait et classe les corrélations d'une matrice"""
        correlations = []
        n_features = len(corr_matrix)

        for i in range(n_features):
            for j in range(i + 1, n_features):
                feature1 = corr_matrix.index[i]
                feature2 = corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]
                correlations.append((feature1, feature2, corr_value))

        # Trier par valeur absolue
        correlations.sort(key=lambda x: abs(x[2]), reverse=True)

        print(f"\n{method_name.upper()} - {group_name}:")
        print("   Paires triées par corrélation (valeur absolue):")

        for feature1, feature2, corr_val in correlations:
            abs_corr = abs(corr_val)
            if abs_corr > 0.8:
                level = "🔴 TRÈS FORTE"
            elif abs_corr > 0.6:
                level = "🟠 FORTE"
            elif abs_corr > 0.4:
                level = "🟡 MODÉRÉE"
            elif abs_corr > 0.2:
                level = "🟢 FAIBLE"
            else:
                level = "⚪ TRÈS FAIBLE"

            # Nettoyer les noms pour l'affichage
            if feature1 == 'event':
                f1_clean = 'event'
            else:
                f1_clean = feature1.replace('_encoded', '').replace(suffix, '')

            if feature2 == 'event':
                f2_clean = 'event'
            else:
                f2_clean = feature2.replace('_encoded', '').replace(suffix, '')

            print(f"     {f1_clean:20s} ↔ {f2_clean:20s}: {corr_val:+.3f} {level}")

        return correlations

    correlations_pearson = extract_correlations(corr_pearson, "Pearson (linéaire)")
    correlations_spearman = extract_correlations(corr_spearman, "Spearman (rang)")

    # ────────────────────────────────────────────────────────────────
    # 6) VISUALISATIONS
    # ────────────────────────────────────────────────────────────────

    print(f"\n📊 GÉNÉRATION DES VISUALISATIONS POUR {group_name}:")

    # Nettoyer les noms pour l'affichage
    display_names = []
    for name in final_features:
        if name == 'event':
            clean_name = 'event'  # Event reste tel quel
        else:
            clean_name = name.replace('_encoded', '').replace(suffix, '')
        display_names.append(clean_name)

    # Créer les copies pour l'affichage
    corr_pearson_display = corr_pearson.copy()
    corr_spearman_display = corr_spearman.copy()
    corr_pearson_display.index = display_names
    corr_pearson_display.columns = display_names
    corr_spearman_display.index = display_names
    corr_spearman_display.columns = display_names

    # Créer la figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Analyse des Corrélations - {group_name}', fontsize=16, fontweight='bold')

    # 1) Heatmap Pearson
    sns.heatmap(corr_pearson_display,
                annot=True,
                cmap='RdBu_r',
                center=0,
                square=True,
                fmt='.2f',
                cbar_kws={'label': 'Corrélation'},
                ax=axes[0, 0],
                linewidths=0.5)
    axes[0, 0].set_title(f'Corrélation de Pearson - {group_name}', fontsize=12)

    # 2) Heatmap Spearman
    sns.heatmap(corr_spearman_display,
                annot=True,
                cmap='RdBu_r',
                center=0,
                square=True,
                fmt='.2f',
                cbar_kws={'label': 'Corrélation'},
                ax=axes[0, 1],
                linewidths=0.5)
    axes[0, 1].set_title(f'Corrélation de Spearman - {group_name}', fontsize=12)

    # 3) Différence Spearman - Pearson
    diff_matrix = corr_spearman_display - corr_pearson_display
    sns.heatmap(diff_matrix,
                annot=True,
                cmap='RdBu_r',
                center=0,
                square=True,
                fmt='.2f',
                cbar_kws={'label': 'Différence'},
                ax=axes[1, 0],
                linewidths=0.5)
    axes[1, 0].set_title(f'Différence (Spearman - Pearson) - {group_name}', fontsize=12)

    # 4) Scatter plot comparaison
    if len(correlations_pearson) > 0:
        pearson_values = [corr[2] for corr in correlations_pearson]
        spearman_values = [corr[2] for corr in correlations_spearman]

        axes[1, 1].scatter(pearson_values, spearman_values, alpha=0.7, s=60)
        axes[1, 1].plot([-1, 1], [-1, 1], 'r--', alpha=0.5, label='y = x')
        axes[1, 1].set_xlabel('Corrélation de Pearson')
        axes[1, 1].set_ylabel('Corrélation de Spearman')
        axes[1, 1].set_title(f'Pearson vs Spearman - {group_name}', fontsize=12)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()

        # Annotations pour les différences importantes
        for i, ((f1, f2, p_val), (_, _, s_val)) in enumerate(zip(correlations_pearson, correlations_spearman)):
            if abs(p_val - s_val) > 0.15:  # Différence significative
                f1_short = f1.replace('_encoded', '').replace(suffix, '')[:8]
                f2_short = f2.replace('_encoded', '').replace(suffix, '')[:8]
                axes[1, 1].annotate(f'{f1_short}↔{f2_short}',
                                    (p_val, s_val),
                                    xytext=(5, 5),
                                    textcoords='offset points',
                                    fontsize=8,
                                    alpha=0.7)

    plt.tight_layout()

    if save_plots:
        filename = f"correlations_{group_name.lower().replace(' ', '_')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"   📁 Graphiques sauvegardés: {filename}")

    plt.show()

    # ────────────────────────────────────────────────────────────────
    # 7) RECOMMANDATIONS
    # ────────────────────────────────────────────────────────────────

    print(f"\n💡 RECOMMANDATIONS POUR {group_name}:")
    print("=" * 60)

    # Corrélations problématiques
    high_corr_pairs = [(f1, f2, corr) for f1, f2, corr in correlations_pearson if abs(corr) > 0.7]
    moderate_corr_pairs = [(f1, f2, corr) for f1, f2, corr in correlations_pearson if 0.5 < abs(corr) <= 0.7]

    if high_corr_pairs:
        print(f"\n🔴 CORRÉLATIONS TRÈS FORTES (>0.7) - ATTENTION:")
        for f1, f2, corr in high_corr_pairs:
            if f1 == 'event':
                f1_clean = 'event'
            else:
                f1_clean = f1.replace('_encoded', '').replace(suffix, '')

            if f2 == 'event':
                f2_clean = 'event'
            else:
                f2_clean = f2.replace('_encoded', '').replace(suffix, '')

            print(f"   • {f1_clean} ↔ {f2_clean}: {corr:+.3f}")
        print(f"     → Risque de multicolinéarité")
        print(f"     → Considérer supprimer une feature ou créer un composite")

    if moderate_corr_pairs:
        print(f"\n🟡 CORRÉLATIONS MODÉRÉES (0.5-0.7) - SURVEILLER:")
        for f1, f2, corr in moderate_corr_pairs:
            if f1 == 'event':
                f1_clean = 'event'
            else:
                f1_clean = f1.replace('_encoded', '').replace(suffix, '')

            if f2 == 'event':
                f2_clean = 'event'
            else:
                f2_clean = f2.replace('_encoded', '').replace(suffix, '')

            print(f"   • {f1_clean} ↔ {f2_clean}: {corr:+.3f}")

    if not high_corr_pairs and not moderate_corr_pairs:
        print(f"\n✅ CORRÉLATIONS ACCEPTABLES")
        print("   → Toutes les corrélations < 0.5")
        print("   → Features suffisamment indépendantes pour le clustering")

    # Différences importantes Pearson vs Spearman
    nonlinear_pairs = [(f1, f2, p_corr, s_corr) for (f1, f2, p_corr), (_, _, s_corr)
                       in zip(correlations_pearson, correlations_spearman)
                       if abs(p_corr - s_corr) > 0.2]

    if nonlinear_pairs:
        print(f"\n🔍 RELATIONS NON-LINÉAIRES DÉTECTÉES:")
        for f1, f2, p_corr, s_corr in nonlinear_pairs:
            f1_clean = f1.replace('_encoded', '').replace(suffix, '')
            f2_clean = f2.replace('_encoded', '').replace(suffix, '')
            print(f"   • {f1_clean} ↔ {f2_clean}:")
            print(f"     Pearson: {p_corr:+.3f} | Spearman: {s_corr:+.3f}")
            print(f"     → Relation monotone mais non-linéaire")

    # Statistiques résumées
    avg_abs_pearson = np.mean([abs(corr) for _, _, corr in correlations_pearson]) if correlations_pearson else 0
    max_abs_pearson = max([abs(corr) for _, _, corr in correlations_pearson]) if correlations_pearson else 0

    print(f"\n📈 RÉSUMÉ - {group_name}:")
    print(f"   Features analysées: {len(final_features)}")
    print(f"   Corrélation absolue moyenne: {avg_abs_pearson:.3f}")
    print(f"   Corrélation absolue maximale: {max_abs_pearson:.3f}")

    if max_abs_pearson < 0.5:
        print("   ✅ Excellent pour clustering (faibles corrélations)")
    elif max_abs_pearson < 0.7:
        print("   🟡 Bon pour clustering (corrélations modérées)")
    else:
        print("   🔴 Attention multicolinéarité (fortes corrélations)")

    # Nettoyer les features temporaires
    for feature_info in encoded_features.values():
        if feature_info['encoded_name'] in df_work.columns:
            df_work.drop(feature_info['encoded_name'], axis=1, inplace=True)

    # Retourner les résultats
    return {
        'pearson': corr_pearson,
        'spearman': corr_spearman,
        'high_correlations': high_corr_pairs,
        'moderate_correlations': moderate_corr_pairs,
        'nonlinear_relations': nonlinear_pairs,
        'stats': {
            'avg_abs_correlation': avg_abs_pearson,
            'max_abs_correlation': max_abs_pearson,
            'n_features': len(final_features),
            'n_observations': final_rows
        }
    }
def create_enhanced_visualizations_with_sessions(results: Dict[str, Any], save_plots: bool = False,
                                                 output_dir: str = ".", groupe1=None,
    groupe2=None,period_atr_stat_session=None):
    """
    Création de 5 visualisations avec segmentation par sessions intraday
    Chaque figure contient 3 graphiques : Global, Session Groupe 1, Session Groupe 2
    """
    setup_plotting_style()

    if not results:
        raise ValueError("Aucune donnée à visualiser")

    valid_results = {k: v for k, v in results.items() if v is not None}
    n_datasets = len(valid_results)
    colors = plt.cm.Set1(np.linspace(0, 1, n_datasets))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # =============================================================================
    # FIGURE 1: Évolutions temporelles avec sessions (3 lignes × 3 graphiques)
    # =============================================================================
    fig1, axes1 = plt.subplots(3, 3, figsize=(24, 18))
    fig1.suptitle('📊 ÉVOLUTIONS TEMPORELLES PAR SESSIONS INTRADAY', fontsize=16, fontweight='bold')

    session_groups = [
        ("GLOBAL", None, "Toutes sessions"),
        (f"SESSIONS {groupe1}", groupe1, f"Sessions {groupe1}"),
        (f"SESSIONS {groupe2}", groupe2, f"Sessions {groupe2}")
    ]

    for row_idx, (group_title, session_filter, group_desc) in enumerate(session_groups):
        # Durée moyenne
        ax_dur = axes1[row_idx, 0]
        # Volume moyen
        ax_vol = axes1[row_idx, 1]
        # Nombre de bougies
        ax_count = axes1[row_idx, 2]

        for idx, (df_name, data) in enumerate(valid_results.items()):
            # Filtrer les données si nécessaire
            if session_filter is not None:
                # Utiliser les données filtrées par session
                if 'session_data_by_group' in data and str(session_filter) in data['session_data_by_group']:
                    session_stats = data['session_data_by_group'][str(session_filter)]['session_stats']
                else:
                    continue  # Pas de données pour ce groupe
            else:
                session_stats = data['session_stats']

            if len(session_stats) == 0:
                continue

            session_stats = session_stats.sort_values('session_date')
            color = colors[idx]

            # Durée
            ax_dur.plot(session_stats['session_date'], session_stats['duration_mean'],
                        marker='o', label=f'{df_name}', linewidth=2, markersize=4, color=color, alpha=0.8)
            if len(session_stats) > 1:
                x_numeric = range(len(session_stats))
                z = np.polyfit(x_numeric, session_stats['duration_mean'], 1)
                p = np.poly1d(z)
                ax_dur.plot(session_stats['session_date'], p(x_numeric), "--", color=color, alpha=0.6, linewidth=1.5)

            # Volume
            ax_vol.plot(session_stats['session_date'], session_stats['volume_mean'],
                        marker='s', label=f'{df_name}', linewidth=2, markersize=4, color=color, alpha=0.8)
            if len(session_stats) > 1:
                z_vol = np.polyfit(x_numeric, session_stats['volume_mean'], 1)
                p_vol = np.poly1d(z_vol)
                ax_vol.plot(session_stats['session_date'], p_vol(x_numeric), "--", color=color, alpha=0.6,
                            linewidth=1.5)

            # Nombre de bougies
            ax_count.plot(session_stats['session_date'], session_stats['candle_count'],
                          marker='^', label=f'{df_name}', linewidth=2, markersize=4, color=color, alpha=0.8)
            if len(session_stats) > 1:
                z_count = np.polyfit(x_numeric, session_stats['candle_count'], 1)
                p_count = np.poly1d(z_count)
                ax_count.plot(session_stats['session_date'], p_count(x_numeric), "--", color=color, alpha=0.6,
                              linewidth=1.5)

        # Configuration des axes
        ax_dur.set_title(f'📈 Durée Moyenne - {group_desc}', fontweight='bold')
        ax_dur.set_ylabel('Durée Moyenne (s)')
        ax_dur.legend()
        ax_dur.grid(True, alpha=0.3)
        ax_dur.tick_params(axis='x', rotation=45)

        ax_vol.set_title(f'📊 Volume Moyen - {group_desc}', fontweight='bold')
        ax_vol.set_ylabel('Volume Moyen')
        ax_vol.legend()
        ax_vol.grid(True, alpha=0.3)
        ax_vol.tick_params(axis='x', rotation=45)

        ax_count.set_title(f'📊 Nombre de Bougies - {group_desc}', fontweight='bold')
        ax_count.set_ylabel('Nombre de Bougies')
        ax_count.legend()
        ax_count.grid(True, alpha=0.3)
        ax_count.tick_params(axis='x', rotation=45)

        if row_idx == 2:  # Dernière ligne
            ax_dur.set_xlabel('Date de Session')
            ax_vol.set_xlabel('Date de Session')
            ax_count.set_xlabel('Date de Session')

    plt.tight_layout()
    if save_plots:
        plt.savefig(f"{output_dir}/trading_temporal_sessions_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.show()

    # =============================================================================
    # FIGURE 2: Distributions des DURÉES par sessions (3 lignes)
    # =============================================================================
    fig2, axes2 = plt.subplots(3, min(4, len(valid_results)), figsize=(6 * min(4, len(valid_results)), 18))
    if len(valid_results) == 1:
        axes2 = axes2.reshape(-1, 1)
    fig2.suptitle('⏱️ DISTRIBUTIONS DES DURÉES PAR SESSIONS INTRADAY', fontsize=16, fontweight='bold')

    for row_idx, (group_title, session_filter, group_desc) in enumerate(session_groups):
        for idx, (df_name, data) in enumerate(valid_results.items()):
            if idx < 4:  # Limite à 4 datasets
                ax = axes2[row_idx, idx] if len(valid_results) > 1 else axes2[row_idx]

                # Filtrer les données si nécessaire
                if session_filter is not None:
                    if 'session_data_by_group' in data and str(session_filter) in data['session_data_by_group']:
                        session_stats = data['session_data_by_group'][str(session_filter)]['session_stats']
                    else:
                        ax.text(0.5, 0.5, f'Pas de données\n{group_desc}', transform=ax.transAxes, ha='center',
                                va='center')
                        ax.set_title(f'⏱️ {df_name} - {group_desc}', fontweight='bold')
                        continue
                else:
                    session_stats = data['session_stats']

                if len(session_stats) == 0:
                    ax.text(0.5, 0.5, f'Pas de données\n{group_desc}', transform=ax.transAxes, ha='center', va='center')
                    ax.set_title(f'⏱️ {df_name} - {group_desc}', fontweight='bold')
                    continue

                duration_data = session_stats['duration_mean']

                ax.hist(duration_data, bins=min(15, len(session_stats)),
                        color=colors[idx], alpha=0.6, edgecolor='black', linewidth=0.5)

                p25 = np.percentile(duration_data, 25)
                p50 = np.percentile(duration_data, 50)
                p75 = np.percentile(duration_data, 75)

                ax.axvline(p25, color='red', linestyle='--', linewidth=2, alpha=0.8, label=f'P25: {p25:.1f}s')
                ax.axvline(p50, color='orange', linestyle='--', linewidth=2, alpha=0.8, label=f'P50: {p50:.1f}s')
                ax.axvline(p75, color='green', linestyle='--', linewidth=2, alpha=0.8, label=f'P75: {p75:.1f}s')

                ax.set_title(f'⏱️ {df_name} - {group_desc}', fontweight='bold')
                ax.set_xlabel('Durée Moyenne (s)')
                ax.set_ylabel('Fréquence')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)

        # Masquer les axes non utilisés
        if len(valid_results) < 4:
            for idx in range(len(valid_results), 4):
                if len(valid_results) > 1 and axes2.shape[1] > idx:
                    axes2[row_idx, idx].set_visible(False)

    plt.tight_layout()
    if save_plots:
        plt.savefig(f"{output_dir}/trading_durations_sessions_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.show()

    # =============================================================================
    # FIGURE 3: Distributions des VOLUMES par sessions (3 lignes)
    # =============================================================================
    fig3, axes3 = plt.subplots(3, min(4, len(valid_results)), figsize=(6 * min(4, len(valid_results)), 18))
    if len(valid_results) == 1:
        axes3 = axes3.reshape(-1, 1)
    fig3.suptitle('📊 DISTRIBUTIONS DES VOLUMES PAR SESSIONS INTRADAY', fontsize=16, fontweight='bold')

    for row_idx, (group_title, session_filter, group_desc) in enumerate(session_groups):
        for idx, (df_name, data) in enumerate(valid_results.items()):
            if idx < 4:
                ax = axes3[row_idx, idx] if len(valid_results) > 1 else axes3[row_idx]

                # Filtrer les données si nécessaire
                if session_filter is not None:
                    if 'session_data_by_group' in data and str(session_filter) in data['session_data_by_group']:
                        session_stats = data['session_data_by_group'][str(session_filter)]['session_stats']
                    else:
                        ax.text(0.5, 0.5, f'Pas de données\n{group_desc}', transform=ax.transAxes, ha='center',
                                va='center')
                        ax.set_title(f'📊 {df_name} - {group_desc}', fontweight='bold')
                        continue
                else:
                    session_stats = data['session_stats']

                if len(session_stats) == 0:
                    ax.text(0.5, 0.5, f'Pas de données\n{group_desc}', transform=ax.transAxes, ha='center', va='center')
                    ax.set_title(f'📊 {df_name} - {group_desc}', fontweight='bold')
                    continue

                volume_data = session_stats['volume_mean']

                ax.hist(volume_data, bins=min(15, len(session_stats)),
                        color=colors[idx], alpha=0.6, edgecolor='black', linewidth=0.5)

                v25 = np.percentile(volume_data, 25)
                v50 = np.percentile(volume_data, 50)
                v75 = np.percentile(volume_data, 75)

                ax.axvline(v25, color='red', linestyle='--', linewidth=2, alpha=0.8, label=f'P25: {v25:.1f}')
                ax.axvline(v50, color='orange', linestyle='--', linewidth=2, alpha=0.8, label=f'P50: {v50:.1f}')
                ax.axvline(v75, color='green', linestyle='--', linewidth=2, alpha=0.8, label=f'P75: {v75:.1f}')

                ax.set_title(f'📊 {df_name} - {group_desc}', fontweight='bold')
                ax.set_xlabel('Volume Moyen')
                ax.set_ylabel('Fréquence')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)

        # Masquer les axes non utilisés
        if len(valid_results) < 4:
            for idx in range(len(valid_results), 4):
                if len(valid_results) > 1 and axes3.shape[1] > idx:
                    axes3[row_idx, idx].set_visible(False)

    plt.tight_layout()
    if save_plots:
        plt.savefig(f"{output_dir}/trading_volumes_sessions_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.show()

    # =============================================================================
    # FIGURE 4: Distributions de l'ATR par sessions (3 lignes)
    # =============================================================================
    fig4, axes4 = plt.subplots(3, min(4, len(valid_results)), figsize=(6 * min(4, len(valid_results)), 18))
    if len(valid_results) == 1:
        axes4 = axes4.reshape(-1, 1)
    fig4.suptitle(f'📈 DISTRIBUTIONS ATR {period_atr_stat_session} PAR SESSIONS INTRADAY', fontsize=16,
                  fontweight='bold')

    for row_idx, (group_title, session_filter, group_desc) in enumerate(session_groups):
        for idx, (df_name, data) in enumerate(valid_results.items()):
            if idx < 4:
                ax = axes4[row_idx, idx] if len(valid_results) > 1 else axes4[row_idx]

                # Filtrer les données si nécessaire
                if session_filter is not None:
                    if 'session_data_by_group' in data and str(session_filter) in data['session_data_by_group']:
                        session_stats = data['session_data_by_group'][str(session_filter)]['session_stats']
                    else:
                        ax.text(0.5, 0.5, f'Pas de données\n{group_desc}', transform=ax.transAxes, ha='center',
                                va='center')
                        ax.set_title(f'📈 {df_name} - {group_desc}', fontweight='bold')
                        continue
                else:
                    session_stats = data['session_stats']

                if len(session_stats) == 0:
                    ax.text(0.5, 0.5, f'Pas de données\n{group_desc}', transform=ax.transAxes, ha='center', va='center')
                    ax.set_title(f'📈 {df_name} - {group_desc}', fontweight='bold')
                    continue

                atr_data = session_stats['atr_mean'].dropna()

                if len(atr_data) > 0:
                    ax.hist(atr_data, bins=min(15, len(atr_data)),
                            color=colors[idx], alpha=0.6, edgecolor='black', linewidth=0.5)

                    atr_p25 = np.percentile(atr_data, 25)
                    atr_p50 = np.percentile(atr_data, 50)
                    atr_p75 = np.percentile(atr_data, 75)

                    ax.axvline(atr_p25, color='red', linestyle='--', linewidth=2, alpha=0.8,
                               label=f'P25: {atr_p25:.3f}')
                    ax.axvline(atr_p50, color='orange', linestyle='--', linewidth=2, alpha=0.8,
                               label=f'P50: {atr_p50:.3f}')
                    ax.axvline(atr_p75, color='green', linestyle='--', linewidth=2, alpha=0.8,
                               label=f'P75: {atr_p75:.3f}')

                    ax.set_xlabel('ATR Moyen par Session')
                    ax.set_ylabel('Fréquence')
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, 'Pas de données\nATR valides', transform=ax.transAxes, ha='center', va='center')

                ax.set_title(f'📈 {df_name} - {group_desc}', fontweight='bold')

        # Masquer les axes non utilisés
        if len(valid_results) < 4:
            for idx in range(len(valid_results), 4):
                if len(valid_results) > 1 and axes4.shape[1] > idx:
                    axes4[row_idx, idx].set_visible(False)

    plt.tight_layout()
    if save_plots:
        plt.savefig(f"{output_dir}/trading_atr_sessions_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.show()

    # =============================================================================
    # FIGURE 5: Distributions des contrats extrêmes par sessions (3 lignes)
    # =============================================================================
    fig5, axes5 = plt.subplots(3, min(4, len(valid_results)), figsize=(6 * min(4, len(valid_results)), 18))
    if len(valid_results) == 1:
        axes5 = axes5.reshape(-1, 1)
    fig5.suptitle('🎯 DISTRIBUTIONS CONTRATS EXTRÊMES (SANS ZÉROS) PAR SESSIONS INTRADAY', fontsize=16,
                  fontweight='bold')

    session_groups = [
        ("GLOBAL", None, "Toutes sessions"),
        (f"SESSIONS {groupe1}", groupe1, f"Sessions {groupe1}"),
        (f"SESSIONS {groupe2}", groupe2, f"Sessions {groupe2}")
    ]

    for row_idx, (group_title, session_filter, group_desc) in enumerate(session_groups):
        for idx, (df_name, data) in enumerate(valid_results.items()):
            if idx < 4:
                ax = axes5[row_idx, idx] if len(valid_results) > 1 else axes5[row_idx]

                # Filtrer les données si nécessaire
                if session_filter is not None:
                    if 'session_data_by_group' in data and str(session_filter) in data['session_data_by_group']:
                        session_stats = data['session_data_by_group'][str(session_filter)]['session_stats']
                    else:
                        ax.text(0.5, 0.5, f'Pas de données\n{group_desc}', transform=ax.transAxes, ha='center',
                                va='center')
                        ax.set_title(f'🎯 {df_name} - {group_desc}', fontweight='bold')
                        continue
                else:
                    session_stats = data['session_stats']

                if len(session_stats) == 0:
                    ax.text(0.5, 0.5, f'Pas de données\n{group_desc}', transform=ax.transAxes, ha='center', va='center')
                    ax.set_title(f'🎯 {df_name} - {group_desc}', fontweight='bold')
                    continue

                # Utiliser les données des contrats extrêmes SANS zéros
                contracts_data = session_stats['extreme_without_zeros'].dropna()

                if len(contracts_data) > 0:
                    ax.hist(contracts_data, bins=min(15, len(contracts_data)),
                            color=colors[idx], alpha=0.6, edgecolor='black', linewidth=0.5)

                    cnz_p25 = np.percentile(contracts_data, 25)
                    cnz_p50 = np.percentile(contracts_data, 50)
                    cnz_p75 = np.percentile(contracts_data, 75)

                    ax.axvline(cnz_p25, color='red', linestyle='--', linewidth=2, alpha=0.8,
                               label=f'P25: {cnz_p25:.3f}')
                    ax.axvline(cnz_p50, color='orange', linestyle='--', linewidth=2, alpha=0.8,
                               label=f'P50: {cnz_p50:.3f}')
                    ax.axvline(cnz_p75, color='green', linestyle='--', linewidth=2, alpha=0.8,
                               label=f'P75: {cnz_p75:.3f}')

                    ax.set_xlabel('Moyenne Contrats Extrêmes (sans zéros)')
                    ax.set_ylabel('Fréquence')
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, 'Pas de données\ncontrats non-zéro', transform=ax.transAxes, ha='center',
                            va='center', fontsize=12)

                ax.set_title(f'🎯 {df_name} - {group_desc}', fontweight='bold')

        # Masquer les axes non utilisés pour cette ligne
        if len(valid_results) < 4:
            for idx in range(len(valid_results), 4):
                if len(valid_results) > 1 and axes5.shape[1] > idx:
                    axes5[row_idx, idx].set_visible(False)

    plt.tight_layout()
    if save_plots:
        plt.savefig(f"{output_dir}/trading_extreme_contracts_sessions_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.show()
 # =============================================================================
    # FIGURE 6: Distributions des volumes above par tick par sessions (3 lignes)
    # =============================================================================
    fig6, axes6 = plt.subplots(3, min(4, len(valid_results)), figsize=(6 * min(4, len(valid_results)), 18))
    if len(valid_results) == 1:
        axes6 = axes6.reshape(-1, 1)
    fig6.suptitle('📊 DISTRIBUTIONS VOLUME ABOVE PAR TICK PAR SESSIONS INTRADAY', fontsize=16,
                  fontweight='bold')

    session_groups = [
        ("GLOBAL", None, "Toutes sessions"),
        (f"SESSIONS {groupe1}", groupe1, f"Sessions {groupe1}"),
        (f"SESSIONS {groupe2}", groupe2, f"Sessions {groupe2}")
    ]

    for row_idx, (group_title, session_filter, group_desc) in enumerate(session_groups):
        for idx, (df_name, data) in enumerate(valid_results.items()):
            if idx < 4:
                ax = axes6[row_idx, idx] if len(valid_results) > 1 else axes6[row_idx]

                # Filtrer les données si nécessaire
                if session_filter is not None:
                    if 'session_data_by_group' in data and str(session_filter) in data['session_data_by_group']:
                        session_stats = data['session_data_by_group'][str(session_filter)]['session_stats']
                    else:
                        ax.text(0.5, 0.5, f'Pas de données\n{group_desc}', transform=ax.transAxes, ha='center',
                                va='center')
                        ax.set_title(f'📊 {df_name} - {group_desc}', fontweight='bold')
                        continue
                else:
                    session_stats = data['session_stats']

                if len(session_stats) == 0:
                    ax.text(0.5, 0.5, f'Pas de données\n{group_desc}', transform=ax.transAxes, ha='center', va='center')
                    ax.set_title(f'📊 {df_name} - {group_desc}', fontweight='bold')
                    continue

                # Utiliser les données des volumes above par tick
                volume_above_data = session_stats['volume_above_per_tick_mean'].dropna()

                if len(volume_above_data) > 0:
                    ax.hist(volume_above_data, bins=min(15, len(volume_above_data)),
                            color=colors[idx], alpha=0.6, edgecolor='black', linewidth=0.5)

                    va_p25 = np.percentile(volume_above_data, 25)
                    va_p50 = np.percentile(volume_above_data, 50)
                    va_p75 = np.percentile(volume_above_data, 75)

                    ax.axvline(va_p25, color='red', linestyle='--', linewidth=2, alpha=0.8,
                               label=f'P25: {va_p25:.3f}')
                    ax.axvline(va_p50, color='orange', linestyle='--', linewidth=2, alpha=0.8,
                               label=f'P50: {va_p50:.3f}')
                    ax.axvline(va_p75, color='green', linestyle='--', linewidth=2, alpha=0.8,
                               label=f'P75: {va_p75:.3f}')

                    ax.set_xlabel('Volume Above Moyen par Tick')
                    ax.set_ylabel('Fréquence')
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, 'Pas de données\nvolume above valides', transform=ax.transAxes, ha='center',
                            va='center', fontsize=12)

                ax.set_title(f'📊 {df_name} - {group_desc}', fontweight='bold')

        # Masquer les axes non utilisés pour cette ligne
        if len(valid_results) < 4:
            for idx in range(len(valid_results), 4):
                if len(valid_results) > 1 and axes6.shape[1] > idx:
                    axes6[row_idx, idx].set_visible(False)

    plt.tight_layout()
    if save_plots:
        plt.savefig(f"{output_dir}/trading_volume_above_per_tick_sessions_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.show()

def filter_data_by_session_group(df: pd.DataFrame, session_group: list, df_name: str) -> pd.DataFrame:
    """
    Filtre les données par groupe de sessions intraday

    Parameters:
    -----------
    df : DataFrame
        Données à filtrer
    session_group : list
        Liste des indices de sessions à conserver
    df_name : str
        Nom du dataset pour les logs

    Returns:
    --------
    DataFrame : Données filtrées
    """
    if 'deltaCustomSessionIndex' not in df.columns:
        logger.warning(f"⚠️ {df_name}: Colonne 'deltaCustomSessionIndex' non trouvée, retour données complètes")
        return df

    # Filtrer par groupe de sessions
    filtered_df = df[df['deltaCustomSessionIndex'].isin(session_group)].copy()

    logger.info(f"📊 {df_name}: {len(filtered_df)}/{len(df)} lignes conservées pour sessions {session_group}")
    return filtered_df


def run_enhanced_trading_analysis_with_sessions(
    df_init_features_train=None,
    df_init_features_test=None,
    df_init_features_val1=None,
    df_init_features_val=None,
    groupe1=None,
    groupe2=None,xtickReversalTickPrice=None,period_atr_stat_session=None) -> Dict[str, Any]:
    """
    Version améliorée avec analyse par sessions intraday et volumes above par tick
    """
    logger.info("🚀 Démarrage de l'analyse trading avec sessions intraday")

    # Filtrer les DataFrames non-None
    dataframes = {
        'df_init_features_train': df_init_features_train,
        'df_init_features_test': df_init_features_test,
        'df_init_features_val1': df_init_features_val1,
        'df_init_features_val': df_init_features_val
    }
    valid_dataframes = {k: v for k, v in dataframes.items() if v is not None}

    if not valid_dataframes:
        raise ValueError("Aucun DataFrame fourni pour l'analyse")

    results = {}
    required_columns = ['timeStampOpeningConvertedtoDate', 'session_id', 'candleDuration', 'sc_volume']

    logger.info(f"Analyse de {len(valid_dataframes)} dataset(s): {list(valid_dataframes.keys())}")

    try:
        # Validation et analyse pour chaque dataset
        for df_name, df in valid_dataframes.items():
            logger.info(f"🔍 Traitement de {df_name}...")

            # Validation
            validation = validate_dataframe_structure(df, df_name, required_columns)
            if not validation['is_valid']:
                logger.error(f"❌ Validation échouée pour {df_name}")
                continue

            if validation['warnings']:
                for warning in validation['warnings']:
                    logger.warning(f"⚠️ {df_name}: {warning}")

            # Calcul des métriques globales (données complètes)
            session_stats_global = calculate_session_metrics_enhanced(df, df_name,xtickReversalTickPrice=xtickReversalTickPrice,period_atr_stat_session=period_atr_stat_session)

            # Calcul des métriques par groupe de sessions
            session_data_by_group = {}

            for group_name, session_indices in [("GROUPE_1", groupe1), ("GROUPE_2", groupe2)]:
                logger.info(f"🔍 Analyse {group_name} (sessions {session_indices}) pour {df_name}")

                # Filtrage des données par groupe de sessions
                df_filtered = filter_data_by_session_group(df, session_indices, df_name)

                if len(df_filtered) > 0:
                    try:
                        session_stats_filtered = calculate_session_metrics_enhanced(df_filtered,
                                                                                    f"{df_name}_{group_name}",xtickReversalTickPrice=xtickReversalTickPrice,period_atr_stat_session=period_atr_stat_session)

                        # Calcul des statistiques pour ce groupe
                        volume_stats = {
                            'avg_volume_per_session': session_stats_filtered['volume_mean'].mean(),
                            'total_volume': session_stats_filtered['volume_sum'].sum(),
                            'duration_volume_correlation': session_stats_filtered['duration_mean'].corr(
                                session_stats_filtered['volume_mean'])
                        }

                        # Statistiques ATR
                        atr_valid = session_stats_filtered['atr_mean'].dropna()
                        atr_stats = {
                            'atr_mean_per_session': session_stats_filtered['atr_mean'],
                            'atr_overall_mean': atr_valid.mean() if len(atr_valid) > 0 else np.nan,
                            'atr_std': atr_valid.std() if len(atr_valid) > 0 else np.nan,
                            'sessions_with_atr': len(atr_valid),
                            'atr_duration_correlation': session_stats_filtered['atr_mean'].corr(
                                session_stats_filtered['duration_mean'])
                        }

                        # Statistiques contrats extrêmes
                        extreme_with_zeros = session_stats_filtered['extreme_with_zeros'].dropna()
                        extreme_without_zeros = session_stats_filtered['extreme_without_zeros'].dropna()

                        extreme_contracts_stats = {
                            'extreme_contracts_with_zeros': session_stats_filtered['extreme_with_zeros'],
                            'extreme_contracts_without_zeros': session_stats_filtered['extreme_without_zeros'],
                            'extreme_with_zeros_mean': extreme_with_zeros.mean() if len(
                                extreme_with_zeros) > 0 else np.nan,
                            'extreme_without_zeros_mean': extreme_without_zeros.mean() if len(
                                extreme_without_zeros) > 0 else np.nan,
                            'extreme_ratio_mean': session_stats_filtered['extreme_ratio'].mean(),
                            'sessions_with_extreme_contracts': len(extreme_without_zeros)
                        }

                        # NOUVEAU: Statistiques volumes above par tick
                        volume_above_per_tick_valid = session_stats_filtered['volume_above_per_tick_mean'].dropna()

                        volume_above_per_tick_stats = {
                            'volume_above_per_tick_mean_series': session_stats_filtered['volume_above_per_tick_mean'],
                            'volume_above_per_tick_overall_mean': volume_above_per_tick_valid.mean() if len(
                                volume_above_per_tick_valid) > 0 else np.nan,
                            'volume_above_ratio_mean': session_stats_filtered['volume_above_ratio'].mean(),
                            'sessions_with_volume_above': len(volume_above_per_tick_valid)
                        }

                        session_data_by_group[str(session_indices)] = {
                            'session_stats': session_stats_filtered,
                            'total_sessions': session_stats_filtered['session_id'].nunique(),
                            'total_candles': session_stats_filtered['candle_count'].sum(),
                            'avg_duration_overall': session_stats_filtered['duration_mean'].mean(),
                            'volume_stats': volume_stats,
                            'atr_stats': atr_stats,
                            'extreme_contracts_stats': extreme_contracts_stats,
                            'volume_above_per_tick_stats': volume_above_per_tick_stats  # NOUVEAU
                        }

                        logger.info(
                            f"✅ {group_name}: {session_data_by_group[str(session_indices)]['total_sessions']} sessions analysées")
                        logger.info(
                            f"   📊 Volumes above par tick: {volume_above_per_tick_stats['sessions_with_volume_above']} sessions")

                    except Exception as e:
                        logger.warning(f"⚠️ Erreur lors de l'analyse du {group_name} pour {df_name}: {e}")
                        session_data_by_group[str(session_indices)] = None
                else:
                    logger.warning(f"⚠️ Aucune donnée trouvée pour {group_name} dans {df_name}")
                    session_data_by_group[str(session_indices)] = None

            # Calcul des statistiques globales avec volumes above par tick
            volume_stats_global = {
                'avg_volume_per_session': session_stats_global['volume_mean'].mean(),
                'total_volume': session_stats_global['volume_sum'].sum(),
                'duration_volume_correlation': session_stats_global['duration_mean'].corr(
                    session_stats_global['volume_mean'])
            }

            # Statistiques ATR globales
            atr_valid_global = session_stats_global['atr_mean'].dropna()
            atr_stats_global = {
                'atr_mean_per_session': session_stats_global['atr_mean'],
                'atr_overall_mean': atr_valid_global.mean() if len(atr_valid_global) > 0 else np.nan,
                'atr_std': atr_valid_global.std() if len(atr_valid_global) > 0 else np.nan,
                'atr_min': atr_valid_global.min() if len(atr_valid_global) > 0 else np.nan,
                'atr_max': atr_valid_global.max() if len(atr_valid_global) > 0 else np.nan,
                'sessions_with_atr': len(atr_valid_global),
                'atr_duration_correlation': session_stats_global['atr_mean'].corr(session_stats_global['duration_mean'])
            }

            # Statistiques contrats extrêmes globales
            extreme_with_zeros_global = session_stats_global['extreme_with_zeros'].dropna()
            extreme_without_zeros_global = session_stats_global['extreme_without_zeros'].dropna()

            extreme_contracts_stats_global = {
                'extreme_contracts_with_zeros': session_stats_global['extreme_with_zeros'],
                'extreme_contracts_without_zeros': session_stats_global['extreme_without_zeros'],
                'extreme_with_zeros_mean': extreme_with_zeros_global.mean() if len(
                    extreme_with_zeros_global) > 0 else np.nan,
                'extreme_without_zeros_mean': extreme_without_zeros_global.mean() if len(
                    extreme_without_zeros_global) > 0 else np.nan,
                'extreme_ratio_mean': session_stats_global['extreme_ratio'].mean(),
                'extreme_count_total': session_stats_global['extreme_count_nonzero'].sum(),
                'sessions_with_extreme_contracts': len(extreme_without_zeros_global),
                'extreme_with_zeros_duration_corr': session_stats_global['extreme_with_zeros'].corr(
                    session_stats_global['duration_mean']),
                'extreme_without_zeros_duration_corr': session_stats_global['extreme_without_zeros'].corr(
                    session_stats_global['duration_mean']),
                'extreme_atr_correlation': session_stats_global['extreme_without_zeros'].corr(
                    session_stats_global['atr_mean'])
            }

            # NOUVEAU: Statistiques volumes above par tick globales
            volume_above_per_tick_valid_global = session_stats_global['volume_above_per_tick_mean'].dropna()

            volume_above_per_tick_stats_global = {
                'volume_above_per_tick_mean_series': session_stats_global['volume_above_per_tick_mean'],
                'volume_above_per_tick_overall_mean': volume_above_per_tick_valid_global.mean() if len(
                    volume_above_per_tick_valid_global) > 0 else np.nan,
                'volume_above_per_tick_std': volume_above_per_tick_valid_global.std() if len(
                    volume_above_per_tick_valid_global) > 0 else np.nan,
                'volume_above_ratio_mean': session_stats_global['volume_above_ratio'].mean(),
                'volume_above_count_total': session_stats_global['volume_above_count'].sum(),
                'sessions_with_volume_above': len(volume_above_per_tick_valid_global),
                'volume_above_duration_correlation': session_stats_global['volume_above_per_tick_mean'].corr(
                    session_stats_global['duration_mean']),
                'volume_above_atr_correlation': session_stats_global['volume_above_per_tick_mean'].corr(
                    session_stats_global['atr_mean']),
                'volume_above_extreme_correlation': session_stats_global['volume_above_per_tick_mean'].corr(
                    session_stats_global['extreme_without_zeros'])
            }

            # Stockage des résultats enrichis
            results[df_name] = {
                'session_stats': session_stats_global,
                'total_sessions': session_stats_global['session_id'].nunique(),
                'total_candles': session_stats_global['candle_count'].sum(),
                'avg_duration_overall': session_stats_global['duration_mean'].mean(),
                'avg_candles_per_session': session_stats_global['candle_count'].mean(),
                'date_range': (session_stats_global['session_date'].min().date(),
                               session_stats_global['session_date'].max().date()),
                'volume_stats': volume_stats_global,
                'atr_stats': atr_stats_global,
                'extreme_contracts_stats': extreme_contracts_stats_global,
                'volume_above_per_tick_stats': volume_above_per_tick_stats_global,  # NOUVEAU
                'session_data_by_group': session_data_by_group,
                'validation': validation
            }

            # Logging enrichi
            logger.info(f"✅ {df_name}: {results[df_name]['total_sessions']} sessions globales analysées")
            logger.info(f"   📊 ATR: {atr_stats_global['sessions_with_atr']} sessions avec ATR valide")
            logger.info(
                f"   🎯 Contrats extrêmes: {extreme_contracts_stats_global['sessions_with_extreme_contracts']} sessions")
            logger.info(
                f"   📊 Volumes above par tick: {volume_above_per_tick_stats_global['sessions_with_volume_above']} sessions")

        # Visualisations avec sessions (maintenant 6 figures)
        if results:
            create_enhanced_visualizations_with_sessions(results,groupe1=groupe1,
    groupe2=groupe2,period_atr_stat_session=period_atr_stat_session)
            logger.info("📊 Visualisations avec sessions intraday générées avec succès (6 figures)")

        # Affichage du résumé enrichi avec sessions
        print_enhanced_summary_statistics_with_sessions(results,groupe1=groupe1,groupe2=groupe2,xtickReversalTickPrice=xtickReversalTickPrice)

        logger.info("🎉 Analyse avec sessions intraday terminée avec succès!")
        return results

    except Exception as e:
        logger.error(f"❌ Erreur lors de l'analyse: {e}")
        raise



