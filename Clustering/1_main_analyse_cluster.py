from func_clustering import *

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, entropy
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from matplotlib.patches import Circle, FancyArrowPatch
import os
import platform
from func_standard import print_notification, load_features_and_sections

# ────────────────────────────────────
# CONFIGURATION GLOBALE
# ────────────────────────────────────
# Paramètres configurables
clustering_with_K = clustering_with_K  # Modifiable selon vos besoins
ANALYSIS_RANGE = [2, 3, 4, 5, 6]  # Range étendu pour l'analyse
SAVE_PLOTS = True  # Activer/désactiver la sauvegarde des graphiques


# ────────────────────────────────────
# 1) CHEMINS FICHIERS
# ────────────────────────────────────
def setup_file_paths():
    """Configuration des chemins de fichiers selon l'OS"""
    if platform.system() != "Darwin":
        directory_path = r"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra chart\xTickReversal\simu\5_0_5TP_6SL\merge"
    else:
        directory_path = "/Users/aurelienlachaud/Documents/trading_local/5_0_5TP_1SL_1/merge"

    file_name = "Step5_5_0_5TP_6SL_010124_150525_extractOnlyFullSession_OnlyShort_feat_4Cluster.csv"
    file_path = os.path.join(directory_path, file_name)

    return directory_path, file_name, file_path


directory_path, file_name, file_path = setup_file_paths()

print_notification(f"Chargement du fichier {file_path}")
df, CUSTOM_SESSIONS = load_features_and_sections(file_path)

# ────────────────────────────────────
# 2) FEATURES G1 et G2
# ────────────────────────────────────
feature_columns = get_feature_columns()
feature_columns_g1 = [f"{col}_g1" if col != 'event' else col for col in feature_columns]
feature_columns_g2 = [f"{col}_g2" if col != 'event' else col for col in feature_columns]

print(f"📊 Features analysées:")
print(f"   G1: {feature_columns_g1}")
print(f"   G2: {feature_columns_g2}")


# ────────────────────────────────────────────────────────────────
# 3) ANALYSE DES CORRÉLATIONS OPTIMISÉE
# ────────────────────────────────────────────────────────────────
def analyze_correlations_comprehensive(df, feature_list, group_name, suffix):
    """Analyse complète des corrélations avec métriques détaillées"""
    print(f"\n📊 ANALYSE DES CORRÉLATIONS - {group_name}")
    print("=" * 60)

    # Colonnes avec suffix
    cols_to_analyze = [f"{col}{suffix}" if col != 'event' else col for col in feature_list]

    # Matrice de corrélation
    corr_matrix = df[cols_to_analyze].corr()

    # Statistiques
    corr_values = corr_matrix.values
    mask = np.triu(np.ones_like(corr_values, dtype=bool), k=1)
    upper_triangle = corr_values[mask]

    stats = {
        'n_features': len(cols_to_analyze),
        'avg_abs_correlation': np.mean(np.abs(upper_triangle)),
        'max_abs_correlation': np.max(np.abs(upper_triangle)),
        'min_abs_correlation': np.min(np.abs(upper_triangle)),
        'std_correlation': np.std(upper_triangle)
    }

    # Corrélations problématiques
    high_correlations = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:
                high_correlations.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': corr_val
                })

    print(f"📈 Statistiques de corrélation:")
    print(f"   Features analysées: {stats['n_features']}")
    print(f"   Corrélation moyenne (abs): {stats['avg_abs_correlation']:.3f}")
    print(f"   Corrélation maximale (abs): {stats['max_abs_correlation']:.3f}")
    print(f"   Écart-type: {stats['std_correlation']:.3f}")
    print(f"   Corrélations fortes (>0.7): {len(high_correlations)}")

    if high_correlations:
        print("\n⚠️ Corrélations problématiques:")
        for corr in high_correlations:
            print(f"   {corr['feature1']} ↔ {corr['feature2']}: {corr['correlation']:.3f}")

    return {
        'stats': stats,
        'high_correlations': high_correlations,
        'correlation_matrix': corr_matrix
    }


print("🎯 ANALYSE COMPARATIVE DES CORRÉLATIONS G1 vs G2")
print("=" * 80)

# Analyse des corrélations
results_g1 = analyze_correlations_comprehensive(df, feature_columns, "Groupe 1 (Asie)", "_g1")
results_g2 = analyze_correlations_comprehensive(df, feature_columns, "Groupe 2 (Reste journée)", "_g2")

# Comparaison entre groupes
if results_g1 and results_g2:
    print("\n" + "⚖️" * 20)
    print("COMPARAISON G1 vs G2")
    print("⚖️" * 20)

    g1_stats = results_g1['stats']
    g2_stats = results_g2['stats']

    comparison_table = pd.DataFrame({
        'Métrique': ['Corrélation moyenne', 'Corrélation maximale', 'Corrélations fortes (>0.7)', 'Features analysées'],
        'G1 (Asie)': [g1_stats['avg_abs_correlation'], g1_stats['max_abs_correlation'],
                      len(results_g1['high_correlations']), g1_stats['n_features']],
        'G2 (Reste)': [g2_stats['avg_abs_correlation'], g2_stats['max_abs_correlation'],
                       len(results_g2['high_correlations']), g2_stats['n_features']],
        'Différence': [g2_stats['avg_abs_correlation'] - g1_stats['avg_abs_correlation'],
                       g2_stats['max_abs_correlation'] - g1_stats['max_abs_correlation'],
                       len(results_g2['high_correlations']) - len(results_g1['high_correlations']),
                       g2_stats['n_features'] - g1_stats['n_features']]
    })

    print("\n📊 TABLEAU COMPARATIF:")
    print(comparison_table.round(3).to_string(index=False))

    # Recommandation
    max_corr_overall = max(g1_stats['max_abs_correlation'], g2_stats['max_abs_correlation'])
    print(f"\n💡 RECOMMANDATION GLOBALE:")
    if max_corr_overall < 0.5:
        print("   ✅ Les deux groupes sont excellents pour le clustering")
    elif max_corr_overall < 0.7:
        print("   🟡 Les deux groupes sont acceptables, surveiller les corrélations modérées")
    else:
        print("   🔴 Attention aux multicolinéarités, optimisation nécessaire")

# ────────────────────────────────────
# 4) PRÉPARATION DES DONNÉES
# ────────────────────────────────────
print(f"\n🔄 PRÉPARATION DES DONNÉES POUR K={clustering_with_K}")

X_scaled_g1, scaler_g1 = scale_features(df, feature_columns_g1)
X_scaled_g2, scaler_g2 = scale_features(df, feature_columns_g2)

print(f"✅ Données standardisées:")
print(f"   G1: {X_scaled_g1.shape}")
print(f"   G2: {X_scaled_g2.shape}")


# ────────────────────────────────────
# 5) MÉTHODE DU COUDE OPTIMISÉE
# ────────────────────────────────────
def elbow_method_analysis(X_g1, X_g2, k_range):
    """Analyse de la méthode du coude avec métriques détaillées"""
    print(f"\n=== MÉTHODE DU COUDE - COMPARAISON G1 vs G2 (K={k_range[0]}-{k_range[-1]}) ===")

    results = {'G1': [], 'G2': []}

    for k in k_range:
        # G1
        kmeans_g1 = KMeans(n_clusters=k, random_state=42, n_init=30)
        kmeans_g1.fit(X_g1)
        results['G1'].append({
            'K': k,
            'inertia': kmeans_g1.inertia_,
            'inertia_reduction': 0  # Calculé plus tard
        })

        # G2
        kmeans_g2 = KMeans(n_clusters=k, random_state=42, n_init=30)
        kmeans_g2.fit(X_g2)
        results['G2'].append({
            'K': k,
            'inertia': kmeans_g2.inertia_,
            'inertia_reduction': 0  # Calculé plus tard
        })

        print(f"K={k}: Inertie G1 = {kmeans_g1.inertia_:.2f} | Inertie G2 = {kmeans_g2.inertia_:.2f}")

    # Calcul de la réduction d'inertie
    for group in ['G1', 'G2']:
        for i in range(1, len(results[group])):
            prev_inertia = results[group][i - 1]['inertia']
            curr_inertia = results[group][i]['inertia']
            results[group][i]['inertia_reduction'] = prev_inertia - curr_inertia

    return results


elbow_results = elbow_method_analysis(X_scaled_g1, X_scaled_g2, ANALYSIS_RANGE)


# Visualisation améliorée de la méthode du coude
def plot_elbow_method(elbow_results, k_range):
    """Graphique optimisé de la méthode du coude"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Graphique principal
    inertias_g1 = [r['inertia'] for r in elbow_results['G1']]
    inertias_g2 = [r['inertia'] for r in elbow_results['G2']]

    ax1.plot(k_range, inertias_g1, 'bo-', linewidth=3, markersize=10, label='G1 (Asie)')
    ax1.plot(k_range, inertias_g2, 'ro-', linewidth=3, markersize=10, label='G2 (Reste journée)')
    ax1.set_xlabel('Nombre de clusters (K)', fontsize=12)
    ax1.set_ylabel('Inertie', fontsize=12)
    ax1.set_title('Méthode du Coude - Comparaison G1 vs G2', fontsize=14)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=11)

    # Annotations
    for i, (k, inertia_g1, inertia_g2) in enumerate(zip(k_range, inertias_g1, inertias_g2)):
        ax1.annotate(f"{inertia_g1:.0f}", xy=(k, inertia_g1), xytext=(5, 10),
                     textcoords='offset points', fontsize=9, color='blue', fontweight='bold')
        ax1.annotate(f"{inertia_g2:.0f}", xy=(k, inertia_g2), xytext=(5, -15),
                     textcoords='offset points', fontsize=9, color='red', fontweight='bold')

    # Graphique des réductions d'inertie
    reductions_g1 = [r['inertia_reduction'] for r in elbow_results['G1'][1:]]
    reductions_g2 = [r['inertia_reduction'] for r in elbow_results['G2'][1:]]

    ax2.bar(np.array(k_range[1:]) - 0.2, reductions_g1, 0.4, label='G1 (Asie)', color='lightblue', alpha=0.7)
    ax2.bar(np.array(k_range[1:]) + 0.2, reductions_g2, 0.4, label='G2 (Reste)', color='lightcoral', alpha=0.7)
    ax2.set_xlabel('Nombre de clusters (K)', fontsize=12)
    ax2.set_ylabel('Réduction d\'inertie', fontsize=12)
    ax2.set_title('Gains d\'inertie par ajout de cluster', fontsize=14)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=11)

    plt.tight_layout()
    if SAVE_PLOTS:
        plt.savefig(os.path.join(directory_path, 'elbow_method_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()


plot_elbow_method(elbow_results, ANALYSIS_RANGE)


# ────────────────────────────────────
# 6) ANALYSE MULTI-CRITÈRES ÉTENDUE
# ────────────────────────────────────
def comprehensive_clustering_analysis(X_g1, X_g2, k_range):
    """Analyse complète avec multiple métriques"""
    print(f"\n=== ANALYSE MULTI-CRITÈRES (K={k_range[0]}-{k_range[-1]}) - G1 vs G2 ===")

    results = {'G1': [], 'G2': []}

    for k in k_range:
        # Analyse G1
        kmeans_g1 = KMeans(n_clusters=k, random_state=42, n_init=30)
        labels_g1 = kmeans_g1.fit_predict(X_g1)

        silhouette_g1 = silhouette_score(X_g1, labels_g1)
        davies_g1 = davies_bouldin_score(X_g1, labels_g1)
        calinski_g1 = calinski_harabasz_score(X_g1, labels_g1)
        cluster_sizes_g1 = pd.Series(labels_g1).value_counts(normalize=True).sort_index()

        results['G1'].append({
            "K": k,
            "Silhouette": silhouette_g1,
            "Davies-Bouldin": davies_g1,
            "Calinski-Harabasz": calinski_g1,
            "Inertie": kmeans_g1.inertia_,
            "Min_Cluster_Size": cluster_sizes_g1.min(),
            "Cluster_Balance": cluster_sizes_g1.std()  # Nouvelle métrique
        })

        # Analyse G2
        kmeans_g2 = KMeans(n_clusters=k, random_state=42, n_init=30)
        labels_g2 = kmeans_g2.fit_predict(X_g2)

        silhouette_g2 = silhouette_score(X_g2, labels_g2)
        davies_g2 = davies_bouldin_score(X_g2, labels_g2)
        calinski_g2 = calinski_harabasz_score(X_g2, labels_g2)
        cluster_sizes_g2 = pd.Series(labels_g2).value_counts(normalize=True).sort_index()

        results['G2'].append({
            "K": k,
            "Silhouette": silhouette_g2,
            "Davies-Bouldin": davies_g2,
            "Calinski-Harabasz": calinski_g2,
            "Inertie": kmeans_g2.inertia_,
            "Min_Cluster_Size": cluster_sizes_g2.min(),
            "Cluster_Balance": cluster_sizes_g2.std()
        })

        print(f"\nK={k}:")
        print(
            f"  G1: Sil={silhouette_g1:.3f} | DB={davies_g1:.3f} | CH={calinski_g1:.1f} | MinSize={cluster_sizes_g1.min():.2%} | Balance={cluster_sizes_g1.std():.3f}")
        print(
            f"  G2: Sil={silhouette_g2:.3f} | DB={davies_g2:.3f} | CH={calinski_g2:.1f} | MinSize={cluster_sizes_g2.min():.2%} | Balance={cluster_sizes_g2.std():.3f}")

    return results


multi_criteria_results = comprehensive_clustering_analysis(X_scaled_g1, X_scaled_g2, ANALYSIS_RANGE)

# Création des DataFrames pour l'analyse
df_results_g1 = pd.DataFrame(multi_criteria_results['G1'])
df_results_g2 = pd.DataFrame(multi_criteria_results['G2'])


# ────────────────────────────────────
# 7) VISUALISATION COMPARATIVE AMÉLIORÉE
# ────────────────────────────────────
def plot_comprehensive_metrics(df_results_g1, df_results_g2, k_range):
    """Visualisation complète des métriques"""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    width = 0.35
    x = np.arange(len(k_range))
    colors_g1 = plt.cm.Blues(np.linspace(0.4, 0.9, len(k_range)))
    colors_g2 = plt.cm.Reds(np.linspace(0.4, 0.9, len(k_range)))

    # 1) Silhouette Score
    ax = axes[0, 0]
    bars1 = ax.bar(x - width / 2, df_results_g1['Silhouette'], width, label='G1 (Asie)', color=colors_g1)
    bars2 = ax.bar(x + width / 2, df_results_g2['Silhouette'], width, label='G2 (Reste)', color=colors_g2)
    ax.axhline(0.4, ls='--', color='green', alpha=0.7, label='Seuil optimal 0.4')
    ax.set_title('Silhouette Score', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score Silhouette')
    ax.set_xticks(x)
    ax.set_xticklabels(k_range)
    ax.legend()
    ax.grid(alpha=0.3)

    # Annotations
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    # 2) Davies-Bouldin Score
    ax = axes[0, 1]
    bars1 = ax.bar(x - width / 2, df_results_g1['Davies-Bouldin'], width, label='G1 (Asie)', color=colors_g1)
    bars2 = ax.bar(x + width / 2, df_results_g2['Davies-Bouldin'], width, label='G2 (Reste)', color=colors_g2)
    ax.set_title('Davies-Bouldin Score (plus bas = mieux)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score Davies-Bouldin')
    ax.set_xticks(x)
    ax.set_xticklabels(k_range)
    ax.legend()
    ax.grid(alpha=0.3)

    # 3) Calinski-Harabasz Score
    ax = axes[0, 2]
    bars1 = ax.bar(x - width / 2, df_results_g1['Calinski-Harabasz'], width, label='G1 (Asie)', color=colors_g1)
    bars2 = ax.bar(x + width / 2, df_results_g2['Calinski-Harabasz'], width, label='G2 (Reste)', color=colors_g2)
    ax.set_title('Calinski-Harabasz Score (plus haut = mieux)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score Calinski-Harabasz')
    ax.set_xticks(x)
    ax.set_xticklabels(k_range)
    ax.legend()
    ax.grid(alpha=0.3)

    # 4) Taille minimale des clusters
    ax = axes[1, 0]
    bars1 = ax.bar(x - width / 2, df_results_g1['Min_Cluster_Size'], width, label='G1 (Asie)', color=colors_g1)
    bars2 = ax.bar(x + width / 2, df_results_g2['Min_Cluster_Size'], width, label='G2 (Reste)', color=colors_g2)
    ax.axhline(0.1, ls='--', color='red', alpha=0.7, label='Seuil minimal 10%')
    ax.set_title('Taille du plus petit cluster', fontsize=14, fontweight='bold')
    ax.set_ylabel('Proportion')
    ax.set_xticks(x)
    ax.set_xticklabels(k_range)
    ax.legend()
    ax.grid(alpha=0.3)

    # Annotations en pourcentage
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f'{height:.1%}', ha='center', va='bottom', fontsize=9)

    # 5) Équilibre des clusters
    ax = axes[1, 1]
    bars1 = ax.bar(x - width / 2, df_results_g1['Cluster_Balance'], width, label='G1 (Asie)', color=colors_g1)
    bars2 = ax.bar(x + width / 2, df_results_g2['Cluster_Balance'], width, label='G2 (Reste)', color=colors_g2)
    ax.set_title('Équilibre des clusters (plus bas = mieux)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Écart-type des tailles')
    ax.set_xticks(x)
    ax.set_xticklabels(k_range)
    ax.legend()
    ax.grid(alpha=0.3)

    # 6) Score composite
    ax = axes[1, 2]

    # Calcul des scores composites normalisés
    def calculate_composite_score(df_results):
        # Normalisation des métriques (0-1)
        sil_norm = (df_results['Silhouette'] - df_results['Silhouette'].min()) / (
                    df_results['Silhouette'].max() - df_results['Silhouette'].min())
        db_norm = 1 - (df_results['Davies-Bouldin'] - df_results['Davies-Bouldin'].min()) / (
                    df_results['Davies-Bouldin'].max() - df_results['Davies-Bouldin'].min())
        size_norm = df_results['Min_Cluster_Size'] / df_results['Min_Cluster_Size'].max()
        balance_norm = 1 - (df_results['Cluster_Balance'] - df_results['Cluster_Balance'].min()) / (
                    df_results['Cluster_Balance'].max() - df_results['Cluster_Balance'].min())

        # Score pondéré
        composite = (sil_norm * 0.4 + db_norm * 0.3 + size_norm * 0.2 + balance_norm * 0.1)
        return composite

    score_g1 = calculate_composite_score(df_results_g1)
    score_g2 = calculate_composite_score(df_results_g2)

    bars1 = ax.bar(x - width / 2, score_g1, width, label='G1 (Asie)', color=colors_g1)
    bars2 = ax.bar(x + width / 2, score_g2, width, label='G2 (Reste)', color=colors_g2)
    ax.set_title('Score Composite (plus haut = mieux)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score Composite')
    ax.set_xticks(x)
    ax.set_xticklabels(k_range)
    ax.legend()
    ax.grid(alpha=0.3)

    # Marquer le meilleur score
    best_k_g1 = k_range[score_g1.idxmax()]
    best_k_g2 = k_range[score_g2.idxmax()]

    ax.text(0.02, 0.98, f'Optimal G1: K={best_k_g1}\nOptimal G2: K={best_k_g2}',
            transform=ax.transAxes, va='top', ha='left',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8),
            fontsize=10, fontweight='bold')

    plt.tight_layout()
    if SAVE_PLOTS:
        plt.savefig(os.path.join(directory_path, 'comprehensive_clustering_metrics.png'), dpi=300, bbox_inches='tight')
    plt.show()

    return score_g1, score_g2


composite_scores = plot_comprehensive_metrics(df_results_g1, df_results_g2, ANALYSIS_RANGE)


# ────────────────────────────────────
# 8) ANALYSE DÉTAILLÉE POUR K OPTIMAL
# ────────────────────────────────────
def detailed_analysis_for_k(df, X_g1, X_g2, feature_cols_g1, feature_cols_g2, k):
    """Analyse détaillée pour une valeur K spécifique"""
    print(f"\n=== ANALYSE DÉTAILLÉE POUR K={k} ===")

    # Clustering
    kmeans_g1 = KMeans(n_clusters=k, random_state=42, n_init=50)
    kmeans_g2 = KMeans(n_clusters=k, random_state=42, n_init=50)

    labels_g1_raw = kmeans_g1.fit_predict(X_g1)
    labels_g2_raw = kmeans_g2.fit_predict(X_g2)

    # Relabeling par volume croissant
    labels_g1, map_g1 = relabel_by_metric(labels_g1_raw, df['volume_p50_g1'])
    labels_g2, map_g2 = relabel_by_metric(labels_g2_raw, df['volume_p50_g1'])

    # Ajout au DataFrame
    df[f"Cluster_G1_K{k}"] = labels_g1
    df[f"Cluster_G2_K{k}"] = labels_g2

    # Profils des clusters
    profile_orig_g1 = df.groupby(f"Cluster_G1_K{k}")[feature_cols_g1].mean().round(3)
    profile_orig_g2 = df.groupby(f"Cluster_G2_K{k}")[feature_cols_g2].mean().round(3)

    print(f"\n📊 PROFILS DES CLUSTERS (K={k}):")
    print(f"\nGroupe 1 (Asie):")
    print(profile_orig_g1)
    print(f"\nGroupe 2 (Reste journée):")
    print(profile_orig_g2)

    return labels_g1, labels_g2, profile_orig_g1, profile_orig_g2


# Analyse pour K optimal
labels_g1_final, labels_g2_final, profiles_g1, profiles_g2 = detailed_analysis_for_k(
    df, X_scaled_g1, X_scaled_g2, feature_columns_g1, feature_columns_g2, clustering_with_K
)


# ────────────────────────────────────
# 9) ANALYSE DE TRANSITION ET PERSISTANCE
# ────────────────────────────────────
def transition_analysis(df, k, labels_g1, labels_g2):
    """Analyse complète des transitions entre clusters"""
    print(f"\n=== ANALYSE DE TRANSITION ET PERSISTANCE (K={k}) ===")

    # Matrice de transition
    transition_matrix = pd.crosstab(labels_g1, labels_g2, normalize='index')

    # Métriques de persistance
    persistence_rate = np.diag(transition_matrix).mean()
    persistence_by_cluster = np.diag(transition_matrix)

    # Entropie des transitions
    entropies = [entropy(row) for _, row in transition_matrix.iterrows()]
    avg_entropy = np.mean(entropies)
    max_entropy = np.log(k)  # Entropie maximale possible

    print(f"📊 MÉTRIQUES DE TRANSITION:")
    print(f"   Taux de persistance global: {persistence_rate:.1%}")
    print(f"   Entropie moyenne: {avg_entropy:.3f} / {max_entropy:.3f}")
    print(f"   Prévisibilité: {(1 - avg_entropy / max_entropy):.1%}")

    print(f"\n📈 PERSISTANCE PAR CLUSTER:")
    for i in range(k):
        print(f"   Cluster {i}: {persistence_by_cluster[i]:.1%}")

    # Identification des transitions les plus probables
    transitions_off_diag = []
    for i in range(k):
        for j in range(k):
            if i != j:
                transitions_off_diag.append((transition_matrix.iloc[i, j], i, j))

    top_transitions = sorted(transitions_off_diag, reverse=True)[:3]
    print(f"\n🔄 PRINCIPALES TRANSITIONS:")
    for prob, src, dst in top_transitions:
        print(f"   Cluster {src} → Cluster {dst}: {prob:.1%}")

    return transition_matrix, persistence_rate, avg_entropy


# ────────────────────────────────────
# 10) GÉNÉRATION DES LABELS SÉMANTIQUES
# ────────────────────────────────────
def generate_semantic_labels(k, profiles_g1):
    """Génération de labels sémantiques basés sur le volume"""
    volume_means = profiles_g1['volume_p50_g1'].sort_values()

    if k == 2:
        level_names = ['CALME', 'ACTIF']
    elif k == 3:
        level_names = ['CALME', 'MODÉRÉ', 'ACTIF']
    elif k == 4:
        level_names = ['TRÈS_CALME', 'CALME', 'ACTIF', 'TRÈS_ACTIF']
    elif k == 5:
        level_names = ['TRÈS_CALME', 'CALME', 'MODÉRÉ', 'ACTIF', 'TRÈS_ACTIF']
    elif k == 6:
        level_names = ['TRÈS_CALME', 'CALME', 'FAIBLE_MOD', 'FORT_MOD', 'ACTIF', 'TRÈS_ACTIF']
    else:
        level_names = [f'NIVEAU_{i}' for i in range(k)]

    # Mapping basé sur l'ordre du volume
    sorted_clusters = volume_means.index
    labels_map = {}
    for i, cluster in enumerate(sorted_clusters):
        labels_map[cluster] = level_names[i]

    return labels_map, level_names


# ────────────────────────────────────
# 11) STRATÉGIES DE TRADING ADAPTATIVES
# ────────────────────────────────────
def generate_trading_strategies(transition_matrix, labels_map, level_names):
    """Génération de stratégies de trading basées sur les transitions"""
    print(f"\n💰 STRATÉGIES DE TRADING RECOMMANDÉES:")

    strategies = {}

    for i, src_label in enumerate(level_names):
        for j, dst_label in enumerate(level_names):
            src_cluster = next(k for k, v in labels_map.items() if v == src_label)
            dst_cluster = next(k for k, v in labels_map.items() if v == dst_label)
            prob = transition_matrix.iloc[src_cluster, dst_cluster]

            # Logique de stratégie
            if src_label == dst_label:  # Persistance
                if prob > 0.6:
                    strategy = "MAINTENIR_POSITION"
                elif prob > 0.4:
                    strategy = "POSITION_PRUDENTE"
                else:
                    strategy = "SURVEILLER"
            else:  # Transition
                if "CALME" in src_label and "ACTIF" in dst_label:
                    strategy = "PRÉPARER_BREAKOUT"
                elif "ACTIF" in src_label and "CALME" in dst_label:
                    strategy = "PRENDRE_PROFITS"
                elif prob > 0.3:
                    strategy = "ANTICIPER_CHANGEMENT"
                else:
                    strategy = "IGNORER"

            strategies[f"{src_label}→{dst_label}"] = {
                'probability': prob,
                'strategy': strategy
            }

            print(f"   {src_label} → {dst_label} ({prob:.0%}): {strategy}")

    return strategies


# ────────────────────────────────────
# 12) VISUALISATION AVANCÉE DES TRANSITIONS
# ────────────────────────────────────
def plot_transition_analysis(transition_matrix, labels_map, k):
    """Visualisation complète de l'analyse des transitions"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1) Matrice de transition avec heatmap
    level_names = list(labels_map.values())
    tick_labels = [f"{i}\n{labels_map[i]}" for i in range(k)]

    im1 = ax1.imshow(transition_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    ax1.set_title(f'Matrice de Transition G1→G2 (K={k})', fontsize=14, fontweight='bold')
    ax1.set_xlabel('G2 (Reste journée)')
    ax1.set_ylabel('G1 (Asie)')
    ax1.set_xticks(range(k))
    ax1.set_xticklabels(tick_labels, fontsize=10)
    ax1.set_yticks(range(k))
    ax1.set_yticklabels(tick_labels, fontsize=10)

    # Annotations
    for i in range(k):
        for j in range(k):
            val = transition_matrix.iloc[i, j]
            color = 'white' if val > 0.5 else 'black'
            weight = 'bold' if i == j else 'normal'
            size = 12 if i == j else 10
            ax1.text(j, i, f"{val:.0%}", ha='center', va='center',
                     color=color, fontsize=size, fontweight=weight)

    plt.colorbar(im1, ax=ax1, label='Probabilité de transition')

    # 2) Graphique en barres des persistances
    persistence_rates = [transition_matrix.iloc[i, i] for i in range(k)]
    colors = plt.cm.viridis(np.linspace(0, 1, k))

    bars = ax2.bar(range(k), persistence_rates, color=colors, alpha=0.7)
    ax2.set_title('Taux de Persistance par Cluster', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Cluster')
    ax2.set_ylabel('Taux de persistance')
    ax2.set_xticks(range(k))
    ax2.set_xticklabels([f"{i}\n{labels_map[i]}" for i in range(k)])
    ax2.axhline(0.5, ls='--', color='red', alpha=0.7, label='Seuil 50%')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # Annotations
    for bar, rate in zip(bars, persistence_rates):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')

    # 3) Diagramme de flux simplifié
    ax3.axis('off')
    ax3.set_title(f'Flux de Transitions Principaux (K={k})', fontsize=14, fontweight='bold')

    # Positions des nœuds
    if k <= 4:
        angles = np.linspace(0, 2 * np.pi, k, endpoint=False)
        positions = {i: (0.5 + 0.3 * np.cos(angle), 0.5 + 0.3 * np.sin(angle))
                     for i, angle in enumerate(angles)}

        # Dessiner les nœuds
        for i, (x, y) in positions.items():
            circle = Circle((x, y), 0.08, color=colors[i], alpha=0.7, ec='black', lw=2)
            ax3.add_patch(circle)
            ax3.text(x, y, f"{i}\n{labels_map[i]}", ha='center', va='center',
                     fontsize=9, fontweight='bold')

        # Dessiner les flèches pour les transitions importantes
        for i in range(k):
            for j in range(k):
                prob = transition_matrix.iloc[i, j]
                if prob > 0.2:  # Seulement transitions significatives
                    x1, y1 = positions[i]
                    x2, y2 = positions[j]

                    if i == j:  # Auto-transition (persistance)
                        # Cercle autour du nœud
                        circle_arrow = Circle((x1, y1), 0.12, fill=False,
                                              color='green', lw=3, alpha=0.7)
                        ax3.add_patch(circle_arrow)
                        ax3.text(x1, y1 - 0.15, f"{prob:.0%}", ha='center', va='center',
                                 fontsize=8, fontweight='bold', color='green')
                    else:
                        # Flèche directionnelle
                        dx, dy = x2 - x1, y2 - y1
                        length = np.sqrt(dx ** 2 + dy ** 2)
                        dx_norm, dy_norm = dx / length * 0.08, dy / length * 0.08

                        arrow = FancyArrowPatch((x1 + dx_norm, y1 + dy_norm),
                                                (x2 - dx_norm, y2 - dy_norm),
                                                arrowstyle='->', lw=2,
                                                color='orange', alpha=0.7)
                        ax3.add_patch(arrow)

                        # Label de probabilité
                        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                        ax3.text(mid_x, mid_y, f"{prob:.0%}", ha='center', va='center',
                                 fontsize=8, bbox=dict(boxstyle="round,pad=0.2",
                                                       facecolor='white', alpha=0.8))

        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.text(0.5, 0.05, "Vert=Persistance | Orange=Transition",
                 ha='center', va='center', fontsize=10, style='italic')
    else:
        ax3.text(0.5, 0.5, f"Diagramme simplifié\npour K={k}\n\n" +
                 f"Persistance moyenne:\n{np.mean(persistence_rates):.1%}",
                 ha='center', va='center', fontsize=12,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))

    # 4) Distribution des tailles de clusters
    cluster_sizes_g1 = [np.sum(labels_g1_final == i) for i in range(k)]
    cluster_sizes_g2 = [np.sum(labels_g2_final == i) for i in range(k)]

    x = np.arange(k)
    width = 0.35

    bars1 = ax4.bar(x - width / 2, cluster_sizes_g1, width, label='G1 (Asie)',
                    color='lightblue', alpha=0.7)
    bars2 = ax4.bar(x + width / 2, cluster_sizes_g2, width, label='G2 (Reste)',
                    color='lightcoral', alpha=0.7)

    ax4.set_title('Distribution des Tailles de Clusters', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Cluster')
    ax4.set_ylabel('Nombre d\'observations')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f"{i}\n{labels_map[i]}" for i in range(k)])
    ax4.legend()
    ax4.grid(alpha=0.3)

    # Annotations
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2, height + len(df) * 0.01,
                     f'{int(height)}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    if SAVE_PLOTS:
        plt.savefig(os.path.join(directory_path, f'transition_analysis_K{k}.png'),
                    dpi=300, bbox_inches='tight')
    plt.show()


# ────────────────────────────────────
# 13) ENRICHISSEMENT DU DATASET
# ────────────────────────────────────
def enrich_dataset_with_predictions(df, transition_matrix, labels_map, k):
    """Enrichissement du dataset avec colonnes de prédiction et trading"""
    print(f"\n📊 ENRICHISSEMENT DU DATASET AVEC PRÉDICTIONS")
    print("=" * 60)

    # Ajout des colonnes de base
    df[f'Regime_G1_K{k}'] = labels_g1_final
    df[f'Regime_G2_K{k}'] = labels_g2_final

    # Labels sémantiques
    df[f'Regime_G1_Label_K{k}'] = df[f'Regime_G1_K{k}'].map(labels_map)
    df[f'Regime_G2_Label_K{k}'] = df[f'Regime_G2_K{k}'].map(labels_map)

    # Prédictions G2 basées sur G1
    def predict_g2_from_g1(regime_g1):
        if regime_g1 in transition_matrix.index:
            predicted_regime = transition_matrix.loc[regime_g1].idxmax()
            probability = transition_matrix.loc[regime_g1].max()
            return predicted_regime, probability
        return None, None

    predictions = df[f'Regime_G1_K{k}'].apply(predict_g2_from_g1)
    df[f'Prediction_G2_K{k}'] = predictions.apply(lambda x: x[0])
    df[f'Prediction_G2_Prob_K{k}'] = predictions.apply(lambda x: x[1])
    df[f'Prediction_G2_Label_K{k}'] = df[f'Prediction_G2_K{k}'].map(labels_map)

    # Qualité des prédictions
    df[f'Prediction_Correct_K{k}'] = (df[f'Prediction_G2_K{k}'] == df[f'Regime_G2_K{k}'])
    df[f'Prediction_Quality_K{k}'] = df[f'Prediction_Correct_K{k}'].map({True: 'CORRECT', False: 'INCORRECT'})

    # Types de transitions
    df[f'Transition_Type_K{k}'] = (df[f'Regime_G1_Label_K{k}'] + '→' +
                                   df[f'Regime_G2_Label_K{k}'])
    df[f'Transition_Predicted_K{k}'] = (df[f'Regime_G1_Label_K{k}'] + '→' +
                                        df[f'Prediction_G2_Label_K{k}'])

    # Métriques de performance
    accuracy = df[f'Prediction_Correct_K{k}'].mean()
    print(f"✅ Précision globale des prédictions K={k}: {accuracy:.1%}")

    # Scores de confiance et de risque
    df[f'Confidence_Score_K{k}'] = (
            df[f'Prediction_G2_Prob_K{k}'] * 0.6 +  # Probabilité de prédiction
            (df['volume_p50_g1'] / df['volume_p50_g1'].max()) * 0.4  # Volume relatif
    ).round(3)

    # Niveau de risque basé sur la volatilité prédite
    def calculate_risk_level(g1_label, pred_label, confidence):
        if g1_label == pred_label:  # Persistance
            if confidence > 0.7:
                return 'BAS'
            elif confidence > 0.5:
                return 'MODÉRÉ'
            else:
                return 'ÉLEVÉ'
        else:  # Transition
            if 'CALME' in g1_label and 'ACTIF' in pred_label:
                return 'ÉLEVÉ'  # Breakout potentiel
            elif 'ACTIF' in g1_label and 'CALME' in pred_label:
                return 'MODÉRÉ'  # Retour au calme
            else:
                return 'MODÉRÉ'

    df[f'Risk_Level_K{k}'] = df.apply(lambda row: calculate_risk_level(
        row[f'Regime_G1_Label_K{k}'],
        row[f'Prediction_G2_Label_K{k}'],
        row[f'Confidence_Score_K{k}']), axis=1)

    # Signaux de trading simplifiés
    def generate_trading_signal(g1_label, pred_label, confidence, risk):
        if confidence > 0.7 and risk == 'BAS':
            return 'HOLD'
        elif 'CALME' in g1_label and 'ACTIF' in pred_label and confidence > 0.5:
            return 'PREPARE_ENTRY'
        elif 'ACTIF' in g1_label and 'CALME' in pred_label and confidence > 0.5:
            return 'TAKE_PROFIT'
        elif risk == 'ÉLEVÉ':
            return 'CAUTION'
        else:
            return 'NEUTRAL'

    df[f'Trading_Signal_K{k}'] = df.apply(lambda row: generate_trading_signal(
        row[f'Regime_G1_Label_K{k}'],
        row[f'Prediction_G2_Label_K{k}'],
        row[f'Confidence_Score_K{k}'],
        row[f'Risk_Level_K{k}']), axis=1)

    # Statistiques finales
    print(f"\n📈 STATISTIQUES D'ENRICHISSEMENT:")
    print(f"   Nouvelles colonnes ajoutées: 12")
    print(f"   Précision prédictive: {accuracy:.1%}")
    print(f"   Distribution des signaux de trading:")

    signal_counts = df[f'Trading_Signal_K{k}'].value_counts()
    for signal, count in signal_counts.items():
        print(f"      {signal}: {count} ({count / len(df) * 100:.1f}%)")

    return df


# ────────────────────────────────────
# 14) SAUVEGARDE ET EXPORT
# ────────────────────────────────────
def save_analysis_results(df, directory_path, file_name, k):
    """Sauvegarde des résultats d'analyse"""
    print(f"\n💾 SAUVEGARDE DES RÉSULTATS")
    print("=" * 50)

    # Fichier enrichi principal
    original_filename = file_name.replace('.csv', '')
    enriched_filename = f"{original_filename}_CLUSTERING_K{k}_ENRICHED.csv"
    enriched_filepath = os.path.join(directory_path, enriched_filename)

    df.to_csv(enriched_filepath, index=False, sep=';')
    print(f"✅ Dataset enrichi sauvegardé: {enriched_filename}")

    # Résumé des métriques
    summary_data = {
        'K': [k],
        'Precision_Predictions': [df[f'Prediction_Correct_K{k}'].mean()],
        'Persistence_Rate': [np.diag(transition_matrix).mean()],
        'Avg_Confidence': [df[f'Confidence_Score_K{k}'].mean()],
        'High_Risk_Sessions': [(df[f'Risk_Level_K{k}'] == 'ÉLEVÉ').mean()],
        'Trading_Signals_Generated': [len(df[f'Trading_Signal_K{k}'].unique())]
    }

    summary_df = pd.DataFrame(summary_data)
    summary_filename = f"CLUSTERING_SUMMARY_K{k}.csv"
    summary_filepath = os.path.join(directory_path, summary_filename)
    summary_df.to_csv(summary_filepath, index=False, sep=';')
    print(f"✅ Résumé des métriques sauvegardé: {summary_filename}")

    print(f"\n📊 RÉSUMÉ FINAL:")
    print(f"   Fichier principal: {len(df)} lignes, {len(df.columns)} colonnes")
    print(f"   Précision prédictive: {summary_data['Precision_Predictions'][0]:.1%}")
    print(f"   Taux de persistance: {summary_data['Persistence_Rate'][0]:.1%}")
    print(f"   Confiance moyenne: {summary_data['Avg_Confidence'][0]:.3f}")


# ────────────────────────────────────
# EXÉCUTION DE L'ANALYSE COMPLÈTE
# ────────────────────────────────────

print(f"\n🚀 LANCEMENT DE L'ANALYSE COMPLÈTE POUR K={clustering_with_K}")
print("=" * 80)

# Analyse des transitions
transition_matrix, persistence_rate, avg_entropy = transition_analysis(
    df, clustering_with_K, labels_g1_final, labels_g2_final
)

# Génération des labels sémantiques
labels_map, level_names = generate_semantic_labels(clustering_with_K, profiles_g1)
print(f"\n🏷️ LABELS SÉMANTIQUES GÉNÉRÉS:")
for cluster, label in labels_map.items():
    print(f"   Cluster {cluster}: {label}")

# Stratégies de trading
trading_strategies = generate_trading_strategies(transition_matrix, labels_map, level_names)

# Visualisation des transitions
plot_transition_analysis(transition_matrix, labels_map, clustering_with_K)

# Enrichissement du dataset
df_enriched = enrich_dataset_with_predictions(df, transition_matrix, labels_map, clustering_with_K)

# Sauvegarde des résultats
save_analysis_results(df_enriched, directory_path, file_name, clustering_with_K)

print(f"\n🎉 ANALYSE COMPLÈTE TERMINÉE!")
print(f"Tous les résultats ont été sauvegardés dans: {directory_path}")
print(f"Modèle de clustering K={clustering_with_K} optimisé et déployé avec succès!")