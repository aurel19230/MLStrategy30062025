import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from statsmodels.stats.power import TTestIndPower
from sklearn.feature_selection import f_classif  # Test F (ANOVA)
from definition import load_data
# Définir l'option pour afficher toutes les colonnes
pd.set_option('display.max_columns', None)
from standard_stat_sc import *

# Si vous voulez également afficher toutes les lignes
pd.set_option('display.max_rows', None)

# Pour afficher plus de caractères dans chaque colonne (éviter la troncation des valeurs)
pd.set_option('display.width', 1000)  # Ajustez ce nombre selon vos besoins


# Chemin vers ton fichier
file_name = "Step5_version2_170924_110325_bugFixTradeResult1_extractOnlyFullSession_OnlyShort_feat_winsorized.csv"
directory_path = "C:\\Users\\aulac\\OneDrive\\Documents\\Trading\\VisualStudioProject\\Sierra chart\\xTickReversal\\simu\\5_0_5TP_1SL\\version2\merge"
file_path = os.path.join(directory_path, file_name)

# Charger les données
df = load_data(file_path)



# ---- FONCTION D'ANALYSE AVEC TEST F (ANOVA) ---- #
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import TTestIndPower
from sklearn.feature_selection import f_classif
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import TTestIndPower
import time
from datetime import timedelta

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import TTestIndPower
from stats_sc.standard_stat_sc import calculate_statistical_power,calculate_statistical_power_job


# ---- CHARGEMENT DES FEATURES À ANALYSER ---- #

if True:
    feature_list = [
        # 'ratio_volRevMove_volImpulsMove',
        # 'ratio_deltaZone1_volZone1',
        # 'ratio_deltaExtrem_volExtrem',
        # 'ratio_volRevMoveZone1_volImpulsMoveExtrem_XRevZone',
         'stoch_overbought',
         'mfi_short_divergence',
         'bull_imbalance_low_1',
        'bull_imbalance_low_2',
    ]
else:
    # Colonnes à exclure explicitement (variables cibles, pnl, dates...)
    excluded_columns = [
        'class_binaire',
        'timestamp',
        'SessionStartEnd',
        'trade_pnl_theoric', 'trade_pnl', 'trade_category','sl_pnl',
        'tp3_pnl', 'tp2_pnl', 'tp1_pnl', 'tp1_pnl_theoric',
        'timeStampOpening', 'date', 'close', 'high', 'low', 'open'
    ]

    # ✅ Sélection correcte des colonnes numériques en excluant explicitement celles mentionnées
    feature_list = df.select_dtypes(include=[np.number]).columns.difference(excluded_columns).tolist()

# Vérification
print(f"📌 Nombre de features sélectionnées : {len(feature_list)}")
print(f"🔹 Liste des features :\n{feature_list}")

# Filtrage des données (classe binaire doit être 0 ou 1)
df_filtered = df[df['class_binaire'].isin([0, 1])]

# ---- EXPLICATION DES RÉSULTATS ---- #
explanation = """
🔍 **Explication des variables du tableau de résultats :**

- **Feature** : Nom de la feature analysée.
- **Sample_Size** : Nombre d'observations utilisées après filtrage des NaN.
- **Effect_Size (Cohen's d)** : Mesure de la séparation entre les deux classes.
  - **> 0.8** : Effet fort ✅
  - **0.5 - 0.8** : Effet moyen ⚠️
  - **< 0.5** : Effet faible ❌

- **P-Value** : Probabilité d'observer la relation par hasard.
  - **< 0.01** : Très significatif ✅✅
  - **0.01 - 0.05** : Significatif ✅
  - **0.05 - 0.10** : Marginalement significatif ⚠️
  - **> 0.10** : Non significatif ❌

- **Fisher_Score (ANOVA F-test)** : Mesure la force discriminante de la feature.
  - **> 20** : Exceptionnellement puissant ✅✅
  - **10 - 20** : Très intéressant ✅
  - **5 - 10** : Modérément intéressant ⚠️
  - **1 - 5** : Faiblement intéressant ❌
  - Dans le trading, même des scores modestes peuvent avoir une valeur s'ils sont stables dans le temps.

- **Power_Analytical** : Puissance statistique basée sur une formule analytique.
- **Power_MonteCarlo** : Puissance statistique estimée via simulations.
- **Required_N** : Nombre d'observations nécessaires pour atteindre **Puissance = 0.8**.
- **Power_Sufficient** : L'échantillon actuel est-il suffisant pour garantir la fiabilité de l'effet observé ?

🎯 **Interprétation des seuils de puissance statistique** :
- ✅ **Puissance ≥ 0.8** : La feature a une distinction nette entre classes. Résultat très fiable.
- ⚠️ **0.6 ≤ Puissance < 0.8** : Impact potentiel, mais fiabilité modérée. À considérer dans un ensemble de signaux.
- ❌ **Puissance < 0.6** : Fiabilité insuffisante. Risque élevé que la relation observée soit due au hasard.

📈 **Considérations spécifiques pour le trading** :
- Une feature avec un Fisher Score modeste mais stable sur différentes périodes peut être plus précieuse qu'une feature avec un score élevé mais instable.
- Surveillez la complémentarité des features : des variables individuellement modestes peuvent créer un signal puissant en combinaison.
- Vérifiez toujours la robustesse après prise en compte des frais de transaction.
- Pour les stratégies haute fréquence, même des effets faibles peuvent être exploitables si la significativité statistique est forte.
"""

print(explanation)
df_filtered = df_filtered.replace([np.inf, -np.inf], np.nan)  # Remplace Inf par NaN
df_filtered = df_filtered.dropna()  # Supprime toutes les lignes contenant NaN
# ---- ANALYSE DE PUISSANCE ---- #
print("\n🔍 **Analyse de puissance statistique pour les features :**")
# Préparation des données
# Préparation des données
X = df_filtered[feature_list].copy()  # DataFrame contenant uniquement les features sélectionnées
y = df_filtered['class_binaire'].copy()  # Série de la cible

from sklearn.preprocessing import StandardScaler
import pandas as pd

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)  # X devient un numpy.ndarray
#
# # Reconvertir en DataFrame en gardant les noms des colonnes :
# X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
X_scaled=X
# Maintenant, la corrélation fonctionnera :
results = correlation_matrices(X_scaled, y)

#
# # Pour accéder aux résultats:
# pearson_matrix = results['pearson_matrix']
# spearman_target = results['spearman_target']

# Appel de la fonction avec X pré-filtré

power_results = calculate_statistical_power_job(X_scaled, y,target_power=0.8, n_simulations_monte=5000,
                                sample_fraction=0.8, verbose=True,
                                method_powerAnaly='both')              # Série de la cible




# Utilisez les résultats comme avant
powerful_features = power_results[power_results['Power_MonteCarlo'] >= 0.5]['Feature'].tolist()
print(f"\n✅ **Features avec une puissance suffisante (≥ 0.5) :** {len(powerful_features)}/{len(feature_list)}")
print(power_results)
print(f"Observations non-NaN pour stoch_overbought: {df_filtered['stoch_overbought'].notna().sum()}")

# ---- VISUALISATION ---- #
def plot_power_analysis(power_results):
    if power_results.empty:
        print("Pas de résultats d'analyse de puissance à visualiser.")
        return

    fig, ax1 = plt.subplots(figsize=(12, 6))

    sns.barplot(x='Feature', y='Power_MonteCarlo', data=power_results, ax=ax1, palette='viridis')
    ax1.axhline(y=0.8, color='r', linestyle='--', label='Puissance cible = 0.8')
    ax1.set_title('Puissance statistique par feature (Monte Carlo)')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)
    ax1.set_ylabel('Puissance')
    ax1.legend()
    plt.show()

# ---- FILTRAGE DES FEATURES PERTINENTES ---- #
powerful_features = power_results[power_results['Power_MonteCarlo'] >= 0.5]['Feature'].tolist()
print(f"\n✅ **Features avec une puissance suffisante (≥ 0.6) :** {len(powerful_features)}/{len(feature_list)}")
if powerful_features:
    print("\n".join(f"- {f}" for f in powerful_features))
plot_power_analysis(power_results)



print("\n\nCode pour les disctribution dans le code suive cette partie")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec
import seaborn as sns


def plot_binary_feature_with_winrate(df, feature_name, target_col='class_binaire', figsize=(14, 8)):
    """
    Plot un histogramme pour une feature binaire (0/1) avec le winrate pour chaque valeur.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame contenant les données
    feature_name : str
        Nom de la feature binaire à analyser
    target_col : str, default='class_binaire'
        Nom de la colonne cible (0/1)
    figsize : tuple, default=(14, 8)
        Taille de la figure
    """
    # Vérifier que la feature existe
    if feature_name not in df.columns:
        print(f"Feature {feature_name} non trouvée dans le DataFrame")
        return

    # Filtrer les NaN
    data = df[[feature_name, target_col]].dropna()

    # Compter le nombre total d'échantillons
    total_samples = len(data)
    print(f"Total d'échantillons pour {feature_name}: {total_samples}")

    # Vérifier si la feature est binaire (0/1)
    unique_values = data[feature_name].unique()
    if not all(val in [0, 1] for val in unique_values):
        print(f"Attention: {feature_name} n'est pas strictement binaire (0/1).")
        print(f"Valeurs uniques: {unique_values}")

    # Calculer les statistiques par valeur de feature
    results = []
    for value in sorted(data[feature_name].unique()):
        subset = data[data[feature_name] == value]
        total = len(subset)
        wins = subset[target_col].sum()  # Somme des 1 dans la colonne cible
        winrate = (wins / total) * 100 if total > 0 else 0

        results.append({
            'Valeur': value,
            'Total': total,
            'Wins': wins,
            'Pertes': total - wins,
            'Winrate': winrate,
            'Pourcentage': (total / total_samples) * 100
        })

    # Créer un DataFrame avec les résultats
    results_df = pd.DataFrame(results)
    print("\nStatistiques par valeur:")
    print(results_df)

    # Créer la figure avec 2 subplots
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 1, height_ratios=[3, 1])

    # Premier subplot: histogramme par classe
    ax1 = fig.add_subplot(gs[0])

    # Histogramme
    sns.histplot(
        data=data,
        x=feature_name,
        hue=target_col,
        palette={0: "royalblue", 1: "crimson"},
        alpha=0.6,
        stat="count",
        discrete=True,
        ax=ax1
    )

    # Calculer les statistiques pour chaque classe
    class0 = data[data[target_col] == 0][feature_name]
    class1 = data[data[target_col] == 1][feature_name]

    # Calculer les moyennes et écarts-types
    mean0, std0 = class0.mean(), class0.std()
    mean1, std1 = class1.mean(), class1.std()

    # Ajouter les statistiques à l'annotation
    ax1.annotate(
        f"Cl.0: µ={mean0:.2f}, σ={std0:.2f}\n"
        f"Cl.1: µ={mean1:.2f}, σ={std1:.2f}",
        xy=(0.98, 0.95),
        xycoords='axes fraction',
        ha='right',
        va='top',
        fontsize='medium',
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8)
    )

    ax1.set_title(f"Distribution de {feature_name} par classe", fontsize=14)
    ax1.set_xlabel(feature_name, fontsize=12)
    ax1.set_ylabel("Fréquence", fontsize=12)

    # Deuxième subplot: barplot des winrates
    ax2 = fig.add_subplot(gs[1])

    # Barplot des winrates
    bar_plot = sns.barplot(
        x='Valeur',
        y='Winrate',
        data=results_df,
        palette='viridis',
        ax=ax2
    )

    # Ajouter les valeurs de winrate sur les barres
    for i, p in enumerate(bar_plot.patches):
        value = results_df.iloc[i]['Valeur']
        winrate = results_df.iloc[i]['Winrate']
        total = results_df.iloc[i]['Total']
        wins = results_df.iloc[i]['Wins']

        # Ajouter le texte au centre de la barre
        ax2.annotate(
            f"{winrate:.2f}%\n({wins}/{total})",
            (p.get_x() + p.get_width() / 2., p.get_height() / 2),
            ha='center',
            va='center',
            fontsize=11,
            color='white',
            fontweight='bold'
        )

    # Calculer l'edge (différence de winrate)
    if len(results_df) == 2:
        edge = abs(results_df.iloc[1]['Winrate'] - results_df.iloc[0]['Winrate'])
        ax2.set_title(f"Winrate par valeur (Edge: {edge:.2f}%)", fontsize=14)
    else:
        ax2.set_title(f"Winrate par valeur", fontsize=14)

    ax2.set_xlabel(feature_name, fontsize=12)
    ax2.set_ylabel("Winrate (%)", fontsize=12)
    ax2.set_ylim(40, 60)  # Ajuster selon vos besoins
    ax2.axhline(y=50, color='r', linestyle='--')  # Ligne de référence à 50%

    # Ajuster le layout
    plt.tight_layout()
    return fig
# Charger la fonction que nous venons de définir
# plot_binary_feature_with_winrate est supposée être déjà définie
def analyze_all_binary_features(df, target_col='class_binaire', min_samples=100, save_dir=None):
    """
    Analyse tous les indicateurs binaires (0/1) dans le DataFrame et affiche
    les winrates pour chaque valeur.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame contenant les données
    target_col : str, default='class_binaire'
        Nom de la colonne cible (0/1)
    min_samples : int, default=100
        Nombre minimum d'échantillons requis pour chaque valeur
    save_dir : str, optional
        Répertoire pour sauvegarder les figures (None = pas de sauvegarde)

    Returns:
    --------
    dict
        Dictionnaire contenant les résultats d'analyse pour chaque feature
    """
    # Trouver toutes les colonnes potentiellement binaires
    binary_features = []
    for col in df.columns:
        if col == target_col:
            continue

        # Vérifier si la colonne ne contient que 0 et 1 (en ignorant NaN)
        unique_values = df[col].dropna().unique()
        if set(unique_values).issubset({0, 1}) and len(unique_values) == 2:
            binary_features.append(col)

    print(f"Features binaires trouvées: {len(binary_features)}")

    # Créer un dictionnaire pour stocker les résultats
    results = {}

    # Analyser chaque feature binaire
    for feature in binary_features:
        print(f"\n{'=' * 50}\nAnalyse de {feature}\n{'=' * 50}")

        # Obtenir les statistiques
        feature_data = df[[feature, target_col]].dropna()

        # Vérifier s'il y a assez d'échantillons
        value_counts = feature_data[feature].value_counts()
        if any(count < min_samples for count in value_counts):
            print(f"Ignoré: Pas assez d'échantillons pour {feature}. Comptage: {value_counts.to_dict()}")
            continue

        # Calculer les statistiques par valeur
        feature_results = []
        for value in sorted(feature_data[feature].unique()):
            subset = feature_data[feature_data[feature] == value]
            total = len(subset)
            wins = subset[target_col].sum()
            winrate = (wins / total) * 100 if total > 0 else 0

            feature_results.append({
                'Valeur': value,
                'Total': total,
                'Wins': wins,
                'Pertes': total - wins,
                'Winrate': winrate
            })

        # Calculer l'edge
        if len(feature_results) == 2:
            edge = abs(feature_results[1]['Winrate'] - feature_results[0]['Winrate'])
        else:
            edge = None

        # Stocker les résultats
        results[feature] = {
            'stats': feature_results,
            'edge': edge
        }

        # Afficher les résultats
        results_df = pd.DataFrame(feature_results)
        print(f"Résultats pour {feature}:")
        print(results_df)
        if edge is not None:
            print(f"Edge (différence de winrate): {edge:.2f}%")

        # Créer et afficher le graphique
        try:
            fig = plot_binary_feature_with_winrate(df, feature)

            # Sauvegarder si demandé
            if save_dir is not None:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                fig.savefig(os.path.join(save_dir, f"{feature}_winrate.png"), dpi=300, bbox_inches='tight')

            plt.show()
        except Exception as e:
            print(f"Erreur lors de la création du graphique pour {feature}: {e}")

    # Résumé final: trier les features par edge
    if results:
        print("\n\n" + "=" * 70)
        print("RÉSUMÉ: FEATURES TRIÉES PAR EDGE (DIFFÉRENCE DE WINRATE)")
        print("=" * 70)

        summary = []
        for feature, data in results.items():
            if data['edge'] is not None:
                summary.append({
                    'Feature': feature,
                    'Edge': data['edge'],
                    'Winrate_0': next((item['Winrate'] for item in data['stats'] if item['Valeur'] == 0), None),
                    'Winrate_1': next((item['Winrate'] for item in data['stats'] if item['Valeur'] == 1), None),
                    'Samples_0': next((item['Total'] for item in data['stats'] if item['Valeur'] == 0), None),
                    'Samples_1': next((item['Total'] for item in data['stats'] if item['Valeur'] == 1), None)
                })

        if summary:
            summary_df = pd.DataFrame(summary)
            summary_df = summary_df.sort_values('Edge', ascending=False)
            pd.set_option('display.max_rows', None)
            print(summary_df)

            return summary_df

    return results
def analyze_all_binary_features(df, target_col='class_binaire', min_samples=100, save_dir=None):
    """
    Analyse tous les indicateurs binaires (0/1) dans le DataFrame et affiche
    les winrates pour chaque valeur.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame contenant les données
    target_col : str, default='class_binaire'
        Nom de la colonne cible (0/1)
    min_samples : int, default=100
        Nombre minimum d'échantillons requis pour chaque valeur
    save_dir : str, optional
        Répertoire pour sauvegarder les figures (None = pas de sauvegarde)

    Returns:
    --------
    dict
        Dictionnaire contenant les résultats d'analyse pour chaque feature
    """
    # Trouver toutes les colonnes potentiellement binaires
    binary_features = []
    for col in df.columns:
        if col == target_col:
            continue

        # Vérifier si la colonne ne contient que 0 et 1 (en ignorant NaN)
        unique_values = df[col].dropna().unique()
        if set(unique_values).issubset({0, 1}) and len(unique_values) == 2:
            binary_features.append(col)

    print(f"Features binaires trouvées: {len(binary_features)}")

    # Créer un dictionnaire pour stocker les résultats
    results = {}

    # Analyser chaque feature binaire
    for feature in binary_features:
        print(f"\n{'=' * 50}\nAnalyse de {feature}\n{'=' * 50}")

        # Obtenir les statistiques
        feature_data = df[[feature, target_col]].dropna()

        # Vérifier s'il y a assez d'échantillons
        value_counts = feature_data[feature].value_counts()
        if any(count < min_samples for count in value_counts):
            print(f"Ignoré: Pas assez d'échantillons pour {feature}. Comptage: {value_counts.to_dict()}")
            continue

        # Calculer les statistiques par valeur
        feature_results = []
        for value in sorted(feature_data[feature].unique()):
            subset = feature_data[feature_data[feature] == value]
            total = len(subset)
            wins = subset[target_col].sum()
            winrate = (wins / total) * 100 if total > 0 else 0

            feature_results.append({
                'Valeur': value,
                'Total': total,
                'Wins': wins,
                'Pertes': total - wins,
                'Winrate': winrate
            })

        # Calculer l'edge
        if len(feature_results) == 2:
            edge = abs(feature_results[1]['Winrate'] - feature_results[0]['Winrate'])
        else:
            edge = None

        # Stocker les résultats
        results[feature] = {
            'stats': feature_results,
            'edge': edge
        }

        # Afficher les résultats
        results_df = pd.DataFrame(feature_results)
        print(f"Résultats pour {feature}:")
        print(results_df)
        if edge is not None:
            print(f"Edge (différence de winrate): {edge:.2f}%")

        # Créer et afficher le graphique
        try:
            fig = plot_binary_feature_with_winrate(df, feature)

            # Sauvegarder si demandé
            if save_dir is not None:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                fig.savefig(os.path.join(save_dir, f"{feature}_winrate.png"), dpi=300, bbox_inches='tight')

            #plt.show()
            print("plot desactivée")
        except Exception as e:
            print(f"Erreur lors de la création du graphique pour {feature}: {e}")

    # Résumé final: trier les features par edge
    if results:
        print("\n\n" + "=" * 70)
        print("RÉSUMÉ: FEATURES TRIÉES PAR EDGE (DIFFÉRENCE DE WINRATE)")
        print("=" * 70)

        summary = []
        for feature, data in results.items():
            if data['edge'] is not None:
                summary.append({
                    'Feature': feature,
                    'Edge': data['edge'],
                    'Winrate_0': next((item['Winrate'] for item in data['stats'] if item['Valeur'] == 0), None),
                    'Winrate_1': next((item['Winrate'] for item in data['stats'] if item['Valeur'] == 1), None),
                    'Samples_0': next((item['Total'] for item in data['stats'] if item['Valeur'] == 0), None),
                    'Samples_1': next((item['Total'] for item in data['stats'] if item['Valeur'] == 1), None)
                })

        if summary:
            summary_df = pd.DataFrame(summary)
            summary_df = summary_df.sort_values('Edge', ascending=False)
            pd.set_option('display.max_rows', None)
            print(summary_df)

            return summary_df

    return results

# Exemple d'utilisation:
summary = analyze_all_binary_features(df_filtered, save_dir="C:/path/to/save/directory")