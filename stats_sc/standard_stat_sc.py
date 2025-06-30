import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from statsmodels.stats.power import TTestIndPower
from sklearn.feature_selection import f_classif  # Test F (ANOVA)

from func_standard import (calculate_atr)
from definition import *
def plot_distributions_short_long_grid(df, features, class_col='class'):
    """
    Crée une unique figure avec jusqu'à 24 features.
    - 4 features par ligne => 4 * 2 = 8 colonnes
    - nrows = nombre de lignes nécessaire pour afficher toutes les features
    - Pour chaque feature, on a 2 subplots contigus :
         - (Short)  => en bleu/orange (class=0 / class=1)
         - (Long)   => en bleu/orange (class=0 / class=1)
    """

    # 1) Limite à 24 features (optionnel si déjà fait plus haut)
    max_features = 24
    features = features[:max_features]
    n_features = len(features)

    # 2) On veut 4 features par ligne, chaque feature occupe 2 colonnes
    #    => ncols = 8
    #    => nrows = ceil(n_features / 4)
    ncols = 8
    nrows = int(np.ceil(n_features / 4))

    # 3) Préparer la figure
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(8 * 2, 4 * nrows),  # 8 colonnes * ~2 en largeur, 4 en hauteur par ligne
                             squeeze=False)

    # 4) Boucle sur chaque feature
    for i, feature in enumerate(features):
        # -- Calcul de la ligne et des 2 colonnes (short, long) --
        row = i // 4
        col_short = (i % 4) * 2
        col_long = col_short + 1

        # Subplots correspondants
        ax_short = axes[row, col_short]
        ax_long  = axes[row, col_long]

        # Filtrage pour short / long
        df_short = df[df['pos_type'] == 'short']
        df_long  = df[df['pos_type'] == 'long']

        # === Distribution SHORT ===
        sns.histplot(
            df_short[df_short[class_col] == 0][feature],
            color='blue',
            kde=True,
            label='Class 0',
            ax=ax_short
        )
        sns.histplot(
            df_short[df_short[class_col] == 1][feature],
            color='orange',
            kde=True,
            label='Class 1',
            ax=ax_short
        )
        ax_short.set_title(f"{feature} - SHORT")
        ax_short.set_xlabel('')
        ax_short.set_ylabel('Fréquence')
        ax_short.legend()

        # === Distribution LONG ===
        sns.histplot(
            df_long[df_long[class_col] == 0][feature],
            color='blue',
            kde=True,
            label='Class 0',
            ax=ax_long
        )
        sns.histplot(
            df_long[df_long[class_col] == 1][feature],
            color='orange',
            kde=True,
            label='Class 1',
            ax=ax_long
        )
        ax_long.set_title(f"{feature} - LONG")
        ax_long.set_xlabel('')
        ax_long.set_ylabel('Fréquence')
        ax_long.legend()

    # 5) Masquer les sous-graphiques inutilisés si < 24 features
    used_subplots = n_features * 2  # chaque feature utilise 2 subplots
    total_subplots = nrows * ncols
    for j in range(used_subplots, total_subplots):
        row_empty = j // ncols
        col_empty = j % ncols
        axes[row_empty, col_empty].axis('off')

    plt.tight_layout()
    plt.show()

# ======================
# 4. Définition des fonctions de traçage
# ======================
def plot_boxplots(df, features, category_col='trade_category', nrows=3, ncols=4):
    """
    Trace des sns.boxplot pour une liste de 'features' avec un ordre personnalisé.

    :param df: DataFrame contenant les données filtrées.
    :param features: Liste de features (colonnes) à tracer.
    :param category_col: Nom de la colonne catégorielle pour l'axe X (ex: 'trade_category').
    :param nrows: Nombre de lignes de subplots.
    :param ncols: Nombre de colonnes de subplots.
    """
    # Définir l'ordre des catégories
    custom_order = ["Trades réussis long", "Trades échoués long", "Trades réussis short", "Trades échoués short"]

    # Préparation de la figure
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3), squeeze=False)

    # Boucle sur chaque feature
    for idx, feature in enumerate(features):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]

        sns.boxplot(
            data=df,
            x=category_col,
            y=feature,
            hue=category_col,
            order=custom_order,  # Appliquer l'ordre personnalisé
            ax=ax,
            showmeans=True,
            palette="Set2",
            dodge=False
        )

        # Supprime la légende si elle existe
        if ax.legend_ is not None:
            ax.legend_.remove()

        # Personnalisation
        ax.set_title(feature)
        ax.set_xlabel('')
        ax.set_ylabel('Valeur')
        ax.tick_params(axis='x', rotation=30)  # Inclinaison des labels en X

    # Masquer les axes vides si le nombre de features est inférieur à nrows*ncols
    total_plots = len(features)
    for idx_empty in range(total_plots, nrows * ncols):
        row = idx_empty // ncols
        col = idx_empty % ncols
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()

def apply_feature_conditions(df, features_conditions):
    mask = np.ones(len(df), dtype=bool)  # Initialisation du masque global à True

    for feature, conditions in features_conditions.items():
        # Check if the feature exists in the dataframe
        if feature in df.columns:
            base_feature = feature
        else:
            # Try to extract the base feature name (for legacy range1, range2 handling)
            base_feature = feature.split('_range')[0]

            # If the base feature still doesn't exist, skip this feature
            if base_feature not in df.columns:
                raise ValueError(f"Feature '{feature}' or '{base_feature}' not found in dataframe. Available columns: {list(df.columns)}")
                continue

        # Filtrer les conditions actives
        active_conditions = [cond for cond in conditions if cond.get('active', False)]
        if not active_conditions:
            continue  # Aucune condition active pour cette feature

        feature_mask = np.zeros(len(df), dtype=bool)  # Initialisation à False

        for condition in active_conditions:
            if condition['type'] == 'greater_than_or_equal':
                feature_mask |= df[base_feature].fillna(-np.inf) >= condition['threshold']
            elif condition['type'] == 'less_than_or_equal':
                feature_mask |= df[base_feature].fillna(np.inf) <= condition['threshold']
            elif condition['type'] == 'between':
                feature_mask |= df[base_feature].fillna(np.nan).between(
                    condition['min'], condition['max'], inclusive='both'
                )
            elif condition['type'] == 'not_between':
                # Nouvelle condition qui exclut ce qui est entre deux valeurs
                feature_mask |= ~df[base_feature].fillna(np.nan).between(
                    condition['min'], condition['max'], inclusive='both'
                )

        mask &= feature_mask  # Intersection avec le masque global

    return df[mask]


def find_consecutive_trades(df, trade_category):
    """
    Trouve les séquences consécutives pour une catégorie spécifique de trades,
    en tenant compte des interruptions par d'autres types de trades.
    """
    filtered_df = df.copy()  # On garde tous les trades pour voir les interruptions

    if filtered_df.empty:
        return 0, None, None

    # Conversion des dates
    if not pd.api.types.is_datetime64_any_dtype(filtered_df['date']):
        filtered_df['date'] = pd.to_datetime(filtered_df['date'])

    # Trier par date
    filtered_df = filtered_df.sort_values('date')

    max_sequence = 0
    current_sequence = 0
    max_start_date = None
    max_end_date = None
    current_start_date = None

    import numpy as np  # Assurez-vous que cette ligne existe au début du fichier

    # Trier par date
    filtered_df = filtered_df.sort_values('date')

    max_sequence = 0
    current_sequence = 0
    max_start_date = None
    max_end_date = None
    current_start_date = None

    import numpy as np

    # Convertir la colonne date en tableau numpy d'une manière différente
    # Cette approche évite complètement l'utilisation de to_pydatetime()
    dates = filtered_df['date'].to_numpy()

    # La ligne pour les catégories reste la même
    categories = filtered_df['trade_category'].values

    for i, (date, category) in enumerate(zip(dates, categories)):
        if category == trade_category:
            if current_sequence == 0:
                current_sequence = 1
                current_start_date = date
            else:
                # Vérifier si le trade précédent était de la même catégorie
                if categories[i - 1] == trade_category:
                    current_sequence += 1
                else:
                    # Réinitialiser si le trade précédent était différent
                    current_sequence = 1
                    current_start_date = date
        else:
            # Si on trouve la plus longue séquence jusqu'ici
            if current_sequence > max_sequence:
                max_sequence = current_sequence
                max_start_date = current_start_date
                max_end_date = dates[i - 1]
            current_sequence = 0

    # Vérifier la dernière séquence
    if current_sequence > max_sequence:
        max_sequence = current_sequence
        max_start_date = current_start_date
        max_end_date = dates[-1]

    return max_sequence, max_start_date, max_end_date
def calculate_performance_metrics(df):
    """
    Calcule les métriques complètes de performance de trading.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame contenant les données de trading avec les colonnes:
        - 'trade_pnl' : Profit/Perte de chaque trade
        - 'pos_type' : Type de position ('long' ou 'short')
        - index : dates des trades

    Returns:
    --------
    dict
        Dictionnaire structuré contenant toutes les métriques de performance
    """
    # 1. Métriques de base sur l'ensemble des trades
    total_trades = len(df)
    winning_trades = df[df['trade_pnl'] > 0]
    losing_trades = df[df['trade_pnl'] < 0]

    num_winning_trades = len(winning_trades)
    num_losing_trades = len(losing_trades)

    # 2. Séparation et analyse par direction
    # Trades longs
    long_trades = df[df['pos_type'] == 'long']
    winning_trades_long = winning_trades[winning_trades['pos_type'] == 'long']
    losing_trades_long = losing_trades[losing_trades['pos_type'] == 'long']

    num_winning_trades_long = len(winning_trades_long)
    num_losing_trades_long = len(losing_trades_long)
    total_trades_long = num_winning_trades_long + num_losing_trades_long

    # Trades shorts
    short_trades = df[df['pos_type'] == 'short']
    winning_trades_short = winning_trades[winning_trades['pos_type'] == 'short']
    losing_trades_short = losing_trades[losing_trades['pos_type'] == 'short']

    num_winning_trades_short = len(winning_trades_short)
    num_losing_trades_short = len(losing_trades_short)
    total_trades_short = num_winning_trades_short + num_losing_trades_short

    # 3. Calcul des profits et pertes
    gross_profit = winning_trades['trade_pnl'].sum() if not winning_trades.empty else 0
    gross_loss = losing_trades['trade_pnl'].sum() if not losing_trades.empty else 0
    net_pnl = gross_profit + gross_loss

    # Profit factor avec gestion de division par zéro
    profit_factor = gross_profit / abs(gross_loss) if abs(gross_loss) > 0 else np.inf if gross_profit > 0 else 0

    # 4. Calcul des win rates
    win_rate = (num_winning_trades / total_trades * 100) if total_trades > 0 else 0
    win_rate_long = (num_winning_trades_long / total_trades_long * 100) if total_trades_long > 0 else 0
    win_rate_short = (num_winning_trades_short / total_trades_short * 100) if total_trades_short > 0 else 0

    # 5. Calcul des moyennes de PnL
    expected_pnl_net = df['trade_pnl'].mean() if not df.empty else 0
    expected_pnl_long = long_trades['trade_pnl'].mean() if not long_trades.empty else 0
    expected_pnl_short = short_trades['trade_pnl'].mean() if not short_trades.empty else 0

    # Moyennes des profits et pertes
    avg_profit_per_win = winning_trades['trade_pnl'].mean() if not winning_trades.empty else 0
    avg_loss_per_loss = losing_trades['trade_pnl'].mean() if not losing_trades.empty else 0

    # 6. Analyse des séquences consécutives
    # Analyse des séquences consécutives pour les longs
    long_win_seq, long_win_start, long_win_end = find_consecutive_trades(df, "Trades réussis long")
    long_lose_seq, long_lose_start, long_lose_end = find_consecutive_trades(df, "Trades échoués long")

    # Analyse des séquences consécutives pour les shorts
    short_win_seq, short_win_start, short_win_end = find_consecutive_trades(df, "Trades réussis short")
    short_lose_seq, short_lose_start, short_lose_end = find_consecutive_trades(df, "Trades échoués short")

    # 7. Identification des meilleurs et pires trades
    def get_extreme_trade(trades_df, extreme_type='max'):
        if trades_df.empty:
            return {'PnL': 0, 'Date': None}
        if extreme_type == 'max':
            idx = trades_df['trade_pnl'].idxmax()
            pnl = trades_df['trade_pnl'].max()
        else:
            idx = trades_df['trade_pnl'].idxmin()
            pnl = trades_df['trade_pnl'].min()
        return {'PnL': pnl, 'Date': idx}

    # 8. Construction du dictionnaire de résultats
    return {
        "Total Trades": total_trades,
        "Trades Réussis": num_winning_trades,
        "Trades Échoués": num_losing_trades,

        "Trades Longs": {
            "Total": total_trades_long,
            "Réussis": num_winning_trades_long,
            "Échoués": num_losing_trades_long,
            "Win Rate": win_rate_long,
            "PnL Moyen": expected_pnl_long,
            "Meilleur Trade": get_extreme_trade(long_trades, 'max'),
            "Pire Trade": get_extreme_trade(long_trades, 'min'),
            "Séquences Consécutives": {
                "Max Trades Gagnants": {
                    "Nombre": long_win_seq,
                    "Date Début": long_win_start,
                    "Date Fin": long_win_end
                },
                "Max Trades Perdants": {
                    "Nombre": long_lose_seq,
                    "Date Début": long_lose_start,
                    "Date Fin": long_lose_end
                }
            }
        },

        "Trades Shorts": {
            "Total": total_trades_short,
            "Réussis": num_winning_trades_short,
            "Échoués": num_losing_trades_short,
            "Win Rate": win_rate_short,
            "PnL Moyen": expected_pnl_short,
            "Meilleur Trade": get_extreme_trade(short_trades, 'max'),
            "Pire Trade": get_extreme_trade(short_trades, 'min'),
            "Séquences Consécutives": {
                "Max Trades Gagnants": {
                    "Nombre": short_win_seq,
                    "Date Début": short_win_start,
                    "Date Fin": short_win_end
                },
                "Max Trades Perdants": {
                    "Nombre": short_lose_seq,
                    "Date Début": short_lose_start,
                    "Date Fin": short_lose_end
                }
            }
        },

        "Performance Globale": {
            "Win Rate Total": win_rate,
            "Gross Profit": gross_profit,
            "Gross Loss": gross_loss,
            "Net PnL": net_pnl,
            "Profit Factor": profit_factor,
            "PnL Moyen par Trade": expected_pnl_net,
            "Profit Moyen (Trades Gagnants)": avg_profit_per_win,
            "Perte Moyenne (Trades Perdants)": avg_loss_per_loss
        }
    }
def print_comparative_performance(metrics_before, metrics_after):
    """
    Affiche une comparaison détaillée et complète des performances avant et après filtrage.

    Parameters:
    -----------
    metrics_before : dict
        Métriques de performance avant filtrage
    metrics_after : dict
        Métriques de performance après filtrage
    """

    def calculate_change(before, after):
        if isinstance(before, (int, float)) and isinstance(after, (int, float)):
            if before == 0:
                return "N/A" if after == 0 else "+∞" if after > 0 else "-∞"
            change = ((after - before) / abs(before)) * 100
            return f"{change:+.2f}%"
        return "N/A"

    #print("\n═══ ANALYSE COMPARATIVE DES PERFORMANCES ═══")

    print("\n📊 STATISTIQUES GLOBALES")
    print("═" * 75)
    print(f"{'Métrique':<35} {'Avant':<15} {'Après':<15} {'Variation':<15}")
    print("─" * 75)

    # Statistiques de base
    basic_metrics = {
        'Nombre total de trades': 'Total Trades',
        'Trades Réussis': 'Trades Réussis',
        'Trades Échoués': 'Trades Échoués'
    }

    for label, key in basic_metrics.items():
        before_val = metrics_before[key]
        after_val = metrics_after[key]
        print(f"{label:<35} {before_val:<15} {after_val:<15} {calculate_change(before_val, after_val):<15}")

    print("\n📈 PERFORMANCE GLOBALE")
    print("-" * 75)

    global_metrics = {
        'Win Rate Total (%)': 'Win Rate Total',
        'Gross Profit': 'Gross Profit',
        'Gross Loss': 'Gross Loss',
        'Net PnL': 'Net PnL',
        'Profit Factor': 'Profit Factor',
        'PnL Moyen par Trade': 'PnL Moyen par Trade',
        'Profit Moyen (Trades Gagnants)': 'Profit Moyen (Trades Gagnants)',
        'Perte Moyenne (Trades Perdants)': 'Perte Moyenne (Trades Perdants)'
    }

    for label, key in global_metrics.items():
        before_val = metrics_before['Performance Globale'][key]
        after_val = metrics_after['Performance Globale'][key]
        print(f"{label:<35} {before_val:15.2f} {after_val:15.2f} {calculate_change(before_val, after_val):<15}")

    # Analyse détaillée par direction
    directions = ['Longs', 'Shorts']
    for direction in directions:
        print(f"\n📊 ANALYSE DES TRADES {direction.upper()}")
        print("═" * 75)

        direction_metrics = {
            'Nombre total': 'Total',
            'Trades Réussis': 'Réussis',
            'Trades Échoués': 'Échoués',
            'Win Rate (%)': 'Win Rate',
            'PnL Moyen': 'PnL Moyen'
        }

        for label, key in direction_metrics.items():
            before_val = metrics_before[f'Trades {direction}'][key]
            after_val = metrics_after[f'Trades {direction}'][key]
            print(f"{label:<35} {before_val:15.2f} {after_val:15.2f} {calculate_change(before_val, after_val):<15}")

        # Meilleurs et pires trades avec leurs dates
        print(f"\n🎯 TRADES EXTRÊMES {direction.upper()}")
        print("─" * 75)

        # Meilleur trade
        best_before = metrics_before[f'Trades {direction}']['Meilleur Trade']
        best_after = metrics_after[f'Trades {direction}']['Meilleur Trade']
        print(f"Meilleur trade avant: {best_before['PnL']:.2f} (Date: {best_before['Date']})")
        print(f"Meilleur trade après: {best_after['PnL']:.2f} (Date: {best_after['Date']})")

        # Pire trade
        worst_before = metrics_before[f'Trades {direction}']['Pire Trade']
        worst_after = metrics_after[f'Trades {direction}']['Pire Trade']
        print(f"Pire trade avant: {worst_before['PnL']:.2f} (Date: {worst_before['Date']})")
        print(f"Pire trade après: {worst_after['PnL']:.2f} (Date: {worst_after['Date']})")

    # Résumé de l'impact du filtrage
    print("\n📑 RÉSUMÉ DE L'IMPACT DU FILTRAGE")
    print("═" * 75)
    trades_removed = metrics_before['Total Trades'] - metrics_after['Total Trades']
    trades_removed_pct = (trades_removed / metrics_before['Total Trades']) * 100

    print(f"Trades filtrés: {trades_removed} ({trades_removed_pct:.2f}% du total)")

    wr_impact = metrics_after['Performance Globale']['Win Rate Total'] - metrics_before['Performance Globale'][
        'Win Rate Total']
    print(f"Impact sur le Win Rate: {wr_impact:+.2f}%")

    pnl_impact = metrics_after['Performance Globale']['Net PnL'] - metrics_before['Performance Globale']['Net PnL']
    print(f"Impact sur le Net PnL: {pnl_impact:+.2f}")

    pf_impact = metrics_after['Performance Globale']['Profit Factor'] - metrics_before['Performance Globale'][
        'Profit Factor']
    print(f"Impact sur le Profit Factor: {pf_impact:+.2f}")

    # Ajouter la section des séquences juste avant le résumé de l'impact
    for direction in ['Longs', 'Shorts']:
        print(f"\n📊 SÉQUENCES CONSÉCUTIVES {direction.upper()}")
        print("═" * 75)

        for period, metrics in [("Avant", metrics_before), ("Après", metrics_after)]:
            sequences = metrics[f"Trades {direction}"]["Séquences Consécutives"]
            print(f"\n{period}:")

            # Affichage des trades gagnants consécutifs
            win_seq = sequences["Max Trades Gagnants"]
            print(f"Trades gagnants consécutifs maximum : {win_seq['Nombre']}")
            if win_seq['Nombre'] > 0:
                print(f"  Période : du {win_seq['Date Début']} au {win_seq['Date Fin']}")

            # Affichage des trades perdants consécutifs
            lose_seq = sequences["Max Trades Perdants"]
            print(f"Trades perdants consécutifs maximum : {lose_seq['Nombre']}")
            if lose_seq['Nombre'] > 0:
                print(f"  Période : du {lose_seq['Date Début']} au {lose_seq['Date Fin']}")
            print()

def calculate_statistical_power(X, y, feature_list=None,
                                alpha=0.05, target_power=0.8, n_simulations=20000,
                                sample_fraction=0.8, verbose=True,
                                method_powerAnaly='both'):
    """
    Calcule la puissance statistique analytique et/ou par simulation Monte Carlo.

    Parameters:
    -----------
    X : pandas.DataFrame
        DataFrame contenant uniquement les features à analyser
    y : pandas.Series
        Série contenant la variable cible binaire (0/1)
    feature_list : list, optional
        Liste des noms de features à analyser. Si None, utilise toutes les colonnes de X
    alpha : float, default=0.05
        Seuil de significativité
    target_power : float, default=0.8
        Puissance statistique cible
    n_simulations : int, default=10000
        Nombre de simulations Monte Carlo
    sample_fraction : float, default=0.8
        Fraction de l'échantillon à utiliser dans chaque simulation
    verbose : bool, default=True
        Afficher les messages d'avancement
    method : str, default='both'
        Méthode de calcul de la puissance à utiliser. Options:
        - 'both': calcule la puissance analytique et Monte Carlo
        - 'analytical': calcule uniquement la puissance analytique
        - 'montecarlo': calcule uniquement la puissance par simulation Monte Carlo

    Returns:
    --------
    pandas.DataFrame
        DataFrame contenant les résultats de l'analyse de puissance
    """
    # Vérifier que la méthode est valide
    valid_methods = ['both', 'analytical', 'montecarlo']
    if method_powerAnaly not in valid_methods:
        raise ValueError(f"La méthode '{method_powerAnaly}' n'est pas valide. Options: {', '.join(valid_methods)}")

    if feature_list is None:
        feature_list = X.columns.tolist()
    else:
        # S'assurer que toutes les features demandées existent dans X
        feature_list = [f for f in feature_list if f in X.columns]

    # Filtrer les colonnes constantes (n'ayant qu'une seule valeur unique)
    constant_features = [col for col in feature_list if X[col].nunique() <= 1]
    if constant_features and verbose:
        print(f"⚠️ {len(constant_features)} features constantes retirées: {constant_features}")

    feature_list = [f for f in feature_list if f not in constant_features]

    results = []
    power_analysis = TTestIndPower()
    total_features = len(feature_list)

    for i, feature in enumerate(feature_list):
        if verbose and i % max(1, total_features // 10) == 0:
            print(f"Traitement: {i + 1}/{total_features} features ({((i + 1) / total_features) * 100:.1f}%)")

        # Préparation des données pour cette feature
        X_feature = X[feature].copy()

        # Filtrer les valeurs NaN
        mask = X_feature.notna() & y.notna()
        X_filtered = X_feature[mask].values
        y_filtered = y[mask].values

        # Séparer les groupes
        group0 = X_filtered[y_filtered == 0]
        group1 = X_filtered[y_filtered == 1]

        # Vérifier que les deux groupes ont suffisamment de données
        if len(group0) <= 1 or len(group1) <= 1:
            if verbose:
                print(f"⚠️ Skipping {feature}: Not enough data in both groups")
            continue

        # Calcul de l'effet de taille (Cohen's d)
        #mean_diff = np.mean(group1) - np.mean(group0)
        mean_diff = np.median(group1) - np.median(group0)

        pooled_std = np.sqrt(((len(group0) - 1) * np.std(group0, ddof=1) ** 2 +
                              (len(group1) - 1) * np.std(group1, ddof=1) ** 2) /
                             (len(group0) + len(group1) - 2))

        if pooled_std == 0:
            if verbose:
                print(f"⚠️ Skipping {feature}: Zero variance in data")
            continue

        effect_size = mean_diff / pooled_std

        # Test statistique de base (t-test de Welch)
        t_stat, p_value = stats.ttest_ind(group0, group1, equal_var=False)

        # Initialiser les valeurs par défaut
        power_analytical = None
        power_monte_carlo = None
        se_monte_carlo = None
        mc_ci_lower = None
        mc_ci_upper = None
        required_n = np.nan

        # Méthode 1: Puissance Analytique (si demandée)
        if method_powerAnaly in ['both', 'analytical']:
            power_analytical = power_analysis.power(
                effect_size=abs(effect_size),  # Utiliser valeur absolue
                nobs1=len(group0),
                alpha=alpha,
                ratio=len(group1) / len(group0)
            )

            # Calcul de la taille d'échantillon requise
            try:
                required_n = power_analysis.solve_power(
                    effect_size=abs(effect_size),
                    power=target_power,
                    alpha=alpha,
                    ratio=len(group1) / len(group0)
                )
            except (ValueError, RuntimeError):
                required_n = np.nan

        # Méthode 2: Puissance Monte Carlo par simulation (si demandée)
        if method_powerAnaly in ['both', 'montecarlo']:
            significant_count = 0
            for _ in range(n_simulations):
                # Échantillonnage aléatoire
                if len(group0) > 1 and len(group1) > 1:
                    sample0_size = max(2, int(len(group0) * sample_fraction))
                    sample1_size = max(2, int(len(group1) * sample_fraction))

                    sample0 = np.random.choice(group0, size=sample0_size, replace=True)
                    sample1 = np.random.choice(group1, size=sample1_size, replace=True)

                    # Test t sur l'échantillon
                    _, p_sim = stats.ttest_ind(sample0, sample1, equal_var=False)
                    if p_sim < alpha:
                        significant_count += 1

            power_monte_carlo = significant_count / n_simulations

            # Calcul de l'erreur standard de la puissance Monte Carlo
            se_monte_carlo = np.sqrt(power_monte_carlo * (1 - power_monte_carlo) / n_simulations)
            mc_ci_lower = max(0, power_monte_carlo - 1.96 * se_monte_carlo)
            mc_ci_upper = min(1, power_monte_carlo + 1.96 * se_monte_carlo)

        # Déterminer quelle puissance utiliser pour la colonne Power_Sufficient
        if method_powerAnaly == 'both':
            power_for_sufficiency = power_monte_carlo
        elif method_powerAnaly == 'analytical':
            power_for_sufficiency = power_analytical
        else:  # montecarlo
            power_for_sufficiency = power_monte_carlo

        # Calcul de la différence entre les méthodes
        delta_power = None
        if method_powerAnaly == 'both' and power_analytical is not None and power_monte_carlo is not None:
            delta_power = abs(power_analytical - power_monte_carlo)

        # Ajouter les résultats
        result_row = {
            'Feature': feature,
            'Sample_Size': len(X_filtered),
            'Group0_Size': len(group0),
            'Group1_Size': len(group1),
            'Effect_Size': effect_size,
            'P_Value': p_value,
            'Required_N': np.ceil(required_n) if not np.isnan(required_n) else np.nan,
            'Power_Sufficient': power_for_sufficiency is not None and power_for_sufficiency >= target_power
        }

        # Ajouter les colonnes spécifiques à la méthode analytique
        if method_powerAnaly in ['both', 'analytical']:
            result_row['Power_Analytical'] = power_analytical

        # Ajouter les colonnes spécifiques à la méthode Monte Carlo
        if method_powerAnaly in ['both', 'montecarlo']:
            result_row['Power_MonteCarlo'] = power_monte_carlo
            result_row['MC_StdError'] = se_monte_carlo
            result_row['MC_CI_Lower'] = mc_ci_lower
            result_row['MC_CI_Upper'] = mc_ci_upper

        # Ajouter la différence entre les méthodes si les deux sont calculées
        if delta_power is not None:
            result_row['Delta_Power'] = delta_power

        results.append(result_row)

    # Créer le DataFrame des résultats
    results_df = pd.DataFrame(results)

    if not results_df.empty:
        # # Trier par la puissance appropriée selon la méthode choisie
        # if method_powerAnaly == 'both' or method_powerAnaly == 'montecarlo':
        #     sort_column = 'Power_MonteCarlo'
        # else:  # analytical
        #     sort_column = 'Power_Analytical'

        # if sort_column in results_df.columns:
        #     results_df = results_df.sort_values(sort_column, ascending=False)

        if verbose:
            print(f"\nAnalyse terminée. {len(results_df)} features analysées.")

            # Statistiques sur les différences de puissance si applicable
            if 'Delta_Power' in results_df.columns:
                mean_delta = results_df['Delta_Power'].mean()
                max_delta = results_df['Delta_Power'].max()
                print(f"Différence moyenne entre les méthodes: {mean_delta:.4f}")
                print(f"Différence maximale entre les méthodes: {max_delta:.4f}")

            # Features avec puissance suffisante
            sufficient_power = results_df[results_df['Power_Sufficient']].shape[0]
            print(
                f"Features avec puissance suffisante (>= {target_power}): {sufficient_power} sur {len(results_df)}")
    else:
        if verbose:
            print("Aucun résultat obtenu. Vérifiez vos données et les paramètres.")

    return results_df


import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import TTestIndPower
from joblib import Parallel, delayed

from scipy.stats import ttest_ind, mannwhitneyu, normaltest, levene
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu, shapiro

def run_single_simulation_auto(group0, group1, sample_fraction, alpha):
    """
    Simulation Monte Carlo qui choisit automatiquement entre Test t et Mann-Whitney.

    """
    sample0_size = max(2, int(len(group0) * sample_fraction))
    sample1_size = max(2, int(len(group1) * sample_fraction))

    sample0 = np.random.choice(group0, size=sample0_size, replace=True)
    sample1 = np.random.choice(group1, size=sample1_size, replace=True)

    # Vérifier la normalité uniquement si les échantillons sont suffisants
    if len(sample0) > 20 and len(sample1) > 20:
        norm_test_0 = normaltest(sample0).pvalue
        norm_test_1 = normaltest(sample1).pvalue
        var_test_p = levene(sample0, sample1).pvalue  # Test de variance

        if norm_test_0 > 0.05 and norm_test_1 > 0.05:  # Si les 2 sont normaux
            _, p_sim = ttest_ind(sample0, sample1, equal_var=(var_test_p > 0.05))
        else:
            _, p_sim = mannwhitneyu(sample0, sample1, alternative='two-sided')
    else:
        _, p_sim = mannwhitneyu(sample0, sample1, alternative='two-sided')  # Cas petit échantillon

    return p_sim < alpha


from scipy.stats import ttest_ind, mannwhitneyu, normaltest, levene
from statsmodels.stats.power import TTestIndPower





def create_full_dataframe_with_filtered_pnl(df_init_features, df_filtered):
        # Création d'une copie du DataFrame initial
        df_full_afterFiltering = df_init_features.copy()

        # Création d'une colonne PnlAfterFiltering initialisée à 0.0 (comme un float)
        # Cela garantit que la colonne sera créée avec un type compatible avec vos valeurs de PnL
        df_full_afterFiltering['PnlAfterFiltering'] = 0.0

        # Identification des lignes qui ont passé le filtrage
        filtered_indices = df_filtered.index

        # Pour ces lignes, on attribue le PnL original à la nouvelle colonne
        # Conversion explicite en float pour s'assurer de la compatibilité
        df_full_afterFiltering.loc[filtered_indices, 'PnlAfterFiltering'] = df_filtered['trade_pnl'].astype(float)

        return df_full_afterFiltering

import pandas as pd

def preprocess_sessions_with_date(df):
    """
    Prétraite les données de sessions en:
    1. Créant une colonne 'session_id' à partir des marqueurs SessionStartEnd
    2. Ajoutant une colonne 'session_date' qui contient la date du dernier enregistrement de chaque session

    Args:
        df (pd.DataFrame): DataFrame avec la colonne SessionStartEnd (10=début, 20=fin) et 'date'

    Returns:
        pd.DataFrame: DataFrame avec les nouvelles colonnes 'session_id' et 'session_date'
    """
    #print("Création des identifiants de session et dates de session")

    # Copier le DataFrame pour éviter de modifier l'original
    df_copy = df.copy()

    # Générer les IDs de session en cumulant les débuts de session
    df_copy['session_id'] = (df_copy['sc_sessionStartEnd'] == 10).cumsum()

    # Identifier les sessions valides (celles qui ont un début et une fin)
    session_valid = df_copy.groupby('session_id')['sc_sessionStartEnd'].transform(lambda x: (10 in x.values) and (20 in x.values))

    # Supprimer les sessions incomplètes (sans fin)
    df_copy.loc[~session_valid, 'session_id'] = 0

    # Assigner la dernière date de chaque session
    df_copy['session_date'] = df_copy.groupby('session_id')['date'].transform('last')

    # Affichage des statistiques
    nb_sessions = df_copy['session_id'].nunique() - (0 in df_copy['session_id'].unique())
    print(f"✓ {nb_sessions} sessions valides identifiées")

    return df_copy



import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import matplotlib.gridspec as gridspec  # Ajout de cette importation


def plot_trading_performance(df):
    """
    Crée une visualisation à quatre niveaux de la performance de trading :
    1. PnL cumulé sur toutes les sessions avec statistiques détaillées
    2. Drawdown cumulé
    3. PnL journalier par session
    4. Profil de performance intra-journalière type sur une session de 23h (de 22h à 21h le lendemain)

    Args:
        df: DataFrame contenant 'session_date', 'PnlAfterFiltering', 'session_id' et 'deltaTimestampOpening'
    """
    # Préparation des données
    # S'assurer que session_date est au format datetime
    if not pd.api.types.is_datetime64_any_dtype(df['session_date']):
        df['session_date'] = pd.to_datetime(df['session_date'])

    # Regrouper par date de session et sommer le PnL
    daily_pnl = df.groupby('session_date')['PnlAfterFiltering'].sum().reset_index()
    daily_pnl = daily_pnl.sort_values('session_date')

    # Calculer le PnL cumulé
    daily_pnl['cumulative_pnl'] = daily_pnl['PnlAfterFiltering'].cumsum()

    # Calculer le drawdown
    daily_pnl['peak'] = daily_pnl['cumulative_pnl'].cummax()
    daily_pnl['drawdown'] = daily_pnl['peak'] - daily_pnl['cumulative_pnl']

    # Identifier les jours avec PnL max et min
    max_pnl_idx = daily_pnl['PnlAfterFiltering'].idxmax()
    min_pnl_idx = daily_pnl['PnlAfterFiltering'].idxmin()
    max_pnl_day = daily_pnl.loc[max_pnl_idx] if max_pnl_idx is not None else None
    min_pnl_day = daily_pnl.loc[min_pnl_idx] if min_pnl_idx is not None else None

    # Calculer le drawdown maximum
    max_drawdown = daily_pnl['drawdown'].max()
    max_dd_idx = daily_pnl['drawdown'].idxmax()
    max_dd_date = daily_pnl.loc[max_dd_idx, 'session_date'] if max_dd_idx is not None else None

    # Calculer les statistiques de performance globales
    # Identifier les trades gagnants et perdants
    winning_trades = df[df['PnlAfterFiltering'] > 0]
    losing_trades = df[df['PnlAfterFiltering'] < 0]

    # Calculer les métriques de performance
    total_trades = len(df[df['PnlAfterFiltering'] != 0])
    winning_trades_count = len(winning_trades)
    losing_trades_count = len(losing_trades)
    winrate = winning_trades_count / total_trades * 100 if total_trades > 0 else 0

    total_profit = winning_trades['PnlAfterFiltering'].sum()
    total_loss = abs(losing_trades['PnlAfterFiltering'].sum())
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

    # Calculer l'expected PnL par trade
    # Filtrer pour n'inclure que les trades effectués
    executed_trades = df[df['PnlAfterFiltering'] != 0]
    expected_pnl = executed_trades['PnlAfterFiltering'].mean() if len(executed_trades) > 0 else 0

    # Fonction d'aide pour formater les dates de manière sécurisée
    def format_date(date_obj):
        if pd.isna(date_obj):
            return "Date inconnue"
        try:
            # Pour pandas Timestamp
            if hasattr(date_obj, 'strftime'):
                return date_obj.strftime('%Y-%m-%d')
            # Pour les dates en string ou autres formats
            return str(date_obj).split(' ')[0]
        except:
            return str(date_obj)

    # Création de la figure avec quatre sous-graphiques
    fig, axes = plt.subplots(4, 1, figsize=(15, 18), sharex=False,
                             gridspec_kw={'height_ratios': [2, 1, 1.5, 2.5]})

    # 1. Graphique du PnL cumulé
    axes[0].plot(daily_pnl['session_date'], daily_pnl['cumulative_pnl'], 'b-', linewidth=2)
    axes[0].set_title('PnL Cumulé', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0].set_ylabel('PnL ($)', fontsize=12)

    # Annotation finale avec les statistiques détaillées
    if len(daily_pnl) > 0:
        final_pnl = daily_pnl['cumulative_pnl'].iloc[-1]

        # Créer un texte avec toutes les statistiques
        stats_text = f"PnL Final: ${final_pnl:.2f}\n"
        stats_text += f"Trades: {total_trades} (Gagnants: {winning_trades_count}, Perdants: {losing_trades_count})\n"
        stats_text += f"Winrate: {winrate:.2f}%\n"
        stats_text += f"Profit: ${total_profit:.2f}, Pertes: ${total_loss:.2f}\n"
        stats_text += f"Profit Factor: {profit_factor:.2f}\n"
        stats_text += f"Expected PnL: ${expected_pnl:.2f}"

        # Obtenir la transformation des coordonnées entre les axes et la figure
        bbox = axes[0].get_position()

        # Placer l'annotation dans l'espace de la figure, juste à droite des axes
        # Note: l'axe des x va de 0 à 1 dans l'espace de la figure
        # Obtenir la transformation des coordonnées entre les axes et la figure
        bbox = axes[0].get_position()

        # Placer l'annotation dans l'espace de la figure, un peu moins à droite
        fig.text(
            bbox.x1 - 0.1,  # Position x: légèrement à gauche de la fin des axes
            bbox.y0 + bbox.height / 2,  # Position y: milieu vertical des axes
            stats_text,
            verticalalignment='center',
            horizontalalignment='left',
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8, edgecolor='gray')
        )

    # 2. Graphique du drawdown
    axes[1].fill_between(daily_pnl['session_date'], 0, daily_pnl['drawdown'], color='r', alpha=0.3)
    axes[1].plot(daily_pnl['session_date'], daily_pnl['drawdown'], 'r-', linewidth=1)
    axes[1].set_title('Drawdown', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylabel('Drawdown ($)', fontsize=12)

    # Annotation du drawdown maximum
    if max_dd_date is not None:
        formatted_date = format_date(max_dd_date)
        axes[1].annotate(f'Max Drawdown: ${max_drawdown:.2f} ({formatted_date})',
                         xy=(max_dd_date, max_drawdown),
                         xytext=(10, -20), textcoords='offset points',
                         arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
                         fontweight='bold', color='darkred')

    # 3. Graphique du PnL journalier
    bar_colors = ['g' if x >= 0 else 'r' for x in daily_pnl['PnlAfterFiltering']]
    axes[2].bar(daily_pnl['session_date'], daily_pnl['PnlAfterFiltering'], color=bar_colors, alpha=0.7)
    axes[2].set_title('PnL Journalier', fontsize=14)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylabel('PnL ($)', fontsize=12)
    #axes[2].set_xlabel('Date', fontsize=12)

    # Formater l'axe des x pour qu'il affiche les dates correctement
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    # Déterminer la fréquence appropriée des ticks en fonction du nombre de jours
    if len(daily_pnl) > 45:
        # Si beaucoup de données, montrer moins de ticks pour éviter l'encombrement
        axes[2].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))  # Lundi
    else:
        # Si peu de données, montrer toutes les dates
        axes[2].xaxis.set_major_locator(mdates.DayLocator())

    # Ajuster la taille et l'angle des étiquettes de date
    for label in axes[2].get_xticklabels():
        label.set_fontsize(8)  # Taille de police 8
        label.set_rotation(45)  # Inclinaison à 45 degrés
        label.set_ha('right')  # Alignement à droite pour éviter le chevauchement
        label.set_rotation_mode('anchor')  # Mode d'ancrage pour une meilleure rotation

    # Annotations pour le PnL journalier max et min
    if max_pnl_day is not None:
        axes[2].annotate(f'Max: ${max_pnl_day["PnlAfterFiltering"]:.2f}',
                         xy=(max_pnl_day['session_date'], max_pnl_day['PnlAfterFiltering']),
                         xytext=(0, 20), textcoords='offset points',
                         arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
                         fontweight='bold', color='darkgreen')

    if min_pnl_day is not None:
        axes[2].annotate(f'Min: ${min_pnl_day["PnlAfterFiltering"]:.2f}',
                         xy=(min_pnl_day['session_date'], min_pnl_day['PnlAfterFiltering']),
                         xytext=(0, -20), textcoords='offset points',
                         arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
                         fontweight='bold', color='darkred')

    # 4. Graphique du profil intra-journalier (tranches de 30 minutes)
    # 4. Graphique du profil intra-journalier (tranches de 30 minutes) - DIVISÉ EN DEUX
    if 'deltaTimestampOpening' in df.columns:
        # Convertir deltaTimestampOpening en numérique si ce n'est pas déjà le cas
        if not pd.api.types.is_numeric_dtype(df['deltaTimestampOpening']):
            df['deltaTimestampOpening'] = pd.to_numeric(df['deltaTimestampOpening'], errors='coerce')

        # Limiter aux données pertinentes (dans la plage de 0 à 1380 minutes)
        df_valid = df[(df['deltaTimestampOpening'] >= 0) & (df['deltaTimestampOpening'] <= 1380)]

        # Arrondir à la tranche de 30 minutes la plus proche
        df_valid['time_bin'] = (df_valid['deltaTimestampOpening'] // 30) * 30

        # Créer une sous-figure avec deux graphiques côte à côte pour la 4ème ligne
        gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=plt.GridSpec(4, 1)[3], width_ratios=[1, 1], wspace=0.3)

        # 4.1 GAUCHE - PnL moyen par intervalle de 30 minutes (graphique existant)
        ax_left = fig.add_subplot(gs[0])

        # Calculer le PnL moyen par intervalle de 30 minutes pour toutes les sessions
        intraday_profile = df_valid.groupby('time_bin')['PnlAfterFiltering'].agg(['sum', 'mean', 'count']).reset_index()
        intraday_profile = intraday_profile.sort_values('time_bin')

        # Fonction pour convertir les minutes depuis l'ouverture (22h) en format d'heure lisible
        def format_trading_time(minutes):
            total_hours = 22 + minutes // 60
            day_marker = "" if total_hours < 24 else "+1j"  # Indicateur de jour suivant
            hours = total_hours % 24
            mins = minutes % 60
            return f"{hours:02d}:{mins:02d}{day_marker}"

        # Créer un tableau complet de toutes les tranches de 30 minutes sur 23h
        all_bins = np.arange(0, 1380 + 30, 30)  # De 0 à 1380 minutes (23h), par tranches de 30 min
        all_times = pd.DataFrame({'time_bin': all_bins})

        # Fusionner avec les données réelles
        complete_profile = all_times.merge(intraday_profile, on='time_bin', how='left').fillna(0)
        complete_profile = complete_profile.sort_values('time_bin')

        # Calculer le PnL cumulatif moyen sur la journée
        complete_profile['cumulative_mean'] = complete_profile['mean'].cumsum()

        # Tracer le graphique des barres de PnL par tranche de 30 minutes
        bar_colors = ['g' if x >= 0 else 'r' for x in complete_profile['mean']]
        ax_left.bar(complete_profile['time_bin'], complete_profile['mean'],
                    width=25, alpha=0.7, color=bar_colors)

        # Ajouter la courbe du PnL cumulatif
        ax_twin = ax_left.twinx()  # Axe secondaire pour le PnL cumulatif
        ax_twin.plot(complete_profile['time_bin'], complete_profile['cumulative_mean'],
                     'b-', linewidth=2, alpha=0.8)
        ax_twin.set_ylabel('PnL Cumulatif Moyen ($)', fontsize=10)

        # Marquer le passage à minuit avec une ligne verticale
        midnight = 120  # 120 minutes après 22h = minuit
        ax_left.axvline(x=midnight, color='k', linestyle='--', alpha=0.5)
        ax_left.text(midnight + 5, ax_left.get_ylim()[1] * 0.9, 'Minuit',
                     fontsize=9, rotation=90, va='top')

        # Configurer l'axe X avec des intervalles de temps lisibles
        # Afficher une étiquette toutes les heures (pour économiser de l'espace)
        hour_ticks = np.arange(0, 1380 + 180, 180)  # Toutes les 3 heures
        ax_left.set_xticks(hour_ticks)
        ax_left.set_xticklabels([format_trading_time(t) for t in hour_ticks],
                                rotation=45, fontsize=8, ha='right')

        # Limiter l'affichage explicitement de 0 à 1380 minutes (23 heures)
        ax_left.set_xlim(0, 1380)

        # Ajouter le titre et les étiquettes
        ax_left.set_title('PnL Moyen par Tranche de 30min', fontsize=12)
        ax_left.set_xlabel('Heure de la journée de trading', fontsize=10)
        ax_left.set_ylabel('PnL Moyen par 30min ($)', fontsize=10)
        ax_left.grid(True, alpha=0.3)

        # 4.2 DROITE - Volume de trades et winrate par intervalle de 30 minutes
        ax_right = fig.add_subplot(gs[1])

        # Ne considérer que les trades qui ont été exécutés (PnL non nul)
        executed_trades = df_valid[df_valid['PnlAfterFiltering'] != 0]

        # Calculer les métriques par tranche de 30 minutes
        volume_by_time = executed_trades.groupby('time_bin').apply(
            lambda x: pd.Series({
                'Réussis': sum(x['PnlAfterFiltering'] > 0),
                'Échoués': sum(x['PnlAfterFiltering'] < 0),
                'Total': len(x)
            })
        ).reset_index()

        # Calculer le winrate
        volume_by_time['Winrate'] = (
                volume_by_time['Réussis'] / volume_by_time['Total'] * 100
        ).replace([np.inf, -np.inf, np.nan], 0)

        # Fusionner avec la grille complète (all_times) pour avoir toutes les tranches
        complete_volume = all_times.merge(volume_by_time, on='time_bin', how='left').fillna(0)

        # Créer le graphique à barres empilées pour le volume (sans label pour ne pas afficher de légende)
        ax_right.bar(
            complete_volume['time_bin'],
            complete_volume['Échoués'],
            width=25, alpha=0.7, color='r'
        )
        ax_right.bar(
            complete_volume['time_bin'],
            complete_volume['Réussis'],
            width=25, alpha=0.7, color='g',
            bottom=complete_volume['Échoués']
        )

        # Ajouter le winrate au-dessus de chaque barre (une seule fois)
        for i, row in complete_volume.iterrows():
            if row['Total'] > 0:  # n'afficher que si au moins un trade
                # Position du texte juste au-dessus de la barre empilée
                bar_top = row['Échoués'] + row['Réussis']
                ax_right.text(
                    row['time_bin'],
                    bar_top + 1,  # petit décalage vertical pour être lisible
                    f"{row['Winrate']:.0f}%",  # Winrate sans décimale
                    ha='center', va='bottom',
                    fontsize=8, rotation=90, color='black'
                )

        # Configurer l'axe X avec les mêmes intervalles que le graphique de gauche
        ax_right.set_xticks(hour_ticks)
        ax_right.set_xticklabels(
            [format_trading_time(t) for t in hour_ticks],
            rotation=45, fontsize=8, ha='right'
        )

        # Marquer le passage à minuit
        ax_right.axvline(x=midnight, color='k', linestyle='--', alpha=0.5)

        # Limiter l'affichage
        ax_right.set_xlim(0, 1380)

        # Ajouter titre/labels
        ax_right.set_title('Volume de Trades et Winrate par Tranche de 30min', fontsize=12)
        ax_right.set_xlabel('Heure de la journée de trading', fontsize=10)
        ax_right.set_ylabel('Nombre de Trades', fontsize=10)
        ax_right.grid(True, alpha=0.3)

        # Enlever la légende (pas de ax_right.legend(...) et pas de label=)
        # Ajouter une annotation sur le nombre de sessions
        session_count = df_valid['session_id'].nunique()
        ax_right.annotate(
            f"Basé sur {session_count} sessions",
            xy=(0.98, 0.02), xycoords='axes fraction',
            ha='right', va='bottom', fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7)
        )

    # Ajuster l'espacement pour accommoder le 4ème graphique
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.4, bottom=0.1)

    # Ajouter un titre global ajusté avec plus d'espace
    fig.suptitle('Analyse de Performance de Trading', fontsize=16, y=0.95)

    # Imprimer quelques statistiques clés
    if len(daily_pnl) > 0:
        print(
            f"Période d'analyse: {format_date(daily_pnl['session_date'].min())} à {format_date(daily_pnl['session_date'].max())}")
        final_pnl = daily_pnl['cumulative_pnl'].iloc[-1]
        print(f"PnL cumulé final: ${final_pnl:.2f}")
        print(f"Drawdown maximum: ${max_drawdown:.2f} le {format_date(max_dd_date)}")

        if max_pnl_day is not None:
            print(
                f"Meilleur jour: ${max_pnl_day['PnlAfterFiltering']:.2f} le {format_date(max_pnl_day['session_date'])}")

        if min_pnl_day is not None:
            print(f"Pire jour: ${min_pnl_day['PnlAfterFiltering']:.2f} le {format_date(min_pnl_day['session_date'])}")
    else:
        print("Aucune donnée disponible pour l'analyse")

    return fig


def correlation_matrices(X, y, figsize=(24, 10), save_path=None):
    """
    Calcule et affiche les matrices de corrélation de Pearson et Spearman,
    ainsi que les corrélations de chaque feature avec la variable cible.
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.gridspec as gridspec
    import matplotlib.colors as mcolors

    # Calcul des matrices de corrélation
    pearson_corr = X.corr(method='pearson')
    spearman_corr = X.corr(method='spearman')

    # Calcul des corrélations avec la cible
    pearson_target = pd.DataFrame({
        'target_correlation': X.corrwith(y, method='pearson')
    })

    spearman_target = pd.DataFrame({
        'target_correlation': X.corrwith(y, method='spearman')
    })

    # Configuration de la figure avec GridSpec pour avoir une disposition personnalisée
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 5, width_ratios=[2, 0.15, 2, 0.15, 0.05])

    # Définition de la colormap pour réutilisation
    cmap = sns.color_palette("coolwarm", as_cmap=True)

    # Matrice de Pearson
    ax_pearson = fig.add_subplot(gs[0])
    mask_pearson = np.triu(np.ones_like(pearson_corr, dtype=bool))
    sns.heatmap(pearson_corr, mask=mask_pearson, annot=True, fmt=".2f", cmap=cmap,
                square=True, linewidths=.5, annot_kws={"size": 9},
                cbar=False, ax=ax_pearson)
    ax_pearson.set_title('Matrice de corrélation de Pearson', fontsize=16)
    ax_pearson.set_xticklabels(ax_pearson.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    ax_pearson.set_yticklabels(ax_pearson.get_yticklabels(), fontsize=9)

    # Corrélation cible - Pearson
    ax_target_p = fig.add_subplot(gs[1])
    sns.heatmap(pearson_target, annot=True, fmt=".2f", cmap=cmap,
                linewidths=.5, annot_kws={"size": 9},
                cbar=False, ax=ax_target_p)
    ax_target_p.set_title('Corrélation\navec Y', fontsize=12)
    ax_target_p.set_xticklabels([])  # Pas de labels X
    ax_target_p.set_yticklabels([])  # Suppression des labels Y

    # Matrice de Spearman
    ax_spearman = fig.add_subplot(gs[2])
    mask_spearman = np.triu(np.ones_like(spearman_corr, dtype=bool))
    sns.heatmap(spearman_corr, mask=mask_spearman, annot=True, fmt=".2f", cmap=cmap,
                square=True, linewidths=.5, annot_kws={"size": 9},
                cbar=False, ax=ax_spearman)
    ax_spearman.set_title('Matrice de corrélation de Spearman', fontsize=16)
    ax_spearman.set_xticklabels(ax_spearman.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    ax_spearman.set_yticklabels([])  # Pas de labels Y

    # Corrélation cible - Spearman
    ax_target_s = fig.add_subplot(gs[3])
    sns.heatmap(spearman_target, annot=True, fmt=".2f", cmap=cmap,
                linewidths=.5, annot_kws={"size": 9},
                cbar=False, ax=ax_target_s)
    ax_target_s.set_title('Corrélation\navec Y', fontsize=12)
    ax_target_s.set_xticklabels([])  # Pas de labels X
    ax_target_s.set_yticklabels([])  # Pas de labels Y

    # Ajout d'une légende commune
    ax_cb = fig.add_subplot(gs[4])
    norm = mcolors.Normalize(vmin=-1, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=ax_cb)
    cbar.set_ticks([-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8])

    # Ajustement des espaces
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.25, wspace=0.05)

    # Sauvegarde de la figure si un chemin est spécifié
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

    # Affichage des corrélations triées avec la variable cible
    pearson_target_sorted = pearson_target.sort_values('target_correlation', ascending=False)
    spearman_target_sorted = spearman_target.sort_values('target_correlation', ascending=False)

    print("Corrélations de Pearson avec la variable cible (triées):")
    print(pearson_target_sorted)
    print("\nCorrélations de Spearman avec la variable cible (triées):")
    print(spearman_target_sorted)

    # Retourne les résultats dans un dictionnaire
    return {
        'pearson_matrix': pearson_corr,
        'spearman_matrix': spearman_corr,
        'pearson_target': pearson_target,
        'spearman_target': spearman_target
    }



import pandas as pd
import numpy as np


def analyze_indicator_winrates(df, indicator_list, class_column='class_binaire', no_trade_value=99):
    """
    Analyse le taux de réussite (winrate) pour chaque indicateur de la liste lorsqu'il est actif.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame contenant les indicateurs et la colonne de classification
    indicator_list : list
        Liste des noms des indicateurs à analyser
    class_column : str, default='class_binaire'
        Nom de la colonne contenant la classification (0=échec, 1=réussite, no_trade_value=pas de trade)
    no_trade_value : int, default=99
        Valeur indiquant l'absence de trade dans la colonne de classification

    Returns:
    --------
    pandas.DataFrame
        DataFrame contenant les résultats d'analyse pour chaque indicateur
    """
    # Filtrer le dataframe pour ne conserver que les lignes où un trade a été effectué
    df_trades = df[df[class_column] != no_trade_value]

    # Calculer le winrate global (sans activation des indicateurs)
    total_trades = len(df_trades)
    successful_trades = len(df_trades[df_trades[class_column] == 1])
    global_winrate = successful_trades / total_trades * 100 if total_trades > 0 else 0

    print(f"Winrate global (tous trades confondus): {global_winrate:.2f}% ({successful_trades}/{total_trades})")
    print("\nWinrate par indicateur activé (valeur = 1):")
    print("-" * 95)
    print(
        f"{'Indicateur':<30} {'Winrate':<10} {'Trades réussis':<15} {'Trades totaux':<15} {'Diff/Global':<15} {'% des trades':<15}")
    print("-" * 95)

    # Calculer le winrate pour chaque indicateur lorsqu'il est activé
    results = []
    for indicator in indicator_list:
        # Filtrer pour les cas où l'indicateur est activé (= 1)
        df_indicator_active = df_trades[df_trades[indicator] == 1]

        # Compter les trades et les succès
        indicator_trades = len(df_indicator_active)
        indicator_success = len(df_indicator_active[df_indicator_active[class_column] == 1])

        # Calculer le winrate de l'indicateur
        indicator_winrate = indicator_success / indicator_trades * 100 if indicator_trades > 0 else 0

        # Calculer la différence avec le winrate global
        winrate_diff = indicator_winrate - global_winrate

        # Calculer le pourcentage de trades pris par rapport au total
        trades_percentage = indicator_trades / total_trades * 100 if total_trades > 0 else 0

        # Stocker les résultats
        results.append({
            'Indicateur': indicator,
            'Winrate': indicator_winrate,
            'Trades_réussis': indicator_success,
            'Trades_totaux': indicator_trades,
            'Diff_Global': winrate_diff,
            'Pourcentage_Trades': trades_percentage
        })

        # Afficher les résultats
        print(
            f"{indicator:<30} {indicator_winrate:.2f}% {indicator_success:<15} {indicator_trades:<15} {winrate_diff:+.2f}% {trades_percentage:.2f}%")

    # Convertir les résultats en dataframe et trier par winrate décroissant
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Winrate', ascending=False)

    print("\nIndicateurs classés par winrate décroissant:")
    print("-" * 95)
    for _, row in results_df.iterrows():
        print(
            f"{row['Indicateur']:<30} {row['Winrate']:.2f}% {row['Trades_réussis']:<15} {row['Trades_totaux']:<15} {row['Diff_Global']:+.2f}% {row['Pourcentage_Trades']:.2f}%")

    return results_df

#
# def print_evaluation_results(results, indicator_type, params):
#     """
#     Affiche les résultats de l'évaluation de manière formatée.
#
#     Parameters:
#     -----------
#     results : dict
#         Dictionnaire contenant les métriques d'évaluation
#     indicator_type : str
#         Type d'indicateur évalué
#     params : dict
#         Paramètres utilisés pour l'évaluation
#     """
#     # Afficher les paramètres
#     print("\n📊 PARAMÈTRES UTILISÉS:")
#     for key, value in params.items():
#         print(f"  {key}: {value}")
#
#     # Afficher les métriques générales
#     print("\n📈 RÉSULTATS DE L'ÉVALUATION:")
#     print(f"  Nombre total d'échantillons: {results.get('total_samples', 0)}")
#
#     # Adapter le nom des bins selon l'indicateur
#     bin0_name = "Survente"
#     bin1_name = "Surachat"
#
#     if indicator_type.lower() == "mfi_divergence":
#         bin0_name = "Divergence Haussière"
#         bin1_name = "Divergence Baissière"
#     elif indicator_type.lower() == "regression_r2":
#         bin0_name = "Volatilité Basse"
#         bin1_name = "Volatilité Haute"
#     elif indicator_type.lower() == "regression_std":
#         bin0_name = "Écart-type Faible"
#         bin1_name = "Écart-type Élevé"
#     elif indicator_type.lower() == "regression_slope":
#         bin0_name = "Volatilité dans les Extrêmes"
#         bin1_name = "Volatilité Modérée"
#     elif indicator_type.lower() == "atr":
#         bin0_name = "ATR Faible"
#         bin1_name = "ATR Modéré"
#     elif indicator_type.lower() == "vwap":
#         bin0_name = "Distance VWAP Extrême"
#         bin1_name = "Distance VWAP Modérée"
#     elif indicator_type.lower() == "percent_bb_simu":
#         bin0_name = "%B Extrême"
#         bin1_name = "%B Modéré"
#     elif indicator_type.lower() == "zscore":
#         bin0_name = "Z-Score Extrême"
#         bin1_name = "Z-Score Modéré"
#
#
#     # Afficher les métriques de bin 0 si disponibles
#     if results.get('bin_0_samples', 0) > 0:
#         print(f"\n  Statistiques du bin 0 ({bin0_name}):")
#         print(f"    • Win Rate: {results.get('bin_0_win_rate', 0):.4f}")
#         print(f"    • Nombre d'échantillons: {results.get('bin_0_samples', 0)}")
#         print(f"    • Nombre de trades gagnants: {results.get('oversold_success_count', 0)}")
#         print(f"    • Pourcentage des données: {results.get('bin_0_pct', 0):.2%}")
#
#     # Afficher les métriques de bin 1 si disponibles
#     if results.get('bin_1_samples', 0) > 0:
#         print(f"\n  Statistiques du bin 1 ({bin1_name}):")
#         print(f"    • Win Rate: {results.get('bin_1_win_rate', 0):.4f}")
#         print(f"    • Nombre d'échantillons: {results.get('bin_1_samples', 0)}")
#         print(f"    • Nombre de trades gagnants: {results.get('overbought_success_count', 0)}")
#         print(f"    • Pourcentage des données: {results.get('bin_1_pct', 0):.2%}")
#
#     # Afficher le spread si les deux bins sont disponibles
#     if results.get('bin_0_samples', 0) > 0 and results.get('bin_1_samples', 0) > 0:
#         print(f"\n  Spread (différence de win rate): {results.get('bin_spread', 0):.4f}")
#
#     # Comparaison avec les attentes
#     print("\n🔍 ANALYSE DES RÉSULTATS:")
#
#     # Vérifier si chaque métrique correspond aux attentes
#     if results.get('bin_0_samples', 0) > 0:
#         bin_0_wr = results.get('bin_0_win_rate', 0)
#         expected_bin_0_wr = MAX_bin_0_win_rate if 'MAX_bin_0_win_rate' in globals() else 0.47
#
#         if bin_0_wr <= expected_bin_0_wr:
#             print(f"  ✅ Bin 0: Win rate ({bin_0_wr:.4f}) conforme aux attentes (≤ {expected_bin_0_wr:.4f})")
#         else:
#             print(f"  ❌ Bin 0: Win rate ({bin_0_wr:.4f}) supérieur aux attentes (> {expected_bin_0_wr:.4f})")
#
#     if results.get('bin_1_samples', 0) > 0:
#         bin_1_wr = results.get('bin_1_win_rate', 0)
#         expected_bin_1_wr = MIN_bin_1_win_rate if 'MIN_bin_1_win_rate' in globals() else 0.53
#
#         if bin_1_wr >= expected_bin_1_wr:
#             print(f"  ✅ Bin 1: Win rate ({bin_1_wr:.4f}) conforme aux attentes (≥ {expected_bin_1_wr:.4f})")
#         else:
#             print(f"  ❌ Bin 1: Win rate ({bin_1_wr:.4f}) inférieur aux attentes (< {expected_bin_1_wr:.4f})")
#
#     # Conclusion
#     print("\n📝 CONCLUSION:")
#     indicator_name = indicator_type.capitalize().replace("_", " ")
#
#     if (results.get('bin_0_samples', 0) > 0 and results.get('bin_0_win_rate', 0) <= expected_bin_0_wr) or \
#             (results.get('bin_1_samples', 0) > 0 and results.get('bin_1_win_rate', 0) >= expected_bin_1_wr):
#         print(f"  L'indicateur {indicator_name} montre une bonne généralisation sur le jeu de données de test.")
#         if results.get('bin_spread', 0) > 0.08:
#             print(f"  Le spread ({results.get('bin_spread', 0):.4f}) est significatif, ce qui est très positif.")
#         else:
#             print(f"  Le spread ({results.get('bin_spread', 0):.4f}) pourrait être amélioré pour plus de robustesse.")
#     else:
#         print(f"  L'indicateur {indicator_name} montre des difficultés à généraliser sur le jeu de données de test.")
#         print("  Considérez de réoptimiser avec plus de données ou d'ajuster les contraintes.")
#



XTICKREVERAL_TICKPRICE = 10  # Nombre de ticks dans la zone above
PERDIOD_ATR_SESSION_ANALYSE=15

def print_enhanced_summary_statistics(valid_results):
    """
    Affiche le résumé statistique enrichi avec ATR et contrats extrêmes
    """
    print("\n" + "=" * 80)
    print("📋 RÉSUMÉ STATISTIQUE GLOBAL ENRICHI (ATR + CONTRATS EXTRÊMES)")
    print("=" * 80)

    for df_name, data in valid_results.items():
        print(f"\n📊 {df_name.upper()}")
        print("-" * 60)
        print(f"📅 Période: {data['date_range'][0]} → {data['date_range'][1]}")
        print(f"🎯 Nombre de sessions: {data['total_sessions']}")
        print(f"🕯️  Total bougies: {data['total_candles']:,}")
        print(f"⏱️  Durée moyenne globale: {data['avg_duration_overall']:.2f}s")
        print(f"📊 Bougies/session: {data['avg_candles_per_session']:.1f}")

        # Statistiques des sessions
        session_stats = data['session_stats']
        print(
            f"📈 Durée min/max par session: {session_stats['duration_mean'].min():.1f}s / {session_stats['duration_mean'].max():.1f}s")
        print(
            f"📊 Bougies min/max par session: {session_stats['candle_count'].min()} / {session_stats['candle_count'].max()}")

        # Statistiques de volume
        vol_stats = data['volume_stats']
        print(f"📈 Volume moyen/session: {vol_stats['avg_volume_per_session']:.2f}")
        print(f"💰 Volume total: {vol_stats['total_volume']:.2f}")
        print(f"🔗 Corrélation durée-volume: {vol_stats['duration_volume_correlation']:.3f}")

        # NOUVEAU: Statistiques ATR
        atr_stats = data['atr_stats']
        print(f"\n📊 ANALYSE ATR 10 PÉRIODES:")
        print(f"   🎯 Sessions avec ATR valide: {atr_stats['sessions_with_atr']}/{data['total_sessions']}")
        if not pd.isna(atr_stats['atr_overall_mean']):
            print(f"   📈 ATR moyen global: {atr_stats['atr_overall_mean']:.4f}")
            print(f"   📏 ATR min/max: {atr_stats['atr_min']:.4f} / {atr_stats['atr_max']:.4f}")
            print(f"   📊 Écart-type ATR: {atr_stats['atr_std']:.4f}")
            print(f"   🔗 Corrélation ATR-Durée: {atr_stats['atr_duration_correlation']:.3f}")
        else:
            print(f"   ⚠️  Pas de données ATR valides (sessions trop courtes)")

        # NOUVEAU: Statistiques contrats extrêmes
        extreme_stats = data['extreme_contracts_stats']
        print(f"\n🎯 ANALYSE CONTRATS EXTRÊMES:")
        print(
            f"   📊 Sessions avec contrats extrêmes: {extreme_stats['sessions_with_extreme_contracts']}/{data['total_sessions']}")

        if not pd.isna(extreme_stats['extreme_with_zeros_mean']):
            print(f"   📈 Contrats extrêmes (avec zéros): {extreme_stats['extreme_with_zeros_mean']:.4f}")
            print(f"   🎯 Contrats extrêmes (sans zéros): {extreme_stats['extreme_without_zeros_mean']:.4f}")
            print(f"   📊 Ratio moyen contrats extrêmes: {extreme_stats['extreme_ratio_mean']:.3f}")
            print(f"   📈 Total contrats extrêmes: {extreme_stats['extreme_count_total']}")
            print(f"   🔗 Corrélation avec-zéros-Durée: {extreme_stats['extreme_with_zeros_duration_corr']:.3f}")
            print(f"   🔗 Corrélation sans-zéros-Durée: {extreme_stats['extreme_without_zeros_duration_corr']:.3f}")
            if not pd.isna(extreme_stats['extreme_atr_correlation']):
                print(f"   🔗 Corrélation contrats-ATR: {extreme_stats['extreme_atr_correlation']:.3f}")
        else:
            print(f"   ⚠️  Pas de données de contrats extrêmes disponibles")

        # Insights d'interprétation
        print(f"\n💡 INSIGHTS:")

        # Analyse de l'évolution temporelle
        if len(session_stats) > 1:
            duration_trend = np.polyfit(range(len(session_stats)), session_stats['duration_mean'], 1)[0]
            trend_text = "📈 amélioration" if duration_trend < 0 else "📉 dégradation"
            print(f"   ⏱️  Tendance durée: {trend_text} ({duration_trend:.3f}s/session)")

        # Analyse de la stabilité
        duration_cv = session_stats['duration_mean'].std() / session_stats['duration_mean'].mean()
        stability = "🔹 stable" if duration_cv < 0.2 else "🔸 variable" if duration_cv < 0.5 else "🔴 instable"
        print(f"   📊 Stabilité: {stability} (CV={duration_cv:.3f})")

        # Efficacité ATR
        if not pd.isna(atr_stats['atr_duration_correlation']):
            atr_efficiency = "📈 positive" if atr_stats['atr_duration_correlation'] < -0.3 else "📊 neutre" if abs(
                atr_stats['atr_duration_correlation']) < 0.3 else "📉 négative"
            print(f"   🎯 Efficacité ATR: {atr_efficiency}")

        # Sélectivité des contrats
        if not pd.isna(extreme_stats['extreme_ratio_mean']):
            selectivity = "🎯 très sélective" if extreme_stats['extreme_ratio_mean'] < 0.1 else "📊 sélective" if \
            extreme_stats['extreme_ratio_mean'] < 0.3 else "📈 peu sélective"
            print(f"   🔍 Sélectivité contrats: {selectivity} ({extreme_stats['extreme_ratio_mean']:.1%})")

import numpy as np

def _percentiles(series, q=(25, 50, 75)):
    """Renvoie un dict {p25, p50, p75} – NaN si < 2 points"""
    s = series.dropna().to_numpy()
    if s.size < 2:
        return {f"p{qi}": np.nan for qi in q}
    pct = np.percentile(s, q)
    return {f"p{qi}": pct[i] for i, qi in enumerate(q)}
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Fonction principale
# ──────────────────────────────────────────────────────────────


# Fonction d'utilisation simplifiée











