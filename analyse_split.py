import pandas as pd
import numpy as np
import os
import time
from numba import jit
from standardFunc_sauv import load_data
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration des périodes de trading
TIME_PERIODS = {
    'Opening': (0, 80),
    'AsieEurope': (80, 800),
    'US': (800, 24 * 60)
}

# Variable globale pour contrôler l'affichage des plots
SHOW_PLOTS = False


def timestamp_to_date_utc_(timestamp):
    date_format = "%Y-%m-%d %H:%M:%S"
    if isinstance(timestamp, pd.Series):
        return timestamp.apply(lambda x: time.strftime(date_format, time.gmtime(x)))
    else:
        return time.strftime(date_format, time.gmtime(timestamp))


@jit(nopython=True)
def calculate_trade_stats(trade_dir, trade_result):
    """Calcule les statistiques de trades pour un segment donné"""
    success = 0
    failure = 0
    for d, r in zip(trade_dir, trade_result):
        if d == 1:  # Long trades
            if r == 1:
                success += 1
            elif r == -1:
                failure += 1
    return success, failure


@jit(nopython=True)
def calculate_split_stats(trade_dir, trade_result):
    """Calcule les statistiques complètes pour un split"""
    long_success = 0
    long_failure = 0
    short_success = 0
    short_failure = 0
    for d, r in zip(trade_dir, trade_result):
        if d == 1:  # Long trades
            if r == 1:
                long_success += 1
            elif r == -1:
                long_failure += 1
        elif d == -1:  # Short trades
            if r == 1:
                short_success += 1
            elif r == -1:
                short_failure += 1
    return long_success, long_failure, short_success, short_failure


@jit(nopython=True)
def calculate_ratios(success, failure):
    """Calcule les ratios avec numba"""
    return float('inf') if failure == 0 else success / failure


@jit(nopython=True)
def calculate_global_stats(trade_dir, trade_result):
    """Calcule les statistiques globales pour tout le dataset"""
    long_success = 0
    long_failure = 0
    short_success = 0
    short_failure = 0
    for d, r in zip(trade_dir, trade_result):
        if d == 1:  # Long trades
            if r == 1:
                long_success += 1
            elif r == -1:
                long_failure += 1
        elif d == -1:  # Short trades
            if r == 1:
                short_success += 1
            elif r == -1:
                short_failure += 1
    long_ratio = float('inf') if long_failure == 0 else long_success / long_failure
    short_ratio = float('inf') if short_failure == 0 else short_success / short_failure
    return long_success, long_failure, short_success, short_failure, long_ratio, short_ratio


def calculate_winrates(df, trade_type='all'):
    """Calcule les winrates pour l'ensemble des trades ou par type"""
    if trade_type == 'long':
        df_filtered = df[df['tradeDir'] == 1]
    elif trade_type == 'short':
        df_filtered = df[df['tradeDir'] == -1]
    else:
        df_filtered = df
    df_filtered = df_filtered[df_filtered['tradeResult'] != 99]
    if len(df_filtered) == 0:
        return 0
    wins = len(df_filtered[df_filtered['tradeResult'] == 1])
    total = len(df_filtered)
    return (wins / total * 100)


def analyze_period_winrates(df, period_name, start_time, end_time):
    """Analyse détaillée des winrates pour une période spécifique"""
    period_df = df[(df['deltaTimestampOpening'] >= start_time) &
                   (df['deltaTimestampOpening'] < end_time)]

    total_trades = len(period_df[period_df['tradeResult'] != 99])
    if total_trades == 0:
        return None

    return {
        'period': period_name,
        'trades_count': total_trades,
        'global_winrate': calculate_winrates(period_df),
        'long_winrate': calculate_winrates(period_df, 'long'),
        'short_winrate': calculate_winrates(period_df, 'short'),
        'long_trades': len(period_df[(period_df['tradeDir'] == 1) & (period_df['tradeResult'] != 99)]),
        'short_trades': len(period_df[(period_df['tradeDir'] == -1) & (period_df['tradeResult'] != 99)]),
        'avg_pnl': period_df[period_df['tradeResult'] != 99]['trade_pnl'].mean()
    }


def print_period_stats(stats):
    """Affichage formaté des statistiques de période"""
    if stats is None:
        print("Pas de trades dans cette période")
        return

    print(f"\n{'-' * 20} {stats['period']} {'-' * 20}")
    print(f"Nombre total de trades: {stats['trades_count']}")
    print(f"Winrate global: {stats['global_winrate']:.2f}%")
    print(f"Long trades: {stats['long_trades']} (Winrate: {stats['long_winrate']:.2f}%)")
    print(f"Short trades: {stats['short_trades']} (Winrate: {stats['short_winrate']:.2f}%)")
    print(f"PnL moyen: {stats['avg_pnl']:.2f}")


def analyze_volume_and_trades_distribution(df, title_prefix="", show_plots=False):
    """Analyse complète de la distribution des volumes et trades par période"""
    total_volume = df['volume'].sum()
    total_trades = len(df[df['tradeResult'] != 99])

    distribution_stats = {
        'total': {
            'volume': total_volume,
            'trades': total_trades,
            'volume_per_trade': total_volume / total_trades if total_trades > 0 else 0
        }
    }

    print(f"\n{title_prefix}Analyse de la distribution des volumes et trades:")
    print("=" * 80)
    print(f"Volume total: {total_volume:,.0f}")
    print(f"Nombre total de trades: {total_trades:,}")
    print(f"Volume moyen par trade: {(total_volume / total_trades if total_trades > 0 else 0):,.2f}")

    for period_name, (start_time, end_time) in TIME_PERIODS.items():
        period_df = df[(df['deltaTimestampOpening'] >= start_time) &
                       (df['deltaTimestampOpening'] < end_time)]
        period_volume = period_df['volume'].sum()
        period_trades = len(period_df[period_df['tradeResult'] != 99])

        stats = {
            'volume': period_volume,
            'volume_pct': (period_volume / total_volume * 100),
            'trades': period_trades,
            'trades_pct': (period_trades / total_trades * 100) if total_trades > 0 else 0,
            'volume_per_trade': period_volume / period_trades if period_trades > 0 else 0
        }

        distribution_stats[period_name] = stats

        #print(f"\n{period_name}:")
        #print(f"  Volume: {period_volume:,.0f} ({stats['volume_pct']:.2f}%)")
        #print(f"  Trades: {period_trades:,} ({stats['trades_pct']:.2f}%)")
        #print(f"  Volume moyen par trade: {stats['volume_per_trade']:,.2f}")

    if SHOW_PLOTS:
        create_volume_trade_visualizations(distribution_stats, title_prefix)

    return distribution_stats


def create_volume_trade_visualizations(stats, title_prefix="", show_plots=False):
    """Crée les visualisations pour l'analyse des volumes et trades"""
    if not SHOW_PLOTS:
        return

    periods = [period for period in stats.keys() if period != 'total']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Camembert des volumes
    volume_sizes = [stats[period]['volume_pct'] for period in periods]
    volume_labels = [f"{period}\n{stats[period]['volume_pct']:.1f}%" for period in periods]
    ax1.pie(volume_sizes, labels=volume_labels, autopct='%1.1f%%', startangle=90)
    ax1.set_title(f'{title_prefix}Distribution du Volume par Période')

    # Barres comparatives volumes vs trades
    x = np.arange(len(periods))
    width = 0.35

    volume_pcts = [stats[period]['volume_pct'] for period in periods]
    trade_pcts = [stats[period]['trades_pct'] for period in periods]

    rects1 = ax2.bar(x - width / 2, volume_pcts, width, label='% Volume')
    rects2 = ax2.bar(x + width / 2, trade_pcts, width, label='% Trades')

    ax2.set_ylabel('Pourcentage')
    ax2.set_title(f'{title_prefix}Comparaison Volume vs Trades')
    ax2.set_xticks(x)
    ax2.set_xticklabels(periods)
    ax2.legend()

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax2.annotate(f'{height:.1f}%',
                         xy=(rect.get_x() + rect.get_width() / 2, height),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    if SHOW_PLOTS:
        plt.show()


def create_comparative_volume_analysis(all_stats, split_indices, show_plots=False):
    """Crée une visualisation comparative des volumes entre les splits"""
    if not SHOW_PLOTS:
        return

    periods = [period for period in all_stats[0].keys() if period != 'total']
    n_splits = len(all_stats)

    fig, ax = plt.subplots(figsize=(15, 8))
    x = np.arange(len(periods))
    width = 0.8 / n_splits

    for i, (stats, split_idx) in enumerate(zip(all_stats, split_indices)):
        volume_pcts = [stats[period]['volume_pct'] for period in periods]
        offset = width * i - width * (n_splits - 1) / 2
        bars = ax.bar(x + offset, volume_pcts, width, label=f'Split {split_idx}')

    ax.set_ylabel('Pourcentage du Volume')
    ax.set_title('Distribution du Volume par Période et par Split')
    ax.set_xticks(x)
    ax.set_xticklabels(periods)
    ax.legend()

    plt.tight_layout()
    if SHOW_PLOTS:
        plt.show()


def plot_winrates(global_stats, title="Analyse des Winrates", show_plots=False):
    """Création des visualisations des winrates"""
    if not SHOW_PLOTS:
        return

    periods = [stat['period'] for stat in global_stats if stat is not None]
    data = {
        'Long': [stat['long_winrate'] for stat in global_stats if stat is not None],
        'Short': [stat['short_winrate'] for stat in global_stats if stat is not None],
        'Global': [stat['global_winrate'] for stat in global_stats if stat is not None]
    }

    plt.figure(figsize=(15, 8))
    x = np.arange(len(periods))
    width = 0.25

    plt.bar(x - width, data['Long'], width, label='Long', color='green', alpha=0.6)
    plt.bar(x, data['Short'], width, label='Short', color='red', alpha=0.6)
    plt.bar(x + width, data['Global'], width, label='Global', color='blue', alpha=0.6)

    plt.xlabel('Périodes de trading')
    plt.ylabel('Winrate (%)')
    plt.title(title)
    plt.xticks(x, periods)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    if SHOW_PLOTS:
        plt.show()


def print_split_metrics(split_df, global_long_ratio, global_short_ratio, ratio_up, ratio_down):
    """Affiche les métriques détaillées d'un split avec les ratios"""
    # Calcul des limites
    long_upper_limit = global_long_ratio * ratio_up
    long_lower_limit = global_long_ratio * ratio_down
    short_upper_limit = global_short_ratio * ratio_up
    short_lower_limit = global_short_ratio * ratio_down

    # Calcul des ratios du split
    trade_dir = split_df['tradeDir'].values
    trade_result = split_df['tradeResult'].values
    long_success, long_failure, short_success, short_failure = calculate_split_stats(trade_dir, trade_result)

    long_ratio = calculate_ratios(long_success, long_failure)
    short_ratio = calculate_ratios(short_success, short_failure)

    print("\nMétriques de trading pour le split:")
    print("=" * 80)
    print(f"Long trades:")
    print(f"  Réussis/Échecs: {long_success}/{long_failure}")
    print(f"  Ratio: {long_ratio:.2f} (Limites: {long_lower_limit:.2f} - {long_upper_limit:.2f})")
    print(
        f"  Status: {'HORS LIMITES' if long_ratio < long_lower_limit or long_ratio > long_upper_limit else 'Dans les limites'}")

    print(f"\nShort trades:")
    print(f"  Réussis/Échecs: {short_success}/{short_failure}")
    print(f"  Ratio: {short_ratio:.2f} (Limites: {short_lower_limit:.2f} - {short_upper_limit:.2f})")
    print(
        f"  Status: {'HORS LIMITES' if short_ratio < short_lower_limit or short_ratio > short_upper_limit else 'Dans les limites'}")

def analyze_splits(nb_split=500, ratio_up=1.5, ratio_down=0.5, show_plots=False):
    """Analyse principale des splits avec winrates, ratios et volumes"""
    # Configuration et chargement des données
    FILE_NAME = "Step1_270324_281124_4TicksRev_6.csv"
    DIRECTORY_PATH = r"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra chart\xTickReversal\simu\4_0_5TP_1SL_newBB\merge"
    file_path = os.path.join(DIRECTORY_PATH, FILE_NAME)

    print("Chargement des données...")
    initial_df = load_data(file_path)
    print("Taille initiale du DataFrame:", len(initial_df))

    # Analyse globale
    print("\nAnalyse globale:")
    print("=" * 80)
    global_volume_stats = analyze_volume_and_trades_distribution(initial_df, "Global - ", show_plots)

    # Suppression des lignes où tradeResult = 99 pour les autres analyses
    df = initial_df[initial_df['tradeResult'] != 99].copy()
    print("\nTaille après suppression des tradeResult = 99:", len(df))

    # Statistiques globales PnL
    df_success = df[df['tradeResult'] == 1]
    df_failure = df[df['tradeResult'] == -1]

    mean_pnl_success = df_success['trade_pnl'].mean()
    mean_pnl_failure = df_failure['trade_pnl'].mean()

    zero_pnl_success = len(df_success[df_success['trade_pnl'] == 0])
    zero_pnl_failure = len(df_failure[df_failure['trade_pnl'] == 0])

    print("\nMoyennes Globales des PnL:")
    print(f"Moyenne PnL Ok: {mean_pnl_success:.2f} (dont {zero_pnl_success} trades avec PnL = 0)")
    print(f"Moyenne PnL Ko: {mean_pnl_failure:.2f} (dont {zero_pnl_failure} trades avec PnL = 0)")

    # Calcul des statistiques globales de trading
    global_trade_dir = df['tradeDir'].values
    global_trade_result = df['tradeResult'].values
    global_stats = calculate_global_stats(global_trade_dir, global_trade_result)
    global_long_s, global_long_f, global_short_s, global_short_f, global_long_ratio, global_short_ratio = global_stats

    print("\nStatistiques globales de trading:")
    print(f"Long trades  - Réussis: {global_long_s}, Échoués: {global_long_f}, Ratio: {global_long_ratio:.2f}")
    print(f"Short trades - Réussis: {global_short_s}, Échoués: {global_short_f}, Ratio: {global_short_ratio:.2f}")

    # Définir les limites pour l'analyse des splits
    long_upper_limit = global_long_ratio * ratio_up
    long_lower_limit = global_long_ratio * ratio_down
    short_upper_limit = global_short_ratio * ratio_up
    short_lower_limit = global_short_ratio * ratio_down

    print(f"\nLimites pour l'analyse des splits:")
    print(f"Long ratio  - Limites: [{long_lower_limit:.2f}, {long_upper_limit:.2f}]")
    print(f"Short ratio - Limites: [{short_lower_limit:.2f}, {short_upper_limit:.2f}]")

    # Analyse des winrates globaux par période
    print("\nAnalyse des winrates par période:")
    print("=" * 80)
    global_period_stats = []
    for period_name, (start_time, end_time) in TIME_PERIODS.items():
        period_stats = analyze_period_winrates(df, period_name, start_time, end_time)
        global_period_stats.append(period_stats)
        #print_period_stats(period_stats)

    # Visualisation des winrates globaux
    plot_winrates(global_period_stats, "Winrates Globaux par Période", show_plots)

    # Analyse par split
    print(f"\nAnalyse détaillée par {nb_split} splits:")
    print("=" * 80)

    split_size = len(df) // nb_split
    splits_hors_limites = []
    all_split_volume_stats = []
    split_indices = []

    for i in range(nb_split):
        start_idx = i * split_size
        end_idx = start_idx + split_size if i < nb_split - 1 else len(df)
        split_df = df.iloc[start_idx:end_idx]

        # Dates du split
        start_date = timestamp_to_date_utc_(split_df['timeStampOpening'].iloc[0])
        end_date = timestamp_to_date_utc_(split_df['timeStampOpening'].iloc[-1])

        print(f"\nSplit {i + 1} ({start_date} - {end_date}):")
        print("-" * 80)

        # Analyse des volumes pour ce split
        split_volume_stats = analyze_volume_and_trades_distribution(split_df, f"Split {i + 1} - ", show_plots)
        all_split_volume_stats.append(split_volume_stats)
        split_indices.append(i + 1)

        # Analyse des winrates par période pour ce split
        print("\nWinrates par période pour ce split:")
        split_period_stats = []
        for period_name, (start_time, end_time) in TIME_PERIODS.items():
            period_stats = analyze_period_winrates(split_df, period_name, start_time, end_time)
            split_period_stats.append(period_stats)
            #print_period_stats(period_stats)

        # AJOUT ICI - après l'analyse des winrates et avant la vérification des limites
        print_split_metrics(split_df, global_long_ratio, global_short_ratio, ratio_up, ratio_down)

        # Statistiques classiques du split
        trade_dir = split_df['tradeDir'].values
        trade_result = split_df['tradeResult'].values
        long_success, long_failure, short_success, short_failure = calculate_split_stats(trade_dir, trade_result)

        # Calcul des ratios
        long_ratio = calculate_ratios(long_success, long_failure)
        short_ratio = calculate_ratios(short_success, short_failure)

        # Vérification des limites
        hors_limites = False
        raisons = []

        if long_ratio != float('inf'):
            if long_ratio > long_upper_limit:
                raisons.append(f"Ratio long trop élevé: {long_ratio:.2f} > {long_upper_limit:.2f}")
                hors_limites = True
            elif long_ratio < long_lower_limit:
                raisons.append(f"Ratio long trop bas: {long_ratio:.2f} < {long_lower_limit:.2f}")
                hors_limites = True

        if short_ratio != float('inf'):
            if short_ratio > short_upper_limit:
                raisons.append(f"Ratio short trop élevé: {short_ratio:.2f} > {short_upper_limit:.2f}")
                hors_limites = True
            elif short_ratio < short_lower_limit:
                raisons.append(f"Ratio short trop bas: {short_ratio:.2f} < {short_lower_limit:.2f}")
                hors_limites = True

        if hors_limites:
            splits_hors_limites.append({
                'split': i + 1,
                'periode': f"{start_date} à {end_date}",
                'ratios': {'long': long_ratio, 'short': short_ratio},
                'raisons': raisons
            })

        # Analyse des winrates par période pour ce split
        print("\nWinrates par période pour ce split:")
        split_period_stats = []
        for period_name, (start_time, end_time) in TIME_PERIODS.items():
            period_stats = analyze_period_winrates(split_df, period_name, start_time, end_time)
            split_period_stats.append(period_stats)
            #print_period_stats(period_stats)

        if any(stat is not None for stat in split_period_stats):
            plot_winrates(split_period_stats, f"Winrates du Split {i + 1}", show_plots)

    # Création du graphique comparatif des volumes entre splits
    create_comparative_volume_analysis(all_split_volume_stats, split_indices, show_plots)

    # Synthèse des splits hors limites
    print("\nSynthèse des splits hors limites:")
    print("=" * 80)
    if splits_hors_limites:
        for split_info in splits_hors_limites:
            print(f"\nSplit {split_info['split']}")
            print(f"Période: {split_info['periode']}")
            print(f"Ratio long: {split_info['ratios']['long']:.2f}")
            print(f"Ratio short: {split_info['ratios']['short']:.2f}")
            print("Raisons:")
            for raison in split_info['raisons']:
                print(f"- {raison}")
    else:
        print("Aucun split hors limites détecté")


if __name__ == "__main__":
    # Demande à l'utilisateur s'il veut voir les graphiques
    user_input = input(
        "Appuyez sur 'b' + Entrée pour afficher les graphiques, ou juste Entrée pour continuer sans graphiques: ").strip()
    show_plots = user_input.lower() == 'b'

    print(f"Mode graphique: {'activé' if show_plots else 'désactivé'}")

    # Lancement de l'analyse avec le paramètre
    analyze_splits(100, ratio_up=1.5, ratio_down=0.6, show_plots=show_plots)