from stats_sc.standard_stat_sc import *
from func_standard import *
from colorama import Fore, Style
import seaborn as sns
import numpy as np
import pandas as pd
import os, sys, platform, io
from pathlib import Path
from contextlib import redirect_stdout
from collections import Counter
from Tools.func_features_preprocessing import *

# ────────────────────────────────────────────────────────────────────────────────
# Paramètres
# ────────────────────────────────────────────────────────────────────────────────

ENV = detect_environment()
DIR = "5_0_5TP_6SL"



# ────────────────────────────────────────────────────────────────────────────────
# Construction du chemin de base selon l'OS
# ────────────────────────────────────────────────────────────────────────────────
if platform.system() != "Darwin":
    DIRECTORY_PATH = Path(
        rf"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra_chart\xTickReversal\simu\{DIR}\merge")
else:
    DIRECTORY_PATH = Path(f"/Users/aurelienlachaud/Documents/trading_local/{DIR}/merge")

# ────────────────────────────────────────────────────────────────────────────────
# Base des noms de fichiers
# ────────────────────────────────────────────────────────────────────────────────
BASE = "Step5_5_0_5TP_6SL_150525_300625_extractOnlyFullSession_Only"
SPLIT_SUFFIX = "_feat"
DIRECTION = "Short"

# ────────────────────────────────────────────────────────────────────────────────
# Construction des chemins de fichiers
# ────────────────────────────────────────────────────────────────────────────────
FILE_NAME = lambda split: f"{BASE}{DIRECTION}{split}.csv"

FILE_PATH_TRAIN  = DIRECTORY_PATH / FILE_NAME(SPLIT_SUFFIX)
FILE_PATH_TEST   = DIRECTORY_PATH / FILE_NAME(SPLIT_SUFFIX)
FILE_PATH_VAL1   = DIRECTORY_PATH / FILE_NAME(SPLIT_SUFFIX)
FILE_PATH_VAL    = DIRECTORY_PATH / FILE_NAME(SPLIT_SUFFIX)
FILE_PATH_UNSEEN = DIRECTORY_PATH / FILE_NAME(SPLIT_SUFFIX)


# ────────────────────────────────────────────────────────────────────────────────
# Filtre global pour tous les algos
# ────────────────────────────────────────────────────────────────────────────────
GLOBAL_MICRO_FILTER = {
    'sc_volume_perTick': [
        {'type': 'between', 'min': 10, 'max': 600, 'active': True}
    ],
    'sc_meanVol_perTick_over1': [
        {'type': 'between', 'min': 25, 'max': 65, 'active': True}
    ],

    'sc_volRev_perTick_Vol_perTick_over1': [
        {'type': 'between', 'min': 0.4, 'max': 7000, 'active': True}
    ],

    'sc_volRev_perTick_volxTicksContZone_perTick_ratio': [
        {'type': 'between', 'min': 0.6, 'max': 7000, 'active': True}
    ],

    'sc_candleDuration': [
        {'type': 'between', 'min': 2.5, 'max': 10000, 'active': True}
    ],


    'sc_is_antiEpuisement_long': [
        {'type': 'between', 'min': 1, 'max': 1, 'active': True}
    ],
    'sc_is_antiSpring_short': [
        {'type': 'between', 'min': 1, 'max': 1, 'active': False}
    ],
}
# ────────────────────────────────────────────────────────────────────────────────
# Algorithmes Short uniquement
# ────────────────────────────────────────────────────────────────────────────────
algoShort1 = {
    'sc_reg_slope_30P_2': [
        {'type': 'between', 'min': 0.2, 'max': 0.95, 'active': True}],

    'sc_diffVolDelta_1_1Ratio': [
        {'type': 'greater_than_or_equal', 'threshold': 0.2, 'active': True}],

    'sc_close_sma_zscore_14': [
        {'type': 'not_between', 'min': -4444, 'max': 2.2, 'active': True}],
    'sc_candleDuration': [
        {'type': 'between', 'min': 2.5, 'max': 70, 'active': True}],
}
algoShort2 = {
    'sc_reg_slope_30P_2': [
        {'type': 'between', 'min': -0.8, 'max': 0.9, 'active': True}],
    'sc_reg_std_30P_2': [
        {'type': 'between', 'min': 1.25, 'max': 3.5, 'active': True}],
    'sc_ratio_delta_vol_VA11P': [
        {'type': 'between', 'min': 0.25, 'max': 1, 'active': True}],
    'sc_candleDuration': [
        {'type': 'between', 'min': 1, 'max': 60, 'active': True}],
}

algoShort3 = {
    'sc_reg_slope_30P_2': [
        {'type': 'between', 'min': 0.1, 'max': 0.65, 'active': True}],
    'sc_is_wr_overbought_short': [
        {'type': 'between', 'min': 1, 'max': 1, 'active': True}],
    'sc_volPocVolRevesalXContRatio': [
        {'type': 'between', 'min': 0.17, 'max': 1, 'active': True}],
    'sc_pocDeltaPocVolRatio': [
        {'type': 'between', 'min': -1, 'max': 0, 'active': True}],
    'sc_candleDuration': [
        {'type': 'between', 'min': 2, 'max': 120, 'active': True}],
}

algoShort4 = {
    'sc_close_sma_zscore_21': [
        {'type': 'not_between', 'min': -150, 'max': 1.6, 'active': True}],
    'sc_reg_std_30P_2': [
        {'type': 'between', 'min': 1.2, 'max': 5, 'active': True}],
    'sc_reg_slope_30P_2': [
        {'type': 'between', 'min': 0.2, 'max': 0.45, 'active': True}],
    'sc_volPocVolRevesalXContRatio': [
        {'type': 'between', 'min': 0.22, 'max': 1, 'active': True}],
    'sc_pocDeltaPocVolRatio': [
        {'type': 'between', 'min': -1, 'max': 0, 'active': True}],
    'sc_candleDuration': [
        {'type': 'between', 'min': 1, 'max': 120, 'active': True}],
}

algoShort5 = {
    'sc_reg_slope_30P_2': [
        {'type': 'between', 'min': 0.32, 'max': 0.8, 'active': True}],
    'sc_ratio_volRevMove_volImpulsMove': [
        {'type': 'between', 'min': 0.6, 'max': 0.9, 'active': True}],
    'sc_candleDuration': [{'type': 'between', 'min': 1, 'max': 50, 'active': True}],
}

algoShort6 = {
    'sc_ratio_volRevMoveZone1_volRevMoveExtrem_xRevZone': [
        {'type': 'between', 'min': 1.7, 'max': 100, 'active': True}],
    'sc_ratio_deltaRevMove_volRevMove': [
        {'type': 'between', 'min': -0.05, 'max': 0.35, 'active': True}],
    'sc_reg_slope_30P_2': [
        {'type': 'between', 'min': -0.5, 'max': 0.9, 'active': True}],
    'sc_diffPriceClose_VA6PPoc': [
        {'type': 'between', 'min': 2, 'max': 30, 'active': True}],
    'sc_candleDuration': [
        {'type': 'between', 'min': 2, 'max': 50, 'active': True}],
}

algoShort7 = {
    'sc_reg_slope_30P_2': [
        {'type': 'between', 'min': -0.1, 'max': 1, 'active': True}],
    'sc_ratio_volRevMove_volImpulsMove': [
        {'type': 'between', 'min': 1.15, 'max': 30, 'active': True}],
    'sc_ratio_deltaRevMove_volRevMove': [
        {'type': 'between', 'min': 0.15, 'max': 1, 'active': True}],
    'sc_candleDuration': [
        {'type': 'between', 'min': 1, 'max': 180, 'active': True}],
}

algoShort8 = {
    'sc_delta_impulsMove_xRevZone_bigStand_extrem': [
        {'type': 'less_than_or_equal', 'threshold': -5, 'active': True}],
    'sc_reg_slope_30P_2': [
        {'type': 'between', 'min': -0.6, 'max': 0.6, 'active': True}],
    'sc_is_imBullWithPoc_light_short': [
        {'type': 'between', 'min': 1, 'max': 1, 'active': True}],
    'sc_candleDuration': [
        {'type': 'between', 'min': 1, 'max': 60, 'active': False}],
}
algoShort9 = {
    'sc_delta_revMove_xRevZone_bigStand_extrem': [
        {'type': 'greater_than_or_equal', 'threshold': 30, 'active': True}, ],

    'sc_reg_slope_30P_2': [
        {'type': 'between', 'min': -0.05, 'max': 0.95, 'active': True}],
    'sc_diffHighPrice_0_1': [
        {'type': 'between', 'min': 2.75, 'max': 100, 'active': True}],
    'sc_candleDuration': [
        {'type': 'between', 'min': 1, 'max': 60, 'active': True}],
}
algoShort10 = { #xxx a verifier car pas disponible sur le dataset que j'ai testé
    'is_rs_range_short': [
        {'type': 'between', 'min': 1, 'max': 1, 'active': True}],
    'sc_candleDuration': [
        {'type': 'between', 'min': 1, 'max': 50, 'active': True}],
    'is_vwap_reversal_pro_short': [
        {'type': 'between', 'min': 1, 'max': 1, 'active': True}],
}

algoShort11 = {
    'sc_is_mfi_overbought_short': [# xxx meilleur sans mfi ?
        {'type': 'between', 'min': 1, 'max': 1, 'active': True}],
    'sc_candleDuration': [
        {'type': 'between', 'min': 5, 'max': 360, 'active': True}],
    'sc_is_rs_range_short': [
        {'type': 'between', 'min': 1, 'max': 1, 'active': True}],

    'sc_volRevVolRevesalXContRatio': [
        {'type': 'between', 'min': 0.6, 'max': 1, 'active': True}],
'sc_deltaRev_volRev_ratio': [
        {'type': 'between', 'min': -1, 'max': 0.3, 'active': True}],
}
algoShort12 = {
    'sc_pocDeltaPocVolRatio': [
        {'type': 'between', 'min': 0.25, 'max': 1, 'active': True}],
    'sc_diffPriceClosePoc_0_0': [
        {'type': 'between', 'min': -0.5, 'max': -0, 'active': True}],
    'sc_reg_std_30P_2': [
        {'type': 'not_between', 'min': 0, 'max': 1.8, 'active': True}],
    'sc_reg_slope_30P_2': [
        {'type': 'between', 'min': 0, 'max': 0.5, 'active': True}],
    'sc_candleDuration': [
        {'type': 'between', 'min': 2, 'max': 50, 'active': True}],
}
algoShort13 = {
    'sc_reg_std_5P_2': [
        {'type': 'between', 'min': 1.2, 'max': 6.5, 'active': True}],
    'sc_reg_slope_10P_2': [
        {'type': 'between', 'min': -0.65, 'max': 0.97, 'active': True}],
    'sc_reg_slope_15P_2': [
        {'type': 'between', 'min': -0.9, 'max': 0.9, 'active': True}],
    'sc_diffPriceClose_VA6PPoc': [
        {'type': 'greater_than_or_equal', 'threshold': 4.5, 'active': True}],
    'sc_candleDuration': [
        {'type': 'between', 'min': 1.5, 'max': 180, 'active': True}],
}

algoShort14 = {
    'sc_ratio_deltaImpulsMove_volImpulsMove': [
        {'type': 'between', 'min': 0.15, 'max': 1, 'active': True}],
    'sc_ratio_deltaRevMove_volRevMove': [
        {'type': 'between', 'min': 0.15, 'max': 0.55, 'active': True}],
    'sc_candleDuration': [
        {'type': 'between', 'min': 1, 'max': 50, 'active': True}],
    'sc_reg_std_30P_2': [
        {'type': 'between', 'min': 1, 'max': 3.5, 'active': True}],
}

algoShort15 = {
    'sc_cum_4DiffVolDeltaRatio': [
        {'type': 'between', 'min': -0.7, 'max': -0.2, 'active': True}],
    'sc_reg_slope_5P_2': [
        {'type': 'between', 'min': -0.1, 'max': 0.9, 'active': True}],
    'is_vwap_reversal_pro_short': [
        {'type': 'between', 'min': 1, 'max': 1, 'active': False}],
    'sc_candleDuration': [
        {'type': 'between', 'min': 1, 'max': 70, 'active': True}],
    'sc_candleSizeTicks': [
        {'type': 'between', 'min': 11, 'max': 30, 'active': True}],
}
algoShort16 = {
    'sc_is_imBullWithPoc_light_short': [
        {'type': 'between', 'min': 1, 'max': 2, 'active': True}],
    'sc_ratio_volZone1_volExtrem': [
        {'type': 'between', 'min': 0.3, 'max': 0.75, 'active': True}],

    'sc_candleDuration': [
        {'type': 'between', 'min': 1, 'max': 50, 'active': True}],
}
algoShort17 = {
    'sc_is_imBullWithPoc_aggressive_short': [
        {'type': 'between', 'min': 1, 'max': 1, 'active': True}],
    'sc_candleDuration': [
        {'type': 'between', 'min': 1, 'max': 60, 'active': True}],
}

# Dictionnaire des algorithmes Short uniquement
algorithmsShort = {
    # "algoShort1": algoShort1,
    "algoShort2": algoShort2,
    "algoShort3": algoShort3,
    "algoShort4": algoShort4,
    "algoShort5": algoShort5,
    "algoShort6": algoShort6,
    "algoShort7": algoShort7,
    "algoShort8": algoShort8,
    # "algoShort10": algoShort10,
    "algoShort11": algoShort11,
    "algoShort12": algoShort12,
    "algoShort13": algoShort13,
    "algoShort14": algoShort14,
    "algoShort15": algoShort15,
    "algoShort16": algoShort14,
    "algoShort17": algoShort17,
}


# ────────────────────────────────────────────────────────────────────────────────
# Fonctions utilitaires
# ────────────────────────────────────────────────────────────────────────────────
def load_and_process_data(file_path):
    """Charger et prétraiter les données d'un fichier CSV."""
    df_init_features, CUSTOM_SESSIONS = load_features_and_sections(file_path)

    cats = [
        "Trades échoués short", "Trades échoués long",
        "Trades réussis short", "Trades réussis long"
    ]
    df_analysis = df_init_features[df_init_features["trade_category"].isin(cats)].copy()
    df_analysis["class"] = np.where(df_analysis["trade_category"].str.contains("échoués"), 0, 1)
    df_analysis["pos_type"] = np.where(df_analysis["trade_category"].str.contains("short"), "Short", "Long")

    return df_init_features, df_analysis, CUSTOM_SESSIONS


def apply_global_filter(df_analysis, global_filter):
    """Applique un filtre global à toutes les données avant l'évaluation des algorithmes."""
    if not global_filter:
        return df_analysis

    print(f"🌍 Application du filtre global sur {len(df_analysis)} trades...")
    df_filtered = apply_feature_conditions(df_analysis, global_filter)
    reduction_pct = (len(df_analysis) - len(df_filtered)) / len(df_analysis) * 100
    print(f"   Trades restants: {len(df_filtered)} (-{reduction_pct:.1f}%)")

    # Afficher les conditions appliquées
    for feature, conditions in global_filter.items():
        for condition in conditions:
            if condition.get('active', True):
                cond_type = condition['type']
                if cond_type == 'greater_than_or_equal':
                    print(f"   • {feature} >= {condition['threshold']}")
                elif cond_type == 'less_than_or_equal':
                    print(f"   • {feature} <= {condition['threshold']}")
                elif cond_type == 'between':
                    print(f"   • {feature} entre {condition['min']} et {condition['max']}")
                elif cond_type == 'not_between':
                    print(f"   • {feature} PAS entre {condition['min']} et {condition['max']}")

    return df_filtered


def evaluate_combined_winrate_by_dataset(df_analysis, df_init_features, algorithms, global_filter=None,
                                         dataset_name="Dataset"):
    """
    Évalue le win rate combiné de tous les algorithmes SHORT après application du filtre global
    """
    print(f"\n🎯 ANALYSE WIN RATE COMBINÉ - {dataset_name}")
    print("=" * 60)

    # 1. Filtrer pour ne garder que les positions SHORT
    df_analysis_short = df_analysis[df_analysis["pos_type"] == "Short"].copy()
    print(f"📊 Positions Short: {len(df_analysis_short)} trades")

    # 2. Appliquer le filtre global
    if global_filter:
        df_analysis_filtered = apply_global_filter(df_analysis_short, global_filter)
        print(f"📊 Après filtre global: {len(df_analysis_filtered)} trades")
    else:
        df_analysis_filtered = df_analysis_short
        print(f"📊 Aucun filtre global: {len(df_analysis_filtered)} trades")

    # 2.5. NOUVEAU: Calculer le Win Rate après filtre global (avant algorithmes)
    if len(df_analysis_filtered) > 0:
        winning_trades_filtered = (df_analysis_filtered["trade_pnl"] > 0).sum()
        win_rate_after_global_filter = (winning_trades_filtered / len(df_analysis_filtered) * 100)
        total_pnl_after_global_filter = df_analysis_filtered["trade_pnl"].sum()
        print(
            f"📊 Win Rate après filtre global: {win_rate_after_global_filter:.2f}% ({winning_trades_filtered}/{len(df_analysis_filtered)})")
        print(f"📊 PnL après filtre global: {total_pnl_after_global_filter:.2f}")
    else:
        win_rate_after_global_filter = 0
        total_pnl_after_global_filter = 0
        print(f"📊 Win Rate après filtre global: 0% (0/0)")

    # 3. Collecter tous les trades sélectionnés par TOUS les algorithmes
    all_selected_trades = pd.DataFrame()

    print(f"\n🔍 Application des {len(algorithms)} algorithmes SHORT:")

    for algo_name, conditions in algorithms.items():
        df_algo_filtered = apply_feature_conditions(df_analysis_filtered, conditions)

        if len(df_algo_filtered) > 0:
            # Marquer ces trades avec l'algorithme qui les a sélectionnés
            df_algo_filtered = df_algo_filtered.copy()
            df_algo_filtered['selected_by_algo'] = algo_name
            all_selected_trades = pd.concat([all_selected_trades, df_algo_filtered], ignore_index=True)
            print(f"   • {algo_name}: {len(df_algo_filtered)} trades sélectionnés")
        else:
            print(f"   • {algo_name}: 0 trades sélectionnés")

    # 4. Supprimer les doublons (un trade peut être sélectionné par plusieurs algos)
    if len(all_selected_trades) > 0:
        # Identifier la colonne d'index des trades
        # Essayer plusieurs noms possibles pour l'identifiant unique des trades
        possible_index_cols = ['tradeIndex', 'trade_index', 'index', 'trade_id', 'id']
        trade_index_col = None

        for col in possible_index_cols:
            if col in all_selected_trades.columns:
                trade_index_col = col
                break

        # Si aucune colonne d'index trouvée, utiliser l'index pandas
        if trade_index_col is None:
            print("   ⚠️  Aucune colonne d'index trouvée, utilisation de l'index pandas")
            all_selected_trades = all_selected_trades.reset_index()
            trade_index_col = 'index'

        print(f"   📋 Utilisation de '{trade_index_col}' comme identifiant unique des trades")

        # Garder trace des algorithmes qui ont sélectionné chaque trade
        trades_by_algo = all_selected_trades.groupby([trade_index_col])['selected_by_algo'].apply(
            lambda x: ', '.join(x.unique())).reset_index()

        # Supprimer les doublons basés sur l'identifiant des trades
        unique_selected_trades = all_selected_trades.drop_duplicates(subset=[trade_index_col]).copy()

        # Ajouter l'info des algorithmes combinés
        unique_selected_trades = unique_selected_trades.merge(trades_by_algo, on=trade_index_col,
                                                              suffixes=('', '_combined'))

        print(f"\n📈 RÉSULTATS COMBINÉS:")
        print(f"   • Total trades sélectionnés (avec doublons): {len(all_selected_trades)}")
        print(f"   • Total trades uniques sélectionnés: {len(unique_selected_trades)}")

        # DEBUG: Vérification des calculs
        print(f"   🔍 VÉRIFICATION CALCULS:")
        print(f"   • Trades originaux SHORT: {len(df_analysis_short)}")
        print(f"   • Trades après filtre global: {len(df_analysis_filtered)}")
        print(f"   • Trades sélectionnés uniques: {len(unique_selected_trades)}")

        # Vérification du pourcentage
        expected_pct = (len(unique_selected_trades) / len(df_analysis_short) * 100) if len(df_analysis_short) > 0 else 0
        print(f"   • %Sél.Final calculé: {expected_pct:.2f}% ({len(unique_selected_trades)}/{len(df_analysis_short)})")

        # Vérification des sessions
        if 'session_id' in df_analysis_short.columns:
            sessions_in_short = df_analysis_short['session_id'].nunique()
            print(f"   • Nombre de sessions avec trades SHORT: {sessions_in_short}")

            # Distribution des trades par session
            session_distribution = df_analysis_short['session_id'].value_counts().sort_index()
            print(f"   • Distribution des trades SHORT par session:")
            for session_id, count in session_distribution.items():
                print(f"     - Session {session_id}: {count} trades")

        # Vérification des trades sélectionnés par session
        if len(unique_selected_trades) > 0 and 'session_id' in unique_selected_trades.columns:
            selected_sessions = unique_selected_trades['session_id'].nunique()
            print(f"   • Sessions avec trades sélectionnés: {selected_sessions}")

            selected_distribution = unique_selected_trades['session_id'].value_counts().sort_index()
            print(f"   • Distribution des trades sélectionnés par session:")
            for session_id, count in selected_distribution.items():
                print(f"     - Session {session_id}: {count} trades sélectionnés")

        # 5. Calculer les métriques combinées
        total_trades = len(unique_selected_trades)
        winning_trades = (unique_selected_trades["trade_pnl"] > 0).sum()
        losing_trades = (unique_selected_trades["trade_pnl"] <= 0).sum()

        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        total_pnl = unique_selected_trades["trade_pnl"].sum()

        profits = unique_selected_trades.loc[unique_selected_trades["trade_pnl"] > 0, "trade_pnl"].sum()
        losses = abs(unique_selected_trades.loc[unique_selected_trades["trade_pnl"] <= 0, "trade_pnl"].sum())
        profit_factor = profits / losses if losses > 0 else float('inf')

        avg_win = unique_selected_trades.loc[
            unique_selected_trades["trade_pnl"] > 0, "trade_pnl"].mean() if winning_trades > 0 else 0
        avg_loss = unique_selected_trades.loc[
            unique_selected_trades["trade_pnl"] <= 0, "trade_pnl"].mean() if losing_trades > 0 else 0

        print(f"\n💰 PERFORMANCE GLOBALE SHORT - {dataset_name}:")
        print(f"   • Win Rate: {win_rate:.2f}% ({winning_trades}/{total_trades})")
        print(f"   • Net PnL: {total_pnl:.2f}")
        print(f"   • Profit Factor: {profit_factor:.2f}")
        print(f"   • Gain moyen: {avg_win:.2f}")
        print(f"   • Perte moyenne: {avg_loss:.2f}")

        # 6. Analyse par algorithme
        print(f"\n🔬 CONTRIBUTION PAR ALGORITHME SHORT:")
        algo_contributions = {}

        for algo_name in algorithms.keys():
            algo_trades = all_selected_trades[all_selected_trades['selected_by_algo'] == algo_name]
            if len(algo_trades) > 0:
                algo_winrate = (algo_trades["trade_pnl"] > 0).sum() / len(algo_trades) * 100
                algo_pnl = algo_trades["trade_pnl"].sum()
                algo_contributions[algo_name] = {
                    'trades': len(algo_trades),
                    'win_rate': algo_winrate,
                    'pnl': algo_pnl
                }
                print(f"   • {algo_name}: {len(algo_trades)} trades, WR: {algo_winrate:.1f}%, PnL: {algo_pnl:.2f}")

        # 7. Analyse des overlaps (trades sélectionnés par plusieurs algos)
        multi_algo_trades = unique_selected_trades[
            unique_selected_trades['selected_by_algo_combined'].str.contains(',')]
        if len(multi_algo_trades) > 0:
            multi_winrate = (multi_algo_trades["trade_pnl"] > 0).sum() / len(multi_algo_trades) * 100
            print(f"\n🔄 TRADES MULTI-ALGORITHMES:")
            print(f"   • {len(multi_algo_trades)} trades sélectionnés par plusieurs algos")
            print(f"   • Win Rate multi-algo: {multi_winrate:.2f}%")

        return {
            'dataset': dataset_name,
            'total_original_trades': len(df_analysis_short),
            'trades_after_global_filter': len(df_analysis_filtered),
            'win_rate_after_global_filter': win_rate_after_global_filter,  # NOUVEAU
            'pnl_after_global_filter': total_pnl_after_global_filter,  # NOUVEAU
            'total_selected_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'net_pnl': total_pnl,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'algo_contributions': algo_contributions,
            'multi_algo_trades': len(multi_algo_trades) if len(multi_algo_trades) > 0 else 0
        }
    else:
        print(f"\n❌ Aucun trade SHORT sélectionné par les algorithmes sur {dataset_name}")
        return {
            'dataset': dataset_name,
            'total_original_trades': len(df_analysis_short),
            'trades_after_global_filter': len(df_analysis_filtered),
            'win_rate_after_global_filter': win_rate_after_global_filter,  # NOUVEAU
            'pnl_after_global_filter': total_pnl_after_global_filter,  # NOUVEAU
            'total_selected_trades': 0,
            'win_rate': 0,
            'net_pnl': 0
        }


def compare_datasets_winrates(results_dict):
    """
    Compare les win rates entre tous les datasets pour les positions SHORT
    """
    print(f"\n🏆 COMPARAISON WIN RATES SHORT ENTRE DATASETS")
    print("=" * 80)

    # Créer un DataFrame pour la comparaison
    comparison_data = []
    for dataset_name, results in results_dict.items():
        comparison_data.append({
            'Dataset': dataset_name,
            'Trades Originaux': results['total_original_trades'],
            'Après Filtre Global': results['trades_after_global_filter'],
            'Trades Sélectionnés': results['total_selected_trades'],
            'Win Rate (%)': results['win_rate'],
            'Net PnL': results['net_pnl'],
            'Profit Factor': results.get('profit_factor', 0),
            '% Sélection': (results['total_selected_trades'] / results['total_original_trades'] * 100) if results[
                                                                                                              'total_original_trades'] > 0 else 0
        })

    df_comparison = pd.DataFrame(comparison_data)

    # Affichage formaté
    print(f"{'Dataset':<10} {'Orig.':<8} {'Filtrés':<8} {'Sélect.':<8} {'WR %':<7} {'PnL':<10} {'PF':<6} {'% Sél.':<7}")
    print("-" * 70)

    for _, row in df_comparison.iterrows():
        print(f"{row['Dataset']:<10} {row['Trades Originaux']:<8} {row['Après Filtre Global']:<8} "
              f"{row['Trades Sélectionnés']:<8} {row['Win Rate (%)']:<7.1f} {row['Net PnL']:<10.2f} "
              f"{row['Profit Factor']:<6.2f} {row['% Sélection']:<7.1f}")

    # Statistiques globales
    total_selected = df_comparison['Trades Sélectionnés'].sum()
    weighted_winrate = (df_comparison['Win Rate (%)'] * df_comparison[
        'Trades Sélectionnés']).sum() / total_selected if total_selected > 0 else 0
    total_pnl = df_comparison['Net PnL'].sum()

    print(f"\n📊 STATISTIQUES GLOBALES SHORT:")
    print(f"   • Total trades sélectionnés: {total_selected}")
    print(f"   • Win Rate pondéré: {weighted_winrate:.2f}%")
    print(f"   • PnL total: {total_pnl:.2f}")

    return df_comparison


def get_session_info(df_init_features, df_analysis):
    """
    Calcule le nombre total de sessions et le nombre de sessions avec au moins un trade
    """
    # Nombre total de sessions (identifié par sc_sessionStartEnd == 10)
    total_sessions = (df_init_features['sc_sessionStartEnd'] == 10).sum()

    # Sessions avec au moins un trade en utilisant session_id
    if 'session_id' in df_analysis.columns:
        sessions_with_trades = df_analysis['session_id'].nunique()
    else:
        # Fallback si session_id n'existe pas
        sessions_with_trades = "N/A"

    return total_sessions, sessions_with_trades


def compare_datasets_winrates_with_sessions(results_dict, datasets_info):
    """
    Compare les win rates entre tous les datasets avec informations de sessions
    """
    print(f"\n🏆 COMPARAISON WIN RATES SHORT ENTRE DATASETS (avec sessions)")
    print("=" * 100)

    # Créer un DataFrame pour la comparaison
    comparison_data = []

    # Mapping des noms de datasets
    dataset_mapping = {
        'Train': datasets_info[0],
        'Test': datasets_info[1],
        'Val1': datasets_info[2],
        'Val': datasets_info[3],
        'Unseen': datasets_info[4]
    }

    for dataset_name, results in results_dict.items():
        # Récupérer les infos de sessions
        if dataset_name in dataset_mapping:
            df_name, df_init_features = dataset_mapping[dataset_name]
            total_sessions = (df_init_features['sc_sessionStartEnd'] == 10).sum()

            # Sessions avec trades réels en utilisant session_id depuis les résultats
            # On doit calculer cela depuis les données d'analyse originales
            sessions_with_trades = "Calc"  # On va le calculer dans main()
        else:
            total_sessions = "N/A"
            sessions_with_trades = "N/A"

        comparison_data.append({
            'Dataset': dataset_name,
            'Sessions Totales': total_sessions,
            'Sessions avec Trades': sessions_with_trades,  # Sera mis à jour dans main()
            'Trades Originaux': results['total_original_trades'],
            'Après Filtre': results['trades_after_global_filter'],
            'Sélectionnés': results['total_selected_trades'],
            'Win Rate (%)': results['win_rate'],
            'Net PnL': results['net_pnl'],
            'Profit Factor': results.get('profit_factor', 0),
            '% Sélection': (results['total_selected_trades'] / results['total_original_trades'] * 100)
            if results['total_original_trades'] > 0 else 0
        })

    df_comparison = pd.DataFrame(comparison_data)

    # Affichage formaté avec sessions
    print(
        f"{'Dataset':<8} {'Sess.Tot':<8} {'Sess.Trd':<8} {'Orig.':<6} {'Filt.':<6} {'Sél.':<6} {'WR %':<6} {'PnL':<9} {'PF':<5} {'%Sél':<5}")
    print("-" * 100)

    for _, row in df_comparison.iterrows():
        print(f"{row['Dataset']:<8} {row['Sessions Totales']:<8} {row['Sessions avec Trades']:<8} "
              f"{row['Trades Originaux']:<6} {row['Après Filtre']:<6} {row['Sélectionnés']:<6} "
              f"{row['Win Rate (%)']:<6.1f} {row['Net PnL']:<9.2f} "
              f"{row['Profit Factor']:<5.2f} {row['% Sélection']:<5.1f}")

    # Statistiques globales
    total_selected = df_comparison['Sélectionnés'].sum()
    weighted_winrate = (df_comparison['Win Rate (%)'] * df_comparison[
        'Sélectionnés']).sum() / total_selected if total_selected > 0 else 0
    total_pnl = df_comparison['Net PnL'].sum()

    print(f"\n📊 STATISTIQUES GLOBALES SHORT:")
    print(f"   • Total trades sélectionnés: {total_selected}")
    print(f"   • Win Rate pondéré: {weighted_winrate:.2f}%")
    print(f"   • PnL total: {total_pnl:.2f}")

    return df_comparison


def analyze_best_algorithms_detailed(results_combined):
    """
    Analyse détaillée des meilleurs algorithmes SHORT avec décomposition par dataset
    """
    print(f"\n🏅 ANALYSE DÉTAILLÉE DES MEILLEURS ALGORITHMES SHORT:")
    print("=" * 80)

    # 1. Collecter toutes les contributions d'algorithmes
    best_algos_overall = {}

    for dataset_name, results in results_combined.items():
        if 'algo_contributions' in results:
            for algo, stats in results['algo_contributions'].items():
                if algo not in best_algos_overall:
                    best_algos_overall[algo] = {
                        'total_trades': 0,
                        'total_pnl': 0,
                        'datasets': []
                    }
                best_algos_overall[algo]['total_trades'] += stats['trades']
                best_algos_overall[algo]['total_pnl'] += stats['pnl']
                best_algos_overall[algo]['datasets'].append({
                    'dataset': dataset_name,
                    'trades': stats['trades'],
                    'win_rate': stats['win_rate'],
                    'pnl': stats['pnl']
                })

    # 2. Trier par PnL total
    sorted_algos = sorted(best_algos_overall.items(), key=lambda x: x[1]['total_pnl'], reverse=True)

    # 3. Affichage du tableau de synthèse
    print(f"\n📊 SYNTHÈSE GLOBALE:")
    print(f"{'Algorithme':<15} {'Total Trades':<12} {'Total PnL':<12} {'Datasets Actifs':<15}")
    print("-" * 60)

    for algo_name, stats in sorted_algos:
        active_datasets = len(stats['datasets'])
        print(f"{algo_name:<15} {stats['total_trades']:<12} {stats['total_pnl']:<12.2f} {active_datasets:<15}")

    # 4. Décomposition détaillée par dataset pour chaque algorithme
    print(f"\n📋 DÉCOMPOSITION PAR DATASET:")
    print("=" * 120)

    datasets_order = ['Train', 'Test', 'Val1', 'Val', 'Unseen']

    for algo_name, stats in sorted_algos:
        print(f"\n🔍 {algo_name} (Total: {stats['total_trades']} trades, {stats['total_pnl']:.2f} PnL)")
        print(f"{'Dataset':<10} {'Trades':<8} {'Win Rate':<10} {'PnL':<12} {'% du Total':<12}")
        print("-" * 60)

        # Créer un dictionnaire pour un accès facile par dataset
        dataset_data = {d['dataset']: d for d in stats['datasets']}

        # Afficher dans l'ordre des datasets
        for dataset in datasets_order:
            if dataset in dataset_data:
                data = dataset_data[dataset]
                pct_trades = (data['trades'] / stats['total_trades'] * 100) if stats['total_trades'] > 0 else 0
                print(
                    f"{dataset:<10} {data['trades']:<8} {data['win_rate']:<10.1f}% {data['pnl']:<12.2f} {pct_trades:<12.1f}%")
            else:
                print(f"{dataset:<10} {'0':<8} {'-':<10} {'0.00':<12} {'0.0':<12}%")

    # 5. Analyse comparative par dataset
    print(f"\n📈 PERFORMANCE PAR DATASET:")
    print("=" * 100)

    # En-tête du tableau
    header = f"{'Algorithme':<15}"
    for dataset in datasets_order:
        header += f"{dataset:<12}"
    header += f"{'Total':<12}"
    print(header)
    print("-" * 100)

    # Lignes de données
    for algo_name, stats in sorted_algos:
        row = f"{algo_name:<15}"
        dataset_data = {d['dataset']: d for d in stats['datasets']}

        for dataset in datasets_order:
            if dataset in dataset_data:
                pnl = dataset_data[dataset]['pnl']
                row += f"{pnl:<12.1f}"
            else:
                row += f"{'0.0':<12}"

        row += f"{stats['total_pnl']:<12.1f}"
        print(row)

    # 6. Statistiques par dataset (tous algorithmes confondus)
    print(f"\n🌍 CONTRIBUTION TOTALE PAR DATASET:")
    print("=" * 60)

    dataset_totals = {}
    for dataset in datasets_order:
        dataset_totals[dataset] = {'trades': 0, 'pnl': 0, 'algos_count': 0}

    for algo_name, stats in best_algos_overall.items():
        for dataset_info in stats['datasets']:
            dataset = dataset_info['dataset']
            if dataset in dataset_totals:
                dataset_totals[dataset]['trades'] += dataset_info['trades']
                dataset_totals[dataset]['pnl'] += dataset_info['pnl']
                dataset_totals[dataset]['algos_count'] += 1

    print(f"{'Dataset':<10} {'Total Trades':<12} {'Total PnL':<12} {'Algos Actifs':<12} {'PnL Moyen/Algo':<15}")
    print("-" * 70)

    for dataset in datasets_order:
        stats = dataset_totals[dataset]
        avg_pnl_per_algo = stats['pnl'] / stats['algos_count'] if stats['algos_count'] > 0 else 0
        print(
            f"{dataset:<10} {stats['trades']:<12} {stats['pnl']:<12.2f} {stats['algos_count']:<12} {avg_pnl_per_algo:<15.2f}")

    return sorted_algos, dataset_totals


def main():
    import warnings
    warnings.filterwarnings('ignore')

    print(f"🎯 ANALYSE WIN RATE GLOBAL - POSITIONS SHORT UNIQUEMENT")
    print(f"Direction: {DIRECTION}")
    print(f"Nombre d'algorithmes: {len(algorithmsShort)}")

    # Chargement des données
    print("\n📂 Chargement des données...")
    df_init_features_train, df_analysis_train, _ = load_and_process_data(FILE_PATH_TRAIN)
    df_init_features_test, df_analysis_test, _ = load_and_process_data(FILE_PATH_TEST)
    df_init_features_val1, df_analysis_val1, _ = load_and_process_data(FILE_PATH_VAL1)
    df_init_features_val, df_analysis_val, _ = load_and_process_data(FILE_PATH_VAL)
    df_init_features_unseen, df_analysis_unseen, _ = load_and_process_data(FILE_PATH_UNSEEN)

    # Informations sur les sessions
    datasets_info = [
        ("TRAIN", df_init_features_train, df_analysis_train),
        ("TEST", df_init_features_test, df_analysis_test),
        ("VAL1", df_init_features_val1, df_analysis_val1),
        ("VAL", df_init_features_val, df_analysis_val),
        ("UNSEEN", df_init_features_unseen, df_analysis_unseen)
    ]

    print("\n📊 INFORMATIONS SESSIONS:")
    session_stats = {}
    for name, df_init, df_analysis in datasets_info:
        # Sessions totales
        total_sessions = (df_init['sc_sessionStartEnd'] == 10).sum()

        # Sessions avec au moins un trade (toutes catégories)
        if 'session_id' in df_analysis.columns:
            sessions_with_trades = df_analysis['session_id'].nunique()
        else:
            sessions_with_trades = "N/A"

        # Sessions avec trades SHORT uniquement
        df_analysis_short = df_analysis[df_analysis["pos_type"] == "Short"]
        if 'session_id' in df_analysis_short.columns and len(df_analysis_short) > 0:
            sessions_with_short_trades = df_analysis_short['session_id'].nunique()
        else:
            sessions_with_short_trades = 0

        session_stats[name] = {
            'total': total_sessions,
            'with_trades': sessions_with_trades,
            'with_short_trades': sessions_with_short_trades
        }

        print(
            f"   • {name}: {total_sessions} sessions totales, {sessions_with_trades} avec trades, {sessions_with_short_trades} avec trades SHORT")

    # Évaluer chaque dataset
    print(f"\n🚀 ÉVALUATION DES ALGORITHMES SHORT...")
    results_combined = {}

    datasets = [
        ("Train", df_analysis_train, df_init_features_train),
        ("Test", df_analysis_test, df_init_features_test),
        ("Val1", df_analysis_val1, df_init_features_val1),
        ("Val", df_analysis_val, df_init_features_val),
        ("Unseen", df_analysis_unseen, df_init_features_unseen)
    ]

    for dataset_name, df_analysis, df_init_features in datasets:
        results_combined[dataset_name] = evaluate_combined_winrate_by_dataset(
            df_analysis, df_init_features, algorithmsShort, GLOBAL_MICRO_FILTER, dataset_name
        )

    # Ajouter les infos de sessions aux résultats
    dataset_name_mapping = {"Train": "TRAIN", "Test": "TEST", "Val1": "VAL1", "Val": "VAL", "Unseen": "UNSEEN"}
    for dataset_name in results_combined.keys():
        mapped_name = dataset_name_mapping.get(dataset_name)
        if mapped_name and mapped_name in session_stats:
            results_combined[dataset_name]['session_stats'] = session_stats[mapped_name]

    # Comparaison finale avec sessions - version améliorée avec Win Rate après filtre global
    print(f"\n🏆 COMPARAISON WIN RATES SHORT ENTRE DATASETS (avec sessions)")
    print("=" * 140)

    print(
        f"{'Dataset':<8} {'Sess.Tot':<8} {'Sess.Trd':<8} {'Sess.Short':<10} {'Orig.':<6} {'Filt.':<6} {'%MacroFilt':<10} {'WR-Global%':<10} {'Sél.':<6} {'Eff.Algo%':<9} {'%Sél.Final':<10} {'Trd/Sess':<8} {'WR-Algo%':<9} {'PnL':<9} {'PF':<5}")
    print("-" * 140)

    comparison_data = []
    for dataset_name, results in results_combined.items():
        session_info = results.get('session_stats', {})

        # Calcul du pourcentage de trades passant le filtre macro
        macro_filter_pct = (results['trades_after_global_filter'] / results['total_original_trades'] * 100) if results[
                                                                                                                   'total_original_trades'] > 0 else 0

        # Calcul du % de sélection par rapport aux trades après filtre global (efficacité des algos)
        algo_efficiency_pct = (results['total_selected_trades'] / results['trades_after_global_filter'] * 100) if \
        results['trades_after_global_filter'] > 0 else 0

        # Calcul du % de sélection finale par rapport aux trades originaux (taux de sélection global)
        final_selection_pct = (results['total_selected_trades'] / results['total_original_trades'] * 100) if results[
                                                                                                                 'total_original_trades'] > 0 else 0

        # Calcul du nombre moyen de trades sélectionnés par session
        sessions_with_short = session_info.get('with_short_trades', 1)  # Éviter division par 0
        trades_per_session = results['total_selected_trades'] / sessions_with_short if sessions_with_short > 0 else 0

        comparison_data.append({
            'Dataset': dataset_name,
            'Sessions Totales': session_info.get('total', 'N/A'),
            'Sessions avec Trades': session_info.get('with_trades', 'N/A'),
            'Sessions avec Short': session_info.get('with_short_trades', 'N/A'),
            'Trades Originaux': results['total_original_trades'],
            'Après Filtre': results['trades_after_global_filter'],
            '% Macro Filtre': macro_filter_pct,
            'WR Après Global Filter (%)': results.get('win_rate_after_global_filter', 0),  # NOUVEAU
            'Sélectionnés': results['total_selected_trades'],
            'Efficacité Algos (%)': algo_efficiency_pct,  # Sélectionnés / Après filtre global
            '% Sélection Finale': final_selection_pct,  # Sélectionnés / Trades originaux (taux final)
            'Trades/Session': trades_per_session,  # Nombre moyen de trades sélectionnés par session
            'Win Rate Algo (%)': results['win_rate'],
            'Net PnL': results['net_pnl'],
            'Profit Factor': results.get('profit_factor', 0),
        })

        print(f"{dataset_name:<8} {session_info.get('total', 'N/A'):<8} {session_info.get('with_trades', 'N/A'):<8} "
              f"{session_info.get('with_short_trades', 'N/A'):<10} {results['total_original_trades']:<6} "
              f"{results['trades_after_global_filter']:<6} {macro_filter_pct:<10.1f} "
              f"{results.get('win_rate_after_global_filter', 0):<10.1f} {results['total_selected_trades']:<6} "
              f"{algo_efficiency_pct:<9.1f} {final_selection_pct:<10.1f} {trades_per_session:<8.1f} {results['win_rate']:<9.1f} {results['net_pnl']:<9.2f} "
              f"{results.get('profit_factor', 0):<5.2f}")

    df_comparison = pd.DataFrame(comparison_data)

    # Statistiques globales
    total_selected = sum([r['total_selected_trades'] for r in results_combined.values()])
    total_after_filter = sum([r['trades_after_global_filter'] for r in results_combined.values()])

    if total_selected > 0:
        weighted_winrate = sum(
            [r['win_rate'] * r['total_selected_trades'] for r in results_combined.values()]) / total_selected
    else:
        weighted_winrate = 0

    if total_after_filter > 0:
        weighted_winrate_after_global = sum(
            [r.get('win_rate_after_global_filter', 0) * r['trades_after_global_filter'] for r in
             results_combined.values()]) / total_after_filter
    else:
        weighted_winrate_after_global = 0

    total_pnl = sum([r['net_pnl'] for r in results_combined.values()])
    total_pnl_after_global = sum([r.get('pnl_after_global_filter', 0) for r in results_combined.values()])

    print(f"\n📊 STATISTIQUES GLOBALES SHORT:")
    print(f"   • Total trades après filtre global: {total_after_filter}")
    print(f"   • Win Rate pondéré après filtre global: {weighted_winrate_after_global:.2f}%")
    print(f"   • PnL total après filtre global: {total_pnl_after_global:.2f}")
    print(f"   • Total trades sélectionnés par algos: {total_selected}")
    print(f"   • Win Rate pondéré après algos: {weighted_winrate:.2f}%")
    print(f"   • PnL total après algos: {total_pnl:.2f}")

    # Nouvelle analyse détaillée des algorithmes
    sorted_algos, dataset_totals = analyze_best_algorithms_detailed(results_combined)

    return results_combined, df_comparison, sorted_algos, dataset_totals


if __name__ == "__main__":
    results_combined, comparison_df, sorted_algos, dataset_totals = main()