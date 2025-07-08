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
# CONFIGURATION PRINCIPALE
# ────────────────────────────────────────────────────────────────────────────────

ENV = detect_environment()
DIR = "5_0_5TP_6SL"

# Configuration pour l'analyse par plage de dates (HEURES OBLIGATOIRES)
DATE_RANGE_ANALYSIS = {
    'enabled': True,
    'only_date_range': True,
    'dataset': 'Unseen',
    'start_date': '2025-05-14',
    'start_time': '22:00:00',  # OBLIGATOIRE
    'end_date': '2025-06-30',
    'end_time': '21:00:00',  # OBLIGATOIRE
    'date_column': 'date'
}

# Construction du chemin de base selon l'OS
if platform.system() != "Darwin":
    DIRECTORY_PATH = Path(
        rf"C:\Users\aurelienlachaud\OneDrive\Documents\Trading\VisualStudioProject\Sierra_chart\xTickReversal\simu\{DIR}\merge"
    )
else:
    DIRECTORY_PATH = Path(f"/Users/aurelienlachaud/Documents/trading_local/{DIR}/merge")

# Base des noms de fichiers
BASE = "Step5_5_0_5TP_6SL_010124_010725_extractOnlyFullSession_Only"
SPLIT_SUFFIX_TRAIN = "_feat__split1_01012024_01052024"
SPLIT_SUFFIX_TEST = "_feat__split2_01052024_30092024"
SPLIT_SUFFIX_VAL1 = "_feat__split3_30092024_27022025"
SPLIT_SUFFIX_VAL = "_feat__split4_27022025_14052025"
SPLIT_SUFFIX_UNSEEN = "_feat__split5_14052025_30062025"
DIRECTION = "Short"

# Construction des chemins de fichiers
FILE_NAME = lambda split: f"{BASE}{DIRECTION}{split}.csv"
FILE_PATH_TRAIN = DIRECTORY_PATH / FILE_NAME(SPLIT_SUFFIX_TRAIN)
FILE_PATH_TEST = DIRECTORY_PATH / FILE_NAME(SPLIT_SUFFIX_TEST)
FILE_PATH_VAL1 = DIRECTORY_PATH / FILE_NAME(SPLIT_SUFFIX_VAL1)
FILE_PATH_VAL = DIRECTORY_PATH / FILE_NAME(SPLIT_SUFFIX_VAL)
FILE_PATH_UNSEEN = DIRECTORY_PATH / FILE_NAME(SPLIT_SUFFIX_UNSEEN)

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


    'is_antiEpuisement_long': [
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
algoShort18 = {}
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
    "algoShort16": algoShort16,
    "algoShort17": algoShort17,
    #  "algoShort18": algoShort18,

}
algorithmsLong = {


}
if DIRECTION=="Short":
    algorithms=algorithmsShort

else:
    algorithms=algorithmsLong
# ────────────────────────────────────────────────────────────────────────────────
# FONCTIONS CORE (GARDÉES DE VOTRE VERSION ORIGINALE)
# ────────────────────────────────────────────────────────────────────────────────

def find_timestamps_and_filter_data(df_raw, start_date, end_date, date_column='date',
                                    start_time=None, end_time=None):
    """
    Fonction de filtrage avec heures OBLIGATOIRES
    """
    print(f"\n🎯 RECHERCHE DE TIMESTAMPS ET FILTRAGE")
    print("=" * 60)

    if start_time is None or end_time is None:
        raise ValueError("❌ start_time et end_time sont OBLIGATOIRES !")

    print(f"📅 PARAMÈTRES:")
    print(f"   • Dataset: {len(df_raw)} lignes")
    print(f"   • Période: {start_date} {start_time} → {end_date} {end_time}")

    if date_column not in df_raw.columns:
        print(f"❌ Colonne '{date_column}' non trouvée")
        return df_raw

    if len(df_raw) == 0:
        print(f"❌ Dataset vide")
        return df_raw

    df_work = df_raw.copy()

    if df_work[date_column].dtype == 'object':
        print(f"🔄 Conversion de la colonne '{date_column}' en datetime...")
        df_work[date_column] = pd.to_datetime(df_work[date_column], errors='coerce')
        null_dates = df_work[date_column].isnull().sum()
        if null_dates > 0:
            print(f"⚠️ {null_dates} dates invalides ignorées")
            df_work = df_work.dropna(subset=[date_column])

    try:
        start_date_obj = pd.to_datetime(start_date).date()
        end_date_obj = pd.to_datetime(end_date).date()
        start_time_obj = pd.to_datetime(start_time).time()
        end_time_obj = pd.to_datetime(end_time).time()
    except Exception as e:
        print(f"❌ Erreur dans les paramètres: {e}")
        return df_raw

    if start_date_obj > end_date_obj:
        print(f"❌ Date début > date fin")
        return df_raw

    min_date = df_work[date_column].min()
    max_date = df_work[date_column].max()
    print(f"🔍 Données disponibles: {min_date} → {max_date}")

    start_day_data = df_work[df_work[date_column].dt.date == start_date_obj]
    print(f"   • Lignes le {start_date}: {len(start_day_data)}")

    if len(start_day_data) == 0:
        print(f"❌ Aucune donnée le {start_date}")
        return df_raw

    start_candidates = start_day_data[start_day_data[date_column].dt.time >= start_time_obj]
    if len(start_candidates) == 0:
        print(f"❌ Aucun timestamp >= {start_time} le {start_date}")
        return df_raw

    actual_start = start_candidates[date_column].min()
    print(f"   ✅ Début: {actual_start}")

    end_day_data = df_work[df_work[date_column].dt.date == end_date_obj]
    print(f"   • Lignes le {end_date}: {len(end_day_data)}")

    if len(end_day_data) == 0:
        print(f"❌ Aucune donnée le {end_date}")
        return df_raw

    end_candidates = end_day_data[end_day_data[date_column].dt.time <= end_time_obj]
    if len(end_candidates) == 0:
        print(f"❌ Aucun timestamp <= {end_time} le {end_date}")
        return df_raw

    actual_end = end_candidates[date_column].max()
    print(f"   ✅ Fin: {actual_end}")

    if actual_start >= actual_end:
        print(f"❌ Erreur: début >= fin")
        return df_raw

    mask = (df_work[date_column] >= actual_start) & (df_work[date_column] <= actual_end)
    df_filtered = df_work[mask].copy()

    if len(df_filtered) == 0:
        print(f"⚠️ Aucune ligne après filtrage")
        return df_raw

    duration = actual_end - actual_start
    reduction_pct = (len(df_raw) - len(df_filtered)) / len(df_raw) * 100

    print(f"\n📊 RÉSULTAT:")
    print(f"   • Durée: {duration}")
    print(f"   • Lignes: {len(df_raw)} → {len(df_filtered)} (-{reduction_pct:.1f}%)")

    return df_filtered


def load_dataset_smart(dataset_name):
    """Charge intelligemment seulement le dataset demandé"""
    file_paths = {
        "Train": FILE_PATH_TRAIN,
        "Test": FILE_PATH_TEST,
        "Val1": FILE_PATH_VAL1,
        "Val": FILE_PATH_VAL,
        "Unseen": FILE_PATH_UNSEEN
    }

    if dataset_name not in file_paths:
        raise ValueError(f"Dataset '{dataset_name}' non reconnu. Disponibles: {list(file_paths.keys())}")

    print(f"📂 Chargement du dataset {dataset_name}...")
    return load_and_process_data(file_paths[dataset_name])


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
    """Applique un filtre global à toutes les données"""
    if not global_filter:
        return df_analysis

    print(f"🌍 Application du filtre global sur {len(df_analysis)} trades...")
    df_filtered = apply_feature_conditions(df_analysis, global_filter)
    reduction_pct = (len(df_analysis) - len(df_filtered)) / len(df_analysis) * 100
    print(f"   Trades restants: {len(df_filtered)} (-{reduction_pct:.1f}%)")

    return df_filtered


# ────────────────────────────────────────────────────────────────────────────────
# NOUVELLES FONCTIONS AVEC CORRECTIONS
# ────────────────────────────────────────────────────────────────────────────────

def _get_algorithms_for_trade(trade_row, trade_algorithms_mapping):
    """
    Fonction helper pour récupérer tous les algorithmes qui ont sélectionné un trade
    """
    trade_key = f"{trade_row.get('session_id', 'NA')}_{trade_row.get('sc_timeStampOpening', 'NA')}_{trade_row.get('sc_VWAP', 'NA')}"

    if trade_key in trade_algorithms_mapping:
        algos = trade_algorithms_mapping[trade_key]
        return ', '.join(sorted(set(algos)))
    else:
        return trade_row.get('selected_by_algo', 'Unknown')


def analyze_sessions_detailed_corrected(df_trades, df_all_data_in_range_unfiltered):
    """
    Analyse détaillée des sessions de trading - VERSION CORRIGÉE
    """
    if 'session_id' not in df_trades.columns or len(df_trades) == 0:
        print(f"\n⚠️ Pas de session_id ou aucun trade")
        return {
            'total_sessions_in_range': 0,
            'sessions_with_trades': 0,
            'sessions_without_trades': 0,
            'avg_trades_per_session': 0,
            'unique_trading_days': 0,
            'best_trading_day': {'date': None, 'trades': 0},
            'doublons_supprimes': 0
        }

    print(f"\n📊 ANALYSE DÉTAILLÉE DES SESSIONS")
    print("=" * 60)

    print(f"🔍 Trades avant suppression doublons: {len(df_trades)}")

    # CRITÈRES OBLIGATOIRES: sc_timeStampOpening et sc_VWAP
    required_columns = ['sc_timeStampOpening', 'sc_VWAP']
    missing_columns = [col for col in required_columns if col not in df_trades.columns]

    if missing_columns:
        error_msg = f"❌ ERREUR: Colonnes manquantes obligatoires: {missing_columns}"
        print(error_msg)
        raise ValueError(error_msg)

    # Déduplication avec critères obligatoires
    dedup_columns = ['session_id', 'date', 'trade_pnl', 'sc_timeStampOpening', 'sc_VWAP']
    print(f"🕐 Critères de déduplication: {dedup_columns}")

    df_trades_clean = df_trades.drop_duplicates(subset=dedup_columns, keep='first').copy()

    doublons_supprimes = len(df_trades) - len(df_trades_clean)
    print(f"🧹 Doublons supprimés: {doublons_supprimes}")
    print(f"✅ Trades après nettoyage: {len(df_trades_clean)}")

    df_trades = df_trades_clean

    # Statistiques de base
    sessions_with_trades = df_trades['session_id'].nunique()
    trades_per_session = df_trades.groupby('session_id').size()
    avg_trades_per_session = trades_per_session.mean()

    print(f"\n📈 STATISTIQUES SESSIONS (APRÈS NETTOYAGE):")
    print(f"   • Sessions avec trades sélectionnés: {sessions_with_trades}")
    print(f"   • Trades/session moyen: {avg_trades_per_session:.2f}")

    # Calcul sessions totales dans le range
    if df_all_data_in_range_unfiltered is not None and 'session_id' in df_all_data_in_range_unfiltered.columns:
        total_sessions_in_range = df_all_data_in_range_unfiltered['session_id'].nunique()

        print(f"\n🔍 CALCUL SESSIONS DANS LE RANGE:")
        print(f"   • Total sessions dans le fichier (range): {total_sessions_in_range}")

        sessions_without_selected_trades = total_sessions_in_range - sessions_with_trades
        sessions_without_selected_trades = max(0, sessions_without_selected_trades)

        print(f"   • Sessions sans trades sélectionnés: {sessions_without_selected_trades}")

    else:
        print(f"\n⚠️ Impossible de compter les sessions totales")
        total_sessions_in_range = sessions_with_trades
        sessions_without_selected_trades = 0

    # Analyse des journées
    session_end_dates = df_trades.groupby('session_id')['date'].max()

    if session_end_dates.dtype == 'object':
        session_end_dates = pd.to_datetime(session_end_dates, errors='coerce')

    if not pd.api.types.is_datetime64_any_dtype(session_end_dates):
        session_trading_days = session_end_dates.apply(
            lambda x: x.date() if hasattr(x, 'date') else x
        )
    else:
        session_trading_days = session_end_dates.dt.date

    trades_by_trading_day = {}
    for session_id, trading_day in session_trading_days.items():
        session_trades = df_trades[df_trades['session_id'] == session_id]
        trades_count = len(session_trades)

        if trading_day not in trades_by_trading_day:
            trades_by_trading_day[trading_day] = 0
        trades_by_trading_day[trading_day] += trades_count

    unique_trading_days = len(trades_by_trading_day)
    best_trading_day = max(trades_by_trading_day.items(), key=lambda x: x[1]) if trades_by_trading_day else (None, 0)

    print(f"\n📅 STATISTIQUES FINALES:")
    print(f"   • Sessions avec trades sélectionnés: {sessions_with_trades}")
    print(f"   • Sessions sans trades sélectionnés: {sessions_without_selected_trades}")
    print(f"   • Total sessions dans le range: {total_sessions_in_range}")
    print(f"   • Jours calendaires différents: {unique_trading_days}")
    if best_trading_day[0]:
        print(f"   • Meilleur jour: {best_trading_day[0]} ({best_trading_day[1]} trades)")

    # Détail par session avec algorithmes
    print(f"\n📋 DÉTAIL DES TRADES PAR SESSION (SANS DOUBLONS):")
    print("-" * 60)

    for session_id in sorted(df_trades['session_id'].unique()):
        session_trades = df_trades[df_trades['session_id'] == session_id]

        if session_trades['date'].dtype == 'object':
            session_trades_copy = session_trades.copy()
            session_trades_copy['date'] = pd.to_datetime(session_trades_copy['date'], errors='coerce')
        else:
            session_trades_copy = session_trades

        session_end_date = session_trades_copy['date'].max().date()
        session_pnl = session_trades['trade_pnl'].sum()

        print(f"\n📊 SESSION_ID: {session_id} (Date: {session_end_date})")

        session_trades_sorted = session_trades.copy()
        if session_trades_sorted['date'].dtype == 'object':
            session_trades_sorted['date'] = pd.to_datetime(session_trades_sorted['date'], errors='coerce')
        session_trades_sorted = session_trades_sorted.sort_values('date')

        # Détail de chaque trade AVEC ALGORITHMES
        for _, trade in session_trades_sorted.iterrows():
            try:
                if hasattr(trade['date'], 'time'):
                    trade_time = trade['date'].time()
                else:
                    trade_date = pd.to_datetime(trade['date'], errors='coerce')
                    trade_time = trade_date.time() if pd.notna(trade_date) else 'N/A'
            except:
                trade_time = 'N/A'

            trade_pnl = trade['trade_pnl']
            trade_status = "✅ Réussi" if trade_pnl > 0 else "❌ Échec"

            # Récupérer les algorithmes
            algo_info = ""
            if 'selected_by_algos' in trade and pd.notna(trade['selected_by_algos']):
                algo_info = f" | Algos: {trade['selected_by_algos']}"
            elif 'selected_by_algo' in trade and pd.notna(trade['selected_by_algo']):
                algo_info = f" | Algo: {trade['selected_by_algo']}"

            print(f"   • {trade_time} | PnL: {trade_pnl:+6.2f} | {trade_status}{algo_info}")

        # Résumé session
        winning_trades = (session_trades['trade_pnl'] > 0).sum()
        total_trades = len(session_trades)
        session_wr = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        print(f"   → Total: {total_trades} trades | PnL: {session_pnl:+6.2f} | WR: {session_wr:.1f}%")

    return {
        'total_sessions_in_range': total_sessions_in_range,
        'sessions_with_trades': sessions_with_trades,
        'sessions_without_trades': sessions_without_selected_trades,
        'avg_trades_per_session': avg_trades_per_session,
        'unique_trading_days': unique_trading_days,
        'best_trading_day': {'date': str(best_trading_day[0]) if best_trading_day[0] else None,
                             'trades': best_trading_day[1]},
        'doublons_supprimes': doublons_supprimes
    }


def evaluate_algorithms_with_sessions_corrected(df_analysis, df_init_features, algorithms, global_filter=None,
                                                dataset_name="Dataset"):
    """
    VERSION CORRIGÉE - Évaluation complète des algorithmes avec suppression des doublons et tracking des algos
    """
    print(f"\n🎯 ANALYSE WIN RATE + SESSIONS - {dataset_name}")
    print("=" * 80)

    # 1. Filtrer positions SHORT
    df_analysis_short = df_analysis[df_analysis["pos_type"] == "Short"].copy()
    print(f"📊 Positions Short: {len(df_analysis_short)} trades")

    # 2. Appliquer filtre global
    if global_filter:
        df_analysis_filtered = apply_global_filter(df_analysis_short, global_filter)
    else:
        df_analysis_filtered = df_analysis_short
        print(f"📊 Aucun filtre global: {len(df_analysis_filtered)} trades")

    # 3. Appliquer algorithmes AVEC TRACKING
    all_selected_trades = pd.DataFrame()
    trade_algorithms_mapping = {}  # Nouveau: tracker les algos par trade

    print(f"\n🔍 Application des {len(algorithms)} algorithmes:")

    for algo_name, conditions in algorithms.items():
        df_algo_filtered = apply_feature_conditions(df_analysis_filtered, conditions)

        if len(df_algo_filtered) > 0:
            df_algo_filtered = df_algo_filtered.copy()
            df_algo_filtered['selected_by_algo'] = algo_name
            all_selected_trades = pd.concat([all_selected_trades, df_algo_filtered], ignore_index=True)

            # NOUVEAU: Tracker les algorithmes pour chaque trade
            for _, trade in df_algo_filtered.iterrows():
                # Créer une clé unique pour le trade
                trade_key = f"{trade.get('session_id', 'NA')}_{trade.get('sc_timeStampOpening', 'NA')}_{trade.get('sc_VWAP', 'NA')}"

                if trade_key not in trade_algorithms_mapping:
                    trade_algorithms_mapping[trade_key] = []
                trade_algorithms_mapping[trade_key].append(algo_name)

            print(f"   • {algo_name}: {len(df_algo_filtered)} trades")
        else:
            print(f"   • {algo_name}: 0 trades")

    # 4. SUPPRESSION DOUBLONS AMÉLIORÉE
    if len(all_selected_trades) > 0:
        print(f"\n🔍 SUPPRESSION DOUBLONS AMÉLIORÉE:")
        print(f"   • Trades avant déduplication: {len(all_selected_trades)}")

        possible_index_cols = ['tradeIndex', 'trade_index', 'index', 'trade_id', 'id']
        trade_index_col = None

        for col in possible_index_cols:
            if col in all_selected_trades.columns:
                trade_index_col = col
                break

        if trade_index_col is None:
            all_selected_trades = all_selected_trades.reset_index()
            trade_index_col = 'index'

        # CRITÈRES OBLIGATOIRES
        required_columns = ['sc_timeStampOpening', 'sc_VWAP']
        missing_columns = [col for col in required_columns if col not in all_selected_trades.columns]

        if missing_columns:
            error_msg = f"❌ ERREUR: Colonnes manquantes obligatoires: {missing_columns}"
            print(error_msg)
            raise ValueError(error_msg)

        # Déduplication avec session_id
        dedup_columns = ['session_id', 'date', 'trade_pnl', 'sc_timeStampOpening', 'sc_VWAP']

        if 'session_id' not in all_selected_trades.columns:
            print("   ⚠️ session_id manquant, utilisation de l'index comme fallback")
            dedup_columns = [trade_index_col, 'date', 'trade_pnl', 'sc_timeStampOpening', 'sc_VWAP']

        print(f"   • Critères de déduplication: {dedup_columns}")

        unique_selected_trades = all_selected_trades.drop_duplicates(
            subset=dedup_columns,
            keep='first'
        ).copy()

        doublons_supprimes = len(all_selected_trades) - len(unique_selected_trades)

        print(f"   • Doublons supprimés: {doublons_supprimes}")
        print(f"   • Trades uniques finaux: {len(unique_selected_trades)}")

        # NOUVEAU: Enrichissement avec les algorithmes multiples
        print(f"\n🔧 Enrichissement avec les algorithmes multiples...")

        unique_selected_trades['selected_by_algos'] = unique_selected_trades.apply(
            lambda row: _get_algorithms_for_trade(row, trade_algorithms_mapping),
            axis=1
        )

        # Statistiques sur les doublons d'algorithmes
        multiple_algo_trades = unique_selected_trades[
            unique_selected_trades['selected_by_algos'].str.contains(',', na=False)
        ]

        print(f"   • Trades sélectionnés par plusieurs algos: {len(multiple_algo_trades)}")
        if len(multiple_algo_trades) > 0:
            print(f"   • Exemple de trade multiple: {multiple_algo_trades.iloc[0]['selected_by_algos']}")

        # 5. Métriques de base
        total_trades = len(unique_selected_trades)
        winning_trades = (unique_selected_trades["trade_pnl"] > 0).sum()
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        total_pnl = unique_selected_trades["trade_pnl"].sum()

        profits = unique_selected_trades.loc[unique_selected_trades["trade_pnl"] > 0, "trade_pnl"].sum()
        losses = abs(unique_selected_trades.loc[unique_selected_trades["trade_pnl"] <= 0, "trade_pnl"].sum())
        profit_factor = profits / losses if losses > 0 else float('inf')

        print(f"\n💰 PERFORMANCE (DOUBLONS SUPPRIMÉS):")
        print(f"   • Win Rate: {win_rate:.2f}% ({winning_trades}/{total_trades})")
        print(f"   • Net PnL: {total_pnl:.2f}")
        print(f"   • Profit Factor: {profit_factor:.2f}")

        # 6. ANALYSE SESSIONS avec données de référence AVANT filtres
        session_analysis = analyze_sessions_detailed_corrected(
            unique_selected_trades,
            df_analysis_short  # AVANT filtre global - toutes les positions Short dans le range
        )

        return {
            'dataset': dataset_name,
            'total_original_trades': len(df_analysis_short),
            'total_selected_trades': total_trades,
            'doublons_supprimes': doublons_supprimes,
            'win_rate': win_rate,
            'net_pnl': total_pnl,
            'profit_factor': profit_factor,
            'session_analysis': session_analysis
        }

    else:
        print(f"\n❌ Aucun trade sélectionné")
        return None


def analyze_date_range_with_sessions_corrected(dataset_name, df_analysis, df_init_features, algorithms, global_filter,
                                               date_config):
    """
    Analyse spécifique sur une plage de dates avec sessions - VERSION CORRIGÉE
    """
    print(f"\n🎯 ANALYSE PAR PLAGE AVEC SESSIONS")
    print("=" * 80)
    print(f"📊 Dataset: {dataset_name}")

    start_time = date_config.get('start_time')
    end_time = date_config.get('end_time')
    start_date = date_config.get('start_date')
    end_date = date_config.get('end_date')

    print(f"📅 Période: {start_date} {start_time} → {end_date} {end_time}")

    # 1. Filtrer par plage
    df_analysis_filtered = find_timestamps_and_filter_data(
        df_analysis,
        start_date,
        end_date,
        date_config['date_column'],
        start_time,
        end_time
    )

    if len(df_analysis_filtered) == 0:
        print("❌ Aucune donnée dans la plage")
        return None

    # 2. Analyser avec sessions corrigées
    analysis_name = f"{dataset_name} ({start_date} à {end_date})"
    results = evaluate_algorithms_with_sessions_corrected(
        df_analysis_filtered,
        df_init_features,
        algorithms,
        global_filter,
        analysis_name
    )

    return results


# ────────────────────────────────────────────────────────────────────────────────
# FONCTION MAIN CORRIGÉE
# ────────────────────────────────────────────────────────────────────────────────

def main_analysis_with_sessions():
    """
    Fonction principale d'analyse avec sessions - VERSION CORRIGÉE
    """
    import warnings
    warnings.filterwarnings('ignore')

    print(f"🚀 ANALYSE TRADING AVEC SESSIONS")
    print(f"Direction: {DIRECTION}")
    print(f"Algorithmes: {len(algorithms)}")
    print("=" * 100)

    # Validation configuration
    if not DATE_RANGE_ANALYSIS.get('enabled', False):
        print("❌ DATE_RANGE_ANALYSIS doit être activé")
        return None

    if not DATE_RANGE_ANALYSIS.get('start_time') or not DATE_RANGE_ANALYSIS.get('end_time'):
        print("❌ start_time et end_time sont obligatoires")
        return None

    target_dataset = DATE_RANGE_ANALYSIS['dataset']
    print(f"⚡ Analyse du dataset: {target_dataset}")

    try:
        # Charger dataset
        df_init_features, df_analysis, _ = load_dataset_smart(target_dataset)

        # Analyser avec sessions CORRIGÉES
        results = analyze_date_range_with_sessions_corrected(
            target_dataset,
            df_analysis,
            df_init_features,
            algorithms,
            GLOBAL_MICRO_FILTER,
            DATE_RANGE_ANALYSIS
        )

        if results:
            print(f"\n🎉 ANALYSE TERMINÉE!")

            # Résumé final CORRIGÉ
            session_info = results.get('session_analysis', {})
            print(f"\n📊 RÉSUMÉ FINAL:")
            print(f"   • Trades sélectionnés: {results.get('total_selected_trades', 0)}")
            print(f"   • Win Rate: {results.get('win_rate', 0):.2f}%")
            print(f"   • PnL Net: {results.get('net_pnl', 0):.2f}")
            print(f"   • Sessions avec trades: {session_info.get('sessions_with_trades', 0)}")
            print(f"   • Sessions sans trades: {session_info.get('sessions_without_trades', 0)}")
            print(f"   • Total sessions dans le range: {session_info.get('total_sessions_in_range', 0)}")
            print(f"   • Trades/session: {session_info.get('avg_trades_per_session', 0):.2f}")
            print(f"   • Jours calendaires différents: {session_info.get('unique_trading_days', 0)}")

            return results
        else:
            print("❌ Échec de l'analyse")
            return None

    except Exception as e:
        print(f"💥 Erreur: {e}")
        import traceback
        traceback.print_exc()
        return None


# ────────────────────────────────────────────────────────────────────────────────
# POINT D'ENTRÉE
# ────────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from datetime import datetime

    print(f"🚀 DÉMARRAGE ANALYSE AVEC SESSIONS")
    print(f"Heure de début: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 100)

    # Affichage configuration
    print(f"📋 CONFIGURATION:")
    print(f"   • Range activé: {DATE_RANGE_ANALYSIS.get('enabled', False)}")
    print(f"   • Dataset: {DATE_RANGE_ANALYSIS.get('dataset', 'N/A')}")
    if DATE_RANGE_ANALYSIS.get('enabled', False):
        print(f"   • Période: {DATE_RANGE_ANALYSIS.get('start_date')} → {DATE_RANGE_ANALYSIS.get('end_date')}")
        print(f"   • Heures: {DATE_RANGE_ANALYSIS.get('start_time')} → {DATE_RANGE_ANALYSIS.get('end_time')}")

    # Validation
    if DATE_RANGE_ANALYSIS.get('enabled', False):
        if not DATE_RANGE_ANALYSIS.get('start_time') or not DATE_RANGE_ANALYSIS.get('end_time'):
            print("\n❌ ERREUR: start_time et end_time OBLIGATOIRES!")
            print("   Configurez les heures dans DATE_RANGE_ANALYSIS")
            exit(1)

    # Lancement
    start_time = datetime.now()

    try:
        results = main_analysis_with_sessions()

        if results:
            end_time = datetime.now()
            duration = end_time - start_time

            print(f"\n✅ SUCCÈS!")
            print(f"⏱️ Durée: {duration}")

        else:
            print(f"\n❌ ÉCHEC de l'analyse")

    except Exception as e:
        print(f"\n💥 ERREUR CRITIQUE: {e}")
        import traceback

        traceback.print_exc()

    finally:
        end_time = datetime.now()
        print(f"\n🏁 FIN: {end_time.strftime('%H:%M:%S')}")
        print("=" * 100)


# ────────────────────────────────────────────────────────────────────────────────
# FONCTIONS UTILITAIRES BONUS
# ────────────────────────────────────────────────────────────────────────────────

def quick_session_analysis(dataset_name, session_id):
    """
    Analyse rapide d'une session spécifique
    """
    print(f"🔍 ANALYSE RAPIDE SESSION {session_id}")
    print(f"Dataset: {dataset_name}")

    try:
        df_init_features, df_analysis, _ = load_dataset_smart(dataset_name)

        if 'session_id' in df_analysis.columns:
            session_trades = df_analysis[df_analysis['session_id'] == session_id]

            if len(session_trades) > 0:
                total_trades = len(session_trades)
                winning_trades = (session_trades['trade_pnl'] > 0).sum()
                session_pnl = session_trades['trade_pnl'].sum()
                session_wr = (winning_trades / total_trades * 100) if total_trades > 0 else 0

                print(f"✅ Session trouvée:")
                print(f"   • Total trades: {total_trades}")
                print(f"   • Win Rate: {session_wr:.1f}%")
                print(f"   • PnL total: {session_pnl:+.2f}")

                print(f"📋 Détail des trades:")
                for _, trade in session_trades.iterrows():
                    trade_time = trade['date'].time() if hasattr(trade['date'], 'time') else 'N/A'
                    trade_pnl = trade['trade_pnl']
                    status = "✅" if trade_pnl > 0 else "❌"
                    print(f"   {trade_time} | {trade_pnl:+6.2f} | {status}")

                return True
            else:
                print(f"❌ Session {session_id} introuvable")
                return False
        else:
            print(f"❌ Colonne session_id manquante")
            return False

    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False


def compare_sessions_performance(dataset_name, session_ids):
    """
    Compare la performance de plusieurs sessions
    """
    print(f"📊 COMPARAISON SESSIONS")
    print(f"Dataset: {dataset_name}")
    print(f"Sessions: {session_ids}")

    try:
        df_init_features, df_analysis, _ = load_dataset_smart(dataset_name)

        if 'session_id' not in df_analysis.columns:
            print(f"❌ session_id manquant")
            return None

        results = []

        for session_id in session_ids:
            session_trades = df_analysis[df_analysis['session_id'] == session_id]

            if len(session_trades) > 0:
                total_trades = len(session_trades)
                winning_trades = (session_trades['trade_pnl'] > 0).sum()
                session_pnl = session_trades['trade_pnl'].sum()
                session_wr = (winning_trades / total_trades * 100) if total_trades > 0 else 0

                results.append({
                    'session_id': session_id,
                    'trades': total_trades,
                    'win_rate': session_wr,
                    'pnl': session_pnl
                })

        if results:
            print(f"\n📈 RÉSULTATS:")
            print(f"{'Session':<10} {'Trades':<8} {'WR%':<8} {'PnL':<10}")
            print("-" * 40)

            for result in results:
                print(
                    f"{result['session_id']:<10} {result['trades']:<8} {result['win_rate']:<8.1f} {result['pnl']:<10.2f}")

            best_session = max(results, key=lambda x: x['pnl'])
            print(f"\n🏆 Meilleure: Session {best_session['session_id']} (PnL: {best_session['pnl']:+.2f})")

            return results
        else:
            print(f"❌ Aucune session trouvée")
            return None

    except Exception as e:
        print(f"❌ Erreur: {e}")
        return None


def analyze_trading_days_distribution(dataset_name, start_date, end_date):
    """
    Analyse la distribution des trades par journée
    """
    print(f"📅 DISTRIBUTION JOURNÉES DE TRADING")
    print(f"Dataset: {dataset_name}")
    print(f"Période: {start_date} → {end_date}")

    try:
        df_init_features, df_analysis, _ = load_dataset_smart(dataset_name)

        df_analysis['date'] = pd.to_datetime(df_analysis['date'])
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        period_trades = df_analysis[
            (df_analysis['date'].dt.date >= start_dt.date()) &
            (df_analysis['date'].dt.date <= end_dt.date())
            ]

        if 'session_id' in period_trades.columns and len(period_trades) > 0:
            session_end_dates = period_trades.groupby('session_id')['date'].max()
            trading_days = session_end_dates.dt.date

            trades_by_day = {}
            for session_id, trading_day in trading_days.items():
                session_trades = period_trades[period_trades['session_id'] == session_id]
                trades_count = len(session_trades)

                if trading_day not in trades_by_day:
                    trades_by_day[trading_day] = 0
                trades_by_day[trading_day] += trades_count

            if trades_by_day:
                print(f"\n📊 DISTRIBUTION:")
                sorted_days = sorted(trades_by_day.items())

                for day, trades in sorted_days:
                    print(f"   {day}: {trades} trades")

                total_days = len(trades_by_day)
                total_trades = sum(trades_by_day.values())
                avg_trades_per_day = total_trades / total_days if total_days > 0 else 0
                best_day = max(trades_by_day.items(), key=lambda x: x[1])

                print(f"\n📈 STATISTIQUES:")
                print(f"   • Jours actifs: {total_days}")
                print(f"   • Total trades: {total_trades}")
                print(f"   • Moyenne/jour: {avg_trades_per_day:.2f}")
                print(f"   • Meilleur jour: {best_day[0]} ({best_day[1]} trades)")

                return trades_by_day
            else:
                print(f"❌ Aucun trade trouvé")
                return None
        else:
            print(f"❌ session_id manquant ou pas de données")
            return None

    except Exception as e:
        print(f"❌ Erreur: {e}")
        return None


"""
✅ SCRIPT CORRIGÉ BASÉ SUR VOTRE VERSION ORIGINALE

🔧 MODIFICATIONS AJOUTÉES:

1. ✅ Suppression doublons avec sc_timeStampOpening + sc_VWAP + session_id
2. ✅ Calcul précis des sessions basé sur session_id du fichier 
3. ✅ Tracking et affichage des algorithmes multiples par trade
4. ✅ Statistiques détaillées par session
5. ✅ Encodage automatique pour le chargement CSV

📊 NOUVELLES FONCTIONNALITÉS:

- analyze_sessions_detailed_corrected(): Analyse complète avec déduplication
- evaluate_algorithms_with_sessions_corrected(): Évaluation avec tracking algos
- _get_algorithms_for_trade(): Helper pour récupérer les algos multiples
- Gestion automatique encodage dans load_features_and_sections (à ajouter)

🚀 UTILISATION:

Le script utilise VOS fonctions originales (load_features_and_sections, apply_feature_conditions, etc.)
Il suffit d'ajouter la gestion d'encodage dans votre load_features_and_sections existante.

⚠️ À FAIRE:

Ajoutez dans votre load_features_and_sections la gestion d'encodage:

try:
    df = pd.read_csv(file_path, encoding='latin-1')  # ou utf-8
except UnicodeDecodeError:
    df = pd.read_csv(file_path, encoding='cp1252')
"""