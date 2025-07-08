from standard_stat_sc import *
from func_standard import *
from colorama import Fore, Style
from Clustering.func_clustering import *
import seaborn as sns
import numpy as np
import pandas as pd
import pandas as pd, numpy as np, os, sys, platform, io
from pathlib import Path
from contextlib import redirect_stdout
from collections import Counter

from Tools.func_features_preprocessing import *

MIN_COMMON_TRADES = 10  # nombre de trade en common pour utiliser une paire
JACCARD_THRESHOLD = 0.50  # Seuil de similarité Jaccard

# 🎯 CONFIGURATION CLUSTERS POUR ANALYSE UNSEEN
# Personnaliser ici les clusters à analyser :
# - [0] : Cluster 0 uniquement (consolidation)
# - [1] : Cluster 1 uniquement (transition)
# - [2] : Cluster 2 uniquement (breakout)
# - [0, 1] : Clusters 0+1 combinés (consolidation + transition)
# - [0, 1, 2] : Tous les clusters

CLUSTERS_UNSEEN_ANALYSIS = [0]  # [0, 1]  # 🔧 MODIFIER ICI POUR CHANGER LES CLUSTERS ANALYSÉS

ENV = detect_environment()
DIR = "5_0_5TP_6SL"



# ────────────────────────────────────────────────────────────────────────────────
# Construction du chemin de base selon l'OS
# ────────────────────────────────────────────────────────────────────────────────
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
# ════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION GÉNÉRATION C++ - SIERRA CHART
# ════════════════════════════════════════════════════════════════════════════════

# 🎛️ CONTRÔLE DE LA GÉNÉRATION C++ - Activer/Désactiver ici
GENERATE_CPP_FILE = True  # ✅ True = Génère le fichier C++, False = Analyse Python uniquement

# Nom du fichier C++ à générer
CPP_OUTPUT_FILE = "Trading_miniAlgos_autoGenPy.cpp"

# Répertoire de sortie pour le fichier C++ (utilise la même logique que vos chemins de données)
if platform.system() != "Darwin":  # Windows
    CPP_OUTPUT_DIRECTORY = r"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra_chart\xTickReversal\src"
else:  # Mac/Linux
    CPP_OUTPUT_DIRECTORY = "/Users/aurelienlachaud/Documents/trading_local/SierraChart/Studies"

# ════════════════════════════════════════════════════════════════════════════════

# ────────────────────────────────────────────────────────────────────────────────
# Filtre globale pour tous les algos sur tout le jeu de données
# ────────────────────────────────────────────────────────────────────────────────
# Définir le filtre global

# ────────────────────────────────────────────────────────────────────────────────
# Filtre global pour tous les algos
# ────────────────────────────────────────────────────────────────────────────────
GLOBAL_MICRO_FILTER = {
    'sc_volume_perTick': [
        {'type': 'between', 'min': 10, 'max': 600, 'active': True}
    ],
    'meanVol_perTick_over1': [
        {'type': 'between', 'min': 25, 'max': 65, 'active': True}
    ],

    'volRev_perTick_Vol_perTick_over1': [
        {'type': 'between', 'min': 0.4, 'max': 7000, 'active': True}
    ],

    'volRev_perTick_volxTicksContZone_perTick_over1': [
        {'type': 'between', 'min': 0.6, 'max': 7000, 'active': True}
    ],

    'sc_candleDuration': [
        {'type': 'between', 'min': 2.5, 'max': 10000, 'active': True}
    ],

    'sc_volCandleMeanOver5Ratio': [
        {'type': 'between', 'min': 0, 'max': 3, 'active': False}
    ],

    'is_antiEpuisement_long': [
        {'type': 'between', 'min': 1, 'max': 1, 'active': True}
    ],
    'is_antiSpring_short': [
        {'type': 'between', 'min': 1, 'max': 1, 'active': False}
    ],
}
ALGO_TYPE_COUNT=18
# ────────────────────────────────────────────────────────────────────────────────
# Algorithmes Short uniquement
# ────────────────────────────────────────────────────────────────────────────────

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
    "algoShort16": algoShort14,
    "algoShort17": algoShort17,
    "algoShort18": algoShort18,

}

algoLong1 = {

}

algoLong2 = {

}

algoLong3 = {

}

algoLong4 = {

}

algoLong5 = {

}

algoLong6 = {

}

algoLong7 = {

}

algoLong8 = {

}

algoLong9 = {

}

algoLong10 = {

}

algoLong11 = {

}

algoLong12 = {

}

algoLong13 = {

}

algoLong14 = {

}

algoLong15 = {

}

algoLong16 = {

}

algoLong17 = {

}

algoLong18 = {

}

algorithmsLong = {
    'algoLong1': algoLong1,
    'algoLong2': algoLong2,
    'algoLong3': algoLong3,
    'algoLong4': algoLong4,
    'algoLong5': algoLong5,
    'algoLong6': algoLong6,
    'algoLong7': algoLong7,
    'algoLong8': algoLong8,
    'algoLong9': algoLong9,
    'algoLong10': algoLong10,
    'algoLong11': algoLong11,
    'algoLong12': algoLong12,
    'algoLong13': algoLong13,
    'algoLong14': algoLong14,
    'algoLong15': algoLong15,
    'algoLong16': algoLong16,
    'algoLong17': algoLong17,
    'algoLong18': algoLong18,
}

if DIRECTION == "Short":
    algorithms = algorithmsShort
elif DIRECTION == "Long":
     algorithms = algorithmsLong
else:
    raise ValueError(f"❌ Direction inconnu pour '{DIRECTION}'")

# ════════════════════════════════════════════════════════════════════════════════
# GÉNÉRATEUR C++ COMPLET INTÉGRÉ - AUCUNE DÉPENDANCE EXTERNE !
# ════════════════════════════════════════════════════════════════════════════════

import re
from typing import Dict, List, Any
from datetime import datetime


# ════════════════════════════════════════════════════════════════════════════════

# ────────────────────────────────────────────────────────────────────────────────
# Chargement et pré‑traitement
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


# Chargement des données d'entraînement et de test
df_init_features_train, df_analysis_train, CUSTOM_SESSIONS_TRAIN = load_and_process_data(FILE_PATH_TRAIN)
count_10 = (df_init_features_train['sc_sessionStartEnd'] == 10).sum()
count_20 = (df_init_features_train['sc_sessionStartEnd'] == 20).sum()
if count_10 != count_20 or count_10 <= 0:
    raise ValueError(
        f"Erreur : Le nombre de 10 et 20 dans sc_sessionStartEnd n'est pas égal ou est nul. 10={count_10}, 20={count_20}")
else:
    Nb_session_train = count_10
    print("Le nombre de sessions dans TRAIN est :", Nb_session_train)

# Chargement des données de test
df_init_features_test, df_analysis_test, CUSTOM_SESSIONS_TEST = load_and_process_data(FILE_PATH_TEST)
count_10 = (df_init_features_test['sc_sessionStartEnd'] == 10).sum()
count_20 = (df_init_features_test['sc_sessionStartEnd'] == 20).sum()
if count_10 != count_20 or count_10 <= 0:
    raise ValueError(
        f"Erreur : Le nombre de 10 et 20 dans sc_sessionStartEnd n'est pas égal ou est nul. 10={count_10}, 20={count_20}")
else:
    Nb_session_test = count_10
    print("Le nombre de sessions dans TEST est :", Nb_session_test)

# Chargement des données de validation 1
df_init_features_val1, df_analysis_val1, CUSTOM_SESSIONS_VAL1 = load_and_process_data(FILE_PATH_VAL1)
count_10 = (df_init_features_val1['sc_sessionStartEnd'] == 10).sum()
count_20 = (df_init_features_val1['sc_sessionStartEnd'] == 20).sum()
if count_10 != count_20 or count_10 <= 0:
    raise ValueError(
        f"Erreur : Le nombre de 10 et 20 dans sc_sessionStartEnd n'est pas égal ou est nul. 10={count_10}, 20={count_20}")
else:
    Nb_session_val1 = count_10
    print("Le nombre de sessions dans VAL1 est :", Nb_session_val1)

# Chargement des données de validation 2
df_init_features_val, df_analysis_val, CUSTOM_SESSIONS_VAL = load_and_process_data(FILE_PATH_VAL)
count_10 = (df_init_features_val['sc_sessionStartEnd'] == 10).sum()
count_20 = (df_init_features_val['sc_sessionStartEnd'] == 20).sum()
if count_10 != count_20 or count_10 <= 0:
    raise ValueError(
        f"Erreur : Le nombre de 10 et 20 dans sc_sessionStartEnd n'est pas égal ou est nul. 10={count_10}, 20={count_20}")
else:
    Nb_session_val = count_10
    print("Le nombre de sessions dans VAL est :", Nb_session_val)

# Chargement des données UNSEEN
df_init_features_unseen, df_analysis_unseen, CUSTOM_SESSIONS_UNSEEN = load_and_process_data(FILE_PATH_UNSEEN)
count_10 = (df_init_features_unseen['sc_sessionStartEnd'] == 10).sum()
count_20 = (df_init_features_unseen['sc_sessionStartEnd'] == 20).sum()
if count_10 != count_20 or count_10 <= 0:
    raise ValueError(
        f"Erreur : Le nombre de 10 et 20 dans sc_sessionStartEnd n'est pas égal ou est nul. 10={count_10}, 20={count_20}")
else:
    Nb_session_unseen = count_10
    print("Le nombre de sessions dans UNSEEN est :", Nb_session_unseen)

# Résumé final
print("\nRésumé des sessions chargées :")
print(f"TRAIN  : {Nb_session_train} sessions")
print(f"TEST   : {Nb_session_test} sessions")
print(f"VAL1   : {Nb_session_val1} sessions")
print(f"VAL    : {Nb_session_val} sessions")
print(f"UNSEEN : {Nb_session_unseen} sessions")

import warnings

warnings.filterwarnings('ignore')

GRAPHIC_ANALYSE = False
if GRAPHIC_ANALYSE:
    # ✅ CORRECT (nouvelle version avec sessions intraday)
    results = run_enhanced_trading_analysis_with_sessions(
        df_init_features_train=df_init_features_train,
        df_init_features_test=df_init_features_test,
        df_init_features_val1=df_init_features_val1,
        df_init_features_val=df_init_features_val,
        groupe1=GROUPE_SESSION_1,
        groupe2=GROUPE_SESSION_2,
        xtickReversalTickPrice=XTICKREVERAL_TICKPRICE, period_atr_stat_session=PERDIOD_ATR_SESSION_ANALYSE
    )
    # Export optionnel vers Excel (maintenant avec ATR et contrats extrêmes)
    export_results_to_excel(results, "analyse_trading_complete.xlsx", directory_path=DIRECTORY_PATH)

# ─────────────────────────────────────────────────────────────
# SUPPRESSION DES POSITIONS LONGUES (on conserve uniquement les shorts)
# ─────────────────────────────────────────────────────────────
df_analysis_train = df_analysis_train[df_analysis_train["pos_type"] == DIRECTION].copy()
df_analysis_test = df_analysis_test[df_analysis_test["pos_type"] == DIRECTION].copy()
df_analysis_val1 = df_analysis_val1[df_analysis_val1["pos_type"] == DIRECTION].copy()
df_analysis_val = df_analysis_val[df_analysis_val["pos_type"] == DIRECTION].copy()
print(f"Données d'entraînement: {len(df_analysis_train)} trades")
print(f"Données de test: {len(df_analysis_test)} trades")
print(f"Données de validation 1: {len(df_analysis_val1)} trades")
print(f"Données de validation 2: {len(df_analysis_val)} trades")
print(f"Données unseen: {len(df_analysis_unseen)} trades")  # NOUVEAU



# ════════════════════════════════════════════════════════════════════════════════
# 🎯 AJOUTEZ ICI LA SUITE DE VOTRE CODE D'ANALYSE !
# ════════════════════════════════════════════════════════════════════════════════

# ────────────────────────────────────────────────────────────────────────────────
# Utilitaires
# ────────────────────────────────────────────────────────────────────────────────
def save_csv(df: pd.DataFrame, path, sep=";") -> Path:
    path = Path(path) if not isinstance(path, Path) else path
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep=sep, index=False)
    print(f"✓ Fichier enregistré: {path}")
    return path


import io
import re
from contextlib import redirect_stdout


def header_print(before: dict, after: dict, name: str) -> None:
    """
    Imprime le rapport comparatif en gardant seulement la section
    « => Performance globale - <name> ».
    Toutes les autres sections sont retirées :
        • 📊 STATISTIQUES GLOBALES
        • 📊 ANALYSE DES TRADES LONGS / SHORTS
        • 🎯 TRADES EXTRÊMES LONGS / SHORTS
        • 📑 RÉSUMÉ DE L'IMPACT DU FILTRAGE
        • 📊 SÉQUENCES CONSÉCUTIVES LONGS / SHORTS
    """
    # 1) Générer le rapport complet
    with io.StringIO() as buf, redirect_stdout(buf):
        print_comparative_performance(before, after)
        report = buf.getvalue()

    # 2) En-têtes à supprimer (tout le bloc jusqu'au prochain en-tête ou la fin)
    headers_to_remove = [
        r"📊 STATISTIQUES GLOBALES",
        r"📊 ANALYSE DES TRADES LONGS",
        r"📊 ANALYSE DES TRADES SHORTS",
        r"🎯 TRADES EXTRÊMES LONGS",
        r"🎯 TRADES EXTRÊMES SHORTS",
        r"📑 RÉSUMÉ DE L'IMPACT DU FILTRAGE",
        r"📊 SÉQUENCES CONSÉCUTIVES LONGS",
        r"📊 SÉQUENCES CONSÉCUTIVES SHORTS",
    ]

    pattern = (
        rf"^({'|'.join(headers_to_remove)})[^\n]*\n"  # ligne-titre complète
        rf"(.*?\n)*?"  # contenu éventuel
        rf"(?=^📊|^📈|^🎯|^📑|^Win Rate après|$\Z)"  # borne suivante
    )

    report = re.sub(pattern, "", report, flags=re.DOTALL | re.MULTILINE)

    # 3) Suffixe « - <name> » uniquement sur PERFORMANCE GLOBALE
    report = report.replace("PERFORMANCE GLOBALE",
                            f"PERFORMANCE GLOBALE - {name}")

    # 4) Affichage final
    print(report)


# Train - seuil p70
p70_mvpt20_train = np.percentile(df_analysis_train['meanVol_perTick_over20'], 70)
print(f"[Train] p70_mvpt20 : {p70_mvpt20_train:.2f}")

# Moyenne et écart-type sur Train pour calculer z-score
mu_train = df_analysis_train['meanVol_perTick_over20'].mean()
sigma_train = df_analysis_train['meanVol_perTick_over20'].std()
print(f"[Train] Mean = {mu_train:.2f}, Std = {sigma_train:.2f}")


# Fonction d'affichage des z-scores
def compute_fz(df_split, split_name):
    df_split['z_meanVol_perTick_over20'] = (df_split['meanVol_perTick_over20'] - mu_train) / sigma_train
    p70_z = np.percentile(df_split['z_meanVol_perTick_over20'], 70)
    print(f"[{split_name}] z-score P70 : {p70_z:.2f}")


z_train = (p70_mvpt20_train - mu_train) / sigma_train
print(f"[Train] z-score P70 : {z_train:.2f}")
# Test
p70_mvpt20_test = np.percentile(df_analysis_test['meanVol_perTick_over20'], 70)
print(f"[Test]  p70_mvpt20 : {p70_mvpt20_test:.2f}")
compute_fz(df_analysis_test, 'Test')

# Val1
p70_mvpt20_val1 = np.percentile(df_analysis_val1['meanVol_perTick_over20'], 70)
print(f"[Val1]  p70_mvpt20 : {p70_mvpt20_val1:.2f}")
compute_fz(df_analysis_val1, 'Val1')

# Val
p70_mvpt20_val = np.percentile(df_analysis_val['meanVol_perTick_over20'], 70)
print(f"[Val]   p70_mvpt20 : {p70_mvpt20_val:.2f}")
compute_fz(df_analysis_val, 'Val')

# Unseen
p70_mvpt20_unseen = np.percentile(df_analysis_unseen['meanVol_perTick_over20'], 70)
print(f"[Unseen] p70_mvpt20 : {p70_mvpt20_unseen:.2f}")
compute_fz(df_analysis_unseen, 'Unseen')


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
                else:
                    raise ValueError(f"❌ Type de condition inconnu pour '{feature}': '{cond_type}'")

    return df_filtered


def evaluate_algorithms_with_global_filter(df_analysis, df_init_features, algorithms, global_filter=None,
                                           dataset_name="Train"):
    """Évalue les algorithmes sur un jeu de données avec un filtre global optionnel."""
    print(f"\033[94m\n{'=' * 80}\nÉVALUATION SUR DATASET {dataset_name}\n{'=' * 80}\033[0m")

    # Appliquer le filtre global AVANT l'évaluation des algorithmes
    if global_filter:
        df_analysis_filtered = apply_global_filter(df_analysis, global_filter)
    else:
        df_analysis_filtered = df_analysis

    results = {}
    to_save = []
    metrics_before = calculate_performance_metrics(df_analysis_filtered)

    for algo_name, cond in algorithms.items():
        print(f"🎯{'-' * 4}ÉVALUATION DE {algo_name} - {dataset_name}{'-' * 4}")
        df_filt = apply_feature_conditions(df_analysis_filtered, cond)

        # CORRECTION: Vérifier que df_filt contient bien des données
        if len(df_filt) == 0:
            print(f"ATTENTION: Aucun trade ne satisfait les conditions pour {algo_name}")
            continue

        # Ajouter un affichage détaillé du PnL pour debug
        pnl_sum = df_filt["trade_pnl"].sum()

        # Compter les trades par type (long/short)
        trade_types = df_filt["pos_type"].value_counts()

        # Calculer le PnL par type de position
        pnl_by_type = df_filt.groupby("pos_type")["trade_pnl"].sum()

        # Créer le dataframe avec PnL filtré
        df_full = preprocess_sessions_with_date(
            create_full_dataframe_with_filtered_pnl(df_init_features, df_filt)
        )

        # CORRECTION: Vérifier et afficher la somme des PnL après filtrage
        pnl_after_filtering = df_full["PnlAfterFiltering"].sum()

        # Comparaison pour détecter les incohérences
        if abs(pnl_sum - pnl_after_filtering) > 1.0:  # Tolérance pour erreurs d'arrondi
            print(
                f"ATTENTION: Incohérence détectée entre la somme des trade_pnl ({pnl_sum}) et PnlAfterFiltering ({pnl_after_filtering})")
            # CORRECTION: Utiliser le PnL original plutôt que celui après filtrage
            use_original_pnl = True
        else:
            use_original_pnl = False

        metrics_after = calculate_performance_metrics(df_filt)
        header_print(metrics_before, metrics_after, f"{algo_name} - {dataset_name}")

        wins_a = (df_filt["trade_pnl"] > 0).sum()
        fails_a = (df_filt["trade_pnl"] <= 0).sum()
        win_rate_a = wins_a / (wins_a + fails_a) * 100 if (wins_a + fails_a) > 0 else 0

        # CORRECTION: Utiliser la somme directe des trade_pnl si une incohérence a été détectée
        if use_original_pnl:
            pnl_a = pnl_sum
            print(f"CORRECTION: Utilisation de la somme directe des trade_pnl comme Net PnL")
        else:
            pnl_a = pnl_after_filtering

        profits_a = df_filt.loc[df_filt["trade_pnl"] > 0, "trade_pnl"].sum()
        losses_a = abs(df_filt.loc[df_filt["trade_pnl"] < 0, "trade_pnl"].sum())
        pf_a = profits_a / losses_a if losses_a else 0

        results[algo_name] = {
            "Nombre de trades": len(df_filt),
            "Net PnL": pnl_a,
            "Win Rate (%)": win_rate_a,
            "Profit Factor": pf_a,
        }

        # on conserve UNIQUEMENT les trades sélectionnés par l'algo
        df_selected = df_full[df_full["PnlAfterFiltering"] != 0].copy()

        # CORRECTION: Si nécessaire, recalculer la colonne PnlAfterFiltering
        if use_original_pnl and 'trade_pnl' in df_selected.columns:
            df_selected["PnlAfterFiltering"] = df_selected["trade_pnl"]
            print(f"CORRECTION: La colonne PnlAfterFiltering a été remplacée par trade_pnl")
        to_save.append((algo_name, df_selected))

    return results, to_save


# ────────────────────────────────────────────────────────────────────────────────
# Évaluation sur les jeux de données d'entraînement et de test
# ────────────────────────────────────────────────────────────────────────────────
# Remplacer les lignes originales par :
results_train, to_save_train = evaluate_algorithms_with_global_filter(
    df_analysis_train, df_init_features_train, algorithms, GLOBAL_MICRO_FILTER, "Train"
)
results_test, to_save_test = evaluate_algorithms_with_global_filter(
    df_analysis_test, df_init_features_test, algorithms, GLOBAL_MICRO_FILTER, "Test"
)
results_val1, to_save_val1 = evaluate_algorithms_with_global_filter(
    df_analysis_val1, df_init_features_val1, algorithms, GLOBAL_MICRO_FILTER, "Val1"
)
results_val, to_save_val = evaluate_algorithms_with_global_filter(
    df_analysis_val, df_init_features_val, algorithms, GLOBAL_MICRO_FILTER, "Val"
)
results_unseen, to_save_unseen = evaluate_algorithms_with_global_filter(
    df_analysis_unseen, df_init_features_unseen, algorithms, GLOBAL_MICRO_FILTER, "Unseen"
)

# ────────────────────────────────────────────────────────────────────────────────
# Création du tableau comparatif avec les deux jeux de données
# ────────────────────────────────────────────────────────────────────────────────
# Créer les DataFrames individuels pour l'entraînement et le test
# Create DataFrames
comparison_train = pd.DataFrame(results_train).T.reset_index().rename(columns={"index": "Algorithme"})
comparison_test = pd.DataFrame(results_test).T.reset_index().rename(columns={"index": "Algorithme"})
comparison_val1 = pd.DataFrame(results_val1).T.reset_index().rename(columns={"index": "Algorithme"})
comparison_val = pd.DataFrame(results_val).T.reset_index().rename(columns={"index": "Algorithme"})
comparison_unseen = pd.DataFrame(results_unseen).T.reset_index().rename(columns={"index": "Algorithme"})


# NEW CODE (FIXED):
def safe_add_exp_pnl(df, dataset_name="Unknown"):
    """Safely add Exp PnL column with error handling"""
    print(f"🔧 Processing {dataset_name} dataset...")

    # Debug - print available columns
    print(f"   Available columns: {list(df.columns)}")

    # Look for PnL column (multiple possible names)
    pnl_col = None
    possible_pnl_names = ['Net PnL', 'PnL', 'pnl', 'net_pnl', 'NetPnL']

    for name in possible_pnl_names:
        if name in df.columns:
            pnl_col = name
            break

    # Look for trade count column
    trade_col = None
    possible_trade_names = ['Nombre de trades', 'trades', 'trade_count', 'Nombre de trades (Train)']

    for name in possible_trade_names:
        if name in df.columns:
            trade_col = name
            break

    if pnl_col and trade_col:
        # Safe division with error handling
        def safe_divide(row):
            try:
                if pd.isna(row[pnl_col]) or pd.isna(row[trade_col]) or row[trade_col] == 0:
                    return 0.0
                return round(row[pnl_col] / row[trade_col], 2)
            except:
                return 0.0

        df["Exp PnL"] = df.apply(safe_divide, axis=1)
        print(f"   ✅ Successfully added Exp PnL using {pnl_col} / {trade_col}")
    else:
        # Fallback - add placeholder column
        df["Exp PnL"] = 0.0
        print(f"   ⚠️  Added placeholder Exp PnL (missing: PnL={pnl_col}, Trades={trade_col})")

    return df


comparison_train = safe_add_exp_pnl(comparison_train, "Train")
comparison_test = safe_add_exp_pnl(comparison_test, "Test")
comparison_val1 = safe_add_exp_pnl(comparison_val1, "Val1")
comparison_val = safe_add_exp_pnl(comparison_val, "Val")
comparison_unseen = safe_add_exp_pnl(comparison_unseen, "Unseen")

# Renommer les colonnes pour distinguer les datasets
comparison_test_renamed = comparison_test.rename(columns={
    "Nombre de trades": "Nombre de trades (Test)",
    "Net PnL": "Net PnL (Test)",
    "Exp PnL": "Exp PnL (Test)",
    "Win Rate (%)": "Win Rate (%) (Test)",
    "Profit Factor": "Profit Factor (Test)",
})

comparison_val1_renamed = comparison_val1.rename(columns={
    "Nombre de trades": "Nombre de trades (Val1)",
    "Net PnL": "Net PnL (Val1)",
    "Exp PnL": "Exp PnL (Val1)",
    "Win Rate (%)": "Win Rate (%) (Val1)",
    "Profit Factor": "Profit Factor (Val1)",
})

comparison_val_renamed = comparison_val.rename(columns={
    "Nombre de trades": "Nombre de trades (Val)",
    "Net PnL": "Net PnL (Val)",
    "Exp PnL": "Exp PnL (Val)",
    "Win Rate (%)": "Win Rate (%) (Val)",
    "Profit Factor": "Profit Factor (Val)",
})
comparison_unseen_renamed = comparison_unseen.rename(columns={
    "Nombre de trades": "Nombre de trades (Unseen)",
    "Net PnL": "Net PnL (Unseen)",
    "Exp PnL": "Exp PnL (Unseen)",
    "Win Rate (%)": "Win Rate (%) (Unseen)",
    "Profit Factor": "Profit Factor (Unseen)",
})


# Fusionner les 4 DataFrames
# Fonction pour créer un DataFrame vide avec les bonnes colonnes
def create_empty_comparison_df(dataset_name):
    """Crée un DataFrame vide avec les colonnes attendues pour un dataset"""
    return pd.DataFrame(columns=[
        "Algorithme",
        f"Nombre de trades ({dataset_name})",
        f"Net PnL ({dataset_name})",
        f"Exp PnL ({dataset_name})",
        f"Win Rate (%) ({dataset_name})",
        f"Profit Factor ({dataset_name})"
    ])


# Vérifier et corriger chaque DataFrame avant le merge
def safe_prepare_comparison(comparison_df, dataset_name):
    """Prépare un DataFrame de comparaison en gérant les cas vides"""
    if len(comparison_df) == 0 or comparison_df.empty:
        print(f"⚠️ Dataset {dataset_name} vide - création d'un DataFrame placeholder")
        return create_empty_comparison_df(dataset_name)
    return comparison_df


# Préparer les DataFrames renommés de manière sécurisée
comparison_test_safe = safe_prepare_comparison(comparison_test_renamed, "Test")
comparison_val1_safe = safe_prepare_comparison(comparison_val1_renamed, "Val1")
comparison_val_safe = safe_prepare_comparison(comparison_val_renamed, "Val")
comparison_unseen_safe = safe_prepare_comparison(comparison_unseen_renamed, "Unseen")

# Fusionner les DataFrames (maintenant tous avec les bonnes colonnes)
comparison_merged = comparison_train.copy()

# Merges avec gestion des DataFrames vides
comparison_merged = pd.merge(comparison_merged, comparison_test_safe[
    ["Algorithme", "Nombre de trades (Test)", "Net PnL (Test)",
     "Exp PnL (Test)", "Win Rate (%) (Test)", "Profit Factor (Test)"]
], on="Algorithme", how="left")

comparison_merged = pd.merge(comparison_merged, comparison_val1_safe[
    ["Algorithme", "Nombre de trades (Val1)", "Net PnL (Val1)",
     "Exp PnL (Val1)", "Win Rate (%) (Val1)", "Profit Factor (Val1)"]
], on="Algorithme", how="left")

comparison_merged = pd.merge(comparison_merged, comparison_val_safe[
    ["Algorithme", "Nombre de trades (Val)", "Net PnL (Val)",
     "Exp PnL (Val)", "Win Rate (%) (Val)", "Profit Factor (Val)"]
], on="Algorithme", how="left")

comparison_merged = pd.merge(comparison_merged, comparison_unseen_safe[
    ["Algorithme", "Nombre de trades (Unseen)", "Net PnL (Unseen)",
     "Exp PnL (Unseen)", "Win Rate (%) (Unseen)", "Profit Factor (Unseen)"]
], on="Algorithme", how="left")



# ────────────────────────────────────────────────────────────────────────────────
# Utilitaires
# ────────────────────────────────────────────────────────────────────────────────
def save_csv(df: pd.DataFrame, path, sep=";") -> Path:
    path = Path(path) if not isinstance(path, Path) else path
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep=sep, index=False)
    print(f"✓ Fichier enregistré: {path}")
    return path

import io
import re
from contextlib import redirect_stdout


def header_print(before: dict, after: dict, name: str) -> None:
    """
    Imprime le rapport comparatif en gardant seulement la section
    « => Performance globale - <name> ».
    Toutes les autres sections sont retirées :
        • 📊 STATISTIQUES GLOBALES
        • 📊 ANALYSE DES TRADES LONGS / SHORTS
        • 🎯 TRADES EXTRÊMES LONGS / SHORTS
        • 📑 RÉSUMÉ DE L'IMPACT DU FILTRAGE
        • 📊 SÉQUENCES CONSÉCUTIVES LONGS / SHORTS
    """
    # 1) Générer le rapport complet
    with io.StringIO() as buf, redirect_stdout(buf):
        print_comparative_performance(before, after)
        report = buf.getvalue()

    # 2) En-têtes à supprimer (tout le bloc jusqu’au prochain en-tête ou la fin)
    headers_to_remove = [
        r"📊 STATISTIQUES GLOBALES",
        r"📊 ANALYSE DES TRADES LONGS",
        r"📊 ANALYSE DES TRADES SHORTS",
        r"🎯 TRADES EXTRÊMES LONGS",
        r"🎯 TRADES EXTRÊMES SHORTS",
        r"📑 RÉSUMÉ DE L'IMPACT DU FILTRAGE",
        r"📊 SÉQUENCES CONSÉCUTIVES LONGS",
        r"📊 SÉQUENCES CONSÉCUTIVES SHORTS",
    ]

    pattern = (
        rf"^({'|'.join(headers_to_remove)})[^\n]*\n"   # ligne-titre complète
        rf"(.*?\n)*?"                                  # contenu éventuel
        rf"(?=^📊|^📈|^🎯|^📑|^Win Rate après|$\Z)"      # borne suivante
    )

    report = re.sub(pattern, "", report, flags=re.DOTALL | re.MULTILINE)

    # 3) Suffixe « - <name> » uniquement sur PERFORMANCE GLOBALE
    report = report.replace("PERFORMANCE GLOBALE",
                            f"PERFORMANCE GLOBALE - {name}")

    # 4) Affichage final
    print(report)



# Train - seuil p70
p70_mvpt20_train = np.percentile(df_analysis_train['meanVol_perTick_over20'], 70)
print(f"[Train] p70_mvpt20 : {p70_mvpt20_train:.2f}")

# Moyenne et écart-type sur Train pour calculer z-score
mu_train = df_analysis_train['meanVol_perTick_over20'].mean()
sigma_train = df_analysis_train['meanVol_perTick_over20'].std()
print(f"[Train] Mean = {mu_train:.2f}, Std = {sigma_train:.2f}")

# Fonction d'affichage des z-scores
def compute_fz(df_split, split_name):
    df_split['z_meanVol_perTick_over20'] = (df_split['meanVol_perTick_over20'] - mu_train) / sigma_train
    p70_z = np.percentile(df_split['z_meanVol_perTick_over20'], 70)
    print(f"[{split_name}] z-score P70 : {p70_z:.2f}")
z_train = (p70_mvpt20_train - mu_train) / sigma_train
print(f"[Train] z-score P70 : {z_train:.2f}")
# Test
p70_mvpt20_test = np.percentile(df_analysis_test['meanVol_perTick_over20'], 70)
print(f"[Test]  p70_mvpt20 : {p70_mvpt20_test:.2f}")
compute_fz(df_analysis_test, 'Test')

# Val1
p70_mvpt20_val1 = np.percentile(df_analysis_val1['meanVol_perTick_over20'], 70)
print(f"[Val1]  p70_mvpt20 : {p70_mvpt20_val1:.2f}")
compute_fz(df_analysis_val1, 'Val1')

# Val
p70_mvpt20_val = np.percentile(df_analysis_val['meanVol_perTick_over20'], 70)
print(f"[Val]   p70_mvpt20 : {p70_mvpt20_val:.2f}")
compute_fz(df_analysis_val, 'Val')

# Unseen
p70_mvpt20_unseen = np.percentile(df_analysis_unseen['meanVol_perTick_over20'], 70)
print(f"[Unseen] p70_mvpt20 : {p70_mvpt20_unseen:.2f}")
compute_fz(df_analysis_unseen, 'Unseen')
#exit(444)
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
                else:
                    raise ValueError(f"❌ Type de condition inconnu pour '{feature}': '{cond_type}'")

    return df_filtered


def evaluate_algorithms_with_global_filter(df_analysis, df_init_features, algorithms, global_filter=None,
                                           dataset_name="Train"):
    """Évalue les algorithmes sur un jeu de données avec un filtre global optionnel."""
    print(f"\033[94m\n{'=' * 80}\nÉVALUATION SUR DATASET {dataset_name}\n{'=' * 80}\033[0m")

    # Appliquer le filtre global AVANT l'évaluation des algorithmes
    if global_filter:
        df_analysis_filtered = apply_global_filter(df_analysis, global_filter)
    else:
        df_analysis_filtered = df_analysis

    results = {}
    to_save = []
    metrics_before = calculate_performance_metrics(df_analysis_filtered)

    for algo_name, cond in algorithms.items():
        print(f"🎯{'-' * 4}ÉVALUATION DE {algo_name} - {dataset_name}{'-' * 4}")
        df_filt = apply_feature_conditions(df_analysis_filtered, cond)

        # CORRECTION: Vérifier que df_filt contient bien des données
        if len(df_filt) == 0:
            print(f"ATTENTION: Aucun trade ne satisfait les conditions pour {algo_name}")
            continue

        # Ajouter un affichage détaillé du PnL pour debug
        pnl_sum = df_filt["trade_pnl"].sum()

        # Compter les trades par type (long/short)
        trade_types = df_filt["pos_type"].value_counts()

        # Calculer le PnL par type de position
        pnl_by_type = df_filt.groupby("pos_type")["trade_pnl"].sum()

        # Créer le dataframe avec PnL filtré
        df_full = preprocess_sessions_with_date(
            create_full_dataframe_with_filtered_pnl(df_init_features, df_filt)
        )

        # CORRECTION: Vérifier et afficher la somme des PnL après filtrage
        pnl_after_filtering = df_full["PnlAfterFiltering"].sum()

        # Comparaison pour détecter les incohérences
        if abs(pnl_sum - pnl_after_filtering) > 1.0:  # Tolérance pour erreurs d'arrondi
            print(
                f"ATTENTION: Incohérence détectée entre la somme des trade_pnl ({pnl_sum}) et PnlAfterFiltering ({pnl_after_filtering})")
            # CORRECTION: Utiliser le PnL original plutôt que celui après filtrage
            use_original_pnl = True
        else:
            use_original_pnl = False

        metrics_after = calculate_performance_metrics(df_filt)
        header_print(metrics_before, metrics_after, f"{algo_name} - {dataset_name}")

        wins_a = (df_filt["trade_pnl"] > 0).sum()
        fails_a = (df_filt["trade_pnl"] <= 0).sum()
        win_rate_a = wins_a / (wins_a + fails_a) * 100 if (wins_a + fails_a) > 0 else 0

        # CORRECTION: Utiliser la somme directe des trade_pnl si une incohérence a été détectée
        if use_original_pnl:
            pnl_a = pnl_sum
            print(f"CORRECTION: Utilisation de la somme directe des trade_pnl comme Net PnL")
        else:
            pnl_a = pnl_after_filtering

        profits_a = df_filt.loc[df_filt["trade_pnl"] > 0, "trade_pnl"].sum()
        losses_a = abs(df_filt.loc[df_filt["trade_pnl"] < 0, "trade_pnl"].sum())
        pf_a = profits_a / losses_a if losses_a else 0

        results[algo_name] = {
            "Nombre de trades": len(df_filt),
            "Net PnL": pnl_a,
            "Win Rate (%)": win_rate_a,
            "Profit Factor": pf_a,
        }

        # on conserve UNIQUEMENT les trades sélectionnés par l'algo
        df_selected = df_full[df_full["PnlAfterFiltering"] != 0].copy()

        # CORRECTION: Si nécessaire, recalculer la colonne PnlAfterFiltering
        if use_original_pnl and 'trade_pnl' in df_selected.columns:
            df_selected["PnlAfterFiltering"] = df_selected["trade_pnl"]
            print(f"CORRECTION: La colonne PnlAfterFiltering a été remplacée par trade_pnl")
        to_save.append((algo_name, df_selected))

    return results, to_save


# ────────────────────────────────────────────────────────────────────────────────
# Évaluation sur les jeux de données d'entraînement et de test
# ────────────────────────────────────────────────────────────────────────────────
# Remplacer les lignes originales par :
results_train, to_save_train = evaluate_algorithms_with_global_filter(
    df_analysis_train, df_init_features_train, algorithms, GLOBAL_MICRO_FILTER, "Train"
)
results_test, to_save_test = evaluate_algorithms_with_global_filter(
    df_analysis_test, df_init_features_test, algorithms, GLOBAL_MICRO_FILTER, "Test"
)
results_val1, to_save_val1 = evaluate_algorithms_with_global_filter(
    df_analysis_val1, df_init_features_val1, algorithms, GLOBAL_MICRO_FILTER, "Val1"
)
results_val, to_save_val = evaluate_algorithms_with_global_filter(
    df_analysis_val, df_init_features_val, algorithms, GLOBAL_MICRO_FILTER, "Val"
)
results_unseen, to_save_unseen = evaluate_algorithms_with_global_filter(
    df_analysis_unseen, df_init_features_unseen, algorithms, GLOBAL_MICRO_FILTER, "Unseen"
)

# ────────────────────────────────────────────────────────────────────────────────
# Création du tableau comparatif avec les deux jeux de données
# ────────────────────────────────────────────────────────────────────────────────
# Créer les DataFrames individuels pour l'entraînement et le test
# Create DataFrames
comparison_train = pd.DataFrame(results_train).T.reset_index().rename(columns={"index": "Algorithme"})
comparison_test = pd.DataFrame(results_test).T.reset_index().rename(columns={"index": "Algorithme"})
comparison_val1 = pd.DataFrame(results_val1).T.reset_index().rename(columns={"index": "Algorithme"})
comparison_val = pd.DataFrame(results_val).T.reset_index().rename(columns={"index": "Algorithme"})
comparison_unseen = pd.DataFrame(results_unseen).T.reset_index().rename(columns={"index": "Algorithme"})


# NEW CODE (FIXED):
def safe_add_exp_pnl(df, dataset_name="Unknown"):
    """Safely add Exp PnL column with error handling"""
    print(f"🔧 Processing {dataset_name} dataset...")

    # Debug - print available columns
    print(f"   Available columns: {list(df.columns)}")

    # Look for PnL column (multiple possible names)
    pnl_col = None
    possible_pnl_names = ['Net PnL', 'PnL', 'pnl', 'net_pnl', 'NetPnL']

    for name in possible_pnl_names:
        if name in df.columns:
            pnl_col = name
            break

    # Look for trade count column
    trade_col = None
    possible_trade_names = ['Nombre de trades', 'trades', 'trade_count', 'Nombre de trades (Train)']

    for name in possible_trade_names:
        if name in df.columns:
            trade_col = name
            break

    if pnl_col and trade_col:
        # Safe division with error handling
        def safe_divide(row):
            try:
                if pd.isna(row[pnl_col]) or pd.isna(row[trade_col]) or row[trade_col] == 0:
                    return 0.0
                return round(row[pnl_col] / row[trade_col], 2)
            except:
                return 0.0

        df["Exp PnL"] = df.apply(safe_divide, axis=1)
        print(f"   ✅ Successfully added Exp PnL using {pnl_col} / {trade_col}")
    else:
        # Fallback - add placeholder column
        df["Exp PnL"] = 0.0
        print(f"   ⚠️  Added placeholder Exp PnL (missing: PnL={pnl_col}, Trades={trade_col})")

    return df
comparison_train = safe_add_exp_pnl(comparison_train, "Train")
comparison_test = safe_add_exp_pnl(comparison_test, "Test")
comparison_val1 = safe_add_exp_pnl(comparison_val1, "Val1")
comparison_val = safe_add_exp_pnl(comparison_val, "Val")
comparison_unseen = safe_add_exp_pnl(comparison_unseen, "Unseen")
# Ajouter les colonnes pour Exp PnL pour tous les datasets
# for df in [comparison_train, comparison_test, comparison_val1, comparison_val, comparison_unseen]:
#     df["Exp PnL"] = (df["Net PnL"] / df["Nombre de trades"]).round(2)

# Renommer les colonnes pour distinguer les datasets
comparison_test_renamed = comparison_test.rename(columns={
    "Nombre de trades": "Nombre de trades (Test)",
    "Net PnL": "Net PnL (Test)",
    "Exp PnL": "Exp PnL (Test)",
    "Win Rate (%)": "Win Rate (%) (Test)",
    "Profit Factor": "Profit Factor (Test)",
})

comparison_val1_renamed = comparison_val1.rename(columns={
    "Nombre de trades": "Nombre de trades (Val1)",
    "Net PnL": "Net PnL (Val1)",
    "Exp PnL": "Exp PnL (Val1)",
    "Win Rate (%)": "Win Rate (%) (Val1)",
    "Profit Factor": "Profit Factor (Val1)",
})

comparison_val_renamed = comparison_val.rename(columns={
    "Nombre de trades": "Nombre de trades (Val)",
    "Net PnL": "Net PnL (Val)",
    "Exp PnL": "Exp PnL (Val)",
    "Win Rate (%)": "Win Rate (%) (Val)",
    "Profit Factor": "Profit Factor (Val)",
})
comparison_unseen_renamed = comparison_unseen.rename(columns={
    "Nombre de trades": "Nombre de trades (Unseen)",
    "Net PnL": "Net PnL (Unseen)",
    "Exp PnL": "Exp PnL (Unseen)",
    "Win Rate (%)": "Win Rate (%) (Unseen)",
    "Profit Factor": "Profit Factor (Unseen)",
})
# Fusionner les 4 DataFrames
# Fonction pour créer un DataFrame vide avec les bonnes colonnes
def create_empty_comparison_df(dataset_name):
    """Crée un DataFrame vide avec les colonnes attendues pour un dataset"""
    return pd.DataFrame(columns=[
        "Algorithme",
        f"Nombre de trades ({dataset_name})",
        f"Net PnL ({dataset_name})",
        f"Exp PnL ({dataset_name})",
        f"Win Rate (%) ({dataset_name})",
        f"Profit Factor ({dataset_name})"
    ])

# Vérifier et corriger chaque DataFrame avant le merge
def safe_prepare_comparison(comparison_df, dataset_name):
    """Prépare un DataFrame de comparaison en gérant les cas vides"""
    if len(comparison_df) == 0 or comparison_df.empty:
        print(f"⚠️ Dataset {dataset_name} vide - création d'un DataFrame placeholder")
        return create_empty_comparison_df(dataset_name)
    return comparison_df

# Préparer les DataFrames renommés de manière sécurisée
comparison_test_safe = safe_prepare_comparison(comparison_test_renamed, "Test")
comparison_val1_safe = safe_prepare_comparison(comparison_val1_renamed, "Val1")
comparison_val_safe = safe_prepare_comparison(comparison_val_renamed, "Val")
comparison_unseen_safe = safe_prepare_comparison(comparison_unseen_renamed, "Unseen")

# Fusionner les DataFrames (maintenant tous avec les bonnes colonnes)
comparison_merged = comparison_train.copy()

# Merges avec gestion des DataFrames vides
comparison_merged = pd.merge(comparison_merged, comparison_test_safe[
    ["Algorithme", "Nombre de trades (Test)", "Net PnL (Test)",
     "Exp PnL (Test)", "Win Rate (%) (Test)", "Profit Factor (Test)"]
], on="Algorithme", how="left")

comparison_merged = pd.merge(comparison_merged, comparison_val1_safe[
    ["Algorithme", "Nombre de trades (Val1)", "Net PnL (Val1)",
     "Exp PnL (Val1)", "Win Rate (%) (Val1)", "Profit Factor (Val1)"]
], on="Algorithme", how="left")

comparison_merged = pd.merge(comparison_merged, comparison_val_safe[
    ["Algorithme", "Nombre de trades (Val)", "Net PnL (Val)",
     "Exp PnL (Val)", "Win Rate (%) (Val)", "Profit Factor (Val)"]
], on="Algorithme", how="left")

comparison_merged = pd.merge(comparison_merged, comparison_unseen_safe[
    ["Algorithme", "Nombre de trades (Unseen)", "Net PnL (Unseen)",
     "Exp PnL (Unseen)", "Win Rate (%) (Unseen)", "Profit Factor (Unseen)"]
], on="Algorithme", how="left")

# Remplacer les NaN par des valeurs par défaut pour les datasets vides
comparison_merged = comparison_merged.fillna({
    "Nombre de trades (Unseen)": 0,
    "Net PnL (Unseen)": 0.0,
    "Exp PnL (Unseen)": 0.0,
    "Win Rate (%) (Unseen)": 0.0,
    "Profit Factor (Unseen)": 0.0
})

print("✅ Merge terminé avec gestion des datasets vides")
# Créer la liste des features utilisées (comme dans le code original)
all_features = sorted({feat for d in algorithms.values() for feat in d})

# Pour chaque feature, inscrire "x" pour les algos qui l'emploient
for feat in all_features:
    comparison_merged[feat] = np.where(
        comparison_merged["Algorithme"].map(lambda a: feat in algorithms[a]), "x", ""
    )

# Ordonner les colonnes pour les 4 datasets
cols_metrics = [
    "Algorithme",
    "Nombre de trades", "Nombre de trades (Test)", "Nombre de trades (Val1)", "Nombre de trades (Val)", "Nombre de trades (Unseen)",
    "Net PnL", "Net PnL (Test)", "Net PnL (Val1)", "Net PnL (Val)", "Net PnL (Unseen)",
    "Exp PnL", "Exp PnL (Test)", "Exp PnL (Val1)", "Exp PnL (Val)", "Exp PnL (Unseen)",
    "Win Rate (%)", "Win Rate (%) (Test)", "Win Rate (%) (Val1)", "Win Rate (%) (Val)", "Win Rate (%) (Unseen)",
    "Profit Factor", "Profit Factor (Test)", "Profit Factor (Val1)", "Profit Factor (Val)", "Profit Factor (Unseen)",
]


# Créer deux tableaux séparés
# NOUVEAU CODE - SÉCURISÉ ✅
def safe_select_columns(df, requested_cols, purpose=""):
    """Sélectionne seulement les colonnes qui existent"""
    existing_cols = [col for col in requested_cols if col in df.columns]
    missing_cols = [col for col in requested_cols if col not in df.columns]

    if missing_cols:
        print(f"⚠️  {purpose} - Colonnes manquantes: {missing_cols}")
        print(f"✅ {purpose} - Utilisation de {len(existing_cols)}/{len(requested_cols)} colonnes disponibles")

    return df[existing_cols].copy()


# Sélection sécurisée pour comparison_metrics
comparison_metrics = safe_select_columns(comparison_merged, cols_metrics, "comparison_metrics")

# Sélection sécurisée pour comparison_features
if 'all_features' in locals():
    feature_cols = ["Algorithme"] + all_features
    comparison_features = safe_select_columns(comparison_merged, feature_cols, "comparison_features")
else:
    print("⚠️  all_features non défini - création avec Algorithme seulement")
    comparison_features = comparison_merged[["Algorithme"]].copy()

print("🎯 Sélection de colonnes corrigée!")

def analyse_doublons_algos(
        algo_dfs: dict[str, pd.DataFrame],
        indicator_columns: list[str] | None = None,
        min_common_trades: int = 20,
        directory_path: str | Path = None  # Ajout du paramètre pour les sauvegardes
) -> None:
    """
    Analyse les doublons entre différents algorithmes de trading.
    Travaille sur un dict {nom_algo: dataframe} au lieu de lire des CSV.

    Args:
        algo_dfs: Dictionnaire {nom_algo: DataFrame}
        indicator_columns: Colonnes à utiliser pour détecter les doublons
        min_common_trades: Nombre min de trades communs pour analyse des paires
        directory_path: Répertoire pour sauvegarder les résultats (optionnel)
    """
    if directory_path is not None:
        directory_path = Path(directory_path) if not isinstance(directory_path, Path) else directory_path
        directory_path.mkdir(parents=True, exist_ok=True)

    if indicator_columns is None:
        indicator_columns = [
            'rsi_', 'macd', 'macd_signal', 'macd_hist',
            'timeElapsed2LastBar', 'timeStampOpening',
            'ratio_deltaRevZone_VolCandle'
        ]

    # 1) Stats par algo + stockage des ensembles uniques
    uniq_sets = {}
    trade_results = {}
    trade_data = {}
    total_rows_before = 0
    file_stats = {}  # Pour statistiques individuelles par algo

    algos = list(algo_dfs.keys())
    for algo, df in algo_dfs.items():
        df = df.copy()
        file_stats[algo] = {'total_rows': len(df)}

        # colonne PnL (on prend la 1ʳᵉ dispo)
        pnl_col = next((c for c in ("PnlAfterFiltering", "trade_pnl") if c in df.columns), None)
        if pnl_col is None:
            print(f"[warn]  Colonne PnL absente pour {algo} – ignoré")
            continue

        total_rows_before += len(df)

        valid_cols = [c for c in indicator_columns if c in df.columns]
        if not valid_cols:
            print(f"Aucune colonne d'indicateur valide trouvée pour {algo}")
            continue

        duplicate_mask = df.duplicated(subset=valid_cols, keep='first')
        dups_int = duplicate_mask.sum()
        file_stats[algo]['duplicates_internal'] = dups_int
        file_stats[algo]['unique_rows'] = len(df) - dups_int

        #print(f"Algorithme {algo}: {len(df)} lignes, {dups_int} doublons internes, {len(df) - dups_int} uniques")

        uniq_sets[algo] = set()
        for _, row in df.drop_duplicates(subset=valid_cols).iterrows():
            key = tuple(row[col] for col in valid_cols)
            uniq_sets[algo].add(key)

            # enregistrement global
            if key not in trade_results:
                trade_results[key] = {}
                trade_data[key] = row.to_dict()

            trade_results[key][algo] = row[pnl_col] > 0

    # 2) matrice inter‑algos
    dup_matrix = pd.DataFrame(0, index=algos, columns=algos, dtype=int)

    for i, a1 in enumerate(algos):
        for a2 in algos[i + 1:]:
            common = len(uniq_sets[a1].intersection(uniq_sets[a2]))
            dup_matrix.loc[a1, a2] = dup_matrix.loc[a2, a1] = common

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    print("\n=== MATRICE DES DOUBLONS ENTRE ALGOS ===")
    print(dup_matrix)

    # Version simplifiée avec noms courts pour meilleure lisibilité
    short_names = {name: f"algo{i}" for i, name in enumerate(algos)}
    readable_matrix = dup_matrix.copy()
    readable_matrix.index = [short_names[name] for name in readable_matrix.index]
    readable_matrix.columns = [short_names[name] for name in readable_matrix.columns]

    # 3) distribution des occurrences
    occ_counts = Counter(len(v) for v in trade_results.values())
    print("\nDistribution des occurrences (nb d'algos dans lesquels apparaît chaque trade) :")
    for k in sorted(occ_counts):
        print(f"  {k} algo(s) : {occ_counts[k]} trades")

    # 4) consolidation globale dé‑dupliquée
    all_keys = set.union(*uniq_sets.values()) if uniq_sets else set()
    global_rows = []
    for key in all_keys:
        if key in trade_data:
            row = trade_data[key]
            row["is_winning_any"] = any(trade_results[key].values())
            row["is_winning_all"] = all(trade_results[key].get(a, False) for a in algos)
            global_rows.append(row)

    global_df = pd.DataFrame(global_rows) if global_rows else pd.DataFrame()

    if not global_df.empty:
        pnl_col = next((c for c in ("PnlAfterFiltering", "trade_pnl") if c in global_df.columns), None)
        if pnl_col:
            wins = (global_df[pnl_col] > 0).sum()
            total = len(global_df)

            # Calculs des métriques globales
            winrate = wins / total * 100 if total > 0 else 0
            total_pnl = global_df[pnl_col].sum()
            total_gains = global_df.loc[global_df[pnl_col] > 0, pnl_col].sum()
            total_losses = global_df.loc[global_df[pnl_col] <= 0, pnl_col].sum()

            # Moyennes et ratios
            avg_win = global_df.loc[global_df[pnl_col] > 0, pnl_col].mean() if wins > 0 else 0
            avg_loss = global_df.loc[global_df[pnl_col] <= 0, pnl_col].mean() if (total - wins) > 0 else 0
            reward_risk_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            expectancy = (winrate / 100 * avg_win) + ((100 - winrate) / 100 * avg_loss)

            print(f"\n=== RÉSUMÉ GLOBAL APRÈS DÉDUPLICATION ===")
            print(f"  Trades totaux avant déduplication: {total_rows_before}")
            print(f"  Trades conservés après déduplication: {total} ({total / total_rows_before * 100:.2f}%)")
            print(f"  Trades réussis: {wins}")
            print(f"  Trades échoués: {total - wins}")
            print(f"  Winrate global: {winrate:.2f}%")
            print(f"  PnL total: {total_pnl:.2f}")
            print(f"  Gains totaux: {total_gains:.2f}")
            print(f"  Pertes totales: {total_losses:.2f}")
            print(f"  Gain moyen: {avg_win:.2f}")
            print(f"  Perte moyenne: {avg_loss:.2f}")
            print(f"  Ratio risque/récompense: {reward_risk_ratio:.2f}")
            print(f" Expectancy par trade: {expectancy:.2f}")

    # 5) Analyse par nombre d'occurrences
    occurrences_stats = {}
    max_occ = max([len(v) for v in trade_results.values()]) if trade_results else 0

    for occ_count in range(2, max_occ + 1):
        trades_with_occ = {k: v for k, v in trade_results.items() if len(v) == occ_count}

        if not trades_with_occ:
            continue

        # Analyse des trades qui apparaissent dans exactement occ_count algos
        winning_trades = []  # Liste des trades gagnants (unanimement)
        losing_trades = []  # Liste des trades non unanimes

        for key, algos_present in trades_with_occ.items():
            # Un trade est considéré comme valide uniquement si TOUS les algos le marquent comme gagnant
            if all(trade_results.get(key, {}).get(algo, False) for algo in algos_present):
                winning_trades.append(key)
            else:
                losing_trades.append(key)

        total_trades = len(winning_trades) + len(losing_trades)
        winrate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0

        # Calculer le PnL total pour les trades gagnants et perdants
        if pnl_col:
            winning_pnl = sum(trade_data.get(key, {}).get(pnl_col, 0) for key in winning_trades)
            losing_pnl = sum(trade_data.get(key, {}).get(pnl_col, 0) for key in losing_trades)
            total_pnl = winning_pnl + losing_pnl

            occurrences_stats[occ_count] = {
                'total_trades': total_trades,
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'winrate': winrate,
                'winning_pnl': winning_pnl,
                'losing_pnl': losing_pnl,
                'total_pnl': total_pnl
            }

            print(f"\n=== ANALYSE DES TRADES APPARAISSANT DANS {occ_count} ALGOS ===")
            print(f"  Nombre total de trades: {total_trades}")
            print(f"  Trades unanimement gagnants: {len(winning_trades)}")
            print(f"  Trades non unanimes (au moins un échec): {len(losing_trades)}")
            print(f"  Winrate (trades unanimement gagnants): {winrate:.2f}%")
            print(f"  PnL total: {total_pnl:.2f}")
            print(f"  PnL des trades gagnants: {winning_pnl:.2f}")
            print(f"  PnL des trades perdants: {losing_pnl:.2f}")

    # 6) analyse des paires d'algos avec > min_common_trades trades communs
    print(f"\n=== ANALYSE DES PAIRES === (> {min_common_trades} trades communs) ===")

    # Éviter les paires redondantes
    analyzed_pairs = set()
    significant_pairs = []

    for i, a1 in enumerate(algos):
        for j, a2 in enumerate(algos[i + 1:], i + 1):
            if dup_matrix.loc[a1, a2] >= min_common_trades:
                significant_pairs.append((a1, a2))
                analyzed_pairs.add((a1, a2))

    pairs_stats = {}

    for a1, a2 in significant_pairs:
        common = uniq_sets[a1].intersection(uniq_sets[a2])
        if len(common) < min_common_trades:
            continue

        # Stats détaillées
        winning_both = 0
        winning_a1_only = 0
        winning_a2_only = 0
        losing_both = 0

        total_pnl = 0
        unanimous_pnl = 0

        for key in common:
            result_a1 = trade_results.get(key, {}).get(a1, False)
            result_a2 = trade_results.get(key, {}).get(a2, False)

            # Compter les cas selon le résultat
            if result_a1 and result_a2:
                winning_both += 1
            elif result_a1 and not result_a2:
                winning_a1_only += 1
            elif not result_a1 and result_a2:
                winning_a2_only += 1
            else:
                losing_both += 1

            # Calculer le PnL si disponible
            if key in trade_data and pnl_col in trade_data[key]:
                pnl = trade_data[key][pnl_col]
                total_pnl += pnl

                # Ajouter au PnL unanime si les deux algos sont d'accord
                if result_a1 == result_a2:
                    unanimous_pnl += pnl

        # Calculer l'accord entre les algos
        agreement_rate = (winning_both + losing_both) / len(common) * 100 if common else 0

        # Stocker les statistiques
        pairs_stats[(a1, a2)] = {
            'common_trades': len(common),
            'winning_both': winning_both,
            'winning_a1_only': winning_a1_only,
            'winning_a2_only': winning_a2_only,
            'losing_both': losing_both,
            'agreement_rate': agreement_rate,
            'total_pnl': total_pnl,
            'unanimous_pnl': unanimous_pnl
        }

        # Calculer la similarité Jaccard pour cette paire
        set1 = uniq_sets[a1]
        set2 = uniq_sets[a2]
        jaccard_sim = calculate_jaccard_similarity(set1, set2)

        # Calculer les win rates
        total_wins_a1 = winning_both + winning_a1_only
        total_wins_a2 = winning_both + winning_a2_only
        winrate_a1_common = (total_wins_a1 / len(common) * 100) if len(common) > 0 else 0
        winrate_a2_common = (total_wins_a2 / len(common) * 100) if len(common) > 0 else 0

        # Win rates globaux
        global_wr_a1 = get_algo_winrate(a1, algo_dfs)
        global_wr_a2 = get_algo_winrate(a2, algo_dfs)

        # Déterminer le statut de diversification (utiliser vos variables locales)
        jaccard_threshold = 0.5  # Ou récupérer depuis vos paramètres
        if jaccard_sim < jaccard_threshold:
            jaccard_color = f"{Fore.GREEN}{jaccard_sim:.1%}{Style.RESET_ALL}"
            diversification_status = "DIVERSIFIÉS"
        else:
            jaccard_color = f"{Fore.RED}{jaccard_sim:.1%}{Style.RESET_ALL}"
            diversification_status = "REDONDANTS"

        # Mettre à jour pairs_stats pour inclure les nouvelles métriques
        pairs_stats[(a1, a2)].update({
            'jaccard_similarity': jaccard_sim,
            'winrate_a1_common': winrate_a1_common,
            'winrate_a2_common': winrate_a2_common,
            'global_wr_a1': global_wr_a1,
            'global_wr_a2': global_wr_a2
        })

        # Modifier l'affichage existant pour ajouter les nouvelles lignes
        print(f"\n>> Analyse de la paire {a1} / {a2} ({diversification_status}):")
        print(f"  Trades communs: {len(common)}")
        print(f"  Gagnants pour les deux: {winning_both}")
        print(f"  Gagnants uniquement pour {a1}: {winning_a1_only}")
        print(f"  Gagnants uniquement pour {a2}: {winning_a2_only}")
        print(f"  Perdants pour les deux: {losing_both}")
        print(f"  Taux d'accord: {agreement_rate:.2f}%")
        print(f"  Win Rate {a1} (trades communs): {winrate_a1_common:.1f}%")
        print(f"  Win Rate {a2} (trades communs): {winrate_a2_common:.1f}%")
        print(f"  Win Rate {a1} (global): {global_wr_a1:.1f}%")
        print(f"  Win Rate {a2} (global): {global_wr_a2:.1f}%")
        print(f"  PnL total: {total_pnl:.2f}")
        print(f"  PnL des trades unanimes: {unanimous_pnl:.2f}")
        print(f"  Taux de Jaccard: {jaccard_color}")

    return pairs_stats, occurrences_stats

# ────────────────────────────────────────────────────────────────────────────────
# CRÉATION DU TABLEAU DE RÉPARTITION PAR SESSIONS INTRADAY
# ────────────────────────────────────────────────────────────────────────────────

def create_sessions_analysis_table(datasets_info_with_results):
    """
    Crée un tableau d'analyse par sessions intraday pour tous les algorithmes et datasets

    Parameters:
    -----------
    datasets_info_with_results : list
        Liste de tuples (dataset_name, algo_dfs, results_dict)

    Returns:
    --------
    pd.DataFrame : Tableau avec répartition par sessions intraday
    """

    # Récupérer tous les algorithmes
    all_algos = set()
    for dataset_name, algo_dfs, results_dict in datasets_info_with_results:
        all_algos.update(algo_dfs.keys())

    all_algos = sorted(list(all_algos))

    # Initialiser le DataFrame
    sessions_analysis = pd.DataFrame({'Algorithme': all_algos})

    # Pour chaque dataset
    for dataset_name, algo_dfs, results_dict in datasets_info_with_results:
        print(f"🔄 Analyse des sessions pour {dataset_name}...")

        # Colonnes pour ce dataset
        col_total = f"{dataset_name}_Total"
        col_total_wr = f"{dataset_name}_Total_WR"
        col_pct_g1 = f"{dataset_name}_%G1"
        col_wr_g1 = f"{dataset_name}_WR_G1"
        col_pct_g2 = f"{dataset_name}_%G2"
        col_wr_g2 = f"{dataset_name}_WR_G2"

        # Initialiser les colonnes
        sessions_analysis[col_total] = 0
        sessions_analysis[col_total_wr] = 0.0
        sessions_analysis[col_pct_g1] = 0.0
        sessions_analysis[col_wr_g1] = 0.0
        sessions_analysis[col_pct_g2] = 0.0
        sessions_analysis[col_wr_g2] = 0.0

        # Analyser chaque algorithme
        for algo_name in all_algos:
            if algo_name not in algo_dfs:
                # Algorithme absent de ce dataset
                continue

            df_algo = algo_dfs[algo_name]

            # Vérifier la présence de la colonne deltaCustomSessionIndex
            if 'deltaCustomSessionIndex' not in df_algo.columns:
                print(f"⚠️ Colonne 'deltaCustomSessionIndex' manquante pour {algo_name} dans {dataset_name}")
                continue

            # Déterminer la colonne PnL
            pnl_col = None
            for col in ['PnlAfterFiltering', 'trade_pnl']:
                if col in df_algo.columns:
                    pnl_col = col
                    break

            if pnl_col is None:
                print(f"⚠️ Colonne PnL manquante pour {algo_name} dans {dataset_name}")
                continue

            # Filtrer les trades avec PnL non nul
            df_trades = df_algo[df_algo[pnl_col] != 0].copy()

            if len(df_trades) == 0:
                continue

            # Calculs globaux
            total_trades = len(df_trades)
            total_wins = (df_trades[pnl_col] > 0).sum()
            total_wr = (total_wins / total_trades * 100) if total_trades > 0 else 0.0

            # Analyser les groupes de sessions
            df_groupe1 = df_trades[df_trades['deltaCustomSessionIndex'].isin(GROUPE_SESSION_1)]
            df_groupe2 = df_trades[df_trades['deltaCustomSessionIndex'].isin(GROUPE_SESSION_2)]

            # Calculs pour groupe 1
            count_g1 = len(df_groupe1)
            pct_g1 = (count_g1 / total_trades * 100) if total_trades > 0 else 0.0
            wins_g1 = (df_groupe1[pnl_col] > 0).sum() if count_g1 > 0 else 0
            wr_g1 = (wins_g1 / count_g1 * 100) if count_g1 > 0 else 0.0

            # Calculs pour groupe 2
            count_g2 = len(df_groupe2)
            pct_g2 = (count_g2 / total_trades * 100) if total_trades > 0 else 0.0
            wins_g2 = (df_groupe2[pnl_col] > 0).sum() if count_g2 > 0 else 0
            wr_g2 = (wins_g2 / count_g2 * 100) if count_g2 > 0 else 0.0

            # Remplir le DataFrame
            mask = sessions_analysis['Algorithme'] == algo_name
            sessions_analysis.loc[mask, col_total] = total_trades
            sessions_analysis.loc[mask, col_total_wr] = round(total_wr, 1)
            sessions_analysis.loc[mask, col_pct_g1] = round(pct_g1, 1)
            sessions_analysis.loc[mask, col_wr_g1] = round(wr_g1, 1)
            sessions_analysis.loc[mask, col_pct_g2] = round(pct_g2, 1)
            sessions_analysis.loc[mask, col_wr_g2] = round(wr_g2, 1)

    return sessions_analysis


def format_sessions_table_for_display(sessions_df):
    """
    Formate le tableau des sessions pour un affichage optimal
    """
    # Créer une copie pour le formatage
    display_df = sessions_df.copy()

    # Formater les colonnes de pourcentage et win rates avec le symbole %
    for col in display_df.columns:
        if col != 'Algorithme':
            if '%' in col or 'WR' in col:
                # Ajouter le symbole % pour les pourcentages et win rates
                display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}%" if pd.notna(x) and x > 0 else "0.0%")
            elif 'Total' in col and 'WR' not in col:
                # Garder les totaux en nombres entiers
                display_df[col] = display_df[col].astype(int)

    return display_df




# ────────────────────────────────────────────────────────────────────────────────
# ANALYSE SUPPLÉMENTAIRE - INSIGHTS PAR SESSIONS
# ────────────────────────────────────────────────────────────────────────────────

def analyze_session_insights_5_datasets(sessions_df):
    """
    Version mise à jour pour analyser les insights des 5 datasets
    """
    print(f"\n{Fore.YELLOW}💡 INSIGHTS PAR SESSIONS INTRADAY (5 DATASETS){Style.RESET_ALL}")
    print("=" * 80)

    datasets = ["Train", "Test", "Val1", "Val", "Unseen"]  # MODIFIÉ

    for dataset in datasets:
        print(f"\n🔍 Dataset {dataset}:")

        # Colonnes pour ce dataset
        col_total = f"{dataset}_Total"
        col_total_wr = f"{dataset}_Total_WR"
        col_wr_g1 = f"{dataset}_WR_G1"
        col_wr_g2 = f"{dataset}_WR_G2"
        col_pct_g1 = f"{dataset}_%G1"
        col_pct_g2 = f"{dataset}_%G2"

        if col_total not in sessions_df.columns:
            continue

        # Filtrer les algorithmes actifs sur ce dataset
        active_algos = sessions_df[sessions_df[col_total] > 0]

        if len(active_algos) == 0:
            print(f"   Aucun algorithme actif")
            continue

        # Moyennes
        avg_total_wr = active_algos[col_total_wr].mean()
        avg_wr_g1 = active_algos[col_wr_g1].mean()
        avg_wr_g2 = active_algos[col_wr_g2].mean()
        avg_pct_g1 = active_algos[col_pct_g1].mean()
        avg_pct_g2 = active_algos[col_pct_g2].mean()

        # Algorithme avec meilleur WR sur groupe 1
        best_g1_idx = active_algos[col_wr_g1].idxmax()
        best_g1_algo = active_algos.loc[best_g1_idx, 'Algorithme']
        best_g1_wr = active_algos.loc[best_g1_idx, col_wr_g1]

        # Algorithme avec meilleur WR sur groupe 2
        best_g2_idx = active_algos[col_wr_g2].idxmax()
        best_g2_algo = active_algos.loc[best_g2_idx, 'Algorithme']
        best_g2_wr = active_algos.loc[best_g2_idx, col_wr_g2]

        print(f"   📊 WR moyen global: {avg_total_wr:.1f}%")
        print(f"   📊 WR moyen Groupe 1: {avg_wr_g1:.1f}% | Groupe 2: {avg_wr_g2:.1f}%")
        print(f"   📊 Répartition moyenne: G1 {avg_pct_g1:.1f}% | G2 {avg_pct_g2:.1f}%")
        print(f"   🏆 Meilleur sur G1: {best_g1_algo} ({best_g1_wr:.1f}%)")
        print(f"   🏆 Meilleur sur G2: {best_g2_algo} ({best_g2_wr:.1f}%)")

        # Identifier les algorithmes avec biais temporel
        bias_threshold = 10.0  # Différence de 10% entre groupes
        biased_algos = []

        for _, row in active_algos.iterrows():
            diff_wr = abs(row[col_wr_g1] - row[col_wr_g2])
            if diff_wr > bias_threshold:
                biased_algos.append({
                    'algo': row['Algorithme'],
                    'diff': diff_wr,
                    'better_group': 'G1' if row[col_wr_g1] > row[col_wr_g2] else 'G2'
                })

        if biased_algos:
            print(f"   ⚠️  Algorithmes avec biais de winrate (>±{bias_threshold}%):")
            for bias in sorted(biased_algos, key=lambda x: x['diff'], reverse=True):
                print(f"      {bias['algo']}: {bias['diff']:.1f}% (meilleur sur {bias['better_group']})")


def create_cluster_analysis_table(datasets_info_with_results):
    """
    Crée un tableau d'analyse par clusters pour le Groupe 2 uniquement

    Parameters:
    -----------
    datasets_info_with_results : list
        Liste de tuples (dataset_name, algo_dfs, results_dict)

    Returns:
    --------
    pd.DataFrame : Tableau avec répartition par clusters
    dict : Dictionnaire des labels de clusters
    """

    # Récupérer tous les algorithmes
    all_algos = set()
    for dataset_name, algo_dfs, results_dict in datasets_info_with_results:
        all_algos.update(algo_dfs.keys())

    all_algos = sorted(list(all_algos))

    # Initialiser le DataFrame
    cluster_analysis = pd.DataFrame({'Algorithme': all_algos})

    # Dictionnaire pour stocker les labels des clusters
    cluster_labels = {}

    # Pour chaque dataset
    for dataset_name, algo_dfs, results_dict in datasets_info_with_results:
        print(f"🔄 Analyse des clusters du Groupe 2 pour {dataset_name}...")

        # Colonnes pour ce dataset
        col_g2_total = f"{dataset_name}_G2_Total"
        col_pct_c2 = f"{dataset_name}_%C2"
        col_wr_c2 = f"{dataset_name}_WR_C2"
        col_pct_c1 = f"{dataset_name}_%C1"
        col_wr_c1 = f"{dataset_name}_WR_C1"
        col_pct_c0 = f"{dataset_name}_%C0"
        col_wr_c0 = f"{dataset_name}_WR_C0"

        # Initialiser les colonnes
        cluster_analysis[col_g2_total] = 0
        cluster_analysis[col_pct_c2] = "-"
        cluster_analysis[col_wr_c2] = "-"
        cluster_analysis[col_pct_c1] = "-"
        cluster_analysis[col_wr_c1] = "-"
        cluster_analysis[col_pct_c0] = "-"
        cluster_analysis[col_wr_c0] = "-"

        # Analyser chaque algorithme
        for algo_name in all_algos:
            if algo_name not in algo_dfs:
                continue

            df_algo = algo_dfs[algo_name]

            # Vérifier les colonnes nécessaires
            required_cols = ['deltaCustomSessionIndex', 'Cluster_G2', 'Regime_G2_Label']
            if not all(col in df_algo.columns for col in required_cols):
                print(f"⚠️ Colonnes manquantes pour {algo_name} dans {dataset_name}")
                continue

            # Déterminer la colonne PnL
            pnl_col = None
            for col in ['PnlAfterFiltering', 'trade_pnl']:
                if col in df_algo.columns:
                    pnl_col = col
                    break

            if pnl_col is None:
                print(f"⚠️ Colonne PnL manquante pour {algo_name} dans {dataset_name}")
                continue

            # Filtrer les trades du groupe 2 avec PnL non nul
            df_groupe2 = df_algo[
                (df_algo['deltaCustomSessionIndex'].isin(GROUPE_SESSION_2)) &
                (df_algo[pnl_col] != 0)
                ].copy()

            if len(df_groupe2) == 0:
                continue

            # Total des trades du groupe 2
            total_g2 = len(df_groupe2)

            # Extraire les labels des clusters (une seule fois)
            if not cluster_labels and len(df_groupe2) > 0:
                for cluster_num in [0, 1, 2]:
                    cluster_rows = df_groupe2[df_groupe2['Cluster_G2'] == cluster_num]
                    if len(cluster_rows) > 0:
                        label = cluster_rows['Regime_G2_Label'].iloc[0]
                        cluster_labels[cluster_num] = label

            # Analyser chaque cluster
            mask = cluster_analysis['Algorithme'] == algo_name
            cluster_analysis.loc[mask, col_g2_total] = total_g2

            for cluster_num in [2, 1, 0]:  # Ordre: C2, C1, C0
                df_cluster = df_groupe2[df_groupe2['Cluster_G2'] == cluster_num]

                if len(df_cluster) > 0:
                    # Calculs pour ce cluster
                    count_cluster = len(df_cluster)
                    pct_cluster = (count_cluster / total_g2 * 100) if total_g2 > 0 else 0.0
                    wins_cluster = (df_cluster[pnl_col] > 0).sum()
                    wr_cluster = (wins_cluster / count_cluster * 100) if count_cluster > 0 else 0.0

                    # Remplir les colonnes
                    if cluster_num == 2:
                        cluster_analysis.loc[mask, col_pct_c2] = round(pct_cluster, 1)
                        cluster_analysis.loc[mask, col_wr_c2] = round(wr_cluster, 1)
                    elif cluster_num == 1:
                        cluster_analysis.loc[mask, col_pct_c1] = round(pct_cluster, 1)
                        cluster_analysis.loc[mask, col_wr_c1] = round(wr_cluster, 1)
                    elif cluster_num == 0:
                        cluster_analysis.loc[mask, col_pct_c0] = round(pct_cluster, 1)
                        cluster_analysis.loc[mask, col_wr_c0] = round(wr_cluster, 1)

    return cluster_analysis, cluster_labels


def format_cluster_table_for_display(cluster_df):
    """
    Formate le tableau des clusters pour un affichage optimal avec coloration
    """
    # Créer une copie pour le formatage
    display_df = cluster_df.copy()

    # Formater les colonnes
    for col in display_df.columns:
        if col != 'Algorithme':
            if 'G2_Total' in col:
                # Les totaux restent en nombres entiers
                display_df[col] = display_df[col].astype(int)
            elif '%C' in col or 'WR_C' in col:
                # Formater les pourcentages et winrates
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:.1f}%" if isinstance(x, (int, float)) and x != "-" else x
                )

    return display_df


def print_cluster_table_with_colors(display_df):
    """
    Affiche le tableau avec coloration spécifique et alignement correct :
    - Colonnes G2_Total en bleu clair
    - Colonnes WR_C0 en orange clair
    - Colonnes WR_C1 en jaune clair
    - Colonnes WR_C2 en vert clair
    """
    from colorama import Fore, Style

    # Définir des couleurs plus claires et contrastées
    LIGHT_ORANGE = '\033[38;5;215m'  # Orange clair
    LIGHT_YELLOW = '\033[38;5;228m'  # Jaune clair
    LIGHT_GREEN = '\033[38;5;157m'  # Vert clair
    LIGHT_BLUE = '\033[38;5;117m'  # Bleu clair

    # Calculer la largeur maximale pour chaque colonne (sans codes couleur)
    col_widths = {}
    for col in display_df.columns:
        # Largeur du nom de colonne
        max_width = len(col)
        # Largeur des valeurs dans cette colonne
        for value in display_df[col]:
            max_width = max(max_width, len(str(value)))
        col_widths[col] = max_width + 2  # Ajouter un peu d'espace

    # Créer l'en-tête avec coloration et alignement correct
    headers = []
    for col in display_df.columns:
        width = col_widths[col]
        if 'G2_Total' in col:
            headers.append(f"{LIGHT_BLUE}{col:>{width}}{Style.RESET_ALL}")
        elif 'WR_C0' in col:
            headers.append(f"{LIGHT_ORANGE}{col:>{width}}{Style.RESET_ALL}")
        elif 'WR_C1' in col:
            headers.append(f"{LIGHT_YELLOW}{col:>{width}}{Style.RESET_ALL}")
        elif 'WR_C2' in col:
            headers.append(f"{LIGHT_GREEN}{col:>{width}}{Style.RESET_ALL}")
        else:
            headers.append(f"{col:>{width}}")

    # Afficher l'en-tête
    header_line = " | ".join(headers)
    print(header_line)

    # Calculer la longueur de la ligne de séparation
    separator_length = sum(col_widths.values()) + (len(display_df.columns) - 1) * 3  # 3 pour " | "
    print("-" * separator_length)

    # Afficher les données avec la même coloration et alignement
    for _, row in display_df.iterrows():
        row_data = []
        for col, value in row.items():
            width = col_widths[col]
            value_str = str(value)

            if 'G2_Total' in col:
                row_data.append(f"{LIGHT_BLUE}{value_str:>{width}}{Style.RESET_ALL}")
            elif 'WR_C0' in col:
                row_data.append(f"{LIGHT_ORANGE}{value_str:>{width}}{Style.RESET_ALL}")
            elif 'WR_C1' in col:
                row_data.append(f"{LIGHT_YELLOW}{value_str:>{width}}{Style.RESET_ALL}")
            elif 'WR_C2' in col:
                row_data.append(f"{LIGHT_GREEN}{value_str:>{width}}{Style.RESET_ALL}")
            else:
                row_data.append(f"{value_str:>{width}}")

        row_line = " | ".join(row_data)
        print(row_line)


# Version alternative avec couleurs encore plus douces pour la lisibilité
def print_cluster_table_with_soft_colors(display_df):
    """
    Version avec des couleurs très douces et claires pour une meilleure lisibilité
    """
    from colorama import Fore, Style

    # Couleurs très douces et claires
    SOFT_ORANGE = '\033[38;2;255;200;150m'  # Orange très clair
    SOFT_YELLOW = '\033[38;2;255;255;180m'  # Jaune très clair
    SOFT_GREEN = '\033[38;2;180;255;180m'  # Vert très clair
    SOFT_BLUE = '\033[38;2;180;220;255m'  # Bleu très clair

    # Calculer la largeur maximale pour chaque colonne
    col_widths = {}
    for col in display_df.columns:
        max_width = len(col)
        for value in display_df[col]:
            max_width = max(max_width, len(str(value)))
        col_widths[col] = max_width + 2

    # Créer l'en-tête avec coloration douce
    headers = []
    for col in display_df.columns:
        width = col_widths[col]
        if 'G2_Total' in col:
            headers.append(f"{SOFT_BLUE}{col:>{width}}{Style.RESET_ALL}")
        elif 'WR_C0' in col:
            headers.append(f"{SOFT_ORANGE}{col:>{width}}{Style.RESET_ALL}")
        elif 'WR_C1' in col:
            headers.append(f"{SOFT_YELLOW}{col:>{width}}{Style.RESET_ALL}")
        elif 'WR_C2' in col:
            headers.append(f"{SOFT_GREEN}{col:>{width}}{Style.RESET_ALL}")
        else:
            headers.append(f"{col:>{width}}")

    # Afficher l'en-tête
    header_line = " | ".join(headers)
    print(header_line)

    # Ligne de séparation
    separator_length = sum(col_widths.values()) + (len(display_df.columns) - 1) * 3
    print("=" * separator_length)

    # Afficher les données
    for _, row in display_df.iterrows():
        row_data = []
        for col, value in row.items():
            width = col_widths[col]
            value_str = str(value)

            if 'G2_Total' in col:
                row_data.append(f"{SOFT_BLUE}{value_str:>{width}}{Style.RESET_ALL}")
            elif 'WR_C0' in col:
                row_data.append(f"{SOFT_ORANGE}{value_str:>{width}}{Style.RESET_ALL}")
            elif 'WR_C1' in col:
                row_data.append(f"{SOFT_YELLOW}{value_str:>{width}}{Style.RESET_ALL}")
            elif 'WR_C2' in col:
                row_data.append(f"{SOFT_GREEN}{value_str:>{width}}{Style.RESET_ALL}")
            else:
                row_data.append(f"{value_str:>{width}}")

        row_line = " | ".join(row_data)
        print(row_line)


# Analyse des doublons pour les données d'entraînement et de test
# ────────────────────────────────────────────────────────────────────────────────
# Analyse des doublons pour les 4 datasets (MODIFIÉ)
# ────────────────────────────────────────────────────────────────────────────────
print(f"{Fore.BLUE}\n{'=' * 80}\nANALYSE DES DOUBLONS SUR LES DONNÉES D'ENTRAÎNEMENT\n{'=' * 80}{Style.RESET_ALL}")
algo_dfs_train = {name: df for name, df in to_save_train}
pairs_stats_train, occurrences_stats_train = analyse_doublons_algos(algo_dfs_train, min_common_trades=MIN_COMMON_TRADES)

print(f"{Fore.BLUE}\n{'=' * 80}\nANALYSE DES DOUBLONS SUR LES DONNÉES DE TEST\n{'=' * 80}{Style.RESET_ALL}")
algo_dfs_test = {name: df for name, df in to_save_test}
pairs_stats_test, occurrences_stats_test = analyse_doublons_algos(algo_dfs_test, min_common_trades=MIN_COMMON_TRADES)

print(f"{Fore.BLUE}\n{'=' * 80}\nANALYSE DES DOUBLONS SUR LES DONNÉES DE VALIDATION 1\n{'=' * 80}{Style.RESET_ALL}")
algo_dfs_val1 = {name: df for name, df in to_save_val1}
pairs_stats_val1, occurrences_stats_val1 = analyse_doublons_algos(algo_dfs_val1, min_common_trades=MIN_COMMON_TRADES)

print(f"{Fore.BLUE}\n{'=' * 80}\nANALYSE DES DOUBLONS SUR LES DONNÉES DE VALIDATION 2\n{'=' * 80}{Style.RESET_ALL}")
algo_dfs_val = {name: df for name, df in to_save_val}
pairs_stats_val, occurrences_stats_val = analyse_doublons_algos(algo_dfs_val, min_common_trades=MIN_COMMON_TRADES)

print(f"{Fore.BLUE}\n{'=' * 80}\nANALYSE DES DOUBLONS SUR LES DONNÉES UNSEEN\n{'=' * 80}{Style.RESET_ALL}")
algo_dfs_unseen = {name: df for name, df in to_save_unseen}
pairs_stats_unseen, occurrences_stats_unseen = analyse_doublons_algos(algo_dfs_unseen, min_common_trades=MIN_COMMON_TRADES)


# Afficher le tableau des métriques pour les 4 datasets
print(f"{Fore.GREEN}\n{'=' * 80}\nSYNTHESE GLOBALE (5 DATASETS)\n{'=' * 80}{Style.RESET_ALL}")

# Afficher le tableau des métriques pour les 5 datasets
print("\n📊 TABLEAU DES MÉTRIQUES (Train / Test / Val1 / Val / Unseen)")
print("=" * 150)  # Augmenter la largeur
print(comparison_metrics.to_string(index=False))

# Afficher le tableau des features
print("\n🔧 TABLEAU DES FEATURES UTILISÉES PAR ALGORITHME")
print("=" * 80)
print(comparison_features.to_string(index=False))

# Préparer les données pour l'analyse
datasets_info_with_results = [
    ("Train", algo_dfs_train, results_train),
    ("Test", algo_dfs_test, results_test),
    ("Val1", algo_dfs_val1, results_val1),
    ("Val", algo_dfs_val, results_val),
    ("Unseen", algo_dfs_unseen, results_unseen)
]

# Créer le tableau d'analyse par sessions
print(f"\n{Fore.CYAN}🔄 Création du tableau d'analyse par sessions intraday...{Style.RESET_ALL}")
sessions_analysis_table = create_sessions_analysis_table(datasets_info_with_results)

# Formater pour l'affichage
sessions_display_table = format_sessions_table_for_display(sessions_analysis_table)

# ────────────────────────────────────────────────────────────────────────────────
# NOUVEAU : ANALYSE PAR CLUSTERS DU GROUPE 2
# ────────────────────────────────────────────────────────────────────────────────

# Créer le tableau d'analyse par clusters
print(f"\n{Fore.CYAN}🔄 Création du tableau d'analyse par clusters du Groupe 2...{Style.RESET_ALL}")
cluster_analysis_table, cluster_labels = create_cluster_analysis_table(datasets_info_with_results)

# Formater pour l'affichage
cluster_display_table = format_cluster_table_for_display(cluster_analysis_table)

# Afficher le tableau avec coloration
print(f"\n{Fore.GREEN}📊 FOCUS GROUPE2 PAR CLUSTER{Style.RESET_ALL}")
print("=" * 150)
print(f"Groupe 2: Sessions {GROUPE_SESSION_2}")
print("=" * 150)

# Utiliser la fonction d'affichage avec coloration
print_cluster_table_with_colors(cluster_display_table)

# Afficher la légende des clusters
print(f"\n{Fore.YELLOW}Légende des clusters :{Style.RESET_ALL}")
for cluster_num in sorted(cluster_labels.keys()):
    print(f"  - Cluster {cluster_num} : {cluster_labels[cluster_num]}")


def create_cluster_0_1_combined_table(cluster_analysis_table):
    """
    Crée un tableau avec les statistiques combinées des clusters 0 + 1
    basé sur le tableau cluster_analysis_table existant

    Parameters:
    -----------
    cluster_analysis_table : pd.DataFrame
        Le tableau d'analyse par clusters existant

    Returns:
    --------
    pd.DataFrame : Tableau avec statistiques combinées clusters 0+1
    """

    # Créer une copie du tableau original pour récupérer les algorithmes
    combined_table = cluster_analysis_table[['Algorithme']].copy()

    # Lister les datasets à traiter
    datasets = ['Train', 'Test', 'Val1', 'Val', 'Unseen']

    for dataset in datasets:
        # Colonnes d'entrée pour ce dataset
        col_g2_total = f"{dataset}_G2_Total"
        col_pct_c0 = f"{dataset}_%C0"
        col_wr_c0 = f"{dataset}_WR_C0"
        col_pct_c1 = f"{dataset}_%C1"
        col_wr_c1 = f"{dataset}_WR_C1"

        # Colonnes de sortie pour ce dataset
        col_nb_c01 = f"{dataset}_C0+1_Nb"
        col_wr_c01 = f"{dataset}_C0+1_WR"

        # Initialiser les nouvelles colonnes
        combined_table[col_nb_c01] = 0
        combined_table[col_wr_c01] = 0.0

        # Vérifier que les colonnes nécessaires existent
        required_cols = [col_g2_total, col_pct_c0, col_wr_c0, col_pct_c1, col_wr_c1]
        missing_cols = [col for col in required_cols if col not in cluster_analysis_table.columns]

        if missing_cols:
            print(f"⚠️ Colonnes manquantes pour {dataset}: {missing_cols}")
            continue

        # Calculer pour chaque algorithme
        for idx, row in cluster_analysis_table.iterrows():
            g2_total = row[col_g2_total]

            # Vérifier si on a des données
            if g2_total == 0:
                combined_table.loc[idx, col_nb_c01] = 0
                combined_table.loc[idx, col_wr_c01] = 0.0
                continue

            # Récupérer les données des clusters 0 et 1
            pct_c0 = row[col_pct_c0] if isinstance(row[col_pct_c0], (int, float)) else 0
            wr_c0 = row[col_wr_c0] if isinstance(row[col_wr_c0], (int, float)) else 0
            pct_c1 = row[col_pct_c1] if isinstance(row[col_pct_c1], (int, float)) else 0
            wr_c1 = row[col_wr_c1] if isinstance(row[col_wr_c1], (int, float)) else 0

            # Calculer le nombre de trades pour chaque cluster
            nb_c0 = int(g2_total * pct_c0 / 100) if pct_c0 > 0 else 0
            nb_c1 = int(g2_total * pct_c1 / 100) if pct_c1 > 0 else 0

            # Nombre total de trades C0+C1
            nb_c01_total = nb_c0 + nb_c1

            if nb_c01_total == 0:
                combined_table.loc[idx, col_nb_c01] = 0
                combined_table.loc[idx, col_wr_c01] = 0.0
                continue

            # Calculer le nombre de trades gagnants pour chaque cluster
            wins_c0 = int(nb_c0 * wr_c0 / 100) if nb_c0 > 0 and wr_c0 > 0 else 0
            wins_c1 = int(nb_c1 * wr_c1 / 100) if nb_c1 > 0 and wr_c1 > 0 else 0

            # Win rate combiné C0+C1
            total_wins = wins_c0 + wins_c1
            wr_c01_combined = (total_wins / nb_c01_total * 100) if nb_c01_total > 0 else 0.0

            # Remplir le tableau
            combined_table.loc[idx, col_nb_c01] = nb_c01_total
            combined_table.loc[idx, col_wr_c01] = round(wr_c01_combined, 1)

    return combined_table


def print_cluster_0_1_table_with_colors(display_df):
    """
    Affiche le tableau des clusters 0+1 avec coloration :
    - Colonnes Nb en bleu clair
    - Colonnes WR en vert clair
    """
    from colorama import Fore, Style

    # Définir des couleurs claires
    LIGHT_BLUE = '\033[38;5;117m'  # Bleu clair pour Nb
    LIGHT_GREEN = '\033[38;5;157m'  # Vert clair pour WR

    print(f"\n{Fore.GREEN}📊 STATISTIQUES COMBINÉES CLUSTERS 0+1{Style.RESET_ALL}")
    print("=" * 120)
    print("Analyse des performances sur les clusters 0 (consolidation) + 1 (transition)")
    print("=" * 120)

    # Calculer la largeur maximale pour chaque colonne
    col_widths = {}
    for col in display_df.columns:
        max_width = len(col)
        for value in display_df[col]:
            max_width = max(max_width, len(str(value)))
        col_widths[col] = max_width + 2

    # Créer l'en-tête avec coloration
    headers = []
    for col in display_df.columns:
        width = col_widths[col]
        if 'C0+1_Nb' in col:
            headers.append(f"{LIGHT_BLUE}{col:>{width}}{Style.RESET_ALL}")
        elif 'C0+1_WR' in col:
            headers.append(f"{LIGHT_GREEN}{col:>{width}}{Style.RESET_ALL}")
        else:
            headers.append(f"{col:>{width}}")

    # Afficher l'en-tête
    header_line = " | ".join(headers)
    print(header_line)

    # Ligne de séparation
    separator_length = sum(col_widths.values()) + (len(display_df.columns) - 1) * 3
    print("-" * separator_length)

    # Afficher les données avec coloration
    for _, row in display_df.iterrows():
        row_data = []
        for col, value in row.items():
            width = col_widths[col]

            # Formater les valeurs
            if 'WR' in col and col != 'Algorithme':
                value_str = f"{value:.1f}%" if value > 0 else "0.0%"
            else:
                value_str = str(value)

            # Appliquer la couleur
            if 'C0+1_Nb' in col:
                row_data.append(f"{LIGHT_BLUE}{value_str:>{width}}{Style.RESET_ALL}")
            elif 'C0+1_WR' in col:
                row_data.append(f"{LIGHT_GREEN}{value_str:>{width}}{Style.RESET_ALL}")
            else:
                row_data.append(f"{value_str:>{width}}")

        row_line = " | ".join(row_data)
        print(row_line)


def analyze_cluster_0_1_insights(combined_table):
    """
    Génère des insights sur les performances combinées des clusters 0+1
    """
    print(f"\n{Fore.YELLOW}💡 INSIGHTS CLUSTERS 0+1 COMBINÉS{Style.RESET_ALL}")
    print("=" * 80)

    datasets = ['Train', 'Test', 'Val1', 'Val', 'Unseen']

    for dataset in datasets:
        col_nb = f"{dataset}_C0+1_Nb"
        col_wr = f"{dataset}_C0+1_WR"

        if col_nb not in combined_table.columns or col_wr not in combined_table.columns:
            continue

        # Filtrer les algorithmes actifs
        active_algos = combined_table[combined_table[col_nb] > 0]

        if len(active_algos) == 0:
            print(f"\n🔍 Dataset {dataset}: Aucun algorithme actif sur clusters 0+1")
            continue

        # Statistiques
        total_trades = active_algos[col_nb].sum()
        avg_trades = active_algos[col_nb].mean()
        avg_wr = active_algos[col_wr].mean()

        # Meilleur algorithme
        best_wr_idx = active_algos[col_wr].idxmax()
        best_algo = active_algos.loc[best_wr_idx, 'Algorithme']
        best_wr = active_algos.loc[best_wr_idx, col_wr]
        best_nb = active_algos.loc[best_wr_idx, col_nb]

        # Algorithme avec le plus de trades
        most_trades_idx = active_algos[col_nb].idxmax()
        most_trades_algo = active_algos.loc[most_trades_idx, 'Algorithme']
        most_trades_nb = active_algos.loc[most_trades_idx, col_nb]
        most_trades_wr = active_algos.loc[most_trades_idx, col_wr]

        print(f"\n🔍 Dataset {dataset} - Clusters 0+1:")
        print(f"   📊 {len(active_algos)} algorithmes actifs")
        print(f"   📊 Total trades: {total_trades}")
        print(f"   📊 Moyenne trades/algo: {avg_trades:.1f}")
        print(f"   📊 WR moyen: {avg_wr:.1f}%")
        print(f"   🏆 Meilleur WR: {best_algo} ({best_wr:.1f}% sur {best_nb} trades)")
        print(f"   📈 Plus de trades: {most_trades_algo} ({most_trades_nb} trades, {most_trades_wr:.1f}% WR)")

        # Algorithmes avec bon équilibre (WR > moyenne ET trades > moyenne/2)
        balanced_algos = active_algos[
            (active_algos[col_wr] > avg_wr) &
            (active_algos[col_nb] > avg_trades / 2)
            ]

        if len(balanced_algos) > 0:
            print(f"   ⚖️  Algorithmes équilibrés (WR > {avg_wr:.1f}% ET trades > {avg_trades / 2:.0f}):")
            for _, row in balanced_algos.head(3).iterrows():
                print(f"      • {row['Algorithme']}: {row[col_wr]:.1f}% WR, {row[col_nb]} trades")


def create_cluster_0_1_summary_comparison():
    """
    Compare les performances moyennes entre datasets pour clusters 0+1
    """
    print(f"\n{Fore.MAGENTA}📋 COMPARAISON INTER-DATASETS - CLUSTERS 0+1{Style.RESET_ALL}")
    print("=" * 80)

    # Cette fonction sera appelée après avoir créé le tableau combiné
    # Elle pourrait comparer les performances entre Train/Test/Val1/Val
    print("💡 Comparaison des performances entre périodes:")
    print("• Train vs Test: Évaluer la robustesse des algorithmes")
    print("• Val1 vs Val: Vérifier la consistance temporelle")
    print("• Clusters 0+1: Focus sur marchés de consolidation/transition")


# ────────────────────────────────────────────────────────────────────────────────
# INTÉGRATION DANS LE CODE PRINCIPAL
# ────────────────────────────────────────────────────────────────────────────────

# Ajouter cette section après la création du tableau cluster_analysis_table :

print(f"\n{Fore.CYAN}🔄 Création du tableau statistiques combinées Clusters 0+1...{Style.RESET_ALL}")

# Créer le tableau combiné clusters 0+1
cluster_0_1_combined = create_cluster_0_1_combined_table(cluster_analysis_table)

# Afficher le tableau avec coloration
print_cluster_0_1_table_with_colors(cluster_0_1_combined)

# Analyser les insights
analyze_cluster_0_1_insights(cluster_0_1_combined)

# Afficher la comparaison
create_cluster_0_1_summary_comparison()

# ----------------------------------------------------------------------
# ANALYSE CONSENSUS (une table par cluster)
# ----------------------------------------------------------------------
def analyze_consensus_for_clusters(target_clusters, cluster_name):

    dataset_groups = {
        # Individuels
        'Train': ['Train'],
        'Test':  ['Test'],
        'Val1':  ['Val1'],
        'Val':   ['Val'],
        'Unseen':['Unseen'],
        # Combos
        'Train + Test': ['Train', 'Test'],
        'Val1 + Val'  : ['Val1', 'Val'],
        'Val + Unseen': ['Val', 'Unseen'],
        'Train + Test + Val1 + Val'            : ['Train', 'Test', 'Val1', 'Val'],
        'Train + Test + Val1 + Val + Unseen'   : ['Train', 'Test', 'Val1', 'Val', 'Unseen'],
        'All Validation'                       : ['Val1', 'Val', 'Unseen']
    }

    results_table = []

    # -------------------------------------------------------------- #
    # BOUCLE SUR CHAQUE GROUPE (Train, Train+Test, …)
    # -------------------------------------------------------------- #
    for group_name, dataset_names in dataset_groups.items():
        print(f"🔄 Analyse consensus {cluster_name} pour {group_name}…")

        all_trades_data = {}

        # -------- 1. COLLECTE DES TRADES --------------------------- #
        for ds_name, algo_dfs, _ in datasets_info_with_results:
            if ds_name not in dataset_names:
                continue

            for algo_name, df_algo in algo_dfs.items():

                # Vérifs rapides
                if {'Cluster_G2', 'deltaCustomSessionIndex'}.difference(df_algo.columns):
                    continue
                pnl_col = next((c for c in ('PnlAfterFiltering', 'trade_pnl') if c in df_algo.columns), None)
                if pnl_col is None:
                    continue

                df_f = df_algo[
                    (df_algo['deltaCustomSessionIndex'].isin(GROUPE_SESSION_2)) &
                    (df_algo['Cluster_G2'].isin(target_clusters)) &
                    (df_algo[pnl_col] != 0)
                ]
                if df_f.empty:
                    continue

                indicator_cols = [
                    'rsi_', 'macd', 'macd_signal', 'macd_hist',
                    'timeElapsed2LastBar', 'timeStampOpening',
                    'ratio_deltaRevZone_volCandle'
                ]
                valid_cols = [c for c in indicator_cols if c in df_f.columns]
                if not valid_cols:
                    continue

                # --- boucle lignes
                for _, row in df_f.iterrows():
                    key = tuple(row[c] for c in valid_cols)
                    trade = all_trades_data.setdefault(key, {
                        'pnl'         : row[pnl_col],
                        'is_winning'  : row[pnl_col] > 0,
                        'algos_signals': set(),

                        # ---- GR1 ----
                        'v25_g1': row.get('volume_p25_g1', 0),
                        'v50_g1': row.get('volume_p50_g1', 0),
                        'v75_g1': row.get('volume_p75_g1', 0),
                        'd25_g1': row.get('duration_p25_g1', 0),
                        'd50_g1': row.get('duration_p50_g1', 0),
                        'd75_g1': row.get('duration_p75_g1', 0),
                        'e25_g1': row.get('extreme_ratio_p25_g1', 0),
                        'e50_g1': row.get('extreme_ratio_g1',     0),  # p50
                        'e75_g1': row.get('extreme_ratio_p75_g1', 0),
                        # FIX: Use correct column names for ATR
                        'a25_g1': row.get('atr_p25_g1', 0),
                        'a50_g1': row.get('atr_p50_g1', 0),
                        'a75_g1': row.get('atr_p75_g1', 0),

                        # ---- GR2 ----
                        'v25_g2': row.get('volume_p25_g2', 0),
                        'v50_g2': row.get('volume_p50_g2', 0),
                        'v75_g2': row.get('volume_p75_g2', 0),
                        'd25_g2': row.get('duration_p25_g2', 0),
                        'd50_g2': row.get('duration_p50_g2', 0),
                        'd75_g2': row.get('duration_p75_g2', 0),
                        'e25_g2': row.get('extreme_ratio_p25_g2', 0),
                        'e50_g2': row.get('extreme_ratio_g2',     0),  # p50
                        'e75_g2': row.get('extreme_ratio_p75_g2', 0),
                        # FIX: Use correct column names for ATR
                        'a25_g2': row.get('atr_p25_g2', 0),
                        'a50_g2': row.get('atr_p50_g2', 0),
                        'a75_g2': row.get('atr_p75_g2', 0),
                    })
                    trade['algos_signals'].add(algo_name)

        # -------- 2. SI AUCUN TRADE -------------------------------- #
        if not all_trades_data:
            results_table.append({'Dataset Group': group_name, **{col: 0 for col in range(1, 34)}})
            continue

        _avg = lambda lst: round(sum(lst) / len(lst), 2) if lst else 0.0

        # -------- 3. EXTRACTION DES LISTES ------------------------- #
        extract = lambda k: [t[k] for t in all_trades_data.values()]
        v25_g1 = extract('v25_g1'); v50_g1 = extract('v50_g1'); v75_g1 = extract('v75_g1')
        d25_g1 = extract('d25_g1'); d50_g1 = extract('d50_g1'); d75_g1 = extract('d75_g1')
        e50_g1 = extract('e50_g1');
        a25_g1 = extract('a25_g1'); a50_g1 = extract('a50_g1'); a75_g1 = extract('a75_g1')

        v25_g2 = extract('v25_g2'); v50_g2 = extract('v50_g2'); v75_g2 = extract('v75_g2')
        d25_g2 = extract('d25_g2'); d50_g2 = extract('d50_g2'); d75_g2 = extract('d75_g2')
        e50_g2 = extract('e50_g2');
        a25_g2 = extract('a25_g2'); a50_g2 = extract('a50_g2'); a75_g2 = extract('a75_g2')

        # -------- 4. CONSENSUS 2 / 3 algos ------------------------- #
        consensus_2 = [t for t in all_trades_data.values() if len(t['algos_signals']) >= 2]
        consensus_3 = [t for t in all_trades_data.values() if len(t['algos_signals']) >= 3]

        consensus_2_count = len(consensus_2)
        consensus_3_count = len(consensus_3)

        consensus_2_wr = round(100 * sum(t['is_winning'] for t in consensus_2) / consensus_2_count, 1) \
                         if consensus_2_count else 0.0
        consensus_3_wr = round(100 * sum(t['is_winning'] for t in consensus_3) / consensus_3_count, 1) \
                         if consensus_3_count else 0.0

        # -------- 5. AJOUT LIGNE RÉSULTATS ------------------------- #
        all_nb = len(all_trades_data)
        all_wr = round(100 * sum(t['is_winning'] for t in all_trades_data.values()) / all_nb, 1)

        results_table.append({
            'Dataset Group': group_name,

            # ── global ──────────────────────────────────────────
            'Tous algos - Nb': all_nb,
            'Tous algos - WR': all_wr,

            # ── GR1 (p25-50-75) ─────────────────────────────────
            'vol25_gr1_avg': _avg(v25_g1),
            'vol50_gr1_avg': _avg(v50_g1),
            'vol75_gr1_avg': _avg(v75_g1),
            'dur25_gr1_avg': _avg(d25_g1),
            'dur50_gr1_avg': _avg(d50_g1),
            'dur75_gr1_avg': _avg(d75_g1),
            'ext50_gr1_avg': _avg(e50_g1),  # extreme-ratio p50
            'atr25_gr1_avg': _avg(a25_g1),  # ← FIXED
            'atr50_gr1_avg': _avg(a50_g1),  # ← FIXED
            'atr75_gr1_avg': _avg(a75_g1),  # ← FIXED

            # ── GR2 (p25-50-75) ─────────────────────────────────
            'vol25_gr2_avg': _avg(v25_g2),
            'vol50_gr2_avg': _avg(v50_g2),
            'vol75_gr2_avg': _avg(v75_g2),
            'dur25_gr2_avg': _avg(d25_g2),
            'dur50_gr2_avg': _avg(d50_g2),
            'dur75_gr2_avg': _avg(d75_g2),
            'ext50_gr2_avg': _avg(e50_g2),  # extreme-ratio p50
            'atr25_gr2_avg': _avg(a25_g2),  # ← FIXED
            'atr50_gr2_avg': _avg(a50_g2),  # ← FIXED
            'atr75_gr2_avg': _avg(a75_g2),  # ← FIXED

            # ── consensus ───────────────────────────────────────
            '≥2 algos - Nb': consensus_2_count,
            '≥2 algos - WR': consensus_2_wr,
            '≥3 algos - Nb': consensus_3_count,
            '≥3 algos - WR': consensus_3_wr
        })

    # -------- FIN boucle dataset_groups -------------------------- #
    return pd.DataFrame(results_table)

import pandas as pd

# GROUPE_SESSION_2 doit être déjà défini dans votre script
# ex. GROUPE_SESSION_2 = [2, 3, 4, 5, 6]
def create_consensus_analysis_tables_5_datasets(datasets_info_with_results):
    """
    Construit trois tableaux de consensus (clusters 0+1, 0 et 2) sur 5 datasets
    (Train, Test, Val1, Val, Unseen) + combinaisons.

    Args:
        datasets_info_with_results: Liste de tuples (dataset_name, algo_dfs, _)
        GROUPE_SESSION_2: Liste des sessions à filtrer

    Returns:
        tuple: (table_cluster_0_1, table_cluster_0, table_cluster_2)
    """

    import pandas as pd

    # ------------------------------------------------------------------ #
    # SOUS-FONCTION : génère un tableau pour la liste `target_clusters`
    # ------------------------------------------------------------------ #
    def analyze_consensus_for_clusters(target_clusters, cluster_name):

        dataset_groups = {
            # Individuels
            'Train': ['Train'],
            'Test': ['Test'],
            'Val1': ['Val1'],
            'Val': ['Val'],
            'Unseen': ['Unseen'],
            # Combos
            'Train + Test': ['Train', 'Test'],
            'Val1 + Val': ['Val1', 'Val'],
            'Val + Unseen': ['Val', 'Unseen'],
            'Train + Test + Val1 + Val': ['Train', 'Test', 'Val1', 'Val'],
            'Train + Test + Val1 + Val + Unseen': ['Train', 'Test', 'Val1', 'Val', 'Unseen'],
            'All Validation': ['Val1', 'Val', 'Unseen']
        }

        results_table = []

        # -------------------------------------------------------------- #
        # BOUCLE SUR CHAQUE GROUPE (Train, Train+Test, …)
        # -------------------------------------------------------------- #
        for group_name, dataset_names in dataset_groups.items():
            print(f"🔄 Analyse consensus {cluster_name} pour {group_name}…")

            all_trades_data = {}

            # -------- 1. COLLECTE DES TRADES --------------------------- #
            for ds_name, algo_dfs, _ in datasets_info_with_results:
                if ds_name not in dataset_names:
                    continue

                for algo_name, df_algo in algo_dfs.items():

                    # Vérifs rapides
                    if {'Cluster_G2', 'deltaCustomSessionIndex'}.difference(df_algo.columns):
                        continue
                    pnl_col = next((c for c in ('PnlAfterFiltering', 'trade_pnl') if c in df_algo.columns), None)
                    if pnl_col is None:
                        continue

                    df_f = df_algo[
                        (df_algo['deltaCustomSessionIndex'].isin(GROUPE_SESSION_2)) &
                        (df_algo['Cluster_G2'].isin(target_clusters)) &
                        (df_algo[pnl_col] != 0)
                        ]
                    if df_f.empty:
                        continue

                    indicator_cols = [
                        'rsi_', 'macd', 'macd_signal', 'macd_hist',
                        'timeElapsed2LastBar', 'timeStampOpening',
                        'ratio_deltaRevZone_volCandle'
                    ]
                    valid_cols = [c for c in indicator_cols if c in df_f.columns]
                    if not valid_cols:
                        continue

                    # --- boucle lignes
                    for _, row in df_f.iterrows():
                        key = tuple(row[c] for c in valid_cols)
                        trade = all_trades_data.setdefault(key, {
                            'pnl': row[pnl_col],
                            'is_winning': row[pnl_col] > 0,
                            'algos_signals': set(),

                            # ---- VOLUME_PERTICK ----
                            'volume_perTick': row.get('volume_perTick', 0),

                            # ---- CANDLE_SIZE_TICKS ----
                            'candleSizeTicks': row.get('candleSizeTicks', 0),

                            # ---- GR1 ----
                            'v25_g1': row.get('volume_p25_g1', 0),
                            'v50_g1': row.get('volume_p50_g1', 0),
                            'v75_g1': row.get('volume_p75_g1', 0),
                            'd25_g1': row.get('duration_p25_g1', 0),
                            'd50_g1': row.get('duration_p50_g1', 0),
                            'd75_g1': row.get('duration_p75_g1', 0),
                            'e25_g1': row.get('extreme_ratio_p25_g1', 0),
                            'e50_g1': row.get('extreme_ratio_g1', 0),  # p50
                            'e75_g1': row.get('extreme_ratio_p75_g1', 0),
                            # ---- ATR GR1 ----
                            'a25_g1': row.get('atr_p25_g1', 0),
                            'a50_g1': row.get('atr_p50_g1', 0),
                            'a75_g1': row.get('atr_p75_g1', 0),
                            # ---- VOL_ABOVE GR1 ----
                            'vol_above_p25_g1': row.get('vol_above_p25_g1', 0),
                            'vol_above_p50_g1': row.get('vol_above_p50_g1', 0),
                            'vol_above_p75_g1': row.get('vol_above_p75_g1', 0),
                            # ---- VOL_MEAN_PER_TICK GR1 ----
                            'volMeanPerTick_p25_g1': row.get('volMeanPerTick_p25_g1', 0),
                            'volMeanPerTick_p50_g1': row.get('volMeanPerTick_p50_g1', 0),
                            'volMeanPerTick_p75_g1': row.get('volMeanPerTick_p75_g1', 0),
                            # ---- WIN/LOSE METRICS GR1 ----
                            'meanVol_perTick_over1_g1': row.get('meanVol_perTick_over1_g1', 0),
                            'meanVol_perTick_over2_g1': row.get('meanVol_perTick_over2_g1', 0),
                            'meanVol_perTick_over5_g1': row.get('meanVol_perTick_over5_g1', 0),
                            'meanVol_perTick_over12_g1': row.get('meanVol_perTick_over12_g1', 0),
                            'meanVol_perTick_over20_g1': row.get('meanVol_perTick_over20_g1', 0),
                            'meanVol_perTick_over30_g1': row.get('meanVol_perTick_over30_g1', 0),
                            'VolCandleMeanOver5Ratio_g1': row.get('VolCandleMeanOver5Ratio_g1', 0),
                            'VolCandleMeanOver12Ratio_g1': row.get('VolCandleMeanOver12Ratio_g1', 0),
                            'VolCandleMeanOver20Ratio_g1': row.get('VolCandleMeanOver20Ratio_g1', 0),
                            'VolCandleMeanOver30Ratio_g1': row.get('VolCandleMeanOver30Ratio_g1', 0),

                            # ---- GR2 ----
                            'v25_g2': row.get('volume_p25_g2', 0),
                            'v50_g2': row.get('volume_p50_g2', 0),
                            'v75_g2': row.get('volume_p75_g2', 0),
                            'd25_g2': row.get('duration_p25_g2', 0),
                            'd50_g2': row.get('duration_p50_g2', 0),
                            'd75_g2': row.get('duration_p75_g2', 0),
                            'e25_g2': row.get('extreme_ratio_p25_g2', 0),
                            'e50_g2': row.get('extreme_ratio_g2', 0),  # p50
                            'e75_g2': row.get('extreme_ratio_p75_g2', 0),
                            # ---- ATR GR2 ----
                            'a25_g2': row.get('atr_p25_g2', 0),
                            'a50_g2': row.get('atr_p50_g2', 0),
                            'a75_g2': row.get('atr_p75_g2', 0),
                            # ---- VOL_ABOVE GR2 ----
                            'vol_above_p25_g2': row.get('vol_above_p25_g2', 0),
                            'vol_above_p50_g2': row.get('vol_above_p50_g2', 0),
                            'vol_above_p75_g2': row.get('vol_above_p75_g2', 0),
                            # ---- VOL_MEAN_PER_TICK GR2 ----
                            'volMeanPerTick_p25_g2': row.get('volMeanPerTick_p25_g2', 0),
                            'volMeanPerTick_p50_g2': row.get('volMeanPerTick_p50_g2', 0),
                            'volMeanPerTick_p75_g2': row.get('volMeanPerTick_p75_g2', 0),
                            # ---- WIN/LOSE METRICS GR2 ----
                            'meanVol_perTick_over1_g2': row.get('meanVol_perTick_over1_g2', 0),
                            'meanVol_perTick_over2_g2': row.get('meanVol_perTick_over2_g2', 0),
                            'meanVol_perTick_over5_g2': row.get('meanVol_perTick_over5_g2', 0),
                            'meanVol_perTick_over12_g2': row.get('meanVol_perTick_over12_g2', 0),
                            'meanVol_perTick_over20_g2': row.get('meanVol_perTick_over20_g2', 0),
                            'meanVol_perTick_over30_g2': row.get('meanVol_perTick_over30_g2', 0),
                            'VolCandleMeanOver5Ratio_g2': row.get('VolCandleMeanOver5Ratio_g2', 0),
                            'VolCandleMeanOver12Ratio_g2': row.get('VolCandleMeanOver12Ratio_g2', 0),
                            'VolCandleMeanOver20Ratio_g2': row.get('VolCandleMeanOver20Ratio_g2', 0),
                            'VolCandleMeanOver30Ratio_g2': row.get('VolCandleMeanOver30Ratio_g2', 0),

                            # ---- WIN/LOSE METRICS INDÉPENDANTS DES GROUPES ----
                            'meanVol_perTick_over1': row.get('meanVol_perTick_over1', 0),
                            'meanVol_perTick_over2': row.get('meanVol_perTick_over2', 0),
                            'meanVol_perTick_over5': row.get('meanVol_perTick_over5', 0),
                            'meanVol_perTick_over12': row.get('meanVol_perTick_over12', 0),
                            'meanVol_perTick_over20': row.get('meanVol_perTick_over20', 0),
                            'meanVol_perTick_over30': row.get('meanVol_perTick_over30', 0),
                            'VolCandleMeanOver5Ratio': row.get('VolCandleMeanOver5Ratio', 0),
                            'VolCandleMeanOver12Ratio': row.get('VolCandleMeanOver12Ratio', 0),
                            'VolCandleMeanOver20Ratio': row.get('VolCandleMeanOver20Ratio', 0),
                            'VolCandleMeanOver30Ratio': row.get('VolCandleMeanOver30Ratio', 0)
                        })
                        trade['algos_signals'].add(algo_name)

            # -------- 2. SI AUCUN TRADE -------------------------------- #
            if not all_trades_data:
                # Créer un dictionnaire avec toutes les colonnes attendues
                empty_result = {'Dataset Group': group_name}
                # Ajouter toutes les colonnes avec valeur 0
                empty_columns = [
                    'Tous algos - Nb', 'Tous algos - WR',
                    'volume_perTick_win', 'volume_perTick_lose',
                    'candleSizeTicks_win', 'candleSizeTicks_lose',
                    'meanVol_perTick_over1_win', 'meanVol_perTick_over1_lose',
                    'meanVol_perTick_over2_win', 'meanVol_perTick_over2_lose',
                    'meanVol_perTick_over5_win', 'meanVol_perTick_over5_lose',
                    'meanVol_perTick_over12_win', 'meanVol_perTick_over12_lose',
                    'meanVol_perTick_over20_win', 'meanVol_perTick_over20_lose',
                    'meanVol_perTick_over30_win', 'meanVol_perTick_over30_lose',
                    'VolCandleMeanOver5Ratio_win', 'VolCandleMeanOver5Ratio_lose',
                    'VolCandleMeanOver12Ratio_win', 'VolCandleMeanOver12Ratio_lose',
                    'VolCandleMeanOver20Ratio_win', 'VolCandleMeanOver20Ratio_lose',
                    'VolCandleMeanOver30Ratio_win', 'VolCandleMeanOver30Ratio_lose',
                    # GR1 columns
                    'vol25_gr1_avg', 'vol50_gr1_avg', 'vol75_gr1_avg',
                    'dur25_gr1_avg', 'dur50_gr1_avg', 'dur75_gr1_avg',
                    'ext50_gr1_avg', 'atr25_gr1_avg', 'atr50_gr1_avg', 'atr75_gr1_avg',
                    'vol_above_p25_gr1_avg', 'vol_above_p50_gr1_avg', 'vol_above_p75_gr1_avg',
                    'volMeanPerTick_p25_gr1_avg', 'volMeanPerTick_p50_gr1_avg', 'volMeanPerTick_p75_gr1_avg',
                    # GR2 columns
                    'vol25_gr2_avg', 'vol50_gr2_avg', 'vol75_gr2_avg',
                    'dur25_gr2_avg', 'dur50_gr2_avg', 'dur75_gr2_avg',
                    'ext50_gr2_avg', 'atr25_gr2_avg', 'atr50_gr2_avg', 'atr75_gr2_avg',
                    'vol_above_p25_gr2_avg', 'vol_above_p50_gr2_avg', 'vol_above_p75_gr2_avg',
                    'volMeanPerTick_p25_gr2_avg', 'volMeanPerTick_p50_gr2_avg', 'volMeanPerTick_p75_gr2_avg',
                    # WIN/LOSE GR1
                    'meanVol_perTick_over1_gr1_win', 'meanVol_perTick_over1_gr1_lose',
                    'meanVol_perTick_over2_gr1_win', 'meanVol_perTick_over2_gr1_lose',
                    'meanVol_perTick_over5_gr1_win', 'meanVol_perTick_over5_gr1_lose',
                    'meanVol_perTick_over12_gr1_win', 'meanVol_perTick_over12_gr1_lose',
                    'meanVol_perTick_over20_gr1_win', 'meanVol_perTick_over20_gr1_lose',
                    'meanVol_perTick_over30_gr1_win', 'meanVol_perTick_over30_gr1_lose',
                    'VolCandleMeanOver5Ratio_gr1_win', 'VolCandleMeanOver5Ratio_gr1_lose',
                    'VolCandleMeanOver12Ratio_gr1_win', 'VolCandleMeanOver12Ratio_gr1_lose',
                    'VolCandleMeanOver20Ratio_gr1_win', 'VolCandleMeanOver20Ratio_gr1_lose',
                    'VolCandleMeanOver30Ratio_gr1_win', 'VolCandleMeanOver30Ratio_gr1_lose',
                    # WIN/LOSE GR2
                    'meanVol_perTick_over1_gr2_win', 'meanVol_perTick_over1_gr2_lose',
                    'meanVol_perTick_over2_gr2_win', 'meanVol_perTick_over2_gr2_lose',
                    'meanVol_perTick_over5_gr2_win', 'meanVol_perTick_over5_gr2_lose',
                    'meanVol_perTick_over12_gr2_win', 'meanVol_perTick_over12_gr2_lose',
                    'meanVol_perTick_over20_gr2_win', 'meanVol_perTick_over20_gr2_lose',
                    'meanVol_perTick_over30_gr2_win', 'meanVol_perTick_over30_gr2_lose',
                    'VolCandleMeanOver5Ratio_gr2_win', 'VolCandleMeanOver5Ratio_gr2_lose',
                    'VolCandleMeanOver12Ratio_gr2_win', 'VolCandleMeanOver12Ratio_gr2_lose',
                    'VolCandleMeanOver20Ratio_gr2_win', 'VolCandleMeanOver20Ratio_gr2_lose',
                    'VolCandleMeanOver30Ratio_gr2_win', 'VolCandleMeanOver30Ratio_gr2_lose',
                    # Consensus
                    '≥2 algos - Nb', '≥2 algos - WR', '≥3 algos - Nb', '≥3 algos - WR'
                ]
                for col in empty_columns:
                    empty_result[col] = 0
                results_table.append(empty_result)
                continue

            _avg = lambda lst: round(sum(lst) / len(lst), 2) if lst else 0.0

            # -------- CALCULS WIN/LOSE SÉPARÉS ------------------------- #
            winning_trades = [t for t in all_trades_data.values() if t['is_winning']]
            losing_trades = [t for t in all_trades_data.values() if not t['is_winning']]

            def _avg_win_lose(metric_key):
                """Calcule les moyennes pour les trades gagnants et perdants"""
                win_values = [t[metric_key] for t in winning_trades if metric_key in t] if winning_trades else []
                lose_values = [t[metric_key] for t in losing_trades if metric_key in t] if losing_trades else []
                return _avg(win_values), _avg(lose_values)

            # -------- 3. EXTRACTION DES LISTES ------------------------- #
            extract = lambda k: [t[k] for t in all_trades_data.values()]
            v25_g1 = extract('v25_g1')
            v50_g1 = extract('v50_g1')
            v75_g1 = extract('v75_g1')
            d25_g1 = extract('d25_g1')
            d50_g1 = extract('d50_g1')
            d75_g1 = extract('d75_g1')
            e50_g1 = extract('e50_g1')
            # Extract ATR data for GR1
            a25_g1 = extract('a25_g1')
            a50_g1 = extract('a50_g1')
            a75_g1 = extract('a75_g1')
            # Extract VOL_ABOVE data for GR1
            vol_above_p25_g1 = extract('vol_above_p25_g1')
            vol_above_p50_g1 = extract('vol_above_p50_g1')
            vol_above_p75_g1 = extract('vol_above_p75_g1')
            # Extract VOL_MEAN_PER_TICK data for GR1
            volMeanPerTick_p25_g1 = extract('volMeanPerTick_p25_g1')
            volMeanPerTick_p50_g1 = extract('volMeanPerTick_p50_g1')
            volMeanPerTick_p75_g1 = extract('volMeanPerTick_p75_g1')

            v25_g2 = extract('v25_g2')
            v50_g2 = extract('v50_g2')
            v75_g2 = extract('v75_g2')
            d25_g2 = extract('d25_g2')
            d50_g2 = extract('d50_g2')
            d75_g2 = extract('d75_g2')
            e50_g2 = extract('e50_g2')
            # Extract ATR data for GR2
            a25_g2 = extract('a25_g2')
            a50_g2 = extract('a50_g2')
            a75_g2 = extract('a75_g2')
            # Extract VOL_ABOVE data for GR2
            vol_above_p25_g2 = extract('vol_above_p25_g2')
            vol_above_p50_g2 = extract('vol_above_p50_g2')
            vol_above_p75_g2 = extract('vol_above_p75_g2')
            # Extract VOL_MEAN_PER_TICK data for GR2
            volMeanPerTick_p25_g2 = extract('volMeanPerTick_p25_g2')
            volMeanPerTick_p50_g2 = extract('volMeanPerTick_p50_g2')
            volMeanPerTick_p75_g2 = extract('volMeanPerTick_p75_g2')

            # -------- 4. CONSENSUS 2 / 3 algos ------------------------- #
            consensus_2 = [t for t in all_trades_data.values() if len(t['algos_signals']) >= 2]
            consensus_3 = [t for t in all_trades_data.values() if len(t['algos_signals']) >= 3]

            consensus_2_count = len(consensus_2)
            consensus_3_count = len(consensus_3)

            consensus_2_wr = round(100 * sum(t['is_winning'] for t in consensus_2) / consensus_2_count, 1) \
                if consensus_2_count else 0.0
            consensus_3_wr = round(100 * sum(t['is_winning'] for t in consensus_3) / consensus_3_count, 1) \
                if consensus_3_count else 0.0

            # -------- 5. AJOUT LIGNE RÉSULTATS ------------------------- #
            all_nb = len(all_trades_data)
            all_wr = round(100 * sum(t['is_winning'] for t in all_trades_data.values()) / all_nb, 1)

            # Calculs WIN/LOSE pour volume_perTick
            volume_perTick_win, volume_perTick_lose = _avg_win_lose('volume_perTick')

            # Calculs WIN/LOSE pour candleSizeTicks
            candleSizeTicks_win, candleSizeTicks_lose = _avg_win_lose('candleSizeTicks')

            # Calculs WIN/LOSE indépendants des groupes (moyennes directes)
            meanVol_perTick_over1_win, meanVol_perTick_over1_lose = _avg_win_lose('meanVol_perTick_over1')
            meanVol_perTick_over2_win, meanVol_perTick_over2_lose = _avg_win_lose('meanVol_perTick_over2')
            meanVol_perTick_over5_win, meanVol_perTick_over5_lose = _avg_win_lose('meanVol_perTick_over5')
            meanVol_perTick_over12_win, meanVol_perTick_over12_lose = _avg_win_lose('meanVol_perTick_over12')
            meanVol_perTick_over20_win, meanVol_perTick_over20_lose = _avg_win_lose('meanVol_perTick_over20')
            meanVol_perTick_over30_win, meanVol_perTick_over30_lose = _avg_win_lose('meanVol_perTick_over30')

            VolCandleMeanOver5Ratio_win, VolCandleMeanOver5Ratio_lose = _avg_win_lose('VolCandleMeanOver5Ratio')
            VolCandleMeanOver12Ratio_win, VolCandleMeanOver12Ratio_lose = _avg_win_lose('VolCandleMeanOver12Ratio')
            VolCandleMeanOver20Ratio_win, VolCandleMeanOver20Ratio_lose = _avg_win_lose('VolCandleMeanOver20Ratio')
            VolCandleMeanOver30Ratio_win, VolCandleMeanOver30Ratio_lose = _avg_win_lose('VolCandleMeanOver30Ratio')

            # Calculs WIN/LOSE pour les nouvelles métriques (ORDRE CORRECT: 1, 2, 5, 12, 20, 30)
            # GR1 WIN/LOSE - meanVol_perTick_over
            meanVol_perTick_over1_g1_win, meanVol_perTick_over1_g1_lose = _avg_win_lose('meanVol_perTick_over1_g1')
            meanVol_perTick_over2_g1_win, meanVol_perTick_over2_g1_lose = _avg_win_lose('meanVol_perTick_over2_g1')
            meanVol_perTick_over5_g1_win, meanVol_perTick_over5_g1_lose = _avg_win_lose('meanVol_perTick_over5_g1')
            meanVol_perTick_over12_g1_win, meanVol_perTick_over12_g1_lose = _avg_win_lose('meanVol_perTick_over12_g1')
            meanVol_perTick_over20_g1_win, meanVol_perTick_over20_g1_lose = _avg_win_lose('meanVol_perTick_over20_g1')
            meanVol_perTick_over30_g1_win, meanVol_perTick_over30_g1_lose = _avg_win_lose('meanVol_perTick_over30_g1')

            # GR1 WIN/LOSE - VolCandleMeanOverRatio
            VolCandleMeanOver5Ratio_g1_win, VolCandleMeanOver5Ratio_g1_lose = _avg_win_lose(
                'VolCandleMeanOver5Ratio_g1')
            VolCandleMeanOver12Ratio_g1_win, VolCandleMeanOver12Ratio_g1_lose = _avg_win_lose(
                'VolCandleMeanOver12Ratio_g1')
            VolCandleMeanOver20Ratio_g1_win, VolCandleMeanOver20Ratio_g1_lose = _avg_win_lose(
                'VolCandleMeanOver20Ratio_g1')
            VolCandleMeanOver30Ratio_g1_win, VolCandleMeanOver30Ratio_g1_lose = _avg_win_lose(
                'VolCandleMeanOver30Ratio_g1')

            # GR2 WIN/LOSE - meanVol_perTick_over
            meanVol_perTick_over1_g2_win, meanVol_perTick_over1_g2_lose = _avg_win_lose('meanVol_perTick_over1_g2')
            meanVol_perTick_over2_g2_win, meanVol_perTick_over2_g2_lose = _avg_win_lose('meanVol_perTick_over2_g2')
            meanVol_perTick_over5_g2_win, meanVol_perTick_over5_g2_lose = _avg_win_lose('meanVol_perTick_over5_g2')
            meanVol_perTick_over12_g2_win, meanVol_perTick_over12_g2_lose = _avg_win_lose('meanVol_perTick_over12_g2')
            meanVol_perTick_over20_g2_win, meanVol_perTick_over20_g2_lose = _avg_win_lose('meanVol_perTick_over20_g2')
            meanVol_perTick_over30_g2_win, meanVol_perTick_over30_g2_lose = _avg_win_lose('meanVol_perTick_over30_g2')

            # GR2 WIN/LOSE - VolCandleMeanOverRatio
            VolCandleMeanOver5Ratio_g2_win, VolCandleMeanOver5Ratio_g2_lose = _avg_win_lose(
                'VolCandleMeanOver5Ratio_g2')
            VolCandleMeanOver12Ratio_g2_win, VolCandleMeanOver12Ratio_g2_lose = _avg_win_lose(
                'VolCandleMeanOver12Ratio_g2')
            VolCandleMeanOver20Ratio_g2_win, VolCandleMeanOver20Ratio_g2_lose = _avg_win_lose(
                'VolCandleMeanOver20Ratio_g2')
            VolCandleMeanOver30Ratio_g2_win, VolCandleMeanOver30Ratio_g2_lose = _avg_win_lose(
                'VolCandleMeanOver30Ratio_g2')

            results_table.append({
                'Dataset Group': group_name,

                # ── global ──────────────────────────────────────────
                'Tous algos - Nb': all_nb,
                'Tous algos - WR': all_wr,

                # ── VOLUME_PERTICK WIN/LOSE ─────────────────────────
                'volume_perTick_win': volume_perTick_win,
                'volume_perTick_lose': volume_perTick_lose,

                # ── CANDLE_SIZE_TICKS WIN/LOSE ──────────────────────
                'candleSizeTicks_win': candleSizeTicks_win,
                'candleSizeTicks_lose': candleSizeTicks_lose,

                # ── WIN/LOSE INDÉPENDANTS DES GROUPES ──────────────
                'meanVol_perTick_over1_win': meanVol_perTick_over1_win,
                'meanVol_perTick_over1_lose': meanVol_perTick_over1_lose,
                'meanVol_perTick_over2_win': meanVol_perTick_over2_win,
                'meanVol_perTick_over2_lose': meanVol_perTick_over2_lose,
                'meanVol_perTick_over5_win': meanVol_perTick_over5_win,
                'meanVol_perTick_over5_lose': meanVol_perTick_over5_lose,
                'meanVol_perTick_over12_win': meanVol_perTick_over12_win,
                'meanVol_perTick_over12_lose': meanVol_perTick_over12_lose,
                'meanVol_perTick_over20_win': meanVol_perTick_over20_win,
                'meanVol_perTick_over20_lose': meanVol_perTick_over20_lose,
                'meanVol_perTick_over30_win': meanVol_perTick_over30_win,
                'meanVol_perTick_over30_lose': meanVol_perTick_over30_lose,
                'VolCandleMeanOver5Ratio_win': VolCandleMeanOver5Ratio_win,
                'VolCandleMeanOver5Ratio_lose': VolCandleMeanOver5Ratio_lose,
                'VolCandleMeanOver12Ratio_win': VolCandleMeanOver12Ratio_win,
                'VolCandleMeanOver12Ratio_lose': VolCandleMeanOver12Ratio_lose,
                'VolCandleMeanOver20Ratio_win': VolCandleMeanOver20Ratio_win,
                'VolCandleMeanOver20Ratio_lose': VolCandleMeanOver20Ratio_lose,
                'VolCandleMeanOver30Ratio_win': VolCandleMeanOver30Ratio_win,
                'VolCandleMeanOver30Ratio_lose': VolCandleMeanOver30Ratio_lose,

                # ── GR1 (p25-50-75) ─────────────────────────────────
                'vol25_gr1_avg': _avg(v25_g1),
                'vol50_gr1_avg': _avg(v50_g1),
                'vol75_gr1_avg': _avg(v75_g1),
                'dur25_gr1_avg': _avg(d25_g1),
                'dur50_gr1_avg': _avg(d50_g1),
                'dur75_gr1_avg': _avg(d75_g1),
                'ext50_gr1_avg': _avg(e50_g1),  # extreme-ratio p50
                'atr25_gr1_avg': _avg(a25_g1),
                'atr50_gr1_avg': _avg(a50_g1),
                'atr75_gr1_avg': _avg(a75_g1),
                'vol_above_p25_gr1_avg': _avg(vol_above_p25_g1),
                'vol_above_p50_gr1_avg': _avg(vol_above_p50_g1),
                'vol_above_p75_gr1_avg': _avg(vol_above_p75_g1),
                'volMeanPerTick_p25_gr1_avg': _avg(volMeanPerTick_p25_g1),
                'volMeanPerTick_p50_gr1_avg': _avg(volMeanPerTick_p50_g1),
                'volMeanPerTick_p75_gr1_avg': _avg(volMeanPerTick_p75_g1),

                # ── GR2 (p25-50-75) ─────────────────────────────────
                'vol25_gr2_avg': _avg(v25_g2),
                'vol50_gr2_avg': _avg(v50_g2),
                'vol75_gr2_avg': _avg(v75_g2),
                'dur25_gr2_avg': _avg(d25_g2),
                'dur50_gr2_avg': _avg(d50_g2),
                'dur75_gr2_avg': _avg(d75_g2),
                'ext50_gr2_avg': _avg(e50_g2),  # extreme-ratio p50
                'atr25_gr2_avg': _avg(a25_g2),
                'atr50_gr2_avg': _avg(a50_g2),
                'atr75_gr2_avg': _avg(a75_g2),
                'vol_above_p25_gr2_avg': _avg(vol_above_p25_g2),
                'vol_above_p50_gr2_avg': _avg(vol_above_p50_g2),
                'vol_above_p75_gr2_avg': _avg(vol_above_p75_g2),
                'volMeanPerTick_p25_gr2_avg': _avg(volMeanPerTick_p25_g2),
                'volMeanPerTick_p50_gr2_avg': _avg(volMeanPerTick_p50_g2),
                'volMeanPerTick_p75_gr2_avg': _avg(volMeanPerTick_p75_g2),

                # ── WIN/LOSE METRICS GR1 ───────────────────────────────
                'meanVol_perTick_over1_gr1_win': meanVol_perTick_over1_g1_win,
                'meanVol_perTick_over1_gr1_lose': meanVol_perTick_over1_g1_lose,
                'meanVol_perTick_over2_gr1_win': meanVol_perTick_over2_g1_win,
                'meanVol_perTick_over2_gr1_lose': meanVol_perTick_over2_g1_lose,
                'meanVol_perTick_over5_gr1_win': meanVol_perTick_over5_g1_win,
                'meanVol_perTick_over5_gr1_lose': meanVol_perTick_over5_g1_lose,
                'meanVol_perTick_over12_gr1_win': meanVol_perTick_over12_g1_win,
                'meanVol_perTick_over12_gr1_lose': meanVol_perTick_over12_g1_lose,
                'meanVol_perTick_over20_gr1_win': meanVol_perTick_over20_g1_win,
                'meanVol_perTick_over20_gr1_lose': meanVol_perTick_over20_g1_lose,
                'meanVol_perTick_over30_gr1_win': meanVol_perTick_over30_g1_win,
                'meanVol_perTick_over30_gr1_lose': meanVol_perTick_over30_g1_lose,
                'VolCandleMeanOver5Ratio_gr1_win': VolCandleMeanOver5Ratio_g1_win,
                'VolCandleMeanOver5Ratio_gr1_lose': VolCandleMeanOver5Ratio_g1_lose,
                'VolCandleMeanOver12Ratio_gr1_win': VolCandleMeanOver12Ratio_g1_win,
                'VolCandleMeanOver12Ratio_gr1_lose': VolCandleMeanOver12Ratio_g1_lose,
                'VolCandleMeanOver20Ratio_gr1_win': VolCandleMeanOver20Ratio_g1_win,
                'VolCandleMeanOver20Ratio_gr1_lose': VolCandleMeanOver20Ratio_g1_lose,
                'VolCandleMeanOver30Ratio_gr1_win': VolCandleMeanOver30Ratio_g1_win,
                'VolCandleMeanOver30Ratio_gr1_lose': VolCandleMeanOver30Ratio_g1_lose,

                # ── WIN/LOSE METRICS GR2 ───────────────────────────────
                'meanVol_perTick_over1_gr2_win': meanVol_perTick_over1_g2_win,
                'meanVol_perTick_over1_gr2_lose': meanVol_perTick_over1_g2_lose,
                'meanVol_perTick_over2_gr2_win': meanVol_perTick_over2_g2_win,
                'meanVol_perTick_over2_gr2_lose': meanVol_perTick_over2_g2_lose,
                'meanVol_perTick_over5_gr2_win': meanVol_perTick_over5_g2_win,
                'meanVol_perTick_over5_gr2_lose': meanVol_perTick_over5_g2_lose,
                'meanVol_perTick_over12_gr2_win': meanVol_perTick_over12_g2_win,
                'meanVol_perTick_over12_gr2_lose': meanVol_perTick_over12_g2_lose,
                'meanVol_perTick_over20_gr2_win': meanVol_perTick_over20_g2_win,
                'meanVol_perTick_over20_gr2_lose': meanVol_perTick_over20_g2_lose,
                'meanVol_perTick_over30_gr2_win': meanVol_perTick_over30_g2_win,
                'meanVol_perTick_over30_gr2_lose': meanVol_perTick_over30_g2_lose,
                'VolCandleMeanOver5Ratio_gr2_win': VolCandleMeanOver5Ratio_g2_win,
                'VolCandleMeanOver5Ratio_gr2_lose': VolCandleMeanOver5Ratio_g2_lose,
                'VolCandleMeanOver12Ratio_gr2_win': VolCandleMeanOver12Ratio_g2_win,
                'VolCandleMeanOver12Ratio_gr2_lose': VolCandleMeanOver12Ratio_g2_lose,
                'VolCandleMeanOver20Ratio_gr2_win': VolCandleMeanOver20Ratio_g2_win,
                'VolCandleMeanOver20Ratio_gr2_lose': VolCandleMeanOver20Ratio_g2_lose,
                'VolCandleMeanOver30Ratio_gr2_win': VolCandleMeanOver30Ratio_g2_win,
                'VolCandleMeanOver30Ratio_gr2_lose': VolCandleMeanOver30Ratio_g2_lose,

                # ── consensus ───────────────────────────────────────
                '≥2 algos - Nb': consensus_2_count,
                '≥2 algos - WR': consensus_2_wr,
                '≥3 algos - Nb': consensus_3_count,
                '≥3 algos - WR': consensus_3_wr
            })

        # -------- FIN boucle dataset_groups -------------------------- #
        return pd.DataFrame(results_table)

    # ------------------------------------------------------------------ #
    # Création des 3 tableaux (clusters 0+1, 0, 2)
    # ------------------------------------------------------------------ #
    table_cluster_0_1 = analyze_consensus_for_clusters([0, 1], "Clusters 0+1")
    table_cluster_0 = analyze_consensus_for_clusters([0], "Cluster 0")
    table_cluster_2 = analyze_consensus_for_clusters([2], "Cluster 2")

    return table_cluster_0_1, table_cluster_0, table_cluster_2


def print_consensus_table_with_colors(df, title):
    """
    Affichage d'un tableau de consensus avec :
      • métriques globales (Tous algos) en bleu
      • consensus ≥2 en vert
      • consensus ≥3 en orange
      • toutes les colonnes *_avg en bleu clair
      • séparation visuelle individuels / combinés
    """
    from colorama import Fore, Style

    BLUE = '\033[38;5;75m'  # Tous algos
    LIGHT_BLUE = '\033[38;5;117m'  # *_avg
    GREEN = '\033[38;5;46m'  # ≥2 algos
    ORANGE = '\033[38;5;208m'  # ≥3 algos
    RESET = Style.RESET_ALL

    # --- forçage string sur tous les noms de colonnes -----------------
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    df = df[[c for c in df.columns if not c.isdigit()]]  # <-- filtre sécurité

    print(f"\n{Fore.CYAN}📊 {title}{Style.RESET_ALL}")
    print("=" * 120)

    # ---------- LARGEUR DES COLONNES ----------------------------------
    col_w = {}
    for col in df.columns:
        col_str = str(col)
        w = len(col_str)
        for v in df[col]:
            w = max(w, len(str(v)))
        col_w[col] = w + 2

    # ---------- ENTÊTE COULEUR ----------------------------------------
    headers = []
    for col in df.columns:
        w = col_w[col]
        cs = str(col)

        if 'Tous algos' in cs:
            headers.append(f"{BLUE}{cs:>{w}}{RESET}")
        elif cs.endswith('_avg'):
            headers.append(f"{LIGHT_BLUE}{cs:>{w}}{RESET}")
        elif '≥2 algos' in cs:
            headers.append(f"{GREEN}{cs:>{w}}{RESET}")
        elif '≥3 algos' in cs:
            headers.append(f"{ORANGE}{cs:>{w}}{RESET}")
        else:
            headers.append(f"{cs:>{w}}")

    print(" | ".join(headers))
    print("-" * (sum(col_w.values()) + 3 * (len(df.columns) - 1)))

    # ---------- DATASETS INDIVIDUELS / COMBINÉS -----------------------
    indiv = ['Train', 'Test', 'Val1', 'Val', 'Unseen']
    combi = [g for g in df['Dataset Group'] if g not in indiv]

    def _print_rows(rows):
        for _, r in rows.iterrows():
            parts = []
            for col, val in r.items():
                w = col_w[col]
                cs = str(col)

                # format valeur
                if cs.endswith('WR'):
                    sval = f"{val:.1f}%" if val else "0.0%"
                elif cs.endswith('_avg'):
                    sval = f"{val:.2f}"
                else:
                    sval = str(val)

                # couleur cellule
                if 'Tous algos' in cs:
                    parts.append(f"{BLUE}{sval:>{w}}{RESET}")
                elif cs.endswith('_avg'):
                    parts.append(f"{LIGHT_BLUE}{sval:>{w}}{RESET}")
                elif '≥2 algos' in cs:
                    parts.append(f"{GREEN}{sval:>{w}}{RESET}")
                elif '≥3 algos' in cs:
                    parts.append(f"{ORANGE}{sval:>{w}}{RESET}")
                else:
                    parts.append(f"{sval:>{w}}")

            print(" | ".join(parts))

    # Individuels
    print(f"{Fore.YELLOW}# DATASETS INDIVIDUELS{RESET}")
    _print_rows(df[df['Dataset Group'].isin(indiv)])

    print("-" * (sum(col_w.values()) + 3 * (len(df.columns) - 1)))

    # Combinés
    print(f"{Fore.YELLOW}# DATASETS COMBINÉS{RESET}")
    _print_rows(df[df['Dataset Group'].isin(combi)])

# Exemple d'utilisation :
# GROUPE_SESSION_2 = [1, 2, 3, 4, 5]  # Définir vos sessions
# table_0_1, table_0, table_2 = create_consensus_analysis_tables_5_datasets(datasets_info_with_results, GROUPE_SESSION_2)
# print_consensus_table_with_colors(table_0_1, "Consensus Analysis - Clusters 0+1")


def analyze_consensus_insights_enhanced(table_cluster_0_1, table_cluster_0, table_cluster_2):
    """
    Génère des insights améliorés sur l'analyse de consensus pour les 3 tableaux
    CORRECTION: Gestion des tableaux vides (clusters sans données)
    """
    from colorama import Fore, Style

    print(f"\n{Fore.YELLOW}💡 INSIGHTS ANALYSE DE CONSENSUS (DÉTAILLÉ){Style.RESET_ALL}")
    print("=" * 100)

    tables_info = [
        (table_cluster_0_1, "Clusters 0+1"),
        (table_cluster_0, "Cluster 0"),
        (table_cluster_2, "Cluster 2")
    ]

    individual_datasets = ['Train', 'Test', 'Val1', 'Val', 'Unseen']

    for table, table_name in tables_info:
        print(f"\n🔍 {table_name}:")

        # Vérifier si le tableau est vide ou mal formé
        if table.empty:
            print(f"   ❌ Tableau vide - aucune donnée pour {table_name}")
            continue

        # Vérifier la structure des colonnes
        if 'Dataset Group' not in table.columns:
            print(f"   ❌ Structure incorrecte - pas de colonne 'Dataset Group' pour {table_name}")
            print(f"   🔧 Colonnes détectées: {list(table.columns)}")
            continue

        # Vérifier que les colonnes essentielles existent
        required_cols = ['Dataset Group', 'Tous algos - Nb', 'Tous algos - WR', '≥2 algos - WR', '≥3 algos - WR']
        missing_cols = [col for col in required_cols if col not in table.columns]

        if missing_cols:
            print(f"   ❌ {table_name} - Colonnes manquantes: {missing_cols}")
            print(f"   🔧 Probable cause: Aucune donnée pour ce cluster")

            # Essayer de trouver des colonnes similaires pour diagnostic
            available_cols = [col for col in table.columns if 'algos' in str(col).lower()]
            if available_cols:
                print(f"   🔧 Colonnes disponibles contenant 'algos': {available_cols}")
            else:
                print(f"   🔧 Aucune colonne contenant 'algos' trouvée")
            continue

        # Analyse des datasets individuels
        print(f"   📊 DATASETS INDIVIDUELS:")
        individual_rows = table[table['Dataset Group'].isin(individual_datasets)]

        if individual_rows.empty:
            print(f"   ⚠️  Aucun dataset individuel trouvé dans {table_name}")
            continue

        for _, row in individual_rows.iterrows():
            dataset = row['Dataset Group']
            try:
                wr_all = row['Tous algos - WR']
                wr_2 = row['≥2 algos - WR']
                wr_3 = row['≥3 algos - WR']
                nb_all = row['Tous algos - Nb']

                if nb_all > 0:
                    improvement_2 = wr_2 - wr_all
                    improvement_3 = wr_3 - wr_all

                    # Métriques moyennes (avec noms corrects)
                    vol_gr1_avg = row.get('vol50_gr1_avg', 0)
                    ext_gr1_avg = row.get('ext50_gr1_avg', 0)
                    dur_gr1_avg = row.get('dur50_gr1_avg', 0)
                    vol_gr2_avg = row.get('vol50_gr2_avg', 0)
                    ext_gr2_avg = row.get('ext50_gr2_avg', 0)
                    dur_gr2_avg = row.get('dur50_gr2_avg', 0)

                    print(
                        f"      • {dataset}: {nb_all} trades | WR: {wr_all:.1f}% → ≥2: {wr_2:.1f}% ({improvement_2:+.1f}pp) → ≥3: {wr_3:.1f}% ({improvement_3:+.1f}pp)")
                    print(
                        f"        Métriques GR1: Vol={vol_gr1_avg:.2f} | Ext={ext_gr1_avg:.2f} | Dur={dur_gr1_avg:.2f}")
                    print(
                        f"        Métriques GR2: Vol={vol_gr2_avg:.2f} | Ext={ext_gr2_avg:.2f} | Dur={dur_gr2_avg:.2f}")
                else:
                    print(f"      • {dataset}: Aucun trade")

            except KeyError as e:
                print(f"   ❌ Erreur d'accès colonne pour {dataset}: {e}")

        # Analyse comparative Train vs Test
        try:
            train_row = table[table['Dataset Group'] == 'Train']
            test_row = table[table['Dataset Group'] == 'Test']

            if len(train_row) > 0 and len(test_row) > 0:
                train_wr = train_row['Tous algos - WR'].iloc[0]
                test_wr = test_row['Tous algos - WR'].iloc[0]

                if train_wr > 0:  # Éviter division par zéro
                    robustesse = (test_wr / train_wr * 100)
                    print(f"   🎯 ROBUSTESSE Train→Test: {train_wr:.1f}% → {test_wr:.1f}% ({robustesse:.0f}% retention)")

                    if robustesse > 90:
                        print(f"      ✅ Très robuste (>90%)")
                    elif robustesse > 70:
                        print(f"      ⚠️  Modérément robuste (70-90%)")
                    else:
                        print(f"      ❌ Peu robuste (<70%)")
                else:
                    print(f"   🎯 ROBUSTESSE Train→Test: Train WR = 0, analyse impossible")

            # Analyse avec Unseen
            unseen_row = table[table['Dataset Group'] == 'Unseen']
            if len(train_row) > 0 and len(unseen_row) > 0:
                train_wr = train_row['Tous algos - WR'].iloc[0]
                unseen_wr = unseen_row['Tous algos - WR'].iloc[0]
                unseen_nb = unseen_row['Tous algos - Nb'].iloc[0]

                if unseen_nb > 0 and train_wr > 0:
                    generalization = (unseen_wr / train_wr * 100)
                    print(
                        f"   🚀 GÉNÉRALISATION Train→Unseen: {train_wr:.1f}% → {unseen_wr:.1f}% ({generalization:.0f}% retention)")

                    if generalization > 90:
                        print(f"      ✅ Excellente généralisation (>90%)")
                    elif generalization > 70:
                        print(f"      ⚠️  Bonne généralisation (70-90%)")
                    else:
                        print(f"      ❌ Généralisation limitée (<70%)")
                else:
                    print(f"   🚀 GÉNÉRALISATION Train→Unseen: Aucun trade détecté sur Unseen ou Train WR = 0")

            # Évolution temporelle Val1 → Val → Unseen
            val1_row = table[table['Dataset Group'] == 'Val1']
            val_row = table[table['Dataset Group'] == 'Val']

            if len(val1_row) > 0 and len(val_row) > 0 and len(unseen_row) > 0:
                val1_wr = val1_row['Tous algos - WR'].iloc[0]
                val_wr = val_row['Tous algos - WR'].iloc[0]
                unseen_wr = unseen_row['Tous algos - WR'].iloc[0]
                unseen_nb = unseen_row['Tous algos - Nb'].iloc[0]

                if unseen_nb > 0:
                    print(
                        f"   📈 ÉVOLUTION TEMPORELLE: Val1 {val1_wr:.1f}% → Val {val_wr:.1f}% → Unseen {unseen_wr:.1f}%")

                    if unseen_wr >= val_wr >= val1_wr:
                        print(f"      📈 Tendance positive (amélioration continue)")
                    elif unseen_wr >= val1_wr and unseen_wr >= val_wr:
                        print(f"      🎯 Performance stable/récupération")
                    else:
                        print(f"      📉 Tendance de dégradation")

        except Exception as e:
            print(f"   ❌ Erreur dans l'analyse comparative: {e}")


def create_unseen_performance_summary(consensus_table_0_1, consensus_table_0, consensus_table_2, comparison_metrics):
    """
    Crée un résumé spécifique des performances sur le dataset Unseen
    CORRECTION: Gestion des tableaux vides (clusters sans données)
    """
    from colorama import Fore, Style

    print(f"\n{Fore.MAGENTA}🚀 RÉSUMÉ PERFORMANCES DATASET UNSEEN{Style.RESET_ALL}")
    print("=" * 80)

    print("📈 Le dataset Unseen représente la période la plus récente (mai 2025)")
    print("🎯 Il constitue le test ultime de généralisation des algorithmes")
    print("⚡ Performances clés à analyser:")
    print()

    # Extraire les données Unseen des tableaux de consensus avec protection

    # Cluster 0+1
    if not consensus_table_0_1.empty and 'Dataset Group' in consensus_table_0_1.columns:
        unseen_cluster_01 = consensus_table_0_1[consensus_table_0_1['Dataset Group'] == 'Unseen']
        if len(unseen_cluster_01) > 0 and 'Tous algos - WR' in consensus_table_0_1.columns:
            wr_01 = unseen_cluster_01['Tous algos - WR'].iloc[0]
            nb_01 = unseen_cluster_01['Tous algos - Nb'].iloc[0]
            print(f"   🔹 Clusters 0+1 (Consolidation + Transition): {nb_01} trades, {wr_01:.1f}% WR")
        else:
            print(f"   🔹 Clusters 0+1 (Consolidation + Transition): ❌ Aucune donnée")
    else:
        print(f"   🔹 Clusters 0+1 (Consolidation + Transition): ❌ Tableau invalide")

    # Cluster 0
    if not consensus_table_0.empty and 'Dataset Group' in consensus_table_0.columns:
        unseen_cluster_0 = consensus_table_0[consensus_table_0['Dataset Group'] == 'Unseen']
        if len(unseen_cluster_0) > 0 and 'Tous algos - WR' in consensus_table_0.columns:
            wr_0 = unseen_cluster_0['Tous algos - WR'].iloc[0]
            nb_0 = unseen_cluster_0['Tous algos - Nb'].iloc[0]
            print(f"   🔹 Cluster 0 (Consolidation): {nb_0} trades, {wr_0:.1f}% WR")
        else:
            print(f"   🔹 Cluster 0 (Consolidation): ❌ Aucune donnée")
    else:
        print(f"   🔹 Cluster 0 (Consolidation): ❌ Tableau invalide")

    # Cluster 2 (avec protection spéciale car souvent vide)
    if not consensus_table_2.empty and 'Dataset Group' in consensus_table_2.columns:
        # Vérifier si les colonnes essentielles existent
        if all(col in consensus_table_2.columns for col in ['Tous algos - WR', 'Tous algos - Nb']):
            unseen_cluster_2 = consensus_table_2[consensus_table_2['Dataset Group'] == 'Unseen']
            if len(unseen_cluster_2) > 0:
                wr_2 = unseen_cluster_2['Tous algos - WR'].iloc[0]
                nb_2 = unseen_cluster_2['Tous algos - Nb'].iloc[0]
                print(f"   🔹 Cluster 2 (Breakout): {nb_2} trades, {wr_2:.1f}% WR")
            else:
                print(f"   🔹 Cluster 2 (Breakout): ❌ Aucune donnée Unseen")
        else:
            print(f"   🔹 Cluster 2 (Breakout): ❌ Pas de données pour ce cluster (probable: cluster vide)")
    else:
        print(f"   🔹 Cluster 2 (Breakout): ❌ Tableau vide ou invalide")

    # Analyser les performances individuelles des algorithmes sur Unseen
    print(f"\n📊 PERFORMANCES PAR ALGORITHME SUR UNSEEN:")

    # Vérifier que comparison_metrics contient les colonnes nécessaires
    required_unseen_cols = ['Nombre de trades (Unseen)', 'Win Rate (%) (Unseen)', 'Net PnL (Unseen)',
                            'Exp PnL (Unseen)']
    missing_cols = [col for col in required_unseen_cols if col not in comparison_metrics.columns]

    if missing_cols:
        print(f"   ❌ Colonnes manquantes dans comparison_metrics: {missing_cols}")
        print(f"   🔧 Colonnes disponibles: {list(comparison_metrics.columns)}")
        return

    # Filtrer les algorithmes qui ont des trades sur Unseen
    try:
        unseen_algos = comparison_metrics[comparison_metrics['Nombre de trades (Unseen)'] > 0].copy()

        if len(unseen_algos) > 0:
            # Trier par Win Rate décroissant
            unseen_algos_sorted = unseen_algos.sort_values('Win Rate (%) (Unseen)', ascending=False)

            print(f"   {len(unseen_algos_sorted)} algorithmes actifs sur Unseen:")

            for _, row in unseen_algos_sorted.head(10).iterrows():  # Top 10
                algo_name = row['Algorithme']
                nb_trades = row['Nombre de trades (Unseen)']
                wr = row['Win Rate (%) (Unseen)']
                pnl = row['Net PnL (Unseen)']
                exp_pnl = row['Exp PnL (Unseen)']

                print(f"      • {algo_name}: {nb_trades} trades, {wr:.1f}% WR, {pnl:.2f} PnL, {exp_pnl:.2f} Exp PnL")
        else:
            print("   ⚠️  Aucun algorithme n'a généré de trades sur Unseen")

        # Comparaison avec les autres datasets
        print(f"\n🔄 COMPARAISON AVEC LES AUTRES DATASETS:")

        if len(unseen_algos) > 0:
            # Calculer les moyennes de performance
            avg_wr_unseen = unseen_algos['Win Rate (%) (Unseen)'].mean()
            avg_trades_unseen = unseen_algos['Nombre de trades (Unseen)'].mean()

            # Comparer avec Train (vérifier d'abord que les colonnes existent)
            if all(col in comparison_metrics.columns for col in ['Nombre de trades', 'Win Rate (%)']):
                train_algos = comparison_metrics[comparison_metrics['Nombre de trades'] > 0]
                if len(train_algos) > 0:
                    avg_wr_train = train_algos['Win Rate (%)'].mean()
                    avg_trades_train = train_algos['Nombre de trades'].mean()

                    wr_retention = (avg_wr_unseen / avg_wr_train * 100) if avg_wr_train > 0 else 0
                    volume_retention = (avg_trades_unseen / avg_trades_train * 100) if avg_trades_train > 0 else 0

                    print(
                        f"   📈 WR moyen: Train {avg_wr_train:.1f}% → Unseen {avg_wr_unseen:.1f}% ({wr_retention:.0f}% retention)")
                    print(
                        f"   📊 Volume moyen: Train {avg_trades_train:.1f} → Unseen {avg_trades_unseen:.1f} trades ({volume_retention:.0f}% retention)")

                    # Évaluation de la qualité de généralisation
                    if wr_retention >= 90 and volume_retention >= 50:
                        print(f"   ✅ EXCELLENTE généralisation (WR>90%, Volume>50%)")
                    elif wr_retention >= 70 and volume_retention >= 30:
                        print(f"   ⚡ BONNE généralisation (WR>70%, Volume>30%)")
                    elif wr_retention >= 50:
                        print(f"   ⚠️  MODÉRÉE généralisation (WR>50%)")
                    else:
                        print(f"   ❌ FAIBLE généralisation (WR<50%)")
                else:
                    print(f"   ⚠️  Aucun algorithme actif sur Train pour comparaison")
            else:
                print(f"   ❌ Colonnes Train manquantes pour comparaison")

        print(f"\n💡 RECOMMANDATIONS:")
        print(f"   🎯 Analyser les algorithmes performants sur Unseen pour identifier les patterns robustes")
        print(f"   🔍 Étudier les différences de market conditions entre périodes")
        print(f"   ⚡ Considérer l'ensemble 'Train + Test + Val1 + Val + Unseen' pour la stratégie finale")

    except Exception as e:
        print(f"   ❌ Erreur dans l'analyse des performances par algorithme: {e}")
        print(f"   🔧 Debug - Colonnes comparison_metrics: {list(comparison_metrics.columns)}")


def print_enhanced_consensus_summary():
    """
    Affiche un résumé amélioré de l'analyse de consensus avec focus sur Unseen
    """

    print(f"\n{Fore.CYAN}📋 RÉSUMÉ GLOBAL ANALYSE DE CONSENSUS{Style.RESET_ALL}")
    print("=" * 80)

    print("🎯 MÉTHODOLOGIE:")
    print("   • Déduplication des trades identiques entre algorithmes")
    print("   • Analyse du consensus (≥2 algos, ≥3 algos)")
    print("   • Focus sur les clusters de marché (0: Consolidation, 1: Transition, 2: Breakout)")
    print("   • Intégration complète du dataset Unseen pour validation finale")

    print(f"\n📊 GROUPES DE DATASETS ANALYSÉS:")
    print("   🔹 Individuels: Train, Test, Val1, Val, Unseen")
    print("   🔹 Combinés: Train+Test, Val1+Val, Val+Unseen")
    print("   🔹 Complets: Train+Test+Val1+Val, Train+Test+Val1+Val+Unseen")
    print("   🔹 Validation: All Validation (Val1+Val+Unseen)")

    print(f"\n⚡ INSIGHTS CLÉS:")
    print("   • Le consensus ≥2 algos améliore généralement le Win Rate")
    print("   • Les clusters 0+1 offrent plus d'opportunités que le cluster 2")
    print("   • Unseen permet d'évaluer la robustesse temporelle")
    print("   • Les métriques GR1/GR2 donnent le contexte de marché")

# ────────────────────────────────────────────────────────────────────────────────
# INTÉGRATION DANS LE CODE PRINCIPAL - VERSION MISE À JOUR
# ────────────────────────────────────────────────────────────────────────────────

print(f"\n{Fore.CYAN}🔄 Création des tableaux d'analyse par consensus (INTÉGRATION UNSEEN)...{Style.RESET_ALL}")

# Créer les tableaux de consensus avec tous les 5 datasets
consensus_table_0_1, consensus_table_0, consensus_table_2 = create_consensus_analysis_tables_5_datasets(
    datasets_info_with_results)

# Afficher les tableaux avec coloration améliorée et intégration Unseen
print_consensus_table_with_colors(consensus_table_0_1, "ANALYSE CONSENSUS - CLUSTERS 0+1 (AVEC UNSEEN)")
print_consensus_table_with_colors(consensus_table_0, "ANALYSE CONSENSUS - CLUSTER 0 UNIQUEMENT (AVEC UNSEEN)")
print_consensus_table_with_colors(consensus_table_2, "ANALYSE CONSENSUS - CLUSTER 2 UNIQUEMENT (AVEC UNSEEN)")

# Analyser les insights améliorés avec focus sur Unseen
analyze_consensus_insights_enhanced(consensus_table_0_1, consensus_table_0, consensus_table_2)
import os
import csv
from datetime import datetime
from pathlib import Path


def sauvegarder_stats_algos(algorithms, dict_stats_algos, DIRECTORY_PATH):
    """
    Sauvegarde les statistiques des algorithmes actifs dans algoStat/{nom_algo}_stat.csv.
    Chaque algorithme activé génère son propre fichier CSV.

    Paramètres:
        algorithms (dict): dictionnaire des algorithmes avec leur statut d'activation
                          ex: {"algoShort1": algoShort1, "algoShort2": algoShort2}
        dict_stats_algos (dict): dictionnaire contenant les stats par algo
                                ex: {"algoShort1": {"trade_pnl": 150.0, "class_binaire": 1,
                                                   "date_trade": "2025-06-10 14:30:25", "split_source": "Train"}}
        DIRECTORY_PATH (str): répertoire principal du projet
    """
    # Créer le répertoire de sortie s'il n'existe pas
    repertoire_sortie = os.path.join(DIRECTORY_PATH, "algoStat")
    os.makedirs(repertoire_sortie, exist_ok=True)

    # Parcourir tous les algorithmes activés
    for algo_name, algo_config in algorithms.items():
        # Vérifier si l'algorithme est activé (contient des filtres)
        if not algo_config or len(algo_config) == 0:
            continue

        # Vérifier si on a des statistiques pour cet algorithme
        if algo_name not in dict_stats_algos:
            print(f"⚠️ Aucune statistique trouvée pour {algo_name}")
            continue

        # Chemin du fichier CSV pour cet algorithme
        fichier_sortie = os.path.join(repertoire_sortie, f"{algo_name}_stat.csv")

        # Supprimer le fichier existant s'il existe
        if os.path.exists(fichier_sortie):
            os.remove(fichier_sortie)

        # Créer le nouveau fichier avec en-têtes
        with open(fichier_sortie, mode='w', encoding='iso-8859-1', newline='') as f:
            writer = csv.writer(f, delimiter=';')

            # Écrire l'en-tête
            writer.writerow(["algo_name", "date", "trade_pnl", "class_binaire", "split_source"])

            # Écrire les données pour cet algorithme
            stats = dict_stats_algos[algo_name]

            # Gérer le cas où stats peut être une liste de trades ou un trade unique
            if isinstance(stats, list):
                # Cas où on a plusieurs trades pour cet algorithme
                for trade_stats in stats:
                    algo_name_val = trade_stats.get("algo_name", algo_name)
                    date_trade = trade_stats.get("date_trade", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    trade_pnl = trade_stats.get("trade_pnl", 0.0)
                    class_binaire = trade_stats.get("class_binaire", 0)
                    split_source = trade_stats.get("split_source", "Unknown")

                    writer.writerow([algo_name_val, date_trade, trade_pnl, class_binaire, split_source])
            else:
                # Cas où on a un seul trade
                algo_name_val = stats.get("algo_name", algo_name)
                date_trade = stats.get("date_trade", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                trade_pnl = stats.get("trade_pnl", 0.0)
                class_binaire = stats.get("class_binaire", 0)
                split_source = stats.get("split_source", "Unknown")

                writer.writerow([algo_name_val, date_trade, trade_pnl, class_binaire, split_source])

        print(f"✅ Fichier créé: {fichier_sortie}")


def construire_dict_stats_algos(to_save_datasets):
    """
    Construit le dictionnaire des statistiques à partir des résultats des algorithmes.
    Utilise les mêmes données filtrées que celles des tableaux de comparaison.
    AJOUTE le filtrage par Groupe 2 (sessions GROUPE_SESSION_2).

    Paramètres:
        to_save_datasets (list): Liste de tuples (dataset_name, to_save_data)
                                où to_save_data est une liste de (algo_name, df_selected)
                                df_selected contient déjà les trades filtrés par l'algorithme

    Retourne:
        dict: Dictionnaire des statistiques par algorithme (Groupe 2 seulement)
    """
    dict_stats_algos = {}

    # Mapping des noms de datasets vers les identifiants de split
    dataset_to_split = {
        "Train": "split1",
        "Test": "split2",
        "Val1": "split3",
        "Val": "split4",
        "Unseen": "unseen"
    }

    for dataset_name, to_save_data in to_save_datasets:
        split_source = dataset_to_split.get(dataset_name, dataset_name.lower())
        print(f"🔄 Traitement des données {dataset_name} ({split_source}) - Groupe 2 seulement...")

        for algo_name, df_selected in to_save_data:
            if algo_name not in dict_stats_algos:
                dict_stats_algos[algo_name] = []

            # DEBUG: Vérifier les colonnes disponibles dans df_selected
           # print(f"   🔍 DEBUG {algo_name}: Colonnes disponibles dans df_selected:")
            date_cols_available = [col for col in df_selected.columns if 'date' in col.lower() or 'time' in col.lower()]
            #print(f"      Colonnes de date: {date_cols_available}")

            # df_selected contient déjà les trades sélectionnés par l'algorithme
            # avec la colonne PnlAfterFiltering != 0

            # Déterminer la colonne PnL (identique au code original)
            pnl_col = None
            for col in ['PnlAfterFiltering', 'trade_pnl']:
                if col in df_selected.columns:
                    pnl_col = col
                    break

            if pnl_col is None:
                print(f"⚠️ Aucune colonne PnL trouvée pour {algo_name} dans {dataset_name}")
                continue

            # Vérifier la présence de la colonne deltaCustomSessionIndex
            if 'deltaCustomSessionIndex' not in df_selected.columns:
                raise ValueError(f"❌ Colonne 'deltaCustomSessionIndex' manquante pour {algo_name} dans {dataset_name}")

            # FILTRAGE 1: Trades avec PnL != 0 (comme avant)
            df_trades = df_selected[df_selected[pnl_col] != 0].copy()

            # FILTRAGE 2: NOUVEAU - Filtrer par Groupe 2 (comme dans create_cluster_analysis_table)
            df_groupe2 = df_trades[df_trades['deltaCustomSessionIndex'].isin(GROUPE_SESSION_2)].copy()

            if len(df_groupe2) == 0:
                print(f"⚠️ Aucun trade du Groupe 2 pour {algo_name} dans {dataset_name}")
                continue

            # Vérifier strictement la présence de date
            if 'date' not in df_groupe2.columns:
                raise ValueError(
                    f"❌ ERREUR CRITIQUE: Colonne 'date' manquante pour {algo_name} dans {dataset_name}! "
                    f"Cette colonne est obligatoire pour extraire la date du trade.")

            #print(f"   ✅ Colonne de date trouvée: date")

            total_avant_g2 = len(df_trades)
            total_apres_g2 = len(df_groupe2)
            # print(f"   {algo_name}: {total_avant_g2} trades → {total_apres_g2} trades Groupe 2 "
            #       f"(sessions {GROUPE_SESSION_2})")

            # Traiter chaque trade du Groupe 2 (extraire date + autres éléments)
            for _, row in df_groupe2.iterrows():
                # EXTRACTION DE LA DATE (strictement depuis date)
                if pd.notna(row['date']):
                    try:
                        if isinstance(row['date'], str):
                            # Si c'est déjà une string, l'utiliser directement
                            date_trade = row['date']
                            # Valider le format en tentant de le parser
                            pd.to_datetime(date_trade)
                        else:
                            # Si c'est un timestamp ou datetime, convertir au format ISO
                            date_trade = pd.to_datetime(row['date']).strftime(
                                "%Y-%m-%d %H:%M:%S")
                    except (ValueError, TypeError) as e:
                        raise ValueError(
                            f"❌ ERREUR: Date invalide dans date pour {algo_name} dans {dataset_name}. "
                            f"Valeur: {row['date']}, Erreur: {e}")
                else:
                    raise ValueError(
                        f"❌ ERREUR: Date manquante dans date pour {algo_name} dans {dataset_name}")

                # EXTRACTION DES AUTRES ÉLÉMENTS (PnL, classe, session)
                trade_pnl = float(row[pnl_col])  # PnL du trade
                class_binaire = 1 if trade_pnl > 0 else 0  # Classification binaire
                session_index = int(row['deltaCustomSessionIndex'])  # Index de session

                # EXTRACTION D'ÉLÉMENTS ADDITIONNELS (optionnels pour enrichissement)
                additional_data = {}

                # Colonnes d'intérêt pour enrichissement (si disponibles)
                optional_cols = {
                    'sc_candleDuration': 'candle_duration',
                    'sc_volume_perTick': 'volume_per_tick',
                    'sc_candleSizeTicks': 'candle_size_ticks',
                    'Cluster_G2': 'cluster_g2',
                    'sc_reg_slope_30P_2': 'reg_slope_30p_2'
                }

                for source_col, target_key in optional_cols.items():
                    if source_col in df_groupe2.columns and pd.notna(row[source_col]):
                        additional_data[target_key] = row[source_col]

                # Ajouter les statistiques du trade (Groupe 2 seulement)
                trade_stats = {
                    "algo_name": algo_name,
                    "date_trade": date_trade,
                    "trade_pnl": trade_pnl,
                    "class_binaire": class_binaire,
                    "split_source": split_source,
                    "session_index": session_index,
                    "date_source_column": "date",  # Source fixe
                    **additional_data  # Données additionnelles
                }

                dict_stats_algos[algo_name].append(trade_stats)

    return dict_stats_algos


# ────────────────────────────────────────────────────────────────────────────────
# INTÉGRATION DANS LE CODE PRINCIPAL
# ────────────────────────────────────────────────────────────────────────────────

# # Ajouter cette section à la fin du script principal, après l'analyse des clusters :
#
# print(f"\n{Fore.CYAN}💾 SAUVEGARDE DES STATISTIQUES D'ALGORITHMES{Style.RESET_ALL}")
# print("=" * 80)
#
# # Préparer les données des 5 datasets
datasets_to_save = [
    ("Train", to_save_train),
    ("Test", to_save_test),
    ("Val1", to_save_val1),
    ("Val", to_save_val),
    ("Unseen", to_save_unseen)
]

# Construire le dictionnaire des statistiques
dict_stats_algos = construire_dict_stats_algos(datasets_to_save)

# Afficher un résumé des statistiques collectées (cohérent avec les analyses Groupe 2)
print("📊 Résumé des statistiques collectées (Groupe 2 seulement - cohérent avec analyses clusters) :")
print(f"🎯 Sessions du Groupe 2 : {GROUPE_SESSION_2}")
print("=" * 80)

for algo_name, stats_list in dict_stats_algos.items():
    if stats_list:
        total_trades = len(stats_list)
        winning_trades = sum(1 for stats in stats_list if stats["class_binaire"] == 1)
        total_pnl = sum(stats["trade_pnl"] for stats in stats_list)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        # Répartition par split
        split_counts = {}
        split_pnl = {}
        for stats in stats_list:
            split = stats["split_source"]
            split_counts[split] = split_counts.get(split, 0) + 1
            split_pnl[split] = split_pnl.get(split, 0) + stats["trade_pnl"]

        splits_str = ", ".join([f"{k}:{v}" for k, v in split_counts.items()])

        # Vérification des sessions (pour debug)
        session_indices = set(stats["session_index"] for stats in stats_list if "session_index" in stats)

        # print(f"  {algo_name}: {total_trades} trades G2, {winning_trades} wins ({win_rate:.1f}%), "
        #       f"PnL: {total_pnl:.2f}")
        # print(f"    Répartition: {splits_str}")
        # print(f"    Sessions présentes: {sorted(session_indices) if session_indices else 'N/A'}")
        # print(f"    Source de date: date")
        #
        # # Vérification de cohérence avec les tableaux cluster analysis
        # print(f"    💡 Ces chiffres doivent correspondre aux colonnes G2_Total des tableaux clusters")

# Sauvegarder les statistiques dans des fichiers CSV séparés
print(f"\n💾 Sauvegarde dans {DIRECTORY_PATH}/algoStat/...")
sauvegarder_stats_algos(algorithms, dict_stats_algos, DIRECTORY_PATH)

print(f"\n✅ Sauvegarde terminée! Fichiers créés dans {DIRECTORY_PATH}/algoStat/")
print("📁 Structure des fichiers générés :")
print("   • Format: {nom_algo}_stat.csv")
print("   • Colonnes: algo_name;date;trade_pnl;class_binaire;split_source")
print("   • Encodage: ISO-8859-1")
print("   • Séparateur: ;")
print(f"   • 🎯 Filtrage: GLOBAL_MICRO_FILTER + Filtres algo + Groupe 2 ({GROUPE_SESSION_2})")
print("   • 📊 Cohérence: Identique aux analyses par clusters du Groupe 2")

# 🚀 LANCER LES NOUVELLES FONCTIONS D'ANALYSE UNSEEN
# ────────────────────────────────────────────────────────────────────────────────

# Créer le résumé spécifique des performances sur Unseen
create_unseen_performance_summary(consensus_table_0_1, consensus_table_0, consensus_table_2, comparison_metrics)

# Afficher le résumé global amélioré
print_enhanced_consensus_summary()

# ────────────────────────────────────────────────────────────────────────────────
# VALIDATION FINALE - VÉRIFICATION DE L'INTÉGRATION UNSEEN
# ────────────────────────────────────────────────────────────────────────────────

print(f"\n{Fore.GREEN}✅ VALIDATION INTÉGRATION UNSEEN{Style.RESET_ALL}")
print("=" * 60)

# Vérifier que Unseen apparaît bien dans tous les tableaux
datasets_to_check = ['Train', 'Test', 'Val1', 'Val', 'Unseen']
missing_datasets = []

for dataset in datasets_to_check:
    if dataset not in consensus_table_0_1['Dataset Group'].values:
        missing_datasets.append(dataset)

if missing_datasets:
    print(f"❌ Datasets manquants: {missing_datasets}")
else:
    print(f"✅ Tous les datasets présents: {datasets_to_check}")

# Compter les nouvelles combinaisons avec Unseen
unseen_combinations = [group for group in consensus_table_0_1['Dataset Group'].values if 'Unseen' in group]
print(f"✅ Nouvelles combinaisons avec Unseen: {len(unseen_combinations)}")
for combo in unseen_combinations:
    print(f"   • {combo}")

print(f"\n🎯 INTÉGRATION UNSEEN TERMINÉE AVEC SUCCÈS!")
print(f"📊 Le dataset Unseen est maintenant pleinement intégré dans toutes les analyses")
print(f"🚀 Vous pouvez désormais évaluer la robustesse temporelle de vos algorithmes")


# ────────────────────────────────────────────────────────────────────────────────
# BONUS : ANALYSE COMPLÉMENTAIRE UNSEEN
# ────────────────────────────────────────────────────────────────────────────────

def create_unseen_detailed_comparison():
    """
    Analyse détaillée comparative entre tous les datasets avec focus sur Unseen
    """
    print(f"\n{Fore.MAGENTA}🔬 ANALYSE DÉTAILLÉE COMPARATIVE (FOCUS UNSEEN){Style.RESET_ALL}")
    print("=" * 80)

    # Analyser les performances par dataset
    datasets = ['Train', 'Test', 'Val1', 'Val', 'Unseen']

    print("📊 ÉVOLUTION DES PERFORMANCES PAR DATASET:")
    print("-" * 50)

    for dataset in datasets:
        # Filtrer les algorithmes actifs sur ce dataset
        col_nb = f'Nombre de trades ({dataset})' if dataset != 'Train' else 'Nombre de trades'
        col_wr = f'Win Rate (%) ({dataset})' if dataset != 'Train' else 'Win Rate (%)'
        col_pnl = f'Net PnL ({dataset})' if dataset != 'Train' else 'Net PnL'

        if col_nb in comparison_metrics.columns:
            active_algos = comparison_metrics[comparison_metrics[col_nb] > 0]

            if len(active_algos) > 0:
                avg_trades = active_algos[col_nb].mean()
                avg_wr = active_algos[col_wr].mean()
                total_pnl = active_algos[col_pnl].sum()
                nb_algos = len(active_algos)

                print(
                    f"🔹 {dataset:8}: {nb_algos:2d} algos | {avg_trades:6.1f} trades/algo | {avg_wr:5.1f}% WR | {total_pnl:8.2f} PnL total")
            else:
                print(f"🔹 {dataset:8}: Aucun algorithme actif")

    # Calculer les corrélations de performance entre datasets
    print(f"\n🔗 CORRÉLATIONS DE PERFORMANCE:")
    print("-" * 40)

    # Créer une matrice de corrélation simple
    datasets_with_data = []
    for dataset in datasets:
        col_wr = f'Win Rate (%) ({dataset})' if dataset != 'Train' else 'Win Rate (%)'
        if col_wr in comparison_metrics.columns:
            datasets_with_data.append((dataset, col_wr))

    print("💡 Analyse des algorithmes performants de manière consistante:")

    # Identifier les algorithmes avec WR > 60% sur plusieurs datasets
    consistent_performers = []

    for _, row in comparison_metrics.iterrows():
        algo_name = row['Algorithme']
        performance_scores = []

        for dataset, col_wr in datasets_with_data:
            wr = row[col_wr]
            if wr > 60:  # Seuil de performance
                performance_scores.append(dataset)

        if len(performance_scores) >= 3:  # Performance sur au moins 3 datasets
            consistent_performers.append((algo_name, performance_scores))

    if consistent_performers:
        print(f"✅ {len(consistent_performers)} algorithmes performants de manière consistante:")
        for algo, datasets_list in consistent_performers:
            print(f"   • {algo}: {', '.join(datasets_list)}")
    else:
        print("⚠️  Aucun algorithme ne montre de performance consistante (WR>60%) sur ≥3 datasets")

    # Focus spécial sur Unseen
    print(f"\n🚀 FOCUS SPÉCIAL DATASET UNSEEN:")
    print("-" * 35)

    if 'Win Rate (%) (Unseen)' in comparison_metrics.columns:
        unseen_performers = comparison_metrics[comparison_metrics['Nombre de trades (Unseen)'] > 0]

        if len(unseen_performers) > 0:
            # Top performers sur Unseen
            top_unseen = unseen_performers.sort_values('Win Rate (%) (Unseen)', ascending=False).head(5)

            print("🏆 TOP 5 ALGORITHMES SUR UNSEEN:")
            for i, (_, row) in enumerate(top_unseen.iterrows(), 1):
                algo = row['Algorithme']
                wr = row['Win Rate (%) (Unseen)']
                trades = row['Nombre de trades (Unseen)']
                pnl = row['Net PnL (Unseen)']

                print(f"   {i}. {algo}: {wr:.1f}% WR ({trades} trades, {pnl:.2f} PnL)")

            # Comparer avec Train
            print(f"\n📈 COMPARAISON TRAIN vs UNSEEN (Top performers):")
            for _, row in top_unseen.head(3).iterrows():
                algo = row['Algorithme']
                wr_train = row['Win Rate (%)']
                wr_unseen = row['Win Rate (%) (Unseen)']
                retention = (wr_unseen / wr_train * 100) if wr_train > 0 else 0

                print(f"   • {algo}: {wr_train:.1f}% → {wr_unseen:.1f}% ({retention:.0f}% retention)")
        else:
            print("❌ Aucun algorithme n'a généré de trades sur Unseen")
    else:
        print("❌ Données Unseen non disponibles dans le tableau des métriques")


def create_final_recommendation_unseen():
    """
    Créer des recommandations finales basées sur l'analyse Unseen
    """
    print(f"\n{Fore.YELLOW}🎯 RECOMMANDATIONS FINALES (BASÉES SUR UNSEEN){Style.RESET_ALL}")
    print("=" * 80)

    print("📋 STRATÉGIE DE SÉLECTION D'ALGORITHMES:")
    print("-" * 45)

    print("1. 🎯 PRIORITÉ 1 - Robustesse temporelle:")
    print("   • Sélectionner les algorithmes performants sur Train + Test + Unseen")
    print("   • Win Rate retention ≥ 70% entre Train et Unseen")
    print("   • Volume de trades suffisant sur toutes les périodes")

    print(f"\n2. 🔄 PRIORITÉ 2 - Diversification:")
    print("   • Combiner algorithmes performants sur différents clusters")
    print("   • Utiliser le consensus ≥2 algos pour améliorer la robustesse")
    print("   • Équilibrer clusters 0+1 (opportunités) vs cluster 2 (breakouts)")

    print(f"\n3. ⚡ PRIORITÉ 3 - Validation finale:")
    print("   • Tester sur l'ensemble 'Train + Test + Val1 + Val + Unseen'")
    print("   • Surveiller les métriques GR1/GR2 pour le contexte de marché")
    print("   • Ajuster selon l'évolution des conditions de marché")

    print(f"\n🚀 PROCHAINES ÉTAPES:")
    print("   • Implémenter les algorithmes sélectionnés en production")
    print("   • Monitorer les performances en temps réel")
    print("   • Mettre à jour les modèles avec nouvelles données")
    print("   • Réévaluer périodiquement avec de nouveaux datasets 'Unseen'")


# Lancer les analyses complémentaires
create_unseen_detailed_comparison()
create_final_recommendation_unseen()


# ────────────────────────────────────────────────────────────────────────────────
# FIN DE L'ANALYSE PAR CLUSTERS
# ────────────────────────────────────────────────────────────────────────────────


# ------------------------------------------------------------------
# AFFICHAGE « RÉPARTITION PAR SESSIONS INTRADAY » (version alignée)
# ------------------------------------------------------------------
from colorama import Fore, Style
# ---------------------------------------------------------------------------
# 1)  Ordre cible pour tous les affichages
# ---------------------------------------------------------------------------
DATASET_ORDER = ['Unseen', 'Val', 'Val1', 'Test', 'Train']   # <= NOUVEL ORDRE

# ---------------------------------------------------------------------------
# 2)  Helper : ré-ordonne dynamiquement les colonnes d’un tableau déjà créé
# ---------------------------------------------------------------------------
def reorder_columns_by_dataset(df: pd.DataFrame,
                               base_cols: list[str] | None = None) -> pd.DataFrame:
    """
    Renv­oie un nouveau DF dont les colonnes sont ré-ordonnées selon DATASET_ORDER.
    On conserve les colonnes de base (p. ex. ['Algorithme']) puis, pour chaque
    dataset, on place toutes les colonnes qui commencent par ce nom.
    """
    if base_cols is None:
        base_cols = ['Algorithme']

    # Colonnes dataset : on garde l’ordre défini in DATASET_ORDER
    ordered_cols = base_cols[:]
    for ds in DATASET_ORDER:
        ordered_cols.extend([c for c in df.columns if c.startswith(ds) and c not in ordered_cols])

    # Colonnes résiduelles (au cas où) en fin de tableau
    ordered_cols.extend([c for c in df.columns if c not in ordered_cols])
    return df[ordered_cols]

# ---------------------------------------------------------------------------
# 3)  Affichage couleur + séparateur fin « | »
# ---------------------------------------------------------------------------
def print_table_with_dataset_separators(display_df: pd.DataFrame,
                                        nb_color='38;5;117',
                                        wr_color='38;5;157',
                                        title: str = "TABLEAU") -> None:
    """
    • Ré-ordonne les colonnes              -> DATASET_ORDER
    • Affiche la table avec un « | » fin   -> séparateur entre datasets
    • Colorise Nb (bleu clair) et WR (vert clair)
    """
    from colorama import Style, Fore

    # Ré-ordonnage
    display_df = reorder_columns_by_dataset(display_df)

    # Calcule largeur max par colonne
    col_w = {c: max(len(c), *(len(str(v)) for v in display_df[c])) + 2
             for c in display_df.columns}

    # Couleurs ANSI
    BLUE  = f'\033[{nb_color}m'
    GREEN = f'\033[{wr_color}m'
    RESET = Style.RESET_ALL

    # ---------- ENTÊTE ----------
    print(f"\n{Fore.GREEN}📊 {title}{Style.RESET_ALL}")
    sep_line = []

    heads = []
    for col in display_df.columns:
        # Ajoute le séparateur si on change de dataset
        ds_name = col.split('_')[0]
        if (ds_name in DATASET_ORDER  # évite Algorithme, etc.
            and heads                                   # pas avant la 1ʳᵉ col
            and heads[-1].split('_')[0] != ds_name):
            sep_line.append('|')
        heads.append(f"{col:>{col_w[col]}}")

    print(" | ".join(heads).replace('| |', '|'))  # header
    print("-" * (sum(col_w.values()) + 3*len(display_df.columns)))  # règle

    # ---------- LIGNES ----------
    for _, row in display_df.iterrows():
        line_parts, last_ds = [], None
        for col in display_df.columns:
            ds_name = col.split('_')[0]
            # Séparateur fin
            if ds_name in DATASET_ORDER and last_ds and ds_name != last_ds:
                line_parts.append('|')
            # Mise en couleur
            val = f"{row[col]:>{col_w[col]}}" if col != 'Algorithme' else f"{row[col]:<{col_w[col]}}"
            if 'Nb' in col:
                val = f"{BLUE}{val}{RESET}"
            elif 'WR' in col:
                val = f"{GREEN}{val}{RESET}"
            line_parts.append(val)
            last_ds = ds_name if ds_name in DATASET_ORDER else last_ds
        print(" | ".join(line_parts).replace('| |', '|'))
print(f"\n{Fore.GREEN}📊 RÉPARTITION PAR SESSIONS INTRADAY{Style.RESET_ALL}")
print("=" * 150)
print(f"Groupe 1: Sessions {GROUPE_SESSION_1} | Groupe 2: Sessions {GROUPE_SESSION_2}")
print("=" * 150)

# ① ré-ordonne les colonnes  ② insère les séparateurs verticaux « | »
print_table_with_dataset_separators(
    sessions_display_table,                      # ton DF déjà prêt
    title="RÉPARTITION PAR SESSIONS INTRADAY"    # titre de la table
)


analyze_session_insights_5_datasets(sessions_analysis_table)

# ────────────────────────────────────────────────────────────────────────────────
# MATRICE GLOBALE DE JACARD
# ────────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────
# MATRICES GLOBALES DE JACCARD pour les 4 datasets
# ────────────────────────────────────────────────────────────────────────────────
print(f"{Fore.YELLOW}\n{'=' * 120}")
print("ANALYSE GLOBALE DE SIMILARITÉ JACCARD (4 DATASETS)")
print(f"{'=' * 120}{Style.RESET_ALL}")

# Définir les datasets avec leurs noms et DataFrames
datasets_info = [
    ("ENTRAINEMENT", algo_dfs_train),
    ("TEST", algo_dfs_test),
    ("VALIDATION 1", algo_dfs_val1),
    ("VALIDATION 2", algo_dfs_val),
    ("UNSEEN", algo_dfs_unseen)  # NOUVEAU
]

# Stocker les matrices pour analyse ultérieure
jaccard_matrices = {}
redundant_pairs_by_dataset = {}

# Analyser chaque dataset
for dataset_name, algo_dfs in datasets_info:
    print(f"\n{Fore.BLUE}{'=' * 120}")
    print(f"MATRICE JACCARD - DONNÉES {dataset_name}")
    print(f"{'=' * 120}{Style.RESET_ALL}")

    # Créer et afficher la matrice Jaccard
    jaccard_matrix = create_full_jaccard_matrix(algo_dfs)
    jaccard_matrices[dataset_name] = jaccard_matrix

    # Afficher la matrice avec analyse
    display_jaccard_matrix(jaccard_matrix=jaccard_matrix,
                           threshold=JACCARD_THRESHOLD,
                           algo_dfs=algo_dfs,
                           min_common_trades=MIN_COMMON_TRADES)

    # Analyser la redondance globale
    redundant_pairs = analyze_global_redundancy(jaccard_matrix, JACCARD_THRESHOLD)
    redundant_pairs_by_dataset[dataset_name] = redundant_pairs

# ────────────────────────────────────────────────────────────────────────────────
# ANALYSE DE CONSISTANCE INTER-DATASETS (NOUVEAU)
# ────────────────────────────────────────────────────────────────────────────────
print(f"{Fore.CYAN}\n{'=' * 120}")
print("ANALYSE DE CONSISTANCE DES REDONDANCES INTER-DATASETS")
print(f"{'=' * 120}{Style.RESET_ALL}")

# Analyser la consistance des paires redondantes
all_redundant_pairs = set()
for dataset_name, pairs in redundant_pairs_by_dataset.items():
    all_redundant_pairs.update(pairs)

if all_redundant_pairs:
    print("\n📊 CONSISTANCE DES PAIRES REDONDANTES ENTRE DATASETS:")

    # Pour chaque paire redondante trouvée, vérifier sur combien de datasets elle apparaît
    consistency_analysis = {}

    for pair in all_redundant_pairs:
        datasets_with_pair = []
        for dataset_name, pairs in redundant_pairs_by_dataset.items():
            if pair in pairs:
                datasets_with_pair.append(dataset_name)

        consistency_analysis[pair] = {
            'datasets': datasets_with_pair,
            'count': len(datasets_with_pair),
            'consistency_rate': len(datasets_with_pair) / len(datasets_info) * 100
        }

    # Trier par niveau de consistance
    sorted_pairs = sorted(consistency_analysis.items(),
                          key=lambda x: x[1]['consistency_rate'],
                          reverse=True)

    print(f"\n🔴 PAIRES CONSTAMMENT REDONDANTES (présentes sur plusieurs datasets):")
    for pair, info in sorted_pairs:
        if info['count'] > 1:  # Présent sur plus d'un dataset
            datasets_str = ", ".join(info['datasets'])
            print(
                f"  {pair[0]} ↔ {pair[1]}: {info['consistency_rate']:.0f}% ({info['count']}/{len(datasets_info)} datasets)")
            print(f"    Présent sur: {datasets_str}")

    # Identifier les paires redondantes uniquement sur un dataset (potentiels faux positifs)
    print(f"\n🟡 PAIRES REDONDANTES SUR UN SEUL DATASET (potentiels faux positifs):")
    for pair, info in sorted_pairs:
        if info['count'] == 1:
            dataset_str = info['datasets'][0]
            print(f"  {pair[0]} ↔ {pair[1]}: Uniquement sur {dataset_str}")

else:
    print("✅ Aucune paire redondante détectée sur l'ensemble des datasets.")

# ────────────────────────────────────────────────────────────────────────────────
# SYNTHÈSE GLOBALE DES REDONDANCES (NOUVEAU)
# ────────────────────────────────────────────────────────────────────────────────
print(f"\n{Fore.GREEN}{'=' * 120}")
print("SYNTHÈSE GLOBALE DES REDONDANCES")
print(f"{'=' * 120}{Style.RESET_ALL}")

print("\n📈 RÉSUMÉ PAR DATASET:")
for dataset_name, pairs in redundant_pairs_by_dataset.items():
    print(f"  {dataset_name}: {len(pairs)} paires redondantes")

if all_redundant_pairs:
    # Calculer le score de redondance global pour chaque algorithme
    algo_redundancy_scores = {}
    all_algos = set()

    # Collecter tous les algorithmes
    for dataset_name, algo_dfs in datasets_info:
        all_algos.update(algo_dfs.keys())

    # Calculer le score de redondance pour chaque algorithme
    for algo in all_algos:
        redundancy_count = 0
        total_possible_pairs = 0

        for pair in all_redundant_pairs:
            if algo in pair:
                redundancy_count += 1

        # Le nombre total de paires possibles pour cet algo
        total_possible_pairs = len(all_algos) - 1

        if total_possible_pairs > 0:
            redundancy_rate = redundancy_count / total_possible_pairs * 100
            algo_redundancy_scores[algo] = redundancy_rate

    # Afficher les algorithmes les plus redondants
    if algo_redundancy_scores:
        sorted_algos = sorted(algo_redundancy_scores.items(),
                              key=lambda x: x[1],
                              reverse=True)

        print(f"\n🔴 ALGORITHMES LES PLUS REDONDANTS:")
        for algo, score in sorted_algos[:5]:  # Top 5
            if score > 0:
                print(f"  {algo}: {score:.1f}% de redondance")

        print(f"\n✅ ALGORITHMES LES MOINS REDONDANTS:")
        for algo, score in sorted_algos[-5:]:  # Bottom 5
            print(f"  {algo}: {score:.1f}% de redondance")

print(f"\n💡 RECOMMANDATIONS:")
if all_redundant_pairs:
    print("  • Considérer la suppression des algorithmes constamment redondants")
    print("  • Prioriser les algorithmes avec faible score de redondance")
    print("  • Vérifier manuellement les paires redondantes sur un seul dataset")
else:
    print("  • Excellente diversification des algorithmes détectée")
    print("  • Aucune optimisation de redondance nécessaire")


# ────────────────────────────────────────────────────────────────────────────────
# MATRICES DE JACCARD PAR CLUSTERS (NOUVEAU)
# ────────────────────────────────────────────────────────────────────────────────

def calculate_jaccard_for_cluster(algo_dfs, target_clusters, dataset_name=""):
    """
    Calcule la matrice de Jaccard pour des clusters spécifiques

    Parameters:
    -----------
    algo_dfs : dict
        Dictionnaire {nom_algo: DataFrame}
    target_clusters : list
        Liste des clusters à analyser (ex: [0] ou [0,1])
    dataset_name : str
        Nom du dataset pour l'affichage

    Returns:
    --------
    pd.DataFrame : Matrice de similarité Jaccard
    """

    # Colonnes d'indicateurs pour identifier les trades uniques
    indicator_columns = [
        'rsi_', 'macd', 'macd_signal', 'macd_hist',
        'timeElapsed2LastBar', 'timeStampOpening',
        'ratio_deltaRevZone_volCandle'
    ]

    # Stocker les ensembles de trades pour chaque algo
    algo_trade_sets = {}

    for algo_name, df_algo in algo_dfs.items():
        # Vérifier les colonnes nécessaires
        if 'Cluster_G2' not in df_algo.columns or 'deltaCustomSessionIndex' not in df_algo.columns:
            continue

        # Déterminer la colonne PnL
        pnl_col = None
        for col in ['PnlAfterFiltering', 'trade_pnl']:
            if col in df_algo.columns:
                pnl_col = col
                break

        if pnl_col is None:
            continue

        # Filtrer les trades du groupe 2 avec les clusters ciblés
        df_filtered = df_algo[
            (df_algo['deltaCustomSessionIndex'].isin(GROUPE_SESSION_2)) &
            (df_algo['Cluster_G2'].isin(target_clusters)) &
            (df_algo[pnl_col] != 0)
            ].copy()

        if len(df_filtered) == 0:
            algo_trade_sets[algo_name] = set()
            continue

        # Créer l'ensemble des trades uniques
        valid_cols = [c for c in indicator_columns if c in df_filtered.columns]
        trade_set = set()

        for _, row in df_filtered.iterrows():
            trade_key = tuple(row[col] for col in valid_cols)
            trade_set.add(trade_key)

        algo_trade_sets[algo_name] = trade_set

    # Créer la matrice de Jaccard
    algos = sorted(algo_trade_sets.keys())
    jaccard_matrix = pd.DataFrame(index=algos, columns=algos, dtype=float)

    for i, algo1 in enumerate(algos):
        for j, algo2 in enumerate(algos):
            if i == j:
                jaccard_matrix.loc[algo1, algo2] = 1.0
            else:
                set1 = algo_trade_sets[algo1]
                set2 = algo_trade_sets[algo2]

                if len(set1) == 0 and len(set2) == 0:
                    jaccard_matrix.loc[algo1, algo2] = 0.0
                else:
                    intersection = len(set1.intersection(set2))
                    union = len(set1.union(set2))
                    jaccard_sim = intersection / union if union > 0 else 0.0
                    jaccard_matrix.loc[algo1, algo2] = jaccard_sim

    return jaccard_matrix


# ────────────────────────────────────────────────────────────────────────────────
# INTÉGRATION DANS LE CODE PRINCIPAL
# ────────────────────────────────────────────────────────────────────────────────

print(f"{Fore.YELLOW}\n{'=' * 120}")
print("ANALYSE DE SIMILARITÉ JACCARD PAR CLUSTERS")
print(f"{'=' * 120}{Style.RESET_ALL}")

# Pour chaque dataset, calculer les matrices de Jaccard par cluster
datasets_for_cluster_jaccard = [
    ("ENTRAINEMENT", algo_dfs_train),
    ("TEST", algo_dfs_test),
    ("VALIDATION 1", algo_dfs_val1),
    ("VALIDATION 2", algo_dfs_val),
    ("UNSEEN", algo_dfs_unseen)  # NOUVEAU
]
# Stocker les résultats pour analyse comparative
jaccard_cluster_results = {
    'cluster_0': {},
    'cluster_0_1': {}
}

for dataset_name, algo_dfs in datasets_for_cluster_jaccard:
    # ────────────────────────────────────────────────────────────────────────────
    # MATRICE JACCARD - CLUSTER 0 (CONSOLIDATION)
    # ────────────────────────────────────────────────────────────────────────────
    print(f"\n{Fore.BLUE}{'=' * 120}")
    print(f"MATRICE JACCARD - CLUSTER 0 (CONSOLIDATION) - DONNÉES {dataset_name}")
    print(f"{'=' * 120}{Style.RESET_ALL}")

    jaccard_matrix_c0 = calculate_jaccard_for_cluster(
        algo_dfs=algo_dfs,
        target_clusters=[0],
        dataset_name=dataset_name
    )

    # Stocker pour analyse ultérieure
    jaccard_cluster_results['cluster_0'][dataset_name] = jaccard_matrix_c0

    # Afficher avec le même format que les matrices globales
    display_jaccard_matrix(
        jaccard_matrix=jaccard_matrix_c0,
        threshold=JACCARD_THRESHOLD,
        algo_dfs=algo_dfs,
        min_common_trades=MIN_COMMON_TRADES
    )

    # ────────────────────────────────────────────────────────────────────────────
    # MATRICE JACCARD - CLUSTERS 0+1 (CONSOLIDATION + TRANSITION)
    # ────────────────────────────────────────────────────────────────────────────
    print(f"\n{Fore.BLUE}{'=' * 120}")
    print(f"MATRICE JACCARD - CLUSTERS 0+1 (CONSOLIDATION + TRANSITION) - DONNÉES {dataset_name}")
    print(f"{'=' * 120}{Style.RESET_ALL}")

    jaccard_matrix_c01 = calculate_jaccard_for_cluster(
        algo_dfs=algo_dfs,
        target_clusters=[0, 1],
        dataset_name=dataset_name
    )

    # Stocker pour analyse ultérieure
    jaccard_cluster_results['cluster_0_1'][dataset_name] = jaccard_matrix_c01

    # Afficher avec le même format
    display_jaccard_matrix(
        jaccard_matrix=jaccard_matrix_c01,
        threshold=JACCARD_THRESHOLD,
        algo_dfs=algo_dfs,
        min_common_trades=MIN_COMMON_TRADES
    )

# ────────────────────────────────────────────────────────────────────────────────
# ANALYSE COMPARATIVE DES REDONDANCES PAR CLUSTER
# ────────────────────────────────────────────────────────────────────────────────

print(f"\n{Fore.CYAN}{'=' * 120}")
print("ANALYSE COMPARATIVE DES REDONDANCES : GLOBAL vs CLUSTER 0 vs CLUSTERS 0+1")
print(f"{'=' * 120}{Style.RESET_ALL}")

# Comparer les paires redondantes entre global et clusters
for dataset_name in ["ENTRAINEMENT", "TEST", "VALIDATION 1", "VALIDATION 2", "UNSEEN"]:  # MODIFIÉ
    print(f"\n📊 DATASET {dataset_name}:")

    # Récupérer les paires redondantes pour chaque analyse
    global_redundant_list = redundant_pairs_by_dataset.get(dataset_name, [])
    global_redundant = set(global_redundant_list)  # Convertir en set

    if dataset_name in jaccard_cluster_results['cluster_0']:
        c0_matrix = jaccard_cluster_results['cluster_0'][dataset_name]
        c0_redundant_list = analyze_global_redundancy(c0_matrix, JACCARD_THRESHOLD)
        c0_redundant = set(c0_redundant_list)  # Convertir la liste en set
    else:
        c0_redundant = set()

    if dataset_name in jaccard_cluster_results['cluster_0_1']:
        c01_matrix = jaccard_cluster_results['cluster_0_1'][dataset_name]
        c01_redundant_list = analyze_global_redundancy(c01_matrix, JACCARD_THRESHOLD)
        c01_redundant = set(c01_redundant_list)  # Convertir la liste en set
    else:
        c01_redundant = set()

    # Analyser les différences
    print(f"  • Paires redondantes globalement : {len(global_redundant)}")
    print(f"  • Paires redondantes sur Cluster 0 : {len(c0_redundant)}")
    print(f"  • Paires redondantes sur Clusters 0+1 : {len(c01_redundant)}")

    # Paires qui deviennent redondantes seulement sur certains clusters
    only_c0 = c0_redundant - global_redundant
    only_c01 = c01_redundant - global_redundant

    if only_c0:
        print(f"  ⚠️  Paires redondantes UNIQUEMENT sur Cluster 0:")
        for pair in sorted(only_c0):
            print(f"     {pair[0]} ↔ {pair[1]}")

    if only_c01:
        print(f"  ⚠️  Paires redondantes UNIQUEMENT sur Clusters 0+1:")
        for pair in sorted(only_c01):
            print(f"     {pair[0]} ↔ {pair[1]}")

# ────────────────────────────────────────────────────────────────────────────────
# AJOUT D'UNE ANALYSE SPÉCIFIQUE UNSEEN vs autres datasets
# ────────────────────────────────────────────────────────────────────────────────
print(f"{Fore.YELLOW}\n{'=' * 120}")
print("AJOUT D'UNE ANALYSE SPÉCIFIQUE UNSEEN vs 4 autres datasets")
print(f"{'=' * 120}{Style.RESET_ALL}")


# Variables pour stocker le nombre de sessions par dataset
# À définir en amont selon vos données
dataset_session_counts = {
    'Train': Nb_session_train,
    'Test': Nb_session_test,
    'Val1': Nb_session_val1,
    'Val': Nb_session_val,
    'Unseen': Nb_session_unseen
}


def get_cluster_name(clusters_list):
    """Génère un nom lisible pour la liste de clusters"""
    if clusters_list == [0]:
        return "Cluster 0 (Consolidation)"
    elif clusters_list == [1]:
        return "Cluster 1 (Transition)"
    elif clusters_list == [2]:
        return "Cluster 2 (Breakout)"
    elif clusters_list == [0, 1]:
        return "Clusters 0+1 (Consolidation + Transition)"
    elif clusters_list == [0, 1, 2]:
        return "Tous Clusters"
    else:
        return f"Clusters {clusters_list}"


def analyze_unseen_performance():
    """
    Analyse spécifique des performances du dataset UNSEEN comparé aux autres
    AVEC FILTRAGE PAR CLUSTERS PERSONNALISABLE
    """
    print(f"\n{Fore.MAGENTA}🎯 ANALYSE SPÉCIFIQUE DATASET UNSEEN{Style.RESET_ALL}")
    print("=" * 120)

    cluster_name = get_cluster_name(CLUSTERS_UNSEEN_ANALYSIS)
    print(f"📊 ANALYSE FOCALISÉE SUR : {cluster_name}")
    print(f"📊 COMPARAISON DES PERFORMANCES UNSEEN vs AUTRES DATASETS:")

    # Fonction pour filtrer les données par clusters
    def get_cluster_filtered_data(algo_dfs, target_clusters):
        """Filtre les données d'algorithmes par clusters spécifiés"""
        filtered_dfs = {}

        for algo_name, df_algo in algo_dfs.items():
            if 'Cluster_G2' not in df_algo.columns or 'deltaCustomSessionIndex' not in df_algo.columns:
                continue

            # Déterminer la colonne PnL
            pnl_col = None
            for col in ['PnlAfterFiltering', 'trade_pnl']:
                if col in df_algo.columns:
                    pnl_col = col
                    break

            if pnl_col is None:
                continue

            # Filtrer par groupe 2 et clusters ciblés
            df_filtered = df_algo[
                (df_algo['deltaCustomSessionIndex'].isin(GROUPE_SESSION_2)) &
                (df_algo['Cluster_G2'].isin(target_clusters)) &
                (df_algo[pnl_col] != 0)
                ].copy()

            if len(df_filtered) > 0:
                filtered_dfs[algo_name] = df_filtered

        return filtered_dfs

    # Appliquer le filtrage par clusters à tous les datasets
    datasets_results_filtered = {}

    all_datasets_dfs = {
        'Train': algo_dfs_train,
        'Test': algo_dfs_test,
        'Val1': algo_dfs_val1,
        'Val': algo_dfs_val,
        'Unseen': algo_dfs_unseen
    }

    for dataset_name, algo_dfs in all_datasets_dfs.items():
        filtered_dfs = get_cluster_filtered_data(algo_dfs, CLUSTERS_UNSEEN_ANALYSIS)

        # Calculer les métriques sur les données filtrées
        filtered_results = {}
        for algo_name, df_filtered in filtered_dfs.items():
            pnl_col = 'PnlAfterFiltering' if 'PnlAfterFiltering' in df_filtered.columns else 'trade_pnl'

            trades_count = len(df_filtered)
            wins = (df_filtered[pnl_col] > 0).sum()
            winrate = (wins / trades_count * 100) if trades_count > 0 else 0
            net_pnl = df_filtered[pnl_col].sum()

            profits = df_filtered.loc[df_filtered[pnl_col] > 0, pnl_col].sum()
            losses = abs(df_filtered.loc[df_filtered[pnl_col] < 0, pnl_col].sum())
            profit_factor = profits / losses if losses > 0 else 0

            filtered_results[algo_name] = {
                'Nombre de trades': trades_count,
                'Win Rate (%)': winrate,
                'Net PnL': net_pnl,
                'Profit Factor': profit_factor
            }

        datasets_results_filtered[dataset_name] = filtered_results

    # Comparer les métriques moyennes avec filtrage par clusters
    for dataset_name, results in datasets_results_filtered.items():
        if not results:
            continue

        avg_trades = np.mean([metrics['Nombre de trades'] for metrics in results.values()])
        avg_winrate = np.mean([metrics['Win Rate (%)'] for metrics in results.values()])
        avg_pnl = np.mean([metrics['Net PnL'] for metrics in results.values()])
        avg_pf = np.mean([metrics['Profit Factor'] for metrics in results.values()])

        print(f"\n🔍 {dataset_name} (sur {cluster_name}):")
        print(f"   📊 Algorithmes actifs: {len(results)}")
        print(f"   📊 Trades moyens/algo: {avg_trades:.1f}")
        print(f"   📊 Win Rate moyen: {avg_winrate:.1f}%")
        print(f"   📊 PnL moyen: {avg_pnl:.2f}")
        print(f"   📊 Profit Factor moyen: {avg_pf:.2f}")

    # Analyser la robustesse des algorithmes sur UNSEEN avec filtrage par clusters
    print(f"\n🎯 ROBUSTESSE DES ALGORITHMES SUR UNSEEN ({cluster_name}):")

    unseen_results = datasets_results_filtered.get('Unseen', {})
    if not unseen_results:
        print("❌ Aucun algorithme actif sur UNSEEN avec les clusters sélectionnés")
        return

    best_on_unseen = []
    for algo_name, unseen_metrics in unseen_results.items():
        unseen_wr = unseen_metrics['Win Rate (%)']
        unseen_trades = unseen_metrics['Nombre de trades']

        # Calculer la performance moyenne sur les autres datasets (filtrés)
        other_wr = []
        other_trades = []

        for dataset_name, results in datasets_results_filtered.items():
            if dataset_name != 'Unseen' and algo_name in results:
                other_wr.append(results[algo_name]['Win Rate (%)'])
                other_trades.append(results[algo_name]['Nombre de trades'])

        if other_wr:
            avg_other_wr = np.mean(other_wr)
            avg_other_trades = np.mean(other_trades)

            # Calculer la rétention de performance
            retention_rate = (unseen_wr / avg_other_wr * 100) if avg_other_wr > 0 else 0

            best_on_unseen.append({
                'algo': algo_name,
                'unseen_wr': unseen_wr,
                'unseen_trades': unseen_trades,
                'avg_other_wr': avg_other_wr,
                'retention_rate': retention_rate
            })

    # Trier par performance sur UNSEEN
    best_on_unseen.sort(key=lambda x: x['unseen_wr'], reverse=True)

    print(f"\n🏆 TOP 10 ALGORITHMES SUR UNSEEN ({cluster_name}):")
    for i, algo_info in enumerate(best_on_unseen[:10]):
        print(f"   {i + 1:2d}. {algo_info['algo']}: {algo_info['unseen_wr']:.1f}% WR "
              f"({algo_info['unseen_trades']} trades) | "
              f"Rétention: {algo_info['retention_rate']:.0f}%")

    # Identifier les algorithmes les plus robustes
    robust_algos = [algo for algo in best_on_unseen
                    if algo['retention_rate'] > 80 and algo['unseen_wr'] > 55]

    if robust_algos:
        print(f"\n✅ ALGORITHMES LES PLUS ROBUSTES (>80% rétention, >55% WR):")
        for algo_info in robust_algos[:5]:
            print(f"   • {algo_info['algo']}: {algo_info['unseen_wr']:.1f}% WR, "
                  f"{algo_info['retention_rate']:.0f}% rétention")

    return datasets_results_filtered


def create_voting_synthesis_tables():
    """
    Crée les tableaux de synthèse pour le voting sur 2 et 3 algorithmes
    EN UTILISANT LES CLUSTERS PERSONNALISÉS AVEC STATISTIQUES PAR SESSION
    """
    print(f"\n{Fore.CYAN}🗳️ TABLEAUX DE SYNTHÈSE - VOTING D'ALGORITHMES{Style.RESET_ALL}")
    print("=" * 140)

    cluster_name = get_cluster_name(CLUSTERS_UNSEEN_ANALYSIS)
    print(f"📊 ANALYSE DE VOTING SUR : {cluster_name}")
    print("🎯 Trades pris uniquement quand plusieurs algorithmes sont d'accord")

    def analyze_voting_consensus(algo_dfs, dataset_name, min_consensus):
        """Analyse le consensus de voting pour un dataset donné avec stats par session"""

        # Colonnes d'indicateurs pour identifier les trades uniques
        indicator_columns = [
            'rsi_', 'macd', 'macd_signal', 'macd_hist',
            'timeElapsed2LastBar', 'timeStampOpening',
            'ratio_deltaRevZone_volCandle'
        ]

        # Collecter tous les trades des algorithmes avec filtrage par clusters
        all_trades_data = {}

        for algo_name, df_algo in algo_dfs.items():
            if 'Cluster_G2' not in df_algo.columns or 'deltaCustomSessionIndex' not in df_algo.columns:
                continue

            # Déterminer la colonne PnL
            pnl_col = None
            for col in ['PnlAfterFiltering', 'trade_pnl']:
                if col in df_algo.columns:
                    pnl_col = col
                    break

            if pnl_col is None:
                continue

            # Filtrer par groupe 2 et clusters ciblés
            df_filtered = df_algo[
                (df_algo['deltaCustomSessionIndex'].isin(GROUPE_SESSION_2)) &
                (df_algo['Cluster_G2'].isin(CLUSTERS_UNSEEN_ANALYSIS)) &
                (df_algo[pnl_col] != 0)
                ].copy()

            if len(df_filtered) == 0:
                continue

            # Créer l'ensemble des trades uniques
            valid_cols = [c for c in indicator_columns if c in df_filtered.columns]
            if not valid_cols:
                continue

            for _, row in df_filtered.iterrows():
                trade_key = tuple(row[col] for col in valid_cols)

                if trade_key not in all_trades_data:
                    all_trades_data[trade_key] = {
                        'pnl': row[pnl_col],
                        'is_winning': row[pnl_col] > 0,
                        'session_id': row['deltaCustomSessionIndex'],
                        'algos_signals': set()
                    }

                all_trades_data[trade_key]['algos_signals'].add(algo_name)

        # Filtrer les trades avec consensus minimum
        consensus_trades = [
            trade for trade in all_trades_data.values()
            if len(trade['algos_signals']) >= min_consensus
        ]

        if not consensus_trades:
            return {
                'nb_trades': 0,
                'nb_wins': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'avg_pnl_per_trade': 0.0,
                'profit_factor': 0.0,
                'nb_sessions': 0,
                'trades_per_session': 0.0,
                'wins_per_session': 0.0
            }

        # Calculer les métriques de base
        nb_trades = len(consensus_trades)
        nb_wins = sum(1 for trade in consensus_trades if trade['is_winning'])
        win_rate = (nb_wins / nb_trades * 100) if nb_trades > 0 else 0.0

        total_pnl = sum(trade['pnl'] for trade in consensus_trades)
        avg_pnl_per_trade = total_pnl / nb_trades if nb_trades > 0 else 0.0

        # Calculer profit factor
        profits = sum(trade['pnl'] for trade in consensus_trades if trade['pnl'] > 0)
        losses = abs(sum(trade['pnl'] for trade in consensus_trades if trade['pnl'] <= 0))
        profit_factor = profits / losses if losses > 0 else 0.0

        # Calculer les statistiques par session
        unique_sessions = set(trade['session_id'] for trade in consensus_trades)
        nb_sessions = len(unique_sessions)

        # Obtenir le nombre total de sessions pour ce dataset
        total_sessions = dataset_session_counts.get(dataset_name, nb_sessions)

        trades_per_session = nb_trades / total_sessions if total_sessions > 0 else 0.0
        wins_per_session = nb_wins / total_sessions if total_sessions > 0 else 0.0

        return {
            'nb_trades': nb_trades,
            'nb_wins': nb_wins,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl_per_trade': avg_pnl_per_trade,
            'profit_factor': profit_factor,
            'nb_sessions': total_sessions,
            'trades_per_session': trades_per_session,
            'wins_per_session': wins_per_session
        }

    # Analyser tous les datasets
    all_datasets_dfs = {
        'Train': algo_dfs_train,
        'Test': algo_dfs_test,
        'Val1': algo_dfs_val1,
        'Val': algo_dfs_val,
        'Unseen': algo_dfs_unseen
    }

    # ────────────────────────────────────────────────────────────────────────────
    # TABLEAU 1 : VOTING AVEC 2 ALGORITHMES MINIMUM
    # ────────────────────────────────────────────────────────────────────────────

    print(f"\n📊 TABLEAU 1 : TRADES AVEC ACCORD DE 2+ ALGORITHMES ({cluster_name})")
    print("=" * 140)

    voting_2_results = []
    for dataset_name, algo_dfs in all_datasets_dfs.items():
        metrics = analyze_voting_consensus(algo_dfs, dataset_name, min_consensus=2)
        voting_2_results.append({
            'Dataset': dataset_name,
            'Nb Sessions': metrics['nb_sessions'],
            'Nb Trades': metrics['nb_trades'],
            'Nb Wins': metrics['nb_wins'],
            'Win Rate (%)': round(metrics['win_rate'], 1),
            'Total PnL': round(metrics['total_pnl'], 2),
            'PnL/Trade': round(metrics['avg_pnl_per_trade'], 2),
            'Profit Factor': round(metrics['profit_factor'], 2),
            'Trades/Session': round(metrics['trades_per_session'], 1),
            'Wins/Session': round(metrics['wins_per_session'], 1)
        })

    voting_2_df = pd.DataFrame(voting_2_results)
    print(voting_2_df.to_string(index=False))

    # ────────────────────────────────────────────────────────────────────────────
    # TABLEAU 2 : VOTING AVEC 3 ALGORITHMES MINIMUM
    # ────────────────────────────────────────────────────────────────────────────

    print(f"\n📊 TABLEAU 2 : TRADES AVEC ACCORD DE 3+ ALGORITHMES ({cluster_name})")
    print("=" * 140)

    voting_3_results = []
    for dataset_name, algo_dfs in all_datasets_dfs.items():
        metrics = analyze_voting_consensus(algo_dfs, dataset_name, min_consensus=3)
        voting_3_results.append({
            'Dataset': dataset_name,
            'Nb Sessions': metrics['nb_sessions'],
            'Nb Trades': metrics['nb_trades'],
            'Nb Wins': metrics['nb_wins'],
            'Win Rate (%)': round(metrics['win_rate'], 1),
            'Total PnL': round(metrics['total_pnl'], 2),
            'PnL/Trade': round(metrics['avg_pnl_per_trade'], 2),
            'Profit Factor': round(metrics['profit_factor'], 2),
            'Trades/Session': round(metrics['trades_per_session'], 1),
            'Wins/Session': round(metrics['wins_per_session'], 1)
        })

    voting_3_df = pd.DataFrame(voting_3_results)
    print(voting_3_df.to_string(index=False))

    # ────────────────────────────────────────────────────────────────────────────
    # ANALYSE COMPARATIVE DES STRATÉGIES DE VOTING
    # ────────────────────────────────────────────────────────────────────────────

    print(f"\n💡 ANALYSE COMPARATIVE - STRATÉGIES DE VOTING ({cluster_name})")
    print("=" * 140)

    # Comparer l'efficacité des deux stratégies
    for i, dataset_name in enumerate(['Train', 'Test', 'Val1', 'Val', 'Unseen']):
        voting_2_metrics = voting_2_results[i]
        voting_3_metrics = voting_3_results[i]

        if voting_2_metrics['Nb Trades'] > 0 and voting_3_metrics['Nb Trades'] > 0:
            print(f"\n🔍 {dataset_name}:")
            print(f"   2+ algos: {voting_2_metrics['Nb Trades']} trades, "
                  f"{voting_2_metrics['Win Rate (%)']}% WR, "
                  f"PnL/trade: {voting_2_metrics['PnL/Trade']}, "
                  f"{voting_2_metrics['Trades/Session']} trades/session")
            print(f"   3+ algos: {voting_3_metrics['Nb Trades']} trades, "
                  f"{voting_3_metrics['Win Rate (%)']}% WR, "
                  f"PnL/trade: {voting_3_metrics['PnL/Trade']}, "
                  f"{voting_3_metrics['Trades/Session']} trades/session")

            # Calculer l'amélioration de qualité
            wr_improvement = voting_3_metrics['Win Rate (%)'] - voting_2_metrics['Win Rate (%)']
            pnl_improvement = voting_3_metrics['PnL/Trade'] - voting_2_metrics['PnL/Trade']

            print(f"   📈 Amélioration 3+ vs 2+: WR {wr_improvement:+.1f}pp, "
                  f"PnL/trade {pnl_improvement:+.2f}")

        elif voting_2_metrics['Nb Trades'] > 0:
            print(f"\n🔍 {dataset_name}: Seulement voting 2+ disponible "
                  f"({voting_2_metrics['Nb Trades']} trades)")
        else:
            print(f"\n🔍 {dataset_name}: Aucun consensus détecté")

    # Recommandations finales
    print(f"\n🎯 RECOMMANDATIONS VOTING ({cluster_name}):")

    # Analyser UNSEEN spécifiquement
    unseen_2 = next(r for r in voting_2_results if r['Dataset'] == 'Unseen')
    unseen_3 = next(r for r in voting_3_results if r['Dataset'] == 'Unseen')

    if unseen_2['Nb Trades'] > 0:
        print(f"   • UNSEEN - Voting 2+: {unseen_2['Win Rate (%)']}% WR sur {unseen_2['Nb Trades']} trades "
              f"({unseen_2['Trades/Session']} trades/session)")

        if unseen_3['Nb Trades'] > 0:
            print(f"   • UNSEEN - Voting 3+: {unseen_3['Win Rate (%)']}% WR sur {unseen_3['Nb Trades']} trades "
                  f"({unseen_3['Trades/Session']} trades/session)")

            if unseen_3['Win Rate (%)'] > unseen_2['Win Rate (%)']:
                print(f"   ✅ Recommandation: Privilégier voting 3+ pour meilleure qualité")
            else:
                print(f"   ⚖️  Recommandation: Voting 2+ offre plus d'opportunités")
        else:
            print(f"   ⚠️  Voting 3+ insuffisant, utiliser voting 2+")
    else:
        print(f"   ❌ Aucun consensus détecté sur UNSEEN avec {cluster_name}")

    return voting_2_df, voting_3_df


def create_unseen_detailed_analysis():
    """
    Analyse détaillée spécifique au dataset UNSEEN avec focus sur la généralisation
    """
    print(f"\n{Fore.CYAN}📈 ANALYSE DÉTAILLÉE DE GÉNÉRALISATION - DATASET UNSEEN{Style.RESET_ALL}")
    print("=" * 120)

    # Analyser les algorithmes qui surperforment ou sous-performent sur UNSEEN
    performance_comparison = []

    for algo_name in algorithms.keys():
        if algo_name not in results_unseen:
            continue

        unseen_metrics = results_unseen[algo_name]

        # Calculer les moyennes sur les datasets de validation (Val1 + Val)
        val_metrics = []
        if algo_name in results_val1:
            val_metrics.append(results_val1[algo_name])
        if algo_name in results_val:
            val_metrics.append(results_val[algo_name])

        if not val_metrics:
            continue

        avg_val_wr = np.mean([m['Win Rate (%)'] for m in val_metrics])
        avg_val_pnl = np.mean([m['Net PnL'] for m in val_metrics])
        avg_val_trades = np.mean([m['Nombre de trades'] for m in val_metrics])

        # Calculer les écarts
        wr_diff = unseen_metrics['Win Rate (%)'] - avg_val_wr
        pnl_diff = unseen_metrics['Net PnL'] - avg_val_pnl
        trades_diff = unseen_metrics['Nombre de trades'] - avg_val_trades

        performance_comparison.append({
            'algo': algo_name,
            'unseen_wr': unseen_metrics['Win Rate (%)'],
            'avg_val_wr': avg_val_wr,
            'wr_diff': wr_diff,
            'unseen_pnl': unseen_metrics['Net PnL'],
            'avg_val_pnl': avg_val_pnl,
            'pnl_diff': pnl_diff,
            'unseen_trades': unseen_metrics['Nombre de trades'],
            'trades_diff': trades_diff
        })

    # Trier par différence de WinRate
    performance_comparison.sort(key=lambda x: x['wr_diff'], reverse=True)

    print(f"\n🚀 ALGORITHMES QUI S'AMÉLIORENT SUR UNSEEN (vs validation):")
    improving_algos = [algo for algo in performance_comparison if algo['wr_diff'] > 5]
    for algo_info in improving_algos[:5]:
        print(f"   • {algo_info['algo']}: {algo_info['wr_diff']:+.1f}pp "
              f"({algo_info['avg_val_wr']:.1f}% → {algo_info['unseen_wr']:.1f}%)")

    print(f"\n📉 ALGORITHMES QUI SE DÉGRADENT SUR UNSEEN (vs validation):")
    declining_algos = [algo for algo in performance_comparison if algo['wr_diff'] < -5]
    declining_algos.sort(key=lambda x: x['wr_diff'])
    for algo_info in declining_algos[:5]:
        print(f"   • {algo_info['algo']}: {algo_info['wr_diff']:+.1f}pp "
              f"({algo_info['avg_val_wr']:.1f}% → {algo_info['unseen_wr']:.1f}%)")

    print(f"\n⚖️  ALGORITHMES STABLES SUR UNSEEN (±5pp):")
    stable_algos = [algo for algo in performance_comparison
                    if -5 <= algo['wr_diff'] <= 5 and algo['unseen_wr'] > 50]
    stable_algos.sort(key=lambda x: x['unseen_wr'], reverse=True)
    for algo_info in stable_algos[:5]:
        print(f"   • {algo_info['algo']}: {algo_info['wr_diff']:+.1f}pp "
              f"(WR: {algo_info['unseen_wr']:.1f}%)")


# ────────────────────────────────────────────────────────────────────────────────
# VÉRIFICATION ET CORRECTION DU CHARGEMENT DES DATASETS
# ────────────────────────────────────────────────────────────────────────────────

# Vérification des tailles de datasets
print(f"\n📊 TAILLES DES DATASETS APRÈS CHARGEMENT:")
print(f"Train: {len(df_analysis_train)} trades")
print(f"Test: {len(df_analysis_test)} trades")
print(f"Val1: {len(df_analysis_val1)} trades")
print(f"Val: {len(df_analysis_val)} trades")
print(f"Unseen: {len(df_analysis_unseen)} trades")

# Affichage du résumé des sessions
print("\nRésumé des sessions chargées :")
print(f"TRAIN  : {Nb_session_train} sessions")
print(f"TEST   : {Nb_session_test} sessions")
print(f"VAL1   : {Nb_session_val1} sessions")
print(f"VAL    : {Nb_session_val} sessions")
print(f"UNSEEN : {Nb_session_unseen} sessions")


# Vérification des dates pour s'assurer que les datasets sont différents
def check_dataset_periods():
    """Vérifie les périodes temporelles des datasets"""

    datasets_info = [
        ("Train", df_analysis_train),
        ("Test", df_analysis_test),
        ("Val1", df_analysis_val1),
        ("Val", df_analysis_val),
        ("Unseen", df_analysis_unseen)
    ]

    print(f"\n📅 VÉRIFICATION DES PÉRIODES TEMPORELLES:")

    for name, df in datasets_info:
        if 'timeStampOpening' in df.columns and len(df) > 0:
            min_date = df['timeStampOpening'].min()
            max_date = df['timeStampOpening'].max()
            print(f"{name}: {min_date} → {max_date} ({len(df)} trades)")
        else:
            print(f"{name}: Pas de colonne timeStampOpening ou dataset vide")


check_dataset_periods()


def create_5_datasets_summary_table():
    """
    Crée un tableau de synthèse pour les 5 datasets avec métriques clés
    AVEC VÉRIFICATION DE LA COHÉRENCE DES DONNÉES ET STATISTIQUES PAR SESSION
    """
    print(f"\n{Fore.GREEN}📋 TABLEAU DE SYNTHÈSE - 5 DATASETS{Style.RESET_ALL}")
    print("=" * 160)

    datasets_results = {
        'Train': results_train,
        'Test': results_test,
        'Val1': results_val1,
        'Val': results_val,
        'Unseen': results_unseen
    }

    summary_data = []

    for dataset_name, results in datasets_results.items():
        if not results:
            continue

        # Calculer les statistiques globales du dataset
        total_trades = sum([metrics['Nombre de trades'] for metrics in results.values()])
        total_wins = sum([int(metrics['Win Rate (%)'] * metrics['Nombre de trades'] / 100)
                          for metrics in results.values()])
        total_pnl = sum([metrics['Net PnL'] for metrics in results.values()])
        avg_winrate = np.mean([metrics['Win Rate (%)'] for metrics in results.values()])
        avg_pf = np.mean([metrics['Profit Factor'] for metrics in results.values()])

        # Obtenir le nombre de sessions pour ce dataset
        nb_sessions = dataset_session_counts.get(dataset_name, 0)

        # Calculer les statistiques par session
        trades_per_session = total_trades / nb_sessions if nb_sessions > 0 else 0
        wins_per_session = total_wins / nb_sessions if nb_sessions > 0 else 0

        # Identifier le meilleur algorithme
        best_algo = max(results.items(), key=lambda x: x[1]['Win Rate (%)'])
        best_algo_name = best_algo[0]
        best_algo_wr = best_algo[1]['Win Rate (%)']

        # Compter les algorithmes avec WR > 55%
        good_algos = sum(1 for metrics in results.values() if metrics['Win Rate (%)'] > 55)

        summary_data.append({
            'Dataset': dataset_name,
            'Nb Sessions': nb_sessions,
            'Nb Algos Actifs': len(results),
            'Total Trades': total_trades,
            'Total Wins': total_wins,
            'PnL Total': round(total_pnl, 2),
            'WR Moyen (%)': round(avg_winrate, 1),
            'PF Moyen': round(avg_pf, 2),
            'Trades/Session': round(trades_per_session, 1),
            'Wins/Session': round(wins_per_session, 1),
            'Algos >55% WR': good_algos,
            'Meilleur Algo': best_algo_name,
            'Meilleur WR (%)': round(best_algo_wr, 1)
        })

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

    # ⚠️ VÉRIFICATION DE COHÉRENCE
    print(f"\n🔍 VÉRIFICATION DE COHÉRENCE DES DONNÉES:")

    # Vérifier si VAL et UNSEEN sont identiques
    val_row = next((row for row in summary_data if row['Dataset'] == 'Val'), None)
    unseen_row = next((row for row in summary_data if row['Dataset'] == 'Unseen'), None)

    if val_row and unseen_row:
        if (val_row['Total Trades'] == unseen_row['Total Trades'] and
                val_row['PnL Total'] == unseen_row['PnL Total']):
            print(f"⚠️ ALERTE: Les datasets VAL et UNSEEN semblent identiques!")
            print(f"   Val: {val_row['Total Trades']} trades, PnL: {val_row['PnL Total']}")
            print(f"   Unseen: {unseen_row['Total Trades']} trades, PnL: {unseen_row['PnL Total']}")
            print(f"💡 Vérifiez que FILE_NAME_UNSEEN pointe vers un fichier différent")
        else:
            print(f"✅ Les datasets VAL et UNSEEN sont bien différents")

    # Analyse comparative (seulement si les datasets sont différents)
    if val_row and unseen_row and val_row['Total Trades'] != unseen_row['Total Trades']:
        print(f"\n💡 INSIGHTS COMPARATIFS:")

        # Comparer UNSEEN avec les autres
        other_rows = [row for row in summary_data if row['Dataset'] != 'Unseen']

        avg_other_wr = np.mean([row['WR Moyen (%)'] for row in other_rows])
        avg_other_pnl = np.mean([row['PnL Total'] for row in other_rows])
        avg_other_trades_per_session = np.mean([row['Trades/Session'] for row in other_rows])
        avg_other_wins_per_session = np.mean([row['Wins/Session'] for row in other_rows])

        print(f"   🎯 UNSEEN vs Autres datasets:")
        print(f"      WR: {unseen_row['WR Moyen (%)']}% vs {avg_other_wr:.1f}% "
              f"({unseen_row['WR Moyen (%)'] - avg_other_wr:+.1f}pp)")
        print(f"      PnL: {unseen_row['PnL Total']} vs {avg_other_pnl:.2f} "
              f"({unseen_row['PnL Total'] - avg_other_pnl:+.2f})")
        print(f"      Trades/Session: {unseen_row['Trades/Session']:.1f} vs {avg_other_trades_per_session:.1f} "
              f"({unseen_row['Trades/Session'] - avg_other_trades_per_session:+.1f})")
        print(f"      Wins/Session: {unseen_row['Wins/Session']:.1f} vs {avg_other_wins_per_session:.1f} "
              f"({unseen_row['Wins/Session'] - avg_other_wins_per_session:+.1f})")

        # Performance relative (RÉTENTION)
        wr_retention = (unseen_row['WR Moyen (%)'] / avg_other_wr * 100) if avg_other_wr > 0 else 0
        print(f"      🎯 RÉTENTION DE PERFORMANCE: {wr_retention:.1f}%")

        if wr_retention > 90:
            print(f"      ✅ Excellente généralisation (>90%)")
        elif wr_retention > 75:
            print(f"      ⚠️  Bonne généralisation (75-90%)")
        else:
            print(f"      ❌ Généralisation limitée (<75%)")

        print(f"\n📖 EXPLICATION DE LA RÉTENTION:")
        print(f"   La rétention mesure à quel point les algorithmes maintiennent")
        print(f"   leur performance sur des données non vues (UNSEEN) par rapport")
        print(f"   à leur performance moyenne sur les autres datasets.")
        print(f"   • >90% = Excellente robustesse (peu de dégradation)")
        print(f"   • 75-90% = Bonne robustesse (dégradation acceptable)")
        print(f"   • <75% = Robustesse limitée (risque d'overfitting)")

    # Analyse des patterns par session
    print(f"\n📊 ANALYSE DES PATTERNS PAR SESSION:")

    for row in summary_data:
        dataset_name = row['Dataset']
        if row['Nb Sessions'] > 0:
            print(f"   {dataset_name}: {row['Trades/Session']:.1f} trades/session, "
                  f"{row['Wins/Session']:.1f} wins/session "
                  f"(WR session: {row['Wins/Session'] / row['Trades/Session'] * 100:.1f}%)")
        else:
            print(f"   {dataset_name}: Pas de données de session disponibles")


def analyze_unseen_vs_val_detailed():
    """
    Analyse détaillée pour comparer VAL et UNSEEN et détecter les doublons
    AVEC STATISTIQUES PAR SESSION
    """
    print(f"\n{Fore.CYAN}🔍 ANALYSE DÉTAILLÉE VAL vs UNSEEN{Style.RESET_ALL}")
    print("=" * 140)

    # Comparer algorithme par algorithme
    common_algos = set(results_val.keys()).intersection(set(results_unseen.keys()))

    if not common_algos:
        print("❌ Aucun algorithme commun entre VAL et UNSEEN")
        return

    print(f"📊 COMPARAISON ALGORITHME PAR ALGORITHME ({len(common_algos)} algos communs):")
    print(f"{'Algorithme':<25} {'VAL WR':<10} {'UNSEEN WR':<12} {'Diff':<8} {'Rétention':<10} {'Status':<15}")
    print("-" * 100)

    identical_count = 0
    retention_scores = []

    for algo in sorted(common_algos):
        val_wr = results_val[algo]['Win Rate (%)']
        unseen_wr = results_unseen[algo]['Win Rate (%)']
        val_trades = results_val[algo]['Nombre de trades']
        unseen_trades = results_unseen[algo]['Nombre de trades']

        diff = unseen_wr - val_wr
        retention = (unseen_wr / val_wr * 100) if val_wr > 0 else 0
        retention_scores.append(retention)

        # Détecter les algorithmes avec performances identiques (suspect)
        if abs(diff) < 0.1 and val_trades == unseen_trades:
            identical_count += 1
            status = "🔴 IDENTIQUE"
        elif abs(diff) < 2.0:
            status = "🟡 TRÈS PROCHE"
        else:
            status = "✅ DIFFÉRENT"

        print(f"{algo:<25} {val_wr:>7.1f}% {unseen_wr:>9.1f}% {diff:>+6.1f}pp {retention:>7.1f}% {status}")

    if identical_count > len(common_algos) * 0.5:  # Plus de 50% identiques
        print(f"\n⚠️ ALERTE: {identical_count}/{len(common_algos)} algorithmes ont des performances identiques!")
        print(f"   Cela suggère fortement que VAL et UNSEEN utilisent les mêmes données")

    # Statistiques de rétention
    avg_retention = np.mean(retention_scores)
    print(f"\n📊 STATISTIQUES DE RÉTENTION:")
    print(f"   Rétention moyenne: {avg_retention:.1f}%")
    print(f"   Rétention min: {min(retention_scores):.1f}%")
    print(f"   Rétention max: {max(retention_scores):.1f}%")
    print(f"   Écart-type: {np.std(retention_scores):.1f}%")

    # Comparer les statistiques par session
    val_sessions = dataset_session_counts.get('Val', 0)
    unseen_sessions = dataset_session_counts.get('Unseen', 0)

    if val_sessions > 0 and unseen_sessions > 0:
        print(f"\n📊 COMPARAISON DES SESSIONS:")
        print(f"   VAL: {val_sessions} sessions")
        print(f"   UNSEEN: {unseen_sessions} sessions")

        if val_sessions == unseen_sessions:
            print(f"   ⚠️ Même nombre de sessions - Vérifiez que les datasets sont différents")
        else:
            print(f"   ✅ Nombres de sessions différents - Datasets probablement distincts")


# ────────────────────────────────────────────────────────────────────────────────
# AJOUT D'UNE ANALYSE DE CORRÉLATION INTER-DATASETS
# ────────────────────────────────────────────────────────────────────────────────

def analyze_inter_dataset_correlation():
    """
    Analyse la corrélation des performances d'algorithmes entre datasets
    """
    print(f"\n{Fore.YELLOW}🔗 ANALYSE DE CORRÉLATION INTER-DATASETS{Style.RESET_ALL}")
    print("=" * 120)

    # Créer une matrice de corrélation des Win Rates
    datasets_results = {
        'Train': results_train,
        'Test': results_test,
        'Val1': results_val1,
        'Val': results_val,
        'Unseen': results_unseen
    }

    # Récupérer tous les algorithmes communs
    common_algos = set.intersection(*[set(results.keys()) for results in datasets_results.values()])

    if len(common_algos) < 3:
        print("❌ Pas assez d'algorithmes communs pour l'analyse de corrélation")
        return

    # Créer la matrice des Win Rates
    correlation_data = {}
    for dataset_name, results in datasets_results.items():
        correlation_data[dataset_name] = [results[algo]['Win Rate (%)'] for algo in sorted(common_algos)]

    correlation_df = pd.DataFrame(correlation_data, index=sorted(common_algos))

    # Calculer la matrice de corrélation
    corr_matrix = correlation_df.corr()

    print(f"📊 MATRICE DE CORRÉLATION DES WIN RATES ({len(common_algos)} algorithmes communs):")
    print(corr_matrix.round(3).to_string())

    # Identifier les corrélations les plus fortes
    print(f"\n🔗 CORRÉLATIONS LES PLUS FORTES:")
    correlations = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            dataset1 = corr_matrix.columns[i]
            dataset2 = corr_matrix.columns[j]
            corr_value = corr_matrix.iloc[i, j]
            correlations.append((dataset1, dataset2, corr_value))

    correlations.sort(key=lambda x: abs(x[2]), reverse=True)

    for dataset1, dataset2, corr_value in correlations:
        if abs(corr_value) > 0.5:  # Seuil de corrélation significative
            direction = "Forte" if corr_value > 0.7 else "Modérée"
            print(f"   • {dataset1} ↔ {dataset2}: {corr_value:.3f} ({direction})")

    # Analyser la prédictibilité d'UNSEEN
    if 'Unseen' in correlation_df.columns:
        print(f"\n🎯 PRÉDICTIBILITÉ D'UNSEEN:")
        unseen_corrs = corr_matrix['Unseen'].drop('Unseen').sort_values(ascending=False)

        print(f"   Corrélations avec UNSEEN:")
        for dataset, corr_val in unseen_corrs.items():
            if abs(corr_val) > 0.3:  # Seuil minimal de prédictibilité
                predictability = "Élevée" if abs(corr_val) > 0.7 else "Modérée"
                print(f"   • {dataset}: {corr_val:.3f} ({predictability})")

        best_predictor = unseen_corrs.idxmax()
        best_corr = unseen_corrs.max()
        print(f"\n   🎯 Meilleur prédicteur d'UNSEEN: {best_predictor} (r={best_corr:.3f})")


def create_final_recommendations():
    """
    Crée des recommandations finales basées sur l'analyse complète des 5 datasets
    AVEC INSIGHTS SUR LES SESSIONS
    """
    print(f"\n{Fore.MAGENTA}🎯 RECOMMANDATIONS FINALES - ANALYSE 5 DATASETS{Style.RESET_ALL}")
    print("=" * 120)

    print("📋 SYNTHÈSE ET RECOMMANDATIONS:")

    # 1. Algorithmes les plus robustes
    print(f"\n1️⃣  ALGORITHMES RECOMMANDÉS (robustes sur tous datasets):")

    robust_scores = {}
    datasets_results = {
        'Train': results_train,
        'Test': results_test,
        'Val1': results_val1,
        'Val': results_val,
        'Unseen': results_unseen
    }

    for algo_name in algorithms.keys():
        scores = []
        for dataset_name, results in datasets_results.items():
            if algo_name in results:
                wr = results[algo_name]['Win Rate (%)']
                trades = results[algo_name]['Nombre de trades']
                # Score pondéré par le nombre de trades (minimum 10 trades)
                if trades >= 10:
                    scores.append(wr)

        if len(scores) >= 4:  # Présent sur au moins 4 datasets
            robust_scores[algo_name] = {
                'avg_wr': np.mean(scores),
                'min_wr': min(scores),
                'std_wr': np.std(scores),
                'consistency': 1 / (1 + np.std(scores)),  # Plus c'est stable, plus le score est élevé
                'datasets_count': len(scores)
            }

    # Trier par consistance et performance
    top_robust = sorted(robust_scores.items(),
                        key=lambda x: (x[1]['consistency'], x[1]['avg_wr']),
                        reverse=True)

    for i, (algo_name, scores) in enumerate(top_robust[:5]):
        print(f"   {i + 1}. {algo_name}: {scores['avg_wr']:.1f}% WR moyen, "
              f"σ={scores['std_wr']:.1f}pp, sur {scores['datasets_count']} datasets")

    # 2. Stratégies de diversification
    print(f"\n2️⃣  STRATÉGIES DE DIVERSIFICATION:")
    print("   • Utiliser les algorithmes les moins corrélés (voir matrice Jaccard)")
    print("   • Privilégier les algorithmes performants sur différents clusters")
    print("   • Éviter les paires constamment redondantes")

    # 3. Gestion des périodes et sessions
    print(f"\n3️⃣  OPTIMISATION TEMPORELLE ET PAR SESSION:")
    print("   • Adapter les algorithmes selon les sessions intraday (Groupe 1 vs Groupe 2)")
    print("   • Utiliser les insights par clusters de marché")
    print("   • Surveiller la dérive temporelle (Train→Test→Val→Unseen)")
    print("   • Optimiser le nombre de trades par session selon le dataset cible")

    # Analyser les patterns de trades par session
    total_sessions = sum(dataset_session_counts.values())
    if total_sessions > 0:
        print(f"\n📊 INSIGHTS SUR LES SESSIONS:")
        print(f"   • Total sessions analysées: {total_sessions}")

        for dataset_name, nb_sessions in dataset_session_counts.items():
            if nb_sessions > 0:
                pct_sessions = (nb_sessions / total_sessions) * 100
                print(f"   • {dataset_name}: {nb_sessions} sessions ({pct_sessions:.1f}%)")

    # 4. Monitoring et maintenance
    print(f"\n4️⃣  MONITORING ET MAINTENANCE:")
    print("   • Surveiller la rétention de performance sur nouvelles données")
    print("   • Recalibrer si la corrélation Train/Production < 0.7")
    print("   • Analyser les nouveaux patterns de redondance")
    print("   • Monitorer les trades/session et wins/session en temps réel")

    print(f"\n✅ ANALYSE TERMINÉE - 5 DATASETS INTÉGRÉS AVEC SUCCÈS")
    print(f"📈 AVEC STATISTIQUES DÉTAILLÉES PAR SESSION")


# ────────────────────────────────────────────────────────────────────────────────
# APPELS DES NOUVELLES FONCTIONS D'ANALYSE UNSEEN AVEC VOTING ET STATS SESSIONS
# ────────────────────────────────────────────────────────────────────────────────

# Lancer l'analyse personnalisée par clusters
datasets_results_filtered = analyze_unseen_performance()

# Créer les tableaux de voting avec stats par session
voting_2_df, voting_3_df = create_voting_synthesis_tables()

# Continuer avec les autres analyses
create_unseen_detailed_analysis()
create_5_datasets_summary_table()
analyze_unseen_vs_val_detailed()

# Appeler les nouvelles fonctions d'analyse
analyze_inter_dataset_correlation()
create_final_recommendations()


print("=" * 120)
print(f"\n{Fore.YELLOW}📊 RAPPEL DES PARAMETRES UTILISES {Style.RESET_ALL}")
print("=" * 120)

print(f"• clustering_with_K : {clustering_with_K}")
print(f"• GROUPE_SESSION_1  : {', '.join(map(str, GROUPE_SESSION_1))}")
print(f"• GROUPE_SESSION_2  : {', '.join(map(str, GROUPE_SESSION_2))}")

# Affichage des features, une par ligne
print("\n• Features utilisées pour le clustering :")
for col in get_feature_columns():
    print(f"  - {col}")

# ════════════════════════════════════════════════════════════════════════════════
# GÉNÉRATION C++ AUTOMATIQUE - VERSION INLINE
# ════════════════════════════════════════════════════════════════════════════════

# Imports nécessaires pour la génération
import re
from typing import Dict, List, Any
from datetime import datetime
from pathlib import Path

# Mapping des noms de features Python vers C++
FEATURE_MAPPING = {
    'sc_meanVol_perTick_over1': 'meanVol_perTick_over1',
    'sc_volRev_perTick_Vol_perTick_over1':'volRev_perTick_Vol_perTick_over1',
    'sc_volRev_perTick_volxTicksContZone_perTick_ratio': 'volRev_perTick_volxTicksContZone_perTick_ratio',
    'sc_is_antiEpuisement_long':'is_antiEpuisement_long',
    'sc_volume_perTick': 'volume_perTick',
    'sc_deltaRev_volRev_ratio':'deltaRev_volRev_ratio',
    'sc_volCandleMeanOver5Ratio': 'volCandleMeanOver5Ratio',
    'sc_candleDuration': 'candleDuration',
    'sc_reg_slope_30P_2': 'slope_30_candle_2',
    'sc_reg_std_30P_2': 'std_slope_30_candle_2',
    'sc_reg_slope_5P_2': 'slope_5_candle_2',
    'sc_reg_slope_10P_2': 'slope_10_candle_2',
    'sc_reg_slope_15P_2': 'slope_15_candle_2',
    'sc_reg_std_5P_2': 'std_slope_5_candle_2',
    'sc_diffVolDelta_1_1Ratio': 'diffVolDelta_1_1Ratio',
    'sc_close_sma_zscore_14': 'sc_close_sma_zscore_14',
    'sc_close_sma_zscore_21': 'sc_close_sma_zscore_21',
    'sc_ratio_delta_vol_VA11P': 'ratio_delta_vol_VA11P',
    'sc_is_wr_overbought_short': 'is_wr_overbought_short',
    'sc_is_wr_oversold_long': 'is_wr_oversold_long',
    'sc_volPocVolRevesalXContRatio': 'volPocVolRevesalXContRatio',
    'sc_ratio_volRevMove_volImpulsMove': 'ratio_volRevMove_volImpulsMove',
    'sc_ratio_volRevMoveZone1_volRevMoveExtrem_xRevZone': 'ratio_volRevMoveZone1_volRevMoveExtrem_XRevZone',
    'sc_ratio_deltaRevMove_volRevMove': 'ratio_deltaRevMove_volRevMove',
    'sc_diffPriceClose_VA6PPoc': 'diffPriceClose_VA6PPoc',
    'sc_delta_impulsMove_xRevZone_bigStand_extrem': 'delta_impulsMove_XRevZone_bigStand_extrem',
    'sc_is_imBullWithPoc_light_short': 'is_imBullWithPoc_light_short',
    'sc_delta_revMove_xRevZone_bigStand_extrem': 'delta_revMove_XRevZone_bigStand_extrem',
    'sc_diffHighPrice_0_1': 'diffHighPrice_0_1',
    'sc_is_rs_range_short': 'is_rs_range_short',
    'sc_is_rs_range_long':  'is_rs_range_long',
    'sc_is_vwap_reversal_pro_short': 'is_vwap_reversal_pro_short',
    'sc_is_vwap_reversal_pro_long': 'is_vwap_reversal_pro_long',
    'sc_is_mfi_overbought_short': 'is_mfi_overbought_short',
    'sc_pocDeltaPocVolRatio': 'pocDeltaPocVolRatio',
    'sc_diffPriceClosePoc_0_0': 'diffPriceClosePoc_0_0',
    'sc_ratio_deltaImpulsMove_volImpulsMove': 'ratio_deltaImpulsMove_volImpulsMove',
    'sc_cum_4DiffVolDeltaRatio': 'cum_4DiffVolDeltaRatio',
    'sc_candleSizeTicks': 'candleSizeTicks',
    'sc_ratio_volZone1_volExtrem': 'ratio_volZone1_volExtrem',
    'sc_is_imBullWithPoc_aggressive_short': 'is_imBullWithPoc_aggressive_short',
    'sc_is_imbBullLightPoc_Low00': 'is_imbBullLightPoc_AtrHigh0_1_short',
    'sc_volRevVolRevesalXContRatio': 'volRevVolRevesalXContRatio'

}

# Mapping des types de conditions
CONDITION_MAPPING = {
    'between': 'CONDITION_BETWEEN',
    'greater_than_or_equal': 'CONDITION_GREATER_THAN_OR_EQUAL',
    'less_than_or_equal': 'CONDITION_LESS_THAN_OR_EQUAL',
    'not_between': 'CONDITION_NOT_BETWEEN'
}

def format_float_value(value):
    """Formate correctement une valeur float pour C++"""
    if isinstance(value, (int, float)):
        if float(value).is_integer():
            return f"{int(value)}.0f"
        else:
            return f"{value}f"
    return f"{value}f"

def extract_algo_number(algo_name: str) -> int:
    """Extrait le numéro de l'algorithme depuis son nom"""
    match = re.search(r'(\d+)$', algo_name)
    if match:
        return int(match.group(1))
    return 1  # Par défaut
# ───────────────────────────────────────────────────────────────────────────────
#  COMPLETE C++ GENERATOR  –  version minimaliste (sans redéfinitions inutiles)
# ───────────────────────────────────────────────────────────────────────────────
from datetime import datetime
from typing import Dict, Any, List

# ───────────────────────────────────────────────────────────────────────────────
#  CLASS  CompleteCppGenerator  –  full version with AlgoDefinition struct
# ───────────────────────────────────────────────────────────────────────────────
from datetime import datetime
from typing import Dict, Any, List


class CompleteCppGenerator:
    """
    Génère un fichier C++ compatible Sierra Chart :
      • accepte des numéros d'algo non continus ;
      • insère la structure AlgoDefinition (typedef + struct) ;
      • n'impose aucune constante supplémentaire (MAX_ALGO_SIMULTANEOUS existe ailleurs).
      • NOUVEAU: inclut le tracking des indices d'algorithmes pour les compteurs.
    """

    # ------------------------------------------------------------------ #
    def __init__(
            self,
            *,
            global_filter: Dict[str, List[Dict[str, Any]]],
            active_short_algos: Dict[str, Any],
            active_long_algos: Dict[str, Any],
            all_short_algos: Dict[str, Any],
            all_long_algos: Dict[str, Any],
    ):
        self.global_filter = global_filter
        self.active_short_algos = active_short_algos
        self.active_long_algos = active_long_algos
        self.all_short_algos = all_short_algos
        self.all_long_algos = all_long_algos

        self.short_slots = self._build_slots(self.active_short_algos, "algoShort")
        self.long_slots = self._build_slots(self.active_long_algos, "algoLong")

    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_slots(names: Dict[str, Any], prefix: str) -> List[str]:
        """Retourne une liste indexable par (numéro-1) ; les trous contiennent ''. """
        if not names:
            return []
        max_idx = max(extract_algo_number(n) for n in names)
        slots = [""] * max_idx
        for n in names:
            slots[extract_algo_number(n) - 1] = n
        return slots

    # ------------------------------------------------------------------ #
    #  Sections C++ (= méthodes privées)
    # ------------------------------------------------------------------ #
    def _header(self) -> List[str]:
        stamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        return [
            "// ==============================================================",
            "//  ⚠️  CODE GÉNÉRÉ AUTOMATIQUEMENT – NE PAS MODIFIER",
            f"//  Généré : {stamp}",
            "// ==============================================================",
            '#include "sierrachart.h"',
            '#include "../include/L-AdvanceSC.h"',
            '#include "../include/Trading_miniAlgos.h"',
            '#include "../include/TradeManagement.h"',
            "",
        ]

    def _struct_definition(self) -> List[str]:
        """Typedef + struct nécessaires aux tableaux d'algos."""
        return [
            "// --------------------------------------------------------------",
            "//  Définition de AlgoDefinition",
            "// --------------------------------------------------------------",
            "typedef bool (*AlgoCheckFunction)(const features_s&);",
            "",
            "struct AlgoDefinition {",
            "    AlgoCheckFunction checkFunction;",
            "    const char*       name;",
            "};",
            "",
        ]

    # def _stub(self, name: str) -> List[str]:
    #     return [
    #         f"bool {name}(const SCStudyInterfaceRef /*sc*/, const features_s& /*fs*/)",
    #         "{",
    #         "    // TODO : implémenter la logique",
    #         "    return false;",
    #         "}",
    #         "",
    #     ]

    def _condition_cpp(self, feat: str, c: Dict[str, Any]) -> str:
        cpp_feat = FEATURE_MAPPING.get(feat, feat)
        ctype = CONDITION_MAPPING[c["type"]]
        if c["type"] in ("between", "not_between"):
            thr, mn, mx = "0.0f", format_float_value(c["min"]), format_float_value(c["max"])
        else:
            thr, mn, mx = format_float_value(c["threshold"]), "0.0f", "0.0f"
        return (f"CheckFeatureCondition(fs.{cpp_feat}, "
                f"{{ {ctype}, {thr}, {mn}, {mx}, true }})")

    def _algo_func(self, name: str, cfg: Dict[str, Any], side: str) -> List[str]:
        func = f"CheckAlgo{extract_algo_number(name)}{side.title()}"
        conds: List[str] = []
        for feat, clist in cfg.items():
            for c in clist:
                if c.get("active", True):
                    conds.append(self._condition_cpp(feat, c))
        if not conds:
            return [f"static bool {func}(const features_s&) {{ return false; }}", ""]
        return [
            f"static bool {func}(const features_s& fs)",
            "{",
            "    return " + "\n        && ".join(conds) + ";",
            "}",
            "",
        ]

    def _algo_array(self, slots: List[str], side: str) -> List[str]:
        arr = f"{side.lower()}Algos"
        out = [f"static const AlgoDefinition {arr}[ALGO_TYPE_COUNT] = {{"]
        for i in range(ALGO_TYPE_COUNT):
            if i < len(slots) and slots[i]:
                func = f"CheckAlgo{i + 1}{side.title()}"
                out.append(f'    {{ {func}, "ALGO_{side.upper()}_{i + 1}" }},')
            else:
                out.append('    { nullptr, "UNUSED" },')
        out.append("};\n")
        return out

    def _global_filter(self, side: str) -> List[str]:
        fname = f"CheckGlobalMicroStructureFilters{side.title()}"
        if not self.global_filter:
            return [f"bool {fname}(const features_s&) {{ return true; }}", ""]
        code = [f"bool {fname}(const features_s& fs)", "{"]
        idx = 1
        for feat, clist in self.global_filter.items():
            for c in clist:
                if c.get("active", True):
                    code.append(f"    // Condition {idx}")
                    code.append(f"    if (!{self._condition_cpp(feat, c)}) return false;")
                    idx += 1
        code += ["    return true;", "}", ""]
        return code

    def _dispatcher(self, side: str, has_algo: bool) -> List[str]:
        """
        Génère le dispatcher avec tracking des indices d'algorithmes.
        MISE À JOUR: Ajoute les indices dans AlgoFilterResult.
        """
        func = f"CheckAlgosFilters{side.title()}"
        arr = f"{side.lower()}Algos"
        order_type = f"ORDDERTYPE_{side.upper()}"

        if not has_algo:
            return [
                f"AlgoFilterResult {func}(const features_s&, const bool[ALGO_TYPE_COUNT])",
                "{",
                "    AlgoFilterResult r{};",
                "    r.isValid = false;",
                f"    r.orderType = {order_type};",
                "    return r;",
                "}",
                "",
            ]

        return [
            f"AlgoFilterResult {func}(const features_s& fs, const bool enabled[ALGO_TYPE_COUNT])",
            "{",
            "    AlgoFilterResult r{};",
            f"    r.orderType = {order_type};",
            "    for (int i = 0; i < ALGO_TYPE_COUNT && "
            "r.algoCount < MAX_ALGO_SIMULTANEOUS; ++i) {",
            f"        if (enabled[i] && {arr}[i].checkFunction && "
            f"{arr}[i].checkFunction(fs)) {{",
            f"            strcpy(r.algoNames[r.algoCount], {arr}[i].name);",
            "            r.algoIndices[r.algoCount] = i;  // ← NOUVEAU: stocker l'indice",
            "            ++r.algoCount;",
            "        }",
            "    }",
            "    r.isValid = r.algoCount > 0;",
            "    return r;",
            "}",
            "",
        ]

    def _usage_example(self) -> List[str]:
        """
        Génère un exemple d'utilisation avec les compteurs de tracking.
        """
        return [
            "// --------------------------------------------------------------",
            "//  EXEMPLE D'UTILISATION AVEC TRACKING DES COMPTEURS",
            "// --------------------------------------------------------------",
            "/*",
            "// Dans votre code principal, ajoutez ces constantes :",
            "#define PERSISTANT29_INT_UNIQUE_SHORT_TRADES           29",
            "#define PERSISTANT30_INT_ALGO_VALIDATIONS_BASE_RESERVED 30  // 30-47 pour SHORT",
            "#define PERSISTANT48_INT_UNIQUE_LONG_TRADES            48",
            "#define PERSISTANT49_INT_ALGO_VALIDATIONS_BASE_LONG    49  // 49-66 pour LONG",
            "",
            "// Exemple d'utilisation pour SHORT :",
            "if (enableShortEntries && baseOK_short && microOK_short) {",
            "    AlgoFilterResult shortAlgoResult{};",
            "    bool shortOK = bypassShortAlgo;",
            "    ",
            "    if (!bypassShortAlgo) {",
            "        shortAlgoResult = CheckAlgosFiltersShort(s_features, algoEnabled);",
            "        shortOK = shortAlgoResult.isValid && shortAlgoResult.algoCount >= minAlgosRequired;",
            "    }",
            "    ",
            "    if (shortOK) {",
            "        // Compteur de trades uniques",
            "        int& uniqueShortTrades = sc.GetPersistentInt(PERSISTANT29_INT_UNIQUE_SHORT_TRADES);",
            "        uniqueShortTrades++;",
            "        ",
            "        // Compteurs par algorithme",
            "        for (int i = 0; i < shortAlgoResult.algoCount; ++i) {",
            "            int algoIndex = shortAlgoResult.algoIndices[i];",
            "            int& algoCounter = sc.GetPersistentInt(PERSISTANT30_INT_ALGO_VALIDATIONS_BASE_RESERVED + algoIndex);",
            "            algoCounter++;",
            "        }",
            "        ",
            "        // Logging détaillé",
            "        SCString algoDetails;",
            "        for (int i = 0; i < shortAlgoResult.algoCount; ++i) {",
            "            int algoIndex = shortAlgoResult.algoIndices[i];",
            "            int currentCount = sc.GetPersistentInt(PERSISTANT30_INT_ALGO_VALIDATIONS_BASE_RESERVED + algoIndex);",
            "            ",
            "            if (i > 0) algoDetails += \", \";",
            "            algoDetails.Format(\"%s%s[%d](%d)\", algoDetails.GetChars(),",
            "                             shortAlgoResult.algoNames[i], algoIndex, currentCount);",
            "        }",
            "        ",
            "        SCString msg;",
            "        msg.Format(\"SHORT Entry | UniqueTrades=%d | Algos=[%s]\",",
            "                   uniqueShortTrades, algoDetails.GetChars());",
            "        sc.AddMessageToLog(msg, 0);",
            "    }",
            "}",
            "*/",
            "",
        ]

    # ------------------------------------------------------------------ #
    def generate_complete_cpp(self) -> str:
        out: List[str] = []

        # 1) En-tête + struct AlgoDefinition
        out += self._header()
        out += self._struct_definition()

        # # 2) Stubs pour chaque algorithme actif
        # for n in sorted(set(self.active_short_algos) | set(self.active_long_algos)):
        #     out += self._stub(n)

        # 3) Filtres globaux
        out += self._global_filter("short")
        out += self._global_filter("long")

        # 4) Fonctions d'algo
        for n in self.active_short_algos:
            out += self._algo_func(n, self.all_short_algos.get(n, {}), "short")
        for n in self.active_long_algos:
            out += self._algo_func(n, self.all_long_algos.get(n, {}), "long")

        # 5) Tableaux des algorithmes
        out += self._algo_array(self.short_slots, "short")
        out += self._algo_array(self.long_slots, "long")

        # 6) Dispatchers (MISE À JOUR avec indices)
        out += self._dispatcher("short", bool(self.active_short_algos))
        out += self._dispatcher("long", bool(self.active_long_algos))

        # 7) Exemple d'utilisation (NOUVEAU)
        out += self._usage_example()

        # 8) Chaîne finale
        return "\n".join(out)


# ════════════════════════════════════════════════════════════════════════════════
# EXÉCUTION DE LA GÉNÉRATION C++ - VERSION INLINE
# ════════════════════════════════════════════════════════════════════════════════

print("📊 Analyse Python terminée !")

# Vérifier si la génération C++ est activée
if GENERATE_CPP_FILE:
    print("\n" + "=" * 80)
    print("🎯 GÉNÉRATION AUTOMATIQUE DU CODE C++")
    print("=" * 80)

    try:
        print("🚀 Utilisation du générateur C++ complet intégré...")

        # Créer tous les dictionnaires d'algorithmes pour le générateur
        all_short_algos = {
            'algoShort1': algoShort1, 'algoShort2': algoShort2, 'algoShort3': algoShort3,
            'algoShort4': algoShort4, 'algoShort5': algoShort5, 'algoShort6': algoShort6,
            'algoShort7': algoShort7, 'algoShort8': algoShort8, 'algoShort9': algoShort9,
            'algoShort10': algoShort10, 'algoShort11': algoShort11, 'algoShort12': algoShort12,
            'algoShort13': algoShort13, 'algoShort14': algoShort14, 'algoShort15': algoShort15,
            'algoShort16': algoShort16, 'algoShort17': algoShort17,
            'algoShort18': algoShort18
        }
        all_long_algos = {
            'algoLong1': algoLong1, 'algoLong2': algoLong2, 'algoLong3': algoLong3,
            'algoLong4': algoLong4, 'algoLong5': algoLong5, 'algoLong6': algoLong6,
            'algoLong7': algoLong7, 'algoLong8': algoLong8, 'algoLong9': algoLong9,
            'algoLong10': algoLong10, 'algoLong11': algoLong11, 'algoLong12': algoLong12,
            'algoLong13': algoLong13, 'algoLong14': algoLong14, 'algoLong15': algoLong15,
            'algoLong16': algoLong16, 'algoLong17': algoLong17, 'algoLong18': algoLong18
        }

        print(f"📊 Configuration détectée :")
        print(f"   • Filtres globaux : {len(GLOBAL_MICRO_FILTER)} conditions")
        print(f"   • Algorithmes Short actifs : {list(algorithmsShort.keys())}")
        print(f"   • Algorithmes Long actifs : {list(algorithmsLong.keys())}")

        # Créer le générateur
        generator = CompleteCppGenerator(
            global_filter=GLOBAL_MICRO_FILTER,
            active_short_algos=algorithmsShort,
            active_long_algos=algorithmsLong,
            all_short_algos=all_short_algos,
            all_long_algos=all_long_algos
        )

        # Générer le code C++
        print("🔄 Génération du code C++ en cours...")
        cpp_code = generator.generate_complete_cpp()

        # Chemin de sortie
        output_path = Path(CPP_OUTPUT_DIRECTORY) / CPP_OUTPUT_FILE
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Écrire le fichier
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(cpp_code)

        # Afficher les résultats
        print(f"\n✅ GÉNÉRATION RÉUSSIE !")
        print(f"📁 Fichier généré : {output_path}")
        print(f"📈 Lignes de code : {len(cpp_code.splitlines())}")
        print(f"📊 Taille du fichier : {len(cpp_code)} caractères")

        # Instructions pour Sierra Chart
        print(f"\n🎯 PROCHAINES ÉTAPES POUR SIERRA CHART :")
        print(f"   1. ✅ Le fichier C++ est prêt : {CPP_OUTPUT_FILE}")
        print(f"   2. 📂 Il est dans le dossier : {CPP_OUTPUT_DIRECTORY}")
        print(f"   3. 🔄 Recompilez Sierra Chart (Ctrl+Shift+F5)")
        print(f"   4. 📊 Ajoutez l'étude à votre graphique")
        print(f"   5. ⚙️ Configurez les paramètres d'algorithmes")

        # Résumé des algorithmes générés
        if algorithmsShort:
            print(f"\n📋 ALGORITHMES SHORT GÉNÉRÉS :")
            for i, algo_name in enumerate(algorithmsShort.keys(), 1):
                print(f"   • CheckAlgo{i}Short() → {algo_name}")

        if algorithmsLong:
            print(f"\n📋 ALGORITHMES LONG GÉNÉRÉS :")
            for i, algo_name in enumerate(algorithmsLong.keys(), 1):
                print(f"   • CheckAlgo{i}Long() → {algo_name}")

        print(f"\n🎊 GÉNÉRATION TERMINÉE AVEC SUCCÈS !")

    except Exception as e:
        print(f"\n❌ ERREUR LORS DE LA GÉNÉRATION C++ :")
        print(f"   {str(e)}")
        print(f"\n🔧 VÉRIFICATIONS :")
        print(f"   • Le répertoire {CPP_OUTPUT_DIRECTORY} est-il accessible ?")
        print(f"   • Avez-vous les droits d'écriture ?")
        print(f"   • Les variables GLOBAL_MICRO_FILTER et algorithmsShort/Long sont-elles définies ?")

else:
    print(f"\n⏭️ GÉNÉRATION C++ DÉSACTIVÉE")
    print(f"💡 Pour activer la génération automatique :")
    print(f"   → Modifiez GENERATE_CPP_FILE = True en haut du fichier")
    print(f"   → Relancez le script")

print(f"\n🏁 SCRIPT TERMINÉ")
print("=" * 80)

# ════════════════════════════════════════════════════════════════════════════════
# 📋 INSTRUCTIONS D'UTILISATION
# ════════════════════════════════════════════════════════════════════════════════
"""
🎯 COMMENT UTILISER CETTE SECTION :

1. 📋 COPIEZ tout ce code à la fin de votre script principal
2. 🎛️ CONFIGUREZ GENERATE_CPP_FILE = True pour activer la génération
3. 🚀 LANCEZ votre script - la génération se fera automatiquement à la fin

AVANTAGES DE CETTE VERSION INLINE :
✅ EXÉCUTION DIRECTE : Pas besoin de if __name__ == "__main__"
✅ MESSAGES CLAIRS : Indications détaillées sur le processus
✅ GESTION D'ERREURS : Diagnostics en cas de problème
✅ INSTRUCTIONS : Guide pour les prochaines étapes Sierra Chart
✅ SIMPLICITÉ : Juste coller à la fin et ça marche !

WORKFLOW :
📊 Votre analyse Python s'exécute
     ⬇️
🔄 Génération automatique du C++ (si activée)
     ⬇️
✅ Fichier C++ prêt pour Sierra Chart
     ⬇️
🎯 Instructions pour la compilation et l'utilisation
"""