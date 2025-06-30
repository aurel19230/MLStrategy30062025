# -*- coding: utf-8 -*-
"""optuna_imbalance_low_high_detection.py – version 5
================================================
- Stratégie "Imbalance Low High Detection" supportant SHORT et LONG
- Intègre **trois** conditions distinctes avec des plages de sampling différentes
- Adapte automatiquement les colonnes selon la direction
- Utilise **deux** datasets de validation pour une plus grande robustesse
- Raccourci clavier « & » pour déclencher un calcul immédiat sur le jeu TEST
- Support pour CSV_TRAIN=None (pas d'utilisation du dataset TRAIN)
- MODIFICATION: TRAIN toujours testé pour info même si non utilisé dans l'optimisation
- ✅ CORRECTION: Bug d'alignement des indices corrigé
"""
from __future__ import annotations

# ─────────────────── CONFIG ──────────────────────────────────────
from pathlib import Path
import sys, time, math, optuna, pandas as pd, numpy as np
import threading

# Remplacer msvcrt par pynput
from pynput import keyboard

# Ajout de colorama pour les affichages colorés
from colorama import init, Fore, Back, Style

# Initialiser colorama (nécessaire pour Windows)
init(autoreset=True)

RANDOM_SEED = 42

# ═══════════════════════ CONFIGURATION DIRECTION ═══════════════════════

DIRECTION = "long"  # "short" ou "long"

# Mapping des colonnes selon la direction
COLUMN_MAPPING_OPTUNA = {
    "short": {
        "volume_col": "sc_bidVolHigh_1",
        "imbalance_col": "sc_bull_imbalance_high_0",
        "description": "Détection imbalances haussières sur les hauts (retournement baissier)"
    },
    "long": {
        "volume_col": "sc_askVolLow_1",
        "imbalance_col": "sc_bear_imbalance_low_0",
        "description": "Détection imbalances baissières sur les bas (retournement haussier)"
    }
}

# Configuration des colonnes pour la direction choisie
VOLUME_COL = COLUMN_MAPPING_OPTUNA[DIRECTION]["volume_col"]
IMBALANCE_COL = COLUMN_MAPPING_OPTUNA[DIRECTION]["imbalance_col"]
STRATEGY_DESC = COLUMN_MAPPING_OPTUNA[DIRECTION]["description"]

# L'objectif est toujours de maximiser le WR
OPTIMIZATION_GOAL = "maximize"
GOAL_DESCRIPTION = "MAXIMISER"

print(f"{Fore.CYAN}🎯 STRATÉGIE: Imbalance Low High Detection - {DIRECTION.upper()}{Style.RESET_ALL}")
print(f"{Fore.YELLOW}📊 {STRATEGY_DESC}{Style.RESET_ALL}")
print(f"{Fore.GREEN}🔧 Colonnes utilisées: {VOLUME_COL} & {IMBALANCE_COL}{Style.RESET_ALL}")

# Fichiers de données
DIR = R"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject" \
      R"\Sierra_chart\xTickReversal\simu\5_0_5TP_6SL\\merge"

# Adaptation du nom de fichier selon la direction
Direction = DIRECTION.capitalize()

# ====== MODIFICATION PRINCIPALE: Contrôle d'utilisation du dataset TRAIN ======
USE_TRAIN_IN_OPTIMIZATION = True  # Mettre False pour désactiver l'utilisation du dataset TRAIN dans l'optimisation
USE_TRAIN_IN_OPTIMIZATION = False  # Mettre True pour utiliser TRAIN dans l'optimisation

# Chemins des fichiers (TRAIN est toujours défini pour pouvoir afficher les stats finales)
CSV_TRAIN = DIR + Rf"\Step5_5_0_5TP_6SL_010124_200625_extractOnlyFullSession_Only{Direction}_feat__split2_01052024_30092024.csv"
CSV_VAL = DIR + Rf"\Step5_5_0_5TP_6SL_010124_200625_extractOnlyFullSession_Only{Direction}_feat__split3_30092024_28022025.csv"
CSV_VAL1 = DIR + Rf"\Step5_5_0_5TP_6SL_010124_200625_extractOnlyFullSession_Only{Direction}_feat__split4_02032025_15052025.csv"
CSV_TEST = DIR + Rf"\Step5_5_0_5TP_6SL_010124_200625_extractOnlyFullSession_Only{Direction}_feat__split5_15052025_20062025.csv"


# Configuration par mode
CONFIGS = {
    "light": {
        "WINRATE_MIN": 0.535,
        "PCT_TRADE_MIN": 0.04,
        "ALPHA": 0.70,
    },
    "aggressive": {
        "WINRATE_MIN": 0.595,
        "PCT_TRADE_MIN": 0.0035,
        "ALPHA": 0,
    }
}

choice = input(
    "Filtrage :\n"
    "  [Entrée] → light (meilleur scénario testé)\n"
    "  a        → agressif\n"
    "  z        → light + poc variable \n"
    "Choix : "
).strip().lower()

if choice == "a":
    cfg = CONFIGS["aggressive"]
    FILTER_POC = False
elif choice == "z":
    cfg = CONFIGS["light"]
    FILTER_POC = True
else:
    cfg = CONFIGS["light"]
    FILTER_POC = False
# ✅ Construction du résumé des paramètres sélectionnés
param_summary = (
    f"🛠️ Paramètres sélectionnés :\n"
    f"▪️ Mode : {'aggressive' if choice == 'a' else 'light'}\n"
    f"▪️ FILTER_POC : {FILTER_POC}\n"
    f"▪️ Config utilisée : {cfg}"
)

# Affichage ou log
print(param_summary)
print(f"\n→ Mode : {'agressif' if choice == 'a' else 'light'}"
      f"{' + poc variable' if FILTER_POC else ''}")

# Affichage de l'état du dataset TRAIN
if not USE_TRAIN_IN_OPTIMIZATION:
    print(
        f"{Fore.YELLOW}⚠️  TRAIN DATASET DÉSACTIVÉ pour l'optimisation - Testé pour information seulement{Style.RESET_ALL}")
else:
    print(f"{Fore.GREEN}✔ TRAIN DATASET ACTIVÉ pour l'optimisation{Style.RESET_ALL}")

print()

WINRATE_MIN = cfg["WINRATE_MIN"]
PCT_TRADE_MIN = cfg["PCT_TRADE_MIN"]
ALPHA = cfg["ALPHA"]

# Paramètres non modifiés par le choix utilisateur
N_TRIALS = 20_000
PRINT_EVERY = 50
FAILED_PENALTY = -0.001

# Gap penalties
LAMBDA_WR = 0.4
LAMBDA_PCT = 0.6

# ── Bornes par condition (adaptées selon la direction) ────────────
if DIRECTION == "short":
    # Paramètres optimisés pour SHORT
    BID_MIN_1, BID_MAX_1 = 3, 8
    BULL_MIN_1, BULL_MAX_1 = 2, 8

    BID_MIN_2, BID_MAX_2 = 8, 17
    BULL_MIN_2, BULL_MAX_2 = 2, 8

    BID_MIN_3, BID_MAX_3 = 14, 60
    BULL_MIN_3, BULL_MAX_3 = 1.5, 8
else:  # LONG
    # Paramètres à optimiser pour LONG (valeurs initiales similaires)
    BID_MIN_1, BID_MAX_1 = 3, 8
    BULL_MIN_1, BULL_MAX_1 = 2, 8

    BID_MIN_2, BID_MAX_2 = 8, 17
    BULL_MIN_2, BULL_MAX_2 = 2, 8

    BID_MIN_3, BID_MAX_3 = 14, 60
    BULL_MIN_3, BULL_MAX_3 = 1.5, 8

# Bornes POC
POS_POC_LOWER_BOUND_MIN, POS_POC_LOWER_BOUND_MAX = -1, 0
POS_POC_UPPER_BOUND_MIN, POS_POC_UPPER_BOUND_MAX = -1, 0
POS_POC_STEP = 0.25

import chardet




# ───────────────────── DATA LOADING ─────────────────────────────
def detect_file_encoding(file_path: str, sample_size: int = 100_000) -> str:
    with open(file_path, 'rb') as f:
        raw = f.read(sample_size)
    return chardet.detect(raw)['encoding']


def load_csv(path: str | Path) -> tuple[pd.DataFrame, int]:
    encoding = detect_file_encoding(str(path))
    if encoding.lower() == "ascii":
        encoding = "ISO-8859-1"

    print(f"{Path(path).name} ➜ encodage détecté: {encoding}")

    df = pd.read_csv(path, sep=";", encoding=encoding, parse_dates=["date"], low_memory=False)

    df["sc_sessionStartEnd"] = pd.to_numeric(df["sc_sessionStartEnd"], errors="coerce")
    df = df.dropna(subset=["sc_sessionStartEnd"])
    df["sc_sessionStartEnd"] = df["sc_sessionStartEnd"].astype(int)

    print(f"{Path(path).name} ➜ uniques sc_sessionStartEnd: {df['sc_sessionStartEnd'].unique()}")

    nb_start = (df["sc_sessionStartEnd"] == 10).sum()
    nb_end = (df["sc_sessionStartEnd"] == 20).sum()
    nb_sessions = min(nb_start, nb_end)

    if nb_start != nb_end:
        print(
            f"{Fore.YELLOW}⚠️ Incohérence sessions: {nb_start} débuts vs {nb_end} fins dans {Path(path).name}{Style.RESET_ALL}")
    else:
        print(f"{Fore.GREEN}✔ {nb_sessions} sessions complètes détectées dans {Path(path).name}{Style.RESET_ALL}")

    df["session_id"] = (df["sc_sessionStartEnd"] == 10).cumsum().astype("int32")

    # ✅ CORRECTION DU BUG: Filtrer SANS reset_index
    df = df[df["class_binaire"].isin([0, 1])].copy()
    # ❌ LIGNE SUPPRIMÉE: df.reset_index(drop=True, inplace=True)

    return df, nb_sessions


# ═══════════════════════ FONCTION DE TEST D'ALIGNEMENT ═══════════════════════

def test_alignment(df_original, df_filtered):
    """Test pour vérifier l'alignement des indices après correction"""
    print(f"\n🔍 TEST D'ALIGNEMENT DES INDICES:")

    # Test sur quelques lignes
    sample_indices = df_filtered.index[:5]
    print(f"  Indices df_filtered[:5]: {sample_indices.tolist()}")

    # Vérifier si les indices existent dans df_original
    if all(idx in df_original.index for idx in sample_indices):
        print(
            f"  class_binaire df_original[sample_indices]: {df_original.loc[sample_indices, 'class_binaire'].tolist()}")
        print(f"  class_binaire df_filtered[:5]: {df_filtered['class_binaire'].iloc[:5].tolist()}")

        # Vérification
        if df_original.loc[sample_indices, 'class_binaire'].equals(df_filtered['class_binaire'].iloc[:5]):
            print(f"  {Fore.GREEN}✅ Alignement correct !{Style.RESET_ALL}")
        else:
            print(f"  {Fore.RED}❌ Problème d'alignement détecté !{Style.RESET_ALL}")
    else:
        print(f"  {Fore.RED}❌ Certains indices n'existent pas dans df_original !{Style.RESET_ALL}")


# ====== CHARGEMENT DES DATASETS (TRAIN toujours chargé pour les stats finales) ======
print(f"{Fore.CYAN}Chargement des données...{Style.RESET_ALL}")

# Chargement complet pour le test d'alignement
TRAIN_COMPLETE = pd.read_csv(CSV_TRAIN, sep=";", encoding='ISO-8859-1', parse_dates=["date"], low_memory=False)
TRAIN, TRAIN_SESSIONS = load_csv(CSV_TRAIN)
VAL, VAL_SESSIONS = load_csv(CSV_VAL)
VAL1, VAL1_SESSIONS = load_csv(CSV_VAL1)
TEST, TEST_SESSIONS = load_csv(CSV_TEST)

# ✅ TEST D'ALIGNEMENT APRÈS CHARGEMENT
test_alignment(TRAIN_COMPLETE, TRAIN)

# Affichage des statistiques
datasets_info = [
    ("TRAIN", TRAIN, TRAIN_SESSIONS),
    ("VAL", VAL, VAL_SESSIONS),
    ("VAL1", VAL1, VAL1_SESSIONS),
    ("TEST", TEST, TEST_SESSIONS)
]

for lbl, d, sessions in datasets_info:
    usage_info = " (utilisé pour optimisation)" if (lbl == "TRAIN" and USE_TRAIN_IN_OPTIMIZATION) else (
        " (info seulement)" if lbl == "TRAIN" else "")
    print(
        f"{lbl:<5} | lignes={len(d):,}  WR brut={(d['class_binaire'] == 1).mean():.2%}  Sessions={sessions}{usage_info}")
print("—————————————————————————————————————————————————————————————\n")


# ───────────────────── MASK BUILDERS (GENERIQUES) ────────────────────────
def imbalance_detection(df: pd.DataFrame, volume_threshold: float, imbalance_threshold: float) -> pd.Series:
    """Condition générique pour détecter les imbalances selon la direction"""
    return (df[VOLUME_COL] > volume_threshold) & (df[IMBALANCE_COL] > imbalance_threshold)


def imbalance_detection_1(df: pd.DataFrame, *, volume_1: float, imbalance_1: float, **kwargs) -> pd.Series:
    """Condition 1"""
    return imbalance_detection(df, volume_1, imbalance_1)


def imbalance_detection_2(df: pd.DataFrame, *, volume_2: float, imbalance_2: float, **kwargs) -> pd.Series:
    """Condition 2"""
    return imbalance_detection(df, volume_2, imbalance_2)


def imbalance_detection_3(df: pd.DataFrame, *, volume_3: float, imbalance_3: float, **kwargs) -> pd.Series:
    """Condition 3"""
    return imbalance_detection(df, volume_3, imbalance_3)


# ───────────────────── METRICS HELPERS ──────────────────────────
def _metrics(df: pd.DataFrame, mask: pd.Series, original_len: int = None) -> tuple[float, float, int, int, int]:
    """Calcule les métriques avec le nombre de sessions couvertes"""
    sub = df.loc[mask]
    if sub.empty:
        return 0.0, 0.0, 0, 0, 0

    wins = int((sub["class_binaire"] == 1).sum())
    total = len(sub)

    base_len = original_len if original_len is not None else len(df)
    pct_trade = total / base_len
    sessions_covered = sub["session_id"].nunique()

    return wins / total, pct_trade, wins, total - wins, sessions_covered


def _metrics_combined(df: pd.DataFrame, m1: pd.Series, m2: pd.Series, m3: pd.Series, original_len: int = None):
    """Calcule les métriques combinées avec le nombre de sessions couvertes"""
    m_u = m1 | m2 | m3
    m_12 = m1 & m2
    m_13 = m1 & m3
    m_23 = m2 & m3
    m_123 = m1 & m2 & m3
    return _metrics(df, m_u, original_len) + _metrics(df, m_12, original_len) + _metrics(df, m_13,
                                                                                         original_len) + _metrics(df,
                                                                                                                  m_23,
                                                                                                                  original_len) + _metrics(
        df, m_123, original_len)


def _metrics_exclusive(df: pd.DataFrame, m1: pd.Series, m2: pd.Series, m3: pd.Series, original_len: int = None):
    """Calcule des métriques détaillées montrant les trades uniques et les chevauchements"""
    m1_only = m1 & ~m2 & ~m3
    m2_only = ~m1 & m2 & ~m3
    m3_only = ~m1 & ~m2 & m3
    m12_only = m1 & m2 & ~m3
    m13_only = m1 & ~m2 & m3
    m23_only = ~m1 & m2 & m3
    m123 = m1 & m2 & m3

    m_u = m1 | m2 | m3

    metrics_global = _metrics(df, m_u, original_len)
    metrics_1_only = _metrics(df, m1_only, original_len)
    metrics_2_only = _metrics(df, m2_only, original_len)
    metrics_3_only = _metrics(df, m3_only, original_len)
    metrics_12_only = _metrics(df, m12_only, original_len)
    metrics_13_only = _metrics(df, m13_only, original_len)
    metrics_23_only = _metrics(df, m23_only, original_len)
    metrics_123 = _metrics(df, m123, original_len)

    return {
        "global": metrics_global,
        "cond1_only": metrics_1_only,
        "cond2_only": metrics_2_only,
        "cond3_only": metrics_3_only,
        "cond12": metrics_12_only,
        "cond13": metrics_13_only,
        "cond23": metrics_23_only,
        "cond123": metrics_123
    }


# ───────────────────── OPTUNA OBJECTIVE ─────────────────────────
best_trial = {
    "score": -math.inf,
    "number": None,
    "score_old": -math.inf,
    "wr_t": 0.0, "pct_t": 0.0, "suc_t": 0, "fail_t": 0, "sess_t": 0,
    "wr_v": 0.0, "pct_v": 0.0, "suc_v": 0, "fail_v": 0, "sess_v": 0,
    "wr_v1": 0.0, "pct_v1": 0.0, "suc_v1": 0, "fail_v1": 0, "sess_v1": 0,
    "wr_t1": 0.0, "pct_t1": 0.0, "suc_t1": 0, "fail_t1": 0, "sess_t1": 0,
    "wr_t2": 0.0, "pct_t2": 0.0, "suc_t2": 0, "fail_t2": 0, "sess_t2": 0,
    "wr_t3": 0.0, "pct_t3": 0.0, "suc_t3": 0, "fail_t3": 0, "sess_t3": 0,
    "wr_v1": 0.0, "pct_v1": 0.0, "suc_v1": 0, "fail_v1": 0, "sess_v1": 0,
    "wr_v2": 0.0, "pct_v2": 0.0, "suc_v2": 0, "fail_v2": 0, "sess_v2": 0,
    "wr_v3": 0.0, "pct_v3": 0.0, "suc_v3": 0, "fail_v3": 0, "sess_v3": 0,
    "wr_v1_1": 0.0, "pct_v1_1": 0.0, "suc_v1_1": 0, "fail_v1_1": 0, "sess_v1_1": 0,
    "wr_v1_2": 0.0, "pct_v1_2": 0.0, "suc_v1_2": 0, "fail_v1_2": 0, "sess_v1_2": 0,
    "wr_v1_3": 0.0, "pct_v1_3": 0.0, "suc_v1_3": 0, "fail_v1_3": 0, "sess_v1_3": 0,
    "avg_gap_wr": 0.0,
    "avg_gap_pct": 0.0,
    "metrics_detail_train": None,
    "metrics_detail_val": None,
    "metrics_detail_val1": None,
    "params": {}
}


def objective(trial: optuna.trial.Trial) -> float:
    # Paramètres adaptés aux noms génériques
    p = {
        "volume_1": trial.suggest_int("volume_1", BID_MIN_1, BID_MAX_1),
        "imbalance_1": trial.suggest_float("imbalance_1", BULL_MIN_1, BULL_MAX_1),
        "volume_2": trial.suggest_int("volume_2", BID_MIN_2, BID_MAX_2),
        "imbalance_2": trial.suggest_float("imbalance_2", BULL_MIN_2, BULL_MAX_2),
        "volume_3": trial.suggest_int("volume_3", BID_MIN_3, BID_MAX_3),
        "imbalance_3": trial.suggest_float("imbalance_3", BULL_MIN_3, BULL_MAX_3),
    }

    val_len = len(VAL)
    val1_len = len(VAL1)

    if FILTER_POC:
        min_value = trial.suggest_float("pos_poc_min", POS_POC_LOWER_BOUND_MIN, POS_POC_LOWER_BOUND_MAX,
                                        step=POS_POC_STEP)
        max_value = trial.suggest_float("pos_poc_max", POS_POC_UPPER_BOUND_MIN, POS_POC_UPPER_BOUND_MAX,
                                        step=POS_POC_STEP)
        p["pos_poc_min"] = min(min_value, max_value)
        p["pos_poc_max"] = max(min_value, max_value)

    # ====== MODIFICATION: TOUJOURS CALCULER LES MÉTRIQUES TRAIN ======
    train_df = TRAIN.copy()
    train_len = len(TRAIN)

    if FILTER_POC:
        poc_min = p["pos_poc_min"]
        poc_max = p["pos_poc_max"]
        train_df = train_df[
            (train_df["diffPriceClosePoc_0_0"] >= poc_min) & (train_df["diffPriceClosePoc_0_0"] <= poc_max)]

    # Masks pour TRAIN (toujours calculées maintenant)
    m1_t = imbalance_detection_1(train_df, **p)
    m2_t = imbalance_detection_2(train_df, **p)
    m3_t = imbalance_detection_3(train_df, **p)

    # Métriques TRAIN (toujours calculées maintenant)
    wr_t1, pct_t1, suc_t1, fail_t1, sess_t1 = _metrics(train_df, m1_t, train_len)
    wr_t2, pct_t2, suc_t2, fail_t2, sess_t2 = _metrics(train_df, m2_t, train_len)
    wr_t3, pct_t3, suc_t3, fail_t3, sess_t3 = _metrics(train_df, m3_t, train_len)

    wr_t, pct_t, suc_t, fail_t, sess_t, *_ = _metrics_combined(train_df, m1_t, m2_t, m3_t, train_len)
    metrics_detail_train = _metrics_exclusive(train_df, m1_t, m2_t, m3_t, train_len)

    val_df = VAL.copy()
    val1_df = VAL1.copy()

    if FILTER_POC:
        poc_min = p["pos_poc_min"]
        poc_max = p["pos_poc_max"]

        val_df = val_df[(val_df["diffPriceClosePoc_0_0"] >= poc_min) & (val_df["diffPriceClosePoc_0_0"] <= poc_max)]
        val1_df = val1_df[(val1_df["diffPriceClosePoc_0_0"] >= poc_min) & (val1_df["diffPriceClosePoc_0_0"] <= poc_max)]

        if trial.number % PRINT_EVERY == 0:
            train_pct = len(train_df) / len(TRAIN)
            print(f"{Fore.CYAN}POC filtré entre {poc_min} et {poc_max} : "
                  f"TR {train_pct:.1%} {'(utilisé)' if USE_TRAIN_IN_OPTIMIZATION else '(info)'}, "
                  f"V1 {len(val_df) / len(VAL):.1%}, "
                  f"V2 {len(val1_df) / len(VAL1):.1%}{Style.RESET_ALL}")

    # Masks pour VAL et VAL1
    m1_v = imbalance_detection_1(val_df, **p)
    m2_v = imbalance_detection_2(val_df, **p)
    m3_v = imbalance_detection_3(val_df, **p)

    m1_v1 = imbalance_detection_1(val1_df, **p)
    m2_v1 = imbalance_detection_2(val1_df, **p)
    m3_v1 = imbalance_detection_3(val1_df, **p)

    # Métriques VAL
    wr_v1, pct_v1, suc_v1, fail_v1, sess_v1 = _metrics(val_df, m1_v, val_len)
    wr_v2, pct_v2, suc_v2, fail_v2, sess_v2 = _metrics(val_df, m2_v, val_len)
    wr_v3, pct_v3, suc_v3, fail_v3, sess_v3 = _metrics(val_df, m3_v, val_len)

    wr_v, pct_v, suc_v, fail_v, sess_v, *_ = _metrics_combined(val_df, m1_v, m2_v, m3_v, val_len)
    metrics_detail_val = _metrics_exclusive(val_df, m1_v, m2_v, m3_v, val_len)

    # Métriques VAL1
    wr_v1_1, pct_v1_1, suc_v1_1, fail_v1_1, sess_v1_1 = _metrics(val1_df, m1_v1, val1_len)
    wr_v1_2, pct_v1_2, suc_v1_2, fail_v1_2, sess_v1_2 = _metrics(val1_df, m2_v1, val1_len)
    wr_v1_3, pct_v1_3, suc_v1_3, fail_v1_3, sess_v1_3 = _metrics(val1_df, m3_v1, val1_len)

    wr_v1, pct_v1, suc_v1, fail_v1, sess_v1, *_ = _metrics_combined(val1_df, m1_v1, m2_v1, m3_v1, val1_len)
    metrics_detail_val1 = _metrics_exclusive(val1_df, m1_v1, m2_v1, m3_v1, val1_len)

    # ====== VÉRIFICATION DES SEUILS ADAPTÉE ======
    # Pour les seuils, on ne considère que les datasets utilisés dans l'optimisation
    datasets_to_check = [(wr_v, pct_v), (wr_v1, pct_v1)]
    if USE_TRAIN_IN_OPTIMIZATION:
        datasets_to_check.append((wr_t, pct_t))

    for wr, pct in datasets_to_check:
        if wr < WINRATE_MIN or pct < PCT_TRADE_MIN:
            return FAILED_PENALTY

    # ====== CALCUL DES ÉCARTS ADAPTÉ ======
    if USE_TRAIN_IN_OPTIMIZATION:
        # Avec TRAIN : écarts entre les 3 datasets
        gap_wr_tv = abs(wr_t - wr_v)
        gap_pct_tv = abs(pct_t - pct_v)
        gap_wr_tv1 = abs(wr_t - wr_v1)
        gap_pct_tv1 = abs(pct_t - pct_v1)
        gap_wr_vv1 = abs(wr_v - wr_v1)
        gap_pct_vv1 = abs(pct_v - pct_v1)

        avg_gap_wr = (gap_wr_tv + gap_wr_tv1 + gap_wr_vv1) / 3
        avg_gap_pct = (gap_pct_tv + gap_pct_tv1 + gap_pct_vv1) / 3

        # Score avec TRAIN
        score = (ALPHA * (wr_t + wr_v + wr_v1) / 3 +
                 (1 - ALPHA) * (pct_t + pct_v + pct_v1) / 3 -
                 LAMBDA_WR * avg_gap_wr -
                 LAMBDA_PCT * avg_gap_pct)
    else:
        # Sans TRAIN : écart seulement entre VAL et VAL1
        gap_wr_vv1 = abs(wr_v - wr_v1)
        gap_pct_vv1 = abs(pct_v - pct_v1)

        avg_gap_wr = gap_wr_vv1
        avg_gap_pct = gap_pct_vv1

        # Score sans TRAIN
        score = (ALPHA * (wr_v + wr_v1) / 2 +
                 (1 - ALPHA) * (pct_v + pct_v1) / 2 -
                 LAMBDA_WR * avg_gap_wr -
                 LAMBDA_PCT * avg_gap_pct)

    global best_trial
    if score > best_trial["score"]:
        best_trial = {
            "number": trial.number,
            "score": score,
            # Métriques TRAIN (toujours stockées maintenant)
            "wr_t": wr_t, "pct_t": pct_t, "suc_t": suc_t, "fail_t": fail_t, "sess_t": sess_t,
            "wr_v": wr_v, "pct_v": pct_v, "suc_v": suc_v, "fail_v": fail_v, "sess_v": sess_v,
            "wr_v1": wr_v1, "pct_v1": pct_v1, "suc_v1": suc_v1, "fail_v1": fail_v1, "sess_v1": sess_v1,
            "wr_t1": wr_t1, "pct_t1": pct_t1, "suc_t1": suc_t1, "fail_t1": fail_t1, "sess_t1": sess_t1,
            "wr_t2": wr_t2, "pct_t2": pct_t2, "suc_t2": suc_t2, "fail_t2": fail_t2, "sess_t2": sess_t2,
            "wr_t3": wr_t3, "pct_t3": pct_t3, "suc_t3": suc_t3, "fail_t3": fail_t3, "sess_t3": sess_t3,
            "wr_v1": wr_v1, "pct_v1": pct_v1, "suc_v1": suc_v1, "fail_v1": fail_v1, "sess_v1": sess_v1,
            "wr_v2": wr_v2, "pct_v2": pct_v2, "suc_v2": suc_v2, "fail_v2": fail_v2, "sess_v2": sess_v2,
            "wr_v3": wr_v3, "pct_v3": pct_v3, "suc_v3": suc_v3, "fail_v3": fail_v3, "sess_v3": sess_v3,
            "wr_v1_1": wr_v1_1, "pct_v1_1": pct_v1_1, "suc_v1_1": suc_v1_1, "fail_v1_1": fail_v1_1,
            "sess_v1_1": sess_v1_1,
            "wr_v1_2": wr_v1_2, "pct_v1_2": pct_v1_2, "suc_v1_2": suc_v1_2, "fail_v1_2": fail_v1_2,
            "sess_v1_2": sess_v1_2,
            "wr_v1_3": wr_v1_3, "pct_v1_3": pct_v1_3, "suc_v1_3": suc_v1_3, "fail_v1_3": fail_v1_3,
            "sess_v1_3": sess_v1_3,
            "avg_gap_wr": avg_gap_wr,
            "avg_gap_pct": avg_gap_pct,
            # Métriques détaillées (toujours stockées pour TRAIN maintenant)
            "metrics_detail_train": metrics_detail_train,
            "metrics_detail_val": metrics_detail_val,
            "metrics_detail_val1": metrics_detail_val1,
            "params": p
        }

    # ====== AFFICHAGE LIVE ADAPTÉ ======
    if USE_TRAIN_IN_OPTIMIZATION:
        print(f"{trial.number:>6} | "
              f"TRAIN {Fore.GREEN}{wr_t:6.2%}{Style.RESET_ALL}/{pct_t:6.2%} | "
              f"VAL {Fore.GREEN}{wr_v:6.2%}{Style.RESET_ALL}/{pct_v:6.2%} | "
              f"VAL1 {Fore.GREEN}{wr_v1:6.2%}{Style.RESET_ALL}/{pct_v1:6.2%}",
              f"{Fore.GREEN}✔{Style.RESET_ALL}" if score > best_trial.get("score_old", -math.inf) else "")
    else:
        print(f"{trial.number:>6} | "
              f"TRAIN {Fore.YELLOW}{wr_t:6.2%}{Style.RESET_ALL}/{pct_t:6.2%}(info) | "
              f"VAL {Fore.GREEN}{wr_v:6.2%}{Style.RESET_ALL}/{pct_v:6.2%} | "
              f"VAL1 {Fore.GREEN}{wr_v1:6.2%}{Style.RESET_ALL}/{pct_v1:6.2%}",
              f"{Fore.GREEN}✔{Style.RESET_ALL}" if score > best_trial.get("score_old", -math.inf) else "")

    best_trial["score_old"] = score
    return score


# Fonction pour afficher les métriques détaillées
def print_detailed_metrics(dataset_name, metrics_detail):
    """Affiche les métriques détaillées par catégorie de trades"""

    wr_g, pct_g, suc_g, fail_g, sess_g = metrics_detail["global"]
    wr_1, pct_1, suc_1, fail_1, sess_1 = metrics_detail["cond1_only"]
    wr_2, pct_2, suc_2, fail_2, sess_2 = metrics_detail["cond2_only"]
    wr_3, pct_3, suc_3, fail_3, sess_3 = metrics_detail["cond3_only"]
    wr_12, pct_12, suc_12, fail_12, sess_12 = metrics_detail["cond12"]
    wr_13, pct_13, suc_13, fail_13, sess_13 = metrics_detail["cond13"]
    wr_23, pct_23, suc_23, fail_23, sess_23 = metrics_detail["cond23"]
    wr_123, pct_123, suc_123, fail_123, sess_123 = metrics_detail["cond123"]

    total_g = suc_g + fail_g
    total_1 = suc_1 + fail_1
    total_2 = suc_2 + fail_2
    total_3 = suc_3 + fail_3
    total_12 = suc_12 + fail_12
    total_13 = suc_13 + fail_13
    total_23 = suc_23 + fail_23
    total_123 = suc_123 + fail_123

    total_details = total_1 + total_2 + total_3 + total_12 + total_13 + total_23 + total_123

    print(f"\n    {Fore.CYAN}[DÉTAIL PAR CATÉGORIE DE TRADES - {dataset_name}]{Style.RESET_ALL}")
    print(f"    Condition 1 uniquement : WR={Fore.GREEN}{wr_1:.2%}{Style.RESET_ALL} | "
          f"Total={Fore.CYAN}{total_1}{Style.RESET_ALL}")
    print(f"    Condition 2 uniquement : WR={Fore.GREEN}{wr_2:.2%}{Style.RESET_ALL} | "
          f"Total={Fore.CYAN}{total_2}{Style.RESET_ALL}")
    print(f"    Condition 3 uniquement : WR={Fore.GREEN}{wr_3:.2%}{Style.RESET_ALL} | "
          f"Total={Fore.CYAN}{total_3}{Style.RESET_ALL}")
    print(f"    Conditions 1+2 : WR={Fore.GREEN}{wr_12:.2%}{Style.RESET_ALL} | "
          f"Total={Fore.CYAN}{total_12}{Style.RESET_ALL}")
    print(f"    Conditions 1+3 : WR={Fore.GREEN}{wr_13:.2%}{Style.RESET_ALL} | "
          f"Total={Fore.CYAN}{total_13}{Style.RESET_ALL}")
    print(f"    Conditions 2+3 : WR={Fore.GREEN}{wr_23:.2%}{Style.RESET_ALL} | "
          f"Total={Fore.CYAN}{total_23}{Style.RESET_ALL}")
    print(f"    Toutes conditions (1+2+3) : WR={Fore.GREEN}{wr_123:.2%}{Style.RESET_ALL} | "
          f"Total={Fore.CYAN}{total_123}{Style.RESET_ALL}")

    print(
        f"    {Fore.YELLOW}Vérification : {total_details} trades catégorisés vs {total_g} total global{Style.RESET_ALL}")
    if total_details != total_g:
        print(f"    {Fore.RED}⚠️ Anomalie détectée: La somme des détails ({total_details}) "
              f"ne correspond pas au total global ({total_g}){Style.RESET_ALL}")


# ───────────────────── HOLD‑OUT TEST ────────────────────────────
def calculate_test_metrics(params: dict):
    print(f"\n{Fore.CYAN}🧮  Calcul sur DATASET TEST, scénario {DIRECTION} {Style.RESET_ALL}\n")

    # ====== RAPPEL DES STATISTIQUES FINALES SUR TOUS LES DATASETS ======
    def calculate_dataset_stats(df, df_name, original_len):
        """Calcule et affiche les stats finales d'un dataset"""
        if df is None:
            return

        dataset_df = df.copy()

        if FILTER_POC and "pos_poc_min" in params and "pos_poc_max" in params:
            poc_min = params["pos_poc_min"]
            poc_max = params["pos_poc_max"]
            poc_min, poc_max = min(poc_min, poc_max), max(poc_min, poc_max)
            dataset_df = dataset_df[
                (dataset_df["diffPriceClosePoc_0_0"] >= poc_min) & (dataset_df["diffPriceClosePoc_0_0"] <= poc_max)]

        m1 = imbalance_detection_1(dataset_df, **params)
        m2 = imbalance_detection_2(dataset_df, **params)
        m3 = imbalance_detection_3(dataset_df, **params)

        wr_u, pct_u, suc_u, fail_u, sess_u, *_ = _metrics_combined(dataset_df, m1, m2, m3, original_len)

        # ====== MODIFICATION: Indiquer si TRAIN est utilisé pour l'optimisation ======
        if df_name == "TRAIN" and not USE_TRAIN_IN_OPTIMIZATION:
            info_suffix = f" {Fore.YELLOW}(info seulement){Style.RESET_ALL}"
        else:
            info_suffix = ""

        print(f"    {df_name}{info_suffix}: WR={Fore.GREEN}{wr_u:.2%}{Style.RESET_ALL}  pct={pct_u:.2%}  "
              f"✓{Fore.GREEN}{suc_u}{Style.RESET_ALL} ✗{Fore.RED}{fail_u}{Style.RESET_ALL}  "
              f"Total={Fore.CYAN}{suc_u + fail_u}{Style.RESET_ALL} (sessions: {sess_u})")

    print(f"{Fore.YELLOW}📊 RAPPEL - Statistiques finales avec paramètres optimaux:{Style.RESET_ALL}")

    if TRAIN is not None:
        calculate_dataset_stats(TRAIN, "TRAIN", len(TRAIN))
    calculate_dataset_stats(VAL, "VAL  ", len(VAL))
    calculate_dataset_stats(VAL1, "VAL1 ", len(VAL1))

    print(f"\n{Fore.YELLOW}🎯 RÉSULTATS SUR DATASET TEST:{Style.RESET_ALL}")

    test_len = len(TEST)
    test_df = TEST.copy()

    if FILTER_POC and "pos_poc_min" in params and "pos_poc_max" in params:
        poc_min = params["pos_poc_min"]
        poc_max = params["pos_poc_max"]

        poc_min, poc_max = min(poc_min, poc_max), max(poc_min, poc_max)

        test_df = test_df[(test_df["diffPriceClosePoc_0_0"] >= poc_min) & (test_df["diffPriceClosePoc_0_0"] <= poc_max)]

        print(f"{Fore.CYAN}POC filtré entre {poc_min} et {poc_max} : "
              f"TEST {len(test_df) / len(TEST):.1%} ({len(test_df)}/{len(TEST)}){Style.RESET_ALL}")

    m1 = imbalance_detection_1(test_df, **params)
    m2 = imbalance_detection_2(test_df, **params)
    m3 = imbalance_detection_3(test_df, **params)

    wr_1, pct_1, suc_1, fail_1, sess_1 = _metrics(test_df, m1, test_len)
    wr_2, pct_2, suc_2, fail_2, sess_2 = _metrics(test_df, m2, test_len)
    wr_3, pct_3, suc_3, fail_3, sess_3 = _metrics(test_df, m3, test_len)

    wr_u, pct_u, suc_u, fail_u, sess_u, *_ = _metrics_combined(test_df, m1, m2, m3, test_len)

    metrics_detail_test = _metrics_exclusive(test_df, m1, m2, m3, test_len)

    print(f"\n{Fore.YELLOW}--- Détail par condition ---{Style.RESET_ALL}")
    print(f"Condition 1: WR={Fore.GREEN}{wr_1:.2%}{Style.RESET_ALL}  pct={pct_1:.2%}  "
          f"✓{Fore.GREEN}{suc_1}{Style.RESET_ALL} ✗{Fore.RED}{fail_1}{Style.RESET_ALL}  "
          f"Total={Fore.CYAN}{suc_1 + fail_1}{Style.RESET_ALL} (sessions: {sess_1})")
    print(f"Condition 2: WR={Fore.GREEN}{wr_2:.2%}{Style.RESET_ALL}  pct={pct_2:.2%}  "
          f"✓{Fore.GREEN}{suc_2}{Style.RESET_ALL} ✗{Fore.RED}{fail_2}{Style.RESET_ALL}  "
          f"Total={Fore.CYAN}{suc_2 + fail_2}{Style.RESET_ALL} (sessions: {sess_2})")
    print(f"Condition 3: WR={Fore.GREEN}{wr_3:.2%}{Style.RESET_ALL}  pct={pct_3:.2%}  "
          f"✓{Fore.GREEN}{suc_3}{Style.RESET_ALL} ✗{Fore.RED}{fail_3}{Style.RESET_ALL}  "
          f"Total={Fore.CYAN}{suc_3 + fail_3}{Style.RESET_ALL} (sessions: {sess_3})")

    print(f"\n{Fore.YELLOW}--- Résultat combiné (union) ---{Style.RESET_ALL}")
    print(f"TEST: WR={Fore.GREEN}{wr_u:.2%}{Style.RESET_ALL}  pct={pct_u:.2%}  "
          f"✓{Fore.GREEN}{suc_u}{Style.RESET_ALL} ✗{Fore.RED}{fail_u}{Style.RESET_ALL}  "
          f"Total={Fore.CYAN}{suc_u + fail_u}{Style.RESET_ALL} (sessions: {sess_u})")

    print_detailed_metrics("TEST", metrics_detail_test)

    is_valid = (wr_u >= WINRATE_MIN and pct_u >= PCT_TRADE_MIN)
    if is_valid:
        print(f"{Fore.GREEN}✅ VALIDE{Style.RESET_ALL}\n\n")
    else:
        print(f"{Fore.RED}❌ REJET{Style.RESET_ALL}")

    return wr_u, pct_u, suc_u, fail_u, sess_u


# ───────────────────── KEYBOARD LISTENING ───────────────────────
RUN_TEST = False


def on_press(key):
    global RUN_TEST
    try:
        if key.char == '&':
            print(f"\n{Fore.YELLOW}🧪  Test demandé via '&'{Style.RESET_ALL}")
            RUN_TEST = True
    except AttributeError:
        pass


def start_keyboard_listener():
    listener = keyboard.Listener(on_press=on_press)
    listener.daemon = True
    listener.start()
    return listener


# ───────────────────── MAIN LOOP (CORRECTION COMPLÈTE) ────────────────────────────────
def main() -> None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    last_best_shown = None

    listener = start_keyboard_listener()
    print(
        f"{Fore.CYAN}Écouteur clavier démarré - appuyez sur '&' à tout moment pour tester sur le dataset TEST{Style.RESET_ALL}")

    # Affichage des colonnes utilisées
    print(f"\n{Fore.YELLOW}═══════════════════════════════════════════════════════════{Style.RESET_ALL}")
    print(f"{Fore.CYAN}📊 Configuration {DIRECTION.upper()} :{Style.RESET_ALL}")
    print(f"   • Volume: {VOLUME_COL}")
    print(f"   • Imbalance: {IMBALANCE_COL}")
    if not USE_TRAIN_IN_OPTIMIZATION:
        print(f"   • {Fore.YELLOW}DATASET TRAIN: TESTÉ POUR INFO SEULEMENT{Style.RESET_ALL}")
    else:
        print(f"   • {Fore.GREEN}DATASET TRAIN: UTILISÉ POUR L'OPTIMISATION{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}═══════════════════════════════════════════════════════════{Style.RESET_ALL}\n")

    for done in range(1, N_TRIALS + 1):
        study.optimize(objective, n_trials=1)

        if RUN_TEST or done % PRINT_EVERY == 0:
            globals()["RUN_TEST"] = False
            if done % PRINT_EVERY == 0 and not RUN_TEST:
                print(f"\n{Fore.YELLOW}🧪  Test automatique (trial {done}){Style.RESET_ALL}")

            # ✅ CORRECTION : Utiliser best_trial["params"] au lieu de study.best_params
            if best_trial.get("params"):
                calculate_test_metrics(best_trial["params"])
            else:
                print(f"{Fore.RED}⚠️ Aucun meilleur trial trouvé encore{Style.RESET_ALL}")

        if best_trial.get("number") is not None:
            print(f"Best trial {best_trial['number']}  value {Fore.GREEN}{best_trial['score']:.4f}{Style.RESET_ALL}",
                  end="\r")

        if (done % PRINT_EVERY == 0 or best_trial.get("number") != last_best_shown):
            bt = best_trial
            print(
                f"\n\n{Fore.YELLOW}*** BEST so far for {DIRECTION} with {param_summary} ▸ trial {bt['number']}  score={Fore.GREEN}{bt['score']:.4f}{Style.RESET_ALL}")
            print(f"    {Fore.CYAN}[Direction: {DIRECTION.upper()} - Imbalance Low High Detection]{Style.RESET_ALL}")

            print(f"    {Fore.CYAN}[GLOBAL - Trades uniques]{Style.RESET_ALL}")

            # ====== MODIFICATION: AFFICHAGE ADAPTÉ SELON L'UTILISATION DE TRAIN ======
            if USE_TRAIN_IN_OPTIMIZATION:
                print(f"    TRAIN WR {Fore.GREEN}{bt['wr_t']:.2%}{Style.RESET_ALL} | "
                      f"pct {bt['pct_t']:.2%} | "
                      f"✓{Fore.GREEN}{bt['suc_t']}{Style.RESET_ALL} "
                      f"✗{Fore.RED}{bt['fail_t']}{Style.RESET_ALL} "
                      f"Total={Fore.CYAN}{bt['suc_t'] + bt['fail_t']}{Style.RESET_ALL} (sessions: {bt['sess_t']})")
            else:
                print(f"    TRAIN WR {Fore.GREEN}{bt['wr_t']:.2%}{Style.RESET_ALL} | "
                      f"pct {bt['pct_t']:.2%} | "
                      f"✓{Fore.GREEN}{bt['suc_t']}{Style.RESET_ALL} "
                      f"✗{Fore.RED}{bt['fail_t']}{Style.RESET_ALL} "
                      f"Total={Fore.CYAN}{bt['suc_t'] + bt['fail_t']}{Style.RESET_ALL} (sessions: {bt['sess_t']}) "
                      f"{Fore.YELLOW}(info seulement){Style.RESET_ALL}")

            print(f"    VAL WR {Fore.GREEN}{bt['wr_v']:.2%}{Style.RESET_ALL} | "
                  f"pct {bt['pct_v']:.2%} | "
                  f"✓{Fore.GREEN}{bt['suc_v']}{Style.RESET_ALL} "
                  f"✗{Fore.RED}{bt['fail_v']}{Style.RESET_ALL} "
                  f"Total={Fore.CYAN}{bt['suc_v'] + bt['fail_v']}{Style.RESET_ALL} (sessions: {bt['sess_v']})")

            print(f"    VAL1 WR {Fore.GREEN}{bt['wr_v1']:.2%}{Style.RESET_ALL} | "
                  f"pct {bt['pct_v1']:.2%} | "
                  f"✓{Fore.GREEN}{bt['suc_v1']}{Style.RESET_ALL} "
                  f"✗{Fore.RED}{bt['fail_v1']}{Style.RESET_ALL} "
                  f"Total={Fore.CYAN}{bt['suc_v1'] + bt['fail_v1']}{Style.RESET_ALL} (sessions: {bt['sess_v1']})")

            print(f"    Gap WR moyen: {Fore.YELLOW}{bt['avg_gap_wr']:.2%}{Style.RESET_ALL} | "
                  f"Gap PCT moyen: {Fore.YELLOW}{bt['avg_gap_pct']:.2%}{Style.RESET_ALL}")

            # ====== MODIFICATION: Toujours afficher les métriques détaillées TRAIN ======
            if bt['metrics_detail_train']:
                train_label = "TRAIN" if USE_TRAIN_IN_OPTIMIZATION else "TRAIN (info seulement)"
                print_detailed_metrics(train_label, bt['metrics_detail_train'])

            if bt['metrics_detail_val']:
                print_detailed_metrics("VAL", bt['metrics_detail_val'])

            if bt['metrics_detail_val1']:
                print_detailed_metrics("VAL1", bt['metrics_detail_val1'])

            # Affichage des paramètres avec noms génériques
            print(f"\n    {Fore.CYAN}[Paramètres optimaux]{Style.RESET_ALL}")
            print(f"    params ➜ {Fore.MAGENTA}{bt['params']}{Style.RESET_ALL}\n")

            last_best_shown = best_trial["number"]

    print(f"\n{Fore.YELLOW}🔚  Fin des essais Optuna.{Style.RESET_ALL}")

    # ═══════════════════ SYNTHÈSE FINALE COMPLÈTE ═══════════════════
    print(f"\n{Fore.YELLOW}{'=' * 80}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}🏆 SYNTHÈSE FINALE - MEILLEUR RÉSULTAT{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{'=' * 80}{Style.RESET_ALL}")

    if best_trial.get("params"):
        bt = best_trial
        print(
            f"\n{Fore.YELLOW}*** BEST FINAL for {DIRECTION} with {param_summary} ▸ trial {bt['number']}  score={Fore.GREEN}{bt['score']:.4f}{Style.RESET_ALL}")
        print(f"    {Fore.CYAN}[Direction: {DIRECTION.upper()} - Imbalance Low High Detection]{Style.RESET_ALL}")

        print(f"\n    {Fore.CYAN}[RÉSULTATS SUR DATASETS D'ENTRAÎNEMENT/VALIDATION]{Style.RESET_ALL}")

        # ====== MODIFICATION: AFFICHAGE ADAPTÉ SELON L'UTILISATION DE TRAIN ======
        if USE_TRAIN_IN_OPTIMIZATION:
            print(f"    TR WR {Fore.GREEN}{bt['wr_t']:.2%}{Style.RESET_ALL} | "
                  f"pct {bt['pct_t']:.2%} | "
                  f"✓{Fore.GREEN}{bt['suc_t']}{Style.RESET_ALL} "
                  f"✗{Fore.RED}{bt['fail_t']}{Style.RESET_ALL} "
                  f"Total={Fore.CYAN}{bt['suc_t'] + bt['fail_t']}{Style.RESET_ALL} (sessions: {bt['sess_t']})")
        else:
            print(f"    TR WR {Fore.GREEN}{bt['wr_t']:.2%}{Style.RESET_ALL} | "
                  f"pct {bt['pct_t']:.2%} | "
                  f"✓{Fore.GREEN}{bt['suc_t']}{Style.RESET_ALL} "
                  f"✗{Fore.RED}{bt['fail_t']}{Style.RESET_ALL} "
                  f"Total={Fore.CYAN}{bt['suc_t'] + bt['fail_t']}{Style.RESET_ALL} (sessions: {bt['sess_t']}) "
                  f"{Fore.YELLOW}(info seulement){Style.RESET_ALL}")

        print(f"    V1 WR {Fore.GREEN}{bt['wr_v']:.2%}{Style.RESET_ALL} | "
              f"pct {bt['pct_v']:.2%} | "
              f"✓{Fore.GREEN}{bt['suc_v']}{Style.RESET_ALL} "
              f"✗{Fore.RED}{bt['fail_v']}{Style.RESET_ALL} "
              f"Total={Fore.CYAN}{bt['suc_v'] + bt['fail_v']}{Style.RESET_ALL} (sessions: {bt['sess_v']})")

        print(f"    V2 WR {Fore.GREEN}{bt['wr_v1']:.2%}{Style.RESET_ALL} | "
              f"pct {bt['pct_v1']:.2%} | "
              f"✓{Fore.GREEN}{bt['suc_v1']}{Style.RESET_ALL} "
              f"✗{Fore.RED}{bt['fail_v1']}{Style.RESET_ALL} "
              f"Total={Fore.CYAN}{bt['suc_v1'] + bt['fail_v1']}{Style.RESET_ALL} (sessions: {bt['sess_v1']})")

        print(f"    Gap WR moyen: {Fore.YELLOW}{bt['avg_gap_wr']:.2%}{Style.RESET_ALL} | "
              f"Gap PCT moyen: {Fore.YELLOW}{bt['avg_gap_pct']:.2%}{Style.RESET_ALL}")

        # Affichage des paramètres optimaux
        print(f"\n    {Fore.CYAN}[Paramètres optimaux]{Style.RESET_ALL}")
        print(f"    params ➜ {Fore.MAGENTA}{bt['params']}{Style.RESET_ALL}")

        print(f"\n{Fore.YELLOW}{'=' * 80}{Style.RESET_ALL}")

        # ✅ CORRECTION PRINCIPALE : Utiliser les bons paramètres pour le test final
        calculate_test_metrics(best_trial["params"])
    else:
        print(
            f"{Fore.RED}❌ Aucun meilleur trial trouvé - impossible de calculer les métriques finales{Style.RESET_ALL}")

    print(f"\n{Fore.YELLOW}{'=' * 80}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}🏁 FIN DE L'OPTIMISATION{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{'=' * 80}{Style.RESET_ALL}")


if __name__ == "__main__":
    main()