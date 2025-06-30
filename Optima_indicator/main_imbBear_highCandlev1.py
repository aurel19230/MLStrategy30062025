
# -*- coding: utf-8 -*-
"""optuna_bear_imbalance.py – version 4
====================================
- Intègre **trois** conditions distinctes avec des plages de sampling différentes.
- Utilise **deux** datasets de validation pour une plus grande robustesse.
- Raccourci clavier « & » (utilisant pynput) pour déclencher un calcul immédiat
  sur le jeu TEST pendant l'optimisation.
- Affichage synthétique à chaque trial + rapport détaillé périodique / sur
  nouveau meilleur score.
- Affichage coloré des résultats avec colorama
- Ajout du nombre de sessions à côté du Total dans l'affichage des résultats
- Ajout d'un affichage détaillé des trades par catégorie exclusive pour chaque dataset
- Adapté pour optimiser une stratégie basée sur des imbalances baissières en haut de bougie
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
DIR = Path(r"C:/Users/aulac/OneDrive/Documents/Trading/VisualStudioProject/"
           r"Sierra chart/xTickReversal/simu/5_0_5TP_6SL/merge")

CSV_TRAIN = DIR / "Step5_5_0_5TP_6SL_010124_150525_extractOnlyFullSession_OnlyShort_feat__split1_01012024_01052024.csv"
CSV_TEST = DIR / "Step5_5_0_5TP_6SL_010124_150525_extractOnlyFullSession_OnlyShort_feat__split2_01052024_01102024.csv"
CSV_VAL1 = DIR / "Step5_5_0_5TP_6SL_010124_150525_extractOnlyFullSession_OnlyShort_feat__split3_01102024_28022025.csv"
CSV_VAL = DIR / "Step5_5_0_5TP_6SL_010124_150525_extractOnlyFullSession_OnlyShort_feat__split4_02032025_14052025.csv"

WINRATE_MIN = 0.52  # WR minimum acceptable
PCT_TRADE_MIN = 0.01  # % de candles tradées minimum
ALPHA = 0.70  # poids du WR dans le score
N_TRIALS = 10_000
PRINT_EVERY = 50
FAILED_PENALTY = -0.001

# Gap penalties
LAMBDA_WR = 0  # 0.60
LAMBDA_PCT = 0  # 0.20

# ── Bornes par condition ────────────────────────────────────────
# Nouvelles bornes pour askVolHigh et bear_imbalance
ASK_MIN_1, ASK_MAX_1 = 1, 6  # Similaire aux bornes originales pour bid
BEAR_MIN_1, BEAR_MAX_1 = 2, 10  # Similaire aux bornes originales pour bull_imbalance

ASK_MIN_2, ASK_MAX_2 = 6, 25   # Chevauchement partiel avec Condition 1
BEAR_MIN_2, BEAR_MAX_2 = 2, 15  # Valeurs plus élevées, explorées différemment

ASK_MIN_3, ASK_MAX_3 = 25, 90   # Élargissement pour capturer plus de trades
BEAR_MIN_3, BEAR_MAX_3 = 1.5, 6 # Élargissement pour améliorer la stabilité

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

    print(f"{path.name} ➜ encodage détecté: {encoding}")

    # Chargement robuste
    df = pd.read_csv(path, sep=";", encoding=encoding, parse_dates=["date"], low_memory=False)

    # 🔧 Correction de SessionStartEnd
    df["SessionStartEnd"] = pd.to_numeric(df["SessionStartEnd"], errors="coerce")
    df = df.dropna(subset=["SessionStartEnd"])
    df["SessionStartEnd"] = df["SessionStartEnd"].astype(int)

    # 🔍 Vérif des valeurs possibles
    print(f"{path.name} ➜ uniques SessionStartEnd: {df['SessionStartEnd'].unique()}")

    # 📊 Compter les sessions avant filtre class_binaire
    nb_start = (df["SessionStartEnd"] == 10).sum()
    nb_end = (df["SessionStartEnd"] == 20).sum()
    nb_sessions = min(nb_start, nb_end)

    if nb_start != nb_end:
        print(
            f"{Fore.YELLOW}⚠️ Incohérence sessions: {nb_start} débuts vs {nb_end} fins dans {path.name}{Style.RESET_ALL}")
    else:
        print(f"{Fore.GREEN}✔ {nb_sessions} sessions complètes détectées dans {path.name}{Style.RESET_ALL}")

    # ✅ Numérotation des sessions (avant filtrage)
    df["session_id"] = (df["SessionStartEnd"] == 10).cumsum().astype("int32")

    # 🧼 Seulement maintenant : filtrage de la cible
    df = df[df["class_binaire"].isin([0, 1])].copy()
    df.reset_index(drop=True, inplace=True)

    return df, nb_sessions


TRAIN, TRAIN_SESSIONS = load_csv(CSV_TRAIN)
VAL, VAL_SESSIONS = load_csv(CSV_VAL)
VAL1, VAL1_SESSIONS = load_csv(CSV_VAL1)
TEST, TEST_SESSIONS = load_csv(CSV_TEST)

# Vérification de la cohérence des sessions (nombre de 10 == nombre de 20)
for name, df in zip(["TRAIN", "VAL", "VAL1", "TEST"], [TRAIN, VAL, VAL1, TEST]):
    start_sessions = df[df["SessionStartEnd"] == 10].shape[0]
    end_sessions = df[df["SessionStartEnd"] == 20].shape[0]
    if start_sessions != end_sessions:
        print(f"ATTENTION: {name} - Incohérence dans les sessions: {start_sessions} débuts vs {end_sessions} fins")

# Affichage des informations sur les datasets
for lbl, d, sessions in zip(("TRAIN", "VAL", "VAL1", "TEST"),
                            (TRAIN, VAL, VAL1, TEST),
                            (TRAIN_SESSIONS, VAL_SESSIONS, VAL1_SESSIONS, TEST_SESSIONS)):
    print(f"{lbl:<5} | lignes={len(d):,}  WR brut={(d['class_binaire'] == 1).mean():.2%}  Sessions={sessions}")
print("—————————————————————————————————————————————————————————————\n")


# ───────────────────── MASK BUILDERS ────────────────────────────

def imbalance_high_rev(df: pd.DataFrame, *, askVolHigh: float, bear_imbalance_high_1: float, **kwargs) -> pd.Series:
    """Condition 1: Imbalance baissière en haut de bougie"""
    return (df["askVolHigh"] > askVolHigh) & (df["bear_imbalance_high_1"] > bear_imbalance_high_1)


def imbalance_high_rev_2(df: pd.DataFrame, *, askVolHigh_2Cond: float, bear_imbalance_high_1_2Cond: float,
                         **kwargs) -> pd.Series:
    """Condition 2: Imbalance baissière en haut de bougie avec paramètres différents"""
    return (df["askVolHigh"] > askVolHigh_2Cond) & (df["bear_imbalance_high_1"] > bear_imbalance_high_1_2Cond)


def imbalance_high_rev_3(df: pd.DataFrame, *, askVolHigh_3Cond: float, bear_imbalance_high_1_3Cond: float,
                         **kwargs) -> pd.Series:
    """Condition 3: Imbalance baissière en haut de bougie avec paramètres encore différents"""
    return (df["askVolHigh"] > askVolHigh_3Cond) & (df["bear_imbalance_high_1"] > bear_imbalance_high_1_3Cond)


# ───────────────────── METRICS HELPERS ──────────────────────────

def _metrics(df: pd.DataFrame, mask: pd.Series) -> tuple[float, float, int, int, int]:
    """Calcule les métriques avec le nombre de sessions couvertes"""
    sub = df.loc[mask]
    if sub.empty:
        return 0.0, 0.0, 0, 0, 0
    wins = int((sub["class_binaire"] == 1).sum())
    total = len(sub)
    # Calculer le nombre de sessions uniques où il y a des trades
    sessions_covered = sub["session_id"].nunique()
    return wins / total, total / len(df), wins, total - wins, sessions_covered


def _metrics_combined(df: pd.DataFrame, m1: pd.Series, m2: pd.Series, m3: pd.Series):
    """Calcule les métriques combinées avec le nombre de sessions couvertes"""
    m_u = m1 | m2 | m3
    m_12 = m1 & m2
    m_13 = m1 & m3
    m_23 = m2 & m3
    m_123 = m1 & m2 & m3
    return _metrics(df, m_u) + _metrics(df, m_12) + _metrics(df, m_13) + _metrics(df, m_23) + _metrics(df, m_123)


def _metrics_exclusive(df: pd.DataFrame, m1: pd.Series, m2: pd.Series, m3: pd.Series):
    """Calcule des métriques détaillées montrant les trades uniques et les chevauchements"""
    # Catégories exclusives
    m1_only = m1 & ~m2 & ~m3  # Uniquement condition 1
    m2_only = ~m1 & m2 & ~m3  # Uniquement condition 2
    m3_only = ~m1 & ~m2 & m3  # Uniquement condition 3
    m12_only = m1 & m2 & ~m3  # Conditions 1 et 2 seulement
    m13_only = m1 & ~m2 & m3  # Conditions 1 et 3 seulement
    m23_only = ~m1 & m2 & m3  # Conditions 2 et 3 seulement
    m123 = m1 & m2 & m3  # Toutes les conditions

    # Global (union)
    m_u = m1 | m2 | m3

    # Calcul des métriques pour chaque catégorie
    metrics_global = _metrics(df, m_u)
    metrics_1_only = _metrics(df, m1_only)
    metrics_2_only = _metrics(df, m2_only)
    metrics_3_only = _metrics(df, m3_only)
    metrics_12_only = _metrics(df, m12_only)
    metrics_13_only = _metrics(df, m13_only)
    metrics_23_only = _metrics(df, m23_only)
    metrics_123 = _metrics(df, m123)

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
# Initialisation de best_trial avec des valeurs par défaut
best_trial = {
    "score": -math.inf,
    "number": None,  # Initialisation de number à None
    "score_old": -math.inf,
    # Autres champs initialisés à des valeurs par défaut
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

    # Nouvelles métriques détaillées
    "metrics_detail_train": None,
    "metrics_detail_val": None,
    "metrics_detail_val1": None,

    "params": {}
}


def objective(trial: optuna.trial.Trial) -> float:
    # Paramètres spécifiques aux trois conditions
    p = {
        "askVolHigh": trial.suggest_int("askVolHigh", ASK_MIN_1, ASK_MAX_1),
        "bear_imbalance_high_1": trial.suggest_float("bear_imbalance_high_1", BEAR_MIN_1, BEAR_MAX_1),
        "askVolHigh_2Cond": trial.suggest_int("askVolHigh_2Cond", ASK_MIN_2, ASK_MAX_2),
        "bear_imbalance_high_1_2Cond": trial.suggest_float("bear_imbalance_high_1_2Cond", BEAR_MIN_2, BEAR_MAX_2),
        "askVolHigh_3Cond": trial.suggest_int("askVolHigh_3Cond", ASK_MIN_3, ASK_MAX_3),
        "bear_imbalance_high_1_3Cond": trial.suggest_float("bear_imbalance_high_1_3Cond", BEAR_MIN_3, BEAR_MAX_3),
    }

    # Masks pour le dataset TRAIN
    m1_t = imbalance_high_rev(TRAIN, **p)
    m2_t = imbalance_high_rev_2(TRAIN, **p)
    m3_t = imbalance_high_rev_3(TRAIN, **p)

    # Masks pour le premier dataset de validation VAL
    m1_v = imbalance_high_rev(VAL, **p)
    m2_v = imbalance_high_rev_2(VAL, **p)
    m3_v = imbalance_high_rev_3(VAL, **p)

    # Masks pour le second dataset de validation VAL1
    m1_v1 = imbalance_high_rev(VAL1, **p)
    m2_v1 = imbalance_high_rev_2(VAL1, **p)
    m3_v1 = imbalance_high_rev_3(VAL1, **p)

    # ────────── Métriques TRAIN ──────────────
    # Métriques individuelles par condition avec sessions
    wr_t1, pct_t1, suc_t1, fail_t1, sess_t1 = _metrics(TRAIN, m1_t)
    wr_t2, pct_t2, suc_t2, fail_t2, sess_t2 = _metrics(TRAIN, m2_t)
    wr_t3, pct_t3, suc_t3, fail_t3, sess_t3 = _metrics(TRAIN, m3_t)

    # Métrique combinée avec sessions
    wr_t, pct_t, suc_t, fail_t, sess_t, *_ = _metrics_combined(TRAIN, m1_t, m2_t, m3_t)

    # Métriques détaillées par catégorie
    metrics_detail_train = _metrics_exclusive(TRAIN, m1_t, m2_t, m3_t)

    # ────────── Métriques VAL (Validation 1) ──────────────
    # Métriques individuelles par condition avec sessions
    wr_v1, pct_v1, suc_v1, fail_v1, sess_v1 = _metrics(VAL, m1_v)
    wr_v2, pct_v2, suc_v2, fail_v2, sess_v2 = _metrics(VAL, m2_v)
    wr_v3, pct_v3, suc_v3, fail_v3, sess_v3 = _metrics(VAL, m3_v)

    # Métrique combinée avec sessions
    wr_v, pct_v, suc_v, fail_v, sess_v, *_ = _metrics_combined(VAL, m1_v, m2_v, m3_v)

    # Métriques détaillées par catégorie
    metrics_detail_val = _metrics_exclusive(VAL, m1_v, m2_v, m3_v)

    # ────────── Métriques VAL1 (Validation 2) ──────────────
    # Métriques individuelles par condition avec sessions
    wr_v1_1, pct_v1_1, suc_v1_1, fail_v1_1, sess_v1_1 = _metrics(VAL1, m1_v1)
    wr_v1_2, pct_v1_2, suc_v1_2, fail_v1_2, sess_v1_2 = _metrics(VAL1, m2_v1)
    wr_v1_3, pct_v1_3, suc_v1_3, fail_v1_3, sess_v1_3 = _metrics(VAL1, m3_v1)

    # Métrique combinée avec sessions
    wr_v1, pct_v1, suc_v1, fail_v1, sess_v1, *_ = _metrics_combined(VAL1, m1_v1, m2_v1, m3_v1)

    # Métriques détaillées par catégorie
    metrics_detail_val1 = _metrics_exclusive(VAL1, m1_v1, m2_v1, m3_v1)

    # Quick threshold veto - Vérification des seuils sur les trois datasets
    if (wr_t < WINRATE_MIN or pct_t < PCT_TRADE_MIN or
            wr_v < WINRATE_MIN or pct_v < PCT_TRADE_MIN or
            wr_v1 < WINRATE_MIN or pct_v1 < PCT_TRADE_MIN):
        return FAILED_PENALTY

    # Calcul des écarts entre les jeux de données
    gap_wr_tv = abs(wr_t - wr_v)
    gap_pct_tv = abs(pct_t - pct_v)

    gap_wr_tv1 = abs(wr_t - wr_v1)
    gap_pct_tv1 = abs(pct_t - pct_v1)

    gap_wr_vv1 = abs(wr_v - wr_v1)
    gap_pct_vv1 = abs(pct_v - pct_v1)

    # Moyenne des écarts
    avg_gap_wr = (gap_wr_tv + gap_wr_tv1 + gap_wr_vv1) / 3
    avg_gap_pct = (gap_pct_tv + gap_pct_tv1 + gap_pct_vv1) / 3

    # Score qui considère les trois datasets et les écarts moyens
    score = (ALPHA * (wr_t + wr_v + wr_v1) / 3 +
             (1 - ALPHA) * (pct_t + pct_v + pct_v1) / 3 -
             LAMBDA_WR * avg_gap_wr -
             LAMBDA_PCT * avg_gap_pct)

    global best_trial
    if score > best_trial["score"]:
        best_trial = {
            "number": trial.number,
            "score": score,
            # Métriques combinées - TRAIN
            "wr_t": wr_t, "pct_t": pct_t, "suc_t": suc_t, "fail_t": fail_t, "sess_t": sess_t,
            # Métriques combinées - VAL
            "wr_v": wr_v, "pct_v": pct_v, "suc_v": suc_v, "fail_v": fail_v, "sess_v": sess_v,
            # Métriques combinées - VAL1
            "wr_v1": wr_v1, "pct_v1": pct_v1, "suc_v1": suc_v1, "fail_v1": fail_v1, "sess_v1": sess_v1,

            # Métriques par condition - TRAIN
            "wr_t1": wr_t1, "pct_t1": pct_t1, "suc_t1": suc_t1, "fail_t1": fail_t1, "sess_t1": sess_t1,
            "wr_t2": wr_t2, "pct_t2": pct_t2, "suc_t2": suc_t2, "fail_t2": fail_t2, "sess_t2": sess_t2,
            "wr_t3": wr_t3, "pct_t3": pct_t3, "suc_t3": suc_t3, "fail_t3": fail_t3, "sess_t3": sess_t3,

            # Métriques par condition - VAL
            "wr_v1": wr_v1, "pct_v1": pct_v1, "suc_v1": suc_v1, "fail_v1": fail_v1, "sess_v1": sess_v1,
            "wr_v2": wr_v2, "pct_v2": pct_v2, "suc_v2": suc_v2, "fail_v2": fail_v2, "sess_v2": sess_v2,
            "wr_v3": wr_v3, "pct_v3": pct_v3, "suc_v3": suc_v3, "fail_v3": fail_v3, "sess_v3": sess_v3,

            # Métriques par condition - VAL1
            "wr_v1_1": wr_v1_1, "pct_v1_1": pct_v1_1, "suc_v1_1": suc_v1_1, "fail_v1_1": fail_v1_1,
            "sess_v1_1": sess_v1_1,
            "wr_v1_2": wr_v1_2, "pct_v1_2": pct_v1_2, "suc_v1_2": suc_v1_2, "fail_v1_2": fail_v1_2,
            "sess_v1_2": sess_v1_2,
            "wr_v1_3": wr_v1_3, "pct_v1_3": pct_v1_3, "suc_v1_3": suc_v1_3, "fail_v1_3": fail_v1_3,
            "sess_v1_3": sess_v1_3,

            # Écarts moyens
            "avg_gap_wr": avg_gap_wr,
            "avg_gap_pct": avg_gap_pct,

            # Métriques détaillées
            "metrics_detail_train": metrics_detail_train,
            "metrics_detail_val": metrics_detail_val,
            "metrics_detail_val1": metrics_detail_val1,

            "params": p
        }

    # Live print avec les trois datasets
    print(f"{trial.number:>6} | "
          f"TR {Fore.GREEN}{wr_t:6.2%}{Style.RESET_ALL}/{pct_t:6.2%} | "
          f"V1 {Fore.GREEN}{wr_v:6.2%}{Style.RESET_ALL}/{pct_v:6.2%} | "
          f"V2 {Fore.GREEN}{wr_v1:6.2%}{Style.RESET_ALL}/{pct_v1:6.2%}",
          f"{Fore.GREEN}✔{Style.RESET_ALL}" if score > best_trial.get("score_old", -math.inf) else "")

    best_trial["score_old"] = score  # helper for symbol
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

    # Calculer les totaux pour chaque catégorie
    total_g = suc_g + fail_g
    total_1 = suc_1 + fail_1
    total_2 = suc_2 + fail_2
    total_3 = suc_3 + fail_3
    total_12 = suc_12 + fail_12
    total_13 = suc_13 + fail_13
    total_23 = suc_23 + fail_23
    total_123 = suc_123 + fail_123

    # Vérification de la somme
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

    # Vérification
    print(
        f"    {Fore.YELLOW}Vérification : {total_details} trades catégorisés vs {total_g} total global{Style.RESET_ALL}")
    if total_details != total_g:
        print(f"    {Fore.RED}⚠️ Anomalie détectée: La somme des détails ({total_details}) "
              f"ne correspond pas au total global ({total_g}){Style.RESET_ALL}")


# ───────────────────── HOLD‑OUT TEST ────────────────────────────

def calculate_test_metrics(params: dict):
    print(f"\n{Fore.CYAN}🧮  Calcul sur DATASET TEST{Style.RESET_ALL}\n")
    m1 = imbalance_high_rev(TEST, **params)
    m2 = imbalance_high_rev_2(TEST, **params)
    m3 = imbalance_high_rev_3(TEST, **params)

    # Calcul des métriques par condition avec sessions
    wr_1, pct_1, suc_1, fail_1, sess_1 = _metrics(TEST, m1)
    wr_2, pct_2, suc_2, fail_2, sess_2 = _metrics(TEST, m2)
    wr_3, pct_3, suc_3, fail_3, sess_3 = _metrics(TEST, m3)

    # Calcul des métriques combinées avec sessions
    wr_u, pct_u, suc_u, fail_u, sess_u, *_ = _metrics_combined(TEST, m1, m2, m3)

    # Calcul des métriques détaillées par catégorie
    metrics_detail_test = _metrics_exclusive(TEST, m1, m2, m3)

    # Affichage détaillé par condition
    print(f"{Fore.YELLOW}--- Détail par condition ---{Style.RESET_ALL}")
    print(f"Condition 1: WR={Fore.GREEN}{wr_1:.2%}{Style.RESET_ALL}  pct={pct_1:.2%}  "
          f"✓{Fore.GREEN}{suc_1}{Style.RESET_ALL} ✗{Fore.RED}{fail_1}{Style.RESET_ALL}  "
          f"Total={Fore.CYAN}{suc_1 + fail_1}{Style.RESET_ALL} (sessions: {TEST_SESSIONS})")
    print(f"Condition 2: WR={Fore.GREEN}{wr_2:.2%}{Style.RESET_ALL}  pct={pct_2:.2%}  "
          f"✓{Fore.GREEN}{suc_2}{Style.RESET_ALL} ✗{Fore.RED}{fail_2}{Style.RESET_ALL}  "
          f"Total={Fore.CYAN}{suc_2 + fail_2}{Style.RESET_ALL} (sessions: {TEST_SESSIONS})")
    print(f"Condition 3: WR={Fore.GREEN}{wr_3:.2%}{Style.RESET_ALL}  pct={pct_3:.2%}  "
          f"✓{Fore.GREEN}{suc_3}{Style.RESET_ALL} ✗{Fore.RED}{fail_3}{Style.RESET_ALL}  "
          f"Total={Fore.CYAN}{suc_3 + fail_3}{Style.RESET_ALL} (sessions: {TEST_SESSIONS})")

    # Affichage résultat combiné
    print(f"\n{Fore.YELLOW}--- Résultat combiné (union) ---{Style.RESET_ALL}")
    print(f"Union: WR={Fore.GREEN}{wr_u:.2%}{Style.RESET_ALL}  pct={pct_u:.2%}  "
          f"✓{Fore.GREEN}{suc_u}{Style.RESET_ALL} ✗{Fore.RED}{fail_u}{Style.RESET_ALL}  "
          f"Total={Fore.CYAN}{suc_u + fail_u}{Style.RESET_ALL} (sessions: {TEST_SESSIONS})")

    # Affichage détaillé par catégorie de trades
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
        # Touche spéciale sans caractère
        pass


# Démarrer listener dans un thread séparé
def start_keyboard_listener():
    listener = keyboard.Listener(on_press=on_press)
    listener.daemon = True  # Le thread sera automatiquement terminé quand le programme principal se termine
    listener.start()
    return listener


# ───────────────────── MAIN LOOP ────────────────────────────────
def main() -> None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    last_best_shown = None

    # Démarrer le listener clavier
    listener = start_keyboard_listener()
    print(
        f"{Fore.CYAN}Écouteur clavier démarré - appuyez sur '&' à tout moment pour tester sur le dataset TEST{Style.RESET_ALL}")

    for done in range(1, N_TRIALS + 1):
        study.optimize(objective, n_trials=1)

        if RUN_TEST:
            globals()["RUN_TEST"] = False
            calculate_test_metrics(study.best_params)

        if best_trial.get("number") is not None:
            print(f"Best trial {best_trial['number']}  value {Fore.GREEN}{best_trial['score']:.4f}{Style.RESET_ALL}",
                  end="\r")

        if (done % PRINT_EVERY == 0 or best_trial.get("number") != last_best_shown):
            bt = best_trial
            print(
                f"\n\n{Fore.YELLOW}*** BEST so far ▸ trial {bt['number']}  score={Fore.GREEN}{bt['score']:.4f}{Style.RESET_ALL}")

            # Affichage global avec trades réussis/échoués/totaux et sessions
            print(f"    {Fore.CYAN}[GLOBAL - Trades uniques]{Style.RESET_ALL}")
            print(f"    TR WR {Fore.GREEN}{bt['wr_t']:.2%}{Style.RESET_ALL} | "
                  f"pct {bt['pct_t']:.2%} | "
                  f"✓{Fore.GREEN}{bt['suc_t']}{Style.RESET_ALL} "
                  f"✗{Fore.RED}{bt['fail_t']}{Style.RESET_ALL} "
                  f"Total={Fore.CYAN}{bt['suc_t'] + bt['fail_t']}{Style.RESET_ALL} (sessions: {TRAIN_SESSIONS})")

            print(f"    V1 WR {Fore.GREEN}{bt['wr_v']:.2%}{Style.RESET_ALL} | "
                  f"pct {bt['pct_v']:.2%} | "
                  f"✓{Fore.GREEN}{bt['suc_v']}{Style.RESET_ALL} "
                  f"✗{Fore.RED}{bt['fail_v']}{Style.RESET_ALL} "
                  f"Total={Fore.CYAN}{bt['suc_v'] + bt['fail_v']}{Style.RESET_ALL} (sessions: {VAL_SESSIONS})")

            print(f"    V2 WR {Fore.GREEN}{bt['wr_v1']:.2%}{Style.RESET_ALL} | "
                  f"pct {bt['pct_v1']:.2%} | "
                  f"✓{Fore.GREEN}{bt['suc_v1']}{Style.RESET_ALL} "
                  f"✗{Fore.RED}{bt['fail_v1']}{Style.RESET_ALL} "
                  f"Total={Fore.CYAN}{bt['suc_v1'] + bt['fail_v1']}{Style.RESET_ALL} (sessions: {VAL1_SESSIONS})")

            # Affichage des écarts moyens
            print(f"    Gap WR moyen: {Fore.YELLOW}{bt['avg_gap_wr']:.2%}{Style.RESET_ALL} | "
                  f"Gap PCT moyen: {Fore.YELLOW}{bt['avg_gap_pct']:.2%}{Style.RESET_ALL}")

            # Affichage détaillé par catégorie de trades pour chaque dataset
            if bt['metrics_detail_train']:
                print_detailed_metrics("TRAIN", bt['metrics_detail_train'])

            if bt['metrics_detail_val']:
                print_detailed_metrics("VAL", bt['metrics_detail_val'])

            if bt['metrics_detail_val1']:
                print_detailed_metrics("VAL1", bt['metrics_detail_val1'])

            # Détail par condition (affichage original)
            print(f"\n    {Fore.CYAN}[Condition 1]{Style.RESET_ALL}")
            print(f"    TR WR {Fore.GREEN}{bt['wr_t1']:.2%}{Style.RESET_ALL} | "
                  f"pct {bt['pct_t1']:.2%} | "
                  f"✓{Fore.GREEN}{bt['suc_t1']}{Style.RESET_ALL} "
                  f"✗{Fore.RED}{bt['fail_t1']}{Style.RESET_ALL} "
                  f"Total={Fore.CYAN}{bt['suc_t1'] + bt['fail_t1']}{Style.RESET_ALL} (sessions: {TRAIN_SESSIONS})")

            print(f"    V1 WR {Fore.GREEN}{bt['wr_v1']:.2%}{Style.RESET_ALL} | "
                  f"pct {bt['pct_v1']:.2%} | "
                  f"✓{Fore.GREEN}{bt['suc_v1']}{Style.RESET_ALL} "
                  f"✗{Fore.RED}{bt['fail_v1']}{Style.RESET_ALL} "
                  f"Total={Fore.CYAN}{bt['suc_v1'] + bt['fail_v1']}{Style.RESET_ALL} (sessions: {VAL_SESSIONS})")

            print(f"    V2 WR {Fore.GREEN}{bt['wr_v1_1']:.2%}{Style.RESET_ALL} | "
                  f"pct {bt['pct_v1_1']:.2%} | "
                  f"✓{Fore.GREEN}{bt['suc_v1_1']}{Style.RESET_ALL} "
                  f"✗{Fore.RED}{bt['fail_v1_1']}{Style.RESET_ALL} "
                  f"Total={Fore.CYAN}{bt['suc_v1_1'] + bt['fail_v1_1']}{Style.RESET_ALL} (sessions: {VAL1_SESSIONS})")

            print(f"\n    {Fore.CYAN}[Condition 2]{Style.RESET_ALL}")
            print(f"    TR WR {Fore.GREEN}{bt['wr_t2']:.2%}{Style.RESET_ALL} | "
                  f"pct {bt['pct_t2']:.2%} | "
                  f"✓{Fore.GREEN}{bt['suc_t2']}{Style.RESET_ALL} "
                  f"✗{Fore.RED}{bt['fail_t2']}{Style.RESET_ALL} "
                  f"Total={Fore.CYAN}{bt['suc_t2'] + bt['fail_t2']}{Style.RESET_ALL} (sessions: {TRAIN_SESSIONS})")

            print(f"    V1 WR {Fore.GREEN}{bt['wr_v2']:.2%}{Style.RESET_ALL} | "
                  f"pct {bt['pct_v2']:.2%} | "
                  f"✓{Fore.GREEN}{bt['suc_v2']}{Style.RESET_ALL} "
                  f"✗{Fore.RED}{bt['fail_v2']}{Style.RESET_ALL} "
                  f"Total={Fore.CYAN}{bt['suc_v2'] + bt['fail_v2']}{Style.RESET_ALL} (sessions: {VAL_SESSIONS})")

            print(f"    V2 WR {Fore.GREEN}{bt['wr_v1_2']:.2%}{Style.RESET_ALL} | "
                  f"pct {bt['pct_v1_2']:.2%} | "
                  f"✓{Fore.GREEN}{bt['suc_v1_2']}{Style.RESET_ALL} "
                  f"✗{Fore.RED}{bt['fail_v1_2']}{Style.RESET_ALL} "
                  f"Total={Fore.CYAN}{bt['suc_v1_2'] + bt['fail_v1_2']}{Style.RESET_ALL} (sessions: {VAL1_SESSIONS})")

            print(f"\n    {Fore.CYAN}[Condition 3]{Style.RESET_ALL}")
            print(f"    TR WR {Fore.GREEN}{bt['wr_t3']:.2%}{Style.RESET_ALL} | "
                  f"pct {bt['pct_t3']:.2%} | "
                  f"✓{Fore.GREEN}{bt['suc_t3']}{Style.RESET_ALL} "
                  f"✗{Fore.RED}{bt['fail_t3']}{Style.RESET_ALL} "
                  f"Total={Fore.CYAN}{bt['suc_t3'] + bt['fail_t3']}{Style.RESET_ALL} (sessions: {TRAIN_SESSIONS})")

            print(f"    V1 WR {Fore.GREEN}{bt['wr_v3']:.2%}{Style.RESET_ALL} | "
                  f"pct {bt['pct_v3']:.2%} | "
                  f"✓{Fore.GREEN}{bt['suc_v3']}{Style.RESET_ALL} "
                  f"✗{Fore.RED}{bt['fail_v3']}{Style.RESET_ALL} "
                  f"Total={Fore.CYAN}{bt['suc_v3'] + bt['fail_v3']}{Style.RESET_ALL} (sessions: {VAL_SESSIONS})")

            print(f"    V2 WR {Fore.GREEN}{bt['wr_v1_3']:.2%}{Style.RESET_ALL} | "
                  f"pct {bt['pct_v1_3']:.2%} | "
                  f"✓{Fore.GREEN}{bt['suc_v1_3']}{Style.RESET_ALL} "
                  f"✗{Fore.RED}{bt['fail_v1_3']}{Style.RESET_ALL} "
                  f"Total={Fore.CYAN}{bt['suc_v1_3'] + bt['fail_v1_3']}{Style.RESET_ALL} (sessions: {VAL1_SESSIONS})")

            print(f"\n    params ➜ {Fore.MAGENTA}{bt['params']}{Style.RESET_ALL}\n")

            last_best_shown = best_trial["number"]

    # Ces deux lignes doivent être alignées avec la définition de la boucle for,
    # pas avec le contenu de la boucle
    print(f"\n{Fore.YELLOW}🔚  Fin des essais Optuna.{Style.RESET_ALL}")
    calculate_test_metrics(study.best_params)


if __name__ == "__main__":
    main()