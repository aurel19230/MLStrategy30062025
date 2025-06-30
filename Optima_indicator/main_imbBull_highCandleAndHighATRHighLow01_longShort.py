# -*- coding: utf-8 -*-
"""optuna_live_monitor.py – version ATR-DiffHighPrice (SIMPLIFIED)
=================================================================
- Optimise dynamiquement diffHighPrice_0_1 en fonction des plages d'ATR
- Utilise **deux** datasets de validation pour une plus grande robustesse
- Raccourci clavier « & » pour déclencher un calcul immédiat sur le jeu TEST
- Affichage synthétique à chaque trial + rapport détaillé périodique
- Filtrage optionnel par is_imBullWithPoc_light_short
- Version simplifiée sans visualisation matplotlib
- Support pour USE_TRAIN_IN_OPTIMIZATION (contrôle d'utilisation du dataset TRAIN)
- MODIFICATION: TRAIN toujours testé pour info même si non utilisé dans l'optimisation
"""

# - 🧠 WR_train  = 62.5%   | pct_train  = 0.53%   | trades = 16   | sessions = 12
# - 📈 WR_val    = 63.11%  | pct_val    = 1.05%   | trades = 103  | sessions = 36
# - 📊 WR_val1   = 69.77%  | pct_val1   = 0.75%   | trades = 43   | sessions = 32
# - 🧪 WR_test   = 63.64%  | pct_test   = 0.60%   | trades = 33   | sessions = 21
# atr_window 12
# - ⚙️ ATR thresholds (respectivement 1 2 3)    : [1.5, 1.7, 1.9]
# - 📐 diff_high_atr (respectivement  1 2 3 4)     : [5.5, 3.75, 5.75, 3.25]

from __future__ import annotations

# ─────────────────── CONFIG ──────────────────────────────────────
from pathlib import Path
import sys, time, math, optuna, pandas as pd, numpy as np
import threading
import warnings
from func_standard import calculate_atr, calculate_atr_thresholds

# Supprimer toutes les visualisations et tous les avertissements
warnings.filterwarnings("ignore")

# Remplacer msvcrt par pynput
try:
    from pynput import keyboard
except ImportError:
    # Créer une version minimale du keyboard listener
    print("Module pynput non disponible - raccourci clavier désactivé")


    class KeyboardListener:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        def join(self):
            pass


    keyboard = type('keyboard', (), {'Listener': KeyboardListener})

# Ajout de colorama pour les affichages colorés
try:
    from colorama import init, Fore, Back, Style

    # Initialiser colorama (nécessaire pour Windows)
    init(autoreset=True)
except ImportError:
    # Créer des fonctions de remplacement si colorama n'est pas disponible
    print("Module colorama non disponible - couleurs désactivées")


    class DummyColor:
        def __getattr__(self, name):
            return ""


    Fore = DummyColor()
    Back = DummyColor()
    Style = DummyColor()

RANDOM_SEED = 42
# 🔧 CONFIGURATION STRATÉGIE
DIRECTION = "short"  # "short" ou "long"

OPTIMIZATION_GOAL = "maximize"
GOAL_DESCRIPTION = "MAXIMISER"
import os
from pathlib import Path

print(
    f"{Fore.CYAN}🎯 STRATÉGIE: ATR CHECK - {DIRECTION.upper()} → {GOAL_DESCRIPTION} le winrate{Style.RESET_ALL}")

# Fichiers de données
DIR = R"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject" \
      R"\Sierra_chart\xTickReversal\simu\5_0_5TP_6SL_samdei soir\\merge"
print("=" * 80)
print("🔍 TEST D'EXISTENCE DU RÉPERTOIRE")
print("=" * 80)

# Test d'existence avec os.path
print(f"📁 Répertoire testé: {DIR}")
print(f"✅ os.path.exists(): {os.path.exists(DIR)}")
print(f"✅ os.path.isdir(): {os.path.isdir(DIR)}")

# ====== MODIFICATION PRINCIPALE: Contrôle d'utilisation du dataset TRAIN ======
USE_TRAIN_IN_OPTIMIZATION = True  # Mettre False pour désactiver l'utilisation du dataset TRAIN dans l'optimisation
#USE_TRAIN_IN_OPTIMIZATION = False  # Mettre False pour désactiver l'utilisation du dataset TRAIN dans l'optimisation

# 🔧 CONFIGURATION FLEXIBLE : TRAIN toujours chargé pour les stats finales
Direction = DIRECTION.capitalize()  # Pour compatibilité avec les noms de fichiers
CSV_TRAIN = DIR + Rf"\Step5_5_0_5TP_6SL_010124_110625_extractOnlyFullSession_Only{Direction}_feat__split2_01052024_30092024.csv"
CSV_VAL = DIR + Rf"\Step5_5_0_5TP_6SL_010124_110625_extractOnlyFullSession_Only{Direction}_feat__split3_30092024_27022025.csv"
CSV_VAL1 = DIR + Rf"\Step5_5_0_5TP_6SL_010124_110625_extractOnlyFullSession_Only{Direction}_feat__split4_27022025_14052025.csv"
CSV_TEST = DIR + Rf"\Step5_5_0_5TP_6SL_010124_110625_extractOnlyFullSession_Only{Direction}_feat__split5_14052025_10062025.csv"

# Ajout de la variable globale ATR_FIXED (True par défaut)
ATR_FIXED = True

# Paramètre pour la fenêtre ATR (sera défini par l'utilisateur)
ATR_WINDOW = 14
ATR_WINDOW_LOW, ATR_WINDOW_HIGH = 3,20
# Paramètres pour les bornes de diffHighPrice
DIFF_HIGH_MIN, DIFF_HIGH_MAX = 2.5, 5.25
DIFF_HIGH_STEP = 0.25  # Pas d'incrémentation

# Paramètres pour l'optimisation
N_TRIALS = 50000  # Nombre total d'essais
PRINT_EVERY = 25  # Fréquence d'affichage détaillé
LAMBDA_WR = 1  # Pénalité pour l'écart de WR entre datasets
LAMBDA_PCT = 0  # Pénalité pour l'écart de PCT entre datasets
FAILED_PENALTY = -1.0  # Pénalité pour les essais échoués

# Variables globales (seront mises à jour dans main())
WINRATE_MIN = 0.535
PCT_TRADE_MIN = 0.001
ALPHA = 0.7
FILTER_IMBULL = False
DEBUG_LOG = True  # Active l'affich
# ───────────────────── DATA LOADING ─────────────────────────────
import chardet


def detect_file_encoding(file_path: str, sample_size: int = 100_000) -> str:
    with open(file_path, 'rb') as f:
        raw = f.read(sample_size)
    return chardet.detect(raw)['encoding']


def load_csv(path: str | Path) -> tuple[pd.DataFrame, int]:
    """Charge le CSV sans filtrage (données brutes)"""
    encoding = detect_file_encoding(str(path))
    if encoding.lower() == "ascii":
        encoding = "ISO-8859-1"

    print(f"{path.name} ➜ encodage détecté: {encoding}")

    # Chargement robuste
    df = pd.read_csv(path, sep=";", encoding=encoding, parse_dates=["date"], low_memory=False)

    # 🔧 Correction de sc_sessionStartEnd
    df["sc_sessionStartEnd"] = pd.to_numeric(df["sc_sessionStartEnd"], errors="coerce")
    df = df.dropna(subset=["sc_sessionStartEnd"])
    df["sc_sessionStartEnd"] = df["sc_sessionStartEnd"].astype(int)

    # 🔍 Vérif des valeurs possibles
    print(f"{path.name} ➜ uniques sc_sessionStartEnd: {df['sc_sessionStartEnd'].unique()}")

    # 📊 Compter les sessions
    nb_start = (df["sc_sessionStartEnd"] == 10).sum()
    nb_end = (df["sc_sessionStartEnd"] == 20).sum()
    nb_sessions = min(nb_start, nb_end)

    if nb_start != nb_end:
        print(
            f"{Fore.YELLOW}⚠️ Incohérence sessions: {nb_start} débuts vs {nb_end} fins dans {path.name}{Style.RESET_ALL}")
    else:
        print(f"{Fore.GREEN}✔ {nb_sessions} sessions complètes détectées dans {path.name}{Style.RESET_ALL}")

    # ✅ Numérotation des sessions
    df["session_id"] = (df["sc_sessionStartEnd"] == 10).cumsum().astype("int32")

    return df, nb_sessions


# Fonction pour charger les données brutes sans filtrage
def load_raw_csv(path: str | Path) -> pd.DataFrame:
    """Charge le CSV brut sans appliquer de filtrage sur class_binaire"""
    encoding = detect_file_encoding(str(path))
    if encoding.lower() == "ascii":
        encoding = "ISO-8859-1"

    # Chargement robuste
    df = pd.read_csv(path, sep=";", encoding=encoding, parse_dates=["date"], low_memory=False)

    # 🔧 Correction de sc_sessionStartEnd mais pas de filtrage
    df["sc_sessionStartEnd"] = pd.to_numeric(df["sc_sessionStartEnd"], errors="coerce")
    df = df.dropna(subset=["sc_sessionStartEnd"])
    df["sc_sessionStartEnd"] = df["sc_sessionStartEnd"].astype(int)

    # ✅ Numérotation des sessions
    df["session_id"] = (df["sc_sessionStartEnd"] == 10).cumsum().astype("int32")

    return df


# Fonction pour calculer l'ATR avec une fenêtre personnalisée
import numba
import numpy as np


def create_atr_masks(df: pd.DataFrame, params: dict) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Crée les masques correspondant aux différentes plages d'ATR"""
    threshold_1 = params["atr_threshold_1"]
    threshold_2 = params["atr_threshold_2"]
    threshold_3 = params["atr_threshold_3"]

    if 'atr_recalc' not in df.columns:
        df['atr_recalc'] = calculate_atr(df, window=ATR_WINDOW)

    mask_atr_1 = df["atr_recalc"] < threshold_1
    mask_atr_2 = (df["atr_recalc"] >= threshold_1) & (df["atr_recalc"] < threshold_2)
    mask_atr_3 = (df["atr_recalc"] >= threshold_2) & (df["atr_recalc"] < threshold_3)
    mask_atr_4 = df["atr_recalc"] >= threshold_3

    return mask_atr_1, mask_atr_2, mask_atr_3, mask_atr_4


# ───────────────────── METRICS HELPERS ──────────────────────────
def _metrics(df: pd.DataFrame, mask: pd.Series, original_len: int = None) -> tuple[float, float, int, int, int]:
    """Calcule les métriques avec le nombre de sessions couvertes"""
    # Appliquer d'abord le filtre class_binaire pour les calculs de métriques
    trading_mask = df["class_binaire"].isin([0, 1])

    # Sous-ensemble des données qui respectent à la fois le filtre trading et le masque fourni
    sub = df.loc[trading_mask & mask]

    if sub.empty:
        return 0.0, 0.0, 0, 0, 0

    # Calculer les succès (class_binaire = 1)
    wins = int((sub["class_binaire"] == 1).sum())
    total = len(sub)  # Nombre total de trades pris

    # IMPORTANT: Toujours utiliser le nombre total d'échantillons class_binaire 0/1 comme base
    # Si original_len n'est pas fourni, le calculer nous-mêmes
    if original_len is None:
        base_len = trading_mask.sum()
    else:
        # Si original_len est fourni, s'assurer qu'il représente bien tous les échantillons class_binaire 0/1
        base_len = original_len

    # Calculer le pourcentage de trades pris par rapport au nombre total d'échantillons valides
    pct_trade = total / base_len

    # Calculer le nombre de sessions uniques où il y a des trades
    sessions_covered = sub["session_id"].nunique()

    return wins / total if total > 0 else 0.0, pct_trade, wins, total - wins, sessions_covered


# Modifiez également la fonction _metrics_by_atr_segment pour appliquer le filtre IMBULL
def _metrics_by_atr_segment(df: pd.DataFrame, params: dict, original_len: int = None, df_name: str = ""):
    """Calcule les métriques détaillées par segment d'ATR"""
    # Filtre pour les données de trading - UNIQUEMENT class_binaire, PAS le filtre IMBULL ici
    trading_mask = df["class_binaire"].isin([0, 1])
    trading_df = df[trading_mask]

    # Si original_len n'est pas fourni, utiliser le nombre d'éléments après filtrage
    if original_len is None:
        original_len = len(trading_df)

    # Si l'ATR n'est pas encore calculé avec la bonne fenêtre, le calculer
    if 'atr_recalc' not in df.columns:
        df['atr_recalc'] = calculate_atr(df, window=ATR_WINDOW)

    # Récupérer les seuils d'ATR depuis les paramètres
    threshold_1 = params["atr_threshold_1"]
    threshold_2 = params["atr_threshold_2"]
    threshold_3 = params["atr_threshold_3"]

    # Créer les masques pour chaque plage d'ATR
    mask_atr_1 = df["atr_recalc"] < threshold_1
    mask_atr_2 = (df["atr_recalc"] >= threshold_1) & (df["atr_recalc"] < threshold_2)
    mask_atr_3 = (df["atr_recalc"] >= threshold_2) & (df["atr_recalc"] < threshold_3)
    mask_atr_4 = df["atr_recalc"] >= threshold_3

    # Récupérer les valeurs de diffHighPrice_0_1 pour chaque plage d'ATR
    diff_high_atr_1 = params["diff_high_atr_1"]
    diff_high_atr_2 = params["diff_high_atr_2"]
    diff_high_atr_3 = params["diff_high_atr_3"]
    diff_high_atr_4 = params["diff_high_atr_4"]

    # Créer les masques pour diffHighPrice_0_1 pour chaque plage d'ATR
    mask_diff_1 = mask_atr_1 & (df["sc_diffHighPrice_0_1"] > diff_high_atr_1)
    mask_diff_2 = mask_atr_2 & (df["sc_diffHighPrice_0_1"] > diff_high_atr_2)
    mask_diff_3 = mask_atr_3 & (df["sc_diffHighPrice_0_1"] > diff_high_atr_3)
    mask_diff_4 = mask_atr_4 & (df["sc_diffHighPrice_0_1"] > diff_high_atr_4)

    if FILTER_IMBULL and "is_imBullWithPoc_light_short" not in df.columns:
        raise ValueError(f"La colonne 'is_imBullWithPoc_light_short' est absente du dataset {df_name}.")

    # Appliquer le filtre IMBULL UNIQUEMENT aux masques diffHighPrice si activé
    if FILTER_IMBULL and "is_imBullWithPoc_light_short" in df.columns:
        # Conversion en booléen

        # Appliquer le filtre après avoir remplacé les NaN par False
        imbull_mask = df["is_imBullWithPoc_light_short"].fillna(0).astype(bool)
        mask_diff_1 = mask_diff_1 & imbull_mask
        mask_diff_2 = mask_diff_2 & imbull_mask
        mask_diff_3 = mask_diff_3 & imbull_mask
        mask_diff_4 = mask_diff_4 & imbull_mask

    # Appliquer le filtre class_binaire aux masques déjà créés
    mask_diff_1 = trading_mask & mask_diff_1
    mask_diff_2 = trading_mask & mask_diff_2
    mask_diff_3 = trading_mask & mask_diff_3
    mask_diff_4 = trading_mask & mask_diff_4

    # Calculer les métriques pour chaque segment
    metrics_1 = _metrics(df, mask_diff_1, original_len)
    metrics_2 = _metrics(df, mask_diff_2, original_len)
    metrics_3 = _metrics(df, mask_diff_3, original_len)
    metrics_4 = _metrics(df, mask_diff_4, original_len)

    # Calculer les métriques pour le masque combiné
    mask_combined = mask_diff_1 | mask_diff_2 | mask_diff_3 | mask_diff_4
    metrics_combined = _metrics(df, mask_combined, original_len)

    # Compter le nombre d'échantillons dans chaque segment d'ATR (sans filtrage class_binaire)
    count_1 = mask_atr_1.sum()
    count_2 = mask_atr_2.sum()
    count_3 = mask_atr_3.sum()
    count_4 = mask_atr_4.sum()

    return {
        "atr_1": metrics_1,
        "atr_2": metrics_2,
        "atr_3": metrics_3,
        "atr_4": metrics_4,
        "combined": metrics_combined,
        # Compte d'échantillons par segment
        "counts": {
            "atr_1": count_1,
            "atr_2": count_2,
            "atr_3": count_3,
            "atr_4": count_4,
        }
    }


# Fonction pour afficher les statistiques ATR individuelles
def print_atr_stats_for_dataset():
    # Extraire les valeurs ATR pour chaque dataset
    train_atr = TRAIN_RAW['atr_recalc'].values
    val_atr = VAL_RAW['atr_recalc'].values
    val1_atr = VAL1_RAW['atr_recalc'].values
    test_atr = TEST_RAW['atr_recalc'].values

    # Afficher les statistiques pour chaque dataset
    print(f"\n{Fore.YELLOW}[STATISTIQUES ATR PAR DATASET - FENÊTRE {ATR_WINDOW}]{Style.RESET_ALL}")

    # TRAIN
    train_pct = np.percentile(train_atr, [0, 25, 50, 75, 100])
    train_mean = np.mean(train_atr)
    print(f"\n{Fore.CYAN}[TRAIN]{Style.RESET_ALL}")
    print(f"  Min: {train_pct[0]:.2f}")
    print(f"  25%: {train_pct[1]:.2f}")
    print(f"  50%: {train_pct[2]:.2f} (médiane)")
    print(f"  75%: {train_pct[3]:.2f}")
    print(f"  Max: {train_pct[4]:.2f}")
    print(f"  Moyenne: {train_mean:.2f}")
    print(f"  Nombre d'échantillons: {len(train_atr):,}")

    # VAL
    val_pct = np.percentile(val_atr, [0, 25, 50, 75, 100])
    val_mean = np.mean(val_atr)
    print(f"\n{Fore.CYAN}[VAL]{Style.RESET_ALL}")
    print(f"  Min: {val_pct[0]:.2f}")
    print(f"  25%: {val_pct[1]:.2f}")
    print(f"  50%: {val_pct[2]:.2f} (médiane)")
    print(f"  75%: {val_pct[3]:.2f}")
    print(f"  Max: {val_pct[4]:.2f}")
    print(f"  Moyenne: {val_mean:.2f}")
    print(f"  Nombre d'échantillons: {len(val_atr):,}")

    # VAL1
    val1_pct = np.percentile(val1_atr, [0, 25, 50, 75, 100])
    val1_mean = np.mean(val1_atr)
    print(f"\n{Fore.CYAN}[VAL1]{Style.RESET_ALL}")
    print(f"  Min: {val1_pct[0]:.2f}")
    print(f"  25%: {val1_pct[1]:.2f}")
    print(f"  50%: {val1_pct[2]:.2f} (médiane)")
    print(f"  75%: {val1_pct[3]:.2f}")
    print(f"  Max: {val1_pct[4]:.2f}")
    print(f"  Moyenne: {val1_mean:.2f}")
    print(f"  Nombre d'échantillons: {len(val1_atr):,}")

    # TEST
    test_pct = np.percentile(test_atr, [0, 25, 50, 75, 100])
    test_mean = np.mean(test_atr)
    print(f"\n{Fore.CYAN}[TEST]{Style.RESET_ALL}")
    print(f"  Min: {test_pct[0]:.2f}")
    print(f"  25%: {test_pct[1]:.2f}")
    print(f"  50%: {test_pct[2]:.2f} (médiane)")
    print(f"  75%: {test_pct[3]:.2f}")
    print(f"  Max: {test_pct[4]:.2f}")
    print(f"  Moyenne: {test_mean:.2f}")
    print(f"  Nombre d'échantillons: {len(test_atr):,}")


# Fonction pour afficher les métriques détaillées par segment d'ATR
def print_atr_segment_metrics(dataset_name, metrics, params):
    """Affiche les métriques détaillées par segment d'ATR"""

    # Récupérer les seuils d'ATR
    threshold_1 = params["atr_threshold_1"]
    threshold_2 = params["atr_threshold_2"]
    threshold_3 = params["atr_threshold_3"]

    # Récupérer les valeurs de diffHighPrice
    diff_high_1 = params["diff_high_atr_1"]
    diff_high_2 = params["diff_high_atr_2"]
    diff_high_3 = params["diff_high_atr_3"]
    diff_high_4 = params["diff_high_atr_4"]

    # Extraire les métriques pour chaque segment
    wr_1, pct_1, suc_1, fail_1, sess_1 = metrics["atr_1"]
    wr_2, pct_2, suc_2, fail_2, sess_2 = metrics["atr_2"]
    wr_3, pct_3, suc_3, fail_3, sess_3 = metrics["atr_3"]
    wr_4, pct_4, suc_4, fail_4, sess_4 = metrics["atr_4"]

    # Extraire les métriques combinées
    wr_c, pct_c, suc_c, fail_c, sess_c = metrics["combined"]

    # Extraire les counts par segment
    count_1 = metrics["counts"]["atr_1"]
    count_2 = metrics["counts"]["atr_2"]
    count_3 = metrics["counts"]["atr_3"]
    count_4 = metrics["counts"]["atr_4"]

    # Calculer les totaux pour chaque segment
    total_1 = suc_1 + fail_1
    total_2 = suc_2 + fail_2
    total_3 = suc_3 + fail_3
    total_4 = suc_4 + fail_4
    total_c = suc_c + fail_c

    print(f"\n    {Fore.CYAN}[MÉTRIQUES PAR SEGMENT ATR - {dataset_name}]{Style.RESET_ALL}")
    print(f"    Segment 1 (ATR < {threshold_1:.1f}, diffHigh > {diff_high_1:.2f}): "
          f"WR={Fore.GREEN}{wr_1:.2%}{Style.RESET_ALL} | "
          f"Trades={Fore.CYAN}{total_1}{Style.RESET_ALL} | "
          f"Échantillons={count_1} ({count_1 / sum([count_1, count_2, count_3, count_4]):.1%})")

    print(f"    Segment 2 ({threshold_1:.1f} ≤ ATR < {threshold_2:.1f}, diffHigh > {diff_high_2:.2f}): "
          f"WR={Fore.GREEN}{wr_2:.2%}{Style.RESET_ALL} | "
          f"Trades={Fore.CYAN}{total_2}{Style.RESET_ALL} | "
          f"Échantillons={count_2} ({count_2 / sum([count_1, count_2, count_3, count_4]):.1%})")

    print(f"    Segment 3 ({threshold_2:.1f} ≤ ATR < {threshold_3:.1f}, diffHigh > {diff_high_3:.2f}): "
          f"WR={Fore.GREEN}{wr_3:.2%}{Style.RESET_ALL} | "
          f"Trades={Fore.CYAN}{total_3}{Style.RESET_ALL} | "
          f"Échantillons={count_3} ({count_3 / sum([count_1, count_2, count_3, count_4]):.1%})")

    print(f"    Segment 4 (ATR ≥ {threshold_3:.1f}, diffHigh > {diff_high_4:.2f}): "
          f"WR={Fore.GREEN}{wr_4:.2%}{Style.RESET_ALL} | "
          f"Trades={Fore.CYAN}{total_4}{Style.RESET_ALL} | "
          f"Échantillons={count_4} ({count_4 / sum([count_1, count_2, count_3, count_4]):.1%})")

    print(f"    {Fore.YELLOW}Total: WR={Fore.GREEN}{wr_c:.2%}{Style.RESET_ALL} | "
          f"Trades={Fore.CYAN}{total_c}{Style.RESET_ALL} | "
          f"Sessions={sess_c}{Style.RESET_ALL}")

    # Vérification des totaux
    verif_total = total_1 + total_2 + total_3 + total_4
    if verif_total != total_c:
        print(f"    {Fore.RED}⚠️ Anomalie détectée: La somme des segments ({verif_total}) "
              f"ne correspond pas au total global ({total_c}){Style.RESET_ALL}")


# ───────────────────── OPTUNA OBJECTIVE ─────────────────────────
# Initialisation de best_trial avec des valeurs par défaut
best_trial = {
    "score": -math.inf,
    "number": None,
    "score_old": -math.inf,
    "atr_window": None,
    # Métriques combinées
    "wr_t": 0.0, "pct_t": 0.0, "suc_t": 0, "fail_t": 0, "sess_t": 0,
    "wr_v": 0.0, "pct_v": 0.0, "suc_v": 0, "fail_v": 0, "sess_v": 0,
    "wr_v1": 0.0, "pct_v1": 0.0, "suc_v1": 0, "fail_v1": 0, "sess_v1": 0,
    # Métriques détaillées par segment d'ATR
    "metrics_train": None,
    "metrics_val": None,
    "metrics_val1": None,
    # Écarts moyens
    "avg_gap_wr": 0.0,
    "avg_gap_pct": 0.0,
    # Paramètres optimaux
    "params": {}
}


# First, move the global declaration to the top of the function
def objective(trial: optuna.trial.Trial) -> float:
    global ATR_WINDOW, best_trial

    # 1) Fenêtre ATR --------------------------------------------------------
    # 1) Fenêtre ATR --------------------------------------------------------
    if not ATR_FIXED:
        ATR_WINDOW = trial.suggest_int("atr_window", ATR_WINDOW_LOW, ATR_WINDOW_HIGH)
        # Recalculer l'ATR pour tous les datasets
        for d in (TRAIN_RAW, VAL_RAW, VAL1_RAW, TEST_RAW):
            d['atr_recalc'] = calculate_atr(d, period=ATR_WINDOW)

    # 2) Seuils ATR ---------------------------------------------------------
    if LOCK_THRESHOLDS and FIXED_THRESHOLDS:
        # Utiliser les seuils fixés manuellement
        a1, a2, a3 = FIXED_THRESHOLDS
        atr_threshold_1 = trial.suggest_float("atr_threshold_1", a1, a1, step=0.1)
        atr_threshold_2 = trial.suggest_float("atr_threshold_2", a2, a2, step=0.1)
        atr_threshold_3 = trial.suggest_float("atr_threshold_3", a3, a3, step=0.1)
    elif not ATR_FIXED and not LOCK_THRESHOLDS:
        # Calculer les seuils adaptés à la fenêtre ATR actuelle
        atr_thresholds = calculate_atr_thresholds([TRAIN_RAW, VAL_RAW, VAL1_RAW, TEST_RAW])

        # Option 1: Fixer exactement aux valeurs calculées
        atr_threshold_1 = trial.suggest_float("atr_threshold_1", atr_thresholds[0], atr_thresholds[0], step=0.1)
        atr_threshold_2 = trial.suggest_float("atr_threshold_2", atr_thresholds[1], atr_thresholds[1], step=0.1)
        atr_threshold_3 = trial.suggest_float("atr_threshold_3", atr_thresholds[2], atr_thresholds[2], step=0.1)

        # Option 2: Permettre une exploration autour des valeurs calculées (recommandé)
        # atr_threshold_1 = trial.suggest_float("atr_threshold_1",
        #                                       max(0.5, atr_thresholds[0] - 0.5),
        #                                       atr_thresholds[0] + 0.5,
        #                                       step=0.1)
        # t2_min = max(atr_threshold_1 + 0.3, atr_thresholds[1] - 0.5)
        # atr_threshold_2 = trial.suggest_float("atr_threshold_2",
        #                                       t2_min,
        #                                       atr_thresholds[1] + 0.5,
        #                                       step=0.1)
        # t3_min = max(atr_threshold_2 + 0.3, atr_thresholds[2] - 0.5)
        # atr_threshold_3 = trial.suggest_float("atr_threshold_3",
        #                                       t3_min,
        #                                       atr_thresholds[2] + 0.5,
        #                                       step=0.1)
    else:
        # Exploration standard des seuils sans adaptation à la fenêtre ATR
        atr_threshold_1 = trial.suggest_float("atr_threshold_1", 1.0, 3.0, step=0.1)
        t2_min = math.ceil((atr_threshold_1 + 0.5) * 10) / 10
        atr_threshold_2 = trial.suggest_float("atr_threshold_2", t2_min, 5.0, step=0.1)
        t3_min = math.ceil((atr_threshold_2 + 0.5) * 10) / 10
        atr_threshold_3 = trial.suggest_float("atr_threshold_3", t3_min, 7.0, step=0.1)

    # Valeurs de diffHighPrice_0_1 pour chaque segment d'ATR
    diff_high_atr_1 = trial.suggest_float("diff_high_atr_1", DIFF_HIGH_MIN, DIFF_HIGH_MAX, step=DIFF_HIGH_STEP)
    diff_high_atr_2 = trial.suggest_float("diff_high_atr_2", DIFF_HIGH_MIN, DIFF_HIGH_MAX, step=DIFF_HIGH_STEP)
    diff_high_atr_3 = trial.suggest_float("diff_high_atr_3", DIFF_HIGH_MIN, DIFF_HIGH_MAX, step=DIFF_HIGH_STEP)
    diff_high_atr_4 = trial.suggest_float("diff_high_atr_4", DIFF_HIGH_MIN, DIFF_HIGH_MAX, step=DIFF_HIGH_STEP)

    # Paramètres complets
    p = {
        "atr_threshold_1": atr_threshold_1,
        "atr_threshold_2": atr_threshold_2,
        "atr_threshold_3": atr_threshold_3,
        "diff_high_atr_1": diff_high_atr_1,
        "diff_high_atr_2": diff_high_atr_2,
        "diff_high_atr_3": diff_high_atr_3,
        "diff_high_atr_4": diff_high_atr_4
    }

    # ====== MODIFICATION: TOUJOURS CALCULER LES MÉTRIQUES TRAIN ======
    # Compter les échantillons après filtrage class_binaire pour TRAIN
    train_filter = TRAIN_RAW["class_binaire"].isin([0, 1])
    train_len = train_filter.sum()

    # Calculer les métriques détaillées par segment d'ATR pour TRAIN (toujours)
    metrics_train = _metrics_by_atr_segment(TRAIN_RAW, p, train_len, "TRAIN")

    # Extraire les métriques combinées TRAIN
    wr_t, pct_t, suc_t, fail_t, sess_t = metrics_train["combined"]

    # Traitement des autres datasets (toujours utilisés)
    val_filter = VAL_RAW["class_binaire"].isin([0, 1])
    val1_filter = VAL1_RAW["class_binaire"].isin([0, 1])
    val_len = val_filter.sum()
    val1_len = val1_filter.sum()

    # Afficher les tailles des datasets pour vérification (optionnel, à retirer plus tard)
    if trial.number == 0:  # Seulement au premier essai
        print(f"Taille des datasets après filtrage class_binaire:")
        print(
            f"  TRAIN: {train_len} échantillons {'(utilisé pour optimisation)' if USE_TRAIN_IN_OPTIMIZATION else '(info seulement)'}")
        print(f"  VAL: {val_len} échantillons")
        print(f"  VAL1: {val1_len} échantillons")

    # Calculer les métriques détaillées par segment d'ATR pour VAL et VAL1
    metrics_val = _metrics_by_atr_segment(VAL_RAW, p, val_len, "VAL")
    metrics_val1 = _metrics_by_atr_segment(VAL1_RAW, p, val1_len, "VAL1")

    # Extraire les métriques combinées
    wr_v, pct_v, suc_v, fail_v, sess_v = metrics_val["combined"]
    wr_v1, pct_v1, suc_v1, fail_v1, sess_v1 = metrics_val1["combined"]

    # ——— Logging brut AVANT les vérifications de seuils ———
    filter_info = " avec IMBULL" if FILTER_IMBULL else ""
    if DEBUG_LOG:
        if USE_TRAIN_IN_OPTIMIZATION:
            print(
                f"TRIAL {trial.number:>5d} | ATRwin={ATR_WINDOW:>2d} | "
                f"ATR_THRESH: {atr_threshold_1:.1f}/{atr_threshold_2:.1f}/{atr_threshold_3:.1f} | "
                f"DH: {diff_high_atr_1:.2f}/{diff_high_atr_2:.2f}/{diff_high_atr_3:.2f}/{diff_high_atr_4:.2f} | "
                f"TR[{Fore.GREEN}{wr_t:.2%}{Style.RESET_ALL}/{pct_t:.2%}] "
                f"V1[{Fore.GREEN}{wr_v:.2%}{Style.RESET_ALL}/{pct_v:.2%}] "
                f"V2[{Fore.GREEN}{wr_v1:.2%}{Style.RESET_ALL}/{pct_v1:.2%}]{filter_info}")
        else:
            print(
                f"TRIAL {trial.number:>5d} | ATRwin={ATR_WINDOW:>2d} | "
                f"ATR_THRESH: {atr_threshold_1:.1f}/{atr_threshold_2:.1f}/{atr_threshold_3:.1f} | "
                f"DH: {diff_high_atr_1:.2f}/{diff_high_atr_2:.2f}/{diff_high_atr_3:.2f}/{diff_high_atr_4:.2f} | "
                f"TR[{Fore.YELLOW}{wr_t:.2%}{Style.RESET_ALL}/{pct_t:.2%}](info) "
                f"V1[{Fore.GREEN}{wr_v:.2%}{Style.RESET_ALL}/{pct_v:.2%}] "
                f"V2[{Fore.GREEN}{wr_v1:.2%}{Style.RESET_ALL}/{pct_v1:.2%}]{filter_info}")

    # ====== VÉRIFICATION DES SEUILS ADAPTÉE ======
    # Pour les seuils, on ne considère que les datasets utilisés dans l'optimisation
    datasets_to_check = [(wr_v, pct_v), (wr_v1, pct_v1)]
    if USE_TRAIN_IN_OPTIMIZATION:
        datasets_to_check.append((wr_t, pct_t))

    for wr, pct in datasets_to_check:
        if wr < WINRATE_MIN or pct < PCT_TRADE_MIN:
            if DEBUG_LOG:
                print(f"{Fore.RED}⚠️ REJET : Seuils WR ou PCT minimaux non atteints{Style.RESET_ALL}")
            return FAILED_PENALTY

    # Vérifier si le nombre de trades dans chaque dataset est suffisant
    min_trades = 10  # Nombre minimum de trades pour que la stratégie soit valide
    trades_to_check = [(suc_v + fail_v), (suc_v1 + fail_v1)]
    if USE_TRAIN_IN_OPTIMIZATION:
        trades_to_check.append(suc_t + fail_t)

    for trades in trades_to_check:
        if trades < min_trades:
            if DEBUG_LOG:
                print(f"{Fore.RED}⚠️ REJET : Nombre de trades insuffisant (<{min_trades}){Style.RESET_ALL}")
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

        # Moyenne des écarts
        avg_gap_wr = (gap_wr_tv + gap_wr_tv1 + gap_wr_vv1) / 3
        avg_gap_pct = (gap_pct_tv + gap_pct_tv1 + gap_pct_vv1) / 3

        # Score qui considère les trois datasets et les écarts moyens
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

    # ——— Vérifier si c'est un nouveau meilleur essai ———
    is_best = score > best_trial["score"]
    if is_best and DEBUG_LOG:
        print(f"{Fore.GREEN}✅ NOUVEAU MEILLEUR ESSAI ! Score: {score:.4f}{Style.RESET_ALL}")

    if is_best:
        best_trial = {
            "number": trial.number,
            "score": score,
            "score_old": score,  # Pour la compatibilité
            # Métriques combinées - TRAIN (toujours calculées maintenant)
            "wr_t": wr_t, "pct_t": pct_t, "suc_t": suc_t, "fail_t": fail_t, "sess_t": sess_t,
            # Métriques combinées - VAL
            "wr_v": wr_v, "pct_v": pct_v, "suc_v": suc_v, "fail_v": fail_v, "sess_v": sess_v,
            # Métriques combinées - VAL1
            "wr_v1": wr_v1, "pct_v1": pct_v1, "suc_v1": suc_v1, "fail_v1": fail_v1, "sess_v1": sess_v1,

            # Métriques détaillées par segment d'ATR (toujours stockées pour TRAIN)
            "metrics_train": metrics_train,
            "metrics_val": metrics_val,
            "metrics_val1": metrics_val1,

            # Écarts moyens
            "avg_gap_wr": avg_gap_wr,
            "avg_gap_pct": avg_gap_pct,

            "atr_window": ATR_WINDOW,
            # Paramètres optimaux
            "params": p
        }
    else:
        best_trial["score_old"] = score  # helper for symbol

    return score


# ───────────────────── HOLD‑OUT TEST ────────────────────────────
def calculate_test_metrics(params: dict):
    print(f"\n{Fore.CYAN}🧮  Calcul sur DATASET TEST{Style.RESET_ALL}\n")

    # ====== RAPPEL DES STATISTIQUES FINALES SUR TOUS LES DATASETS ======
    def calculate_dataset_stats(df, df_name, original_len):
        """Calcule et affiche les stats finales d'un dataset"""
        dataset_df = df.copy()

        # Filtrer selon class_binaire pour le calcul
        original_len_filtered = dataset_df["class_binaire"].isin([0, 1]).sum()

        # Calculer les métriques détaillées par segment d'ATR
        metrics = _metrics_by_atr_segment(dataset_df, params, original_len_filtered, df_name)

        # Extraire les métriques combinées
        wr_u, pct_u, suc_u, fail_u, sess_u = metrics["combined"]

        # ====== MODIFICATION: Indiquer si TRAIN est utilisé pour l'optimisation ======
        if df_name == "TRAIN" and not USE_TRAIN_IN_OPTIMIZATION:
            info_suffix = f" {Fore.YELLOW}(info seulement){Style.RESET_ALL}"
        else:
            info_suffix = ""

        print(f"    {df_name}{info_suffix}: WR={Fore.GREEN}{wr_u:.2%}{Style.RESET_ALL}  pct={pct_u:.2%}  "
              f"✓{Fore.GREEN}{suc_u}{Style.RESET_ALL} ✗{Fore.RED}{fail_u}{Style.RESET_ALL}  "
              f"Total={Fore.CYAN}{suc_u + fail_u}{Style.RESET_ALL} (sessions: {sess_u})")

    print(f"{Fore.YELLOW}📊 RAPPEL - Statistiques finales avec paramètres optimaux:{Style.RESET_ALL}")

    calculate_dataset_stats(TRAIN_RAW, "TRAIN", len(TRAIN_RAW))
    calculate_dataset_stats(VAL_RAW, "VAL  ", len(VAL_RAW))
    calculate_dataset_stats(VAL1_RAW, "VAL1 ", len(VAL1_RAW))

    print(f"\n{Fore.YELLOW}🎯 RÉSULTATS SUR DATASET TEST:{Style.RESET_ALL}")

    # Nombre d'échantillons après filtrage class_binaire
    test_len = TEST_RAW["class_binaire"].isin([0, 1]).sum()

    # Calculer les métriques détaillées par segment d'ATR
    metrics_test = _metrics_by_atr_segment(TEST_RAW, params, test_len)

    # Extraire les métriques combinées
    wr_c, pct_c, suc_c, fail_c, sess_c = metrics_test["combined"]

    # Afficher les métriques détaillées par segment
    print_atr_segment_metrics("TEST", metrics_test, params)

    # Vérifier si la stratégie est valide sur le jeu de test
    is_valid = (wr_c >= WINRATE_MIN and pct_c >= PCT_TRADE_MIN)
    if is_valid:
        print(f"{Fore.GREEN}✅ VALIDE{Style.RESET_ALL}\n\n")
    else:
        print(f"{Fore.RED}❌ REJET{Style.RESET_ALL}")

    return wr_c, pct_c, suc_c, fail_c, sess_c


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
    except:
        # Ignorer toutes les autres erreurs
        pass


# Démarrer listener dans un thread séparé
def start_keyboard_listener():
    try:
        listener = keyboard.Listener(on_press=on_press)
        listener.daemon = True  # Le thread sera automatiquement terminé quand le programme principal se termine
        listener.start()
        print(
            f"{Fore.CYAN}Écouteur clavier démarré - appuyez sur '&' à tout moment pour tester sur le dataset TEST{Style.RESET_ALL}")
        return listener
    except:
        print("Écouteur clavier indisponible")
        return None


# Fonction pour afficher les détails du meilleur essai
def print_best_trial_details(bt, trial_number=None):
    """Affiche les détails complets du meilleur essai"""
    if bt["number"] is None:
        print(f"\n{Fore.YELLOW}Aucun essai valide trouvé pour le moment{Style.RESET_ALL}")
        return

    # Récupérer les paramètres du meilleur essai
    params = bt["params"]

    # Extraire les métriques principales
    wr_t, pct_t = bt["wr_t"], bt["pct_t"]
    wr_v, pct_v = bt["wr_v"], bt["pct_v"]
    wr_v1, pct_v1 = bt["wr_v1"], bt["pct_v1"]

    # Afficher l'en-tête
    if trial_number:
        print(f"\n{Fore.CYAN}══════════════════ RÉSUMÉ APRÈS {trial_number} ESSAIS ══════════════════{Style.RESET_ALL}")
    else:
        print(f"\n{Fore.CYAN}══════════════════ MEILLEUR ESSAI ══════════════════{Style.RESET_ALL}")

    print(f"{Fore.YELLOW}Meilleur essai #{bt['number']} (score: {bt['score']:.4f}){Style.RESET_ALL}")

    # Afficher les paramètres optimaux
    print(f"\n{Fore.GREEN}PARAMÈTRES OPTIMAUX:{Style.RESET_ALL}")
    print(f"  • Fenêtre ATR: {bt['atr_window']}")
    print(
        f"  • Seuils ATR: {params['atr_threshold_1']:.1f} | {params['atr_threshold_2']:.1f} | {params['atr_threshold_3']:.1f}")
    print(f"  • DiffHighPrice par segment: "
          f"{params['diff_high_atr_1']:.2f} | {params['diff_high_atr_2']:.2f} | "
          f"{params['diff_high_atr_3']:.2f} | {params['diff_high_atr_4']:.2f}")

    # Afficher les métriques globales avec plus de détails sur les trades
    print(f"\n{Fore.GREEN}MÉTRIQUES PRINCIPALES:{Style.RESET_ALL}")

    # ====== MODIFICATION: AFFICHAGE ADAPTÉ SELON L'UTILISATION DE TRAIN ======
    total_t = bt['suc_t'] + bt['fail_t']
    if USE_TRAIN_IN_OPTIMIZATION:
        print(f"  • TRAIN:  WR={Fore.GREEN}{wr_t:.2%}{Style.RESET_ALL} | "
              f"Trades={total_t} (succès: {bt['suc_t']}, échecs: {bt['fail_t']}) | "
              f"PCT={pct_t:.2%} | Sessions={bt['sess_t']}")
    else:
        print(f"  • TRAIN:  WR={Fore.GREEN}{wr_t:.2%}{Style.RESET_ALL} | "
              f"Trades={total_t} (succès: {bt['suc_t']}, échecs: {bt['fail_t']}) | "
              f"PCT={pct_t:.2%} | Sessions={bt['sess_t']} {Fore.YELLOW}(info seulement){Style.RESET_ALL}")

    total_v = bt['suc_v'] + bt['fail_v']
    total_v1 = bt['suc_v1'] + bt['fail_v1']

    print(f"  • VAL1:   WR={Fore.GREEN}{wr_v:.2%}{Style.RESET_ALL} | "
          f"Trades={total_v} (succès: {bt['suc_v']}, échecs: {bt['fail_v']}) | "
          f"PCT={pct_v:.2%} | Sessions={bt['sess_v']}")
    print(f"  • VAL2:   WR={Fore.GREEN}{wr_v1:.2%}{Style.RESET_ALL} | "
          f"Trades={total_v1} (succès: {bt['suc_v1']}, échecs: {bt['fail_v1']}) | "
          f"PCT={pct_v1:.2%} | Sessions={bt['sess_v1']}")

    # Afficher les écarts
    print(f"  • Écarts: WR={bt['avg_gap_wr']:.2%} | PCT={bt['avg_gap_pct']:.2%}")

    # ====== MODIFICATION: Toujours afficher les métriques détaillées TRAIN ======
    if bt["metrics_train"]:
        train_label = "TRAIN" if USE_TRAIN_IN_OPTIMIZATION else "TRAIN (info seulement)"
        print_atr_segment_metrics(train_label, bt["metrics_train"], params)

    if bt["metrics_val"]:
        print_atr_segment_metrics("VAL", bt["metrics_val"], params)

    if bt["metrics_val1"]:
        print_atr_segment_metrics("VAL1", bt["metrics_val1"], params)

    print(f"\n{Fore.CYAN}═══════════════════════════════════════════════════════{Style.RESET_ALL}")


def calculate_test_metrics_and_save(params, trial_number, atr_window):
    """Calcule les métriques sur TEST et enregistre si valide sur tous les datasets"""
    print(f"\n{Fore.CYAN}🧮  Calcul sur DATASET TEST{Style.RESET_ALL}\n")

    # Nombre d'échantillons après filtrage class_binaire
    test_len = TEST_RAW["class_binaire"].isin([0, 1]).sum()

    # Calculer les métriques détaillées par segment d'ATR
    metrics_test = _metrics_by_atr_segment(TEST_RAW, params, test_len)

    # Extraire les métriques combinées
    wr_test, pct_test, suc_test, fail_test, sess_test = metrics_test["combined"]

    # Afficher les métriques détaillées par segment
    print_atr_segment_metrics("TEST", metrics_test, params)

    # Vérifier si la stratégie est valide sur le jeu de test
    test_valid = (wr_test >= WINRATE_MIN and pct_test >= PCT_TRADE_MIN)

    if test_valid:
        print(f"{Fore.GREEN}✅ VALIDE SUR TEST{Style.RESET_ALL}\n\n")

        # Récupérer les métriques best_trial pour les autres datasets
        wr_val, pct_val = best_trial["wr_v"], best_trial["pct_v"]
        wr_val1, pct_val1 = best_trial["wr_v1"], best_trial["pct_v1"]
        wr_train, pct_train = best_trial["wr_t"], best_trial["pct_t"]

        # ====== MODIFICATION: VÉRIFICATION ADAPTÉE SELON L'UTILISATION DE TRAIN ======
        if USE_TRAIN_IN_OPTIMIZATION:
            all_valid = (wr_train >= WINRATE_MIN and pct_train >= PCT_TRADE_MIN and
                         wr_val >= WINRATE_MIN and pct_val >= PCT_TRADE_MIN and
                         wr_val1 >= WINRATE_MIN and pct_val1 >= PCT_TRADE_MIN)
        else:
            # Si TRAIN n'est pas utilisé pour l'optimisation, on ne le vérifie pas pour la validation
            all_valid = (wr_val >= WINRATE_MIN and pct_val >= PCT_TRADE_MIN and
                         wr_val1 >= WINRATE_MIN and pct_val1 >= PCT_TRADE_MIN)

        if all_valid:
            validation_sets = "tous les ensembles" if USE_TRAIN_IN_OPTIMIZATION else "VAL, VAL1 et TEST"
            print(f"{Fore.GREEN}💯 STRATÉGIE VALIDE SUR {validation_sets.upper()}{Style.RESET_ALL}")

            # Créer un dictionnaire pour stocker les résultats
            trial_result = {
                "trial_number": trial_number,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "atr_window": atr_window,
                "score": best_trial["score"],
                "use_train_in_optimization": USE_TRAIN_IN_OPTIMIZATION,
                # Métriques WR et PCT pour chaque dataset (TRAIN toujours inclus pour info)
                "wr_train": wr_train, "pct_train": pct_train,
                "wr_val": wr_val, "pct_val": pct_val,
                "wr_val1": wr_val1, "pct_val1": pct_val1,
                "wr_test": wr_test, "pct_test": pct_test,
                # Nombre de trades pour chaque dataset (TRAIN toujours inclus pour info)
                "trades_train": best_trial["suc_t"] + best_trial["fail_t"],
                "trades_val": best_trial["suc_v"] + best_trial["fail_v"],
                "trades_val1": best_trial["suc_v1"] + best_trial["fail_v1"],
                "trades_test": suc_test + fail_test,
                # Sessions couvertes (TRAIN toujours inclus pour info)
                "sessions_train": best_trial["sess_t"],
                "sessions_val": best_trial["sess_v"],
                "sessions_val1": best_trial["sess_v1"],
                "sessions_test": sess_test,
                # Paramètres
                "atr_threshold_1": params["atr_threshold_1"],
                "atr_threshold_2": params["atr_threshold_2"],
                "atr_threshold_3": params["atr_threshold_3"],
                "diff_high_atr_1": params["diff_high_atr_1"],
                "diff_high_atr_2": params["diff_high_atr_2"],
                "diff_high_atr_3": params["diff_high_atr_3"],
                "diff_high_atr_4": params["diff_high_atr_4"],
                # Écarts
                "avg_gap_wr": best_trial["avg_gap_wr"],
                "avg_gap_pct": best_trial["avg_gap_pct"],
                # Configuration
                "filter_imbull": FILTER_IMBULL,
            }

            # Ajouter à la liste des essais valides
            valid_trials.append(trial_result)

            # Sauvegarder immédiatement dans un fichier CSV et obtenir le nom
            filename = save_valid_trials_to_csv()

            print(f"{Fore.YELLOW}📊 Essai #{trial_number} mémorisé dans {filename}{Style.RESET_ALL}")

            # Retourner les métriques et le nom du fichier
            return wr_test, pct_test, suc_test, fail_test, sess_test, filename
        else:
            validation_sets = "tous les ensembles" if USE_TRAIN_IN_OPTIMIZATION else "VAL/VAL1"
            print(f"{Fore.YELLOW}⚠️ Stratégie valide sur TEST mais pas sur {validation_sets}{Style.RESET_ALL}")

        # Si pas valide, retourner juste les métriques
        return wr_test, pct_test, suc_test, fail_test, sess_test


import pandas as pd
import os
from datetime import datetime

# Structure pour mémoriser les essais valides sur les 4 ensembles
valid_trials = []


# Fonction pour sauvegarder les essais valides dans un CSV
def save_valid_trials_to_csv():
    """Sauvegarde la liste des essais valides dans un fichier CSV et retourne le nom du fichier"""
    if not valid_trials:
        return None  # Pas d'essais valides à sauvegarder

    # Création d'un DataFrame pandas
    df = pd.DataFrame(valid_trials)

    # Définir le nom du fichier avec un timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"valid_trials_ATRHIGHLOW_{DIRECTION}_{timestamp}.csv"

    # CORRECTION: Créer le chemin complet en convertissant DIR en Path
    filepath = Path(DIR) / filename

    # Déterminer si le fichier existe déjà
    file_exists = os.path.isfile(filepath)

    # Écrire dans le fichier CSV (avec en-têtes si nouveau fichier)
    df.to_csv(filepath, mode='a', header=not file_exists, index=False, sep=';')

    print(f"{Fore.CYAN}Résultats sauvegardés dans : {filepath}{Style.RESET_ALL}")

    return filepath  # Retourne le chemin complet


# ───────────────────── MAIN LOOP (CORRECTION COMPLÈTE) ────────────────────────────────
def main():
    global ATR_WINDOW, ATR_FIXED, LOCK_THRESHOLDS, FIXED_THRESHOLDS, RUN_TEST, FILTER_IMBULL, valid_trials, USE_TRAIN_IN_OPTIMIZATION

    # Variable pour suivre le dernier fichier CSV créé
    last_csv_filename = None

    # ====== CHOIX SIMPLE ET DIRECT ======
    # Choix utilisateur -----------------------------------------------------
    print("Mode d'optimisation :")
    print(f"  [Entrée] → ATR variable ({ATR_WINDOW_LOW}-{ATR_WINDOW_HIGH}) avec filtre IMBULL activé")
    print("  1        → ATR fixe avec filtre IMBULL activé + TRAIN")
    print(f"  2        → ATR variable ({ATR_WINDOW_LOW}-{ATR_WINDOW_HIGH}) sans filtre IMBULL")
    print("  3        → ATR fixe sans filtre IMBULL")
    print("  4        → ATR variable sans TRAIN dans l'optimisation")
    print("  5        → ATR fixe sans TRAIN dans l'optimisation")
    mode = input("Choix : ").strip().lower()

    # Configuration selon le mode choisi
    if mode == "1":
        ATR_FIXED = True
        FILTER_IMBULL = True
        mode_description = "ATR fixe avec filtre IMBULL activé"
        print(f"{Fore.GREEN}Mode 1 sélectionné: {mode_description}{Style.RESET_ALL}")
    elif mode == "2":
        ATR_FIXED = False
        FILTER_IMBULL = False
        mode_description = f"ATR variable ({ATR_WINDOW_LOW}-{ATR_WINDOW_HIGH}) sans filtre IMBULL"
        print(f"{Fore.GREEN}Mode 2 sélectionné: {mode_description}{Style.RESET_ALL}")
    elif mode == "3":
        ATR_FIXED = True
        FILTER_IMBULL = False
        mode_description = "ATR fixe sans filtre IMBULL"
        print(f"{Fore.GREEN}Mode 3 sélectionné: {mode_description}{Style.RESET_ALL}")
    elif mode == "4":
        ATR_FIXED = False
        FILTER_IMBULL = True
        mode_description = "ATR variable"
        print(f"{Fore.GREEN}Mode 4 sélectionné: {mode_description}{Style.RESET_ALL}")
    elif mode == "5":
        ATR_FIXED = True
        FILTER_IMBULL = True
        mode_description = "ATR fixe"
        print(f"{Fore.GREEN}Mode 5 sélectionné: {mode_description}{Style.RESET_ALL}")
    else:  # Par défaut ou touche Entrée - VOTRE CHOIX PRÉFÉRÉ
        ATR_FIXED = False  # ATR variable
        FILTER_IMBULL = True  # IMBULL activé
        # TRAIN : Utiliser la valeur définie dans le script (pas de question)

        if USE_TRAIN_IN_OPTIMIZATION:
            mode_description = f"ATR variable ({ATR_WINDOW_LOW}-{ATR_WINDOW_HIGH}) avec filtre IMBULL activé + TRAIN"
        else:
            mode_description = f"ATR variable ({ATR_WINDOW_LOW}-{ATR_WINDOW_HIGH}) avec filtre IMBULL activé sans TRAIN dans l'optimisation"

        print(f"{Fore.GREEN}Mode par défaut sélectionné: {mode_description}{Style.RESET_ALL}")

    # ✅ Construction du résumé des paramètres sélectionnés
    config_summary = {
        "WINRATE_MIN": WINRATE_MIN,
        "PCT_TRADE_MIN": PCT_TRADE_MIN,
        "ALPHA": ALPHA
    }

    param_summary = (
        f"🛠️ Paramètres sélectionnés :\n"
        f"▪️ Mode : {mode_description}\n"
        f"▪️ ATR_FIXED : {ATR_FIXED}\n"
        f"▪️ FILTER_IMBULL : {FILTER_IMBULL}\n"
        f"▪️ USE_TRAIN_IN_OPTIMIZATION : {USE_TRAIN_IN_OPTIMIZATION}\n"
        f"▪️ Config utilisée : {config_summary}"
    )

    # Affichage de l'état du dataset TRAIN
    if not USE_TRAIN_IN_OPTIMIZATION:
        print(
            f"{Fore.YELLOW}⚠️  TRAIN DATASET DÉSACTIVÉ pour l'optimisation - Testé pour information seulement{Style.RESET_ALL}")
    else:
        print(f"{Fore.GREEN}✔ TRAIN DATASET ACTIVÉ pour l'optimisation{Style.RESET_ALL}")

    # ====== CHOIX ATR FIXE OU VARIABLE ======
    print(f"\n{Fore.YELLOW}Configuration ATR:{Style.RESET_ALL}")
    print("  [Entrée] → ATR variable (optimisation de la fenêtre)")
    print("  f        → ATR fixe (fenêtre constante)")
    atr_choice = input("Choix : ").strip().lower()

    if atr_choice == "f":
        ATR_FIXED = True
        atr_in = input("Fenêtre ATR (défaut 14) : ").strip()
        ATR_WINDOW = int(atr_in) if atr_in.isdigit() else 14
        print(f"{Fore.CYAN}Fenêtre ATR fixée à {ATR_WINDOW}{Style.RESET_ALL}")
    else:
        ATR_FIXED = False
        print(f"{Fore.CYAN}Fenêtre ATR variable entre {ATR_WINDOW_LOW} et {ATR_WINDOW_HIGH}{Style.RESET_ALL}")

    # Afficher l'état du filtre IMBULL
    if FILTER_IMBULL:
        print(
            f"{Fore.YELLOW}Filtre IMBULL activé: Condition is_imBullWithPoc_light_short ajoutée aux autres filtres{Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}Filtre IMBULL désactivé: Seuls les filtres standards sont appliqués{Style.RESET_ALL}")

    # Chargement datasets ---------------------------------------------------
    print(f"{Fore.CYAN}Chargement des données…{Style.RESET_ALL}")
    global TRAIN_RAW, VAL_RAW, VAL1_RAW, TEST_RAW
    TRAIN_RAW, VAL_RAW, VAL1_RAW, TEST_RAW = (load_raw_csv(p) for p in (CSV_TRAIN, CSV_VAL, CSV_VAL1, CSV_TEST))

    # Calculer l'ATR pour tous les datasets avec la fenêtre initiale
    for d in (TRAIN_RAW, VAL_RAW, VAL1_RAW, TEST_RAW):
        d['atr_recalc'] = calculate_atr(d, period=ATR_WINDOW)

    # Quartiles ATR globaux -------------------------------------------------
    all_atr = np.concatenate([d['atr_recalc'].values for d in (TRAIN_RAW, VAL_RAW, VAL1_RAW, TEST_RAW)])
    q25, q50, q75 = np.quantile(all_atr, [0.25, 0.50, 0.75])
    quartiles = [round(q25, 1), round(q50, 1), round(q75, 1)]
    print(f"Seuils proposés (quartiles) : {quartiles[0]}, {quartiles[1]}, {quartiles[2]}")

    if input("Figer ces seuils ? (o/n) : ").strip().lower() == 'o':
        LOCK_THRESHOLDS = True
        FIXED_THRESHOLDS = quartiles
        print("Seuils ATR figés pour toute l'optimisation.")
    else:
        LOCK_THRESHOLDS = False
        print("Seuils ATR variables selon optimisation.")

    # Étude Optuna ----------------------------------------------------------
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))

    # Listener clavier ------------------------------------------------------
    start_keyboard_listener()

    # Initialiser la liste des essais valides
    valid_trials = []

    # Statistiques des essais
    total_trials = 0
    valid_trial_count = 0

    # Affichage de la configuration finale
    print(f"\n{Fore.YELLOW}═══════════════════════════════════════════════════════════{Style.RESET_ALL}")
    print(f"{Fore.CYAN}📊 ConfigurationATR CHECK - {DIRECTION.upper()}:{Style.RESET_ALL}")
    print(
        f"   • ATR: {'Fixe' if ATR_FIXED else 'Variable'} ({'fenêtre ' + str(ATR_WINDOW) if ATR_FIXED else f'plage {ATR_WINDOW_LOW}-{ATR_WINDOW_HIGH}'})")
    print(f"   • Filtre IMBULL: {'Activé' if FILTER_IMBULL else 'Désactivé'}")
    if not USE_TRAIN_IN_OPTIMIZATION:
        print(f"   • {Fore.YELLOW}DATASET TRAIN: TESTÉ POUR INFO SEULEMENT{Style.RESET_ALL}")
    else:
        print(f"   • {Fore.GREEN}DATASET TRAIN: UTILISÉ POUR L'OPTIMISATION{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}═══════════════════════════════════════════════════════════{Style.RESET_ALL}\n")

    # Boucle optimisation ---------------------------------------------------
    for done in range(1, N_TRIALS + 1):
        study.optimize(objective, n_trials=1)
        total_trials += 1

        # Vérifier si l'essai courant est valide (pas penalty)
        if study.trials[-1].value > FAILED_PENALTY:
            valid_trial_count += 1

        # Affichage périodique
        if done % PRINT_EVERY == 0 or done == 1:  # Afficher aussi après le premier essai
            valid_pct = valid_trial_count / total_trials * 100 if total_trials > 0 else 0
            print(f"\n{Fore.CYAN}Trial {done}/{N_TRIALS} – "
                  f"Essais ok: {valid_trial_count}/{total_trials} ({valid_pct:.1f}%){Style.RESET_ALL}")
            print(f"Best value : {study.best_value if study.best_trial else 'N/A'}")

            # ✅ CORRECTION : Afficher le contexte avec param_summary
            if best_trial["number"] is not None:
                print(
                    f"\n{Fore.YELLOW}*** BEST so far for {DIRECTION} with {param_summary} ▸ trial {best_trial['number']}  score={Fore.GREEN}{best_trial['score']:.4f}{Style.RESET_ALL}")
                print_best_trial_details(best_trial, done)
            else:
                print(
                    f"\n{Fore.YELLOW}*** BEST so far for {DIRECTION} with {param_summary} ▸ trial None  score=-inf{Style.RESET_ALL}")

            # Exécuter un test périodique
            if done % PRINT_EVERY == 0:
                RUN_TEST = True

        # Test sur demande ou périodique
        if RUN_TEST and best_trial["number"] is not None:
            RUN_TEST = False
            print(f"\n{Fore.YELLOW}Test demandé après {done} essais{Style.RESET_ALL}")

            # ✅ CORRECTION : Utiliser best_trial["params"] au lieu de study.best_params
            result = calculate_test_metrics_and_save(best_trial["params"], best_trial["number"],
                                                     best_trial["atr_window"])

            # Si un nom de fichier a été retourné, le mémoriser
            if result and isinstance(result, tuple) and len(result) > 5:
                filename = result[5]  # Le nom du fichier est en position 5 du tuple
                if filename:
                    last_csv_filename = filename

    # ═══════════════════ SYNTHÈSE FINALE COMPLÈTE ═══════════════════
    print(f"\n{Fore.YELLOW}{'=' * 80}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}🏆 SYNTHÈSE FINALE - MEILLEUR RÉSULTATATRCHECK{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{'=' * 80}{Style.RESET_ALL}")

    if best_trial.get("params"):
        bt = best_trial
        print(
            f"\n{Fore.YELLOW}*** BEST FINAL for {DIRECTION} with {param_summary} ▸ trial {bt['number']}  score={Fore.GREEN}{bt['score']:.4f}{Style.RESET_ALL}")
        print(f"    {Fore.CYAN}[Direction: {DIRECTION.upper()} - ATR checking]{Style.RESET_ALL}")

        print(f"\n    {Fore.CYAN}[RÉSULTATS SUR DATASETS D'ENTRAÎNEMENT/VALIDATION]{Style.RESET_ALL}")

        # ====== MODIFICATION: AFFICHAGE ADAPTÉ SELON L'UTILISATION DE TRAIN ======
        total_t = bt['suc_t'] + bt['fail_t']
        if USE_TRAIN_IN_OPTIMIZATION:
            print(f"    TR WR {Fore.GREEN}{bt['wr_t']:.2%}{Style.RESET_ALL} | "
                  f"pct {bt['pct_t']:.2%} | "
                  f"✓{Fore.GREEN}{bt['suc_t']}{Style.RESET_ALL} "
                  f"✗{Fore.RED}{bt['fail_t']}{Style.RESET_ALL} "
                  f"Total={Fore.CYAN}{total_t}{Style.RESET_ALL} (sessions: {bt['sess_t']})")
        else:
            print(f"    TR WR {Fore.GREEN}{bt['wr_t']:.2%}{Style.RESET_ALL} | "
                  f"pct {bt['pct_t']:.2%} | "
                  f"✓{Fore.GREEN}{bt['suc_t']}{Style.RESET_ALL} "
                  f"✗{Fore.RED}{bt['fail_t']}{Style.RESET_ALL} "
                  f"Total={Fore.CYAN}{total_t}{Style.RESET_ALL} (sessions: {bt['sess_t']}) "
                  f"{Fore.YELLOW}(info seulement){Style.RESET_ALL}")

        total_v = bt['suc_v'] + bt['fail_v']
        total_v1 = bt['suc_v1'] + bt['fail_v1']

        print(f"    V1 WR {Fore.GREEN}{bt['wr_v']:.2%}{Style.RESET_ALL} | "
              f"pct {bt['pct_v']:.2%} | "
              f"✓{Fore.GREEN}{bt['suc_v']}{Style.RESET_ALL} "
              f"✗{Fore.RED}{bt['fail_v']}{Style.RESET_ALL} "
              f"Total={Fore.CYAN}{total_v}{Style.RESET_ALL} (sessions: {bt['sess_v']})")

        print(f"    V2 WR {Fore.GREEN}{bt['wr_v1']:.2%}{Style.RESET_ALL} | "
              f"pct {bt['pct_v1']:.2%} | "
              f"✓{Fore.GREEN}{bt['suc_v1']}{Style.RESET_ALL} "
              f"✗{Fore.RED}{bt['fail_v1']}{Style.RESET_ALL} "
              f"Total={Fore.CYAN}{total_v1}{Style.RESET_ALL} (sessions: {bt['sess_v1']})")

        print(f"    Gap WR moyen: {Fore.YELLOW}{bt['avg_gap_wr']:.2%}{Style.RESET_ALL} | "
              f"Gap PCT moyen: {Fore.YELLOW}{bt['avg_gap_pct']:.2%}{Style.RESET_ALL}")

        # Affichage des paramètres optimaux
        params = bt["params"]
        print(f"\n    {Fore.CYAN}[Paramètres optimaux]{Style.RESET_ALL}")
        print(f"    • Fenêtre ATR: {bt['atr_window']}")
        print(
            f"    • Seuils ATR: {params['atr_threshold_1']:.1f} | {params['atr_threshold_2']:.1f} | {params['atr_threshold_3']:.1f}")
        print(f"    • DiffHighPrice par segment: "
              f"{params['diff_high_atr_1']:.2f} | {params['diff_high_atr_2']:.2f} | "
              f"{params['diff_high_atr_3']:.2f} | {params['diff_high_atr_4']:.2f}")
        print(f"    params ➜ {Fore.MAGENTA}{params}{Style.RESET_ALL}")

        # ====== MODIFICATION: Toujours afficher les métriques détaillées TRAIN ======
        if bt["metrics_train"]:
            train_label = "TRAIN" if USE_TRAIN_IN_OPTIMIZATION else "TRAIN (info seulement)"
            print_atr_segment_metrics(train_label, bt["metrics_train"], params)

        if bt["metrics_val"]:
            print_atr_segment_metrics("VAL", bt["metrics_val"], params)

        if bt["metrics_val1"]:
            print_atr_segment_metrics("VAL1", bt["metrics_val1"], params)

        print(f"\n{Fore.YELLOW}{'=' * 80}{Style.RESET_ALL}")

        # ✅ CORRECTION PRINCIPALE : Utiliser les bons paramètres pour le test final
        calculate_test_metrics(best_trial["params"])
    else:
        print(
            f"{Fore.RED}❌ Aucun meilleur trial trouvé - impossible de calculer les métriques finales{Style.RESET_ALL}")

    # Afficher un résumé des essais valides
    if valid_trials:
        print(f"\n{Fore.GREEN}🏆 RÉSUMÉ DES ESSAIS VALIDES{Style.RESET_ALL}")
        validation_msg = "tous les ensembles" if USE_TRAIN_IN_OPTIMIZATION else "VAL, VAL1 et TEST (TRAIN info seulement)"
        print(f"(Valides sur {validation_msg})")
        print(f"Nombre d'essais valides: {len(valid_trials)}")

        # Afficher les 5 meilleurs essais (triés par score)
        best_valid = sorted(valid_trials, key=lambda x: x["score"], reverse=True)[:5]
        for i, trial in enumerate(best_valid):
            print(f"\n{i + 1}. Essai #{trial['trial_number']} (Score: {trial['score']:.4f})")
            print(f"   ATR Window: {trial['atr_window']}")
            # Toujours afficher TRAIN avec indication si info seulement
            train_info = " (info)" if not trial['use_train_in_optimization'] else ""
            print(
                f"   WR: Train={trial['wr_train']:.2%}{train_info}, Val={trial['wr_val']:.2%}, Val1={trial['wr_val1']:.2%}, Test={trial['wr_test']:.2%}")
            print(
                f"   PCT: Train={trial['pct_train']:.2%}{train_info}, Val={trial['pct_val']:.2%}, Val1={trial['pct_val1']:.2%}, Test={trial['pct_test']:.2%}")

        # Sauvegarder une dernière fois pour être sûr et obtenir le nom de fichier
        filename = save_valid_trials_to_csv()
        if filename:
            last_csv_filename = filename

        # Rappeler le nom du fichier où les essais ont été sauvegardés
        if last_csv_filename:
            print(
                f"\n{Fore.YELLOW}📊 Tous les essais valides ont été sauvegardés dans {last_csv_filename}{Style.RESET_ALL}")
        else:
            print(f"\n{Fore.YELLOW}📊 Tous les essais valides ont été sauvegardés.{Style.RESET_ALL}")
    else:
        validation_sets = "tous les ensembles" if USE_TRAIN_IN_OPTIMIZATION else "VAL, VAL1 et TEST"
        print(f"\n{Fore.RED}❌ Aucun essai valide sur {validation_sets} n'a été trouvé{Style.RESET_ALL}")

    # Proposer de tester sur le jeu de test
    if best_trial["number"] is not None and input(
            "\nCalculer les métriques sur le jeu de TEST ? (o/n): ").strip().lower() == 'o':
        # ✅ CORRECTION : Utiliser best_trial["params"] au lieu de study.best_params
        calculate_test_metrics(best_trial["params"])

    print(f"\n{Fore.YELLOW}{'=' * 80}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}🏁 FIN DE L'OPTIMISATION ATRK CHECKING{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{'=' * 80}{Style.RESET_ALL}")


if __name__ == "__main__":
    main()