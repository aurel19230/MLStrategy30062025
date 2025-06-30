# -*- coding: utf-8 -*-
"""optuna_vwap_reversal_pro_optimizer.py
==========================================
- Optimise VWAP Reversal Pro pour différentes directions (SHORT/LONG)
- Basé sur la structure de l'optimiseur Williams %R
- Utilise les fonctions de Tools.func_features_preprocessing
- AJOUT: Gestion USE_TRAIN_IN_OPTIMIZATION pour contrôler l'utilisation du dataset TRAIN
- ✅ CORRECTION: Bug d'alignement des indices corrigé
"""

from __future__ import annotations
from pathlib import Path
import sys, time, math, optuna, pandas as pd, numpy as np
import threading
import warnings
from datetime import datetime
import chardet
from Tools.func_features_preprocessing import vwap_reversal_pro_optimized

STOP_OPTIMIZATION = False  # touche ² => arrêt propre
TEST_ON_DEMAND = False  # touche & => test immédiat
from pynput import keyboard


# ────────── Key flags ──────────────────────────────────────────


def _on_press(key):
    global STOP_OPTIMIZATION, TEST_ON_DEMAND
    if hasattr(key, "char"):
        if key.char == "²":
            print("\n🛑  Arrêt demandé (²)")
            STOP_OPTIMIZATION = True
        elif key.char == "&":
            print("\n🧪  Test demandé (&) – sera exécuté après ce trial")
            TEST_ON_DEMAND = True


keyboard.Listener(on_press=_on_press, daemon=True).start()

# Désactiver les avertissements
warnings.filterwarnings("ignore")

# Import des modules pour l'interface
try:
    from pynput import keyboard
except ImportError:
    print("Module pynput non disponible - raccourci clavier désactivé")


    class KeyboardListener:
        def __init__(self, *args, **kwargs): pass

        def start(self): pass

        def join(self): pass


    keyboard = type('keyboard', (), {'Listener': KeyboardListener})

try:
    from colorama import init, Fore, Back, Style

    init(autoreset=True)
except ImportError:
    print("Module colorama non disponible - couleurs désactivées")


    class DummyColor:
        def __getattr__(self, name): return ""


    Fore = DummyColor()
    Back = DummyColor()
    Style = DummyColor()

# ═══════════════════════ CONFIGURATION ═══════════════════════════

RANDOM_SEED = 42

# 🔧 CONFIGURATION STRATÉGIE
DIRECTION = "short"  # "short" ou "long"

# ====== MODIFICATION PRINCIPALE: Contrôle d'utilisation du dataset TRAIN ======
USE_TRAIN_IN_OPTIMIZATION = True  # Mettre False pour désactiver l'utilisation du dataset TRAIN dans l'optimisation
USE_TRAIN_IN_OPTIMIZATION = False  # Mettre True pour utiliser TRAIN dans l'optimisation

# Pour VWAP Reversal, l'objectif est toujours de maximiser le WR
OPTIMIZATION_GOAL = "maximize"
GOAL_DESCRIPTION = "MAXIMISER"
import os
from pathlib import Path

print(
    f"{Fore.CYAN}🎯 STRATÉGIE: VWAP Reversal Pro - {DIRECTION.upper()} → {GOAL_DESCRIPTION} le winrate{Style.RESET_ALL}")

# Affichage de l'état du dataset TRAIN
if not USE_TRAIN_IN_OPTIMIZATION:
    print(
        f"{Fore.YELLOW}⚠️  TRAIN DATASET DÉSACTIVÉ pour l'optimisation - Testé pour information seulement{Style.RESET_ALL}")
else:
    print(f"{Fore.GREEN}✔ TRAIN DATASET ACTIVÉ pour l'optimisation{Style.RESET_ALL}")

# Fichiers de données
DIR = R"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject" \
      R"\Sierra_chart\xTickReversal\simu\5_0_5TP_6SL\merge"
print("=" * 80)
print("🔍 TEST D'EXISTENCE DU RÉPERTOIRE")
print("=" * 80)

# Test d'existence avec os.path
print(f"📁 Répertoire testé: {DIR}")
print(f"✅ os.path.exists(): {os.path.exists(DIR)}")
print(f"✅ os.path.isdir(): {os.path.isdir(DIR)}")

# 🔧 CONFIGURATION FLEXIBLE : Le dataset TRAIN est toujours chargé pour les stats finales
Direction = DIRECTION.capitalize()  # Pour compatibilité avec les noms de fichiers
# Chemins des fichiers (TRAIN est toujours défini pour pouvoir afficher les stats finales)
CSV_TRAIN = DIR + Rf"\Step5_5_0_5TP_6SL_010124_200625_extractOnlyFullSession_Only{DIRECTION}_feat__split2_01052024_30092024.csv"
CSV_VAL = DIR + Rf"\Step5_5_0_5TP_6SL_010124_200625_extractOnlyFullSession_Only{DIRECTION}_feat__split3_30092024_28022025.csv"
CSV_VAL1 = DIR + Rf"\Step5_5_0_5TP_6SL_010124_200625_extractOnlyFullSession_Only{DIRECTION}_feat__split4_02032025_15052025.csv"
CSV_TEST = DIR + Rf"\Step5_5_0_5TP_6SL_010124_200625_extractOnlyFullSession_Only{DIRECTION}_feat__split5_15052025_20062025.csv"

# 🔧 PARAMÈTRES D'OPTIMISATION VWAP REVERSAL PRO
# Ranges basés sur vos paramètres existants avec une marge pour l'exploration
LOOKBACK_MIN, LOOKBACK_MAX = 12, 40
MOMENTUM_MIN, MOMENTUM_MAX = 3, 20
Z_WINDOW_MIN, Z_WINDOW_MAX = 20, 60
ATR_PERIOD_MIN, ATR_PERIOD_MAX = 8, 38
ATR_MULT_MIN, ATR_MULT_MAX = 1.2, 3.1
EMA_FILTER_MIN, EMA_FILTER_MAX = 25, 90
VOL_LOOKBACK_MIN, VOL_LOOKBACK_MAX = 3, 17
VOL_RATIO_MIN_MIN, VOL_RATIO_MIN_MAX = 0.1, 0.6

print(f"📊 Optimisation VWAP Reversal Pro - Direction: {DIRECTION}")

# 🔧 CRITÈRES DE VALIDATION
WINRATE_MIN = 0.53  # WR minimum acceptable
PCT_TRADE_MIN = 0.01  # % de candles tradées minimum
MIN_TRADES = 10  # Nombre minimum de trades pour validation

# Paramètres d'optimisation Optuna
N_TRIALS = 5000
PRINT_EVERY = 10
ALPHA = 0.70  # Poids du WR dans le score
LAMBDA_WR = 0.5  # Pénalité pour l'écart de WR entre datasets
LAMBDA_PCT = 0.5  # Pénalité pour l'écart de PCT entre datasets
FAILED_PENALTY = -1.0  # Pénalité pour échec


# ═══════════════════════ CHARGEMENT DES DONNÉES ═══════════════════════

def detect_file_encoding(file_path: str, sample_size: int = 100_000) -> str:
    with open(file_path, 'rb') as f:
        raw = f.read(sample_size)
    return chardet.detect(raw)['encoding']


def load_csv_complete(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Charge les données complètes ET filtrées séparément
    Returns:
        df_complete: DataFrame avec TOUTES les bougies chronologiques (pour VWAP)
        df_filtered: DataFrame avec seulement class_binaire ∈ {0, 1} (pour les métriques)
        nb_sessions: Nombre de sessions
    """
    path = Path(path)
    encoding = detect_file_encoding(str(path))
    if encoding.lower() == "ascii":
        encoding = "ISO-8859-1"

    print(f"{path.name} ➜ encodage détecté: {encoding}")

    # Chargement COMPLET sans filtrage
    df_complete = pd.read_csv(path, sep=";", encoding=encoding, parse_dates=["date"], low_memory=False)

    # Vérifier que les colonnes VWAP nécessaires existent
    required_cols = ['sc_VWAP', 'sc_close', 'sc_high', 'sc_low', 'sc_volume']
    missing_cols = [col for col in required_cols if col not in df_complete.columns]
    if missing_cols:
        print(f"{Fore.RED}❌ Colonnes manquantes: {missing_cols}{Style.RESET_ALL}")
        raise ValueError(f"Colonnes manquantes dans le dataset: {missing_cols}")

    # Correction de sc_sessionStartEnd
    df_complete["sc_sessionStartEnd"] = pd.to_numeric(df_complete["sc_sessionStartEnd"], errors="coerce")
    df_complete = df_complete.dropna(subset=["sc_sessionStartEnd"])
    df_complete["sc_sessionStartEnd"] = df_complete["sc_sessionStartEnd"].astype(int)

    # Compter les sessions
    nb_start = (df_complete["sc_sessionStartEnd"] == 10).sum()
    nb_end = (df_complete["sc_sessionStartEnd"] == 20).sum()
    nb_sessions = min(nb_start, nb_end)

    if nb_start != nb_end:
        print(f"{Fore.YELLOW}⚠️ Incohérence sessions: {nb_start} débuts vs {nb_end} fins{Style.RESET_ALL}")
    else:
        print(f"{Fore.GREEN}✔ {nb_sessions} sessions complètes détectées{Style.RESET_ALL}")

    # Numérotation des sessions
    df_complete["session_id"] = (df_complete["sc_sessionStartEnd"] == 10).cumsum().astype("int32")

    # ✅ CORRECTION DU BUG: Créer le dataset FILTRÉ SANS reset_index
    df_filtered = df_complete[df_complete["class_binaire"].isin([0, 1])].copy()
    # ❌ LIGNE SUPPRIMÉE: df_filtered.reset_index(drop=True, inplace=True)

    print(f"📊 Données complètes: {len(df_complete):,} bougies")
    print(f"📊 Données filtrées: {len(df_filtered):,} bougies ({len(df_filtered) / len(df_complete):.1%})")

    return df_complete, df_filtered, nb_sessions


# ═══════════════════════ CALCUL VWAP REVERSAL PRO ═══════════════════════
def calculate_vwap_reversal_signal(df_complete: pd.DataFrame, df_filtered: pd.DataFrame,
                                   params: dict, direction: str) -> pd.Series:
    """
    Calcule le signal VWAP Reversal Pro sur données COMPLÈTES
    ✅ Version optimisée utilisant l'alignement par index (comme Williams %R)

    Args:
        df_complete: DataFrame avec toutes les bougies chronologiques
        df_filtered: DataFrame avec seulement class_binaire ∈ {0, 1}
        params: Dictionnaire des paramètres VWAP Reversal
        direction: "short" ou "long"

    Returns:
        signal: Serie avec signal (0/1) indexée sur df_filtered
    """

    # 1) Calculer le signal sur données COMPLÈTES (chronologiques)
    signal_complete, status_df = vwap_reversal_pro_optimized(
        df_complete,
        lookback=params['lookback'],
        momentum=params['momentum'],
        z_window=params['z_window'],
        atr_period=params['atr_period'],
        atr_mult=params['atr_mult'],
        ema_filter=params['ema_filter'],
        vol_lookback=params['vol_lookback'],
        vol_ratio_min=params['vol_ratio_min'],
        direction=direction,
        atr_ma="sma"
    )

    # 2) ✅ APPROCHE PAR INDICES (plus rapide et fiable)
    # Ajouter le signal calculé au DataFrame complet
    df_complete['vwap_signal'] = signal_complete

    # Récupérer les signaux correspondant aux indices du DataFrame filtré
    signal_filtered = df_complete.loc[df_filtered.index, 'vwap_signal'].to_numpy()

    # Gestion des NaN éventuels (au cas où certains indices n'existent pas)
    signal_filtered = np.where(np.isnan(signal_filtered), 0, signal_filtered)

    # Debug : vérifier l'alignement (seulement au premier appel)
    if not hasattr(calculate_vwap_reversal_signal, '_debug_printed'):
        print(f"🔧 Alignement VWAP: {len(signal_filtered)} signaux récupérés pour {len(df_filtered)} lignes filtrées")
        print(
            f"   Signaux actifs: {np.sum(signal_filtered == 1)} / {len(signal_filtered)} ({np.mean(signal_filtered):.2%})")
        calculate_vwap_reversal_signal._debug_printed = True

    return pd.Series(signal_filtered.astype(int), index=df_filtered.index)


# ═══════════════════════ MÉTRIQUES ═══════════════════════

def _metrics(df: pd.DataFrame, mask: pd.Series, original_len: int = None) -> tuple[float, float, int, int, int]:
    """Calcule les métriques pour un masque donné"""
    sub = df.loc[mask]
    if sub.empty:
        return 0.0, 0.0, 0, 0, 0

    wins = int((sub["class_binaire"] == 1).sum())
    losses = int((sub["class_binaire"] == 0).sum())
    total = wins + losses

    if len(sub) != total:
        print(f"⚠️ Warning: {len(sub) - total} lignes avec class_binaire invalide ignorées")

    base_len = original_len if original_len is not None else len(df)
    pct_trade = total / base_len
    sessions_covered = sub["session_id"].nunique()

    return wins / total if total > 0 else 0.0, pct_trade, wins, losses, sessions_covered


# ═══════════════════════ FONCTION DE TEST D'ALIGNEMENT ═══════════════════════

def test_alignment(df_complete, df_filtered):
    """Test pour vérifier l'alignement des indices après correction"""
    print(f"\n🔍 TEST D'ALIGNEMENT DES INDICES:")

    # Test sur quelques lignes
    sample_indices = df_filtered.index[:5]
    print(f"  Indices df_filtered[:5]: {sample_indices.tolist()}")
    print(f"  class_binaire df_complete[sample_indices]: {df_complete.loc[sample_indices, 'class_binaire'].tolist()}")
    print(f"  class_binaire df_filtered[:5]: {df_filtered['class_binaire'].iloc[:5].tolist()}")

    # Vérification
    if df_complete.loc[sample_indices, 'class_binaire'].equals(df_filtered['class_binaire'].iloc[:5]):
        print(f"  {Fore.GREEN}✅ Alignement correct !{Style.RESET_ALL}")
    else:
        print(f"  {Fore.RED}❌ Problème d'alignement détecté !{Style.RESET_ALL}")

    # Vérification plus complète
    all_match = df_complete.loc[df_filtered.index, 'class_binaire'].equals(df_filtered['class_binaire'])
    if all_match:
        print(f"  {Fore.GREEN}✅ Alignement complet vérifié sur {len(df_filtered)} lignes !{Style.RESET_ALL}")
    else:
        print(f"  {Fore.RED}❌ Problèmes d'alignement sur le dataset complet !{Style.RESET_ALL}")


# ═══════════════════════ OPTIMISATION OPTUNA ═══════════════════════

# Variables globales pour stocker les datasets
TRAIN_COMPLETE, TRAIN_FILTERED = None, None
VAL_COMPLETE, VAL_FILTERED = None, None
VAL1_COMPLETE, VAL1_FILTERED = None, None
TEST_COMPLETE, TEST_FILTERED = None, None

best_trial = {
    "score": -math.inf,
    "number": None,
    "wr_t": 0.0, "pct_t": 0.0, "suc_t": 0, "fail_t": 0, "sess_t": 0,
    "wr_v": 0.0, "pct_v": 0.0, "suc_v": 0, "fail_v": 0, "sess_v": 0,
    "wr_v1": 0.0, "pct_v1": 0.0, "suc_v1": 0, "fail_v1": 0, "sess_v1": 0,
    "avg_gap_wr": 0.0, "avg_gap_pct": 0.0,
    "params": {}
}


def objective(trial: optuna.trial.Trial) -> float:
    global best_trial

    # Suggérer les paramètres
    params = {
        "lookback": trial.suggest_int("lookback", LOOKBACK_MIN, LOOKBACK_MAX),
        "momentum": trial.suggest_int("momentum", MOMENTUM_MIN, MOMENTUM_MAX),
        "z_window": trial.suggest_int("z_window", Z_WINDOW_MIN, Z_WINDOW_MAX),
        "atr_period": trial.suggest_int("atr_period", ATR_PERIOD_MIN, ATR_PERIOD_MAX),
        "atr_mult": trial.suggest_float("atr_mult", ATR_MULT_MIN, ATR_MULT_MAX, step=0.1),
        "ema_filter": trial.suggest_int("ema_filter", EMA_FILTER_MIN, EMA_FILTER_MAX),
        "vol_lookback": trial.suggest_int("vol_lookback", VOL_LOOKBACK_MIN, VOL_LOOKBACK_MAX),
        "vol_ratio_min": trial.suggest_float("vol_ratio_min", VOL_RATIO_MIN_MIN, VOL_RATIO_MIN_MAX, step=0.05)
    }

    # ====== MODIFICATION: TOUJOURS CALCULER LES MÉTRIQUES TRAIN ======
    signal_train = calculate_vwap_reversal_signal(TRAIN_COMPLETE, TRAIN_FILTERED, params, DIRECTION)
    mask_train = signal_train == 1
    train_len = len(TRAIN_FILTERED)
    wr_t, pct_t, suc_t, fail_t, sess_t = _metrics(TRAIN_FILTERED, mask_train, train_len)

    # VAL et VAL1 obligatoires
    signal_val = calculate_vwap_reversal_signal(VAL_COMPLETE, VAL_FILTERED, params, DIRECTION)
    signal_val1 = calculate_vwap_reversal_signal(VAL1_COMPLETE, VAL1_FILTERED, params, DIRECTION)

    mask_val = signal_val == 1
    mask_val1 = signal_val1 == 1

    val_len = len(VAL_FILTERED)
    val1_len = len(VAL1_FILTERED)

    wr_v, pct_v, suc_v, fail_v, sess_v = _metrics(VAL_FILTERED, mask_val, val_len)
    wr_v1, pct_v1, suc_v1, fail_v1, sess_v1 = _metrics(VAL1_FILTERED, mask_val1, val1_len)

    # ====== VÉRIFICATION DES SEUILS ADAPTÉE ======
    # Pour les seuils, on ne considère que les datasets utilisés dans l'optimisation
    datasets_to_check = [(wr_v, pct_v, suc_v + fail_v), (wr_v1, pct_v1, suc_v1 + fail_v1)]
    if USE_TRAIN_IN_OPTIMIZATION:
        datasets_to_check.append((wr_t, pct_t, suc_t + fail_t))

    for wr, pct, total_trades in datasets_to_check:
        if wr < WINRATE_MIN or pct < PCT_TRADE_MIN or total_trades < MIN_TRADES:
            return FAILED_PENALTY

    # Affichage conditionnel compact
    if trial.number % 10 == 0 or (wr_v >= WINRATE_MIN and wr_v1 >= WINRATE_MIN):
        if USE_TRAIN_IN_OPTIMIZATION:
            print(f"Trial {trial.number:>5d} | "
                  f"LB={params['lookback']:>2d} Mom={params['momentum']:>2d} Z={params['z_window']:>2d} "
                  f"ATR={params['atr_period']:>2d}/{params['atr_mult']:.1f} "
                  f"EMA={params['ema_filter']:>2d} Vol={params['vol_lookback']:>2d}/{params['vol_ratio_min']:.2f} | "
                  f"TR[{Fore.GREEN}{wr_t:.2%}{Style.RESET_ALL}/{pct_t:.2%}] "
                  f"V1[{Fore.GREEN}{wr_v:.2%}{Style.RESET_ALL}/{pct_v:.2%}] "
                  f"V2[{Fore.GREEN}{wr_v1:.2%}{Style.RESET_ALL}/{pct_v1:.2%}]")
        else:
            print(f"Trial {trial.number:>5d} | "
                  f"LB={params['lookback']:>2d} Mom={params['momentum']:>2d} Z={params['z_window']:>2d} "
                  f"ATR={params['atr_period']:>2d}/{params['atr_mult']:.1f} "
                  f"EMA={params['ema_filter']:>2d} Vol={params['vol_lookback']:>2d}/{params['vol_ratio_min']:.2f} | "
                  f"TR[{Fore.YELLOW}{wr_t:.2%}{Style.RESET_ALL}/{pct_t:.2%}](info) "
                  f"V1[{Fore.GREEN}{wr_v:.2%}{Style.RESET_ALL}/{pct_v:.2%}] "
                  f"V2[{Fore.GREEN}{wr_v1:.2%}{Style.RESET_ALL}/{pct_v1:.2%}]")

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

    # Mise à jour du meilleur essai
    if score > best_trial["score"]:
        print(f"{Fore.GREEN}✅ NOUVEAU MEILLEUR ESSAI ! Score: {score:.4f}{Style.RESET_ALL}")
        best_trial = {
            "number": trial.number,
            "score": score,
            # Métriques TRAIN (toujours stockées maintenant)
            "wr_t": wr_t, "pct_t": pct_t, "suc_t": suc_t, "fail_t": fail_t, "sess_t": sess_t,
            "wr_v": wr_v, "pct_v": pct_v, "suc_v": suc_v, "fail_v": fail_v, "sess_v": sess_v,
            "wr_v1": wr_v1, "pct_v1": pct_v1, "suc_v1": suc_v1, "fail_v1": fail_v1, "sess_v1": sess_v1,
            "avg_gap_wr": avg_gap_wr,
            "avg_gap_pct": avg_gap_pct,
            "params": params
        }

    return score


# ═══════════════════════ TEST SUR HOLD-OUT ═══════════════════════

def calculate_test_metrics(params: dict):
    """Calcule les métriques sur le dataset TEST"""
    print(f"\n{Fore.CYAN}🧮 Calcul sur DATASET TEST{Style.RESET_ALL}\n")

    # ====== RAPPEL DES STATISTIQUES FINALES SUR TOUS LES DATASETS ======
    def calculate_dataset_stats(df_complete, df_filtered, df_name, original_len):
        """Calcule et affiche les stats finales d'un dataset"""
        if df_complete is None or df_filtered is None:
            return

        # Calcul du signal
        signal = calculate_vwap_reversal_signal(df_complete, df_filtered, params, DIRECTION)
        mask = signal == 1

        wr, pct, suc, fail, sess = _metrics(df_filtered, mask, original_len)

        # ====== MODIFICATION: Indiquer si TRAIN est utilisé pour l'optimisation ======
        if df_name == "TRAIN" and not USE_TRAIN_IN_OPTIMIZATION:
            info_suffix = f" {Fore.YELLOW}(info seulement){Style.RESET_ALL}"
        else:
            info_suffix = ""

        print(f"    {df_name}{info_suffix}: WR={Fore.GREEN}{wr:.2%}{Style.RESET_ALL}  pct={pct:.2%}  "
              f"✓{Fore.GREEN}{suc}{Style.RESET_ALL} ✗{Fore.RED}{fail}{Style.RESET_ALL}  "
              f"Total={Fore.CYAN}{suc + fail}{Style.RESET_ALL} (sessions: {sess})")

    print(f"{Fore.YELLOW}📊 RAPPEL - Statistiques finales avec paramètres optimaux:{Style.RESET_ALL}")

    if TRAIN_COMPLETE is not None:
        calculate_dataset_stats(TRAIN_COMPLETE, TRAIN_FILTERED, "TRAIN", len(TRAIN_FILTERED))
    calculate_dataset_stats(VAL_COMPLETE, VAL_FILTERED, "VAL  ", len(VAL_FILTERED))
    calculate_dataset_stats(VAL1_COMPLETE, VAL1_FILTERED, "VAL1 ", len(VAL1_FILTERED))

    print(f"\n{Fore.YELLOW}🎯 RÉSULTATS SUR DATASET TEST:{Style.RESET_ALL}")

    # Calcul du signal
    signal_test = calculate_vwap_reversal_signal(TEST_COMPLETE, TEST_FILTERED, params, DIRECTION)
    mask_test = signal_test == 1

    test_len = len(TEST_FILTERED)
    wr_test, pct_test, suc_test, fail_test, sess_test = _metrics(TEST_FILTERED, mask_test, test_len)

    # Affichage détaillé des paramètres
    print(f"VWAP Reversal Pro {DIRECTION.upper()}")
    print(f"Paramètres:")
    print(f"  • Lookback: {params['lookback']}")
    print(f"  • Momentum: {params['momentum']}")
    print(f"  • Z-Window: {params['z_window']}")
    print(f"  • ATR: Period={params['atr_period']}, Mult={params['atr_mult']:.2f}")
    print(f"  • EMA Filter: {params['ema_filter']}")
    print(f"  • Volume: Lookback={params['vol_lookback']}, Ratio={params['vol_ratio_min']:.2f}")

    print(f"\nRésultats:")
    print(f"WR={Fore.GREEN}{wr_test:.2%}{Style.RESET_ALL} | "
          f"PCT={pct_test:.2%} | "
          f"Trades={suc_test + fail_test} (✓{suc_test} ✗{fail_test}) | "
          f"Sessions={sess_test}")

    # Validation
    is_valid = (wr_test >= WINRATE_MIN and pct_test >= PCT_TRADE_MIN and
                (suc_test + fail_test) >= MIN_TRADES)

    print(f"\nCritères: WR ≥ {WINRATE_MIN:.1%}, PCT ≥ {PCT_TRADE_MIN:.1%}, Trades ≥ {MIN_TRADES}")

    if is_valid:
        print(f"{Fore.GREEN}✅ VALIDE{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}❌ REJET{Style.RESET_ALL}")

    return wr_test, pct_test, suc_test, fail_test, sess_test, is_valid


# ═══════════════════════ PROGRAMME PRINCIPAL ═══════════════════════

def main():
    global TRAIN_COMPLETE, TRAIN_FILTERED, VAL_COMPLETE, VAL_FILTERED
    global VAL1_COMPLETE, VAL1_FILTERED, TEST_COMPLETE, TEST_FILTERED
    global TEST_ON_DEMAND, STOP_OPTIMIZATION

    print(f"{Fore.CYAN}═══════════════════════════════════════════════════════════{Style.RESET_ALL}")
    print(f"{Fore.CYAN}   OPTIMISATION VWAP REVERSAL PRO - {DIRECTION.upper()}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}═══════════════════════════════════════════════════════════{Style.RESET_ALL}\n")

    print(f"{Fore.CYAN}Chargement des données...{Style.RESET_ALL}")

    # ====== MODIFICATION: TRAIN toujours chargé mais usage conditionnel ======
    TRAIN_COMPLETE, TRAIN_FILTERED, train_sessions = load_csv_complete(CSV_TRAIN)
    VAL_COMPLETE, VAL_FILTERED, val_sessions = load_csv_complete(CSV_VAL)
    VAL1_COMPLETE, VAL1_FILTERED, val1_sessions = load_csv_complete(CSV_VAL1)
    TEST_COMPLETE, TEST_FILTERED, test_sessions = load_csv_complete(CSV_TEST)

    # ✅ TEST D'ALIGNEMENT APRÈS CHARGEMENT
    test_alignment(TRAIN_COMPLETE, TRAIN_FILTERED)

    print(f"\n📊 RÉSUMÉ DES DATASETS :")
    print("─" * 60)

    datasets_info = []
    if TRAIN_COMPLETE is not None:
        usage_info = " (utilisé pour optimisation)" if USE_TRAIN_IN_OPTIMIZATION else " (info seulement)"
        datasets_info.append(("TRAIN", TRAIN_FILTERED, train_sessions, usage_info))

    datasets_info.extend([
        ("VAL", VAL_FILTERED, val_sessions, ""),
        ("VAL1", VAL1_FILTERED, val1_sessions, ""),
        ("TEST", TEST_FILTERED, test_sessions, "")
    ])

    for label, df_filtered, sessions, usage in datasets_info:
        if df_filtered is not None:
            print(f"{label:5} | lignes filtrées={len(df_filtered):,} | "
                  f"WR brut={(df_filtered['class_binaire'] == 1).mean():.2%} | Sessions={sessions}{usage}")

    print("─" * 60)

    # Configuration de l'étude Optuna
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED)
    )

    print(f"\n{Fore.CYAN}Début de l'optimisation ({N_TRIALS} essais)...{Style.RESET_ALL}\n")

    # Boucle d'optimisation
    start_time = time.time()

    for done in range(1, N_TRIALS + 1):

        # 1) sortie propre demandée ?
        if STOP_OPTIMIZATION:
            break

        # 2) un seul trial
        study.optimize(objective, n_trials=1)

        # 3) test immédiat (&) avec les meilleurs params courants
        if TEST_ON_DEMAND and best_trial["number"] is not None:
            TEST_ON_DEMAND = False
            print("\n🧪 TEST à la volée – meilleurs paramètres connus")
            calculate_test_metrics(best_trial["params"])

        # 4) récap + test périodique toutes PRINT_EVERY itérations
        if done % PRINT_EVERY == 0:
            elapsed = time.time() - start_time
            print(
                f"\n{Fore.CYAN}"
                f"══════════════════ RÉSUMÉ APRÈS {done} ESSAIS "
                f"(VWAP REVERSAL PRO - {DIRECTION.upper()}) "
                f"[{elapsed / 60:.1f} min] "
                f"═════════════════"
                f"{Style.RESET_ALL}"
            )
            if best_trial["number"] is not None:
                params = best_trial["params"]
                print(f"Meilleur: Trial #{best_trial['number']} | Score={best_trial['score']:.4f}")
                print("Paramètres optimaux:")
                print(f"  • Lookback={params['lookback']}, Momentum={params['momentum']}, "
                      f"Z-Window={params['z_window']}")
                print(f"  • ATR: Period={params['atr_period']}, Mult={params['atr_mult']:.2f}")
                print(f"  • EMA={params['ema_filter']}, Vol: LB={params['vol_lookback']}, "
                      f"Ratio={params['vol_ratio_min']:.2f}")

                # ====== MODIFICATION: AFFICHAGE ADAPTÉ SELON L'UTILISATION DE TRAIN ======
                if USE_TRAIN_IN_OPTIMIZATION:
                    metrics_parts = [
                        f"TR[{Fore.GREEN}{best_trial['wr_t']:.2%}{Style.RESET_ALL}/"
                        f"{best_trial['pct_t']:.2%}]",
                        f"V1[{Fore.GREEN}{best_trial['wr_v']:.2%}{Style.RESET_ALL}/"
                        f"{best_trial['pct_v']:.2%}]",
                        f"V2[{Fore.GREEN}{best_trial['wr_v1']:.2%}{Style.RESET_ALL}/"
                        f"{best_trial['pct_v1']:.2%}]"
                    ]
                else:
                    metrics_parts = [
                        f"TR[{Fore.YELLOW}{best_trial['wr_t']:.2%}{Style.RESET_ALL}/"
                        f"{best_trial['pct_t']:.2%}](info)",
                        f"V1[{Fore.GREEN}{best_trial['wr_v']:.2%}{Style.RESET_ALL}/"
                        f"{best_trial['pct_v']:.2%}]",
                        f"V2[{Fore.GREEN}{best_trial['wr_v1']:.2%}{Style.RESET_ALL}/"
                        f"{best_trial['pct_v1']:.2%}]"
                    ]

                print("Métriques: " + " ".join(metrics_parts))

                # 🧮 test systématique sur CSV_TEST
                calculate_test_metrics(best_trial["params"])

    # RÉSUMÉ FINAL
    print(f"\n{Fore.YELLOW}══════════════════ RÉSUMÉ FINAL ══════════════════{Style.RESET_ALL}")
    if best_trial["number"] is not None:
        params = best_trial["params"]
        print(f"\n🏆 MEILLEUR ESSAI #{best_trial['number']} (Score: {best_trial['score']:.4f})")
        print(f"🎯 Stratégie: VWAP Reversal Pro - {DIRECTION.upper()}")

        # ====== MODIFICATION: Indication du mode d'utilisation de TRAIN ======
        if USE_TRAIN_IN_OPTIMIZATION:
            print(f"🔧 Mode: TRAIN utilisé pour l'optimisation")
        else:
            print(f"🔧 Mode: TRAIN testé pour information seulement")

        print(f"\n📋 PARAMÈTRES OPTIMAUX:")
        print(f"params_{DIRECTION} = {{")
        print(f"    'lookback': {params['lookback']},")
        print(f"    'momentum': {params['momentum']},")
        print(f"    'z_window': {params['z_window']},")
        print(f"    'atr_period': {params['atr_period']},")
        print(f"    'atr_mult': {params['atr_mult']:.10f},")
        print(f"    'ema_filter': {params['ema_filter']},")
        print(f"    'vol_lookback': {params['vol_lookback']},")
        print(f"    'vol_ratio_min': {params['vol_ratio_min']:.10f},")
        print(f"}}")

        print(f"\n📊 MÉTRIQUES FINALES:")

        # ====== MODIFICATION: AFFICHAGE ADAPTÉ SELON L'UTILISATION DE TRAIN ======
        if USE_TRAIN_IN_OPTIMIZATION:
            metrics_to_show = [("TRAIN", "t"), ("VAL", "v"), ("VAL1", "v1")]
        else:
            metrics_to_show = [("TRAIN (info)", "t"), ("VAL", "v"), ("VAL1", "v1")]

        for label, prefix in metrics_to_show:
            wr = best_trial[f"wr_{prefix}"]
            pct = best_trial[f"pct_{prefix}"]
            suc = best_trial[f"suc_{prefix}"]
            fail = best_trial[f"fail_{prefix}"]
            total = suc + fail
            print(f"  • {label:12}: WR={wr:.2%} | Trades={total} (✓{suc}, ✗{fail}) | PCT={pct:.2%}")

        print(f"\n🧪 TEST FINAL:")
        calculate_test_metrics(best_trial["params"])

    elapsed_total = time.time() - start_time
    print(f"\n✅ Fin de l'optimisation. Durée totale: {elapsed_total / 60:.1f} minutes")
    print(f"   Moyenne par essai: {elapsed_total / done:.2f} secondes")


if __name__ == "__main__":
    main()