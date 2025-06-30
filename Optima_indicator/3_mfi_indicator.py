# -*- coding: utf-8 -*-
"""optuna_mfi_optimizer_multi_zone.py
====================================
- Optimise MFI (Money Flow Index) pour différentes directions (SHORT/LONG) et zones (OVERBOUGHT/OVERSOLD)
- Logique adaptative selon la stratégie :
  * SHORT + OVERBOUGHT : Maximiser MFI (entrer en SHORT en surachat)
  * SHORT + OVERSOLD   : Minimiser MFI (éviter SHORT en survente)
  * LONG + OVERSOLD    : Maximiser MFI (entrer en LONG en survente)
  * LONG + OVERBOUGHT  : Minimiser MFI (éviter LONG en surachat)
- Support pour USE_TRAIN_IN_OPTIMIZATION : contrôle l'utilisation du dataset TRAIN dans l'optimisation
- ✅ CORRECTION: Bug d'alignement des indices corrigé
"""

from __future__ import annotations
from pathlib import Path
import sys, time, math, optuna, pandas as pd, numpy as np
import threading
import warnings
from datetime import datetime
import chardet
from Tools.func_features_preprocessing import compute_mfi

# ────────── Key flags ──────────────────────────────────────────

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
DIRECTION = "Short"  # "Short" ou "Long"
ZONE = "overbought"  # "overbought" ou "oversold"

# ====== MODIFICATION PRINCIPALE: Contrôle d'utilisation du dataset TRAIN ======
USE_TRAIN_IN_OPTIMIZATION = True  # Mettre False pour désactiver l'utilisation du dataset TRAIN dans l'optimisation
USE_TRAIN_IN_OPTIMIZATION = False  # Mettre True pour utiliser TRAIN dans l'optimisation

# Déduction automatique de l'objectif
if (DIRECTION == "Short" and ZONE == "overbought") or (DIRECTION == "Long" and ZONE == "oversold"):
    OPTIMIZATION_GOAL = "maximize"  # Maximiser le winrate
    GOAL_DESCRIPTION = "MAXIMISER"
else:
    OPTIMIZATION_GOAL = "minimize"  # Minimiser le winrate
    GOAL_DESCRIPTION = "MINIMISER"

print(f"{Fore.CYAN}🎯 STRATÉGIE: {DIRECTION} en zone {ZONE.upper()} → {GOAL_DESCRIPTION} le winrate{Style.RESET_ALL}")

# Affichage de l'état du dataset TRAIN
if not USE_TRAIN_IN_OPTIMIZATION:
    print(
        f"{Fore.YELLOW}⚠️  TRAIN DATASET DÉSACTIVÉ pour l'optimisation - Testé pour information seulement{Style.RESET_ALL}")
else:
    print(f"{Fore.GREEN}✔ TRAIN DATASET ACTIVÉ pour l'optimisation{Style.RESET_ALL}")

# Fichiers de données
DIR = R"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject" \
      R"\Sierra_chart\xTickReversal\simu\5_0_5TP_6SL\merge"

# ====== MODIFICATION: TRAIN est toujours défini pour pouvoir afficher les stats finales ======
# Chemins des fichiers (TRAIN est toujours défini pour pouvoir afficher les stats finales)
CSV_TRAIN = DIR + Rf"\Step5_5_0_5TP_6SL_010124_200625_extractOnlyFullSession_Only{DIRECTION}_feat__split2_01052024_30092024.csv"
CSV_VAL = DIR + Rf"\Step5_5_0_5TP_6SL_010124_200625_extractOnlyFullSession_Only{DIRECTION}_feat__split3_30092024_28022025.csv"
CSV_VAL1 = DIR + Rf"\Step5_5_0_5TP_6SL_010124_200625_extractOnlyFullSession_Only{DIRECTION}_feat__split4_02032025_15052025.csv"
CSV_TEST = DIR + Rf"\Step5_5_0_5TP_6SL_010124_200625_extractOnlyFullSession_Only{DIRECTION}_feat__split5_15052025_20062025.csv"

# 🔧 PARAMÈTRES D'OPTIMISATION ADAPTATIFS POUR MFI
if ZONE == "overbought":
    # Zone surachat : MFI élevé (ex: 70 à 90)
    PERIOD_MIN, PERIOD_MAX = 30, 98
    THRESHOLD_MIN, THRESHOLD_MAX = 65, 99
    THRESHOLD_STEP = 1
    ZONE_DESCRIPTION = "surachat (MFI élevé proche de 100)"
elif ZONE == "oversold":
    # Zone survente : MFI bas (ex: 10 à 40)
    PERIOD_MIN, PERIOD_MAX = 12, 60
    THRESHOLD_MIN, THRESHOLD_MAX = 5, 45
    THRESHOLD_STEP = 1
    ZONE_DESCRIPTION = "survente (MFI bas proche de 0)"

print(f"📊 Zone {ZONE}: {ZONE_DESCRIPTION}")
print(f"🔧 Threshold range: [{THRESHOLD_MIN}, {THRESHOLD_MAX}]")

# 🔧 CRITÈRES DE VALIDATION ADAPTATIFS
if OPTIMIZATION_GOAL == "maximize":
    # Pour maximiser : critères standards
    WINRATE_MIN = 0.525  # WR minimum acceptable
    PCT_TRADE_MIN = 0.02  # % de candles tradées minimum
    COMPARISON_OPERATOR = "≥"
else:
    # Pour minimiser : critères inversés
    WINRATE_MAX = 0.53  # WR maximum acceptable (< 50%)
    PCT_TRADE_MIN = 0.045  # % de candles tradées minimum (inchangé)
    COMPARISON_OPERATOR = "≤"

MIN_TRADES = 10  # Nombre minimum de trades pour validation

# Paramètres d'optimisation Optuna
N_TRIALS = 50000
PRINT_EVERY = 25
ALPHA = 0.70  # Poids du WR dans le score
LAMBDA_WR = 0.7  # Pénalité pour l'écart de WR entre datasets
LAMBDA_PCT = 0.3  # Pénalité pour l'écart de PCT entre datasets
FAILED_PENALTY = -1.0 if OPTIMIZATION_GOAL == "maximize" else 1.0  # Pénalité adaptative

# ✅ Construction du résumé des paramètres sélectionnés
param_summary = (
    f"🛠️ Paramètres sélectionnés :\n"
    f"▪️ Direction : {DIRECTION}\n"
    f"▪️ Zone : {ZONE} ({ZONE_DESCRIPTION})\n"
    f"▪️ Objectif : {GOAL_DESCRIPTION} le winrate\n"
    f"▪️ Period range : [{PERIOD_MIN}, {PERIOD_MAX}]\n"
    f"▪️ Threshold range : [{THRESHOLD_MIN}, {THRESHOLD_MAX}]\n"
    f"▪️ Dataset TRAIN : {'Utilisé pour optimisation' if USE_TRAIN_IN_OPTIMIZATION else 'Info seulement'}\n"
    f"▪️ ALPHA : {ALPHA}, LAMBDA_WR : {LAMBDA_WR}, LAMBDA_PCT : {LAMBDA_PCT}"
)

# Affichage du résumé
print(param_summary)


# ═══════════════════════ CHARGEMENT DES DONNÉES ═══════════════════════

def detect_file_encoding(file_path: str, sample_size: int = 100_000) -> str:
    with open(file_path, 'rb') as f:
        raw = f.read(sample_size)
    return chardet.detect(raw)['encoding']


def load_csv_complete(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Charge les données complètes ET filtrées séparément
    Returns:
        df_complete: DataFrame avec TOUTES les bougies chronologiques (pour MFI)
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


# ═══════════════════════ CALCUL MFI AVEC LOGIQUE ADAPTATIVE ═══════════════════════

def calculate_mfi_signal_adaptive(df_complete: pd.DataFrame, df_filtered: pd.DataFrame,
                                  period: int, threshold: float, direction: str, zone: str) -> pd.Series:
    """
    🔧 Calcule MFI sur données COMPLÈTES avec logique adaptative selon direction/zone
    ✅ Utilise l'alignement par index pour éviter les problèmes de merge sur flottants

    Args:
        df_complete: DataFrame avec toutes les bougies chronologiques
        df_filtered: DataFrame avec seulement class_binaire ∈ {0, 1}
        period: Période pour MFI
        threshold: Seuil de déclenchement
        direction: "Short" ou "Long"
        zone: "overbought" ou "oversold"

    Returns:
        signal: Serie avec signal (0/1) indexée sur df_filtered
    """

    # 1) Calculer MFI sur données COMPLÈTES (chronologiques)
    sc_high = pd.to_numeric(df_complete['sc_high'], errors='coerce')
    sc_low = pd.to_numeric(df_complete['sc_low'], errors='coerce')
    sc_close = pd.to_numeric(df_complete['sc_close'], errors='coerce')
    sc_volume = pd.to_numeric(df_complete['sc_volume'], errors='coerce')
    session_starts = (df_complete['sc_sessionStartEnd'] == 10).values

    # MFI calculé sur TOUTE la chronologie
    mfi_complete = compute_mfi(sc_high.values, sc_low.values, sc_close.values, sc_volume.values,
                               session_starts=session_starts, period=period, fill_value=50)

    # 2) ✅ APPROCHE PAR INDICES (plus rapide et fiable)
    df_complete['mfi'] = mfi_complete
    mfi_filtered = df_complete.loc[df_filtered.index, 'mfi'].to_numpy()

    # Gestion des NaN éventuels
    mfi_filtered = np.where(np.isnan(mfi_filtered), 50, mfi_filtered)

    # 3) 🔧 LOGIQUE ADAPTATIVE pour créer le signal (robuste à la casse)
    direction_upper = direction.capitalize()  # "short" -> "Short", "SHORT" -> "Short"
    zone_lower = zone.lower()  # "OVERBOUGHT" -> "overbought", "OverBought" -> "overbought"

    if direction_upper == "Short" and zone_lower == "overbought":
        # SHORT en surachat : signal = 1 quand MFI >= threshold (valeur élevée)
        signal_condition = mfi_filtered >= threshold
        signal_description = f"SHORT en SURACHAT (MFI >= {threshold})"

    elif direction_upper == "Short" and zone_lower == "oversold":
        # SHORT en survente : signal = 1 quand MFI <= threshold (valeur faible)
        signal_condition = mfi_filtered <= threshold
        signal_description = f"SHORT en SURVENTE (MFI <= {threshold})"

    elif direction_upper == "Long" and zone_lower == "oversold":
        # LONG en survente : signal = 1 quand MFI <= threshold (valeur faible)
        signal_condition = mfi_filtered <= threshold
        signal_description = f"LONG en SURVENTE (MFI <= {threshold})"

    elif direction_upper == "Long" and zone_lower == "overbought":
        # LONG en surachat : signal = 1 quand MFI >= threshold (valeur élevée)
        signal_condition = mfi_filtered >= threshold
        signal_description = f"LONG en SURACHAT (MFI >= {threshold})"

    else:
        raise ValueError(f"Combinaison non supportée: direction={direction_upper}, zone={zone_lower}. "
                         f"Valeurs autorisées: direction=['Short', 'Long'], zone=['overbought', 'oversold']")

    # Debug : afficher la logique utilisée (seulement au premier appel)
    if not hasattr(calculate_mfi_signal_adaptive, '_debug_printed'):
        print(f"🔧 Logique signal: {signal_description}")
        print(f"🔧 Alignement MFI: {len(mfi_filtered)} valeurs récupérées pour {len(df_filtered)} lignes filtrées")
        print(
            f"   Signaux actifs: {np.sum(signal_condition)} / {len(signal_condition)} ({np.mean(signal_condition):.2%})")
        calculate_mfi_signal_adaptive._debug_printed = True

    return pd.Series(signal_condition.astype(int), index=df_filtered.index)


# ═══════════════════════ MÉTRIQUES INCHANGÉES ═══════════════════════

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


# ═══════════════════════ OPTIMISATION OPTUNA ADAPTATIVE ═══════════════════════

# Variables globales pour stocker les datasets (TRAIN toujours chargé maintenant)
TRAIN_COMPLETE, TRAIN_FILTERED = None, None
VAL_COMPLETE, VAL_FILTERED = None, None
VAL1_COMPLETE, VAL1_FILTERED = None, None
TEST_COMPLETE, TEST_FILTERED = None, None

# 🔧 INITIALISATION ADAPTATIVE DU MEILLEUR TRIAL
if OPTIMIZATION_GOAL == "maximize":
    best_trial_init_score = -math.inf
else:
    best_trial_init_score = math.inf

best_trial = {
    "score": best_trial_init_score,
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
    period = trial.suggest_int("period", PERIOD_MIN, PERIOD_MAX)
    threshold = trial.suggest_float("threshold", THRESHOLD_MIN, THRESHOLD_MAX, step=THRESHOLD_STEP)

    params = {
        "period": period,
        "threshold": threshold
    }

    # ====== MODIFICATION: TOUJOURS CALCULER LES MÉTRIQUES TRAIN ======
    signal_train = calculate_mfi_signal_adaptive(TRAIN_COMPLETE, TRAIN_FILTERED,
                                                 period, threshold, DIRECTION, ZONE)
    mask_train = signal_train == 1
    train_len = len(TRAIN_FILTERED)
    wr_t, pct_t, suc_t, fail_t, sess_t = _metrics(TRAIN_FILTERED, mask_train, train_len)

    # VAL et VAL1 obligatoires
    signal_val = calculate_mfi_signal_adaptive(VAL_COMPLETE, VAL_FILTERED,
                                               period, threshold, DIRECTION, ZONE)
    signal_val1 = calculate_mfi_signal_adaptive(VAL1_COMPLETE, VAL1_FILTERED,
                                                period, threshold, DIRECTION, ZONE)

    mask_val = signal_val == 1
    mask_val1 = signal_val1 == 1

    val_len = len(VAL_FILTERED)
    val1_len = len(VAL1_FILTERED)

    wr_v, pct_v, suc_v, fail_v, sess_v = _metrics(VAL_FILTERED, mask_val, val_len)
    wr_v1, pct_v1, suc_v1, fail_v1, sess_v1 = _metrics(VAL1_FILTERED, mask_val1, val1_len)

    # ====== AFFICHAGE ADAPTÉ SELON L'UTILISATION DE TRAIN ======
    if USE_TRAIN_IN_OPTIMIZATION:
        print(f"Trial {trial.number:>5d} | Period={period:>2d} Threshold={threshold:>6.1f} | "
              f"TR[{Fore.GREEN}{wr_t:.2%}{Style.RESET_ALL}/{pct_t:.2%}] "
              f"V1[{Fore.GREEN}{wr_v:.2%}{Style.RESET_ALL}/{pct_v:.2%}] "
              f"V2[{Fore.GREEN}{wr_v1:.2%}{Style.RESET_ALL}/{pct_v1:.2%}]")
    else:
        print(f"Trial {trial.number:>5d} | Period={period:>2d} Threshold={threshold:>6.1f} | "
              f"TR[{Fore.YELLOW}{wr_t:.2%}{Style.RESET_ALL}/{pct_t:.2%}](info) "
              f"V1[{Fore.GREEN}{wr_v:.2%}{Style.RESET_ALL}/{pct_v:.2%}] "
              f"V2[{Fore.GREEN}{wr_v1:.2%}{Style.RESET_ALL}/{pct_v1:.2%}]")

    # ====== VÉRIFICATIONS ADAPTÉES SELON L'UTILISATION DE TRAIN ======
    datasets_to_check = [(wr_v, pct_v), (wr_v1, pct_v1)]
    if USE_TRAIN_IN_OPTIMIZATION:
        datasets_to_check.append((wr_t, pct_t))

    for wr, pct in datasets_to_check:
        if OPTIMIZATION_GOAL == "maximize":
            if wr < WINRATE_MIN or pct < PCT_TRADE_MIN:
                return FAILED_PENALTY
        else:
            if wr > WINRATE_MAX or pct < PCT_TRADE_MIN:
                return FAILED_PENALTY

    if ((suc_v + fail_v) < MIN_TRADES or (suc_v1 + fail_v1) < MIN_TRADES):
        return FAILED_PENALTY

    # ====== CALCUL DES ÉCARTS ADAPTÉ SELON L'UTILISATION DE TRAIN ======
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
        if OPTIMIZATION_GOAL == "maximize":
            score = (ALPHA * (wr_t + wr_v + wr_v1) / 3 +
                     (1 - ALPHA) * (pct_t + pct_v + pct_v1) / 3 -
                     LAMBDA_WR * avg_gap_wr -
                     LAMBDA_PCT * avg_gap_pct)
        else:
            score = (ALPHA * (wr_t + wr_v + wr_v1) / 3 -
                     (1 - ALPHA) * (pct_t + pct_v + pct_v1) / 3 +
                     LAMBDA_WR * avg_gap_wr +
                     LAMBDA_PCT * avg_gap_pct)
    else:
        # Sans TRAIN : écart seulement entre VAL et VAL1
        gap_wr_vv1 = abs(wr_v - wr_v1)
        gap_pct_vv1 = abs(pct_v - pct_v1)

        avg_gap_wr = gap_wr_vv1
        avg_gap_pct = gap_pct_vv1

        # Score sans TRAIN
        if OPTIMIZATION_GOAL == "maximize":
            score = (ALPHA * (wr_v + wr_v1) / 2 +
                     (1 - ALPHA) * (pct_v + pct_v1) / 2 -
                     LAMBDA_WR * avg_gap_wr -
                     LAMBDA_PCT * avg_gap_pct)
        else:
            score = (ALPHA * (wr_v + wr_v1) / 2 -
                     (1 - ALPHA) * (pct_v + pct_v1) / 2 +
                     LAMBDA_WR * avg_gap_wr +
                     LAMBDA_PCT * avg_gap_pct)

    # 🔧 MISE À JOUR ADAPTATIVE DU MEILLEUR ESSAI
    is_better = False
    if OPTIMIZATION_GOAL == "maximize":
        is_better = score > best_trial["score"]
    else:
        is_better = score < best_trial["score"]

    if is_better:
        print(f"{Fore.GREEN}✅ NOUVEAU MEILLEUR ESSAI ! Score: {score:.4f}{Style.RESET_ALL}")
        best_trial = {
            "number": trial.number,
            "score": score,
            "wr_t": wr_t, "pct_t": pct_t, "suc_t": suc_t, "fail_t": fail_t, "sess_t": sess_t,
            "wr_v": wr_v, "pct_v": pct_v, "suc_v": suc_v, "fail_v": fail_v, "sess_v": sess_v,
            "wr_v1": wr_v1, "pct_v1": pct_v1, "suc_v1": suc_v1, "fail_v1": fail_v1, "sess_v1": sess_v1,
            "avg_gap_wr": avg_gap_wr,
            "avg_gap_pct": avg_gap_pct,
            "params": params
        }

    return score


# ═══════════════════════ TEST SUR HOLD-OUT ADAPTATIF ═══════════════════════

def calculate_test_metrics_adaptive(params: dict):
    """Calcule les métriques sur le dataset TEST avec validation adaptative"""
    print(f"\n{Fore.CYAN}🧮 Calcul sur DATASET TEST{Style.RESET_ALL}\n")

    period = params["period"]
    threshold = params["threshold"]

    # ====== RAPPEL DES STATISTIQUES FINALES SUR TOUS LES DATASETS ======
    def calculate_dataset_stats(dataset_name, df_complete, df_filtered):
        """Calcule et affiche les stats finales d'un dataset"""
        if df_complete is None or df_filtered is None:
            return

        signal = calculate_mfi_signal_adaptive(df_complete, df_filtered,
                                               params['period'], params['threshold'],
                                               DIRECTION, ZONE)
        mask = signal == 1
        wr, pct, suc, fail, sess = _metrics(df_filtered, mask, len(df_filtered))

        # ====== MODIFICATION: Indiquer si TRAIN est utilisé pour l'optimisation ======
        if dataset_name == "TRAIN" and not USE_TRAIN_IN_OPTIMIZATION:
            info_suffix = f" {Fore.YELLOW}(info seulement){Style.RESET_ALL}"
        else:
            info_suffix = ""

        print(f"    {dataset_name}{info_suffix}: WR={Fore.GREEN}{wr:.2%}{Style.RESET_ALL}  pct={pct:.2%}  "
              f"✓{Fore.GREEN}{suc}{Style.RESET_ALL} ✗{Fore.RED}{fail}{Style.RESET_ALL}  "
              f"Total={Fore.CYAN}{suc + fail}{Style.RESET_ALL} (sessions: {sess})")

    print(f"{Fore.YELLOW}📊 RAPPEL - Statistiques finales avec paramètres optimaux:{Style.RESET_ALL}")
    print(f"{Fore.CYAN}    Paramètres: Period={period}, Threshold={threshold:.1f}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}    Stratégie: {DIRECTION} en zone {ZONE.upper()}{Style.RESET_ALL}")

    calculate_dataset_stats("TRAIN", TRAIN_COMPLETE, TRAIN_FILTERED)
    calculate_dataset_stats("VAL  ", VAL_COMPLETE, VAL_FILTERED)
    calculate_dataset_stats("VAL1 ", VAL1_COMPLETE, VAL1_FILTERED)

    print(f"\n{Fore.YELLOW}🎯 RÉSULTATS SUR DATASET TEST:{Style.RESET_ALL}")

    # Calcul adaptatif du signal sur TEST
    signal_test = calculate_mfi_signal_adaptive(TEST_COMPLETE, TEST_FILTERED,
                                                period, threshold, DIRECTION, ZONE)
    mask_test = signal_test == 1

    test_len = len(TEST_FILTERED)
    wr_test, pct_test, suc_test, fail_test, sess_test = _metrics(TEST_FILTERED, mask_test, test_len)

    # Affichage
    print(f"MFI {DIRECTION} {ZONE.upper()} (Period={period}, Threshold={threshold:.1f})")
    print(f"WR={Fore.GREEN}{wr_test:.2%}{Style.RESET_ALL} | "
          f"PCT={pct_test:.2%} | "
          f"Trades={suc_test + fail_test} (✓{suc_test} ✗{fail_test}) | "
          f"Sessions={sess_test}")

    # 🔧 VALIDATION ADAPTATIVE
    if OPTIMIZATION_GOAL == "maximize":
        is_valid = (wr_test >= WINRATE_MIN and pct_test >= PCT_TRADE_MIN and
                    (suc_test + fail_test) >= MIN_TRADES)
        criteria = f"WR {COMPARISON_OPERATOR} {WINRATE_MIN:.1%}"
    else:
        is_valid = (wr_test <= WINRATE_MAX and pct_test >= PCT_TRADE_MIN and
                    (suc_test + fail_test) >= MIN_TRADES)
        criteria = f"WR {COMPARISON_OPERATOR} {WINRATE_MAX:.1%}"

    print(f"Critères: {criteria}, PCT ≥ {PCT_TRADE_MIN:.1%}, Trades ≥ {MIN_TRADES}")

    if is_valid:
        print(f"{Fore.GREEN}✅ VALIDE{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}❌ REJET{Style.RESET_ALL}")

    return wr_test, pct_test, suc_test, fail_test, sess_test, is_valid


# ═══════════════════════ PROGRAMME PRINCIPAL (AVEC MODIFICATIONS) ═══════════════════════

def main():
    global TRAIN_COMPLETE, TRAIN_FILTERED, VAL_COMPLETE, VAL_FILTERED
    global VAL1_COMPLETE, VAL1_FILTERED, TEST_COMPLETE, TEST_FILTERED

    print(f"{Fore.CYAN}Chargement des données...{Style.RESET_ALL}")

    # ====== MODIFICATION: TRAIN toujours chargé maintenant ======
    TRAIN_COMPLETE, TRAIN_FILTERED, train_sessions = load_csv_complete(CSV_TRAIN)
    VAL_COMPLETE, VAL_FILTERED, val_sessions = load_csv_complete(CSV_VAL)
    VAL1_COMPLETE, VAL1_FILTERED, val1_sessions = load_csv_complete(CSV_VAL1)
    TEST_COMPLETE, TEST_FILTERED, test_sessions = load_csv_complete(CSV_TEST)

    # ✅ TEST D'ALIGNEMENT APRÈS CHARGEMENT
    test_alignment(TRAIN_COMPLETE, TRAIN_FILTERED)

    print(f"\n📊 RÉSUMÉ DES DATASETS :")

    datasets_info = [
        ("TRAIN", TRAIN_FILTERED, train_sessions),
        ("VAL", VAL_FILTERED, val_sessions),
        ("VAL1", VAL1_FILTERED, val1_sessions),
        ("TEST", TEST_FILTERED, test_sessions)
    ]

    for label, df_filtered, sessions in datasets_info:
        if df_filtered is not None:
            # ====== MODIFICATION: Indiquer le statut d'utilisation pour TRAIN ======
            usage_info = ""
            if label == "TRAIN":
                if USE_TRAIN_IN_OPTIMIZATION:
                    usage_info = " (utilisé pour optimisation)"
                else:
                    usage_info = " (info seulement)"

            print(f"{label:5} | lignes filtrées={len(df_filtered):,} | "
                  f"WR brut={(df_filtered['class_binaire'] == 1).mean():.2%} | Sessions={sessions}{usage_info}")

    print("─" * 60)

    # 🔧 CONFIGURATION ADAPTATIVE DE L'ÉTUDE OPTUNA
    study = optuna.create_study(
        direction=OPTIMIZATION_GOAL,
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED)
    )

    # Boucle d'optimisation
    for done in range(1, N_TRIALS + 1):
        study.optimize(objective, n_trials=1)

        if done % PRINT_EVERY == 0:
            print(
                f"\n{Fore.CYAN}"
                f"══════════════════ RÉSUMÉ APRÈS {done} ESSAIS "
                f"({DIRECTION.upper()} · {ZONE.upper()}) "
                f"[{Path(CSV_TEST).name}] "
                f"═════════════════"
                f"{Style.RESET_ALL}"
            )

            # ✅ CORRECTION : Afficher le contexte avec param_summary
            if best_trial["number"] is not None:
                print(
                    f"\n{Fore.YELLOW}*** BEST so far for {DIRECTION} {ZONE} with {param_summary} ▸ trial {best_trial['number']}  score={Fore.GREEN}{best_trial['score']:.4f}{Style.RESET_ALL}")

                params = best_trial["params"]
                print(f"Meilleur: Trial #{best_trial['number']} | "
                      f"Period={params['period']} Threshold={params['threshold']:.1f} | "
                      f"Score={best_trial['score']:.4f}")

                metrics_parts = []
                # ====== MODIFICATION: Affichage adapté selon l'utilisation de TRAIN ======
                if USE_TRAIN_IN_OPTIMIZATION:
                    metrics_parts.append(
                        f"TR[{Fore.GREEN}{best_trial['wr_t']:.2%}{Style.RESET_ALL}/{best_trial['pct_t']:.2%}]")
                else:
                    metrics_parts.append(
                        f"TR[{Fore.YELLOW}{best_trial['wr_t']:.2%}{Style.RESET_ALL}/{best_trial['pct_t']:.2%}](info)")

                metrics_parts.extend([
                    f"V1[{Fore.GREEN}{best_trial['wr_v']:.2%}{Style.RESET_ALL}/{best_trial['pct_v']:.2%}]",
                    f"V2[{Fore.GREEN}{best_trial['wr_v1']:.2%}{Style.RESET_ALL}/{best_trial['pct_v1']:.2%}]"
                ])

                print(" ".join(metrics_parts))

                # ✅ CORRECTION : Utiliser best_trial["params"] au lieu de study.best_params
                wr_test, pct_test, suc_test, fail_test, sess_test, is_valid = calculate_test_metrics_adaptive(
                    best_trial["params"])
            else:
                print(
                    f"\n{Fore.YELLOW}*** BEST so far for {DIRECTION} {ZONE} with {param_summary} ▸ trial None  score=-inf{Style.RESET_ALL}")

    # ═══════════════════ SYNTHÈSE FINALE COMPLÈTE ═══════════════════
    print(f"\n{Fore.YELLOW}{'=' * 80}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}🏆 SYNTHÈSE FINALE - MEILLEUR RÉSULTAT MFI OPTIMIZER{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{'=' * 80}{Style.RESET_ALL}")

    if best_trial.get("params"):
        bt = best_trial
        print(
            f"\n{Fore.YELLOW}*** BEST FINAL for {DIRECTION} {ZONE} with {param_summary} ▸ trial {bt['number']}  score={Fore.GREEN}{bt['score']:.4f}{Style.RESET_ALL}")
        print(
            f"    {Fore.CYAN}[Direction: {DIRECTION} - Zone: {ZONE.upper()} - Objectif: {GOAL_DESCRIPTION} WR]{Style.RESET_ALL}")

        print(f"\n    {Fore.CYAN}[RÉSULTATS SUR DATASETS D'ENTRAÎNEMENT/VALIDATION]{Style.RESET_ALL}")

        # ====== AFFICHAGE ADAPTÉ SELON L'UTILISATION DE TRAIN ======
        if USE_TRAIN_IN_OPTIMIZATION:
            total_t = bt['suc_t'] + bt['fail_t']
            print(f"    TR WR {Fore.GREEN}{bt['wr_t']:.2%}{Style.RESET_ALL} | "
                  f"pct {bt['pct_t']:.2%} | "
                  f"✓{Fore.GREEN}{bt['suc_t']}{Style.RESET_ALL} "
                  f"✗{Fore.RED}{bt['fail_t']}{Style.RESET_ALL} "
                  f"Total={Fore.CYAN}{total_t}{Style.RESET_ALL} (sessions: {bt['sess_t']})")
        else:
            total_t = bt['suc_t'] + bt['fail_t']
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
        print(f"    • Stratégie: {DIRECTION} en zone {ZONE.upper()} → {GOAL_DESCRIPTION} WR")
        print(f"    • Zone: {ZONE_DESCRIPTION}")
        print(f"    • Dataset TRAIN: {'Utilisé pour optimisation' if USE_TRAIN_IN_OPTIMIZATION else 'Info seulement'}")
        print(f"    • Period: {params['period']}")
        print(f"    • Threshold: {params['threshold']:.1f}")

        # Logique de signal MFI
        if DIRECTION == "Short" and ZONE == "overbought":
            signal_logic = f"Signal = 1 si MFI >= {params['threshold']:.1f} (SHORT en surachat)"
        elif DIRECTION == "Short" and ZONE == "oversold":
            signal_logic = f"Signal = 1 si MFI <= {params['threshold']:.1f} (SHORT en survente)"
        elif DIRECTION == "Long" and ZONE == "oversold":
            signal_logic = f"Signal = 1 si MFI <= {params['threshold']:.1f} (LONG en survente)"
        elif DIRECTION == "Long" and ZONE == "overbought":
            signal_logic = f"Signal = 1 si MFI >= {params['threshold']:.1f} (LONG en surachat)"

        print(f"    • Logique: {signal_logic}")

        # Critères de validation
        if OPTIMIZATION_GOAL == "maximize":
            print(f"    • Critères: WR ≥ {WINRATE_MIN:.1%}, PCT ≥ {PCT_TRADE_MIN:.1%}, Trades ≥ {MIN_TRADES}")
        else:
            print(f"    • Critères: WR ≤ {WINRATE_MAX:.1%}, PCT ≥ {PCT_TRADE_MIN:.1%}, Trades ≥ {MIN_TRADES}")

        print(f"    params ➜ {Fore.MAGENTA}{params}{Style.RESET_ALL}")

        print(f"\n{Fore.YELLOW}{'=' * 80}{Style.RESET_ALL}")

        # ✅ CORRECTION PRINCIPALE : Utiliser les bons paramètres pour le test final
        print(f"{Fore.YELLOW}🎯 RÉSULTATS SUR DATASET TEST:{Style.RESET_ALL}")
        wr_test, pct_test, suc_test, fail_test, sess_test, is_valid = calculate_test_metrics_adaptive(
            best_trial["params"])

    else:
        print(
            f"{Fore.RED}❌ Aucun meilleur trial trouvé - impossible de calculer les métriques finales{Style.RESET_ALL}")

    print(f"\n{Fore.YELLOW}{'=' * 80}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}🏁 FIN DE L'OPTIMISATION MFI{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{'=' * 80}{Style.RESET_ALL}")


if __name__ == "__main__":
    main()