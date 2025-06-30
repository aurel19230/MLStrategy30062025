# -*- coding: utf-8 -*-
"""
optuna_antispring_optimizer_ultra_fast.py
==========================================
Version ULTRA-RAPIDE avec optimisations Numba et pré-calculs
SANS FALLBACK - Erreur claire si problème avec calculate_slopes_and_r2_numba
"""

from __future__ import annotations
from pathlib import Path
import sys, math, warnings, optuna, pandas as pd, numpy as np, chardet
from typing import Tuple
import time
from Tools.func_features_preprocessing import *

# ────────── Numba pour calculs ultra-rapides ──────────────────────────────────
try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
    print("✅ Numba disponible - accélération activée")
except ImportError:
    print("❌ Numba non disponible - performance dégradée")
    NUMBA_AVAILABLE = False


    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


    prange = range

# ────────── Import de votre fonction optimisée ──────────────────────────────────
try:
    from Tools.func_features_preprocessing import calculate_slopes_and_r2_numba

    SLOPES_FUNCTION_AVAILABLE = True
    print("✅ Fonction calculate_slopes_and_r2_numba importée - utilisation de votre optimisation")
except ImportError:
    print("❌ ERREUR CRITIQUE: Impossible d'importer calculate_slopes_and_r2_numba")
    print("   Vérifiez que Tools.func_features_preprocessing contient cette fonction")
    sys.exit("ARRÊT CRITIQUE: fonction slopes manquante - pas de fallback autorisé")

warnings.filterwarnings("ignore")

try:
    from colorama import init, Fore, Back, Style

    init(autoreset=True)
except ImportError:
    print("Module colorama non disponible - couleurs désactivées")


    class DummyColor:
        def __getattr__(self, name):
            return ""


    Fore = DummyColor()
    Back = DummyColor()
    Style = DummyColor()

# ═══════════════════════ CONFIGURATION ULTRA-RAPIDE ═══════════════════════════

RANDOM_SEED = 42
DIRECTION = "LONG"

# ══════════════ FLAGS D'ACTIVATION DES DATASETS ══════════════
USE_TRAIN_IN_OPTIMIZATION = True
USE_TEST_IN_OPTIMIZATION = False  # TEST = objectif principal
USE_VAL_IN_OPTIMIZATION = True
USE_VAL1_IN_OPTIMIZATION = True
USE_UNSEEN_IN_OPTIMIZATION = False

# 🔧 PARAMÈTRES D'OPTIMISATION (RÉDUITS POUR RAPIDITÉ)
VOLUME_THRESHOLD_MIN = -5
VOLUME_THRESHOLD_MAX = 1
DURATION_THRESHOLD_MIN = 1
DURATION_THRESHOLD_MAX = 8
THRESHOLD_STEP = 0.01  # Pas plus gros pour moins d'essais

SLOPE_PERIOD_MIN = 3  # Plage réduite
SLOPE_PERIOD_MAX = 10  # Plage réduite

# 🔧 CRITÈRES DE VALIDATION (ASSOUPLIS)
WINRATE_MIN = 0.505  # Légèrement assoupli
PCT_TRADE_MIN = 0.08  # Légèrement assoupli
MIN_TRADES = 50  # Réduit pour plus de flexibilité
REQUIRE_ALL_POSITIVE_IMPROVEMENTS = True  # NOUVEAU: Toutes les améliorations doivent être positives

# 🆕 MODE ANTI/DIRECT (NOUVEAU)
FORCE_ANTI_MODE = 0 # None=optimise les 2 modes, 0=force ANTI, 1=force DIRECT

# Paramètres Optuna (RÉDUITS)
N_TRIALS = 15000  # Moins d'essais mais plus rapides
PRINT_EVERY = 100  # Affichage plus fréquent

print(f"{Fore.CYAN}🚀 OPTIMISEUR ANTI-SPRING ULTRA-RAPIDE{Style.RESET_ALL}")
print(f"📊 Plages optimisées pour rapidité:")
print(f"   Volume: [{VOLUME_THRESHOLD_MIN}, {VOLUME_THRESHOLD_MAX}] (step: {THRESHOLD_STEP})")
print(f"   Duration: [{DURATION_THRESHOLD_MIN}, {DURATION_THRESHOLD_MAX}] (step: {THRESHOLD_STEP})")
print(f"🎯 Critères de validation:")
print(f"   Winrate min: {WINRATE_MIN:.1%}")
print(f"   % trades min: {PCT_TRADE_MIN:.1%}")
print(f"   Trades min: {MIN_TRADES}")
print(f"   Toutes améliorations positives: {REQUIRE_ALL_POSITIVE_IMPROVEMENTS}")
print(f"🔧 Conditions de filtrage: 4 types (V<D<, V>D<, V<D>, V>D>)")
if FORCE_ANTI_MODE is None:
    print(f"🔄 Mode: OPTIMISÉ (ANTI + DIRECT) - 8 stratégies")
elif FORCE_ANTI_MODE == 0:
    print(f"🔄 Mode: FORCÉ ANTI - 4 stratégies")
elif FORCE_ANTI_MODE == 1:
    print(f"🔄 Mode: FORCÉ DIRECT - 4 stratégies")
print()

# Fichiers
DIR = r"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra_chart\xTickReversal\simu\5_0_5TP_6SL\merge"
TEMPLATE = (DIR + rf"\Step5_5_0_5TP_6SL_010124_200625_extractOnlyFullSession_Only{DIRECTION}_feat__split{{split}}.csv")

CSV_TRAIN = TEMPLATE.format(split="1_01012024_01052024")
CSV_TEST = TEMPLATE.format(split="2_01052024_30092024")
CSV_VAL = TEMPLATE.format(split="3_30092024_28022025")
CSV_VAL1 = TEMPLATE.format(split="4_02032025_15052025")
CSV_UNSEEN = TEMPLATE.format(split="5_15052025_20062025")


# ═══════════════════════ CHARGEMENT OPTIMISÉ ═══════════════════════

def _enc(p: str, n=40_000):
    """Détection encodage optimisée"""
    with open(p, "rb") as f:
        raw = f.read(n)
    enc = chardet.detect(raw)["encoding"]
    return "ISO-8859-1" if enc and enc.lower() == "ascii" else enc


def load_csv_ultra_fast(path):
    """Chargement CSV ultra-optimisé"""
    print(f"📥 Chargement rapide: {Path(path).name}")

    df = pd.read_csv(path, sep=";", encoding=_enc(path))

    # Nettoyage minimal mais efficace
    df["sc_sessionStartEnd"] = pd.to_numeric(df["sc_sessionStartEnd"], errors="coerce").astype("Int16")
    df = df.dropna(subset=["sc_sessionStartEnd"])
    df["session_id"] = (df["sc_sessionStartEnd"] == 10).cumsum().astype("int32")

    # Filtrage trades valides
    df_trades = df[df["class_binaire"].isin([0, 1])].copy()

    return df, df_trades


# ═══════════════════════ UTILISATION DE VOTRE FONCTION OPTIMISÉE ═══════════════════════


@njit
def apply_filter_numba(volume_slopes, duration_slopes, class_binaire,
                       vol_threshold, dur_threshold, condition_type, anti_mode):
    """
    Application ultra-rapide du filtre spring avec Numba
    condition_type: 0=V<D< | 1=V>D< | 2=V<D> | 3=V>D>
    anti_mode: 0=ANTI (garde tout SAUF springs) | 1=DIRECT (garde SEULEMENT springs)
    """
    n = len(volume_slopes)

    # Masque Spring avec condition variable
    spring_mask = np.zeros(n, dtype=np.bool_)

    for i in range(n):
        if condition_type == 0:  # V<D< : Volume < et Duration
            spring_mask[i] = (volume_slopes[i] < vol_threshold) and (duration_slopes[i] < dur_threshold)
        elif condition_type == 1:  # V>D< : Volume > et Duration
            spring_mask[i] = (volume_slopes[i] > vol_threshold) and (duration_slopes[i] < dur_threshold)
        elif condition_type == 2:  # V<D> : Volume < et Duration >
            spring_mask[i] = (volume_slopes[i] < vol_threshold) and (duration_slopes[i] > dur_threshold)
        elif condition_type == 3:  # V>D> : Volume > et Duration >
            spring_mask[i] = (volume_slopes[i] > vol_threshold) and (duration_slopes[i] > dur_threshold)

    # Choisir le masque selon le mode
    if anti_mode == 0:
        keep_mask = ~spring_mask  # Mode ANTI : garde tout SAUF les springs
    else:
        keep_mask = spring_mask  # Mode DIRECT : garde SEULEMENT les springs

    # Compter les résultats
    original_trades = n
    filtered_trades = np.sum(keep_mask)
    spring_detected = np.sum(spring_mask)

    if anti_mode == 0:
        spring_removed = spring_detected  # En mode ANTI, on "supprime" les springs
    else:
        spring_removed = original_trades - spring_detected  # En mode DIRECT, on "supprime" les non-springs

    if filtered_trades == 0:
        return 0.0, 0.0, 0, spring_removed

    # Calculer winrates
    original_wins = np.sum(class_binaire)
    filtered_wins = np.sum(class_binaire[keep_mask])

    original_wr = original_wins / original_trades
    filtered_wr = filtered_wins / filtered_trades

    return original_wr, filtered_wr, filtered_trades, spring_removed


# ═══════════════════════ CLASSES DE DONNÉES PRÉ-CALCULÉES ═══════════════════════

class FastDataset:
    """Container optimisé pour un dataset avec pré-calculs"""

    def __init__(self, name, df_complete, df_trades):
        self.name = name
        print(f"📊 Préparation {name}...")

        # Données de base
        self.n_trades = len(df_trades)
        self.original_wr = df_trades['class_binaire'].mean()

        # Conversion en arrays pour votre fonction
        self.class_binaire = df_trades['class_binaire'].values.astype(np.int8)

        # Préparation des données complètes pour calculate_slopes_and_r2_numba
        self.df_complete_sorted = df_complete.sort_values(['session_id', 'date']).reset_index(drop=True)

        # Arrays globaux pour votre fonction
        self.durations = self.df_complete_sorted['sc_candleDuration'].values.astype(np.float64)
        self.volumes = self.df_complete_sorted['sc_volume_perTick'].values.astype(np.float64)

        # Session starts pour votre fonction - CONVERSION EXPLICITE EN NUMPY
        self.session_starts = (self.df_complete_sorted['sc_sessionStartEnd'] == 10).values.astype(np.bool_)

        # Mapping des trades vers les indices complets
        self.trade_indices_in_complete = []
        for trade_idx in df_trades.index:
            # Trouver la position dans df_complete_sorted
            matching_positions = np.where(self.df_complete_sorted.index == trade_idx)[0]
            if len(matching_positions) > 0:
                self.trade_indices_in_complete.append(matching_positions[0])

        self.trade_indices_in_complete = np.array(self.trade_indices_in_complete, dtype=np.int32)

        print(f"   ✅ {len(self.trade_indices_in_complete)} trades mappés sur {len(self.durations)} bougies complètes")

    def calculate_slopes_fast(self, n_periods):
        """
        Calcul ultra-rapide des slopes avec VOTRE fonction optimisée
        """
        print(f"   🔧 Calcul slopes {self.name} avec calculate_slopes_and_r2_numba (période {n_periods})...")

        # PAR :
        vol_slopes_complete, dur_slopes_complete, r2_vol, r2_dur = calculate_slopes_with_optimized_function(
            self.durations, self.volumes, self.session_starts, n_periods
        )

        # Extraire seulement les slopes correspondant aux trades
        vol_slopes_trades = vol_slopes_complete[self.trade_indices_in_complete]
        dur_slopes_trades = dur_slopes_complete[self.trade_indices_in_complete]

        # Validation des résultats
        if len(vol_slopes_trades) != self.n_trades:
            raise ValueError(f"Taille mismatch: {len(vol_slopes_trades)} slopes vs {self.n_trades} trades")

        # Gestion des NaN (remplacer par 0)
        vol_slopes_trades = np.nan_to_num(vol_slopes_trades, nan=0.0)
        dur_slopes_trades = np.nan_to_num(dur_slopes_trades, nan=0.0)

        print(f"   ✅ Slopes calculées: Vol[{np.min(vol_slopes_trades):.4f}, {np.max(vol_slopes_trades):.4f}], "
              f"Dur[{np.min(dur_slopes_trades):.4f}, {np.max(dur_slopes_trades):.4f}]")

        return vol_slopes_trades, dur_slopes_trades, self.class_binaire


# ═══════════════════════ GESTIONNAIRE DE CACHE INTELLIGENT ═══════════════════════

class SlopeCache:
    """Cache intelligent pour éviter les recalculs"""

    def __init__(self):
        self.cache = {}  # {period: {dataset_name: (vol_slopes, dur_slopes, class_binaire)}}
        self.hit_count = 0
        self.miss_count = 0

    def get_slopes(self, dataset, period):
        """Récupère les slopes du cache ou les calcule"""
        if period not in self.cache:
            self.cache[period] = {}

        if dataset.name in self.cache[period]:
            self.hit_count += 1
            return self.cache[period][dataset.name]

        # Cache miss - calculer
        self.miss_count += 1
        start_time = time.time()
        vol_slopes, dur_slopes, class_binaire = dataset.calculate_slopes_fast(period)
        calc_time = time.time() - start_time

        # Stocker dans le cache
        self.cache[period][dataset.name] = (vol_slopes, dur_slopes, class_binaire)

        if self.miss_count <= 5:  # Afficher seulement les premiers
            print(f"   🔄 Cache miss: {dataset.name} période {period} calculée en {calc_time:.2f}s")

        return vol_slopes, dur_slopes, class_binaire

    def get_stats(self):
        """Statistiques du cache"""
        total = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total * 100 if total > 0 else 0
        return f"Cache: {self.hit_count} hits, {self.miss_count} miss (taux: {hit_rate:.1f}%)"


# ═══════════════════════ OPTIMISATION OPTUNA ULTRA-RAPIDE ═══════════════════════

# Variables globales
FAST_DATASETS = {}
SLOPE_CACHE = SlopeCache()

best_trial = {
    "score": -math.inf,
    "number": None,
    "all_metrics": {},
    "params": {}
}


def objective_ultra_fast(trial: optuna.trial.Trial) -> float:
    """Fonction objective ultra-rapide avec Numba et cache"""
    global best_trial

    # Suggérer les paramètres - AVEC MODE ANTI/DIRECT CONFIGURABLE
    volume_threshold = trial.suggest_float("volume_threshold",
                                           VOLUME_THRESHOLD_MIN, VOLUME_THRESHOLD_MAX,
                                           step=THRESHOLD_STEP)
    duration_threshold = trial.suggest_float("duration_threshold",
                                             DURATION_THRESHOLD_MIN, DURATION_THRESHOLD_MAX,
                                             step=THRESHOLD_STEP)
    slope_period = trial.suggest_int("slope_period", SLOPE_PERIOD_MIN, SLOPE_PERIOD_MAX)
    condition_type = trial.suggest_categorical("condition_type", [0, 1, 2, 3])

    # Mode ANTI/DIRECT : optimisé ou forcé selon FORCE_ANTI_MODE
    if FORCE_ANTI_MODE is None:
        anti_mode = trial.suggest_categorical("anti_mode", [0, 1])  # Optimise les 2 modes
    else:
        anti_mode = FORCE_ANTI_MODE  # Force le mode choisi

    params = {
        "volume_threshold": volume_threshold,
        "duration_threshold": duration_threshold,
        "slope_period": slope_period,
        "condition_type": condition_type,
        "anti_mode": anti_mode
    }

    # Calculer métriques pour tous les datasets
    all_metrics = {}
    winrates_for_score = []
    all_improvements_positive = True  # Flag pour vérifier toutes les améliorations

    for dataset_name, dataset in FAST_DATASETS.items():
        # Récupérer slopes (avec cache intelligent)
        vol_slopes, dur_slopes, class_binaire = SLOPE_CACHE.get_slopes(dataset, slope_period)

        if len(vol_slopes) == 0:
            continue

        # Application ultra-rapide du filtre avec Numba - AVEC ANTI_MODE
        original_wr, filtered_wr, filtered_trades, spring_removed = apply_filter_numba(
            vol_slopes, dur_slopes, class_binaire,
            volume_threshold, duration_threshold, condition_type, anti_mode
        )

        retention_rate = filtered_trades / len(class_binaire)
        winrate_improvement = filtered_wr - original_wr

        # Vérifier si amélioration positive
        if winrate_improvement <= 0:
            all_improvements_positive = False

        metrics = {
            'original_trades': len(class_binaire),
            'filtered_trades': filtered_trades,
            'retention_rate': retention_rate,
            'original_winrate': original_wr,
            'filtered_winrate': filtered_wr,
            'winrate_improvement': winrate_improvement,
            'spring_trades_removed': spring_removed
        }

        all_metrics[dataset_name] = metrics

        # Validation selon les flags
        should_validate = (
                (dataset_name == 'TRAIN' and USE_TRAIN_IN_OPTIMIZATION) or
                (dataset_name == 'TEST' and USE_TEST_IN_OPTIMIZATION) or
                (dataset_name == 'VAL' and USE_VAL_IN_OPTIMIZATION) or
                (dataset_name == 'VAL1' and USE_VAL1_IN_OPTIMIZATION) or
                (dataset_name == 'UNSEEN' and USE_UNSEEN_IN_OPTIMIZATION)
        )

        if should_validate:
            # Vérifications existantes
            if (filtered_trades < MIN_TRADES or
                    retention_rate < PCT_TRADE_MIN or
                    filtered_wr < WINRATE_MIN):
                return -math.inf

            winrates_for_score.append(filtered_wr)

    # NOUVELLE VALIDATION: Toutes les améliorations doivent être positives
    if REQUIRE_ALL_POSITIVE_IMPROVEMENTS and not all_improvements_positive:
        return -math.inf

    # Score
    if not winrates_for_score:
        return -math.inf

    score = np.mean(winrates_for_score)

    # Bonus TEST si pas dans optimisation
    if not USE_TEST_IN_OPTIMIZATION and 'TEST' in all_metrics:
        score += all_metrics['TEST']['winrate_improvement'] * 2

    # Affichage compact avec condition
    if trial.number % 20 == 0:  # Affichage moins fréquent pour rapidité
        condition_symbols = ["V<D<", "V>D<", "V<D>", "V>D>"]
        condition_str = condition_symbols[condition_type]

        print(
            f"T{trial.number:>4d} | V{volume_threshold:>+5.1f} D{duration_threshold:>+5.1f} P{slope_period:>2d} {condition_str} | ",
            end="")

        for name in ['TEST', 'TRAIN', 'VAL', 'VAL1']:
            if name in all_metrics:
                m = all_metrics[name]
                color = Fore.CYAN if name == 'TEST' else Fore.GREEN
                print(f"{name[0]}{color}{m['winrate_improvement']:+.1%}{Style.RESET_ALL}/"
                      f"{m['retention_rate']:.0%} ", end="")

        print(f"S={score:+.3f}")

    # Mise à jour du meilleur
    if score > best_trial["score"]:
        if trial.number % 20 == 0 or score > best_trial["score"] + 0.01:
            print(f"{Fore.GREEN}✅ NOUVEAU RECORD ! T{trial.number} Score: {score:+.4f}{Style.RESET_ALL}")

        best_trial = {
            "number": trial.number,
            "score": score,
            "all_metrics": all_metrics,
            "params": params
        }

    return score


# ═══════════════════════ PROGRAMME PRINCIPAL ULTRA-RAPIDE ═══════════════════════

def main():
    print(f"{Fore.CYAN}🚀 DÉMARRAGE OPTIMISEUR ULTRA-RAPIDE{Style.RESET_ALL}")

    start_time = time.time()

    # Chargement ultra-rapide
    print(f"\n📥 Chargement ultra-rapide des datasets...")
    datasets_info = [
        ("TRAIN", CSV_TRAIN),
        ("TEST", CSV_TEST),
        ("VAL", CSV_VAL),
        ("VAL1", CSV_VAL1),
        ("UNSEEN", CSV_UNSEEN)
    ]

    for name, path in datasets_info:
        df_complete, df_trades = load_csv_ultra_fast(path)
        if not df_complete.empty and not df_trades.empty:
            FAST_DATASETS[name] = FastDataset(name, df_complete, df_trades)

    load_time = time.time() - start_time
    print(f"✅ Datasets chargés en {load_time:.1f}s")

    # Pré-calcul d'une période pour initialiser
    print(f"\n🔄 Pré-calcul initial (période {SLOPE_PERIOD_MIN})...")
    init_start = time.time()

    for dataset in FAST_DATASETS.values():
        SLOPE_CACHE.get_slopes(dataset, SLOPE_PERIOD_MIN)

    init_time = time.time() - init_start
    print(f"✅ Pré-calcul terminé en {init_time:.1f}s")

    # Optimisation ultra-rapide
    print(f"\n🏃‍♂️ Optimisation ultra-rapide - {N_TRIALS} trials...")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED)
    )

    opt_start = time.time()

    # Boucle d'optimisation
    for done in range(1, N_TRIALS + 1):
        study.optimize(objective_ultra_fast, n_trials=1)

        if done % PRINT_EVERY == 0:
            elapsed = time.time() - opt_start
            speed = done / elapsed
            eta = (N_TRIALS - done) / speed / 60 if speed > 0 else 0

            print(f"\n{Fore.YELLOW}📊 Après {done} trials ({speed:.1f} trials/s) - ETA: {eta:.1f}min{Style.RESET_ALL}")
            print(f"{SLOPE_CACHE.get_stats()}")

            if best_trial["number"] is not None:
                bt = best_trial
                params = bt["params"]
                condition_symbols = ["V<D<", "V>D<", "V<D>", "V>D>"]
                condition_desc = condition_symbols[params['condition_type']]

                print(f"🏆 Meilleur: T{bt['number']} | Vol={params['volume_threshold']:+.3f} "
                      f"Dur={params['duration_threshold']:+.3f} Period={params['slope_period']} "
                      f"Cond={params['condition_type']}({condition_desc}) | Score={bt['score']:+.4f}")

                # Dans l'affichage des résultats
                if 'TEST' in bt["all_metrics"]:
                    test_m = bt["all_metrics"]['TEST']

                    # Calculs détaillés
                    original_trades = test_m['original_trades']
                    filtered_trades = test_m['filtered_trades']
                    trades_removed = original_trades - filtered_trades

                    original_winners = int(original_trades * test_m['original_winrate'])
                    filtered_winners = int(filtered_trades * test_m['filtered_winrate'])

                    tp_removed = original_winners - filtered_winners  # Vrais positifs supprimés
                    tn_removed = trades_removed - tp_removed  # Vrais négatifs supprimés

                    print(f"🎯 TEST Détaillé:")
                    print(f"   📈 TP supprimés: {tp_removed:,} bons trades perdus")
                    print(f"   📉 TN supprimés: {tn_removed:,} mauvais trades évités")
                    print(f"   ⚖️ Ratio TN/TP: {tn_removed / tp_removed:.2f} (>1.0 = bon filtre)")

                # Affichage détaillé de tous les datasets avec indicateur de validité
                print("📈 Détail datasets:")
                all_positive = True
                for name in ['TEST', 'TRAIN', 'VAL', 'VAL1', 'UNSEEN']:
                    if name in bt["all_metrics"]:
                        m = bt["all_metrics"][name]
                        improvement_positive = m['winrate_improvement'] > 0
                        if not improvement_positive:
                            all_positive = False

                        color = Fore.CYAN if name == 'TEST' else Fore.GREEN
                        improvement_color = Fore.GREEN if improvement_positive else Fore.RED
                        indicator = "✅" if improvement_positive else "❌"

                        print(
                            f"   {name}: {indicator} {improvement_color}{m['winrate_improvement']:+.2%}{Style.RESET_ALL} "
                            f"WR | {m['retention_rate']:.1%} trades | "
                            f"{m['original_winrate']:.1%}→{m['filtered_winrate']:.1%}")

                validity_indicator = "🟢 VALIDE" if all_positive else "🔴 INVALIDE"
                print(f"🎯 Statut global: {validity_indicator} (toutes améliorations positives: {all_positive})")

    # Résultats finaux
    total_time = time.time() - start_time
    print(f"\n🏁 Optimisation terminée en {total_time:.1f}s")
    print(f"⚡ Performance: {N_TRIALS / total_time:.1f} trials/seconde")
    print(f"📊 {SLOPE_CACHE.get_stats()}")

    if best_trial["number"] is not None:
        bt = best_trial
        params = bt["params"]

        print(f"\n{Fore.CYAN}🏆 RÉSULTATS FINAUX ULTRA-RAPIDES{Style.RESET_ALL}")
        print(f"Paramètres optimaux:")
        print(f"  Volume threshold: {params['volume_threshold']:+.6f}")
        print(f"  Duration threshold: {params['duration_threshold']:+.6f}")
        print(f"  Slope period: {params['slope_period']} bougies")
        print(
            f"  Condition type: {params['condition_type']} ({['V<D<', 'V>D<', 'V<D>', 'V>D>'][params['condition_type']]})")
        print(f"  Score: {bt['score']:+.4f}")

        print(f"\nCode à utiliser:")
        print(f"VOLUME_THRESHOLD = {params['volume_threshold']}")
        print(f"DURATION_THRESHOLD = {params['duration_threshold']}")
        print(f"SLOPE_PERIOD = {params['slope_period']}")
        print(
            f"CONDITION_TYPE = {params['condition_type']}  # {['V<D<', 'V>D<', 'V<D>', 'V>D>'][params['condition_type']]}")

        print(f"\n📊 Résumé par dataset:")
        for name, metrics in bt["all_metrics"].items():
            improvement_color = Fore.GREEN if metrics['winrate_improvement'] > 0 else Fore.RED
            print(f"{name}: {metrics['original_trades']:,} → {metrics['filtered_trades']:,} "
                  f"({improvement_color}{metrics['winrate_improvement']:+.2%}{Style.RESET_ALL})")

    else:
        print(f"{Fore.RED}❌ Aucun résultat valide trouvé{Style.RESET_ALL}")


if __name__ == "__main__":
    main()