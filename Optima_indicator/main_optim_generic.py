# -*- coding: utf-8 -*-
"""optuna_imbalance_low_high_detection.py – version 8 SIMPLIFIÉE + SAUVEGARDE
================================================
- Stratégie "Imbalance Low High Detection" supportant SHORT et LONG
- SIMPLIFICATION RADICALE: Une seule logique de filtrage global
- Suppression des 3 conditions artificielles
- Optimisation directe des features importantes
- NOUVEAU: Sauvegarde automatique des résultats valides
- Raccourci clavier « & » pour déclencher un calcul immédiat sur le jeu TEST
"""
from __future__ import annotations

# ─────────────────── CONFIG ──────────────────────────────────────
from pathlib import Path
import sys, time, math, optuna, pandas as pd, numpy as np
import threading
import json
from datetime import datetime

# Remplacer msvcrt par pynput
from pynput import keyboard

# Ajout de colorama pour les affichages colorés
from colorama import init, Fore, Back, Style

# Initialiser colorama (nécessaire pour Windows)
init(autoreset=True)

RANDOM_SEED = 42

# ═══════════════════════ CONFIGURATION DIRECTION ═══════════════════════

DIRECTION = "short"  # "short" ou "long"

# Mapping des colonnes selon la direction
COLUMN_MAPPING = {
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
VOLUME_COL = COLUMN_MAPPING[DIRECTION]["volume_col"]
IMBALANCE_COL = COLUMN_MAPPING[DIRECTION]["imbalance_col"]
STRATEGY_DESC = COLUMN_MAPPING[DIRECTION]["description"]

print(f"{Fore.CYAN}🎯 STRATÉGIE SIMPLIFIÉE: Imbalance Low High Detection - {DIRECTION.upper()}{Style.RESET_ALL}")
print(f"{Fore.YELLOW}📊 {STRATEGY_DESC}{Style.RESET_ALL}")
print(f"{Fore.GREEN}🔧 Colonnes utilisées: {VOLUME_COL} & {IMBALANCE_COL}{Style.RESET_ALL}")

# ═══════════════════════ CONFIGURATION SIMPLIFIÉE DES FEATURES ═══════════════════════

# Configuration simplifiée - Une seule logique de filtrage
FEATURES_CONFIG = {
    # === FEATURES PRINCIPALES D'IMBALANCE ===
    "is_imBullWithPoc_light_short": {
        "name": "is_imBullWithPoc_light_short",
        "active": True,
        "min_bound": 1,
        "max_bound": 1,
        "step": 1,
        "condition_type": "between"
    },
    "poc_positionVSClose": {
        "name": "sc_diffPriceClosePoc_0_0",
        "active": True,
        "min_bound": -1.25,
        "max_bound": 0,
        "step": 0.25,
        "condition_type": "between"
    },

    # === FEATURES GLOBALES ADDITIONNELLES ===
    "candle_size": {
        "name": "sc_candleSizeTicks",
        "active": True,
        "min_bound": 10.0,
        "max_bound": 60.0,
        "step": 0.25,
        "condition_type": "between"
    },
    "sc_volPocVolRevesalXContRatio": {
        "name": "sc_volPocVolRevesalXContRatio",
        "active": True,
        "min_bound": 0,
        "max_bound": 1,
        "step": 0.025,
        "condition_type": "between"
    },
    "volRevVolRevesalXContRatio": {
        "name": "volRevVolRevesalXContRatio",
        "active": True,  # Désactivée par défaut
        "min_bound": 0,
        "max_bound": 1,
        "step": 0.025,
        "condition_type": "between"
    },
    "deltaRev_volRev_ratio": {
        "name": "deltaRev_volRev_ratio",
        "active": True,  # Désactivée par défaut
        "min_bound": -1,
        "max_bound": 1,
        "step": 0.025,
        "condition_type": "between"
    }
}

# Configuration interactive des features
print(f"\n{Fore.CYAN}🔧 Configuration Simplifiée des Features:{Style.RESET_ALL}")
for feature_key, config in FEATURES_CONFIG.items():
    current_status = "ACTIVÉE" if config["active"] else "DÉSACTIVÉE"
    print(f"  {feature_key}: {config['name']} - {current_status}")

choice_features = input(
    "\nConfiguration features :\n"
    "  [Entrée] → Garder la configuration actuelle\n"
    "  c        → Configurer manuellement chaque feature\n"
    "Choix : "
).strip().lower()

if choice_features == "c":
    print(f"\n{Fore.YELLOW}Configuration manuelle des features:{Style.RESET_ALL}")
    for feature_key, config in FEATURES_CONFIG.items():
        current_status = "o" if config["active"] else "n"
        choice = input(f"  Activer {feature_key} ({config['name']}) ? [o/n, défaut={current_status}]: ").strip().lower()
        if choice in ['o', 'y', 'yes', 'oui']:
            FEATURES_CONFIG[feature_key]["active"] = True
        elif choice in ['n', 'no', 'non']:
            FEATURES_CONFIG[feature_key]["active"] = False

# Affichage de la configuration finale
print(f"\n{Fore.GREEN}✅ Configuration finale des features:{Style.RESET_ALL}")
active_features = []
for feature_key, config in FEATURES_CONFIG.items():
    status = f"{Fore.GREEN}ACTIVÉE{Style.RESET_ALL}" if config["active"] else f"{Fore.RED}DÉSACTIVÉE{Style.RESET_ALL}"
    print(f"  {feature_key}: {config['name']} - {status}")
    if config["active"]:
        active_features.append(feature_key)

print(f"\n{Fore.CYAN}📊 {len(active_features)} feature(s) active(s) sur {len(FEATURES_CONFIG)}{Style.RESET_ALL}")

# Fichiers de données
DIR = R"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject" \
      R"\Sierra_chart\xTickReversal\simu\5_0_5TP_6SL\merge"

# Adaptation du nom de fichier selon la direction
Direction = DIRECTION.capitalize()

# ====== CONTRÔLE D'UTILISATION DU DATASET TRAIN ======
USE_TRAIN_IN_OPTIMIZATION = False  # Mettre True pour utiliser TRAIN dans l'optimisation

# Chemins des fichiers
CSV_TRAIN = DIR + Rf"\Step5_5_0_5TP_6SL_010124_160625_extractOnlyFullSession_Only{Direction}_feat__split2_01052024_01102024.csv"
CSV_VAL = DIR + Rf"\Step5_5_0_5TP_6SL_010124_160625_extractOnlyFullSession_Only{Direction}_feat__split3_01102024_28022025.csv"
CSV_VAL1 = DIR + Rf"\Step5_5_0_5TP_6SL_010124_160625_extractOnlyFullSession_Only{Direction}_feat__split4_02032025_15052025.csv"
CSV_TEST = DIR + Rf"\Step5_5_0_5TP_6SL_010124_160625_extractOnlyFullSession_Only{Direction}_feat__split5_15052025_16062025.csv"

# ───────────────────── CONFIGURATION SAUVEGARDE ─────────────────────────────

# Répertoire de sauvegarde des résultats valides (utilise DIR par défaut)
default_results_dir = Path(DIR)  # Même dossier que les CSV

# Choix du répertoire de sauvegarde
print(f"\n{Fore.CYAN}📁 Configuration du répertoire de sauvegarde:{Style.RESET_ALL}")
print(f"   Répertoire par défaut: {default_results_dir}")

choice_dir = input(
    "Répertoire de sauvegarde :\n"
    "  [Entrée] → Utiliser le répertoire des données (DIR)\n"
    "  p        → Spécifier un chemin personnalisé\n"
    "Choix : "
).strip().lower()

if choice_dir == "p":
    custom_path = input("Entrez le chemin du répertoire de sauvegarde: ").strip()
    try:
        RESULTS_DIR = Path(custom_path)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"{Fore.GREEN}✅ Répertoire personnalisé configuré: {RESULTS_DIR}{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.YELLOW}⚠️ Erreur avec le chemin personnalisé: {e}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}   Utilisation du répertoire par défaut{Style.RESET_ALL}")
        RESULTS_DIR = default_results_dir
else:
    RESULTS_DIR = default_results_dir

# S'assurer que le répertoire existe (DIR existe déjà normalement)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"{Fore.CYAN}📁 Répertoire de sauvegarde final: {RESULTS_DIR}{Style.RESET_ALL}")
print(f"{Fore.GREEN}💾 Les résultats VALIDES seront automatiquement sauvegardés dans: {RESULTS_DIR}{Style.RESET_ALL}")

# Configuration par mode
CONFIGS = {
    "light": {
        "WINRATE_MIN": 0.59,
        "PCT_TRADE_MIN": 0.004,
        "ALPHA": 0.70,
    },
    "aggressive": {
        "WINRATE_MIN": 0.61,
        "PCT_TRADE_MIN": 0.012,
        "ALPHA": 0.70,
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

# Construction du résumé des paramètres sélectionnés
param_summary = (
    f"🛠️ Paramètres sélectionnés :\n"
    f"▪️ Mode : {'aggressive' if choice == 'a' else 'light'}\n"
    f"▪️ FILTER_POC : {FILTER_POC}\n"
    f"▪️ Features actives : {len(active_features)}/{len(FEATURES_CONFIG)}\n"
    f"▪️ Config utilisée : {cfg}"
)

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
LAMBDA_WR = 0.2
LAMBDA_PCT = 0.8

# Bornes POC (si activé)
POS_POC_LOWER_BOUND_MIN, POS_POC_LOWER_BOUND_MAX = -1, 0
POS_POC_UPPER_BOUND_MIN, POS_POC_UPPER_BOUND_MAX = -1, 0
POS_POC_STEP = 0.25

import chardet

# ───────────────────── AJOUT EN DÉBUT DE SCRIPT ─────────────────────────────

# Récupération du nom du script et de la date de lancement
import os

SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]  # Nom sans extension
LAUNCH_DATETIME = datetime.now().strftime("%Y%m%d_%H%M%S")  # Date/heure de lancement


# ───────────────────── FONCTIONS DE SAUVEGARDE AMÉLIORÉES ─────────────────────────────

def save_valid_result(params, feature_filter, test_metrics, trial_number):
    """Sauvegarde automatiquement les résultats valides dans un fichier JSON."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    result = {
        "timestamp": timestamp,
        "script_name": SCRIPT_NAME,
        "launch_datetime": LAUNCH_DATETIME,
        "direction": DIRECTION,
        "trial_number": trial_number,
        "validation_status": "VALID",
        "test_metrics": {
            "winrate": test_metrics[0],
            "pct_trade": test_metrics[1],
            "success": test_metrics[2],
            "failure": test_metrics[3],
            "sessions": test_metrics[4]
        },
        "optimized_params": params,
        "feature_filter": feature_filter,
        "config_used": {
            "winrate_min": WINRATE_MIN,
            "pct_trade_min": PCT_TRADE_MIN,
            "alpha": ALPHA,
            "filter_poc": FILTER_POC,
            "use_train": USE_TRAIN_IN_OPTIMIZATION
        }
    }

    # Nom du fichier avec script et date de lancement
    filename = RESULTS_DIR / f"valid_results_{DIRECTION}_{SCRIPT_NAME}_{LAUNCH_DATETIME}_{timestamp}.json"

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"{Fore.GREEN}💾 Résultat VALIDE sauvegardé dans: {filename}{Style.RESET_ALL}")

        # NOUVEAU: Log enrichi avec toutes les informations
        log_entry = {
            "timestamp": timestamp,
            "script_name": SCRIPT_NAME,
            "launch_datetime": LAUNCH_DATETIME,
            "direction": DIRECTION,
            "trial": trial_number,
            "filename": filename.name,

            # ===== RÉSULTATS TEST =====
            "test": {
                "wr": f"{test_metrics[0]:.2%}",
                "pct": f"{test_metrics[1]:.2%}",
                "success": test_metrics[2],
                "failure": test_metrics[3],
                "total_trades": test_metrics[2] + test_metrics[3],
                "sessions": test_metrics[4]
            },

            # ===== RÉSULTATS DATASETS D'OPTIMISATION =====
            "optimization_results": {
                "train": {
                    "wr": f"{best_trial['wr_t']:.2%}",
                    "pct": f"{best_trial['pct_t']:.2%}",
                    "success": best_trial['suc_t'],
                    "failure": best_trial['fail_t'],
                    "total_trades": best_trial['suc_t'] + best_trial['fail_t'],
                    "sessions": best_trial['sess_t'],
                    "usage": "info_only" if not USE_TRAIN_IN_OPTIMIZATION else "optimization"
                },
                "val": {
                    "wr": f"{best_trial['wr_v']:.2%}",
                    "pct": f"{best_trial['pct_v']:.2%}",
                    "success": best_trial['suc_v'],
                    "failure": best_trial['fail_v'],
                    "total_trades": best_trial['suc_v'] + best_trial['fail_v'],
                    "sessions": best_trial['sess_v']
                },
                "val1": {
                    "wr": f"{best_trial['wr_v1']:.2%}",
                    "pct": f"{best_trial['pct_v1']:.2%}",
                    "success": best_trial['suc_v1'],
                    "failure": best_trial['fail_v1'],
                    "total_trades": best_trial['suc_v1'] + best_trial['fail_v1'],
                    "sessions": best_trial['sess_v1']
                }
            },

            # ===== GAPS ENTRE DATASETS =====
            "gaps": {
                "avg_gap_wr": f"{best_trial['avg_gap_wr']:.2%}",
                "avg_gap_pct": f"{best_trial['avg_gap_pct']:.2%}"
            },

            # ===== PARAMÈTRES OPTIMAUX =====
            "optimized_params": params,

            # ===== SCORE FINAL =====
            "optimization_score": best_trial['score'],

            # ===== CONFIGURATION =====
            "config": {
                "winrate_min": WINRATE_MIN,
                "pct_trade_min": PCT_TRADE_MIN,
                "alpha": ALPHA,
                "filter_poc": FILTER_POC,
                "use_train_in_optimization": USE_TRAIN_IN_OPTIMIZATION,
                "active_features": [key for key, config in FEATURES_CONFIG.items() if config["active"]]
            }
        }

        # Append au fichier de log global avec nom de script
        log_filename = RESULTS_DIR / f"valid_results_log_{DIRECTION}_{SCRIPT_NAME}_{LAUNCH_DATETIME}.json"
        try:
            with open(log_filename, 'r', encoding='utf-8') as f:
                log_data = json.load(f)
        except FileNotFoundError:
            log_data = {
                "script_name": SCRIPT_NAME,
                "launch_datetime": LAUNCH_DATETIME,
                "results": []
            }

        log_data["results"].append(log_entry)

        with open(log_filename, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)

        print(f"{Fore.CYAN}📋 Log enrichi mis à jour: {log_filename}{Style.RESET_ALL}")

    except Exception as e:
        print(f"{Fore.RED}❌ Erreur sauvegarde: {e}{Style.RESET_ALL}")


def show_previous_valid_results():
    """Affiche les résultats valides précédents s'ils existent avec plus de détails."""
    log_filename = RESULTS_DIR / f"valid_results_log_{DIRECTION}_{SCRIPT_NAME}_{LAUNCH_DATETIME}.json"

    try:
        with open(log_filename, 'r', encoding='utf-8') as f:
            log_data = json.load(f)

        results = log_data.get("results", [])
        if not results:
            print(f"{Fore.YELLOW}📁 Aucun résultat valide précédent trouvé pour ce script.{Style.RESET_ALL}")
            return

        script_info = log_data.get("script_name", "N/A")
        launch_info = log_data.get("launch_datetime", "N/A")

        print(f"\n{Fore.CYAN}📋 Résultats valides précédents pour {DIRECTION.upper()} - {script_info}:{Style.RESET_ALL}")
        print(
            f"{Fore.YELLOW}🚀 Lancement: {launch_info[:8]}/{launch_info[9:15]} (cette session: {LAUNCH_DATETIME[:8]}/{LAUNCH_DATETIME[9:15]}){Style.RESET_ALL}")

        # Affichage en tableau condensé
        print(
            f"{'#':<3} {'Date/Heure':<17} {'Trial':<6} {'TEST':<15} {'TRAIN':<15} {'VAL':<15} {'VAL1':<15} {'Score':<8}")
        print("-" * 100)

        for i, result in enumerate(results[-10:], 1):  # Afficher les 10 derniers
            timestamp = result['timestamp']
            date_str = f"{timestamp[:8]}/{timestamp[9:15]}"

            # Récupération des données (compatibilité avec ancien et nouveau format)
            if 'test' in result:
                # Nouveau format enrichi
                test_info = f"{result['test']['wr']}/{result['test']['pct']}"
                train_info = f"{result['optimization_results']['train']['wr']}/{result['optimization_results']['train']['pct']}"
                val_info = f"{result['optimization_results']['val']['wr']}/{result['optimization_results']['val']['pct']}"
                val1_info = f"{result['optimization_results']['val1']['wr']}/{result['optimization_results']['val1']['pct']}"
                score = f"{result.get('optimization_score', 'N/A'):.3f}" if isinstance(result.get('optimization_score'),
                                                                                       (int, float)) else "N/A"
            else:
                # Ancien format (compatibilité)
                test_info = f"{result.get('wr', 'N/A')}/{result.get('pct', 'N/A')}"
                train_info = val_info = val1_info = "N/A"
                score = "N/A"

            print(
                f"{i:<3} {date_str:<17} {result['trial']:<6} {test_info:<15} {train_info:<15} {val_info:<15} {val1_info:<15} {score:<8}")

        if len(results) > 10:
            print(f"... et {len(results) - 10} autres résultats")

        # Affichage détaillé du dernier résultat
        if results and 'test' in results[-1]:
            print(f"\n{Fore.YELLOW}📊 Détail du dernier résultat:{Style.RESET_ALL}")
            last = results[-1]
            train_usage = last['optimization_results']['train']['usage']
            train_suffix = f" ({Fore.YELLOW}info only{Style.RESET_ALL})" if train_usage == "info_only" else ""

            print(
                f"  TRAIN{train_suffix}: {last['optimization_results']['train']['wr']} WR, {last['optimization_results']['train']['pct']} PCT, {last['optimization_results']['train']['total_trades']} trades")
            print(
                f"  VAL:   {last['optimization_results']['val']['wr']} WR, {last['optimization_results']['val']['pct']} PCT, {last['optimization_results']['val']['total_trades']} trades")
            print(
                f"  VAL1:  {last['optimization_results']['val1']['wr']} WR, {last['optimization_results']['val1']['pct']} PCT, {last['optimization_results']['val1']['total_trades']} trades")
            print(f"  TEST:  {last['test']['wr']} WR, {last['test']['pct']} PCT, {last['test']['total_trades']} trades")
            print(f"  Features actives: {len(last['config']['active_features'])}")

    except FileNotFoundError:
        print(f"{Fore.YELLOW}📁 Aucun historique de résultats valides trouvé pour ce script.{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}❌ Erreur lecture historique: {e}{Style.RESET_ALL}")

# ───────────────────── FONCTIONS SIMPLIFIÉES ─────────────────────────────

def apply_feature_conditions(df, feature_filters):
    """Applique les conditions de filtrage sur les features."""
    if not feature_filters:
        return df

    mask = pd.Series(True, index=df.index)

    for feature_name, conditions in feature_filters.items():
        if feature_name not in df.columns:
            print(f"{Fore.YELLOW}⚠️ Feature '{feature_name}' non trouvée dans le dataset{Style.RESET_ALL}")
            continue

        for condition in conditions:
            if not condition.get('active', True):
                continue

            cond_type = condition['type']

            if cond_type == 'greater_than':
                mask &= (df[feature_name] > condition['threshold'])
            elif cond_type == 'greater_than_or_equal':
                mask &= (df[feature_name] >= condition['threshold'])
            elif cond_type == 'less_than':
                mask &= (df[feature_name] < condition['threshold'])
            elif cond_type == 'less_than_or_equal':
                mask &= (df[feature_name] <= condition['threshold'])
            elif cond_type == 'between':
                mask &= (df[feature_name] >= condition['min']) & (df[feature_name] <= condition['max'])
            elif cond_type == 'not_between':
                mask &= ~((df[feature_name] >= condition['min']) & (df[feature_name] <= condition['max']))
            else:
                raise ValueError(f"❌ Type de condition inconnu pour '{feature_name}': '{cond_type}'")

    return df[mask].copy()


def create_feature_filter_from_params(params):
    """Crée un dictionnaire de filtres à partir des paramètres optimisés."""
    feature_filter = {}

    for feature_key, config in FEATURES_CONFIG.items():
        if not config["active"]:
            continue

        feature_name = config["name"]
        condition_type = config["condition_type"]

        if feature_name not in feature_filter:
            feature_filter[feature_name] = []

        condition = {"active": True}

        if condition_type == "greater_than":
            condition.update({
                "type": "greater_than",
                "threshold": params[f"{feature_key}_threshold"]
            })
        elif condition_type == "greater_than_or_equal":
            condition.update({
                "type": "greater_than_or_equal",
                "threshold": params[f"{feature_key}_threshold"]
            })
        elif condition_type == "less_than":
            condition.update({
                "type": "less_than",
                "threshold": params[f"{feature_key}_threshold"]
            })
        elif condition_type == "less_than_or_equal":
            condition.update({
                "type": "less_than_or_equal",
                "threshold": params[f"{feature_key}_threshold"]
            })
        elif condition_type == "between":
            condition.update({
                "type": "between",
                "min": params[f"{feature_key}_min"],
                "max": params[f"{feature_key}_max"]
            })
        elif condition_type == "not_between":
            condition.update({
                "type": "not_between",
                "min": params[f"{feature_key}_min"],
                "max": params[f"{feature_key}_max"]
            })

        feature_filter[feature_name].append(condition)

    return feature_filter


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

    df = df[df["class_binaire"].isin([0, 1])].copy()
    df.reset_index(drop=True, inplace=True)

    return df, nb_sessions


# Chargement des datasets
TRAIN, TRAIN_SESSIONS = load_csv(CSV_TRAIN)
VAL, VAL_SESSIONS = load_csv(CSV_VAL)
VAL1, VAL1_SESSIONS = load_csv(CSV_VAL1)
TEST, TEST_SESSIONS = load_csv(CSV_TEST)

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


# ───────────────────── METRICS HELPERS SIMPLIFIÉS ──────────────────────────
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


# ───────────────────── OPTUNA OBJECTIVE SIMPLIFIÉ ─────────────────────────
best_trial = {
    "score": -math.inf,
    "number": None,
    "score_old": -math.inf,
    "wr_t": 0.0, "pct_t": 0.0, "suc_t": 0, "fail_t": 0, "sess_t": 0,
    "wr_v": 0.0, "pct_v": 0.0, "suc_v": 0, "fail_v": 0, "sess_v": 0,
    "wr_v1": 0.0, "pct_v1": 0.0, "suc_v1": 0, "fail_v1": 0, "sess_v1": 0,
    "avg_gap_wr": 0.0,
    "avg_gap_pct": 0.0,
    "params": {},
    "feature_filter": {}
}


def objective(trial: optuna.trial.Trial) -> float:
    # ====== OPTIMISATION SIMPLIFIÉE DES FEATURES ======
    p = {}

    for feature_key, config in FEATURES_CONFIG.items():
        if not config["active"]:
            continue

        condition_type = config["condition_type"]

        if condition_type in ["greater_than", "greater_than_or_equal", "less_than", "less_than_or_equal"]:
            # Paramètre unique: seuil
            if config.get("step") and isinstance(config["step"], (int, float)):
                p[f"{feature_key}_threshold"] = trial.suggest_float(
                    f"{feature_key}_threshold",
                    config["min_bound"],
                    config["max_bound"],
                    step=config["step"]
                )
            else:
                p[f"{feature_key}_threshold"] = trial.suggest_float(
                    f"{feature_key}_threshold",
                    config["min_bound"],
                    config["max_bound"]
                )
        elif condition_type in ["between", "not_between"]:
            # Deux paramètres: min et max
            if config.get("step") and isinstance(config["step"], (int, float)):
                min_val = trial.suggest_float(
                    f"{feature_key}_min",
                    config["min_bound"],
                    config["max_bound"],
                    step=config["step"]
                )
                max_val = trial.suggest_float(
                    f"{feature_key}_max",
                    config["min_bound"],
                    config["max_bound"],
                    step=config["step"]
                )
            else:
                min_val = trial.suggest_float(
                    f"{feature_key}_min",
                    config["min_bound"],
                    config["max_bound"]
                )
                max_val = trial.suggest_float(
                    f"{feature_key}_max",
                    config["min_bound"],
                    config["max_bound"]
                )

            # S'assurer que min <= max
            p[f"{feature_key}_min"] = min(min_val, max_val)
            p[f"{feature_key}_max"] = max(min_val, max_val)

    # Création du filtre de features à partir des paramètres
    feature_filter = create_feature_filter_from_params(p)

    val_len = len(VAL)
    val1_len = len(VAL1)

    if FILTER_POC:
        min_value = trial.suggest_float("pos_poc_min", POS_POC_LOWER_BOUND_MIN, POS_POC_LOWER_BOUND_MAX,
                                        step=POS_POC_STEP)
        max_value = trial.suggest_float("pos_poc_max", POS_POC_UPPER_BOUND_MIN, POS_POC_UPPER_BOUND_MAX,
                                        step=POS_POC_STEP)
        p["pos_poc_min"] = min(min_value, max_value)
        p["pos_poc_max"] = max(min_value, max_value)

    # ====== TRAITEMENT SIMPLIFIÉ DES DATASETS ======
    train_df = TRAIN.copy()
    train_len = len(TRAIN)
    val_df = VAL.copy()
    val1_df = VAL1.copy()

    # Application du filtre POC si activé
    if FILTER_POC:
        poc_min = p["pos_poc_min"]
        poc_max = p["pos_poc_max"]

        train_df = train_df[
            (train_df["diffPriceClosePoc_0_0"] >= poc_min) & (train_df["diffPriceClosePoc_0_0"] <= poc_max)]
        val_df = val_df[(val_df["diffPriceClosePoc_0_0"] >= poc_min) & (val_df["diffPriceClosePoc_0_0"] <= poc_max)]
        val1_df = val1_df[(val1_df["diffPriceClosePoc_0_0"] >= poc_min) & (val1_df["diffPriceClosePoc_0_0"] <= poc_max)]

        if trial.number % PRINT_EVERY == 0:
            train_pct = len(train_df) / len(TRAIN)
            print(f"{Fore.CYAN}POC filtré entre {poc_min} et {poc_max} : "
                  f"TR {train_pct:.1%} {'(utilisé)' if USE_TRAIN_IN_OPTIMIZATION else '(info)'}, "
                  f"V1 {len(val_df) / len(VAL):.1%}, "
                  f"V2 {len(val1_df) / len(VAL1):.1%}{Style.RESET_ALL}")

    # ====== APPLICATION DU FILTRE GLOBAL SIMPLIFIÉ ======
    train_filtered = apply_feature_conditions(train_df, feature_filter)
    val_filtered = apply_feature_conditions(val_df, feature_filter)
    val1_filtered = apply_feature_conditions(val1_df, feature_filter)

    # ====== CALCUL DES MÉTRIQUES SIMPLIFIÉES ======
    wr_t, pct_t, suc_t, fail_t, sess_t = _metrics(train_filtered, pd.Series(True, index=train_filtered.index),
                                                  train_len)
    wr_v, pct_v, suc_v, fail_v, sess_v = _metrics(val_filtered, pd.Series(True, index=val_filtered.index), val_len)
    wr_v1, pct_v1, suc_v1, fail_v1, sess_v1 = _metrics(val1_filtered, pd.Series(True, index=val1_filtered.index),
                                                       val1_len)

    # ====== VÉRIFICATION DES SEUILS ======
    datasets_to_check = [(wr_v, pct_v), (wr_v1, pct_v1)]
    if USE_TRAIN_IN_OPTIMIZATION:
        datasets_to_check.append((wr_t, pct_t))

    for wr, pct in datasets_to_check:
        if wr < WINRATE_MIN or pct < PCT_TRADE_MIN:
            return FAILED_PENALTY

    # ====== CALCUL DU SCORE SIMPLIFIÉ ======
    if USE_TRAIN_IN_OPTIMIZATION:
        # Avec TRAIN
        gap_wr_tv = abs(wr_t - wr_v)
        gap_pct_tv = abs(pct_t - pct_v)
        gap_wr_tv1 = abs(wr_t - wr_v1)
        gap_pct_tv1 = abs(pct_t - pct_v1)
        gap_wr_vv1 = abs(wr_v - wr_v1)
        gap_pct_vv1 = abs(pct_v - pct_v1)

        avg_gap_wr = (gap_wr_tv + gap_wr_tv1 + gap_wr_vv1) / 3
        avg_gap_pct = (gap_pct_tv + gap_pct_tv1 + gap_pct_vv1) / 3

        score = (ALPHA * (wr_t + wr_v + wr_v1) / 3 +
                 (1 - ALPHA) * (pct_t + pct_v + pct_v1) / 3 -
                 LAMBDA_WR * avg_gap_wr -
                 LAMBDA_PCT * avg_gap_pct)
    else:
        # Sans TRAIN
        gap_wr_vv1 = abs(wr_v - wr_v1)
        gap_pct_vv1 = abs(pct_v - pct_v1)

        avg_gap_wr = gap_wr_vv1
        avg_gap_pct = gap_pct_vv1

        score = (ALPHA * (wr_v + wr_v1) / 2 +
                 (1 - ALPHA) * (pct_v + pct_v1) / 2 -
                 LAMBDA_WR * avg_gap_wr -
                 LAMBDA_PCT * avg_gap_pct)

    # ====== MISE À JOUR DU MEILLEUR TRIAL ======
    global best_trial
    if score > best_trial["score"]:
        best_trial = {
            "number": trial.number,
            "score": score,
            "wr_t": wr_t, "pct_t": pct_t, "suc_t": suc_t, "fail_t": fail_t, "sess_t": sess_t,
            "wr_v": wr_v, "pct_v": pct_v, "suc_v": suc_v, "fail_v": fail_v, "sess_v": sess_v,
            "wr_v1": wr_v1, "pct_v1": pct_v1, "suc_v1": suc_v1, "fail_v1": fail_v1, "sess_v1": sess_v1,
            "avg_gap_wr": avg_gap_wr,
            "avg_gap_pct": avg_gap_pct,
            "params": p,
            "feature_filter": feature_filter
        }

    # ====== AFFICHAGE LIVE SIMPLIFIÉ ======
    if USE_TRAIN_IN_OPTIMIZATION:
        print(f"{trial.number:>6} | "
              f"TRAIN {Fore.GREEN}{wr_t:6.2%}{Style.RESET_ALL}/{pct_t:6.2%} | "
              f"VAL {Fore.GREEN}{wr_v:6.2%}{Style.RESET_ALL}/{pct_v:6.2%} | "
              f"VAL1 {Fore.GREEN}{wr_v1:6.2%}{Style.RESET_ALL}/{pct_v1:6.2%} | "
              f"Score {score:.4f}",
              f"{Fore.GREEN}✔{Style.RESET_ALL}" if score > best_trial.get("score_old", -math.inf) else "")
    else:
        print(f"{trial.number:>6} | "
              f"TRAIN {Fore.YELLOW}{wr_t:6.2%}{Style.RESET_ALL}/{pct_t:6.2%}(info) | "
              f"VAL {Fore.GREEN}{wr_v:6.2%}{Style.RESET_ALL}/{pct_v:6.2%} | "
              f"VAL1 {Fore.GREEN}{wr_v1:6.2%}{Style.RESET_ALL}/{pct_v1:6.2%} | "
              f"Score {score:.4f}",
              f"{Fore.GREEN}✔{Style.RESET_ALL}" if score > best_trial.get("score_old", -math.inf) else "")

    best_trial["score_old"] = score
    return score


# ───────────────────── HOLD‑OUT TEST SIMPLIFIÉ ────────────────────────────
def calculate_test_metrics(params: dict, feature_filter: dict = None):
    print(f"\n{Fore.CYAN}🧮  Calcul SIMPLIFIÉ sur DATASET TEST, scénario {DIRECTION} {Style.RESET_ALL}\n")

    # Fonction pour calculer et afficher les stats finales d'un dataset
    def calculate_dataset_stats(df, df_name, original_len):
        if df is None:
            return

        dataset_df = df.copy()

        # Application du filtre POC
        if FILTER_POC and "pos_poc_min" in params and "pos_poc_max" in params:
            poc_min = params["pos_poc_min"]
            poc_max = params["pos_poc_max"]
            poc_min, poc_max = min(poc_min, poc_max), max(poc_min, poc_max)
            dataset_df = dataset_df[
                (dataset_df["diffPriceClosePoc_0_0"] >= poc_min) & (dataset_df["diffPriceClosePoc_0_0"] <= poc_max)]

        # Application du filtre de features
        filtered_df = apply_feature_conditions(dataset_df, feature_filter)
        wr_u, pct_u, suc_u, fail_u, sess_u = _metrics(filtered_df, pd.Series(True, index=filtered_df.index),
                                                      original_len)

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

    # Application du filtre POC sur TEST
    if FILTER_POC and "pos_poc_min" in params and "pos_poc_max" in params:
        poc_min = params["pos_poc_min"]
        poc_max = params["pos_poc_max"]
        poc_min, poc_max = min(poc_min, poc_max), max(poc_min, poc_max)
        test_df = test_df[(test_df["diffPriceClosePoc_0_0"] >= poc_min) & (test_df["diffPriceClosePoc_0_0"] <= poc_max)]

        print(f"{Fore.CYAN}POC filtré entre {poc_min} et {poc_max} : "
              f"TEST {len(test_df) / len(TEST):.1%} ({len(test_df)}/{len(TEST)}){Style.RESET_ALL}")

    # Application du filtre de features
    test_filtered = apply_feature_conditions(test_df, feature_filter)
    wr_u, pct_u, suc_u, fail_u, sess_u = _metrics(test_filtered, pd.Series(True, index=test_filtered.index), test_len)

    print(f"\n{Fore.YELLOW}--- Résultat final ---{Style.RESET_ALL}")
    print(f"TEST: WR={Fore.GREEN}{wr_u:.2%}{Style.RESET_ALL}  pct={pct_u:.2%}  "
          f"✓{Fore.GREEN}{suc_u}{Style.RESET_ALL} ✗{Fore.RED}{fail_u}{Style.RESET_ALL}  "
          f"Total={Fore.CYAN}{suc_u + fail_u}{Style.RESET_ALL} (sessions: {sess_u})")

    is_valid = (wr_u >= WINRATE_MIN and pct_u >= PCT_TRADE_MIN)
    if is_valid:
        print(f"{Fore.GREEN}✅ VALIDE{Style.RESET_ALL}")

        # 💾 SAUVEGARDE AUTOMATIQUE DES RÉSULTATS VALIDES
        test_metrics = (wr_u, pct_u, suc_u, fail_u, sess_u)
        save_valid_result(params, feature_filter, test_metrics, best_trial.get("number", "unknown"))

        print()
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


# ───────────────────── MAIN LOOP SIMPLIFIÉ ────────────────────────────────
def main() -> None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    last_best_shown = None

    listener = start_keyboard_listener()
    print(
        f"{Fore.CYAN}Écouteur clavier démarré - appuyez sur '&' à tout moment pour tester sur le dataset TEST{Style.RESET_ALL}")

    # Affichage de la configuration
    print(f"\n{Fore.YELLOW}═══════════════════════════════════════════════════════════{Style.RESET_ALL}")
    print(f"{Fore.CYAN}📊 Configuration {DIRECTION.upper()} SIMPLIFIÉE :{Style.RESET_ALL}")
    print(f"   • Volume: {VOLUME_COL}")
    print(f"   • Imbalance: {IMBALANCE_COL}")

    # Affichage des features actives
    print(f"   • Features actives:")
    for feature_key, config in FEATURES_CONFIG.items():
        if config["active"]:
            print(f"     ↳ {feature_key}: {config['name']} ({config['condition_type']})")

    if not USE_TRAIN_IN_OPTIMIZATION:
        print(f"   • {Fore.YELLOW}DATASET TRAIN: TESTÉ POUR INFO SEULEMENT{Style.RESET_ALL}")
    else:
        print(f"   • {Fore.GREEN}DATASET TRAIN: UTILISÉ POUR L'OPTIMISATION{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}═══════════════════════════════════════════════════════════{Style.RESET_ALL}\n")

    # 📋 Affichage des résultats valides précédents
    show_previous_valid_results()

    for done in range(1, N_TRIALS + 1):
        study.optimize(objective, n_trials=1)

        if RUN_TEST or done % PRINT_EVERY == 0:
            globals()["RUN_TEST"] = False
            if done % PRINT_EVERY == 0 and not RUN_TEST:
                print(f"\n{Fore.YELLOW}🧪  Test automatique (trial {done}){Style.RESET_ALL}")

            if best_trial.get("params"):
                calculate_test_metrics(best_trial["params"], best_trial.get("feature_filter"))
            else:
                print(f"{Fore.RED}⚠️ Aucun meilleur trial trouvé encore{Style.RESET_ALL}")

        if best_trial.get("number") is not None:
            print(f"Best trial {best_trial['number']}  value {Fore.GREEN}{best_trial['score']:.4f}{Style.RESET_ALL}",
                  end="\r")

        if (done % PRINT_EVERY == 0 or best_trial.get("number") != last_best_shown):
            bt = best_trial
            print(
                f"\n\n{Fore.YELLOW}*** BEST SIMPLIFIÉ for {DIRECTION} ▸ trial {bt['number']}  score={Fore.GREEN}{bt['score']:.4f}{Style.RESET_ALL}")
            print(f"    {Fore.CYAN}[Direction: {DIRECTION.upper()} - Logique Simplifiée]{Style.RESET_ALL}")

            if USE_TRAIN_IN_OPTIMIZATION:
                print(f"    TRAIN WR {Fore.GREEN}{bt['wr_t']:.2%}{Style.RESET_ALL} | pct {bt['pct_t']:.2%} | "
                      f"✓{Fore.GREEN}{bt['suc_t']}{Style.RESET_ALL} ✗{Fore.RED}{bt['fail_t']}{Style.RESET_ALL} "
                      f"Total={Fore.CYAN}{bt['suc_t'] + bt['fail_t']}{Style.RESET_ALL} (sessions: {bt['sess_t']})")
            else:
                print(f"    TRAIN WR {Fore.GREEN}{bt['wr_t']:.2%}{Style.RESET_ALL} | pct {bt['pct_t']:.2%} | "
                      f"✓{Fore.GREEN}{bt['suc_t']}{Style.RESET_ALL} ✗{Fore.RED}{bt['fail_t']}{Style.RESET_ALL} "
                      f"Total={Fore.CYAN}{bt['suc_t'] + bt['fail_t']}{Style.RESET_ALL} (sessions: {bt['sess_t']}) "
                      f"{Fore.YELLOW}(info seulement){Style.RESET_ALL}")

            print(f"    VAL WR {Fore.GREEN}{bt['wr_v']:.2%}{Style.RESET_ALL} | pct {bt['pct_v']:.2%} | "
                  f"✓{Fore.GREEN}{bt['suc_v']}{Style.RESET_ALL} ✗{Fore.RED}{bt['fail_v']}{Style.RESET_ALL} "
                  f"Total={Fore.CYAN}{bt['suc_v'] + bt['fail_v']}{Style.RESET_ALL} (sessions: {bt['sess_v']})")

            print(f"    VAL1 WR {Fore.GREEN}{bt['wr_v1']:.2%}{Style.RESET_ALL} | pct {bt['pct_v1']:.2%} | "
                  f"✓{Fore.GREEN}{bt['suc_v1']}{Style.RESET_ALL} ✗{Fore.RED}{bt['fail_v1']}{Style.RESET_ALL} "
                  f"Total={Fore.CYAN}{bt['suc_v1'] + bt['fail_v1']}{Style.RESET_ALL} (sessions: {bt['sess_v1']})")

            print(f"    Gap WR: {Fore.YELLOW}{bt['avg_gap_wr']:.2%}{Style.RESET_ALL} | "
                  f"Gap PCT: {Fore.YELLOW}{bt['avg_gap_pct']:.2%}{Style.RESET_ALL}")

            # Affichage des paramètres optimaux
            print(f"\n    {Fore.CYAN}[Paramètres optimaux]{Style.RESET_ALL}")
            print(f"    params ➜ {Fore.MAGENTA}{bt['params']}{Style.RESET_ALL}")

            # Affichage du filtre simplifié
            if bt.get('feature_filter'):
                print(f"\n    {Fore.MAGENTA}[Filtre Simplifié]{Style.RESET_ALL}")
                for feature, conditions in bt['feature_filter'].items():
                    for condition in conditions:
                        if condition.get('active', True):
                            cond_type = condition['type']
                            if cond_type in ['greater_than', 'less_than']:
                                op = ">" if cond_type == 'greater_than' else "<"
                                print(f"    {feature} {op} {condition['threshold']}")
                            elif cond_type in ['between', 'not_between']:
                                op = "entre" if cond_type == 'between' else "PAS entre"
                                print(f"    {feature} {op} {condition['min']} et {condition['max']}")
            print()

            last_best_shown = best_trial["number"]

    print(f"\n{Fore.YELLOW}🔚  Fin des essais Optuna.{Style.RESET_ALL}")

    # SYNTHÈSE FINALE
    print(f"\n{Fore.YELLOW}{'=' * 80}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}🏆 SYNTHÈSE FINALE SIMPLIFIÉE{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{'=' * 80}{Style.RESET_ALL}")

    if best_trial.get("params"):
        bt = best_trial
        print(
            f"\n{Fore.YELLOW}*** BEST FINAL SIMPLIFIÉ for {DIRECTION} ▸ trial {bt['number']}  score={Fore.GREEN}{bt['score']:.4f}{Style.RESET_ALL}")

        print(f"\n    {Fore.CYAN}[RÉSULTATS FINAUX]{Style.RESET_ALL}")

        if USE_TRAIN_IN_OPTIMIZATION:
            print(f"    TR WR {Fore.GREEN}{bt['wr_t']:.2%}{Style.RESET_ALL} | pct {bt['pct_t']:.2%} | "
                  f"Total={Fore.CYAN}{bt['suc_t'] + bt['fail_t']}{Style.RESET_ALL}")
        else:
            print(f"    TR WR {Fore.GREEN}{bt['wr_t']:.2%}{Style.RESET_ALL} | pct {bt['pct_t']:.2%} | "
                  f"Total={Fore.CYAN}{bt['suc_t'] + bt['fail_t']}{Style.RESET_ALL} "
                  f"{Fore.YELLOW}(info seulement){Style.RESET_ALL}")

        print(f"    V1 WR {Fore.GREEN}{bt['wr_v']:.2%}{Style.RESET_ALL} | pct {bt['pct_v']:.2%} | "
              f"Total={Fore.CYAN}{bt['suc_v'] + bt['fail_v']}{Style.RESET_ALL}")

        print(f"    V2 WR {Fore.GREEN}{bt['wr_v1']:.2%}{Style.RESET_ALL} | pct {bt['pct_v1']:.2%} | "
              f"Total={Fore.CYAN}{bt['suc_v1'] + bt['fail_v1']}{Style.RESET_ALL}")

        # Test final
        calculate_test_metrics(best_trial["params"], best_trial.get("feature_filter"))
    else:
        print(f"{Fore.RED}❌ Aucun meilleur trial trouvé{Style.RESET_ALL}")

    print(f"\n{Fore.YELLOW}{'=' * 80}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}🏁 FIN DE L'OPTIMISATION SIMPLIFIÉE{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{'=' * 80}{Style.RESET_ALL}")


if __name__ == "__main__":
    main()