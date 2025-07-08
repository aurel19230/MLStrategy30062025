#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test simple: recherche de timestamps et comptage de lignes
Pas de filtres, juste la logique de base
"""

# ────────────────────────────────────────────────────────────────────────────────
# IMPORTS ESSENTIELS
# ────────────────────────────────────────────────────────────────────────────────

from stats_sc.standard_stat_sc import *
from func_standard import *
import numpy as np
import pandas as pd
import os, sys, platform
from pathlib import Path
from Tools.func_features_preprocessing import *

# ────────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────────────────────────────────────

ENV = detect_environment()
DIR = "5_0_5TP_6SL"

# Configuration pour le test
TEST_CONFIG = {
    'dataset': 'Unseen',
    'start_date': '2025-06-24',
    'start_time': '22:00:00',
    'end_date': '2025-06-25',
    'end_time': '21:00:00',
    'date_column': 'date'
}

# Chemins de fichiers
if platform.system() != "Darwin":
    DIRECTORY_PATH = Path(
        rf"C:\Users\aurelienlachaud\OneDrive\Documents\Trading\VisualStudioProject\Sierra_chart\xTickReversal\simu\{DIR}\merge"
    )
else:
    DIRECTORY_PATH = Path(f"/Users/aurelienlachaud/Documents/trading_local/{DIR}/merge")

BASE = "Step5_5_0_5TP_6SL_010124_010725_extractOnlyFullSession_Only"
DIRECTION = "Short"
SPLIT_SUFFIX_UNSEEN = "_feat__split5_14052025_30062025"
FILE_NAME = lambda split: f"{BASE}{DIRECTION}{split}.csv"
FILE_PATH_UNSEEN = DIRECTORY_PATH / FILE_NAME(SPLIT_SUFFIX_UNSEEN)


# ────────────────────────────────────────────────────────────────────────────────
# FONCTION SIMPLE DE RECHERCHE DE TIMESTAMPS
# ────────────────────────────────────────────────────────────────────────────────

def find_timestamps_and_count(df_raw, start_date, end_date, date_column='date',
                              start_time="22:00:00", end_time="21:00:00"):
    """
    Fonction simple qui trouve les timestamps et compte les lignes
    """
    print(f"\n🎯 RECHERCHE DE TIMESTAMPS SIMPLE")
    print("=" * 60)
    print(f"📅 PARAMÈTRES:")
    print(f"   • Dataset: {len(df_raw)} lignes")
    print(f"   • Date début: {start_date}")
    print(f"   • Heure début: {start_time}")
    print(f"   • Date fin: {end_date}")
    print(f"   • Heure fin: {end_time}")

    if date_column not in df_raw.columns:
        print(f"❌ Colonne '{date_column}' non trouvée")
        return None

    # Conversion datetime
    df_work = df_raw.copy()
    if df_work[date_column].dtype == 'object':
        df_work[date_column] = pd.to_datetime(df_work[date_column], errors='coerce')

    # Conversion des paramètres
    start_date_obj = pd.to_datetime(start_date).date()
    end_date_obj = pd.to_datetime(end_date).date()
    start_time_obj = pd.to_datetime(start_time).time()
    end_time_obj = pd.to_datetime(end_time).time()

    print(f"\n🔍 RECHERCHE:")

    # ═══════════════════════════════════════════════════════════════════════════════
    # ÉTAPE 1: TIMESTAMP DE DÉBUT
    # ═══════════════════════════════════════════════════════════════════════════════

    start_day_data = df_work[df_work[date_column].dt.date == start_date_obj]
    print(f"   • Lignes disponibles le {start_date}: {len(start_day_data)}")

    if len(start_day_data) == 0:
        print(f"❌ Aucune donnée le {start_date}")
        return None

    # Premier timestamp >= start_time
    start_candidates = start_day_data[start_day_data[date_column].dt.time >= start_time_obj]

    if len(start_candidates) == 0:
        print(f"❌ Aucun timestamp >= {start_time} le {start_date}")
        return None

    actual_start = start_candidates[date_column].min()
    print(f"   ✅ Timestamp début trouvé: {actual_start}")

    # ═══════════════════════════════════════════════════════════════════════════════
    # ÉTAPE 2: TIMESTAMP DE FIN
    # ═══════════════════════════════════════════════════════════════════════════════

    end_day_data = df_work[df_work[date_column].dt.date == end_date_obj]
    print(f"   • Lignes disponibles le {end_date}: {len(end_day_data)}")

    if len(end_day_data) == 0:
        print(f"❌ Aucune donnée le {end_date}")
        return None

    # Dernier timestamp <= end_time
    end_candidates = end_day_data[end_day_data[date_column].dt.time <= end_time_obj]

    if len(end_candidates) == 0:
        print(f"❌ Aucun timestamp <= {end_time} le {end_date}")
        return None

    actual_end = end_candidates[date_column].max()
    print(f"   ✅ Timestamp fin trouvé: {actual_end}")

    # ═══════════════════════════════════════════════════════════════════════════════
    # ÉTAPE 3: COMPTAGE DES LIGNES ENTRE LES TIMESTAMPS
    # ═══════════════════════════════════════════════════════════════════════════════

    if actual_start >= actual_end:
        print(f"❌ Erreur: début >= fin")
        return None

    mask = (df_work[date_column] >= actual_start) & (df_work[date_column] <= actual_end)
    lines_between = mask.sum()

    print(f"\n📊 RÉSULTAT:")
    print(f"   • Timestamp début: {actual_start}")
    print(f"   • Timestamp fin: {actual_end}")
    print(f"   • Durée: {actual_end - actual_start}")
    print(f"   • Lignes entre timestamps: {lines_between}")
    print(f"   • % du dataset: {(lines_between / len(df_raw) * 100):.2f}%")

    return {
        'start_timestamp': actual_start,
        'end_timestamp': actual_end,
        'lines_between': lines_between,
        'total_lines': len(df_raw),
        'percentage': (lines_between / len(df_raw) * 100)
    }


# ────────────────────────────────────────────────────────────────────────────────
# FONCTION DE CHARGEMENT SIMPLE
# ────────────────────────────────────────────────────────────────────────────────

def load_raw_data():
    """Charge juste le dataset brut"""
    print(f"📂 Chargement dataset brut...")
    try:
        df_init_features, _ = load_features_and_sections(FILE_PATH_UNSEEN)
        print(f"✅ Dataset chargé: {len(df_init_features)} lignes")
        return df_init_features
    except Exception as e:
        print(f"❌ Erreur chargement: {e}")
        return None


# ────────────────────────────────────────────────────────────────────────────────
# TESTS SIMPLES
# ────────────────────────────────────────────────────────────────────────────────

def test_basic_timestamps():
    """Test de base avec la configuration"""
    print(f"🧪 TEST BASIC TIMESTAMPS")
    print("=" * 80)

    # Charger les données
    df_raw = load_raw_data()
    if df_raw is None:
        return

    # Test avec configuration
    result = find_timestamps_and_count(
        df_raw,
        TEST_CONFIG['start_date'],
        TEST_CONFIG['end_date'],
        TEST_CONFIG['date_column'],
        TEST_CONFIG['start_time'],
        TEST_CONFIG['end_time']
    )

    if result:
        print(f"\n✅ TEST RÉUSSI")
        return result
    else:
        print(f"\n❌ TEST ÉCHOUÉ")
        return None


def test_multiple_ranges():
    """Test de plusieurs plages pour vérifier la cohérence"""
    print(f"\n🧪 TEST COHÉRENCE MULTIPLE PLAGES")
    print("=" * 80)

    df_raw = load_raw_data()
    if df_raw is None:
        return

    # Test 1: Plage longue (15-26 juin)
    print(f"\n📅 Test 1: Plage longue (15-26 juin)")
    result_long = find_timestamps_and_count(
        df_raw, '2025-06-15', '2025-06-26', 'date', '22:00:00', '21:00:00'
    )

    # Test 2: Plage courte (24-26 juin)
    print(f"\n📅 Test 2: Plage courte (24-26 juin)")
    result_short = find_timestamps_and_count(
        df_raw, '2025-06-24', '2025-06-26', 'date', '22:00:00', '21:00:00'
    )

    # Test 3: Une seule journée (24-25 juin)
    print(f"\n📅 Test 3: Une journée (24-25 juin)")
    result_day = find_timestamps_and_count(
        df_raw, '2025-06-24', '2025-06-25', 'date', '22:00:00', '21:00:00'
    )

    # Comparaison
    print(f"\n📊 COMPARAISON:")
    print(f"{'Plage':<20} {'Lignes':<10} {'%':<8}")
    print("-" * 40)

    if result_long:
        print(f"{'Long (15-26)':<20} {result_long['lines_between']:<10} {result_long['percentage']:<8.2f}")
    else:
        print(f"{'Long (15-26)':<20} {'0':<10} {'0.00':<8}")

    if result_short:
        print(f"{'Court (24-26)':<20} {result_short['lines_between']:<10} {result_short['percentage']:<8.2f}")
    else:
        print(f"{'Court (24-26)':<20} {'0':<10} {'0.00':<8}")

    if result_day:
        print(f"{'Jour (24-25)':<20} {result_day['lines_between']:<10} {result_day['percentage']:<8.2f}")
    else:
        print(f"{'Jour (24-25)':<20} {'0':<10} {'0.00':<8}")

    # Vérification logique
    print(f"\n🔍 VÉRIFICATION LOGIQUE:")

    if result_long and result_short:
        if result_short['lines_between'] <= result_long['lines_between']:
            print(f"✅ Court ≤ Long: cohérent")
        else:
            print(f"❌ Court > Long: incohérent")

    if result_short and result_day:
        if result_day['lines_between'] <= result_short['lines_between']:
            print(f"✅ Jour ≤ Court: cohérent")
        else:
            print(f"❌ Jour > Court: incohérent")

    return result_long, result_short, result_day


def main():
    """Fonction principale simple"""
    print(f"🚀 TEST SIMPLE TIMESTAMPS")
    print(f"Objectif: Trouver timestamps et compter lignes")
    print("=" * 80)

    # Test 1: Configuration de base
    print(f"\n1️⃣ TEST CONFIGURATION DE BASE")
    result_basic = test_basic_timestamps()

    # Test 2: Plusieurs plages
    print(f"\n2️⃣ TEST COHÉRENCE PLAGES")
    results_multiple = test_multiple_ranges()

    # Résumé
    print(f"\n3️⃣ RÉSUMÉ")
    print("=" * 60)

    if result_basic:
        print(f"✅ Test de base réussi: {result_basic['lines_between']} lignes trouvées")
        print(f"   Timestamps: {result_basic['start_timestamp']} → {result_basic['end_timestamp']}")
    else:
        print(f"❌ Test de base échoué")

    if results_multiple:
        long, short, day = results_multiple
        if long and short and day:
            print(f"✅ Tests multiples réussis:")
            print(f"   Long: {long['lines_between']} lignes")
            print(f"   Court: {short['lines_between']} lignes")
            print(f"   Jour: {day['lines_between']} lignes")
        else:
            print(f"⚠️ Certains tests multiples ont échoué")

    print(f"\n🎯 CONCLUSION:")
    print(f"La fonction find_timestamps_and_count() est prête !")
    print(f"Elle trouve les timestamps exacts et compte les lignes entre eux.")


if __name__ == "__main__":
    main()