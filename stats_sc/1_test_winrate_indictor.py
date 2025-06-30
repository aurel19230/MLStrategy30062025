import pandas as pd
import numpy as np

# =========================================================
# 1) Paramètres généraux
# =========================================================
TEST_SC = True  # True ➜ on teste les versions sc_, False ➜ on les ignore

# Chemins des fichiers - MODIFIÉ: peut être None
pathShort = r"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra_chart\xTickReversal\simu\5_0_5TP_6SL\merge\Step5_5_0_5TP_6SL_010124_270625_extractOnlyFullSession_OnlyShort_feat__split4_02032025_15052025.csv"
pathLong = r"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra_chart\xTickReversal\simu\5_0_5TP_6SL\merge\Step5_5_0_5TP_6SL_010124_270625_extractOnlyFullSession_OnlyLong_feat__split4_02032025_15052025.csv"
pathLong=None
# Pour désactiver l'analyse d'un côté, mettre le path à None
# pathShort = None  # Désactiver l'analyse SHORT
# pathLong = None   # Désactiver l'analyse LONG

# → Renseigner ici (ou laisser None)
start_period = None  # "5/15/2025 10:00:00 PM"   # None#  ➜ pas de borne basse
end_period = None  # "5/16/2025 08:59:56 PM"   # None#  ➜ pas de borne haute

# =========================================================
# 2) Listes d'indicateurs MISE À JOUR
# =========================================================
indicators_short = [
    "is_imBullWithPoc_light_short", "is_imBullWithPoc_aggressive_short",
    "is_mfi_overbought_short", "is_rs_range_short",
    "is_wr_overbought_short", "is_vwap_reversal_pro_short",
    'is_vptRatio_low_volume_32', 'is_vptRatio_high_volume_32',
    'is_vptRatio_conservative_outbound_51',
    # 🆕 NOUVEAUX INDICATEURS MICROSTRUCTURE
    'is_antiSpring_short',  # 1 = trade validé après filtrage anti-spring
    'is_antiEpuisement_long',  # 1 = trade validé après filtrage anti-épuisement
]

indicators_long = [
    "is_imBearWithPoc_light_long", "is_imBearWithPoc_aggressive_long",
    "is_mfi_oversold_long", "is_rs_range_long",
    "is_wr_oversold_long", "is_vwap_reversal_pro_long",
    'is_vptRatio_low_volume_32', 'is_vptRatio_high_volume_32',
    "is_vptRatio_conservative_outbound_51",
    # 🆕 NOUVEAUX INDICATEURS MICROSTRUCTURE
    'is_antiEpuisement_long',  # 1 = trade validé après filtrage anti-épuisement
    'is_antiSpring_short',
]


# =========================================================
# 3) Utilitaires MODIFIÉS pour logique ANTI inversée
# =========================================================
def filter_by_period(df, start, end):
    """Filtre df sur la colonne 'date' si start/end ne sont pas None."""
    # Conversion robuste en datetime (si ce n'est déjà fait)
    if not np.issubdtype(df['date'].dtype, np.datetime64):
        df['date'] = pd.to_datetime(df['date'], errors='coerce', dayfirst=False)  # ici m/d/yyyy
    mask = pd.Series(True, index=df.index)
    if start is not None:
        mask &= df['date'] >= pd.to_datetime(start)
    if end is not None:
        mask &= df['date'] <= pd.to_datetime(end)
    return df.loc[mask]


def winrate(df, col):
    """
    Calcule le winrate pour les trades où col == 1.
    Logique normale : garde les trades où l'indicateur == 1.
    """
    subset = df[(df[col] == 1) & (df['class_binaire'].isin([0, 1]))]
    if subset.empty:
        return np.nan, 0
    return round((subset['class_binaire'] == 1).mean() * 100, 2), len(subset)


def winrate_anti(df, col):
    """
    🆕 Calcule le winrate pour les indicateurs ANTI quand ils sont à 1 (trades validés).
    LOGIQUE INVERSÉE : 1 = trade validé après filtrage, 0 = trade rejeté par le filtre.
    """
    subset = df[(df[col] == 1) & (df['class_binaire'].isin([0, 1]))]
    if subset.empty:
        return np.nan, 0
    return round((subset['class_binaire'] == 1).mean() * 100, 2), len(subset)


def compare_pairs(df, base_cols, side, test_sc=True):
    """Version mise à jour pour gérer les indicateurs ANTI avec logique inversée."""
    rows = []

    # Indicateurs ANTI (logique inversée : on teste quand == 1)
    anti_indicators = ['is_antiSpring_short', 'is_antiEpuisement_long']

    for base in base_cols:
        # Détecter si c'est un indicateur ANTI
        if base in anti_indicators:
            # 🔄 Logique inversée : on teste quand l'indicateur == 1 (trades validés)
            wr_b, n_b = winrate_anti(df, base)
            signal_name = base.replace('_short', '').replace('_long', '') + ' (=1)'
        else:
            # 📊 Logique normale : on teste quand l'indicateur == 1
            wr_b, n_b = winrate(df, base)
            signal_name = base.replace('_short', '').replace('_long', '')

        row = {'Signal': signal_name,
               f'WR {side} %': wr_b,
               f'N {side}': n_b}

        if test_sc:
            sc = 'sc_' + base
            if sc in df.columns:
                if base in anti_indicators:
                    wr_s, n_s = winrate_anti(df, sc)  # Logique inversée pour SC aussi
                else:
                    wr_s, n_s = winrate(df, sc)  # Logique normale
            else:
                wr_s, n_s = np.nan, 0
            row.update({f'WR SC_{side} %': wr_s, f'N SC_{side}': n_s,
                        f'Δ WR (SC-base)': (None if np.isnan(wr_b) or np.isnan(wr_s)
                                            else round(wr_s - wr_b, 2))})
        rows.append(row)
    return pd.DataFrame(rows)


def global_wr(df):
    """Win-rate tous trades (class_binaire ∈ {0,1})."""
    valid = df[df['class_binaire'].isin([0, 1])]
    if valid.empty:
        return np.nan, 0
    return round((valid['class_binaire'] == 1).mean() * 100, 2), len(valid)


def analyze_microstructure(df, indicator_name, side_name):
    """Analyse détaillée d'un indicateur microstructure avec logique inversée."""
    if indicator_name not in df.columns:
        print(f"  ❌ {indicator_name}: MANQUANTE dans {side_name}")
        return

    print(f"\n📊 {indicator_name} ({side_name}):")

    # Winrate quand indicateur = 1 (trades validés après filtrage)
    wr_1, n_1 = winrate_anti(df, indicator_name)
    print(f"  🟢 Quand = 1 (trades validés) : {wr_1:.2f}% ({n_1} trades)")

    # Winrate quand indicateur = 0 (trades rejetés par le filtre)
    subset_0 = df[(df[indicator_name] == 0) & (df['class_binaire'].isin([0, 1]))]
    if not subset_0.empty:
        wr_0 = round((subset_0['class_binaire'] == 1).mean() * 100, 2)
        n_0 = len(subset_0)
        print(f"  🔴 Quand = 0 (trades rejetés): {wr_0:.2f}% ({n_0} trades)")

        if not np.isnan(wr_1) and not np.isnan(wr_0):
            print(f"  📈 Efficacité du filtre: {wr_1 - wr_0:+.2f}% points")
            print(f"  📉 % trades rejetés: {n_0 / (n_0 + n_1) * 100:.1f}%")
    else:
        print(f"  🔴 Quand = 0 (trades rejetés): Aucun trade rejeté")


# =========================================================
# 4) NOUVEAU: Vérification des chemins et chargement conditionnel
# =========================================================
print("📥 Vérification des chemins et lecture des CSV…")

# Variables pour stocker les DataFrames (None si pas de fichier)
df_short = None
df_long = None

# Chargement conditionnel SHORT
if pathShort is not None:
    try:
        print(f"  📁 Chargement SHORT: {pathShort}")
        df_short = pd.read_csv(pathShort, sep=';', encoding='ISO-8859-1',
                               parse_dates=['date'], dayfirst=False)
        df_short = filter_by_period(df_short, start_period, end_period)
        print(f"  ✅ SHORT chargé: {len(df_short)} lignes")
    except Exception as e:
        print(f"  ❌ Erreur chargement SHORT: {e}")
        df_short = None
else:
    print("  ⏭️  SHORT ignoré (path = None)")

# Chargement conditionnel LONG
if pathLong is not None:
    try:
        print(f"  📁 Chargement LONG: {pathLong}")
        df_long = pd.read_csv(pathLong, sep=';', encoding='ISO-8859-1',
                              parse_dates=['date'], dayfirst=False)
        df_long = filter_by_period(df_long, start_period, end_period)
        print(f"  ✅ LONG chargé: {len(df_long)} lignes")
    except Exception as e:
        print(f"  ❌ Erreur chargement LONG: {e}")
        df_long = None
else:
    print("  ⏭️  LONG ignoré (path = None)")

# Vérification qu'au moins un fichier est chargé
if df_short is None and df_long is None:
    print("\n❌ ERREUR: Aucun fichier n'a pu être chargé. Arrêt du script.")
    exit(1)

print(f"\n⏱️  Période analysée : "
      f"{start_period or 'début des données'} → {end_period or 'fin des données'}")

# =========================================================
# 5) Vérification des colonnes microstructure (conditionnel)
# =========================================================
print("\n🔍 Vérification colonnes microstructure:")

if df_short is not None:
    micro_cols_short = ['is_antiSpring_short']
    for col in micro_cols_short:
        if col in df_short.columns:
            unique_vals = df_short[col].value_counts().sort_index()
            print(f"  SHORT {col}: {dict(unique_vals)}")
        else:
            print(f"  ❌ SHORT {col}: MANQUANTE")
else:
    print("  ⏭️  Vérification SHORT ignorée (données non chargées)")

if df_long is not None:
    micro_cols_long = ['is_antiEpuisement_long']
    for col in micro_cols_long:
        if col in df_long.columns:
            unique_vals = df_long[col].value_counts().sort_index()
            print(f"  LONG {col}: {dict(unique_vals)}")
        else:
            print(f"  ❌ LONG {col}: MANQUANTE")
else:
    print("  ⏭️  Vérification LONG ignorée (données non chargées)")

# =========================================================
# 6) Calculs conditionnels
# =========================================================
print("\n=== WIN-RATE GLOBAL ===")

# Calculs SHORT
if df_short is not None:
    wr_g_short, n_g_short = global_wr(df_short)
    summary_short = compare_pairs(df_short, indicators_short, 'SHORT', test_sc=TEST_SC)
    print(f"SHORT : {wr_g_short:.2f}%  ({n_g_short} trades)")
else:
    wr_g_short, n_g_short = np.nan, 0
    summary_short = pd.DataFrame()
    print("SHORT : Non calculé (données non disponibles)")

# Calculs LONG
if df_long is not None:
    wr_g_long, n_g_long = global_wr(df_long)
    summary_long = compare_pairs(df_long, indicators_long, 'LONG', test_sc=TEST_SC)
    print(f"LONG  : {wr_g_long:.2f}%  ({n_g_long} trades)")
else:
    wr_g_long, n_g_long = np.nan, 0
    summary_long = pd.DataFrame()
    print("LONG  : Non calculé (données non disponibles)")

# =========================================================
# 7) Affichage conditionnel avec focus microstructure
# =========================================================
if df_short is not None and not summary_short.empty:
    print("\n=== RÉSUMÉ SHORT ===")
    print(summary_short.sort_values('Signal').to_string(index=False))
else:
    print("\n=== RÉSUMÉ SHORT ===")
    print("Aucune donnée SHORT à afficher")

if df_long is not None and not summary_long.empty:
    print("\n=== RÉSUMÉ LONG ===")
    print(summary_long.sort_values('Signal').to_string(index=False))
else:
    print("\n=== RÉSUMÉ LONG ===")
    print("Aucune donnée LONG à afficher")

# =========================================================
# 8) 🆕 ANALYSE DÉTAILLÉE MICROSTRUCTURE - LOGIQUE INVERSÉE (conditionnel)
# =========================================================
print("\n" + "=" * 60)
print("🔬 ANALYSE DÉTAILLÉE MICROSTRUCTURE (LOGIQUE INVERSÉE)")
print("=" * 60)

# Test is_antiSpring_short (seulement si df_short existe)
if df_short is not None:
    analyze_microstructure(df_short, 'is_antiSpring_short', 'SHORT')
else:
    print("\n⏭️  Analyse is_antiSpring_short ignorée (données SHORT non disponibles)")

# Test is_antiEpuisement_long (seulement si df_long existe)
if df_long is not None:
    analyze_microstructure(df_long, 'is_antiEpuisement_long', 'LONG')
else:
    print("\n⏭️  Analyse is_antiEpuisement_long ignorée (données LONG non disponibles)")

print("\n" + "=" * 60)
print("✅ Analyse terminée - Les valeurs (=1) montrent les performances")
print("   des trades VALIDÉS par le filtrage ANTI (trades de qualité)")
print("   Les valeurs (=0) montrent les trades REJETÉS par le filtre")
print("=" * 60)

# Option d'export CSV (conditionnel)
if df_short is not None and not summary_short.empty:
    # summary_short.to_csv("wr_comparatif_short_with_microstructure_inverse.csv", sep=';', index=False)
    pass

if df_long is not None and not summary_long.empty:
    # summary_long.to_csv("wr_comparatif_long_with_microstructure_inverse.csv", sep=';', index=False)
    pass