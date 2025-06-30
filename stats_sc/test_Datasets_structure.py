import pandas as pd
from pathlib import Path
import chardet
from colorama import Fore, Style, init
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuration globale
SLOPE_PERIODS = 5
MAX_WORKERS = 4  # Parallélisation
# Initialiser colorama pour Windows
init(autoreset=True)
from Tools.func_features_preprocessing import *

# Configuration
DIRECTION = "LONG"  # ou "SHORT" selon votre besoin

DIR = (r"C:\\Users\\aulac\\OneDrive\\Documents\\Trading\\VisualStudioProject"
       r"\\Sierra_chart\\xTickReversal\\simu\\5_0_5TP_6SL_samedi2706\\merge")
TEMPLATE = (DIR +
            rf"\\Step5_5_0_5TP_6SL_010124_200625_extractOnlyFullSession_Only{DIRECTION}_feat__split{{split}}.csv")

CSV_TRAIN = TEMPLATE.format(split="1_01012024_01052024")
CSV_TEST = TEMPLATE.format(split="2_01052024_30092024")
CSV_VAL = TEMPLATE.format(split="3_30092024_28022025")
CSV_VAL1 = TEMPLATE.format(split="4_02032025_15052025")
CSV_UNSEEN = TEMPLATE.format(split="5_15052025_20062025")


def detect_file_encoding(file_path: str) -> str:
    """
    Détecte l'encodage d'un fichier
    """
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read(100000)  # Lire les premiers 100KB
            result = chardet.detect(raw_data)
            return result['encoding'] or 'utf-8'
    except Exception as e:
        print(f"❌ Erreur détection encodage: {e}")
        return 'utf-8'


def load_csv_complete(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Charge les données complètes ET filtrées séparément
    Returns:
        df_complete: DataFrame avec TOUTES les bougies chronologiques (pour Rogers-Satchell)
        df_filtered: DataFrame avec seulement class_binaire ∈ {0, 1} (pour les métriques)
        nb_sessions: Nombre de sessions
    """
    path = Path(path)

    if not path.exists():
        print(f"❌ Fichier introuvable: {path}")
        return pd.DataFrame(), pd.DataFrame(), 0

    encoding = detect_file_encoding(str(path))
    if encoding.lower() == "ascii":
        encoding = "ISO-8859-1"

    print(f"{path.name} ➜ encodage détecté: {encoding}")

    try:
        # Chargement COMPLET sans filtrage
        df_complete = pd.read_csv(path, sep=";", encoding=encoding, parse_dates=["date"], low_memory=False)
    except Exception as e:
        print(f"❌ Erreur chargement {path.name}: {e}")
        return pd.DataFrame(), pd.DataFrame(), 0

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


def analyze_winrate_performance(df_filtered, name):
    """
    Analyse les performances brutes AVANT filtrage par algorithme
    """
    if df_filtered.empty:
        print(f"❌ {name}: DataFrame vide")
        return None

    # Distribution des classes
    class_counts = df_filtered['class_binaire'].value_counts().sort_index()
    total_trades = len(df_filtered)

    # Calculs de base
    wins = class_counts.get(1, 0)  # Trades réussis
    losses = class_counts.get(0, 0)  # Trades échoués
    winrate = (wins / total_trades * 100) if total_trades > 0 else 0

    # Analyse par session
    session_stats = df_filtered.groupby('session_id').agg({
        'class_binaire': ['count', 'sum', 'mean']
    }).round(3)
    session_stats.columns = ['trades_per_session', 'wins_per_session', 'winrate_per_session']

    # Métriques avancées par session
    session_winrates = session_stats['winrate_per_session'] * 100
    session_trade_counts = session_stats['trades_per_session']

    stats = {
        'name': name,
        'total_trades': total_trades,
        'wins': wins,
        'losses': losses,
        'winrate': winrate,
        'trades_per_session_mean': session_trade_counts.mean(),
        'trades_per_session_std': session_trade_counts.std(),
        'trades_per_session_median': session_trade_counts.median(),
        'winrate_per_session_mean': session_winrates.mean(),
        'winrate_per_session_std': session_winrates.std(),
        'winrate_per_session_median': session_winrates.median(),
        'best_session_winrate': session_winrates.max(),
        'worst_session_winrate': session_winrates.min(),
        'sessions_count': len(session_stats),
        'profitable_sessions': (session_winrates > 50).sum(),
        'profitable_sessions_pct': (session_winrates > 50).mean() * 100
    }

    return stats, session_stats


def analyze_temporal_patterns(df_filtered, name):
    """
    Analyse les patterns temporels qui pourraient expliquer les différences de performance
    """
    if df_filtered.empty or 'date' not in df_filtered.columns:
        return None

    # Extraire les composantes temporelles
    df_temp = df_filtered.copy()
    df_temp['hour'] = df_temp['date'].dt.hour
    df_temp['day_of_week'] = df_temp['date'].dt.dayofweek  # 0=Lundi, 6=Dimanche
    df_temp['month'] = df_temp['date'].dt.month

    # Analyse par heure
    hourly_stats = df_temp.groupby('hour').agg({
        'class_binaire': ['count', 'mean']
    }).round(3)
    hourly_stats.columns = ['trades_count', 'winrate']
    hourly_stats['winrate'] *= 100

    # Analyse par jour de la semaine
    daily_stats = df_temp.groupby('day_of_week').agg({
        'class_binaire': ['count', 'mean']
    }).round(3)
    daily_stats.columns = ['trades_count', 'winrate']
    daily_stats['winrate'] *= 100

    # Analyse par mois
    monthly_stats = df_temp.groupby('month').agg({
        'class_binaire': ['count', 'mean']
    }).round(3)
    monthly_stats.columns = ['trades_count', 'winrate']
    monthly_stats['winrate'] *= 100

    return {
        'hourly': hourly_stats,
        'daily': daily_stats,
        'monthly': monthly_stats
    }


def analyze_market_conditions(df_filtered, name):
    """
    Analyse les conditions de marché qui pourraient affecter les performances
    """
    if df_filtered.empty:
        return None

    # Colonnes d'intérêt pour l'analyse des conditions de marché
    market_cols = []
    for col in df_filtered.columns:
        if any(keyword in col.lower() for keyword in ['volume', 'volatility', 'spread', 'duration', 'atr']):
            market_cols.append(col)

    if not market_cols:
        return None

    # Diviser les trades en gagnants/perdants
    winners = df_filtered[df_filtered['class_binaire'] == 1]
    losers = df_filtered[df_filtered['class_binaire'] == 0]

    comparisons = {}
    for col in market_cols:
        if col in df_filtered.columns:
            try:
                win_mean = winners[col].mean()
                loss_mean = losers[col].mean()
                overall_mean = df_filtered[col].mean()

                comparisons[col] = {
                    'winners_mean': win_mean,
                    'losers_mean': loss_mean,
                    'overall_mean': overall_mean,
                    'difference': win_mean - loss_mean,
                    'difference_pct': ((win_mean - loss_mean) / overall_mean * 100) if overall_mean != 0 else 0
                }
            except:
                continue

    return comparisons


def analyze_market_regimes(datasets_dict, dataset_names):
    """
    Analyse approfondie des différences de régimes de marché entre datasets
    Focus sur les indicateurs qui caractérisent TEST vs TOUS les autres datasets
    """

    print("\n" + "=" * 80)
    print("🔍 ANALYSE DES RÉGIMES DE MARCHÉ - DIAGNOSTIC TEST vs TOUS")
    print("=" * 80)

    # ====== 1. ANALYSE TEMPORELLE DES BOUGIES ======
    print("\n📊 1. ANALYSE TEMPORELLE DES BOUGIES")
    print("-" * 50)

    candle_stats = {}

    for name, (df_complete, df_filtered) in datasets_dict.items():
        if df_complete.empty:
            continue

        # Statistiques des durées de bougies
        durations = df_complete['sc_candleDuration'].dropna()
        volumes = df_complete['sc_volume_perTick'].dropna()

        stats_dict = {
            'duration_mean': durations.mean(),
            'duration_median': durations.median(),
            'duration_std': durations.std(),
            'duration_cv': durations.std() / durations.mean() * 100,
            'duration_skew': stats.skew(durations),
            'duration_kurt': stats.kurtosis(durations),
            'volume_mean': volumes.mean(),
            'volume_median': volumes.median(),
            'volume_std': volumes.std(),
            'volume_cv': volumes.std() / volumes.mean() * 100,
            'duration_p95': durations.quantile(0.95),
            'duration_p05': durations.quantile(0.05),
            'fast_candles_pct': (durations < 10).mean() * 100,  # % bougies < 10s
            'slow_candles_pct': (durations > 300).mean() * 100,  # % bougies > 5min
        }

        candle_stats[name] = stats_dict

    # Affichage comparatif
    df_stats = pd.DataFrame(candle_stats).T

    print("DURÉE MOYENNE DES BOUGIES:")
    for name in dataset_names:
        if name in df_stats.index:
            print(
                f"{name:12}: {df_stats.loc[name, 'duration_mean']:6.1f}s (CV: {df_stats.loc[name, 'duration_cv']:5.1f}%)")

    print("\nVOLUME MOYEN PAR TICK:")
    for name in dataset_names:
        if name in df_stats.index:
            print(f"{name:12}: {df_stats.loc[name, 'volume_mean']:6.1f} (CV: {df_stats.loc[name, 'volume_cv']:5.1f}%)")

    print("\nDISTRIBUTION DES VITESSES:")
    for name in dataset_names:
        if name in df_stats.index:
            print(
                f"{name:12}: {df_stats.loc[name, 'fast_candles_pct']:4.1f}% rapides (<10s) | {df_stats.loc[name, 'slow_candles_pct']:4.1f}% lentes (>5min)")

    # ====== 2. ANALYSE DES PATTERNS VPT ======
    print("\n📈 2. ANALYSE DES INDICATEURS VPT")
    print("-" * 50)

    vpt_analysis = {}

    for name, (df_complete, df_filtered) in datasets_dict.items():
        if df_filtered.empty:
            continue

        # Chercher toutes les colonnes VPT
        vpt_cols = [col for col in df_filtered.columns if 'vpt' in col.lower()]

        if not vpt_cols:
            continue

        winners = df_filtered[df_filtered['class_binaire'] == 1]
        losers = df_filtered[df_filtered['class_binaire'] == 0]

        vpt_diffs = {}
        for col in vpt_cols:
            if col in df_filtered.columns:
                win_mean = winners[col].mean()
                loss_mean = losers[col].mean()
                diff = win_mean - loss_mean
                diff_pct = (diff / df_filtered[col].mean() * 100) if df_filtered[col].mean() != 0 else 0

                vpt_diffs[col] = {
                    'difference': diff,
                    'difference_pct': diff_pct,
                    'overall_mean': df_filtered[col].mean()
                }

        vpt_analysis[name] = vpt_diffs

    # Identifier les VPT avec des comportements opposés - TEST vs TOUS
    divergent_signals = []

    if 'TEST' in vpt_analysis:
        test_vpt = vpt_analysis['TEST']

        # Calculer moyennes des autres datasets
        other_datasets = {k: v for k, v in vpt_analysis.items() if k != 'TEST'}

        for vpt_name in test_vpt.keys():
            # Calculer moyenne des effets pour tous les autres datasets
            other_effects = []
            for other_name, other_vpt in other_datasets.items():
                if vpt_name in other_vpt:
                    other_effects.append(other_vpt[vpt_name]['difference_pct'])

            if other_effects:
                test_effect = test_vpt[vpt_name]['difference_pct']
                others_avg = np.mean(other_effects)
                divergence = abs(test_effect - others_avg)

                # Signaux divergents si |écart| > 15%
                if divergence > 15:
                    divergent_signals.append({
                        'vpt': vpt_name,
                        'test_effect': test_effect,
                        'others_avg': others_avg,
                        'divergence_strength': divergence,
                        'is_opposite': (test_effect * others_avg < 0)
                    })

        # Trier par force de divergence
        divergent_signals.sort(key=lambda x: x['divergence_strength'], reverse=True)

        print(f"🚨 {len(divergent_signals)} VPT avec COMPORTEMENTS DIVERGENTS détectés:")
        for signal in divergent_signals[:8]:  # Top 8
            opposite_flag = " [OPPOSÉ]" if signal['is_opposite'] else ""
            print(f"   {signal['vpt'][:35]:35}{opposite_flag}")
            print(f"      TEST:   {signal['test_effect']:+6.1f}% | AUTRES: {signal['others_avg']:+6.1f}%")
            print(f"      Écart:  {signal['divergence_strength']:5.1f}%")

    # ====== 3. ANALYSE DES CONDITIONS TEMPORELLES ======
    print("\n⏰ 3. ANALYSE DES CONDITIONS TEMPORELLES")
    print("-" * 50)

    temporal_analysis = {}

    for name, (df_complete, df_filtered) in datasets_dict.items():
        if df_filtered.empty or 'date' not in df_filtered.columns:
            continue

        df_temp = df_filtered.copy()
        df_temp['hour'] = df_temp['date'].dt.hour
        df_temp['day_of_week'] = df_temp['date'].dt.dayofweek

        # Analyser la distribution temporelle des trades
        hourly_dist = df_temp['hour'].value_counts().sort_index()
        daily_dist = df_temp['day_of_week'].value_counts().sort_index()

        # Analyser les performances par heure/jour
        hourly_perf = df_temp.groupby('hour')['class_binaire'].agg(['count', 'mean'])
        daily_perf = df_temp.groupby('day_of_week')['class_binaire'].agg(['count', 'mean'])

        temporal_analysis[name] = {
            'hourly_dist': hourly_dist,
            'daily_dist': daily_dist,
            'hourly_perf': hourly_perf,
            'daily_perf': daily_perf,
            'peak_trading_hour': hourly_dist.idxmax(),
            'peak_trading_day': daily_dist.idxmax(),
            'best_hour': hourly_perf['mean'].idxmax(),
            'worst_hour': hourly_perf['mean'].idxmin(),
            'trading_concentration': (hourly_dist.max() / hourly_dist.sum()) * 100
        }

    print("CONCENTRATION DU TRADING:")
    for name in dataset_names:
        if name in temporal_analysis:
            analysis = temporal_analysis[name]
            print(
                f"{name:12}: Pic à {analysis['peak_trading_hour']:2d}h ({analysis['trading_concentration']:4.1f}% du volume)")

    # Analyse spécifique TEST vs AUTRES
    if 'TEST' in temporal_analysis:
        test_temp = temporal_analysis['TEST']

        print(f"\n🎯 SPÉCIFICITÉS TEMPORELLES DE TEST:")
        print(f"   Heure de pic:     {test_temp['peak_trading_hour']}h")
        print(f"   Meilleure heure:  {test_temp['best_hour']}h")
        print(f"   Pire heure:       {test_temp['worst_hour']}h")
        print(f"   Concentration:    {test_temp['trading_concentration']:.1f}%")

        # Comparer avec les autres
        other_concentrations = []
        for name, analysis in temporal_analysis.items():
            if name != 'TEST':
                other_concentrations.append(analysis['trading_concentration'])

        if other_concentrations:
            avg_concentration = np.mean(other_concentrations)
            concentration_diff = test_temp['trading_concentration'] - avg_concentration

            print(f"\n   Comparaison concentration:")
            print(f"   TEST: {test_temp['trading_concentration']:.1f}% vs AUTRES: {avg_concentration:.1f}%")
            if abs(concentration_diff) > 5:
                status = "plus concentré" if concentration_diff > 0 else "plus dispersé"
                print(f"   → TEST est {status} temporellement ({concentration_diff:+.1f}%)")

    # ====== 4. RÉGIMES DE VOLATILITÉ ======
    print("\n🌪️  4. ANALYSE DES RÉGIMES DE VOLATILITÉ")
    print("-" * 50)

    volatility_analysis = {}

    for name, (df_complete, df_filtered) in datasets_dict.items():
        if df_complete.empty:
            continue

        # Chercher colonnes de volatilité
        vol_cols = []
        for col in df_complete.columns:
            if any(keyword in col.lower() for keyword in ['atr', 'volatility', 'range', 'spread']):
                vol_cols.append(col)

        if vol_cols:
            # Calculer moyennes de volatilité
            vol_means = {}
            for col in vol_cols:
                vol_means[col] = df_complete[col].mean()

            volatility_analysis[name] = vol_means

    if volatility_analysis:
        print("NIVEAUX DE VOLATILITÉ MOYENS:")
        vol_cols_found = set()
        for analysis in volatility_analysis.values():
            vol_cols_found.update(analysis.keys())

        for col in list(vol_cols_found)[:5]:  # Top 5 indicateurs
            print(f"\n{col[:30]:30}:")
            for name in dataset_names:
                if name in volatility_analysis and col in volatility_analysis[name]:
                    print(f"   {name:12}: {volatility_analysis[name][col]:8.3f}")

        # Analyse spécifique TEST vs AUTRES
        if 'TEST' in volatility_analysis:
            test_vol = volatility_analysis['TEST']

            print(f"\n🎯 COMPARAISON VOLATILITÉ TEST vs AUTRES:")

            for col in list(vol_cols_found)[:3]:  # Top 3 indicateurs
                if col in test_vol:
                    test_value = test_vol[col]
                    other_values = []

                    for name, vol_data in volatility_analysis.items():
                        if name != 'TEST' and col in vol_data:
                            other_values.append(vol_data[col])

                    if other_values:
                        others_avg = np.mean(other_values)
                        diff_pct = ((test_value - others_avg) / others_avg * 100) if others_avg != 0 else 0

                        status = "🔴" if abs(diff_pct) > 20 else "🟡" if abs(diff_pct) > 10 else "🟢"
                        direction = "plus élevé" if diff_pct > 0 else "plus bas"

                        print(f"   {status} {col[:25]:25}: TEST {direction} de {abs(diff_pct):4.1f}%")

    # ====== 5. RANKING ANALYSE : TEST vs ALL ======
    print("\n🏆 5. POSITIONNEMENT DE TEST vs TOUS LES DATASETS")
    print("-" * 60)

    quality_percentile = None

    if 'TEST' in df_stats.index and len(df_stats) > 1:
        test_stats = df_stats.loc['TEST']

        print("CLASSEMENTS DE TEST (1=meilleur selon contexte, 5=pire):")

        # Durée moyenne (rang selon vitesse - plus court peut être mieux ou pire selon stratégie)
        duration_rank = (df_stats['duration_mean'] < test_stats['duration_mean']).sum() + 1
        duration_percentile = (len(df_stats) - duration_rank + 1) / len(df_stats) * 100
        print(f"⏱️  Vitesse bougies     : {duration_rank}/{len(df_stats)} (Percentile: {duration_percentile:.0f}%)")

        # Volume par tick (plus élevé généralement mieux)
        volume_rank = (df_stats['volume_mean'] > test_stats['volume_mean']).sum() + 1
        volume_percentile = (len(df_stats) - volume_rank + 1) / len(df_stats) * 100
        print(f"📊 Volume/tick         : {volume_rank}/{len(df_stats)} (Percentile: {volume_percentile:.0f}%)")

        # Stabilité durée (CV plus bas = plus stable = mieux)
        cv_duration_rank = (df_stats['duration_cv'] < test_stats['duration_cv']).sum() + 1
        cv_duration_percentile = (len(df_stats) - cv_duration_rank + 1) / len(df_stats) * 100
        print(f"🎯 Stabilité durée     : {cv_duration_rank}/{len(df_stats)} (Percentile: {cv_duration_percentile:.0f}%)")

        # Stabilité volume (CV plus bas = plus stable = mieux)
        cv_volume_rank = (df_stats['volume_cv'] < test_stats['volume_cv']).sum() + 1
        cv_volume_percentile = (len(df_stats) - cv_volume_rank + 1) / len(df_stats) * 100
        print(f"📈 Stabilité volume    : {cv_volume_rank}/{len(df_stats)} (Percentile: {cv_volume_percentile:.0f}%)")

        # Bougies rapides (selon stratégie, peut être bon ou mauvais)
        fast_rank = (df_stats['fast_candles_pct'] < test_stats['fast_candles_pct']).sum() + 1
        fast_percentile = (len(df_stats) - fast_rank + 1) / len(df_stats) * 100
        print(f"⚡ Bougies rapides     : {fast_rank}/{len(df_stats)} (Percentile: {fast_percentile:.0f}%)")

        # Score global de "qualité" des conditions de marché
        # Plus le rang est bas, meilleures sont les conditions (stabilité élevée = bon)
        quality_score = cv_duration_rank + cv_volume_rank
        max_quality_score = 2 * len(df_stats)
        quality_percentile = (max_quality_score - quality_score + 2) / max_quality_score * 100

        print(f"\n🎯 SCORE QUALITÉ CONDITIONS: {quality_percentile:.0f}% (stabilité générale)")

        if quality_percentile >= 80:
            print("   ✅ TEST = Conditions de marché EXCELLENTES")
        elif quality_percentile >= 60:
            print("   🟢 TEST = Conditions de marché BONNES")
        elif quality_percentile >= 40:
            print("   🟡 TEST = Conditions de marché MOYENNES")
        elif quality_percentile >= 20:
            print("   🟠 TEST = Conditions de marché DIFFICILES")
        else:
            print("   🔴 TEST = Conditions de marché TRÈS DIFFICILES")

        # Analyse détaillée des écarts
        print(f"\n📊 ÉCARTS SIGNIFICATIFS DE TEST:")

        # Durée
        duration_diff_pct = (test_stats['duration_mean'] - df_stats['duration_mean'].mean()) / df_stats[
            'duration_mean'].mean() * 100
        if abs(duration_diff_pct) > 15:
            direction = "plus rapides" if duration_diff_pct < 0 else "plus lentes"
            print(f"   ⏱️  Bougies {direction} de {abs(duration_diff_pct):.1f}% vs moyenne")

        # Volume
        volume_diff_pct = (test_stats['volume_mean'] - df_stats['volume_mean'].mean()) / df_stats[
            'volume_mean'].mean() * 100
        if abs(volume_diff_pct) > 15:
            direction = "plus élevé" if volume_diff_pct > 0 else "plus bas"
            print(f"   📊 Volume {direction} de {abs(volume_diff_pct):.1f}% vs moyenne")

        # Stabilité
        cv_diff = test_stats['duration_cv'] - df_stats['duration_cv'].mean()
        if abs(cv_diff) > 100:  # CV en %
            direction = "moins stable" if cv_diff > 0 else "plus stable"
            print(f"   🎯 Durées {direction} de {abs(cv_diff):.0f} points de CV vs moyenne")

    # ====== 6. RECOMMANDATIONS FINALES ======
    print("\n💡 RECOMMANDATIONS BASÉES SUR L'ANALYSE TEST vs TOUS")
    print("=" * 70)

    recommendations = []
    priority_actions = []

    if 'TEST' in df_stats.index:
        # Basé sur la qualité des conditions
        if quality_percentile and quality_percentile < 40:
            recommendations.append("🎯 PRIORITÉ 1: Conditions de marché difficiles détectées pour TEST")
            priority_actions.append("   → Adapter les paramètres à la volatilité élevée")
            priority_actions.append("   → Considérer des stops plus larges ou des TP plus courts")

        # Basé sur les VPT divergents
        if len(divergent_signals) > 5:
            recommendations.append("🎯 PRIORITÉ 2: Instabilité majeure des signaux VPT")
            priority_actions.append("   → Re-calibrer TOUS les seuils VPT pour TEST")
            priority_actions.append("   → Implémenter une détection de régime de marché")
        elif len(divergent_signals) > 2:
            recommendations.append("🎯 PRIORITÉ 2: Quelques signaux VPT instables")
            priority_actions.append("   → Re-vérifier les VPT les plus divergents")

        # Basé sur la temporalité
        if 'TEST' in temporal_analysis:
            test_conc = temporal_analysis['TEST']['trading_concentration']
            if test_conc > 15:
                recommendations.append("🎯 PRIORITÉ 4: Trading très concentré temporellement")
                priority_actions.append("   → Optimiser spécifiquement pour les heures de pic")

        # Recommandations générales
        if not recommendations:
            recommendations.append("✅ Les différences TEST vs AUTRES sont dans la normale")
            priority_actions.append("   → Optimisation fine des hyperparamètres suffisante")
            priority_actions.append("   → Surveiller la stabilité sur données futures")

        print("ACTIONS RECOMMANDÉES:")
        for rec in recommendations:
            print(rec)

        for action in priority_actions:
            print(action)

    print(f"\n🔬 INDICATEURS CLÉS À SURVEILLER:")
    print(f"   1. Stabilité temporelle des VPT (actuellement {len(divergent_signals)} divergents)")
    print(f"   2. Niveau de volatilité relative vs autres périodes")
    print(f"   3. Concentration temporelle des trades")
    print(f"   4. Corrélations features-performance")

    return {
        'candle_stats': df_stats,
        'vpt_analysis': vpt_analysis,
        'temporal_analysis': temporal_analysis,
        'volatility_analysis': volatility_analysis,
        'divergent_signals': divergent_signals,
        'quality_score': quality_percentile,
        'recommendations': recommendations
    }


@njit
def fast_percentile(arr, q):
    """Calcul rapide de percentile avec numba"""
    sorted_arr = np.sort(arr)
    n = len(sorted_arr)
    index = (n - 1) * q / 100.0
    lower = int(index)
    upper = lower + 1
    weight = index - lower

    if upper >= n:
        return sorted_arr[-1]
    return sorted_arr[lower] * (1 - weight) + sorted_arr[upper] * weight


@njit
def fast_stats_calculation(values):
    """Calcul ultra-rapide des statistiques de base avec numba"""
    if len(values) == 0:
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    mean_val = np.mean(values)
    median_val = np.median(values)
    std_val = np.std(values)
    min_val = np.min(values)
    max_val = np.max(values)
    q25_val = fast_percentile(values, 25)
    q75_val = fast_percentile(values, 75)

    return np.array([mean_val, median_val, std_val, min_val, max_val, q25_val, q75_val])


def print_winrate_analysis(stats_dict):
    """
    Affiche l'analyse détaillée des winrates
    """
    print("\n" + "=" * 80)
    print("🎯 ANALYSE DES PERFORMANCES BRUTES (AVANT FILTRAGE ALGORITHME)")
    print("=" * 80)

    for name, (stats, session_stats) in stats_dict.items():
        if stats is None:
            continue

        print(f"\n=== {name} ===")
        print(f"📊 Total trades: {stats['total_trades']:,}")
        print(f"✅ Wins: {stats['wins']:,} | ❌ Losses: {stats['losses']:,}")
        print(f"🎯 Winrate global: {stats['winrate']:.2f}%")

        print(f"\n📈 ANALYSE PAR SESSION:")
        print(f"🏟️  Sessions analysées: {stats['sessions_count']}")
        print(
            f"📊 Trades/session - Moyenne: {stats['trades_per_session_mean']:.1f} | Médiane: {stats['trades_per_session_median']:.1f} | Écart-type: {stats['trades_per_session_std']:.1f}")
        print(
            f"🎯 Winrate/session - Moyenne: {stats['winrate_per_session_mean']:.2f}% | Médiane: {stats['winrate_per_session_median']:.2f}% | Écart-type: {stats['winrate_per_session_std']:.2f}%")
        print(
            f"🏆 Meilleure session: {stats['best_session_winrate']:.2f}% | 💀 Pire session: {stats['worst_session_winrate']:.2f}%")
        print(
            f"💰 Sessions profitables (>50%): {stats['profitable_sessions']}/{stats['sessions_count']} ({stats['profitable_sessions_pct']:.1f}%)")


def print_comparative_winrate_analysis(stats_dict):
    """
    Analyse comparative des winrates entre datasets
    """
    print("\n" + "=" * 80)
    print("🔍 ANALYSE COMPARATIVE DES WINRATES")
    print("=" * 80)

    # Extraire les stats valides
    valid_stats = {name: stats for name, (stats, _) in stats_dict.items() if stats is not None}

    if len(valid_stats) < 2:
        print("❌ Pas assez de datasets pour une analyse comparative")
        return

    # Classement par winrate
    sorted_by_winrate = sorted(valid_stats.items(), key=lambda x: x[1]['winrate'], reverse=True)

    print("🏆 CLASSEMENT PAR WINRATE GLOBAL:")
    for i, (name, stats) in enumerate(sorted_by_winrate, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
        print(f"{medal} {name:12}: {stats['winrate']:6.2f}% ({stats['total_trades']:,} trades)")

    # Analyse des écarts
    best_name, best_stats = sorted_by_winrate[0]
    worst_name, worst_stats = sorted_by_winrate[-1]

    print(f"\n📊 ANALYSE DES ÉCARTS:")
    print(f"🏆 Meilleur: {best_name} ({best_stats['winrate']:.2f}%)")
    print(f"💀 Pire: {worst_name} ({worst_stats['winrate']:.2f}%)")
    print(f"📏 Écart: {best_stats['winrate'] - worst_stats['winrate']:.2f} points de %")

    # Analyse de la consistance (écart-type des winrates par session)
    print(f"\n🎯 ANALYSE DE LA CONSISTANCE (Écart-type winrate/session):")
    sorted_by_consistency = sorted(valid_stats.items(), key=lambda x: x[1]['winrate_per_session_std'])

    for name, stats in sorted_by_consistency:
        consistency_score = "🟢" if stats['winrate_per_session_std'] < 20 else "🟡" if stats[
                                                                                         'winrate_per_session_std'] < 30 else "🔴"
        print(
            f"{consistency_score} {name:12}: {stats['winrate_per_session_std']:5.2f}% (plus c'est bas, plus c'est consistant)")

    # Focus sur TEST si il sous-performe
    if 'TEST' in valid_stats:
        test_stats = valid_stats['TEST']
        test_rank = next(i for i, (name, _) in enumerate(sorted_by_winrate, 1) if name == 'TEST')

        if test_rank > 2:  # Si TEST n'est pas dans le top 2
            print(f"\n🔍 ANALYSE SPÉCIFIQUE - POURQUOI TEST SOUS-PERFORME:")
            print(f"📉 Position: {test_rank}/{len(valid_stats)} avec {test_stats['winrate']:.2f}% de winrate")

            # Comparaison avec le meilleur
            if best_name != 'TEST':
                print(f"📊 Vs {best_name}:")
                print(f"   • Écart winrate: -{best_stats['winrate'] - test_stats['winrate']:.2f} points")
                print(
                    f"   • Trades/session: TEST={test_stats['trades_per_session_mean']:.1f} vs {best_name}={best_stats['trades_per_session_mean']:.1f}")
                print(
                    f"   • Consistance: TEST={test_stats['winrate_per_session_std']:.2f}% vs {best_name}={best_stats['winrate_per_session_std']:.2f}%")
                print(
                    f"   • Sessions profitables: TEST={test_stats['profitable_sessions_pct']:.1f}% vs {best_name}={best_stats['profitable_sessions_pct']:.1f}%")


def print_temporal_analysis(temporal_dict):
    """
    Affiche l'analyse temporelle
    """
    print("\n" + "=" * 80)
    print("⏰ ANALYSE DES PATTERNS TEMPORELS")
    print("=" * 80)

    for name, patterns in temporal_dict.items():
        if patterns is None:
            continue

        print(f"\n=== {name} ===")

        # Analyse horaire
        if 'hourly' in patterns and not patterns['hourly'].empty:
            hourly = patterns['hourly']
            best_hour = hourly['winrate'].idxmax()
            worst_hour = hourly['winrate'].idxmin()

            print(f"🕐 PERFORMANCE PAR HEURE:")
            print(
                f"🏆 Meilleure heure: {best_hour}h ({hourly.loc[best_hour, 'winrate']:.2f}% - {hourly.loc[best_hour, 'trades_count']} trades)")
            print(
                f"💀 Pire heure: {worst_hour}h ({hourly.loc[worst_hour, 'winrate']:.2f}% - {hourly.loc[worst_hour, 'trades_count']} trades)")

            # Top 3 heures
            top_hours = hourly.nlargest(3, 'winrate')
            print(
                f"📊 Top 3 heures: {', '.join([f'{h}h({wr:.1f}%)' for h, wr in zip(top_hours.index, top_hours['winrate'])])}")

        # Analyse par jour de la semaine
        if 'daily' in patterns and not patterns['daily'].empty:
            daily = patterns['daily']
            days_names = ['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim']

            print(f"\n📅 PERFORMANCE PAR JOUR:")
            for day_idx in daily.index:
                day_name = days_names[day_idx] if day_idx < len(days_names) else f'Jour{day_idx}'
                print(
                    f"   {day_name}: {daily.loc[day_idx, 'winrate']:5.2f}% ({daily.loc[day_idx, 'trades_count']} trades)")


def print_market_conditions_analysis(market_dict):
    """
    Affiche l'analyse des conditions de marché
    """
    print("\n" + "=" * 80)
    print("📈 ANALYSE DES CONDITIONS DE MARCHÉ")
    print("=" * 80)

    for name, conditions in market_dict.items():
        if conditions is None or not conditions:
            continue

        print(f"\n=== {name} ===")
        print("🔍 Différences moyennes entre trades gagnants et perdants:")

        # Trier par importance de l'écart
        sorted_conditions = sorted(conditions.items(),
                                   key=lambda x: abs(x[1]['difference_pct']),
                                   reverse=True)

        for col, stats in sorted_conditions[:10]:  # Top 10 des différences
            direction = "📈" if stats['difference'] > 0 else "📉"
            print(f"{direction} {col:25}: {stats['difference']:8.2f} ({stats['difference_pct']:+6.1f}%)")


def calculate_candle_stats(df_complete, df_name):
    """
    Calcule les statistiques de durée des bougies et volume par tick pour un dataset
    """
    if df_complete.empty:
        print(f"❌ {df_name}: DataFrame vide")
        return None

    stats = {'count': len(df_complete)}

    # === ANALYSE DURÉE DES BOUGIES ===
    if 'sc_candleDuration' in df_complete.columns:
        durations = df_complete['sc_candleDuration'].dropna()

        if len(durations) > 0:
            stats.update({
                'duration_count': len(durations),
                'duration_mean_seconds': durations.mean(),
                'duration_mean_minutes': durations.mean() / 60,
                'duration_median_seconds': durations.median(),
                'duration_std_seconds': durations.std(),
                'duration_min_seconds': durations.min(),
                'duration_max_seconds': durations.max(),
                'duration_q25': durations.quantile(0.25),
                'duration_q75': durations.quantile(0.75)
            })
        else:
            print(f"⚠️  {df_name}: Aucune donnée de durée valide")
    else:
        print(f"⚠️  {df_name}: Colonne 'sc_candleDuration' introuvable")

    # === ANALYSE VOLUME PAR TICK ===
    if 'sc_volume_perTick' in df_complete.columns:
        volumes = df_complete['sc_volume_perTick'].dropna()

        if len(volumes) > 0:
            stats.update({
                'volume_count': len(volumes),
                'volume_mean': volumes.mean(),
                'volume_median': volumes.median(),
                'volume_std': volumes.std(),
                'volume_min': volumes.min(),
                'volume_max': volumes.max(),
                'volume_q25': volumes.quantile(0.25),
                'volume_q75': volumes.quantile(0.75)
            })
        else:
            print(f"⚠️  {df_name}: Aucune donnée de volume par tick valide")
    else:
        print(f"⚠️  {df_name}: Colonne 'sc_volume_perTick' introuvable")

    # === AFFICHAGE ===
    print(f"\n=== {df_name} ===")
    print(f"📊 Nombre total de bougies: {stats['count']:,}")

    # Durée des bougies
    if 'duration_mean_seconds' in stats:
        print(f"\n🕐 DURÉE DES BOUGIES:")
        print(
            f"⏱️  Durée moyenne: {stats['duration_mean_seconds']:.2f} secondes ({stats['duration_mean_minutes']:.2f} minutes)")
        print(f"📈 Médiane: {stats['duration_median_seconds']:.2f}s")
        print(f"📊 Écart-type: {stats['duration_std_seconds']:.2f}s")
        print(f"📉 Min: {stats['duration_min_seconds']:.0f}s | Max: {stats['duration_max_seconds']:.0f}s")
        print(f"📦 Q25: {stats['duration_q25']:.1f}s | Q75: {stats['duration_q75']:.1f}s")

    # Volume par tick
    if 'volume_mean' in stats:
        print(f"\n📈 VOLUME PAR TICK:")
        print(f"📊 Volume moyen: {stats['volume_mean']:.2f}")
        print(f"📈 Médiane: {stats['volume_median']:.2f}")
        print(f"📊 Écart-type: {stats['volume_std']:.2f}")
        print(f"📉 Min: {stats['volume_min']:.0f} | Max: {stats['volume_max']:.0f}")
        print(f"📦 Q25: {stats['volume_q25']:.1f} | Q75: {stats['volume_q75']:.1f}")

    return stats


def check_temporal_consistency(datasets):
    """
    Vérifie si les durées sont cohérentes avec les timestamps
    """
    print("\n" + "=" * 50)
    print("🕐 VÉRIFICATION COHÉRENCE TEMPORELLE")
    print("=" * 50)

    for df, name in datasets:
        if df.empty:
            print(f"{name:12}: DataFrame vide - analyse impossible")
            continue

        if 'date' in df.columns and len(df) > 1:
            try:
                # Calculer les différences réelles entre timestamps
                df_sorted = df.sort_values('date').copy()
                time_diffs = df_sorted['date'].diff().dt.total_seconds().dropna()

                if len(time_diffs) > 0:
                    actual_mean = time_diffs.mean()
                    if 'sc_candleDuration' in df.columns:
                        reported_mean = df['sc_candleDuration'].mean()
                        if pd.notna(reported_mean) and reported_mean > 0:
                            diff_pct = abs(actual_mean - reported_mean) / actual_mean * 100

                            status = "✅" if diff_pct <= 10 else "⚠️"
                            print(
                                f"{status} {name:12}: Réel={actual_mean:.1f}s | Rapporté={reported_mean:.1f}s | Écart={diff_pct:.1f}%")

                            if diff_pct > 10:
                                print(f"   🔍 {name}: Écart significatif détecté!")
                        else:
                            print(f"❌ {name:12}: Durées rapportées invalides")
                    else:
                        print(f"❌ {name:12}: Colonne sc_candleDuration manquante")
                else:
                    print(f"❌ {name:12}: Impossible de calculer les différences temporelles")
            except Exception as e:
                print(f"❌ {name:12}: Erreur analyse temporelle - {e}")
        else:
            print(f"❌ {name:12}: Colonne 'date' manquante ou données insuffisantes")


def analyze_pre_trade_conditions(datasets_dict, n_candles_before=10):
    """
    Analyse les conditions des N bougies précédant chaque trade
    pour identifier les différences entre datasets
    """

    print(f"\n🔍 ANALYSE DES {n_candles_before} BOUGIES PRÉCÉDANT CHAQUE TRADE")
    print("=" * 80)

    results = {}

    for name, (df_complete, df_filtered) in datasets_dict.items():
        if df_complete.empty or df_filtered.empty:
            continue

        print(f"\n📊 DATASET: {name}")
        print("-" * 50)

        # Préparer les données
        df_complete_sorted = df_complete.sort_values(['session_id', 'date']).reset_index(drop=True)

        # Identifier les indices des trades dans df_complete
        trade_indices = []
        for _, trade_row in df_filtered.iterrows():
            # Trouver l'indice correspondant dans df_complete
            matching_indices = df_complete_sorted[
                (df_complete_sorted['date'] == trade_row['date']) &
                (df_complete_sorted['session_id'] == trade_row['session_id'])
                ].index

            if len(matching_indices) > 0:
                trade_indices.append(matching_indices[0])

        print(f"Trades identifiés: {len(trade_indices)}")

        # Analyser les conditions pré-trade
        pre_trade_conditions = []

        for trade_idx in trade_indices:
            # Vérifier qu'on a assez de bougies avant
            if trade_idx >= n_candles_before:
                # Extraire les N bougies précédentes
                start_idx = trade_idx - n_candles_before
                pre_candles = df_complete_sorted.iloc[start_idx:trade_idx]

                # Calculer les métriques
                conditions = calculate_pre_trade_metrics(pre_candles, n_candles_before)
                if conditions:
                    pre_trade_conditions.append(conditions)

        if pre_trade_conditions:
            # Agréger les résultats
            aggregated = aggregate_pre_trade_conditions(pre_trade_conditions, name)
            results[name] = aggregated

            # Afficher les résultats
            display_pre_trade_analysis(aggregated, name)

    # Analyse comparative
    if len(results) > 1:
        compare_pre_trade_conditions(results)

    return results


def calculate_pre_trade_metrics(pre_candles, n_candles):
    """
    Calcule les métriques pour les bougies pré-trade
    """
    if len(pre_candles) != n_candles:
        return None

    # Vérifier les colonnes nécessaires
    required_cols = ['sc_candleDuration', 'sc_volume_perTick']
    if not all(col in pre_candles.columns for col in required_cols):
        return None

    try:
        # Métriques de base
        conditions = {
            # Durée des bougies
            'duration_mean': pre_candles['sc_candleDuration'].mean(),
            'duration_median': pre_candles['sc_candleDuration'].median(),
            'duration_std': pre_candles['sc_candleDuration'].std(),
            'duration_min': pre_candles['sc_candleDuration'].min(),
            'duration_max': pre_candles['sc_candleDuration'].max(),

            # Volume par tick
            'volume_per_tick_mean': pre_candles['sc_volume_perTick'].mean(),
            'volume_per_tick_median': pre_candles['sc_volume_perTick'].median(),
            'volume_per_tick_std': pre_candles['sc_volume_perTick'].std(),
            'volume_per_tick_min': pre_candles['sc_volume_perTick'].min(),
            'volume_per_tick_max': pre_candles['sc_volume_perTick'].max(),

            # Vitesse des bougies
            'fast_candles_pct': (pre_candles['sc_candleDuration'] < 10).mean() * 100,
            'slow_candles_pct': (pre_candles['sc_candleDuration'] > 300).mean() * 100,

            # Volume des bougies (si disponible)
            'volume_candle_mean': None,
            'volume_candle_std': None,
        }

        # Volume des bougies si disponible
        if 'sc_volume' in pre_candles.columns:
            conditions['volume_candle_mean'] = pre_candles['sc_volume'].mean()
            conditions['volume_candle_std'] = pre_candles['sc_volume'].std()

        # ATR si disponible
        if 'sc_atr' in pre_candles.columns:
            conditions['atr_mean'] = pre_candles['sc_atr'].mean()
            conditions['atr_std'] = pre_candles['sc_atr'].std()
        else:
            conditions['atr_mean'] = None
            conditions['atr_std'] = None

        # Tendance des durées (accélération/décélération)
        durations = pre_candles['sc_candleDuration'].values
        if len(durations) >= 3:
            # Calculer la tendance (régression linéaire simple)
            x = np.arange(len(durations))
            trend_slope = np.polyfit(x, durations, 1)[0]
            conditions['duration_trend'] = trend_slope
        else:
            conditions['duration_trend'] = 0

        # Volatilité des volumes
        volumes = pre_candles['sc_volume_perTick'].values
        if len(volumes) >= 2:
            conditions['volume_volatility'] = np.std(volumes) / np.mean(volumes) if np.mean(volumes) > 0 else 0
        else:
            conditions['volume_volatility'] = 0

        return conditions

    except Exception as e:
        print(f"   ⚠️ Erreur calcul métriques: {e}")
        return None


def aggregate_pre_trade_conditions(conditions_list, dataset_name):
    """
    Agrège les conditions pré-trade pour un dataset
    """
    if not conditions_list:
        return None

    # Convertir en DataFrame pour faciliter les calculs
    df_conditions = pd.DataFrame(conditions_list)

    # Calculer les statistiques agrégées
    aggregated = {}

    for col in df_conditions.columns:
        if df_conditions[col].dtype in ['int64', 'float64']:
            aggregated[f'{col}_mean'] = df_conditions[col].mean()
            aggregated[f'{col}_std'] = df_conditions[col].std()
            aggregated[f'{col}_median'] = df_conditions[col].median()
            aggregated[f'{col}_q25'] = df_conditions[col].quantile(0.25)
            aggregated[f'{col}_q75'] = df_conditions[col].quantile(0.75)

    aggregated['n_trades_analyzed'] = len(conditions_list)

    return aggregated


def display_pre_trade_analysis(aggregated, dataset_name):
    """
    Affiche l'analyse des conditions pré-trade
    """
    if not aggregated:
        print(f"❌ Pas de données pour {dataset_name}")
        return

    print(f"Trades analysés: {aggregated['n_trades_analyzed']}")

    print(f"\n🕐 DURÉE DES BOUGIES PRÉ-TRADE:")
    print(f"   Moyenne: {aggregated['duration_mean_mean']:6.1f}s ± {aggregated['duration_mean_std']:5.1f}s")
    print(f"   Médiane: {aggregated['duration_median_mean']:6.1f}s")
    print(f"   Min/Max: {aggregated['duration_min_mean']:6.1f}s / {aggregated['duration_max_mean']:6.1f}s")
    print(f"   Variabilité: {aggregated['duration_std_mean']:6.1f}s")
    print(f"   Tendance: {aggregated['duration_trend_mean']:+6.2f}s/bougie")

    print(f"\n📊 VOLUME PAR TICK PRÉ-TRADE:")
    print(f"   Moyenne: {aggregated['volume_per_tick_mean_mean']:6.1f} ± {aggregated['volume_per_tick_mean_std']:5.1f}")
    print(f"   Médiane: {aggregated['volume_per_tick_median_mean']:6.1f}")
    print(f"   Volatilité: {aggregated['volume_volatility_mean']:6.3f}")

    print(f"\n⚡ VITESSE DES BOUGIES PRÉ-TRADE:")
    print(f"   Rapides (<10s): {aggregated['fast_candles_pct_mean']:5.1f}%")
    print(f"   Lentes (>5min): {aggregated['slow_candles_pct_mean']:5.1f}%")

    if aggregated.get('atr_mean_mean'):
        print(f"\n🌪️ ATR PRÉ-TRADE:")
        print(f"   Moyenne: {aggregated['atr_mean_mean']:6.3f} ± {aggregated['atr_mean_std']:6.3f}")


def compare_pre_trade_conditions(results):
    """
    Compare les conditions pré-trade entre datasets
    """
    print(f"\n🔍 COMPARAISON DES CONDITIONS PRÉ-TRADE ENTRE DATASETS")
    print("=" * 80)

    if 'TEST' not in results:
        print("❌ TEST non trouvé dans les résultats")
        return

    test_results = results['TEST']
    other_datasets = {k: v for k, v in results.items() if k != 'TEST'}

    print(f"\n📊 ANALYSE COMPARATIVE (TEST vs AUTRES)")
    print("-" * 60)

    # Métriques clés à comparer
    key_metrics = [
        ('duration_mean_mean', 'Durée moyenne', 's'),
        ('duration_std_mean', 'Variabilité durée', 's'),
        ('volume_per_tick_mean_mean', 'Volume/tick moyen', ''),
        ('volume_volatility_mean', 'Volatilité volume', ''),
        ('fast_candles_pct_mean', 'Bougies rapides', '%'),
        ('duration_trend_mean', 'Tendance durée', 's/bougie'),
    ]

    significant_differences = []

    for metric, description, unit in key_metrics:
        if metric not in test_results:
            continue

        test_value = test_results[metric]

        # Calculer moyenne des autres datasets
        other_values = []
        for name, data in other_datasets.items():
            if metric in data:
                other_values.append(data[metric])

        if not other_values:
            continue

        others_mean = np.mean(other_values)
        difference_pct = ((test_value - others_mean) / others_mean * 100) if others_mean != 0 else 0

        # Seuil de significativité
        is_significant = abs(difference_pct) > 10

        status = "🔴" if abs(difference_pct) > 25 else "🟡" if abs(difference_pct) > 15 else "🟢"
        direction = "plus élevé" if difference_pct > 0 else "plus bas"

        print(
            f"{status} {description:20}: TEST {direction} de {abs(difference_pct):5.1f}% ({test_value:.2f}{unit} vs {others_mean:.2f}{unit})")

        if is_significant:
            significant_differences.append({
                'metric': description,
                'test_value': test_value,
                'others_mean': others_mean,
                'difference_pct': difference_pct,
                'unit': unit
            })

    # Analyse détaillée des différences significatives
    if significant_differences:
        print(f"\n🚨 DIFFÉRENCES SIGNIFICATIVES DÉTECTÉES:")
        print("-" * 60)

        for diff in significant_differences:
            print(f"\n📍 {diff['metric']}:")
            print(f"   TEST: {diff['test_value']:.2f}{diff['unit']}")
            print(f"   AUTRES: {diff['others_mean']:.2f}{diff['unit']}")
            print(f"   ÉCART: {diff['difference_pct']:+.1f}%")

            # Interprétation
            if 'durée' in diff['metric'].lower():
                if diff['difference_pct'] > 0:
                    print(f"   ➜ TEST opère sur des bougies plus lentes avant trade")
                else:
                    print(f"   ➜ TEST opère sur des bougies plus rapides avant trade")
            elif 'volume' in diff['metric'].lower():
                if diff['difference_pct'] > 0:
                    print(f"   ➜ TEST trade dans des conditions de volume plus élevé")
                else:
                    print(f"   ➜ TEST trade dans des conditions de volume plus faible")
            elif 'tendance' in diff['metric'].lower():
                if diff['difference_pct'] > 0:
                    print(f"   ➜ TEST trade quand les bougies s'accélèrent")
                else:
                    print(f"   ➜ TEST trade quand les bougies ralentissent")

    # Diagnostic final
    print(f"\n💡 DIAGNOSTIC PRÉ-TRADE:")
    print("-" * 40)

    if len(significant_differences) == 0:
        print("✅ Les conditions pré-trade de TEST sont similaires aux autres")
        print("   → Le problème n'est probablement PAS dans les conditions d'entrée")
        print("   → Chercher dans les signaux, le timing d'exécution, ou la gestion")
    else:
        print(f"⚠️ {len(significant_differences)} différences majeures détectées")
        print("   → Les conditions pré-trade de TEST sont distinctes")
        print("   → Ces différences peuvent expliquer la sous-performance")

        # Recommandations basées sur les différences
        for diff in significant_differences:
            if 'rapides' in diff['metric'].lower() and diff['difference_pct'] > 0:
                print("   → Recommandation: Filtrer les périodes de bougies trop rapides")
            elif 'variabilité' in diff['metric'].lower() and diff['difference_pct'] > 0:
                print("   → Recommandation: Éviter les périodes trop volatiles")


def analyze_pre_trade_by_performance(datasets_dict, n_candles_before=10):
    """
    Analyse spécifique: comparer les conditions pré-trade des wins vs losses
    """
    print(f"\n🎯 ANALYSE PRÉ-TRADE: WINS vs LOSSES")
    print("=" * 60)

    for name, (df_complete, df_filtered) in datasets_dict.items():
        if df_complete.empty or df_filtered.empty:
            continue

        print(f"\n📊 {name}:")
        print("-" * 30)

        # Séparer wins et losses
        wins = df_filtered[df_filtered['class_binaire'] == 1]
        losses = df_filtered[df_filtered['class_binaire'] == 0]

        print(f"Wins: {len(wins)}, Losses: {len(losses)}")

        # Analyser les conditions pré-trade pour chaque groupe
        for group_name, group_data in [("WINS", wins), ("LOSSES", losses)]:
            if len(group_data) < 10:  # Minimum de trades pour l'analyse
                continue

            # Échantillonner si trop de données
            sample_size = min(500, len(group_data))
            sample_data = group_data.sample(n=sample_size, random_state=42)

            pre_conditions = []

            for _, trade_row in sample_data.iterrows():
                # Trouver la position dans df_complete
                df_complete_sorted = df_complete.sort_values(['session_id', 'date'])

                matching_rows = df_complete_sorted[
                    (df_complete_sorted['date'] == trade_row['date']) &
                    (df_complete_sorted['session_id'] == trade_row['session_id'])
                    ]

                if len(matching_rows) > 0:
                    trade_idx = matching_rows.index[0]
                    complete_idx = df_complete_sorted.index.get_loc(trade_idx)

                    if complete_idx >= n_candles_before:
                        start_idx = complete_idx - n_candles_before
                        pre_candles = df_complete_sorted.iloc[start_idx:complete_idx]

                        conditions = calculate_pre_trade_metrics(pre_candles, n_candles_before)
                        if conditions:
                            pre_conditions.append(conditions)

            if pre_conditions:
                avg_conditions = aggregate_pre_trade_conditions(pre_conditions, f"{name}_{group_name}")

                if avg_conditions:
                    print(f"\n{group_name} ({len(pre_conditions)} échantillons):")
                    print(f"  Durée moy: {avg_conditions['duration_mean_mean']:5.1f}s")
                    print(f"  Vol/tick:  {avg_conditions['volume_per_tick_mean_mean']:5.1f}")
                    print(f"  Rapides:   {avg_conditions['fast_candles_pct_mean']:4.1f}%")


# Fonction principale à intégrer dans le script principal
def add_pre_trade_analysis_to_main(datasets_dict, n_candles_before=10):
    """
    Fonction à ajouter dans la fonction main() du script principal
    """

    # Analyse des conditions pré-trade
    print("\n" + "=" * 80)
    print("🔍 ANALYSE DES CONDITIONS PRÉ-TRADE")
    print("=" * 80)

    pre_trade_results = analyze_pre_trade_conditions(datasets_dict, n_candles_before)

    # Analyse wins vs losses
    analyze_pre_trade_by_performance(datasets_dict, n_candles_before)

    return pre_trade_results


import pandas as pd
import numpy as np
from pathlib import Path

# ===== ÉTAPE 1: SUPPRIMER COMPLÈTEMENT calculate_slopes_pretrade_numba =====

# ❌ SUPPRIMEZ CETTE FONCTION ENTIÈREMENT DE VOTRE CODE :
"""
@njit
def calculate_slopes_pretrade_numba(values, window=10):
    # FONCTION À SUPPRIMER - ELLE FAIT DU CLIPPING FORCÉ
    pass
"""


# ===== ÉTAPE 2: FONCTION DE REMPLACEMENT UNIQUE =====
def calculate_pre_trade_slopes_CLEAN(pre_candles, debug_mode=False):
    """
    Version PROPRE utilisant UNIQUEMENT calculate_slopes_and_r2_numba
    Plus de duplication, plus de clipping forcé
    """
    try:
        durations = pre_candles['sc_candleDuration'].values
        volumes = pre_candles['sc_volume_perTick'].values

        if len(durations) < 3 or len(volumes) < 3:
            return {
                'duration_slope': 0, 'duration_r2': 0, 'duration_std': 0,
                'volume_slope': 0, 'volume_r2': 0, 'volume_std': 0
            }

        # Créer session_starts (première bougie = début de session)
        session_starts_dur = np.zeros(len(durations), dtype=bool)
        session_starts_vol = np.zeros(len(volumes), dtype=bool)
        session_starts_dur[0] = True
        session_starts_vol[0] = True

        if debug_mode:
            print(f"   🔍 CLEAN: Calcul slopes avec calculate_slopes_and_r2_numba SANS clipping...")
            print(f"   🔍 Durées: {len(durations)} valeurs, range=[{np.min(durations):.1f}, {np.max(durations):.1f}]")
            print(f"   🔍 Volumes: {len(volumes)} valeurs, range=[{np.min(volumes):.1f}, {np.max(volumes):.1f}]")

        # ✅ UTILISATION UNIQUE DE VOTRE FONCTION OFFICIELLE
        duration_slopes, r2_dur, std_dur = calculate_slopes_and_r2_numba(
            durations, session_starts_dur, SLOPE_PERIODS,
            clip_slope=False,  # ← SANS CLIPPING
            include_close_bar=True
        )

        volume_slopes, r2_vol, std_vol = calculate_slopes_and_r2_numba(
            volumes, session_starts_vol, SLOPE_PERIODS,
            clip_slope=False,  # ← SANS CLIPPING
            include_close_bar=True
        )

        # Extraire les dernières valeurs valides
        dur_slope = 0
        vol_slope = 0
        dur_r2 = 0
        vol_r2 = 0
        dur_std = 0
        vol_std = 0

        if len(duration_slopes) > 0:
            valid_dur_slopes = duration_slopes[~np.isnan(duration_slopes)]
            if len(valid_dur_slopes) > 0:
                dur_slope = valid_dur_slopes[-1]
                dur_r2 = r2_dur[~np.isnan(duration_slopes)][-1] if len(r2_dur[~np.isnan(duration_slopes)]) > 0 else 0
                dur_std = std_dur[~np.isnan(duration_slopes)][-1] if len(std_dur[~np.isnan(duration_slopes)]) > 0 else 0

        if len(volume_slopes) > 0:
            valid_vol_slopes = volume_slopes[~np.isnan(volume_slopes)]
            if len(valid_vol_slopes) > 0:
                vol_slope = valid_vol_slopes[-1]
                vol_r2 = r2_vol[~np.isnan(volume_slopes)][-1] if len(r2_vol[~np.isnan(volume_slopes)]) > 0 else 0
                vol_std = std_vol[~np.isnan(volume_slopes)][-1] if len(std_vol[~np.isnan(volume_slopes)]) > 0 else 0

        if debug_mode:
            print(f"   ✅ CLEAN: Slopes calculées SANS clipping:")
            print(f"      Duration slope: {dur_slope:.6f} (R²={dur_r2:.3f})")
            print(f"      Volume slope: {vol_slope:.6f} (R²={vol_r2:.3f})")

        return {
            'duration_slope': dur_slope,
            'duration_r2': dur_r2,
            'duration_std': dur_std,
            'volume_slope': vol_slope,
            'volume_r2': vol_r2,
            'volume_std': vol_std
        }

    except Exception as e:
        if debug_mode:
            print(f"   ❌ ERREUR calculate_slopes_and_r2_numba: {e}")

        # Fallback numpy simple
        try:
            x = np.arange(len(durations))
            dur_slope = np.polyfit(x, durations, 1)[0] if len(durations) >= 3 else 0
            vol_slope = np.polyfit(x, volumes, 1)[0] if len(volumes) >= 3 else 0

            if debug_mode:
                print(f"   🔧 FALLBACK numpy: dur={dur_slope:.6f}, vol={vol_slope:.6f}")

            return {
                'duration_slope': dur_slope, 'duration_r2': 0, 'duration_std': 0,
                'volume_slope': vol_slope, 'volume_r2': 0, 'volume_std': 0
            }
        except Exception as e2:
            if debug_mode:
                print(f"   ❌ ERREUR fallback: {e2}")
            return {
                'duration_slope': 0, 'duration_r2': 0, 'duration_std': 0,
                'volume_slope': 0, 'volume_r2': 0, 'volume_std': 0
            }



# ===== ÉTAPE 3: FONCTION MÉTRIQUES NETTOYÉE =====

def calculate_pre_trade_metrics_CLEAN(pre_candles):
    """
    Version NETTOYÉE - utilise uniquement calculate_slopes_and_r2_numba
    """
    try:
        required_cols = ['sc_candleDuration', 'sc_volume_perTick']
        if not all(col in pre_candles.columns for col in required_cols):
            return None

        durations = pre_candles['sc_candleDuration'].values
        volumes = pre_candles['sc_volume_perTick'].values

        if len(durations) == 0 or len(volumes) == 0:
            return None
        if np.all(np.isnan(durations)) or np.all(np.isnan(volumes)):
            return None

        # Calculs de base
        stats = {
            'duration_mean': np.nanmean(durations),
            'duration_median': np.nanmedian(durations),
            'duration_std': np.nanstd(durations),
            'volume_mean': np.nanmean(volumes),
            'volume_median': np.nanmedian(volumes),
            'volume_std': np.nanstd(volumes),
            'fast_candles_pct': np.mean(durations < 10) * 100,
            'slow_candles_pct': np.mean(durations > 300) * 100,
        }

        # Coefficients de variation
        stats['duration_cv'] = stats['duration_std'] / stats['duration_mean'] if stats['duration_mean'] > 0 else 0
        stats['volume_cv'] = stats['volume_std'] / stats['volume_mean'] if stats['volume_mean'] > 0 else 0

        # ✅ SLOPES AVEC LA FONCTION PROPRE
        if len(durations) >= 3:
            slope_results = calculate_pre_trade_slopes_CLEAN(pre_candles)

            stats['duration_trend'] = slope_results['duration_slope']
            stats['duration_slope_stdev'] = slope_results['duration_std']
            stats['duration_slope_r2'] = slope_results['duration_r2']
            stats['volume_trend'] = slope_results['volume_slope']
            stats['volume_slope_stdev'] = slope_results['volume_std']
            stats['volume_slope_r2'] = slope_results['volume_r2']
        else:
            return None

        # Métriques supplémentaires
        stats['duration_volatility'] = stats['duration_cv']
        stats['volume_volatility'] = stats['volume_cv']

        if len(durations) >= 5:
            mid = len(durations) // 2
            stats['duration_acceleration'] = np.mean(durations[mid:]) - np.mean(durations[:mid])
            stats['volume_acceleration'] = np.mean(volumes[mid:]) - np.mean(volumes[:mid])
        else:
            stats['duration_acceleration'] = 0
            stats['volume_acceleration'] = 0

        return stats

    except Exception as e:
        print(f"   ❌ Erreur calculate_pre_trade_metrics_CLEAN: {e}")
        return None


def analyze_pre_trade_conditions_complete(datasets_dict, n_candles_before=10):
    """
    Version COMPLÈTE OPTIMISÉE - Analyse TOUS les trades sans échantillonnage
    """

    print(f"\n🔍 ANALYSE COMPLÈTE DES {n_candles_before} BOUGIES PRÉ-TRADE (TOUS LES TRADES)")
    print("=" * 80)

    results = {}

    for name, (df_complete, df_filtered) in datasets_dict.items():
        if df_complete.empty or df_filtered.empty:
            continue

        print(f"\n📊 DATASET: {name}")
        print("-" * 50)
        print(f"Analyse de {len(df_filtered):,} trades sur {len(df_complete):,} bougies...")

        # ===== OPTIMISATION MAJEURE 1: INDEX MULTI-NIVEAUX =====
        print("   Préparation des index...")

        # Créer un index optimisé par session
        df_complete_reset = df_complete.reset_index(drop=True)
        df_complete_reset['row_number'] = df_complete_reset.index

        # Trier par session et date pour recherche séquentielle
        df_complete_sorted = df_complete_reset.sort_values(['session_id', 'date']).reset_index(drop=True)

        # ===== OPTIMISATION MAJEURE 2: GROUPBY PAR SESSION =====
        print("   Groupement par sessions...")

        # Grouper par session pour traitement parallèle
        session_groups = df_complete_sorted.groupby('session_id')
        session_indices = {}

        for session_id, group in session_groups:
            session_indices[session_id] = {
                'data': group.reset_index(drop=True),
                'start_idx': group.index[0] if len(group) > 0 else 0
            }

        print(f"   {len(session_indices)} sessions indexées")

        # ===== OPTIMISATION MAJEURE 3: TRAITEMENT VECTORISÉ =====
        print("   Traitement vectorisé des trades...")

        pre_trade_stats = []
        total_trades = len(df_filtered)
        processed = 0
        skipped = 0

        # Grouper les trades par session aussi
        trades_by_session = df_filtered.groupby('session_id')

        for session_id, session_trades in trades_by_session:
            if session_id not in session_indices:
                skipped += len(session_trades)
                continue

            session_data = session_indices[session_id]['data']

            if len(session_data) < n_candles_before + 1:
                skipped += len(session_trades)
                continue

            # Traitement vectorisé de tous les trades de cette session
            session_stats = process_session_trades_vectorized(
                session_trades, session_data, n_candles_before
            )

            pre_trade_stats.extend(session_stats)
            processed += len(session_trades)

            # Affichage progression
            if processed % 500 == 0 or processed == total_trades:
                pct = (processed / total_trades) * 100
                print(f"   Progression: {processed:,}/{total_trades:,} trades ({pct:.1f}%)")

        print(f"   ✅ Analysés: {len(pre_trade_stats):,} | Ignorés: {skipped:,}")

        if pre_trade_stats:
            # Analyse des résultats sans perte de données
            analysis = analyze_complete_stats(pre_trade_stats, name)
            results[name] = analysis
            display_complete_results(analysis, name)

    # Comparaison complète
    if len(results) > 1:
        compare_datasets_complete(results)

    return results


def process_session_trades_vectorized(session_trades, session_data, n_candles_before):
    """
    Traitement vectorisé optimisé de tous les trades d'une session
    """
    session_stats = []

    # Convertir les dates en index pour recherche rapide
    session_dates = pd.to_datetime(session_data['date'])

    for _, trade_row in session_trades.iterrows():
        trade_date = pd.to_datetime(trade_row['date'])

        # Recherche vectorisée de la position
        # Trouver toutes les bougies avant ou égales à la date du trade
        before_trade_mask = session_dates <= trade_date
        before_trade_indices = np.where(before_trade_mask)[0]

        if len(before_trade_indices) < n_candles_before + 1:
            continue

        # Position du trade (dernière bougie avant ou égale)
        trade_position = before_trade_indices[-1]

        # Vérifier qu'on a assez de bougies avant
        if trade_position < n_candles_before:
            continue

        # Extraire les N bougies précédentes
        start_pos = trade_position - n_candles_before
        end_pos = trade_position

        pre_candles = session_data.iloc[start_pos:end_pos]

        if len(pre_candles) == n_candles_before:
            stats = calculate_pre_trade_metrics_CLEAN(pre_candles)
            if stats:
                stats['trade_result'] = trade_row['class_binaire']
                stats['session_id'] = trade_row['session_id']
                stats['trade_date'] = trade_row['date']
                session_stats.append(stats)

    return session_stats
def calculate_pre_trade_metrics_complete_debug_fixed(pre_candles):
    """
    Version corrigée utilisant DIRECTEMENT calculate_slopes_and_r2_numba
    """
    try:
        required_cols = ['sc_candleDuration', 'sc_volume_perTick']
        if not all(col in pre_candles.columns for col in required_cols):
            return None

        durations = pre_candles['sc_candleDuration'].values
        volumes = pre_candles['sc_volume_perTick'].values

        if len(durations) == 0 or len(volumes) == 0:
            return None
        if np.all(np.isnan(durations)) or np.all(np.isnan(volumes)):
            return None

        # Calculs de base
        stats = {
            'duration_mean': np.nanmean(durations),
            'duration_median': np.nanmedian(durations),
            'duration_std': np.nanstd(durations),
            'volume_mean': np.nanmean(volumes),
            'volume_median': np.nanmedian(volumes),
            'volume_std': np.nanstd(volumes),
            'fast_candles_pct': np.mean(durations < 10) * 100,
            'slow_candles_pct': np.mean(durations > 300) * 100,
        }

        # Coefficients de variation
        if stats['duration_mean'] > 0:
            stats['duration_cv'] = stats['duration_std'] / stats['duration_mean']
        else:
            stats['duration_cv'] = 0

        if stats['volume_mean'] > 0:
            stats['volume_cv'] = stats['volume_std'] / stats['volume_mean']
        else:
            stats['volume_cv'] = 0

        # ✅ SLOPES AVEC VOTRE FONCTION OFFICIELLE calculate_slopes_and_r2_numba
        if len(durations) >= 3:
            # Créer session_starts (première bougie = début de session)
            session_starts_dur = np.zeros(len(durations), dtype=bool)
            session_starts_vol = np.zeros(len(volumes), dtype=bool)
            session_starts_dur[0] = True
            session_starts_vol[0] = True

            # Appel direct à votre fonction
            duration_slopes, r2_dur, std_dur = calculate_slopes_and_r2_numba(
                durations, session_starts_dur, SLOPE_PERIODS,
                clip_slope=False,  # ← SANS CLIPPING
                include_close_bar=True
            )

            volume_slopes, r2_vol, std_vol = calculate_slopes_and_r2_numba(
                volumes, session_starts_vol, SLOPE_PERIODS,
                clip_slope=False,  # ← SANS CLIPPING
                include_close_bar=True
            )

            # Extraire les dernières valeurs valides
            dur_slope = 0
            vol_slope = 0
            dur_r2 = 0
            vol_r2 = 0
            dur_std = 0
            vol_std = 0

            if len(duration_slopes) > 0:
                valid_indices_dur = ~np.isnan(duration_slopes)
                if np.any(valid_indices_dur):
                    dur_slope = duration_slopes[valid_indices_dur][-1]
                    dur_r2 = r2_dur[valid_indices_dur][-1]
                    dur_std = std_dur[valid_indices_dur][-1]

            if len(volume_slopes) > 0:
                valid_indices_vol = ~np.isnan(volume_slopes)
                if np.any(valid_indices_vol):
                    vol_slope = volume_slopes[valid_indices_vol][-1]
                    vol_r2 = r2_vol[valid_indices_vol][-1]
                    vol_std = std_vol[valid_indices_vol][-1]

            stats['duration_trend'] = dur_slope
            stats['duration_slope_stdev'] = dur_std
            stats['duration_slope_r2'] = dur_r2
            stats['volume_trend'] = vol_slope
            stats['volume_slope_stdev'] = vol_std
            stats['volume_slope_r2'] = vol_r2

        else:
            return None

        # Autres métriques
        stats['duration_volatility'] = stats['duration_cv']
        stats['volume_volatility'] = stats['volume_cv']

        if len(durations) >= 5:
            mid = len(durations) // 2
            first_half_dur = np.mean(durations[:mid])
            second_half_dur = np.mean(durations[mid:])
            stats['duration_acceleration'] = second_half_dur - first_half_dur

            first_half_vol = np.mean(volumes[:mid])
            second_half_vol = np.mean(volumes[mid:])
            stats['volume_acceleration'] = second_half_vol - first_half_vol
        else:
            stats['duration_acceleration'] = 0
            stats['volume_acceleration'] = 0

        return stats

    except Exception as e:
        print(f"   ❌ Erreur calculate_pre_trade_metrics_complete_debug_fixed: {e}")
        return None

def process_session_trades_CLEAN(session_trades, session_data, n_candles_before):
    """
    Version nettoyée du traitement de session
    """
    session_stats = []

    session_dates_np = session_data['date'].values.astype('datetime64[ns]')

    for _, trade_row in session_trades.iterrows():
        trade_date_np = np.datetime64(trade_row['date'])

        mask = session_dates_np <= trade_date_np
        indices = np.where(mask)[0]

        if len(indices) < n_candles_before + 1:
            continue

        trade_position = indices[-1]

        if trade_position < n_candles_before:
            continue

        start_pos = trade_position - n_candles_before
        end_pos = trade_position

        pre_candles = session_data.iloc[start_pos:end_pos]

        if len(pre_candles) == n_candles_before:
            # ✅ UTILISE LA FONCTION NETTOYÉE
            stats = calculate_pre_trade_metrics_CLEAN(pre_candles)
            if stats:
                stats.update({
                    'trade_result': trade_row['class_binaire'],
                    'session_id': trade_row['session_id'],
                    'trade_date': trade_row['date']
                })
                session_stats.append(stats)

    return session_stats


# ===== ÉTAPE 6: FONCTION PRINCIPALE NETTOYÉE =====

def analyze_pre_trade_conditions_CLEAN(datasets_dict, n_candles_before=10):
    """
    Version NETTOYÉE de l'analyse pré-trade
    Utilise uniquement calculate_slopes_and_r2_numba
    """
    print(f"\n🧹 ANALYSE PRÉ-TRADE NETTOYÉE (SANS DUPLICATION)")
    print("=" * 80)

    results = {}

    for name, (df_complete, df_filtered) in datasets_dict.items():
        if df_complete.empty or df_filtered.empty:
            continue

        print(f"\n🧹 DATASET CLEAN: {name}")
        print("-" * 50)

        # Préparation optimisée
        df_complete_sorted = df_complete.sort_values(['session_id', 'date']).reset_index(drop=True)
        session_groups = df_complete_sorted.groupby('session_id', sort=False)
        session_indices = {session_id: group.reset_index(drop=True)
                          for session_id, group in session_groups}

        pre_trade_stats = []
        total_trades = len(df_filtered)
        processed = 0

        trades_by_session = df_filtered.groupby('session_id', sort=False)

        for session_id, session_trades in trades_by_session:
            if session_id not in session_indices:
                continue

            session_data = session_indices[session_id]
            if len(session_data) < n_candles_before + 1:
                continue

            # ✅ TRAITEMENT NETTOYÉ
            session_stats = process_session_trades_CLEAN(
                session_trades, session_data, n_candles_before
            )

            pre_trade_stats.extend(session_stats)
            processed += len(session_trades)

            if processed % 1000 == 0:
                pct = (processed / total_trades) * 100
                print(f"   🧹 Clean progression: {processed:,}/{total_trades:,} ({pct:.1f}%)")

        print(f"   ✅ CLEAN: {len(pre_trade_stats):,} trades analysés")

        if pre_trade_stats:
            analysis = analyze_complete_stats_fast(pre_trade_stats, name)
            results[name] = analysis
            display_complete_results_fast(analysis, name)

    return results
def analyze_complete_stats(pre_trade_stats, dataset_name):
    """
    Analyse complète sans perte d'information
    """
    df_stats = pd.DataFrame(pre_trade_stats)

    # Séparer wins et losses
    wins = df_stats[df_stats['trade_result'] == 1]
    losses = df_stats[df_stats['trade_result'] == 0]

    analysis = {
        'dataset': dataset_name,
        'total_trades': len(df_stats),
        'wins_count': len(wins),
        'losses_count': len(losses),
        'winrate': len(wins) / len(df_stats) * 100 if len(df_stats) > 0 else 0,
        'overall': {},
        'wins': {},
        'losses': {},
        'differences': {},
        'raw_data': df_stats  # Garder les données brutes
    }

    # Toutes les métriques numériques
    numeric_cols = [col for col in df_stats.columns if df_stats[col].dtype in ['int64', 'float64']]
    numeric_cols = [col for col in numeric_cols if col != 'trade_result']  # Exclure le résultat

    for col in numeric_cols:
        # Statistiques globales
        analysis['overall'][col] = {
            'mean': df_stats[col].mean(),
            'median': df_stats[col].median(),
            'std': df_stats[col].std(),
            'min': df_stats[col].min(),
            'max': df_stats[col].max(),
            'q25': df_stats[col].quantile(0.25),
            'q75': df_stats[col].quantile(0.75)
        }

        # Statistiques wins
        if len(wins) > 0:
            analysis['wins'][col] = {
                'mean': wins[col].mean(),
                'median': wins[col].median(),
                'std': wins[col].std()
            }

        # Statistiques losses
        if len(losses) > 0:
            analysis['losses'][col] = {
                'mean': losses[col].mean(),
                'median': losses[col].median(),
                'std': losses[col].std()
            }

        # Différences wins vs losses
        if len(wins) > 0 and len(losses) > 0:
            wins_mean = analysis['wins'][col]['mean']
            losses_mean = analysis['losses'][col]['mean']
            overall_mean = analysis['overall'][col]['mean']

            diff_abs = wins_mean - losses_mean
            diff_pct = (diff_abs / overall_mean * 100) if overall_mean != 0 else 0

            analysis['differences'][col] = {
                'abs': diff_abs,
                'pct': diff_pct,
                'wins_mean': wins_mean,
                'losses_mean': losses_mean
            }

    return analysis


def display_complete_results(analysis, dataset_name):
    """
    Affichage complet des résultats - MODIFIÉ pour inclure les infos slopes officielles
    """
    print(f"\n📈 RÉSULTATS COMPLETS {dataset_name}:")
    print(f"   Total trades: {analysis['total_trades']:,}")
    print(f"   Wins: {analysis['wins_count']:,} | Losses: {analysis['losses_count']:,}")
    print(f"   Winrate échantillon: {analysis['winrate']:.2f}%")

    # Conditions pré-trade moyennes
    if 'duration_mean' in analysis['overall']:
        dur_data = analysis['overall']['duration_mean']
        vol_data = analysis['overall']['volume_mean']

        print(f"\n   🕐 CONDITIONS PRÉ-TRADE MOYENNES:")
        print(f"   • Durée: {dur_data['mean']:5.1f}s (médiane: {dur_data['median']:5.1f}s)")
        print(f"   • Volume/tick: {vol_data['mean']:5.1f} (médiane: {vol_data['median']:5.1f})")

        if 'fast_candles_pct' in analysis['overall']:
            fast_data = analysis['overall']['fast_candles_pct']
            slow_data = analysis['overall']['slow_candles_pct']
            print(f"   • Rapides (<10s): {fast_data['mean']:4.1f}%")
            print(f"   • Lentes (>5min): {slow_data['mean']:4.1f}%")

    # Top différences wins vs losses avec info slopes
    if analysis['differences']:
        print(f"\n   🎯 TOP DIFFÉRENCES WINS vs LOSSES:")

        # Trier par importance de la différence
        sorted_diffs = sorted(analysis['differences'].items(),
                              key=lambda x: abs(x[1]['pct']), reverse=True)

        significant_count = 0
        for metric, diff_data in sorted_diffs:
            if abs(diff_data['pct']) > 3:  # Seuil 3% pour être inclusif
                significant_count += 1
                if significant_count <= 5:  # Top 5
                    direction = "plus élevé" if diff_data['pct'] > 0 else "plus bas"

                    # Ajouter info sur la méthode de calcul pour les trends
                    method_info = ""
                    if 'trend' in metric:
                        method_info = f" (calculate_slopes_and_r2_numba, {SLOPE_PERIODS} périodes)"

                    print(f"   • {metric}: WINS {direction} de {abs(diff_data['pct']):.1f}%{method_info}")

        if significant_count == 0:
            print(f"   • Aucune différence significative (>3%) détectée")


# FONCTIONS MANQUANTES À AJOUTER À VOTRE SCRIPT

def analyze_wins_vs_losses_slopes(analysis_results):
    """
    Analyse détaillée WINS vs LOSSES sur les slopes
    FONCTION MANQUANTE CRITIQUE pour valider nos hypothèses
    """
    print(f"\n🎯 ANALYSE SLOPES: WINS vs LOSSES (calculate_slopes_and_r2_numba)")
    print("=" * 70)

    slope_comparisons = {}

    for name, data in analysis_results.items():
        if 'raw_data' not in data or data['raw_data'].empty:
            continue

        df = data['raw_data']
        wins = df[df['trade_result'] == 1]
        losses = df[df['trade_result'] == 0]

        if len(wins) == 0 or len(losses) == 0:
            continue

        # Calculs des moyennes slopes
        wins_vol_trend = wins['volume_trend'].mean()
        losses_vol_trend = losses['volume_trend'].mean()
        wins_dur_trend = wins['duration_trend'].mean()
        losses_dur_trend = losses['duration_trend'].mean()

        # Calcul des différences en pourcentage
        vol_diff_abs = wins_vol_trend - losses_vol_trend
        dur_diff_abs = wins_dur_trend - losses_dur_trend

        # Pourcentages (par rapport à la moyenne générale)
        overall_vol = df['volume_trend'].mean()
        overall_dur = df['duration_trend'].mean()

        vol_diff_pct = (vol_diff_abs / abs(overall_vol) * 100) if abs(overall_vol) > 0.001 else 0
        dur_diff_pct = (dur_diff_abs / abs(overall_dur) * 100) if abs(overall_dur) > 0.001 else 0

        slope_comparisons[name] = {
            'wins_vol_trend': wins_vol_trend,
            'losses_vol_trend': losses_vol_trend,
            'wins_dur_trend': wins_dur_trend,
            'losses_dur_trend': losses_dur_trend,
            'vol_diff_abs': vol_diff_abs,
            'dur_diff_abs': dur_diff_abs,
            'vol_diff_pct': vol_diff_pct,
            'dur_diff_pct': dur_diff_pct
        }

        print(f"\n📊 {name}:")
        print(f"   Volume trend - WINS: {wins_vol_trend:+.4f} | LOSSES: {losses_vol_trend:+.4f}")
        print(f"   → Différence: WINS plus {'élevé' if vol_diff_abs > 0 else 'bas'} de {abs(vol_diff_pct):.1f}%")

        print(f"   Duration trend - WINS: {wins_dur_trend:+.4f} | LOSSES: {losses_dur_trend:+.4f}")
        print(f"   → Différence: WINS plus {'élevé' if dur_diff_abs > 0 else 'bas'} de {abs(dur_diff_pct):.1f}%")

    return slope_comparisons


def validate_test_timing_hypothesis(slope_comparisons):
    """
    Valide spécifiquement l'hypothèse du timing tardif de TEST
    """
    print(f"\n🚨 VALIDATION HYPOTHÈSE: TEST trade en retard (épuisement)")
    print("=" * 60)

    if 'TEST' not in slope_comparisons:
        print("❌ TEST non trouvé dans les comparaisons")
        return

    test_data = slope_comparisons['TEST']
    other_datasets = {k: v for k, v in slope_comparisons.items() if k != 'TEST'}

    print(f"📊 PATTERN TEST:")
    print(f"   Volume trend WINS vs LOSSES: {test_data['vol_diff_pct']:+.1f}%")
    print(f"   Duration trend WINS vs LOSSES: {test_data['dur_diff_pct']:+.1f}%")

    # Calculer moyenne des autres datasets
    if other_datasets:
        other_vol_diffs = [data['vol_diff_pct'] for data in other_datasets.values()]
        other_dur_diffs = [data['dur_diff_pct'] for data in other_datasets.values()]

        avg_vol_diff_others = sum(other_vol_diffs) / len(other_vol_diffs)
        avg_dur_diff_others = sum(other_dur_diffs) / len(other_dur_diffs)

        print(f"\n📊 PATTERN AUTRES DATASETS (moyenne):")
        print(f"   Volume trend WINS vs LOSSES: {avg_vol_diff_others:+.1f}%")
        print(f"   Duration trend WINS vs LOSSES: {avg_dur_diff_others:+.1f}%")

        print(f"\n🎯 COMPARAISON TEST vs AUTRES:")

        # Test de l'hypothèse volume
        vol_divergence = test_data['vol_diff_pct'] - avg_vol_diff_others
        print(f"   Volume trend divergence: {vol_divergence:+.1f}%")

        if test_data['vol_diff_pct'] > 0 and avg_vol_diff_others < 0:
            print("   ✅ HYPOTHÈSE CONFIRMÉE: TEST trade sur volume croissant vs autres sur volume décroissant")
            print("   → TEST entre tard (épuisement) vs autres entrent tôt (accumulation)")
        elif abs(vol_divergence) > 50:
            print(f"   ⚠️  FORTE DIVERGENCE: TEST a un pattern volume très différent")
        else:
            print("   🔶 Pas de divergence majeure sur le volume trend")

        # Test de l'hypothèse durée
        dur_divergence = test_data['dur_diff_pct'] - avg_dur_diff_others
        print(f"   Duration trend divergence: {dur_divergence:+.1f}%")

        if test_data['dur_diff_pct'] > 0 and avg_dur_diff_others < 0:
            print("   ✅ HYPOTHÈSE CONFIRMÉE: TEST trade sur bougies qui ralentissent vs autres sur accélération")
        elif abs(dur_divergence) > 50:
            print(f"   ⚠️  FORTE DIVERGENCE: TEST a un pattern durée très différent")
        else:
            print("   🔶 Pas de divergence majeure sur le duration trend")

        # Diagnostic final
        print(f"\n💡 DIAGNOSTIC FINAL:")

        vol_confirmed = test_data['vol_diff_pct'] > 0 and avg_vol_diff_others < 0
        dur_confirmed = test_data['dur_diff_pct'] > 0 and avg_dur_diff_others < 0

        if vol_confirmed and dur_confirmed:
            print("   🔴 TIMING TARDIF CONFIRMÉ: TEST entre en phase d'épuisement")
            print("   → Volume qui monte (participation tardive) + bougies qui ralentissent")
            print("   → Recommandation: Filtrer ces conditions ou inverser la logique")
        elif vol_confirmed or dur_confirmed:
            print("   🟡 TIMING TARDIF PARTIELLEMENT CONFIRMÉ")
            print("   → Une des deux conditions d'épuisement détectée")
        else:
            print("   🟢 TIMING TARDIF NON CONFIRMÉ")
            print("   → Les patterns slopes ne confirment pas l'hypothèse d'épuisement")


def compare_slope_patterns_detailed(slope_comparisons):
    """
    Comparaison détaillée des patterns slopes entre tous les datasets
    """
    print(f"\n📈 COMPARAISON DÉTAILLÉE DES PATTERNS SLOPES")
    print("=" * 70)

    # Tableau de comparaison
    print(f"{'Dataset':<15} {'Vol.Trend':<10} {'Dur.Trend':<10} {'Interprétation':<30}")
    print("-" * 70)

    for name, data in slope_comparisons.items():
        vol_direction = "+" if data['vol_diff_pct'] > 0 else "-"
        dur_direction = "+" if data['dur_diff_pct'] > 0 else "-"

        # Classification du pattern
        if data['vol_diff_pct'] > 0 and data['dur_diff_pct'] < 0:
            pattern = "Volume↑ Durée↓ (Momentum)"
        elif data['vol_diff_pct'] < 0 and data['dur_diff_pct'] < 0:
            pattern = "Volume↓ Durée↓ (Spring)"
        elif data['vol_diff_pct'] > 0 and data['dur_diff_pct'] > 0:
            pattern = "Volume↑ Durée↑ (Épuisement)"
        elif data['vol_diff_pct'] < 0 and data['dur_diff_pct'] > 0:
            pattern = "Volume↓ Durée↑ (Consolidation)"
        else:
            pattern = "Pattern neutre"

        print(
            f"{name:<15} {vol_direction}{abs(data['vol_diff_pct']):>4.1f}%    {dur_direction}{abs(data['dur_diff_pct']):>4.1f}%    {pattern:<30}")

    # Recommandations par pattern
    print(f"\n💡 RECOMMANDATIONS PAR PATTERN:")
    print("-" * 40)

    for name, data in slope_comparisons.items():
        if data['vol_diff_pct'] > 0 and data['dur_diff_pct'] > 0:
            print(f"🔴 {name}: ÉVITER - Pattern d'épuisement détecté")
        elif data['vol_diff_pct'] < 0 and data['dur_diff_pct'] < 0:
            print(f"🟢 {name}: OPTIMAL - Pattern spring/compression")
        elif data['vol_diff_pct'] > 0 and data['dur_diff_pct'] < 0:
            print(f"🟡 {name}: BON - Pattern momentum classique")
        else:
            print(f"🔶 {name}: MOYEN - Pattern consolidation")


# FONCTION PRINCIPALE À AJOUTER DANS main()
def add_missing_slopes_analysis(results):
    """
    Ajoute l'analyse manquante des slopes
    À appeler après add_ultra_fast_pre_trade_analysis()
    """
    print(f"\n" + "=" * 80)
    print("🔍 ANALYSE SLOPES DÉTAILLÉE - VALIDATION HYPOTHÈSES")
    print("=" * 80)

    # Analyse wins vs losses slopes
    slope_comparisons = analyze_wins_vs_losses_slopes(results)

    # Validation hypothèse TEST
    validate_test_timing_hypothesis(slope_comparisons)

    # Comparaison détaillée
    compare_slope_patterns_detailed(slope_comparisons)

    return slope_comparisons


def compare_datasets_complete(results):
    """
    Comparaison complète entre tous les datasets
    """
    print(f"\n🔍 COMPARAISON COMPLÈTE DES CONDITIONS PRÉ-TRADE")
    print("=" * 70)

    if 'TEST' not in results:
        print("❌ TEST introuvable dans les résultats")
        return

    test_data = results['TEST']['overall']
    other_datasets = {k: v for k, v in results.items() if k != 'TEST'}

    print(f"\n📊 TEST vs AUTRES DATASETS:")
    print("-" * 50)

    # Métriques principales pour comparaison
    key_metrics = [
        ('duration_mean', 'Durée moyenne pré-trade', 's'),
        ('duration_std', 'Variabilité durée pré-trade', 's'),
        ('volume_mean', 'Volume/tick moyen pré-trade', ''),
        ('volume_volatility', 'Volatilité volume pré-trade', ''),
        ('fast_candles_pct', 'Bougies rapides pré-trade', '%'),
        ('slow_candles_pct', 'Bougies lentes pré-trade', '%'),
        ('duration_trend', 'Tendance durée pré-trade', 's/bougie'),
        ('duration_acceleration', 'Accélération durée pré-trade', 's')
    ]

    significant_differences = []

    for metric, description, unit in key_metrics:
        if metric not in test_data or 'mean' not in test_data[metric]:
            continue

        test_value = test_data[metric]['mean']

        # Calculer moyenne des autres datasets
        other_values = []
        for name, data in other_datasets.items():
            if metric in data['overall'] and 'mean' in data['overall'][metric]:
                other_values.append(data['overall'][metric]['mean'])

        if not other_values:
            continue

        others_mean = np.mean(other_values)
        others_std = np.std(other_values)

        # Calculer l'écart
        if others_mean != 0:
            difference_pct = (test_value - others_mean) / others_mean * 100
        else:
            difference_pct = 0

        # Déterminer la significativité
        is_very_significant = abs(difference_pct) > 25
        is_significant = abs(difference_pct) > 15
        is_notable = abs(difference_pct) > 10

        if is_very_significant:
            status = "🔴"
        elif is_significant:
            status = "🟡"
        elif is_notable:
            status = "🟠"
        else:
            status = "🟢"

        direction = "plus élevé" if difference_pct > 0 else "plus bas"

        print(f"{status} {description:30}: TEST {direction} de {abs(difference_pct):5.1f}%")
        print(f"    TEST: {test_value:.2f}{unit} | AUTRES: {others_mean:.2f}±{others_std:.2f}{unit}")

        if is_significant:
            significant_differences.append({
                'metric': description,
                'test_value': test_value,
                'others_mean': others_mean,
                'difference_pct': difference_pct,
                'unit': unit
            })

    # Synthèse des différences majeures
    if significant_differences:
        print(f"\n🚨 SYNTHÈSE DES DIFFÉRENCES MAJEURES:")
        print("-" * 50)

        for diff in significant_differences:
            print(f"\n📍 {diff['metric']}:")
            print(f"   Écart: {diff['difference_pct']:+.1f}%")

            # Interprétation business
            if 'durée' in diff['metric'].lower() and 'moyenne' in diff['metric'].lower():
                if diff['difference_pct'] > 0:
                    print(f"   → TEST trade après des séquences de bougies plus lentes")
                    print(f"   → Possible attente trop longue ou hésitation avant signal")
                else:
                    print(f"   → TEST trade après des séquences de bougies plus rapides")
                    print(f"   → Possible réaction impulsive ou trading dans la nervosité")

            elif 'rapides' in diff['metric'].lower():
                if diff['difference_pct'] > 0:
                    print(f"   → TEST trade plus souvent après des périodes agitées")
                    print(f"   → Risque de signaux dans le bruit de marché")
                else:
                    print(f"   → TEST trade plus souvent après des périodes calmes")
                    print(f"   → Meilleur timing potentiel")

            elif 'tendance' in diff['metric'].lower():
                if diff['difference_pct'] > 0:
                    print(f"   → TEST trade quand le marché s'accélère")
                    print(f"   → Possible entrée tardive dans le mouvement")
                else:
                    print(f"   → TEST trade quand le marché ralentit")
                    print(f"   → Possible anticipation ou retournement")

    else:
        print(f"\n✅ AUCUNE DIFFÉRENCE MAJEURE PRÉ-TRADE DÉTECTÉE")
        print("   → Les conditions d'entrée de TEST sont similaires aux autres")
        print("   → Le problème est probablement ailleurs:")
        print("     • Signaux ou algorithme de sélection")
        print("     • Timing d'exécution précis")
        print("     • Gestion des trades (stops/TP)")
        print("     • Facteurs post-entrée")


def analyze_complete_stats_fast(pre_trade_stats, dataset_name):
    """Version rapide de l'analyse des stats"""
    # Conversion ultra-rapide en array numpy
    data_array = []
    for stat in pre_trade_stats:
        row = [
            stat.get('duration_mean', 0),
            stat.get('volume_mean', 0),
            stat.get('duration_trend', 0),
            stat.get('volume_trend', 0),
            stat.get('trade_result', 0)
        ]
        data_array.append(row)

    data_np = np.array(data_array)

    # Séparation vectorisée wins/losses
    wins_mask = data_np[:, 4] == 1
    losses_mask = data_np[:, 4] == 0

    wins_data = data_np[wins_mask]
    losses_data = data_np[losses_mask]

    analysis = {
        'dataset': dataset_name,
        'total_trades': len(data_np),
        'wins_count': len(wins_data),
        'losses_count': len(losses_data),
        'winrate': len(wins_data) / len(data_np) * 100 if len(data_np) > 0 else 0,
        'raw_data': pd.DataFrame(pre_trade_stats)  # Conversion finale seulement
    }

    return analysis


def display_complete_results_fast(analysis, dataset_name):
    """Affichage rapide des résultats"""
    print(f"\n⚡ RÉSULTATS ULTRA-RAPIDES {dataset_name}:")
    print(f"   Trades: {analysis['total_trades']:,} | Winrate: {analysis['winrate']:.2f}%")


def calculate_pre_trade_slopes_debug(pre_candles):
    """
    Version DEBUG pour identifier pourquoi les slopes sont NaN
    """
    print(f"   🔍 DEBUG: Analyse des données pré-trade...")

    durations = pre_candles['sc_candleDuration'].values
    volumes = pre_candles['sc_volume_perTick'].values

    print(f"   🔍 Durées: {len(durations)} valeurs, min={np.nanmin(durations):.2f}, max={np.nanmax(durations):.2f}")
    print(f"   🔍 Volumes: {len(volumes)} valeurs, min={np.nanmin(volumes):.2f}, max={np.nanmax(volumes):.2f}")

    # Vérifier session_starts
    if 'sc_sessionStartEnd' in pre_candles.columns:
        session_starts = (pre_candles['sc_sessionStartEnd'] == 10).values
        print(f"   🔍 Session starts: {session_starts.sum()} sur {len(session_starts)} (valeurs True)")
    else:
        print(f"   ❌ Colonne 'sc_sessionStartEnd' manquante !")
        session_starts = np.zeros(len(durations), dtype=bool)
        session_starts[0] = True  # Forcer au moins un début de session
        print(f"   🔧 Session starts forcée: 1 début de session créé")

    try:
        # Tenter avec votre fonction
        print(f"   🔍 Appel calculate_slopes_and_r2_numba...")
        duration_slopes, r2_dur, std_dur = calculate_slopes_and_r2_numba(
            durations, session_starts, SLOPE_PERIODS,  clip_slope=False,
                                  include_close_bar=True)

        print(f"   🔍 Résultats durée: slopes={len(duration_slopes)}, r2={len(r2_dur)}, std={len(std_dur)}")
        if len(duration_slopes) > 0:
            print(f"   🔍 Première slope durée: {duration_slopes[0]}")
            print(f"   🔍 Dernière slope durée: {duration_slopes[-1]}")

        volume_slopes, r2_vol, std_vol = calculate_slopes_and_r2_numba(
            volumes, session_starts, SLOPE_PERIODS,  clip_slope=False,
                                  include_close_bar=True)


        print(f"   🔍 Résultats volume: slopes={len(volume_slopes)}, r2={len(r2_vol)}, std={len(std_vol)}")
        if len(volume_slopes) > 0:
            print(f"   🔍 Première slope volume: {volume_slopes[0]}")
            print(f"   🔍 Dernière slope volume: {volume_slopes[-1]}")

        # Retourner la dernière slope valide
        duration_slope = duration_slopes[-1] if len(duration_slopes) > 0 and not np.isnan(duration_slopes[-1]) else 0
        volume_slope = volume_slopes[-1] if len(volume_slopes) > 0 and not np.isnan(volume_slopes[-1]) else 0

        return {
            'duration_slope': duration_slope,
            'duration_r2': r2_dur[-1] if len(r2_dur) > 0 else 0,
            'duration_std': std_dur[-1] if len(std_dur) > 0 else 0,
            'volume_slope': volume_slope,
            'volume_r2': r2_vol[-1] if len(r2_vol) > 0 else 0,
            'volume_std': std_vol[-1] if len(std_vol) > 0 else 0
        }

    except Exception as e:
        print(f"   ❌ ERREUR calculate_slopes_and_r2_numba: {e}")

        # FALLBACK: Utiliser numpy pour debug
        print(f"   🔧 FALLBACK: Utilisation numpy.polyfit...")
        try:
            x = np.arange(len(durations))
            duration_slope = np.polyfit(x, durations, 1)[0] if len(durations) >= 3 else 0
            volume_slope = np.polyfit(x, volumes, 1)[0] if len(volumes) >= 3 else 0

            print(f"   🔧 Slope durée (numpy): {duration_slope}")
            print(f"   🔧 Slope volume (numpy): {volume_slope}")

            return {
                'duration_slope': duration_slope,
                'duration_r2': 0,
                'duration_std': 0,
                'volume_slope': volume_slope,
                'volume_r2': 0,
                'volume_std': 0
            }
        except Exception as e2:
            print(f"   ❌ ERREUR numpy aussi: {e2}")
            return {
                'duration_slope': 0, 'duration_r2': 0, 'duration_std': 0,
                'volume_slope': 0, 'volume_r2': 0, 'volume_std': 0
            }


@njit
def fast_stats_calculation(values):
    """Calcul ultra-rapide des statistiques de base avec numba"""
    if len(values) == 0:
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    mean_val = np.mean(values)
    median_val = np.median(values)
    std_val = np.std(values)
    min_val = np.min(values)
    max_val = np.max(values)
    q25_val = fast_percentile(values, 25)
    q75_val = fast_percentile(values, 75)

    return np.array([mean_val, median_val, std_val, min_val, max_val, q25_val, q75_val])
@njit
def fast_candle_classification(durations):
    """Classification rapide des types de bougies"""
    if len(durations) == 0:
        return np.array([0.0, 0.0, 0.0])

    fast_count = np.sum(durations < 10)
    slow_count = np.sum(durations > 300)
    medium_count = len(durations) - fast_count - slow_count

    total = len(durations)
    return np.array([
        fast_count / total * 100,
        slow_count / total * 100,
        medium_count / total * 100
    ])
def calculate_pre_trade_slopes_ultra_fast(pre_candles):
    """
    Version optimisée de la fonction slopes pour l'ultra-rapide
    """
    try:
        durations = pre_candles['sc_candleDuration'].values
        volumes = pre_candles['sc_volume_perTick'].values
        session_starts = (pre_candles['sc_sessionStartEnd'] == 10).values

        # Appel direct à votre fonction (pas d'optimisation possible ici)
        duration_slopes, r2_dur, std_dur = calculate_slopes_and_r2_numba(
            durations, session_starts, SLOPE_PERIODS,clip_slope=False,
                                  include_close_bar=True)
        volume_slopes, r2_vol, std_vol = calculate_slopes_and_r2_numba(volumes, session_starts, SLOPE_PERIODS,  clip_slope=False,
include_close_bar=True)

        # Retour optimisé (accès direct)
        return {
            'duration_slope': duration_slopes[-1] if len(duration_slopes) > 0 else 0,
            'duration_r2': r2_dur[-1] if len(r2_dur) > 0 else 0,
            'duration_std': std_dur[-1] if len(std_dur) > 0 else 0,
            'volume_slope': volume_slopes[-1] if len(volume_slopes) > 0 else 0,
            'volume_r2': r2_vol[-1] if len(r2_vol) > 0 else 0,
            'volume_std': std_vol[-1] if len(std_vol) > 0 else 0
        }
    except:
        return {
            'duration_slope': 0, 'duration_r2': 0, 'duration_std': 0,
            'volume_slope': 0, 'volume_r2': 0, 'volume_std': 0
        }




# VERSION ULTRA-RAPIDE DE LA FONCTION PRINCIPALE


# REMPLACEZ ENTIÈREMENT votre fonction analyze_pre_trade_conditions_ultra_fast par cette version :

# NOUVELLE FONCTION : Métriques avec votre logique adaptée
def calculate_pre_trade_metrics_complete_debug_with_adaptation(pre_candles):
    """
    Version finale utilisant votre logique de slopes adaptée
    """
    try:
        required_cols = ['sc_candleDuration', 'sc_volume_perTick']
        if not all(col in pre_candles.columns for col in required_cols):
            return None

        durations = pre_candles['sc_candleDuration'].values
        volumes = pre_candles['sc_volume_perTick'].values

        if len(durations) == 0 or len(volumes) == 0:
            return None
        if np.all(np.isnan(durations)) or np.all(np.isnan(volumes)):
            return None

        # Calculs de base (rapides)
        stats = {
            'duration_mean': np.nanmean(durations),
            'duration_median': np.nanmedian(durations),
            'duration_std': np.nanstd(durations),
            'volume_mean': np.nanmean(volumes),
            'volume_median': np.nanmedian(volumes),
            'volume_std': np.nanstd(volumes),
            'fast_candles_pct': np.mean(durations < 10) * 100,
            'slow_candles_pct': np.mean(durations > 300) * 100,
        }

        # Coefficients de variation
        stats['duration_cv'] = stats['duration_std'] / stats['duration_mean'] if stats['duration_mean'] > 0 else 0
        stats['volume_cv'] = stats['volume_std'] / stats['volume_mean'] if stats['volume_mean'] > 0 else 0

        # SLOPES avec votre logique adaptée
        if len(durations) >= 3:
            # DEBUG: Afficher uniquement pour les premiers trades
            debug_mode = len(durations) == SLOPE_PERIODS  # Debug seulement si taille exacte
            slope_results = calculate_pre_trade_slopes_CLEAN(pre_candles, debug_mode)

            stats['duration_trend'] = slope_results['duration_slope']
            stats['duration_slope_stdev'] = slope_results['duration_std']
            stats['duration_slope_r2'] = slope_results['duration_r2']
            stats['volume_trend'] = slope_results['volume_slope']
            stats['volume_slope_stdev'] = slope_results['volume_std']
            stats['volume_slope_r2'] = slope_results['volume_r2']
        else:
            return None

        # Métriques supplémentaires
        stats['duration_volatility'] = stats['duration_cv']
        stats['volume_volatility'] = stats['volume_cv']

        if len(durations) >= 5:
            mid = len(durations) // 2
            stats['duration_acceleration'] = np.mean(durations[mid:]) - np.mean(durations[:mid])
            stats['volume_acceleration'] = np.mean(volumes[mid:]) - np.mean(volumes[:mid])
        else:
            stats['duration_acceleration'] = 0
            stats['volume_acceleration'] = 0

        return stats

    except Exception as e:
        if debug_mode:
            print(f"   ❌ Erreur calculate_pre_trade_metrics: {e}")
        return None

def analyze_pre_trade_conditions_ultra_fast(datasets_dict, n_candles_before=10):
    """
    Version ULTRA-RAPIDE de l'analyse complète - CORRIGÉE
    """
    print(f"\n🚀 ANALYSE ULTRA-RAPIDE DES {n_candles_before} BOUGIES PRÉ-TRADE")
    print("=" * 80)

    results = {}

    for name, (df_complete, df_filtered) in datasets_dict.items():
        if df_complete.empty or df_filtered.empty:
            continue

        print(f"\n⚡ DATASET: {name}")
        print("-" * 50)
        print(f"Mode ultra-rapide: {len(df_filtered):,} trades sur {len(df_complete):,} bougies...")

        # Optimisation 1: Index pré-calculé ultra-rapide
        df_complete_reset = df_complete.reset_index(drop=True)
        df_complete_sorted = df_complete_reset.sort_values(['session_id', 'date']).reset_index(drop=True)

        # Optimisation 2: Groupby optimisé
        session_groups = df_complete_sorted.groupby('session_id', sort=False)
        session_indices = {}

        for session_id, group in session_groups:
            session_indices[session_id] = group.reset_index(drop=True)

        print(f"   🔥 {len(session_indices)} sessions indexées (ultra-rapide)")

        # Optimisation 3: Traitement ultra-vectorisé CORRIGÉ
        pre_trade_stats = []
        total_trades = len(df_filtered)
        processed = 0

        trades_by_session = df_filtered.groupby('session_id', sort=False)

        for session_id, session_trades in trades_by_session:
            if session_id not in session_indices:
                continue

            session_data = session_indices[session_id]

            if len(session_data) < n_candles_before + 1:
                continue

            # CORRECTION: Utiliser la bonne fonction avec les bons arguments
            session_stats = process_session_trades_CLEAN(
                session_trades, session_data, n_candles_before
            )

            pre_trade_stats.extend(session_stats)
            processed += len(session_trades)

            # Affichage optimisé (moins fréquent)
            if processed % 1000 == 0 or processed == total_trades:
                pct = (processed / total_trades) * 100
                print(f"   ⚡ Progression: {processed:,}/{total_trades:,} ({pct:.1f}%)")

        print(f"   ✅ ULTRA-RAPIDE: {len(pre_trade_stats):,} trades analysés")

        if pre_trade_stats:
            # Analyse rapide des résultats
            analysis = analyze_complete_stats_fast(pre_trade_stats, name)
            results[name] = analysis
            display_complete_results_fast(analysis, name)

    return results



# AJOUTEZ CETTE FONCTION COMPLÈTE D'ANALYSE DES PATTERNS :

def analyze_spring_momentum_exhaustion_patterns(analysis_results):
    """
    Analyse complète des patterns Spring, Momentum et Épuisement avec filtrage
    """
    print(f"\n" + "=" * 80)
    print("🔍 ANALYSE COMPLÈTE DES PATTERNS - SPRING vs MOMENTUM vs ÉPUISEMENT")
    print("=" * 80)

    # Configuration des patterns
    pattern_config = {
        'spring': {
            'name': 'Spring Pattern',
            'description': 'Volume↓ + Duration↓ (Accumulation discrète avant explosion)',
            'volume_condition': lambda vol_trend: vol_trend < -0.005,
            'duration_condition': lambda dur_trend: dur_trend < -0.005,
            'expected_winrate': 'High (52-54%)',
            'interpretation': 'Optimal timing - Early entry before momentum',
            'recommendation': '🟢 TRADE - Best conditions'
        },
        'momentum': {
            'name': 'Momentum Pattern',
            'description': 'Volume↑ + Duration↓ (Momentum with acceleration)',
            'volume_condition': lambda vol_trend: vol_trend > 0.005,
            'duration_condition': lambda dur_trend: dur_trend < -0.005,
            'expected_winrate': 'Good (51-53%)',
            'interpretation': 'Good timing - Entry during acceleration',
            'recommendation': '🟡 TRADE - Good conditions'
        },
        'spring_momentum': {
            'name': 'Spring + Momentum Combined',
            'description': 'Either Spring OR Momentum conditions',
            'volume_condition': lambda vol_trend: True,  # Will be handled in logic
            'duration_condition': lambda dur_trend: True,  # Will be handled in logic
            'expected_winrate': 'Good (51-53%)',
            'interpretation': 'Combined optimal conditions',
            'recommendation': '🟡 TRADE - Good conditions combined'
        },
        'exhaustion': {
            'name': 'Exhaustion Pattern',
            'description': 'Volume↑ + Duration↑ (Late entry - movement exhaustion)',
            'volume_condition': lambda vol_trend: vol_trend > 0.005,
            'duration_condition': lambda dur_trend: dur_trend > 0.005,
            'expected_winrate': 'Poor (49-51%)',
            'interpretation': 'Late timing - Entry on exhaustion',
            'recommendation': '🔴 AVOID - Poor conditions'
        },
        'all_trades': {
            'name': 'All Trades (No Filter)',
            'description': 'Original performance without any filtering',
            'volume_condition': lambda vol_trend: True,
            'duration_condition': lambda dur_trend: True,
            'expected_winrate': 'Baseline',
            'interpretation': 'Original algorithm performance',
            'recommendation': '⚪ BASELINE'
        }
    }

    results_summary = {}

    for dataset_name, data in analysis_results.items():
        if 'raw_data' not in data or data['raw_data'].empty:
            continue

        df = data['raw_data']
        original_trades = len(df)
        original_winrate = data['winrate']

        print(f"\n📊 DATASET: {dataset_name}")
        print("=" * 50)
        print(f"Original Performance: {original_trades:,} trades, {original_winrate:.2f}% winrate")
        print()

        dataset_results = {
            'original_trades': original_trades,
            'original_winrate': original_winrate,
            'patterns': {}
        }

        # Analyser chaque pattern
        for pattern_name, pattern_config_item in pattern_config.items():
            if pattern_name == 'spring_momentum':
                # Logique combinée Spring + Momentum
                spring_mask = (df['volume_trend'] < -0.005) & (df['duration_trend'] < -0.005)
                momentum_mask = (df['volume_trend'] > 0.005) & (df['duration_trend'] < -0.005)
                pattern_mask = spring_mask | momentum_mask
            elif pattern_name == 'all_trades':
                # Tous les trades
                pattern_mask = pd.Series([True] * len(df), index=df.index)
            else:
                # Patterns individuels
                vol_mask = pattern_config_item['volume_condition'](df['volume_trend'])
                dur_mask = pattern_config_item['duration_condition'](df['duration_trend'])
                pattern_mask = vol_mask & dur_mask

            # Filtrer les données
            filtered_df = df[pattern_mask]
            filtered_trades = len(filtered_df)

            if filtered_trades > 0:
                filtered_wins = (filtered_df['trade_result'] == 1).sum()
                filtered_winrate = (filtered_wins / filtered_trades) * 100
                trades_retained_pct = (filtered_trades / original_trades) * 100
                winrate_improvement = filtered_winrate - original_winrate
            else:
                filtered_winrate = 0
                trades_retained_pct = 0
                winrate_improvement = 0

            # Stocker les résultats
            pattern_results = {
                'trades_count': filtered_trades,
                'winrate': filtered_winrate,
                'trades_retained_pct': trades_retained_pct,
                'winrate_improvement': winrate_improvement,
                'config': pattern_config_item
            }

            dataset_results['patterns'][pattern_name] = pattern_results

            # Affichage détaillé
            if pattern_name != 'all_trades':  # Skip baseline in detailed display
                status_emoji = "🟢" if winrate_improvement > 1 else "🟡" if winrate_improvement > -0.5 else "🔴"
                print(f"{status_emoji} {pattern_config_item['name']:25}")
                print(f"   Trades: {filtered_trades:6,} ({trades_retained_pct:5.1f}% retained)")
                print(f"   Winrate: {filtered_winrate:5.2f}% ({winrate_improvement:+5.2f}% vs original)")
                print(f"   {pattern_config_item['recommendation']}")
                print()

        results_summary[dataset_name] = dataset_results

    # TABLEAU RÉCAPITULATIF GLOBAL
    print(f"\n📈 TABLEAU RÉCAPITULATIF - PERFORMANCE PAR PATTERN")
    print("=" * 120)

    # En-tête du tableau
    header = f"{'Dataset':<12} {'Pattern':<20} {'Trades':<8} {'%Kept':<6} {'WR%':<6} {'Δ WR':<6} {'Status':<15} {'Recommendation':<20}"
    print(header)
    print("-" * 120)

    # Données pour chaque dataset et pattern
    for dataset_name, dataset_data in results_summary.items():
        first_row = True

        # Ordre des patterns pour affichage
        pattern_order = ['all_trades', 'spring', 'momentum', 'spring_momentum', 'exhaustion']

        for pattern_name in pattern_order:
            if pattern_name in dataset_data['patterns']:
                pattern_data = dataset_data['patterns'][pattern_name]

                # Nom du dataset seulement sur la première ligne
                dataset_display = dataset_name if first_row else ""
                first_row = False

                # Formatage des données
                trades_display = f"{pattern_data['trades_count']:,}"
                retained_display = f"{pattern_data['trades_retained_pct']:.1f}%"
                winrate_display = f"{pattern_data['winrate']:.2f}"
                improvement_display = f"{pattern_data['winrate_improvement']:+.2f}"

                # Status basé sur l'amélioration
                if pattern_name == 'all_trades':
                    status = "BASELINE"
                elif pattern_data['winrate_improvement'] > 2:
                    status = "EXCELLENT"
                elif pattern_data['winrate_improvement'] > 1:
                    status = "VERY GOOD"
                elif pattern_data['winrate_improvement'] > 0:
                    status = "GOOD"
                elif pattern_data['winrate_improvement'] > -1:
                    status = "NEUTRAL"
                else:
                    status = "POOR"

                # Recommandation simplifiée
                if pattern_name == 'all_trades':
                    recommendation = "BASELINE"
                elif pattern_data['winrate_improvement'] > 1:
                    recommendation = "🟢 TRADE"
                elif pattern_data['winrate_improvement'] > -0.5:
                    recommendation = "🟡 CONSIDER"
                else:
                    recommendation = "🔴 AVOID"

                # Pattern name display
                pattern_display = pattern_data['config']['name']

                print(
                    f"{dataset_display:<12} {pattern_display:<20} {trades_display:<8} {retained_display:<6} {winrate_display:<6} {improvement_display:<6} {status:<15} {recommendation:<20}")

        print("-" * 120)  # Separator between datasets

    # ANALYSE CROSS-DATASET DES PATTERNS
    print(f"\n🔍 ANALYSE CROSS-DATASET DES PATTERNS")
    print("=" * 80)

    pattern_summary = {}
    for pattern_name in ['spring', 'momentum', 'spring_momentum', 'exhaustion']:
        pattern_performances = []
        pattern_retentions = []

        for dataset_name, dataset_data in results_summary.items():
            if pattern_name in dataset_data['patterns']:
                pattern_data = dataset_data['patterns'][pattern_name]
                pattern_performances.append(pattern_data['winrate_improvement'])
                pattern_retentions.append(pattern_data['trades_retained_pct'])

        if pattern_performances:
            avg_improvement = sum(pattern_performances) / len(pattern_performances)
            avg_retention = sum(pattern_retentions) / len(pattern_retentions)

            pattern_summary[pattern_name] = {
                'avg_improvement': avg_improvement,
                'avg_retention': avg_retention,
                'datasets_count': len(pattern_performances)
            }

    print(f"{'Pattern':<20} {'Avg ΔWR':<8} {'Avg %Kept':<10} {'Overall Rating':<15}")
    print("-" * 60)

    # Trier par amélioration moyenne
    sorted_patterns = sorted(pattern_summary.items(), key=lambda x: x[1]['avg_improvement'], reverse=True)

    for pattern_name, summary in sorted_patterns:
        config = pattern_config[pattern_name]

        # Rating global
        if summary['avg_improvement'] > 2:
            rating = "🟢 EXCELLENT"
        elif summary['avg_improvement'] > 1:
            rating = "🟢 VERY GOOD"
        elif summary['avg_improvement'] > 0:
            rating = "🟡 GOOD"
        elif summary['avg_improvement'] > -1:
            rating = "🟡 NEUTRAL"
        else:
            rating = "🔴 POOR"

        print(
            f"{config['name']:<20} {summary['avg_improvement']:+6.2f}% {summary['avg_retention']:7.1f}%   {rating:<15}")

    # RECOMMANDATIONS FINALES
    print(f"\n💡 RECOMMANDATIONS STRATÉGIQUES FINALES")
    print("=" * 80)

    best_pattern = sorted_patterns[0]
    worst_pattern = sorted_patterns[-1]

    print(f"🏆 MEILLEUR PATTERN: {pattern_config[best_pattern[0]]['name']}")
    print(f"   Amélioration moyenne: {best_pattern[1]['avg_improvement']:+.2f}%")
    print(f"   Rétention moyenne: {best_pattern[1]['avg_retention']:.1f}%")
    print(f"   {pattern_config[best_pattern[0]]['recommendation']}")

    print(f"\n💀 PIRE PATTERN: {pattern_config[worst_pattern[0]]['name']}")
    print(f"   Amélioration moyenne: {worst_pattern[1]['avg_improvement']:+.2f}%")
    print(f"   Rétention moyenne: {worst_pattern[1]['avg_retention']:.1f}%")
    print(f"   {pattern_config[worst_pattern[0]]['recommendation']}")

    # Recommandations par dataset
    print(f"\n📊 RECOMMANDATIONS PAR DATASET:")
    for dataset_name, dataset_data in results_summary.items():
        best_pattern_for_dataset = None
        best_improvement = -999

        for pattern_name, pattern_data in dataset_data['patterns'].items():
            if pattern_name != 'all_trades' and pattern_data['winrate_improvement'] > best_improvement:
                best_improvement = pattern_data['winrate_improvement']
                best_pattern_for_dataset = pattern_name

        if best_pattern_for_dataset:
            config = pattern_config[best_pattern_for_dataset]
            pattern_data = dataset_data['patterns'][best_pattern_for_dataset]

            print(f"\n🎯 {dataset_name}:")
            print(f"   Meilleur pattern: {config['name']}")
            print(
                f"   Impact: {pattern_data['winrate_improvement']:+.2f}% winrate, {pattern_data['trades_retained_pct']:.1f}% trades kept")

            if pattern_data['winrate_improvement'] > 1:
                print(f"   ✅ RECOMMANDATION: Implémenter ce filtre")
            elif pattern_data['winrate_improvement'] > 0:
                print(f"   🟡 RECOMMANDATION: Tester en condition réelle")
            else:
                print(f"   ❌ RECOMMANDATION: Garder la configuration actuelle")

    return results_summary


# AJOUTEZ CETTE FONCTION DANS LA FONCTION main() APRÈS add_missing_slopes_analysis :
# ===== SUPPRESSION DES DOUBLONS =====
# Gardez seulement UNE SEULE version de fast_candle_classification

@njit
def fast_candle_classification(durations):
    """Classification rapide des types de bougies"""
    if len(durations) == 0:
        return np.array([0.0, 0.0, 0.0])

    fast_count = np.sum(durations < 10)
    slow_count = np.sum(durations > 300)
    medium_count = len(durations) - fast_count - slow_count

    total = len(durations)
    return np.array([
        fast_count / total * 100,
        slow_count / total * 100,
        medium_count / total * 100
    ])


# ===== VERSION SANS FALLBACK AVEC AFFICHAGE SLOPES =====

def calculate_pre_trade_slopes_DISPLAY(pre_candles, debug_mode=True):
    """
    Version qui AFFICHE les slopes calculées par calculate_slopes_and_r2_numba
    SANS AUCUN FALLBACK
    """
    try:
        durations = pre_candles['sc_candleDuration'].values
        volumes = pre_candles['sc_volume_perTick'].values

        if len(durations) < 3 or len(volumes) < 3:
            if debug_mode:
                print(f"   ❌ Pas assez de données: durées={len(durations)}, volumes={len(volumes)}")
            return {
                'duration_slope': 0, 'duration_r2': 0, 'duration_std': 0,
                'volume_slope': 0, 'volume_r2': 0, 'volume_std': 0
            }

        # Créer session_starts
        session_starts_dur = np.zeros(len(durations), dtype=bool)
        session_starts_vol = np.zeros(len(volumes), dtype=bool)
        session_starts_dur[0] = True
        session_starts_vol[0] = True

        if debug_mode:
            print(f"   🔧 INPUT pour calculate_slopes_and_r2_numba:")
            print(f"      Durées: {len(durations)} valeurs [{np.min(durations):.1f}, {np.max(durations):.1f}]")
            print(f"      Volumes: {len(volumes)} valeurs [{np.min(volumes):.1f}, {np.max(volumes):.1f}]")
            print(f"      SLOPE_PERIODS: {SLOPE_PERIODS}")
            print(f"      clip_slope: False")

        # ✅ APPEL DIRECT À VOTRE FONCTION - AUCUN FALLBACK
        duration_slopes, r2_dur, std_dur = calculate_slopes_and_r2_numba(
            durations, session_starts_dur, SLOPE_PERIODS,
            clip_slope=False,  # ← SANS CLIPPING
            include_close_bar=True
        )

        volume_slopes, r2_vol, std_vol = calculate_slopes_and_r2_numba(
            volumes, session_starts_vol, SLOPE_PERIODS,
            clip_slope=False,  # ← SANS CLIPPING
            include_close_bar=True
        )

        if debug_mode:
            print(f"   📊 RÉSULTATS calculate_slopes_and_r2_numba:")
            print(f"      Duration slopes: array de {len(duration_slopes)} valeurs")
            if len(duration_slopes) > 0:
                print(f"         Première: {duration_slopes[0]:.6f}")
                print(f"         Dernière: {duration_slopes[-1]:.6f}")
                print(f"         NaN count: {np.sum(np.isnan(duration_slopes))}")

            print(f"      Volume slopes: array de {len(volume_slopes)} valeurs")
            if len(volume_slopes) > 0:
                print(f"         Première: {volume_slopes[0]:.6f}")
                print(f"         Dernière: {volume_slopes[-1]:.6f}")
                print(f"         NaN count: {np.sum(np.isnan(volume_slopes))}")

        # Extraction des dernières valeurs valides
        dur_slope = 0
        vol_slope = 0
        dur_r2 = 0
        vol_r2 = 0
        dur_std = 0
        vol_std = 0

        if len(duration_slopes) > 0:
            valid_dur_mask = ~np.isnan(duration_slopes)
            if np.any(valid_dur_mask):
                dur_slope = duration_slopes[valid_dur_mask][-1]
                dur_r2 = r2_dur[valid_dur_mask][-1] if len(r2_dur) > 0 else 0
                dur_std = std_dur[valid_dur_mask][-1] if len(std_dur) > 0 else 0

        if len(volume_slopes) > 0:
            valid_vol_mask = ~np.isnan(volume_slopes)
            if np.any(valid_vol_mask):
                vol_slope = volume_slopes[valid_vol_mask][-1]
                vol_r2 = r2_vol[valid_vol_mask][-1] if len(r2_vol) > 0 else 0
                vol_std = std_vol[valid_vol_mask][-1] if len(std_vol) > 0 else 0

        if debug_mode:
            print(f"   ✅ SLOPES FINALES EXTRAITES:")
            print(f"      Duration slope: {dur_slope:.6f} (R²={dur_r2:.3f}, Std={dur_std:.3f})")
            print(f"      Volume slope: {vol_slope:.6f} (R²={vol_r2:.3f}, Std={vol_std:.3f})")

        return {
            'duration_slope': dur_slope,
            'duration_r2': dur_r2,
            'duration_std': dur_std,
            'volume_slope': vol_slope,
            'volume_r2': vol_r2,
            'volume_std': vol_std
        }

    except Exception as e:
        if debug_mode:
            print(f"   ❌ ERREUR FATALE calculate_slopes_and_r2_numba: {e}")
            print(f"   ❌ AUCUN FALLBACK - RETOUR DE ZÉROS")

        # ❌ AUCUN FALLBACK - RETOUR ZÉROS SEULEMENT
        return {
            'duration_slope': 0, 'duration_r2': 0, 'duration_std': 0,
            'volume_slope': 0, 'volume_r2': 0, 'volume_std': 0
        }


# Stratégie de filtrage pour améliorer TEST sans dégrader les autres datasets
# Stratégie de filtrage pour améliorer TEST sans dégrader les autres datasets

def apply_anti_spring_universal_filter(
        df,
        volume_slope_col='volume_slope',
        duration_slope_col='duration_slope'):

    # --- auto-détection : si le nom par défaut est absent,
    #     on bascule sur la nouvelle convention -------------
    if volume_slope_col not in df.columns and 'volume_trend' in df.columns:
        volume_slope_col = 'volume_trend'
    if duration_slope_col not in df.columns and 'duration_trend' in df.columns:
        duration_slope_col = 'duration_trend'

    # si les colonnes manquent toujours → on ne filtre pas
    if (volume_slope_col not in df.columns) or (duration_slope_col not in df.columns):
        print("⚠️  Colonnes slope introuvables – filtre ignoré")
        return df.copy()

    # -------- filtre Anti-Spring --------
    spring_mask      = (df[volume_slope_col] < 0) & (df[duration_slope_col] < 0)
    anti_spring_mask = ~spring_mask
    return df[anti_spring_mask]

def analyze_anti_spring_impact(datasets, label_col='class_binaire'):
    """
    datasets :  • dict  {name: df}
                • ou liste de tuples (df, name)   **ou**   (name, df)
    """
    import pandas as pd
    results = {}

    # 1) normaliser l’itération
    items = datasets.items() if isinstance(datasets, dict) else datasets

    for pair in items:
        # — déballer intelligemment —
        if isinstance(pair, tuple) and len(pair) == 2:
            # Cas (df, name)  ou  (name, df)
            if isinstance(pair[0], pd.DataFrame):
                df, dataset_name = pair            # (df, name)
            else:
                dataset_name, df = pair            # (name, df)
        else:
            raise ValueError("Chaque élément doit être un tuple (df, name) ou (name, df)")

        # 2) filtrer 0 / 1
        valid_mask = df[label_col].isin([0, 1])
        df_valid   = df[valid_mask]

        orig_trades = len(df_valid)
        orig_wr     = df_valid[label_col].mean() if orig_trades else 0

        # 3) appliquer le filtre anti-Spring
        df_filtered = apply_anti_spring_universal_filter(df_valid)
        filt_trades = len(df_filtered)
        filt_wr     = df_filtered[label_col].mean() if filt_trades else 0

        results[dataset_name] = {
            'original_trades':  orig_trades,
            'filtered_trades':  filt_trades,
            'retention_rate':   filt_trades / orig_trades * 100 if orig_trades else 0,
            'original_winrate': orig_wr,
            'filtered_winrate': filt_wr,
            'winrate_improvement': filt_wr - orig_wr
        }

        print(f"📊 {dataset_name}:"
              f"  Trades {orig_trades:,} → {filt_trades:,}"
              f"  ({results[dataset_name]['retention_rate']:.1f}% retenus) |"
              f"  WR {orig_wr:5.2%} → {filt_wr:5.2%}"
              f"  ({results[dataset_name]['winrate_improvement']:+.2%})")

    return results


def adaptive_pattern_filter(df, dataset_type,
                            volume_slope_col='volume_slope',
                            duration_slope_col='duration_slope'):

    # --- si les slopes n’existent pas on ne filtre pas ---
    if volume_slope_col not in df.columns or duration_slope_col not in df.columns:
        print(f"⚠️  {dataset_type}: colonnes de slope absentes → aucun filtre appliqué")
        exit(45)
        return df.copy()

    # (logique inchangée ensuite)
    if dataset_type == "TEST":
        condition = ~((df[volume_slope_col] < 0) & (df[duration_slope_col] < 0))
    elif dataset_type == "TRAIN":
        condition = df[volume_slope_col] > 0
    elif dataset_type == "VALIDATION":
        condition = ~((df[volume_slope_col] > 0) & (df[duration_slope_col] < -0.5))
    else:  # VALIDATION 1, UNSEEN, etc.
        condition = ~((df[volume_slope_col] < 0) & (df[duration_slope_col] < 0)) \
                    & (df[volume_slope_col] > -0.1)

    filtered_df = df[condition]
    print(f"🎯 {dataset_type}: {len(filtered_df)}/{len(df)} trades conservés "
          f"({len(filtered_df)/len(df)*100:.1f} %)")
    return filtered_df



# Fonction principale pour tester la stratégie
def test_universal_anti_spring_strategy(datasets_dict,
                                        label_col='class_binaire'):
    """
    Test la stratégie anti-Spring universelle + filtre adaptatif
    """
    print("🔍 TEST STRATÉGIE ANTI-SPRING UNIVERSELLE")
    print("=" * 60)

    # --- 1) normaliser l’itération -----------------------------
    if isinstance(datasets_dict, dict):
        items = datasets_dict.items()          # {'TRAIN': df, ...}
    else:                                      # [(df, name) ou (name, df)]
        items = [(name, df) if isinstance(pair[0], str) else (pair[1], pair[0])
                 for pair in datasets_dict]

    # --- 2) stratégie 1 : anti-Spring universel ----------------
    print("\n📊 STRATÉGIE 1 : Anti-Spring universel")
    universal_results = analyze_anti_spring_impact(datasets_dict,
                                                   label_col=label_col)

    # --- 3) stratégie 2 : filtre adaptatif ---------------------
    print("\n📊 STRATÉGIE 2 : Filtre adaptatif par dataset")
    adaptive_results = {}

    for dataset_name, df in items:
        if label_col not in df.columns:
            raise KeyError(f"Colonne '{label_col}' absente de {dataset_name}")

        original_wr   = df[label_col].mean()
        filtered_df   = adaptive_pattern_filter(df, dataset_name)
        filtered_wr   = filtered_df[label_col].mean() if len(filtered_df) else 0

        adaptive_results[dataset_name] = {
            'original_winrate' : original_wr,
            'filtered_winrate' : filtered_wr,
            'improvement'      : filtered_wr - original_wr
        }

    # --- 4) comparaison ---------------------------------------
    print("\n📈 COMPARAISON DES STRATÉGIES")
    print("=" * 60)
    for dataset_name in adaptive_results:
        univ_imp  = universal_results[dataset_name]['winrate_improvement']
        adapt_imp = adaptive_results[dataset_name]['improvement']
        better    = "Universelle" if univ_imp > adapt_imp else "Adaptative"

        print(f"{dataset_name}:  anti-Spring {univ_imp:+.2%} | "
              f"adaptatif {adapt_imp:+.2%}  →  🏆 {better}")

    return universal_results, adaptive_results




def calculate_pre_trade_metrics_DISPLAY(pre_candles):
    """
    Version qui AFFICHE les métriques et utilise calculate_slopes_and_r2_numba
    SANS FALLBACK
    """
    try:
        required_cols = ['sc_candleDuration', 'sc_volume_perTick']
        if not all(col in pre_candles.columns for col in required_cols):
            print(f"   ❌ Colonnes manquantes: {[col for col in required_cols if col not in pre_candles.columns]}")
            return None

        durations = pre_candles['sc_candleDuration'].values
        volumes = pre_candles['sc_volume_perTick'].values

        if len(durations) == 0 or len(volumes) == 0:
            print(f"   ❌ Arrays vides: dur={len(durations)}, vol={len(volumes)}")
            return None
        if np.all(np.isnan(durations)) or np.all(np.isnan(volumes)):
            print(f"   ❌ Toutes les valeurs sont NaN")
            return None

        print(f"   📊 Calcul métriques pré-trade:")
        print(f"      Input: {len(durations)} durées, {len(volumes)} volumes")

        # Calculs de base
        stats = {
            'duration_mean': np.nanmean(durations),
            'duration_median': np.nanmedian(durations),
            'duration_std': np.nanstd(durations),
            'volume_mean': np.nanmean(volumes),
            'volume_median': np.nanmedian(volumes),
            'volume_std': np.nanstd(volumes),
            'fast_candles_pct': np.mean(durations < 10) * 100,
            'slow_candles_pct': np.mean(durations > 300) * 100,
        }

        # Coefficients de variation
        stats['duration_cv'] = stats['duration_std'] / stats['duration_mean'] if stats['duration_mean'] > 0 else 0
        stats['volume_cv'] = stats['volume_std'] / stats['volume_mean'] if stats['volume_mean'] > 0 else 0

        print(f"      Stats de base calculées ✓")

        # ✅ SLOPES AVEC AFFICHAGE - SANS FALLBACK
        if len(durations) >= 3:
            print(f"   🔧 Calcul des slopes avec votre fonction:")
            slope_results = calculate_pre_trade_slopes_DISPLAY(pre_candles, debug_mode=True)

            stats['duration_trend'] = slope_results['duration_slope']
            stats['duration_slope_stdev'] = slope_results['duration_std']
            stats['duration_slope_r2'] = slope_results['duration_r2']
            stats['volume_trend'] = slope_results['volume_slope']
            stats['volume_slope_stdev'] = slope_results['volume_std']
            stats['volume_slope_r2'] = slope_results['volume_r2']

            print(f"   ✅ Slopes intégrées dans stats")
        else:
            print(f"   ❌ Pas assez de données pour les slopes ({len(durations)} < 3)")
            return None

        # Métriques supplémentaires
        stats['duration_volatility'] = stats['duration_cv']
        stats['volume_volatility'] = stats['volume_cv']

        if len(durations) >= 5:
            mid = len(durations) // 2
            stats['duration_acceleration'] = np.mean(durations[mid:]) - np.mean(durations[:mid])
            stats['volume_acceleration'] = np.mean(volumes[mid:]) - np.mean(volumes[:mid])
        else:
            stats['duration_acceleration'] = 0
            stats['volume_acceleration'] = 0

        print(f"   ✅ Métriques complètes calculées")
        return stats

    except Exception as e:
        print(f"   ❌ ERREUR calculate_pre_trade_metrics_DISPLAY: {e}")
        return None


def process_session_trades_DISPLAY(session_trades, session_data, n_candles_before):
    """
    Version avec affichage pour le traitement de session
    """
    session_stats = []
    print(f"   📊 Traitement session: {len(session_trades)} trades")

    session_dates_np = session_data['date'].values.astype('datetime64[ns]')

    for i, (_, trade_row) in enumerate(session_trades.iterrows()):
        if i == 0:  # Afficher seulement pour le premier trade
            print(f"   🔍 Exemple trade #{i + 1}:")

        trade_date_np = np.datetime64(trade_row['date'])

        mask = session_dates_np <= trade_date_np
        indices = np.where(mask)[0]

        if len(indices) < n_candles_before + 1:
            if i == 0:
                print(f"      ❌ Pas assez de bougies: {len(indices)} < {n_candles_before + 1}")
            continue

        trade_position = indices[-1]

        if trade_position < n_candles_before:
            if i == 0:
                print(f"      ❌ Position trop proche du début: {trade_position} < {n_candles_before}")
            continue

        start_pos = trade_position - n_candles_before
        end_pos = trade_position

        pre_candles = session_data.iloc[start_pos:end_pos]

        if len(pre_candles) == n_candles_before:
            if i == 0:
                print(f"      ✅ Extraction pré-trade: {len(pre_candles)} bougies")
                print(f"         Période: {pre_candles['date'].iloc[0]} à {pre_candles['date'].iloc[-1]}")

            # ✅ UTILISE LA VERSION AVEC AFFICHAGE
            stats = calculate_pre_trade_metrics_DISPLAY(pre_candles) if i == 0 else calculate_pre_trade_metrics_CLEAN(
                pre_candles)

            if stats:
                stats.update({
                    'trade_result': trade_row['class_binaire'],
                    'session_id': trade_row['session_id'],
                    'trade_date': trade_row['date']
                })
                session_stats.append(stats)

                if i == 0:
                    print(f"      ✅ Stats ajoutées pour trade #{i + 1}")

    print(f"   ✅ Session terminée: {len(session_stats)} trades analysés")
    return session_stats


def analyze_pre_trade_conditions_DISPLAY(datasets_dict, n_candles_before=10):
    """
    Version AVEC AFFICHAGE des slopes - SANS FALLBACK
    """
    print(f"\n🔧 ANALYSE PRÉ-TRADE AVEC AFFICHAGE DES SLOPES")
    print("=" * 80)

    results = {}

    for name, (df_complete, df_filtered) in datasets_dict.items():
        if df_complete.empty or df_filtered.empty:
            continue

        print(f"\n🔧 DATASET: {name}")
        print("-" * 50)

        # Préparation optimisée
        df_complete_sorted = df_complete.sort_values(['session_id', 'date']).reset_index(drop=True)
        session_groups = df_complete_sorted.groupby('session_id', sort=False)
        session_indices = {session_id: group.reset_index(drop=True)
                           for session_id, group in session_groups}

        pre_trade_stats = []
        total_trades = len(df_filtered)
        processed = 0

        trades_by_session = df_filtered.groupby('session_id', sort=False)

        # TRAITER SEULEMENT LA PREMIÈRE SESSION POUR AFFICHAGE
        first_session = True

        for session_id, session_trades in trades_by_session:
            if session_id not in session_indices:
                continue

            session_data = session_indices[session_id]
            if len(session_data) < n_candles_before + 1:
                continue

            if first_session:
                print(f"\n🔍 AFFICHAGE DÉTAILLÉ SESSION #{session_id}:")
                session_stats = process_session_trades_DISPLAY(
                    session_trades, session_data, n_candles_before
                )
                first_session = False
            else:
                # Sessions suivantes sans affichage
                session_stats = process_session_trades_CLEAN(
                    session_trades, session_data, n_candles_before
                )

            pre_trade_stats.extend(session_stats)
            processed += len(session_trades)

            if processed % 1000 == 0:
                pct = (processed / total_trades) * 100
                print(f"   Progression: {processed:,}/{total_trades:,} ({pct:.1f}%)")

        print(f"   ✅ TOTAL: {len(pre_trade_stats):,} trades analysés pour {name}")

        if pre_trade_stats:
            analysis = analyze_complete_stats_fast(pre_trade_stats, name)
            results[name] = analysis
            display_complete_results_fast(analysis, name)

    return results


# ===== FONCTION MODIFIÉE DANS MAIN() =====

def add_ultra_fast_pre_trade_analysis_DISPLAY(datasets_dict, n_candles_before=10):
    """
    Version avec affichage des slopes pour debugging
    """
    print(f"\n🔧 ANALYSE AVEC AFFICHAGE DES SLOPES")
    print(f"🔧 Utilisation de calculate_slopes_and_r2_numba SANS fallback")

    results = analyze_pre_trade_conditions_DISPLAY(datasets_dict, n_candles_before)

    print(f"\n✅ ANALYSE AVEC AFFICHAGE TERMINÉE")
    return results


# ===== REMPLACEZ DANS VOTRE MAIN() =====
"""
# Remplacez cette ligne:
pre_trade_results = add_ultra_fast_pre_trade_analysis(datasets_dict, n_candles_before=SLOPE_PERIODS)

# Par cette ligne:
pre_trade_results = add_ultra_fast_pre_trade_analysis_DISPLAY(datasets_dict, n_candles_before=SLOPE_PERIODS)
"""
def add_pattern_analysis_to_main(pre_trade_results):
    """
    Ajoute l'analyse des patterns Spring/Momentum/Épuisement au script principal
    """
    pattern_results = analyze_spring_momentum_exhaustion_patterns(pre_trade_results)
    return pattern_results


# DANS VOTRE FONCTION main(), AJOUTEZ CETTE LIGNE APRÈS slopes_analysis :

# slopes_analysis = add_missing_slopes_analysis(pre_trade_results)
#
# # NOUVELLE LIGNE À AJOUTER :
# pattern_analysis = add_pattern_analysis_to_main(pre_trade_results)

# FONCTION BONUS : Configuration des seuils de patterns
def configure_pattern_thresholds():
    """
    Configuration des seuils pour les patterns - adaptée aux pentes clippées [-1, +1]
    """
    return {
        'spring_volume_threshold': -5,  # Volume baisse forte (20%)
        'spring_duration_threshold': 5,  # Accélération forte (20%)
        'momentum_volume_threshold': 5,  # Volume monte modéré (10%)
        'momentum_duration_threshold': -5,  # Accélération modérée (15%)
        'exhaustion_volume_threshold': 5,  # Volume monte fort (20%)
        'exhaustion_duration_threshold': 5  # Ralentissement modéré (10%)
    }


# ===== CORRECTION 1: Fonction pour merger les résultats slopes avec les datasets originaux =====

def merge_slopes_with_datasets(pre_trade_results, datasets_dict):
    """
    Merge les slopes calculées dans pre_trade_results avec les datasets originaux
    """
    merged_datasets = {}

    for dataset_name, (df_complete, df_filtered) in datasets_dict.items():
        if dataset_name in pre_trade_results and 'raw_data' in pre_trade_results[dataset_name]:
            slopes_df = pre_trade_results[dataset_name]['raw_data']

            print(f"📊 Merging slopes pour {dataset_name}:")
            print(f"   Original filtered: {len(df_filtered):,} trades")
            print(f"   Slopes calculées: {len(slopes_df):,} trades")

            # Merger sur date et session_id
            df_merged = df_filtered.merge(
                slopes_df[['trade_date', 'session_id', 'volume_trend', 'duration_trend', 'trade_result']],
                left_on=['date', 'session_id'],
                right_on=['trade_date', 'session_id'],
                how='inner'
            )

            print(f"   Après merge: {len(df_merged):,} trades avec slopes")

            # Vérifier les colonnes slopes
            if 'volume_trend' in df_merged.columns and 'duration_trend' in df_merged.columns:
                print(f"   ✅ Colonnes slopes présentes")
                print(
                    f"   📊 Volume trend range: [{df_merged['volume_trend'].min():.4f}, {df_merged['volume_trend'].max():.4f}]")
                print(
                    f"   📊 Duration trend range: [{df_merged['duration_trend'].min():.4f}, {df_merged['duration_trend'].max():.4f}]")
            else:
                print(f"   ❌ Colonnes slopes manquantes")

            merged_datasets[dataset_name] = df_merged
        else:
            print(f"❌ {dataset_name}: Pas de résultats slopes disponibles")
            merged_datasets[dataset_name] = df_filtered

    return merged_datasets


# ===== CORRECTION 2: Fonction apply_anti_spring_universal_filter corrigée =====

def apply_anti_spring_universal_filter_fixed(df,
                                             volume_slope_col='volume_trend',
                                             duration_slope_col='duration_trend'):
    """
    Version corrigée qui vérifie d'abord la présence des colonnes
    """
    print(f"   🔍 Colonnes disponibles: {list(df.columns)[:10]}...")  # Afficher les 10 premières colonnes

    # Vérifier la présence des colonnes slopes
    if volume_slope_col not in df.columns:
        print(f"   ❌ Colonne '{volume_slope_col}' manquante")
        return df.copy()

    if duration_slope_col not in df.columns:
        print(f"   ❌ Colonne '{duration_slope_col}' manquante")
        return df.copy()

    print(f"   ✅ Colonnes slopes trouvées: {volume_slope_col}, {duration_slope_col}")

    # Statistiques avant filtrage
    original_count = len(df)
    original_winrate = df['class_binaire'].mean() if 'class_binaire' in df.columns else 0

    # Définir le pattern Spring: Volume↓ ET Duration↓
    spring_mask = (df[volume_slope_col] < 1.5) & (df[duration_slope_col] <1.5) #pour les < et > semble inversé
    spring_count = ~spring_mask.sum()

    print(
        f"   📊 Spring pattern détecté: {spring_count:,}/{original_count:,} trades ({spring_count / original_count * 100:.1f}%)")

    # Appliquer le filtre Anti-Spring
    anti_spring_mask = ~spring_mask

    filtered_df = df[anti_spring_mask]

    filtered_count = len(filtered_df)
    filtered_winrate = filtered_df['class_binaire'].mean() if 'class_binaire' in filtered_df.columns else 0

    print(f"   📊 Après filtre: {filtered_count:,} trades retenus ({filtered_count / original_count * 100:.1f}%)")
    print(f"   📊 Winrate: {original_winrate:.4f} → {filtered_winrate:.4f} ({filtered_winrate - original_winrate:+.4f})")

    return filtered_df


# ===== CORRECTION 3: Fonction analyze_anti_spring_impact corrigée =====

def analyze_anti_spring_impact_fixed(datasets, label_col='class_binaire'):
    """
    Version corrigée qui utilise les datasets avec slopes
    """
    results = {}

    # Normaliser l'itération
    if isinstance(datasets, dict):
        items = datasets.items()
    else:
        items = datasets

    for dataset_name, df in items:
        print(f"\n📊 Analyse Anti-Spring: {dataset_name}")
        print("-" * 50)

        # Filtrer les trades valides (0/1)
        if label_col not in df.columns:
            print(f"   ❌ Colonne '{label_col}' manquante dans {dataset_name}")
            continue

        valid_mask = df[label_col].isin([0, 1])
        df_valid = df[valid_mask]

        orig_trades = len(df_valid)
        orig_wr = df_valid[label_col].mean() if orig_trades > 0 else 0

        print(f"   📊 Données valides: {orig_trades:,} trades")

        # Appliquer le filtre anti-Spring
        df_filtered = apply_anti_spring_universal_filter_fixed(df_valid)

        filt_trades = len(df_filtered)
        filt_wr = df_filtered[label_col].mean() if filt_trades > 0 else 0

        retention_rate = filt_trades / orig_trades * 100 if orig_trades > 0 else 0
        winrate_improvement = filt_wr - orig_wr

        results[dataset_name] = {
            'original_trades': orig_trades,
            'filtered_trades': filt_trades,
            'retention_rate': retention_rate,
            'original_winrate': orig_wr,
            'filtered_winrate': filt_wr,
            'winrate_improvement': winrate_improvement
        }

        print(f"   ✅ RÉSULTAT: {orig_trades:,} → {filt_trades:,} trades ({retention_rate:.1f}% retenus)")
        print(f"   ✅ WINRATE: {orig_wr:.4f} → {filt_wr:.4f} ({winrate_improvement:+.4f})")

    return results

def test_universal_anti_spring_strategy_fixed(datasets_dict, pre_trade_results, label_col='class_binaire'):
    """
    Version corrigée qui utilise les résultats slopes
    """
    print("🔍 TEST STRATÉGIE ANTI-SPRING UNIVERSELLE (VERSION CORRIGÉE)")
    print("=" * 80)

    # Étape 1: Merger les slopes avec les datasets originaux
    print("\n📊 ÉTAPE 1: Merge des slopes avec les datasets")
    print("-" * 60)

    merged_datasets = merge_slopes_with_datasets(pre_trade_results, datasets_dict)

    # Étape 2: Analyser l'impact du filtre anti-Spring
    print("\n📊 ÉTAPE 2: Application du filtre anti-Spring")
    print("-" * 60)

    universal_results = analyze_anti_spring_impact_fixed(merged_datasets, label_col=label_col)

    # Étape 3: Afficher le résumé
    print("\n📈 RÉSUMÉ DE L'IMPACT ANTI-SPRING")
    print("=" * 70)

    print(f"{'Dataset':<15} {'Original':<8} {'Filtré':<8} {'Rétention':<10} {'WR Orig':<8} {'WR Filt':<8} {'Δ WR':<8}")
    print("-" * 70)

    for dataset_name, result in universal_results.items():
        print(f"{dataset_name:<15} "
              f"{result['original_trades']:>7,} "
              f"{result['filtered_trades']:>7,} "
              f"{result['retention_rate']:>8.1f}% "
              f"{result['original_winrate']:>7.2%} "
              f"{result['filtered_winrate']:>7.2%} "
              f"{result['winrate_improvement']:>+7.2%}")

    # Étape 4: Analyse spécifique TEST
    if 'TEST' in universal_results:
        test_result = universal_results['TEST']
        print(f"\n🎯 FOCUS TEST:")
        print(f"   Amélioration winrate: {test_result['winrate_improvement']:+.2%}")

        if test_result['winrate_improvement'] > 0.01:  # +1%
            print(f"   ✅ SUCCÈS: Le filtre anti-Spring améliore TEST")
        elif test_result['winrate_improvement'] > 0:
            print(f"   🟡 MODÉRÉ: Légère amélioration")
        else:
            print(f"   ❌ ÉCHEC: Pas d'amélioration")

    return universal_results


def main():
    """
    Fonction principale
    """
    global SLOPE_PERIODS

    print("🚀 DÉMARRAGE DE L'ANALYSE COMPLÈTE DES PERFORMANCES DE TRADING")
    print("=" * 80)
    print(f"📂 Direction analysée: {DIRECTION}")
    print(f"📁 Répertoire: {DIR}")
    print(f"⚙️  Périodes slope: {SLOPE_PERIODS}")  # Nouveau paramètre affiché
    print()

    # ====== CHARGEMENT DES DONNÉES ======
    print("📥 CHARGEMENT DES FICHIERS")
    print("=" * 30)

    try:
        TRAIN_COMPLETE, TRAIN_FILTERED, train_sessions = load_csv_complete(CSV_TRAIN)
        TEST_COMPLETE, TEST_FILTERED, test_sessions = load_csv_complete(CSV_TEST)
        VAL_COMPLETE, VAL_FILTERED, val_sessions = load_csv_complete(CSV_VAL)
        VAL1_COMPLETE, VAL1_FILTERED, val1_sessions = load_csv_complete(CSV_VAL1)
        UNSEEN_COMPLETE, UNSEEN_FILTERED, unseen_sessions = load_csv_complete(CSV_UNSEEN)
    except Exception as e:
        print(f"❌ Erreur lors du chargement des données: {e}")
        return

    # ====== NOUVELLE ANALYSE: WINRATES BRUTS ======
    filtered_datasets = [
        (TRAIN_FILTERED, "TRAIN"),
        (TEST_FILTERED, "TEST"),
        (VAL_FILTERED, "VALIDATION"),
        (VAL1_FILTERED, "VALIDATION 1"),
        (UNSEEN_FILTERED, "UNSEEN")
    ]

    # Analyse des winrates
    winrate_stats = {}
    for df, name in filtered_datasets:
        result = analyze_winrate_performance(df, name)
        if result:
            winrate_stats[name] = result

    # Affichage des analyses de winrate
    print_winrate_analysis(winrate_stats)
    print_comparative_winrate_analysis(winrate_stats)

    # ====== NOUVELLE ANALYSE: PATTERNS TEMPORELS ======
    temporal_patterns = {}
    for df, name in filtered_datasets:
        patterns = analyze_temporal_patterns(df, name)
        if patterns:
            temporal_patterns[name] = patterns

    print_temporal_analysis(temporal_patterns)

    # ====== NOUVELLE ANALYSE: CONDITIONS DE MARCHÉ ======
    market_conditions = {}
    for df, name in filtered_datasets:
        conditions = analyze_market_conditions(df, name)
        if conditions:
            market_conditions[name] = conditions

    print_market_conditions_analysis(market_conditions)

    # ====== ANALYSE DES STATISTIQUES DE BOUGIES (ANCIEN CODE) ======
    print("\n" + "=" * 60)
    print("🔍 ANALYSE DES STATISTIQUES DE BOUGIES")
    print("=" * 60)

    datasets = [
        (TRAIN_COMPLETE, "TRAIN"),
        (TEST_COMPLETE, "TEST"),
        (VAL_COMPLETE, "VALIDATION"),
        (VAL1_COMPLETE, "VALIDATION 1"),
        (UNSEEN_COMPLETE, "UNSEEN")
    ]

    all_stats = {}

    for df, name in datasets:
        stats = calculate_candle_stats(df, name)
        if stats:
            all_stats[name] = stats

    # ====== RÉSUMÉ GLOBAL ======
    if all_stats:
        print("\n" + "=" * 60)
        print("📋 RÉSUMÉ GLOBAL")
        print("=" * 60)

        # Calcul des moyennes pondérées
        total_candles = sum(stats['count'] for stats in all_stats.values())

        # Durées
        stats_with_duration = {name: stats for name, stats in all_stats.items() if 'duration_mean_seconds' in stats}
        if stats_with_duration:
            weighted_avg_duration = sum(stats['duration_mean_seconds'] * stats['duration_count']
                                        for stats in stats_with_duration.values()) / sum(
                stats['duration_count'] for stats in stats_with_duration.values())

            print("🕐 DURÉES DES BOUGIES:")
            for name, stats in stats_with_duration.items():
                print(
                    f"{name:12}: {stats['duration_mean_seconds']:6.2f}s ({stats['duration_mean_minutes']:5.2f} min) - {stats['duration_count']:,} bougies")
            print(f"🎯 MOYENNE PONDÉRÉE: {weighted_avg_duration:.2f}s ({weighted_avg_duration / 60:.2f} min)")

        # Volumes
        stats_with_volume = {name: stats for name, stats in all_stats.items() if 'volume_mean' in stats}
        if stats_with_volume:
            weighted_avg_volume = sum(stats['volume_mean'] * stats['volume_count']
                                      for stats in stats_with_volume.values()) / sum(
                stats['volume_count'] for stats in stats_with_volume.values())

            print(f"\n📈 VOLUME PAR TICK:")
            for name, stats in stats_with_volume.items():
                print(f"{name:12}: {stats['volume_mean']:8.2f} - {stats['volume_count']:,} bougies")
            print(f"🎯 MOYENNE PONDÉRÉE: {weighted_avg_volume:.2f}")

        print(f"\n📊 TOTAL BOUGIES: {total_candles:,}")

        # ====== ANALYSE COMPARATIVE ======
        if len(all_stats) > 1:
            print("\n" + "=" * 60)
            print("🔍 ANALYSE COMPARATIVE")
            print("=" * 60)

            # Comparaison durées
            if stats_with_duration and len(stats_with_duration) > 1:
                print("🕐 DURÉES:")
                duration_means = [stats['duration_mean_seconds'] for stats in stats_with_duration.values()]
                duration_names = list(stats_with_duration.keys())

                min_duration = min(duration_means)
                max_duration = max(duration_means)
                min_idx = duration_means.index(min_duration)
                max_idx = duration_means.index(max_duration)

                print(f"⚡ Plus rapide: {duration_names[min_idx]} ({min_duration:.2f}s)")
                print(f"🐌 Plus lent: {duration_names[max_idx]} ({max_duration:.2f}s)")
                print(f"📏 Écart: {max_duration - min_duration:.2f}s ({(max_duration - min_duration) / 60:.2f} min)")

                print(f"\n📊 VARIABILITÉ DURÉES (Coefficient de variation):")
                for name, stats in stats_with_duration.items():
                    cv = (stats['duration_std_seconds'] / stats['duration_mean_seconds']) * 100
                    print(f"{name:12}: {cv:.1f}%")

            # Comparaison volumes
            if stats_with_volume and len(stats_with_volume) > 1:
                print(f"\n📈 VOLUMES:")
                volume_means = [stats['volume_mean'] for stats in stats_with_volume.values()]
                volume_names = list(stats_with_volume.keys())

                min_volume = min(volume_means)
                max_volume = max(volume_means)
                min_idx = volume_means.index(min_volume)
                max_idx = volume_means.index(max_volume)

                print(f"📉 Plus faible: {volume_names[min_idx]} ({min_volume:.2f})")
                print(f"📈 Plus élevé: {volume_names[max_idx]} ({max_volume:.2f})")
                print(f"📏 Écart: {max_volume - min_volume:.2f}")

                print(f"\n📊 VARIABILITÉ VOLUMES (Coefficient de variation):")
                for name, stats in stats_with_volume.items():
                    cv = (stats['volume_std'] / stats['volume_mean']) * 100
                    print(f"{name:12}: {cv:.1f}%")

    else:
        print("❌ Aucune donnée valide trouvée dans les fichiers")

    # ====== VÉRIFICATION TEMPORELLE ======
    check_temporal_consistency(datasets)

    # ====== DIAGNOSTIC COMPLET TEST vs TOUS ======
    # ====== ANALYSE DES CONDITIONS PRÉ-TRADE ======
    datasets_dict = {
        'TRAIN': (TRAIN_COMPLETE, TRAIN_FILTERED),
        'TEST': (TEST_COMPLETE, TEST_FILTERED),
        'VALIDATION': (VAL_COMPLETE, VAL_FILTERED),
        'VALIDATION 1': (VAL1_COMPLETE, VAL1_FILTERED),
        'UNSEEN': (UNSEEN_COMPLETE, UNSEEN_FILTERED)
    }

    # Analyser les 10 bougies précédant chaque trade
    pre_trade_results = add_ultra_fast_pre_trade_analysis_DISPLAY(datasets_dict, n_candles_before=SLOPE_PERIODS)
    slopes_analysis = add_missing_slopes_analysis(pre_trade_results)
    # Analyse complète des patterns Spring/Momentum/Épuisement
    pattern_analysis = add_pattern_analysis_to_main(pre_trade_results)
    dataset_names = ['TRAIN', 'TEST', 'VALIDATION', 'VALIDATION 1', 'UNSEEN']
    results = analyze_market_regimes(datasets_dict, dataset_names)

    # ====== RECOMMANDATIONS FINALES CONSOLIDÉES ======
    print("\n" + "=" * 80)
    print("💡 RECOMMANDATIONS CONSOLIDÉES POUR AMÉLIORER TEST")
    print("=" * 80)

    # Analyser les points faibles du dataset TEST
    if 'TEST' in winrate_stats and winrate_stats['TEST'][0] is not None:
        test_stats = winrate_stats['TEST'][0]

        print("🔍 DIAGNOSTIC FINAL TEST:")
        print(f"• Winrate: {test_stats['winrate']:.2f}% (Position: 5/5)")
        print(f"• Consistance: {test_stats['winrate_per_session_std']:.2f}% d'écart-type")
        print(f"• Sessions profitables: {test_stats['profitable_sessions_pct']:.1f}%")
        if results['quality_score']:
            print(f"• Score qualité conditions: {results['quality_score']:.0f}%")
        print(f"• VPT divergents détectés: {len(results['divergent_signals'])}")

        print("\n🎯 PLAN D'ACTION PRIORITAIRE:")

        # Consolidation des recommandations
        all_recommendations = []

        # De l'analyse winrate
        if test_stats['winrate_per_session_std'] > 10:
            all_recommendations.append("📊 CONSISTANCE: Forte variabilité entre sessions")

        # De l'analyse des régimes de marché
        if results['quality_score'] and results['quality_score'] < 50:
            all_recommendations.append("🌪️ CONDITIONS: Régime de marché difficile")

        if len(results['divergent_signals']) > 3:
            all_recommendations.append("📈 VPT: Signaux instables temporellement")

        # Actions concrètes
        print("1. ANALYSE DÉTAILLÉE DES SESSIONS PROBLÉMATIQUES")
        print("   → Identifier les 10 pires sessions de TEST")
        print("   → Analyser leurs conditions de marché communes")

        print("2. RECALIBRAGE DES INDICATEURS VPT")
        print(f"   → Re-tester les {len(results['divergent_signals'])} VPT divergents")
        print("   → Implémenter des seuils adaptatifs par régime")

        print("3. FILTRAGE TEMPOREL INTELLIGENT")
        print("   → Éviter les heures/jours les moins performants")
        print("   → Concentrer sur les créneaux optimaux")

        print("4. ADAPTATION AUX CONDITIONS DE MARCHÉ")
        print("   → Ajuster stops/TP selon la volatilité")
        print("   → Filtrer les périodes de micro-volatilité")

    print(f"\n🔬 SURVEILLANCE CONTINUE:")
    print(f"   • Stabilité VPT: {len(results['divergent_signals'])} indicateurs à surveiller")
    print(f"   • Qualité conditions: Maintenir >60%")
    print(f"   • Consistance sessions: Viser <8% d'écart-type")
    print(f"   • Winrate global: Objectif >52%")

    print("\n" + "=" * 80)
    print("✅ ANALYSE COMPLÈTE TERMINÉE")
    print("=" * 80)
    # Après l'analyse pré-trade existante:
    pre_trade_results = add_ultra_fast_pre_trade_analysis_DISPLAY(datasets_dict, n_candles_before=SLOPE_PERIODS)
    slopes_analysis = add_missing_slopes_analysis(pre_trade_results)
    pattern_analysis = add_pattern_analysis_to_main(pre_trade_results)

    universal_results = test_universal_anti_spring_strategy_fixed(
        datasets_dict,
        pre_trade_results,
        label_col='class_binaire'
    )
if __name__ == "__main__":
    main()