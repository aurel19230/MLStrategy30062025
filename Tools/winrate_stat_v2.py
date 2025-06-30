import os
import numpy as np
import pandas as pd
from numba import njit
from standardFunc_sauv import CUSTOM_SESSIONS, sessions_selection

# Pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.expand_frame_repr', False)

FILE_NAME_ = "Step5_4_0_6TP_1SL_080919_141024_extractOnlyFullSession_OnlyShort_feat_winsorized.csv"
DIRECTORY_PATH_ = r"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra chart\xTickReversal\simu\4_0_6TP_1SL\merge"
FILE_PATH_ = os.path.join(DIRECTORY_PATH_, FILE_NAME_)

# Time periods dictionary
time_periods_dict = {
    'Opening': {
        'start': 0,
        'end': 65,
        'selected': False,
        'description': "22:00-23:05",
        'session_type': 1
    },
    'Asie': {
        'start': 65,
        'end': 535,
        'selected': True,
        'description': "23:05-6:55",
        'session_type': 1
    },
    'OpenUk': {
        'start': 535,
        'end': 545,
        'selected': True,
        'description': "6:55-07:05",
        'session_type': 1
    },
    'preOpenEurope': {
        'start': 545,
        'end': 595,
        'selected': True,
        'description': "07:05-07:55",
        'session_type': 1
    },
    'OpenEurope': {
        'start': 595,
        'end': 605,
        'selected': True,
        'description': "07:55-08:05",
        'session_type': 1
    },
    'MorningEurope': {
        'start': 605,
        'end': 865,
        'selected': True,
        'description': "08:05-12:25",
        'session_type': 1
    },
    'preOpenUS': {
        'start': 875,
        'end': 925,
        'selected': True,
        'description': "12:25-13:25",
        'session_type': 2
    },
    'MoringUS': {
        'start': 935,
        'end': 1065,
        'selected': True,
        'description': "13:25-15:45",
        'session_type': 2
    },
    'AfternonUS': {
        'start': 1065,
        'end': 1195,
        'selected': True,
        'description': "15:45-17:55",
        'session_type': 2
    },
    'Evening': {
        'start': 1195,
        'end': 1280,
        'selected': True,
        'description': "17:55-19:20",
        'session_type': 2
    },
    'Close': {
        'start': 1280,
        'end': 1380,
        'selected': True,
        'description': "19:20-21:00",
        'session_type': 2
    }
}

# List of features to test


features_to_test = [
        "deltaTimestampOpeningSection1min",
        "deltaTimestampOpeningSection5min",
        "deltaTimestampOpeningSection30min",
        "VolAbvState",
        "VolBlwState",
        "candleSizeTicks",
        "diffPriceClosePoc_0_0",
        "diffPriceClosePoc_0_1",
        "diffPriceClosePoc_0_2",
        "diffPriceClosePoc_0_3",
        "diffPriceClosePoc_0_4",
        "diffPriceClosePoc_0_5",
        "diffHighPrice_0_1",
        "diffHighPrice_0_2",
        "diffHighPrice_0_3",
        "diffHighPrice_0_4",
        "diffHighPrice_0_5",
        "diffLowPrice_0_1",
        "diffLowPrice_0_2",
        "diffLowPrice_0_3",
        "diffLowPrice_0_4",
        "diffLowPrice_0_5",
        "diffPriceCloseVWAP",
        "diffPriceCloseVWAPbyIndex",
        "atr",
        "bandWidthBB",
        "perctBB",
        "perct_VA6P",
        "ratio_delta_vol_VA6P",
        "diffPriceClose_VA6PPoc",
        "diffPriceClose_VA6PvaH",
        "diffPriceClose_VA6PvaL",
        "perct_VA11P",
        "ratio_delta_vol_VA11P",
        "diffPriceClose_VA11PPoc",
        "diffPriceClose_VA11PvaH",
        "diffPriceClose_VA11PvaL",
        "perct_VA16P",
        "ratio_delta_vol_VA16P",
        "diffPriceClose_VA16PPoc",
        "diffPriceClose_VA16PvaH",
        "diffPriceClose_VA16PvaL",
        "perct_VA21P",
        "ratio_delta_vol_VA21P",
        "diffPriceClose_VA21PPoc",
        "diffPriceClose_VA21PvaH",
        "diffPriceClose_VA21PvaL",
        "overlap_ratio_VA_6P_11P",
        "overlap_ratio_VA_6P_16P",
        "overlap_ratio_VA_6P_21P",
        "overlap_ratio_VA_11P_21P",
        "poc_diff_6P_11P",
        "poc_diff_ratio_6P_11P",
        "poc_diff_6P_16P",
        "poc_diff_ratio_6P_16P",
        "poc_diff_6P_21P",
        "poc_diff_ratio_6P_21P",
        "poc_diff_11P_21P",
        "poc_diff_ratio_11P_21P",
        "market_regimeADX",
        "market_regimeADX_state",
        "range_strength_10_32",
        "range_strength_5_23",
        "is_in_range_10_32",
        "is_in_range_5_23",
        "bearish_reversal_force",
        "bullish_reversal_force",
        "bearish_ask_bid_ratio",
        "bullish_ask_bid_ratio",
        "ratioDeltaBlw",
        "ratioDeltaAbv",
        "diffVolCandle_0_1Ratio",
        "diffVolDelta_0_1Ratio",
        "cumDiffVolDeltaRatio",
        "VolPocVolCandleRatio",
        "pocDeltaPocVolRatio",
        "VolAbv_vol_ratio",
        "VolBlw_vol_ratio",
        "asymetrie_volume",
        "VolCandleMeanxRatio",
        "bearish_ask_ratio",
        "bearish_bid_ratio",
        "bullish_ask_ratio",
        "bullish_bid_ratio",
        "bearish_ask_score",
        "bearish_bid_score",
        "bearish_imnbScore_score",
        "bullish_ask_score",
        "bullish_bid_score",
        "bullish_imnbScore_score",
        "bull_imbalance_low_1",
        "bull_imbalance_low_2",
        "bull_imbalance_low_3",
        "bull_imbalance_high_0",
        "bull_imbalance_high_1",
        "bull_imbalance_high_2",
        "bear_imbalance_low_0",
        "bear_imbalance_low_1",
        "bear_imbalance_low_2",
        "bear_imbalance_high_1",
        "bear_imbalance_high_2",
        "bear_imbalance_high_3",
        "imbalance_score_low",
        "imbalance_score_high",
        "finished_auction_high",
        "finished_auction_low",
        "bearish_ask_abs_ratio_abv",
        "bearish_bid_abs_ratio_abv",
        "bearish_abs_diff_abv",
        "bullish_ask_abs_ratio_blw",
        "bullish_bid_abs_ratio_blw",
        "bullish_abs_diff_blw",
        "bearish_askBigStand_abs_ratio_abv",
        "bearish_askBigStand_abs_ratio_abv_special",
        "bearish_bidBigStand_abs_ratio_abv",
        "bearish_bidBigStand_abs_ratio_abv_special",
        "bearish_bigStand_abs_diff_abv",
        "bullish_askBigStand_abs_ratio_blw",
        "bullish_askBigStand_abs_ratio_blw_special",
        "bullish_bidBigStand_abs_ratio_blw",
        "bullish_bidBigStand_abs_ratio_blw_special",
        "bullish_bigStand_abs_diff_blw",
        "bearish_askBigHigh_abs_ratio_abv",
        "bearish_askBigHigh_abs_ratio_abv_special",
        "bearish_bidBigHigh_abs_ratio_abv",
        "bearish_bidBigHigh_abs_ratio_abv_special",
        "bearish_bigHigh_abs_diff_abv",
        "bullish_askBigHigh_abs_ratio_blw",
        "bullish_askBigHigh_abs_ratio_blw_special",
        "bullish_bidBigHigh_abs_ratio_blw",
        "bullish_bidBigHigh_abs_ratio_blw_special",
        "bullish_bigHigh_abs_diff_blw",
        "bearish_extrem_revIntensity_ratio_extrem",
        "bullish_extrem_revIntensity_ratio_extrem",
        "bearish_extrem_zone_volume_ratio_extrem",
        "bullish_extrem_zone_volume_ratio_extrem",
        "bearish_extrem_pressure_ratio_extrem",
        "bearish_extrem_pressure_ratio_special_extrem",
        "bullish_extrem_pressure_ratio_extrem",
        "bullish_extrem_pressure_ratio_special_extrem",
        "bearish_extrem_abs_ratio_extrem",
        "bearish_extrem_abs_ratio_special_extrem",
        "bullish_extrem_abs_ratio_extrem",
        "bullish_extrem_abs_ratio_special_extrem",
        "bearish_extrem_vs_rest_activity_extrem",
        "bearish_extrem_vs_rest_activity_special_extrem",
        "bullish_extrem_vs_rest_activity_extrem",
        "bullish_extrem_vs_rest_activity_special_extrem",
        "bearish_continuation_vs_reversal_extrem",
        "bearish_continuation_vs_reversal_special_extrem",
        "bullish_continuation_vs_reversal_extrem",
        "bullish_continuation_vs_reversal_special_extrem",
        "bearish_repeat_ticks_ratio_extrem",
        "bearish_repeat_ticks_ratio_special_extrem",
        "bullish_repeat_ticks_ratio_extrem",
        "bullish_repeat_ticks_ratio_special_extrem",
        "bearish_big_trade_ratio_extrem",
        "bearish_big_trade_ratio_special_extrem",
        "bearish_big_trade_imbalance_extrem",
        "bearish_big_trade_imbalance_special_extrem",
        "bullish_big_trade_ratio_extrem",
        "bullish_big_trade_ratio_special_extrem",
        "bullish_big_trade_imbalance_extrem",
        "bullish_big_trade_imbalance_special_extrem",
        "bearish_asc_dsc_ratio",
        "bearish_asc_dsc_ratio_special",
        "bearish_asc_dynamics",
        "bearish_dsc_dynamics",
        "bullish_asc_dsc_ratio",
        "bullish_asc_dsc_ratio_special",
        "bullish_asc_dynamics",
        "bullish_dsc_dynamics",
        "bearish_asc_ask_bid_imbalance",
        "bearish_dsc_ask_bid_imbalance",
        "bearish_imbalance_evolution",
        "bearish_asc_ask_bid_delta_imbalance",
        "bearish_dsc_ask_bid_delta_imbalance",
        "bullish_asc_ask_bid_imbalance",
        "bullish_dsc_ask_bid_imbalance",
        "bullish_imbalance_evolution",
        "bullish_asc_ask_bid_delta_imbalance",
        "bullish_dsc_ask_bid_delta_imbalance",
        "extrem_asc_ratio_bearish_extrem",
        "bearish_extrem_dsc_ratio_extrem",
        "extrem_zone_significance_bearish_extrem",
        "extrem_ask_bid_imbalance_bearish_extrem",
        "extrem_ask_bid_imbalance_bearish_special_extrem",
        "extrem_asc_dsc_comparison_bearish_extrem",
        "extrem_asc_ratio_bullish_extrem",
        "extrem_dsc_ratio_bullish_extrem",
        "extrem_zone_significance_bullish_extrem",
        "extrem_ask_bid_imbalance_bullish",
        "extrem_ask_bid_imbalance_bullish_special_extrem",
        "extrem_asc_dsc_comparison_bullish_extrem",
        "bearish_absorption_ratio",
        "bearish_absorption_ratio_special",
        "bullish_absorption_ratio",
        "bullish_absorption_ratio_special",
        "bearish_big_trade_ratio2_extrem",
        "bearish_big_trade_ratio2_special_extrem",
        "bullish_big_trade_ratio2_extrem",
        "bullish_big_trade_ratio2_extrem_special",
        "absorption_intensity_repeat_bearish_vol",
        "bearish_absorption_intensity_repeat_count",
        "bearish_absorption_intensity_repeat_count_special",
        "bearish_repeatAskBid_ratio",
        "absorption_intensity_repeat_bullish_vol",
        "bullish_absorption_intensity_repeat_count",
        "bullish_absorption_intensity_repeat_count_special",
        "bullish_repeatAskBid_ratio",
        "count_AbvBlw_asym_ratio",
        "count_blw_tot_ratio",
        "count_abv_tot_ratio",
        "bearish_volume_ratio_6Tick",
        "bullish_volume_ratio_6Tick",
        "bearish_relatif_ratio_6Tick",
        "bullish_relatif_ratio_6Tick",
        "bearish_relatifDelta_ratio_6Tick",
        "bullish_relatifDelta_ratio_6Tick",
        "bearish_buyer_pressure_6Tick",
        "bullish_buyer_pressure_6Tick",
        "bearish_seller_pressure_6Tick",
        "bullish_seller_pressure_6Tick",
        "bearish_absorption_6Tick",
        "bearish_absorption_6Tick_special",
        "bullish_absorption_6Tick",
        "bullish_absorption_6Tick_special",
        "bearish_repeat_ticks_ratio_6Tick",
        "bullish_repeat_ticks_ratio_6Tick",
        "bearish_price_dynamics_comparison_6Tick",
        "bearish_price_dynamics_comparison_6Tick_special",
        "bullish_price_dynamics_comparison_6Tick",
        "bullish_price_dynamics_comparison_6Tick_special",
        "bearish_activity_bid_ask_ratio_6Tick",
        "bullish_activity_ask_bid_ratio_6Tick",
        "bearish_repeat_ticks_imbalance_6Tick",
        "bullish_repeat_ticks_imbalance_6Tick",
        "bearish_ticks_imbalance_6Tick",
        "bullish_ticks_imbalance_6Tick",
        "vol_volatility_score",
        "price_volume_dynamic",
        "bearish_absorption_score",
        "bullish_absorption_score",
        "bearish_market_context_score",
        "bullish_market_context_score",
        "bearish_combined_pressure",
        "bullish_combined_pressure",
        "naked_poc_dist_above",
        "naked_poc_dist_below",
        "timeStampOpening"
    ]



operators = ['!=', '<', '>', '==']


@njit
def calculate_stats_numba(class_binaire):
    """Optimized statistics calculation with numba"""
    success = np.sum(class_binaire == 1)
    failure = np.sum(class_binaire == 0)
    filtered = np.sum(class_binaire == 99)
    total = success + failure
    win_rate = (success / total * 100) if total > 0 else 0
    return success, failure, filtered, total, win_rate


@njit
def apply_filter_numba(feature_values, class_binaire, operator_type, threshold):
    """Numba-optimized filter application"""
    # operator_type: 0 for !=, 1 for <, 2 for >, 3 for ==
    result = class_binaire.copy()

    if operator_type == 0:  # !=
        mask = feature_values != threshold
    elif operator_type == 1:  #
        mask = feature_values < threshold
    elif operator_type == 2:  # >
        mask = feature_values > threshold
    else:  # ==
        mask = feature_values == threshold

    result[mask] = 99
    return result


def get_thresholds(feature_values):
    """Calculate thresholds for a feature"""
    clean_values = feature_values[~np.isnan(feature_values)]
    unique_values = np.unique(clean_values)

    if len(unique_values) > 20:
        # Calculer les valeurs min et max
        min_val = np.min(clean_values)
        max_val = np.max(clean_values)

        # Créer 11 points équidistants (pour avoir 10 intervalles)
        thresholds = np.linspace(min_val, max_val, 21)
        return np.unique(thresholds), True
    return unique_values, False


def operator_to_type(operator):
    """Convert operator string to numeric type for numba"""
    return {'!=': 0, '<': 1, '>': 2, '==': 3}[operator]


def process_feature(feature_values, class_binaire, base_winrate, base_total_trades):
    """Process a single feature and return results"""
    thresholds, is_using_percentiles = get_thresholds(feature_values)
    current_operators = ['<', '>'] if is_using_percentiles else operators

    print(f"Thresholds: {thresholds}")
    print(f"Operators used: {current_operators}")

    feature_results = []

    for threshold in thresholds:
        for operator in current_operators:
            operator_type = operator_to_type(operator)
            filtered_class = apply_filter_numba(feature_values, class_binaire, operator_type, threshold)
            stats = calculate_stats_numba(filtered_class)

            # Calculate tick_pnl: successes * 1.5 - failures * 1
            tick_pnl = (stats[0] * 1.5) - (stats[1] * 1.1)

            feature_results.append({
                'operator': operator,
                'threshold': threshold,
                'winrate_improvement': stats[4] - base_winrate,
                'trades_kept_ratio': stats[3] / base_total_trades if base_total_trades > 0 else 0,
                'new_winrate': stats[4],
                'success': stats[0],
                'failure': stats[1],
                'filtered': stats[2],
                'total': stats[3],
                'tick_pnl': tick_pnl  # Add the new field
            })

    return feature_results


def main():
    # Load data and apply initial filtering
    df_filtered = pd.read_csv(FILE_PATH_, sep=';', encoding='iso-8859-1', nrows=10000000)
    df_filtered = sessions_selection(df_filtered, CUSTOM_SECTIONS, time_periods_dict=time_periods_dict,
                                     results_directory=None)

    # Seuil minimal pour le nombre de trades
    MIN_TRADES_COUNT = 1000  # Vous pouvez ajuster cette valeur selon vos besoins

    # Calculate base statistics
    base_class_binaire = df_filtered['class_binaire'].values
    base_stats = calculate_stats_numba(base_class_binaire)
    base_winrate = base_stats[4]
    base_total_trades = base_stats[3]
    base_success = base_stats[0]
    base_failure = base_stats[1]

    print(f"\nBase statistics:")
    print(f"Total trades: {base_total_trades:,}")
    print(f"Successful trades: {base_success:,}")
    print(f"Failed trades: {base_failure:,}")
    print(f"Base win rate: {base_winrate:.2f}%")

    all_results = []

    # Process each feature
    for feature in features_to_test:
        print(f"\nProcessing feature: {feature}")
        feature_values = df_filtered[feature].values
        class_binaire = df_filtered['class_binaire'].values

        feature_results = process_feature(feature_values, class_binaire,
                                          base_winrate, base_total_trades)

        # Add feature name to results
        for result in feature_results:
            result['feature'] = feature
            all_results.append(result)

    # Convert results to DataFrame and sort
    results_df = pd.DataFrame(all_results)

    # Filtrer les résultats avec un nombre minimum de trades
    results_df = results_df[results_df['total'] >= MIN_TRADES_COUNT]

    results_df_sorted = results_df.sort_values(by='winrate_improvement', ascending=False)

    # Display results
    print(
        f"\nTop results based on Win Rate Improvement (Reference total trades: {base_total_trades:,}, Success: {base_success:,}, Failure: {base_failure:,})")

    # Ajouter les explications des opérateurs
    print("\nInterprétation des opérateurs de filtrage :")
    print("operator < threshold : conserve les trades où la valeur est >= threshold")
    print("operator > threshold : conserve les trades où la valeur est <= threshold")
    print("operator == threshold : conserve les trades où la valeur est != threshold")
    print("operator != threshold : conserve les trades où la valeur est == threshold")

    print(f"\nShowing only results with {MIN_TRADES_COUNT:,}+ trades:")

    # Renommer les colonnes pour plus de clarté
    results_df_sorted.columns = results_df_sorted.columns.map({
        'feature': 'feature',
        'operator': 'operator',
        'threshold': 'threshold',
        'winrate_improvement': 'winrate_improvement',
        'trades_kept_ratio': 'trades_kept_ratio',
        'total': 'trades_total',
        'success': 'trades_success',
        'failure': 'trades_failure',
        'new_winrate': 'new_winrate',
        'tick_pnl': 'tick_pnl'  # Add the new column mapping
    })

    # Mettre à jour les noms de colonnes dans columns_to_display pour correspondre aux nouveaux noms
    columns_to_display = ['feature', 'operator', 'threshold', 'winrate_improvement',
                          'trades_kept_ratio', 'trades_total', 'trades_success',
                          'trades_failure', 'new_winrate', 'tick_pnl']  # Add tick_pnl to

    print(results_df_sorted[columns_to_display].head(100).to_string(index=False))


if __name__ == "__main__":
    main()