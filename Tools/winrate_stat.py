
import os
from standardFunc_sauv import (load_data, split_sessions, print_notification,
                               plot_calibrationCurve_distrib, plot_fp_tp_rates, check_gpu_availability,
                               timestamp_to_date_utc, calculate_and_display_sessions,
                               calculate_and_display_sessions,
                               calculate_weighted_adjusted_score_custom, sigmoidCustom,
                               custom_metric_ProfitBased_gpu, create_weighted_logistic_obj_gpu,
                               optuna_options, train_finalModel_analyse, init_dataSet, compute_confusion_matrix_cupy,
                               CUSTOM_SECTIONS, sessions_selection, calculate_normalized_objectives, run_cross_validation_optuna, setup_metric_dict,
                               process_RFE_filteringg, calculate_fold_stats, add_session_id, update_fold_metrics, initialize_metrics_dict, setup_xgb_params, cv_config)
import pandas as pd
FILE_NAME_ = "Step5_4_0_6TP_1SL_080919_141024_extractOnlyFullSession_OnlyShort_feat_winsorized.csv"
# FILE_NAME_ = "Step5_4_0_6TP_1SL_080919_141024_extractOnly220LastFullSession_OnlyShort_feat_winsorized.csv"
# FILE_NAME_ = "Step5_4_0_6TP_1SL_080919_141024_extractOnlyFullSession_OnlyShort_feat_winsorizedScaledWithNanVal.csv"
# FILE_NAME_ = "Step5_4_0_6TP_1SL_080919_141024_extractOnlyFullSession_OnlyShort_feat.csv"
DIRECTORY_PATH_ = r"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra chart\xTickReversal\simu\4_0_6TP_1SL\merge"

FILE_PATH_ = os.path.join(DIRECTORY_PATH_, FILE_NAME_)
#df_init = load_data(FILE_PATH_)
df_init = pd.read_csv(FILE_PATH_, sep=';', encoding='iso-8859-1', nrows=100000000)

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

df_filtered = sessions_selection(df_init, CUSTOM_SECTIONS, time_periods_dict=time_periods_dict,
                                     results_directory=None)


def calculate_stats(df, condition_name=""):
    success = len(df[df['class_binaire'] == 1])
    failure = len(df[df['class_binaire'] == 0])
    filtered = len(df[df['class_binaire'] == 99])
    total = success + failure
    win_rate = (success / total * 100) if total > 0 else 0

    print(f"\nStatistiques {condition_name}:")
    print(f"Trades réussis (1): {success}")
    print(f"Trades échoués (0): {failure}")
    print(f"Trades filtrés (99): {filtered}")
    print(f"Total trades actifs: {total}")
    print(f"Win Rate: {win_rate:.2f}%")

    return {'success': success, 'failure': failure, 'filtered': filtered, 'total': total, 'win_rate': win_rate}


def calculate_session_stats_before(df, time_periods_dict):
    stats_per_session = {}
    timestamps = df['deltaTimestampOpening'].values

    for name, info in time_periods_dict.items():
        if info['selected']:
            mask_period = (timestamps >= info['start']) & (timestamps < info['end'])
            session_data = df[mask_period]

            success = len(session_data[session_data['class_binaire'] == 1])
            failure = len(session_data[session_data['class_binaire'] == 0])
            filtered = len(session_data[session_data['class_binaire'] == 99])
            total = success + failure
            win_rate = (success / total * 100) if total > 0 else 0

            stats_per_session[name] = {
                'success': success,
                'failure': failure,
                'filtered': filtered,
                'total': total,
                'win_rate': win_rate
            }

    return stats_per_session


def calculate_session_stats(df, time_periods_dict, stats_before_per_session):
    print("\nUtilisation des périodes du dictionnaire:")
    timestamps = df['deltaTimestampOpening'].values

    for name, info in time_periods_dict.items():
        if info['selected']:
            mask_period = (timestamps >= info['start']) & (timestamps < info['end'])
            session_data = df[mask_period]

            success = len(session_data[session_data['class_binaire'] == 1])
            failure = len(session_data[session_data['class_binaire'] == 0])
            filtered = len(session_data[session_data['class_binaire'] == 99])
            total = success + failure
            win_rate = (success / total * 100) if total > 0 else 0

            # Calculer le changement de win rate pour cette session
            stats_before = stats_before_per_session[name]
            win_rate_before = stats_before['win_rate']
            win_rate_change = win_rate - win_rate_before

            print(f"- {name}: {info['start']} à {info['end']} ({info['description']}) (activée)")
            print(f"  Statistiques pour {name}:")
            print(f"  - Minutes depuis 22h: {info['start']} - {info['end']}")
            print(f"  - Heures: {info['description']}")
            print(f"  - Trades réussis: {success}")
            print(f"  - Trades échoués: {failure}")
            print(f"  - Trades filtrés: {filtered}")
            print(f"  - Total trades: {total}")
            print(f"  - Win Rate: {win_rate:.2f}%")
            print(f"  - Changement du Win Rate: {win_rate_change:+.2f}%\n")




# Exemple d'utilisation
def main():
    # Calcul des stats initiales
    print("\n=== AVANT FILTRAGE ===")
    stats_before = calculate_stats(df_filtered, "initiales")
    stats_before_per_session = calculate_session_stats_before(df_filtered, time_periods_dict)

    # Application des filtres
    df_filtered.loc[df_filtered['diffPriceClosePoc_0_0'] != -0.75, 'class_binaire'] = 99
    #df_filtered.loc[df_filtered['naked_poc_dist_below'] < 1.49+0.25, 'class_binaire'] = 99
    #df_filtered.loc[df_filtered['bearish_dsc_ask_bid_delta_imbalance'] < 0.076923, 'class_binaire'] = 99

    # Calcul des stats après filtrage
    print("\n=== APRÈS FILTRAGE diffPriceClosePoc_0_2 > 0.5 ===")
    # Statistiques par session après filtrage
    print("\n=== STATISTIQUES PAR SESSION APRÈS FILTRAGE ===")
    calculate_session_stats(df_filtered, time_periods_dict, stats_before_per_session)

    stats_after = calculate_stats(df_filtered, "après filtrage diffLowPrice_0_1")

    trades_affected = stats_before['total'] - stats_after['total']
    win_rate_change_global = stats_after['win_rate'] - stats_before['win_rate']

    print("\n=== IMPACT GLOBAL DU FILTRAGE ===")
    print(f"\nTrades filtrés par la condition: {trades_affected}")
    print(f"Changement du Win Rate: {win_rate_change_global:+.2f}%")

if __name__ == "__main__":
    main()