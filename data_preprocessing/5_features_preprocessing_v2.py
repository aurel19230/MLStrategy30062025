
from func_standard import calculate_naked_poc_distances, CUSTOM_SESSIONS,diviser_fichier_par_sessions, \
    save_features_with_sessions,process_reg_slope_replacement

from Tools.func_features_preprocessing import *
import numpy as np
from stats_sc.standard_stat_sc import *
from Clustering.func_clustering import *


import platform  # <-- Ajouter cette ligne

diffDivBy0 = np.nan
addDivBy0 = np.nan
valueX = np.nan
valueY = np.nan
# Définition de la fonction calculate_max_ratio
import warnings
from pandas.errors import PerformanceWarning
# Nom du fichier
# Ignorer tous les avertissements de performance pandas
warnings.filterwarnings("ignore", category=PerformanceWarning)
direction="Short"
file_name =        f"Step4_5_0_5TP_6SL_150525_300625_extractOnlyFullSession_Only{direction}.csv"
file_name_unseen = f"Step5_5_0_5TP_6SL_010124_270625_extractOnlyFullSession_Only{direction}_feat__split5_15052025_27062025.csv"
DIR="5_0_5TP_6SL"

file_nameEvent = "Calendrier_Evenements_Macroeconomiques_2024_2025_AvecDoubleEvent.csv"

USSE_SPLIT_SESSION=True #pour effectuer les split des sessions
USE_DEFAUT_PARAM_4_SPLIT_SESSION = True #pour prendre les semarquations des splits par defaut


# Chemin du répertoire
#directory_path =  r"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra_chart\xTickReversal\simu\5_0_5TP_1SL\version2\merge\extend"
if platform.system() != "Darwin":
    directory_path = rf"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra_chart\xTickReversal\simu\{DIR}\merge"
    directory_path = rf"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra_chart\xTickReversal\simu\\{DIR}\\merge"
    directory_path_unseen = rf"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra_chart\xTickReversal\simu\{DIR}\merge"
    directory_path_unseen=rf"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra_chart\xTickReversal\simu\{DIR}\merge"
    directory_pathEvent = r"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra_chart\xTickReversal\simu\5_0_5TP_6SL_Train_Test_Val1_Val\merge"
else:
    directory_path = "/Users/aurelienlachaud/Documents/trading_local/5_0_5TP_1SL_1/merge"
# Construction du chemin complet du fichier
file_path = os.path.join(directory_path, file_name)
file_path_unseen = os.path.join(directory_path_unseen, file_name_unseen)

file_pathEvent = os.path.join(directory_pathEvent, file_nameEvent)

if REPLACE_NAN:
    print(
        f"\nINFO : Implémenter dans le code => les valeurs NaN seront remplacées par {REPLACED_NANVALUE_BY} et un index")
else:
    print(
        f"\nINFO : Implémenter dans le code => les valeurs NaN ne seront pas remplacées par une valeur choisie par l'utilisateur mais laissé à NAN")


def calculate_features_and_sessionsStat(df, file_path_, trained_models=None):
    import pandas as pd  # Import explicite
    import os

    # Calcul de la moyenne de trade_pnl pour chaque classe
    mean_pnl = df.groupby('class_binaire')['trade_pnl'].mean()

    print("\nMoyenne de trade_pnl par classe:")
    print(f"Classe 0 (Perdants): {mean_pnl[0]:.2f}")
    print(f"Classe 1 (Gagnants): {mean_pnl[1]:.2f}")
    stats_pnl = df.groupby('class_binaire')['trade_pnl'].agg(['count', 'mean', 'std'])
    print("\nStatistiques de trade_pnl par classe:")
    print(stats_pnl)

    # Afficher la liste complète des colonnes
    all_columns = df.columns.tolist()

    # Imprimer la liste
    print("Liste complète des colonnes:")
    for col in all_columns:
        print(col)

    print_notification("Début du calcul des features")
    # Calcul des features
    features_df = pd.DataFrame()
    features_df['deltaTimestampOpening'] = df['sc_deltaTimestampOpening']
    # Session 1 minute
    features_df['deltaTimestampOpeningSession1min'] = df['sc_deltaTimestampOpening'].apply(
        lambda x: min(int(np.floor(x / 1)) * 1, 1379))  # 23h = 1380 minutes - 1

    unique_sections = sorted(features_df['deltaTimestampOpeningSession1min'].unique())
    section_to_index = {section: index for index, section in enumerate(unique_sections)}
    features_df['deltaTimestampOpeningSession1index'] = features_df['deltaTimestampOpeningSession1min'].map(
        section_to_index)

    # Session 5 minutes
    features_df['deltaTimestampOpeningSession5min'] = df['sc_deltaTimestampOpening'].apply(
        lambda x: min(int(np.floor(x / 5)) * 5, 1375))  # Dernier multiple de 5 < 1380

    unique_sections = sorted(features_df['deltaTimestampOpeningSession5min'].unique())
    section_to_index = {section: index for index, section in enumerate(unique_sections)}
    features_df['deltaTimestampOpeningSession5index'] = features_df['deltaTimestampOpeningSession5min'].map(
        section_to_index)

    # Session 15 minutes
    features_df['deltaTimestampOpeningSession15min'] = df['sc_deltaTimestampOpening'].apply(
        lambda x: min(int(np.floor(x / 15)) * 15, 1365))  # Dernier multiple de 15 < 1380

    unique_sections = sorted(features_df['deltaTimestampOpeningSession15min'].unique())
    section_to_index = {section: index for index, section in enumerate(unique_sections)}
    features_df['deltaTimestampOpeningSession15index'] = features_df['deltaTimestampOpeningSession15min'].map(
        section_to_index)

    # Session 30 minutes
    features_df['deltaTimestampOpeningSession30min'] = df['sc_deltaTimestampOpening'].apply(
        lambda x: min(int(np.floor(x / 30)) * 30, 1350))  # Dernier multiple de 30 < 1380

    unique_sections = sorted(features_df['deltaTimestampOpeningSession30min'].unique())
    section_to_index = {section: index for index, section in enumerate(unique_sections)}
    features_df['deltaTimestampOpeningSession30index'] = features_df['deltaTimestampOpeningSession30min'].map(
        section_to_index)

    # Custom session
    features_df['deltaCustomSessionMin'] = df['sc_deltaTimestampOpening'].apply(
        lambda x: get_custom_section(x, CUSTOM_SESSIONS)['start']
    )

    # Application sur features_df
    features_df['deltaCustomSessionIndex'] = features_df['deltaTimestampOpening'].apply(
        lambda x: get_custom_section_index(x, CUSTOM_SESSIONS)
    )
    # Utilisation
    windows = [
        #6, 14, 21,30, 40,
         10,50]
    for window in windows:
        slope_r2_df = apply_optimized_slope_r2_calculation(df, window)
        features_df = pd.concat([features_df, slope_r2_df], axis=1)

    windows_sma = [6, 14, 21, 30]
    for window in windows_sma:
        ratio, zscore = enhanced_close_to_sma_ratio(df, window,diffDivBy0=diffDivBy0,DEFAULT_DIV_BY0=DEFAULT_DIV_BY0,valueX=valueX,fill_value=0)
        features_df[f'close_sma_ratio_{window}'] = ratio
        features_df[f'close_sma_zscore_{window}'] = zscore

    # Ajout de la colonne à features_df
    features_df['linear_slope_prevSession'] = calculate_previous_session_slope(df, features_df)

    # Appliquer la fonction

    candle_rev_tick = calculate_candle_rev_tick(df)

    # Calculer les features d'absorption
    absorption_features = calculate_absorpsion_features(df, candle_rev_tick)

    # Ajouter les colonnes à features_df
    features_df = pd.concat([features_df, absorption_features], axis=1)

    # Features précédentes
    features_df['volAbvState'] = np.where(df['sc_volAbv'] == 0, 0, 1)
    features_df['volBlwState'] = np.where(df['sc_volBlw'] == 0, 0, 1)
    features_df['candleSizeTicks'] = np.where(df['sc_candleSizeTicks'] < 4, np.nan, df['sc_candleSizeTicks'])
    features_df['diffPriceClosePoc_0_0'] = df['sc_close'] - df['sc_pocPrice']
    features_df['diffPriceClosePoc_0_1'] = df['sc_close'] - df['sc_pocPrice'].shift(1)
    features_df['diffPriceClosePoc_0_2'] = df['sc_close'] - df['sc_pocPrice'].shift(2)
    features_df['diffPriceClosePoc_0_3'] = df['sc_close'] - df['sc_pocPrice'].shift(3)
    features_df['diffPriceClosePoc_0_4'] = df['sc_close'] - df['sc_pocPrice'].shift(4)
    features_df['diffPriceClosePoc_0_5'] = df['sc_close'] - df['sc_pocPrice'].shift(5)
    # features_df['diffPriceClosePoc_0_6'] = df['sc_close'] - df['sc_pocPrice'].shift(6)

    features_df['diffPocPrice_0_1'] = df['sc_pocPrice'] - df['sc_pocPrice'].shift(1)
    features_df['diffPocPrice_1_2'] = df['sc_pocPrice'].shift(1) - df['sc_pocPrice'].shift(2)
    features_df['diffPocPrice_2_3'] = df['sc_pocPrice'].shift(2) - df['sc_pocPrice'].shift(3)
    features_df['diffPocPrice_0_2'] = df['sc_pocPrice'] - df['sc_pocPrice'].shift(2)



    features_df['diffHighPrice_0_1'] = df['sc_high'] - df['sc_high'].shift(1)
    features_df['diffHighPrice_0_2'] = df['sc_high'] - df['sc_high'].shift(2)
    features_df['diffHighPrice_0_3'] = df['sc_high'] - df['sc_high'].shift(3)
    features_df['diffHighPrice_0_4'] = df['sc_high'] - df['sc_high'].shift(4)
    features_df['diffHighPrice_0_5'] = df['sc_high'] - df['sc_high'].shift(5)
    # features_df['diffHighPrice_0_6'] = df['sc_high'] - df['sc_high'].shift(6)


    features_df['diffLowPrice_0_1'] = df['sc_low'] - df['sc_low'].shift(1)
    features_df['diffLowPrice_0_2'] = df['sc_low'] - df['sc_low'].shift(2)
    features_df['diffLowPrice_0_3'] = df['sc_low'] - df['sc_low'].shift(3)
    features_df['diffLowPrice_0_4'] = df['sc_low'] - df['sc_low'].shift(4)
    features_df['diffLowPrice_0_5'] = df['sc_low'] - df['sc_low'].shift(5)
    # features_df['diffLowPrice_0_6'] = df['sc_low'] - df['sc_low'].shift(6)
    # features_df['diffLowPrice_0_6'] = df['sc_low'] - df['sc_low'].shift(6)


    features_df['diffPriceCloseVWAP'] = df['sc_close'] - df['sc_VWAP']

    # Créer les conditions pour chaque plage
    conditions = [
        (df['sc_close'] >= df['sc_VWAP']) & (df['sc_close'] <= df['sc_VWAPsd1Top']),
        (df['sc_close'] > df['sc_VWAPsd1Top']) & (df['sc_close'] <= df['sc_VWAPsd2Top']),
        (df['sc_close'] > df['sc_VWAPsd2Top']) & (df['sc_close'] <= df['sc_VWAPsd3Top']),
        (df['sc_close'] > df['sc_VWAPsd3Top']) & (df['sc_close'] <= df['sc_VWAPsd4Top']),
        (df['sc_close'] > df['sc_VWAPsd4Top']),
        (df['sc_close'] < df['sc_VWAP']) & (df['sc_close'] >= df['sc_VWAPsd1Bot']),
        (df['sc_close'] < df['sc_VWAPsd1Bot']) & (df['sc_close'] >= df['sc_VWAPsd2Bot']),
        (df['sc_close'] < df['sc_VWAPsd2Bot']) & (df['sc_close'] >= df['sc_VWAPsd3Bot']),
        (df['sc_close'] < df['sc_VWAPsd3Bot']) & (df['sc_close'] >= df['sc_VWAPsd4Bot']),
        (df['sc_close'] < df['sc_VWAPsd4Bot'])
    ]

    # Créer les valeurs correspondantes pour chaque condition
    values = [1, 2, 3, 4, 5, -1, -2, -3, -4, -5]

    # Utiliser np.select pour créer la nouvelle feature
    features_df['diffPriceCloseVWAPbyIndex'] = np.select(conditions, values, default=0)

    features_df['atr'] = df['sc_atr']
    features_df['bandWidthBB'] = df['sc_bandWidthBB']
    features_df['perctBB'] = df['sc_perctBB']



    # Apply the function for different periods

    # Liste des périodes à analyser
    periods = [6, 11, 16, 21]
    for nbPeriods in periods:
        # Calculate the percentage of the value area using pd.notnull()
        value_area = valueArea_pct(df, nbPeriods)
        features_df[f'perct_VA{nbPeriods}P'] = np.where(
            pd.notnull(value_area),
            value_area,
            np.nan
        )

        # Calcul du ratio delta volume
        features_df[f'ratio_delta_vol_VA{nbPeriods}P'] = np.where(
            df[f'sc_vaVol_{nbPeriods}periods'] != 0,
            df[f'sc_vaDelta_{nbPeriods}periods'] / df[f'sc_vaVol_{nbPeriods}periods'],
            np.nan
        )

        # Différence entre le prix de clôture et le POC
        features_df[f'diffPriceClose_VA{nbPeriods}PPoc'] = np.where(
            df[f'sc_vaPoc_{nbPeriods}periods'] != 0,
            df['sc_close'] - df[f'sc_vaPoc_{nbPeriods}periods'],
            np.nan
        )

        # Différence entre le prix de clôture et VAH
        features_df[f'diffPriceClose_VA{nbPeriods}PvaH'] = np.where(
            df[f'sc_vaH_{nbPeriods}periods'] != 0,
            df['sc_close'] - df[f'sc_vaH_{nbPeriods}periods'],
            np.nan
        )

        # Différence entre le prix de clôture et VAL
        features_df[f'diffPriceClose_VA{nbPeriods}PvaL'] = np.where(
            df[f'sc_vaL_{nbPeriods}periods'] != 0,
            df['sc_close'] - df[f'sc_vaL_{nbPeriods}periods'],
            np.nan
        )

    # Génération des combinaisons de périodes
    period_combinations = [(6, 11), (6, 16), (6, 21), (11, 21)]

    for nbPeriods1, nbPeriods2 in period_combinations:
        # --- Proposition 1 : Chevauchement des zones de valeur ---

        # Récupération des VAH et VAL pour les deux périodes

        vaH_p1 = df[f'sc_vaH_{nbPeriods1}periods']
        vaL_p1 = df[f'sc_vaL_{nbPeriods1}periods']
        vaH_p2 = df[f'sc_vaH_{nbPeriods2}periods']
        vaL_p2 = df[f'sc_vaL_{nbPeriods2}periods']

        # Calcul du chevauchement
        min_VAH = np.minimum(vaH_p1, vaH_p2)
        max_VAL = np.maximum(vaL_p1, vaL_p2)
        overlap = np.maximum(0, min_VAH - max_VAL)

        # Calcul de l'étendue totale des zones de valeur combinées
        max_VAH_total = np.maximum(vaH_p1, vaH_p2)
        min_VAL_total = np.minimum(vaL_p1, vaL_p2)
        total_range = max_VAH_total - min_VAL_total

        # Calcul du ratio de chevauchement normalisé
        condition = (total_range != 0) & (vaH_p1 != 0) & (vaH_p2 != 0) & (vaL_p1 != 0) & (vaL_p2 != 0)
        overlap_ratio = np.where(condition, overlap / total_range, np.nan)

        # Ajout de la nouvelle feature au dataframe features_df
        features_df[f'overlap_ratio_VA_{nbPeriods1}P_{nbPeriods2}P'] = overlap_ratio

        # --- Proposition 2 : Analyse des POC ---

        # Récupération des POC pour les deux périodes
        poc_p1 = df[f'sc_vaPoc_{nbPeriods1}periods']
        poc_p2 = df[f'sc_vaPoc_{nbPeriods2}periods']

        # Calcul de la différence absolue entre les POC
        condition = (poc_p1 != 0) & (poc_p2 != 0)
        poc_diff = np.where(condition, poc_p1 - poc_p2, np.nan)

        # Calcul de la valeur moyenne des POC pour normalisation
        average_POC = (poc_p1 + poc_p2) / 2

        # Calcul du ratio de différence normalisé
        condition = (average_POC != 0) & (poc_p1 != 0) & (poc_p2 != 0)
        poc_diff_ratio = np.where(condition, np.abs(poc_diff) / average_POC, np.nan)

        # Ajout des nouvelles features au dataframe features_df
        features_df[f'poc_diff_{nbPeriods1}P_{nbPeriods2}P'] = poc_diff
        features_df[f'poc_diff_ratio_{nbPeriods1}P_{nbPeriods2}P'] = poc_diff_ratio

    # Appliquer range_strength sur une copie de df pour ne pas modifier df
    df_copy1 = df.copy()
    df_with_range_strength_10_32, range_strength_percent_in_range_10_32 = range_strength(df_copy1, 'range_strength_10_32',
                                                                                         window=10, atr_multiple=3.2,
                                                                                         min_strength=0.1)
    df_with_range_strength_5_23, range_strength_percent_in_range_5_23 = range_strength(df_copy1, 'range_strength_5_23',
                                                                                       window=5, atr_multiple=2.3,
                                                                                       min_strength=0.1)



    # Appliquer detect_market_regime sur une copie de df pour ne pas modifier df
    df_copy = df.copy()
    df_with_regime, regimeAdx_pct_infThreshold = detect_market_regimeADX(df_copy, period=14, adx_threshold=25)
    # Ajouter la colonne 'market_regime' à features_df
    features_df['market_regimeADX'] = df_with_regime['market_regimeADX']
    features_df['is_in_range_10_32'] = df_with_range_strength_10_32['range_strength_10_32'].notna().astype(int)
    features_df['is_in_range_5_23'] = df_with_range_strength_5_23['range_strength_5_23'].notna().astype(int)

    conditions = [
        (features_df['market_regimeADX'] < 25),
        (features_df['market_regimeADX'] >= 25) & (features_df['market_regimeADX'] < 50),
        (features_df['market_regimeADX'] >= 50) & (features_df['market_regimeADX'] < 75),
        (features_df['market_regimeADX'] >= 75)
    ]

    choices = [0, 1, 2, 3]

    features_df['market_regimeADX_state'] = np.select(conditions, choices, default=np.nan)

    # Nouvelles features - Force du renversement
    features_df['bearish_reversal_force'] = np.where(df['sc_volume'] != 0, df['sc_volAbv'] / df['sc_volume'],
                                                     addDivBy0 if DEFAULT_DIV_BY0 else valueX)
    features_df['bullish_reversal_force'] = np.where(df['sc_volume'] != 0, df['sc_volBlw'] / df['sc_volume'],
                                                     addDivBy0 if DEFAULT_DIV_BY0 else valueX)


    # Nouvelles features - Features de Momentum:



    # Relatif volume evol
    features_df['diffVolCandle_0_1Ratio'] = np.where(df['sc_volume'].shift(1) != 0,
                                                     (df['sc_volume'] - df['sc_volume'].shift(1)) / df['sc_volume'].shift(1),
                                                     diffDivBy0 if DEFAULT_DIV_BY0 else valueX)

    # Relatif delta evol
    features_df['diffVolDelta_0_0Ratio'] = np.where(df['sc_volume'] != 0,
                                                    df['sc_delta'] / df['sc_volume'], diffDivBy0 if DEFAULT_DIV_BY0 else valueX)
    features_df['diffVolDelta_1_1Ratio'] = np.where(df['sc_volume'].shift(1) != 0,
                                                    df['sc_delta'].shift(1) / df['sc_volume'].shift(1),
                                                    diffDivBy0 if DEFAULT_DIV_BY0 else valueX)
    features_df['diffVolDelta_2_2Ratio'] = np.where(df['sc_volume'].shift(2) != 0,
                                                    df['sc_delta'].shift(2) / df['sc_volume'].shift(2),
                                                    diffDivBy0 if DEFAULT_DIV_BY0 else valueX)
    features_df['diffVolDelta_3_3Ratio'] = np.where(df['sc_volume'].shift(3) != 0,
                                                    df['sc_delta'].shift(3) / df['sc_volume'].shift(3),
                                                    diffDivBy0 if DEFAULT_DIV_BY0 else valueX)
    features_df['diffVolDelta_0_1Ratio'] = np.where(df['sc_delta'].shift(1) != 0,
                                                    (df['sc_delta'] - df['sc_delta'].shift(1)) / df['sc_delta'].shift(1),
                                                    diffDivBy0 if DEFAULT_DIV_BY0 else valueX)
    # Définir la période comme variable
    # Moyenne des volumes sur les nb_periods dernières périodes (de t-1 à t-nb_periods)

    features_df['volume_perTick'] =df['sc_volume'] /(2 * df['sc_candleSizeTicks'])

    for x in [1, 3, 5, 10, 20]:
        nb_periods = x
        col_name = f'meanVol_perTick_over{nb_periods}'
        col_ratio = f'ratioVolPerTick_over{nb_periods}'

        if nb_periods == 0:
            # Cas spécial : pas de moyenne, on prend directement la bougie précédente
            features_df[col_name] = (df['sc_volume'].shift(1) / (2 * df['sc_candleSizeTicks'].shift(1)))
        else:
            # Cas normal : moyenne sur nb_periods
            features_df[col_name] = (df['sc_volume'].shift(1).rolling(window=nb_periods, min_periods=1).mean()
                                     / (2 * df['sc_candleSizeTicks'].shift(1)))

        # Volume par tick de la bougie courante
        current_vol_per_tick = df['sc_volume'] / (2 * df['sc_candleSizeTicks'])

        # Ratio : volume par tick courant / moyenne volume par tick précédentes
        features_df[col_ratio] = np.where(
            features_df[col_name] != 0,  # Si la moyenne précédente est non nulle
            current_vol_per_tick / features_df[col_name],  # -> Ratio normal
            addDivBy0 if DEFAULT_DIV_BY0 else valueX  # Sinon -> valeur spécifique
        )



    # Somme des deltas sur les mêmes nb_periods périodes
    # Cumul des deltas sur 4 bougies (bougie actuelle incluse)
    delta_sum_4 = sum(df['sc_delta'].shift(i) for i in range(0, 4))  # Bougie actuelle + 3 précédentes

    # Moyenne des volumes sur 4 bougies (bougie actuelle incluse)
    mean_vol_4 = df['sc_volume'].rolling(window=4, min_periods=1).mean()

    # Calcul du ratio cumulé delta / volume
    features_df['cum_4DiffVolDeltaRatio'] = np.where(
        mean_vol_4 != 0,
        delta_sum_4 / mean_vol_4,
        diffDivBy0 if DEFAULT_DIV_BY0 else valueX
    )
    # Nouvelles features - Features de Volume Profile:
    # Importance du POC
    features_df['volcontZone_zoneReversal'] = np.where(
        (df['sc_candleDir'] == -1) & (df['sc_candleSizeTicks'] >= 11),
        df['sc_volAbv'] + df['sc_vol_xTicksContZone'],
        np.where(
            (df['sc_candleDir'] == 1) & (df['sc_candleSizeTicks'] >= 11),
            df['sc_volBlw'] + df['sc_vol_xTicksContZone'],
            0  # 0 si les conditions ne sont pas remplies
        )
    )

    volRev = np.where(df['sc_candleDir'] == -1, df['sc_volAbv'], df['sc_volBlw'])
    deltaRev = np.where(df['sc_candleDir'] == -1, df['sc_deltaAbv'], df['sc_deltaBlw'])



    features_df['volPocVolCandleRatio'] = np.where(df['sc_volume'] != 0, df['sc_volPOC'] / df['sc_volume'],
                                                   addDivBy0 if DEFAULT_DIV_BY0 else valueX)

    features_df['volPocVolRevesalXContRatio'] = np.where(features_df['volcontZone_zoneReversal'] != 0, df['sc_volPOC'] / features_df['volcontZone_zoneReversal'],
                                                   addDivBy0 if DEFAULT_DIV_BY0 else valueX)

    features_df['pocDeltaPocVolRatio'] = np.where(df['sc_volPOC'] != 0, df['sc_deltaPOC'] / df['sc_volPOC'],
                                                  diffDivBy0 if DEFAULT_DIV_BY0 else valueX)

    # Asymétrie du volume
    features_df['volAbv_vol_ratio'] = np.where(df['sc_volume'] != 0, (df['sc_volAbv']) / df['sc_volume'],
                                               diffDivBy0 if DEFAULT_DIV_BY0 else valueX)
    features_df['volBlw_vol_ratio'] = np.where(df['sc_volume'] != 0, (df['sc_volBlw']) / df['sc_volume'],
                                               diffDivBy0 if DEFAULT_DIV_BY0 else valueX)

    features_df['asymetrie_volume'] = np.where(df['sc_volume'] != 0, (df['sc_volAbv'] - df['sc_volBlw']) / df['sc_volume'],
                                               diffDivBy0 if DEFAULT_DIV_BY0 else valueX)

    features_df['volRevVolRevesalXContRatio'] = np.where(features_df['volcontZone_zoneReversal'] != 0, volRev / features_df['volcontZone_zoneReversal'],
                                                   addDivBy0 if DEFAULT_DIV_BY0 else valueX)
    features_df['volRevVolCandle_ratio'] = np.where(df['sc_volume'] != 0, volRev  / df['sc_volume'],
                                                   addDivBy0 if DEFAULT_DIV_BY0 else valueX)
    features_df['deltaRev_volRev_ratio'] = np.where(volRev != 0, deltaRev / volRev,
                                                         addDivBy0 if DEFAULT_DIV_BY0 else valueX)



    # Nouvelles features - Features Cumulatives sur les 5 dernières bougies:
    # Volume spike
    for x in [5, 10, 20, 30]:
        nb_periods = x

        # Nom des colonnes à générer
        col_mean = f'volMeanOver{nb_periods}'
        col_ratio = f'volCandleMeanOver{nb_periods}Ratio'

        # ---- 1️⃣ Calcul de la moyenne des volumes sur nb_periods bougies précédentes (sans la bougie actuelle) ----
        features_df[col_mean] = df['sc_volume'].shift(1).rolling(window=nb_periods, min_periods=1).mean()

        # ---- 2️⃣ Calcul du ratio : volume actuel / moyenne des volumes précédents ----
        features_df[col_ratio] = np.where(
            features_df[col_mean] != 0,  # Si la moyenne précédente est non nulle
            df['sc_volume'] / features_df[col_mean],  # -> On calcule le ratio normal
            addDivBy0 if DEFAULT_DIV_BY0 else valueX  # Sinon (moyenne = 0) -> valeur spécifique
        )

    Imb_Div0=0
    Imb_zone=0
    # Nouvelles features - Order Flow:
    # Imbalances haussières
    features_df['bull_imbalance_low_1'] = np.where(
        df['sc_bidVolLow'] == 0,
        Imb_Div0,
        np.where(
            (df['sc_bidVolLow'] >= 0) & (df['sc_bidVolLow'] <= 0),
            Imb_zone,
            df['sc_askVolLow_1'] / df['sc_bidVolLow']
        )
    )

    volRev_perTick=volRev/(2*5)
    features_df['volRev_perTick']=volRev_perTick
    features_df['volxTicksContZone_perTick']=df['sc_vol_xTicksContZone']/(2*5)

    features_df['volRev_perTick_Vol_perTick_over1']=volRev_perTick/ features_df['meanVol_perTick_over1']

    features_df['volRev_perTick_volxTicksContZone_perTick_ratio']=volRev_perTick/ features_df['volxTicksContZone_perTick']


    # Imbalances haussières
    # Version simplifiée avec intervalle
    features_df['bull_imbalance_low_2'] = np.where(
        df['sc_bidVolLow_1'] == 0,
        Imb_Div0,
        np.where(
            (df['sc_bidVolLow_1'] >= 0) & (df['sc_bidVolLow_1'] <= 0),
            Imb_zone,
            df['sc_askVolLow_2'] / df['sc_bidVolLow_1']
        )
    )

    # Version simplifiée avec intervalle
    features_df['bull_imbalance_low_3'] = np.where(
        df['sc_bidVolLow_2'] == 0,
        Imb_Div0,
        np.where(
            (df['sc_bidVolLow_2'] >= 0) & (df['sc_bidVolLow_2'] <= 0),
            Imb_zone,
            df['sc_askVolLow_3'] / df['sc_bidVolLow_2']
        )
    )


    features_df['bull_imbalance_high_0'] = np.where(
        df['sc_bidVolHigh_1'] == 0,
        Imb_Div0,
        np.where(
            (df['sc_bidVolHigh_1'] >= 0) & (df['sc_bidVolHigh_1'] <= 0),
            Imb_zone,
            df['sc_askVolHigh'] / df['sc_bidVolHigh_1']
        )
    )

    features_df['bull_imbalance_high_1'] = np.where(
        df['sc_bidVolHigh_2'] == 0,
        Imb_Div0,
        np.where(
            (df['sc_bidVolHigh_2'] >= 0) & (df['sc_bidVolHigh_2'] <= 0),
            Imb_zone,
            df['sc_askVolHigh_1'] / df['sc_bidVolHigh_2']
        )
    )

    features_df['bull_imbalance_high_2'] = np.where(
        df['sc_bidVolHigh_3'] == 0,
        Imb_Div0,
        np.where(
            (df['sc_bidVolHigh_3'] >= 0) & (df['sc_bidVolHigh_3'] <= 0),
            Imb_zone,
            df['sc_askVolHigh_2'] / df['sc_bidVolHigh_3']
        )
    )

    # Imbalances baissières
    features_df['bear_imbalance_low_0'] = np.where(
        df['sc_askVolLow_1'] != 0,
        df['sc_bidVolLow'] / df['sc_askVolLow_1'],
        Imb_Div0 if DEFAULT_DIV_BY0 else (
            calculate_max_ratio(
                df['sc_bidVolLow'] / df['sc_askVolLow_1'],
                df['sc_askVolLow_1'] != 0
            )
        )
    )

    features_df['bear_imbalance_low_1'] = np.where(
        df['sc_askVolLow_2'] != 0,
        df['sc_bidVolLow_1'] / df['sc_askVolLow_2'],
        Imb_Div0 if DEFAULT_DIV_BY0 else (
            calculate_max_ratio(
                df['sc_bidVolLow_1'] / df['sc_askVolLow_2'],
                df['sc_askVolLow_2'] != 0
            )
        )
    )

    features_df['bear_imbalance_low_2'] = np.where(
        df['sc_askVolLow_3'] != 0,
        df['sc_bidVolLow_2'] / df['sc_askVolLow_3'],
        Imb_Div0 if DEFAULT_DIV_BY0 else (
            calculate_max_ratio(
                df['sc_bidVolLow_2'] / df['sc_askVolLow_3'],
                df['sc_askVolLow_3'] != 0
            )
        )
    )

    # Imbalances baissières
    features_df['bear_imbalance_low_0'] = np.where(
        df['sc_askVolLow_1'] == 0,
        Imb_Div0,
        np.where(
            (df['sc_askVolLow_1'] >= 0) & (df['sc_askVolLow_1'] <= 0),
            Imb_zone,
            df['sc_bidVolLow'] / df['sc_askVolLow_1']
        )
    )

    features_df['bear_imbalance_low_1'] = np.where(
        df['sc_askVolLow_2'] == 0,
        Imb_Div0,
        np.where(
            (df['sc_askVolLow_2'] >= 0) & (df['sc_askVolLow_2'] <= 0),
            Imb_zone,
            df['sc_bidVolLow_1'] / df['sc_askVolLow_2']
        )
    )

    features_df['bear_imbalance_low_2'] = np.where(
        df['sc_askVolLow_3'] == 0,
        Imb_Div0,
        np.where(
            (df['sc_askVolLow_3'] >= 0) & (df['sc_askVolLow_3'] <= 0),
            Imb_zone,
            df['sc_bidVolLow_2'] / df['sc_askVolLow_3']
        )
    )

    features_df['bear_imbalance_high_1'] = np.where(
        df['sc_askVolHigh'] != 0,
        df['sc_bidVolHigh_1'] / df['sc_askVolHigh'],
        Imb_Div0 if DEFAULT_DIV_BY0 else (
            calculate_max_ratio(
                df['sc_bidVolHigh_1'] / df['sc_askVolHigh'],
                df['sc_askVolHigh'] != 0
            )
        )
    )

    features_df['bear_imbalance_high_2'] = np.where(
        df['sc_askVolHigh_1'] != 0,
        df['sc_bidVolHigh_2'] / df['sc_askVolHigh_1'],
        Imb_Div0 if DEFAULT_DIV_BY0 else (
            calculate_max_ratio(
                df['sc_bidVolHigh_2'] / df['sc_askVolHigh_1'],
                df['sc_askVolHigh_1'] != 0
            )
        )
    )

    features_df['bear_imbalance_high_3'] = np.where(
        df['sc_askVolHigh_2'] != 0,
        df['sc_bidVolHigh_3'] / df['sc_askVolHigh_2'],
        Imb_Div0 if DEFAULT_DIV_BY0 else (
            calculate_max_ratio(
                df['sc_bidVolHigh_3'] / df['sc_askVolHigh_2'],
                df['sc_askVolHigh_2'] != 0
            )
        )
    )
    # Score d'Imbalance Asymétrique
    sell_pressureLow = df['sc_bidVolLow'] + df['sc_bidVolLow_1']
    buy_pressureLow = df['sc_askVolLow_1'] + df['sc_askVolLow_2']
    total_volumeLow = buy_pressureLow + sell_pressureLow
    features_df['imbalance_score_low'] = np.where(total_volumeLow != 0,
                                                  (buy_pressureLow - sell_pressureLow) / total_volumeLow,
                                                  diffDivBy0 if DEFAULT_DIV_BY0 else valueX)

    sell_pressureHigh = df['sc_bidVolHigh_1'] + df['sc_bidVolHigh_2']
    buy_pressureHigh = df['sc_askVolHigh'] + df['sc_askVolHigh_1']
    total_volumeHigh = sell_pressureHigh + buy_pressureHigh
    features_df['imbalance_score_high'] = np.where(total_volumeHigh != 0,
                                                   (sell_pressureHigh - buy_pressureHigh) / total_volumeHigh,
                                                   diffDivBy0 if DEFAULT_DIV_BY0 else valueX)

    # Finished Auction
    features_df['finished_auction_high'] = (df['sc_bidVolHigh'] == 0).astype(int)
    features_df['finished_auction_low'] = (df['sc_askVolLow'] == 0).astype(int)
    features_df['staked00_high'] = ((df['sc_bidVolHigh'] == 0) & (df['sc_bidVolHigh_1'] == 0)).astype(int)
    features_df['staked00_low'] = ((df['sc_askVolLow'] == 0) & (df['sc_askVolLow_1'] == 0)).astype(int)

    #dist_above, dist_below = calculate_naked_poc_distances(df)

    #features_df["naked_poc_dist_above"] = dist_above
    #features_df["naked_poc_dist_below"] = dist_below
    print_notification("Ajout des informations sur les class et les trades")


    features_df['diffPriceCloseVAH_0'] = df ['sc_close']- df ['sc_VA_high_0']
    features_df['diffPriceCloseVAL_0'] = df ['sc_close']- df ['sc_VA_low_0']
    features_df['ratio_delta_vol_VA_0'] = np.where(
        df['sc_VA_vol_0'] != 0,  # Condition
        df['sc_VA_delta_0'] / df['sc_VA_vol_0'],  # Valeur si la condition est vraie
        np.nan  # Valeur si la condition est fausse
    )

    #add processing metrics
    features_df['class_binaire'] = df['class_binaire']
    features_df['date'] = df['date']
    features_df['trade_category'] = df['trade_category']

    # Enregistrement des fichiers
    print_notification("Début de l'enregistrement des fichiers")

    # Extraire le nom du fichier et le répertoire
    file_dir = os.path.dirname(file_path_)
    file_name = os.path.basename(file_path_)


    # Vérification des colonnes manquantes
    colonnes_manquantes = [col for col in colonnes_a_transferer if col not in df.columns]

    # Si des colonnes sont manquantes, lever une erreur avec la liste détaillée
    if colonnes_manquantes:
        raise ValueError(f"Les colonnes suivantes sont manquantes dans le DataFrame source :\n" +
                         "\n".join(f"- {col}" for col in colonnes_manquantes))

    # Si toutes les colonnes sont présentes, effectuer le transfert
    for colonne in colonnes_a_transferer:
        features_df[colonne] = df[colonne]

    #recopie les données sierra chart et met des 0 si pas asser de données en début de session
    # Liste des fenêtres
    # Usage example:
    windows_list = [5, 10, 15, 30]
    session_starts = (df['sc_sessionStartEnd'] == 10).values
    df_results = process_reg_slope_replacement(df, session_starts, windows_list, reg_feature_prefix="sc_reg_slope_")
    # Fusionner avec features_df (assurez-vous que l'index est aligné)
    features_df = pd.concat([features_df, df_results], axis=1)

    df_results = process_reg_slope_replacement(df, session_starts, windows_list, reg_feature_prefix="sc_reg_std_")
    # Fusionner avec features_df (assurez-vous que l'index est aligné)
    features_df = pd.concat([features_df, df_results], axis=1)

    print("Transfert réussi ! Toutes les colonnes ont été copiées avec succès.")



    # Liste de toutes les colonnes à inclure si la condition est remplie
    colonnes_a_inclure = [
        "sc_ratio_vol_volCont_zoneA_xTicksContZone",
        "sc_ratio_delta_volCont_zoneA_xTicksContZone",
        "sc_ratio_vol_volCont_zoneB_xTicksContZone",
        "sc_ratio_delta_volCont_zoneB_xTicksContZone",
        "sc_ratio_vol_volCont_zoneC_xTicksContZone",
        "sc_ratio_delta_volCont_zoneC_xTicksContZone"
    ]

    # Vérifier si la colonne spécifique est présente dans df
    if "sc_ratio_vol_volCont_zoneA_xTicksContZone" in df.columns:
        # Vérifier que toutes les colonnes existent dans df
        colonnes_existantes = [col for col in colonnes_a_inclure if col in df.columns]

        # Si features_df n'existe pas encore, le créer avec ces colonnes
        if 'features_df' not in locals():
            features_df = df[colonnes_existantes].copy()
        # Sinon, ajouter ces colonnes à features_df existant
        else:
            for col in colonnes_existantes:
                features_df[col] = df[col]

        print(f"Colonnes ajoutées à features_df: {colonnes_existantes}")
    else:
        print("La colonne 'sc_ratio_vol_volCont_ZoneA_xTicksContZone' n'est pas présente dans df")
        exit(12)


    features_df['candleDuration'] = df['sc_timeStampOpening'].shift(-1) - df['sc_timeStampOpening']
    # Création de la colonne avec des valeurs par défaut de 0
    #
    features_df['vix_slope_12_up_15'] = 0

    features_df = compute_consecutive_trend_feature(
        df=df,
        features_df=features_df,
        target_col='sc_vix_slope_12',
        n=15,
        trend_type='up',
        output_col='vix_slope_12_up_15'
    )

    # ✅ S'assurer que df contient la colonne session_id
    df["session_id"] = (df["sc_sessionStartEnd"] == 10).cumsum().astype("int32")

    # ✅ S'assurer que features_df a aussi la colonne session_id
    features_df["session_id"] = df["session_id"].copy()
    features_df = add_moving_percentiles4VolEtDuration(features_df, df)

    print("xxxxxxxxxxxxxxxxx DEBUT ZONE SPECIFIQUE xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

    # ════════════════════════════════════════════════════════════════════════════════
    # DEBUT DEFINITION SPECIAL
    # ════════════════════════════════════════════════════════════════════════════════

    # ════════════════════════════════════════════════════════════════════════════════
    # 1️⃣  PARAMS – valeurs optimales regroupées  (maj 18 / 06 / 2025)
    # ════════════════════════════════════════════════════════════════════════════════

    PARAMS = {
        # ----------------------------------------------------------------------------
        # SPECIAL 1 : LIGHT - Volume Imbalance avec POC
        # ----------------------------------------------------------------------------
        "light_short": {
            'volume_1': 3, 'imbalance_1': 2.8765577559341775,
            'volume_2': 16, 'imbalance_2': 3.7502271206794373,
            'volume_3': 20, 'imbalance_3': 1.6128252330193062
        },
        "light_long": {
            'volume_1': 3, 'imbalance_1': 4.1399822426260675,
            'volume_2': 8, 'imbalance_2': 2.098378937495904,
            'volume_3': 43, 'imbalance_3': 6.91531591271347
        },

        # ----------------------------------------------------------------------------
        # SPECIAL 2 : AGGRESSIVE - Volume Imbalance avec POC
        # ----------------------------------------------------------------------------
        "agg_short": {
            'volume_1': 3, 'imbalance_1': 5.374412696579563,
            'volume_2': 17, 'imbalance_2': 5.096884736935089,
            'volume_3': 26, 'imbalance_3': 6.346103887296557
        },
        "agg_long": {
            'volume_1': 6, 'imbalance_1': 4.170026814921757,
            'volume_2': 16, 'imbalance_2': 3.434828137256047,
            'volume_3': 27, 'imbalance_3': 7.359269622191843
        },

        # ----------------------------------------------------------------------------
        # SPECIAL 3 : MFI (Money Flow Index)
        # ----------------------------------------------------------------------------
        "mfi": {
            "oversold_period": 40, "oversold_threshold": 35.0,
            "overbought_period": 5, "overbought_threshold": 78.0
        },

        # ----------------------------------------------------------------------------
        # SPECIAL 4 : RS RANGE (Rogers-Satchell)
        # ----------------------------------------------------------------------------
        # "rs": {
        #     "period_short": 30, "low_short": 0.000202, "high_short": 0.000220,
        #     "period_long": 29, "low_long": 0.000204, "high_long": 0.000210
        # },

        "rs": {
            "period_short": 11, "low_short": 0.000174 , "high_short": 0.000180 ,
            "period_long": 17 , "low_long": 0.000189 , "high_long": 0.000194
        },

        # ----------------------------------------------------------------------------
        # SPECIAL 5 : WILLIAMS %R
        # ----------------------------------------------------------------------------
        "wr": {
            "period_short": 15, "th_short": -10.0,
            "period_long": 51, "th_long": -95.0
        },

        # ----------------------------------------------------------------------------
        # SPECIAL 6 : VWAP REVERSAL PRO
        # ----------------------------------------------------------------------------
        "vwap_short": {
            'lookback': 19, 'momentum': 18, 'z_window': 22,
            'atr_period': 19, 'atr_mult': 2.70,
            'ema_filter': 29, 'vol_lookback': 12, 'vol_ratio_min': 0.60
        },
        "vwap_long": {
            'lookback': 38, 'momentum': 11, 'z_window': 45,
            'atr_period': 32, 'atr_mult': 1.20,
            'ema_filter': 77, 'vol_lookback': 8, 'vol_ratio_min': 0.25
        },

        # ----------------------------------------------------------------------------
        # SPECIAL 7 : IMB Light POC avec ATR
        # ----------------------------------------------------------------------------
        "atr_high": {
            "atr_threshold_1": 1.5, "atr_threshold_2": 1.7, "atr_threshold_3": 1.9,
            "diff_high_atr_1": 5.5, "diff_high_atr_2": 3.75,
            "diff_high_atr_3": 5.75, "diff_high_atr_4": 3.25,
            "atr_window": 12
        },
        "atr_low": {
            "atr_threshold_1": 1.5, "atr_threshold_2": 1.7, "atr_threshold_3": 1.9,
            "diff_low_atr_1": 5.5, "diff_low_atr_2": 3.75,
            "diff_low_atr_3": 5.75, "diff_low_atr_4": 3.25,
            "atr_window": 12
        },
        # ----------------------------------------------------------------------------
        # SPECIAL 8 : Volume Per Tick - Profil Light (plus de trades)
        # ----------------------------------------------------------------------------
        "volume_per_tick_light_short": {
            "period": 32, "threshold_low": 0.65, "threshold_high": 2.75,
            "direction": "Short", "profile": "light",
            "expected_trade_pct": 8.5, "expected_winrate": 0.55  # 7.53-11.49% trades, WR ~52-63%
            # Features générées: vptRatio_light_ratio_period_32, is_vptRatio_light_signal_outband, etc.
        },
        "volume_per_tick_light_long": {
            "period": 51, "threshold_low": 1.00, "threshold_high": 3.17,
            "direction": "Long", "profile": "light",
            "expected_trade_pct": 36.0, "expected_winrate": 0.525  # 34.54-38.87% trades, WR ~51-53%
            # Features générées: vptRatio_light_ratio_period_51, is_vptRatio_light_signal_outband, etc.
        },

        # ----------------------------------------------------------------------------
        # SPECIAL 9 : Volume Per Tick - Profil Aggressive (moins de trades, meilleur WR)
        # ----------------------------------------------------------------------------
        "volume_per_tick_aggressive_short": {
            "period": 19, "threshold_low": 0.50, "threshold_high": 3.05,
            "direction": "Short", "profile": "aggressive",
            "expected_trade_pct": 2.9, "expected_winrate": 0.57  # 2.28-3.68% trades, WR ~54-61%
            # Features générées: vptRatio_aggressive_ratio_period_19, is_vptRatio_aggressive_signal_outband, etc.
        },
        "volume_per_tick_aggressive_long": {
            "period": 51, "threshold_low": 1.00, "threshold_high": 3.17,
            "direction": "Long", "profile": "aggressive",
            "expected_trade_pct": 36.0, "expected_winrate": 0.525  # Même performance que light Long
            # Features générées: vptRatio_aggressive_ratio_period_51, is_vptRatio_aggressive_signal_outband, etc.
        },

        # ----------------------------------------------------------------------------
        # SPECIAL 10 : Volume Per Tick - Configuration par défaut (recommandée)
        # ----------------------------------------------------------------------------
        "volume_per_tick": {
            "period": 32, "threshold_low": 0.65, "threshold_high": 2.75,
            "direction": "Long"  # Direction par défaut - Compromis optimal
            # Features générées: vptRatio_ratio_period_32, is_vptRatio_signal_outband, etc.
        },

        # ----------------------------------------------------------------------------
        # SPECIAL 11 : Volume Per Tick - Configurations spécialisées
        # ----------------------------------------------------------------------------
        "volume_per_tick_conservative": {
            "period": 51, "threshold_low": 1.00, "threshold_high": 3.17,
            "direction": "Long", "profile": "conservative"  # Maximum de trades avec WR décent
            # Features générées: vptRatio_conservative_ratio_period_51, is_vptRatio_conservative_signal_outband, etc.
        },
        "volume_per_tick_selective": {
            "period": 19, "threshold_low": 0.50, "threshold_high": 3.05,
            "direction": "Short", "profile": "selective"  # Minimum de trades avec WR élevé
            # Features générées: vptRatio_selective_ratio_period_19, is_vptRatio_selective_signal_outband, etc.
        },
    # ----------------------------------------------------------------------------
    # SPECIAL 12 – Microstructure Anti-Spring & Anti-Épuisement
    # ----------------------------------------------------------------------------
        "micro_antispring": {
            "short_period": 5,
            "short_vol_threshold": -0.860,
            "short_dur_threshold": 6.280,
            "short_condition": 0  # V<D<
        },
        "micro_antiepuisement": {
            "long_period": 6,
            "long_vol_threshold": -4.820,
            "long_dur_threshold": 2.350,
            "long_condition": 3  # V>D>
        }

    }

    generate_trading_config_header(
        params=PARAMS,
        output_path=r"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra_chart\xTickReversal\include\Special_indicator_autoGenPy.h"
    )

    # 2. Optionnel : Génération du chargeur CSV
    generate_csv_loader_header(
        csv_path=r"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra_chart\xTickReversal\include\Special_indicator_config.csv",
        output_h_path=r"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra_chart\xTickReversal\include\Special_indicator_loader_header_autoGenPy.h"
    )

    print("\n🎯 RÉSUMÉ :")
    print("✅ Fichier .h généré avec namespace Special_indicator_hardcode")
    print("✅ Fichier CSV généré pour utilisation dynamique")
    print("✅ Chargeur CSV généré pour namespace Special_indicator_fromCsv")

    #exit(112)
    # ════════════════════════════════════════════════════════════════════════════════
    # 2️⃣  APPLY – appel séquentiel des fonctions
    # ════════════════════════════════════════════════════════════════════════════════

    # ----------------------------------------------------------------------------
    # SPECIAL 1 – Light - Volume Imbalance avec POC
    # ----------------------------------------------------------------------------
    features_df = add_ImBullWithPoc(df, features_df, "is_imBullWithPoc_light_short",
                                    PARAMS["light_short"], "short")
    features_df = add_ImBullWithPoc(df, features_df, "is_imBearWithPoc_light_long",
                                    PARAMS["light_long"], "long")

    # ----------------------------------------------------------------------------
    # SPECIAL 2 – Aggressive - Volume Imbalance avec POC
    # ----------------------------------------------------------------------------
    features_df = add_ImBullWithPoc(df, features_df, "is_imBullWithPoc_aggressive_short",
                                    PARAMS["agg_short"], "short")
    features_df = add_ImBullWithPoc(df, features_df, "is_imBearWithPoc_aggressive_long",
                                    PARAMS["agg_long"], "long")

    # ----------------------------------------------------------------------------
    # SPECIAL 3 – MFI (Money Flow Index)
    # ----------------------------------------------------------------------------
    features_df = add_mfi(
        df, features_df,
        overbought_period=PARAMS["mfi"]["overbought_period"],
        oversold_period=PARAMS["mfi"]["oversold_period"],
        overbought_threshold=PARAMS["mfi"]["overbought_threshold"],
        oversold_threshold=PARAMS["mfi"]["oversold_threshold"]
    )

    # ----------------------------------------------------------------------------
    # SPECIAL 4 – RS Range (Rogers-Satchell)
    # ----------------------------------------------------------------------------
    features_df = add_rs(
        df, features_df,
        period_short_range=PARAMS["rs"]["period_short"],
        rs_short_low_threshold_range=PARAMS["rs"]["low_short"],
        rs_short_high_threshold_range=PARAMS["rs"]["high_short"],
        period_long_range=PARAMS["rs"]["period_long"],
        rs_long_low_threshold_range=PARAMS["rs"]["low_long"],
        rs_long_high_threshold_range=PARAMS["rs"]["high_long"]
    )

    # ----------------------------------------------------------------------------
    # SPECIAL 5 – Williams %R
    # ----------------------------------------------------------------------------
    features_df = add_williams_r_harmonized(
        df, features_df,
        period_short_overbought=PARAMS["wr"]["period_short"],
        threshold_short_overbought=PARAMS["wr"]["th_short"],
        period_long_oversold=PARAMS["wr"]["period_long"],
        threshold_long_oversold=PARAMS["wr"]["th_long"],
        use_adaptive_logic=True,  # 🔑 Utilise la logique de l'optimizer
        debug_mode=True  # 🔍 Affiche les statistiques détaillées
    )

    # ----------------------------------------------------------------------------
    # SPECIAL 6 – VWAP Reversal Pro
    # ----------------------------------------------------------------------------
    # df : DataFrame complet  •  features_df : DataFrame des features
    # ↓ Pour aligner strictement avec la version C++, on fixe atr_ma='sma'
    features_df = add_vwap_reversal_pro(
        df_full=df, features_df=features_df, direction='short',
        atr_ma='sma',  # ← MM ATR identique C++
        **PARAMS["vwap_short"]
    )
    features_df = add_vwap_reversal_pro(
        df_full=df, features_df=features_df, direction='long',
        atr_ma='sma',
        **PARAMS["vwap_long"]
    )

    # ----------------------------------------------------------------------------
    # SPECIAL 7 – IMB Light POC + ATR
    # ----------------------------------------------------------------------------
    features_df = add_imbBullLightPoc_Atr_HighLow(
        df, features_df, "is_imbBullLightPoc_AtrHigh0_1_short", PARAMS["atr_high"]
    )
    features_df = add_imbBullLightPoc_Atr_HighLow(
        df, features_df, "is_imbBearLightPoc_AtrLow0_0_long", PARAMS["atr_low"]
    )

    # ----------------------------------------------------------------------------
    # SPECIAL 8 to 11 : Volume Per Tick - Tous profils (Light, Aggressive, Default, Conservative, Selective)
    # ----------------------------------------------------------------------------

    print("\n🎯 2. Application de tous les profils Volume Per Tick :")

    volume_per_tick_configs = [
        "volume_per_tick_light_short",  # period=32 dans PARAMS
        "volume_per_tick_light_long",  # period=51 dans PARAMS
        "volume_per_tick_aggressive_short",  # period=19 dans PARAMS
        "volume_per_tick_aggressive_long",  # period=51 dans PARAMS
        "volume_per_tick",  # period=32 dans PARAMS
        "volume_per_tick_conservative",  # period=51 dans PARAMS
        "volume_per_tick_selective"  # period=19 dans PARAMS
    ]

    for config_name in volume_per_tick_configs:
        features_df = add_volume_per_tick(df, features_df,
                                          config_name=config_name,
                                          PARAMS=PARAMS)

    # ----------------------------------------------------------------------------
    # SPECIAL 12 – Microstructure Anti-Spring & Anti-Épuisement
    # ----------------------------------------------------------------------------
    features_df = add_micro_antiSpringEpuissement(
        df, features_df,
        short_period=PARAMS["micro_antispring"]["short_period"],
        short_vol_threshold=PARAMS["micro_antispring"]["short_vol_threshold"],
        short_dur_threshold=PARAMS["micro_antispring"]["short_dur_threshold"],
        short_condition=PARAMS["micro_antispring"]["short_condition"],
        long_period=PARAMS["micro_antiepuisement"]["long_period"],
        long_vol_threshold=PARAMS["micro_antiepuisement"]["long_vol_threshold"],
        long_dur_threshold=PARAMS["micro_antiepuisement"]["long_dur_threshold"],
        long_condition=PARAMS["micro_antiepuisement"]["long_condition"]
    )

    # --------------------  FIN  --------------------

    features_df = add_rsi(df, features_df, period=5)
    features_df = add_macd(df, features_df, short_period=4, long_period=8, signal_period=5)

    ##is_rangeSlope is_extremSlope
    features_df = add_regression_slope(df, features_df, period_range=28, period_extrem=30,
                         slope_range_threshold_low=0.2715994135835932 , slope_range_threshold_high=0.3842233665393566 ,
                         slope_extrem_threshold_low=-0.299867692475089, slope_extrem_threshold_high=0.5280173704178184 )

    ##is_atr_extremLow is_atr_range
    features_df = add_atr(df, features_df, atr_period_range=12, atr_period_extrem=25,
                atr_low_threshold_range= 2.2906256448366022, atr_high_threshold_range= 2.6612737528788495,
                atr_low_threshold_extrem=1.5360453527266502)

    #is_zscore_range is_zscore_extrem  zscore_extrem zscore_range
    features_df = add_zscore(df, features_df,
                  period_range=48, period_extrem=0,
                  zscore_range_threshold_low=-0.3435, zscore_range_threshold_high=0.2173,
                  zscore_extrem_threshold_low=-0, zscore_extrem_threshold_high=0)

    #is_range_volatility std_range  is_range_volatility  is_extrem_volatility
    features_df = add_std_regression(df, features_df,
                      period_range=13, period_extrem=46,
                      std_low_threshold_range=0.8468560606715232, std_high_threshold_range=0.9342162977682715,
                      std_low_threshold_extrem=1.3744010118158592 ,std_high_threshold_extrem=5.0650563837214735)

    #r2_range r2_extrem is_range_volatility_r2  is_extrem_volatility_r2
    features_df = add_r2_regression(df, features_df,
                        period_range=21, period_extrem=53,
                        r2_low_threshold_range=0.10281944161014597, r2_high_threshold_range= 0.18548155526479806,
                        r2_low_threshold_extrem= 0.020222174899276364, r2_high_threshold_extrem=0.8665041310372354)
    #is_stoch_oversold is_stoch_overbought
    features_df = add_stochastic_force_indicators(df, features_df,
                                        k_period_overbought=42, d_period_overbought=41,
                                        k_period_oversold=105, d_period_oversold=169,
                                        overbought_threshold=93, oversold_threshold=21,
                                        fi_short=4, fi_long=4)


    #is_mfi_shortDiv is_mfi_antiShortDiv
    features_df = add_mfi_divergence(df, features_df,
                           mfi_period_bearish=7, div_lookback_bearish=11,
                            mfi_period_antiBear=14,div_lookback_antiBear=18,
                           min_price_increase=0.00074, min_mfi_decrease=8.48e-05,
                           min_price_decrease=0.00018, min_mfi_increase=0.00093)




    column_settings=init_column_settings(candle_rev_tick)

    columns_to_process = list(column_settings.keys())

    # Vérification de l'existence des colonnes
    # Vérification des colonnes manquantes dans features_df
    missing_columns = [column for column in columns_to_process if column not in features_df.columns]

    # Vérification des colonnes supplémentaires dans features_df
    columns_to_exclude = ['class_binaire', 'date', 'trade_category']

    extra_columns = [column for column in features_df.columns
                     if column not in columns_to_process
                     and column not in columns_to_exclude]
    if missing_columns or extra_columns:
        if missing_columns:
            print("Erreur : Les colonnes suivantes sont manquantes dans features_df :")
            for column in missing_columns:
                print(f"- {column}")

        if extra_columns:
            print("Erreur : Les colonnes suivantes sont présentes dans features_df mais pas dans columns_to_process :")
            for column in extra_columns:
                print(f"- {column}")

        print("Le processus va s'arrêter en raison de différences dans les colonnes.")
        exit(1)  # Arrête le script avec un code d'erreur

    print(
        "Toutes les features nécessaires sont présentes et aucune colonne supplémentaire n'a été détectée. Poursuite du traitement.")

    # Utilisation
    # Appliquer la fonction à features_df
    features_NANReplacedVal_df, nan_replacement_values = replace_nan_and_inf(features_df.copy(), columns_to_process,
                                                                            REPLACE_NAN)
    # Initialisation des DataFrames avec le même index que le DataFrame d'entrée
    outliersTransform_df = pd.DataFrame(index=features_NANReplacedVal_df.index)
    winsorized_scaledWithNanValue_df = pd.DataFrame(index=features_NANReplacedVal_df.index)

    total_features = len(columns_to_process)

    for i, columnName in enumerate(columns_to_process):
        # Récupérer les paramètres de la colonne
        (
            transformation_method,
            transformation_params,
            floorInf_booleen,
            cropSup_booleen,
            floorInf_percent,
            cropSup_percent,
            _
        ) = column_settings[columnName]

        # Selon la méthode, on applique le traitement adéquat
        if transformation_method == "winsor":
            # On applique la winsorisation
            outliersTransform_df[columnName] = apply_winsorization(
                features_NANReplacedVal_df,
                columnName,
                floorInf_booleen,
                cropSup_booleen,
                floorInf_percent,
                cropSup_percent,
                nan_replacement_values
            )
        elif transformation_method == "log":
            #outliersTransform_df[columnName] = np.log(features_NANReplacedVal_df['columnName'])

            #outliersTransform_df[columnName] = transformer.fit_transform(features_NANReplacedVal_df[[columnName]])
            outliersTransform_df[columnName] = np.sqrt(features_NANReplacedVal_df[columnName])

        elif transformation_method == "Yeo-Johnson":
            # Exemple: placeholder d'une fonction Yeo-Johnson
            # ==============================================
            def apply_yeo_johnson(features_NANReplacedVal_df, columnName, transformation_params=None, standardize=False):
                """
                Applique la transformation Yeo-Johnson via scikit-learn sur une seule colonne.
                Si la colonne contient des NaN, la transformation est ignorée.
                """
                # Vérification des valeurs NaN
                from sklearn.preprocessing import PowerTransformer

                if features_NANReplacedVal_df[columnName].isna().any():
                    print(f"⚠️ Warning: Column '{columnName}' contains NaN values. Transformation skipped.")
                    exit(27)
                    return features_NANReplacedVal_df[columnName]

                # Récupération des valeurs valides
                valid_values = features_NANReplacedVal_df[[columnName]].values  # Scikit-learn exige un 2D array

                if valid_values.size == 0:
                    raise ValueError(
                        f"🚨 Error: No valid values found in '{columnName}', cannot apply Yeo-Johnson transformation.")

                # Application de la transformation Yeo-Johnson via PowerTransformer
                transformer = PowerTransformer(method='yeo-johnson', standardize=standardize)
                transformed_values = transformer.fit_transform(valid_values)

                # Mise à jour des valeurs transformées dans le DataFrame
                transformed_column = features_NANReplacedVal_df[columnName].copy()
                transformed_column[:] = transformed_values.flatten()

                return transformed_column

            outliersTransform_df[columnName] = apply_yeo_johnson(
                features_NANReplacedVal_df,
                columnName,
                transformation_params
            )
            # ==============================================

        else:
            print("error aucune transformation")

            exit(45)

    features_NANReplacedVal_df = features_NANReplacedVal_df[columns_to_process]

    print(f"arpès on a features_NANReplacedVal_df:{features_NANReplacedVal_df.shape}")
    print(f"arpès on a outliersTransform_df:{outliersTransform_df.shape}")

    print("\n")
    print("Vérification finale :")
    print(f"   - Nombre de colonnes dans outliersTransform_df : {len(outliersTransform_df.columns)}")

    print(f"\n")

    # print(f"   - Nombre de colonnes dans winsorized_scaledWithNanValue_df : {len(winsorized_scaledWithNanValue_df.columns)}")
    # assert len(outliersTransform_df.columns) == len(winsorized_scaledWithNanValue_df.columns), "Le nombre de colonnes ne correspond pas entre les DataFrames"


    print_notification(
        "Ajout de 'volume', 'sc_timeStampOpening', class_binaire', 'date', 'candleDir', 'sc_volAbv','sc_VWAP','sc_high','sc_low','open','sc_close','sc_bidVolHigh_1',''sc_askVolHigh'',trade_category', 'sc_sessionStartEnd' pour permettre la suite des traitements")
    # Colonnes à ajouter
    # ───────────────────────────────────────────────────────────
    # ───────────────────────────────────────────────────────────
    # 0) LISTES DE COLONNES À INJECTER
    # ───────────────────────────────────────────────────────────
    columns_to_add1 = [
        # === Données de base ===
        'sc_volume', 'sc_delta', 'sc_timeStampOpening', 'sc_deltaTimestampOpening',
        'sc_high', 'sc_low', 'sc_close', 'sc_open',
        'sc_pocPrice',

        # === Volume par niveaux ===
        'sc_askVolLow', 'sc_askVolHigh', 'sc_bidVolLow', 'sc_bidVolHigh',
        'sc_bidVolLow_1', 'sc_bidVolHigh_1', 'sc_askVolLow_1', 'sc_askVolHigh_1',
        'sc_bidVolLow_2', 'sc_bidVolHigh_2', 'sc_askVolLow_2', 'sc_askVolHigh_2',
        'sc_bidVolLow_3', 'sc_bidVolHigh_3', 'sc_askVolLow_3', 'sc_askVolHigh_3',
        'sc_bidVolLow_4', 'sc_bidVolHigh_4', 'sc_askVolLow_4', 'sc_askVolHigh_4',

        # === Volume et delta directionnels ===
        'sc_candleDir', 'sc_volAbv', 'sc_volBlw', 'sc_deltaAbv', 'sc_deltaBlw',
        'sc_volPOC', 'sc_deltaPOC',

        # === VWAP et déviations ===
        'sc_VWAP',
        'sc_VWAPsd1Top', 'sc_VWAPsd2Top', 'sc_VWAPsd3Top', 'sc_VWAPsd4Top',
        'sc_VWAPsd1Bot', 'sc_VWAPsd2Bot', 'sc_VWAPsd3Bot', 'sc_VWAPsd4Bot',

        # === Indicateurs techniques ===
        'sc_atr', 'sc_bandWidthBB', 'sc_perctBB',
        'sc_vol_xTicksContZone',  # ✅ sc_candleSizeTicks supprimé d'ici

        # === Value Area période actuelle (0) ===
        'sc_VA_high_0', 'sc_VA_low_0', 'sc_VA_vol_0', 'sc_VA_delta_0',

        # === Value Area périodes multiples ===
        'sc_vaVol_6periods', 'sc_vaVol_11periods', 'sc_vaVol_16periods', 'sc_vaVol_21periods',
        'sc_vaDelta_6periods', 'sc_vaDelta_11periods', 'sc_vaDelta_16periods', 'sc_vaDelta_21periods',
        'sc_vaPoc_6periods', 'sc_vaPoc_11periods', 'sc_vaPoc_16periods', 'sc_vaPoc_21periods',
        'sc_vaH_6periods', 'sc_vaH_11periods', 'sc_vaH_16periods', 'sc_vaH_21periods',
        'sc_vaL_6periods', 'sc_vaL_11periods', 'sc_vaL_16periods', 'sc_vaL_21periods',

        # === Ratios de zones ===
        'sc_ratio_vol_volCont_zoneA_xTicksContZone', 'sc_ratio_delta_volCont_zoneA_xTicksContZone',
        'sc_ratio_vol_volCont_zoneB_xTicksContZone', 'sc_ratio_delta_volCont_zoneB_xTicksContZone',
        'sc_ratio_vol_volCont_zoneC_xTicksContZone', 'sc_ratio_delta_volCont_zoneC_xTicksContZone',

        # === Régression et VIX ===
        'sc_vix_slope_12',

        # === Métadonnées et sessions ===
        'sc_nb_completedBar', 'sc_sessionStartEnd', 'session_id',

        # === Classes et trades ===
        'class_binaire', 'trade_category', 'date',
        'trade_pnl', 'tp1_pnl', 'tp2_pnl', 'tp3_pnl', 'sl_pnl'
    ]

    columns_to_add2 = [
        'sc_candleDuration',
        'sc_close_sma_zscore_14',
        'sc_close_sma_zscore_21',
        'sc_close_sma_zscore_40',
        'sc_cum_4DiffVolDeltaRatio',
        'sc_candleSizeTicks',
        'sc_volume_perTick',
        'sc_volMeanOver5',
        'sc_volCandleMeanOver5Ratio',
        'sc_volMeanOver10',
        'sc_volCandleMeanOver10Ratio',
        'sc_meanVol_perTick_over1',
        'sc_meanVol_perTick_over3',
        'sc_meanVol_perTick_over5',
        'sc_volRev_perTick_Vol_perTick_over1',
        'sc_volRev_perTick_volxTicksContZone_perTick_ratio',

        'sc_diffHighPrice_0_1',
        'sc_diffPriceClose_VA6PPoc',
        'sc_diffPriceClosePoc_0_0',
        'sc_diffVolDelta_1_1Ratio',
        'sc_bull_imbalance_high_0',
        'sc_bear_imbalance_low_0',
        'sc_is_imBullWithPoc_aggressive_short',
        'sc_is_imBearWithPoc_aggressive_long',
        'sc_is_imBullWithPoc_light_short',
        'sc_is_imBearWithPoc_light_long',
        'sc_is_imbBullLightPoc_AtrHigh0_1_short',
        'sc_is_imbBearLightPoc_AtrLow0_1_long',
        'sc_is_mfi_overbought_short',
        'sc_is_mfi_oversold_long',
        'sc_is_wr_oversold_long',
        'sc_is_wr_overbought_short',
        'sc_is_rs_range_short',
        'sc_is_rs_range_long',
        'sc_is_vwap_reversal_pro_short',
        'sc_is_vwap_reversal_pro_long',
        'sc_is_antiSpring_short',
        'sc_is_antiEpuisement_long',
        'sc_volume_slope_antiSpring_short',
        'sc_duration_slope_antiSpring_short',
        'sc_volume_slope_antiEpuisement_long',
        'sc_duration_slope_antiEpuisement_long',
        'sc_pocDeltaPocVolRatio',
        'sc_ratio_delta_vol_VA11P',
        'sc_volRev',
        'sc_deltaRev',
        'sc_volcontZone_zoneReversal',
        'sc_volPocVolRevesalXContRatio',
        'sc_volRevVolRevesalXContRatio',
        'sc_deltaRev_volRev_ratio',
        'sc_reg_r2_10P_2'
    ]
    columns_to_add3 = list(dict.fromkeys(columns_to_add1 + columns_to_add2))

    # ───────────────────────────────────────────────────────────
    # 1) VÉRIFICATION DE LA PRÉSENCE DANS df
    # ───────────────────────────────────────────────────────────
    missing = [c for c in columns_to_add3 if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans df : {missing}")

    # ───────────────────────────────────────────────────────────
    # 2) DÉTERMINATION DES COLONNES À AJOUTER  →  to_append
    # ───────────────────────────────────────────────────────────
    to_append = [c for c in columns_to_add3 if c not in features_NANReplacedVal_df.columns]

    # (to_append reste accessible pour tout usage ultérieur)
    if to_append:
        features_NANReplacedVal_df = pd.concat(
            [features_NANReplacedVal_df, df[to_append]],
            axis=1,
            verify_integrity=True
        )
        # Exemple si tu veux enrichir outliersTransform_df également :
        # outliersTransform_df = pd.concat([outliersTransform_df, df[to_append]], axis=1)

    # ───────────────────────────────────────────────────────────
    # 3) ORDRE FINAL DES COLONNES
    # ───────────────────────────────────────────────────────────
    first_columns = [
        'session_id', 'nb_completedBar',
        'deltaTimestampOpening',
        'deltaTimestampOpeningSession1min', 'deltaTimestampOpeningSession1index',
        'deltaTimestampOpeningSession5min', 'deltaTimestampOpeningSession5index',
        'deltaTimestampOpeningSession15min', 'deltaTimestampOpeningSession15index',
        'deltaTimestampOpeningSession30min', 'deltaTimestampOpeningSession30index',
        'deltaCustomSessionMin', 'deltaCustomSessionIndex'
    ]

    sc_columns = columns_to_add2

    ordered, used = [], set()

    # 3-a  Colonnes prioritaires
    for col in first_columns:
        if col in features_NANReplacedVal_df.columns:
            ordered.append(col);
            used.add(col)

    # 3-b  sc_XXX suivi de XXX
    for sc in sc_columns:
        base = sc[3:]
        if sc in features_NANReplacedVal_df.columns:
            ordered.append(sc);
            used.add(sc)
            if base in features_NANReplacedVal_df.columns and base not in used:
                ordered.append(base);
                used.add(base)

    # 3-c  Reste
    ordered.extend([c for c in features_NANReplacedVal_df.columns if c not in used])
    features_NANReplacedVal_df = features_NANReplacedVal_df[ordered]

    # ───────────────────────────────────────────────────────────
    # 4) CONTRÔLES FINALS
    # ───────────────────────────────────────────────────────────
    required = ['sc_sessionStartEnd', 'sc_timeStampOpening']
    for col in required:
        if col not in features_NANReplacedVal_df.columns:
            raise ValueError(f"Colonne requise manquante : {col}")

    if features_NANReplacedVal_df['sc_sessionStartEnd'].iloc[0] != 10:
        raise ValueError("La première ligne du DataFrame doit commencer par un sc_sessionStartEnd = 10.")

    # ───────────────────────────────────────────────────────────
    # 5) (OPTIONNEL) AJOUT DE dateSession
    # ───────────────────────────────────────────────────────────
    # Désactive-le si tu n’en as plus besoin
    # features_NANReplacedVal_df['dateSession'] = (
    #     pd.to_datetime(features_NANReplacedVal_df['sc_timeStampOpening'], unit='s')
    # ).dt.date

    # ────────────────────────────────
    # 1) Préparation des tableaux
    # ────────────────────────────────
    sess_arr = features_NANReplacedVal_df['sc_sessionStartEnd'].to_numpy(np.int64)
    ts_arr   = features_NANReplacedVal_df['sc_timeStampOpening'].to_numpy(np.int64)  # secondes Unix


    # ────────────────────────────────
    # 3) Exécution rapide
    # ────────────────────────────────
    date_ns = compute_dates_numba(sess_arr, ts_arr)            # tableau int64(ns)
    features_NANReplacedVal_df['dateSession'] = pd.to_datetime(date_ns, utc=True)

    print("✅ Colonne dateSession calculée et ajoutée (Numba).")

    features_NANReplacedVal_df['dayOfWeek'] = features_NANReplacedVal_df['dateSession'].dt.dayofweek


    if to_append:
        outliersTransform_df = pd.concat(
            [outliersTransform_df, df[to_append]],
            axis=1,
            verify_integrity=True  # stoppe si un doublon persiste
        )

    # winsorized_scaledWithNanValue_df = pd.concat([winsorized_scaledWithNanValue_df, columns_df], axis=1)

    print_notification(
        "Colonnes 'sc_timeStampOpening','session_id','class_binaire', 'candleDir', 'date','sc_VWAP', 'trade_category', 'sc_sessionStartEnd' , 'sc_close', "
        "'trade_pnl', 'tp1_pnl','tp2_pnl','tp3_pnl','sl_pnl','trade_pnl_theoric','tp1_pnl_theoric','sl_pnl_theoric' ajoutées")

    # Filtrer les lignes valides : tendance haussière lente et class_binaire ∈ {0, 1}
    mask = (features_df['vix_slope_12_up_15'] == 1) & (features_df['class_binaire'].isin([0, 1]))

    # Calcul de la moyenne sur les lignes valides
    mean_val = features_df.loc[mask, 'class_binaire'].mean()

    # Créer une nouvelle colonne avec cette valeur partout (ou seulement sur les lignes valides si souhaité)
    nb_sample = features_df['vix_slope_12_up_15'].sum()
    nb_trades = features_df['vix_slope_12_up_15'].isin([0, 1]).sum()


    # Affichage
    # Masque : lignes où vix_slope_12_up_15 == 1 et class_binaire ∈ {0, 1}
    mask = (features_df['vix_slope_12_up_15'] == 1) & (features_df['class_binaire'].isin([0, 1]))

    # Nombre de cas où vix_slope_12_up_15 == 1 (tous types de class_binaire)
    nb_samples = (features_df['vix_slope_12_up_15'] == 1).sum()

    # Masque pour les cas vix_slope_12_up_15 == 1 et class_binaire ∈ {0, 1}
    mask = (features_df['vix_slope_12_up_15'] == 1) & (features_df['class_binaire'].isin([0, 1]))

    # Nombre de trades valides (class_binaire 0 ou 1)
    nb_trades = mask.sum()

    # Nombre de gagnants (class_binaire == 1)
    nb_wins = (features_df.loc[mask, 'class_binaire'] == 1).sum()

    # Winrate parmi les trades valides
    winrate = nb_wins / nb_trades if nb_trades > 0 else np.nan

    # Affichage
    print("🔍 Analyse sur vix_slope_12_up_15 == 1")
    print(f"- Nombre total de samples (peu importe class_binaire) : {nb_samples}")
    print(f"- Nombre de trades valides (class_binaire ∈ {{0,1}}) : {nb_trades}")
    print(f"- Nombre de gains (class_binaire == 1) : {nb_wins}")
    print(f"- Winrate : {winrate:.2%}")


    file_without_extension = os.path.splitext(file_name)[0]
    file_without_extension = file_without_extension.replace("Step4", "Step5")

    # Créer le nouveau nom de fichier pour les features originales
    new_file_name = file_without_extension + '_feat.csv'

    # Construire le chemin complet du nouveau fichier
    feat_file = os.path.join(file_dir, new_file_name)

    XTICKREVERAL_TICKPRICE = 10  # Nombre de ticks dans la zone above
    PERDIOD_ATR_SESSION_ANALYSE=8
    import Clustering.func_clustering as fctCustering
    features_NANReplacedVal_df = create_dataframe_with_group_indicators(
        df=features_NANReplacedVal_df,
        groupe1_sessions=fctCustering.GROUPE_SESSION_1,
        groupe2_sessions=fctCustering.GROUPE_SESSION_2,xtickReversalTickPrice=XTICKREVERAL_TICKPRICE,period_atr_stat_session=PERDIOD_ATR_SESSION_ANALYSE)

    print("---" * 50)
    print("ajoute les evénements")
    events_df = pd.read_csv(file_pathEvent, sep=';', encoding='ISO-8859-1')
    # Conversion explicite de la colonne 'Date' en datetime (sécurité)
    events_df['Date'] = pd.to_datetime(events_df['Date'], errors='coerce', dayfirst=True)  # dayfirst=True si format français (DD/MM/YYYY)
    # Vérification rapide
    print(events_df['Date'].head())
    # Créer un set des dates d'événements (pour recherche rapide)
    event_dates = set(events_df['Date'].dt.date.dropna())  # .dropna() pour éviter les NaT
    features_NANReplacedVal_df['event'] = features_NANReplacedVal_df['dateSession'].dt.date.isin(event_dates).astype(int)


    # Vérification du nombre de 10 et 20 dans sc_sessionStartEnd
    count_10 = (features_NANReplacedVal_df['sc_sessionStartEnd'] == 10).sum()
    count_20 = (features_NANReplacedVal_df['sc_sessionStartEnd'] == 20).sum()

    if count_10 != count_20 or count_10 <= 0:
        raise ValueError(f"Erreur : Le nombre de 10 et 20 dans sc_sessionStartEnd n'est pas égal ou est nul. 10={count_10}, 20={count_20}")

    # Filtrer uniquement les lignes où sc_sessionStartEnd == 10
    features_NANReplacedVal_df_reduced  = features_NANReplacedVal_df[features_NANReplacedVal_df['sc_sessionStartEnd'] == 10].copy()

    print("---" * 50)
    print("ajoute Information Groupe 1 et Groupe 2 (matrice de passage etc) ")

    feature_columns = get_feature_columns()
    feature_columns_g1 = [f"{col}_g1" if col != 'event' else col for col in feature_columns]
    feature_columns_g2 = [f"{col}_g2" if col != 'event' else col for col in feature_columns]
    if trained_models is None:
        # ═══ MODE ENTRAÎNEMENT ═══
        print("🔄 MODE ENTRAÎNEMENT - Création des modèles")

        # Scaling et fit des scalers
        X_scaled_g1, scaler_g1 = scale_features(features_NANReplacedVal_df_reduced, feature_columns_g1)
        X_scaled_g2, scaler_g2 = scale_features(features_NANReplacedVal_df_reduced, feature_columns_g2)

        # Clustering - fit des modèles (UN SEUL PAR GROUPE)
        kmeans_g1 = KMeans(n_clusters=clustering_with_K, random_state=42, n_init=50)  # ← n_init augmenté pour stabilité
        kmeans_g2 = KMeans(n_clusters=clustering_with_K, random_state=42, n_init=50)

        labels_g1_raw = kmeans_g1.fit_predict(X_scaled_g1)
        labels_g2_raw = kmeans_g2.fit_predict(X_scaled_g2)

        # Sauvegarde des modèles entraînés
        trained_models_output = {
            'scaler_g1': scaler_g1,
            'scaler_g2': scaler_g2,
            'kmeans_g1': kmeans_g1,
            'kmeans_g2': kmeans_g2
        }

    else:
        # ═══ MODE PRÉDICTION ═══
        print("🔮 MODE PRÉDICTION - Utilisation des modèles pré-entraînés")

        # Récupération des modèles sauvegardés
        scaler_g1 = trained_models['scaler_g1']
        scaler_g2 = trained_models['scaler_g2']
        kmeans_g1 = trained_models['kmeans_g1']
        kmeans_g2 = trained_models['kmeans_g2']

        # Transform avec les scalers pré-entraînés
        X_scaled_g1 = scaler_g1.transform(features_NANReplacedVal_df_reduced[feature_columns_g1])
        X_scaled_g2 = scaler_g2.transform(features_NANReplacedVal_df_reduced[feature_columns_g2])

        # Prédiction avec les modèles pré-entraînés
        labels_g1_raw = kmeans_g1.predict(X_scaled_g1)
        labels_g2_raw = kmeans_g2.predict(X_scaled_g2)

        # Pas de modèles à retourner en mode prédiction
        trained_models_output = None

    # ════════════════════════════════════════════════════════════════
    # TRAITEMENT COMMUN (ENTRAÎNEMENT + PRÉDICTION)
    # ════════════════════════════════════════════════════════════════

    # Relabel par volume croissant (BUG FIX: utiliser volume_p50_g2 pour G2)
    labels_g1, map_g1 = relabel_by_metric(labels_g1_raw, features_NANReplacedVal_df_reduced['volume_p50_g1'])
    labels_g2, map_g2 = relabel_by_metric(labels_g2_raw, features_NANReplacedVal_df_reduced['volume_p50_g2'])

    # Assigner dans le DataFrame principal
    features_NANReplacedVal_df_reduced["Cluster_G1"] = labels_g1
    features_NANReplacedVal_df_reduced["Cluster_G2"] = labels_g2

    # Profils G1
    df_scaled_g1 = pd.DataFrame(X_scaled_g1, columns=feature_columns_g1)
    df_scaled_g1["Cluster"] = labels_g1
    profile_scaled_g1 = df_scaled_g1.groupby("Cluster").mean().round(3)
    profile_orig_g1 = features_NANReplacedVal_df_reduced.groupby("Cluster_G1")[feature_columns_g1].mean().round(3)

    # Profils G2
    df_scaled_g2 = pd.DataFrame(X_scaled_g2, columns=feature_columns_g2)
    df_scaled_g2["Cluster"] = labels_g2
    profile_scaled_g2 = df_scaled_g2.groupby("Cluster").mean().round(3)
    profile_orig_g2 = features_NANReplacedVal_df_reduced.groupby("Cluster_G2")[feature_columns_g2].mean().round(3)

    # Matrice de transition (pour affichage optionnel)
    transition_matrix = pd.crosstab(features_NANReplacedVal_df_reduced['Cluster_G1'],
                                    features_NANReplacedVal_df_reduced['Cluster_G2'],
                                    normalize='index')
    print("\nProbabilités de transition (lignes=G1, colonnes=G2):")
    print(transition_matrix.round(3))

    # ────────────────────────────────────────────────────────────────
    # 1) COLONNES DE STATUT DES SESSIONS
    # ────────────────────────────────────────────────────────────────

    print("\n🏷️  ÉTAPE 1: STATU T DES SESSIONS")
    print("\n🏷️  ÉTAPE 1: STATUT DES SESSIONS")

    # Labels pour les clusters (logique adaptative selon clustering_with_K)
    if clustering_with_K == 2:
        cluster_labels = {0: 'CANDLE_MEAN_LOW', 1: 'CANDLE_MEAN_HIGH_1'}
    elif clustering_with_K == 3:
        cluster_labels = {0: 'CANDLE_MEAN_LOW', 1: 'VOL_CADL_MED', 2: 'CANDLE_MEAN_HIGH_1'}
    elif clustering_with_K == 4:
        cluster_labels = {0: 'CANDLE_MEAN_LOW', 1: 'VOL_CADL_MED', 2: 'CANDLE_MEAN_HIGH_1', 3: 'CANDLE_MEAN_HIGH_2'}
    elif clustering_with_K == 5:
        cluster_labels = {0: 'CANDLE_MEAN_LOW', 1: 'VOL_CADL_MED', 2: 'CANDLE_MEAN_HIGH_1', 3: 'CANDLE_MEAN_HIGH_2',
                          4: 'CANDLE_MEAN_HIGH_3'}
    elif clustering_with_K == 6:
        cluster_labels = {0: 'CANDLE_MEAN_LOW', 1: 'VOL_CADL_MED', 2: 'CANDLE_MEAN_HIGH_1', 3: 'CANDLE_MEAN_HIGH_2',
                          4: 'CANDLE_MEAN_HIGH_3', 5: 'CANDLE_MEAN_HIGH_4'}
    else:
        cluster_labels = {i: f'NIVEAU_{i}' for i in range(clustering_with_K)}

    # Statut G1 (Asie)
    features_NANReplacedVal_df_reduced['Regime_G1'] = labels_g1.copy()
    features_NANReplacedVal_df_reduced['Regime_G1_Label'] = features_NANReplacedVal_df_reduced['Regime_G1'].map(
        cluster_labels)

    # Statut G2 (Reste journée)
    features_NANReplacedVal_df_reduced['Regime_G2'] = labels_g2.copy()
    features_NANReplacedVal_df_reduced['Regime_G2_Label'] = features_NANReplacedVal_df_reduced['Regime_G2'].map(
        cluster_labels)

    # Message adaptatif selon le nombre de clusters
    cluster_names = list(cluster_labels.values())
    print(f"✅ Colonnes créées: Regime_G1, Regime_G1_Label, Regime_G2, Regime_G2_Label")
    print(f"🏷️  Labels utilisés pour K={clustering_with_K}: {cluster_names}")


    # ────────────────────────────────────────────────────────────────
    # 2) PRÉDICTION G2 À PARTIR DE G1
    # ────────────────────────────────────────────────────────────────

    print("\n🔮 ÉTAPE 2: PRÉDICTION G2 À PARTIR DE G1")

    # Utiliser la matrice de transition pour prédire
    # 1️⃣ Créer la matrice de transition
    transition_matrix_final = pd.crosstab(
        features_NANReplacedVal_df_reduced['Regime_G1'], features_NANReplacedVal_df_reduced['Regime_G2'], normalize='index'
    )

    print("="*50)
    print("🔍 ANALYSE DU DATAFRAME PRINCIPAL")
    print("="*50)

    # 1. Informations générales sur features_NANReplacedVal_df_reduced

    # print(features_NANReplacedVal_df_reduced.dtypes)
    # 3️⃣ Appliquer la prédiction
    #predictions = features_NANReplacedVal_df_reduced['Regime_G1'].apply(predict_g2_from_g1,transition_matrix_final)
    predictions = features_NANReplacedVal_df_reduced['Regime_G1'].apply(
        lambda x: predict_g2_from_g1(x, transition_matrix_final)
    )
    # 4️⃣ Séparer les résultats en deux colonnes
    features_NANReplacedVal_df_reduced['Prediction_G2'] = predictions.apply(lambda x: x[0])
    features_NANReplacedVal_df_reduced['Prediction_G2_Probability'] = predictions.apply(lambda x: x[1])

    # Labels textuels pour les prédictions
    features_NANReplacedVal_df_reduced['Prediction_G2_Label'] = features_NANReplacedVal_df_reduced['Prediction_G2'].map({
        0: 'CANDLE_MEAN_LOW',
        1: 'VOL_CADL_MED' if clustering_with_K >= 3 else 'CANDLE_MEAN_HIGH',
        2: 'CANDLE_MEAN_HIGH' if clustering_with_K >= 3 else None
    }).fillna('CANDLE_MEAN_HIGH')

    print(f"✅ Colonnes créées: Prediction_G2, Prediction_G2_Label, Prediction_G2_Probability")

    # ────────────────────────────────────────────────────────────────
    # 3) QUALITÉ DE LA PRÉDICTION
    # ────────────────────────────────────────────────────────────────

    print("\n🎯 ÉTAPE 3: QUALITÉ DE LA PRÉDICTION")

    # Exactitude de la prédiction
    features_NANReplacedVal_df_reduced['Prediction_Correct'] = (features_NANReplacedVal_df_reduced['Prediction_G2'] == features_NANReplacedVal_df_reduced['Regime_G2'])
    features_NANReplacedVal_df_reduced['Prediction_Quality'] = features_NANReplacedVal_df_reduced['Prediction_Correct'].map({True: 'CORRECT', False: 'INCORRECT'})

    # Type de transition réelle
    features_NANReplacedVal_df_reduced['Transition_Type'] = features_NANReplacedVal_df_reduced['Regime_G1_Label'] + 'to' + features_NANReplacedVal_df_reduced['Regime_G2_Label']

    # Transition prédite
    features_NANReplacedVal_df_reduced['Transition_Predicted'] = features_NANReplacedVal_df_reduced['Regime_G1_Label'] + 'to' + features_NANReplacedVal_df_reduced['Prediction_G2_Label']

    accuracy = features_NANReplacedVal_df_reduced['Prediction_Correct'].mean()
    print(f"✅ Précision globale des prédictions: {accuracy:.1%}")
    print(f"✅ Colonnes créées: Prediction_Correct, Prediction_Quality, Transition_Type, Transition_Predicted")


    # 1️⃣ Identifier les colonnes supplémentaires à ajouter
    cols_extra = [col for col in features_NANReplacedVal_df_reduced.columns if col not in features_NANReplacedVal_df.columns and col != 'dateSession']

    # 2️⃣ Filtrer les lignes de sc_sessionStartEnd == 10 dans features_NANReplacedVal_df
    df_10 = features_NANReplacedVal_df[features_NANReplacedVal_df['sc_sessionStartEnd'] == 10][['dateSession']].drop_duplicates()

    # 3️⃣ Fusionner sur la clé dateSession pour obtenir les valeurs des colonnes supplémentaires
    df_merge = df_10.merge(features_NANReplacedVal_df_reduced[['dateSession'] + cols_extra], on='dateSession', how='left')

    # 4️⃣ Mettre à jour features_NANReplacedVal_df avec les colonnes supplémentaires (par dateSession)
    for col in cols_extra:
        features_NANReplacedVal_df = features_NANReplacedVal_df.merge(
            df_merge[['dateSession', col]],
            on='dateSession',
            how='left'
        )

    # 5️⃣ Vérification rapide
    print("✅ Colonnes ajoutées :", cols_extra)
    print(features_NANReplacedVal_df.head())

    print("---" * 50)
    print("Création du fichier de simultation pour les clusters (utilisation avec main_analyse_cluser")
    print_notification(f"Enregistrement du fichier de features non modifiées pour l'aanalyse des cluster : {feat_file}")
    new_file_name4Cluster = file_without_extension + '_feat_4Cluster.csv'

    # Construire le chemin complet du nouveau fichier
    feat_file4Cluster = os.path.join(file_dir, new_file_name4Cluster)

    # Colonnes à conserver
    cols_to_keep = [
        'session_id',
        'dateSession',
        'dayOfWeek',

        # Groupe 1 - Percentiles P25, P50, P75
        'volume_p25_g1', 'volume_p50_g1', 'volume_p75_g1',
        'atr_p25_g1', 'atr_p50_g1', 'atr_p75_g1',
        'duration_p25_g1', 'duration_p50_g1', 'duration_p75_g1',
        'vol_above_p25_g1', 'vol_above_p50_g1', 'vol_above_p75_g1',
        'volMeanPerTick_p25_g1', 'volMeanPerTick_p50_g1', 'volMeanPerTick_p75_g1',

        # Groupe 1 - Métriques Win/Lose
        'meanVol_perTick_over1_g1', 'meanVol_perTick_over2_g1', 'meanVol_perTick_over5_g1', 'meanVol_perTick_over12_g1',
        'meanVol_perTick_over20_g1', 'meanVol_perTick_over30_g1',
        'volCandleMeanOver5Ratio_g1', 'volCandleMeanOver12Ratio_g1', 'volCandleMeanOver20Ratio_g1',
        'volCandleMeanOver30Ratio_g1',

        # Groupe 1 - Autres indicateurs
        'extreme_ratio_g1', 'volume_spread_g1', 'volume_above_spread_g1',
        'atr_spread_g1', 'duration_spread_g1',

        # Groupe 2 - Percentiles P25, P50, P75
        'volume_p25_g2', 'volume_p50_g2', 'volume_p75_g2',
        'atr_p25_g2', 'atr_p50_g2', 'atr_p75_g2',
        'duration_p25_g2', 'duration_p50_g2', 'duration_p75_g2',
        'vol_above_p25_g2', 'vol_above_p50_g2', 'vol_above_p75_g2',
        'volMeanPerTick_p25_g2', 'volMeanPerTick_p50_g2', 'volMeanPerTick_p75_g2',

        # Groupe 2 - Métriques Win/Lose
        'meanVol_perTick_over1_g2', 'meanVol_perTick_over2_g2', 'meanVol_perTick_over5_g2', 'meanVol_perTick_over12_g2',
        'meanVol_perTick_over20_g2', 'meanVol_perTick_over30_g2',
        'volCandleMeanOver5Ratio_g2', 'volCandleMeanOver12Ratio_g2', 'volCandleMeanOver20Ratio_g2',
        'volCandleMeanOver30Ratio_g2',

        # Groupe 2 - Autres indicateurs
        'extreme_ratio_g2', 'volume_spread_g2', 'volume_above_spread_g2',
        'atr_spread_g2', 'duration_spread_g2'
    ]

    # Nouveau DataFrame final
    features_NANReplacedVal_Session_4Cluster_df = features_NANReplacedVal_df_reduced[cols_to_keep]
    # Lire le fichier des événements


    # Ajouter la colonne "event" au dataframe principal
    features_NANReplacedVal_Session_4Cluster_df = features_NANReplacedVal_Session_4Cluster_df.copy()

    features_NANReplacedVal_Session_4Cluster_df['event'] = features_NANReplacedVal_Session_4Cluster_df['dateSession'].dt.date.isin(event_dates).astype(int)

    # ════════════════════════════════════════════════════════════════
    # RETOUR ADAPTÉ SELON LE MODE
    # ════════════════════════════════════════════════════════════════

    if trained_models is None:
        # Mode entraînement : retourner les modèles
        return feat_file, feat_file4Cluster, features_NANReplacedVal_df, features_NANReplacedVal_Session_4Cluster_df, trained_models_output
    else:
        # Mode prédiction : retour standard
        return feat_file, feat_file4Cluster, features_NANReplacedVal_df, features_NANReplacedVal_Session_4Cluster_df

# ════════════════════════════════════════════════════════════════
# 1️⃣ PREMIER APPEL : MODE ENTRAÎNEMENT (données d'entraînement)
# ════════════════════════════════════════════════════════════════

print("🔄 TRAITEMENT DES DONNÉES D'ENTRAÎNEMENT")
df = load_data(file_path)

# Premier appel avec trained_models=None pour entraîner les modèles
feat_file, feat_file4Cluster, features_NANReplacedVal_df, features_NANReplacedVal_Session_4Cluster_df, trained_models = (
    calculate_features_and_sessionsStat(df, file_path, trained_models=None))

print("---" * 50)
# Sauvegarder le fichier des features originales
print_notification(f"Enregistrement du fichier de features non modifiées pour Train, Test, Val1 et Val : {feat_file}")
save_features_with_sessions(features_NANReplacedVal_df, CUSTOM_SESSIONS, feat_file)
# Sauvegarde du fichier final
save_features_with_sessions(features_NANReplacedVal_Session_4Cluster_df, CUSTOM_SESSIONS, feat_file4Cluster)

print_notification(f"Debut du split du fichier Step 5 pour former Train, test, val1 et Val")

# Appeler la fonction avec le DataFrame et les paramètres
if (USSE_SPLIT_SESSION):
    diviser_fichier_par_sessions(features_NANReplacedVal_df, directory_path, feat_file,
                                 use_default_params=USE_DEFAUT_PARAM_4_SPLIT_SESSION,
                                 encoding_used="ISO-8859-1")  # ISO-8859-1

# ════════════════════════════════════════════════════════════════
# 2️⃣ DEUXIÈME APPEL : MODE PRÉDICTION (données unseen)
# ════════════════════════════════════════════════════════════════

print("\n🔮 TRAITEMENT DES DONNÉES UNSEEN AVEC MODÈLES PRÉ-ENTRAÎNÉS")
df_unseen = load_data(file_path_unseen)

# Deuxième appel avec les modèles pré-entraînés pour garantir la cohérence
feat_file_unseen, feat_file4Cluster_unseen, features_NANReplacedVal_df_unseen, features_NANReplacedVal_Session_4Cluster_df_unseen = (
    calculate_features_and_sessionsStat(df_unseen, file_path_unseen, trained_models=trained_models))

print("---" * 50)
# Sauvegarder le fichier des features originales
print_notification(f"Enregistrement du fichier de features non modifiées pour *unseen : {feat_file_unseen}")
save_features_with_sessions(features_NANReplacedVal_df_unseen, CUSTOM_SESSIONS, feat_file_unseen)
# Sauvegarde du fichier final
save_features_with_sessions(features_NANReplacedVal_Session_4Cluster_df_unseen, CUSTOM_SESSIONS, feat_file4Cluster_unseen)

# ════════════════════════════════════════════════════════════════
# 📊 RÉSUMÉ DES MODÈLES UTILISÉS
# ════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("📊 RÉSUMÉ DU TRAITEMENT")
print("="*60)
print("✅ Données d'entraînement : Modèles créés et sauvegardés")
print("✅ Données unseen : Modèles pré-entraînés utilisés")
print("✅ Cohérence garantie entre les deux jeux de données")
print(f"✅ Modèles sauvegardés : {list(trained_models.keys())}")
print("="*60)