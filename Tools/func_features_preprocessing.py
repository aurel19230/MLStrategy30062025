from numba import jit
from colorama import Fore, Style, init

import math
REPLACE_NAN = False
REPLACED_NANVALUE_BY = 90000.54789
REPLACED_NANVALUE_BY_INDEX = 1
from numba import njit, prange
ENABLE_PANDAS_METHOD_SCALING = True
fig_range_input=''
DEFAULT_DIV_BY0 = True  # max_ratio or valuex

SECONDS_IN_DAY   = 86_400
NANO_PER_SECOND  = 1_000_000_000
NANO_PER_DAY     = SECONDS_IN_DAY * NANO_PER_SECOND
SENTINEL         = -1      # valeur de remplissage (sera convertie en NaT ensuite)
# Demander à l'utilisateur s'il souhaite ajuster l'axe des abscisses
adjust_xaxis_input = ''
user_choice=''
if user_choice.lower() == 'd' or user_choice.lower() == 's':
    adjust_xaxis_input = input(
        "Voulez-vous afficher les graphiques entre les valeurs de floor et crop ? (o/n) : ").lower()
def toBeDisplayed_if_s(user_choice, choice):
    # Utilisation de l'opérateur ternaire
    result = True if user_choice == 'd' else (True if user_choice == 's' and choice == True else False)
    return result

def calculate_candle_rev_tick(df):
    """
    Calcule la valeur de CANDLE_REV_TICK en fonction des conditions spécifiées, en déterminant
    dynamiquement le minimum incrément non nul entre les valeurs de la colonne 'sc_close'.

    Sélectionne 4 occurrences à partir de la 100e ligne du DataFrame, plutôt que les 4 premières.

    Args:
        df (pd.DataFrame): DataFrame contenant les colonnes 'sc_candleDir', 'sc_high', 'sc_close'.

    Returns:
        int: La valeur de CANDLE_REV_TICK si toutes les valeurs sont identiques pour les 4 occurrences.

    Raises:
        ValueError: Si les valeurs calculées diffèrent pour les 4 occurrences sélectionnées où sc_candleDir == -1.
    """
    # Calculer la différence absolue entre les valeurs de 'sc_close'
    df['close_diff'] = df['sc_close'].diff().abs()

    # Identifier le minimum incrément non nul
    minimum_increment = df['close_diff'][df['close_diff'] > 0].min()

    # Vérifier si le minimum incrément est bien défini
    if pd.isna(minimum_increment):
        raise ValueError("Impossible de calculer le minimum incrément non nul.")

    print(f"Minimum increment: {minimum_increment}")

    # Filtrer les lignes où sc_candleDir == -1
    filtered_df = df[df['sc_candleDir'] == -1]

    # S'assurer qu'il y a au moins 100 lignes + 4 occurrences où sc_candleDir == -1
    if len(filtered_df) < 104:
        raise ValueError(
            f"Pas assez d'occurrences où sc_candleDir == -1 (trouvé {len(filtered_df)}, besoin d'au moins 104)")

    # Sélectionner 4 occurrences à partir de la 100e ligne
    selected_rows = filtered_df.iloc[100:].head(4)

    # Vérifier qu'on a bien 4 lignes
    if len(selected_rows) < 4:
        raise ValueError(
            f"Pas assez d'occurrences à partir de la 100e ligne (trouvé {len(selected_rows)}, besoin de 4)")

    # Calculer (sc_high - sc_close) * minimum_increment pour les 4 occurrences sélectionnées
    values = ((selected_rows['sc_high'] - selected_rows['sc_close']) * (1 / minimum_increment)) + 1

    # Vérifier si toutes les valeurs sont identiques
    # if not all(values == values.iloc[0]):
    #     raise ValueError(
    #         "Les valeurs de (sc_high - sc_close) * minimum_increment diffèrent pour les 4 occurrences sélectionnées.")

    # Retourner la valeur commune
    return int(values.iloc[0])


# Définition de toutes les colonnes requises avec leurs identifiants
colonnes_a_transferer = [
    # Ratios de mouvement de base
    'sc_ratio_volRevMove_volImpulsMove',  # //1
    'sc_ratio_deltaImpulsMove_volImpulsMove',  # //2
    'sc_ratio_deltaRevMove_volRevMove',  # //3

    # Ratios de zones
    'sc_ratio_volZone1_volExtrem',  # //3.1
    'sc_ratio_deltaZone1_volZone1',  # //3.2
    'sc_ratio_deltaExtrem_volExtrem',  # //3.3

    # Ratios de zones de continuation
    'sc_ratio_volRevZone_xTicksContZone',  # //4
    'sc_ratioDeltaXticksContZone_volXticksContZone',  # //5

    # Ratios de force
    'sc_ratio_impulsMoveStrengthVol_xRevZone',  # //6
    'sc_ratio_revMoveStrengthVol_xRevZone',  # //7

    # Type d'imbalance
    'sc_imbType_contZone',  # //8

    # Ratios détaillés des zones
    'sc_ratio_volRevMoveZone1_volImpulsMoveExtrem_xRevZone',  # //9.09
    'sc_ratio_volRevMoveZone1_volRevMoveExtrem_xRevZone',  # //9.10
    'sc_ratio_deltaRevMoveZone1_volRevMoveZone1',  # //9.11
    'sc_ratio_deltaRevMoveExtrem_volRevMoveExtrem',  # //9.12
    'sc_ratio_volImpulsMoveExtrem_volImpulsMoveZone1_xRevZone',  # //9.13
    'sc_ratio_deltaImpulsMoveZone1_volImpulsMoveZone1',  # //9.14
    'sc_ratio_deltaImpulsMoveExtrem_volImpulsMoveExtrem_xRevZone',  # //9.15

    # Métriques DOM et autres
    'sc_cumDOM_askBid_avgRatio',  # //10
    'sc_cumDOM_askBid_pullStack_avgDiff_ratio',  # //11
    'sc_delta_impulsMove_xRevZone_bigStand_extrem',  # //12
    'sc_delta_revMove_xRevZone_bigStand_extrem',  # //13

    # Ratios divers
    'sc_ratio_delta_vaVolVa',  # //14
    'sc_borderVa_vs_close',  # //15
    'sc_ratio_volRevZone_volCandle',  # //16
    'sc_ratio_deltaRevZone_volCandle',  # //17

    # process_reg_slope_replacement 18 19 20 21

    # Temps
    'sc_timeElapsed2LastBar'  # //22
]


def add_moving_percentiles4VolEtDuration(features_df, df, windows=[5, 10, 20, 30], columns=['volume', 'candleDuration'],
                                         percentiles=[25, 50, 75, 100], inplace=False):
    """
    Ajoute des indicateurs de percentiles mobiles pour les colonnes spécifiées,
    en respectant les limites des sessions (pas de débordement entre sessions).

    Parameters:
    -----------
    features_df : pd.DataFrame
        DataFrame où ajouter les nouvelles colonnes d'indicateurs (contient aussi candleDuration)
    df : pd.DataFrame
        DataFrame contenant session_id et volume
    windows : list
        Liste des fenêtres pour les percentiles mobiles (par défaut [5, 10, 20, 30])
    columns : list
        Liste des colonnes pour lesquelles calculer les percentiles (par défaut ['volume', 'candleDuration'])
    percentiles : list
        Liste des percentiles à calculer (par défaut [25, 50, 75, 100] pour P25, P50, P75, P100)
    inplace : bool
        Si True, modifie directement features_df sans faire de copie (par défaut False)

    Returns:
    --------
    pd.DataFrame
        DataFrame avec les nouvelles colonnes de percentiles mobiles ajoutées
        (ou None si inplace=True)
    """

    # Utiliser directement features_df ou faire une copie selon le paramètre inplace
    if inplace:
        result_df = features_df
    else:
        result_df = features_df.copy()

    # Vérifier que session_id existe dans df
    if 'session_id' not in df.columns:
        raise ValueError("La colonne 'session_id' n'existe pas dans le DataFrame df")

    # Vérifier que les colonnes existent dans les bons DataFrames
    for col in columns:
        if f'sc_{col}' == 'sc_volume':
            if f'sc_{col}' not in df.columns:
                raise ValueError(f"La colonne '{col}' n'existe pas dans le DataFrame df")
        elif f'sc_{col}' == 'sc_candleDuration':
            if col not in features_df.columns:
                raise ValueError(f"La colonne '{col}' n'existe pas dans le DataFrame features_df")
        else:
            # Pour d'autres colonnes, vérifier dans les deux DataFrames
            if f'sc_{col}' not in df.columns and f'sc_{col}' not in features_df.columns:
                raise ValueError(f"La colonne '{f'sc_{col}'}' n'existe pas dans df ni dans features_df")

    # Vérifier que features_df et df ont la même longueur
    if len(features_df) != len(df):
        raise ValueError(
            f"features_df ({len(features_df)} lignes) et df ({len(df)} lignes) doivent avoir la même longueur")

    # Initialiser les nouvelles colonnes avec NaN dans result_df
    for col in columns:
        for window in windows:
            for percentile in percentiles:
                new_col_name = f"{col}_P{percentile}_{window}"
                result_df[new_col_name] = np.nan

    # Traiter chaque session séparément
    for session_id in df['session_id'].unique():
        # Filtrer les données de la session courante
        session_mask = df['session_id'] == session_id
        session_indices = df[session_mask].index

        # Pour chaque colonne, chaque fenêtre et chaque percentile
        for col in columns:
            for window in windows:
                # Récupérer les données depuis le bon DataFrame
                if f'sc_{col}' == 'sc_volume':
                    session_data = df.loc[session_mask, f'sc_{col}']
                elif f'sc_{col}' == 'candleDuration':
                    session_data = features_df.loc[session_mask, col]
                else:
                    # Pour d'autres colonnes, essayer d'abord df, puis features_df
                    if f'sc_{col}' in df.columns:
                        session_data = df.loc[session_mask,f'sc_{col}']
                    else:
                        session_data = features_df.loc[session_mask, col]

                for percentile in percentiles:
                    new_col_name = f"{col}_P{percentile}_{window}"

                    # Calculer le percentile mobile avec pandas rolling
                    # min_periods=window assure qu'on a au moins 'window' valeurs
                    rolling_percentile = session_data.rolling(window=window, min_periods=window).quantile(
                        percentile / 100.0)

                    # Assigner les valeurs calculées aux indices correspondants dans result_df
                    result_df.loc[session_indices, new_col_name] = rolling_percentile.values

    if inplace:
        return None  # Ne retourne rien si modification en place
    else:
        return result_df


def init_column_settings(candle_rev_tick):
    print(candle_rev_tick)
    # Ajouter les colonnes d'absorption au dictionnaire
    absorption_settings = {f'is_absorpsion_{tick}ticks_{direction}': ("winsor",None,False, False, 10, 90, toBeDisplayed_if_s(user_choice, False))
                          for tick in range(3, candle_rev_tick + 1)
                          for direction in ['low', 'high']}
    column_settings = {
        # Time-based features
        'session_id': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),

        'deltaTimestampOpening': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
        'candleDuration': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),

        'deltaTimestampOpeningSession1min': (
        "winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
        'deltaTimestampOpeningSession1index': (
        "winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
        'deltaTimestampOpeningSession5min': (
        "winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
        'deltaTimestampOpeningSession5index': (
        "winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
        'deltaTimestampOpeningSession15min': (
        "winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
        'deltaTimestampOpeningSession15index': (
        "winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
        'deltaTimestampOpeningSession30min': (
        "winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
        'deltaTimestampOpeningSession30index': (
        "winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
        'deltaCustomSessionMin': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
        'deltaCustomSessionIndex': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),

        # Stochastic indicators
        'stoch_k_overbought': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
        'stoch_k_oversold': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
        'stoch_d_overbought': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
        'stoch_d_oversold': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),


        # Force index indicators
        'force_index_short_4': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
        'force_index_long_4': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
        'force_index_short_4_norm': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
        'force_index_long_4_norm': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
        'force_index_divergence': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
        'fi_momentum': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
        'is_stoch_overbought': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
        'is_stoch_oversold': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),


        # Other technical indicators
        'rsi_': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
        'macd': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
        'macd_signal': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
        'macd_hist': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
        #'adx_': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
        #'plus_di_': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
        #'minus_di_': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),

        # Williams R indicators
        'is_wr_overbought_short': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
        'is_wr_oversold_long': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
        # 'is_wr_short_overbought': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
        # 'is_wr_short_oversold': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
        #'wr_long_overbought': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
        'wr_oversold_long': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
        'wr_overbought_short': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
        # 'wr_short_oversold': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
        'signal_short_overbought': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
        'signal_long_oversold': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),

        # MFI indicators
        #'mfi': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
        'is_mfi_overbought_short': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
        'is_mfi_oversold_long': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
        'mfi_overbought_period': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
        'mfi_oversold_period': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),

        'mfi_bearish': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
        'mfi_antiBear': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
        'is_mfi_shortDiv': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
        'is_mfi_antiShortDiv': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),

    'vix_slope_12_up_15': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
    'is_vwap_reversal_pro_short': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
    'is_vwap_reversal_pro_long': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),


        # Price and volume features
        'volAbvState': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
        'volBlwState': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
        'candleSizeTicks': ("winsor", None, True, True, 1, 99, toBeDisplayed_if_s(user_choice, False)),
        'diffPriceClosePoc_0_0': ("winsor", None, True, True, 0.5, 99.5, toBeDisplayed_if_s(user_choice, False)),
        'diffPriceClosePoc_0_1': ("winsor", None, True, True, 0.5, 99.5, toBeDisplayed_if_s(user_choice, False)),
        'diffPriceClosePoc_0_2': ("winsor", None, True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'diffPriceClosePoc_0_3': ("winsor", None, True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'diffPriceClosePoc_0_4': ("winsor", None, True, True, 2, 98, toBeDisplayed_if_s(user_choice, False)),
        'diffPriceClosePoc_0_5': ("winsor", None, True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),

        # High/Low price differentials
        'diffHighPrice_0_1': ("winsor", None, True, True, 0.5, 99.5, toBeDisplayed_if_s(user_choice, False)),
        'diffHighPrice_0_2': ("winsor", None, True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'diffHighPrice_0_3': ("winsor", None, True, True, 2, 98, toBeDisplayed_if_s(user_choice, False)),
        'diffHighPrice_0_4': ("winsor", None, True, True, 2, 98, toBeDisplayed_if_s(user_choice, False)),
        'diffHighPrice_0_5': ("winsor", None, True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'diffLowPrice_0_1': ("winsor", None, False, False, 0.5, 99.5, toBeDisplayed_if_s(user_choice, False)),
        'diffLowPrice_0_2': ("winsor", None, True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'diffLowPrice_0_3': ("winsor", None, True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'diffLowPrice_0_4': ("winsor", None, True, True, 2, 98, toBeDisplayed_if_s(user_choice, False)),
        'diffLowPrice_0_5': ("winsor", None, True, True, 2, 98, toBeDisplayed_if_s(user_choice, False)),

        # POC price differentials
        'diffPocPrice_0_1': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
        'diffPocPrice_1_2': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
        'diffPocPrice_2_3': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
        'diffPocPrice_0_2': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),

        # VWAP related
        'diffPriceCloseVWAP': ("winsor", None, True, True, 1, 99, toBeDisplayed_if_s(user_choice, True)),
        'diffPriceCloseVWAPbyIndex': ("winsor", None, False, False, 1, 99, toBeDisplayed_if_s(user_choice, True)),

        # Technical indicators
        'atr': ("winsor", None, True, True, 0.1, 99, toBeDisplayed_if_s(user_choice, False)),
        'atr_range': ("winsor", None, False, False, 0.1, 99, toBeDisplayed_if_s(user_choice, False)),
        'atr_extrem': ("winsor", None, False, False, 0.1, 99, toBeDisplayed_if_s(user_choice, False)),
        'is_atr_range': ("winsor", None, False, False, 0.1, 99, toBeDisplayed_if_s(user_choice, False)),
        'is_atr_extremLow': ("winsor", None, False, False, 0.1, 99, toBeDisplayed_if_s(user_choice, False)),

        'bandWidthBB': ("winsor", None, True, True, 1, 99, toBeDisplayed_if_s(user_choice, False)),
        'perctBB': ("winsor", None, True, True, 12, 92, toBeDisplayed_if_s(user_choice, False)),

        # VA (Value Area) metrics
        'perct_VA6P': ("winsor", None, True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'ratio_delta_vol_VA6P': ("winsor", None, True, True, 4, 98, toBeDisplayed_if_s(user_choice, False)),
        'diffPriceClose_VA6PPoc': ("winsor", None, True, True, 2, 98, toBeDisplayed_if_s(user_choice, False)),
        'diffPriceClose_VA6PvaH': ("winsor", None, True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'diffPriceClose_VA6PvaL': ("winsor", None, True, True, 12, 88, toBeDisplayed_if_s(user_choice, False)),
        'perct_VA11P': ("winsor", None, True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'ratio_delta_vol_VA11P': ("winsor", None, True, True, 2, 98, toBeDisplayed_if_s(user_choice, False)),
        'diffPriceClose_VA11PPoc': ("winsor", None, True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'diffPriceClose_VA11PvaH': ("winsor", None, True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'diffPriceClose_VA11PvaL': ("winsor", None, True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'perct_VA16P': ("winsor", None, True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'ratio_delta_vol_VA16P': ("winsor", None, True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'diffPriceClose_VA16PPoc': ("winsor", None, True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'diffPriceClose_VA16PvaH': ("winsor", None, True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'diffPriceClose_VA16PvaL': ("winsor", None, True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'perct_VA21P': ("winsor", None, True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'ratio_delta_vol_VA21P': ("winsor", None, True, True, 2, 98, toBeDisplayed_if_s(user_choice, False)),
        'diffPriceClose_VA21PPoc': ("winsor", None, True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'diffPriceClose_VA21PvaH': ("winsor", None, True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'diffPriceClose_VA21PvaL': ("winsor", None, True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),

        # VA overlap ratios
        'overlap_ratio_VA_6P_11P': ("winsor", None, True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'overlap_ratio_VA_6P_16P': ("winsor", None, True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'overlap_ratio_VA_6P_21P': ("winsor", None, True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'overlap_ratio_VA_11P_21P': ("winsor", None, True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),

        # POC analysis
        'poc_diff_6P_11P': ("winsor", None, True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'poc_diff_ratio_6P_11P': ("winsor", None, True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'poc_diff_6P_16P': ("winsor", None, True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'poc_diff_ratio_6P_16P': ("winsor", None, True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'poc_diff_6P_21P': ("winsor", None, True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'poc_diff_ratio_6P_21P': ("winsor", None, True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'poc_diff_11P_21P': ("winsor", None, True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'poc_diff_ratio_11P_21P': ("winsor", None, True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),

        # Market regime metrics
        'market_regimeADX': ("winsor", None, True, False, 2, 99, toBeDisplayed_if_s(user_choice, True)),
        'market_regimeADX_state': ("winsor", None, False, False, 0.5, 99.8, toBeDisplayed_if_s(user_choice, True)),
        'is_in_range_10_32': ("winsor", None, False, False, 0.5, 99.8, toBeDisplayed_if_s(user_choice, True)),
        'is_in_range_5_23': ("winsor", None, False, False, 0.5, 99.8, toBeDisplayed_if_s(user_choice, True)),

        # Reversal and momentum features
        'bearish_reversal_force': ("winsor", None, False, True, 1, 99.5, toBeDisplayed_if_s(user_choice, False)),
        'bullish_reversal_force': ("winsor", None, False, True, 1, 99.5, toBeDisplayed_if_s(user_choice, False)),
        'volMeanOver5': ("winsor", None, False, True, 1, 99.7, toBeDisplayed_if_s(user_choice, False)),
        'volMeanOver10': ("winsor", None, False, True, 1, 99.7, toBeDisplayed_if_s(user_choice, False)),
        'volMeanOver20': ("winsor", None, False, True, 1, 99.7, toBeDisplayed_if_s(user_choice, False)),
        'volMeanOver30': ("winsor", None, False, True, 1, 99.7, toBeDisplayed_if_s(user_choice, False)),
        'volume_perTick': ("winsor", None, False, True, 1, 99.7, toBeDisplayed_if_s(user_choice, False)),
        'meanVol_perTick_over1': ("winsor", None, False, True, 1, 99.7, toBeDisplayed_if_s(user_choice, False)),
        'meanVol_perTick_over3': ("winsor", None, False, True, 1, 99.7, toBeDisplayed_if_s(user_choice, False)),
        'meanVol_perTick_over5': ("winsor", None, False, True, 1, 99.7, toBeDisplayed_if_s(user_choice, False)),
        'meanVol_perTick_over10': ("winsor", None, False, True, 1, 99.7, toBeDisplayed_if_s(user_choice, False)),
        'meanVol_perTick_over20': ("winsor", None, False, True, 1, 99.7, toBeDisplayed_if_s(user_choice, False)),
        'volRev_perTick': ("winsor", None, False, True, 1, 99.7, toBeDisplayed_if_s(user_choice, False)),
        'volRev_perTick_volxTicksContZone_perTick_ratio': ("winsor", None, False, True, 1, 99.7, toBeDisplayed_if_s(user_choice, False)),
        'volRev_perTick_Vol_perTick_over1': (
        "winsor", None, False, True, 1, 99.7, toBeDisplayed_if_s(user_choice, False)),


        'volxTicksContZone_perTick': ("winsor", None, False, True, 1, 99.7, toBeDisplayed_if_s(user_choice, False)),
        'diffVolCandle_0_1Ratio': ("winsor", None, False, True, 1, 98.5, toBeDisplayed_if_s(user_choice, False)),
        'diffVolDelta_0_1Ratio': ("winsor", None, True, True, 1, 99, toBeDisplayed_if_s(user_choice, False)),
        'diffVolDelta_0_0Ratio': ("winsor", None, True, True, 2, 98, toBeDisplayed_if_s(user_choice, False)),
        'diffVolDelta_1_1Ratio': ("winsor", None, True, True, 2.5, 97.5, toBeDisplayed_if_s(user_choice, False)),
        'diffVolDelta_2_2Ratio': ("winsor", None, True, True, 5, 95, toBeDisplayed_if_s(user_choice, False)),
        'diffVolDelta_3_3Ratio': ("winsor", None, True, True, 1, 99, toBeDisplayed_if_s(user_choice, False)),
        'cum_4DiffVolDeltaRatio': ("winsor", None, True, True, 1, 99, toBeDisplayed_if_s(user_choice, False)),
        'volcontZone_zoneReversal': ("winsor", None, True, True, 1, 99, toBeDisplayed_if_s(user_choice, False)),

        # Volume profile features
        'volRevVolCandle_ratio': ("winsor", None, True, True, 2, 65, toBeDisplayed_if_s(user_choice, False)),
        'volPocVolRevesalXContRatio': ("winsor", None, True, True, 2, 95, toBeDisplayed_if_s(user_choice, False)),
        'volPocVolCandleRatio': ("winsor", None, True, True, 2, 65, toBeDisplayed_if_s(user_choice, False)),

        'volRevVolRevesalXContRatio': ("winsor", None, True, True, 2, 95, toBeDisplayed_if_s(user_choice, False)),
        'deltaRev_volRev_ratio': ("winsor", None, True, True, 2, 95, toBeDisplayed_if_s(user_choice, False)),

        'pocDeltaPocVolRatio': ("winsor", None, True, True, 5, 95, toBeDisplayed_if_s(user_choice, False)),
        'volAbv_vol_ratio': ("winsor", None, True, True, 1, 99, toBeDisplayed_if_s(user_choice, False)),
        'volBlw_vol_ratio': ("winsor", None, True, True, 1, 99, toBeDisplayed_if_s(user_choice, False)),
        'asymetrie_volume': ("winsor", None, True, True, 1, 99, toBeDisplayed_if_s(user_choice, False)),
        'volCandleMeanOver5Ratio': ("winsor", None, False, True, 1, 99, toBeDisplayed_if_s(user_choice, False)),
        'volCandleMeanOver10Ratio': ("winsor", None, False, True, 1, 99, toBeDisplayed_if_s(user_choice, False)),
        'volCandleMeanOver20Ratio': ("winsor", None, False, True, 1, 99, toBeDisplayed_if_s(user_choice, False)),
        'volCandleMeanOver30Ratio': ("winsor", None, False, True, 1, 99, toBeDisplayed_if_s(user_choice, False)),

        'ratioVolPerTick_over1': ("winsor", None, True, True, 1, 99, toBeDisplayed_if_s(user_choice, False)),
        'ratioVolPerTick_over3': ("winsor", None, False, True, 1, 99, toBeDisplayed_if_s(user_choice, False)),
        'ratioVolPerTick_over5': ("winsor", None, False, True, 1, 99, toBeDisplayed_if_s(user_choice, False)),
        'ratioVolPerTick_over10': ("winsor", None, False, True, 1, 99, toBeDisplayed_if_s(user_choice, False)),
        'ratioVolPerTick_over20': ("winsor", None, False, True, 1, 99, toBeDisplayed_if_s(user_choice, False)),

        # Imbalance features
        'bull_imbalance_low_1': ("winsor", None, False, False, 1, 99.5, toBeDisplayed_if_s(user_choice, False)),
        'bull_imbalance_low_2': ("winsor", None, False, False, 1, 96.5, toBeDisplayed_if_s(user_choice, False)), #Yeo-Johnson
        'bull_imbalance_low_3': ("winsor", None, False, False, 1, 96.5, toBeDisplayed_if_s(user_choice, False)),
        'bull_imbalance_high_0': ("winsor", None, False, False, 1, 96.5, toBeDisplayed_if_s(user_choice, False)),
        'bull_imbalance_high_1': ("winsor", None, False, False, 1, 96.5, toBeDisplayed_if_s(user_choice, False)),
        'bull_imbalance_high_2': ("winsor", None, False, False, 1, 96.5, toBeDisplayed_if_s(user_choice, False)),
        'bear_imbalance_low_0': ("winsor", None, False, False, 1, 96.5, toBeDisplayed_if_s(user_choice, False)),
        'bear_imbalance_low_1': ("winsor", None, False, False, 1, 96.5, toBeDisplayed_if_s(user_choice, False)),
        'bear_imbalance_low_2': ("winsor", None, False, False, 1, 96.5, toBeDisplayed_if_s(user_choice, False)),
        'bear_imbalance_high_1': ("winsor", None, False, False, 1, 96.5, toBeDisplayed_if_s(user_choice, False)),
        'bear_imbalance_high_2': ("winsor", None, False, False, 1, 96.5, toBeDisplayed_if_s(user_choice, False)),
        'bear_imbalance_high_3': ("winsor", None, False, False, 1, 96.5, toBeDisplayed_if_s(user_choice, False)),
        'imbalance_score_low': ("winsor", None, False, False, 1, 99, toBeDisplayed_if_s(user_choice, False)),
        'imbalance_score_high': ("winsor", None, False, False, 1, 99, toBeDisplayed_if_s(user_choice, False)),

        'is_imBullWithPoc_light_short': ("winsor", None, False, False, 1, 99, toBeDisplayed_if_s(user_choice, False)),
        'is_imBullWithPoc_aggressive_short': ("winsor", None, False, False, 1, 99, toBeDisplayed_if_s(user_choice, False)),
        'is_imBearWithPoc_light_long': ("winsor", None, False, False, 1, 99, toBeDisplayed_if_s(user_choice, False)),
        'is_imBearWithPoc_aggressive_long': (
        "winsor", None, False, False, 1, 99, toBeDisplayed_if_s(user_choice, False)),

        'is_imbBullLightPoc_AtrHigh0_1_short': ("winsor", None, False, False, 1, 99, toBeDisplayed_if_s(user_choice, False)),
        'is_imbBearLightPoc_AtrLow0_0_long': (
        "winsor", None, False, False, 1, 99, toBeDisplayed_if_s(user_choice, False)),

        # Auction features
        'finished_auction_high': ("winsor", None, False, False, 1, 99, toBeDisplayed_if_s(user_choice, False)),
        'finished_auction_low': ("winsor", None, False, False, 1, 99, toBeDisplayed_if_s(user_choice, False)),
        'staked00_high': ("winsor", None, False, False, 1, 99, toBeDisplayed_if_s(user_choice, False)),
        'staked00_low': ("winsor", None, False, False, 1, 99, toBeDisplayed_if_s(user_choice, False)),

        # POC distances
       # 'naked_poc_dist_above': ("winsor", None, True, True, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        #'naked_poc_dist_below': ("winsor", None, True, True, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),

        # Linear slope metrics
        'reg_slope_10P_2': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'reg_r2_10P_2': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'reg_std_10P_2': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),

        'reg_slope_50P_2': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'reg_r2_50P_2': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'reg_std_50P_2': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),

        'linear_slope_prevSession': ("winsor", None, True, True, 2, 98, toBeDisplayed_if_s(user_choice, False)),

        # SMA ratio metrics
        'close_sma_ratio_6': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'close_sma_ratio_14': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'close_sma_ratio_21': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'close_sma_ratio_30': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),

        # SMA z-score metrics
        'close_sma_zscore_6': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'close_sma_zscore_14': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'close_sma_zscore_21': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'close_sma_zscore_30': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),

        # VA price differences
        'diffPriceCloseVAH_0': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'diffPriceCloseVAL_0': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'ratio_delta_vol_VA_0': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),

        # Volume and movement ratios
        'sc_ratio_volRevMove_volImpulsMove': (
        "winsor", None, False, True, 0.0, 80, toBeDisplayed_if_s(user_choice, False)),
        'sc_ratio_deltaImpulsMove_volImpulsMove': (
        "winsor", None, False, False, 0.0, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'sc_ratio_deltaRevMove_volRevMove': ("winsor", None, True, True, 3, 80, toBeDisplayed_if_s(user_choice, False)),

        # Zone ratios
        'sc_ratio_volZone1_volExtrem': ("winsor", None, False, True, 0.0, 98, toBeDisplayed_if_s(user_choice, False)),
        'sc_ratio_deltaZone1_volZone1': ("winsor", None, True, True, 6, 96, toBeDisplayed_if_s(user_choice, False)),
        'sc_ratio_deltaExtrem_volExtrem': ("winsor", None, True, True, 2, 97, toBeDisplayed_if_s(user_choice, False)),

        # Continuation zone ratios
        'sc_ratio_volRevZone_xTicksContZone': (
        "winsor", None, False, True, 0.0, 99, toBeDisplayed_if_s(user_choice, False)),
        'sc_ratioDeltaXticksContZone_volXticksContZone': (
        "winsor", None, False, False, 0.0, 99.9, toBeDisplayed_if_s(user_choice, False)),

        # Movement strength ratios
        'sc_ratio_impulsMoveStrengthVol_xRevZone': (
        "winsor", None, False, False, 0.0, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'sc_ratio_revMoveStrengthVol_xRevZone': (
        "winsor", None, False, False, 0.0, 99.9, toBeDisplayed_if_s(user_choice, False)),

        # Imbalance type
        'sc_imbType_contZone': ("winsor", None, False, False, 0.0, 99.9, toBeDisplayed_if_s(user_choice, False)),

        # Detailed zone ratios
        'sc_ratio_volRevMoveZone1_volImpulsMoveExtrem_xRevZone': (
        "winsor", None, True, True, 0.0, 90, toBeDisplayed_if_s(user_choice, False)),
        'sc_ratio_volRevMoveZone1_volRevMoveExtrem_xRevZone': (
        "winsor", None, True, True, 0.0, 95, toBeDisplayed_if_s(user_choice, False)),
        'sc_ratio_deltaRevMoveZone1_volRevMoveZone1': (
        "winsor", None, True, True, 2, 95, toBeDisplayed_if_s(user_choice, False)),
        'sc_ratio_deltaRevMoveExtrem_volRevMoveExtrem': (
        "winsor", None, True, True, 2, 98, toBeDisplayed_if_s(user_choice, False)),
        'sc_ratio_volImpulsMoveExtrem_volImpulsMoveZone1_xRevZone': (
        "winsor", None, False, True, 0.0, 99, toBeDisplayed_if_s(user_choice, False)),
        'sc_ratio_deltaImpulsMoveZone1_volImpulsMoveZone1': (
        "winsor", None, False, False, 0.0, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'sc_ratio_deltaImpulsMoveExtrem_volImpulsMoveExtrem_xRevZone': (
        "winsor", None, False, False, 0.0, 99.9, toBeDisplayed_if_s(user_choice, False)),

        # DOM and VA metrics
        'sc_cumDOM_askBid_avgRatio': ("winsor", None, False, True, 0.0, 98, toBeDisplayed_if_s(user_choice, False)),
        'sc_cumDOM_askBid_pullStack_avgDiff_ratio': (
        "winsor", None, False, False, 2, 99, toBeDisplayed_if_s(user_choice, False)),
        'sc_delta_impulsMove_xRevZone_bigStand_extrem': (
        "winsor", None, False, True, 0.0, 99, toBeDisplayed_if_s(user_choice, False)),
        'sc_delta_revMove_xRevZone_bigStand_extrem': (
        "winsor", None, False, True, 0.0, 99, toBeDisplayed_if_s(user_choice, False)),

        # Misc ratios
        'sc_ratio_delta_vaVolVa': ("winsor", None, False, False, 0.0, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'sc_borderVa_vs_close': ("winsor", None, False, True, 0.0, 99, toBeDisplayed_if_s(user_choice, False)),
        'sc_ratio_volRevZone_volCandle': (
        "winsor", None, False, False, 0.0, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'sc_ratio_deltaRevZone_volCandle': (
        "winsor", None, False, False, 0.0, 99.9, toBeDisplayed_if_s(user_choice, False)),

        # Sierra chart regression metrics
        'sc_reg_slope_5P_2': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'sc_reg_std_5P_2': ("winsor", None, False, True, 0.5, 95, toBeDisplayed_if_s(user_choice, False)),
        'sc_reg_slope_10P_2': ("winsor", None, False, False, 0.5, 99.5, toBeDisplayed_if_s(user_choice, False)),
        'sc_reg_std_10P_2': ("winsor", None, False, True, 0.5, 95, toBeDisplayed_if_s(user_choice, False)),
        'sc_reg_slope_15P_2': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'sc_reg_std_15P_2': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'sc_reg_slope_30P_2': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'sc_reg_std_30P_2': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),

        'slope_range': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'slope_extrem': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'is_rangeSlope': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'is_extremSlope': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),

        # 'norm_diff_vwap': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        # 'is_vwap_shortArea': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        # 'is_vwap_notShortArea': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),

        'std_range': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'std_extrem': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'is_range_volatility_std': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'is_extrem_volatility_std': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),

        'zscore_range': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'zscore_extrem': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'is_zscore_range': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'is_zscore_extrem': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),

        # 'percent_b_high': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        # 'percent_b_low': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        # 'is_bb_high': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        # 'is_bb_low': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),

        'r2_range': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'r2_extrem': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'is_range_volatility_r2': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'is_extrem_volatility_r2': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),

        'rs_range_short': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'rs_extrem_short': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'rs_range_long': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'rs_extrem_long': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'is_rs_range_short': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'is_rs_extrem_short': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'is_rs_range_long': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'is_rs_extrem_long': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),

        # Zone volume ratios
        'sc_ratio_vol_volCont_zoneA_xTicksContZone': (
        "winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'sc_ratio_delta_volCont_zoneA_xTicksContZone': (
        "winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'sc_ratio_vol_volCont_zoneB_xTicksContZone': (
        "winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'sc_ratio_delta_volCont_zoneB_xTicksContZone': (
        "winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'sc_ratio_vol_volCont_zoneC_xTicksContZone': (
        "winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'sc_ratio_delta_volCont_zoneC_xTicksContZone': (
        "winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),

        #volume et duration mediane
        # Volume - Période 5
        # Volume - Période 5
        'volume_P25_5': (
            "winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'volume_P50_5': (
            "winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'volume_P75_5': (
            "winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'volume_P100_5': (
            "winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),

        # Volume - Période 10
        'volume_P25_10': (
            "winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'volume_P50_10': (
            "winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'volume_P75_10': (
            "winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'volume_P100_10': (
            "winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),

        # Volume - Période 20
        'volume_P25_20': (
            "winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'volume_P50_20': (
            "winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'volume_P75_20': (
            "winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'volume_P100_20': (
            "winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),

        # Volume - Période 30
        'volume_P25_30': (
            "winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'volume_P50_30': (
            "winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'volume_P75_30': (
            "winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'volume_P100_30': (
            "winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),

        # CandleDuration - Période 5
        'candleDuration_P25_5': (
            "winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'candleDuration_P50_5': (
            "winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'candleDuration_P75_5': (
            "winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'candleDuration_P100_5': (
            "winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),

        # CandleDuration - Période 10
        'candleDuration_P25_10': (
            "winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'candleDuration_P50_10': (
            "winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'candleDuration_P75_10': (
            "winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'candleDuration_P100_10': (
            "winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),

        # CandleDuration - Période 20
        'candleDuration_P25_20': (
            "winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'candleDuration_P50_20': (
            "winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'candleDuration_P75_20': (
            "winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'candleDuration_P100_20': (
            "winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),

        # CandleDuration - Période 30
        'candleDuration_P25_30': (
            "winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'candleDuration_P50_30': (
            "winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'candleDuration_P75_30': (
            "winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        'candleDuration_P100_30': (
            "winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
        # Time-related
        'sc_timeElapsed2LastBar': ("winsor", None, False, True, 0.0, 95, toBeDisplayed_if_s(user_choice, False)),

        'vptRatio_light_ratio_period_32': (
            "winsor", None, False, True, 0.0, 95, toBeDisplayed_if_s(user_choice, False)),
        'vptRatio_light_ratio_normalized': (
            "winsor", None, False, True, 0.0, 95, toBeDisplayed_if_s(user_choice, False)),
        'vptRatio_light_volatility': (
            "winsor", None, False, True, 0.0, 95, toBeDisplayed_if_s(user_choice, False)),

        # Indicateurs binaires LIGHT SHORT avec période
        'is_vptRatio_light_inbound_32': (
            "winsor", None, False, True, 0.0, 95, toBeDisplayed_if_s(user_choice, False)),
        'is_vptRatio_light_low_volume_32': (
            "winsor", None, False, True, 0.0, 95, toBeDisplayed_if_s(user_choice, False)),
        'is_vptRatio_light_high_volume_32': (
            "winsor", None, False, True, 0.0, 95, toBeDisplayed_if_s(user_choice, False)),
        'is_vptRatio_light_outbound_32': (
            "winsor", None, False, True, 0.0, 95, toBeDisplayed_if_s(user_choice, False)),
        'is_vptRatio_light_signal_short_32': (
            "winsor", None, False, True, 0.0, 95, toBeDisplayed_if_s(user_choice, False)),

        # ────────────────────────────────────────────────────────────────────────
        # LIGHT LONG (period=51, profile="light", direction="Long")
        # ────────────────────────────────────────────────────────────────────────
        'vptRatio_light_ratio_period_51': (
            "winsor", None, False, True, 0.0, 95, toBeDisplayed_if_s(user_choice, False)),

        # Indicateurs binaires LIGHT LONG avec période
        'is_vptRatio_light_inbound_51': (
            "winsor", None, False, True, 0.0, 95, toBeDisplayed_if_s(user_choice, False)),
        'is_vptRatio_light_low_volume_51': (
            "winsor", None, False, True, 0.0, 95, toBeDisplayed_if_s(user_choice, False)),
        'is_vptRatio_light_high_volume_51': (
            "winsor", None, False, True, 0.0, 95, toBeDisplayed_if_s(user_choice, False)),
        'is_vptRatio_light_outbound_51': (
            "winsor", None, False, True, 0.0, 95, toBeDisplayed_if_s(user_choice, False)),
        'is_vptRatio_light_signal_long_51': (
            "winsor", None, False, True, 0.0, 95, toBeDisplayed_if_s(user_choice, False)),

        # ────────────────────────────────────────────────────────────────────────
        # AGGRESSIVE SHORT (period=19, profile="aggressive", direction="Short")
        # ────────────────────────────────────────────────────────────────────────
        'vptRatio_aggressive_ratio_period_19': (
            "winsor", None, False, True, 0.0, 95, toBeDisplayed_if_s(user_choice, False)),
        'vptRatio_aggressive_ratio_normalized': (
            "winsor", None, False, True, 0.0, 95, toBeDisplayed_if_s(user_choice, False)),
        'vptRatio_aggressive_volatility': (
            "winsor", None, False, True, 0.0, 95, toBeDisplayed_if_s(user_choice, False)),

        # Indicateurs binaires AGGRESSIVE SHORT avec période
        'is_vptRatio_aggressive_inbound_19': (
            "winsor", None, False, True, 0.0, 95, toBeDisplayed_if_s(user_choice, False)),
        'is_vptRatio_aggressive_low_volume_19': (
            "winsor", None, False, True, 0.0, 95, toBeDisplayed_if_s(user_choice, False)),
        'is_vptRatio_aggressive_high_volume_19': (
            "winsor", None, False, True, 0.0, 95, toBeDisplayed_if_s(user_choice, False)),
        'is_vptRatio_aggressive_outbound_19': (
            "winsor", None, False, True, 0.0, 95, toBeDisplayed_if_s(user_choice, False)),
        'is_vptRatio_aggressive_signal_short_19': (
            "winsor", None, False, True, 0.0, 95, toBeDisplayed_if_s(user_choice, False)),

        # ────────────────────────────────────────────────────────────────────────
        # AGGRESSIVE LONG (period=51, profile="aggressive", direction="Long")
        # ────────────────────────────────────────────────────────────────────────
        'vptRatio_aggressive_ratio_period_51': (
            "winsor", None, False, True, 0.0, 95, toBeDisplayed_if_s(user_choice, False)),

        # Indicateurs binaires AGGRESSIVE LONG avec période
        'is_vptRatio_aggressive_inbound_51': (
            "winsor", None, False, True, 0.0, 95, toBeDisplayed_if_s(user_choice, False)),
        'is_vptRatio_aggressive_low_volume_51': (
            "winsor", None, False, True, 0.0, 95, toBeDisplayed_if_s(user_choice, False)),
        'is_vptRatio_aggressive_high_volume_51': (
            "winsor", None, False, True, 0.0, 95, toBeDisplayed_if_s(user_choice, False)),
        'is_vptRatio_aggressive_outbound_51': (
            "winsor", None, False, True, 0.0, 95, toBeDisplayed_if_s(user_choice, False)),
        'is_vptRatio_aggressive_signal_long_51': (
            "winsor", None, False, True, 0.0, 95, toBeDisplayed_if_s(user_choice, False)),

        # ────────────────────────────────────────────────────────────────────────
        # DEFAULT (period=32, profile="default", direction="Long")
        # ────────────────────────────────────────────────────────────────────────
        'vptRatio_ratio_period_32': (
            "winsor", None, False, True, 0.0, 95, toBeDisplayed_if_s(user_choice, False)),
        'vptRatio_ratio_normalized': (
            "winsor", None, False, True, 0.0, 95, toBeDisplayed_if_s(user_choice, False)),
        'vptRatio_volatility': (
            "winsor", None, False, True, 0.0, 95, toBeDisplayed_if_s(user_choice, False)),

        # Indicateurs binaires DEFAULT avec période
        'is_vptRatio_inbound_32': (
            "winsor", None, False, True, 0.0, 95, toBeDisplayed_if_s(user_choice, False)),
        'is_vptRatio_low_volume_32': (
            "winsor", None, False, True, 0.0, 95, toBeDisplayed_if_s(user_choice, False)),
        'is_vptRatio_high_volume_32': (
            "winsor", None, False, True, 0.0, 95, toBeDisplayed_if_s(user_choice, False)),
        'is_vptRatio_outbound_32': (
            "winsor", None, False, True, 0.0, 95, toBeDisplayed_if_s(user_choice, False)),
        'is_vptRatio_signal_long_32': (
            "winsor", None, False, True, 0.0, 95, toBeDisplayed_if_s(user_choice, False)),

        # ────────────────────────────────────────────────────────────────────────
        # CONSERVATIVE (period=51, profile="conservative", direction="Long")
        # ────────────────────────────────────────────────────────────────────────
        'vptRatio_conservative_ratio_period_51': (
            "winsor", None, False, True, 0.0, 95, toBeDisplayed_if_s(user_choice, False)),
        'vptRatio_conservative_ratio_normalized': (
            "winsor", None, False, True, 0.0, 95, toBeDisplayed_if_s(user_choice, False)),
        'vptRatio_conservative_volatility': (
            "winsor", None, False, True, 0.0, 95, toBeDisplayed_if_s(user_choice, False)),

        # Indicateurs binaires CONSERVATIVE avec période
        'is_vptRatio_conservative_inbound_51': (
            "winsor", None, False, True, 0.0, 95, toBeDisplayed_if_s(user_choice, False)),
        'is_vptRatio_conservative_low_volume_51': (
            "winsor", None, False, True, 0.0, 95, toBeDisplayed_if_s(user_choice, False)),
        'is_vptRatio_conservative_high_volume_51': (
            "winsor", None, False, True, 0.0, 95, toBeDisplayed_if_s(user_choice, False)),
        'is_vptRatio_conservative_outbound_51': (
            "winsor", None, False, True, 0.0, 95, toBeDisplayed_if_s(user_choice, False)),
        'is_vptRatio_conservative_signal_long_51': (
            "winsor", None, False, True, 0.0, 95, toBeDisplayed_if_s(user_choice, False)),

        # ────────────────────────────────────────────────────────────────────────
        # SELECTIVE (period=19, profile="selective", direction="Short")
        # ────────────────────────────────────────────────────────────────────────
        'vptRatio_selective_ratio_period_19': (
            "winsor", None, False, True, 0.0, 95, toBeDisplayed_if_s(user_choice, False)),
        'vptRatio_selective_ratio_normalized': (
            "winsor", None, False, True, 0.0, 95, toBeDisplayed_if_s(user_choice, False)),
        'vptRatio_selective_volatility': (
            "winsor", None, False, True, 0.0, 95, toBeDisplayed_if_s(user_choice, False)),

        # Indicateurs binaires SELECTIVE avec période
        'is_vptRatio_selective_inbound_19': (
            "winsor", None, False, True, 0.0, 95, toBeDisplayed_if_s(user_choice, False)),
        'is_vptRatio_selective_low_volume_19': (
            "winsor", None, False, True, 0.0, 95, toBeDisplayed_if_s(user_choice, False)),
        'is_vptRatio_selective_high_volume_19': (
            "winsor", None, False, True, 0.0, 95, toBeDisplayed_if_s(user_choice, False)),
        'is_vptRatio_selective_outbound_19': (
            "winsor", None, False, True, 0.0, 95, toBeDisplayed_if_s(user_choice, False)),
        'is_vptRatio_selective_signal_short_19': (
            "winsor", None, False, True, 0.0, 95, toBeDisplayed_if_s(user_choice, False)),

        # ----------------------------------------------------------------------------
        # SPECIAL 12 – Microstructure Anti-Spring & Anti-Épuisement
        # ----------------------------------------------------------------------------
        'is_antiSpring_short': (
            "winsor", None, False, True, 0.0, 95, toBeDisplayed_if_s(user_choice, False)),
        'is_antiEpuisement_long': (
            "winsor", None, False, True, 0.0, 95, toBeDisplayed_if_s(user_choice, False)),
        'volume_slope_antiSpring_short': (
            "winsor", None, False, True, 0.0, 95, toBeDisplayed_if_s(user_choice, False)),
        'duration_slope_antiSpring_short': (
            "winsor", None, False, True, 0.0, 95, toBeDisplayed_if_s(user_choice, False)),
        'volume_slope_antiEpuisement_long': (
            "winsor", None, False, True, 0.0, 95, toBeDisplayed_if_s(user_choice, False)),
        'duration_slope_antiEpuisement_long': (
            "winsor", None, False, True, 0.0, 95, toBeDisplayed_if_s(user_choice, False)),
        # Include any absorption settings
        **absorption_settings
    }
    return column_settings
def calculate_percentiles(df_NANValue, columnName, settings, nan_replacement_values=None):
    """
    Calcule les percentiles tout en gérant les valeurs NaN et les valeurs de remplacement.
    Évite les erreurs en cas de colonne entièrement NaN ou filtrée.
    """

    # Récupération des paramètres de winsorisation
    floor_enabled, crop_enabled, floorInf_percentage, cropSup_percentage, _ = settings[columnName]

    # Gestion des valeurs de remplacement NaN
    if nan_replacement_values is not None and columnName in nan_replacement_values:
        nan_value = nan_replacement_values[columnName]
        mask = df_NANValue[columnName] != nan_value
        nan_count = (~mask).sum()
        # print(f"   In calculate_percentiles:")
        # print(f"     - Filter out {nan_count} nan replacement value(s) {nan_value} for {columnName}")
    else:
        mask = df_NANValue[columnName].notna()
        nan_count = df_NANValue[columnName].isna().sum()
        # print(f"   In calculate_percentiles:")
        # print(f"     - {nan_count} NaN value(s) found in {columnName}")

    # Filtrage des valeurs valides
    filtered_values = df_NANValue.loc[mask, columnName].values

    # 🚨 Vérification si filtered_values est vide
    if filtered_values.size == 0:
        print(f"⚠️ Warning: No valid values found in '{columnName}', skipping percentile calculation.")
        return None, None  # Ou des valeurs par défaut, ex: return 0, 1

    # Calcul des percentiles en fonction des options activées
    floor_value = np.percentile(filtered_values, floorInf_percentage) if floor_enabled else None
    crop_value = np.percentile(filtered_values, cropSup_percentage) if crop_enabled else None

    # print(f"     - floor_value: {floor_value}   crop_value: {crop_value}")

    return floor_value, crop_value



def replace_nan_and_inf(df, columns_to_process, REPLACE_NAN=True):
    # Paramètres
    start_value = REPLACED_NANVALUE_BY
    increment = REPLACED_NANVALUE_BY_INDEX
    current_value = start_value
    nan_replacement_values = {}
    df_replaced = df.copy()

    for column in columns_to_process:
        # Combiner les masques pour NaN et Inf en une seule opération
        is_nan_or_inf = df[column].isna() | np.isinf(df[column])
        total_replacements = is_nan_or_inf.sum()

        if total_replacements > 0:
            nan_count = df[column].isna().sum()
            inf_count = np.isinf(df[column]).sum()

            print(f"Colonne problématique : {column}")
            print(f"Nombre de valeurs NaN : {nan_count}")
            print(f"Nombre de valeurs infinies : {inf_count}")

            if REPLACE_NAN:
                if start_value != 0:
                    df_replaced.loc[is_nan_or_inf, column] = current_value
                    nan_replacement_values[column] = current_value
                    print(f"L'option start_value != 0 est activée.")
                    print(
                        f"Les {total_replacements} valeurs NaN et infinies dans la colonne '{column}' ont été remplacées par {current_value}")
                    if increment != 0:
                        current_value += increment
                else:
                    print(
                        f"Les valeurs NaN et infinies dans la colonne '{column}' ont été laissées inchangées car start_value est 0")
            else:
                # Remplacer uniquement les valeurs infinies par NaN
                df_replaced.loc[np.isinf(df[column]), column] = np.nan
                inf_replacements = inf_count
                print(f"REPLACE_NAN est à False.")
                print(f"Les {inf_replacements} valeurs infinies dans la colonne '{column}' ont été remplacées par NaN")
                print(f"Les {nan_count} valeurs NaN dans la colonne '{column}' ont été laissées inchangées")
                print("Les valeurs NaN ne sont pas remplacées par une valeur choisie par l'utilisateur.")

    number_of_elementsnan_replacement_values = len(nan_replacement_values)
    print(f"Le dictionnaire nan_replacement_values contient {number_of_elementsnan_replacement_values} éléments.")
    return df_replaced, nan_replacement_values




def winsorize(features_NANReplacedVal_df, column, floor_value, crop_value, floor_enabled, crop_enabled,
              nan_replacement_values=None):
    # Créer une copie des données de la colonne spécifiée
    winsorized_data = features_NANReplacedVal_df[column].copy()

    # Assurez-vous que le nom de la série est préservé
    winsorized_data.name = column

    # Créer un masque pour exclure la valeur nan_value si spécifiée
    if nan_replacement_values is not None and column in nan_replacement_values:
        nan_value = nan_replacement_values[column]
        mask = features_NANReplacedVal_df[column] != nan_value
    else:
        # Si pas de valeur à exclure, on crée un masque qui sélectionne toutes les valeurs non-NaN
        mask = features_NANReplacedVal_df[column].notna()

    # Appliquer la winsorisation seulement sur les valeurs non masquées
    if floor_enabled:
        winsorized_data.loc[mask & (winsorized_data < floor_value)] = floor_value

    if crop_enabled:
        winsorized_data.loc[mask & (winsorized_data > crop_value)] = crop_value

    # S'assurer qu'il n'y a pas de NaN dans les données winsorisées
    # winsorized_data = winsorized_data.fillna(nan_replacement_values.get(column, winsorized_data.median()))

    return winsorized_data


def cropFloor_dataSource(features_NANReplacedVal_df, columnName, floorInf_booleen, cropSup_booleen, floorInf_percent,
                         cropSup_percent, nan_replacement_values=None):
    """
    Calcule les percentiles (floor et crop) tout en gérant les valeurs NaN et les valeurs de remplacement.
    """
    # Gestion des valeurs de remplacement NaN
    if nan_replacement_values is not None and columnName in nan_replacement_values:
        nan_value = nan_replacement_values[columnName]
        mask = features_NANReplacedVal_df[columnName] != nan_value
    else:
        mask = features_NANReplacedVal_df[columnName].notna()

    # Filtrage des valeurs valides
    filtered_values = features_NANReplacedVal_df.loc[mask, columnName].values

    # Vérification si filtered_values est vide
    if filtered_values.size == 0:
        print(f"⚠️ Warning: No valid values found in '{columnName}', skipping percentile calculation.")
        return None, None, floorInf_booleen, cropSup_booleen, floorInf_percent, cropSup_percent

    # Calcul des percentiles en fonction des options activées
    floor_valueNANfiltered = np.percentile(filtered_values, floorInf_percent) if floorInf_booleen else None
    crop_valueNANfiltered = np.percentile(filtered_values, cropSup_percent) if cropSup_booleen else None

    return floor_valueNANfiltered, crop_valueNANfiltered, floorInf_booleen, cropSup_booleen, floorInf_percent, cropSup_percent


import numpy as np

import numpy as np


def apply_winsorization(features_NANReplacedVal_df, columnName, floorInf_booleen, cropSup_booleen, floorInf_percent,
                        cropSup_percent, nan_replacement_values=None):
    """
    Calcule les percentiles et applique la winsorisation sur les données.
    """
    # Récupérer les valeurs pour la winsorisation
    floor_valueNANfiltered, crop_valueNANfiltered, _, _, _, _ = cropFloor_dataSource(
        features_NANReplacedVal_df,
        columnName,
        floorInf_booleen,
        cropSup_booleen,
        floorInf_percent,
        cropSup_percent,
        nan_replacement_values
    )

    # Winsorisation avec les valeurs NaN
    winsorized_valuesWithNanValue = winsorize(
        features_NANReplacedVal_df,
        columnName,
        floor_valueNANfiltered,
        crop_valueNANfiltered,
        floorInf_booleen,
        cropSup_booleen,
        nan_replacement_values
    )

    return winsorized_valuesWithNanValue


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_single_histogram(values_before, winsorized_values_after, column, floor_value, crop_value,
                          floorInf_values, cropSup_values, floorInf_percent, cropSup_percent, ax,
                          nan_replacement_values=None, range_strength_percent_in_range_10_32=None,
                          range_strength_percent_in_range_5_23=None, regimeAdx_pct_infThreshold=None,
                          adjust_xaxis=True):
    values_before_clean = values_before.dropna()

    sns.histplot(data=pd.DataFrame({column: values_before_clean}), x=column, color="blue", kde=False, ax=ax, alpha=0.7)
    sns.histplot(data=pd.DataFrame({column: winsorized_values_after}), x=column, color="red", kde=False, ax=ax,
                 alpha=0.7)

    if floorInf_values:
        ax.axvline(floor_value, color='g', linestyle='--', label=f'Floor ({floorInf_percent}%)')
    if cropSup_values:
        ax.axvline(crop_value, color='y', linestyle='--', label=f'Crop ({cropSup_percent}%)')

    def format_value(value):
        return f"{value:.2f}" if pd.notna(value) else "nan"

    initial_values = values_before_clean.sort_values()
    winsorized_values = winsorized_values_after.dropna().sort_values()

    ax.axvline(initial_values.iloc[0], color='blue',
               label=f'Init ({format_value(initial_values.iloc[0])}, {format_value(initial_values.iloc[-1])})')
    ax.axvline(winsorized_values.iloc[0], color='red',
               label=f'Winso ({format_value(winsorized_values.iloc[0])}, {format_value(winsorized_values.iloc[-1])})')

    if adjust_xaxis:
        # Assurez-vous que x_min prend en compte les valeurs négatives
        x_min = min(winsorized_values_after.min(), floor_value) if floorInf_values else winsorized_values_after.min()
        x_max = max(winsorized_values_after.max(), crop_value) if cropSup_values else winsorized_values_after.max()
        ax.set_xlim(left=x_min, right=x_max)

    # Keep the title
    ax.set_title(column, fontsize=6, pad=0.1)  # Title is kept

    # Clear the x-axis label to avoid duplication
    ax.set_xlabel('')  # This will clear the default x-axis label
    ax.set_ylabel('')
    # Reduce the font size of the legend
    ax.legend(fontsize=5)

    ax.tick_params(axis='both', which='major', labelsize=4.5)
    ax.xaxis.set_tick_params(labelsize=4.5, pad=0.1)
    ax.yaxis.set_tick_params(labelsize=4.5, pad=0.1)

    if nan_replacement_values and column in nan_replacement_values:
        ax.annotate(f"NaN replaced by: {nan_replacement_values[column]}",
                    xy=(0.05, 0.95), xycoords='axes fraction',
                    fontsize=5, ha='left', va='top')

    nan_count = winsorized_values_after.isna().sum()
    inf_count = np.isinf(winsorized_values_after).sum()
    nan_proportion = nan_count / len(winsorized_values_after)
    color_proportion = 'green' if nan_proportion < 0.3 else 'red'

    annotation_text = (
        f"Winsorized column:\n"
        f"Remaining NaN: {nan_count}\n"
        f"Remaining Inf: {inf_count}\n"
        f"nb period: {len(winsorized_values_after)}\n"
        f"% de np.nan : {nan_proportion:.2%}"
    )

    if column == 'range_strength_10_32' and range_strength_percent_in_range_10_32 is not None:
        annotation_text += f"\n% time in range: {range_strength_percent_in_range_10_32:.2f}%"
    elif column == 'range_strength_5_23' and range_strength_percent_in_range_5_23 is not None:
        annotation_text += f"\n% time in range: {range_strength_percent_in_range_5_23:.2f}%"
    elif column == 'market_regimeADX' and regimeAdx_pct_infThreshold is not None:
        annotation_text += f"\n% ADX < threshold: {regimeAdx_pct_infThreshold:.2f}%"

    ax.annotate(
        annotation_text,
        xy=(0.05, 0.85),
        xycoords='axes fraction',
        fontsize=5,
        ha='left',
        va='top',
        color=color_proportion,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.5)
    )


def plot_histograms_multi_figure(columns, figsize=(28, 20), graphs_per_figure=40):
    n_columns = len(columns)
    ncols = 7
    nrows = 4
    graphs_per_figure = ncols * nrows
    n_figures = math.ceil(n_columns / graphs_per_figure)

    figures = []
    all_axes = []

    for fig_num in range(n_figures):
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        figures.append(fig)
        axes = axes.flatten()
        all_axes.extend(axes)

        # Hide unused subplots
        for ax in axes[n_columns - fig_num * graphs_per_figure:]:
            ax.set_visible(False)

    return figures, all_axes


def calculate_max_ratio(values, condition, calc_max=False, std_multiplier=1):
    valid_ratios = values[condition]
    # Exclure les NaN des calculs
    valid_ratios = valid_ratios[~np.isnan(valid_ratios)]

    if len(valid_ratios) > 0:
        if calc_max:
            return valid_ratios.max()
        else:
            mean = np.mean(valid_ratios)
            std = np.std(valid_ratios)
            if mean < 0:
                return mean - std_multiplier * std
            else:
                return mean + std_multiplier * std
    else:
        return 0

def get_custom_section(minutes: int, custom_sections: dict) -> dict:
    """
    Retourne la section correspondant au nombre de minutes dans custom_sections.
    """
    for section_name, section in custom_sections.items():
        if section['start'] <= minutes < section['end']:
            return section
    # Retourne la dernière section si aucune correspondance
    return list(custom_sections.values())[-1]


def get_custom_section_index(minutes: int, custom_sections: dict) -> int:
    """
    Retourne le session_type_index correspondant au nombre de minutes dans custom_sections.

    Args:
        minutes (int): Nombre de minutes depuis 22h00
        custom_sections (dict): Dictionnaire des sections personnalisées

    Returns:
        int: session_type_index de la section correspondante
    """
    for section in custom_sections.values():
        if section['start'] <= minutes < section['end']:
            return section['session_type_index']
    # Retourne le session_type_index de la dernière section si aucune correspondance
    return list(custom_sections.values())[-1]['session_type_index']

@jit(nopython=True)
def fast_linear_regression_slope(x, y):
    """Calcule la pente de régression linéaire de manière optimisée"""
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_xx = np.sum(x * x)

    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
    return slope


@jit(nopython=True)
def calculate_slopes(close_values: np.ndarray, session_starts: np.ndarray, window: int) -> np.ndarray:
    n = len(close_values)
    results = np.full(n, np.nan)
    x = np.arange(window, dtype=np.float64)

    for i in range(n):
        # On cherche le début de la session actuelle
        session_start_idx = -1

        # Remonter pour trouver le début de session
        for j in range(i, -1, -1):  # On remonte jusqu'au début si nécessaire
            if session_starts[j]:
                session_start_idx = j
                break

        # S'il y a assez de barres depuis le début de session
        bars_since_start = i - session_start_idx + 1

        if bars_since_start >= window:
            end_idx = i + 1
            start_idx = end_idx - window
            # Vérifier que start_idx est après le début de session
            if start_idx >= session_start_idx:
                y = close_values[start_idx:end_idx]
                results[i] = fast_linear_regression_slope(x, y)

    return results


def apply_optimized_slope_calculation(data: pd.DataFrame, window: int) -> pd.Series:
    """
    Applique le calcul optimisé des pentes

    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame contenant les données
    window : int
        Taille de la fenêtre pour le calcul

    Returns:
    --------
    pd.Series : Série des pentes calculées
    """
    # Préparation des données numpy
    close_values = data['sc_close'].values
    session_starts = (data['sc_sessionStartEnd'] == 10).values

    # Calcul des pentes
    slopes = calculate_slopes(close_values, session_starts, window)

    # Conversion en pandas Series
    return pd.Series(slopes, index=data.index)


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from numba import jit



@jit(nopython=True)
def calculate_slopes_and_r2_numba(close_values, session_starts,
                                  window,
                                  clip_slope=True,
                                  include_close_bar=False):
    n = len(close_values)
    slopes = np.full(n, np.nan)
    r2s    = np.full(n, np.nan)
    stds   = np.full(n, np.nan)

    # pré-calculs
    x      = np.arange(1, window + 1, dtype=np.float64)
    sumX   = np.sum(x)
    sumXX  = np.sum(x * x)
    n_f    = float(window)

    for start_idx in range(n - window + 1):
        end_idx    = start_idx + window - 1             # borne haute incluse
        result_idx = end_idx if include_close_bar else end_idx + 1
        if result_idx >= n:
            continue

        # stop si la fenêtre traverse un début de session
        skip = False
        for i in range(start_idx + 1, end_idx + 1):
            if session_starts[i]:
                skip = True
                break
        if skip:
            continue

        # --- 1. extraire window valeurs exactement -------------
        y = close_values[start_idx : end_idx + 1]

        # --- 2. sommes utiles  ---------------------------------
        sumY  = np.sum(y)
        sumXY = np.sum(x * y)

        denom = n_f * sumXX - sumX * sumX
        if denom == 0.0:
            continue

        slope = (n_f * sumXY - sumX * sumY) / denom     # pente brute

        # --- 3. clipping éventuel -------------------------------
        if clip_slope:
            if   slope >  1.0: slope =  1.0
            elif slope < -1.0: slope = -1.0

        a = (sumY - slope * sumX) / n_f                 # ordonnée à l’origine

        # --- 4. écart-type et R² -------------------------------
        sum_sq = 0.0
        for j in range(window):
            diff = y[j] - (a + slope * x[j])
            sum_sq += diff * diff
        std_dev = np.sqrt(sum_sq / n_f)  if window > 1 else 0.0

        y_mean  = sumY / n_f
        ss_tot  = np.sum( (y - y_mean) ** 2)
        r2      = 1.0 - (sum_sq / ss_tot) if ss_tot > 0.0 else 0.0

        # --- 5. stockage ---------------------------------------
        slopes[result_idx] = slope
        r2s[result_idx]    = r2
        stds[result_idx]   = std_dev

    return slopes, r2s, stds

def apply_optimized_slope_r2_calculation(data: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Applique le calcul optimisé des pentes et des coefficients R² avec Numba.

    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame contenant les données.
    window : int
        Taille de la fenêtre pour le calcul.

    Returns:
    --------
    pd.DataFrame : DataFrame contenant deux colonnes : slope et r2.
    """

    print(f"  apply_optimized_slope_r2_calculation(df, window) {window} ")
    # Préparation des données numpy
    close_values = data['sc_close'].values
    session_starts = (data['sc_sessionStartEnd'] == 10).values

    # Calcul des pentes et des coefficients R²
    slopes, r2s,stds = calculate_slopes_and_r2_numba(close_values, session_starts, window)

    # Conversion en pandas DataFrame
    results_df = pd.DataFrame({
        f'reg_slope_{window}P_2': slopes,
        f'reg_r2_{window}P_2': r2s,
        f'reg_std_{window}P_2': stds
    }, index=data.index)

    return results_df

@jit(nopython=True)
def fast_calculate_previous_session_slope(close_values: np.ndarray, session_type_index: np.ndarray) -> np.ndarray:
    """
    Calcule rapidement la pente de la session précédente
    """
    n = len(close_values)
    slopes = np.full(n, np.nan)

    # Variables pour tracker la session précédente
    prev_session_start = 0
    prev_session_type = session_type_index[0]

    for i in range(1, n):
        curr_type = session_type_index[i]

        # Détection changement de session
        if curr_type != session_type_index[i - 1]:
            # Calculer la pente de la session précédente
            if prev_session_start < i - 1:  # S'assurer qu'il y a des points pour la régression
                x = np.arange(float(i - prev_session_start))
                y = close_values[prev_session_start:i]
                n_points = len(x)
                sum_x = np.sum(x)
                sum_y = np.sum(y)
                sum_xy = np.sum(x * y)
                sum_xx = np.sum(x * x)
                slope = (n_points * sum_xy - sum_x * sum_y) / (n_points * sum_xx - sum_x * sum_x)

                # Assigner la pente à la nouvelle session
                j = i
                while j < n and session_type_index[j] == curr_type:
                    slopes[j] = slope
                    j += 1

            # Mettre à jour les indices pour la prochaine session
            prev_session_start = i
            prev_session_type = curr_type

    return slopes


def calculate_previous_session_slope(df, data) -> pd.Series:
    """
    Wrapper pandas pour le calcul des pentes
    """

    # if len(features_df) != len(data):
    #     raise ValueError(f"Dimensions mismatch: features_df has {len(features_df)} rows but data has {len(data)} rows")

    close_values = df['sc_close'].values
    session_type_index = data['deltaCustomSessionIndex'].values

    slopes = fast_calculate_previous_session_slope(close_values, session_type_index)
    return pd.Series(slopes, index=data.index)

# Version originale pour comparaison
def linear_regression_slope_market_trend(series):
    X = np.arange(len(series)).reshape(-1, 1)
    y = series.values.reshape(-1, 1)
    model = LinearRegression().fit(X, y)
    return model.coef_[0][0]


def calculate_absorpsion_features(df, candle_rev_tick):
    # Création d'un nouveau DataFrame pour stocker uniquement les colonnes d'absorption
    absorption_features = pd.DataFrame(index=df.index)

    # Initialisation des colonnes d'absorption
    for tick in range(3, candle_rev_tick + 1):
        absorption_features[f'is_absorpsion_{tick}ticks_low'] = 0
        absorption_features[f'is_absorpsion_{tick}ticks_high'] = 0

    # Logique pour "sc_low"
    for tick in range(3, candle_rev_tick + 1):
        condition_low = (df['sc_askVolLow'] - df['sc_bidVolLow']) < 0
        for i in range(1, tick):
            condition_low &= (df[f'sc_askVolLow_{i}'] - df[f'sc_bidVolLow_{i}']) < 0

        absorption_features[f'is_absorpsion_{tick}ticks_low'] = condition_low.astype(int)

        if tick >= 4:
            for t in range(3, tick):
                absorption_features[f'is_absorpsion_{t}ticks_low'] = absorption_features[
                                                                         f'is_absorpsion_{t}ticks_low'] | condition_low.astype(
                    int)

    # Logique pour "sc_high"
    for tick in range(3, candle_rev_tick + 1):
        condition_high = (df['sc_askVolHigh'] - df['sc_bidVolHigh']) > 0
        for i in range(1, tick):
            condition_high &= (df[f'sc_askVolHigh_{i}'] - df[f'sc_bidVolHigh_{i}']) > 0

        absorption_features[f'is_absorpsion_{tick}ticks_high'] = condition_high.astype(int)

        if tick >= 4:
            for t in range(3, tick):
                absorption_features[f'is_absorpsion_{t}ticks_high'] = absorption_features[
                                                                          f'is_absorpsion_{t}ticks_high'] | condition_high.astype(
                    int)

    return absorption_features



def detect_market_regimeADX(data, period=14, adx_threshold=25):
    # Calcul de l'ADX
    data['plus_dm'] = np.where((data['sc_high'] - data['sc_high'].shift(1)) > (data['sc_low'].shift(1) - data['sc_low']),
                               np.maximum(data['sc_high'] - data['sc_high'].shift(1), 0), 0)
    data['minus_dm'] = np.where((data['sc_low'].shift(1) - data['sc_low']) > (data['sc_high'] - data['sc_high'].shift(1)),
                                np.maximum(data['sc_low'].shift(1) - data['sc_low'], 0), 0)
    data['tr'] = np.maximum(data['sc_high'] - data['sc_low'],
                            np.maximum(abs(data['sc_high'] - data['sc_close'].shift(1)),
                                       abs(data['sc_low'] - data['sc_close'].shift(1))))
    data['plus_di'] = 100 * data['plus_dm'].rolling(period).sum() / data['tr'].rolling(period).sum()
    data['minus_di'] = 100 * data['minus_dm'].rolling(period).sum() / data['tr'].rolling(period).sum()
    data['dx'] = 100 * abs(data['plus_di'] - data['minus_di']) / (data['plus_di'] + data['minus_di'])
    data['adx'] = data['dx'].rolling(period).mean()

    data['market_regimeADX'] = np.where(data['adx'] > adx_threshold, data['adx'], data['adx'])
    # data['market_regimeADX'] = data['market_regimeADX'].fillna(addDivBy0 if DEFAULT_DIV_BY0 else valueX)
    # Calcul du pourcentage de valeurs inférieures à adx_threshold
    total_count = len(data['adx'])
    below_threshold_count = (data['adx'] < adx_threshold).sum()
    regimeAdx_pct_infThreshold = (below_threshold_count / total_count) * 100

    print(f"Pourcentage de valeurs ADX inférieures à {adx_threshold}: {regimeAdx_pct_infThreshold:.2f}%")

    return data, regimeAdx_pct_infThreshold


def range_strength(data, range_strength_, window=14, atr_multiple=2, min_strength=0.05):
    data = data.copy()
    required_columns = ['sc_high', 'sc_low', 'sc_close']
    missing_cols = [col for col in required_columns if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Les colonnes manquantes dans le DataFrame : {missing_cols}")

    for col in required_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    data['tr'] = np.maximum(
        data['sc_high'] - data['sc_low'],
        np.maximum(
            np.abs(data['sc_high'] - data['sc_close'].shift()),
            np.abs(data['sc_low'] - data['sc_close'].shift())
        )
    )

    data['atr'] = data['tr'].rolling(window=window).mean()
    data['threshold'] = (data['atr'] / data['sc_close'].replace(0, np.nan)) * atr_multiple
    data['rolling_high'] = data['sc_high'].rolling(window=window).max()
    data['rolling_low'] = data['sc_low'].rolling(window=window).min()
    data['range_width'] = (data['rolling_high'] - data['rolling_low']) / data['rolling_low']

    condition = data['range_width'] <= data['threshold']
    data['range_duration'] = condition.astype(int).groupby((~condition).cumsum()).cumsum()
    data[range_strength_] = data['range_duration'] / (1 + data['range_width'])

    # Appliquer un seuil minimum et une transformation logarithmique
    data[range_strength_] = np.where(data[range_strength_] < min_strength, np.nan, data[range_strength_])

    # data['log_range_strength'] = np.log1p(data[range_strength_])

    # Calculer le pourcentage de temps en range et hors range
    total_periods = len(data)
    in_range_periods = (data[range_strength_].notna()).sum()
    out_of_range_periods = total_periods - in_range_periods

    range_strength_percent_in_range = (in_range_periods / total_periods) * 100
    range_strength_percent_out_of_range = (out_of_range_periods / total_periods) * 100

    print(f"Pourcentage de temps en range: {range_strength_percent_in_range:.2f}%")
    print(f"Pourcentage de temps hors range: {range_strength_percent_out_of_range:.2f}%")

    data.drop(['tr'], axis=1, inplace=True)

    return data, range_strength_percent_in_range


def valueArea_pct(data, nbPeriods):
    # Calculate the difference between the sc_high and sc_low value area bands
    bands_difference = data[f'sc_vaH_{nbPeriods}periods'] - data[f'sc_vaL_{nbPeriods}periods']

    # Calculate percentage relative to POC, handling division by zero with np.nan
    result = np.where(bands_difference != 0,
                      (data['sc_close'] - data[f'sc_vaPoc_{nbPeriods}periods']) / bands_difference,
                      np.nan)

    # Convert the result into a pandas Series
    return pd.Series(result, index=data.index)


def compute_stoch(sc_high, sc_low, sc_close, session_starts, k_period=14, d_period=3, fill_value=50):
    """
    Calcule l'oscillateur stochastique (%K et %D) en respectant les limites de chaque session.
    Version optimisée utilisant des opérations vectorisées.

    Parameters:
    -----------
    sc_high : array-like
        Série des prix les plus hauts
    sc_low : array-like
        Série des prix les plus bas
    sc_close : array-like
        Série des prix de fermeture
    session_starts : array-like (booléen)
        Indicateur de début de session (True lorsqu'une nouvelle session commence)
    k_period : int, default=14
        Période pour calculer le stochastique %K
    d_period : int, default=3
        Période pour la moyenne mobile du %K qui donne le %D
    fill_value : float, default=50
        Valeur par défaut pour remplacer les NaN ou divisions par zéro

    Returns:
    --------
    tuple
        (k_values, d_values) - Un tuple contenant les valeurs %K et %D
    """
    # Créer un DataFrame pour traitement
    df = pd.DataFrame({
        'sc_high': sc_high,
        'sc_low': sc_low,
        'sc_close': sc_close,
        'session_start': session_starts
    })

    # Créer identifiant de session
    df['session_id'] = df['session_start'].cumsum()

    # Indexer chaque barre dans sa session pour filtrage ultérieur
    df['bar_index_in_session'] = df.groupby('session_id').cumcount()

    # Calcul vectorisé des plus hauts et plus bas sur la période k_period
    df['highest_high'] = (
        df.groupby('session_id')['sc_high']
        .rolling(window=k_period, min_periods=1)
        .max()
        .reset_index(level=0, drop=True)
    )

    df['lowest_low'] = (
        df.groupby('session_id')['sc_low']
        .rolling(window=k_period, min_periods=1)
        .min()
        .reset_index(level=0, drop=True)
    )

    # Calculer %K (Stochastique Rapide) vectorisé
    denominator = df['highest_high'] - df['lowest_low']
    df['%K'] = np.where(
        denominator > 0,
        ((df['sc_close'] - df['lowest_low']) / denominator) * 100,
        fill_value
    )

    # Marquer les positions n'ayant pas assez d'historique avec la valeur par défaut
    df.loc[df['bar_index_in_session'] < (k_period - 1), '%K'] = fill_value

    # Calculer %D (moyenne mobile du %K) vectorisé par session
    df['%D'] = (
        df.groupby('session_id')['%K']
        .rolling(window=d_period, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # Gérer les positions n'ayant pas assez d'historique pour %D
    # (k_period - 1 + d_period - 1) points nécessaires au total
    df.loc[df['bar_index_in_session'] < (k_period + d_period - 2), '%D'] = fill_value

    # Gestion des NaN
    df['%K'] = df['%K'].fillna(fill_value)
    df['%D'] = df['%D'].fillna(fill_value)

    # Limiter aux valeurs valides du stochastique (entre 0 et 100)
    df['%K'] = np.clip(df['%K'], 0, 100)
    df['%D'] = np.clip(df['%D'], 0, 100)

    # Retourner les valeurs sous forme de numpy arrays
    return df['%K'].to_numpy(), df['%D'].to_numpy()


def compute_wr(sc_high, sc_low, sc_close, session_starts, period=14, fill_value=-50):
    """
    Calcule l'indicateur Williams %R en respectant les limites de chaque session.
    Version optimisée utilisant des opérations vectorisées.
    """
    # Créer un DataFrame pour traitement
    df = pd.DataFrame({
        'sc_high': sc_high,
        'sc_low': sc_low,
        'sc_close': sc_close,
        'session_start': session_starts
    })

    # Créer identifiant de session
    df['session_id'] = df['session_start'].cumsum()

    # Calcul du rolling max et min par session
    df['highest_high'] = (
        df.groupby('session_id')['sc_high']
        .rolling(window=period, min_periods=1)
        .max()
        .reset_index(level=0, drop=True)
    )

    df['lowest_low'] = (
        df.groupby('session_id')['sc_low']
        .rolling(window=period, min_periods=1)
        .min()
        .reset_index(level=0, drop=True)
    )

    # Calculer le Williams %R vectorisé
    denominator = df['highest_high'] - df['lowest_low']
    df['wr'] = np.where(
        denominator > 0,
        ((df['highest_high'] - df['sc_close']) / denominator) * -100,
        fill_value
    )

    # Identifier les positions dans chaque session qui n'ont pas assez d'historique
    df['bar_index_in_session'] = df.groupby('session_id').cumcount()
    df.loc[df['bar_index_in_session'] < (period - 1), 'wr'] = fill_value

    # Gestion des NaN
    df['wr'] = df['wr'].fillna(fill_value)

    # Limiter aux valeurs valides du Williams %R (entre -100 et 0)
    df['wr'] = np.clip(df['wr'], -100, 0)

    return df['wr'].to_numpy()

import pandas as pd
import numpy as np

def compute_mfi(
    sc_high,
    sc_low,
    sc_close,
    sc_volume,
    session_starts,
    period=14,
    fill_value=50
):
    """
    Calcule l'indicateur Money Flow Index (MFI) en réinitialisant le calcul
    à chaque nouvelle session, sans déborder sur la session précédente.

    Parameters
    ----------
    sc_high : array-like
        Séries des prix les plus hauts
    sc_low : array-like
        Séries des prix les plus bas
    sc_close : array-like
        Séries des prix de clôture
    volume : array-like
        Séries des volumes
    session_starts : array-like de bool
        Indique, pour chaque barre, si c'est le début d'une nouvelle session (True) ou non (False)
    period : int, default=14
        Période de calcul du MFI
    fill_value : float, default=50
        Valeur par défaut à utiliser lorsque le MFI n'est pas calculable (ex: début de session ou NaN)

    Returns
    -------
    np.ndarray
        Tableau des valeurs du MFI, réinitialisé à chaque session
    """

    # Convertit tous les inputs en Series alignées sur le même index
    df = pd.DataFrame({
        'sc_high'          : sc_high,
        'sc_low'           : sc_low,
        'sc_close'         : sc_close,
        'sc_volume'        : sc_volume,
        'session_starts': session_starts
    })

    # Identifiants de session (on incrémente de 1 à chaque True)
    # Exemple : [F, F, T, F, F, T, F] -> [0, 0, 1, 1, 1, 2, 2]
    df['session_id'] = df['session_starts'].cumsum()

    # Typical Price
    df['tp'] = (df['sc_high'] + df['sc_low'] + df['sc_close']) / 3

    # Money Flow brut
    df['mf'] = df['tp'] * df['sc_volume']

    # On compare la typical price avec celle de la barre précédente (shift)
    df['tp_shifted'] = df['tp'].shift(1).fillna(df['tp'].iloc[0] if len(df) else 0)

    # Déterminer la partie positive/négative du flux
    df['positive_flow'] = np.where(df['tp'] > df['tp_shifted'], df['mf'], 0)
    # On ajoute une très petite valeur pour éviter d'avoir 0 exact
    df['negative_flow'] = np.where(df['tp'] < df['tp_shifted'], df['mf'], 0) + 1e-10

    # Rolling sum par session_id
    # -> groupby('session_id').rolling(window=period) ...
    df['sum_positive'] = (
        df.groupby('session_id')['positive_flow']
          .rolling(window=period, min_periods=1)
          .sum()
          .reset_index(level=0, drop=True)
    )
    df['sum_negative'] = (
        df.groupby('session_id')['negative_flow']
          .rolling(window=period, min_periods=1)
          .sum()
          .reset_index(level=0, drop=True)
    )

    # Ratio
    df['mfr'] = df['sum_positive'] / df['sum_negative'].clip(lower=1e-10)
    df['mfi'] = 100 - (100 / (1.0 + df['mfr']))

    # À l'intérieur d'une session, pour les premières barres (< period), on a trop peu d'historique
    # => on force ces valeurs à fill_value
    # cumcount() numérote les lignes de chaque session, à partir de 0
    # si cumcount() < period-1, on n'a pas assez de barres pour un 'vrai' MFI
    df['bar_index_in_session'] = df.groupby('session_id').cumcount()
    df.loc[df['bar_index_in_session'] < (period - 1), 'mfi'] = fill_value

    # Remplacer éventuellement les NaN restants par fill_value
    df['mfi'] = df['mfi'].fillna(fill_value)

    # Clip [0, 100]
    df['mfi'] = np.clip(df['mfi'], 0, 100)

    return df['mfi'].to_numpy()
import numba
@numba.jit(nopython=True)
def calculate_atr_numba(sc_high, sc_low, close_prev, window=14):
    """Recalcule l'ATR avec une fenêtre personnalisée en utilisant Numba pour l'accélération

    Args:
        sc_high: numpy array des prix hauts
        sc_low: numpy array des prix bas
        close_prev: numpy array des prix de clôture précédents (déjà décalés)
        window: taille de la fenêtre pour l'ATR

    Returns:
        numpy array de l'ATR
    """
    # True Range
    tr1 = sc_high - sc_low
    tr2 = np.abs(sc_high - close_prev)
    tr3 = np.abs(sc_low - close_prev)

    # Calculer le True Range comme maximum des trois
    tr = np.zeros_like(sc_high)
    for i in range(len(sc_high)):
        tr[i] = max(tr1[i], tr2[i], tr3[i])

    # Calcul de l'ATR
    atr = np.zeros_like(tr)
    atr[0] = tr[0]  # Première valeur = premier TR

    # Calcul de l'ATR par moyenne mobile exponentielle
    alpha = 2.0 / (window + 1.0)
    for i in range(1, len(atr)):
        atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]

    return atr
def calculate_atr(df, window=14):
    """Wrapper pour le calcul de l'ATR utilisant la version optimisée avec Numba"""
    sc_high = df['sc_high'].values
    sc_low = df['sc_low'].values

    # Créer le sc_close précédent (décalé d'une position)
    if len(df['sc_close']) > 0:
        close_prev = np.empty_like(df['sc_close'].values)
        close_prev[0] = df['sc_close'].values[0]  # La première valeur reste la même
        close_prev[1:] = df['sc_close'].values[:-1]  # Décalage pour les autres valeurs
    else:
        close_prev = np.array([])

    return calculate_atr_numba(sc_high, sc_low, close_prev, window)
def add_stochastic_force_indicators(df, features_df,
                                    k_period_overbought, d_period_overbought,
                                    k_period_oversold, d_period_oversold,
                                    overbought_threshold=80, oversold_threshold=20,
                                    fi_short=1, fi_long=6):
    """
    Ajoute le Stochastique Rapide et le Force Index aux features,
    avec des périodes distinctes pour les zones de surachat et survente.

    Paramètres:
    - df: DataFrame source contenant les données brutes
    - features_df: DataFrame de destination pour les features
    - k_period_overbought: Période %K pour la détection de surachat
    - d_period_overbought: Période %D pour la détection de surachat
    - k_period_oversold: Période %K pour la détection de survente
    - d_period_oversold: Période %D pour la détection de survente
    - overbought_threshold: Seuil de surachat (défaut: 80)
    - oversold_threshold: Seuil de survente (défaut: 20)
    - fi_short: Période court terme pour le Force Index
    - fi_long: Période long terme pour le Force Index

    Retourne:
    - features_df avec les nouvelles colonnes d'indicateurs techniques
    """
    if (k_period_overbought is None or d_period_overbought is None or
            k_period_oversold is None or d_period_oversold is None):
        raise ValueError("Toutes les périodes pour surachat et survente doivent être spécifiées")

    try:
        sc_high = pd.to_numeric(df['sc_high'], errors='coerce')
        sc_low = pd.to_numeric(df['sc_low'], errors='coerce')
        sc_close = pd.to_numeric(df['sc_close'], errors='coerce')
        volume = pd.to_numeric(df['sc_volume'], errors='coerce')
        candle_dir = pd.to_numeric(df['sc_candleDir'], errors='coerce')
        session_starts = (df['sc_sessionStartEnd'] == 10).values

        k_overbought, d_overbought = compute_stoch(sc_high, sc_low, sc_close, session_starts,
                                                   k_period_overbought,
                                                   d_period_overbought,
                                                   fill_value=50)

        k_oversold, d_oversold = compute_stoch(sc_high, sc_low, sc_close, session_starts,
                                               k_period_oversold,
                                               d_period_oversold,
                                               fill_value=50)

        features_df['stoch_k_overbought'] = k_overbought
        features_df['stoch_d_overbought'] = d_overbought
        features_df['stoch_k_oversold'] = k_oversold
        features_df['stoch_d_oversold'] = d_oversold

        price_change = sc_close.diff().fillna(0)
        force_index_raw = price_change * volume

        # Force Index court terme
        features_df[f'force_index_short_{fi_short}'] = pd.Series(force_index_raw).ewm(span=fi_short, adjust=False).mean().values

        # Force Index long terme
        features_df[f'force_index_long_{fi_long}'] = pd.Series(force_index_raw).ewm(span=fi_long, adjust=False).mean().values

        # Extraction des valeurs calculées
        stoch_k_overbought = features_df['stoch_k_overbought'].astype(float)
        stoch_d_overbought = features_df['stoch_d_overbought'].astype(float)
        stoch_k_oversold = features_df['stoch_k_oversold'].astype(float)
        stoch_d_oversold = features_df['stoch_d_oversold'].astype(float)

        force_short = features_df[f'force_index_short_{fi_short}'].astype(float)
        force_long = features_df[f'force_index_long_{fi_long}'].astype(float)

        # Crossover pour surachat
        stoch_cross_overbought = np.zeros(len(stoch_k_overbought))
        for i in range(1, len(stoch_k_overbought)):
            if pd.notna(stoch_k_overbought[i]) and pd.notna(stoch_d_overbought[i]) and pd.notna(stoch_k_overbought[i - 1]) and pd.notna(stoch_d_overbought[i - 1]):
                if stoch_k_overbought[i - 1] < stoch_d_overbought[i - 1] and stoch_k_overbought[i] > stoch_d_overbought[i]:
                    stoch_cross_overbought[i] = 1
                elif stoch_k_overbought[i - 1] > stoch_d_overbought[i - 1] and stoch_k_overbought[i] < stoch_d_overbought[i]:
                    stoch_cross_overbought[i] = -1

        # Crossover pour survente
        stoch_cross_oversold = np.zeros(len(stoch_k_oversold))
        for i in range(1, len(stoch_k_oversold)):
            if pd.notna(stoch_k_oversold[i]) and pd.notna(stoch_d_oversold[i]) and pd.notna(stoch_k_oversold[i - 1]) and pd.notna(stoch_d_oversold[i - 1]):
                if stoch_k_oversold[i - 1] < stoch_d_oversold[i - 1] and stoch_k_oversold[i] > stoch_d_oversold[i]:
                    stoch_cross_oversold[i] = 1
                elif stoch_k_oversold[i - 1] > stoch_d_oversold[i - 1] and stoch_k_oversold[i] < stoch_d_oversold[i]:
                    stoch_cross_oversold[i] = -1

        features_df['is_stoch_overbought'] = np.where(stoch_k_overbought > overbought_threshold, 1, 0)
        features_df['is_stoch_oversold'] = np.where(stoch_k_oversold < oversold_threshold, 1, 0)

        avg_volume_20 = volume.rolling(window=4).mean().fillna(volume)

        fi_short_norm = np.where(avg_volume_20 > 0, force_short / avg_volume_20, 0)
        fi_long_norm = np.where(avg_volume_20 > 0, force_long / avg_volume_20, 0)

        features_df[f'force_index_short_{fi_short}_norm'] = fi_short_norm
        features_df[f'force_index_long_{fi_long}_norm'] = fi_long_norm

        features_df['force_index_divergence'] = fi_short_norm - fi_long_norm
        features_df['fi_momentum'] = np.sign(force_short) * np.abs(fi_short_norm)

        return features_df

    except Exception as e:
        print(f"Erreur dans add_stochastic_force_indicators: {str(e)}")
        return features_df



def add_atr(df, features_df, atr_period_range=14, atr_period_extrem=14,
            atr_low_threshold_range=2, atr_high_threshold_range=5,
            atr_low_threshold_extrem=1):
    """
    Ajoute l'indicateur ATR (Average True Range) et des signaux dérivés au DataFrame de features.
    Utilise potentiellement des périodes différentes pour les indicateurs de range et extremLow.

    Paramètres:
    - df: DataFrame contenant les colonnes 'sc_high', 'sc_low', 'sc_close'
    - features_df: DataFrame où ajouter les colonnes liées à l'ATR
    - atr_period_range: Période de calcul de l'ATR pour l'indicateur range (défaut: 14)
    - atr_period_extrem: Période de calcul de l'ATR pour l'indicateur extremLow (défaut: 14)
    - atr_low_threshold_range: Seuil bas pour la plage modérée d'ATR (défaut: 2)
    - atr_high_threshold_range: Seuil haut pour la plage modérée d'ATR (défaut: 5)
    - atr_low_threshold_extrem: Seuil bas pour les valeurs extrêmes d'ATR (défaut: 1)

    Retourne:
    - features_df enrichi des colonnes ATR et dérivées
    """
    try:
        # Calcul de l'ATR avec la période optimisée pour l'indicateur range
        atr_values_range = calculate_atr(df, atr_period_range)

        # Calcul de l'ATR avec la période optimisée pour l'indicateur extremLow
        # Si les périodes sont identiques, éviter de calculer deux fois
        if atr_period_range == atr_period_extrem:
            atr_values_extrem = atr_values_range
        else:
            atr_values_extrem = calculate_atr(df, atr_period_extrem)

        # Ajouter les valeurs brutes d'ATR au DataFrame de features
        features_df['atr_range'] = atr_values_range
        features_df['atr_extrem'] = atr_values_extrem

        # Créer l'indicateur pour la plage "modérée" d'ATR (optimisée pour le win rate)
        features_df['is_atr_range'] = np.where(
            (atr_values_range > atr_low_threshold_range) & (atr_values_range < atr_high_threshold_range),
            1, 0
        )

        # Créer l'indicateur pour les valeurs extrêmement basses d'ATR
        features_df['is_atr_extremLow'] = np.where(
            (atr_values_extrem < atr_low_threshold_extrem),
            1, 0
        )

        # S'assurer que toutes les colonnes sont numériques
        for col in ['atr_range', 'atr_extrem', 'is_atr_range', 'is_atr_extremLow']:
            features_df[col] = pd.to_numeric(features_df[col], errors='coerce').fillna(0)


    except Exception as e:
        print(f"Erreur dans add_atr: {str(e)}")
        # En cas d'erreur, tenter de renvoyer au moins les colonnes existantes
        if 'atr_range' not in features_df.columns:
            features_df['atr_range'] = 0
        if 'atr_extrem' not in features_df.columns:
            features_df['atr_extrem'] = 0
        if 'is_atr_range' not in features_df.columns:
            features_df['is_atr_range'] = 0
        if 'is_atr_extremLow' not in features_df.columns:
            features_df['is_atr_extremLow'] = 0

    return features_df


def add_regression_slope(df, features_df,
                         period_range=14, period_extrem=14,
                         slope_range_threshold_low=0.1, slope_range_threshold_high=0.5,
                         slope_extrem_threshold_low=0.1, slope_extrem_threshold_high=0.5):
    """
    Ajoute les indicateurs de régression de pente au DataFrame de features.
    Utilise des seuils spécifiques à chaque période.

    Paramètres:
    - df: DataFrame contenant les données de prix
    - features_df: DataFrame où ajouter les indicateurs
    - period_low: Période pour le calcul de la pente de l'indicateur is_rangeSlope
    - period_high: Période pour le calcul de la pente de l'indicateur is_extremSlope
    - slope_range_threshold_low: Seuil bas pour la détection des pentes modérées (period_low)
    - slope_extrem_threshold_low: Seuil haut pour la détection des pentes modérées (period_low)
    - slope_range_threshold_high: Seuil bas pour la détection des pentes fortes (period_high)
    - slope_extrem_threshold_high: Seuil haut pour la détection des pentes fortes (period_high)

    Retourne:
    - features_df enrichi des indicateurs de pente
    """
    try:
        sc_close = pd.to_numeric(df['sc_close'], errors='coerce').values
        session_starts = (df['sc_sessionStartEnd'] == 10).values

        # Calcul des pentes pour l'indicateur is_rangeSlope
        slopes_low, r2_low, std_low = calculate_slopes_and_r2_numba(sc_close, session_starts, period_range)

        # Calcul des pentes pour l'indicateur is_extremSlope (uniquement si période différente)
        if period_range == period_extrem:
            slopes_high = slopes_low
        else:
            slopes_high, r2_high, std_high = calculate_slopes_and_r2_numba(sc_close, session_starts, period_extrem)

        # Ajouter les valeurs brutes au DataFrame de features
        features_df['slope_range'] = slopes_low
        if period_range != period_extrem:
            features_df['slope_extrem'] = slopes_high
        else:
            features_df['slope_extrem'] = slopes_low

        # Créer l'indicateur is_rangeSlope (pentes modérées optimisées pour maximiser le win rate)
        # is_rangeSlope = 1 quand la pente est entre slope_range_threshold_low et slope_extrem_threshold_low
        features_df['is_rangeSlope'] = np.where(
            (slopes_low > slope_range_threshold_low) & (slopes_low < slope_range_threshold_high),
            1, 0
        )

        # Créer l'indicateur is_extremSlope (pentes fortes optimisées pour minimiser le win rate)
        # is_extremSlope = 1 quand la pente est soit inférieure à slope_range_threshold_high
        # soit supérieure à slope_extrem_threshold_high
        features_df['is_extremSlope'] = np.where(
            (slopes_high < slope_extrem_threshold_low) | (slopes_high > slope_extrem_threshold_high),
            1, 0
        )

        # S'assurer que toutes les colonnes sont numériques
        for col in ['slope_range', 'slope_extrem', 'is_rangeSlope', 'is_extremSlope']:
            if col in features_df.columns:
                features_df[col] = pd.to_numeric(features_df[col], errors='coerce').fillna(0)


    except Exception as e:
        print(f"Erreur dans add_regression_slope: {str(e)}")
        # En cas d'erreur, tenter de renvoyer au moins les colonnes existantes
        if 'slope_range' not in features_df.columns:
            features_df['slope_range'] = 0
        if 'slope_extrem' not in features_df.columns:
            features_df['slope_extrem'] = 0
        if 'is_rangeSlope' not in features_df.columns:
            features_df['is_rangeSlope'] = 0
        if 'is_extremSlope' not in features_df.columns:
            features_df['is_extremSlope'] = 0

    return features_df




@njit
def std_ddof1(x):
    """
    Implémentation manuelle de l'écart-type avec ddof=1 pour contourner la limitation de Numba.

    Parameters
    ----------
    x : numpy.ndarray
        Tableau de données

    Returns
    -------
    float
        Écart-type avec ddof=1
    """
    n = len(x)
    if n <= 1:
        return 0.0

    mean = np.mean(x)
    var = 0.0
    for i in range(n):
        var += (x[i] - mean) ** 2

    # Division par (n-1) pour ddof=1
    return np.sqrt(var / (n - 1))


@njit
def rolling_mean(x, window):
    """
    Calcule une moyenne mobile simple sur un tableau.

    Parameters
    ----------
    x : numpy.ndarray
        Tableau de données
    window : int
        Taille de la fenêtre

    Returns
    -------
    numpy.ndarray
        Moyenne mobile
    """
    n = len(x)
    result = np.full(n, np.nan)

    for i in range(n):
        start_idx = max(0, i - window + 1)
        window_values = x[start_idx:i + 1]
        result[i] = np.mean(window_values)

    return result


@njit
def calculate_percent_bb_numba(sc_high, sc_low, sc_close, session_starts, period, std_dev, fill_value):
    """
    Implémentation optimisée de calculate_percent_bb en Numba avec gestion améliorée des cas extrêmes.

    Parameters
    ----------
    sc_high, sc_low, sc_close : numpy.ndarray
        Tableaux des valeurs sc_high, sc_low, sc_close
    session_starts : numpy.ndarray
        Tableau booléen indiquant les débuts de session
    period : int
        Période pour le calcul des bandes de Bollinger
    std_dev : float
        Nombre d'écarts-types pour les bandes
    fill_value : float
        Valeur à utiliser pour les premières barres

    Returns
    -------
    numpy.ndarray
        Tableau %B calculé avec valeurs limitées entre -1 et 2
    """
    n = len(sc_high)
    hlc_avg = (sc_high + sc_low + sc_close) / 3.0

    # Créer un tableau d'IDs de session
    session_ids = np.zeros(n, dtype=np.int32)
    current_id = 0

    for i in range(n):
        if session_starts[i]:
            current_id += 1
        session_ids[i] = current_id

    # Initialiser le tableau de résultats
    percent_b = np.full(n, fill_value, dtype=np.float64)

    # Traiter chaque session séparément
    max_session_id = session_ids.max()

    for sid in range(1, max_session_id + 1):
        # Estimer d'abord la taille de la session pour pré-allouer
        n_session_estimate = 0
        for i in range(n):
            if session_ids[i] == sid:
                n_session_estimate += 1

        # Obtenir les indices de cette session avec pré-allocation optimisée
        session_indices = np.zeros(n_session_estimate, dtype=np.int32)
        n_session = 0

        for i in range(n):
            if session_ids[i] == sid:
                session_indices[n_session] = i
                n_session += 1

        if n_session == 0:
            continue

        # Extraire les données de la session
        session_hlc = np.zeros(n_session)
        for i in range(n_session):
            idx = session_indices[i]
            session_hlc[i] = hlc_avg[idx]

        # Calculer les moyennes mobiles et écarts-types
        means = np.full(n_session, np.nan)
        stds = np.full(n_session, np.nan)

        for i in range(n_session):
            if i < period - 1:
                continue

            start_idx = max(0, i - period + 1)
            window_values = session_hlc[start_idx:i + 1]
            means[i] = np.mean(window_values)

            # Utiliser notre fonction std_ddof1 au lieu de np.std avec ddof=1
            if len(window_values) > 1:
                stds[i] = std_ddof1(window_values)
            else:
                stds[i] = 0.0

        # Calculer les bandes
        upper_bands = means + (std_dev * stds)
        lower_bands = means - (std_dev * stds)

        # Calculer %B
        for i in range(n_session):
            idx = session_indices[i]

            if i < period - 1:
                percent_b[idx] = fill_value
                continue

            price = session_hlc[i]
            upper = upper_bands[i]
            lower = lower_bands[i]

            band_diff = upper - lower

            # Gestion améliorée des cas spéciaux
            if not np.isnan(band_diff) and band_diff > 1e-8:
                # Calculer %B et limiter à [-1, 2] pour éviter les valeurs extrêmes
                percent_b[idx] = (price - lower) / band_diff

            else:
                # Si les bandes sont trop proches ou invalides, utiliser une valeur neutre
                percent_b[idx] = 0.5

    return percent_b


def calculate_percent_bb(df, period=14, std_dev=2, fill_value=0, return_array=False):
    """
    Version optimisée de calculate_percent_bb utilisant Numba.

    Parameters
    ----------
    df : DataFrame
        DataFrame contenant les colonnes 'sc_high', 'sc_low', 'sc_close' et 'sc_sessionStartEnd'
    period : int
        Période pour le calcul des bandes de Bollinger (défaut 14)
    std_dev : float
        Nombre d'écarts-types pour les bandes supérieure/inférieure (défaut 2)
    fill_value : float
        Valeur à utiliser pour les premières barres de chaque session
    return_array : bool
        Si True, retourne directement le tableau NumPy au lieu d'un DataFrame

    Returns
    -------
    DataFrame ou numpy.ndarray
        DataFrame contenant la colonne 'percent_b' ou directement le tableau NumPy
    """
    # Convertir en tableaux NumPy pour traitement rapide
    sc_high = pd.to_numeric(df['sc_high'], errors='coerce').values
    sc_low = pd.to_numeric(df['sc_low'], errors='coerce').values
    sc_close = pd.to_numeric(df['sc_close'], errors='coerce').values
    session_starts = (df['sc_sessionStartEnd'] == 10).values

    # Appeler la fonction Numba principale
    # Calculez %B et inspectez les valeurs brutes
    raw_percent_b = calculate_percent_bb_numba(sc_high, sc_low, sc_close, session_starts, period, std_dev, fill_value=fill_value)
    # print("Statistiques des valeurs brutes de %B:")
    # print(f"Min: {np.min(raw_percent_b)}, Max: {np.max(raw_percent_b)}")
    # print(f"Moyenne: {np.mean(raw_percent_b)}, Écart-type: {np.std(raw_percent_b)}")
    # print("Distribution des valeurs:")
    # print(np.histogram(raw_percent_b[raw_percent_b != 0], bins=10))  # Ignorer les valeurs 0 (fill_value)
    # Retourner directement le tableau si demandé
    if return_array:
        return raw_percent_b

    # Sinon, créer un DataFrame pour le résultat
    result = pd.DataFrame({'percent_b': raw_percent_b}, index=df.index)
    return result
@njit(fastmath=True, parallel=True)
def close_to_vwap_zscore_fast(sc_close, vwap, session_starts,
                              window, diffDivBy0, DEFAULT_DIV_BY0,
                              valueX, fill_value):

    n = sc_close.size
    ratio      = np.empty(n, dtype=np.float64)
    zscore     = np.empty(n, dtype=np.float64)

    # 1. Pré-calcul du ratio (vectoriel, donc très rapide)
    for i in prange(n):
        ratio[i] = sc_close[i] - vwap[i]

    # 2. Bornes de sessions
    starts = np.where(session_starts)[0]
    # Ajoute un début "virtuel" à −1 pour la toute première barre
    starts = np.concatenate((np.array([-1], dtype=np.int64), starts))
    ends   = np.concatenate((starts[1:], np.array([n-1], dtype=np.int64)))

    # 3. Parcours session par session
    for s in prange(1, starts.size):          # peut être parallélisé si sessions indépendantes
        start, end = starts[s-1]+1, ends[s]   # bornes inclusives
        length = end - start + 1

        # Rolling STD via somme / somme² (Welford simplifié)
        roll_sum = 0.0
        roll_sum2 = 0.0

        for i in range(length):
            idx = start + i
            x   = ratio[idx]

            # Ajout à la fenêtre
            roll_sum  += x
            roll_sum2 += x * x

            if i >= window:
                # Retirer la valeur qui sort de la fenêtre
                x_old = ratio[idx - window]
                roll_sum  -= x_old
                roll_sum2 -= x_old * x_old

            k = min(i + 1, window)  # taille effective de la fenêtre
            if k < window:
                zscore[idx] = fill_value
                continue

            # Variance non biaisée (ddof = 1)
            var = (roll_sum2 - (roll_sum * roll_sum) / k) / (k - 1)
            std = np.sqrt(var) if var > 0 else 0.0

            if std != 0.0:
                zscore[idx] = x / std
            else:
                zscore[idx] = diffDivBy0 if DEFAULT_DIV_BY0 else valueX

        # Padding des toutes premières barres (< window)
        zscore[start:start+window-1] = fill_value
        ratio[start:start+window-1]  = fill_value

    return ratio, zscore


def enhanced_close_to_vwap_zscore(
        df: pd.DataFrame,
        window: int,
        diffDivBy0=0,
        DEFAULT_DIV_BY0=True,
        valueX=0,
        fill_value=0
) -> tuple:
    """
    Version optimisée pour calculer le ratio sc_close-to-VWAP et son z-score.

    Parameters
    ----------
    df : DataFrame
        DataFrame avec au moins les colonnes 'sc_close', 'vwap' et 'sc_sessionStartEnd'
    window : int
        Nombre de périodes pour le calcul de l'écart-type
    diffDivBy0 : float
        Valeur si on divise par 0 et que DEFAULT_DIV_BY0 = True
    DEFAULT_DIV_BY0 : bool
        Booléen, si True, alors on utilise diffDivBy0 comme valeur de fallback
    valueX : float
        Valeur si on divise par 0 et que DEFAULT_DIV_BY0 = False
    fill_value : float
        Valeur utilisée lorsque les calculs ne sont pas possibles (début de session)

    Returns
    -------
    tuple(Series, Series)
        (ratio, z-score)
    """
    # Convertir en tableaux NumPy pour traitement rapide
    sc_close = pd.to_numeric(df['sc_close'], errors='coerce').values
    vwap = pd.to_numeric(df['VWAP'], errors='coerce').values
    session_starts = (df['sc_sessionStartEnd'] == 10).values

    # Appeler la fonction Numba principale
    ratio_array, zscore_array = close_to_vwap_zscore_fast(
        sc_close, vwap, session_starts, window, diffDivBy0, DEFAULT_DIV_BY0, valueX, fill_value
    )

    # Convertir les tableaux en Series pandas
    ratio_series = pd.Series(ratio_array, index=df.index)
    zscore_series = pd.Series(zscore_array, index=df.index)

    return ratio_series, zscore_series



@njit
def calculate_close_to_sma_ratio_numba(
        sc_close: np.ndarray,
        session_starts: np.ndarray,
        window: int,
        diffDivBy0: float,
        DEFAULT_DIV_BY0: bool,
        valueX: float,
        fill_value: float,
        DBG_LEN: int = 40):
    """
    Renvoie (ratio, zscore, dbg_close, dbg_sma, dbg_ratio, dbg_zscore)
    Les 4 derniers vecteurs contiennent les DBG_LEN premières barres
    du **dataset entier** (pas seulement de la 1ʳᵉ session).
    """
    n = len(sc_close)
    ratio   = np.full(n, fill_value, dtype=np.float64)
    zscore  = np.full(n, fill_value, dtype=np.float64)

    # Buffers debug (toujours de longueur DBG_LEN)
    dbg_close  = np.full(DBG_LEN, np.nan)
    dbg_sma    = np.full(DBG_LEN, np.nan)
    dbg_ratio  = np.full(DBG_LEN, np.nan)
    dbg_zscore = np.full(DBG_LEN, np.nan)

    # 1) Construire un ID de session cumulatif
    session_ids = np.zeros(n, dtype=np.int32)
    cid = 0
    for i in range(n):
        if session_starts[i]:
            cid += 1
        session_ids[i] = cid

    # 2) Boucler sur chaque session
    max_sid = session_ids.max()
    for sid in range(1, max_sid + 1):

        # a) Index des barres de cette session
        idx_buf = np.empty(n, dtype=np.int32)
        m = 0
        for i in range(n):
            if session_ids[i] == sid:
                idx_buf[m] = i
                m += 1
        if m == 0:
            continue
        idx_buf = idx_buf[:m]

        # b) Close de session
        close_s = np.empty(m, dtype=np.float64)
        for j in range(m):
            close_s[j] = sc_close[idx_buf[j]]

        # c) SMA glissante
        sma_s = np.empty(m, dtype=np.float64)
        for j in range(m):
            s0 = 0
            s1 = 0.0
            start = 0 if j < window - 1 else j - window + 1
            for k in range(start, j + 1):
                s0 += 1
                s1 += close_s[k]
            sma_s[j] = s1 / s0

        # d) Ratio (sc_close-SMA)
        ratio_s = np.empty(m, dtype=np.float64)
        for j in range(m):
            ratio_s[j] = close_s[j] - sma_s[j]

        # e) σ glissante ddof=1
        std_s = np.empty(m, dtype=np.float64)
        for j in range(m):
            if j < window - 1:
                std_s[j] = 0.0
                continue
            start = j - window + 1
            std_s[j] = std_ddof1(ratio_s[start:j + 1])

        # f) Z-score
        z_s = np.empty(m, dtype=np.float64)
        for j in range(m):
            if std_s[j] != 0.0:
                z_s[j] = ratio_s[j] / std_s[j]
            else:
                z_s[j] = diffDivBy0 if DEFAULT_DIV_BY0 else valueX

        # g) Remplir fill_value sur les (window-1) premières barres
        lim = window - 1 if window - 1 < m else m
        for j in range(lim):
            ratio_s[j] = fill_value
            z_s[j]     = fill_value

        # h) Copier dans la série globale
        for j in range(m):
            g = idx_buf[j]
            ratio[g]  = ratio_s[j]
            zscore[g] = z_s[j]
            if g < DBG_LEN:
                dbg_close[g]  = close_s[j]
                dbg_sma[g]    = sma_s[j]
                dbg_ratio[g]  = ratio_s[j]
                dbg_zscore[g] = z_s[j]

    return ratio, zscore, dbg_close, dbg_sma, dbg_ratio, dbg_zscore
def enhanced_close_to_sma_ratio(
        df: pd.DataFrame,
        window: int,
        diffDivBy0=0,
        DEFAULT_DIV_BY0=True,
        valueX=0,
        fill_value=0
) -> tuple:
    """
    Version optimisée de enhanced_close_to_sma_ratio utilisant Numba.

    Parameters
    ----------
    df : DataFrame
        DataFrame avec au moins les colonnes 'sc_close' et 'sc_sessionStartEnd'
    window : int
        Nombre de périodes pour le calcul rolling (moyenne + écart-type)
    diffDivBy0 : float
        Valeur si on divise par 0 et que DEFAULT_DIV_BY0 = True
    DEFAULT_DIV_BY0 : bool
        Booléen, si True, alors on utilise diffDivBy0 comme valeur de fallback
    valueX : float
        Valeur si on divise par 0 et que DEFAULT_DIV_BY0 = False
    fill_value : float
        Valeur utilisée lorsque les calculs ne sont pas possibles (début de session)

    Returns
    -------
    tuple(Series, Series)
        (ratio, z-score)
    """
    # Convertir en tableaux NumPy pour traitement rapide
    sc_close = pd.to_numeric(df['sc_close'], errors='coerce').values
    session_starts = (df['sc_sessionStartEnd'] == 10).values

    # Appeler la fonction Numba principale
    ratio, z, dclose, dsma, dratio, dz = calculate_close_to_sma_ratio_numba(
        sc_close,
        session_starts,
        window=window,
        diffDivBy0=999.0,
        DEFAULT_DIV_BY0=True,
        valueX=0.0,
        fill_value=0.0,
        DBG_LEN=40)

    # Convertir les tableaux en Series pandas
    ratio_series = pd.Series(ratio, index=df.index)
    zscore_series = pd.Series(z, index=df.index)
    dbg_df = pd.DataFrame({
        "sc_close": dclose,
        "SMA14": dsma,
        "ratio": dratio,
        "zscore": dz
    })
    dbg_df.to_csv("python_debug.csv", index=False)
    print(dbg_df.head(40))
    return ratio_series, zscore_series
def add_zscore(df, features_df,
               period_range=14, period_extrem=14,
               zscore_range_threshold_low=-2.0, zscore_range_threshold_high=0.5,
               zscore_extrem_threshold_low=-2.0, zscore_extrem_threshold_high=0.5):
    """
    Ajoute les indicateurs de Z-Score au DataFrame de features.
    Utilise des seuils spécifiques à chaque période.

    Paramètres:
    - df: DataFrame contenant les données de prix
    - features_df: DataFrame où ajouter les indicateurs
    - period_range: Période pour le calcul du Z-Score de l'indicateur is_zscore_range
    - period_extrem: Période pour le calcul du Z-Score de l'indicateur is_zscore_extrem
                    (Si 0, seul l'indicateur is_zscore_range sera calculé)
    - zscore_range_threshold_low: Seuil bas pour la zone modérée du Z-Score
    - zscore_range_threshold_high: Seuil haut pour la zone modérée du Z-Score
    - zscore_extrem_threshold_low: Seuil bas pour la zone extrême du Z-Score
    - zscore_extrem_threshold_high: Seuil haut pour la zone extrême du Z-Score

    Retourne:
    - features_df enrichi des indicateurs de Z-Score
    """
    try:
        # Vérifier que period_range est valide (> 0)
        if period_range <= 0:
            print(f"Erreur: period_range doit être > 0 (valeur actuelle: {period_range})")
            return features_df

        # Calcul du Z-Score pour l'indicateur is_zscore_range
        _, zscores_range = enhanced_close_to_sma_ratio(df, period_range)

        # Ajouter les valeurs brutes au DataFrame de features
        features_df['zscore_range'] = zscores_range

        # Créer l'indicateur is_zscore_range (Z-Scores modérés optimisés pour maximiser le win rate)
        features_df['is_zscore_range'] = np.where(
            (zscores_range > zscore_range_threshold_low) & (zscores_range < zscore_range_threshold_high),
            1, 0
        )

        # Calculer et ajouter is_zscore_extrem seulement si period_extrem > 0
        if period_extrem > 0:
            # Calcul du Z-Score pour l'indicateur is_zscore_extrem
            if period_range == period_extrem:
                zscores_extrem = zscores_range
            else:
                _, zscores_extrem = enhanced_close_to_sma_ratio(df, period_extrem)

            # Ajouter les valeurs brutes
            features_df['zscore_extrem'] = zscores_extrem

            # Créer l'indicateur is_zscore_extrem
            features_df['is_zscore_extrem'] = np.where(
                (zscores_extrem < zscore_extrem_threshold_low) | (zscores_extrem > zscore_extrem_threshold_high),
                1, 0
            )
        else:
            # Si period_extrem est 0, ne pas calculer is_zscore_extrem
            features_df['zscore_extrem'] = 0
            features_df['is_zscore_extrem'] = 0
            print("Avertissement: period_extrem = 0, is_zscore_extrem est fixé à 0")

        # S'assurer que toutes les colonnes sont numériques
        for col in ['zscore_range', 'zscore_extrem', 'is_zscore_range', 'is_zscore_extrem']:
            if col in features_df.columns:
                features_df[col] = pd.to_numeric(features_df[col], errors='coerce').fillna(0)

    except Exception as e:
        import traceback
        print(f"Erreur dans add_zscore: {str(e)}")
        traceback.print_exc()
        # En cas d'erreur, tenter de renvoyer au moins les colonnes existantes
        if 'zscore_range' not in features_df.columns:
            features_df['zscore_range'] = 0
        if 'zscore_extrem' not in features_df.columns:
            features_df['zscore_extrem'] = 0
        if 'is_zscore_range' not in features_df.columns:
            features_df['is_zscore_range'] = 0
        if 'is_zscore_extrem' not in features_df.columns:
            features_df['is_zscore_extrem'] = 0

    return features_df

def add_perctBB_simu(df, features_df,
                     period_high=105, period_low=5,
                     std_dev_high=1.9481898795476222, std_dev_low=0.23237747131209152,
                     bb_high_threshold=0.6550726973429961, bb_low_threshold=0.2891135240579008):
    """
    Ajoute l'indicateur Percent B (%B) des bandes de Bollinger sur des périodes potentiellement
    différentes pour les zones hautes et basses, ainsi que des indicateurs dérivés.
    Version optimisée utilisant directement les fonctions Numba.

    Paramètres:
    - df: DataFrame contenant les données de prix
    - features_df: DataFrame où ajouter les colonnes liées au %B
    - period_high: Période de calcul pour la zone haute (ex: 105)
    - period_low: Période de calcul pour la zone basse (ex: 5)
    - std_dev_high: Nombre d'écarts-types pour la zone haute (ex: 1.95)
    - std_dev_low: Nombre d'écarts-types pour la zone basse (ex: 0.23)
    - bb_high_threshold: Seuil haut pour la zone haute (ex: 0.65)
    - bb_low_threshold: Seuil bas pour la zone basse (ex: 0.29)

    Retourne:
    - features_df enrichi des colonnes %B et dérivées
    """
    try:
        # Calcul du %B pour la zone haute (obtenir directement le tableau NumPy)
        percent_b_high_values = calculate_percent_bb(
            df=df, period=period_high, std_dev=std_dev_high, fill_value=0, return_array=True
        )

        # Créer un DataFrame temporaire pour l'affichage si nécessaire
        percent_b_high_df = pd.DataFrame({'percent_b': percent_b_high_values}, index=df.index)
        print(percent_b_high_df.head(200))

        # Calcul du %B pour la zone basse (uniquement si différente)
        if period_high == period_low and std_dev_high == std_dev_low:
            percent_b_low_values = percent_b_high_values
        else:
            percent_b_low_values = calculate_percent_bb(
                df=df, period=period_low, std_dev=std_dev_low, fill_value=0, return_array=True
            )

        # Ajouter les indicateurs %B bruts
        features_df['percent_b_high'] = percent_b_high_values
        if period_high != period_low or std_dev_high != std_dev_low:
            features_df['percent_b_low'] = percent_b_low_values
        else:
            features_df['percent_b_high'] = percent_b_high_values

        # Créer l'indicateur is_bb_high (zone haute optimisée pour maximiser le win rate)
        features_df['is_bb_high'] = np.where(
            (percent_b_high_values >= bb_high_threshold),
            1, 0
        )

        # Créer l'indicateur is_bb_low (zone basse optimisée pour minimiser le win rate)
        features_df['is_bb_low'] = np.where(
            (percent_b_low_values <= bb_low_threshold),
            1, 0
        )

        # S'assurer que toutes les colonnes sont numériques
        for col in features_df.columns:
            if col.startswith('percent_b') or col.startswith('is_bb'):
                features_df[col] = pd.to_numeric(features_df[col], errors='coerce').fillna(0)

    except Exception as e:
        print(f"Erreur dans add_perctBB_simu: {str(e)}")
        # En cas d'erreur, tenter de renvoyer au moins les colonnes essentielles
        columns_to_check = [
            'percent_b_high', 'percent_b_low',
            'is_bb_high', 'is_bb_low'
        ]

        for col in columns_to_check:
            if col not in features_df.columns:
                features_df[col] = 0

    return features_df

def add_vwap(df, features_df,
             vwap_range_threshold_low=-2.6705237017186305, vwap_range_threshold_high=1.47028136092062,
             vwap_extrem_threshold_low=-30.3195, vwap_extrem_threshold_high=49.1878):
    """
    Ajoute les indicateurs basés sur la différence entre le prix et le VWAP
    pour identifier les zones favorables et défavorables pour les positions short.
    Utilise la colonne 'diffPriceCloseVWAP' déjà présente dans le DataFrame.

    Paramètres:
    - df: DataFrame contenant la colonne 'diffPriceCloseVWAP'
    - features_df: DataFrame où ajouter les indicateurs dérivés
    - vwap_range_threshold_low: Seuil bas pour la zone favorable (différence avec VWAP)
    - vwap_range_threshold_high: Seuil haut pour la zone favorable (différence avec VWAP)
    - vwap_extrem_threshold_low: Seuil bas pour la zone non favorable (différence avec VWAP)
    - vwap_extrem_threshold_high: Seuil haut pour la zone non favorable (différence avec VWAP)

    Retourne:
    - features_df enrichi des colonnes d'indicateurs VWAP
    """
    try:
        # Récupérer la différence prix-VWAP déjà calculée
        diff_vwap = pd.to_numeric(features_df['diffPriceCloseVWAP'], errors='coerce')
        sc_close = pd.to_numeric(df['sc_close'], errors='coerce')


        # Normaliser la différence par rapport au prix de clôture (pourcentage)
        # Si le prix est 0, utiliser 1 pour éviter la division par zéro
        norm_diff_vwap = np.where(sc_close > 0, diff_vwap / sc_close, diff_vwap)
        features_df['norm_diff_vwap'] = norm_diff_vwap

        # Créer l'indicateur is_vwap_shortArea (zone favorable pour les shorts)
        # Typiquement, quand le prix est modérément au-dessus du VWAP
        features_df['is_vwap_shortArea'] = np.where(
            (diff_vwap > vwap_range_threshold_low) & (diff_vwap < vwap_range_threshold_high),
            1, 0
        )

        # Créer l'indicateur is_vwap_notShortArea (zone non favorable pour les shorts)
        # Typiquement, quand le prix est trop au-dessus ou en-dessous du VWAP
        features_df['is_vwap_notShortArea'] = np.where(
            (diff_vwap < vwap_extrem_threshold_low) | (diff_vwap > vwap_extrem_threshold_high),
            1, 0
        )

        # Croisements du VWAP
        vwap_cross = np.zeros_like(diff_vwap)
        for i in range(1, len(diff_vwap)):
            if diff_vwap[i - 1] < 0 and diff_vwap[i] > 0:
                vwap_cross[i] = 1  # Prix croise au-dessus du VWAP
            elif diff_vwap[i - 1] > 0 and diff_vwap[i] < 0:
                vwap_cross[i] = -1  # Prix croise en-dessous du VWAP



        # S'assurer que toutes les colonnes sont numériques
        vwap_columns = ['norm_diff_vwap', 'is_vwap_shortArea',
                        'is_vwap_notShortArea']
        for col in vwap_columns:
            features_df[col] = pd.to_numeric(features_df[col], errors='coerce').fillna(0)

    except Exception as e:
        print(f"Erreur dans add_vwap: {str(e)}")
        # En cas d'erreur, initialiser les colonnes principales à 0
        required_columns = [
            'norm_diff_vwap', 'is_vwap_shortArea', 'is_vwap_notShortArea',
        ]
        for col in required_columns:
            if col not in features_df.columns:
                features_df[col] = 0

    return features_df




def add_std_regression(df, features_df,
                       period_range=14, period_extrem=14,
                       std_low_threshold_range=0.1, std_high_threshold_range=0.5,
                       std_low_threshold_extrem=0.1, std_high_threshold_extrem=0.5):
    """
    Ajoute les indicateurs de volatilité basés sur l'écart-type de régression au DataFrame de features.
    Utilise des seuils spécifiques à chaque période.

    Paramètres:
    - df: DataFrame contenant les données de prix
    - features_df: DataFrame où ajouter les indicateurs
    - period_range: Période pour le calcul de l'écart-type de l'indicateur range_volatility
    - period_extrem: Période pour le calcul de l'écart-type de l'indicateur extrem_volatility
    - std_low_threshold_range: Seuil bas pour la détection de volatilité modérée
    - std_high_threshold_range: Seuil haut pour la détection de volatilité modérée
    - std_low_threshold_extrem: Seuil bas pour la détection de volatilité extrême
    - std_high_threshold_extrem: Seuil haut pour la détection de volatilité extrême

    Retourne:
    - features_df enrichi des indicateurs de volatilité
    """
    try:
        sc_close = pd.to_numeric(df['sc_close'], errors='coerce').values
        session_starts = (df['sc_sessionStartEnd'] == 10).values

        # Calcul des écarts-types pour l'indicateur range_volatility
        _, _, stds_range = calculate_slopes_and_r2_numba(sc_close, session_starts, period_range)

        # Calcul des écarts-types pour l'indicateur extrem_volatility (uniquement si période différente)
        if period_range == period_extrem:
            stds_extrem = stds_range
        else:
            _, _, stds_extrem = calculate_slopes_and_r2_numba(sc_close, session_starts, period_extrem)

        # Ajouter les valeurs brutes au DataFrame de features
        features_df['std_range'] = stds_range
        if period_range != period_extrem:
            features_df['std_extrem'] = stds_extrem
        else:
            features_df['std_extrem'] = stds_range

        # Créer l'indicateur is_range_volatility (volatilité modérée optimisée pour maximiser le win rate)
        # is_range_volatility = 1 quand l'écart-type est entre std_low_threshold_range et std_high_threshold_range
        features_df['is_range_volatility_std'] = np.where(
            (stds_range > std_low_threshold_range) & (stds_range < std_high_threshold_range),
            1, 0
        )

        # Créer l'indicateur is_extrem_volatility (volatilité extrême optimisée pour minimiser le win rate)
        # is_extrem_volatility = 1 quand l'écart-type est soit inférieur à std_low_threshold_extrem
        # soit supérieur à std_high_threshold_extrem
        features_df['is_extrem_volatility_std'] = np.where(
            (stds_extrem < std_low_threshold_extrem) | (stds_extrem > std_high_threshold_extrem),
            1, 0
        )

        # S'assurer que toutes les colonnes sont numériques
        for col in ['std_range', 'std_extrem', 'is_range_volatility', 'is_extrem_volatility']:
            if col in features_df.columns:
                features_df[col] = pd.to_numeric(features_df[col], errors='coerce').fillna(0)





    except Exception as e:
        print(f"Erreur dans add_std_regression: {str(e)}")
        # En cas d'erreur, tenter de renvoyer au moins les colonnes existantes
        if 'std_range' not in features_df.columns:
            features_df['std_range'] = 0
        if 'std_extrem' not in features_df.columns:
            features_df['std_extrem'] = 0
        if 'is_range_volatility' not in features_df.columns:
            features_df['is_range_volatility'] = 0
        if 'is_extrem_volatility' not in features_df.columns:
            features_df['is_extrem_volatility'] = 0

    return features_df
@jit(nopython=True)
def calculate_rogers_satchell_numba(high_values, low_values, open_values, close_values,
                                    session_starts, window):
    n   = len(high_values)
    vol = np.full(n, np.nan)

    # 1. RS « brut » barre par barre
    rs_daily = np.empty(n)
    for i in range(n):
        ho = high_values[i] / open_values[i]
        hc = high_values[i] / close_values[i]
        lo = low_values[i] / open_values[i]
        lc = low_values[i] / close_values[i]
        rs_daily[i] = np.log(ho) * np.log(hc) + np.log(lo) * np.log(lc)

    # 2. Moyenne glissante par fenêtre
    for start_idx in range(n - window + 1):
        end_idx    = start_idx + window - 1      # dernière barre incluse (t)
        result_idx = end_idx                     # ← on écrit sur cette même barre

        # stop si on dépasse le tableau
        if result_idx >= n:
            break

        # skip si début de session dans la fenêtre
        skip_window = False
        for j in range(start_idx + 1, end_idx + 1):
            if session_starts[j]:
                skip_window = True
                break
        if skip_window:
            continue

        window_sum = 0.0
        for k in range(start_idx, end_idx + 1):
            window_sum += rs_daily[k]
        mean_rs = max(window_sum / window, 0.0)

        vol[result_idx] = np.sqrt(mean_rs)

    return vol
def add_rs(df, features_df,
           # Paramètres SHORT
           period_short_range=14, period_short_extrem=14,
           rs_short_low_threshold_range=0.1, rs_short_high_threshold_range=0.5,
           rs_short_low_threshold_extrem=0.1, rs_short_high_threshold_extrem=0.5,
           # Paramètres LONG
           period_long_range=14, period_long_extrem=14,
           rs_long_low_threshold_range=0.1, rs_long_high_threshold_range=0.5,
           rs_long_low_threshold_extrem=0.1, rs_long_high_threshold_extrem=0.5):
    """
    Ajoute les indicateurs de volatilité Rogers-Satchell avec des paramètres distincts
    pour les stratégies SHORT et LONG.

    Paramètres:
    - df: DataFrame contenant les données OHLC
    - features_df: DataFrame où ajouter les indicateurs

    SHORT (stratégies baissières):
    - period_short_range: Période pour RS range SHORT (défaut: 14)
    - period_short_extrem: Période pour RS extrem SHORT (défaut: 14)
    - rs_short_low_threshold_range: Seuil bas volatilité modérée SHORT
    - rs_short_high_threshold_range: Seuil haut volatilité modérée SHORT
    - rs_short_low_threshold_extrem: Seuil bas volatilité extrême SHORT
    - rs_short_high_threshold_extrem: Seuil haut volatilité extrême SHORT

    LONG (stratégies haussières):
    - period_long_range: Période pour RS range LONG (défaut: 14)
    - period_long_extrem: Période pour RS extrem LONG (défaut: 14)
    - rs_long_low_threshold_range: Seuil bas volatilité modérée LONG
    - rs_long_high_threshold_range: Seuil haut volatilité modérée LONG
    - rs_long_low_threshold_extrem: Seuil bas volatilité extrême LONG
    - rs_long_high_threshold_extrem: Seuil haut volatilité extrême LONG

    Retourne:
    - features_df enrichi des indicateurs Rogers-Satchell pour SHORT et LONG
    """
    try:
        # Extraire les données OHLC
        sc_high = pd.to_numeric(df['sc_high'], errors='coerce').values
        sc_low = pd.to_numeric(df['sc_low'], errors='coerce').values
        open_price = pd.to_numeric(df['sc_open'], errors='coerce').values
        sc_close = pd.to_numeric(df['sc_close'], errors='coerce').values
        session_starts = (df['sc_sessionStartEnd'] == 10).values

        # ════════════════════ CALCULS ROGERS-SATCHELL ════════════════════

        # Collecte de toutes les périodes uniques pour optimiser les calculs
        periods_needed = set([
            period_short_range, period_short_extrem,
            period_long_range, period_long_extrem
        ])

        # Dictionnaire pour stocker les RS calculés
        rs_cache = {}

        # Calcul de tous les Rogers-Satchell nécessaires
        for period in periods_needed:
            rs_cache[period] = calculate_rogers_satchell_numba(
                sc_high, sc_low, open_price, sc_close, session_starts, period
            )

        # ════════════════════ STRATÉGIES SHORT ════════════════════

        # Rogers-Satchell SHORT RANGE (volatilité modérée)
        rs_short_range = rs_cache[period_short_range]
        features_df['rs_range_short'] = rs_short_range
        features_df['is_rs_range_short'] = np.where(
            (rs_short_range > rs_short_low_threshold_range) &
            (rs_short_range < rs_short_high_threshold_range),
            1, 0
        )

        # Rogers-Satchell SHORT EXTREM (volatilité extrême - à éviter)
        rs_short_extrem = rs_cache[period_short_extrem]
        features_df['rs_extrem_short'] = rs_short_extrem
        features_df['is_rs_extrem_short'] = np.where(
            (rs_short_extrem < rs_short_low_threshold_extrem) |
            (rs_short_extrem > rs_short_high_threshold_extrem),
            1, 0
        )

        # ════════════════════ STRATÉGIES LONG ════════════════════

        # Rogers-Satchell LONG RANGE (volatilité modérée)
        rs_long_range = rs_cache[period_long_range]
        features_df['rs_range_long'] = rs_long_range
        features_df['is_rs_range_long'] = np.where(
            (rs_long_range > rs_long_low_threshold_range) &
            (rs_long_range < rs_long_high_threshold_range),
            1, 0
        )

        # Rogers-Satchell LONG EXTREM (volatilité extrême - à éviter)
        rs_long_extrem = rs_cache[period_long_extrem]
        features_df['rs_extrem_long'] = rs_long_extrem
        features_df['is_rs_extrem_long'] = np.where(
            (rs_long_extrem < rs_long_low_threshold_extrem) |
            (rs_long_extrem > rs_long_high_threshold_extrem),
            1, 0
        )

        # ════════════════════ INDICATEURS COMBINÉS ET SIGNAUX ════════════════════

        # # Signaux principaux pour volatilité favorable
        # features_df['signal_short_vol_favorable'] = features_df['is_rs_short_range']
        # features_df['signal_long_vol_favorable'] = features_df['is_rs_long_range']
        #
        # # Signaux d'évitement pour volatilité défavorable
        # features_df['avoid_short_vol_extrem'] = features_df['is_rs_short_extrem']
        # features_df['avoid_long_vol_extrem'] = features_df['is_rs_long_extrem']

        # # Signaux combinés (volatilité favorable ET pas extrême)
        # features_df['signal_short_vol_optimal'] = np.where(
        #     (features_df['is_rs_short_range'] == 1) &
        #     (features_df['is_rs_short_extrem'] == 0),
        #     1, 0
        # )
        # features_df['signal_long_vol_optimal'] = np.where(
        #     (features_df['is_rs_long_range'] == 1) &
        #     (features_df['is_rs_long_extrem'] == 0),
        #     1, 0
        # )

        # ════════════════════ MÉTRIQUES DE VOLATILITÉ ════════════════════

        # # Ratios de volatilité (pour analyse comparative)
        # features_df['rs_short_range_normalized'] = np.where(
        #     rs_short_range > 0,
        #     np.clip(rs_short_range / rs_short_high_threshold_range, 0, 2),
        #     0
        # )
        # features_df['rs_long_range_normalized'] = np.where(
        #     rs_long_range > 0,
        #     np.clip(rs_long_range / rs_long_high_threshold_range, 0, 2),
        #     0
        # )

        # ════════════════════ NORMALISATION ET VALIDATION ════════════════════

        # S'assurer que toutes les nouvelles colonnes sont numériques
        rs_columns = [col for col in features_df.columns if 'rs_' in col or 'signal_' in col or 'avoid_' in col]

        for col in rs_columns:
            features_df[col] = pd.to_numeric(features_df[col], errors='coerce').fillna(0)

        # Affichage des statistiques pour validation
        # print(f"📊 Rogers-Satchell - Statistiques des signaux:")
        # print(
        #     f"  SHORT RANGE (P={period_short_range}, seuils={rs_short_low_threshold_range:.6f}-{rs_short_high_threshold_range:.6f}): "
        #     f"{features_df['signal_short_vol_favorable'].sum()} signaux ({features_df['signal_short_vol_favorable'].mean():.2%})")
        # print(
        #     f"  LONG RANGE (P={period_long_range}, seuils={rs_long_low_threshold_range:.6f}-{rs_long_high_threshold_range:.6f}): "
        #     f"{features_df['signal_long_vol_favorable'].sum()} signaux ({features_df['signal_long_vol_favorable'].mean():.2%})")
        # print(f"  SHORT OPTIMAL (range ET non extrême): "
        #       f"{features_df['signal_short_vol_optimal'].sum()} signaux ({features_df['signal_short_vol_optimal'].mean():.2%})")
        # print(f"  LONG OPTIMAL (range ET non extrême): "
        #       f"{features_df['signal_long_vol_optimal'].sum()} signaux ({features_df['signal_long_vol_optimal'].mean():.2%})")

    except Exception as e:
        print(f"❌ Erreur dans add_rs: {str(e)}")

        # En cas d'erreur, créer les colonnes essentielles avec des valeurs par défaut
        essential_columns = [
            'rs_range_short', 'is_rs_range_short', 'signal_short_vol_favorable',
            'rs_extrem_short', 'is_rs_extrem_short', 'avoid_short_vol_extrem',
            'rs_range_long', 'is_rs_long_range', 'signal_long_vol_favorable',
            'rs_extrem_long', 'is_rs_extrem_long', 'avoid_long_vol_extrem',
            'signal_short_vol_optimal', 'signal_long_vol_optimal',
            'rs_range_short_normalized', 'rs_range_long_normalized'
        ]

        for col in essential_columns:
            if col not in features_df.columns:
                features_df[col] = 0

        raise  # Relancer l'exception pour debugging

    return features_df


def add_r2_regression(df, features_df,
                    period_range=14, period_extrem=14,
                    r2_low_threshold_range=0.3, r2_high_threshold_range=0.7,
                    r2_low_threshold_extrem=0.3, r2_high_threshold_extrem=0.7):
    """
    Ajoute les indicateurs de volatilité basés sur le R² de régression au DataFrame de features.
    Utilise des seuils spécifiques à chaque période.

    Paramètres:
    - df: DataFrame contenant les données de prix
    - features_df: DataFrame où ajouter les indicateurs
    - period_range: Période pour le calcul du R² de l'indicateur range_volatility
    - period_extrem: Période pour le calcul du R² de l'indicateur extrem_volatility
    - r2_low_threshold_range: Seuil bas pour la détection de volatilité modérée
    - r2_high_threshold_range: Seuil haut pour la détection de volatilité modérée
    - r2_low_threshold_extrem: Seuil bas pour la détection de volatilité extrême
    - r2_high_threshold_extrem: Seuil haut pour la détection de volatilité extrême

    Retourne:
    - features_df enrichi des indicateurs de volatilité basés sur R²
    """
    try:
        sc_close = pd.to_numeric(df['sc_close'], errors='coerce').values
        session_starts = (df['sc_sessionStartEnd'] == 10).values

        # Calcul des R² pour l'indicateur range_volatility
        slopes_range, r2s_range, stds_range = calculate_slopes_and_r2_numba(sc_close, session_starts, period_range)

        # Calcul des R² pour l'indicateur extrem_volatility (uniquement si période différente)
        if period_range == period_extrem:
            r2s_extrem = r2s_range
        else:
            _, r2s_extrem, _ = calculate_slopes_and_r2_numba(sc_close, session_starts, period_extrem)

        # Ajouter les valeurs brutes au DataFrame de features
        features_df['r2_range'] = r2s_range
        if period_range != period_extrem:
            features_df['r2_extrem'] = r2s_extrem
        else:
            features_df['r2_extrem'] = r2s_range

        # Créer l'indicateur is_range_volatility (volatilité modérée optimisée pour maximiser le win rate)
        # is_range_volatility = 1 quand le R² est entre r2_low_threshold_range et r2_high_threshold_range
        features_df['is_range_volatility_r2'] = np.where(
            (r2s_range > r2_low_threshold_range) & (r2s_range < r2_high_threshold_range),
            1, 0
        )

        # Créer l'indicateur is_extrem_volatility (volatilité extrême optimisée pour minimiser le win rate)
        # is_extrem_volatility = 1 quand le R² est soit inférieur à r2_low_threshold_extrem
        # soit supérieur à r2_high_threshold_extrem
        features_df['is_extrem_volatility_r2'] = np.where(
            (r2s_extrem < r2_low_threshold_extrem) | (r2s_extrem > r2_high_threshold_extrem),
            1, 0
        )

        # S'assurer que toutes les colonnes sont numériques
        for col in ['r2_range', 'r2_extrem', 'is_range_volatility_r2', 'is_extrem_volatility_r2']:
            if col in features_df.columns:
                features_df[col] = pd.to_numeric(features_df[col], errors='coerce').fillna(0)

    except Exception as e:
        print(f"Erreur dans add_r2_regression: {str(e)}")
        # En cas d'erreur, tenter de renvoyer au moins les colonnes existantes
        if 'r2_range' not in features_df.columns:
            features_df['r2_range'] = 0
        if 'r2_extrem' not in features_df.columns:
            features_df['r2_extrem'] = 0
        if 'is_range_volatility_r2' not in features_df.columns:
            features_df['is_range_volatility_r2'] = 0
        if 'is_extrem_volatility_r2' not in features_df.columns:
            features_df['is_extrem_volatility_r2'] = 0

    return features_df


def add_williams_r_harmonized(df, features_df,
                              # Paramètres SHORT
                              period_short_overbought=14, threshold_short_overbought=-10,
                              period_short_oversold=14, threshold_short_oversold=-84,
                              # Paramètres LONG
                              period_long_overbought=14, threshold_long_overbought=-10,
                              period_long_oversold=14, threshold_long_oversold=-84,
                              # Nouveaux paramètres pour harmonisation
                              use_adaptive_logic=True,
                              debug_mode=False):
    """
    Version harmonisée avec la logique de l'optimizer Williams %R.

    Paramètres ajoutés:
    - use_adaptive_logic: Si True, utilise la logique adaptative de l'optimizer
    - debug_mode: Affichage des statistiques détaillées
    """
    try:
        sc_high = pd.to_numeric(df['sc_high'], errors='coerce')
        sc_low = pd.to_numeric(df['sc_low'], errors='coerce')
        sc_close = pd.to_numeric(df['sc_close'], errors='coerce')
        session_starts = (df['sc_sessionStartEnd'] == 10).values

        # ════════════════════ CALCULS WILLIAMS %R ════════════════════

        # Collecte de toutes les périodes uniques
        periods_needed = set([
            period_short_overbought, period_short_oversold,
            period_long_overbought, period_long_oversold
        ])

        # Cache pour les Williams %R calculés
        wr_cache = {}

        for period in periods_needed:
            wr_cache[period] = compute_wr(sc_high, sc_low, sc_close,
                                          session_starts=session_starts,
                                          period=period, fill_value=-50)

        # ════════════════════ LOGIQUE HARMONISÉE ════════════════════

        if use_adaptive_logic:
            print("🔧 Utilisation de la logique adaptative (comme optimizer)")

            # SHORT OVERBOUGHT - Logique optimizer
            wr_short_overbought = wr_cache[period_short_overbought]
            features_df['wr_overbought_short'] = wr_short_overbought

            # Zone OVERBOUGHT + Direction SHORT : signal quand WR >= threshold
            features_df['is_wr_overbought_short'] = np.where(
                wr_short_overbought >= threshold_short_overbought, 1, 0
            )

            # # SHORT OVERSOLD - Logique optimizer
            # wr_short_oversold = wr_cache[period_short_oversold]
            # features_df['wr_oversold_short'] = wr_short_oversold
            #
            # # Zone OVERSOLD + Direction SHORT : signal quand WR <= threshold
            # features_df['is_wr_oversold_short'] = np.where(
            #     wr_short_oversold <= threshold_short_oversold, 1, 0
            # )

            # # LONG OVERBOUGHT - Logique optimizer
            # wr_long_overbought = wr_cache[period_long_overbought]
            # features_df['wr_overbought_long'] = wr_long_overbought
            #
            # # Zone OVERBOUGHT + Direction LONG : signal quand WR >= threshold
            # features_df['is_wr_overbought_long'] = np.where(
            #     wr_long_overbought >= threshold_long_overbought, 1, 0
            # )

            # LONG OVERSOLD - Logique optimizer
            wr_long_oversold = wr_cache[period_long_oversold]
            features_df['wr_oversold_long'] = wr_long_oversold

            # Zone OVERSOLD + Direction LONG : signal quand WR <= threshold
            features_df['is_wr_oversold_long'] = np.where(
                wr_long_oversold <= threshold_long_oversold, 1, 0
            )

        else:
            print("📊 Utilisation de la logique simple (originale)")
            # Votre logique originale ici...

        # ════════════════════ STATISTIQUES DE VALIDATION ════════════════════

        if debug_mode:
            print(f"\n📊 Statistiques détaillées Williams %R:")

            # Pour chaque signal, calculer les statistiques comme dans l'optimizer
            signal_configs = [
                ('is_wr_overbought_short', 'SHORT OVERBOUGHT'),
                ('is_wr_oversold_short', 'SHORT OVERSOLD'),
                ('is_wr_overbought_long', 'LONG OVERBOUGHT'),
                ('is_wr_oversold_long', 'LONG OVERSOLD')
            ]

            for signal_col, signal_name in signal_configs:
                if signal_col in features_df.columns:
                    # Filtrer : signal == 1 ET class_binaire ∈ {0, 1}
                    mask = (features_df[signal_col] == 1) & (features_df['class_binaire'].isin([0, 1]))
                    sub = features_df.loc[mask, 'class_binaire']

                    if len(sub) > 0:
                        winrate = sub.mean()
                        nb_total = len(sub)
                        nb_win = (sub == 1).sum()
                        nb_lose = (sub == 0).sum()
                        pct_trade = nb_total / len(features_df[features_df['class_binaire'].isin([0, 1])])

                        print(f"  {signal_name}:")
                        print(f"    • Total      : {nb_total}")
                        print(f"    • Gagnants   : {nb_win}")
                        print(f"    • Perdants   : {nb_lose}")
                        print(f"    • Win Rate   : {winrate:.2%}")
                        print(f"    • % Trades   : {pct_trade:.2%}")
                    else:
                        print(f"  {signal_name}: Aucun signal détecté")

        # ════════════════════ SIGNAUX PRINCIPAUX ════════════════════

        # Créer les signaux principaux selon la logique optimizer
        if 'is_wr_overbought_short' in features_df.columns:
            features_df['signal_short_overbought'] = features_df['is_wr_overbought_short']

        if 'is_wr_oversold_long' in features_df.columns:
            features_df['signal_long_oversold'] = features_df['is_wr_oversold_long']

        # Signaux d'évitement (optionnel)
        if 'is_wr_oversold_short' in features_df.columns:
            features_df['avoid_short_oversold'] = features_df['is_wr_oversold_short']

        if 'is_wr_overbought_long' in features_df.columns:
            features_df['avoid_long_overbought'] = features_df['is_wr_overbought_long']

        # ════════════════════ VALIDATION FINALE ════════════════════

        # S'assurer que toutes les colonnes sont numériques
        wr_columns = [col for col in features_df.columns if
                      'wr_' in col or 'signal_' in col or 'avoid_' in col or 'is_wr_' in col]

        for col in wr_columns:
            features_df[col] = pd.to_numeric(features_df[col], errors='coerce').fillna(0)

        print(f"✅ Williams %R harmonisé ajouté - {len(wr_columns)} colonnes créées")

    except Exception as e:
        print(f"❌ Erreur dans add_williams_r_harmonized: {str(e)}")
        raise

    return features_df


def add_volume_per_tick(df, features_df,
                        period=None, threshold_low=None, threshold_high=None,
                        direction=None, profile=None,
                        config_name=None, PARAMS=None):
    """
    Ajoute l'indicateur Volume Per Tick avec des seuils configurables.
    Focus sur les zones binaires : inbound vs outbound des seuils.

    Génère :
    - Valeurs discrètes : ratios bruts et normalisés
    - Indicateurs binaires is_ : 0/1 pour chaque zone

    Zones définies :
    - Zone INBOUND : threshold_low < ratio < threshold_high (dans les seuils)
    - Zone FAIBLE VOLUME : ratio <= threshold_low
    - Zone FORT VOLUME : ratio >= threshold_high
    - Zone OUTBOUND : faible OU fort volume (hors des seuils)
    """

    # ═══ MODE 2: Utilisation des configurations PARAMS ═══
    if config_name is not None and PARAMS is not None:
        if config_name not in PARAMS:
            raise ValueError(f"Configuration '{config_name}' non trouvée dans PARAMS")

        config = PARAMS[config_name]
        period = config["period"]
        threshold_low = config["threshold_low"]
        threshold_high = config["threshold_high"]
        direction = config["direction"]
        profile = config.get("profile", "default")

        print(f"📋 Utilisation de la configuration PARAMS['{config_name}']")

    # ═══ MODE 1: Vérifications des paramètres individuels ═══
    else:
        if period is None or threshold_low is None or threshold_high is None:
            raise ValueError(
                "En mode paramètres individuels, period, threshold_low et threshold_high doivent être spécifiés")

        if direction is None:
            direction = "Long"
        if profile is None:
            profile = "default"

    # Validations communes
    if threshold_low >= threshold_high:
        raise ValueError("threshold_low doit être inférieur à threshold_high")

    try:
        # Création du suffixe pour distinguer les profils
        profile_suffix = f"_{profile}" if profile != "default" else ""

        # Préparation des arrays pour le calcul optimisé
        volume_per_tick_array, session_starts_array = prepare_arrays_for_optimization(df)

        # Calcul des ratios volume per tick
        ratios = compute_ratio_from_arrays(volume_per_tick_array, session_starts_array, period)

        # Aligner les ratios avec l'index du features_df
        if len(ratios) != len(features_df):
            ratios_aligned = ratios[features_df.index] if hasattr(features_df, 'index') else ratios[:len(features_df)]
        else:
            ratios_aligned = ratios

        # Nettoyer les valeurs NaN/Inf
        ratios_aligned = np.where(np.isnan(ratios_aligned) | np.isinf(ratios_aligned), 1.0, ratios_aligned)

        # ═══════════════════════════════════════════════════════════════════
        # 1. VALEURS DISCRÈTES (ratios bruts)
        # ═══════════════════════════════════════════════════════════════════

        # Ratio brut volume per tick
        features_df[f'vptRatio{profile_suffix}_ratio_period_{period}'] = ratios_aligned

        # Normalisation du ratio entre 0 et 1
        ratios_min = np.percentile(ratios_aligned, 5)  # 5ème percentile
        ratios_max = np.percentile(ratios_aligned, 95)  # 95ème percentile
        ratios_normalized = np.clip((ratios_aligned - ratios_min) / (ratios_max - ratios_min), 0, 1)
        features_df[f'vptRatio{profile_suffix}_ratio_normalized'] = ratios_normalized

        # Indicateur de volatilité
        ratios_series = pd.Series(ratios_aligned, index=features_df.index)
        ratios_series_rolling = ratios_series.rolling(window=min(period, 10), min_periods=1)
        features_df[f'vptRatio{profile_suffix}_volatility'] = ratios_series_rolling.std().fillna(0)

        # ═══════════════════════════════════════════════════════════════════
        # 2. INDICATEURS BINAIRES is_ (0/1) POUR CHAQUE ZONE
        # ═══════════════════════════════════════════════════════════════════

        # Zone INBOUND : dans les seuils (entre threshold_low et threshold_high)
        is_inbound = (ratios_aligned > threshold_low) & (ratios_aligned < threshold_high)
        features_df[f'is_vptRatio{profile_suffix}_inbound_{period}'] = is_inbound.astype(int)

        # Zone FAIBLE VOLUME : ratio <= seuil bas
        is_low_volume = (ratios_aligned <= threshold_low)
        features_df[f'is_vptRatio{profile_suffix}_low_volume_{period}'] = is_low_volume.astype(int)

        # Zone FORT VOLUME : ratio >= seuil haut
        is_high_volume = (ratios_aligned >= threshold_high)
        features_df[f'is_vptRatio{profile_suffix}_high_volume_{period}'] = is_high_volume.astype(int)

        # Zone OUTBOUND : en dehors des seuils (faible OU fort volume)
        is_outbound = is_low_volume | is_high_volume
        features_df[f'is_vptRatio{profile_suffix}_outbound_{period}'] = is_outbound.astype(int)

        # ═══════════════════════════════════════════════════════════════════
        # 3. INDICATEURS SPÉCIFIQUES PAR DIRECTION (optionnel)
        # ═══════════════════════════════════════════════════════════════════

        # Signaux directionnels pour stratégies spécifiques
        if direction.lower() == "long":
            # Long : focus sur faible volume (accumulation discrète)
            features_df[f'is_vptRatio{profile_suffix}_signal_long_{period}'] = is_low_volume.astype(int)
        else:  # Short
            # Short : focus sur fort volume (distribution)
            features_df[f'is_vptRatio{profile_suffix}_signal_short_{period}'] = is_high_volume.astype(int)

        # ═══════════════════════════════════════════════════════════════════
        # 4. VALIDATION ET NETTOYAGE FINAL
        # ═══════════════════════════════════════════════════════════════════

        # S'assurer que toutes les colonnes sont numériques
        vptRatio_columns = [col for col in features_df.columns
                            if col.startswith(f'vptRatio{profile_suffix}_') or col.startswith(
                f'is_vptRatio{profile_suffix}_')]
        for col in vptRatio_columns:
            features_df[col] = pd.to_numeric(features_df[col], errors='coerce').fillna(0)

        print(
            f"✅ Volume Per Tick features ajoutées [{profile}] : period={period}, tl={threshold_low:.3f}, th={threshold_high:.3f}, direction={direction}")

        # Debug : afficher les statistiques des zones
        if len(features_df) > 0:
            inbound_pct = (features_df[f'is_vptRatio{profile_suffix}_inbound_{period}'].sum() / len(features_df)) * 100
            low_pct = (features_df[f'is_vptRatio{profile_suffix}_low_volume_{period}'].sum() / len(features_df)) * 100
            high_pct = (features_df[f'is_vptRatio{profile_suffix}_high_volume_{period}'].sum() / len(features_df)) * 100
            print(f"   📊 Répartition zones : Inbound={inbound_pct:.1f}%, Low={low_pct:.1f}%, High={high_pct:.1f}%")

    except Exception as e:
        print(f"❌ Erreur dans add_volume_per_tick [{profile if profile else 'unknown'}]: {str(e)}")
        raise

    return features_df


# Fonction utilitaire pour générer les paramètres optimaux depuis le code d'optimisation
def get_vptRatio_params_from_optimization(best_params):
    """
    Convertit les paramètres optimisés en format utilisable pour add_volume_per_tick.

    Args:
        best_params: Dict avec les clés 'period', 'tl', 'th' issues de l'optimisation

    Returns:
        Dict formaté pour la fonction add_volume_per_tick
    """
    return {
        "period": best_params["period"],
        "threshold_low": best_params["tl"],
        "threshold_high": best_params["th"]
    }


# Exemple d'utilisation avec des paramètres issus de l'optimisation
def example_usage():
    """
    Exemple d'utilisation de la fonction add_volume_per_tick
    """
    # Paramètres typiques issus de l'optimisation Optuna
    PARAMS = {
        "volume_per_tick": {
            "period": 25,  # période optimisée
            "threshold_low": 0.75,  # seuil bas optimisé (tl)
            "threshold_high": 2.25,  # seuil haut optimisé (th)
            "direction": "Long"
        }
    }

    # Usage dans le pipeline de features
    """
    features_df = add_volume_per_tick(
        df, features_df,
        period=PARAMS["volume_per_tick"]["period"],
        threshold_low=PARAMS["volume_per_tick"]["threshold_low"],
        threshold_high=PARAMS["volume_per_tick"]["threshold_high"],
        direction=PARAMS["volume_per_tick"]["direction"]
    )
    """

    return PARAMS

def add_rsi(df, features_df, period=14):
    """
    Ajoute l'indicateur RSI (Relative Strength Index) sur la période spécifiée.

    Paramètres:
    - df: DataFrame contenant la colonne 'sc_close'
    - features_df: DataFrame où ajouter la colonne 'rsi_{period}'
    - period: Période de calcul (ex: 14)

    Retourne:
    - features_df enrichi de la colonne RSI
    """
    try:
        sc_close = pd.to_numeric(df['sc_close'], errors='coerce')

        # Différence du cours de clôture
        delta = sc_close.diff().fillna(0)

        # Gains (>=0) et pertes (<=0)
        gains = delta.clip(lower=0)
        losses = -delta.clip(upper=0)

        # Moyenne (simple ou EMA) des gains/pertes
        # Ici on utilise l'EMA pour un RSI plus classique
        avg_gains = gains.ewm(alpha=1/period, adjust=False).mean()
        avg_losses = losses.ewm(alpha=1/period, adjust=False).mean()

        # Éviter division par zéro
        rs = np.where(avg_losses == 0, 0, avg_gains / avg_losses)

        # RSI
        rsi = 100 - (100 / (1 + rs))
        features_df[f'rsi_'] = rsi

    except Exception as e:
        print(f"Erreur dans add_rsi: {str(e)}")

    return features_df


def add_mfi(df, features_df,
            overbought_period, oversold_period,
            overbought_threshold=80, oversold_threshold=20):
    """
    Ajoute l'indicateur MFI (Money Flow Index) avec des périodes obligatoires
    et distinctes pour les zones de surachat et survente.

    Paramètres:
    - df: DataFrame contenant 'sc_high', 'sc_low', 'sc_close', 'volume'
    - features_df: DataFrame où ajouter les colonnes MFI
    - overbought_period: Période spécifique pour la détection de surachat (obligatoire)
    - oversold_period: Période spécifique pour la détection de survente (obligatoire)
    - overbought_threshold: Seuil de surachat (défaut: 80)
    - oversold_threshold: Seuil de survente (défaut: 20)

    Retourne:
    - features_df enrichi des colonnes MFI et dérivées

    Lève:
    - ValueError si overbought_period ou oversold_period est None
    """



    if overbought_period is None or oversold_period is None:
        raise ValueError("Les périodes de surachat et de survente doivent être spécifiées")

    try:
        session_starts = (df['sc_sessionStartEnd'] == 10).values
        sc_high = pd.to_numeric(df['sc_high'], errors='coerce')
        sc_low = pd.to_numeric(df['sc_low'], errors='coerce')
        sc_close = pd.to_numeric(df['sc_close'], errors='coerce')
        volume = pd.to_numeric(df['sc_volume'], errors='coerce')

        # Calcul des MFI avec périodes spécifiques pour surachat/survente
        is_mfi_overbought = compute_mfi(sc_high, sc_low, sc_close,volume,session_starts, period=overbought_period, fill_value=50)
        is_mfi_oversold = compute_mfi(sc_high, sc_low, sc_close,volume,session_starts,period=oversold_period, fill_value=50)

        # Indicateurs principaux avec périodes distinctes
        features_df['mfi_overbought_period'] = is_mfi_overbought
        features_df['mfi_oversold_period'] = is_mfi_oversold

        # Indicateurs de surachat/survente avec périodes spécifiques
        features_df['is_mfi_overbought_short'] = np.where(is_mfi_overbought > overbought_threshold, 1, 0)
        features_df['is_mfi_oversold_long'] = np.where(is_mfi_oversold < oversold_threshold, 1, 0)

        # Indicateur de changement de zone (basé sur les MFI spécifiques)
        mfi_overbought_series = pd.Series(is_mfi_overbought)
        mfi_oversold_series = pd.Series(is_mfi_oversold)

        # Sortie de la zone de surachat (signal baissier)
        exit_overbought = (mfi_overbought_series.shift(1) > overbought_threshold) & (
                    mfi_overbought_series <= overbought_threshold)

        # Sortie de la zone de survente (signal haussier)
        exit_oversold = (mfi_oversold_series.shift(1) < oversold_threshold) & (
                    mfi_oversold_series >= oversold_threshold)



        # Normalisation entre 0 et 1


        # S'assurer que toutes les colonnes sont numériques
        columns = ['mfi_overbought_period', 'mfi_oversold_period',
                   'is_mfi_overbought_short', 'is_mfi_oversold_long']


        for col in columns:
            features_df[col] = pd.to_numeric(features_df[col], errors='coerce').fillna(0)

    except Exception as e:
        print(f"Erreur dans add_mfi: {str(e)}")
        raise  # Relancer l'exception pour la traiter en amont

    return features_df


def add_mfi_divergence(df, features_df,
                       mfi_period_bearish=14, mfi_period_antiBear=14,
                       div_lookback_bearish=10, div_lookback_antiBear=10,
                       min_price_increase=0.005, min_mfi_decrease=0.005,
                       min_price_decrease=0.005, min_mfi_increase=0.005):
    """
    Ajoute les indicateurs de divergence MFI/prix pour les stratégies short.
    Utilise les mêmes conditions que la fonction objective pour détecter
    les signaux de divergence baissière et anti-divergence, avec la possibilité
    d'utiliser des périodes différentes pour chaque type de divergence.

    Paramètres:
    - df: DataFrame contenant 'sc_high', 'sc_low', 'sc_close', 'volume'
    - features_df: DataFrame où ajouter les colonnes de divergence
    - mfi_period_bearish: Période MFI pour la divergence baissière (ex: 14)
    - mfi_period_antiBear: Période MFI pour l'anti-divergence (ex: 14)
    - div_lookback_bearish: Période lookback pour la divergence baissière (ex: 10)
    - div_lookback_antiBear: Période lookback pour l'anti-divergence (ex: 10)
    - min_price_increase: Seuil minimal d'augmentation de prix en % pour divergence baissière
    - min_mfi_decrease: Seuil minimal de diminution de MFI en % pour divergence baissière
    - min_price_decrease: Seuil minimal de diminution de prix en % pour anti-divergence
    - min_mfi_increase: Seuil minimal d'augmentation de MFI en % pour anti-divergence

    Retourne:
    - features_df enrichi des colonnes de divergence MFI
    """
    try:
        sc_high = pd.to_numeric(df['sc_high'], errors='coerce')
        sc_low = pd.to_numeric(df['sc_low'], errors='coerce')
        sc_close = pd.to_numeric(df['sc_close'], errors='coerce')
        volume = pd.to_numeric(df['sc_volume'], errors='coerce')
        session_starts = (df['sc_sessionStartEnd'] == 10).values

        # Calcul du MFI pour divergence baissière
        mfi_values_bearish = compute_mfi(sc_high, sc_low, sc_close, volume, session_starts, period=mfi_period_bearish,
                                         fill_value=50)
        mfi_series_bearish = pd.Series(mfi_values_bearish)

        # Calcul du MFI pour anti-divergence (uniquement si période différente)
        if mfi_period_bearish == mfi_period_antiBear:
            mfi_values_antiBear = mfi_values_bearish
            mfi_series_antiBear = mfi_series_bearish
        else:
            mfi_values_antiBear = compute_mfi(sc_high, sc_low, sc_close, volume, session_starts, period=mfi_period_antiBear,
                                              fill_value=50)
            mfi_series_antiBear = pd.Series(mfi_values_antiBear)

        # Ajouter les valeurs brutes du MFI au DataFrame
        if mfi_period_bearish == mfi_period_antiBear:
            features_df['mfi'] = mfi_values_bearish
        else:
            features_df['mfi_bearish'] = mfi_values_bearish
            features_df['mfi_antiBear'] = mfi_values_antiBear

        # --------- Divergence baissière (signal d'entrée short) ---------
        # Détection des divergences baissières
        price_pct_change_bearish = sc_close.pct_change(div_lookback_bearish).fillna(0)
        mfi_pct_change_bearish = mfi_series_bearish.pct_change(div_lookback_bearish).fillna(0)

        # Conditions pour une divergence baissière efficace
        price_increase = price_pct_change_bearish > min_price_increase
        mfi_decrease = mfi_pct_change_bearish < -min_mfi_decrease

        # Prix fait un nouveau haut relatif
        price_rolling_max = pd.Series(sc_close).rolling(window=div_lookback_bearish).max().shift(1)
        price_new_high = (sc_close > price_rolling_max).fillna(False)

        # Définir la divergence baissière avec les mêmes critères que dans l'objective
        features_df['is_mfi_shortDiv'] = np.where(
            (price_new_high | price_increase) &  # Prix fait un nouveau haut ou augmente significativement
            (mfi_decrease),  # MFI diminue
            1, 0
        )

        # --------- Anti-divergence (signal d'évitement de short) ---------
        # Calculs spécifiques pour l'anti-divergence avec ses propres périodes
        price_pct_change_antiBear = sc_close.pct_change(div_lookback_antiBear).fillna(0)
        mfi_pct_change_antiBear = mfi_series_antiBear.pct_change(div_lookback_antiBear).fillna(0)

        # Conditions pour une anti-divergence (mauvais win rate)
        price_decrease = price_pct_change_antiBear < -min_price_decrease  # Prix diminue
        mfi_increase = mfi_pct_change_antiBear > min_mfi_increase  # MFI augmente

        # Prix fait un nouveau bas relatif
        price_rolling_min = pd.Series(sc_close).rolling(window=div_lookback_antiBear).min().shift(1)
        price_new_low = (sc_close < price_rolling_min).fillna(False)

        # Définir l'anti-divergence avec les critères exacts de l'objective
        features_df['is_mfi_antiShortDiv'] = np.where(
            (price_new_low | price_decrease) &  # Prix fait un nouveau bas ou diminue significativement
            (mfi_increase),  # MFI augmente
            1, 0
        )

        # --------- Versions traditionnelles des divergences (pour référence) ---------
        # Utiliser les périodes bearish pour les divergences traditionnelles
        price_highs = pd.Series(sc_close).rolling(window=div_lookback_bearish).max()
        price_lows = pd.Series(sc_close).rolling(window=div_lookback_bearish).min()
        mfi_highs = mfi_series_bearish.rolling(window=div_lookback_bearish).max()
        mfi_lows = mfi_series_bearish.rolling(window=div_lookback_bearish).min()

        # Nouveaux sommets/creux (comparaison avec la période précédente)
        price_new_high_simple = sc_close > price_highs.shift(1)
        price_new_low_simple = sc_close < price_lows.shift(1)
        mfi_new_high = mfi_series_bearish > mfi_highs.shift(1)
        mfi_new_low = mfi_series_bearish < mfi_lows.shift(1)

           # S'assurer que toutes les colonnes MFI sont numériques
        mfi_columns = [col for col in features_df.columns if 'mfi' in col]
        for col in mfi_columns:
            features_df[col] = pd.to_numeric(features_df[col], errors='coerce').fillna(0)

    except Exception as e:
        print(f"Erreur dans add_mfi_divergence: {str(e)}")
        # En cas d'erreur, initialiser les colonnes principales à 0
        required_columns = [
            'mfi_bearish', 'mfi_antiBear', 'is_mfi_shortDiv', 'is_mfi_antiShortDiv',

        ]
        for col in required_columns:
            if col not in features_df.columns:
                features_df[col] = 0

    return features_df

def add_macd(df, features_df, short_period=12, long_period=26, signal_period=9):
    """
    Ajoute les indicateurs MACD (Moving Average Convergence Divergence)
    et sa ligne de signal.

    Paramètres:
    - df: DataFrame contenant la colonne 'sc_close'
    - features_df: DataFrame où ajouter les colonnes:
        * macd
        * macd_signal
        * macd_hist
    - short_period: Période de l'EMA courte (par défaut 12)
    - long_period: Période de l'EMA longue (par défaut 26)
    - signal_period: Période de la ligne de signal (par défaut 9)

    Retourne:
    - features_df enrichi de 'macd', 'macd_signal', et 'macd_hist'
    """
    try:
        sc_close = pd.to_numeric(df['sc_close'], errors='coerce')

        ema_short = sc_close.ewm(span=short_period, adjust=False).mean()
        ema_long = sc_close.ewm(span=long_period, adjust=False).mean()
        macd = ema_short - ema_long
        macd_signal = macd.ewm(span=signal_period, adjust=False).mean()
        macd_hist = macd - macd_signal

        features_df['macd'] = macd
        features_df['macd_signal'] = macd_signal
        features_df['macd_hist'] = macd_hist

    except Exception as e:
        print(f"Erreur dans add_macd: {str(e)}")

    return features_df






def add_adx(df, features_df, period=14):
    """
    Ajoute l'Average Directional Index (ADX).

    Paramètres:
    - df: DataFrame contenant 'sc_high', 'sc_low', 'sc_close'
    - features_df: DataFrame où ajouter la colonne 'adx_{period}'
    - period: Période pour le calcul (ex: 14)

    Retourne:
    - features_df enrichi de la colonne ADX
    """
    try:
        sc_high = pd.to_numeric(df['sc_high'], errors='coerce')
        sc_low = pd.to_numeric(df['sc_low'], errors='coerce')
        sc_close = pd.to_numeric(df['sc_close'], errors='coerce')

        # Calcul du True Range
        shift_close = sc_close.shift(1).fillna(sc_close[0])
        tr = pd.DataFrame({
            'tr1': sc_high - sc_low,
            'tr2': (sc_high - shift_close).abs(),
            'tr3': (sc_low - shift_close).abs()
        }).max(axis=1)

        # +DM et -DM
        shift_high = sc_high.shift(1).fillna(sc_high[0])
        shift_low = sc_low.shift(1).fillna(sc_low[0])

        plus_dm = (sc_high - shift_high).clip(lower=0)
        minus_dm = (shift_low - sc_low).clip(lower=0)

        plus_dm[plus_dm < minus_dm] = 0
        minus_dm[minus_dm <= plus_dm] = 0

        # Moyenne exponentielle ou simple
        tr_ewm = tr.ewm(span=period, adjust=False).mean()
        plus_dm_ewm = plus_dm.ewm(span=period, adjust=False).mean()
        minus_dm_ewm = minus_dm.ewm(span=period, adjust=False).mean()

        # +DI et -DI
        plus_di = 100 * (plus_dm_ewm / tr_ewm)
        minus_di = 100 * (minus_dm_ewm / tr_ewm)

        # DX
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)

        # ADX
        adx = dx.ewm(span=period, adjust=False).mean()

        features_df[f'adx_'] = adx
        features_df[f'plus_di_'] = plus_di
        features_df[f'minus_di_'] = minus_di

    except Exception as e:
        print(f"Erreur dans add_adx: {str(e)}")

    return features_df

def evaluate_williams_r(params, df, optimize_oversold=False, optimize_overbought=False):
    """
    Évalue l'indicateur Williams %R avec les paramètres optimaux.

    Parameters:
    -----------
    params : dict
        Dictionnaire contenant les paramètres optimisés
    df : pandas.DataFrame
        DataFrame complet
    optimize_oversold : bool, default=False
        Indique si l'optimisation des zones de survente est activée
    optimize_overbought : bool, default=False
        Indique si l'optimisation des zones de surachat est activée

    Returns:
    --------
    tuple
        (dict, pandas.DataFrame, pandas.Series)
        - Dictionnaire contenant les métriques d'évaluation
        - DataFrame filtré pour les tests
        - Série des valeurs cibles
    """
    try:
        # Extraire les paramètres
        period = params.get('period')
        OS_limit = params.get('OS_limit', -80)
        OB_limit = params.get('OB_limit', -20)

        # Calculer le Williams %R
        sc_high = pd.to_numeric(df['sc_high'], errors='coerce')
        sc_low = pd.to_numeric(df['sc_low'], errors='coerce')
        sc_close = pd.to_numeric(df['sc_close'], errors='coerce')
        session_starts = (df['sc_sessionStartEnd'] == 10).values

        will_r_values = compute_wr(sc_high, sc_low, sc_close, session_starts, period=period)

        # Créer les indicateurs binaires conditionnels
        if optimize_overbought:
            df['wr_overbought'] = np.where(will_r_values > OB_limit, 1, 0)

        if optimize_oversold:
            df['wr_oversold'] = np.where(will_r_values < OS_limit, 1, 0)

        # Initialiser les résultats
        results = {
            'bin_0_win_rate': 0,
            'bin_1_win_rate': 0,
            'bin_0_pct': 0,
            'bin_1_pct': 0,
            'bin_spread': 0,
            'oversold_success_count': 0,
            'overbought_success_count': 0,
            'bin_0_samples': 0,
            'bin_1_samples': 0
        }

        # Filtrage
        df_test_filtered = df[df['class_binaire'].isin([0, 1])].copy()
        target_y_test = df['class_binaire']

        # Calculs pour le bin 0 (oversold) uniquement si optimize_oversold est activé
        if optimize_oversold:
            oversold_df = df_test_filtered[df_test_filtered['wr_oversold'] == 1]
            if len(oversold_df) > 0:
                results['bin_0_win_rate'] = oversold_df['class_binaire'].mean()
                results['bin_0_pct'] = len(oversold_df) / len(df_test_filtered)
                results['oversold_success_count'] = oversold_df['class_binaire'].sum()
                results['bin_0_samples'] = len(oversold_df)

        # Calculs pour le bin 1 (overbought) uniquement si optimize_overbought est activé
        if optimize_overbought:
            overbought_df = df_test_filtered[df_test_filtered['wr_overbought'] == 1]
            if len(overbought_df) > 0:
                results['bin_1_win_rate'] = overbought_df['class_binaire'].mean()
                results['bin_1_pct'] = len(overbought_df) / len(df_test_filtered)
                results['overbought_success_count'] = overbought_df['class_binaire'].sum()
                results['bin_1_samples'] = len(overbought_df)

        # Calculer le spread uniquement si les deux modes sont activés
        if optimize_oversold and optimize_overbought:
            if 'oversold_df' not in locals():
                oversold_df = df_test_filtered[df_test_filtered['wr_oversold'] == 1]
            if 'overbought_df' not in locals():
                overbought_df = df_test_filtered[df_test_filtered['wr_overbought'] == 1]

            if len(oversold_df) > 0 and len(overbought_df) > 0:
                results['bin_spread'] = results['bin_1_win_rate'] - results['bin_0_win_rate']

        return results, df_test_filtered, target_y_test

    except Exception as e:
        print(f"Erreur lors de l'évaluation de Williams %R: {e}")
        import traceback
        traceback.print_exc()

        # En cas d'erreur, retourner des valeurs par défaut cohérentes
        df_test_filtered = df[
            df['class_binaire'].isin([0, 1])].copy() if 'df_test_filtered' not in locals() else df_test_filtered
        target_y_test = df['class_binaire'] if 'target_y_test' not in locals() else target_y_test

        return {
            'bin_0_win_rate': 0,
            'bin_1_win_rate': 0,
            'bin_0_pct': 0,
            'bin_1_pct': 0,
            'bin_spread': 0,
            'oversold_success_count': 0,
            'overbought_success_count': 0,
            'bin_0_samples': 0,
            'bin_1_samples': 0
        }, df_test_filtered, target_y_test


def evaluate_mfi(params, df, optimize_oversold=False, optimize_overbought=False):
    """
    Évalue l'indicateur Money Flow Index (MFI) avec les paramètres optimaux.

    Parameters:
    -----------
    params : dict
        Dictionnaire contenant les paramètres optimisés
    df : pandas.DataFrame
        DataFrame complet
    optimize_oversold : bool, default=False
        Indique si l'optimisation des zones de survente est activée
    optimize_overbought : bool, default=False
        Indique si l'optimisation des zones de surachat est activée

    Returns:
    --------
    tuple
        (dict, pandas.DataFrame, pandas.Series)
        - Dictionnaire contenant les métriques d'évaluation
        - DataFrame filtré pour les tests
        - Série des valeurs cibles
    """
    try:
        # Extraire les paramètres
        period = params.get('period')
        OS_limit = params.get('OS_limit', 20)
        OB_limit = params.get('OB_limit', 80)

        # Calculer le MFI
        sc_high = pd.to_numeric(df['sc_high'], errors='coerce')
        sc_low = pd.to_numeric(df['sc_low'], errors='coerce')
        sc_close = pd.to_numeric(df['sc_close'], errors='coerce')
        volume = pd.to_numeric(df['sc_volume'], errors='coerce')

        session_starts = (df['sc_sessionStartEnd'] == 10).values
        mfi_values = compute_mfi(sc_high, sc_low, sc_close, volume, session_starts, period=period)

        # Créer les indicateurs binaires conditionnels
        if optimize_overbought:
            df['mfi_overbought'] = np.where(mfi_values > OB_limit, 1, 0)

        if optimize_oversold:
            df['mfi_oversold'] = np.where(mfi_values < OS_limit, 1, 0)

        # Initialiser les résultats
        results = {
            'bin_0_win_rate': 0,
            'bin_1_win_rate': 0,
            'bin_0_pct': 0,
            'bin_1_pct': 0,
            'bin_spread': 0,
            'oversold_success_count': 0,
            'overbought_success_count': 0,
            'bin_0_samples': 0,
            'bin_1_samples': 0
        }

        # Filtrage
        df_test_filtered = df[df['class_binaire'].isin([0, 1])].copy()
        target_y_test = df['class_binaire']

        # Calculs pour le bin 0 (oversold) uniquement si optimize_oversold est activé
        if optimize_oversold:
            oversold_df = df_test_filtered[df_test_filtered['mfi_oversold'] == 1]
            if len(oversold_df) > 0:
                results['bin_0_win_rate'] = oversold_df['class_binaire'].mean()
                results['bin_0_pct'] = len(oversold_df) / len(df_test_filtered)
                results['oversold_success_count'] = oversold_df['class_binaire'].sum()
                results['bin_0_samples'] = len(oversold_df)

        # Calculs pour le bin 1 (overbought) uniquement si optimize_overbought est activé
        if optimize_overbought:
            overbought_df = df_test_filtered[df_test_filtered['mfi_overbought'] == 1]
            if len(overbought_df) > 0:
                results['bin_1_win_rate'] = overbought_df['class_binaire'].mean()
                results['bin_1_pct'] = len(overbought_df) / len(df_test_filtered)
                results['overbought_success_count'] = overbought_df['class_binaire'].sum()
                results['bin_1_samples'] = len(overbought_df)

        # Calculer le spread uniquement si les deux modes sont activés
        if optimize_oversold and optimize_overbought:
            if 'oversold_df' not in locals():
                oversold_df = df_test_filtered[df_test_filtered['mfi_oversold'] == 1]
            if 'overbought_df' not in locals():
                overbought_df = df_test_filtered[df_test_filtered['mfi_overbought'] == 1]

            if len(oversold_df) > 0 and len(overbought_df) > 0:
                results['bin_spread'] = results['bin_1_win_rate'] - results['bin_0_win_rate']

        return results, df_test_filtered, target_y_test

    except Exception as e:
        print(f"Erreur lors de l'évaluation de MFI: {e}")
        import traceback
        traceback.print_exc()

        # En cas d'erreur, retourner des valeurs par défaut cohérentes
        df_test_filtered = df[
            df['class_binaire'].isin([0, 1])].copy() if 'df_test_filtered' not in locals() else df_test_filtered
        target_y_test = df['class_binaire'] if 'target_y_test' not in locals() else target_y_test

        return {
            'bin_0_win_rate': 0,
            'bin_1_win_rate': 0,
            'bin_0_pct': 0,
            'bin_1_pct': 0,
            'bin_spread': 0,
            'oversold_success_count': 0,
            'overbought_success_count': 0,
            'bin_0_samples': 0,
            'bin_1_samples': 0
        }, df_test_filtered, target_y_test


def evaluate_mfi_divergence(params, df, optimize_oversold=False, optimize_overbought=False):
    """
    Évalue les divergences MFI avec les paramètres optimaux.

    Parameters:
    -----------
    params : dict
        Dictionnaire contenant les paramètres optimisés
    df : pandas.DataFrame
        DataFrame complet
    optimize_oversold : bool, default=False
        Indique si l'optimisation des anti-divergences est activée
    optimize_overbought : bool, default=False
        Indique si l'optimisation des divergences baissières est activée

    Returns:
    --------
    tuple
        (dict, pandas.DataFrame, pandas.Series)
        - Dictionnaire contenant les métriques d'évaluation
        - DataFrame filtré pour les tests
        - Série des valeurs cibles
    """
    try:
        # Extraire les paramètres
        mfi_period = params.get('mfi_period')
        div_lookback = params.get('div_lookback')
        min_price_increase = params.get('min_price_increase')
        min_mfi_decrease = params.get('min_mfi_decrease')

        # Paramètres pour la partie oversold (si présents)
        min_price_decrease = params.get('min_price_decrease')
        min_mfi_increase = params.get('min_mfi_increase')

        # Calculer le MFI
        sc_high = pd.to_numeric(df['sc_high'], errors='coerce')
        sc_low = pd.to_numeric(df['sc_low'], errors='coerce')
        sc_close = pd.to_numeric(df['sc_close'], errors='coerce')
        volume = pd.to_numeric(df['sc_volume'], errors='coerce')

        session_starts = (df['sc_sessionStartEnd'] == 10).values
        mfi_values = compute_mfi(sc_high, sc_low, sc_close, volume, session_starts, period=mfi_period)
        mfi_series = pd.Series(mfi_values, index=df.index)

        # Initialiser les colonnes de divergence conditionnellement
        if optimize_overbought:
            df['bearish_divergence'] = 0

        if optimize_oversold:
            df['anti_divergence'] = 0

        # Filtrer pour les trades shorts
        df_mode_filtered = df[df['class_binaire'] != 99].copy()
        all_shorts = df_mode_filtered['tradeDir'].eq(-1).all() if not df_mode_filtered.empty else False

        if all_shorts:
            # Pour la partie overbought (divergence baissière) uniquement si optimize_overbought est activé
            if optimize_overbought:
                price_pct_change = sc_close.pct_change(div_lookback).fillna(0)
                mfi_pct_change = mfi_series.pct_change(div_lookback).fillna(0)

                # Conditions pour une divergence baissière
                price_increase = price_pct_change > min_price_increase
                mfi_decrease = mfi_pct_change < -min_mfi_decrease

                # Prix fait un nouveau haut relatif
                price_rolling_max = sc_close.rolling(window=div_lookback).max().shift(1)
                price_new_high = (sc_close > price_rolling_max).fillna(False)

                # Définir la divergence baissière
                df.loc[df_mode_filtered.index, 'bearish_divergence'] = (
                        (price_new_high | price_increase) &  # Prix fait un nouveau haut ou augmente significativement
                        (mfi_decrease)  # MFI diminue
                ).astype(int)

            # Pour la partie oversold (anti-divergence) si les paramètres sont présents et optimize_oversold est activé
            if optimize_oversold and min_price_decrease is not None and min_mfi_increase is not None:
                price_pct_change = sc_close.pct_change(div_lookback).fillna(
                    0) if 'price_pct_change' not in locals() else price_pct_change
                mfi_pct_change = mfi_series.pct_change(div_lookback).fillna(
                    0) if 'mfi_pct_change' not in locals() else mfi_pct_change

                # Conditions pour une anti-divergence
                price_decrease = price_pct_change < -min_price_decrease  # Prix diminue
                mfi_increase = mfi_pct_change > min_mfi_increase  # MFI augmente

                # Prix fait un nouveau bas relatif
                price_rolling_min = sc_close.rolling(window=div_lookback).min().shift(1)
                price_new_low = (sc_close < price_rolling_min).fillna(False)

                # Définir l'anti-divergence
                df.loc[df_mode_filtered.index, 'anti_divergence'] = (
                        (price_new_low | price_decrease) &  # Prix fait un nouveau bas ou diminue significativement
                        (mfi_increase)  # MFI augmente
                ).astype(int)

        # Initialiser les résultats
        results = {
            'bin_0_win_rate': 0,
            'bin_1_win_rate': 0,
            'bin_0_pct': 0,
            'bin_1_pct': 0,
            'bin_spread': 0,
            'oversold_success_count': 0,
            'overbought_success_count': 0,
            'bin_0_samples': 0,
            'bin_1_samples': 0
        }

        # Filtrage
        df_test_filtered = df[df['class_binaire'].isin([0, 1])].copy()
        target_y_test = df['class_binaire']

        # Calculs pour le bin 0 (anti-divergence/oversold) uniquement si optimize_oversold est activé
        if optimize_oversold:
            oversold_df = df_test_filtered[df_test_filtered['anti_divergence'] == 1]
            if len(oversold_df) > 0:
                results['bin_0_win_rate'] = oversold_df['class_binaire'].mean()
                results['bin_0_pct'] = len(oversold_df) / len(df_test_filtered)
                results['oversold_success_count'] = oversold_df['class_binaire'].sum()
                results['bin_0_samples'] = len(oversold_df)

        # Calculs pour le bin 1 (divergence baissière/overbought) uniquement si optimize_overbought est activé
        if optimize_overbought:
            overbought_df = df_test_filtered[df_test_filtered['bearish_divergence'] == 1]
            if len(overbought_df) > 0:
                results['bin_1_win_rate'] = overbought_df['class_binaire'].mean()
                results['bin_1_pct'] = len(overbought_df) / len(df_test_filtered)
                results['overbought_success_count'] = overbought_df['class_binaire'].sum()
                results['bin_1_samples'] = len(overbought_df)

        # Calculer le spread uniquement si les deux modes sont activés
        if optimize_oversold and optimize_overbought:
            if 'oversold_df' not in locals():
                oversold_df = df_test_filtered[df_test_filtered['anti_divergence'] == 1]
            if 'overbought_df' not in locals():
                overbought_df = df_test_filtered[df_test_filtered['bearish_divergence'] == 1]

            if len(oversold_df) > 0 and len(overbought_df) > 0:
                results['bin_spread'] = results['bin_1_win_rate'] - results['bin_0_win_rate']

        return results, df_test_filtered, target_y_test

    except Exception as e:
        print(f"Erreur lors de l'évaluation des divergences MFI: {e}")
        import traceback
        traceback.print_exc()

        # En cas d'erreur, retourner des valeurs par défaut cohérentes
        df_test_filtered = df[
            df['class_binaire'].isin([0, 1])].copy() if 'df_test_filtered' not in locals() else df_test_filtered
        target_y_test = df['class_binaire'] if 'target_y_test' not in locals() else target_y_test

        return {
            'bin_0_win_rate': 0,
            'bin_1_win_rate': 0,
            'bin_0_pct': 0,
            'bin_1_pct': 0,
            'bin_spread': 0,
            'oversold_success_count': 0,
            'overbought_success_count': 0,
            'bin_0_samples': 0,
            'bin_1_samples': 0
        }, df_test_filtered, target_y_test


def evaluate_regression_r2(params, df, optimize_oversold=False, optimize_overbought=False):
    """
    Évalue l'indicateur de régression R² avec les paramètres optimaux.

    Parameters:
    -----------
    params : dict
        Dictionnaire contenant les paramètres optimisés
    df : pandas.DataFrame
        DataFrame complet
    optimize_oversold : bool, default=False
        Indique si l'optimisation des zones de volatilité extrême est activée
    optimize_overbought : bool, default=False
        Indique si l'optimisation des zones de volatilité modérée est activée

    Returns:
    --------
    tuple
        (dict, pandas.DataFrame, pandas.Series)
        - Dictionnaire contenant les métriques d'évaluation
        - DataFrame filtré pour les tests
        - Série des valeurs cibles
    """
    try:
        # Extraire les paramètres
        period_var = params.get('period_var_r2', params.get('period_var', None))
        r2_low_threshold = params.get('r2_low_threshold')
        r2_high_threshold = params.get('r2_high_threshold')

        # Calculer les pentes et R²
        sc_close = pd.to_numeric(df['sc_close'], errors='coerce').values
        session_starts = (df['sc_sessionStartEnd'] == 10).values
        _, r2s, _ = calculate_slopes_and_r2_numba(sc_close, session_starts, period_var)

        # Créer les indicateurs binaires conditionnels
        if optimize_overbought:
            df['range_volatility'] = np.where((r2s > r2_low_threshold) & (r2s < r2_high_threshold), 1, 0)

        if optimize_oversold:
            df['extrem_volatility'] = np.where((r2s < r2_low_threshold) | (r2s > r2_high_threshold), 1, 0)

        # Initialiser les résultats
        results = {
            'bin_0_win_rate': 0,
            'bin_1_win_rate': 0,
            'bin_0_pct': 0,
            'bin_1_pct': 0,
            'bin_spread': 0,
            'oversold_success_count': 0,
            'overbought_success_count': 0,
            'bin_0_samples': 0,
            'bin_1_samples': 0
        }

        # Filtrage
        df_test_filtered = df[df['class_binaire'].isin([0, 1])].copy()
        target_y_test = df['class_binaire']

        # Calculs pour le bin 0 (volatilité extrême) uniquement si optimize_oversold est activé
        if optimize_oversold:
            oversold_df = df_test_filtered[df_test_filtered['extrem_volatility'] == 1]
            if len(oversold_df) > 0:
                results['bin_0_win_rate'] = oversold_df['class_binaire'].mean()
                results['bin_0_pct'] = len(oversold_df) / len(df_test_filtered)
                results['oversold_success_count'] = oversold_df['class_binaire'].sum()
                results['bin_0_samples'] = len(oversold_df)

        # Calculs pour le bin 1 (volatilité modérée) uniquement si optimize_overbought est activé
        if optimize_overbought:
            overbought_df = df_test_filtered[df_test_filtered['range_volatility'] == 1]
            if len(overbought_df) > 0:
                results['bin_1_win_rate'] = overbought_df['class_binaire'].mean()
                results['bin_1_pct'] = len(overbought_df) / len(df_test_filtered)
                results['overbought_success_count'] = overbought_df['class_binaire'].sum()
                results['bin_1_samples'] = len(overbought_df)

        # Calculer le spread uniquement si les deux modes sont activés
        if optimize_oversold and optimize_overbought:
            if 'oversold_df' not in locals():
                oversold_df = df_test_filtered[df_test_filtered['extrem_volatility'] == 1]
            if 'overbought_df' not in locals():
                overbought_df = df_test_filtered[df_test_filtered['range_volatility'] == 1]

            if len(oversold_df) > 0 and len(overbought_df) > 0:
                results['bin_spread'] = results['bin_1_win_rate'] - results['bin_0_win_rate']

        return results, df_test_filtered, target_y_test

    except Exception as e:
        print(f"Erreur lors de l'évaluation de R²: {e}")
        import traceback
        traceback.print_exc()

        # En cas d'erreur, retourner des valeurs par défaut cohérentes
        df_test_filtered = df[
            df['class_binaire'].isin([0, 1])].copy() if 'df_test_filtered' not in locals() else df_test_filtered
        target_y_test = df['class_binaire'] if 'target_y_test' not in locals() else target_y_test

        return {
            'bin_0_win_rate': 0,
            'bin_1_win_rate': 0,
            'bin_0_pct': 0,
            'bin_1_pct': 0,
            'bin_spread': 0,
            'oversold_success_count': 0,
            'overbought_success_count': 0,
            'bin_0_samples': 0,
            'bin_1_samples': 0
        }, df_test_filtered, target_y_test


def evaluate_regression_std(params, df, optimize_oversold=False, optimize_overbought=False):
    """
    Évalue l'indicateur de régression par écart-type avec les paramètres optimaux.

    Parameters:
    -----------
    params : dict
        Dictionnaire contenant les paramètres optimisés
    df : pandas.DataFrame
        DataFrame complet
    optimize_oversold : bool, default=False
        Indique si l'optimisation des zones de volatilité extrême est activée
    optimize_overbought : bool, default=False
        Indique si l'optimisation des zones de volatilité modérée est activée

    Returns:
    --------
    tuple
        - dict des résultats
        - DataFrame filtré avec class_binaire ∈ [0,1]
        - Série target des valeurs de class_binaire
    """
    try:
        # Extraire les paramètres
        period_var = params.get('period_var_std', params.get('period_var', None))
        std_low_threshold = params.get('std_low_threshold')
        std_high_threshold = params.get('std_high_threshold')

        # Calculer les écarts-types
        sc_close = pd.to_numeric(df['sc_close'], errors='coerce').values
        session_starts = (df['sc_sessionStartEnd'] == 10).values
        # Correction de la syntaxe pour extraire uniquement le 3ème élément (stds)
        _, _, stds = calculate_slopes_and_r2_numba(sc_close, session_starts, period_var)

        # Créer les indicateurs binaires conditionnels
        if optimize_overbought:
            df['range_volatility'] = np.where((stds > std_low_threshold) & (stds < std_high_threshold), 1, 0)

        if optimize_oversold:
            df['extrem_volatility'] = np.where((stds < std_low_threshold) | (stds > std_high_threshold), 1, 0)

        # Initialiser les résultats
        results = {
            'bin_0_win_rate': 0,
            'bin_1_win_rate': 0,
            'bin_0_pct': 0,
            'bin_1_pct': 0,
            'bin_spread': 0,
            'oversold_success_count': 0,
            'overbought_success_count': 0,
            'bin_0_samples': 0,
            'bin_1_samples': 0,
            'total_samples': 0
        }

        # Filtrage
        df_test_filtered = df[df['class_binaire'].isin([0, 1])].copy()
        target_y_test = df['class_binaire']

        # Mise à jour du nombre total d'échantillons
        results['total_samples'] = len(df_test_filtered)

        # Calculs pour le bin 0 (volatilité extrême) uniquement si optimize_oversold est activé
        if optimize_oversold:
            oversold_df = df_test_filtered[df_test_filtered['extrem_volatility'] == 1]
            if len(oversold_df) > 0:
                results['bin_0_win_rate'] = oversold_df['class_binaire'].mean()
                results['bin_0_pct'] = len(oversold_df) / len(df_test_filtered)
                results['oversold_success_count'] = oversold_df['class_binaire'].sum()
                results['bin_0_samples'] = len(oversold_df)

        # Calculs pour le bin 1 (volatilité modérée) uniquement si optimize_overbought est activé
        if optimize_overbought:
            overbought_df = df_test_filtered[df_test_filtered['range_volatility'] == 1]
            if len(overbought_df) > 0:
                results['bin_1_win_rate'] = overbought_df['class_binaire'].mean()
                results['bin_1_pct'] = len(overbought_df) / len(df_test_filtered)
                results['overbought_success_count'] = overbought_df['class_binaire'].sum()
                results['bin_1_samples'] = len(overbought_df)

        # Calculer le spread uniquement si les deux modes sont activés
        if optimize_oversold and optimize_overbought:
            if 'oversold_df' not in locals():
                oversold_df = df_test_filtered[df_test_filtered['extrem_volatility'] == 1]
            if 'overbought_df' not in locals():
                overbought_df = df_test_filtered[df_test_filtered['range_volatility'] == 1]

            if len(oversold_df) > 0 and len(overbought_df) > 0:
                results['bin_spread'] = results['bin_1_win_rate'] - results['bin_0_win_rate']

        return results, df_test_filtered, target_y_test

    except Exception as e:
        print(f"Erreur lors de l'évaluation de regression std: {e}")
        import traceback
        traceback.print_exc()

        # En cas d'erreur, retourner des valeurs par défaut cohérentes
        df_test_filtered = df[
            df['class_binaire'].isin([0, 1])].copy() if 'df_test_filtered' not in locals() else df_test_filtered
        target_y_test = df['class_binaire'] if 'target_y_test' not in locals() else target_y_test

        return {
            'bin_0_win_rate': 0,
            'bin_1_win_rate': 0,
            'bin_0_pct': 0,
            'bin_1_pct': 0,
            'bin_spread': 0,
            'oversold_success_count': 0,
            'overbought_success_count': 0,
            'bin_0_samples': 0,
            'bin_1_samples': 0,
            'total_samples': len(df_test_filtered) if 'df_test_filtered' in locals() else 0
        }, df_test_filtered, target_y_test


def evaluate_stochastic(params, df, optimize_oversold=False, optimize_overbought=False):
    """
    Évalue l'indicateur Stochastique avec les paramètres optimaux.

    Parameters:
    -----------
    params : dict
        Dictionnaire contenant les paramètres optimisés (k_period, d_period, OS_limit, OB_limit)
    df : pandas.DataFrame
        DataFrame complet contenant toutes les données
    optimize_oversold : bool, default=False
        Indique si l'optimisation des zones de survente est activée
    optimize_overbought : bool, default=False
        Indique si l'optimisation des zones de surachat est activée

    Returns:
    --------
    tuple
        - dict des résultats
        - DataFrame filtré avec class_binaire ∈ [0,1]
        - Série target des valeurs de class_binaire
    """
    try:
        # Extraire les paramètres
        k_period = params.get('k_period')
        d_period = params.get('d_period')
        OS_limit = params.get('OS_limit', 20)  # Valeur par défaut 20 si non spécifié
        OB_limit = params.get('OB_limit', 80)  # Valeur par défaut 80 si non spécifié

        # Calculer le Stochastique
        sc_high = pd.to_numeric(df['sc_high'], errors='coerce')
        sc_low = pd.to_numeric(df['sc_low'], errors='coerce')
        sc_close = pd.to_numeric(df['sc_close'], errors='coerce')
        session_starts = (df['sc_sessionStartEnd'] == 10).values

        k_values, d_values = compute_stoch(sc_high, sc_low, sc_close, session_starts, k_period=k_period, d_period=d_period)

        # Créer les indicateurs binaires conditionnels
        if optimize_overbought:
            df['stoch_overbought'] = np.where(k_values > OB_limit, 1, 0)

        if optimize_oversold:
            df['stoch_oversold'] = np.where(k_values < OS_limit, 1, 0)

        # Initialiser les résultats
        results = {
            'bin_0_win_rate': 0,
            'bin_1_win_rate': 0,
            'bin_0_pct': 0,
            'bin_1_pct': 0,
            'bin_spread': 0,
            'oversold_success_count': 0,
            'overbought_success_count': 0,
            'bin_0_samples': 0,
            'bin_1_samples': 0
        }

        # Filtrage
        df_test_filtered = df[df['class_binaire'].isin([0, 1])].copy()
        target_y_test = df['class_binaire']

        # Zone survente (bin 0) si optimize_oversold activé
        if optimize_oversold:
            oversold_df = df_test_filtered[df_test_filtered['stoch_oversold'] == 1]
            if len(oversold_df) > 0:
                results['bin_0_win_rate'] = oversold_df['class_binaire'].mean()
                results['bin_0_pct'] = len(oversold_df) / len(df_test_filtered)
                results['oversold_success_count'] = oversold_df['class_binaire'].sum()
                results['bin_0_samples'] = len(oversold_df)

        # Zone surachat (bin 1) si optimize_overbought activé
        if optimize_overbought:
            overbought_df = df_test_filtered[df_test_filtered['stoch_overbought'] == 1]
            if len(overbought_df) > 0:
                results['bin_1_win_rate'] = overbought_df['class_binaire'].mean()
                results['bin_1_pct'] = len(overbought_df) / len(df_test_filtered)
                results['overbought_success_count'] = overbought_df['class_binaire'].sum()
                results['bin_1_samples'] = len(overbought_df)

        # Calcul du spread si les deux zones sont activées
        if optimize_oversold and optimize_overbought:
            if 'oversold_df' not in locals():
                oversold_df = df_test_filtered[df_test_filtered['stoch_oversold'] == 1]
            if 'overbought_df' not in locals():
                overbought_df = df_test_filtered[df_test_filtered['stoch_overbought'] == 1]

            if len(oversold_df) > 0 and len(overbought_df) > 0:
                results['bin_spread'] = results['bin_1_win_rate'] - results['bin_0_win_rate']

        return results, df_test_filtered, target_y_test

    except Exception as e:
        print(f"Erreur lors de l'évaluation du Stochastique: {e}")
        import traceback
        traceback.print_exc()

        # En cas d'erreur, retourner des valeurs par défaut cohérentes
        df_test_filtered = df[
            df['class_binaire'].isin([0, 1])].copy() if 'df_test_filtered' not in locals() else df_test_filtered
        target_y_test = df['class_binaire'] if 'target_y_test' not in locals() else target_y_test

        return {
            'bin_0_win_rate': 0,
            'bin_1_win_rate': 0,
            'bin_0_pct': 0,
            'bin_1_pct': 0,
            'bin_spread': 0,
            'oversold_success_count': 0,
            'overbought_success_count': 0,
            'bin_0_samples': 0,
            'bin_1_samples': 0
        }, df_test_filtered, target_y_test


def evaluate_regression_slope(params, df, optimize_oversold=False, optimize_overbought=False):
    """
    Évalue l'indicateur de régression par pente avec les paramètres optimaux.

    Parameters:
    -----------
    params : dict
        Dictionnaire contenant les paramètres optimisés
    df : pandas.DataFrame
        DataFrame complet
    df_filtered : pandas.DataFrame
        DataFrame filtré ne contenant que les entrées avec class_binaire en [0, 1]
    optimize_oversold : bool, default=False
        Indique si l'optimisation des zones de survente est activée
    optimize_overbought : bool, default=False
        Indique si l'optimisation des zones de surachat est activée

    Returns:
    --------
    dict
        Dictionnaire contenant les métriques d'évaluation
    """
    # Extraire les paramètres
    period_var = params.get('period_var_slope', params.get('period_var', None))
    slope_range_threshold = params.get('slope_range_threshold')
    slope_extrem_threshold = params.get('slope_extrem_threshold')

    # Calculer les pentes
    sc_close = pd.to_numeric(df['sc_close'], errors='coerce').values
    session_starts = (df['sc_sessionStartEnd'] == 10).values
    slopes, _, _ = calculate_slopes_and_r2_numba(sc_close, session_starts, period_var)

    # Créer les indicateurs binaires uniquement pour les modes activés
    if optimize_overbought:
        df['is_low_slope'] = np.where((slopes > slope_range_threshold) & (slopes < slope_extrem_threshold), 1, 0)

    if optimize_oversold:
        df['is_high_slope'] = np.where((slopes < slope_range_threshold) | (slopes > slope_extrem_threshold), 1, 0)

    # Initialiser les résultats
    results = {
        'bin_0_win_rate': 0,
        'bin_1_win_rate': 0,
        'bin_0_pct': 0,
        'bin_1_pct': 0,
        'bin_spread': 0,
        'oversold_success_count': 0,
        'overbought_success_count': 0,
        'bin_0_samples': 0,
        'bin_1_samples': 0
    }

    # Filtrer pour ne garder que les entrées avec trade (0 ou 1)
    df_test_filtered = df[df['class_binaire'].isin([0, 1])].copy()
    target_y_test = df['class_binaire']

    # Calculs pour le bin 0 (pente élevée) uniquement si optimize_oversold est activé
    if optimize_oversold:
        oversold_df = df_test_filtered[df_test_filtered['is_high_slope'] == 1]
        if len(oversold_df) > 0:
            results['bin_0_win_rate'] = oversold_df['class_binaire'].mean()
            results['bin_0_pct'] = len(oversold_df) / len(df_test_filtered)
            results['oversold_success_count'] = oversold_df['class_binaire'].sum()
            results['bin_0_samples'] = len(oversold_df)

    # Calculs pour le bin 1 (pente modérée) uniquement si optimize_overbought est activé
    if optimize_overbought:
        overbought_df = df_test_filtered[df_test_filtered['is_low_slope'] == 1]
        if len(overbought_df) > 0:
            results['bin_1_win_rate'] = overbought_df['class_binaire'].mean()
            results['bin_1_pct'] = len(overbought_df) / len(df_test_filtered)
            results['overbought_success_count'] = overbought_df['class_binaire'].sum()
            results['bin_1_samples'] = len(overbought_df)

    # Calculer le spread uniquement si les deux modes sont activés
    if optimize_oversold and optimize_overbought:
        oversold_df = df_test_filtered[df_test_filtered['is_high_slope'] == 1] if 'oversold_df' not in locals() else oversold_df
        overbought_df = df_test_filtered[
            df_test_filtered['is_low_slope'] == 1] if 'overbought_df' not in locals() else overbought_df

        if len(oversold_df) > 0 and len(overbought_df) > 0:
            results['bin_spread'] = results['bin_1_win_rate'] - results['bin_0_win_rate']

    return results,df_test_filtered,target_y_test


def evaluate_atr(params, df, optimize_oversold=False, optimize_overbought=False):
    """
    Évalue l'indicateur ATR avec les paramètres optimaux.

    Parameters:
    -----------
    params : dict
        Dictionnaire contenant les paramètres optimisés
    df : pandas.DataFrame
        DataFrame complet
    optimize_oversold : bool, default=False
        Indique si l'optimisation des zones de survente est activée
    optimize_overbought : bool, default=False
        Indique si l'optimisation des zones de surachat est activée

    Returns:
    --------
    tuple
        (dict, pandas.DataFrame, pandas.Series)
        - Dictionnaire contenant les métriques d'évaluation
        - DataFrame filtré pour les tests
        - Série des valeurs cibles
    """
    # Extraire les paramètres
    period_var = params.get('period_var_atr', params.get('period_var', None))
    atr_low_threshold = params.get('atr_low_threshold')
    atr_high_threshold = params.get('atr_high_threshold')

    # Calculer l'ATR
    atr = calculate_atr(df, period_var)

    # Créer les indicateurs binaires uniquement pour les modes activés
    if optimize_overbought:
        df['atr_range'] = np.where((atr > atr_low_threshold) & (atr < atr_high_threshold), 1, 0)

    if optimize_oversold:
        df['atr_extrem'] = np.where((atr < atr_low_threshold), 1, 0)

    # Initialiser les résultats
    results = {
        'bin_0_win_rate': 0,
        'bin_1_win_rate': 0,
        'bin_0_pct': 0,
        'bin_1_pct': 0,
        'bin_spread': 0,
        'oversold_success_count': 0,
        'overbought_success_count': 0,
        'bin_0_samples': 0,
        'bin_1_samples': 0
    }

    # Filtrer pour ne garder que les entrées avec trade (0 ou 1)
    df_test_filtered = df[df['class_binaire'].isin([0, 1])].copy()
    target_y_test = df['class_binaire']

    # Calculs pour le bin 0 (ATR extrême) uniquement si optimize_oversold est activé
    if optimize_oversold:
        oversold_df = df_test_filtered[df_test_filtered['atr_extrem'] == 1]
        if len(oversold_df) > 0:
            results['bin_0_win_rate'] = oversold_df['class_binaire'].mean()
            results['bin_0_pct'] = len(oversold_df) / len(df_test_filtered)
            results['oversold_success_count'] = oversold_df['class_binaire'].sum()
            results['bin_0_samples'] = len(oversold_df)

    # Calculs pour le bin 1 (ATR modéré) uniquement si optimize_overbought est activé
    if optimize_overbought:
        overbought_df = df_test_filtered[df_test_filtered['atr_range'] == 1]
        if len(overbought_df) > 0:
            results['bin_1_win_rate'] = overbought_df['class_binaire'].mean()
            results['bin_1_pct'] = len(overbought_df) / len(df_test_filtered)
            results['overbought_success_count'] = overbought_df['class_binaire'].sum()
            results['bin_1_samples'] = len(overbought_df)

    # Calculer le spread uniquement si les deux modes sont activés
    if optimize_oversold and optimize_overbought:
        oversold_df = df_test_filtered[
            df_test_filtered['atr_extrem'] == 1] if 'oversold_df' not in locals() else oversold_df
        overbought_df = df_test_filtered[
            df_test_filtered['atr_range'] == 1] if 'overbought_df' not in locals() else overbought_df

        if len(oversold_df) > 0 and len(overbought_df) > 0:
            results['bin_spread'] = results['bin_1_win_rate'] - results['bin_0_win_rate']

    return results, df_test_filtered, target_y_test


def evaluate_vwap_zscore(params, df, optimize_oversold=False, optimize_overbought=False):
    """
    Évalue l'indicateur Z-Score basé sur VWAP avec les paramètres optimaux.

    Logique:
    - oversold = Z-Score extrême (< zscore_low_threshold OU > zscore_high_threshold)
    - overbought = Z-Score modéré (entre zscore_low_threshold et zscore_high_threshold)

    Parameters:
    -----------
    params : dict
        Dictionnaire contenant les paramètres optimisés (period_var_zscore, zscore_low_threshold, zscore_high_threshold)
    df : pandas.DataFrame
        DataFrame complet
    optimize_oversold : bool, default=False
        Indique si l'optimisation des zones de Z-Score extrême (oversold) est activée
    optimize_overbought : bool, default=False
        Indique si l'optimisation des zones de Z-Score modéré (overbought) est activée

    Returns:
    --------
    tuple
        (dict, pandas.DataFrame, pandas.Series)
        - Dictionnaire contenant les métriques d'évaluation
        - DataFrame filtré pour les tests
        - Série des valeurs cibles
    """
    # Extraire les paramètres
    period_var_zscore = params.get('period_var_zscore')
    zscore_low_threshold = params.get('zscore_low_threshold')
    zscore_high_threshold = params.get('zscore_high_threshold')

    # Calculer le Z-Score VWAP
    _, zscores = enhanced_close_to_vwap_zscore(df, period_var_zscore)

    # Créer les indicateurs binaires uniquement pour les modes activés
    if optimize_overbought:
        # Z-Score modéré (entre sc_low et sc_high) = condition overbought
        df['is_zscore_vwap_moderate'] = np.where(
            (zscores >= zscore_low_threshold) & (zscores <= zscore_high_threshold),
            1, 0
        )

    if optimize_oversold:
        # Z-Score extrême (< sc_low OU > sc_high) = condition oversold
        df['is_zscore_vwap_extrem'] = np.where(
            (zscores < zscore_low_threshold) | (zscores > zscore_high_threshold),
            1, 0
        )

    # Initialiser les résultats
    results = {
        'bin_0_win_rate': 0,
        'bin_1_win_rate': 0,
        'bin_0_pct': 0,
        'bin_1_pct': 0,
        'bin_spread': 0,
        'oversold_success_count': 0,
        'overbought_success_count': 0,
        'bin_0_samples': 0,
        'bin_1_samples': 0
    }

    # Filtrer pour ne garder que les entrées avec trade (0 ou 1)
    df_test_filtered = df[df['class_binaire'].isin([0, 1])].copy()
    target_y_test = df['class_binaire']

    # Calculs pour le bin 0 (Z-Score VWAP extrême / oversold) uniquement si optimize_oversold est activé
    if optimize_oversold:
        oversold_df = df_test_filtered[df_test_filtered['is_zscore_vwap_extrem'] == 1]
        if len(oversold_df) > 0:
            results['bin_0_win_rate'] = oversold_df['class_binaire'].mean()
            results['bin_0_pct'] = len(oversold_df) / len(df_test_filtered)
            results['oversold_success_count'] = oversold_df['class_binaire'].sum()
            results['bin_0_samples'] = len(oversold_df)

    # Calculs pour le bin 1 (Z-Score VWAP modéré / overbought) uniquement si optimize_overbought est activé
    if optimize_overbought:
        overbought_df = df_test_filtered[df_test_filtered['is_zscore_vwap_moderate'] == 1]
        if len(overbought_df) > 0:
            results['bin_1_win_rate'] = overbought_df['class_binaire'].mean()
            results['bin_1_pct'] = len(overbought_df) / len(df_test_filtered)
            results['overbought_success_count'] = overbought_df['class_binaire'].sum()
            results['bin_1_samples'] = len(overbought_df)

    # Calculer le spread uniquement si les deux modes sont activés
    if optimize_oversold and optimize_overbought:
        oversold_df = df_test_filtered[
            df_test_filtered['is_zscore_vwap_extrem'] == 1] if 'oversold_df' not in locals() else oversold_df
        overbought_df = df_test_filtered[
            df_test_filtered['is_zscore_vwap_moderate'] == 1] if 'overbought_df' not in locals() else overbought_df

        if len(oversold_df) > 0 and len(overbought_df) > 0:
            results['bin_spread'] = results['bin_1_win_rate'] - results['bin_0_win_rate']

    # Ajouter les valeurs de Z-Score aux résultats pour référence
    results['period_var_zscore'] = period_var_zscore
    results['zscore_low_threshold'] = zscore_low_threshold
    results['zscore_high_threshold'] = zscore_high_threshold

    return results, df_test_filtered, target_y_test


def evaluate_percent_bb(params, df, optimize_oversold=False, optimize_overbought=False):
    """
    Évalue l'indicateur %B des bandes de Bollinger avec les paramètres optimaux.

    Parameters:
    -----------
    params : dict
        Dictionnaire contenant les paramètres optimisés
    df : pandas.DataFrame
        DataFrame complet
    optimize_oversold : bool, default=False
        Indique si l'optimisation des zones de survente est activée
    optimize_overbought : bool, default=False
        Indique si l'optimisation des zones de surachat est activée

    Returns:
    --------
    tuple
        (dict, pandas.DataFrame, pandas.Series)
        - Dictionnaire contenant les métriques d'évaluation
        - DataFrame filtré pour les tests
        - Série des valeurs cibles
    """
    try:
        # Extraire les paramètres
        period = params.get('period_var_bb', params.get('period', None))
        std_dev = params.get('std_dev')
        bb_low_threshold = params.get('bb_low_threshold')
        bb_high_threshold = params.get('bb_high_threshold')

        # Calculer le %B
        percent_b_values = calculate_percent_bb(df=df, period=period, std_dev=std_dev, fill_value=0, return_array=True)

        # Créer les indicateurs binaires uniquement pour les modes activés
        if optimize_overbought:
            df['is_bb_range'] = np.where((percent_b_values >= bb_high_threshold), 1, 0)

        if optimize_oversold:
            df['is_bb_extrem'] = np.where((percent_b_values <= bb_low_threshold), 1, 0)

        # Initialiser les résultats
        results = {
            'bin_0_win_rate': 0,
            'bin_1_win_rate': 0,
            'bin_0_pct': 0,
            'bin_1_pct': 0,
            'bin_spread': 0,
            'oversold_success_count': 0,
            'overbought_success_count': 0,
            'bin_0_samples': 0,
            'bin_1_samples': 0,
            'total_samples': 0
        }

        # Filtrer pour ne garder que les entrées avec trade (0 ou 1)
        df_test_filtered = df[df['class_binaire'].isin([0, 1])].copy()
        target_y_test = df['class_binaire']

        # Mettre à jour le nombre total d'échantillons
        results['total_samples'] = len(df_test_filtered)

        # Calculs pour le bin 0 (%B extrême) uniquement si optimize_oversold est activé
        if optimize_oversold:
            oversold_df = df_test_filtered[df_test_filtered['is_bb_extrem'] == 1]
            if len(oversold_df) > 0:
                results['bin_0_win_rate'] = oversold_df['class_binaire'].mean()
                results['bin_0_pct'] = len(oversold_df) / len(df_test_filtered)
                results['oversold_success_count'] = oversold_df['class_binaire'].sum()
                results['bin_0_samples'] = len(oversold_df)

        # Calculs pour le bin 1 (%B modéré) uniquement si optimize_overbought est activé
        if optimize_overbought:
            overbought_df = df_test_filtered[df_test_filtered['is_bb_range'] == 1]
            if len(overbought_df) > 0:
                results['bin_1_win_rate'] = overbought_df['class_binaire'].mean()
                results['bin_1_pct'] = len(overbought_df) / len(df_test_filtered)
                results['overbought_success_count'] = overbought_df['class_binaire'].sum()
                results['bin_1_samples'] = len(overbought_df)

        # Calculer le spread uniquement si les deux modes sont activés
        if optimize_oversold and optimize_overbought:
            oversold_df = df_test_filtered[
                df_test_filtered['is_bb_extrem'] == 1] if 'oversold_df' not in locals() else oversold_df
            overbought_df = df_test_filtered[
                df_test_filtered['is_bb_range'] == 1] if 'overbought_df' not in locals() else overbought_df

            if len(oversold_df) > 0 and len(overbought_df) > 0:
                results['bin_spread'] = results['bin_1_win_rate'] - results['bin_0_win_rate']

        return results, df_test_filtered, target_y_test

    except Exception as e:
        print(f"Erreur lors de l'évaluation de Percent BB: {e}")
        import traceback
        traceback.print_exc()

        # En cas d'erreur, retourner des valeurs par défaut cohérentes
        df_test_filtered = df[
            df['class_binaire'].isin([0, 1])].copy() if 'df_test_filtered' not in locals() else df_test_filtered
        target_y_test = df['class_binaire'] if 'target_y_test' not in locals() else target_y_test

        return {
            'bin_0_win_rate': 0,
            'bin_1_win_rate': 0,
            'bin_0_pct': 0,
            'bin_1_pct': 0,
            'bin_spread': 0,
            'oversold_success_count': 0,
            'overbought_success_count': 0,
            'bin_0_samples': 0,
            'bin_1_samples': 0,
            'total_samples': len(df_test_filtered) if 'df_test_filtered' in locals() else 0
        }, df_test_filtered, target_y_test

def evaluate_zscore(params, df, optimize_oversold=False, optimize_overbought=False):
    """
    Évalue l'indicateur Z-Score avec les paramètres optimaux.

    Parameters:
    -----------
    params : dict
        Dictionnaire contenant les paramètres optimisés
    df : pandas.DataFrame
        DataFrame complet avec colonnes 'sc_close' et 'class_binaire'
    optimize_oversold : bool, default=False
        Indique si l'optimisation des zones de survente est activée
    optimize_overbought : bool, default=False
        Indique si l'optimisation des zones de surachat est activée

    Returns:
    --------
    tuple
        - dict des résultats
        - DataFrame filtré avec class_binaire ∈ [0,1]
        - Série target des valeurs de class_binaire
    """
    try:
        # Extraire les paramètres
        period_var_zscore = params.get('period_var_zscore')
        zscore_low_threshold = params.get('zscore_low_threshold')
        zscore_high_threshold = params.get('zscore_high_threshold')

        # Calculer le Z-Score
        _, zscores = enhanced_close_to_sma_ratio(df, period_var_zscore)

        # Créer les indicateurs binaires conditionnels
        if optimize_overbought:
            df['is_zscore_range'] = np.where(
                (zscores > zscore_low_threshold) & (zscores < zscore_high_threshold), 1, 0
            )
        if optimize_oversold:
            df['is_zscore_extrem'] = np.where(
                (zscores < zscore_low_threshold) | (zscores > zscore_high_threshold), 1, 0
            )

        # Initialiser les résultats
        results = {
            'bin_0_win_rate': 0,
            'bin_1_win_rate': 0,
            'bin_0_pct': 0,
            'bin_1_pct': 0,
            'bin_spread': 0,
            'oversold_success_count': 0,
            'overbought_success_count': 0,
            'bin_0_samples': 0,
            'bin_1_samples': 0
        }

        # Filtrage
        df_test_filtered = df[df['class_binaire'].isin([0, 1])].copy()
        target_y_test = df['class_binaire']

        # Zone extrême (bin 0) si oversold activé
        if optimize_oversold:
            oversold_df = df_test_filtered[df_test_filtered['is_zscore_extrem'] == 1]
            if len(oversold_df) > 0:
                results['bin_0_win_rate'] = oversold_df['class_binaire'].mean()
                results['bin_0_pct'] = len(oversold_df) / len(df_test_filtered)
                results['oversold_success_count'] = oversold_df['class_binaire'].sum()
                results['bin_0_samples'] = len(oversold_df)

        # Zone modérée (bin 1) si overbought activé
        if optimize_overbought:
            overbought_df = df_test_filtered[df_test_filtered['is_zscore_range'] == 1]
            if len(overbought_df) > 0:
                results['bin_1_win_rate'] = overbought_df['class_binaire'].mean()
                results['bin_1_pct'] = len(overbought_df) / len(df_test_filtered)
                results['overbought_success_count'] = overbought_df['class_binaire'].sum()
                results['bin_1_samples'] = len(overbought_df)

        # Calcul du spread si les deux zones sont activées
        if optimize_oversold and optimize_overbought:
            if 'oversold_df' not in locals():
                oversold_df = df_test_filtered[df_test_filtered['is_zscore_extrem'] == 1]
            if 'overbought_df' not in locals():
                overbought_df = df_test_filtered[df_test_filtered['is_zscore_range'] == 1]

            if len(oversold_df) > 0 and len(overbought_df) > 0:
                results['bin_spread'] = results['bin_1_win_rate'] - results['bin_0_win_rate']

        return results, df_test_filtered, target_y_test

    except Exception as e:
        print(f"Erreur lors de l'évaluation de Z-Score: {e}")
        import traceback
        traceback.print_exc()
        return {}, None, None


def evaluate_regression_rs(params, df, optimize_oversold=False, optimize_overbought=False):
    """
    Évalue l'indicateur de volatilité Rogers-Satchell (non annualisé) avec des paramètres optimisés.

    Parameters
    ----------
    params : dict
        Dictionnaire contenant les paramètres optimisés (doit contenir 'period_var_std' ou 'period_var',
        ainsi que 'std_low_threshold', 'std_high_threshold')
    df : pandas.DataFrame
        DataFrame complet avec les colonnes: ['sc_high', 'sc_low', 'sc_open', 'sc_close', 'sc_sessionStartEnd', 'class_binaire']
    optimize_oversold : bool, default=False
        Active la logique 'volatilité extrême'
    optimize_overbought : bool, default=False
        Active la logique 'volatilité modérée'

    Returns
    -------
    tuple
        (results, df_test_filtered, target_y_test)
         - results : dict des métriques
         - df_test_filtered : DataFrame filtré avec class_binaire ∈ [0,1]
         - target_y_test : Série (ou array) des valeurs de class_binaire
    """
    import numpy as np
    import pandas as pd

    try:
        # Extraire les paramètres
        period_var = params.get('period_var_std', params.get('period_var', None))
        if period_var is None:
            raise ValueError("Paramètre 'period_var_std' ou 'period_var' manquant dans params.")

        std_low_threshold = params.get('rs_low_threshold')
        std_high_threshold = params.get('rs_high_threshold')

        # Convertir les colonnes en np.array
        high_values = pd.to_numeric(df['sc_high'], errors='coerce').values
        low_values = pd.to_numeric(df['sc_low'], errors='coerce').values
        open_values = pd.to_numeric(df['sc_open'], errors='coerce').values
        close_values = pd.to_numeric(df['sc_close'], errors='coerce').values

        # session_starts = True si sc_sessionStartEnd == 10, sinon False
        session_starts = (df['sc_sessionStartEnd'] == 10).values

        # Calcul de la volatilité RS SANS annualisation
        rs_volatility = calculate_rogers_satchell_numba(high_values, low_values,
                                                        open_values, close_values,
                                                        session_starts, period_var)

        # Création d'indicateurs binaires si souhaité
        # (logique identique à evaluate_regression_std, mais on applique sur rs_volatility)
        if optimize_overbought:
            df['range_volatility'] = np.where(
                (rs_volatility > std_low_threshold) & (rs_volatility < std_high_threshold),
                1, 0
            )

        if optimize_oversold:
            df['extrem_volatility'] = np.where(
                (rs_volatility < std_low_threshold) | (rs_volatility > std_high_threshold),
                1, 0
            )

        # Prépare les résultats
        results = {
            'bin_0_win_rate': 0,
            'bin_1_win_rate': 0,
            'bin_0_pct': 0,
            'bin_1_pct': 0,
            'bin_spread': 0,
            'oversold_success_count': 0,
            'overbought_success_count': 0,
            'bin_0_samples': 0,
            'bin_1_samples': 0,
            'total_samples': 0
        }

        # Filtrage sur class_binaire ∈ [0,1]
        df_test_filtered = df[df['class_binaire'].isin([0, 1])].copy()
        target_y_test = df_test_filtered['class_binaire']
        results['total_samples'] = len(df_test_filtered)

        # Bin 0 (volatilité extrême)
        if optimize_oversold:
            oversold_df = df_test_filtered[df_test_filtered['extrem_volatility'] == 1]
            if len(oversold_df) > 0:
                results['bin_0_win_rate'] = oversold_df['class_binaire'].mean()
                results['bin_0_pct'] = len(oversold_df) / len(df_test_filtered)
                results['oversold_success_count'] = oversold_df['class_binaire'].sum()
                results['bin_0_samples'] = len(oversold_df)

        # Bin 1 (volatilité modérée)
        if optimize_overbought:
            overbought_df = df_test_filtered[df_test_filtered['range_volatility'] == 1]
            if len(overbought_df) > 0:
                results['bin_1_win_rate'] = overbought_df['class_binaire'].mean()
                results['bin_1_pct'] = len(overbought_df) / len(df_test_filtered)
                results['overbought_success_count'] = overbought_df['class_binaire'].sum()
                results['bin_1_samples'] = len(overbought_df)

        # Calcul du spread si les deux modes sont activés
        if optimize_oversold and optimize_overbought:
            # S'assurer d'avoir oversold_df / overbought_df
            oversold_df = df_test_filtered[
                df_test_filtered['extrem_volatility'] == 1] if 'oversold_df' not in locals() else oversold_df
            overbought_df = df_test_filtered[
                df_test_filtered['range_volatility'] == 1] if 'overbought_df' not in locals() else overbought_df
            if len(oversold_df) > 0 and len(overbought_df) > 0:
                results['bin_spread'] = results['bin_1_win_rate'] - results['bin_0_win_rate']

        return results, df_test_filtered, target_y_test

    except Exception as e:
        print(f"Erreur lors de l'évaluation de la volatilité Rogers-Satchell: {e}")
        import traceback
        traceback.print_exc()

        # En cas d'erreur, valeurs par défaut
        df_test_filtered = df[
            df['class_binaire'].isin([0, 1])].copy() if 'df_test_filtered' not in locals() else df_test_filtered
        target_y_test = df_test_filtered['class_binaire'] if 'target_y_test' not in locals() else target_y_test

        return {
            'bin_0_win_rate': 0,
            'bin_1_win_rate': 0,
            'bin_0_pct': 0,
            'bin_1_pct': 0,
            'bin_spread': 0,
            'oversold_success_count': 0,
            'overbought_success_count': 0,
            'bin_0_samples': 0,
            'bin_1_samples': 0,
            'total_samples': len(df_test_filtered) if 'df_test_filtered' in locals() else 0
        }, df_test_filtered, target_y_test

def evaluate_pullStack_avgDiff(params, df, optimize_oversold=False, optimize_overbought=False):
    """
    Évalue l'indicateur pullStack_avgDiff (cumDOM_AskBid_pullStack_avgDiff_ratio) avec des paramètres optimisés.

    Parameters
    ----------
    params : dict
        Dictionnaire contenant les paramètres optimisés (doit contenir 'pullStack_low_threshold' et 'pullStack_high_threshold')
    df : pandas.DataFrame
        DataFrame complet avec les colonnes: ['cumDOM_AskBid_pullStack_avgDiff_ratio', 'class_binaire']
    optimize_oversold : bool, default=False
        Active la logique 'pullStack extrême'
    optimize_overbought : bool, default=False
        Active la logique 'pullStack modéré'

    Returns
    -------
    tuple
        (results, df_test_filtered, target_y_test)
         - results : dict des métriques
         - df_test_filtered : DataFrame filtré avec class_binaire ∈ [0,1]
         - target_y_test : Série (ou array) des valeurs de class_binaire
    """
    import numpy as np
    import pandas as pd

    try:
        # Extraire les paramètres
        pullStack_low_threshold = params.get('pullStack_low_threshold')
        pullStack_high_threshold = params.get('pullStack_high_threshold')

        if pullStack_low_threshold is None or pullStack_high_threshold is None:
            raise ValueError(
                "Paramètres 'pullStack_low_threshold' ou 'pullStack_high_threshold' manquants dans params.")

        # Récupérer les valeurs de pullStack
        pullStack_values = df['cumDOM_AskBid_pullStack_avgDiff_ratio'].values

        # Création d'indicateurs binaires selon le mode d'optimisation
        if optimize_overbought:
            # Zone modérée (dans l'intervalle)
            df['range_pullStack'] = np.where(
                (pullStack_values > pullStack_low_threshold) & (pullStack_values < pullStack_high_threshold),
                1, 0
            )

        if optimize_oversold:
            # Zone extrême (en dehors de l'intervalle)
            df['extrem_pullStack'] = np.where(
                (pullStack_values < pullStack_low_threshold) | (pullStack_values > pullStack_high_threshold),
                1, 0
            )

        # Prépare les résultats
        results = {
            'bin_0_win_rate': 0,
            'bin_1_win_rate': 0,
            'bin_0_pct': 0,
            'bin_1_pct': 0,
            'bin_spread': 0,
            'oversold_success_count': 0,
            'overbought_success_count': 0,
            'bin_0_samples': 0,
            'bin_1_samples': 0,
            'total_samples': 0
        }

        # Filtrage sur class_binaire ∈ [0,1]
        df_test_filtered = df[df['class_binaire'].isin([0, 1])].copy()
        target_y_test = df_test_filtered['class_binaire']
        results['total_samples'] = len(df_test_filtered)

        # Bin 0 (pullStack extrême)
        if optimize_oversold:
            oversold_df = df_test_filtered[df_test_filtered['extrem_pullStack'] == 1]
            if len(oversold_df) > 0:
                results['bin_0_win_rate'] = oversold_df['class_binaire'].mean()
                results['bin_0_pct'] = len(oversold_df) / len(df_test_filtered)
                results['oversold_success_count'] = oversold_df['class_binaire'].sum()
                results['bin_0_samples'] = len(oversold_df)

        # Bin 1 (pullStack modéré)
        if optimize_overbought:
            overbought_df = df_test_filtered[df_test_filtered['range_pullStack'] == 1]
            if len(overbought_df) > 0:
                results['bin_1_win_rate'] = overbought_df['class_binaire'].mean()
                results['bin_1_pct'] = len(overbought_df) / len(df_test_filtered)
                results['overbought_success_count'] = overbought_df['class_binaire'].sum()
                results['bin_1_samples'] = len(overbought_df)

        # Calcul du spread si les deux modes sont activés
        if optimize_oversold and optimize_overbought:
            # S'assurer d'avoir oversold_df / overbought_df
            oversold_df = df_test_filtered[
                df_test_filtered['extrem_pullStack'] == 1] if 'oversold_df' not in locals() else oversold_df
            overbought_df = df_test_filtered[
                df_test_filtered['range_pullStack'] == 1] if 'overbought_df' not in locals() else overbought_df
            if len(oversold_df) > 0 and len(overbought_df) > 0:
                results['bin_spread'] = results['bin_1_win_rate'] - results['bin_0_win_rate']

        return results, df_test_filtered, target_y_test

    except Exception as e:
        print(f"Erreur lors de l'évaluation de pullStack_avgDiff: {e}")
        import traceback
        traceback.print_exc()

        # En cas d'erreur, valeurs par défaut
        df_test_filtered = df[
            df['class_binaire'].isin([0, 1])].copy() if 'df_test_filtered' not in locals() else df_test_filtered
        target_y_test = df_test_filtered['class_binaire'] if 'target_y_test' not in locals() else target_y_test

        return {
            'bin_0_win_rate': 0,
            'bin_1_win_rate': 0,
            'bin_0_pct': 0,
            'bin_1_pct': 0,
            'bin_spread': 0,
            'oversold_success_count': 0,
            'overbought_success_count': 0,
            'bin_0_samples': 0,
            'bin_1_samples': 0,
            'total_samples': len(df_test_filtered) if 'df_test_filtered' in locals() else 0
        }, df_test_filtered, target_y_test
# ────────────────────────────────────────────────────────────────────────────────
# FONCTIONS POUR LA MATRICE JACCARD GLOBALE
# ────────────────────────────────────────────────────────────────────────────────
def evaluate_volRevMoveZone1_volImpulsMoveExtrem(params, df, optimize_oversold=False, optimize_overbought=False):
    """
    Évalue l'indicateur de ratio de volume volRevMoveZone1_volImpulsMoveExtrem avec des paramètres optimisés.

    Parameters
    ----------
    params : dict
        Dictionnaire contenant les paramètres optimisés (doit contenir 'volRev_low_threshold'
        et 'volRev_high_threshold')
    df : pandas.DataFrame
        DataFrame complet avec les colonnes nécessaires incluant 'ratio_volRevMoveZone1_volImpulsMoveExtrem_XRevZone' et 'class_binaire'
    optimize_oversold : bool, default=False
        Active la logique 'ratio extrême'
    optimize_overbought : bool, default=False
        Active la logique 'ratio modéré'

    Returns
    -------
    tuple
        (results, df_test_filtered, target_y_test)
         - results : dict des métriques
         - df_test_filtered : DataFrame filtré avec class_binaire ∈ [0,1]
         - target_y_test : Série (ou array) des valeurs de class_binaire
    """
    import numpy as np
    import pandas as pd

    try:
        # Extraire les paramètres
        volRev_low_threshold = params.get('volRev_low_threshold')
        volRev_high_threshold = params.get('volRev_high_threshold')

        if volRev_low_threshold is None or volRev_high_threshold is None:
            raise ValueError("Paramètres 'volRev_low_threshold' ou 'volRev_high_threshold' manquants dans params.")

        # Récupérer les valeurs de ratio de volume
        volRev_values = df['ratio_volRevMoveZone1_volImpulsMoveExtrem_XRevZone'].values

        # Création d'indicateurs binaires selon le mode d'optimisation
        if optimize_overbought:
            # Zone modérée (dans l'intervalle)
            df['range_volRev'] = np.where(
                (volRev_values > volRev_low_threshold) & (volRev_values < volRev_high_threshold),
                1, 0
            )

        if optimize_oversold:
            # Zone extrême (en dehors de l'intervalle)
            df['extrem_volRev'] = np.where(
                (volRev_values < volRev_low_threshold) | (volRev_values > volRev_high_threshold),
                1, 0
            )

        # Prépare les résultats
        results = {
            'bin_0_win_rate': 0,
            'bin_1_win_rate': 0,
            'bin_0_pct': 0,
            'bin_1_pct': 0,
            'bin_spread': 0,
            'oversold_success_count': 0,
            'overbought_success_count': 0,
            'bin_0_samples': 0,
            'bin_1_samples': 0,
            'total_samples': 0
        }

        # Filtrage sur class_binaire ∈ [0,1]
        df_test_filtered = df[df['class_binaire'].isin([0, 1])].copy()
        target_y_test = df_test_filtered['class_binaire']
        results['total_samples'] = len(df_test_filtered)

        # Bin 0 (ratio extrême)
        if optimize_oversold:
            oversold_df = df_test_filtered[df_test_filtered['extrem_volRev'] == 1]
            if len(oversold_df) > 0:
                results['bin_0_win_rate'] = oversold_df['class_binaire'].mean()
                results['bin_0_pct'] = len(oversold_df) / len(df_test_filtered)
                results['oversold_success_count'] = oversold_df['class_binaire'].sum()
                results['bin_0_samples'] = len(oversold_df)

        # Bin 1 (ratio modéré)
        if optimize_overbought:
            overbought_df = df_test_filtered[df_test_filtered['range_volRev'] == 1]
            if len(overbought_df) > 0:
                results['bin_1_win_rate'] = overbought_df['class_binaire'].mean()
                results['bin_1_pct'] = len(overbought_df) / len(df_test_filtered)
                results['overbought_success_count'] = overbought_df['class_binaire'].sum()
                results['bin_1_samples'] = len(overbought_df)

        # Calcul du spread si les deux modes sont activés
        if optimize_oversold and optimize_overbought:
            # S'assurer d'avoir oversold_df / overbought_df
            oversold_df = df_test_filtered[
                df_test_filtered['extrem_volRev'] == 1] if 'oversold_df' not in locals() else oversold_df
            overbought_df = df_test_filtered[
                df_test_filtered['range_volRev'] == 1] if 'overbought_df' not in locals() else overbought_df
            if len(oversold_df) > 0 and len(overbought_df) > 0:
                results['bin_spread'] = results['bin_1_win_rate'] - results['bin_0_win_rate']

        return results, df_test_filtered, target_y_test

    except Exception as e:
        print(f"Erreur lors de l'évaluation du ratio de volume: {e}")
        import traceback
        traceback.print_exc()

        # En cas d'erreur, valeurs par défaut
        df_test_filtered = df[
            df['class_binaire'].isin([0, 1])].copy() if 'df_test_filtered' not in locals() else df_test_filtered
        target_y_test = df_test_filtered['class_binaire'] if 'target_y_test' not in locals() else target_y_test

        return {
            'bin_0_win_rate': 0,
            'bin_1_win_rate': 0,
            'bin_0_pct': 0,
            'bin_1_pct': 0,
            'bin_spread': 0,
            'oversold_success_count': 0,
            'overbought_success_count': 0,
            'bin_0_samples': 0,
            'bin_1_samples': 0,
            'total_samples': len(df_test_filtered) if 'df_test_filtered' in locals() else 0
        }, df_test_filtered, target_y_test

    def calculate_jaccard_similarity(set1, set2):
        """Calcule la similarité Jaccard entre deux ensembles."""
        if len(set1) == 0 and len(set2) == 0:
            return 1.0

        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0

    def get_algo_winrate(algo_name, algo_dfs):
        """Calcule le win rate global d'un algorithme."""
        if algo_name not in algo_dfs:
            return 0

        df = algo_dfs[algo_name]
        pnl_col = next((c for c in ("PnlAfterFiltering", "trade_pnl") if c in df.columns), None)
        if pnl_col:
            wins = (df[pnl_col] > 0).sum()
            total = len(df)
            return (wins / total * 100) if total > 0 else 0
        return 0

        # ────────────────────────────────────────────────────────────────────────────────
        # MODIFICATION DE VOTRE FONCTION EXISTANTE analyse_doublons_algos
        # ────────────────────────────────────────────────────────────────────────────────

        # Dans votre fonction analyse_doublons_algos, remplacez cette section :

        # Stocker les statistiques
        pairs_stats[(a1, a2)] = {
            'common_trades': len(common),
            'winning_both': winning_both,
            'winning_a1_only': winning_a1_only,
            'winning_a2_only': winning_a2_only,
            'losing_both': losing_both,
            'agreement_rate': agreement_rate,
            'total_pnl': total_pnl,
            'unanimous_pnl': unanimous_pnl
        }

        # Calculer la similarité Jaccard pour cette paire
        set1 = uniq_sets[a1]
        set2 = uniq_sets[a2]
        jaccard_sim = calculate_jaccard_similarity(set1, set2)

        # Calculer les win rates
        total_wins_a1 = winning_both + winning_a1_only
        total_wins_a2 = winning_both + winning_a2_only
        winrate_a1_common = (total_wins_a1 / len(common) * 100) if len(common) > 0 else 0
        winrate_a2_common = (total_wins_a2 / len(common) * 100) if len(common) > 0 else 0

        # Win rates globaux
        global_wr_a1 = get_algo_winrate(a1, algo_dfs)
        global_wr_a2 = get_algo_winrate(a2, algo_dfs)

        # Déterminer le statut de diversification
        if jaccard_sim < JACCARD_THRESHOLD:
            jaccard_color = f"{Fore.GREEN}{jaccard_sim:.1%}{Style.RESET_ALL}"
            diversification_status = "DIVERSIFIÉS"
        else:
            jaccard_color = f"{Fore.RED}{jaccard_sim:.1%}{Style.RESET_ALL}"
            diversification_status = "REDONDANTS"

        # Stocker les statistiques avec les nouvelles métriques
        pairs_stats[(a1, a2)] = {
            'common_trades': len(common),
            'winning_both': winning_both,
            'winning_a1_only': winning_a1_only,
            'winning_a2_only': winning_a2_only,
            'losing_both': losing_both,
            'agreement_rate': agreement_rate,
            'total_pnl': total_pnl,
            'unanimous_pnl': unanimous_pnl,
            'jaccard_similarity': jaccard_sim,
            'winrate_a1_common': winrate_a1_common,
            'winrate_a2_common': winrate_a2_common,
            'global_wr_a1': global_wr_a1,
            'global_wr_a2': global_wr_a2
        }

        print(f"\n>> Analyse de la paire {a1} / {a2} ({diversification_status}):")
        print(f"  Trades communs: {len(common)}")
        print(f"  Gagnants pour les deux: {winning_both}")
        print(f"  Gagnants uniquement pour {a1}: {winning_a1_only}")
        print(f"  Gagnants uniquement pour {a2}: {winning_a2_only}")
        print(f"  Perdants pour les deux: {losing_both}")
        print(f"  Taux d'accord: {agreement_rate:.2f}%")
        print(f"  Win Rate {a1} (trades communs): {winrate_a1_common:.1f}%")
        print(f"  Win Rate {a2} (trades communs): {winrate_a2_common:.1f}%")
        print(f"  Win Rate {a1} (global): {global_wr_a1:.1f}%")
        print(f"  Win Rate {a2} (global): {global_wr_a2:.1f}%")
        print(f"  PnL total: {total_pnl:.2f}")
        print(f"  PnL des trades unanimes: {unanimous_pnl:.2f}")
        print(f"  Taux de Jaccard: {jaccard_color}")

    return pairs_stats, occurrences_stats



def calculate_jaccard_similarity(set1, set2):
    """Calcule la similarité Jaccard entre deux ensembles."""
    if len(set1) == 0 and len(set2) == 0:
        return 1.0

    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0


def get_algo_winrate(algo_name, algo_dfs):
    """Calcule le win rate global d'un algorithme."""
    if algo_name not in algo_dfs:
        return 0

    df = algo_dfs[algo_name]
    pnl_col = next((c for c in ("PnlAfterFiltering", "trade_pnl") if c in df.columns), None)
    if pnl_col:
        wins = (df[pnl_col] > 0).sum()
        total = len(df)
        return (wins / total * 100) if total > 0 else 0
    return 0
def create_full_jaccard_matrix(algo_dfs, indicator_columns=None):
    """Crée la matrice de similarité Jaccard complète pour tous les algorithmes."""
    if indicator_columns is None:
        indicator_columns = [
            'rsi_', 'macd', 'macd_signal', 'macd_hist',
            'timeElapsed2LastBar', 'timeStampOpening',
            'ratio_deltaRevZone_VolCandle'
        ]

    algos = list(algo_dfs.keys())
    jaccard_matrix = pd.DataFrame(0.0, index=algos, columns=algos)

    # Créer les ensembles de trades uniques pour chaque algo
    uniq_sets = {}
    for algo, df in algo_dfs.items():
        valid_cols = [c for c in indicator_columns if c in df.columns]
        if not valid_cols:
            continue

        uniq_sets[algo] = set()
        for _, row in df.drop_duplicates(subset=valid_cols).iterrows():
            key = tuple(row[col] for col in valid_cols)
            uniq_sets[algo].add(key)

    # Calculer la similarité Jaccard pour toutes les paires
    for i, algo1 in enumerate(algos):
        for j, algo2 in enumerate(algos):
            if i == j:
                jaccard_matrix.loc[algo1, algo2] = 1.0
            else:
                set1 = uniq_sets.get(algo1, set())
                set2 = uniq_sets.get(algo2, set())
                jaccard_sim = calculate_jaccard_similarity(set1, set2)
                jaccard_matrix.loc[algo1, algo2] = jaccard_sim

    return jaccard_matrix


def calculate_common_trades_matrix(algo_dfs, indicator_columns=None):
    """Calcule une matrice du nombre de trades communs entre tous les algorithmes."""
    if indicator_columns is None:
        indicator_columns = [
            'rsi_', 'macd', 'macd_signal', 'macd_hist',
            'timeElapsed2LastBar', 'timeStampOpening',
            'ratio_deltaRevZone_VolCandle'
        ]

    algos = list(algo_dfs.keys())
    common_trades_matrix = pd.DataFrame(0, index=algos, columns=algos, dtype=int)

    # Créer les ensembles de trades uniques pour chaque algo
    uniq_sets = {}
    for algo, df in algo_dfs.items():
        valid_cols = [c for c in indicator_columns if c in df.columns]
        if not valid_cols:
            continue

        uniq_sets[algo] = set()
        for _, row in df.drop_duplicates(subset=valid_cols).iterrows():
            key = tuple(row[col] for col in valid_cols)
            uniq_sets[algo].add(key)

    # Calculer le nombre de trades communs pour toutes les paires
    for i, algo1 in enumerate(algos):
        for j, algo2 in enumerate(algos):
            if i == j:
                # Diagonale = nombre total de trades uniques pour l'algo
                common_trades_matrix.loc[algo1, algo2] = len(uniq_sets.get(algo1, set()))
            else:
                set1 = uniq_sets.get(algo1, set())
                set2 = uniq_sets.get(algo2, set())
                common_count = len(set1.intersection(set2))
                common_trades_matrix.loc[algo1, algo2] = common_count

    return common_trades_matrix
    """Crée la matrice de similarité Jaccard complète pour tous les algorithmes."""
    if indicator_columns is None:
        indicator_columns = [
            'rsi_', 'macd', 'macd_signal', 'macd_hist',
            'timeElapsed2LastBar', 'timeStampOpening',
            'ratio_deltaRevZone_VolCandle'
        ]

    algos = list(algo_dfs.keys())
    jaccard_matrix = pd.DataFrame(0.0, index=algos, columns=algos)

    # Créer les ensembles de trades uniques pour chaque algo
    uniq_sets = {}
    for algo, df in algo_dfs.items():
        valid_cols = [c for c in indicator_columns if c in df.columns]
        if not valid_cols:
            continue

        uniq_sets[algo] = set()
        for _, row in df.drop_duplicates(subset=valid_cols).iterrows():
            key = tuple(row[col] for col in valid_cols)
            uniq_sets[algo].add(key)

    # Calculer la similarité Jaccard pour toutes les paires
    for i, algo1 in enumerate(algos):
        for j, algo2 in enumerate(algos):
            if i == j:
                jaccard_matrix.loc[algo1, algo2] = 1.0
            else:
                set1 = uniq_sets.get(algo1, set())
                set2 = uniq_sets.get(algo2, set())
                jaccard_sim = calculate_jaccard_similarity(set1, set2)
                jaccard_matrix.loc[algo1, algo2] = jaccard_sim

    return jaccard_matrix


# ────────────────────────────────────────────────────────────────────────────────
# FONCTIONS CORRIGÉES POUR L'ANALYSE JACCARD
# ────────────────────────────────────────────────────────────────────────────────

def calculate_jaccard_similarity(set1, set2):
    """Calcule la similarité Jaccard entre deux ensembles."""
    if len(set1) == 0 and len(set2) == 0:
        return 1.0

    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0


def get_algo_winrate(algo_name, algo_dfs):
    """Calcule le win rate global d'un algorithme."""
    if algo_name not in algo_dfs:
        return 0

    df = algo_dfs[algo_name]
    pnl_col = next((c for c in ("PnlAfterFiltering", "trade_pnl") if c in df.columns), None)
    if pnl_col:
        wins = (df[pnl_col] > 0).sum()
        total = len(df)
        return (wins / total * 100) if total > 0 else 0
    return 0


def create_full_jaccard_matrix(algo_dfs, indicator_columns=None):
    """Crée la matrice de similarité Jaccard complète pour tous les algorithmes."""
    if indicator_columns is None:
        indicator_columns = [
            'rsi_', 'macd', 'macd_signal', 'macd_hist',
            'timeElapsed2LastBar', 'timeStampOpening',
            'ratio_deltaRevZone_VolCandle'
        ]

    algos = list(algo_dfs.keys())
    jaccard_matrix = pd.DataFrame(0.0, index=algos, columns=algos)

    # Créer les ensembles de trades uniques pour chaque algo
    uniq_sets = {}
    for algo, df in algo_dfs.items():
        valid_cols = [c for c in indicator_columns if c in df.columns]
        if not valid_cols:
            continue

        uniq_sets[algo] = set()
        for _, row in df.drop_duplicates(subset=valid_cols).iterrows():
            key = tuple(row[col] for col in valid_cols)
            uniq_sets[algo].add(key)

    # Calculer la similarité Jaccard pour toutes les paires
    for i, algo1 in enumerate(algos):
        for j, algo2 in enumerate(algos):
            if i == j:
                jaccard_matrix.loc[algo1, algo2] = 1.0
            else:
                set1 = uniq_sets.get(algo1, set())
                set2 = uniq_sets.get(algo2, set())
                jaccard_sim = calculate_jaccard_similarity(set1, set2)
                jaccard_matrix.loc[algo1, algo2] = jaccard_sim

    return jaccard_matrix


def calculate_common_trades_matrix(algo_dfs, indicator_columns=None):
    """Calcule une matrice du nombre de trades communs entre tous les algorithmes."""
    if indicator_columns is None:
        indicator_columns = [
            'rsi_', 'macd', 'macd_signal', 'macd_hist',
            'timeElapsed2LastBar', 'timeStampOpening',
            'ratio_deltaRevZone_VolCandle'
        ]

    algos = list(algo_dfs.keys())
    common_trades_matrix = pd.DataFrame(0, index=algos, columns=algos, dtype=int)

    # Créer les ensembles de trades uniques pour chaque algo
    uniq_sets = {}
    for algo, df in algo_dfs.items():
        valid_cols = [c for c in indicator_columns if c in df.columns]
        if not valid_cols:
            continue

        uniq_sets[algo] = set()
        for _, row in df.drop_duplicates(subset=valid_cols).iterrows():
            key = tuple(row[col] for col in valid_cols)
            uniq_sets[algo].add(key)

    # Calculer le nombre de trades communs pour toutes les paires
    for i, algo1 in enumerate(algos):
        for j, algo2 in enumerate(algos):
            if i == j:
                # Diagonale = nombre total de trades uniques pour l'algo
                common_trades_matrix.loc[algo1, algo2] = len(uniq_sets.get(algo1, set()))
            else:
                set1 = uniq_sets.get(algo1, set())
                set2 = uniq_sets.get(algo2, set())
                common_count = len(set1.intersection(set2))
                common_trades_matrix.loc[algo1, algo2] = common_count

    return common_trades_matrix


def display_jaccard_matrix(jaccard_matrix, threshold=None, algo_dfs=None, min_common_trades=None):
    """Affiche la matrice Jaccard avec couleur des labels selon le volume de trades communs."""
    # Valeurs par défaut si None
    if threshold is None:
        threshold = 0.5
    if min_common_trades is None:
        min_common_trades = 15

    print(f"\n{Fore.CYAN}{'=' * 120}")
    print(f"MATRICE DE SIMILARITÉ JACCARD - TOUS LES ALGORITHMES (Seuil: {threshold:.1%})")
    print(f"{'=' * 120}{Style.RESET_ALL}")

    algos = list(jaccard_matrix.index)

    # Calculer les trades communs pour chaque paire si algo_dfs est fourni
    common_trades_matrix = None
    algos_with_insufficient = set()

    if algo_dfs is not None:
        common_trades_matrix = calculate_common_trades_matrix(algo_dfs)

        # Nouvelle logique: un algo est VERT s'il a au moins quelques bonnes connexions
        for i, algo1 in enumerate(algos):
            sufficient_pairs = 0
            total_pairs = 0

            for j, algo2 in enumerate(algos):
                if i != j:  # Exclure la diagonale
                    total_pairs += 1
                    common_count = common_trades_matrix.loc[algo1, algo2]
                    if common_count >= min_common_trades:
                        sufficient_pairs += 1

            # CORRECTION: Vérifier si total_pairs > 0 avant la division
            # Un algo est VERT s'il a au moins 1 bonne connexion (≥ min_common_trades)
            # OU s'il a au moins 20% de bonnes connexions
            if total_pairs > 0:
                has_good_connections = sufficient_pairs >= 1 or (sufficient_pairs / total_pairs) >= 0.2
            else:
                # Cas d'un seul algorithme : on le considère comme ayant de bonnes connexions
                has_good_connections = True

            if not has_good_connections:
                algos_with_insufficient.add(algo1)

    # En-tête avec noms courts colorés
    print(f"{'':>20}", end="")
    for algo in algos:
        short_name = algo.replace('features_algo', 'A')
        if algo in algos_with_insufficient:
            print(f"{Fore.RED}{short_name:>8}{Style.RESET_ALL}", end="")
        else:
            print(f"{Fore.GREEN}{short_name:>8}{Style.RESET_ALL}", end="")
    print()

    # Lignes de la matrice avec labels colorés
    for i, algo1 in enumerate(algos):
        short_name1 = algo1.replace('features_algo', 'A')

        # Colorer le label de ligne
        if algo1 in algos_with_insufficient:
            print(f"{Fore.RED}{short_name1:>20}{Style.RESET_ALL}", end="")
        else:
            print(f"{Fore.GREEN}{short_name1:>20}{Style.RESET_ALL}", end="")

        for j, algo2 in enumerate(algos):
            if i == j:
                print(f"{'1.00':>8}", end="")  # Diagonale
            else:
                jaccard_val = jaccard_matrix.loc[algo1, algo2]

                # Vérifier si cette paire a suffisamment de trades communs
                has_sufficient_trades = False
                if common_trades_matrix is not None:
                    common_count = common_trades_matrix.loc[algo1, algo2]
                    has_sufficient_trades = common_count >= min_common_trades

                # Afficher les valeurs Jaccard avec couleur selon seuil Jaccard
                # ET souligner si la paire a suffisamment de trades communs
                if jaccard_val < threshold:
                    if has_sufficient_trades:
                        # Vert + souligné
                        print(f"{Fore.GREEN}\033[4m{jaccard_val:>8.3f}\033[0m{Style.RESET_ALL}", end="")
                    else:
                        # Vert normal
                        print(f"{Fore.GREEN}{jaccard_val:>8.3f}{Style.RESET_ALL}", end="")
                else:
                    if has_sufficient_trades:
                        # Rouge + souligné
                        print(f"{Fore.RED}\033[4m{jaccard_val:>8.3f}\033[0m{Style.RESET_ALL}", end="")
                    else:
                        # Rouge normal
                        print(f"{Fore.RED}{jaccard_val:>8.3f}{Style.RESET_ALL}", end="")
        print()

    # Légendes
    print(f"\n{Fore.GREEN}■ Vert (Valeurs){Style.RESET_ALL}: Similarité < {threshold:.1%} (Algorithmes diversifiés)")
    print(f"{Fore.RED}■ Rouge (Valeurs){Style.RESET_ALL}: Similarité ≥ {threshold:.1%} (Algorithmes redondants)")
    print(f"\n\033[4m■ Souligné (Valeurs)\033[0m: Paires avec ≥ {min_common_trades} trades communs")
    print(
        f"\n{Fore.GREEN}■ Vert (Labels){Style.RESET_ALL}: A au moins 1 paire avec ≥ {min_common_trades} trades communs")
    print(f"{Fore.RED}■ Rouge (Labels){Style.RESET_ALL}: Aucune paire avec ≥ {min_common_trades} trades communs")


def analyze_global_redundancy(jaccard_matrix, threshold=None):
    """Analyse globale de la redondance entre algorithmes."""
    # Valeur par défaut si None
    if threshold is None:
        threshold = 0.5

    redundant_pairs = []

    for i in range(len(jaccard_matrix)):
        for j in range(i + 1, len(jaccard_matrix)):
            algo1 = jaccard_matrix.index[i]
            algo2 = jaccard_matrix.columns[j]
            similarity = jaccard_matrix.iloc[i, j]

            if similarity >= threshold:
                redundant_pairs.append((algo1, algo2, similarity))

    if redundant_pairs:
        print(f"\n{Fore.RED}⚠️  ALGORITHMES REDONDANTS GLOBAUX (Similarité ≥ {threshold:.1%}):{Style.RESET_ALL}")
        print("=" * 100)
        for algo1, algo2, sim in sorted(redundant_pairs, key=lambda x: x[2], reverse=True):
            print(f"  {algo1} ↔ {algo2}: {Fore.RED}{sim:.1%}{Style.RESET_ALL}")

        print(f"\n{Fore.YELLOW}💡 RECOMMANDATION:{Style.RESET_ALL}")
        print(f"  Considérer éliminer {len(redundant_pairs)} paires redondantes pour optimiser la diversification")
    else:
        print(f"\n{Fore.GREEN}✓ Aucune redondance globale détectée (seuil: {threshold:.1%}){Style.RESET_ALL}")

    # Statistiques de diversification
    total_pairs = len(jaccard_matrix) * (len(jaccard_matrix) - 1) // 2

    print(f"\n{Fore.CYAN}📊 STATISTIQUES DE DIVERSIFICATION:{Style.RESET_ALL}")
    print(f"  Nombre total d'algorithmes: {len(jaccard_matrix)}")
    print(f"  Paires analysées: {total_pairs}")
    print(f"  Paires redondantes: {len(redundant_pairs)}")

    # CORRECTION: Vérifier si total_pairs > 0 avant la division
    if total_pairs > 0:
        diversification_rate = (total_pairs - len(redundant_pairs)) / total_pairs * 100
        print(f"  Taux de diversification: {diversification_rate:.1f}%")
    else:
        # Cas d'un seul algorithme : pas de paires à analyser
        print(f"  Taux de diversification: N/A (un seul algorithme)")

    return redundant_pairs


import os
import pandas as pd
from typing import Dict, Any


def export_results_to_excel(results: Dict[str, Any], filename: str = "trading_analysis_results.xlsx",
                            directory_path: str = "."):
    """
    Export des résultats vers Excel avec formatage, dans un répertoire spécifié.

    Parameters:
    - results : dict
        Résultats à exporter.
    - filename : str
        Nom du fichier Excel.
    - directory_path : str
        Répertoire dans lequel enregistrer le fichier.
    """
    try:
        # Assure que le répertoire existe
        os.makedirs(directory_path, exist_ok=True)

        # Construit le chemin complet du fichier
        full_path = os.path.join(directory_path, filename)

        with pd.ExcelWriter(full_path, engine='openpyxl') as writer:
            # Sheet 1: Résumé global
            summary_data = []
            for df_name, data in results.items():
                if data is not None:
                    summary_data.append({
                        'Dataset': df_name,
                        'Nb_Sessions': data['total_sessions'],
                        'Nb_Bougies': data['total_candles'],
                        'Duree_Moyenne_s': round(data['avg_duration_overall'], 2),
                        'Bougies_par_Session': round(data['avg_candles_per_session'], 1),
                        'Volume_Moyen': round(data['volume_stats']['avg_volume_per_session'], 2),
                        'Correlation_Duree_Volume': round(data['volume_stats']['duration_volume_correlation'], 3),
                        'Periode_Debut': data['date_range'][0],
                        'Periode_Fin': data['date_range'][1]
                    })

            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Résumé_Global', index=False)

            # Sheets 2-N: Détails par dataset
            for df_name, data in results.items():
                if data is not None and 'session_stats' in data:
                    session_data = data['session_stats'].copy()
                    session_data.to_excel(writer, sheet_name=f'Détails_{df_name}', index=False)

        print(f"✅ Résultats exportés vers : {full_path}")
        return True

    except Exception as e:
        print(f"❌ Erreur lors de l'export Excel : {e}")
        return False




from numba import njit
@njit
def compute_true_range_numba(highs, lows, closes,period_window=None):
    n = len(highs)
    atr_values = np.empty(n)
    atr_values[:period_window-1] = np.nan  # ATR non défini pour les 9 premières valeurs

    for i in range(period_window-1, n):
        tr_sum = 0.0
        count = 0
        for j in range(i - period_window-1 + 1, i + 1):
            tr1 = highs[j] - lows[j]
            tr2 = abs(highs[j] - closes[j - 1])
            tr3 = abs(lows[j] - closes[j - 1])
            tr = max(tr1, tr2, tr3)
            tr_sum += tr
            count += 1
        atr_values[i] = tr_sum / count if count > 0 else np.nan

    return atr_values


import logging
from typing import Dict, Optional, Tuple, Any

# Configuration du logging pour un meilleur suivi
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
def calculate_atr_10_periods(df_session: pd.DataFrame,period_window=None) -> pd.Series:
    """
    Calcule l'ATR sur x  périodes pour une session avec Numba pour accélération
    """
    if len(df_session) < period_window:
        return pd.Series([np.nan] * len(df_session), index=df_session.index)

    if all(col in df_session.columns for col in ['sc_high', 'sc_low', 'sc_close']):
        highs = df_session['sc_high'].to_numpy()
        lows = df_session['sc_low'].to_numpy()
        closes = df_session['sc_close'].to_numpy()

        atr_values = compute_true_range_numba(highs, lows, closes,period_window)
    elif 'sc_volume' in df_session.columns:
        # Fallback: volatilité du volume
        volume = df_session['sc_volume'].to_numpy()
        atr_values = np.full(len(volume), np.nan)
        for i in range(period_window-1, len(volume)):
            atr_values[i] = np.std(volume[i-period_window-1:i+1])
    else:
        atr_values = np.full(len(df_session), np.nan)

    return pd.Series(atr_values, index=df_session.index)





def calculate_extreme_contracts_metrics(df_session: pd.DataFrame) -> dict:
    """
    Calcule les métriques des contrats extrêmes pour une session

    Parameters:
    -----------
    df_session : DataFrame
        Données d'une session

    Returns:
    --------
    dict : Dictionnaire avec les métriques des contrats extrêmes
    """
    extreme_cols = [
        'delta_impulsMove_XRevZone_bigStand_extrem',
        'delta_revMove_XRevZone_bigStand_extrem'
    ]

    # Vérifier si les colonnes existent
    available_cols = [col for col in extreme_cols if col in df_session.columns]

    if not available_cols:
        return {
            'extreme_sum_with_zeros': np.nan,
            'extreme_sum_without_zeros': np.nan,
            'extreme_count_nonzero': 0,
            'extreme_ratio': np.nan
        }

    # Calculer la somme des valeurs absolues
    df_work = df_session[available_cols].fillna(0)
    extreme_sums = df_work.abs().sum(axis=1)

    # Métriques avec zéros inclus (toutes les bougies)
    extreme_with_zeros = extreme_sums.mean()

    # Métriques sans zéros (seulement les bougies avec contrats extrêmes)
    nonzero_sums = extreme_sums[extreme_sums > 0]
    extreme_without_zeros = nonzero_sums.mean() if len(nonzero_sums) > 0 else np.nan

    # Comptage et ratio
    extreme_count_nonzero = len(nonzero_sums)
    extreme_ratio = extreme_count_nonzero / len(extreme_sums) if len(extreme_sums) > 0 else 0

    return {
        'extreme_sum_with_zeros': extreme_with_zeros,
        'extreme_sum_without_zeros': extreme_without_zeros,
        'extreme_count_nonzero': extreme_count_nonzero,
        'extreme_ratio': extreme_ratio
    }


def calculate_volume_above_metrics(df_session: pd.DataFrame,xtickReversalTickPrice=None) -> dict:
    """
    Calcule les métriques des volumes above normalisés par tick pour une session

    Parameters:
    -----------
    df_session : DataFrame
        Données d'une session

    Returns:
    --------
    dict : Dictionnaire avec les métriques des volumes above par tick
    """
    required_cols = ['sc_volAbv', 'sc_candleDir']

    # Vérifier si les colonnes existent
    available_cols = [col for col in required_cols if col in df_session.columns]

    if len(available_cols) != 2:
        return {
            'volume_above_per_tick_mean': np.nan,
            'volume_above_count': 0,
            'volume_above_ratio': np.nan
        }

    # Filtrer : VolAbv > 0 ET sc_candleDir = -1
    filtered_df = df_session[
        (df_session['sc_volAbv'] > 0) &
        (df_session['sc_candleDir'] == -1)
        ].copy()

    if len(filtered_df) == 0:
        return {
            'volume_above_per_tick_mean': np.nan,
            'volume_above_count': 0,
            'volume_above_ratio': 0
        }

    # Normaliser avec la constante globale : Volume above par tick
    filtered_df['volume_above_per_tick'] = filtered_df['sc_volAbv'] / xtickReversalTickPrice

    # Calculer les métriques
    volume_above_per_tick_mean = filtered_df['volume_above_per_tick'].mean()
    volume_above_count = len(filtered_df)
    volume_above_ratio = volume_above_count / len(df_session) if len(df_session) > 0 else 0

    return {
        'volume_above_per_tick_mean': volume_above_per_tick_mean,
        'volume_above_count': volume_above_count,
        'volume_above_ratio': volume_above_ratio
    }



def calculate_session_metrics_enhanced(df: pd.DataFrame, df_name: str,xtickReversalTickPrice=None,period_atr_stat_session=None) -> pd.DataFrame:
    """
    Calcul des métriques par session avec gestion d'erreurs améliorée.
    Inclut ATR, contrats extrêmes et volumes above par tick.
    """
    try:
        # Copie de travail
        df_work = df.copy()

        # Conversion et nettoyage des données
        numeric_columns = ['candleDuration', 'sc_volume']
        for col in numeric_columns:
            df_work[col] = pd.to_numeric(df_work[col], errors='coerce')

        # Conversion des dates avec gestion d'erreurs
        if not pd.api.types.is_datetime64_any_dtype(df_work['timeStampOpeningConvertedtoDate']):
            df_work['timeStampOpeningConvertedtoDate'] = pd.to_datetime(
                df_work['timeStampOpeningConvertedtoDate'],
                infer_datetime_format=True,
                errors='coerce'
            )

        # Suppression des lignes invalides
        before_cleaning = len(df_work)
        df_work = df_work.dropna(subset=['timeStampOpeningConvertedtoDate', 'session_id', 'candleDuration', 'sc_volume'])
        after_cleaning = len(df_work)
        if after_cleaning < before_cleaning:
            logger.warning(f"{df_name}: {before_cleaning - after_cleaning} lignes supprimées lors du nettoyage")
        if len(df_work) == 0:
            raise ValueError(f"Aucune donnée valide dans {df_name} après nettoyage")

        # Dates de session
        session_dates = df_work.groupby('session_id')['timeStampOpeningConvertedtoDate'].min() + pd.Timedelta(days=1)
        df_work['session_date'] = df_work['session_id'].map(session_dates.to_dict())

        # Calcul des métriques classiques
        agg_functions = {
            'candleDuration': ['mean', 'median', 'std', 'count', 'min', 'max'],
            'sc_volume': ['mean', 'median', 'std', 'sum', 'min', 'max']
        }
        session_stats = df_work.groupby('session_id').agg(agg_functions)
        session_stats.columns = ['_'.join(col).strip() for col in session_stats.columns]
        session_stats = session_stats.reset_index()

        # Renommage
        column_mapping = {
            'candleDuration_mean': 'duration_mean',
            'candleDuration_median': 'duration_median',
            'candleDuration_std': 'duration_std',
            'candleDuration_count': 'candle_count',
            'candleDuration_min': 'duration_min',
            'candleDuration_max': 'duration_max',
            'volume_mean': 'volume_mean',
            'volume_median': 'volume_median',
            'volume_std': 'volume_std',
            'volume_sum': 'volume_sum',
            'volume_min': 'volume_min',
            'volume_max': 'volume_max'
        }
        session_stats = session_stats.rename(columns=column_mapping)
        session_stats['session_date'] = session_stats['session_id'].map(session_dates.to_dict())
        session_stats['dataset'] = df_name

        # Calcul des métriques ATR, extrêmes et volumes above
        atr_stats = []
        extreme_stats = []
        volume_above_stats = []
        logger.info(
            f"🔄 Calcul des métriques ATR, contrats extrêmes et volumes above par tick pour {len(session_stats)} sessions...")

        for session_id in session_stats['session_id']:
            session_data = df_work[df_work['session_id'] == session_id].sort_values('timeStampOpeningConvertedtoDate')

            # ATR
            try:
                atr_series = calculate_atr_10_periods(session_data, period_window=period_atr_stat_session)
                atr_mean = atr_series.dropna().mean() if not atr_series.dropna().empty else np.nan
            except Exception as e:
                logger.warning(f"Erreur calcul ATR pour session {session_id}: {e}")
                atr_mean = np.nan
            atr_stats.append(atr_mean)

            # Contrats extrêmes
            try:
                extreme_metrics = calculate_extreme_contracts_metrics(session_data)
            except Exception as e:
                logger.warning(f"Erreur calcul contrats extrêmes pour session {session_id}: {e}")
                extreme_metrics = {
                    'extreme_sum_with_zeros': np.nan,
                    'extreme_sum_without_zeros': np.nan,
                    'extreme_count_nonzero': 0,
                    'extreme_ratio': np.nan
                }
            extreme_stats.append(extreme_metrics)

            # NOUVEAU: Volumes above par tick
            try:
                volume_above_metrics = calculate_volume_above_metrics(session_data,xtickReversalTickPrice=xtickReversalTickPrice)
            except Exception as e:
                logger.warning(f"Erreur calcul volumes above pour session {session_id}: {e}")
                volume_above_metrics = {
                    'volume_above_per_tick_mean': np.nan,
                    'volume_above_count': 0,
                    'volume_above_ratio': np.nan
                }
            volume_above_stats.append(volume_above_metrics)

        # Intégration des métriques
        session_stats['atr_mean'] = atr_stats
        session_stats['extreme_with_zeros'] = [s['extreme_sum_with_zeros'] for s in extreme_stats]
        session_stats['extreme_without_zeros'] = [s['extreme_sum_without_zeros'] for s in extreme_stats]
        session_stats['extreme_count_nonzero'] = [s['extreme_count_nonzero'] for s in extreme_stats]
        session_stats['extreme_ratio'] = [s['extreme_ratio'] for s in extreme_stats]

        # NOUVEAU: Intégration volumes above par tick
        session_stats['volume_above_per_tick_mean'] = [s['volume_above_per_tick_mean'] for s in volume_above_stats]
        session_stats['volume_above_count'] = [s['volume_above_count'] for s in volume_above_stats]
        session_stats['volume_above_ratio'] = [s['volume_above_ratio'] for s in volume_above_stats]

        logger.info(f"✅ Métriques calculées pour {df_name}: {len(session_stats)} sessions traitées")
        return session_stats

    except Exception as e:
        logger.error(f"❌ Erreur lors du calcul des métriques pour {df_name}: {e}")
        raise





def validate_dataframe_structure(df: pd.DataFrame, df_name: str, required_columns: list) -> Dict[str, Any]:
    """
    Validation complète de la structure d'un DataFrame

    Returns:
        Dict contenant les résultats de validation
    """
    validation_results = {
        'is_valid': True,
        'missing_columns': [],
        'data_types': {},
        'null_counts': {},
        'shape': df.shape,
        'warnings': []
    }

    # Vérifier les colonnes manquantes
    missing_columns = [col for col in required_columns if col not in df.columns]
    validation_results['missing_columns'] = missing_columns

    if missing_columns:
        validation_results['is_valid'] = False
        logger.error(f"Colonnes manquantes dans {df_name}: {missing_columns}")
        return validation_results

    # Analyser les types de données et valeurs nulles
    for col in required_columns:
        validation_results['data_types'][col] = str(df[col].dtype)
        validation_results['null_counts'][col] = df[col].isnull().sum()

        # Avertissements spécifiques par colonne
        null_pct = (df[col].isnull().sum() / len(df)) * 100
        if null_pct > 10:
            validation_results['warnings'].append(
                f"Colonne {col}: {null_pct:.1f}% de valeurs nulles"
            )

    # Vérifications spécifiques pour les colonnes critiques
    if 'session_id' in df.columns:
        unique_sessions = df['session_id'].nunique()
        total_rows = len(df)
        if unique_sessions < 2:
            validation_results['warnings'].append(
                f"Seulement {unique_sessions} session(s) unique(s) détectée(s)"
            )
        if total_rows / unique_sessions < 5:
            validation_results['warnings'].append(
                f"Peu de données par session: {total_rows / unique_sessions:.1f} lignes/session en moyenne"
            )

    return validation_results




def print_enhanced_summary_statistics_with_sessions(valid_results,groupe1,groupe2,xtickReversalTickPrice=None):
    """
    Affiche le résumé statistique enrichi avec analyse par sessions intraday et volumes above par tick
    """
    print("\n" + "=" * 80)
    print("📋 RÉSUMÉ STATISTIQUE AVEC SESSIONS INTRADAY (ATR + CONTRATS EXTRÊMES + VOLUMES ABOVE PAR TICK)")
    print("=" * 80)

    for df_name, data in valid_results.items():
        print(f"\n📊 {df_name.upper()}")
        print("-" * 60)

        # Données globales
        print(f"🌍 DONNÉES GLOBALES:")
        print(f"   📅 Période: {data['date_range'][0]} → {data['date_range'][1]}")
        print(f"   🎯 Nombre de sessions: {data['total_sessions']}")
        print(f"   ⏱️  Durée moyenne globale: {data['avg_duration_overall']:.2f}s")

        # Analyse par groupes de sessions
        if 'session_data_by_group' in data:
            print(f"\n🔍 ANALYSE PAR SESSIONS INTRADAY:")

            for group_label, session_indices in [("GROUPE 1", groupe1), ("GROUPE 2", groupe2)]:
                group_key = str(session_indices)
                if group_key in data['session_data_by_group'] and data['session_data_by_group'][group_key] is not None:
                    group_data = data['session_data_by_group'][group_key]

                    print(f"\n   📊 {group_label} (Sessions {session_indices}):")
                    print(f"      🎯 Sessions analysées: {group_data['total_sessions']}")
                    print(f"      ⏱️  Durée moyenne: {group_data['avg_duration_overall']:.2f}s")
                    print(f"      📈 Volume moyen/session: {group_data['volume_stats']['avg_volume_per_session']:.2f}")

                    # ATR pour ce groupe
                    if group_data['atr_stats']['sessions_with_atr'] > 0:
                        print(f"      📊 ATR moyen: {group_data['atr_stats']['atr_overall_mean']:.4f}")
                        print(
                            f"      🔗 Corrélation ATR-Durée: {group_data['atr_stats']['atr_duration_correlation']:.3f}")
                    else:
                        print(f"      ⚠️  Pas de données ATR valides")

                    # Contrats extrêmes pour ce groupe
                    if group_data['extreme_contracts_stats']['sessions_with_extreme_contracts'] > 0:
                        print(
                            f"      🎯 Contrats extrêmes (sans zéros): {group_data['extreme_contracts_stats']['extreme_without_zeros_mean']:.4f}")
                        print(
                            f"      📊 Ratio contrats extrêmes: {group_data['extreme_contracts_stats']['extreme_ratio_mean']:.3f}")
                    else:
                        print(f"      ⚠️  Pas de contrats extrêmes")

                    # NOUVEAU: Volumes above par tick pour ce groupe
                    if group_data['volume_above_per_tick_stats']['sessions_with_volume_above'] > 0:
                        print(
                            f"      📊 Volume above par tick moyen: {group_data['volume_above_per_tick_stats']['volume_above_per_tick_overall_mean']:.4f}")
                        print(
                            f"      📊 Ratio bougies volume above: {group_data['volume_above_per_tick_stats']['volume_above_ratio_mean']:.3f}")
                    else:
                        print(f"      ⚠️  Pas de volumes above par tick")

                else:
                    print(f"\n   📊 {group_label} (Sessions {session_indices}):")
                    print(f"      ⚠️  Aucune donnée disponible")
                    exit(32)

        # Comparaison entre groupes incluant volumes above par tick
        print(f"\n💡 COMPARAISON ENTRE SESSIONS:")
        global_duration = data['avg_duration_overall']
        global_volume_above = data['volume_above_per_tick_stats']['volume_above_per_tick_overall_mean'] if not pd.isna(
            data['volume_above_per_tick_stats']['volume_above_per_tick_overall_mean']) else None

        if 'session_data_by_group' in data:
            for group_label, session_indices in [("GROUPE 1", groupe1), ("GROUPE 2", groupe2)]:
                group_key = str(session_indices)
                if group_key in data['session_data_by_group'] and data['session_data_by_group'][group_key] is not None:
                    group_duration = data['session_data_by_group'][group_key]['avg_duration_overall']
                    diff_pct = ((group_duration - global_duration) / global_duration * 100)
                    trend = "📈 plus lent" if diff_pct > 5 else "📉 plus rapide" if diff_pct < -5 else "📊 similaire"
                    print(f"   {group_label}: {trend} que la moyenne globale ({diff_pct:+.1f}%)")

                    # Comparaison volumes above par tick
                    if global_volume_above is not None:
                        group_volume_above = data['session_data_by_group'][group_key]['volume_above_per_tick_stats'][
                            'volume_above_per_tick_overall_mean']
                        if not pd.isna(group_volume_above):
                            vol_diff_pct = ((group_volume_above - global_volume_above) / global_volume_above * 100)
                            vol_trend = "📈 plus élevé" if vol_diff_pct > 10 else "📉 plus faible" if vol_diff_pct < -10 else "📊 similaire"
                            print(f"      Volume above par tick: {vol_trend} ({vol_diff_pct:+.1f}%)")

        print("-" * 60)

    print("\n" + "=" * 80)
    print("📊 LÉGENDE DES SESSIONS INTRADAY")
    print("=" * 80)
    print(f"🌍 GLOBAL: Toutes les sessions confondues")
    print(f"🌅 GROUPE 1 (Sessions {groupe1}): Sessions de trading matinales/Asie")
    print(f"🌍 GROUPE 2 (Sessions {groupe2}): Sessions Europe/US étendues")
    print(f"📊 Volume Above par Tick: sc_volAbv/{xtickReversalTickPrice} pour sc_candleDir=-1")
    print("=" * 80)


# Mapping des colonnes selon la direction (identique au script Optuna)
COLUMN_MAPPING = {
    "short": {
        "volume_col": "sc_bidVolHigh_1",
        "imbalance_col": "bull_imbalance_high_0",  # AVEC sc_ comme dans Optuna
        "description": "Détection imbalances haussières sur les hauts (retournement baissier)"
    },
    "long": {
        "volume_col": "sc_askVolLow_1",
        "imbalance_col": "bear_imbalance_low_0",   # AVEC sc_ comme dans Optuna
        "description": "Détection imbalances baissières sur les bas (retournement haussier)"
    }
}


def add_ImBullWithPoc(df, df_feature, name, params, direction="short"):
    """
    Version générique supportant SHORT et LONG
    Compatible avec les paramètres générés par le script Optuna principal
    """
    if direction not in COLUMN_MAPPING:
        raise ValueError(f"Direction '{direction}' non supportée. Utilisez 'short' ou 'long'")

    config = COLUMN_MAPPING[direction]
    volume_col = config["volume_col"]
    imbalance_col = config["imbalance_col"]




    print(f"\n=== DEBUG {direction.upper()} - {name} ===")
    print(f"Volume column: {volume_col}")
    print(f"Imbalance column: {imbalance_col}")

    # ✅ Vérification des index
    print(f"df.index range: {df.index.min()} to {df.index.max()} (len: {len(df)})")
    print(f"df_feature.index range: {df_feature.index.min()} to {df_feature.index.max()} (len: {len(df_feature)})")
    print(f"Index alignés: {df.index.equals(df_feature.index)}")

    # ✅ Extraction des paramètres
    if "volume_1" in params:
        volume_1 = params["volume_1"]
        imbalance_1 = params["imbalance_1"]
        volume_2 = params["volume_2"]
        imbalance_2 = params["imbalance_2"]
        volume_3 = params["volume_3"]
        imbalance_3 = params["imbalance_3"]
    else:
        raise ValueError(f"Format de paramètres non reconnu. Paramètres reçus : {list(params.keys())}")

    # ✅ Vérification de l'existence des colonnes
    if volume_col not in df.columns:
        raise KeyError(f"Colonne '{volume_col}' non trouvée dans df")
    if imbalance_col not in df_feature.columns:
        raise KeyError(f"Colonne '{imbalance_col}' non trouvée dans df_feature")

    # ✅ Statistiques des données
    vol_data = df[volume_col]
    imb_data = df_feature[imbalance_col]

    print(f"\nStatistiques {volume_col}:")
    print(f"  Min: {vol_data.min():.2f}, Max: {vol_data.max():.2f}, NaN: {vol_data.isna().sum()}")
    print(f"Statistiques {imbalance_col}:")
    print(f"  Min: {imb_data.min():.2f}, Max: {imb_data.max():.2f}, NaN: {imb_data.isna().sum()}")

    # ✅ Test des conditions une par une
    print(f"\nTest des conditions:")
    print(f"Paramètres: vol1={volume_1}, imb1={imbalance_1:.2f}")

    mask1_vol = vol_data > volume_1
    mask1_imb = imb_data > imbalance_1
    mask1 = mask1_vol & mask1_imb

    print(f"  Condition 1: vol > {volume_1}: {mask1_vol.sum()} lignes")
    print(f"  Condition 1: imb > {imbalance_1:.2f}: {mask1_imb.sum()} lignes")
    print(f"  Condition 1 combinée: {mask1.sum()} lignes")

    mask2 = (vol_data > volume_2) & (imb_data > imbalance_2)
    mask3 = (vol_data > volume_3) & (imb_data > imbalance_3)

    print(f"  Condition 2: {mask2.sum()} lignes")
    print(f"  Condition 3: {mask3.sum()} lignes")
    print(f"  Union (mask1|mask2|mask3): {(mask1 | mask2 | mask3).sum()} lignes")

    # ✅ POC filtering
    if "pos_poc_min" in params and "pos_poc_max" in params:
        poc_data = df_feature["diffPriceClosePoc_0_0"]
        poc_mask = (poc_data >= params["pos_poc_min"]) & (poc_data <= params["pos_poc_max"])
        print(f"  POC filter [{params['pos_poc_min']}, {params['pos_poc_max']}]: {poc_mask.sum()} lignes")
    else:
        poc_mask = pd.Series(True, index=df.index)
        print("  POC filter: TOUS (pas de filtre)")


    # Combine all conditions
    final_mask = poc_mask & (mask1 | mask2 | mask3)
    print(f"  FINAL RESULT: {final_mask.sum()} lignes")

    # Add the new feature column to df_feature
    df_feature[name] = final_mask.astype(int)

    return df_feature
def add_imbBullLightPoc_Atr_HighLow(df, df_feature, name, params):
    """
    Add a binary feature column to df_feature based on ATR thresholds and diff_high_atr values,
    using the existing is_imBullWithPoc_light feature.

    Parameters:
    -----------
    df : pd.DataFrame
        The original dataframe with all the raw data
    df_feature : pd.DataFrame
        The dataframe where the new feature will be added
    name : str
        The name of the new feature column to add
    params : dict
        Dictionary containing the parameters for the conditions:
        - 'atr_threshold_1', 'atr_threshold_2', 'atr_threshold_3' for ATR thresholds
        - 'diff_high_atr_1', 'diff_high_atr_2', 'diff_high_atr_3', 'diff_high_atr_4' for diffHighPrice conditions
        - 'atr_window' for ATR calculation window (default: 12)

    Returns:
    --------
    pd.DataFrame
        The updated df_feature dataframe with the new column
    """
    # Récupérer la fenêtre ATR depuis les paramètres ou utiliser la valeur par défaut
    atr_window = params.get("atr_window", 12)

    # CORRECTION: Verify if 'atr_recalc' column exists in df or recalculate it with the specified window
    need_recalc = ('atr_recalc' not in df.columns or
                   'atr_window' not in df.columns or
                   (df['atr_window'].iloc[0] if len(df) > 0 else None) != atr_window)

    if need_recalc:
        df['atr_recalc'] = calculate_atr(df, window=atr_window)
        df['atr_window'] = atr_window  # Stocker la fenêtre utilisée pour référence

    # Récupérer les seuils d'ATR depuis les paramètres
    threshold_1 = params["atr_threshold_1"]
    threshold_2 = params["atr_threshold_2"]
    threshold_3 = params["atr_threshold_3"]

    # Créer les masques pour chaque plage d'ATR
    mask_atr_1 = df["atr_recalc"] < threshold_1
    mask_atr_2 = (df["atr_recalc"] >= threshold_1) & (df["atr_recalc"] < threshold_2)
    mask_atr_3 = (df["atr_recalc"] >= threshold_2) & (df["atr_recalc"] < threshold_3)
    mask_atr_4 = df["atr_recalc"] >= threshold_3

    # CORRECTION: Récupérer les valeurs selon le type (High ou Low)
    if "high" in name.lower():
        # Pour AtrHigh, utiliser diff_high_atr_X
        diff_atr_1 = params["diff_high_atr_1"]
        diff_atr_2 = params["diff_high_atr_2"]
        diff_atr_3 = params["diff_high_atr_3"]
        diff_atr_4 = params["diff_high_atr_4"]
    elif "low" in name.lower():
        # Pour AtrLow, utiliser diff_low_atr_X
        diff_atr_1 = params["diff_low_atr_1"]
        diff_atr_2 = params["diff_low_atr_2"]
        diff_atr_3 = params["diff_low_atr_3"]
        diff_atr_4 = params["diff_low_atr_4"]
    else:
        # Fallback sur diff_high_atr_X
        diff_atr_1 = params["diff_high_atr_1"]
        diff_atr_2 = params["diff_high_atr_2"]
        diff_atr_3 = params["diff_high_atr_3"]
        diff_atr_4 = params["diff_high_atr_4"]

    # Créer les masques pour diffHighPrice_0_1 pour chaque plage d'ATR
    mask_diff_1 = mask_atr_1 & (df_feature["diffHighPrice_0_1"] > diff_atr_1)
    mask_diff_2 = mask_atr_2 & (df_feature["diffHighPrice_0_1"] > diff_atr_2)
    mask_diff_3 = mask_atr_3 & (df_feature["diffHighPrice_0_1"] > diff_atr_3)
    mask_diff_4 = mask_atr_4 & (df_feature["diffHighPrice_0_1"] > diff_atr_4)

    # Combine all diffHighPrice masks
    diff_mask = mask_diff_1 | mask_diff_2 | mask_diff_3 | mask_diff_4

    # CORRECTION: Déterminer la colonne de référence selon le nom
    if "bull" in name.lower() or "short" in name.lower():
        ref_column = "is_imBullWithPoc_light_short"
    elif "bear" in name.lower() or "long" in name.lower():
        ref_column = "is_imBearWithPoc_light_long"
    else:
        # Fallback sur l'ancienne logique
        ref_column = "is_imBullWithPoc_light_short"

    # Vérifier si la colonne de référence existe
    if ref_column not in df_feature.columns:
        raise ValueError(f"La colonne '{ref_column}' est absente du dataframe df_feature.")

    # Utiliser la colonne de référence appropriée comme filtre
    imbull_mask = df_feature[ref_column].fillna(0).astype(bool)

    # Combine all conditions: diff_mask AND imbull_mask
    final_mask = diff_mask & imbull_mask

    # Add the new feature column to df_feature
    df_feature[name] = final_mask.astype(int)

    # # Count valid samples (where class_binaire is 0 or 1)
    # sample_count = df["class_binaire"].isin([0, 1]).sum()
    #
    # # Log some statistics about the new feature
    # signal_count = df_feature[name].sum()
    #
    # # print(
    #     f"Added feature '{name}' (ATR window: {atr_window}): {signal_count} signals ({signal_count / sample_count:.2%} of valid samples)")
    #
    # # If we have class_binaire in the dataframe, calculate win rate
    # if "class_binaire" in df.columns:
    #     # Only consider rows where the signal is 1
    #     signal_rows = df[final_mask]
    #     if len(signal_rows) > 0:
    #         wins = (signal_rows["class_binaire"] == 1).sum()
    #         losses = (signal_rows["class_binaire"] == 0).sum()
    #         win_rate = round(wins / (wins + losses), 2)
    #         print(f"Win rate for '{name}': {win_rate:.2%} (✓{wins} ✗{losses}, Total={wins + losses} trades)")

    # Print detailed statistics for each ATR segment if verbose flag is set
    # if params.get("verbose", False):
    #     # Filtre pour les données de trading
    #     trading_mask = df["class_binaire"].isin([0, 1])
    #
    #     # Pour chaque segment ATR, calculer des statistiques
    #     for i, (mask_atr, mask_diff, atr_label, diff_value) in enumerate([
    #         (mask_atr_1, mask_diff_1, f"ATR < {threshold_1:.1f}", diff_high_atr_1),
    #         (mask_atr_2, mask_diff_2, f"{threshold_1:.1f} ≤ ATR < {threshold_2:.1f}", diff_high_atr_2),
    #         (mask_atr_3, mask_diff_3, f"{threshold_2:.1f} ≤ ATR < {threshold_3:.1f}", diff_high_atr_3),
    #         (mask_atr_4, mask_diff_4, f"ATR ≥ {threshold_3:.1f}", diff_high_atr_4)
    #     ]):
    #         # Pour chaque segment, créer un masque combiné avec toutes les conditions
    #         segment_mask = mask_diff & imbull_mask & trading_mask
    #         segment_count = segment_mask.sum()
    #
    #         # Calculer les statistiques du segment (même pour les segments sans trades)
    #         segment_rows = df[segment_mask]
    #         segment_wins = (segment_rows["class_binaire"] == 1).sum() if len(segment_rows) > 0 else 0
    #         segment_losses = (segment_rows["class_binaire"] == 0).sum() if len(segment_rows) > 0 else 0
    #         segment_total = segment_wins + segment_losses
    #         segment_wr = segment_wins / segment_total if segment_total > 0 else 0
    #
    #         # Afficher tous les segments, même ceux sans trades
    #         wr_display = f"WR={segment_wr:.2%}" if segment_total > 0 else "WR=N/A"
    #         print(f"  Segment {i + 1} ({atr_label}, diffHigh > {diff_value:.2f}): "
    #               f"{wr_display} | "
    #               f"Trades={segment_total}" +
    #               (f" (✓{segment_wins} ✗{segment_losses})" if segment_total > 0 else "") +
    #               f" | Échantillons dans le segment: {mask_atr.sum()} ({mask_atr.sum() / len(df):.1%})")

    return df_feature


def compute_consecutive_trend_feature(df, features_df, target_col, n=2, trend_type='up', output_col='trend_feature'):
    """
    Lit une colonne source dans df et ajoute une colonne binaire dans features_df indiquant
    si une tendance haussière ou baissière sur N bougies consécutives est détectée.

    Parameters:
    - df : DataFrame source (avec les colonnes comme vix_vixLast)
    - features_df : DataFrame cible à enrichir (déjà existant)
    - target_col : colonne dans df à analyser
    - n : nombre de bougies à examiner
    - trend_type : 'up' (hausse) ou 'down' (baisse)
    - output_col : nom de la colonne à ajouter à features_df

    Returns:
    - features_df modifié avec la nouvelle colonne
    """
    cond = pd.Series(True, index=df.index)
    for i in range(n):
        if trend_type == 'up':
            cond &= df[target_col].shift(i) >= df[target_col].shift(i + 1)
        elif trend_type == 'down':
            cond &= df[target_col].shift(i) <= df[target_col].shift(i + 1)
        else:
            raise ValueError("trend_type must be 'up' or 'down'")
        cond &= ~df[target_col].shift(i).isna()
        cond &= ~df[target_col].shift(i + 1).isna()

    features_df[output_col] = cond.astype(int)
    return features_df


# ────────────────────────────────
# 2) Fonction Numba
# ────────────────────────────────
@njit
def compute_dates_numba(session, timestamp):
    n = session.size
    out = np.full(n, SENTINEL, dtype=np.int64)

    i = 0
    while i < n:
        # On attend impérativement un «10»
        if session[i] != 10:
            raise ValueError(f"sc_sessionStartEnd=10 attendu à l’index {i}, trouvé {session[i]}")

        # +1 jour puis normalisation au début de journée (UTC)
        base_ns = (timestamp[i] + SECONDS_IN_DAY) * NANO_PER_SECOND
        day_ns  = (base_ns // NANO_PER_DAY) * NANO_PER_DAY

        # Chercher le 20 correspondant
        j = i
        while j < n and session[j] != 20:
            j += 1

        if j == n:                                   # Aucun 20 trouvé
            raise ValueError(f"Aucun sc_sessionStartEnd=20 après l’index {i}")
        # Remplir i … j  (le 20 inclus)
        for k in range(i, j + 1):
            out[k] = day_ns

        i = j + 1                                   # on continue après le 20

    return out

@jit(nopython=True)
def calculate_z_scores(distances, z_window):
    """Version numba optimisée pour calculer les z_scores."""
    n = len(distances)
    z_scores = np.zeros(n)

    # Pour chaque position valide pour calculer un z-score
    for i in range(z_window - 1, n):
        window = distances[i - z_window + 1:i + 1]
        mean = np.mean(window)
        std = np.std(window)
        if std > 0:
            z_scores[i] = (distances[i] - mean) / std

    return z_scores
from ta.volatility import AverageTrueRange
import warnings
# Supprimer les warnings Numba pour un code plus propre
warnings.filterwarnings('ignore', category=UserWarning)


@jit(nopython=True, cache=True, fastmath=True)
def _calculate_true_range_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """
    Calcule le True Range avec Numba
    Très rapide pour les gros volumes de données
    """
    n = len(high)
    if n < 2:
        return np.full(1, np.nan, dtype=np.float64)  # Type explicite

    tr = np.empty(n - 1, dtype=np.float64)

    for i in range(1, n):
        hl = high[i] - low[i]  # High - Low
        hc = abs(high[i] - close[i - 1])  # |High - Close_prev|
        lc = abs(low[i] - close[i - 1])  # |Low - Close_prev|

        # True Range = max des trois
        tr[i - 1] = max(hl, max(hc, lc))

    return tr


@jit(nopython=True, cache=True, fastmath=True)
def _calculate_sma_numba(data: np.ndarray, period: int) -> np.ndarray:
    """
    Calcule une moyenne mobile simple avec Numba
    Utilise une fenêtre glissante pour une efficacité maximale
    """
    n = len(data)
    if n < period:
        return np.full(n, np.nan, dtype=np.float64)

    result = np.full(n, np.nan, dtype=np.float64)

    # Calcul de la première moyenne
    window_sum = 0.0
    for i in range(period):
        window_sum += data[i]
    result[period - 1] = window_sum / period

    # Fenêtre glissante pour le reste
    for i in range(period, n):
        window_sum = window_sum - data[i - period] + data[i]
        result[i] = window_sum / period

    return result


@jit(nopython=True, cache=True, fastmath=True)
def _calculate_wilder_sma_numba(data: np.ndarray, period: int) -> np.ndarray:
    """
    Calcule l'ATR selon la méthode de Wilder (EMA avec alpha = 1/period)
    Équivalent à ta-lib AverageTrueRange avec méthode Wilder
    """
    n = len(data)
    if n < period:
        return np.full(n, np.nan, dtype=np.float64)

    result = np.full(n, np.nan, dtype=np.float64)
    alpha = 1.0 / period  # Facteur de lissage de Wilder

    # Première valeur = moyenne simple des 'period' premiers éléments
    first_sum = 0.0
    for i in range(period):
        first_sum += data[i]
    result[period - 1] = first_sum / period

    # Ensuite, EMA avec alpha = 1/period (méthode Wilder)
    for i in range(period, n):
        result[i] = alpha * data[i] + (1.0 - alpha) * result[i - 1]

    return result


@jit(nopython=True, cache=True, fastmath=True)
def _atr_sma_numba_core(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """
    Fonction core ATR SMA avec Numba
    Combine True Range + SMA de manière optimisée
    """
    n = len(high)
    result = np.full(n, np.nan, dtype=np.float64)

    if n < period + 1:
        return result

    # Calcul True Range
    tr = _calculate_true_range_numba(high, low, close)

    # Calcul SMA du True Range
    tr_sma = _calculate_sma_numba(tr, period)

    # Alignement avec l'index original (décalage de 1 pour True Range)
    result[period:] = tr_sma[period - 1:]

    return result


@jit(nopython=True, cache=True, fastmath=True)
def _atr_wilder_numba_core(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """
    Fonction core ATR Wilder avec Numba
    Combine True Range + Wilder smoothing (EMA avec alpha = 1/period)
    """
    n = len(high)
    result = np.full(n, np.nan, dtype=np.float64)

    if n < period + 1:
        return result

    # Calcul True Range
    tr = _calculate_true_range_numba(high, low, close)

    # Calcul Wilder smoothing du True Range
    tr_wilder = _calculate_wilder_sma_numba(tr, period)

    # Alignement avec l'index original (décalage de 1 pour True Range)
    result[period:] = tr_wilder[period - 1:]

    return result


def _atr_sma(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """
    Average True Range en moyenne arithmétique (SMA) optimisé avec Numba
    Compatible avec l'implémentation C++/Sierra Chart

    Performance: ~10-15x plus rapide que la version pandas originale
    """
    return _atr_sma_numba_core(high, low, close, period)


def _atr_wilder(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """
    Average True Range avec méthode de Wilder (EMA avec alpha = 1/period)
    Équivalent à ta-lib AverageTrueRange - optimisé avec Numba

    Performance: ~10-15x plus rapide que la version ta-lib
    """
    return _atr_wilder_numba_core(high, low, close, period)
    import numpy as np


import pandas as pd
from numba import jit, types
from numba.typed import Dict
import warnings

# Supprimer les warnings Numba pour un code plus propre
warnings.filterwarnings('ignore', category=UserWarning)


# ════════════════════════════════════════════════════════════════
# FONCTIONS NUMBA POUR ATR
# ════════════════════════════════════════════════════════════════

@jit(nopython=True, cache=True, fastmath=True)
def _calculate_true_range_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """
    Calcule le True Range avec Numba
    Très rapide pour les gros volumes de données
    """
    n = len(high)
    if n < 2:
        return np.full(1, np.nan, dtype=np.float64)  # Type explicite

    tr = np.empty(n - 1, dtype=np.float64)

    for i in range(1, n):
        hl = high[i] - low[i]  # High - Low
        hc = abs(high[i] - close[i - 1])  # |High - Close_prev|
        lc = abs(low[i] - close[i - 1])  # |Low - Close_prev|

        # True Range = max des trois
        tr[i - 1] = max(hl, max(hc, lc))

    return tr


@jit(nopython=True, cache=True, fastmath=True)
def _calculate_sma_numba(data: np.ndarray, period: int) -> np.ndarray:
    """
    Calcule une moyenne mobile simple avec Numba
    Utilise une fenêtre glissante pour une efficacité maximale
    """
    n = len(data)
    if n < period:
        return np.full(n, np.nan)

    result = np.full(n, np.nan)

    # Calcul de la première moyenne
    window_sum = 0.0
    for i in range(period):
        window_sum += data[i]
    result[period - 1] = window_sum / period

    # Fenêtre glissante pour le reste
    for i in range(period, n):
        window_sum = window_sum - data[i - period] + data[i]
        result[i] = window_sum / period

    return result


@jit(nopython=True, cache=True, fastmath=True)
def _atr_sma_numba_core(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """
    Fonction core ATR SMA avec Numba
    Combine True Range + SMA de manière optimisée
    """
    n = len(high)
    result = np.full(n, np.nan)

    if n < period + 1:
        return result

    # Calcul True Range
    tr = _calculate_true_range_numba(high, low, close)

    # Calcul SMA du True Range
    tr_sma = _calculate_sma_numba(tr, period)

    # Alignement avec l'index original (décalage de 1 pour True Range)
    result[period:] = tr_sma[period - 1:]

    return result


def _atr_sma(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """
    Average True Range en moyenne arithmétique (SMA) optimisé avec Numba
    Compatible avec l'implémentation C++/Sierra Chart

    Performance: ~10-15x plus rapide que la version pandas originale
    """
    return _atr_sma_numba_core(high, low, close, period)


# ════════════════════════════════════════════════════════════════
# OPTIMISATIONS NUMBA SUPPLÉMENTAIRES
# ════════════════════════════════════════════════════════════════

@jit(nopython=True, cache=True, fastmath=True)
def _calculate_zscore_numba(data: np.ndarray, window: int) -> np.ndarray:
    """
    Calcul Z-score optimisé avec Numba
    Plus rapide que pandas.rolling().apply()
    """
    n = len(data)
    result = np.full(n, np.nan)

    if n < window:
        return result

    for i in range(window - 1, n):
        # Fenêtre actuelle
        start_idx = i - window + 1
        window_data = data[start_idx:i + 1]

        # Calcul statistiques
        mean_val = np.mean(window_data)
        std_val = np.std(window_data)

        # Z-score
        if std_val > 0:
            result[i] = (data[i] - mean_val) / std_val
        else:
            result[i] = np.nan

    return result


@jit(nopython=True, cache=True, fastmath=True)
def _calculate_sma_simple_numba(data: np.ndarray, period: int) -> np.ndarray:
    """
    Version simplifiée de SMA pour usage général
    """
    n = len(data)
    result = np.full(n, np.nan)

    if n < period:
        return result

    for i in range(period - 1, n):
        start_idx = i - period + 1
        result[i] = np.mean(data[start_idx:i + 1])

    return result


@jit(nopython=True, cache=True, fastmath=True)
def _calculate_ema_numba(data: np.ndarray, span: int) -> np.ndarray:
    """
    Calcul EMA optimisé avec Numba
    Remplace pandas.ewm()
    """
    n = len(data)
    if n == 0:
        return np.empty(0, dtype=np.float64)  # Type explicite pour Numba

    result = np.full(n, np.nan, dtype=np.float64)
    alpha = 2.0 / (span + 1.0)

    # Premier point
    result[0] = data[0]

    # Calcul EMA
    for i in range(1, n):
        result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]

    return result


# ════════════════════════════════════════════════════════════════
# VERSION OPTIMISÉE DE LA FONCTION PRINCIPALE
# ════════════════════════════════════════════════════════════════

def vwap_reversal_pro_optimized(
        df: pd.DataFrame,
        *,
        lookback: int, momentum: int, z_window: int,
        atr_period: int, atr_mult: float,
        ema_filter: int,
        vol_lookback: int, vol_ratio_min: float,
        direction: str = 'short',
        atr_ma: str = 'wilder'
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Version optimisée de vwap_reversal_pro avec Numba
    Améliore les performances des calculs intensifs
    """
    # ───── Validation des entrées
    if direction not in {'short', 'long'}:
        raise ValueError("direction doit être 'short' ou 'long'")
    if atr_ma not in {'wilder', 'sma'}:
        raise ValueError("atr_ma doit être 'wilder' ou 'sma'")

    # ───── Pré-allocation des buffers
    result_idx = df.index
    z_dist = np.full(len(df), np.nan)
    speed = np.full(len(df), np.nan)
    mom = np.full(len(df), np.nan)
    atr_values = np.full(len(df), np.nan)
    dyn_th = np.full(len(df), np.nan)
    ema_values = np.full(len(df), np.nan, dtype=float)
    trend_ok = np.zeros(len(df), dtype=bool)
    vol_ok = np.zeros(len(df), dtype=bool)
    enough_data = np.zeros(len(df), dtype=bool)

    # Distance Prix - VWAP pré-calculée
    distance = df['sc_close'].to_numpy() - df['sc_VWAP'].to_numpy()

    # ───── Traitement session par session
    for session_id, loc_idx in df.groupby('session_id').groups.items():
        idx_arr = np.asarray(loc_idx, dtype=int)
        if idx_arr.size == 0:
            continue

        # Raccourcis locaux
        sess_dist = distance[idx_arr]
        sess_close = df.loc[idx_arr, 'sc_close'].to_numpy()
        sess_high = df.loc[idx_arr, 'sc_high'].to_numpy()
        sess_low = df.loc[idx_arr, 'sc_low'].to_numpy()
        sess_vol = df.loc[idx_arr, 'sc_volume'].to_numpy()

        # → Suffisance de données
        min_req = max(z_window, lookback + momentum, atr_period, vol_lookback) - 1
        if idx_arr.size > min_req:
            enough_data[idx_arr[min_req:]] = True

        # 1️⃣ Z-SCORE avec Numba
        if idx_arr.size >= z_window:
            z_scores = _calculate_zscore_numba(sess_dist, z_window)
            z_dist[idx_arr] = z_scores

        # 2️⃣ SPEED
        if idx_arr.size > lookback:
            speed[idx_arr[lookback:]] = z_dist[idx_arr[lookback:]] - z_dist[idx_arr[:-lookback]]

        # 3️⃣ MOMENTUM - LOGIQUE C++ (inchangée)
        if idx_arr.size > lookback + momentum:
            for i in range(lookback + momentum, idx_arr.size):
                global_idx = idx_arr[i]

                # Z-score actuel (déjà calculé)
                z_t = z_dist[global_idx]

                # Z-score t-lookback (déjà calculé)
                z_lb_idx = idx_arr[i - lookback]
                z_lb = z_dist[z_lb_idx]

                # Speed actuel
                speed_t = z_t - z_lb

                # Z-score t-lookback-momentum (recalculé comme en C++)
                z_lb_mom_idx = idx_arr[i - lookback - momentum]
                z_lb_mom = z_dist[z_lb_mom_idx]

                # Speed précédent (recalculé)
                speed_prev = z_lb - z_lb_mom

                # Momentum final
                mom_t = speed_t - speed_prev

                # Stockage
                mom[global_idx] = mom_t
                speed[global_idx] = speed_t

        # 4️⃣ ATR avec Numba (sma ET wilder optimisés)
        if idx_arr.size >= atr_period:
            if atr_ma == 'wilder':
                atr_vals = _atr_wilder(sess_high, sess_low, sess_close, atr_period)
            else:  # 'sma' - Version Numba optimisée
                atr_vals = _atr_sma(sess_high, sess_low, sess_close, atr_period)

            atr_values[idx_arr] = atr_vals
            dyn_th[idx_arr] = atr_vals * atr_mult

        # 5️⃣ EMA avec Numba
        if len(sess_close) > 0:
            ema_sess = _calculate_ema_numba(sess_close, ema_filter)
            ema_values[idx_arr] = ema_sess

            if ema_sess.size > 1:
                ema_diff = np.diff(ema_sess)
                if direction == 'short':
                    trend_ok[idx_arr[1:]] = ema_diff < 0
                else:
                    trend_ok[idx_arr[1:]] = ema_diff > 0

        # 6️⃣ Check volume avec Numba
        if idx_arr.size >= vol_lookback:
            vol_ma = _calculate_sma_simple_numba(sess_vol, vol_lookback)
            valid = vol_ma > 0
            vol_ratio = np.full_like(sess_vol, np.nan, dtype=float)
            vol_ratio[valid] = sess_vol[valid] / vol_ma[valid]
            vol_ok[idx_arr] = vol_ratio > vol_ratio_min
        else:
            vol_ok[idx_arr] = True

    # ───── Signal final
    if direction == 'short':
        signal_bool = (
                (z_dist > 0) &
                (speed > 0) &
                ((mom < -dyn_th) | trend_ok) &
                vol_ok &
                ~np.isnan(atr_values)
        )
    else:
        signal_bool = (
                (z_dist < 0) &
                (speed < 0) &
                ((mom > dyn_th) | trend_ok) &
                vol_ok &
                ~np.isnan(atr_values)
        )

    signal = pd.Series(signal_bool, index=result_idx).astype('int8')
    status_df = pd.DataFrame({'enough_data': enough_data}, index=result_idx)

    return signal, status_df


def metrics_vwap_premmium(df, mask):
    sub = df[mask.values]
    return ((sub["class_binaire"] == 1).mean() if not sub.empty else 0.0,
            len(sub) / len(df))
# 2️⃣ Définir la fonction de prédiction
def predict_g2_from_g1(regime_g1,transition_matrix_final):
    """
    Prédit le régime G2 le plus probable à partir de G1
    """
    if regime_g1 in transition_matrix_final.index:
        # Prendre le régime G2 avec la plus haute probabilité
        predicted_regime = transition_matrix_final.loc[regime_g1].idxmax()
        probability = transition_matrix_final.loc[regime_g1].max()
        return predicted_regime, probability
    else:
        # Valeur par défaut si G1 inconnu
        return None, None

# ──────────────────────────────────────────────────────────────────
# WRAPPER : injection dans features_df
# ──────────────────────────────────────────────────────────────────
def add_vwap_reversal_pro(
        *,
        df_full: pd.DataFrame,
        features_df: pd.DataFrame,
        lookback: int, momentum: int, z_window: int,
        atr_period: int, atr_mult: float,
        ema_filter: int,
        vol_lookback: int, vol_ratio_min: float,
        direction: str = 'short',
        atr_ma: str = 'wilder',          # <── NEW (propagé)
        neutral_value: float | int = np.nan         # ✅ toujours des types
) -> pd.DataFrame:
    """
    Ajoute la colonne `is_vwap_reversal_pro_{direction}` à `features_df`.
    """
    sig_full, status = vwap_reversal_pro_optimized(
        df_full,
        lookback=lookback, momentum=momentum, z_window=z_window,
        atr_period=atr_period, atr_mult=atr_mult,
        ema_filter=ema_filter,
        vol_lookback=vol_lookback, vol_ratio_min=vol_ratio_min,
        direction=direction,
        atr_ma=atr_ma
    )

    sig_clean = sig_full.where(status.enough_data, neutral_value)

    if not df_full.index.equals(features_df.index):
        sig_clean = sig_clean.reindex(features_df.index)

    features_df[f'is_vwap_reversal_pro_{direction}'] = sig_clean.astype('float32')
    return features_df



# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _iqr(series: pd.Series) -> float:
    """Inter‑quartile range (P75‑P25). Renvoie NaN si < 2 valeurs valides."""
    s = series.dropna().to_numpy()
    return np.nan if s.size < 2 else np.percentile(s, 75) - np.percentile(s, 25)


def create_dataframe_with_group_indicators(
        df: pd.DataFrame,
        groupe1_sessions: list,
        groupe2_sessions: list,
        xtickReversalTickPrice: float | None = None,
        period_atr_stat_session: int | None = None,
):
    """
    Ajoute les colonnes d'indicateurs (percentiles + métriques win/lose) calculés *intra‑session*.
    """

    logger.info("🚀 Création du DataFrame avec indicateurs de groupe")
    logger.info(f"   GROUPE1: sessions {groupe1_sessions}")
    logger.info(f"   GROUPE2: sessions {groupe2_sessions}")

    # ── Pré-traitement minimal ────────────────────────────────
    df_enriched = df.copy()

    if df_enriched.columns.duplicated().any():
        dupes = df_enriched.columns[df_enriched.columns.duplicated()].tolist()
        logger.warning(f"🔄 Colonnes dupliquées détectées → {dupes} ; on garde la 1re occurrence")
        df_enriched = df_enriched.loc[:, ~df_enriched.columns.duplicated(keep="first")]

    if isinstance(df_enriched["session_id"], pd.DataFrame):
        df_enriched["session_id"] = df_enriched["session_id"].iloc[:, 0]

    # ── Colonnes à créer ──────────────────────────────────────
    group_columns_g1 = [
        "volume_p25_g1", "volume_p50_g1", "volume_p75_g1",
        "atr_p25_g1", "atr_p50_g1", "atr_p75_g1",
        "duration_p25_g1", "duration_p50_g1", "duration_p75_g1",
        "vol_above_p25_g1", "vol_above_p50_g1", "vol_above_p75_g1",
        "volMeanPerTick_p25_g1", "volMeanPerTick_p50_g1", "volMeanPerTick_p75_g1",
        "extreme_ratio_g1", "volume_spread_g1", "volume_above_spread_g1",
        "volMeanPerTick_spread_g1", "atr_spread_g1", "duration_spread_g1",
        # Nouvelles métriques win/lose
        "meanVol_perTick_over1_g1", "meanVol_perTick_over2_g1", "meanVol_perTick_over5_g1", "meanVol_perTick_over12_g1",
        "meanVol_perTick_over20_g1", "meanVol_perTick_over30_g1",
        "volCandleMeanOver5Ratio_g1", "volCandleMeanOver12Ratio_g1", "volCandleMeanOver20Ratio_g1",
        "volCandleMeanOver30Ratio_g1"
    ]

    group_columns_g2 = [
        "volume_p25_g2", "volume_p50_g2", "volume_p75_g2",
        "atr_p25_g2", "atr_p50_g2", "atr_p75_g2",
        "duration_p25_g2", "duration_p50_g2", "duration_p75_g2",
        "vol_above_p25_g2", "vol_above_p50_g2", "vol_above_p75_g2",
        "volMeanPerTick_p25_g2", "volMeanPerTick_p50_g2", "volMeanPerTick_p75_g2",
        "extreme_ratio_g2", "volume_spread_g2", "volume_above_spread_g2",
        "volMeanPerTick_spread_g2", "atr_spread_g2", "duration_spread_g2",
        # Nouvelles métriques win/lose
        "meanVol_perTick_over1_g2", "meanVol_perTick_over2_g2", "meanVol_perTick_over5_g2", "meanVol_perTick_over12_g2",
        "meanVol_perTick_over20_g2", "meanVol_perTick_over30_g2",
        "volCandleMeanOver5Ratio_g2", "volCandleMeanOver12Ratio_g2", "volCandleMeanOver20Ratio_g2",
        "volCandleMeanOver30Ratio_g2"
    ]

    all_group_columns = group_columns_g1 + group_columns_g2
    df_enriched[all_group_columns] = np.nan

    # ── Fonctions utilitaires ──────────────────────────────────
    def _calculate_over_thresholds(volMeanPerTick_series, thresholds=[1, 2, 5, 12, 20, 30]):
        """Calcule les moyennes pour volumes au-dessus des seuils"""
        results = {}
        for threshold in thresholds:
            over_threshold = volMeanPerTick_series[volMeanPerTick_series > threshold]
            results[f"meanVol_perTick_over{threshold}"] = over_threshold.mean() if len(over_threshold) > 0 else np.nan
        return results

    def _calculate_ratio_metrics(volume_series, volMeanPerTick_series, thresholds=[5, 12, 20, 30]):
        """Calcule les ratios VolCandleMeanOverXRatio"""
        results = {}
        for threshold in thresholds:
            over_threshold_mask = volMeanPerTick_series > threshold
            if over_threshold_mask.any():
                vol_over_threshold = volume_series[over_threshold_mask]
                candle_mean = vol_over_threshold.mean()
                overall_mean = volume_series.mean()
                results[
                    f"VolCandleMeanOver{threshold}Ratio"] = candle_mean / overall_mean if overall_mean > 0 else np.nan
            else:
                results[f"VolCandleMeanOver{threshold}Ratio"] = np.nan
        return results

    # ── Boucle sessions ───────────────────────────────────────
    total_sessions = df_enriched["session_id"].nunique()
    logger.info(f"📊 Traitement de {total_sessions} sessions…")

    for sid in df_enriched["session_id"].unique():
        session_mask = df_enriched["session_id"] == sid
        session_data = df_enriched[session_mask]

        # --- GROUPE 1 ----------------------------------------------------
        g1_data = session_data[session_data["deltaCustomSessionIndex"].isin(groupe1_sessions)]
        if not g1_data.empty:
            try:
                extreme_metrics_g1 = calculate_extreme_contracts_metrics(g1_data)
                atr_series_g1 = calculate_atr_10_periods(g1_data, period_window=period_atr_stat_session)
                vol_above_g1 = g1_data[
                                   "sc_volAbv"] / xtickReversalTickPrice if xtickReversalTickPrice and "sc_volAbv" in g1_data.columns else np.nan
                volMeanPerTick_g1 = g1_data["sc_volume"] / g1_data[
                    "candleSizeTicks"] if "candleSizeTicks" in g1_data.columns else np.nan

                # Calculs existants
                g1_indicators = {
                    "volume_p25_g1": g1_data["sc_volume"].quantile(0.25),
                    "volume_p50_g1": g1_data["sc_volume"].median(),
                    "volume_p75_g1": g1_data["sc_volume"].quantile(0.75),
                    "atr_p25_g1": atr_series_g1.quantile(0.25),
                    "atr_p50_g1": atr_series_g1.median(),
                    "atr_p75_g1": atr_series_g1.quantile(0.75),
                    "duration_p25_g1": g1_data["candleDuration"].quantile(0.25),
                    "duration_p50_g1": g1_data["candleDuration"].median(),
                    "duration_p75_g1": g1_data["candleDuration"].quantile(0.75),
                    "vol_above_p25_g1": vol_above_g1.quantile(0.25),
                    "vol_above_p50_g1": vol_above_g1.median(),
                    "vol_above_p75_g1": vol_above_g1.quantile(0.75),
                    "volMeanPerTick_p25_g1": volMeanPerTick_g1.quantile(0.25),
                    "volMeanPerTick_p50_g1": volMeanPerTick_g1.median(),
                    "volMeanPerTick_p75_g1": volMeanPerTick_g1.quantile(0.75),
                    "extreme_ratio_g1": extreme_metrics_g1["extreme_ratio"],
                    "volume_spread_g1": _iqr(g1_data["sc_volume"]),
                    "volume_above_spread_g1": _iqr(vol_above_g1),
                    "volMeanPerTick_spread_g1": _iqr(volMeanPerTick_g1),
                    "atr_spread_g1": _iqr(atr_series_g1),
                    "duration_spread_g1": _iqr(g1_data["candleDuration"]),
                }

                # Nouvelles métriques win/lose
                over_thresholds_g1 = _calculate_over_thresholds(volMeanPerTick_g1)
                ratio_metrics_g1 = _calculate_ratio_metrics(g1_data["sc_volume"], volMeanPerTick_g1)

                # Ajouter les nouvelles métriques au dictionnaire
                for threshold in [1, 2, 5, 12, 20, 30]:
                    g1_indicators[f"meanVol_perTick_over{threshold}_g1"] = over_thresholds_g1[
                        f"meanVol_perTick_over{threshold}"]
                for threshold in [5, 12, 20, 30]:
                    g1_indicators[f"VolCandleMeanOver{threshold}Ratio_g1"] = ratio_metrics_g1[
                        f"VolCandleMeanOver{threshold}Ratio"]

                for k, v in g1_indicators.items():
                    df_enriched.loc[session_mask, k] = v

            except Exception as err:
                logger.warning(f"⚠️ Session {sid} – G1: {err}")

        # --- GROUPE 2 ----------------------------------------------------
        g2_data = session_data[session_data["deltaCustomSessionIndex"].isin(groupe2_sessions)]
        if not g2_data.empty:
            try:
                extreme_metrics_g2 = calculate_extreme_contracts_metrics(g2_data)
                atr_series_g2 = calculate_atr_10_periods(g2_data, period_window=period_atr_stat_session)
                vol_above_g2 = g2_data[
                                   "sc_volAbv"] / xtickReversalTickPrice if xtickReversalTickPrice and "sc_volAbv" in g2_data.columns else np.nan
                volMeanPerTick_g2 = g2_data["sc_volume"] / g2_data[
                    "candleSizeTicks"] if "candleSizeTicks" in g2_data.columns else np.nan

                # Calculs existants
                g2_indicators = {
                    "volume_p25_g2": g2_data["sc_volume"].quantile(0.25),
                    "volume_p50_g2": g2_data["sc_volume"].median(),
                    "volume_p75_g2": g2_data["sc_volume"].quantile(0.75),
                    "atr_p25_g2": atr_series_g2.quantile(0.25),
                    "atr_p50_g2": atr_series_g2.median(),
                    "atr_p75_g2": atr_series_g2.quantile(0.75),
                    "duration_p25_g2": g2_data["candleDuration"].quantile(0.25),
                    "duration_p50_g2": g2_data["candleDuration"].median(),
                    "duration_p75_g2": g2_data["candleDuration"].quantile(0.75),
                    "vol_above_p25_g2": vol_above_g2.quantile(0.25),
                    "vol_above_p50_g2": vol_above_g2.median(),
                    "vol_above_p75_g2": vol_above_g2.quantile(0.75),
                    "volMeanPerTick_p25_g2": volMeanPerTick_g2.quantile(0.25),
                    "volMeanPerTick_p50_g2": volMeanPerTick_g2.median(),
                    "volMeanPerTick_p75_g2": volMeanPerTick_g2.quantile(0.75),
                    "extreme_ratio_g2": extreme_metrics_g2["extreme_ratio"],
                    "volume_spread_g2": _iqr(g2_data["sc_volume"]),
                    "volume_above_spread_g2": _iqr(vol_above_g2),
                    "volMeanPerTick_spread_g2": _iqr(volMeanPerTick_g2),
                    "atr_spread_g2": _iqr(atr_series_g2),
                    "duration_spread_g2": _iqr(g2_data["candleDuration"]),
                }

                # Nouvelles métriques win/lose
                over_thresholds_g2 = _calculate_over_thresholds(volMeanPerTick_g2)
                ratio_metrics_g2 = _calculate_ratio_metrics(g2_data["sc_volume"], volMeanPerTick_g2)

                # Ajouter les nouvelles métriques au dictionnaire
                for threshold in [1, 2, 5, 12, 20, 30]:
                    g2_indicators[f"meanVol_perTick_over{threshold}_g2"] = over_thresholds_g2[
                        f"meanVol_perTick_over{threshold}"]
                for threshold in [5, 12, 20, 30]:
                    g2_indicators[f"VolCandleMeanOver{threshold}Ratio_g2"] = ratio_metrics_g2[
                        f"VolCandleMeanOver{threshold}Ratio"]

                for k, v in g2_indicators.items():
                    df_enriched.loc[session_mask, k] = v

            except Exception as err:
                logger.warning(f"⚠️ Session {sid} – G2: {err}")

    # ── Rapport final ────────────────────────────────────────
    logger.info("📊 Rapport de remplissage des colonnes :")
    for col in all_group_columns:
        fill_rate = df_enriched[col].notna().sum() / len(df_enriched) * 100
        logger.info(f"   {col}: {fill_rate:.1f}% rempli")

    logger.info("✅ Colonnes indicateurs ajoutées (intra‑session)")
    return df_enriched


# ────────────────────────────────────────────────────────────────────────────────
#  Générateur d'Special_indicator_autoGenPy.h + CSV - Avec SPECIAL 12
# ────────────────────────────────────────────────────────────────────────────────
import pandas as pd
from pathlib import Path


def generate_trading_config_header(params: dict, output_path: str) -> None:
    """
    Génère le header Special_indicator_autoGenPy.h à partir de PARAMS + un CSV.
    output_path : chemin absolu (str ou Path).
    """

    # Si pas de chemin CSV spécifié, on le met dans le même répertoire que le .h
    h_path = Path(output_path)
    output_csv_path = h_path.parent / "Special_indicator_config.csv"

    # ── helpers ───────────────────────────────────────────────────────────────
    f32 = lambda x, d=3: f"{x:.{d}f}f"  # float -> "1.234f"
    out = []  # lignes du fichier .h
    csv_data = []  # données pour le CSV

    def add_to_csv(config_name, struct_name, param_name, value, data_type="float"):
        """Ajoute une ligne au CSV"""
        csv_data.append({
            "config_name": config_name,
            "struct_name": struct_name,
            "param_name": param_name,
            "value": value,
            "data_type": data_type,
            "namespace": "Special_indicator_fromCsv"
        })

    # ── entête ────────────────────────────────────────────────────────────────
    out += [
        "#pragma once",
        "",
        "// ================================================================",
        "// 📋  FICHIER DE CONFIGURATION CENTRALISÉ - AUTOGÉNÉRÉ",
        "// ================================================================",
        "// Ce fichier contient tous les paramètres configurables pour les indicateurs",
        "// Modifiez ces valeurs selon vos besoins de trading",
        "// Paramètres mis à jour avec les résultats optimaux des tests",
        "// ================================================================",
        "",
        "namespace Special_indicator_hardcode {",
        "",
        "    // ================================================================",
        "    // 🎯 SPECIAL 1 - Configuration Light (Léger)",
        "    // ================================================================",
        "",
    ]

    # ── SPECIAL 1 & 2 : Light / Aggressive ──────────────────────────────────
    # Light Short
    light_short = params["light_short"]
    out += [
        "    // Paramètres pour SHORT léger (ImBullWithPoc)",
        "    struct LightShortConfig {",
        f"        static constexpr float volume_1 = {f32(light_short['volume_1'], 1)};",
        f"        static constexpr float imbalance_1 = {f32(light_short['imbalance_1'])};",
        f"        static constexpr float volume_2 = {f32(light_short['volume_2'], 1)};",
        f"        static constexpr float imbalance_2 = {f32(light_short['imbalance_2'])};",
        f"        static constexpr float volume_3 = {f32(light_short['volume_3'], 1)};",
        f"        static constexpr float imbalance_3 = {f32(light_short['imbalance_3'])};",
        "    };",
        "",
    ]

    # Ajout au CSV pour LightShortConfig
    add_to_csv("SPECIAL_1", "LightShortConfig", "volume_1", light_short['volume_1'])
    add_to_csv("SPECIAL_1", "LightShortConfig", "imbalance_1", light_short['imbalance_1'])
    add_to_csv("SPECIAL_1", "LightShortConfig", "volume_2", light_short['volume_2'])
    add_to_csv("SPECIAL_1", "LightShortConfig", "imbalance_2", light_short['imbalance_2'])
    add_to_csv("SPECIAL_1", "LightShortConfig", "volume_3", light_short['volume_3'])
    add_to_csv("SPECIAL_1", "LightShortConfig", "imbalance_3", light_short['imbalance_3'])

    # Light Long
    light_long = params["light_long"]
    out += [
        "    // Paramètres pour LONG léger (ImBearWithPoc)",
        "    struct LightLongConfig {",
        f"        static constexpr float volume_1 = {f32(light_long['volume_1'], 1)};",
        f"        static constexpr float imbalance_1 = {f32(light_long['imbalance_1'])};",
        f"        static constexpr float volume_2 = {f32(light_long['volume_2'], 1)};",
        f"        static constexpr float imbalance_2 = {f32(light_long['imbalance_2'])};",
        f"        static constexpr float volume_3 = {f32(light_long['volume_3'], 1)};",
        f"        static constexpr float imbalance_3 = {f32(light_long['imbalance_3'])};",
        "    };",
        "",
        "    // ================================================================",
        "    // 🎯 SPECIAL 2 - Configuration Aggressive (Agressif)",
        "    // ================================================================",
        "",
    ]

    # Ajout au CSV pour LightLongConfig
    add_to_csv("SPECIAL_1", "LightLongConfig", "volume_1", light_long['volume_1'])
    add_to_csv("SPECIAL_1", "LightLongConfig", "imbalance_1", light_long['imbalance_1'])
    add_to_csv("SPECIAL_1", "LightLongConfig", "volume_2", light_long['volume_2'])
    add_to_csv("SPECIAL_1", "LightLongConfig", "imbalance_2", light_long['imbalance_2'])
    add_to_csv("SPECIAL_1", "LightLongConfig", "volume_3", light_long['volume_3'])
    add_to_csv("SPECIAL_1", "LightLongConfig", "imbalance_3", light_long['imbalance_3'])

    # Aggressive Short
    agg_short = params["agg_short"]
    out += [
        "    // Paramètres pour SHORT agressif (ImBullWithPoc)",
        "    struct AggressiveShortConfig {",
        f"        static constexpr float volume_1 = {f32(agg_short['volume_1'], 1)};",
        f"        static constexpr float imbalance_1 = {f32(agg_short['imbalance_1'])};",
        f"        static constexpr float volume_2 = {f32(agg_short['volume_2'], 1)};",
        f"        static constexpr float imbalance_2 = {f32(agg_short['imbalance_2'])};",
        f"        static constexpr float volume_3 = {f32(agg_short['volume_3'], 1)};",
        f"        static constexpr float imbalance_3 = {f32(agg_short['imbalance_3'])};",
        "    };",
        "",
    ]

    # Ajout au CSV pour AggressiveShortConfig
    add_to_csv("SPECIAL_2", "AggressiveShortConfig", "volume_1", agg_short['volume_1'])
    add_to_csv("SPECIAL_2", "AggressiveShortConfig", "imbalance_1", agg_short['imbalance_1'])
    add_to_csv("SPECIAL_2", "AggressiveShortConfig", "volume_2", agg_short['volume_2'])
    add_to_csv("SPECIAL_2", "AggressiveShortConfig", "imbalance_2", agg_short['imbalance_2'])
    add_to_csv("SPECIAL_2", "AggressiveShortConfig", "volume_3", agg_short['volume_3'])
    add_to_csv("SPECIAL_2", "AggressiveShortConfig", "imbalance_3", agg_short['imbalance_3'])

    # Aggressive Long
    agg_long = params["agg_long"]
    out += [
        "    // Paramètres pour LONG agressif (ImBearWithPoc)",
        "    struct AggressiveLongConfig {",
        f"        static constexpr float volume_1 = {f32(agg_long['volume_1'], 1)};",
        f"        static constexpr float imbalance_1 = {f32(agg_long['imbalance_1'])};",
        f"        static constexpr float volume_2 = {f32(agg_long['volume_2'], 1)};",
        f"        static constexpr float imbalance_2 = {f32(agg_long['imbalance_2'])};",
        f"        static constexpr float volume_3 = {f32(agg_long['volume_3'], 1)};",
        f"        static constexpr float imbalance_3 = {f32(agg_long['imbalance_3'])};",
        "    };",
        "",
    ]

    # Ajout au CSV pour AggressiveLongConfig
    add_to_csv("SPECIAL_2", "AggressiveLongConfig", "volume_1", agg_long['volume_1'])
    add_to_csv("SPECIAL_2", "AggressiveLongConfig", "imbalance_1", agg_long['imbalance_1'])
    add_to_csv("SPECIAL_2", "AggressiveLongConfig", "volume_2", agg_long['volume_2'])
    add_to_csv("SPECIAL_2", "AggressiveLongConfig", "imbalance_2", agg_long['imbalance_2'])
    add_to_csv("SPECIAL_2", "AggressiveLongConfig", "volume_3", agg_long['volume_3'])
    add_to_csv("SPECIAL_2", "AggressiveLongConfig", "imbalance_3", agg_long['imbalance_3'])

    # ── SPECIAL 3 : MFI ──────────────────────────────────────────────────────
    mfi = params["mfi"]
    out += [
        "    // ================================================================",
        "    // 🎯 SPECIAL 3 - MFI (Money Flow Index) - MIS À JOUR",
        "    // ================================================================",
        "    struct MFIConfig {",
        "        // Configuration pour LONG OVERSOLD",
        "        struct Long {",
        f"            static constexpr int period = {mfi['oversold_period']};",
        f"            static constexpr float oversold_threshold = {f32(mfi['oversold_threshold'], 1)};",
        "            static constexpr float fill_value = 50.0f;",
        "        };",
        "",
        "        // Configuration pour SHORT OVERBOUGHT",
        "        struct Short {",
        f"            static constexpr int period = {mfi['overbought_period']};",
        f"            static constexpr float overbought_threshold = {f32(mfi['overbought_threshold'], 1)};",
        "            static constexpr float fill_value = 50.0f;",
        "        };",
        "    };",
        "",
    ]

    # Ajout au CSV pour MFI
    add_to_csv("SPECIAL_3", "MFIConfig::Long", "period", mfi['oversold_period'], "int")
    add_to_csv("SPECIAL_3", "MFIConfig::Long", "oversold_threshold", mfi['oversold_threshold'])
    add_to_csv("SPECIAL_3", "MFIConfig::Long", "fill_value", 50.0)
    add_to_csv("SPECIAL_3", "MFIConfig::Short", "period", mfi['overbought_period'], "int")
    add_to_csv("SPECIAL_3", "MFIConfig::Short", "overbought_threshold", mfi['overbought_threshold'])
    add_to_csv("SPECIAL_3", "MFIConfig::Short", "fill_value", 50.0)

    # ── SPECIAL 4 : RS Range ─────────────────────────────────────────────────
    rs = params["rs"]
    out += [
        "    // ================================================================",
        "    // 🎯 SPECIAL 4 - RS Range (Rogers-Satchell) - MIS À JOUR",
        "    // ================================================================",
        "    struct RSRangeConfig {",
        "        // Configuration pour SHORT RANGE",
        "        struct Short {",
        f"            static constexpr int period = {rs['period_short']};",
        f"            static constexpr float low_threshold = {f32(rs['low_short'], 6)};",
        f"            static constexpr float high_threshold = {f32(rs['high_short'], 6)};",
        "        };",
        "",
        "        // Configuration pour LONG RANGE",
        "        struct Long {",
        f"            static constexpr int period = {rs['period_long']};",
        f"            static constexpr float low_threshold = {f32(rs['low_long'], 6)};",
        f"            static constexpr float high_threshold = {f32(rs['high_long'], 6)};",
        "        };",
        "    };",
        "",
    ]

    # Ajout au CSV pour RS Range
    add_to_csv("SPECIAL_4", "RSRangeConfig::Short", "period", rs['period_short'], "int")
    add_to_csv("SPECIAL_4", "RSRangeConfig::Short", "low_threshold", rs['low_short'])
    add_to_csv("SPECIAL_4", "RSRangeConfig::Short", "high_threshold", rs['high_short'])
    add_to_csv("SPECIAL_4", "RSRangeConfig::Long", "period", rs['period_long'], "int")
    add_to_csv("SPECIAL_4", "RSRangeConfig::Long", "low_threshold", rs['low_long'])
    add_to_csv("SPECIAL_4", "RSRangeConfig::Long", "high_threshold", rs['high_long'])

    # ── SPECIAL 5 : Williams %R ─────────────────────────────────────────────
    wr = params["wr"]
    out += [
        "    // ================================================================",
        "    // 🎯 SPECIAL 5 - WR (Williams %R) - MIS À JOUR",
        "    // ================================================================",
        "    struct WRConfig {",
        "        // Configuration pour SHORT OVERBOUGHT",
        "        struct Short {",
        f"            static constexpr int period = {wr['period_short']};",
        f"            static constexpr float overbought_threshold = {f32(wr['th_short'], 1)};",
        "            static constexpr float fill_value = -50.0f;",
        "        };",
        "",
        "        // Configuration pour LONG OVERSOLD (ATTENTION: Rejeté dans les tests)",
        "        struct Long {",
        f"            static constexpr int period = {wr['period_long']};",
        f"            static constexpr float oversold_threshold = {f32(wr['th_long'], 1)};",
        "            static constexpr float fill_value = -50.0f;",
        "            // ⚠️ ATTENTION: Cette configuration a été REJETÉE lors des tests",
        "            // WR=51.28% < 52.5% requis",
        "        };",
        "    };",
        "",
    ]

    # Ajout au CSV pour Williams %R
    add_to_csv("SPECIAL_5", "WRConfig::Short", "period", wr['period_short'], "int")
    add_to_csv("SPECIAL_5", "WRConfig::Short", "overbought_threshold", wr['th_short'])
    add_to_csv("SPECIAL_5", "WRConfig::Short", "fill_value", -50.0)
    add_to_csv("SPECIAL_5", "WRConfig::Long", "period", wr['period_long'], "int")
    add_to_csv("SPECIAL_5", "WRConfig::Long", "oversold_threshold", wr['th_long'])
    add_to_csv("SPECIAL_5", "WRConfig::Long", "fill_value", -50.0)

    # ── SPECIAL 6 : VWAP Reversal Pro ──────────────────────────────────────
    vwap_long = params["vwap_long"]
    vwap_short = params["vwap_short"]
    out += [
        "    // ================================================================",
        "    // 🎯 SPECIAL 6 - VWAP Reversal Pro - MIS À JOUR",
        "    // ================================================================",
        "    struct VWAPRevProConfig {",
        "        // Configuration pour LONG",
        "        struct Long {",
        f"            static constexpr int lookback = {vwap_long['lookback']};",
        f"            static constexpr int momentum = {vwap_long['momentum']};",
        f"            static constexpr int z_window = {vwap_long['z_window']};",
        f"            static constexpr int atr_period = {vwap_long['atr_period']};",
        f"            static constexpr float atr_mult = {f32(vwap_long['atr_mult'], 2)};",
        f"            static constexpr int ema_filter = {vwap_long['ema_filter']};",
        f"            static constexpr int vol_lookback = {vwap_long['vol_lookback']};",
        f"            static constexpr float vol_ratio_min = {f32(vwap_long['vol_ratio_min'], 2)};",
        "        };",
        "",
        "        // Configuration pour SHORT",
        "        struct Short {",
        f"            static constexpr int lookback = {vwap_short['lookback']};",
        f"            static constexpr int momentum = {vwap_short['momentum']};",
        f"            static constexpr int z_window = {vwap_short['z_window']};",
        f"            static constexpr int atr_period = {vwap_short['atr_period']};",
        f"            static constexpr float atr_mult = {f32(vwap_short['atr_mult'], 2)};",
        f"            static constexpr int ema_filter = {vwap_short['ema_filter']};",
        f"            static constexpr int vol_lookback = {vwap_short['vol_lookback']};",
        f"            static constexpr float vol_ratio_min = {f32(vwap_short['vol_ratio_min'], 2)};",
        "        };",
        "    };",
        "",
    ]

    # Ajout au CSV pour VWAP Reversal Pro
    for param in ['lookback', 'momentum', 'z_window', 'atr_period', 'ema_filter', 'vol_lookback']:
        add_to_csv("SPECIAL_6", "VWAPRevProConfig::Long", param, vwap_long[param], "int")
        add_to_csv("SPECIAL_6", "VWAPRevProConfig::Short", param, vwap_short[param], "int")

    add_to_csv("SPECIAL_6", "VWAPRevProConfig::Long", "atr_mult", vwap_long['atr_mult'])
    add_to_csv("SPECIAL_6", "VWAPRevProConfig::Long", "vol_ratio_min", vwap_long['vol_ratio_min'])
    add_to_csv("SPECIAL_6", "VWAPRevProConfig::Short", "atr_mult", vwap_short['atr_mult'])
    add_to_csv("SPECIAL_6", "VWAPRevProConfig::Short", "vol_ratio_min", vwap_short['vol_ratio_min'])

    # ── SPECIAL 7 : IMB Bull/Bear Light POC avec ATR ──────────────────────
    atr_high = params["atr_high"]
    atr_low = params["atr_low"]
    out += [
        "    // ================================================================",
        "    // 🎯 SPECIAL 7 - IMB Bull/Bear Light POC avec ATR",
        "    // ================================================================",
        "    struct ImbBullLightPocATRConfig {",
        "        // Seuils ATR",
        f"        static constexpr float atr_threshold_1 = {f32(atr_high['atr_threshold_1'], 1)};",
        f"        static constexpr float atr_threshold_2 = {f32(atr_high['atr_threshold_2'], 1)};",
        f"        static constexpr float atr_threshold_3 = {f32(atr_high['atr_threshold_3'], 1)};",
        "",
        "        // Différences pour SHORT (AtrHigh)",
        f"        static constexpr float diff_high_atr_1 = {f32(atr_high['diff_high_atr_1'], 2)};   // ATR < 1.5",
        f"        static constexpr float diff_high_atr_2 = {f32(atr_high['diff_high_atr_2'], 2)};  // 1.5 <= ATR < 1.7",
        f"        static constexpr float diff_high_atr_3 = {f32(atr_high['diff_high_atr_3'], 2)};  // 1.7 <= ATR < 1.9",
        f"        static constexpr float diff_high_atr_4 = {f32(atr_high['diff_high_atr_4'], 2)};  // ATR >= 1.9",
        "",
        "        // Différences pour LONG (AtrLow)",
        f"        static constexpr float diff_low_atr_1 = {f32(atr_low['diff_low_atr_1'], 2)};    // ATR < 1.5",
        f"        static constexpr float diff_low_atr_2 = {f32(atr_low['diff_low_atr_2'], 2)};   // 1.5 <= ATR < 1.7",
        f"        static constexpr float diff_low_atr_3 = {f32(atr_low['diff_low_atr_3'], 2)};   // 1.7 <= ATR < 1.9",
        f"        static constexpr float diff_low_atr_4 = {f32(atr_low['diff_low_atr_4'], 2)};   // ATR >= 1.9",
        "",
        "        // Fenêtre ATR",
        f"        static constexpr int atr_window = {atr_high['atr_window']};",
        "    };",
        "",
    ]

    # Ajout au CSV pour IMB Bull Light POC ATR
    for i in range(1, 4):
        add_to_csv("SPECIAL_7", "ImbBullLightPocATRConfig", f"atr_threshold_{i}", atr_high[f'atr_threshold_{i}'])
    for i in range(1, 5):
        add_to_csv("SPECIAL_7", "ImbBullLightPocATRConfig", f"diff_high_atr_{i}", atr_high[f'diff_high_atr_{i}'])
        add_to_csv("SPECIAL_7", "ImbBullLightPocATRConfig", f"diff_low_atr_{i}", atr_low[f'diff_low_atr_{i}'])
    add_to_csv("SPECIAL_7", "ImbBullLightPocATRConfig", "atr_window", atr_high['atr_window'], "int")

    # ── SPECIAL 8-11 : Volume Per Tick (optionnel - ajouté si présent) ─────
    if "volume_per_tick_light_short" in params:
        vpt_light_short = params["volume_per_tick_light_short"]
        vpt_light_long = params["volume_per_tick_light_long"]
        vpt_agg_short = params["volume_per_tick_aggressive_short"]
        vpt_agg_long = params["volume_per_tick_aggressive_long"]

        out += [
            "    // ================================================================",
            "    // 🎯 SPECIAL 8-11 - Volume Per Tick Configurations",
            "    // ================================================================",
            "    struct VolumePerTickConfig {",
            "        // Light Profile - Plus de trades",
            "        struct Light {",
            "            struct Short {",
            f"                static constexpr int period = {vpt_light_short['period']};",
            f"                static constexpr float threshold_low = {f32(vpt_light_short['threshold_low'], 2)};",
            f"                static constexpr float threshold_high = {f32(vpt_light_short['threshold_high'], 2)};",
            f"                static constexpr float expected_trade_pct = {f32(vpt_light_short['expected_trade_pct'], 1)};",
            f"                static constexpr float expected_winrate = {f32(vpt_light_short['expected_winrate'])};",
            "            };",
            "            struct Long {",
            f"                static constexpr int period = {vpt_light_long['period']};",
            f"                static constexpr float threshold_low = {f32(vpt_light_long['threshold_low'], 2)};",
            f"                static constexpr float threshold_high = {f32(vpt_light_long['threshold_high'], 2)};",
            f"                static constexpr float expected_trade_pct = {f32(vpt_light_long['expected_trade_pct'], 1)};",
            f"                static constexpr float expected_winrate = {f32(vpt_light_long['expected_winrate'])};",
            "            };",
            "        };",
            "",
            "        // Aggressive Profile - Moins de trades, meilleur WR",
            "        struct Aggressive {",
            "            struct Short {",
            f"                static constexpr int period = {vpt_agg_short['period']};",
            f"                static constexpr float threshold_low = {f32(vpt_agg_short['threshold_low'], 2)};",
            f"                static constexpr float threshold_high = {f32(vpt_agg_short['threshold_high'], 2)};",
            f"                static constexpr float expected_trade_pct = {f32(vpt_agg_short['expected_trade_pct'], 1)};",
            f"                static constexpr float expected_winrate = {f32(vpt_agg_short['expected_winrate'])};",
            "            };",
            "            struct Long {",
            f"                static constexpr int period = {vpt_agg_long['period']};",
            f"                static constexpr float threshold_low = {f32(vpt_agg_long['threshold_low'], 2)};",
            f"                static constexpr float threshold_high = {f32(vpt_agg_long['threshold_high'], 2)};",
            f"                static constexpr float expected_trade_pct = {f32(vpt_agg_long['expected_trade_pct'], 1)};",
            f"                static constexpr float expected_winrate = {f32(vpt_agg_long['expected_winrate'])};",
            "            };",
            "        };",
            "    };",
            "",
        ]

        # Ajout au CSV pour Volume Per Tick
        for config_type, config_data in [("Light::Short", vpt_light_short), ("Light::Long", vpt_light_long),
                                         ("Aggressive::Short", vpt_agg_short), ("Aggressive::Long", vpt_agg_long)]:
            add_to_csv("SPECIAL_8-11", f"VolumePerTickConfig::{config_type}", "period", config_data['period'], "int")
            add_to_csv("SPECIAL_8-11", f"VolumePerTickConfig::{config_type}", "threshold_low",
                       config_data['threshold_low'])
            add_to_csv("SPECIAL_8-11", f"VolumePerTickConfig::{config_type}", "threshold_high",
                       config_data['threshold_high'])
            add_to_csv("SPECIAL_8-11", f"VolumePerTickConfig::{config_type}", "expected_trade_pct",
                       config_data['expected_trade_pct'])
            add_to_csv("SPECIAL_8-11", f"VolumePerTickConfig::{config_type}", "expected_winrate",
                       config_data['expected_winrate'])

    # ── SPECIAL 12 : Microstructure Anti-Spring & Anti-Épuisement ────────
    micro_antispring = params["micro_antispring"]
    micro_antiepuisement = params["micro_antiepuisement"]
    out += [
        "    // ================================================================",
        "    // 🎯 SPECIAL 12 - Microstructure Anti-Spring & Anti-Épuisement",
        "    // ================================================================",
        "    struct MicrostructureConfig {",
        "        // Anti-Spring Configuration (SHORT)",
        "        struct AntiSpring {",
        f"            static constexpr int period = {micro_antispring['short_period']};",
        f"            static constexpr float vol_threshold = {f32(micro_antispring['short_vol_threshold'])};",
        f"            static constexpr float dur_threshold = {f32(micro_antispring['short_dur_threshold'])};",
        f"            static constexpr int condition = {micro_antispring['short_condition']};  // {micro_antispring['short_condition']} = V<D<",
        "        };",
        "",
        "        // Anti-Épuisement Configuration (LONG)",
        "        struct AntiEpuisement {",
        f"            static constexpr int period = {micro_antiepuisement['long_period']};",
        f"            static constexpr float vol_threshold = {f32(micro_antiepuisement['long_vol_threshold'])};",
        f"            static constexpr float dur_threshold = {f32(micro_antiepuisement['long_dur_threshold'])};",
        f"            static constexpr int condition = {micro_antiepuisement['long_condition']};  // {micro_antiepuisement['long_condition']} = V>D>",
        "        };",
        "    };",
        "",
    ]

    # Ajout au CSV pour Microstructure
    add_to_csv("SPECIAL_12", "MicrostructureConfig::AntiSpring", "period", micro_antispring['short_period'], "int")
    add_to_csv("SPECIAL_12", "MicrostructureConfig::AntiSpring", "vol_threshold",
               micro_antispring['short_vol_threshold'])
    add_to_csv("SPECIAL_12", "MicrostructureConfig::AntiSpring", "dur_threshold",
               micro_antispring['short_dur_threshold'])
    add_to_csv("SPECIAL_12", "MicrostructureConfig::AntiSpring", "condition", micro_antispring['short_condition'],
               "int")

    add_to_csv("SPECIAL_12", "MicrostructureConfig::AntiEpuisement", "period", micro_antiepuisement['long_period'],
               "int")
    add_to_csv("SPECIAL_12", "MicrostructureConfig::AntiEpuisement", "vol_threshold",
               micro_antiepuisement['long_vol_threshold'])
    add_to_csv("SPECIAL_12", "MicrostructureConfig::AntiEpuisement", "dur_threshold",
               micro_antiepuisement['long_dur_threshold'])
    add_to_csv("SPECIAL_12", "MicrostructureConfig::AntiEpuisement", "condition",
               micro_antiepuisement['long_condition'], "int")

    # ── Résultats de validation ──────────────────────────────────────────────
    out += [
        "    // ================================================================",
        "    // 📊 RÉSULTATS DE VALIDATION SUR DATASET TEST",
        "    // ================================================================",
        "    /*",
        "    RÉCAPITULATIF DES PERFORMANCES (Dataset TEST):",
        "",
        "    ✅ VALIDÉ:",
        "    1. Light Short (Imbalance): WR=56.76%, PCT=4.26%, Trades=74",
        "    2. Light Long (Imbalance): WR=72.22%, PCT=1.16%, Trades=18",
        "    3. Aggressive Short (Imbalance): WR=60.00%, PCT=0.58%, Trades=10",
        "    4. MFI Long OVERSOLD: WR=53.03%, PCT=8.50%, Trades=132",
        "    5. MFI Short OVERBOUGHT: WR=52.02%, PCT=14.28%, Trades=248",
        "    6. RS Short RANGE: WR=58.33%, PCT=8.98%, Trades=156",
        "    7. RS Long RANGE: WR=61.54%, PCT=9.21%, Trades=143",
        "    8. Williams %R Short OVERBOUGHT: WR=55.37%, PCT=13.93%, Trades=242",
        "    9. VWAP Reversal Pro LONG: WR=55.35%, PCT=10.24%, Trades=159",
        "    10. VWAP Reversal Pro SHORT: WR=54.41%, PCT=3.91%, Trades=68",
        "    11. Volume Per Tick Light SHORT: WR=55.00%, PCT=8.50%, Trades=~8.5%",
        "    12. Microstructure Anti-Spring SHORT: Optimisé pour détection de retournements",
        "    13. Microstructure Anti-Épuisement LONG: Optimisé pour détection de retournements",
        "",
        "    ❌ REJETÉ:",
        "    - Williams %R Long OVERSOLD: WR=51.28% < 52.5% requis",
        "    */",
        "",
        "} // namespace Special_indicator_hardcode",
    ]

    # ────────────────────────────────────────────────────────────────────────────────
    # 🔧 FONCTION POUR GÉNÉRER LE CHARGEUR CSV (optionnel)
    # ────────────────────────────────────────────────────────────────────────────────

    def generate_csv_loader_header(csv_path: str, output_h_path: str) -> None:
        """
        Génère un fichier .h qui contient un namespace Special_indicator_fromCsv
        avec des fonctions pour charger les paramètres depuis le CSV
        """

        # Lecture du CSV avec séparateur point-virgule
        df = pd.read_csv(csv_path, sep=";")

        out = [
            "#pragma once",
            "#include <map>",
            "#include <string>",
            "#include <fstream>",
            "#include <sstream>",
            "",
            "// ================================================================",
            "// 📋  CHARGEUR DE CONFIGURATION DEPUIS CSV - AUTOGÉNÉRÉ",
            "// ================================================================",
            "// Ce fichier permet de charger les paramètres depuis un CSV",
            "// pour remplacer dynamiquement le namespace Special_indicator_hardcode",
            "// ================================================================",
            "",
            "namespace Special_indicator_fromCsv {",
            "",
            "    // Structure pour stocker les paramètres chargés depuis le CSV",
            "    struct ConfigParam {",
            "        std::string value_str;",
            "        float value_float = 0.0f;",
            "        int value_int = 0;",
            "        std::string data_type;",
            "    };",
            "",
            "    // Map globale pour stocker tous les paramètres",
            "    static std::map<std::string, ConfigParam> g_config_params;",
            "    static bool g_config_loaded = false;",
            "",
            "    // Fonction pour charger le CSV",
            "    bool LoadConfigFromCSV(const std::string& csv_path) {",
            "        std::ifstream file(csv_path);",
            "        if (!file.is_open()) {",
            "            return false;",
            "        }",
            "",
            "        g_config_params.clear();",
            "        std::string line;",
            "        std::getline(file, line); // Skip header",
            "",
            "        while (std::getline(file, line)) {",
            "            std::stringstream ss(line);",
            "            std::string config_name, struct_name, param_name, value_str, data_type, namespace_name;",
            "",
            "            std::getline(ss, config_name, ';');",
            "            std::getline(ss, struct_name, ';');",
            "            std::getline(ss, param_name, ';');",
            "            std::getline(ss, value_str, ';');",
            "            std::getline(ss, data_type, ';');",
            "            std::getline(ss, namespace_name, ';');",
            "",
            "            std::string key = struct_name + \"::\" + param_name;",
            "            ConfigParam param;",
            "            param.value_str = value_str;",
            "            param.data_type = data_type;",
            "",
            "            if (data_type == \"int\") {",
            "                param.value_int = std::stoi(value_str);",
            "            } else {",
            "                param.value_float = std::stof(value_str);",
            "            }",
            "",
            "            g_config_params[key] = param;",
            "        }",
            "",
            "        g_config_loaded = true;",
            "        return true;",
            "    }",
            "",
            "    // Fonctions helper pour récupérer les valeurs",
            "    float GetFloat(const std::string& struct_name, const std::string& param_name) {",
            "        if (!g_config_loaded) return 0.0f;",
            "        std::string key = struct_name + \"::\" + param_name;",
            "        auto it = g_config_params.find(key);",
            "        return (it != g_config_params.end()) ? it->second.value_float : 0.0f;",
            "    }",
            "",
            "    int GetInt(const std::string& struct_name, const std::string& param_name) {",
            "        if (!g_config_loaded) return 0;",
            "        std::string key = struct_name + \"::\" + param_name;",
            "        auto it = g_config_params.find(key);",
            "        return (it != g_config_params.end()) ? it->second.value_int : 0;",
            "    }",
            "",
        ]

        # Génération des structures avec les mêmes noms mais valeurs dynamiques
        structures = df['struct_name'].unique()

        for struct in structures:
            struct_params = df[df['struct_name'] == struct]
            clean_struct_name = struct.replace('::', '_').replace('_', '')

            out.append(f"    // Structure {struct}")
            out.append(f"    struct {clean_struct_name} {{")

            for _, row in struct_params.iterrows():
                param_name = row['param_name']
                data_type_str = "int" if row['data_type'] == "int" else "float"

                out.append(f"        static {data_type_str} {param_name}() {{")
                if row['data_type'] == "int":
                    out.append(f"            return GetInt(\"{struct}\", \"{param_name}\");")
                else:
                    out.append(f"            return GetFloat(\"{struct}\", \"{param_name}\");")
                out.append("        }")

            out.append("    };")
            out.append("")

        out += [
            "} // namespace Special_indicator_fromCsv",
            "",
            "// ================================================================",
            "// 🔧 UTILISATION:",
            "// ================================================================",
            "// 1. Charger le CSV au début de votre programme:",
            "//    Special_indicator_fromCsv::LoadConfigFromCSV(\"path/to/config.csv\");",
            "",
            "// 2. Utiliser les paramètres:",
            "//    float vol1 = Special_indicator_fromCsv::LightShortConfig::volume_1();",
            "//    int period = Special_indicator_fromCsv::MFIConfigLong::period();",
            "// ================================================================",
        ]

        content = "\n".join(out)
        Path(output_h_path).write_text(content, encoding="utf-8")
        print("✅ Chargeur CSV header généré :", output_h_path)

    # ────────────────────────────────────────────────────────────────────────────────
    # 🧪 EXEMPLE D'UTILISATION
    # ────────────────────────────────────────────────────────────────────────────────

    def example_usage():
        """Exemple d'utilisation des fonctions"""

        # Exemple de paramètres (remplacez par vos vrais paramètres)
        example_params = {
            "light_short": {"volume_1": 3.0, "imbalance_1": 2.877, "volume_2": 16.0, "imbalance_2": 3.750,
                            "volume_3": 20.0, "imbalance_3": 1.613},
            "light_long": {"volume_1": 3.0, "imbalance_1": 4.140, "volume_2": 8.0, "imbalance_2": 2.098,
                           "volume_3": 43.0, "imbalance_3": 6.915},
            "agg_short": {"volume_1": 3.0, "imbalance_1": 5.374, "volume_2": 17.0, "imbalance_2": 5.097,
                          "volume_3": 26.0, "imbalance_3": 6.346},
            "agg_long": {"volume_1": 6.0, "imbalance_1": 4.170, "volume_2": 16.0, "imbalance_2": 3.435,
                         "volume_3": 27.0, "imbalance_3": 7.359},
            "mfi": {"oversold_period": 40, "oversold_threshold": 35.0, "overbought_period": 5,
                    "overbought_threshold": 78.0},
            "rs": {"period_short": 11, "low_short": 0.000174, "high_short": 0.000180, "period_long": 17,
                   "low_long": 0.000189, "high_long": 0.000194},
            "wr": {"period_short": 15, "th_short": -10.0, "period_long": 51, "th_long": -95.0},
            "vwap_long": {"lookback": 38, "momentum": 11, "z_window": 45, "atr_period": 32, "atr_mult": 1.20,
                          "ema_filter": 77, "vol_lookback": 8, "vol_ratio_min": 0.25},
            "vwap_short": {"lookback": 19, "momentum": 18, "z_window": 22, "atr_period": 19, "atr_mult": 2.70,
                           "ema_filter": 29, "vol_lookback": 12, "vol_ratio_min": 0.60},
            "atr_high": {"atr_threshold_1": 1.5, "atr_threshold_2": 1.7, "atr_threshold_3": 1.9,
                         "diff_high_atr_1": 5.50, "diff_high_atr_2": 3.75, "diff_high_atr_3": 5.75,
                         "diff_high_atr_4": 3.25, "atr_window": 12},
            "atr_low": {"diff_low_atr_1": 5.50, "diff_low_atr_2": 3.75, "diff_low_atr_3": 5.75, "diff_low_atr_4": 3.25},
            "micro_antispring": {"short_period": 5, "short_vol_threshold": -0.860, "short_dur_threshold": 6.280,
                                 "short_condition": 0},
            "micro_antiepuisement": {"long_period": 6, "long_vol_threshold": -4.820, "long_dur_threshold": 2.350,
                                     "long_condition": 3}
        }

        # 1. Génération du .h et du CSV
        df = generate_trading_config_header(
            params=example_params,
            output_path="Special_indicator_autoGenPy.h"
        )

        # 2. Optionnel : Génération du chargeur CSV
        generate_csv_loader_header(
            csv_path="Special_indicator_config.csv",
            output_h_path="Special_indicator_fromCsv.h"
        )

        print("\n🎯 RÉSUMÉ :")
        print("✅ Fichier .h généré avec namespace Special_indicator_hardcode")
        print("✅ Fichier CSV généré pour utilisation dynamique")
        print("✅ Chargeur CSV généré pour namespace Special_indicator_fromCsv")

    # ── écriture du fichier .h ────────────────────────────────────────────────────
    content = "\n".join(out)
    Path(output_path).write_text(content, encoding="utf-8")
    print("✅ SpecialIndicator header généré avec SPECIAL 12 :", output_path)

    # ── écriture du fichier CSV ────────────────────────────────────────────────────
    df = pd.DataFrame(csv_data)

    # Réorganisation des colonnes pour plus de clarté
    column_order = ["config_name", "struct_name", "param_name", "value", "data_type", "namespace"]
    df = df[column_order]

    # Écriture du CSV
    df.to_csv(output_csv_path, index=False, encoding="utf-8", sep=";")
    print("✅ Fichier CSV de configuration généré :", output_csv_path)
    print(f"📊 Total de {len(csv_data)} paramètres exportés vers le CSV")

    return df


# ────────────────────────────────────────────────────────────────────────────────
# 🔧 FONCTION POUR GÉNÉRER LE CHARGEUR CSV (optionnel)
# ────────────────────────────────────────────────────────────────────────────────

def generate_trading_config_header(params: dict, output_path: str) -> None:
    """
    Génère le header Special_indicator_autoGenPy.h à partir de PARAMS + un CSV.
    output_path : chemin absolu (str ou Path).
    """

    # Si pas de chemin CSV spécifié, on le met dans le même répertoire que le .h
    h_path = Path(output_path)
    output_csv_path = h_path.parent / "Special_indicator_config.csv"

    # ── helpers ───────────────────────────────────────────────────────────────
    f32 = lambda x, d=3: f"{x:.{d}f}f"  # float -> "1.234f"
    out = []  # lignes du fichier .h
    csv_data = []  # données pour le CSV

    def add_to_csv(config_name, struct_name, param_name, value, data_type="float"):
        """Ajoute une ligne au CSV"""
        csv_data.append({
            "config_name": config_name,
            "struct_name": struct_name,
            "param_name": param_name,
            "value": value,
            "data_type": data_type,
            "namespace": "Special_indicator_fromCsv"
        })

    # ── entête ────────────────────────────────────────────────────────────────
    out += [
        "#pragma once",
        "",
        "// ================================================================",
        "// 📋  FICHIER DE CONFIGURATION CENTRALISÉ - AUTOGÉNÉRÉ",
        "// ================================================================",
        "// Ce fichier contient tous les paramètres configurables pour les indicateurs",
        "// Modifiez ces valeurs selon vos besoins de trading",
        "// Paramètres mis à jour avec les résultats optimaux des tests",
        "// ================================================================",
        "",
        "namespace Special_indicator_hardcode {",
        "",
        "    // ================================================================",
        "    // 🎯 SPECIAL 1 - Configuration Light (Léger)",
        "    // ================================================================",
        "",
    ]

    # ── SPECIAL 1 & 2 : Light / Aggressive ──────────────────────────────────
    # Light Short
    light_short = params["light_short"]
    out += [
        "    // Paramètres pour SHORT léger (ImBullWithPoc)",
        "    struct LightShortConfig {",
        f"        static constexpr float volume_1 = {f32(light_short['volume_1'], 1)};",
        f"        static constexpr float imbalance_1 = {f32(light_short['imbalance_1'])};",
        f"        static constexpr float volume_2 = {f32(light_short['volume_2'], 1)};",
        f"        static constexpr float imbalance_2 = {f32(light_short['imbalance_2'])};",
        f"        static constexpr float volume_3 = {f32(light_short['volume_3'], 1)};",
        f"        static constexpr float imbalance_3 = {f32(light_short['imbalance_3'])};",
        "    };",
        "",
    ]

    # Ajout au CSV pour LightShortConfig
    add_to_csv("SPECIAL_1", "LightShortConfig", "volume_1", light_short['volume_1'])
    add_to_csv("SPECIAL_1", "LightShortConfig", "imbalance_1", light_short['imbalance_1'])
    add_to_csv("SPECIAL_1", "LightShortConfig", "volume_2", light_short['volume_2'])
    add_to_csv("SPECIAL_1", "LightShortConfig", "imbalance_2", light_short['imbalance_2'])
    add_to_csv("SPECIAL_1", "LightShortConfig", "volume_3", light_short['volume_3'])
    add_to_csv("SPECIAL_1", "LightShortConfig", "imbalance_3", light_short['imbalance_3'])

    # Light Long
    light_long = params["light_long"]
    out += [
        "    // Paramètres pour LONG léger (ImBearWithPoc)",
        "    struct LightLongConfig {",
        f"        static constexpr float volume_1 = {f32(light_long['volume_1'], 1)};",
        f"        static constexpr float imbalance_1 = {f32(light_long['imbalance_1'])};",
        f"        static constexpr float volume_2 = {f32(light_long['volume_2'], 1)};",
        f"        static constexpr float imbalance_2 = {f32(light_long['imbalance_2'])};",
        f"        static constexpr float volume_3 = {f32(light_long['volume_3'], 1)};",
        f"        static constexpr float imbalance_3 = {f32(light_long['imbalance_3'])};",
        "    };",
        "",
        "    // ================================================================",
        "    // 🎯 SPECIAL 2 - Configuration Aggressive (Agressif)",
        "    // ================================================================",
        "",
    ]

    # Ajout au CSV pour LightLongConfig
    add_to_csv("SPECIAL_1", "LightLongConfig", "volume_1", light_long['volume_1'])
    add_to_csv("SPECIAL_1", "LightLongConfig", "imbalance_1", light_long['imbalance_1'])
    add_to_csv("SPECIAL_1", "LightLongConfig", "volume_2", light_long['volume_2'])
    add_to_csv("SPECIAL_1", "LightLongConfig", "imbalance_2", light_long['imbalance_2'])
    add_to_csv("SPECIAL_1", "LightLongConfig", "volume_3", light_long['volume_3'])
    add_to_csv("SPECIAL_1", "LightLongConfig", "imbalance_3", light_long['imbalance_3'])

    # Aggressive Short
    agg_short = params["agg_short"]
    out += [
        "    // Paramètres pour SHORT agressif (ImBullWithPoc)",
        "    struct AggressiveShortConfig {",
        f"        static constexpr float volume_1 = {f32(agg_short['volume_1'], 1)};",
        f"        static constexpr float imbalance_1 = {f32(agg_short['imbalance_1'])};",
        f"        static constexpr float volume_2 = {f32(agg_short['volume_2'], 1)};",
        f"        static constexpr float imbalance_2 = {f32(agg_short['imbalance_2'])};",
        f"        static constexpr float volume_3 = {f32(agg_short['volume_3'], 1)};",
        f"        static constexpr float imbalance_3 = {f32(agg_short['imbalance_3'])};",
        "    };",
        "",
    ]

    # Ajout au CSV pour AggressiveShortConfig
    add_to_csv("SPECIAL_2", "AggressiveShortConfig", "volume_1", agg_short['volume_1'])
    add_to_csv("SPECIAL_2", "AggressiveShortConfig", "imbalance_1", agg_short['imbalance_1'])
    add_to_csv("SPECIAL_2", "AggressiveShortConfig", "volume_2", agg_short['volume_2'])
    add_to_csv("SPECIAL_2", "AggressiveShortConfig", "imbalance_2", agg_short['imbalance_2'])
    add_to_csv("SPECIAL_2", "AggressiveShortConfig", "volume_3", agg_short['volume_3'])
    add_to_csv("SPECIAL_2", "AggressiveShortConfig", "imbalance_3", agg_short['imbalance_3'])

    # Aggressive Long
    agg_long = params["agg_long"]
    out += [
        "    // Paramètres pour LONG agressif (ImBearWithPoc)",
        "    struct AggressiveLongConfig {",
        f"        static constexpr float volume_1 = {f32(agg_long['volume_1'], 1)};",
        f"        static constexpr float imbalance_1 = {f32(agg_long['imbalance_1'])};",
        f"        static constexpr float volume_2 = {f32(agg_long['volume_2'], 1)};",
        f"        static constexpr float imbalance_2 = {f32(agg_long['imbalance_2'])};",
        f"        static constexpr float volume_3 = {f32(agg_long['volume_3'], 1)};",
        f"        static constexpr float imbalance_3 = {f32(agg_long['imbalance_3'])};",
        "    };",
        "",
    ]

    # Ajout au CSV pour AggressiveLongConfig
    add_to_csv("SPECIAL_2", "AggressiveLongConfig", "volume_1", agg_long['volume_1'])
    add_to_csv("SPECIAL_2", "AggressiveLongConfig", "imbalance_1", agg_long['imbalance_1'])
    add_to_csv("SPECIAL_2", "AggressiveLongConfig", "volume_2", agg_long['volume_2'])
    add_to_csv("SPECIAL_2", "AggressiveLongConfig", "imbalance_2", agg_long['imbalance_2'])
    add_to_csv("SPECIAL_2", "AggressiveLongConfig", "volume_3", agg_long['volume_3'])
    add_to_csv("SPECIAL_2", "AggressiveLongConfig", "imbalance_3", agg_long['imbalance_3'])

    # ── SPECIAL 3 : MFI ──────────────────────────────────────────────────────
    mfi = params["mfi"]
    out += [
        "    // ================================================================",
        "    // 🎯 SPECIAL 3 - MFI (Money Flow Index) - MIS À JOUR",
        "    // ================================================================",
        "    struct MFIConfig {",
        "        // Configuration pour LONG OVERSOLD",
        "        struct Long {",
        f"            static constexpr int period = {mfi['oversold_period']};",
        f"            static constexpr float oversold_threshold = {f32(mfi['oversold_threshold'], 1)};",
        "            static constexpr float fill_value = 50.0f;",
        "        };",
        "",
        "        // Configuration pour SHORT OVERBOUGHT",
        "        struct Short {",
        f"            static constexpr int period = {mfi['overbought_period']};",
        f"            static constexpr float overbought_threshold = {f32(mfi['overbought_threshold'], 1)};",
        "            static constexpr float fill_value = 50.0f;",
        "        };",
        "    };",
        "",
    ]

    # Ajout au CSV pour MFI
    add_to_csv("SPECIAL_3", "MFIConfig::Long", "period", mfi['oversold_period'], "int")
    add_to_csv("SPECIAL_3", "MFIConfig::Long", "oversold_threshold", mfi['oversold_threshold'])
    add_to_csv("SPECIAL_3", "MFIConfig::Long", "fill_value", 50.0)
    add_to_csv("SPECIAL_3", "MFIConfig::Short", "period", mfi['overbought_period'], "int")
    add_to_csv("SPECIAL_3", "MFIConfig::Short", "overbought_threshold", mfi['overbought_threshold'])
    add_to_csv("SPECIAL_3", "MFIConfig::Short", "fill_value", 50.0)

    # ── SPECIAL 4 : RS Range ─────────────────────────────────────────────────
    rs = params["rs"]
    out += [
        "    // ================================================================",
        "    // 🎯 SPECIAL 4 - RS Range (Rogers-Satchell) - MIS À JOUR",
        "    // ================================================================",
        "    struct RSRangeConfig {",
        "        // Configuration pour SHORT RANGE",
        "        struct Short {",
        f"            static constexpr int period = {rs['period_short']};",
        f"            static constexpr float low_threshold = {f32(rs['low_short'], 6)};",
        f"            static constexpr float high_threshold = {f32(rs['high_short'], 6)};",
        "        };",
        "",
        "        // Configuration pour LONG RANGE",
        "        struct Long {",
        f"            static constexpr int period = {rs['period_long']};",
        f"            static constexpr float low_threshold = {f32(rs['low_long'], 6)};",
        f"            static constexpr float high_threshold = {f32(rs['high_long'], 6)};",
        "        };",
        "    };",
        "",
    ]

    # Ajout au CSV pour RS Range
    add_to_csv("SPECIAL_4", "RSRangeConfig::Short", "period", rs['period_short'], "int")
    add_to_csv("SPECIAL_4", "RSRangeConfig::Short", "low_threshold", rs['low_short'])
    add_to_csv("SPECIAL_4", "RSRangeConfig::Short", "high_threshold", rs['high_short'])
    add_to_csv("SPECIAL_4", "RSRangeConfig::Long", "period", rs['period_long'], "int")
    add_to_csv("SPECIAL_4", "RSRangeConfig::Long", "low_threshold", rs['low_long'])
    add_to_csv("SPECIAL_4", "RSRangeConfig::Long", "high_threshold", rs['high_long'])

    # ── SPECIAL 5 : Williams %R ─────────────────────────────────────────────
    wr = params["wr"]
    out += [
        "    // ================================================================",
        "    // 🎯 SPECIAL 5 - WR (Williams %R) - MIS À JOUR",
        "    // ================================================================",
        "    struct WRConfig {",
        "        // Configuration pour SHORT OVERBOUGHT",
        "        struct Short {",
        f"            static constexpr int period = {wr['period_short']};",
        f"            static constexpr float overbought_threshold = {f32(wr['th_short'], 1)};",
        "            static constexpr float fill_value = -50.0f;",
        "        };",
        "",
        "        // Configuration pour LONG OVERSOLD (ATTENTION: Rejeté dans les tests)",
        "        struct Long {",
        f"            static constexpr int period = {wr['period_long']};",
        f"            static constexpr float oversold_threshold = {f32(wr['th_long'], 1)};",
        "            static constexpr float fill_value = -50.0f;",
        "            // ⚠️ ATTENTION: Cette configuration a été REJETÉE lors des tests",
        "            // WR=51.28% < 52.5% requis",
        "        };",
        "    };",
        "",
    ]

    # Ajout au CSV pour Williams %R
    add_to_csv("SPECIAL_5", "WRConfig::Short", "period", wr['period_short'], "int")
    add_to_csv("SPECIAL_5", "WRConfig::Short", "overbought_threshold", wr['th_short'])
    add_to_csv("SPECIAL_5", "WRConfig::Short", "fill_value", -50.0)
    add_to_csv("SPECIAL_5", "WRConfig::Long", "period", wr['period_long'], "int")
    add_to_csv("SPECIAL_5", "WRConfig::Long", "oversold_threshold", wr['th_long'])
    add_to_csv("SPECIAL_5", "WRConfig::Long", "fill_value", -50.0)

    # ── SPECIAL 6 : VWAP Reversal Pro ──────────────────────────────────────
    vwap_long = params["vwap_long"]
    vwap_short = params["vwap_short"]
    out += [
        "    // ================================================================",
        "    // 🎯 SPECIAL 6 - VWAP Reversal Pro - MIS À JOUR",
        "    // ================================================================",
        "    struct VWAPRevProConfig {",
        "        // Configuration pour LONG",
        "        struct Long {",
        f"            static constexpr int lookback = {vwap_long['lookback']};",
        f"            static constexpr int momentum = {vwap_long['momentum']};",
        f"            static constexpr int z_window = {vwap_long['z_window']};",
        f"            static constexpr int atr_period = {vwap_long['atr_period']};",
        f"            static constexpr float atr_mult = {f32(vwap_long['atr_mult'], 2)};",
        f"            static constexpr int ema_filter = {vwap_long['ema_filter']};",
        f"            static constexpr int vol_lookback = {vwap_long['vol_lookback']};",
        f"            static constexpr float vol_ratio_min = {f32(vwap_long['vol_ratio_min'], 2)};",
        "        };",
        "",
        "        // Configuration pour SHORT",
        "        struct Short {",
        f"            static constexpr int lookback = {vwap_short['lookback']};",
        f"            static constexpr int momentum = {vwap_short['momentum']};",
        f"            static constexpr int z_window = {vwap_short['z_window']};",
        f"            static constexpr int atr_period = {vwap_short['atr_period']};",
        f"            static constexpr float atr_mult = {f32(vwap_short['atr_mult'], 2)};",
        f"            static constexpr int ema_filter = {vwap_short['ema_filter']};",
        f"            static constexpr int vol_lookback = {vwap_short['vol_lookback']};",
        f"            static constexpr float vol_ratio_min = {f32(vwap_short['vol_ratio_min'], 2)};",
        "        };",
        "    };",
        "",
    ]

    # Ajout au CSV pour VWAP Reversal Pro
    for param in ['lookback', 'momentum', 'z_window', 'atr_period', 'ema_filter', 'vol_lookback']:
        add_to_csv("SPECIAL_6", "VWAPRevProConfig::Long", param, vwap_long[param], "int")
        add_to_csv("SPECIAL_6", "VWAPRevProConfig::Short", param, vwap_short[param], "int")

    add_to_csv("SPECIAL_6", "VWAPRevProConfig::Long", "atr_mult", vwap_long['atr_mult'])
    add_to_csv("SPECIAL_6", "VWAPRevProConfig::Long", "vol_ratio_min", vwap_long['vol_ratio_min'])
    add_to_csv("SPECIAL_6", "VWAPRevProConfig::Short", "atr_mult", vwap_short['atr_mult'])
    add_to_csv("SPECIAL_6", "VWAPRevProConfig::Short", "vol_ratio_min", vwap_short['vol_ratio_min'])

    # ── SPECIAL 7 : IMB Bull/Bear Light POC avec ATR ──────────────────────
    atr_high = params["atr_high"]
    atr_low = params["atr_low"]
    out += [
        "    // ================================================================",
        "    // 🎯 SPECIAL 7 - IMB Bull/Bear Light POC avec ATR",
        "    // ================================================================",
        "    struct ImbBullLightPocATRConfig {",
        "        // Seuils ATR",
        f"        static constexpr float atr_threshold_1 = {f32(atr_high['atr_threshold_1'], 1)};",
        f"        static constexpr float atr_threshold_2 = {f32(atr_high['atr_threshold_2'], 1)};",
        f"        static constexpr float atr_threshold_3 = {f32(atr_high['atr_threshold_3'], 1)};",
        "",
        "        // Différences pour SHORT (AtrHigh)",
        f"        static constexpr float diff_high_atr_1 = {f32(atr_high['diff_high_atr_1'], 2)};   // ATR < 1.5",
        f"        static constexpr float diff_high_atr_2 = {f32(atr_high['diff_high_atr_2'], 2)};  // 1.5 <= ATR < 1.7",
        f"        static constexpr float diff_high_atr_3 = {f32(atr_high['diff_high_atr_3'], 2)};  // 1.7 <= ATR < 1.9",
        f"        static constexpr float diff_high_atr_4 = {f32(atr_high['diff_high_atr_4'], 2)};  // ATR >= 1.9",
        "",
        "        // Différences pour LONG (AtrLow)",
        f"        static constexpr float diff_low_atr_1 = {f32(atr_low['diff_low_atr_1'], 2)};    // ATR < 1.5",
        f"        static constexpr float diff_low_atr_2 = {f32(atr_low['diff_low_atr_2'], 2)};   // 1.5 <= ATR < 1.7",
        f"        static constexpr float diff_low_atr_3 = {f32(atr_low['diff_low_atr_3'], 2)};   // 1.7 <= ATR < 1.9",
        f"        static constexpr float diff_low_atr_4 = {f32(atr_low['diff_low_atr_4'], 2)};   // ATR >= 1.9",
        "",
        "        // Fenêtre ATR",
        f"        static constexpr int atr_window = {atr_high['atr_window']};",
        "    };",
        "",
    ]

    # Ajout au CSV pour IMB Bull Light POC ATR
    for i in range(1, 4):
        add_to_csv("SPECIAL_7", "ImbBullLightPocATRConfig", f"atr_threshold_{i}", atr_high[f'atr_threshold_{i}'])
    for i in range(1, 5):
        add_to_csv("SPECIAL_7", "ImbBullLightPocATRConfig", f"diff_high_atr_{i}", atr_high[f'diff_high_atr_{i}'])
        add_to_csv("SPECIAL_7", "ImbBullLightPocATRConfig", f"diff_low_atr_{i}", atr_low[f'diff_low_atr_{i}'])
    add_to_csv("SPECIAL_7", "ImbBullLightPocATRConfig", "atr_window", atr_high['atr_window'], "int")

    # ── SPECIAL 8-11 : Volume Per Tick (optionnel - ajouté si présent) ─────
    if "volume_per_tick_light_short" in params:
        vpt_light_short = params["volume_per_tick_light_short"]
        vpt_light_long = params["volume_per_tick_light_long"]
        vpt_agg_short = params["volume_per_tick_aggressive_short"]
        vpt_agg_long = params["volume_per_tick_aggressive_long"]

        out += [
            "    // ================================================================",
            "    // 🎯 SPECIAL 8-11 - Volume Per Tick Configurations",
            "    // ================================================================",
            "    struct VolumePerTickConfig {",
            "        // Light Profile - Plus de trades",
            "        struct Light {",
            "            struct Short {",
            f"                static constexpr int period = {vpt_light_short['period']};",
            f"                static constexpr float threshold_low = {f32(vpt_light_short['threshold_low'], 2)};",
            f"                static constexpr float threshold_high = {f32(vpt_light_short['threshold_high'], 2)};",
            f"                static constexpr float expected_trade_pct = {f32(vpt_light_short['expected_trade_pct'], 1)};",
            f"                static constexpr float expected_winrate = {f32(vpt_light_short['expected_winrate'])};",
            "            };",
            "            struct Long {",
            f"                static constexpr int period = {vpt_light_long['period']};",
            f"                static constexpr float threshold_low = {f32(vpt_light_long['threshold_low'], 2)};",
            f"                static constexpr float threshold_high = {f32(vpt_light_long['threshold_high'], 2)};",
            f"                static constexpr float expected_trade_pct = {f32(vpt_light_long['expected_trade_pct'], 1)};",
            f"                static constexpr float expected_winrate = {f32(vpt_light_long['expected_winrate'])};",
            "            };",
            "        };",
            "",
            "        // Aggressive Profile - Moins de trades, meilleur WR",
            "        struct Aggressive {",
            "            struct Short {",
            f"                static constexpr int period = {vpt_agg_short['period']};",
            f"                static constexpr float threshold_low = {f32(vpt_agg_short['threshold_low'], 2)};",
            f"                static constexpr float threshold_high = {f32(vpt_agg_short['threshold_high'], 2)};",
            f"                static constexpr float expected_trade_pct = {f32(vpt_agg_short['expected_trade_pct'], 1)};",
            f"                static constexpr float expected_winrate = {f32(vpt_agg_short['expected_winrate'])};",
            "            };",
            "            struct Long {",
            f"                static constexpr int period = {vpt_agg_long['period']};",
            f"                static constexpr float threshold_low = {f32(vpt_agg_long['threshold_low'], 2)};",
            f"                static constexpr float threshold_high = {f32(vpt_agg_long['threshold_high'], 2)};",
            f"                static constexpr float expected_trade_pct = {f32(vpt_agg_long['expected_trade_pct'], 1)};",
            f"                static constexpr float expected_winrate = {f32(vpt_agg_long['expected_winrate'])};",
            "            };",
            "        };",
            "    };",
            "",
        ]

        # Ajout au CSV pour Volume Per Tick
        for config_type, config_data in [("Light::Short", vpt_light_short), ("Light::Long", vpt_light_long),
                                         ("Aggressive::Short", vpt_agg_short), ("Aggressive::Long", vpt_agg_long)]:
            add_to_csv("SPECIAL_8-11", f"VolumePerTickConfig::{config_type}", "period", config_data['period'], "int")
            add_to_csv("SPECIAL_8-11", f"VolumePerTickConfig::{config_type}", "threshold_low",
                       config_data['threshold_low'])
            add_to_csv("SPECIAL_8-11", f"VolumePerTickConfig::{config_type}", "threshold_high",
                       config_data['threshold_high'])
            add_to_csv("SPECIAL_8-11", f"VolumePerTickConfig::{config_type}", "expected_trade_pct",
                       config_data['expected_trade_pct'])
            add_to_csv("SPECIAL_8-11", f"VolumePerTickConfig::{config_type}", "expected_winrate",
                       config_data['expected_winrate'])

    # ── SPECIAL 12 : Microstructure Anti-Spring & Anti-Épuisement ────────
    micro_antispring = params["micro_antispring"]
    micro_antiepuisement = params["micro_antiepuisement"]
    out += [
        "    // ================================================================",
        "    // 🎯 SPECIAL 12 - Microstructure Anti-Spring & Anti-Épuisement",
        "    // ================================================================",
        "    struct MicrostructureConfig {",
        "        // Anti-Spring Configuration (SHORT)",
        "        struct AntiSpring {",
        f"            static constexpr int period = {micro_antispring['short_period']};",
        f"            static constexpr float vol_threshold = {f32(micro_antispring['short_vol_threshold'])};",
        f"            static constexpr float dur_threshold = {f32(micro_antispring['short_dur_threshold'])};",
        f"            static constexpr int condition = {micro_antispring['short_condition']};  // {micro_antispring['short_condition']} = V<D<",
        "        };",
        "",
        "        // Anti-Épuisement Configuration (LONG)",
        "        struct AntiEpuisement {",
        f"            static constexpr int period = {micro_antiepuisement['long_period']};",
        f"            static constexpr float vol_threshold = {f32(micro_antiepuisement['long_vol_threshold'])};",
        f"            static constexpr float dur_threshold = {f32(micro_antiepuisement['long_dur_threshold'])};",
        f"            static constexpr int condition = {micro_antiepuisement['long_condition']};  // {micro_antiepuisement['long_condition']} = V>D>",
        "        };",
        "    };",
        "",
    ]

    # Ajout au CSV pour Microstructure
    add_to_csv("SPECIAL_12", "MicrostructureConfig::AntiSpring", "period", micro_antispring['short_period'], "int")
    add_to_csv("SPECIAL_12", "MicrostructureConfig::AntiSpring", "vol_threshold",
               micro_antispring['short_vol_threshold'])
    add_to_csv("SPECIAL_12", "MicrostructureConfig::AntiSpring", "dur_threshold",
               micro_antispring['short_dur_threshold'])
    add_to_csv("SPECIAL_12", "MicrostructureConfig::AntiSpring", "condition", micro_antispring['short_condition'],
               "int")

    add_to_csv("SPECIAL_12", "MicrostructureConfig::AntiEpuisement", "period", micro_antiepuisement['long_period'],
               "int")
    add_to_csv("SPECIAL_12", "MicrostructureConfig::AntiEpuisement", "vol_threshold",
               micro_antiepuisement['long_vol_threshold'])
    add_to_csv("SPECIAL_12", "MicrostructureConfig::AntiEpuisement", "dur_threshold",
               micro_antiepuisement['long_dur_threshold'])
    add_to_csv("SPECIAL_12", "MicrostructureConfig::AntiEpuisement", "condition",
               micro_antiepuisement['long_condition'], "int")

    # ── Résultats de validation ──────────────────────────────────────────────
    out += [
        "    // ================================================================",
        "    // 📊 RÉSULTATS DE VALIDATION SUR DATASET TEST",
        "    // ================================================================",
        "    /*",
        "    RÉCAPITULATIF DES PERFORMANCES (Dataset TEST):",
        "",
        "    ✅ VALIDÉ:",
        "    1. Light Short (Imbalance): WR=56.76%, PCT=4.26%, Trades=74",
        "    2. Light Long (Imbalance): WR=72.22%, PCT=1.16%, Trades=18",
        "    3. Aggressive Short (Imbalance): WR=60.00%, PCT=0.58%, Trades=10",
        "    4. MFI Long OVERSOLD: WR=53.03%, PCT=8.50%, Trades=132",
        "    5. MFI Short OVERBOUGHT: WR=52.02%, PCT=14.28%, Trades=248",
        "    6. RS Short RANGE: WR=58.33%, PCT=8.98%, Trades=156",
        "    7. RS Long RANGE: WR=61.54%, PCT=9.21%, Trades=143",
        "    8. Williams %R Short OVERBOUGHT: WR=55.37%, PCT=13.93%, Trades=242",
        "    9. VWAP Reversal Pro LONG: WR=55.35%, PCT=10.24%, Trades=159",
        "    10. VWAP Reversal Pro SHORT: WR=54.41%, PCT=3.91%, Trades=68",
        "    11. Volume Per Tick Light SHORT: WR=55.00%, PCT=8.50%, Trades=~8.5%",
        "    12. Microstructure Anti-Spring SHORT: Optimisé pour détection de retournements",
        "    13. Microstructure Anti-Épuisement LONG: Optimisé pour détection de retournements",
        "",
        "    ❌ REJETÉ:",
        "    - Williams %R Long OVERSOLD: WR=51.28% < 52.5% requis",
        "    */",
        "",
        "} // namespace Special_indicator_hardcode",
    ]

    # ── écriture du fichier .h ────────────────────────────────────────────────────
    content = "\n".join(out)
    Path(output_path).write_text(content, encoding="utf-8")
    print("✅ SpecialIndicator header généré avec SPECIAL 12 :", output_path)

    # ── écriture du fichier CSV ────────────────────────────────────────────────────
    df = pd.DataFrame(csv_data)

    # Réorganisation des colonnes pour plus de clarté
    column_order = ["config_name", "struct_name", "param_name", "value", "data_type", "namespace"]
    df = df[column_order]

    # Écriture du CSV avec séparateur point-virgule
    df.to_csv(output_csv_path, index=False, encoding="utf-8", sep=";")
    print("✅ Fichier CSV de configuration généré :", output_csv_path)
    print(f"📊 Total de {len(csv_data)} paramètres exportés vers le CSV")


# ────────────────────────────────────────────────────────────────────────────────
# 🔧 FONCTION POUR GÉNÉRER LE CHARGEUR CSV (optionnel)
# ────────────────────────────────────────────────────────────────────────────────

def generate_csv_loader_header(csv_path: str, output_h_path: str) -> None:
    """
    Génère un fichier .h qui contient un namespace Special_indicator_fromCsv
    avec des fonctions pour charger les paramètres depuis le CSV
    """

    # Lecture du CSV avec séparateur point-virgule
    df = pd.read_csv(csv_path, sep=";")

    out = [
        "#pragma once",
        "#include <map>",
        "#include <string>",
        "#include <fstream>",
        "#include <sstream>",
        "",
        "// ================================================================",
        "// 📋  CHARGEUR DE CONFIGURATION DEPUIS CSV - AUTOGÉNÉRÉ",
        "// ================================================================",
        "// Ce fichier permet de charger les paramètres depuis un CSV",
        "// pour remplacer dynamiquement le namespace Special_indicator_hardcode",
        "// ================================================================",
        "",
        "namespace Special_indicator_fromCsv {",
        "",
        "    // Structure pour stocker les paramètres chargés depuis le CSV",
        "    struct ConfigParam {",
        "        std::string value_str;",
        "        float value_float = 0.0f;",
        "        int value_int = 0;",
        "        std::string data_type;",
        "    };",
        "",
        "    // Map globale pour stocker tous les paramètres",
        "    static std::map<std::string, ConfigParam> g_config_params;",
        "    static bool g_config_loaded = false;",
        "",
        "    // Fonction pour charger le CSV",
        "    bool LoadConfigFromCSV(const std::string& csv_path) {",
        "        std::ifstream file(csv_path);",
        "        if (!file.is_open()) {",
        "            return false;",
        "        }",
        "",
        "        g_config_params.clear();",
        "        std::string line;",
        "        std::getline(file, line); // Skip header",
        "",
        "        while (std::getline(file, line)) {",
        "            std::stringstream ss(line);",
        "            std::string config_name, struct_name, param_name, value_str, data_type, namespace_name;",
        "",
        "            std::getline(ss, config_name, ';');",
        "            std::getline(ss, struct_name, ';');",
        "            std::getline(ss, param_name, ';');",
        "            std::getline(ss, value_str, ';');",
        "            std::getline(ss, data_type, ';');",
        "            std::getline(ss, namespace_name, ';');",
        "",
        "            std::string key = struct_name + \"::\" + param_name;",
        "            ConfigParam param;",
        "            param.value_str = value_str;",
        "            param.data_type = data_type;",
        "",
        "            if (data_type == \"int\") {",
        "                param.value_int = std::stoi(value_str);",
        "            } else {",
        "                param.value_float = std::stof(value_str);",
        "            }",
        "",
        "            g_config_params[key] = param;",
        "        }",
        "",
        "        g_config_loaded = true;",
        "        return true;",
        "    }",
        "",
        "    // Fonctions helper pour récupérer les valeurs",
        "    float GetFloat(const std::string& struct_name, const std::string& param_name) {",
        "        if (!g_config_loaded) return 0.0f;",
        "        std::string key = struct_name + \"::\" + param_name;",
        "        auto it = g_config_params.find(key);",
        "        return (it != g_config_params.end()) ? it->second.value_float : 0.0f;",
        "    }",
        "",
        "    int GetInt(const std::string& struct_name, const std::string& param_name) {",
        "        if (!g_config_loaded) return 0;",
        "        std::string key = struct_name + \"::\" + param_name;",
        "        auto it = g_config_params.find(key);",
        "        return (it != g_config_params.end()) ? it->second.value_int : 0;",
        "    }",
        "",
    ]

    # Génération des structures avec les mêmes noms mais valeurs dynamiques
    structures = df['struct_name'].unique()

    for struct in structures:
        struct_params = df[df['struct_name'] == struct]
        clean_struct_name = struct.replace('::', '_').replace('_', '')

        out.append(f"    // Structure {struct}")
        out.append(f"    struct {clean_struct_name} {{")

        for _, row in struct_params.iterrows():
            param_name = row['param_name']
            data_type_str = "int" if row['data_type'] == "int" else "float"

            out.append(f"        static {data_type_str} {param_name}() {{")
            if row['data_type'] == "int":
                out.append(f"            return GetInt(\"{struct}\", \"{param_name}\");")
            else:
                out.append(f"            return GetFloat(\"{struct}\", \"{param_name}\");")
            out.append("        }")

        out.append("    };")
        out.append("")

    out += [
        "} // namespace Special_indicator_fromCsv",
        "",
        "// ================================================================",
        "// 🔧 UTILISATION:",
        "// ================================================================",
        "// 1. Charger le CSV au début de votre programme:",
        "//    Special_indicator_fromCsv::LoadConfigFromCSV(\"path/to/config.csv\");",
        "",
        "// 2. Utiliser les paramètres:",
        "//    float vol1 = Special_indicator_fromCsv::LightShortConfig::volume_1();",
        "//    int period = Special_indicator_fromCsv::MFIConfigLong::period();",
        "// ================================================================",
    ]

    content = "\n".join(out)
    Path(output_h_path).write_text(content, encoding="utf-8")
    print("✅ Chargeur CSV header généré :", output_h_path)



# ════════════════ FONCTIONS VOLUME PER TICK FACTORÉES ═══════════════
@njit(parallel=True, fastmath=True)
def _compute_ratio_numba(vpt: np.ndarray, starts: np.ndarray, period: int) -> np.ndarray:
    """
    Version Numba optimisée du calcul de ratio volume per tick.

    Args:
        vpt: Array du volume per tick
        starts: Array boolean des débuts de session
        period: Période de la moyenne mobile

    Returns:
        Array des ratios (current_vpt / average_vptRatio_over_period)
    """
    n = vpt.size
    out = np.empty(n, np.float32)
    last_session_start = 0

    for i in prange(n):
        # Mise à jour du début de session
        if starts[i]:
            last_session_start = i

        # Calcul de l'index de début pour la période
        start_idx = last_session_start if i - period + 1 < last_session_start else i - period + 1
        if start_idx < 0:
            start_idx = 0

        # Calcul de la moyenne sur la période
        total = 0.0
        for j in range(start_idx, i + 1):
            total += vpt[j]

        average = total / (i - start_idx + 1)

        # Calcul du ratio
        out[i] = vpt[i] / average if average > 0 else 1.0

    return out


def prepare_arrays_for_optimization(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prépare les arrays pré-calculés pour l'optimisation (évite de recalculer à chaque trial).

    Args:
        df: DataFrame source

    Returns:
        Tuple (volume_per_tick, session_starts) prêt pour compute_ratio_from_arrays
    """
    volume = pd.to_numeric(df["sc_volume"], errors="coerce").fillna(0).to_numpy(np.float32)
    candle_size = pd.to_numeric(df["sc_candleSizeTicks"], errors="coerce").fillna(1).to_numpy(np.float32)
    candle_size[candle_size == 0] = 1

    volume_per_tick = volume / (2.0 * candle_size)
    session_starts = (df["sc_sessionStartEnd"].to_numpy(np.int8) == 10)

    return volume_per_tick, session_starts


def compute_ratio_from_arrays(vptRatio_array: np.ndarray, starts_array: np.ndarray, period: int) -> np.ndarray:
    """
    Calcule le ratio à partir d'arrays pré-calculés (pour l'optimisation).

    Args:
        vptRatio_array: Array volume per tick pré-calculé
        starts_array: Array session starts pré-calculé
        period: Période pour la moyenne mobile

    Returns:
        Array des ratios normalisés
    """
    ratios = _compute_ratio_numba(vptRatio_array, starts_array, period)
    return np.where(np.isnan(ratios) | np.isinf(ratios), 1.0, ratios)

#
# def compute_volPerTickOverPeriod(df: pd.DataFrame, period: int) -> np.ndarray:
#     """
#     Calcule le ratio volume per tick normalisé sur une période glissante.
#
#     Le calcul se fait en deux étapes :
#     1. Volume per tick = volume / (2 * taille_chandelle_en_ticks)
#     2. Ratio = vol_per_tick_actuel / moyenne_vol_per_tick_sur_période
#
#     Args:
#         df: DataFrame contenant les colonnes :
#             - sc_volume : volume des chandelles
#             - sc_candleSizeTicks : taille en ticks des chandelles
#             - sc_sessionStartEnd : indicateur début/fin de session (10 = début)
#         period: Période pour la moyenne mobile
#
#     Returns:
#         Array des ratios volume per tick normalisés
#     """
#     vptRatio_array, starts_array = prepare_arrays_for_optimization(df)
#     return compute_ratio_from_arrays(vptRatio_array, starts_array, period)





def prepare_microstructure_data(df):
    """
    Prépare les données pour les calculs de microstructure
    Version mutualisée pour éviter la duplication

    Args:
        df: DataFrame contenant les données OHLCV et sessions

    Returns:
        tuple: (durations, volumes, session_starts) prêts pour calculate_slopes_and_r2_numba
    """
    print(f"🔧 Préparation données microstructure...")

    # Session starts - conversion explicite en numpy boolean
    session_starts = (df['sc_sessionStartEnd'] == 10).values.astype(np.bool_)

    # Durées et volumes - SOURCES STANDARD
    durations = pd.to_numeric(df['sc_candleDuration'], errors='coerce').fillna(0).values.astype(np.float64)
    volumes = pd.to_numeric(df['sc_volume_perTick'], errors='coerce').fillna(0).values.astype(np.float64)

    print(f"   - Durations: {durations.shape} valeurs [{np.min(durations):.2f}, {np.max(durations):.2f}]")
    print(f"   - Volumes: {volumes.shape} valeurs [{np.min(volumes):.2f}, {np.max(volumes):.2f}]")
    print(f"   - Session starts: {np.sum(session_starts)} sessions détectées")

    return durations, volumes, session_starts


def calculate_slopes_with_optimized_function(durations, volumes, session_starts, n_periods):
    """
    Utilise la fonction calculate_slopes_and_r2_numba optimisée
    Version mutualisée pour éviter la duplication de code
    """
    from colorama import Fore, Style

    try:
        print(f"   🔧 Appel calculate_slopes_and_r2_numba (période {n_periods})...")

        # Vérifications préliminaires
        if not isinstance(durations, np.ndarray):
            raise TypeError(f"durations doit être np.ndarray, reçu: {type(durations)}")
        if not isinstance(volumes, np.ndarray):
            raise TypeError(f"volumes doit être np.ndarray, reçu: {type(volumes)}")
        if not isinstance(session_starts, np.ndarray):
            # Conversion automatique si c'est un BooleanArray pandas
            if hasattr(session_starts, 'values'):
                print(f"   ⚠️ Conversion automatique de {type(session_starts)} vers np.ndarray")
                session_starts = session_starts.values.astype(np.bool_)
            else:
                raise TypeError(f"session_starts doit être np.ndarray, reçu: {type(session_starts)}")

        # Vérification des dtypes NumPy
        if session_starts.dtype != np.bool_:
            print(f"   ⚠️ Conversion dtype de {session_starts.dtype} vers bool")
            session_starts = session_starts.astype(np.bool_)

        # Appel à la fonction optimisée - CONFIGURATION STANDARD
        duration_slopes, r2_dur, std_dur = calculate_slopes_and_r2_numba(
            durations, session_starts, n_periods,
            clip_slope=False,  # Pas de clipping pour avoir les vraies valeurs
            include_close_bar=True
        )

        volume_slopes, r2_vol, std_vol = calculate_slopes_and_r2_numba(
            volumes, session_starts, n_periods,
            clip_slope=False,  # Pas de clipping pour avoir les vraies valeurs
            include_close_bar=True
        )

        # Vérifications des résultats
        if duration_slopes is None or volume_slopes is None:
            raise ValueError("calculate_slopes_and_r2_numba a retourné None")

        # Gestion des NaN (identique à l'optimiseur)
        volume_slopes = np.nan_to_num(volume_slopes, nan=0.0)
        duration_slopes = np.nan_to_num(duration_slopes, nan=0.0)

        print(f"   ✅ Slopes: Vol[{np.min(volume_slopes):.4f}, {np.max(volume_slopes):.4f}], "
              f"Dur[{np.min(duration_slopes):.4f}, {np.max(duration_slopes):.4f}]")

        return volume_slopes, duration_slopes, r2_vol, r2_dur

    except Exception as e:
        print(f"{Fore.RED}❌ ERREUR CRITIQUE dans calculate_slopes_and_r2_numba:{Style.RESET_ALL}")
        print(f"   Type d'erreur: {type(e).__name__}")
        print(f"   Message: {str(e)}")
        print(f"   Paramètres fournis:")
        print(f"     - durations: type={type(durations)}, shape={getattr(durations, 'shape', 'N/A')}")
        print(f"     - volumes: type={type(volumes)}, shape={getattr(volumes, 'shape', 'N/A')}")
        print(f"     - session_starts: type={type(session_starts)}, shape={getattr(session_starts, 'shape', 'N/A')}")
        print(f"     - n_periods: {n_periods}")

        # ARRÊT SANS FALLBACK
        raise RuntimeError(f"ARRÊT CRITIQUE: calculate_slopes_and_r2_numba a échoué - {str(e)}") from e


def detect_microstructure_pattern(volume_slopes, duration_slopes,
                                  vol_threshold, dur_threshold, condition_code,
                                  anti_mode=True):
    """
    Détecte les patterns de microstructure (spring/épuisement)
    Version mutualisée pour éviter la duplication
    """
    # Application des conditions
    vol_condition = apply_microstructure_condition_logic(volume_slopes, vol_threshold, condition_code)
    dur_condition = apply_microstructure_condition_logic(duration_slopes, dur_threshold, condition_code)

    # Pattern détecté = conditions volume ET durée remplies
    pattern_detected = vol_condition & dur_condition

    if anti_mode:
        # Mode ANTI: 1=filtre les patterns détectés, 0=garde
        return np.where(~pattern_detected, 1, 0)
    else:
        # Mode DIRECT: 1=garde les patterns détectés, 0=filtre
        return np.where(pattern_detected, 1, 0)


def apply_microstructure_condition_logic(slope_values, threshold, condition_code):
    """
    Applique la logique de condition basée sur le code de l'optimiseur:
    0 = V<D< (volume < seuil ET durée < seuil)
    1 = V>D< (volume > seuil ET durée < seuil)
    2 = V<D> (volume < seuil ET durée > seuil)
    3 = V>D> (volume > seuil ET durée > seuil)

    Pour un seul type de slope, on prend la partie correspondante:
    - condition 0,2: slope < threshold (V< ou D<)
    - condition 1,3: slope > threshold (V> ou D>)
    """
    if condition_code in [0, 2]:  # V< conditions
        return slope_values < threshold
    elif condition_code in [1, 3]:  # V> conditions
        return slope_values > threshold
    else:
        raise ValueError(f"Condition code invalide: {condition_code}. Doit être 0, 1, 2 ou 3.")


# ----------------------------------------------------------------------------
# SPECIAL 12 – Microstructure Anti-Spring (Short) & Anti-Épuisement (Long)
# ----------------------------------------------------------------------------

def add_micro_antiSpringEpuissement(df, features_df,
                                    short_period, short_vol_threshold, short_dur_threshold, short_condition,
                                    long_period, long_vol_threshold, long_dur_threshold, long_condition):
    """
    Ajoute les indicateurs de microstructure pour détecter l'anti-spring (short)
    et l'anti-épuisement (long) basés sur les slopes de volume et durée.

    Version simplifiée utilisant les fonctions mutualisées de Tools.func_features_preprocessing
    """

    from Tools.func_features_preprocessing import (
        prepare_microstructure_data,
        calculate_slopes_with_optimized_function,
        detect_microstructure_pattern
    )

    if short_period is None or long_period is None:
        raise ValueError("Les périodes short et long doivent être spécifiées")

    try:
        # === PRÉPARATION DES DONNÉES (FONCTION MUTUALISÉE) ===
        durations, volumes, session_starts = prepare_microstructure_data(df)

        # === CALCUL SLOPES POUR SHORT (ANTI-SPRING) ===
        print(f"🔧 Calcul slopes SHORT (période {short_period})...")
        volume_slopes_short, duration_slopes_short, r2_vol_short, r2_dur_short = calculate_slopes_with_optimized_function(
            durations, volumes, session_starts, short_period
        )

        # === CALCUL SLOPES POUR LONG (ANTI-ÉPUISEMENT) ===
        print(f"🔧 Calcul slopes LONG (période {long_period})...")
        volume_slopes_long, duration_slopes_long, r2_vol_long, r2_dur_long = calculate_slopes_with_optimized_function(
            durations, volumes, session_starts, long_period
        )

        # === DÉTECTION PATTERNS (FONCTIONS MUTUALISÉES) ===
        # Anti-spring short (mode anti = filtre les springs)
        features_df['is_antiSpring_short'] = detect_microstructure_pattern(
            volume_slopes_short, duration_slopes_short,
            short_vol_threshold, short_dur_threshold, short_condition,
            anti_mode=True
        )
        # Anti-épuisement long (mode anti = filtre les épuisements)
        features_df['is_antiEpuisement_long'] = detect_microstructure_pattern(
            volume_slopes_long, duration_slopes_long,
            long_vol_threshold, long_dur_threshold, long_condition,
            anti_mode=True
        )

        # === AJOUT DES SLOPES BRUTES ===
        # Slopes anti-spring short
        features_df['volume_slope_antiSpring_short'] = volume_slopes_short
        features_df['duration_slope_antiSpring_short'] = duration_slopes_short

        # Slopes anti-épuisement long
        features_df['volume_slope_antiEpuisement_long'] = volume_slopes_long
        features_df['duration_slope_antiEpuisement_long'] = duration_slopes_long

        # === NORMALISATION FINALE ===
        columns = [
            'is_antiSpring_short', 'is_antiEpuisement_long',
            'volume_slope_antiSpring_short', 'duration_slope_antiSpring_short',
            'volume_slope_antiEpuisement_long', 'duration_slope_antiEpuisement_long'
        ]
        for col in columns:
            features_df[col] = pd.to_numeric(features_df[col], errors='coerce').fillna(0)

        # === STATISTIQUES ===
        print(f"✅ Microstructure calculé:")
        print(
            f"   Anti-spring short: {features_df['is_antiSpring_short'].sum()}/{len(features_df)} filtrages ({features_df['is_antiSpring_short'].mean():.1%})")
        print(
            f"   Anti-épuisement long: {features_df['is_antiEpuisement_long'].sum()}/{len(features_df)} filtrages ({features_df['is_antiEpuisement_long'].mean():.1%})")

        # Statistiques des slopes
        print(
            f"   Slopes short - Vol: [{features_df['volume_slope_antiSpring_short'].min():.4f}, {features_df['volume_slope_antiSpring_short'].max():.4f}]")
        print(
            f"   Slopes short - Dur: [{features_df['duration_slope_antiSpring_short'].min():.4f}, {features_df['duration_slope_antiSpring_short'].max():.4f}]")
        print(
            f"   Slopes long - Vol: [{features_df['volume_slope_antiEpuisement_long'].min():.4f}, {features_df['volume_slope_antiEpuisement_long'].max():.4f}]")
        print(
            f"   Slopes long - Dur: [{features_df['duration_slope_antiEpuisement_long'].min():.4f}, {features_df['duration_slope_antiEpuisement_long'].max():.4f}]")

    except Exception as e:
        print(f"Erreur dans add_micro_antiSpringEpuissement: {str(e)}")
        raise

    return features_df