import pandas as pd
import numpy as np
import optuna
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import datetime

warnings.filterwarnings('ignore')
from func_standard import *
import numpy as np


########################################################
# CALLBACK POUR LES LOGS TOUTES LES 100 ITERATIONS
########################################################
def logging_callback(study, trial):
    if trial.number % 100 == 0:
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        val = trial.value
        if val is None or not np.isfinite(val):
            value_str = str(val)
        else:
            value_str = f"{val:.2f}"

        askbid_low_sorted = trial.user_attrs.get("askbid_low_sorted", None)
        askbid_high_sorted = trial.user_attrs.get("askbid_high_sorted", None)
        pullstack_low_sorted = trial.user_attrs.get("pullstack_low_sorted", None)
        pullstack_high_sorted = trial.user_attrs.get("pullstack_high_sorted", None)

        best_trial_number = study.best_trial.number
        best_val = study.best_value
        if best_val is None or not np.isfinite(best_val):
            best_value_str = str(best_val)
        else:
            best_value_str = f"{best_val:.2f}"

        print(
            f"[I {now_str}] Trial {trial.number} finished with value: {value_str} "
            f"and used parameters (sorted): "
            f"askbid=[{askbid_low_sorted}, {askbid_high_sorted}], "
            f"pullstack=[{pullstack_low_sorted}, {pullstack_high_sorted}]. "
            f"Best is trial {best_trial_number} with value: {best_value_str}"
        )

########################################################
# VALIDATION DES SESSIONS (10->20)
########################################################
def validate_sessions_10_20_only(df):
    df_10_20 = df[df['SessionStartEnd'].isin([10, 20])].copy()
    if not pd.api.types.is_datetime64_any_dtype(df_10_20['date']):
        df_10_20['date'] = pd.to_datetime(df_10_20['date'])
    df_10_20 = df_10_20.sort_values('date').reset_index(drop=True)

    current_session_start_day = None
    current_session_start_ts = None
    sessions_data = []

    for _, row in df_10_20.iterrows():
        row_ts = row['date']
        row_day = row_ts.normalize()
        status = row['SessionStartEnd']

        if status == 10:
            if current_session_start_day is not None:
                raise Exception("Session incomplète : nouveau '10' avant un '20'.")
            current_session_start_day = row_day
            current_session_start_ts = row_ts

        elif status == 20:
            if current_session_start_day is None:
                raise Exception("Barre '20' sans '10'.")
            if (row_day - current_session_start_day).days != 1:
                raise Exception("Dates incohérentes...")

            sessions_data.append({
                'start_date': current_session_start_day,
                'end_date': row_day,
                'start_bar_ts': current_session_start_ts,
                'end_bar_ts': row_ts
            })
            current_session_start_day = None
            current_session_start_ts = None

    if current_session_start_day is not None:
        raise Exception("Session incomplète : dernier '10' sans '20'.")

    return sessions_data


@njit(parallel=True)
def calculate_max_drawdown_numba(pnl_array):
    peak = pnl_array[0]
    max_dd = 0.0
    for i in prange(len(pnl_array)):
        val = pnl_array[i]
        if val > peak:
            peak = val
        dd = peak - val
        if dd > max_dd:
            max_dd = dd
    return max_dd

########################################################
# 1) STATS SESSIONS (SANS DÉTAIL) => AVANT OPTIMISATION
########################################################
def compute_and_print_sessions_stats_no_details(df, sessions_data):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    global_pnl = 0.0
    global_history = []

    print("\n=== STATS DES SESSIONS ===")
    for i, sess in enumerate(sessions_data, start=1):
        # Timestamps réels pour éviter l'exclusion des trades de début et fin
        start_ts = sess['start_bar_ts']  # ex: 2025-01-05 23:00:00
        end_ts   = sess['end_bar_ts']    # ex: 2025-01-06 21:59:46

        # Correction : On filtre avec les vrais timestamps au lieu de start_day/end_day
        # On ignore les "pas de trade" (class_binaire=99)
        rows_sess = df[
            (df['date'] >= start_ts) &
            (df['date'] <= end_ts) &
            (df['class_binaire'].isin([0, 1]))
            ]
        # Vérification : nombre de trades capturés dans la session
        print(f"Session {i}: de {start_ts} à {end_ts} → Trades trouvés : {len(rows_sess)}")

        # PNL de la session
        session_pnl = rows_sess['trade_pnl'].sum()
        session_cum = rows_sess['trade_pnl'].cumsum()
        # session_cum est un Series
        session_cum_np = session_cum.to_numpy()  # ou session_cum.values

        # On passe le tableau NumPy à la fonction Numba
        session_dd = calculate_max_drawdown_numba(session_cum_np)

        trades_real = rows_sess[df['trade_category'].str.contains("Trades échoués|Trades réussis", na=False)]
        nb_trades = len(trades_real)
        nb_wins = (trades_real['trade_pnl'] > 0).sum()
        if nb_trades > 0:
            win_rate = 100.0 * nb_wins / nb_trades
            expected_pnl = session_pnl / nb_trades
        else:
            win_rate = 0.0
            expected_pnl = 0.0

        # Au fur et à mesure :
        global_history.append(global_pnl)

        # Puis, au moment de calculer le drawdown :
        import numpy as np

        global_history_np = np.array(global_history, dtype=np.float64)
        global_dd = calculate_max_drawdown_numba(global_history_np)



        # Affichage des statistiques avec timestamps réels
        print(f"\nSession #{i} du {end_ts.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  - Date début session        : {start_ts.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  - Date fin session          : {end_ts.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  - Win Rate (fin session)    : {win_rate:.2f}%")
        print(f"  - PNL fin de session        : {session_pnl:.2f}")
        print(f"  - Max drawdown (session)    : {session_dd:.2f}")
        print(f"  - Nombre de trades pris     : {nb_trades}")
        print(f"  - Expected PNL/trade        : {expected_pnl:.2f}")
        print(f"  - PNL cumulé historique     : {global_pnl:.2f}")
        print(f"  - Max drawdown historique   : {global_dd:.2f}")



########################################################
# 2) STATS SESSIONS (AVEC DÉTAIL) => APRES OPTIMISATION
########################################################
def compute_and_print_sessions_stats_with_details(df_opt, sessions_data):
    """
    Reprend les sessions trouvées sur df initial et applique les calculs sur df_opt.
    Utilise start_bar_ts et end_bar_ts pour inclure correctement tous les trades.
    """
    df_opt = df_opt.copy().sort_values('date')
    if not pd.api.types.is_datetime64_any_dtype(df_opt['date']):
        df_opt['date'] = pd.to_datetime(df_opt['date'])

    global_pnl = 0.0
    global_history = []

    print("\n=== STATS DES SESSIONS (APRES OPTIMISATION, AVEC DETAILS, MÊMES BORNES) ===")

    for i, sess in enumerate(sessions_data, start=1):
        start_ts = sess['start_bar_ts']  # ex: 2025-01-05 23:00:00
        end_ts = sess['end_bar_ts']  # ex: 2025-01-06 21:59:46

        # Correction : Filtrage avec les vrais timestamps !
        rows_sess = df_opt[
            (df_opt['date'] >= start_ts) &
            (df_opt['date'] <= end_ts) &
            (df_opt['class_binaire'].isin([0, 1]))  # On garde uniquement les trades passés (réussis=1 ou échoués=0)
            ]
        trades_real = rows_sess[rows_sess['trade_category'].str.contains("Trades échoués|Trades réussis", na=False)]
        # Vérifier combien de trades sont bien pris
        print(f"Session {i}: de {start_ts} à {end_ts} → Trades trouvés : {len(rows_sess)}")

        # PNL de la session
        session_pnl = trades_real['trade_pnl'].sum()
        session_cum = trades_real['trade_pnl'].cumsum()
        session_dd = calculate_max_drawdown_numba(session_cum)

        nb_trades = len(trades_real)
        winning_trades = trades_real[trades_real['trade_pnl'] > 0]
        losing_trades = trades_real[trades_real['trade_pnl'] <= 0]

        nb_wins = len(winning_trades)
        nb_losses = len(losing_trades)

        if nb_trades > 0:
            win_rate = 100.0 * nb_wins / nb_trades
            expected_pnl = session_pnl / nb_trades
        else:
            win_rate = 0.0
            expected_pnl = 0.0

        # Mise à jour global
        global_pnl += session_pnl
        global_history.append(global_pnl)
        global_dd = calculate_max_drawdown_numba(pd.Series(global_history))

        # Affichage des résultats avec vrais timestamps
        print(f"\nSession #{i} du {end_ts.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  - Date début session        : {start_ts.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  - Date fin session          : {end_ts.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  - Win Rate (fin session)    : {win_rate:.2f}%")
        print(f"  - PNL fin de session        : {session_pnl:.2f}")
        print(f"  - Max drawdown (session)    : {session_dd:.2f}")
        print(f"  - Nombre de trades pris     : {nb_trades}")

        print(f"  - Trades gagnés             : {nb_wins}")
        for _, row_w in winning_trades.iterrows():
            print(f"      * {row_w['date']} => PNL={row_w['trade_pnl']:.2f}")

        print(f"  - Trades échoués            : {nb_losses}")
        for _, row_l in losing_trades.iterrows():
            print(f"      * {row_l['date']} => PNL={row_l['trade_pnl']:.2f}")

        print(f"  - Expected PNL/trade        : {expected_pnl:.2f}")
        print(f"  - PNL cumulé historique     : {global_pnl:.2f}")
        print(f"  - Max drawdown historique   : {global_dd:.2f}")


########################################################
# OPTIMISATION
########################################################
class RatioOptimizer:
    def __init__(self, shorts_df):
        self.shorts_df = shorts_df[shorts_df['class_binaire'].isin([0,1])].copy()
        self.optimization_history = []
        self.best_result = None

    def objective(self, trial):
        askbid_low_raw = trial.suggest_float('askbid_low', 0, 291)
        askbid_high_raw = trial.suggest_float('askbid_high', 0, 291)
        pullstack_low_raw = trial.suggest_float('pullstack_low', -63, 50)
        pullstack_high_raw = trial.suggest_float('pullstack_high', -63, 50)

        askbid_low, askbid_high = sorted([askbid_low_raw, askbid_high_raw])
        pullstack_low, pullstack_high = sorted([pullstack_low_raw, pullstack_high_raw])

        trades_in_range = self.shorts_df[
            (self.shorts_df['cumDOM_AskBid_avgRatio'].between(askbid_low, askbid_high)) &
            (self.shorts_df['cumDOM_AskBid_pullStack_avgDiff_ratio'].between(pullstack_low, pullstack_high))
        ]
        n_trades = len(trades_in_range)
        if n_trades < 10:
            return float('-inf')

        total_pnl = trades_in_range['trade_pnl'].sum()
        win_rate = (trades_in_range['trade_pnl'] > 0).mean() * 100
        avg_pnl = total_pnl / n_trades

        trial.set_user_attr("askbid_low_sorted", askbid_low)
        trial.set_user_attr("askbid_high_sorted", askbid_high)
        trial.set_user_attr("pullstack_low_sorted", pullstack_low)
        trial.set_user_attr("pullstack_high_sorted", pullstack_high)

        result = {
            'askbid_range_used': (askbid_low, askbid_high),
            'pullstack_range_used': (pullstack_low, pullstack_high),
            'total_pnl': total_pnl,
            'num_trades': n_trades,
            'win_rate': win_rate,
            'avg_pnl': avg_pnl
        }
        self.optimization_history.append(result)

        if (self.best_result is None) or (total_pnl > self.best_result['total_pnl']):
            self.best_result = result

        return total_pnl

    def optimize(self, n_trials=10000):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trials, callbacks=[logging_callback],n_jobs=-1)
        return study


########################################################
# PLOTLY
########################################################
def analyze_and_visualize_results(optimizer):
    import pandas as pd
    history = pd.DataFrame(optimizer.optimization_history)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Distribution PNL vs Ratios (3D)',
            'PNL vs Nombre de Trades',
            'Win Rate vs PNL',
            'Distribution des Plages (Heatmap)'
        ),
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter'}],
               [{'type': 'scatter'}, {'type': 'heatmap'}]]
    )

    # 1. Scatter 3D
    fig.add_trace(
        go.Scatter3d(
            x=history['askbid_range_used'].apply(lambda x: (x[0] + x[1]) / 2),
            y=history['pullstack_range_used'].apply(lambda x: (x[0] + x[1]) / 2),
            z=history['total_pnl'],
            mode='markers',
            marker=dict(
                size=8,
                color=history['win_rate'],
                colorscale='Viridis',
                opacity=0.8,
                showscale=True,
                colorbar=dict(title='Win Rate')
            ),
            name='PNL'
        ),
        row=1, col=1
    )

    # 2. PNL vs #Trades
    fig.add_trace(
        go.Scatter(
            x=history['num_trades'],
            y=history['total_pnl'],
            mode='markers',
            marker=dict(
                color=history['win_rate'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Win Rate')
            ),
            name='PNL vs #Trades'
        ),
        row=1, col=2
    )

    # 3. Win Rate vs PNL
    fig.add_trace(
        go.Scatter(
            x=history['total_pnl'],
            y=history['win_rate'],
            mode='markers+text',
            text=history['num_trades'],
            textposition='top center',
            marker=dict(
                size=10,
                color=history['num_trades'],
                colorscale='Bluered',
                showscale=True,
                colorbar=dict(title='# Trades')
            ),
            name='Win Rate vs PNL'
        ),
        row=2, col=1
    )

    # 4. Heatmap
    fig.add_trace(
        go.Heatmap(
            z=history['total_pnl'],
            colorscale='RdBu',
            colorbar=dict(title='PNL')
        ),
        row=2, col=2
    )

    fig.update_layout(
        height=1000,
        width=1200,
        title_text="Analyse de l'Optimisation des Ratios",
        showlegend=False
    )
    return fig


def print_optimization_results(optimizer):
    print("\n=== Résultats de l'Optimisation des Ratios ===")
    if optimizer.best_result is None:
        print("Aucune solution valide (pas assez de trades).")
        return

    r = optimizer.best_result
    print("\nMeilleure combinaison trouvée :")
    print(f"  - Plage Ask/Bid Ratio : {r['askbid_range_used'][0]:.2f} - {r['askbid_range_used'][1]:.2f}")
    print(f"  - Plage Pull/Stack    : {r['pullstack_range_used'][0]:.2f} - {r['pullstack_range_used'][1]:.2f}")
    print("\nPerformance :")
    print(f"  - PNL Total           : {r['total_pnl']:.2f}")
    print(f"  - Nombre de trades    : {r['num_trades']}")
    print(f"  - Win Rate            : {r['win_rate']:.1f}%")
    print(f"  - PNL moyen/trade     : {r['avg_pnl']:.2f}")


########################################################
# CONSTRUCTION DF OPTIMISE (SI BESOIN)
########################################################
def build_optimized_df(df, best_result):
    """
    Conserve :
      - Barres de session (10 ou 20)
      - Barres dont (askbid, pullstack) in best_result
    """
    askbid_low, askbid_high = best_result['askbid_range_used']
    pullstack_low, pullstack_high = best_result['pullstack_range_used']

    mask = (
        (df['SessionStartEnd'].isin([10, 20])) |
        (
            df['cumDOM_AskBid_avgRatio'].between(askbid_low, askbid_high) &
            df['cumDOM_AskBid_pullStack_avgDiff_ratio'].between(pullstack_low, pullstack_high)
        )
    )
    df_opt = df[mask].copy()
    df_opt = df_opt.sort_values('date').reset_index(drop=True)
    return df_opt


def compute_and_print_sessions_stats_with_details_on_df_opt(df_opt, sessions_data):
    """
    Reprend les sessions trouvées sur df initial et applique les calculs sur df_opt.
    Utilise start_bar_ts et end_bar_ts pour inclure correctement tous les trades.
    """
    df_opt = df_opt.copy().sort_values('date')
    if not pd.api.types.is_datetime64_any_dtype(df_opt['date']):
        df_opt['date'] = pd.to_datetime(df_opt['date'])

    global_pnl = 0.0
    global_history = []

    print("\n=== STATS DES SESSIONS (APRES OPTIMISATION, AVEC DETAILS, MÊMES BORNES) ===")

    for i, sess in enumerate(sessions_data, start=1):
        start_ts = sess['start_bar_ts']  # ex: 2025-01-06 23:01:46
        end_ts = sess['end_bar_ts']  # ex: 2025-01-07 21:57:02

        # Correction : On filtre avec les vrais timestamps
        rows_sess = df_opt[(df_opt['date'] >= start_ts) & (df_opt['date'] <= end_ts)]

        # Vérification : nombre de trades capturés dans la session
        print(f"Session {i}: de {start_ts} à {end_ts} → Trades trouvés : {len(rows_sess)}")

        # PNL de la session
        session_pnl = rows_sess['trade_pnl'].sum()
        session_cum = rows_sess['trade_pnl'].cumsum()

        # ✅ Convertir en tableau NumPy
        session_cum_np = session_cum.to_numpy(dtype=np.float64)  # ou .values

        # ✅ Appeler la fonction avec un array compatible avec Numba
        session_dd = calculate_max_drawdown_numba(session_cum_np)

        trades_real = rows_sess[rows_sess['trade_category'].str.contains("Trades échoués|Trades réussis", na=False)]
        nb_trades = len(trades_real)
        winning_trades = trades_real[trades_real['trade_pnl'] > 0]
        losing_trades = trades_real[trades_real['trade_pnl'] <= 0]

        nb_wins = len(winning_trades)
        nb_losses = len(losing_trades)

        if nb_trades > 0:
            win_rate = 100.0 * nb_wins / nb_trades
            expected_pnl = session_pnl / nb_trades
        else:
            win_rate = 0.0
            expected_pnl = 0.0


        global_pnl += session_pnl
        global_history.append(global_pnl)

        # ✅ Correction : Convertir la liste en un tableau NumPy AVANT d'appeler Numba
        global_history_np = np.array(global_history, dtype=np.float64)
        global_dd = calculate_max_drawdown_numba(global_history_np)

        # ✅ Correction : affichage des **vrais timestamps**
        print(f"\nSession #{i} du {end_ts.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  - Date début session        : {start_ts.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  - Date fin session          : {end_ts.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  - Win Rate (fin session)    : {win_rate:.2f}%")
        print(f"  - PNL fin de session        : {session_pnl:.2f}")
        print(f"  - Max drawdown (session)    : {session_dd:.2f}")
        print(f"  - Nombre de trades pris     : {nb_trades}")

        print(f"  - Trades gagnés             : {nb_wins}")
        for _, row_w in winning_trades.iterrows():
            print(f"      * {row_w['date']} => PNL={row_w['trade_pnl']:.2f}")

        print(f"  - Trades échoués            : {nb_losses}")
        for _, row_l in losing_trades.iterrows():
            print(f"      * {row_l['date']} => PNL={row_l['trade_pnl']:.2f}")

        print(f"  - Expected PNL/trade        : {expected_pnl:.2f}")
        print(f"  - PNL cumulé historique     : {global_pnl:.2f}")
        print(f"  - Max drawdown historique   : {global_dd:.2f}")


def preprocess_trades_data(df, direction='short'):
    """
    Prétraite les données de trading en conservant soit les trades shorts soit les longs,
    et en standardisant toutes les autres lignes.

    Args:
        df (pd.DataFrame): DataFrame d'entrée contenant les données de trading
        direction (str): Direction des trades à conserver ('short' ou 'long')
                        Par défaut : 'short'

    Returns:
        pd.DataFrame: DataFrame prétraité où seuls les trades de la direction spécifiée
                     sont conservés, les autres lignes étant standardisées

    Raises:
        ValueError: Si la direction spécifiée n'est pas 'short' ou 'long'
    """
    # Validation de l'argument direction
    if direction not in ['short', 'long']:
        raise ValueError("La direction doit être 'short' ou 'long'")

    # Création d'une copie pour éviter de modifier les données originales
    df_processed = df.copy()

    # Définition des catégories selon la direction choisie
    trade_categories = {
        'short': ['Trades réussis short', 'Trades échoués short'],
        'long': ['Trades réussis long', 'Trades échoués long']
    }

    # Sélection des catégories appropriées
    selected_categories = trade_categories[direction]

    # Création d'un masque pour identifier les lignes qui ne sont pas des trades de la direction choisie
    non_selected_mask = ~df_processed['trade_category'].isin(selected_categories)

    # Application des valeurs par défaut pour toutes les autres lignes
    df_processed.loc[non_selected_mask, 'trade_category'] = 'Pas de trade'
    df_processed.loc[non_selected_mask, 'trade_pnl'] = 0
    df_processed.loc[non_selected_mask, 'class_binaire'] = 99

    # Affichage d'un résumé des modifications
    total_rows = len(df_processed)
    kept_trades = (~non_selected_mask).sum()
    standardized_rows = non_selected_mask.sum()

    print(f"\nRésumé du prétraitement ({direction}):")
    print(f"- Nombre total de lignes: {total_rows}")
    print(f"- Trades {direction} conservés: {kept_trades}")
    print(f"- Lignes standardisées: {standardized_rows}")
    print(f"- Catégories conservées: {', '.join(selected_categories)}")

    return df_processed


########################################################
# MAIN
########################################################
def main():
    import optuna
    import os
    # On réduit la verbosité globale d'Optuna au niveau WARN
    # pour ne pas avoir leurs logs "Trial X finished..."
    optuna.logging.set_verbosity(optuna.logging.WARN)
    # 1) Charger le CSV
    DIRECTORY_PATH = r"C:\\Users\\aulac\\OneDrive\\Documents\\Trading\\VisualStudioProject\\Sierra chart\\xTickReversal\\simu\\\\5_0_4TP_0SL\\merge_old"

    FILE_NAME_ = "Step5_5_0_4TP_0SL_050125_200125_extractOnlyFullSession_OnlyShort_feat_winsorized.csv"
    FILE_PATH = os.path.join(DIRECTORY_PATH, FILE_NAME_)

    df_init, CUSTOM_SESSIONS = load_features_and_sections(FILE_PATH)
    # Chargement des données
    df_init, CUSTOM_SESSIONS = load_features_and_sections(FILE_PATH)

    # Prétraitement des données avec la direction spécifiée
    df_processed = preprocess_trades_data(df_init, direction='short')  # ou 'long'
    print(df_processed)

    # Le reste du code utilise maintenant df_processed
    sessions_data = validate_sessions_10_20_only(df_processed)
    print(sessions_data)
    compute_and_print_sessions_stats_no_details(df_processed, sessions_data)
    print("\n[Optimisation]")
    #shorts_df = df[df['trade_category'].isin(['Trades réussis short', 'Trades échoués short'])]
    optimizer = RatioOptimizer(df_processed)
    study = optimizer.optimize(n_trials=12000
                               )
    print_optimization_results(optimizer)

    # 4) Construire un DF optimisé (optionnel)
    df_opt = build_optimized_df(df_processed, optimizer.best_result)
    print(f"\nDF optimisé : {len(df_opt)} lignes")

    # 5) Affichage des stats AVEC détails,
    #    en réutilisant la liste sessions_data (trouvée sur df complet)
    print("\n[Stats APRES Optimisation : AVEC DETAILS, mêmes sessions qu'avant]")
    compute_and_print_sessions_stats_with_details_on_df_opt(df_opt, sessions_data)

    # 6) Visualisation
    fig = analyze_and_visualize_results(optimizer)
    fig.show()

if __name__ == "__main__":
    main()
