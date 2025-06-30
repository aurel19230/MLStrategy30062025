import pandas as pd
import numpy as np
import optuna
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import datetime
from standard_stat_sc import *

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
    def __init__(self, df, params_dict):
        self.df = df[df['class_binaire'].isin([0,1])].copy()
        self.optimization_history = []
        self.best_result = None
        self.params_dict = params_dict  # Stocke le dictionnaire de paramètres

    def objective(self, trial):
        """
        Méthode objective pour Optuna.
        Boucle automatiquement sur self.params_dict pour générer les valeurs _low/_high,
        applique le filtre sur self.df,
        calcule le PNL et met à jour self.best_result si besoin.
        """
        import numpy as np

        # Dictionnaire local qui stockera les valeurs "low/high" suggérées
        param_values = {}

        # 1) Génération dynamique des suggestions
        # ---------------------------------------
        # Pour chaque "param_name", on a une entrée (low_bound, high_bound) dans params_dict.
        # Chaque bound est un tuple : (type, min, max), ex: ('float', 0, 291)
        #
        # ex : params_dict = {
        #    'cumDOM_AskBid_avgRatio': [('float', 0, 291), ('float', 0, 291)],
        #    'cumDOM_AskBid_pullStack_avgDiff_ratio': [('float', -63, 50), ('float', -63, 50)]
        # }
        #
        # => On va créer param_name_low et param_name_high pour chaque variable.
        #
        for param_name, bounds in self.params_dict.items():
            if bounds == 'binary':
                # Pour un paramètre binaire, on génère la valeur 0 ou 1
                binary_val = trial.suggest_int(param_name, 0, 1)
                # On stocke la même valeur en low/high
                param_values[f"{param_name}_low"] = binary_val
                param_values[f"{param_name}_high"] = binary_val
            else:
                low_bound, high_bound = bounds
                low_type, low_min, low_max = low_bound
                high_type, high_min, high_max = high_bound

                if low_type == 'float':
                    low_value = trial.suggest_float(f"{param_name}_low", low_min, low_max)
                elif low_type == 'int':
                    low_value = trial.suggest_int(f"{param_name}_low", int(low_min), int(low_max))

                if high_type == 'float':
                    high_value = trial.suggest_float(f"{param_name}_high", high_min, high_max)
                elif high_type == 'int':
                    high_value = trial.suggest_int(f"{param_name}_high", int(high_min), int(high_max))

                param_values[f"{param_name}_low"], param_values[f"{param_name}_high"] = sorted([low_value, high_value])

        # 2) Filtrage automatique dans self.df
        # -------------------------------------------
        # On part d'un masque "tout True", puis on combine
        condition = np.ones(len(self.df), dtype=bool)

        # Pour chaque param_name du dictionnaire, on récupère param_name_low / param_name_high
        for param_name in self.params_dict.keys():
            low_key = f"{param_name}_low"
            high_key = f"{param_name}_high"
            low_val = param_values[low_key]
            high_val = param_values[high_key]

            # On applique un filtre "df[param_name] between low_val/high_val"
            condition &= self.df[param_name].between(low_val, high_val)

        # On récupère les lignes filtrées
        trades_in_range = self.df[condition]

        # Si pas assez de trades, on renvoie -inf pour pénaliser cette config
        n_trades = len(trades_in_range)
        if n_trades < 10:
            return float('-inf')

        # 3) Calcul du PNL
        # ----------------
        total_pnl = trades_in_range['trade_pnl'].sum()
        # (On peut calculer le win_rate, avg_pnl, etc.)
        # ex:
        win_rate = (trades_in_range['trade_pnl'] > 0).mean() * 100
        avg_pnl = total_pnl / n_trades

        # 4) Construction du "result" pour l'historique
        # ---------------------------------------------
        # On enregistre TOUTES les valeurs dans param_values (pour pouvoir recréer le filtre après).
        # On y ajoute le PNL, etc.
        result = {
            'param_values': param_values,
            # Ex: {'cumDOM_AskBid_avgRatio_low': 12.3, 'cumDOM_AskBid_avgRatio_high': 150.0, ...}
            'total_pnl': total_pnl,
            'num_trades': n_trades,
            'win_rate': win_rate,
            'avg_pnl': avg_pnl
        }
        self.optimization_history.append(result)

        # 5) Mise à jour de self.best_result
        # ----------------------------------
        # Si c'est le meilleur PNL jusqu'ici, on remplace
        if (self.best_result is None) or (total_pnl > self.best_result['total_pnl']):
            self.best_result = result

        # 6) Retour de la valeur => Optuna cherche à la maximiser
        # -------------------------------------------------------
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

    param_values = r['param_values']  # dict
    # Ex: { "cumDOM_AskBid_avgRatio_low": 12.3, "cumDOM_AskBid_avgRatio_high": 100.0, ... }

    # Parcourir param_values par paires
    for k, v in param_values.items():
        if k.endswith("_low"):
            param_name = k[:-4]  # "cumDOM_AskBid_avgRatio"
            low_val = v
            high_val = param_values.get(f"{param_name}_high", None)
            if high_val is not None:
                print(f"  - Plage pour {param_name} : {low_val:.2f} - {high_val:.2f}")

    print("\nPerformance :")
    print(f"  - PNL Total           : {r['total_pnl']:.2f}")
    if 'num_trades' in r:
        print(f"  - Nombre de trades    : {r['num_trades']}")
    if 'win_rate' in r:
        print(f"  - Win Rate            : {r['win_rate']:.1f}%")
    if 'avg_pnl' in r:
        print(f"  - PNL moyen/trade     : {r['avg_pnl']:.2f}")


########################################################
# CONSTRUCTION DF OPTIMISE (SI BESOIN)
########################################################
def build_optimized_df(df, best_result):
    """
    Filtre df en fonction des paramètres qui ont donné le meilleur PNL.
    Conserve aussi les barres (10 ou 20).
    """
    if best_result is None:
        raise ValueError("best_result is None => Impossible de construire df_opt.")

    param_values = best_result['param_values']  # On récupère le dict de paramètres

    # 1) Filtrage dynamique
    condition = np.ones(len(df), dtype=bool)

    # param_values est un dict du genre {'cumDOM_AskBid_avgRatio_low': X, 'cumDOM_AskBid_avgRatio_high': Y, ...}
    # On va boucler dessus de façon groupée => on peut re-parcourir param_config
    # ou reconstituer le pattern param_name = ..._low / ..._high.

    # Option A) Re-parcourir param_config si on l'a => "s'il faut"
    # Option B) ou parse param_values.keys()

    # EXEMPLE en regroupant par "param_name" => on repère le suffixe "_low" / "_high"
    # param_name = key[:-4] => "cumDOM_AskBid_avgRatio"
    # suffix = key[-4:] => "_low"

    for k, val in param_values.items():
        if k.endswith("_low"):
            param_name = k[:-4]  # remove suffix
            low_val = val
            high_val = param_values.get(f"{param_name}_high", None)
            if high_val is None:
                # skip or raise
                continue

            condition &= df[param_name].between(low_val, high_val)

    # 2) On combine ce mask avec "barres 10 ou 20"
    # si c'est la logique voulue : on garde TOUTES les barres de session + barres in range
    # => mask_session = df['SessionStartEnd'].isin([10,20])
    # => mask_final = mask_session | condition
    mask_session = df['SessionStartEnd'].isin([10, 20])
    mask_final = mask_session | condition

    df_opt = df[mask_final].copy()
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
    from pynput import keyboard
    import random

    # Définir une seed pour la reproductibilité
    SEED = 50  # Vous pouvez choisir n'importe quelle valeur entière

    # Initialiser les générateurs de nombres aléatoires
    np.random.seed(SEED)
    random.seed(SEED)

    # Configuration d'Optuna avec la seed
    sampler = optuna.samplers.TPESampler(seed=SEED)  # Le sampler TPE est couramment utilisé

    # Variable globale pour signaler l'arrêt
    global STOP_OPTIMIZATION
    STOP_OPTIMIZATION = False

    # Fonction de rappel pour arrêter Optuna
    def callback_optuna_stop(study, trial):
        global STOP_OPTIMIZATION
        if STOP_OPTIMIZATION:
            print("\n\nArrêt demandé par l'utilisateur - Finalisation de l'essai en cours...")
            study.stop()

    # Gestionnaire d'événements clavier
    def on_press(key):
        global STOP_OPTIMIZATION
        try:
            if hasattr(key, 'char') and key.char == '²':
                print("\nTouche '²' détectée - Signal d'arrêt envoyé")
                STOP_OPTIMIZATION = True
        except AttributeError:
            pass

    # Démarrer l'écouteur de clavier dans un thread séparé
    listener = keyboard.Listener(on_press=on_press)
    listener.daemon = True  # Pour que le thread se termine quand le programme principal se termine
    listener.start()

    # Le reste de votre code reste inchangé
    # On réduit la verbosité globale d'Optuna au niveau WARN
    optuna.logging.set_verbosity(optuna.logging.WARN)

    # 1) Charger le CSV
    DIRECTORY_PATH = r"C:\\Users\\aulac\\OneDrive\\Documents\\Trading\\VisualStudioProject\\Sierra chart\\xTickReversal\\simu\\\\5_0_5TP_1SL\\merge_I1_I2"
    FILE_NAME_ = "Step5_5_0_5TP_1SL_150924_280225_bugFixTradeResult_extractOnlyFullSession_OnlyShort_feat_winsorized.csv"
    FILE_PATH = os.path.join(DIRECTORY_PATH, FILE_NAME_)

    df_init, CUSTOM_SESSIONS = load_features_and_sections(FILE_PATH)

    #add  session_id
    df_withsessionID= preprocess_sessions_with_date(df_init)

    # Prétraitement des données avec la direction spécifiée
    df_processed = preprocess_trades_data(df_withsessionID, direction='short')
    print(df_processed)

    # Le reste du code utilise maintenant df_processed
    sessions_data = validate_sessions_10_20_only(df_processed)
    print(sessions_data)
    compute_and_print_sessions_stats_no_details(df_processed, sessions_data)
    print("\n[Optimisation]")

    param_config = {
        #'cumDOM_AskBid_avgRatio': [('float', 0, 291), ('float', 0, 291)],
        #'meanVolx': [('float', 0, 7000), ('float', 0, 7000)],
        'VolAbv_vol_ratio': [('float', 0, 1), ('float', 0, 1)],
        'finished_auction_low': 'binary',
        'finished_auction_high': 'binary'
    }
    #'new_param2': 'binary'  # Indique qu'on veut juste 0 ou 1

    # Modification de la classe RatioOptimizer pour intégrer notre callback
    # Création de l'étude avec le sampler configuré
    class ModifiedRatioOptimizer(RatioOptimizer):
        def optimize(self, n_trials=10000):
            # Utiliser le sampler avec la seed définie
            study = optuna.create_study(direction='maximize', sampler=sampler)
            callbacks = [logging_callback, callback_optuna_stop]
            study.optimize(self.objective, n_trials=n_trials, callbacks=callbacks, n_jobs=-1)
            return study

    # Création de l'optimiseur modifié avec le dictionnaire de paramètres
    optimizer = ModifiedRatioOptimizer(df_processed, param_config)

    # Informer l'utilisateur comment arrêter
    print("\nL'optimisation est en cours. Appuyez sur la touche '²' pour arrêter.\n")

    # Lancer l'optimisation
    study = optimizer.optimize(n_trials=10000)
    print_optimization_results(optimizer)

    # 4) Construire un DF optimisé (optionnel)
    df_opt = build_optimized_df(df_processed, optimizer.best_result)
    print(f"\nDF optimisé : {len(df_opt)} lignes")

    # 5) Affichage des stats AVEC détails
    print("\n[Stats APRES Optimisation : AVEC DETAILS, mêmes sessions qu'avant]")
    compute_and_print_sessions_stats_with_details_on_df_opt(df_opt, sessions_data)

    # Arrêter proprement l'écouteur de clavier
    listener.stop()


if __name__ == "__main__":
    main()
