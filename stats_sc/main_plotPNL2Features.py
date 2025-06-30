import pandas as pd
import numpy as np
import optuna
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import datetime

warnings.filterwarnings('ignore')

########################################################
# CALLBACK POUR LES LOGS TOUTES LES 100 ITERATIONS
########################################################
def logging_callback(study, trial):
    """
    Callback qui n'affiche un log que toutes les 100 itérations.
    Gère aussi le cas où trial.value est None ou infini.
    """
    if trial.number % 100 == 0:
        now_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Gestion de la valeur
        if trial.value is None or not np.isfinite(trial.value):
            value_str = str(trial.value)  # ex: 'None', 'inf', '-inf'
        else:
            value_str = f"{trial.value:.2f}"

        # Paramètres triés
        askbid_low_sorted = trial.user_attrs.get("askbid_low_sorted", None)
        askbid_high_sorted = trial.user_attrs.get("askbid_high_sorted", None)
        pullstack_low_sorted = trial.user_attrs.get("pullstack_low_sorted", None)
        pullstack_high_sorted = trial.user_attrs.get("pullstack_high_sorted", None)

        # Meilleure valeur trouvée jusque-là
        best_trial_number = study.best_trial.number
        # Idem, on sécurise l'affichage
        if study.best_value is None or not np.isfinite(study.best_value):
            best_value_str = str(study.best_value)
        else:
            best_value_str = f"{study.best_value:.2f}"

        print(
            f"[I {now_str}] Trial {trial.number} finished with value: {value_str} "
            f"and used parameters (sorted): "
            f"askbid=[{askbid_low_sorted}, {askbid_high_sorted}], "
            f"pullstack=[{pullstack_low_sorted}, {pullstack_high_sorted}]. "
            f"Best is trial {best_trial_number} with value: {best_value_str}"
        )

class RatioOptimizer:
    """
    Classe qui gère l'optimisation des plages de ratios pour maximiser le PNL des trades.
    Cette classe utilise l'optimisation bayésienne via Optuna pour trouver les meilleures
    plages de valeurs pour deux ratios clés:
    - cumDOM_AskBid_avgRatio (0 à 291)
    - cumDOM_AskBid_pullStack_avgDiff_ratio (-63 à 50)
    """

    def __init__(self, shorts_df):
        """
        Initialise l'optimiseur avec les données des trades shorts.

        Args:
            shorts_df (pd.DataFrame): DataFrame contenant les trades shorts
        """
        self.shorts_df = shorts_df
        self.optimization_history = []  # Stocke tous les essais (plages + performance)
        self.best_result = None  # Stocke le meilleur résultat trouvé (PNL max)

    def objective(self, trial):
        """
        Fonction objective pour l'optimisation bayésienne.
        Elle évalue une combinaison de plages de ratios et retourne le PNL (à maximiser).

        Args:
            trial (optuna.trial.Trial): Un essai Optuna

        Returns:
            float: PNL total des trades filtrés
        """
        # Propositions d'Optuna (valeurs "brutes")
        askbid_low_raw = trial.suggest_float('askbid_low', 0, 4)
        askbid_high_raw = trial.suggest_float('askbid_high', 0, 4)
        pullstack_low_raw = trial.suggest_float('pullstack_low', -0.3, 7)
        pullstack_high_raw = trial.suggest_float('pullstack_high', -0.3, 7)

        # On trie pour obtenir (low <= high)
        askbid_low, askbid_high = sorted([askbid_low_raw, askbid_high_raw])
        pullstack_low, pullstack_high = sorted([pullstack_low_raw, pullstack_high_raw])

        # Filtrer les trades
        trades_in_range = self.shorts_df[
            (self.shorts_df['ratio_volRevMove_volImpulsMove'].between(askbid_low, askbid_high)) &
            (self.shorts_df['cumDOM_AskBid_pullStack_avgDiff_ratio'].between(pullstack_low, pullstack_high))
        ]

        n_trades = len(trades_in_range)

        # Si trop peu de trades, on pénalise la solution
        if n_trades < 10:
            return float('-inf')

        # Calculer les métriques
        total_pnl = trades_in_range['trade_pnl'].sum()
        win_rate = (trades_in_range['trade_pnl'] > 0).mean() * 100
        avg_pnl = total_pnl / n_trades

        # Stocker la version triée dans user_attrs, pour la récupérer dans le callback
        trial.set_user_attr("askbid_low_sorted", askbid_low)
        trial.set_user_attr("askbid_high_sorted", askbid_high)
        trial.set_user_attr("pullstack_low_sorted", pullstack_low)
        trial.set_user_attr("pullstack_high_sorted", pullstack_high)

        # Stocker les résultats dans l'historique (on y met aussi la version brute si besoin)
        result = {
            'askbid_range_used': (askbid_low, askbid_high),
            'pullstack_range_used': (pullstack_low, pullstack_high),
            'askbid_range_raw': (askbid_low_raw, askbid_high_raw),
            'pullstack_range_raw': (pullstack_low_raw, pullstack_high_raw),
            'total_pnl': total_pnl,
            'num_trades': n_trades,
            'win_rate': win_rate,
            'avg_pnl': avg_pnl
        }
        self.optimization_history.append(result)

        # Mettre à jour le "meilleur résultat"
        if self.best_result is None or total_pnl > self.best_result['total_pnl']:
            self.best_result = result

        # Optuna va chercher à maximiser ce return
        return total_pnl

    def optimize(self, n_trials=100):
        """
        Lance l'optimisation avec un nombre spécifié d'essais.

        Args:
            n_trials (int): Nombre d'essais pour l'optimisation

        Returns:
            optuna.Study: L'étude Optuna complétée
        """
        # On choisit "maximize" puisqu'on cherche à maximiser le PNL
        study = optuna.create_study(direction='maximize')

        # On associe le callback "logging_callback"
        study.optimize(self.objective, n_trials=n_trials, callbacks=[logging_callback])

        return study


def analyze_and_visualize_results(optimizer):
    """
    Crée une visualisation détaillée des résultats de l'optimisation.

    Args:
        optimizer (RatioOptimizer): L'optimiseur avec les résultats

    Returns:
        plotly.graph_objects.Figure: La figure avec les visualisations
    """
    # Créer une figure avec plusieurs sous-graphiques
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

    history = pd.DataFrame(optimizer.optimization_history)

    # 1. Scatter 3D des résultats
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

    # 2. PNL vs Nombre de trades
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

    # 4. Heatmap (simple exemple, ici on ne calcule pas de grille)
    fig.add_trace(
        go.Heatmap(
            z=history['total_pnl'],
            colorscale='RdBu',
            colorbar=dict(title='PNL')
        ),
        row=2, col=2
    )

    # Mise en forme finale
    fig.update_layout(
        height=1000,
        width=1200,
        title_text="Analyse de l'Optimisation des Ratios",
        showlegend=False
    )

    return fig


def print_optimization_results(optimizer):
    """
    Affiche un résumé détaillé des résultats de l'optimisation.

    Args:
        optimizer (RatioOptimizer): L'optimiseur avec les résultats
    """
    print("\n=== Résultats de l'Optimisation des Ratios ===")

    if optimizer.best_result is None:
        print("Aucune solution valide trouvée (pas assez de trades dans toutes les plages testées).")
        return

    result = optimizer.best_result
    print("\nMeilleure combinaison trouvée :")
    print(f"Plage Ask/Bid Ratio utilisé    : {result['askbid_range_used'][0]:.2f} - {result['askbid_range_used'][1]:.2f}")
    print(f"Plage Pull/Stack Ratio utilisé : {result['pullstack_range_used'][0]:.2f} - {result['pullstack_range_used'][1]:.2f}")
    print("\nPerformance :")
    print(f"PNL Total              : {result['total_pnl']:.2f}")
    print(f"Nombre de trades       : {result['num_trades']}")
    print(f"Win Rate               : {result['win_rate']:.1f}%")
    print(f"PNL moyen par trade    : {result['avg_pnl']:.2f}")


def main():
    """
    Fonction principale qui orchestre le processus d'optimisation complet.
    """
    # On réduit la verbosité globale d'Optuna au niveau WARN
    # pour ne pas avoir leurs logs "Trial X finished..."
    optuna.logging.set_verbosity(optuna.logging.WARN)

    # Charger les données
    import os
    # Chemin vers ton fichier
    file_name = "Step5_5_0_5TP_1SL_150924_280225_bugFixTradeResult_extractOnlyFullSession_OnlyShort_feat_winsorized.csv"
    directory_path = "C:\\Users\\aulac\\OneDrive\\Documents\\Trading\\VisualStudioProject\\Sierra chart\\xTickReversal\\simu\\5_0_5TP_1SL\\\merge_I1_I2"
    csv_path = os.path.join(directory_path, file_name)

    print("Chargement des données...")
    df = pd.read_csv(csv_path, sep=';', encoding='cp1252', low_memory=False)

    # Filtrer pour ne garder que les shorts
    shorts_df = df[df['trade_category'].isin(['Trades réussis short', 'Trades échoués short'])]
    print(f"Nombre total de trades shorts : {len(shorts_df)}")

    # Instancier et lancer l'optimisation
    print("\nDémarrage de l'optimisation...")
    optimizer = RatioOptimizer(shorts_df)

    # Nombre d'essais (peut être ajusté à la hausse)
    study = optimizer.optimize(n_trials=10000)

    # Résultats texte
    print_optimization_results(optimizer)

    # Visualisations
    fig = analyze_and_visualize_results(optimizer)
    fig.show()


if __name__ == "__main__":
    main()
