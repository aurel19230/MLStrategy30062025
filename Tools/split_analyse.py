import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import time
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import TimeSeriesSplit
from standardFunc_sauv import load_data
import numba as nb
from numba import jit
import matplotlib.pyplot as plt

# Configuration pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


def timestamp_to_date_utc_(timestamp):
    date_format = "%Y-%m-%d %H:%M:%S"
    if isinstance(timestamp, pd.Series):
        return timestamp.apply(lambda x: time.strftime(date_format, time.gmtime(x)))
    else:
        return time.strftime(date_format, time.gmtime(timestamp))


def calculate_time_difference(start_date_str, end_date_str):
    if isinstance(start_date_str, str):
        date_format = "%Y-%m-%d %H:%M:%S"
        start_date = datetime.strptime(start_date_str, date_format)
        end_date = datetime.strptime(end_date_str, date_format)
    else:
        start_date = start_date_str
        end_date = end_date_str

    diff = relativedelta(end_date, start_date)
    return diff


@jit(nopython=True)
def calculate_ratios(class_values):
    """
    Calcule les ratios de classes rapidement
    """
    class_1 = np.sum(class_values == 1)
    class_0 = np.sum(class_values == 0)
    ratio = class_1 / class_0 if class_0 > 0 else 0
    return class_1, class_0, ratio


@jit(nopython=True)
def calculate_fold_stats(val_data):
    """
    Calcule les statistiques pour un fold
    """
    class_values = val_data[val_data != 99]
    class_1, class_0, ratio = calculate_ratios(class_values)
    return len(class_values), class_1, class_0, ratio


@jit(nopython=True)
def calculate_stats_numba(class_values):
    """
    Version optimisée du calcul des statistiques
    """
    total = len(class_values)
    wins = np.sum(class_values == 1)
    losses = np.sum(class_values == 0)
    excluded = total - (wins + losses)
    winrate = wins / float(wins + losses) if (wins + losses) > 0 else 0
    return total, wins, losses, excluded, winrate


def calculate_stats(df, title):
    """
    Calcule et affiche les statistiques d'un DataFrame
    """
    total = len(df)
    wins = (df['class_binaire'] == 1).sum()
    losses = (df['class_binaire'] == 0).sum()
    excluded = total - wins - losses
    winrate = wins / (wins + losses) if (wins + losses) > 0 else 0

    print(f"\n{title}:")
    print(f"Total observations: {total:,d}")
    print(f"Wins: {wins:,d}")
    print(f"Losses: {losses:,d}")
    print(f"Exclus (class_binaire == 99): {excluded:,d}")
    print(f"Winrate: {winrate:.2%}")

    return {
        'total': total,
        'wins': wins,
        'losses': losses,
        'excluded': excluded,
        'winrate': winrate
    }


@jit(nopython=True)
def calculate_session_stats_numba(timestamps, session_markers, class_values):
    """
    Calcule les statistiques de session avec Numba
    Returns: list of tuples (start_idx, end_idx, n_observations)
    """
    sessions = []
    current_session_start = -1

    for i in range(len(session_markers)):
        if session_markers[i] == 10:  # Début de session
            current_session_start = i
        elif session_markers[i] == 20 and current_session_start != -1:  # Fin de session
            sessions.append((current_session_start, i, i - current_session_start + 1))
            current_session_start = -1

    return sessions


def calculate_session_averages(df, removed_sessions=None):
    """
    Calcule le nombre moyen d'observations par session avec optimisation Numba
    """
    # Conversion en arrays numpy pour Numba
    timestamps = df['timeStampOpening'].values
    session_markers = df['SessionStartEnd'].values
    class_values = df['class_binaire'].values

    # Calcul des statistiques avec Numba
    sessions = calculate_session_stats_numba(timestamps, session_markers, class_values)

    cleaned_sessions = []
    print("\nAnalyse détaillée des sessions:")
    print("=" * 50)

    # Traitement des résultats
    for i, (start_idx, end_idx, n_obs) in enumerate(sessions, 1):
        cleaned_sessions.append(n_obs)
        print(f"Session {i}:")
        print(f"  Début: {timestamp_to_date_utc_(timestamps[start_idx])}")
        print(f"  Fin: {timestamp_to_date_utc_(timestamps[end_idx])}")
        print(f"  Observations: {n_obs}")

    cleaned_avg = np.mean(cleaned_sessions) if cleaned_sessions else 0
    cleaned_total_sessions = len(cleaned_sessions)

    # Pour les sessions supprimées
    if removed_sessions:
        print("\nSessions supprimées:")
        print("=" * 50)
        removed_observations = [session['statistics']['trades'] for session in removed_sessions]

        for i, (session, obs) in enumerate(zip(removed_sessions, removed_observations), 1):
            print(f"Session supprimée {i}:")
            print(f"  Début: {session['start_date']}")
            print(f"  Fin: {session['end_date']}")
            print(f"  Observations: {obs}")

        removed_avg = np.mean(removed_observations) if removed_observations else 0
        removed_total_sessions = len(removed_sessions)
    else:
        removed_avg = 0
        removed_total_sessions = 0
        removed_observations = []

    print("\nRésumé des sessions:")
    print("=" * 50)
    print(f"Sessions conservées:")
    print(f"  Nombre total de sessions: {cleaned_total_sessions}")
    print(f"  Observations par session: {cleaned_sessions}")
    print(f"  Moyenne d'observations: {cleaned_avg:.2f}")
    if cleaned_sessions:
        print(f"  Min observations: {min(cleaned_sessions)}")
        print(f"  Max observations: {max(cleaned_sessions)}")
        print(f"  Écart-type: {np.std(cleaned_sessions):.2f}")

    if removed_sessions:
        print(f"\nSessions supprimées:")
        print(f"  Nombre total de sessions: {removed_total_sessions}")
        print(f"  Observations par session: {removed_observations}")
        print(f"  Moyenne d'observations: {removed_avg:.2f}")
        if removed_observations:
            print(f"  Min observations: {min(removed_observations)}")
            print(f"  Max observations: {max(removed_observations)}")
            print(f"  Écart-type: {np.std(removed_observations):.2f}")

    return {
        'cleaned_avg': cleaned_avg,
        'cleaned_total_sessions': cleaned_total_sessions,
        'cleaned_sessions_details': cleaned_sessions,
        'removed_avg': removed_avg,
        'removed_total_sessions': removed_total_sessions,
        'removed_sessions_details': removed_observations
    }


def analyze_splits(df, X_train_full, n_splits=80, filter_ratio_up=None, filter_ratio_down=None,
                   display_all_folds=False):
    df_filtered = df[df['class_binaire'] != 99].copy()
    tscv = TimeSeriesSplit(n_splits=n_splits)
    split_stats = {}
    anomalous_folds = []

    # Premier passage pour calculer les ratios
    val_ratios = []
    for i, (train_index, val_index) in enumerate(tscv.split(df_filtered)):
        val_data = df_filtered.iloc[val_index]
        val_class_1 = (val_data['class_binaire'] == 1).sum()
        val_class_0 = (val_data['class_binaire'] == 0).sum()
        val_ratio = val_class_1 / val_class_0 if val_class_0 > 0 else 0
        val_ratios.append(val_ratio)

    mean_ratio = np.mean(val_ratios)
    threshold_up = mean_ratio * (1 + filter_ratio_up / 100) if filter_ratio_up is not None else float('inf')
    threshold_down = mean_ratio * (1 - filter_ratio_down / 100) if filter_ratio_down is not None else 0

    # Deuxième passage pour l'analyse complète
    for i, (train_index, val_index) in enumerate(tscv.split(df_filtered)):
        train_data = df_filtered.iloc[train_index]
        val_data = df_filtered.iloc[val_index]

        start_time = val_data['timeStampOpening'].iloc[0]
        end_time = val_data['timeStampOpening'].iloc[-1]

        train_total, train_class_1, train_class_0, train_ratio = calculate_fold_stats(
            train_data['class_binaire'].values)
        val_total, val_class_1, val_class_0, val_ratio = calculate_fold_stats(val_data['class_binaire'].values)

        split_stats[i + 1] = {
            'start_time': start_time,
            'end_time': end_time,
            'train_n_trades': train_total,
            'val_n_trades': val_total,
            'train_class_1': int(train_class_1),
            'train_class_0': int(train_class_0),
            'val_class_1': int(val_class_1),
            'val_class_0': int(val_class_0),
            'train_ratio': train_ratio,
            'val_ratio': val_ratio
        }

        if display_all_folds:
            start_date = timestamp_to_date_utc_(start_time)
            end_date = timestamp_to_date_utc_(end_time)
            percentage_diff = ((val_ratio - mean_ratio) / mean_ratio) * 100 if mean_ratio > 0 else 0
            print(f"\nFold {i + 1}/{n_splits}:")
            print(f"Période de validation - Du {start_date} au {end_date}")
            print(f"Train - Total: {train_total}, Classe [1]: {train_class_1}, "
                  f"Classe [0]: {train_class_0}, Ratio: {train_ratio:.2f}")
            print(f"Val   - Total: {val_total}, Classe [1]: {val_class_1}, "
                  f"Classe [0]: {val_class_0}, Ratio: {val_ratio:.2f}")
            print(f"Écart par rapport à la moyenne ({mean_ratio:.2f}): {percentage_diff:+.1f}%")

        # Vérification des seuils haut et bas
        if val_ratio > threshold_up or val_ratio < threshold_down:
            start_date = timestamp_to_date_utc_(start_time)
            end_date = timestamp_to_date_utc_(end_time)
            percentage_diff = ((val_ratio - mean_ratio) / mean_ratio) * 100
            anomaly_type = "au-dessus" if val_ratio > threshold_up else "en-dessous"

            anomalous_folds.append({
                'fold': i + 1,
                'ratio': val_ratio,
                'start_date': start_date,
                'end_date': end_date,
                'stats': split_stats[i + 1],
                'anomaly_type': anomaly_type,
                'percentage_diff': percentage_diff
            })

    if anomalous_folds:
        print(f"\n=== Folds avec ratio de validation anormal ===")
        print(f"Moyenne des ratios de validation: {mean_ratio:.2f}")
        print(f"Seuils de détection: {threshold_down:.2f} (bas) à {threshold_up:.2f} (haut)\n")

        for fold in anomalous_folds:
            print(f"Fold {fold['fold']}/{n_splits} ({fold['anomaly_type']} de la moyenne, "
                  f"écart: {fold['percentage_diff']:+.1f}%):")
            print(f"Période de validation - Du {fold['start_date']} ({fold['stats']['start_time']}) "
                  f"au {fold['end_date']} ({fold['stats']['end_time']})")
            print(f"Train - Total: {fold['stats']['train_n_trades']}, "
                  f"Classe [1]: {fold['stats']['train_class_1']}, "
                  f"Classe [0]: {fold['stats']['train_class_0']}, "
                  f"Ratio: {fold['stats']['train_ratio']:.2f}")
            print(f"Val   - Total: {fold['stats']['val_n_trades']}, "
                  f"Classe [1]: {fold['stats']['val_class_1']}, "
                  f"Classe [0]: {fold['stats']['val_class_0']}, "
                  f"Ratio: {fold['stats']['val_ratio']:.2f}\n")

    return split_stats, anomalous_folds


@jit(nopython=True)
def find_session_boundaries_numba(session_markers, timestamps, start_idx, end_idx):
    """
    Version optimisée de la recherche des limites de session
    """
    sessions = []
    current_idx = start_idx

    # Remonter au début de la session
    while current_idx > 0 and session_markers[current_idx] != 10:
        current_idx -= 1
    first_session_start = current_idx

    current_idx = first_session_start
    while current_idx <= end_idx:
        if session_markers[current_idx] == 10:
            session_start = current_idx
            temp_idx = current_idx
            while temp_idx < len(session_markers) and session_markers[temp_idx] != 20:
                temp_idx += 1
            session_end = min(temp_idx, len(session_markers) - 1)
            sessions.append((session_start, session_end))
        current_idx += 1

    return sessions


def find_all_session_boundaries(df, fold_info):
    """
    Trouve toutes les sessions impactées par un fold anormal
    """
    session_markers = df['SessionStartEnd'].values
    timestamps = df['timeStampOpening'].values

    start_idx = np.where(timestamps == fold_info['stats']['start_time'])[0][0]
    end_idx = np.where(timestamps == fold_info['stats']['end_time'])[0][0]

    session_boundaries = find_session_boundaries_numba(session_markers, timestamps, start_idx, end_idx)

    sessions = []
    for start_idx, end_idx in session_boundaries:
        sessions.append({
            'start_idx': int(start_idx),
            'end_idx': int(end_idx),
            'start_time': timestamps[start_idx],
            'end_time': timestamps[end_idx],
            'rows': df.loc[start_idx:end_idx]
        })

    return sessions


def analyze_and_remove_anomalous_periods(df, anomalous_folds):
    """
    Analyse et affiche les sessions à supprimer pour chaque fold anormal.
    """
    all_sessions_to_remove = []
    print("\nDétail des sessions à supprimer par fold anormal :")
    print("=" * 100)

    for fold in anomalous_folds:
        print(f"\nFold {fold['fold']} (ratio anormal: {fold['stats']['val_ratio']:.2f}):")
        fold_start_date = timestamp_to_date_utc_(fold['stats']['start_time'])
        fold_end_date = timestamp_to_date_utc_(fold['stats']['end_time'])
        print(f"Période du fold - Du {fold_start_date} au {fold_end_date}")

        sessions = find_all_session_boundaries(df, fold)
        first_session_start = sessions[0]['start_time']
        last_session_end = sessions[-1]['end_time']

        print(f"Sessions impactées ({len(sessions)}):")
        total_trades = total_wins = total_losses = 0

        for i, session in enumerate(sessions, 1):
            session_start_date = timestamp_to_date_utc_(session['start_time'])
            session_end_date = timestamp_to_date_utc_(session['end_time'])

            session_data = session['rows']
            trades = len(session_data[session_data['class_binaire'] != 99])
            wins = len(session_data[session_data['class_binaire'] == 1])
            losses = len(session_data[session_data['class_binaire'] == 0])

            print(f"  Session {i}:")
            print(f"    Début : {session_start_date}")
            print(f"    Fin   : {session_end_date}")
            print(f"    Trades: {trades} (Wins: {wins}, Losses: {losses})")

            total_trades += trades
            total_wins += wins
            total_losses += losses

            all_sessions_to_remove.append({
                'start_time': session['start_time'],
                'end_time': session['end_time'],
                'start_date': session_start_date,
                'end_date': session_end_date,
                'fold_number': fold['fold'],
                'statistics': {
                    'trades': trades,
                    'wins': wins,
                    'losses': losses
                }
            })

        print(f"  Résumé du fold:")
        print(f"    Période totale - Du {timestamp_to_date_utc_(first_session_start)} "
              f"au {timestamp_to_date_utc_(last_session_end)}")
        print(f"    Total trades: {total_trades} (Wins: {total_wins}, Losses: {total_losses})")
        winrate = (total_wins / (total_wins + total_losses) * 100) if total_wins + total_losses > 0 else 0
        print(f"    Winrate: {winrate:.2f}%")
        print("-" * 80)

    df_cleaned = df.copy()
    rows_to_drop = []

    for session in all_sessions_to_remove:
        mask = (df_cleaned['timeStampOpening'] >= session['start_time']) & \
               (df_cleaned['timeStampOpening'] <= session['end_time'])
        rows_to_drop.extend(df_cleaned[mask].index)

    df_cleaned = df_cleaned.drop(rows_to_drop)

    return df_cleaned, all_sessions_to_remove


def create_comparative_visualizations(df_initial, df_cleaned):
    """
    Crée une visualisation comparative des datasets avant/après nettoyage
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    def prepare_data(df):
        """Prépare les données pour les visualisations"""
        total_trades = len(df)
        no_trades = len(df[df['class_binaire'] == 99])
        active_trades = len(df[df['class_binaire'].isin([0, 1])])
        successes = len(df[df['class_binaire'] == 1])
        failures = len(df[df['class_binaire'] == 0])

        return {
            'total': total_trades,
            'no_trades': no_trades,
            'active_trades': active_trades,
            'successes': successes,
            'failures': failures
        }

    def plot_position_pie(stats, ax, title):
        """Crée le graphique de répartition des positions"""
        sizes = [stats['no_trades'], stats['active_trades']]
        labels = [f'Aucun trade\n{stats["no_trades"]}',
                  f'Trades actifs\n{stats["active_trades"]}']
        colors = ['#3498db', '#e67e22']

        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                          autopct='%1.1f%%',
                                          pctdistance=0.85,
                                          labeldistance=1.05)
        plt.setp(texts, size=10, weight="bold")
        plt.setp(autotexts, size=9, weight="bold")
        ax.set_title(f'{title}\nTotal: {stats["total"]}', pad=20, size=14)

    def plot_success_bars(stats, ax, title):
        """Crée le graphique de répartition des résultats"""
        total_active = stats['successes'] + stats['failures']
        success_rate = (stats['successes'] / total_active * 100) if total_active > 0 else 0
        failure_rate = (stats['failures'] / total_active * 100) if total_active > 0 else 0

        ax.bar(['Trades'], [failure_rate], color='red', alpha=0.7, width=0.5,
               label='% trades échoués')
        ax.bar(['Trades'], [success_rate], bottom=[failure_rate], color='green',
               alpha=0.7, width=0.5, label='% trades réussis')

        # Ajout des pourcentages dans les barres
        ax.text(0, failure_rate / 2, f'{failure_rate:.1f}%', ha='center',
                va='center', color='white', weight='bold', size=12)
        ax.text(0, failure_rate + success_rate / 2, f'{success_rate:.1f}%',
                ha='center', va='center', color='white', weight='bold', size=12)

        ax.set_ylim(0, 110)
        ax.set_title(title, pad=20, size=14)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=12)

    def plot_monthly_distribution(df, ax, title):
        """Crée le graphique de distribution mensuelle"""
        df = df.copy()
        df['date'] = pd.to_datetime(df['timeStampOpening'].apply(lambda x:
                                                                 pd.Timestamp.fromtimestamp(x).strftime('%Y-%m-%d')))
        df['month'] = df['date'].dt.strftime('%y-%m')  # Format YY-MM

        monthly_data = df[df['class_binaire'].isin([0, 1])].groupby(['month', 'class_binaire']).size().unstack(
            fill_value=0)

        if not monthly_data.empty:
            monthly_total = monthly_data.sum(axis=1)
            monthly_pct = monthly_data.div(monthly_total, axis=0) * 100

            ax.stackplot(range(len(monthly_pct.index)),
                         [monthly_pct[col] for col in monthly_pct.columns],
                         labels=['Échecs', 'Succès'],
                         colors=['red', 'green'],
                         alpha=0.7)

            # Afficher tous les ticks avec des étiquettes plus petites
            ax.set_xticks(range(len(monthly_pct.index)))
            ax.set_xticklabels(monthly_pct.index,
                               rotation=45,
                               ha='right',
                               fontsize=8)

            ax.set_ylim(0, 100)
            ax.set_title(title, pad=20, size=14)
            ax.legend(fontsize=12, loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='y', which='major', labelsize=12)

            # Ajustement des marges et étiquettes
            ax.margins(x=0.01)
            ax.set_xlabel('Mois', fontsize=12, labelpad=10)
            ax.set_ylabel('Pourcentage', fontsize=12, labelpad=10)

        else:
            ax.text(0.5, 0.5, 'Pas de données disponibles',
                    ha='center', va='center', transform=ax.transAxes)

    def plot_success_pie(stats, ax, title):
        """Crée le graphique de répartition succès/échec"""
        sizes = [stats['failures'], stats['successes']]
        labels = [f'Trades échoués\n{stats["failures"]}',
                  f'Trades réussis\n{stats["successes"]}']
        colors = ['red', 'green']

        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                          autopct='%1.1f%%',
                                          pctdistance=0.85,
                                          labeldistance=1.05)
        plt.setp(texts, size=12, weight="bold")
        plt.setp(autotexts, size=12, weight="bold")
        ax.set_title(title, pad=20, size=14)

    # Création des figures avec une taille augmentée
    fig = plt.figure(figsize=(25, 15))
    fig2 = plt.figure(figsize=(25, 15))

    # Ajout des titres
    fig.suptitle('Données initiales', fontsize=18, y=0.95)
    fig2.suptitle('Données après nettoyage', fontsize=18, y=0.95)

    # Calcul des statistiques
    initial_stats = prepare_data(df_initial)
    cleaned_stats = prepare_data(df_cleaned)

    # Configuration des subplots avec plus d'espace
    gs = plt.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.5,
                      left=0.05, right=0.95, bottom=0.1, top=0.9)
    gs2 = plt.GridSpec(2, 2, figure=fig2, hspace=0.4, wspace=0.5,
                       left=0.05, right=0.95, bottom=0.1, top=0.9)

    # Dataset initial
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    plot_position_pie(initial_stats, ax1, "Répartition des positions")
    plot_monthly_distribution(df_initial, ax2, "Distribution mensuelle")
    plot_success_bars(initial_stats, ax3, "Résultats des trades")
    plot_success_pie(initial_stats, ax4, "Répartition succès/échec")

    # Dataset nettoyé
    ax5 = fig2.add_subplot(gs2[0, 0])
    ax6 = fig2.add_subplot(gs2[0, 1])
    ax7 = fig2.add_subplot(gs2[1, 0])
    ax8 = fig2.add_subplot(gs2[1, 1])

    plot_position_pie(cleaned_stats, ax5, "Répartition des positions")
    plot_monthly_distribution(df_cleaned, ax6, "Distribution mensuelle")
    plot_success_bars(cleaned_stats, ax7, "Résultats des trades")
    plot_success_pie(cleaned_stats, ax8, "Répartition succès/échec")

    # Ajustement des layouts
    fig.tight_layout(rect=[0, 0.03, 1, 0.92])
    fig2.tight_layout(rect=[0, 0.03, 1, 0.92])

    return fig, fig2


# Exemple d'utilisation:
# fig = create_comparative_visualizations(df_initial, df_cleaned)
# plt.show()

def main():
    # Définition des chemins et chargement des données
    FILE_NAME_ = "Step5_4_0_6TP_1SL_080919_141024_extractOnly220LastFullSession_OnlyShort_feat_winsorized.csv"
    DIRECTORY_PATH_ = r"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra chart\xTickReversal\simu\\4_0_6TP_1SL\merge"
    file_path = os.path.join(DIRECTORY_PATH_, FILE_NAME_)

    print("\nVoulez-vous afficher tous les folds ? (o/n)")
    display_all = input().lower() == 'o'

    initial_df = load_data(file_path)
    print("Taille initiale du DataFrame:", len(initial_df))

    # Analyser les splits et identifier les folds anormaux
    split_stats, anomalous_folds = analyze_splits(
        df=initial_df,
        X_train_full=initial_df,
        n_splits=550,
        filter_ratio_up=400,  # 400% au-dessus de la moyenne
        filter_ratio_down=80,  # 80% en-dessous de la moyenne
        display_all_folds=display_all
    )

    # Analyser et supprimer les sessions anormales
    df_cleaned, removed_sessions = analyze_and_remove_anomalous_periods(initial_df, anomalous_folds)

    # Calculer les statistiques
    initial_stats = calculate_stats(initial_df, "Avant nettoyage")
    cleaned_stats = calculate_stats(df_cleaned, "Après nettoyage")
    session_stats = calculate_session_averages(df_cleaned, removed_sessions)

    # Préparation des métadonnées
    metadata = {
        'cleaning_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'original_rows': len(initial_df),
        'cleaned_rows': len(df_cleaned),
        'removed_rows': len(initial_df) - len(df_cleaned),
        'removed_sessions': len(removed_sessions),
        'filter_ratio_up': 400,
        'filter_ratio_down': 80,
        'mean_ratio_before': initial_stats['wins'] / initial_stats['losses'] if initial_stats['losses'] > 0 else 0,
        'mean_ratio_after': cleaned_stats['wins'] / cleaned_stats['losses'] if cleaned_stats['losses'] > 0 else 0,
        'cleaned_sessions_count': session_stats['cleaned_total_sessions'],
        'cleaned_avg_observations_per_session': session_stats['cleaned_avg'],
        'removed_sessions_count': session_stats['removed_total_sessions'],
        'removed_avg_observations_per_session': session_stats['removed_avg']
    }

    # Sauvegarder les résultats
    #timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f"{file_path}_cleaned.csv"
    #metadata_filename = f"metadata_{timestamp}.json"

    output_path = os.path.join(DIRECTORY_PATH_, output_filename)
    #metadata_path = os.path.join(DIRECTORY_PATH_, metadata_filename)

    print(f'\nEcriture de {output_path}')
    print("=" * 50)

    #df_cleaned.to_csv(output_path, index=False, sep=';')
    #with open(metadata_path, 'w') as f:
     #   json.dump(metadata, f, indent=4)

    # Afficher le résumé final
    print("\nRésumé final de l'opération de nettoyage:")
    print("=" * 80)
    print(f"Données initiales:")
    print(f"  - Nombre total d'observations: {initial_stats['total']:,d}")
    print(f"  - Wins: {initial_stats['wins']:,d}")
    print(f"  - Losses: {initial_stats['losses']:,d}")
    print(f"  - Winrate initial: {initial_stats['winrate']:.2%}")

    print(f"\nDonnées nettoyées:")
    print(f"  - Nombre total d'observations: {cleaned_stats['total']:,d}")
    print(f"  - Wins: {cleaned_stats['wins']:,d}")
    print(f"  - Losses: {cleaned_stats['losses']:,d}")
    print(f"  - Winrate final: {cleaned_stats['winrate']:.2%}")
    print(f"  - Amélioration du winrate: {(cleaned_stats['winrate'] - initial_stats['winrate']) * 100:.2f} points")
    print(f"  - Nombre de sessions restantes: {session_stats['cleaned_total_sessions']}")
    print(f"  - Moyenne d'observations par session: {session_stats['cleaned_avg']:.1f}")

    print(f"\nSessions supprimées:")
    print(f"  - Nombre de sessions: {session_stats['removed_total_sessions']}")
    print(f"  - Moyenne d'observations par session: {session_stats['removed_avg']:.1f}")
    print(f"  - Total observations supprimées: {len(initial_df) - len(df_cleaned):,d}")
    print(f"  - Pourcentage de données conservées: {(len(df_cleaned) / len(initial_df)) * 100:.2f}%")

    print(f"\nFichiers de sortie:")
    print(f"  - Données nettoyées: {output_filename}")
    #print(f"  - Métadonnées: {metadata_filename}")

    return initial_df,df_cleaned, removed_sessions, metadata


if __name__ == "__main__":
    try:
        print("Début de l'analyse et du nettoyage des données...")
        initial_df, df_cleaned, removed_sessions, metadata = main()

        # Création des visualisations
        fig1, fig2 = create_comparative_visualizations(initial_df, df_cleaned)
        plt.show()

        print("\nOpération terminée avec succès!")

    except Exception as e:
        print(f"\nUne erreur s'est produite: {str(e)}")
        raise