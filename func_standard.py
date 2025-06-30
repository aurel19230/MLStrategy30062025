import calendar

from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, \
    average_precision_score, matthews_corrcoef, precision_recall_curve, precision_score
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler
# Put this at the top of your script or in a utility file
import matplotlib.cbook
if not hasattr(matplotlib.cbook, '_Stack') and hasattr(matplotlib.cbook, 'Stack'):
    matplotlib.cbook._Stack = matplotlib.cbook.Stack
from numba import njit, prange

import matplotlib.ticker as ticker
from PIL import Image
import sys
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.feature_selection import RFECV
import csv
import math
from functools import partial
from sklearn.model_selection import KFold, TimeSeriesSplit
import json
import shutil
from parameters import *
from definition import *
from func_xgb import *
from func_lightgbm import *
from func_RF import *
from func_SVC import *
import chardet
x_train_vifStatPCA_initDataSet_DEBUG_global = pd.DataFrame()

def setup_plotting_style():
    """Configuration optimisée du style des graphiques"""
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams.update({
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9
    })
def detect_file_encoding(file_path):
    """
    Détecte l'encodage d'un fichier.

    Args:
        file_path: Chemin du fichier

    Returns:
        Encodage détecté (str)
    """
    with open(file_path, 'rb') as file:
        # Lire un échantillon du fichier (les premiers 100 Ko devraient suffire)
        raw_data = file.read(100000)
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        confidence = result['confidence']
        print_notification(f"Encodage détecté : {encoding} (confiance : {confidence:.2f})")
        return encoding
def load_csv_with_encoding(file_path, encodings=None):
    """
    Essaie de charger un fichier CSV avec différents encodages.

    Args:
        file_path: Chemin du fichier CSV
        encodings: Liste d'encodages à essayer (si None, détectera automatiquement)

    Returns:
        DataFrame pandas ou None si échec
    """
    if encodings is None:
        # Si aucun encodage fourni, essayer de détecter
        detected_encoding = detect_file_encoding(file_path)
        encodings = [detected_encoding, 'latin1', 'utf-8', 'cp1252', 'ISO-8859-1']

    # Supprimer les doublons et placer l'encodage détecté en premier
    encodings = list(dict.fromkeys(encodings))

    for encoding in encodings:
        try:
            print_notification(f"Tentative de chargement avec l'encodage: {encoding}")

            # Ajout de low_memory=False pour éviter les DtypeWarning
            df = pd.read_csv(file_path, sep=';', encoding=encoding, low_memory=False)

            # Afficher les types de données détectés pour les colonnes
            print_notification("Types de données détectés :")
            for col, dtype in df.dtypes.items():
                print(f"  - {col}: {dtype}")

            print_notification(f"Chargement réussi avec l'encodage: {encoding}")
            return df, encoding
        except UnicodeDecodeError:
            print_notification(f"Échec avec l'encodage: {encoding}")
        except Exception as e:
            print_notification(f"Autre erreur lors du chargement avec {encoding}: {e}")

    return None, None



CUSTOM_SESSIONS = {
    "Opening": {
        "start": 0,
        "end": 120,
        "session_type_index": 0,
        "selected": False,
        "description": "22:00-23:05",
    },
    "Asie": {
        "start": 120,
        "end": 535,
        "session_type_index": 1,
        "selected": True,
        "description": "23:05-6:55",
    },
    "OpenUk": {
        "start": 535,
        "end": 545,
        "session_type_index": 2,
        "selected": False,
        "description": "6:55-07:05",
    },
    "preOpenEurope": {
        "start": 545,
        "end": 595,
        "session_type_index": 2,
        "selected": True,
        "description": "07:05-07:55",
    },
    "OpenEurope": {  # 8h = 480 minutes -> 475-485
        "start": 595,
        "end": 605,
        "session_type_index": 2,
        "selected": False,
        "description": "07:55-08:05",
    },
    "MorningEurope": {
        "start": 605,
        "end": 865,
        "session_type_index": 2,
        "selected": True,
        "description": "08:05-12:25",
    },
    "OpenpreOpenUS": {  # 12h30 = 750 minutes -> 745-755
        "start": 865,
        "end": 875,
        "session_type_index": 3,
        "selected": False,
        "description": "12:25-12:35",
    },
    "preOpenUS": {
        "start": 875,
        "end": 925,
        "session_type_index": 3,
        "selected": True,
        "description": "12:35-13:25",
    },
    "OpenMoringUS": {  # 13h30 = 810 minutes -> 805-815
        "start": 925,
        "end": 935,
        "session_type_index": 3,
        "selected": False,
        "description": "13:25-13:35",
    },
    "MoringUS": {
        "start": 935,
        "end": 1065,
        "session_type_index": 3,
        "selected": True,
        "description": "13:35-15:45",
    },
    "AfternonUS": {
        "start": 1065,
        "end": 1195,
        "session_type_index": 4,
        "selected": True,
        "description": "15:45-17:55",
    },
    "Evening": {
        "start": 1195,
        "end": 1280,
        "session_type_index": 5,
        "selected": True,
        "description": "17:55-19:20",
    },
    "Close": {
        "start": 1280,
        "end": 1380,
        "session_type_index": 6,
        "selected": True,
        "description": "19:20-21:00",
    }
}

from functools import reduce
from statsmodels.stats.outliers_influence import variance_inflation_factor


# Variable globale pour suivre si la fonction a déjà été appelée
_first_call_save_r_trialesults = True


def detect_environment():
    """
    Détecte l'environnement d'exécution
    Returns: 'colab', 'pycharm', ou 'other'
    """
    # Vérification pour Google Colab
    try:
        from google.colab import drive
        return 'colab'
    except ImportError:
        # Vérification pour PyCharm
        if 'PYCHARM_HOSTED' in os.environ or 'PYCHARM_MATPLOTLIB_PORT' in os.environ:
            return 'pycharm'
        return 'other'


import torch


def date_to_timestamp_utc(year, month, day, hour, minute, second):  # from c++ encoding
    timeinfo = (year, month, day, hour, minute, second)
    return calendar.timegm(timeinfo)


# Fonction pour convertir une chaîne de caractères en timestamp Unix
def convert_to_unix_timestamp(date_string):
    date_format = "%Y-%m-%d %H:%M:%S"  # Modifier le format de date
    dt = datetime.strptime(date_string, date_format)
    return int(dt.timestamp())


# Fonction pour convertir un timestamp Unix en chaîne de caractères de date
def convert_from_unix_timestamp(unix_timestamp):
    date_format = "%Y-%m-%d %H:%M:%S"  # Modifier le format de date
    dt = datetime.fromtimestamp(unix_timestamp)
    return dt.strftime(date_format)


def plot_feature_histograms_by_class(data, className, column_settings, figsize=(32, 24)):
    columns = list(column_settings.keys())
    n_columns = len(columns)
    ncols = 6
    nrows = (n_columns + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    for i, column in enumerate(columns):
        # Compter le nombre d'observations pour chaque classe
        class_0_count = data[data[className] == 0].shape[0]
        class_1_count = data[data[className] == 1].shape[0]

        # Tracer l'histogramme pour la classe 0 en rouge avec densité
        sns.histplot(data=data[data[className] == 0], x=column, color='red', label='Classe 0', bins=300, kde=True,
                     stat='density', common_norm=True, ax=axes[i], alpha=0.5)

        # Tracer l'histogramme pour la classe 1 en bleu avec densité
        sns.histplot(data=data[data[className] == 1], x=column, color='blue', label='Classe 1', bins=300, kde=True,
                     stat='density', common_norm=True, ax=axes[i], alpha=0.5)

        # Ajouter un titre
        axes[i].set_title(f'{column}')

        # Supprimer le label de l'axe des abscisses
        axes[i].set_xlabel('')

        # Ajouter le label de l'axe des ordonnées
        axes[i].set_ylabel('')

        # Ajouter une légende
        axes[i].legend()

        # Ajouter le décompte des classes au graphique
        axes[i].text(0.95, 0.95, f'Classe 0: {class_0_count}\nClasse 1: {class_1_count}', transform=axes[i].transAxes,
                     fontsize=8, horizontalalignment='right', verticalalignment='top',
                     bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()
    plt.show()


import pandas as pd
import numpy as np

import pandas as pd
import numpy as np


def split_sessions(df, test_size=0.2, min_train_sessions=2, min_test_sessions=2):
    session_col = "sc_sessionStartEnd"

    # Identifier les sessions complètes
    session_starts = df[df[session_col] == 10].index
    session_ends = df[df[session_col] == 20].index

    print(f"Nombre de débuts de session détectés : {len(session_starts)}")
    print(f"Nombre de fins de session détectées : {len(session_ends)}")

    sessions = []
    excluded_rows = 0
    orphan_starts = 0
    orphan_ends = 0

    for start, end in zip(session_starts, session_ends):
        if start < end:  # Vérifier que la session est complète
            session = df.loc[start:end]
            if session.iloc[0][session_col] == 10 and session.iloc[-1][session_col] == 20:
                sessions.append(session)
                # print(f"Session extraite de l'index {start} à {end}, comprenant {len(session)} lignes. "
                # f"Valeur de {session_col} au début: {session.iloc[0][session_col]}, "
                # f"à la fin: {session.iloc[-1][session_col]}")
            else:
                excluded_rows += len(session)
                print(colored(f"Session invalide ignorée de l'index {start} à {end}. "
                              f"Valeur de {session_col} au début: {session.iloc[0][session_col]}, "
                              f"à la fin: {session.iloc[-1][session_col]}", "red"))
        else:
            excluded_rows += end - start + 1
            orphan_starts += 1
            orphan_ends += 1

    # Gérer les débuts ou fins de session orphelins
    if len(session_starts) > len(session_ends):
        orphan_starts += len(session_starts) - len(session_ends)
    elif len(session_ends) > len(session_starts):
        orphan_ends += len(session_ends) - len(session_starts)

    total_sessions = len(sessions)
    print(f"Nombre de sessions complètes et valides extraites : {total_sessions}")

    # Compter les lignes avant la première session et après la dernière
    rows_before_first = session_starts[0] - df.index[0] if len(session_starts) > 0 else 0
    rows_after_last = df.index[-1] - session_ends[-1] if len(session_ends) > 0 else 0
    excluded_rows += rows_before_first + rows_after_last

    # print(f"Lignes avant la première session : {rows_before_first}")
    # print(f"Lignes après la dernière session : {rows_after_last}")
    # print(f"Débuts de session orphelins : {orphan_starts}")
    # print(f"Fins de session orphelines : {orphan_ends}")
    # print(f"Nombre total de lignes exclues : {excluded_rows}")

    # Vérifier s'il y a des sessions orphelines et lever une erreur si c'est le cas
    if orphan_starts > 0 or orphan_ends > 0:
        error_message = f"Erreur : Sessions orphelines détectées. Débuts orphelins : {orphan_starts}, Fins orphelines : {orphan_ends}"
        raise ValueError(error_message)

    if not sessions:
        raise ValueError("Aucune session complète et valide détectée dans les données.")

    # Calculer le nombre de sessions de test nécessaires
    test_sessions_count = max(min_test_sessions, int(total_sessions * test_size))
    min_required_sessions = min_train_sessions + test_sessions_count

    if total_sessions < min_required_sessions:
        raise ValueError(f"Nombre insuffisant de sessions. "
                         f"Trouvé : {total_sessions}, "
                         f"Minimum requis : {min_required_sessions} "
                         f"({min_train_sessions} pour l'entraînement, {test_sessions_count} pour le test). "
                         f"Veuillez fournir plus de données.")

    # Ajuster le nombre de sessions de test si nécessaire
    if total_sessions - test_sessions_count < min_train_sessions:
        test_sessions_count = total_sessions - min_train_sessions

    # Diviser les sessions en train et test
    train_sessions = sessions[:-test_sessions_count]
    test_sessions = sessions[-test_sessions_count:]

    # Vérification finale
    if len(train_sessions) < min_train_sessions:
        raise ValueError(f"Nombre insuffisant de sessions d'entraînement. "
                         f"Trouvé : {len(train_sessions)}, Minimum requis : {min_train_sessions}")
    if len(test_sessions) < min_test_sessions:
        raise ValueError(f"Nombre insuffisant de sessions de test. "
                         f"Trouvé : {len(test_sessions)}, Minimum requis : {min_test_sessions}")

    # Combiner les sessions en DataFrames
    train_df = pd.concat(train_sessions)
    test_df = pd.concat(test_sessions)

    # Afficher les informations
    print(f"Nombre total de lignes dans le DataFrame original : {len(df)}")
    print(f"Nombre de lignes dans le train : {len(train_df)}")
    print(f"Nombre de lignes dans le test : {len(test_df)}")
    print(f"Nombre de sessions dans le train : {len(train_sessions)}")
    print(f"Nombre de sessions dans le test : {len(test_sessions)}")

    total_included_rows = len(train_df) + len(test_df)
    print(f"Nombre total de lignes incluses : {total_included_rows}")
    print(f"Différence avec le total : {len(df) - total_included_rows}")
    print("\n")
    return train_df, len(train_sessions), test_df, len(test_sessions)


from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

if platform.system() != "Darwin":  # "Darwin" est le nom interne de macOS
    import cupy as cp
else:
    print("CuPy ne sera pas importé sur macOS.")


def check_gpu_availability():
    torch_available = torch.cuda.is_available()
    if platform.system() != "Darwin":  # "Darwin" est le nom interne de macOS
        import cupy as cp
        cupy_available = cp.cuda.is_available()
    else:
        print("CuPy ne sera pas utilisé sur macOS.")

    if platform.system() != "Darwin":  # "Darwin" est le nom interne de macOS
        import cupy as cp

        if not (torch_available and cupy_available):
            print("Erreur : GPU n'est pas disponible pour PyTorch et/ou CuPy. Le programme va s'arrêter.")
            if not torch_available:
                print("PyTorch ne détecte pas de GPU.")
            if not cupy_available:
                print("CuPy ne détecte pas de GPU.")
            exit(1)
        print("GPU est disponible. Utilisation de CUDA pour les calculs.")
        print(f"GPU détecté par PyTorch : {torch.cuda.get_device_name(0)}")
        print(f"GPU détecté par CuPy : {cp.cuda.runtime.getDeviceProperties(cp.cuda.Device())['name'].decode()}")
        # Affichage de la mémoire GPU disponible
        torch_memory = torch.cuda.get_device_properties(0).total_memory
        cupy_memory = cp.cuda.runtime.memGetInfo()[1]

        print(f"Mémoire GPU totale (PyTorch) : {torch_memory / 1e9:.2f} GB")
        print(f"Mémoire GPU totale (CuPy) : {cupy_memory / 1e9:.2f} GB")
        # Vérification de la version CUDA
        torch_cuda_version = torch.version.cuda
        cupy_cuda_version = cp.cuda.runtime.runtimeGetVersion()

        print(f"Version CUDA pour PyTorch : {torch_cuda_version}")
        print(f"Version CUDA pour CuPy : {cupy_cuda_version}")

        if torch_cuda_version != cupy_cuda_version:
            print("Attention : Les versions CUDA pour PyTorch et CuPy sont différentes.")
            print("Cela pourrait causer des problèmes de compatibilité.")

    else:
        print("CuPy et Torcn ne seront pas importés sur macOS.")


def plot_calibrationCurve_distrib(y_true, y_pred_proba, n_bins=200, strategy='uniform',
                                  optimal_optuna_threshold=None, show_histogram=True, num_sessions=25,
                                  results_directory=None, config=None):
    """
    Trace une courbe de calibration et une distribution des prédictions.
    ...
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss

    if config is None or 'device_' not in config:
        raise ValueError("Le paramètre 'config' doit contenir une clé 'device_'.")

    if results_directory is None:
        raise ValueError("Le paramètre 'results_directory' doit être spécifié.")
    os.makedirs(results_directory, exist_ok=True)

    y_true = np.array(y_true)

    # Vérifier la configuration CUDA et importer Cupy si nécessaire
    if config['device_'] == 'cuda':
        try:
            import cupy as cp
            # Si y_pred_proba est un tableau CuPy, on le convertit
            if isinstance(y_pred_proba, cp.ndarray):
                y_pred_proba = y_pred_proba.get()
        except ImportError:
            raise ImportError("CuPy n'est pas installé, mais 'device_' est défini sur 'cuda'.")
    # Sinon, on ne touche pas à y_pred_proba s'il est déjà NumPy

    # Vérification de la longueur
    if len(y_true) != len(y_pred_proba):
        raise ValueError("'y_true' et 'y_pred_proba' doivent avoir la même longueur.")

    # Vérification du seuil
    if optimal_optuna_threshold is None:
        raise ValueError("The 'optimal_optuna_threshold' parameter must be provided.")

    # Début du plotting
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # 1) Calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=50, strategy=strategy)
    brier_score = brier_score_loss(y_true, y_pred_proba)

    # Amélioration de la visualisation
    axes[0].plot(prob_pred, prob_true, marker='o', linewidth=2, markersize=4,
                 color='blue', label='Calibration curve')
    axes[0].plot([0, 1], [0, 1], linestyle='--', color='gray',
                 label='Perfectly calibrated', linewidth=1.5)

    # Ajout d'une grille plus fine
    axes[0].grid(True, which='major', linestyle='-', alpha=0.3)
    axes[0].grid(True, which='minor', linestyle=':', alpha=0.2)
    axes[0].minorticks_on()

    # Ajustement des limites et des marges
    axes[0].set_xlim(-0.05, 1.05)
    axes[0].set_ylim(-0.05, 1.05)

    axes[0].set_title(f'Calibration Curve (Reliability Diagram)\nBrier Score: {brier_score:.4f}',
                      fontsize=12, pad=15)
    axes[0].set_xlabel('Mean Predicted Probability', fontsize=12, labelpad=10)
    axes[0].set_ylabel('Fraction of Positives', fontsize=12, labelpad=10)
    axes[0].legend(loc='upper left', fontsize=10, framealpha=0.9)

    # 2) Histogram / Distribution (TP, FP, TN, FN)
    if show_histogram:
        bins = np.linspace(0, 1, n_bins + 1)
        bin_width = bins[1] - bins[0]
        bin_centers = bins[:-1] + bin_width / 2

        tp_counts = np.zeros(n_bins)
        fp_counts = np.zeros(n_bins)
        tn_counts = np.zeros(n_bins)
        fn_counts = np.zeros(n_bins)

        bin_indices = np.digitize(y_pred_proba, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        for i in range(len(y_true)):
            bin_idx = bin_indices[i]
            actual = y_true[i]
            predicted_proba = y_pred_proba[i]
            predicted_label = int(predicted_proba >= optimal_optuna_threshold)

            if actual == 1 and predicted_label == 1:
                tp_counts[bin_idx] += 1
            elif actual == 0 and predicted_label == 1:
                fp_counts[bin_idx] += 1
            elif actual == 0 and predicted_label == 0:
                tn_counts[bin_idx] += 1
            elif actual == 1 and predicted_label == 0:
                fn_counts[bin_idx] += 1

        bottom_tn = np.zeros(n_bins)
        bottom_fp = tn_counts
        bottom_fn = tn_counts + fp_counts
        bottom_tp = tn_counts + fp_counts + fn_counts

        axes[1].bar(bin_centers, tn_counts, width=bin_width, label='TN', alpha=0.7, color='green')
        axes[1].bar(bin_centers, fp_counts, width=bin_width, bottom=bottom_tn, label='FP', alpha=0.7, color='red')
        axes[1].bar(bin_centers, fn_counts, width=bin_width, bottom=bottom_fp, label='FN', alpha=0.7, color='orange')
        axes[1].bar(bin_centers, tp_counts, width=bin_width, bottom=bottom_fn, label='TP', alpha=0.7, color='blue')

        axes[1].axvline(x=optimal_optuna_threshold, color='black', linestyle='--',
                        label=f'Threshold ({optimal_optuna_threshold:.2f})')
        axes[1].set_title('Repartition des TP, FP, TN, FN per Bin Test Set', fontsize=12)
        axes[1].set_xlabel('Proportion de prédictions négatives (fonction du choix du seuil)', fontsize=12)
        axes[1].set_ylabel('Count', fontsize=12)
        axes[1].legend(loc='upper center', fontsize=10, ncol=5)
        axes[1].grid(True)

        total_tp = int(np.sum(tp_counts))
        total_fp = int(np.sum(fp_counts))
        total_tn = int(np.sum(tn_counts))
        total_fn = int(np.sum(fn_counts))
        total_samples = len(y_true)

        accuracy = (total_tp + total_tn) / total_samples
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        total_trades = total_tp + total_fp
        winrate = total_tp / total_trades * 100 if total_trades > 0 else 0

        annotation_text = (
            f'Total Samples: {total_samples}\n'
            f'TP: {total_tp}\nFP: {total_fp}\nTN: {total_tn}\nFN: {total_fn}\n'
            f'Winrate: {winrate:.2f}%\n'
            f'Accuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\n'
            f'Recall: {recall:.4f}\nF1 Score: {f1_score:.4f}\n'
            f'Nombre de Session: {num_sessions}'
        )

        axes[1].text(0.02, 0.98, annotation_text,
                     transform=axes[1].transAxes, va='top', ha='left', fontsize=10,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(results_directory, 'calibration_and_distribution.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def to_numpy(array):
    return cp.asnumpy(array) if isinstance(array, cp.ndarray) else array


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


# import cupy as cp  # Décommentez si vous utilisez des tableaux CuPy

def to_numpy(array):
    """Convertit les tableaux CuPy en tableaux NumPy, si applicable."""
    try:
        import cupy as cp
        if isinstance(array, cp.ndarray):
            return cp.asnumpy(array)
    except ImportError:
        pass
    return np.asarray(array)


def plot_fp_tp_rates(X=None, y_label=None, y_pred_proba_afterSig=None, feature_deltaTime_name=None,
                     optimal_optuna_threshold=None, dataset_name=None, results_directory=None):
    """
    Trace les taux de FP et TP empilés en fonction du temps sur une période de 22h à 21h.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12), facecolor='white')

    def format_time_label(minutes):
        total_minutes = minutes + (22 * 60)  # Ajouter 22h en minutes
        total_minutes = total_minutes % (24 * 60)  # Modulo pour revenir à 0 après 24h
        hours = total_minutes // 60
        minutes = total_minutes % 60
        return f"{int(hours):02d}:{int(minutes):02d}"

    def plot_rates(ax, n_bins):
        bins = np.linspace(0, 1380, n_bins + 1)

        # Convertir explicitement les tableaux en NumPy
        feature_values_np = to_numpy(X[feature_deltaTime_name].values)
        y_label_np = to_numpy(y_label)
        y_predProba_np = to_numpy(y_pred_proba_afterSig)
        print(f"Longueur de feature_values_np : {len(feature_values_np)}")
        print(f"Longueur de y_test_label_np : {len(y_label_np)}")
        print(f"Longueur de y_test_predProba_np : {len(y_predProba_np)}")
        df = pd.DataFrame({
            'feature': feature_values_np,
            'y_true': y_label_np,
            'y_pred': y_predProba_np >= optimal_optuna_threshold
        })

        # Création des bins et calcul des taux
        df['bin'] = pd.cut(df['feature'], bins=bins, include_lowest=True)
        grouped = df.groupby('bin')

        rates = grouped.apply(lambda x: pd.Series({
            'FP_rate': ((x['y_pred'] == 1) & (x['y_true'] == 0)).sum() / len(x) if len(x) > 0 else 0,
            'TP_rate': ((x['y_pred'] == 1) & (x['y_true'] == 1)).sum() / len(x) if len(x) > 0 else 0
        }))

        # Récupérer les bords gauches des bins pour positionner les barres
        bin_edges = rates.index.categories.left
        bin_widths = rates.index.categories.right - rates.index.categories.left

        # Ajuster la largeur des barres pour un léger espacement
        bar_widths = bin_widths * 0.98

        # Tracer les barres empilées
        ax.bar(bin_edges, rates['TP_rate'], width=bar_widths, align='edge',
               color='green', label='Taux de Vrais Positifs', alpha=0.7)
        ax.bar(bin_edges, rates['FP_rate'], width=bar_widths, align='edge',
               bottom=rates['TP_rate'], color='red', label='Taux de Faux Positifs', alpha=0.7)

        ax.set_xlim(bins[0], bins[-1])
        ax.set_ylim(0, 0.30)  # Limite de l'axe Y à 30%

        # Configuration des labels de l'axe X
        hour_marks = np.arange(0, 1381, 60)
        hour_labels = [format_time_label(m) for m in hour_marks]

        ax.set_xticks(hour_marks)
        ax.set_xticklabels(hour_labels, rotation=45, ha='right', fontsize=10)

        ax.set_ylabel('Taux', fontsize=12)
        ax.set_title(f'Taux de FP et TP par {feature_deltaTime_name} (bins={n_bins})', fontsize=14)
        ax.legend(loc='upper right', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)

    # Tracer les deux graphiques avec les différents nombres de bins
    plot_rates(ax1, 25)
    plot_rates(ax2, 75)

    # Ajustement de la mise en page
    plt.tight_layout()

    if results_directory:
        file_path = os.path.join(results_directory, f'fp_tp_rates_{dataset_name}_by_{feature_deltaTime_name}.png')
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Graphique sauvegardé sous : {file_path}")

    plt.close()


import numba as nb


def calculate_weighted_adjusted_score_custom(scores, weight_split, nb_split_weight, std_penalty_factor=1.0):
    """
    Calcule le score ajusté pondéré avec gestion spéciale du cas nb_split_weight=0

    Args:
        scores: Liste des scores PNL pour chaque split (Peut être NumPy, liste Python, ou CuPy)
        weight_split: Poids à appliquer aux splits les plus anciens
        nb_split_weight: Nombre de splits auxquels appliquer le poids spécifique
        std_penalty_factor: Facteur de pénalité pour l'écart-type
    """
    # Import numpy ici si ce n'est pas déjà fait plus haut dans le code
    import numpy as np

    # Si `scores` est un tableau CuPy, convertissez-le en NumPy
    # Vérification par hasattr:
    if hasattr(scores, 'get'):
        scores = scores.get()  # Convertit un tableau CuPy en tableau NumPy

    # Si `scores` est une liste de tableaux CuPy, convertissez chaque élément
    # dans le cas où `scores` est une liste.
    # Cela dépend de la structure de `scores`. Si c'est directement un Cupy array,
    # la conversion ci-dessus suffit. Sinon, vous pouvez faire:
    # scores = [s.get() if hasattr(s, 'get') else s for s in scores]

    scores = np.array(scores)

    if nb_split_weight > len(scores):
        raise ValueError("nb_split_weight ne peut pas être supérieur au nombre de scores.")

    if nb_split_weight == 0:
        # Cas sans pondération : calcul traditionnel
        mean = np.mean(scores)
        std = np.std(scores, ddof=1)  # ddof=1 pour écart-type non biaisé
        return mean - std_penalty_factor * std, mean, std

    # Cas avec pondération
    weights = np.array([weight_split] * nb_split_weight + [1] * (len(scores) - nb_split_weight))
    weights = weights / np.sum(weights)

    weighted_mean = np.sum(scores * weights)
    squared_diff = (scores - weighted_mean) ** 2
    weighted_variance = np.sum(weights * squared_diff) / (1 - np.sum(weights ** 2))
    weighted_std = np.sqrt(weighted_variance)

    return weighted_mean - std_penalty_factor * weighted_std, weighted_mean, weighted_std


def analyze_shap_values(model, X, y, dataset_name, create_dependence_plots=False, max_dependence_plots=3,
                        save_dir='./shap_dependencies_results/'):
    """
    Analyse les valeurs SHAP pour un ensemble de données et génère des visualisations.
    Version corrigée avec calcul approprié des importances SHAP et tri par valeur absolue.

    Parameters:
    -----------
    model : object
        Le modèle entraîné pour lequel calculer les valeurs SHAP.
    X : pandas.DataFrame
        Les features de l'ensemble de données.
    y : pandas.Series
        Les labels de l'ensemble de données.
    dataset_name : str
        Le nom de l'ensemble de données, utilisé pour nommer les fichiers de sortie.
    create_dependence_plots : bool, optional (default=False)
        Si True, crée des graphiques de dépendance pour les features les plus importantes.
    max_dependence_plots : int, optional (default=3)
        Le nombre maximum de graphiques de dépendance à créer si create_dependence_plots est True.
    save_dir : str, optional (default='./shap_dependencies_results/')
        Le répertoire où sauvegarder les graphiques générés et le fichier CSV.

    Returns:
    --------
    Tuple[numpy.ndarray, shap.Explanation, dict]
        - Les valeurs SHAP calculées
        - L'objet d'explication SHAP
        - Un dictionnaire contenant les résultats clés de l'analyse
    """
    # Création du répertoire principal de sauvegarde
    os.makedirs(save_dir, exist_ok=True)

    # Création du sous-répertoire pour les graphiques de dépendance
    dependence_dir = os.path.join(save_dir, 'dependence_plot')
    os.makedirs(dependence_dir, exist_ok=True)

    print('save_dir:',save_dir)
    print('dependence_dir:',dependence_dir)


    print(f"\n=== Analyse SHAP pour {dataset_name} ===")
    print(f"Dimensions des données: {X.shape}")

    # Création de l'explainer SHAP
    try:
        explainer = shap.TreeExplainer(model)
        print("TreeExplainer créé avec succès")
    except Exception as e:
        print(f"Erreur lors de la création de l'explainer: {str(e)}")
        traceback.print_exc()
        raise

    # Calcul des valeurs SHAP
    try:
        shap_values_explanation = explainer(X)
        shap_values = explainer.shap_values(X)
        print("Valeurs SHAP calculées avec succès")

        # Affichage des dimensions des valeurs SHAP
        if isinstance(shap_values, list):
            print(f"Type de sortie: Liste de {len(shap_values)} tableaux")
            for i, sv in enumerate(shap_values):
                print(f"Shape des valeurs SHAP classe {i}: {sv.shape}")
        else:
            print(f"Shape des valeurs SHAP: {shap_values.shape}")

    except Exception as e:
        print(f"Erreur lors du calcul des valeurs SHAP: {str(e)}")
        traceback.print_exc()
        raise

    # Gestion des problèmes de classification binaire
    if isinstance(shap_values, list) and len(shap_values) == 2:
        print("Détection de classification binaire - utilisation des valeurs de la classe positive")
        shap_values = shap_values[1]

    # Création du graphique résumé
    try:
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        plt.title(f"SHAP Feature Importance - {dataset_name}")
        plt.tight_layout()
        summary_path = os.path.join(save_dir, f'shap_importance_{dataset_name}.png')
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Graphique résumé SHAP créé et sauvegardé dans: {summary_path}")
    except Exception as e:
        print(f"Erreur lors de la création du graphique résumé: {str(e)}")
        traceback.print_exc()

    # Calcul des importances SHAP
    try:
        # Calcul de la moyenne des valeurs absolues des valeurs SHAP (importance globale)
        feature_importance = np.abs(shap_values).mean(axis=0)

        # Calcul de la moyenne des valeurs SHAP (effet moyen signé)
        feature_effect = np.mean(shap_values, axis=0)

        # Appliquer le signe de l'effet moyen à l'importance
        signed_feature_importance = feature_importance * np.sign(feature_effect)

        # Création du DataFrame avec les importances
        shap_df = pd.DataFrame({
            'feature': X.columns,
            'importance': signed_feature_importance,
        })

        # Ajout de la colonne des valeurs absolues pour le tri
        shap_df['abs_importance'] = np.abs(shap_df['importance'])

        # Tri par importance absolue décroissante
        shap_df = shap_df.sort_values('abs_importance', ascending=False)

        # Calcul des métriques cumulatives
        total_abs_importance = shap_df['abs_importance'].sum()
        shap_df['importance_percentage'] = (shap_df['abs_importance'] / total_abs_importance) * 100
        shap_df['cumulative_importance_percentage'] = shap_df['importance_percentage'].cumsum()

        # Créer le graphique personnalisé au lieu du summary_plot
        create_custom_importance_plot(shap_df, dataset_name, save_dir)

        # Nettoyage avant sauvegarde
        shap_df_save = shap_df.drop('abs_importance', axis=1)

        # Sauvegarde du CSV
        csv_path = os.path.join(save_dir, f'shap_values_{dataset_name}.csv')
        shap_df_save.to_csv(csv_path, index=False, sep=';')
        print(f"\nValeurs SHAP sauvegardées dans: {csv_path}")

        # Création du dictionnaire results avant la création des graphiques de dépendance
        results = {
            'feature_importance': shap_df_save,
            'top_10_features': shap_df['feature'].head(40).tolist(),
            'features_for_80_percent': shap_df[shap_df['cumulative_importance_percentage'] <= 80].shape[0]
        }

        print(f"\nTop 10 features basées sur l'analyse SHAP:")
        print(results['top_10_features'][:10])  # N'affiche que les 10 premiers
        print(
            f"\nNombre de features nécessaires pour expliquer 80% de l'importance : {results['features_for_80_percent']}")

    except Exception as e:
        print(f"Erreur lors du calcul des importances: {str(e)}")
        traceback.print_exc()
        raise

        # Création des graphiques de dépendance si demandé
    if create_dependence_plots:
        print("\nCréation des graphiques de dépendance...")
        most_important_features = shap_df['feature'].head(max_dependence_plots)

        for idx, feature in enumerate(most_important_features):
            try:
                plt.figure(figsize=(10, 6))
                shap.dependence_plot(feature, shap_values, X, show=False)
                plt.title(f"SHAP Dependence Plot - {feature} - {dataset_name}")
                plt.tight_layout()

                # Ajout du préfixe numérique au nom du fichier
                dep_plot_path = os.path.join(dependence_dir, f'{idx}_shap_dependence_{feature}_{dataset_name}.png')
                plt.savefig(dep_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Graphique de dépendance créé pour {feature}")
            except Exception as e:
                print(f"Erreur lors de la création du graphique de dépendance pour {feature}: {str(e)}")
                traceback.print_exc()

    print(f"\nAnalyse SHAP terminée - Tous les résultats sont sauvegardés dans: {save_dir}")

    return shap_values, shap_values_explanation, results


def compare_feature_importance(shap_values_train, shap_values_test, X_train, X_test,
                               save_dir='./shap_dependencies_results/', top_n=20):
    """
    Compare l'importance des features entre l'ensemble d'entraînement et de test.
    Si X_test est None, affiche uniquement les importances de l'ensemble d'entraînement.

    Parameters:
    -----------
    shap_values_train : numpy.ndarray
        Valeurs SHAP pour l'ensemble d'entraînement
    shap_values_test : numpy.ndarray ou None
        Valeurs SHAP pour l'ensemble de test, peut être None si pas de test
    X_train : pandas.DataFrame
        DataFrame contenant les features d'entraînement
    X_test : pandas.DataFrame ou None
        DataFrame contenant les features de test, peut être None si pas de test
    save_dir : str, optional
        Répertoire où sauvegarder le graphique (par défaut './shap_dependencies_results/')
    top_n : int, optional
        Nombre de top features à afficher dans le graphique (par défaut 20)

    Returns:
    --------
    pandas.DataFrame
        DataFrame contenant les importances des features et leurs différences si X_test est disponible,
        sinon uniquement les importances de l'ensemble d'entraînement
    """
    # Créer le répertoire de sauvegarde s'il n'existe pas
    os.makedirs(save_dir, exist_ok=True)

    importance_train = np.abs(shap_values_train).mean(0)

    if X_test is not None and shap_values_test is not None:
        # Vérifier que les colonnes sont identiques dans X_train et X_test
        if not all(X_train.columns == X_test.columns):
            raise ValueError("Les colonnes de X_train et X_test doivent être identiques.")

        # Calcul des importances pour l'ensemble de test
        importance_test = np.abs(shap_values_test).mean(0)

        # Création du DataFrame avec les deux ensembles
        importance_df = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance_Train': importance_train,
            'Importance_Test': importance_test
        })

        # Calcul de la différence
        importance_df['Difference'] = importance_df['Importance_Train'] - importance_df['Importance_Test']
        importance_df = importance_df.sort_values('Difference', key=abs, ascending=False)

        # Sélectionner les top_n features pour la visualisation de différence
        top_features = importance_df.head(top_n)

        plt.figure(figsize=(12, 8))
        sns.barplot(x='Difference', y='Feature', hue='Feature', data=top_features, palette='coolwarm', legend=False)
        plt.title(f"Top {top_n} Differences in Feature Importance (Train - Test)")
        plt.xlabel("Difference in Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'feature_importance_difference.png'))
        plt.close()

        print(f"Graphique de différence d'importance des features sauvegardé dans {save_dir}")
        print("\nTop differences in feature importance:")
        print(importance_df.head(10))
    else:
        # Si X_test est None, on crée un DataFrame uniquement avec les importances d'entraînement
        importance_df = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance_Train': importance_train
        })

        # Tri par importance décroissante
        importance_df = importance_df.sort_values('Importance_Train', ascending=False)

        # Sélectionner les top_n features pour la visualisation
        top_features = importance_df.head(top_n)

        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance_Train', y='Feature', hue='Feature', data=top_features, palette='viridis',
                    legend=False)
        plt.title(f"Top {top_n} Feature Importance (Training Set Only)")
        plt.xlabel("Feature Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'feature_importance_train_only.png'))
        plt.close()

        print(f"Graphique d'importance des features d'entraînement sauvegardé dans {save_dir}")
        print("\nTop feature importance (training set only):")
        print(importance_df.head(10))

    return importance_df


def compare_shap_distributions(shap_values_train=None, shap_explanation_train=None, shap_values_test=None,
                               shap_explanation_test=None,
                               X_train=None, y_train_label=None, X_test=None, y_test_label=None,
                               top_n=10,
                               save_dir='./shap_dependencies_results/'):
    """
    Compare les distributions des valeurs SHAP entre l'ensemble d'entraînement et de test.

    Parameters:
    -----------
    shap_values_train : numpy.ndarray
        Valeurs SHAP pour l'ensemble d'entraînement
    shap_explanation_train : shap.Explanation
        Objet Explanation SHAP pour l'ensemble d'entraînement (pour les scatter plots)
    shap_values_test : numpy.ndarray
        Valeurs SHAP pour l'ensemble de test
    shap_explanation_test : shap.Explanation
        Objet Explanation SHAP pour l'ensemble de test (non utilisé actuellement)
    X_train : pandas.DataFrame
        DataFrame contenant les features d'entraînement
    y_train_label : numpy.ndarray
        Labels de l'ensemble d'entraînement
    X_test : pandas.DataFrame
        DataFrame contenant les features de test
    y_test_label : numpy.ndarray
        Labels de l'ensemble de test
    top_n : int, optional
        Nombre de top features à comparer (par défaut 10)
    save_dir : str, optional
        Répertoire où sauvegarder les graphiques (par défaut './shap_dependencies_results/')

    Returns:
    --------
    None
        Sauvegarde les graphiques dans le répertoire spécifié
    """
    # Créer le répertoire principal de sauvegarde s'il n'existe pas
    os.makedirs(save_dir, exist_ok=True)

    # Créer le sous-répertoire pour les analyses combinées
    combined_dir = os.path.join(save_dir, 'shap_combined')
    os.makedirs(combined_dir, exist_ok=True)

    # Vérifier que les colonnes sont identiques dans X_train et X_test
    if not all(X_train.columns == X_test.columns):
        raise ValueError("Les colonnes de X_train et X_test doivent être identiques.")

    # Calculer l'importance des features basée sur les valeurs SHAP absolues moyennes
    feature_importance = np.abs(shap_values_train).mean(0)
    top_features = X_train.columns[np.argsort(feature_importance)[-top_n:]]

    # Pour chaque feature importante
    for idx, feature in enumerate(reversed(top_features)):
        # Créer une figure avec 3 sous-graphiques côte à côte
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

        # 1. Distribution Train/Test
        sns.kdeplot(data=shap_values_train[:, X_train.columns.get_loc(feature)],
                    label='Train', fill=True, ax=ax1)
        sns.kdeplot(data=shap_values_test[:, X_test.columns.get_loc(feature)],
                    label='Test', fill=True, ax=ax1)
        ax1.set_title("Train/Test Distribution")
        ax1.set_xlabel("SHAP Value")
        ax1.set_ylabel("Density")
        ax1.legend()

        # 2. Distribution par classe sur train
        feature_idx = X_train.columns.get_loc(feature)

        # Calcul des statistiques
        mean_class_0 = shap_values_train[y_train_label == 0, feature_idx].mean()
        mean_class_1 = shap_values_train[y_train_label == 1, feature_idx].mean()
        count_class_0 = sum(y_train_label == 0)
        count_class_1 = sum(y_train_label == 1)

        # Histogramme à la place du KDE
        ax2.hist(shap_values_train[y_train_label == 0, feature_idx],
                 bins=50, alpha=0.5, color='blue', label='Classe 0', density=True)
        ax2.hist(shap_values_train[y_train_label == 1, feature_idx],
                 bins=50, alpha=0.5, color='red', label='Classe 1', density=True)

        # Ajout des statistiques sur le graphique
        stats_text = f"Classe 0 (n={count_class_0}):\nMoyenne SHAP = {mean_class_0:.6f}\n\n" \
                     f"Classe 1 (n={count_class_1}):\nMoyenne SHAP = {mean_class_1:.6f}"

        # Positionnement du texte dans le coin supérieur droit
        ax2.text(0.95, 0.95, stats_text,
                 transform=ax2.transAxes,
                 verticalalignment='top',
                 horizontalalignment='right',
                 bbox=dict(boxstyle='round,pad=0.5',
                           facecolor='white',
                           alpha=0.8,
                           edgecolor='gray'))

        ax2.set_title("Distribution par Classe suur x_train")
        ax2.set_xlabel("SHAP Value")
        ax2.set_ylabel("Density")
        ax2.legend()

        # 3. Scatter plot SHAP
        feature_idx = list(X_train.columns).index(feature)
        shap.plots.scatter(shap_explanation_train[:, feature_idx], ax=ax3, show=False)
        ax3.set_title("SHAP Scatter Plot sur x_train")

        # Ajout du titre global
        plt.suptitle(f"SHAP Analysis - {feature}", fontsize=16, y=1.05)
        plt.tight_layout()

        # Sauvegarde de la figure dans le sous-répertoire shap_combined
        save_path = os.path.join(combined_dir, f'{idx}_shap_combined_analysis_{feature}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

        print(f"Analyse combinée créée pour la feature: {feature} (rang {idx})")

    print(f"Les graphiques de distribution SHAP ont été sauvegardés dans {combined_dir}")

    # Afficher un récapitulatif des features analysées
    print("Features analysées : " + " | ".join(f"{i}. {feature}" for i, feature in enumerate(top_features, 1)))
    # Dans la fonction compare_shap_distributions, avant la boucle des features
    # Calculer la somme des SHAP values pour chaque exemple
    shap_sums_train = shap_values_train.sum(axis=1)  # somme sur toutes les features

    print("\nAnalyse des sommes SHAP:")
    print(f"Moyenne des sommes SHAP pour classe 0: {shap_sums_train[y_train_label == 0].mean():.6f}")
    print(f"Moyenne des sommes SHAP pour classe 1: {shap_sums_train[y_train_label == 1].mean():.6f}")

    # Créer un nouveau plot pour la distribution des sommes
    plt.figure(figsize=(10, 6))
    plt.hist(shap_sums_train[y_train_label == 0], bins=50, alpha=0.5, color='blue', label='Classe 0', density=True)
    plt.hist(shap_sums_train[y_train_label == 1], bins=50, alpha=0.5, color='red', label='Classe 1', density=True)
    plt.title("Distribution des sommes des valeurs SHAP par classe")
    plt.xlabel("Somme des valeurs SHAP")
    plt.ylabel("Densité")
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'shap_sums_distribution_perClass.png'))

    # Créer une figure pour les top 20 features
    plt.figure(figsize=(16, 14))
    shap.summary_plot(shap_values_train, X_train, show=False, max_display=20)
    plt.title("Top 20 Features", fontsize=14, pad=20)
    plt.savefig(os.path.join(save_dir, 'shap_summary_plot_split1_top20.png'),
                bbox_inches='tight',
                dpi=300,
                pad_inches=1)
    plt.close()

    # Créer une figure séparée pour les features 21-40
    plt.figure(figsize=(16, 14))
    feature_importance = np.abs(shap_values_train).mean(0)
    sorted_idx = np.argsort(feature_importance)
    X_train_subset = X_train.iloc[:, sorted_idx[-40:-20]]
    shap_values_subset = shap_values_train[:, sorted_idx[-40:-20]]
    shap.summary_plot(shap_values_subset, X_train_subset, show=False, max_display=20)
    plt.title("Features 21-40", fontsize=14, pad=20)
    plt.savefig(os.path.join(save_dir, 'shap_summary_plot_split2_21_40.png'),
                bbox_inches='tight',
                dpi=300,
                pad_inches=1)
    plt.close()

    from PIL import Image

    def merge_images_horizontal(image1_path, image2_path, output_path, spacing=20):
        """
        Fusionner deux images horizontalement (côte à côte)

        Args:
            image1_path (str): Chemin vers la première image
            image2_path (str): Chemin vers la deuxième image
            output_path (str): Chemin pour sauvegarder l'image fusionnée
            spacing (int): Espace en pixels entre les images
        """
        # Ouvrir les images
        img1 = Image.open(image1_path)
        img2 = Image.open(image2_path)

        # Obtenir les dimensions
        width1, height1 = img1.size
        width2, height2 = img2.size

        # Utiliser la hauteur maximale
        max_height = max(height1, height2)

        # Redimensionner les images si nécessaire pour avoir la même hauteur
        if height1 != max_height:
            new_width1 = int(width1 * (max_height / height1))
            img1 = img1.resize((new_width1, max_height), Image.Resampling.LANCZOS)
            width1 = new_width1

        if height2 != max_height:
            new_width2 = int(width2 * (max_height / height2))
            img2 = img2.resize((new_width2, max_height), Image.Resampling.LANCZOS)
            width2 = new_width2

        # Créer une nouvelle image avec la largeur combinée (+ l'espacement)
        merged_image = Image.new('RGB', (width1 + spacing + width2, max_height), 'white')

        # Coller les images
        merged_image.paste(img1, (0, 0))
        merged_image.paste(img2, (width1 + spacing, 0))

        # Sauvegarder l'image fusionnée
        merged_image.save(output_path, 'PNG', dpi=(300, 300))

    # Utilisation
    image1_path = os.path.join(save_dir, 'shap_summary_plot_split1_top20.png')
    image2_path = os.path.join(save_dir, 'shap_summary_plot_split2_21_40.png')
    output_path = os.path.join(save_dir, 'shap_summary_plot_combined.png')

    merge_images_horizontal(image1_path, image2_path, output_path)


def compare_mean_shap_values(shap_values_train, shap_values_test, X_train, save_dir='./shap_dependencies_results/'):
    """
    Compare les valeurs SHAP moyennes entre l'ensemble d'entraînement et de test.

    Parameters:
    -----------
    shap_values_train : numpy.ndarray
        Valeurs SHAP pour l'ensemble d'entraînement
    shap_values_test : numpy.ndarray
        Valeurs SHAP pour l'ensemble de test
    X_train : pandas.DataFrame
        DataFrame contenant les features d'entraînement
    save_dir : str, optional
        Répertoire où sauvegarder le graphique (par défaut './shap_dependencies_results/')

    Returns:
    --------
    pandas.DataFrame
        DataFrame contenant la comparaison des valeurs SHAP moyennes
    """
    # Créer le répertoire de sauvegarde s'il n'existe pas
    os.makedirs(save_dir, exist_ok=True)

    mean_shap_train = shap_values_train.mean(axis=0)
    mean_shap_test = shap_values_test.mean(axis=0)

    shap_comparison = pd.DataFrame({
        'Feature': X_train.columns,
        'Mean SHAP Train': mean_shap_train,
        'Mean SHAP Test': mean_shap_test,
        'Difference': mean_shap_train - mean_shap_test
    })

    shap_comparison = shap_comparison.sort_values('Difference', key=abs, ascending=False)

    plt.figure(figsize=(12, 8))
    plt.scatter(shap_comparison['Mean SHAP Train'], shap_comparison['Mean SHAP Test'], alpha=0.5)
    plt.plot([-1, 1], [-1, 1], 'r--')  # ligne de référence y=x
    for i, txt in enumerate(shap_comparison['Feature']):
        plt.annotate(txt, (shap_comparison['Mean SHAP Train'].iloc[i], shap_comparison['Mean SHAP Test'].iloc[i]))
    plt.xlabel('Mean SHAP Value (Train)')
    plt.ylabel('Mean SHAP Value (Test)')
    plt.title('Comparison of Mean SHAP Values: Train vs Test')
    plt.tight_layout()

    # Sauvegarder le graphique dans le répertoire spécifié
    plt.savefig(os.path.join(save_dir, 'mean_shap_comparison.png'))
    plt.close()

    print("Top differences in mean SHAP values:")
    print(shap_comparison.head(10))

    return shap_comparison


def main_shap_analysis(final_model, X_train, y_train_label, X_test, y_test_label,
                       save_dir='./shap_dependencies_results/'):
    """Fonction principale pour l'analyse SHAP."""
    shap_comparison=None
    # Analyse SHAP sur l'ensemble d'entraînement et de test
    shap_values_train, shap_explanation_train, resulat_train_shap_feature_importance = analyze_shap_values(final_model,
                                                                                                           X_train,
                                                                                                           y_train_label,
                                                                                                           "Training_Set",
                                                                                                           create_dependence_plots=True,
                                                                                                           max_dependence_plots=40,
                                                                                                           save_dir=save_dir)

    if X_test is not None:
        shap_values_test, shap_explanation_test, resulat_test_shap_feature_importance = analyze_shap_values(final_model,
                                                                                                            X_test,
                                                                                                            y_test_label,
                                                                                                            "Test_Set",
                                                                                                            create_dependence_plots=True,
                                                                                                            max_dependence_plots=40,
                                                                                                            save_dir=save_dir)
    else:
        shap_values_test, shap_explanation_test, resulat_test_shap_feature_importance = None, None, None

    # Comparaison des importances de features et des distributions SHAP
    importance_df = compare_feature_importance(shap_values_train, shap_values_test, X_train, X_test, save_dir=save_dir)
    if X_test is not None:
        compare_shap_distributions(
            shap_values_train=shap_values_train, shap_explanation_train=shap_explanation_train,
            shap_values_test=shap_values_test, shap_explanation_test=shap_explanation_test,
            X_train=X_train, y_train_label=y_train_label, X_test=X_test, y_test_label=y_test_label,
            top_n=40, save_dir=save_dir)

        # Comparaison des valeurs SHAP moyennes
        shap_comparison = compare_mean_shap_values(
            shap_values_train=shap_values_train, shap_values_test=shap_values_test, X_train=X_train, save_dir=save_dir)

    return importance_df, shap_comparison, shap_values_train, shap_values_test, resulat_test_shap_feature_importance


def analyze_xgboost_trees(model, feature_names, nan_value, max_trees=None):
    """
    Analyse les arbres XGBoost pour identifier tous les splits et ceux impliquant des NaN.

    :param model: Modèle XGBoost entraîné (Booster ou XGBClassifier/XGBRegressor)
    :param feature_names: Liste des noms des features
    :param nan_value: Valeur utilisée pour remplacer les NaN (peut être np.nan)
    :param max_trees: Nombre maximum d'arbres à analyser (None pour tous les arbres)
    :return: Tuple (DataFrame de tous les splits, DataFrame des splits NaN)
    """
    # Si le modèle est un Booster, l'utiliser directement ; sinon, obtenir le Booster depuis le modèle
    if isinstance(model, xgb.Booster):
        booster = model
    else:
        booster = model.get_booster()

    # Obtenir les arbres sous forme de DataFrame
    trees_df = booster.trees_to_dataframe()

    if max_trees is not None:
        trees_df = trees_df[trees_df['Tree'] < max_trees].copy()

    # Filtrer pour ne garder que les splits (nœuds non-feuilles)
    all_splits = trees_df[trees_df['Feature'] != 'Leaf'].copy()

    # Check if 'Depth' is already present; if not, calculate it
    if pd.isna(nan_value):
        # Lorsque nan_value est np.nan, identifier les splits impliquant des valeurs manquantes
        nan_splits = all_splits[all_splits['Missing'] != all_splits['Yes']].copy()
        print(f"Utilisation de la condition pour np.nan. Nombre de splits NaN trouvés : {len(nan_splits)}")
    else:
        # Lorsque nan_value est une valeur spécifique, identifier les splits sur cette valeur
        all_splits.loc[:, 'Split'] = all_splits['Split'].astype(float)
        nan_splits = all_splits[np.isclose(all_splits['Split'], nan_value, atol=1e-8)].copy()
        print(
            f"Utilisation de la condition pour nan_value={nan_value}. Nombre de splits NaN trouvés : {len(nan_splits)}")
        if len(nan_splits) > 0:
            print("Exemples de valeurs de split considérées comme NaN:")
            print(nan_splits['Split'].head())

    return all_splits, nan_splits


def extract_decision_rules(model, nan_value, importance_threshold=0.01):
    """
    Extrait les règles de décision importantes impliquant la valeur de remplacement des NaN ou les valeurs NaN.
    :param model: Modèle XGBoost entraîné (Booster ou XGBClassifier/XGBRegressor)
    :param nan_value: Valeur utilisée pour remplacer les NaN (peut être np.nan)
    :param importance_threshold: Seuil de gain pour inclure une règle
    :return: Liste des règles de décision importantes
    """
    if isinstance(model, xgb.Booster):
        booster = model
    else:
        booster = model.get_booster()

    trees_df = booster.trees_to_dataframe()

    # Ajouter la colonne Depth en comptant le nombre de tirets dans 'ID'
    if 'Depth' not in trees_df.columns:
        trees_df['Depth'] = trees_df['ID'].apply(lambda x: x.count('-'))

    # Vérifiez que les colonnes attendues sont présentes dans chaque ligne avant de traiter
    expected_columns = ['Tree', 'Depth', 'Feature', 'Gain', 'Split', 'Missing', 'Yes']

    # Filtrer les lignes avec des NaN si c'est nécessaire
    trees_df = trees_df.dropna(subset=expected_columns, how='any')

    if pd.isna(nan_value):
        important_rules = trees_df[
            (trees_df['Missing'] != trees_df['Yes']) & (trees_df['Gain'] > importance_threshold)
            ]
    else:
        important_rules = trees_df[
            (trees_df['Split'] == nan_value) & (trees_df['Gain'] > importance_threshold)
            ]

    rules = []
    for _, row in important_rules.iterrows():
        try:
            # Check if necessary columns exist in the row before constructing the rule
            if all(col in row for col in expected_columns):
                rule = f"Arbre {row['Tree']}, Profondeur {row['Depth']}"
                feature = row['Feature']
                gain = row['Gain']

                if pd.isna(nan_value):
                    missing_direction = row['Missing']
                    rule += f": SI {feature} < {row['Split']} OU {feature} est NaN (valeurs manquantes vont vers le nœud {missing_direction}) ALORS ... (gain: {gain:.4f})"
                else:
                    rule += f": SI {feature} == {nan_value} ALORS ... (gain: {gain:.4f})"

                rules.append(rule)
            else:
                print(f"Colonnes manquantes dans cette ligne : {row.to_dict()}")
                continue

        except IndexError as e:
            # Gérer l'erreur d'index hors limites
            # print(f"Erreur lors de l'extraction des informations de la règle: {e}")
            continue
        except Exception as e:
            # Gérer toute autre erreur inattendue
            print(f"Erreur inattendue: {e}")
            continue

    return rules


def analyze_nan_impact(model, X_train, feature_names, nan_value, shap_values=None,
                       features_per_plot=35, verbose_nan_rule=False,
                       save_dir='./nan_analysis_results/'):
    """
    Analyse l'impact des valeurs NaN ou des valeurs de remplacement des NaN sur le modèle XGBoost.

    Parameters:
    -----------
    model : xgboost.Booster ou xgboost.XGBClassifier/XGBRegressor
        Modèle XGBoost entraîné
    X_train : pandas.DataFrame
        Données d'entrée
    feature_names : list
        Liste des noms des features
    nan_value : float ou np.nan
        Valeur utilisée pour remplacer les NaN
    shap_values : numpy.array, optional
        Valeurs SHAP pré-calculées
    features_per_plot : int, default 35
        Nombre de features à afficher par graphique
    verbose_nan_rule : bool, default False
        Si True, affiche les règles de décision détaillées
    save_dir : str, default './nan_analysis_results/'
        Répertoire où sauvegarder les résultats

    Returns:
    --------
    dict
        Dictionnaire contenant les résultats de l'analyse
    """
    os.makedirs(save_dir, exist_ok=True)

    results = {}

    # 1. Analyser les splits impliquant les valeurs NaN
    all_splits, nan_splits = analyze_xgboost_trees(model, feature_names, nan_value)
    results['total_splits'] = len(all_splits)
    results['nan_splits'] = len(nan_splits)
    results['nan_splits_percentage'] = (len(nan_splits) / len(all_splits)) * 100

    # Stocker les distributions pour une utilisation ultérieure
    all_splits_dist = all_splits['Feature'].value_counts()
    nan_splits_dist = nan_splits['Feature'].value_counts()

    # 2. Visualiser la distribution des splits NaN
    plt.figure(figsize=(12, 6))
    sns.countplot(x='Feature', data=nan_splits, order=nan_splits_dist.index)
    plt.title("Distribution des splits impliquant des valeurs NaN par feature")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'nan_splits_distribution.png'))
    plt.close()

    # 3. Analyser la profondeur des splits NaN
    if 'Depth' not in nan_splits.columns and 'ID' in nan_splits.columns:
        nan_splits['Depth'] = nan_splits['ID'].apply(lambda x: x.count('-'))

    if 'Depth' in nan_splits.columns:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Feature', y='Depth', data=nan_splits)
        plt.title("Profondeur des splits impliquant des valeurs NaN par feature")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'nan_splits_depth.png'))
        plt.close()

    # 4. Extraire les règles de décision importantes
    important_rules = extract_decision_rules(model, nan_value)
    results['important_rules'] = important_rules

    if verbose_nan_rule:
        print("\nRègles de décision importantes impliquant des valeurs NaN :")
        for rule in important_rules:
            print(rule)

    # 5. Analyser l'importance des features avec des valeurs NaN

    # Calculs pour l'analyse SHAP et NaN
    shap_mean = np.abs(shap_values).mean(axis=0)
    nan_counts = X_train.isna().sum() if pd.isna(nan_value) else (X_train == nan_value).sum()
    nan_percentages = (nan_counts / len(X_train)) * 100

    nan_fi_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': shap_mean,
        'Total_NaN': nan_counts,
        'Percentage_NaN': nan_percentages
    }).sort_values('Importance', ascending=False)

    # Générer les graphiques par lots
    num_features = len(nan_fi_df)
    for i in range((num_features + features_per_plot - 1) // features_per_plot):
        subset_df = nan_fi_df.iloc[i * features_per_plot: (i + 1) * features_per_plot]

        fig, ax1 = plt.subplots(figsize=(14, 8))
        sns.barplot(x='Feature', y='Importance', data=subset_df, ax=ax1, color='skyblue')
        ax1.set_xlabel('Feature', fontsize=12)
        ax1.set_ylabel('Importance (SHAP)', color='blue', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='blue')

        ax2 = ax1.twinx()
        sns.lineplot(x='Feature', y='Percentage_NaN', data=subset_df, ax=ax2, color='red', marker='o')
        ax2.set_ylabel('Pourcentage de NaN (%)', color='red', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.yaxis.set_major_formatter(ticker.PercentFormatter())

        plt.title(f"Importance des features (SHAP) et pourcentage de NaN (X_Train)\n"
                  f"(Features {i * features_per_plot + 1} à {min((i + 1) * features_per_plot, num_features)})",
                  fontsize=14)

        # Incliner les étiquettes de l'axe x à 45 degrés
        ax1.set_xticks(range(len(subset_df)))
        ax1.set_xticklabels(subset_df['Feature'], rotation=45, ha='right', va='top')

        # Ajuster l'espace pour améliorer la visibilité
        plt.tight_layout()
        fig.subplots_adjust(bottom=0.2)  # Ajustez cette valeur si nécessaire

        plt.savefig(os.path.join(save_dir, f'nan_features_shap_importance_percentage_{i + 1}.png'))
        plt.close()

    # Calcul et visualisation de la corrélation
    correlation = nan_fi_df['Importance'].corr(nan_fi_df['Percentage_NaN'])
    results['shap_nan_correlation'] = correlation

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='Percentage_NaN', y='Importance', data=nan_fi_df)
    plt.title('Relation entre le pourcentage de NaN et l\'importance des features (SHAP)')
    plt.xlabel('Pourcentage de NaN (%)')
    plt.ylabel('Importance (SHAP)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'shap_importance_vs_percentage_nan.png'))
    plt.close()

    return results


# 1. Fonction pour analyser les erreurs

def analyze_errors(X_test, y_test_label, y_pred_threshold, y_pred_proba, feature_names, save_dir=None,
                   top_features=None):
    """
    Analyse les erreurs de prédiction du modèle et génère des visualisations.

    Parameters:
    -----------
    X_test : pd.DataFrame
        Les features de l'ensemble de test.
    y_test_label : array-like
        Les vraies étiquettes de l'ensemble de test.
    y_pred_threshold : array-like
        Les prédictions du modèle après application du seuil.
    y_pred_proba : array-like
        Les probabilités prédites par le modèle.
    feature_names : list
        Liste des noms des features.
    save_dir : str, optional
        Le répertoire où sauvegarder les résultats de l'analyse (par défaut './analyse_error/').
    top_features : list, optional
        Liste des 10 features les plus importantes (si disponible).

    Returns:
    --------
    tuple
        (results_df, error_df) : DataFrames contenant respectivement tous les résultats et les cas d'erreur.
    """
    # Créer le répertoire de sauvegarde s'il n'existe pas
    os.makedirs(save_dir, exist_ok=True)

    # Vérifier et convertir les arrays CuPy en NumPy
    def convert_to_numpy(arr):
        return arr.get() if isinstance(arr, cp.ndarray) else arr

    # Convertir y_test_label, y_pred_threshold, y_pred_proba en NumPy si nécessaire
    y_test_label = convert_to_numpy(y_test_label)
    y_pred_threshold = convert_to_numpy(y_pred_threshold)
    y_pred_proba = convert_to_numpy(y_pred_proba)

    # Créer un dictionnaire pour stocker toutes les données
    data = {
        'true_label': y_test_label,
        'predicted_label': y_pred_threshold,
        'prediction_probability': y_pred_proba
    }

    # Ajouter les features au dictionnaire, en les convertissant en NumPy si nécessaire
    for feature in feature_names:
        data[feature] = convert_to_numpy(X_test[feature])

    # Créer le DataFrame en une seule fois
    results_df = pd.DataFrame(data)

    # Ajouter les colonnes d'erreur
    results_df['is_error'] = results_df['true_label'] != results_df['predicted_label']
    results_df['error_type'] = np.where(results_df['is_error'],
                                        np.where(results_df['true_label'] == 1, 'False Negative', 'False Positive'),
                                        'Correct')

    # Analyse des erreurs
    error_distribution = results_df['error_type'].value_counts(normalize=True)
    print("Distribution des erreurs:")
    print(error_distribution)

    # Sauvegarder la distribution des erreurs
    error_distribution.to_csv(os.path.join(save_dir, 'error_distribution.csv'), index=False, sep=';')

    # Analyser les features pour les cas d'erreur
    error_df = results_df[results_df['is_error']]

    print("\nMoyenne des features pour les erreurs vs. prédictions correctes:")
    feature_means = results_df.groupby('error_type')[feature_names].mean()
    with pd.option_context('display.max_columns', None, 'display.width', None):
        print(feature_means)

    # Sauvegarder les moyennes des features
    feature_means.to_csv(os.path.join(save_dir, 'feature_means_by_error_type.csv'), index=False, sep=';')

    # Visualiser la distribution des probabilités de prédiction pour les erreurs
    plt.figure(figsize=(10, 6))
    sns.histplot(data=error_df, x='prediction_probability', hue='true_label', bins=20)
    plt.title('Distribution des probabilités de prédiction pour les erreurs')
    plt.savefig(os.path.join(save_dir, 'error_probability_distribution.png'))
    plt.close()

    # Identifier les cas les plus confiants mais erronés
    most_confident_errors = error_df.sort_values('prediction_probability', ascending=False).head(5)
    print("\nLes 5 erreurs les plus confiantes:")
    print(most_confident_errors[['true_label', 'predicted_label', 'prediction_probability']])

    # Sauvegarder les erreurs les plus confiantes
    most_confident_errors.to_csv(os.path.join(save_dir, 'most_confident_errors.csv'), index=False, sep=';')

    # Visualisations supplémentaires
    plt.figure(figsize=(12, 10))
    sns.heatmap(error_df[feature_names].corr(), annot=False, cmap='coolwarm')
    plt.title('Corrélations des features pour les erreurs')
    plt.savefig(os.path.join(save_dir, 'error_features_correlation.png'))
    plt.close()

    # Identification des erreurs
    errors = X_test[y_test_label != y_pred_threshold]
    print("Nombre d'erreurs:", len(errors))

    # Créer un subplot pour chaque feature (si top_features est fourni)
    if top_features and len(top_features) >= 40:
        fig, axes = plt.subplots(5, 8, figsize=(60, 30))  # Grille 5x8
        fig.suptitle('Distribution des 40 features les plus importantes par type d\'erreur', fontsize=16)

        for i, feature in enumerate(top_features[:40]):
            row = i // 8  # Division entière par 8 (nombre de colonnes)
            col = i % 8  # Modulo 8 pour obtenir la colonne
            sns.boxplot(x='error_type', y=feature, data=results_df, ax=axes[row, col])
            axes[row, col].set_title(f'{i + 1}. {feature}')
            axes[row, col].set_xlabel('')
            if col == 0:
                axes[row, col].set_ylabel('Valeur')
            else:
                axes[row, col].set_ylabel('')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'top_40_features_distribution_by_error.png'),
                    dpi=300, bbox_inches='tight')
        print("Graphique combiné sauvegardé sous 'top_40_features_distribution_by_error.png'")
        plt.close()

    # Sauvegarder les DataFrames spécifiques demandés
    results_df.to_csv(os.path.join(save_dir, 'model_results_analysis.csv'), index=False, sep=';')
    error_df.to_csv(os.path.join(save_dir, 'model_errors_analysis.csv'), index=False, sep=';')
    print(f"\nLes résultats de l'analyse ont été sauvegardés dans le répertoire : {save_dir}")

    return results_df, error_df


# 2. Fonction pour analyser les erreurs confiantes
def analyze_confident_errors(shap_values, confident_errors, X_test, feature_names, important_features, n=5):
    for idx in confident_errors.index[:n]:
        print(f"-----------------> Analyse de l'erreur à l'index {idx}:")
        print(f"Vrai label: {confident_errors.loc[idx, 'true_label']}")
        print(f"Label prédit: {confident_errors.loc[idx, 'predicted_label']}")
        print(f"Probabilité de prédiction: {confident_errors.loc[idx, 'prediction_probability']:.4f}")

        print("\nValeurs des features importantes:")
        for feature in important_features:
            value = X_test.loc[idx, feature]
            print(f"{feature}: {value:.4f}")

        print("\nTop 5 features influentes (SHAP) pour ce cas:")
        case_feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': np.abs(shap_values[0])
        }).sort_values('importance', ascending=False)

        print(case_feature_importance.head())
        print(f"<----------------- Fin Analyse de l'erreur à l'index {idx}:")


# 3. Fonction pour visualiser les erreurs confiantes
def plot_confident_errors(shap_values, confident_errors, X_test, feature_names, n=5, results_directory=None):
    # Vérifier le nombre d'erreurs confiantes disponibles
    num_errors = len(confident_errors)
    if num_errors == 0:
        print("Aucune erreur confiante trouvée.")
        return

    # Ajuster n si nécessaire
    n = min(n, num_errors)

    for i, idx in enumerate(confident_errors.index[:n]):
        plt.figure(figsize=(10, 6))

        # Vérifier si shap_values est une liste (pour les modèles à plusieurs classes)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Prendre les valeurs SHAP pour la classe positive

        shap.summary_plot(shap_values, X_test.loc[idx:idx], plot_type="bar", feature_names=feature_names, show=False)
        plt.title(f"Erreur {i + 1}: Vrai {confident_errors.loc[idx, 'true_label']}, "
                  f"Prédit {confident_errors.loc[idx, 'predicted_label']} "
                  f"(Prob: {confident_errors.loc[idx, 'prediction_probability']:.4f})")
        plt.tight_layout()
        plt.savefig(f'confident_error_shap_{i + 1}.png')

        plt.close()

    if n > 0:
        # Create a summary image combining all individual plots
        images = [Image.open(f'confident_error_shap_{i + 1}.png') for i in range(n)]
        widths, heights = zip(*(i.size for i in images))

        max_width = max(widths)
        total_height = sum(heights)

        new_im = Image.new('RGB', (max_width, total_height))

        y_offset = 0
        for im in images:
            new_im.paste(im, (0, y_offset))
            y_offset += im.size[1]

        new_im.save(os.path.join(results_directory, 'confident_errors_shap_combined.png'), dpi=(300, 300),
                    bbox_inches='tight')
        # Clean up individual images
        for i in range(n):
            os.remove(f'confident_error_shap_{i + 1}.png')

        print(f"Image combinée des {n} erreurs confiantes sauvegardée sous 'confident_errors_shap_combined.png'")
    else:
        print("Pas assez d'erreurs confiantes pour créer une image combinée.")


# 4. Fonction pour comparer les erreurs vs les prédictions correctes
def compare_errors_vs_correct(confident_errors, correct_predictions, X_test, important_features, results_directory):
    error_data = X_test.loc[confident_errors.index]
    correct_data = X_test.loc[correct_predictions.index]

    comparison_data = []
    for feature in important_features:
        error_mean = error_data[feature].mean()
        correct_mean = correct_data[feature].mean()
        difference = error_mean - correct_mean

        comparison_data.append({
            'Feature': feature,
            'Erreurs Confiantes (moyenne)': error_mean,
            'Prédictions Correctes (moyenne)': correct_mean,
            'Différence': difference
        })

    comparison_df = pd.DataFrame(comparison_data)
    print("\nComparaison des features importantes:")
    with pd.option_context('display.max_columns', None, 'display.width', None):
        print(comparison_df)

    # Visualisation
    plt.figure(figsize=(12, 6))
    bar_width = 0.35
    index = np.arange(len(important_features))

    plt.bar(index, comparison_df['Erreurs Confiantes (moyenne)'], bar_width, label='Erreurs Confiantes')
    plt.bar(index + bar_width, comparison_df['Prédictions Correctes (moyenne)'], bar_width,
            label='Prédictions Correctes')

    plt.xlabel('Features')
    plt.ylabel('Valeur Moyenne')
    plt.title('Comparaison des features importantes: Erreurs vs Prédictions Correctes')
    plt.xticks(index + bar_width / 2, important_features, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_directory, 'compare_errors_vs_correct.png'), dpi=300, bbox_inches='tight')
    plt.close()


# You already have CuPy imported as cp, so we will use it.
def analyze_predictions_by_range(X_test, y_pred_proba, shap_values_all, prob_min=0.90, prob_max=1.00,
                                 top_n_features=None,
                                 output_dir=None):
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)

    # Helper function to convert CuPy arrays to NumPy
    def convert_to_numpy(arr):
        # Check if cp is available and arr is a cp array
        return arr.get() if isinstance(arr, cp.ndarray) else arr

    # Convert y_pred_proba and shap_values_all to NumPy if necessary
    y_pred_proba = convert_to_numpy(y_pred_proba)
    shap_values_all = convert_to_numpy(shap_values_all)

    # 1. Identifier les échantillons dans la plage de probabilités spécifiée
    prob_mask = (y_pred_proba >= prob_min) & (y_pred_proba <= prob_max)
    selected_samples = X_test[prob_mask]
    selected_proba = y_pred_proba[prob_mask]

    # Vérifier s'il y a des échantillons dans la plage spécifiée
    if len(selected_samples) == 0:
        print(f"Aucun échantillon trouvé dans la plage de probabilités {prob_min:.2f} - {prob_max:.2f}")
        return

    # print(f"Nombre d'échantillons dans la plage {prob_min:.2f} - {prob_max:.2f}: {len(selected_samples)}")

    # Calculer l'importance des features basée sur les valeurs SHAP
    feature_importance = np.abs(shap_values_all).mean(0)

    # Sélectionner les top features si spécifié
    if top_n_features is not None and top_n_features < len(X_test.columns):
        top_features_indices = np.argsort(feature_importance)[-top_n_features:]
        top_features = X_test.columns[top_features_indices]
        selected_samples = selected_samples[top_features]
        X_test_top = X_test[top_features]
        shap_values_selected = shap_values_all[prob_mask][:, top_features_indices]
    else:
        top_features = X_test.columns
        X_test_top = X_test
        shap_values_selected = shap_values_all[prob_mask]

    # 2. Examiner ces échantillons
    with open(os.path.join(output_dir, 'selected_samples_details.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Échantillons avec probabilités entre {prob_min:.2f} et {prob_max:.2f}:\n")
        for i, (idx, row) in enumerate(selected_samples.iterrows()):
            f.write(f"\nÉchantillon {i + 1} (Probabilité: {selected_proba[i]:.4f}):\n")
            for feature, value in row.items():
                f.write(f"  {feature}: {value}\n")

    # 3. Analyser les statistiques de ces échantillons
    stats = selected_samples.describe()
    stats.to_csv(os.path.join(output_dir, 'selected_samples_statistics.csv'), index=False, sep=';')

    # 4. Comparer avec les statistiques globales
    comparison_list = []
    for feature in top_features:
        global_mean = X_test_top[feature].mean()
        selected_mean = selected_samples[feature].mean()
        diff = selected_mean - global_mean
        comparison_list.append({
            'Feature': feature,
            'Global_Mean': global_mean,
            'Selected_Mean': selected_mean,
            'Difference': diff
        })
    comparison = pd.DataFrame(comparison_list)
    comparison.to_csv(os.path.join(output_dir, 'global_vs_selected_comparison.csv'), index=False, sep=';')

    # 5. Visualiser les distributions
    # Créer le sous-dossier 'distribution_features' dans output_dir
    distribution_dir = os.path.join(output_dir, 'distribution_features')
    os.makedirs(distribution_dir, exist_ok=True)

    # Générer et sauvegarder les graphiques dans le sous-dossier
    for feature in top_features:
        plt.figure(figsize=(10, 6))
        sns.histplot(X_test_top[feature], kde=True, label='Global')
        sns.histplot(selected_samples[feature], kde=True, label=f'Selected ({prob_min:.2f} - {prob_max:.2f})')
        plt.title(f"Distribution de {feature}")
        plt.legend()

        # Enregistrer dans le sous-dossier
        plt.savefig(os.path.join(distribution_dir, f'distribution_{feature}.png'))
        plt.close()

    # 6. Analyse des corrélations entre les features pour ces échantillons
    correlation_matrix = selected_samples.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title(f"Matrice de corrélation pour les échantillons avec probabilités entre {prob_min:.2f} et {prob_max:.2f}")
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
    plt.close()

    # 7. Comparer les valeurs SHAP pour ces échantillons
    shap.summary_plot(shap_values_selected, selected_samples, plot_type="bar", show=False)
    plt.title(
        f"Importance des features SHAP pour les échantillons avec probabilités entre {prob_min:.2f} et {prob_max:.2f}")
    plt.savefig(os.path.join(output_dir, 'shap_importance.png'))
    plt.close()

    print(f"Analyse terminée. Les résultats ont été sauvegardés dans le dossier '{output_dir}'.")


def check_original_alignment(y_train_label, df_pnl_data_train):
    """
    Vérifie si les données d'origine y_train_label et df_pnl_data_train sont alignées.

    1. Vérifie que les longueurs sont identiques.
    2. Si ce sont des pandas Series, vérifie que leurs indices sont identiques.
    3. Vérifie la cohérence métier :
       - Pour chaque échantillon où y_train_label == 1, la valeur correspondante dans df_pnl_data_train doit être > 0.
       - Pour chaque échantillon où y_train_label == 0, la valeur correspondante dans df_pnl_data_train doit être < 0.

    Parameters
    ----------
    y_train_label : np.ndarray ou pd.Series
        Tableau ou Series des labels d'entraînement.
    df_pnl_data_train : np.ndarray ou pd.Series
        Tableau ou Series des valeurs pnl correspondantes.

    Raises
    ------
    ValueError : Si une des vérifications échoue.
    """
    # Vérifier les longueurs
    if len(y_train_label) != len(df_pnl_data_train):
        raise ValueError("Les longueurs de y_train_label et df_pnl_data_train sont différentes.")

    # Si ce sont des Series, vérifier que leurs index sont identiques
    if isinstance(y_train_label, pd.Series) and isinstance(df_pnl_data_train, pd.Series):
        if not y_train_label.index.equals(df_pnl_data_train.index):
            raise ValueError("Les indices des Series ne sont pas identiques.")

    # Convertir en arrays pour faciliter la vérification métier
    y_array = y_train_label if isinstance(y_train_label, np.ndarray) else y_train_label.to_numpy()
    pnl_array = df_pnl_data_train if isinstance(df_pnl_data_train, np.ndarray) else df_pnl_data_train.to_numpy()

    # Vérifier la cohérence métier
    mask_positive = (y_array == 1)
    mask_zero = (y_array == 0)

    if not np.all(pnl_array[mask_positive] > 0):
        bad_indices = np.where(pnl_array[mask_positive] <= 0)[0]
        raise ValueError(f"Pour certains indices où y_train_label==1, le pnl n'est pas > 0: indices {bad_indices}")

    if not np.all(pnl_array[mask_zero] < 0):
        bad_indices = np.where(pnl_array[mask_zero] >= 0)[0]
        raise ValueError(f"Pour certains indices où y_train_label==0, le pnl n'est pas < 0: indices {bad_indices}")

    print("Les données d'origine sont alignées et cohérentes.")


def filter_features(X_train, y_train_label, X_test=None, y_test_label=None,
                    df_pnl_data_train=None, df_pnl_data_test=None,
                    selected_columns_manual=None, config=None,
                    results_directory=None, X_train_full=None):
    """
    Filtre les features en utilisant diverses méthodes (VIF, corrélation, information mutuelle)
    et nettoie les dataframes des valeurs NaN et Inf.

    Parameters:
    -----------
    X_train : pandas.DataFrame
        DataFrame contenant les features d'entraînement
    y_train_label : pandas.Series
        Variable cible pour l'entraînement
    X_test : pandas.DataFrame, optional
        DataFrame contenant les features de test
    y_test_label : pandas.Series, optional
        Variable cible pour le test
    df_pnl_data_train : pandas.DataFrame, optional
        Données PnL pour l'entraînement
    df_pnl_data_test : pandas.DataFrame, optional
        Données PnL pour le test
    selected_columns_manual : list, optional
        Liste des colonnes déjà sélectionnées manuellement
    config : dict, optional
        Configuration des paramètres d'analyse et filtrage
    results_directory : str, optional
        Répertoire où stocker les résultats
    X_train_full : pandas.DataFrame, optional
        DataFrame complet des features d'entraînement avant sélection manuelle

    Returns:
    --------
    dict
        Dictionnaire contenant:
        - 'selected_columns': liste des colonnes sélectionnées après filtrage
        - 'X_train': X_train nettoyé des valeurs NaN et Inf
        - 'y_train_label': y_train_label nettoyé des valeurs NaN et Inf
        - 'X_test': X_test nettoyé des valeurs NaN et Inf (si fourni)
        - 'y_test_label': y_test_label nettoyé des valeurs NaN et Inf (si fourni)
        - 'df_pnl_data_train': df_pnl_data_train nettoyé des valeurs NaN et Inf (si fourni)
        - 'df_pnl_data_test': df_pnl_data_test nettoyé des valeurs NaN et Inf (si fourni)
        - 'scaler': l'objet scaler utilisé (si scaling appliqué)
    """
    import pandas as pd
    import numpy as np

    # Valeurs par défaut
    if config is None:
        config = {}
    if selected_columns_manual is None:
        selected_columns_manual = list(X_train.columns)
    result = {}
    # # Résultat à retourner
    # result = {
    #     'X_train': X_train,
    #     'y_train_label': y_train_label
    # }
    #
    # # Ajouter les éléments optionnels s'ils sont fournis
    # if X_test is not None:
    #     result['X_test'] = X_test
    # if y_test_label is not None:
    #     result['y_test_label'] = y_test_label
    # if df_pnl_data_train is not None:
    #     result['df_pnl_data_train'] = df_pnl_data_train
    # if df_pnl_data_test is not None:
    #     result['df_pnl_data_test'] = df_pnl_data_test

    # Affichage des informations sur les features après exclusion manuelle
    if X_train_full is not None:
        print(f"\nFeatures X_train_full après exclusion manuelle des features (short + 99)(a verivier AL)):")
        compute_display_statistic(X=X_train_full, name="X_train_full",
                                  config=config, compute_feature_stat=False)

    print(f"Features X_train après exclusion manuelle des features (sur trades short après exclusion de 99):")
    compute_display_statistic(X=X_train, name="X_train",
                              config=config, compute_feature_stat=False)

    # Affichage des valeurs NaN dans les ensembles de données
    print(f"\nValeurs NaN : X_train={X_train.isna().sum().sum()}, y_train_label={y_train_label.isna().sum()}", end="")
    if X_test is not None and y_test_label is not None:
        print(f", X_test={X_test.isna().sum().sum()}, y_test_label={y_test_label.isna().sum()}\n")
    else:
        print("\n")

    print(f"Dimensions de X_train: {X_train.shape} (lignes, colonnes)", end="")
    if X_test is not None:
        print(f" | Dimensions de X_test: {X_test.shape} (lignes, colonnes)")
    else:
        print("")

    print(f"Nb de features après exlusion manuelle: {len(selected_columns_manual)}\n")

    # Variables pour stocker les masques après nettoyage
    mask_train_noInfNan = None
    mask_test_noInfNan = None

    # Nettoyage des valeurs NaN et Inf si demandé dans la configuration
    if config.get("remove_inf_nan_afterFeaturesSelections", False):
        if df_pnl_data_train is not None:
            X_train_noInfNan, y_train_label_noInfNan, df_pnl_data_train_noInfNan, mask_train_noInfNan = remove_nan_inf(
                X=X_train, y=y_train_label, df_pnl_data=df_pnl_data_train, dataset_name="train")
            # result['X_train_noInfNan'] = X_train_noInfNan
            # result['y_train_label_noInfNan'] = y_train_label_noInfNan
            # result['df_pnl_data_train'] = df_pnl_data_train
        else:  # df_pnl_data_train is None:
            X_train_noInfNan, y_train_label_noInfNan, _, mask_train_noInfNan = remove_nan_inf(
                X=X_train, y=y_train_label, df_pnl_data=None, dataset_name="train")
            # result['X_train_noInfNan'] = X_train_noInfNan
            # result['y_train_label_noInfNan'] = y_train_label_noInfNan
            print("df_pnl_data_train is None")
            exit(56)

        if X_test is not None and y_test_label is not None:
            if df_pnl_data_test is not None:
                X_test_noInfNan, y_test_label_noInfNan, df_pnl_data_test_noInfNan, mask_test_noInfNan = remove_nan_inf(
                    X=X_test, y=y_test_label, df_pnl_data=df_pnl_data_test, dataset_name="test")
                # result['X_test_noInfNan'] = X_test_noInfNan
                # result['y_test_label_noInfNan'] = y_test_label_noInfNan
                # result['df_pnl_data_test_noInfNan'] = df_pnl_data_test_noInfNan
            else:
                X_test_noInfNan, y_test_label_noInfNan, _, mask_test_noInfNan = remove_nan_inf(
                    X=X_test, y=y_test_label, df_pnl_data=None, dataset_name="test")
                # result['X_test_noInfNan'] = X_test_noInfNan
                # result['y_test_label_noInfNan'] = y_test_label_noInfNan
                # print("df_pnl_data_test is None")
                exit(56)

    result['mask_test_noInfNan'] = mask_test_noInfNan
    result['mask_train_noInfNan'] = mask_train_noInfNan
    # Application éventuelle du scaling (normalisation/standardisation) des features
    chosen_scaler = config.get('scaler_choice', scalerChoice.SCALER_ROBUST)

    if chosen_scaler != scalerChoice.SCALER_DISABLE:
        # print(f"\n-- Scaler {chosen_scaler} actif ---\n")
        # Appliquer le scaling
        if X_test_noInfNan is not None and y_test_label is not None:
            X_train_scaled, X_test_scaled, y_train_label_scaled, y_test_label_scaled, scaler = apply_data_feature_scaling(
                X_train_noInfNan, X_test_noInfNan, y_train_label_noInfNan, y_test_label_noInfNan,
                mask_train_noInfNan, mask_test_noInfNan,
                chosen_scaler=chosen_scaler,
                results_directory=results_directory,
                config=config
            )
            # On conserve les données originales pour le retour mais on utilise les données scalées pour le filtrage
            X_train_for_filtering = X_train_scaled
            y_train_label_for_filtering = y_train_label_scaled
        else:
            # Cas où X_test ou y_test_label n'est pas fourni
            X_train_scaled, _, y_train_label_scaled, _, scaler = apply_data_feature_scaling(
                X_train, X_train.copy(), y_train_label, y_train_label.copy(),
                mask_train_noInfNan, mask_train_noInfNan,
                chosen_scaler=chosen_scaler,
                results_directory=results_directory,
                config=config
            )
            # On conserve les données originales pour le retour mais on utilise les données scalées pour le filtrage
            X_train_for_filtering = X_train_scaled
            y_train_label_for_filtering = y_train_label_scaled

        # result['scaler'] = scaler
    else:
        print("\n-- Pas de scaler actif ---\n")
        # Si pas de scaling, on utilise les données originales pour le filtrage
        X_train_for_filtering = X_train
        y_train_label_for_filtering = y_train_label

    # Filtrage automatique des features basé sur VIF, corrélation et information mutuelle (si activé)
    auto_filtering_mode = config.get('auto_filtering_mode', None)
    is_compute_vif = config.get('compute_vif', True)  # Par défaut True pour garder le comportement d'origine

    if auto_filtering_mode != AutoFilteringOptions.DISPLAY_MODE_NOFILTERING:
        if chosen_scaler == scalerChoice.SCALER_DISABLE:
            print(
                f"\n!!!!!!!!!!!!!!!!!!!!!! POur la mutuelle information et MRMR il est préférable de normaliser les données  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")

        vif_stat_pca_result = compute_display_statistic(
            X=X_train_for_filtering, Y=y_train_label_for_filtering,
            name="X_train", config=config,
            compute_feature_stat=True,
            is_compute_vif=is_compute_vif)

        result['vif_stat_pca_result'] = vif_stat_pca_result
        # #compute pca on xtest
        # if stats_result is not None:
        #     # Extraire les informations du résultat
        #     #selected_columns_afterVifCorrMiFiltering = stats_result['retained_columns']
        #     X_train_modified = stats_result['X_with_pca_vif_stat']
        #     pca_info = stats_result.get('pca_info', {})
        #
        #     # Mettre à jour X_train
        #     #result['X_train'] = X_train_modified
        #
        #     # Pour X_test, appliquer seulement la PCA si l'objet PCA est disponible
        #     if X_test_noInfNan is not None and pca_info.get('applied', False):
        #         pca_object = pca_info.get('pca_object')
        #
        #         if pca_object is not None:
        #             # Extraire les informations nécessaires
        #             high_corr_columns_used4pca = pca_info.get('high_corr_columns_used4pca', [])
        #             pca_columns_created = pca_info.get('pca_columns_created', [])
        #
        #             # Cas 1 : liste des colonnes source vide
        #             if not high_corr_columns_used4pca:
        #                 raise ValueError(
        #                     "Impossible d'appliquer la PCA à X_test : la liste des colonnes sources pour la PCA est vide.")
        #
        #             # Cas 2 : certaines colonnes sources sont manquantes dans X_test
        #             missing_input_columns = [col for col in high_corr_columns_used4pca if col not in X_test_noInfNan.columns]
        #             if missing_input_columns:
        #                 raise ValueError(
        #                     f"Impossible d'appliquer la PCA à X_test : colonnes manquantes dans X_test : {missing_input_columns}")
        #
        #             # ✅ Tout est bon, on peut appliquer la PCA
        #             print(f"✅ Application de la PCA à X_test pour créer : {pca_columns_created}")
        #             # Copier X_test pour le modifier
        #             X_test_noInfNan_copy = X_test_noInfNan.copy()
        #
        #             # Appliquer transform() avec le même objet PCA
        #             X_test_pca = pca_object.transform(X_test_noInfNan_copy[high_corr_columns_used4pca])
        #
        #             # Ajouter les colonnes PCA générées
        #             for i, col_name in enumerate(pca_columns_created):
        #                 if i < X_test_pca.shape[1]:
        #                     X_test_noInfNan_copy[col_name] = X_test_pca[:, i]
        #                 else:
        #                     raise ValueError(f"La composante PCA {col_name} est manquante dans la transformation.")
        #
        #             # Extraire les colonnes finales de X_train (après filtrage et PCA)
        #             x_train_columns = list(X_train_modified.columns)
        #
        #             # Vérifier que toutes les colonnes de X_train sont bien dans X_test
        #             missing_in_test = [col for col in x_train_columns if col not in X_test_noInfNan_copy.columns]
        #             if missing_in_test:
        #                 raise ValueError(
        #                     f"Colonnes manquantes dans X_test après transformation PCA : {missing_in_test}")
        #
        #             # Réordonner et filtrer X_test selon les colonnes de X_train
        #             X_test_noInfNan_copy = X_test_noInfNan_copy[x_train_columns]
        #             result['X_test'] = X_test_noInfNan_copy
        #             print("✅ PCA appliquée à X_test avec succès.")
        #
        #         else:
        #             raise ValueError(
        #                 "Objet PCA non disponible : aucune transformation PCA ne peut être appliquée à X_test.")
        #
        #     else:
        #         raise ValueError("X_test est None ou pca_info['applied'] est False : impossible d'appliquer la PCA.")
        #
        # else:
        #     print("Aucune colonne retournée par compute_display_statistic")
        #     selected_columns_afterVifCorrMiFiltering = list(X_train.columns)
        #     exit(81)
    # rajouter dans result les colonnes de df_feature_vifAndStat_selected qui sont les colonnes selection après vif et stat
    for key in result:
        print(f"- {key}")
    return result,scaler


#  explanation = """
#     🔍 **Explication des variables du tableau de résultats :**
#
#     - **Feature** : Nom de la feature analysée.
#     - **Sample_Size** : Nombre d'observations utilisées après filtrage des NaN.
#     - **Effect_Size (Cohen's d)** : Mesure de la séparation entre les deux classes.
#       - **> 0.8** : Effet fort ✅
#       - **0.5 - 0.8** : Effet moyen ⚠️
#       - **< 0.5** : Effet faible ❌
#
#     - **P-Value** : Probabilité d'observer la relation par hasard.
#       - **< 0.01** : Très significatif ✅✅
#       - **0.01 - 0.05** : Significatif ✅
#       - **0.05 - 0.10** : Marginalement significatif ⚠️
#       - **> 0.10** : Non significatif ❌
#
#     - **Fisher_Score (ANOVA F-test)** : Mesure la force discriminante de la feature.
#       - **> 20** : Exceptionnellement puissant ✅✅
#       - **10 - 20** : Très intéressant ✅
#       - **5 - 10** : Modérément intéressant ⚠️
#       - **1 - 5** : Faiblement intéressant ❌
#
#     - **MI (Information Mutuelle)** : Quantifie la dépendance entre la feature et la cible.
#       - **> 0.1** : Dépendance forte ✅✅
#       - **0.05 - 0.1** : Dépendance significative ✅
#       - **0.01 - 0.05** : Dépendance modérée ⚠️
#       - **< 0.01** : Dépendance faible ❌
#
#     - **mRMR_Score** : Information Mutuelle avec la cible moins redondance moyenne avec autres features.
#       - **> 0.05** : Signal fort et unique ✅✅
#       - **0.02 - 0.05** : Signal modéré peu redondant ✅
#       - **0.01 - 0.02** : Signal faible ou partiellement redondant ⚠️
#       - **< 0.01** : Signal très faible ou hautement redondant ❌
#
#     - **VIF (Variance Inflation Factor)** : Mesure la multicolinéarité avec les autres features.
#       - **< 2.5** : Excellente indépendance ✅✅
#       - **2.5 - 5** : Bonne indépendance ✅
#       - **5 - 7.5** : Redondance modérée, à évaluer ⚠️
#       - **> 7.5** : Forte redondance, potentiellement problématique ❌
#       - Crucial en trading pour éviter le surapprentissage et garantir la stabilité du modèle.
#
#     - **lr_target (Corrélation de Pearson)** : Mesure la relation linéaire avec la cible.
#       - **|r| > 0.3** : Corrélation linéaire forte en trading ✅✅
#       - **0.2 < |r| < 0.3** : Corrélation linéaire modérée ✅
#       - **0.1 < |r| < 0.2** : Corrélation linéaire faible ⚠️
#       - **|r| < 0.1** : Corrélation linéaire négligeable ❌
#       - Note: En trading, même des corrélations de 0.02-0.05 peuvent être significatives si elles sont persistantes et exploitables à haute fréquence.
#
#     - **nlr_target (Corrélation de Spearman)** : Mesure la relation monotone (non-linéaire) avec la cible.
#       - **|ρ| > 0.25** : Corrélation de rang forte en trading ✅✅
#       - **0.15 < |ρ| < 0.25** : Corrélation de rang modérée ✅
#       - **0.08 < |ρ| < 0.15** : Corrélation de rang faible ⚠️
#       - **|ρ| < 0.08** : Corrélation de rang négligeable ❌
#       - Particulièrement utile pour détecter des patterns de marché non-linéaires.
#
#     - **Power_Analytical** : Puissance statistique basée sur une formule analytique.
#     - **Power_MonteCarlo** : Puissance statistique estimée via simulations.
#     - **Required_N** : Nombre d'observations nécessaires pour atteindre **Puissance = 0.8**.
#     - **Power_Sufficient** : L'échantillon actuel est-il suffisant pour garantir la fiabilité de l'effet observé ?
#
#     🎯 **Interprétation des seuils de puissance statistique** :
#     - ✅ **Puissance ≥ 0.8** : La feature a une distinction nette entre classes. Résultat très fiable.
#     - ⚠️ **0.6 ≤ Puissance < 0.8** : Impact potentiel, mais fiabilité modérée. À considérer dans un ensemble de signaux.
#     - ❌ **Puissance < 0.6** : Fiabilité insuffisante. Risque élevé que la relation observée soit due au hasard.
#
#     📈 **Considérations spécifiques pour le trading** :
#     - Une feature avec un score modeste mais stable sur différentes périodes peut être plus précieuse qu'une feature avec un score élevé mais instable.
#     - La différence entre les corrélations de Pearson (lr_target) et Spearman (nlr_target) peut révéler la nature de la relation avec le marché:
#       - Si Spearman > Pearson: Présence probable d'une relation non-linéaire (ex: effets de seuil, comportements asymétriques).
#       - Si Pearson > Spearman: Relation principalement linéaire, mais potentiellement influencée par des valeurs extrêmes.
#     - En trading algorithmique, des corrélations faibles (même 0.02-0.05) peuvent générer un avantage significatif si elles sont exploitées systématiquement et à grande échelle.
#     - Le VIF est crucial pour construire des modèles robustes: des features avec un fort pouvoir prédictif mais un VIF élevé peuvent déstabiliser le modèle en conditions réelles.
#     - Pour les stratégies haute fréquence, privilégiez des VIF plus stricts (<3-4) et des signaux faiblement corrélés entre eux.
#     - Pour les stratégies à plus long terme, des VIF jusqu'à 7-8 peuvent être acceptables si la feature apporte une information précieuse.
#     - Les scores mRMR élevés sont particulièrement précieux car ils identifient des signaux à la fois informatifs et non redondants.
#     - Testez toujours la stabilité temporelle de ces métriques sur différentes périodes de marché pour vérifier la persistance du signal.
#     """
#
#     print(explanation)
#     print("=" * 60)
#     print(f"Nombre total de features après filtrage manuel: {len(selected_columns_manual)}")
#     print(
#         f"Nombre total de features après filtrage VIF, CORR et MI: {len(selected_columns_afterVifCorrMiFiltering)}")
#     print("=" * 60)
#
#     # Nous ne faisons pas exit(77) ici pour permettre à la fonction de continuer
# else:
#     # Pas de filtrage automatique, on garde les features sélectionnées manuellement
#     selected_columns_afterVifCorrMiFiltering = selected_columns_manual
#     print("\nRésumé:")
#     print(f"Nombre total de features filtrées manuellement: {len(selected_columns_manual)}")
#     print(
#         f"Nombre total de features (filtrage VIF, CORR et MI désactivé): {len(selected_columns_afterVifCorrMiFiltering)}")

def init_dataSet(df_init_features=None, nanvalue_to_newval=None, config=None, CUSTOM_SESSIONS_=None,
                 results_directory=None):
    """
    Initialise et prépare les données pour l'entraînement et l'évaluation d'un modèle de trading.

    Args:
        df_init_features: DataFrame avec les caractéristiques brutes
        nanvalue_to_newval: Valeur à utiliser pour remplacer les NaN
        config: Dictionnaire de configuration
        CUSTOM_SESSIONS_: Définition des sessions de trading à utiliser
        results_directory: Répertoire pour sauvegarder les résultats

    Returns:
        Tuple contenant les ensembles de données préparés
    """
    # Récupération des colonnes sélectionnées manuellement depuis la configuration
    selected_columns_manual = config.get('selected_columns_manual', None)
    if selected_columns_manual is None:
        raise ValueError("La configuration doit contenir la clé 'selected_columns_manual'.")

    # Filtrage des données par sessions de trading si activé
    if (config['enable_script_CUSTOM_SESSIONS'] == True):
        df_filtered = sessions_selection(df_init_features, CUSTOM_SESSIONS_=CUSTOM_SESSIONS_,
                                         results_directory=results_directory)
    else:
        df_filtered = df_init_features

    # Gestion des valeurs NaN dans le dataset
    if nanvalue_to_newval is not None:
        df_filtered = df_filtered.fillna(nanvalue_to_newval).infer_objects(copy=False)
        nan_value = nanvalue_to_newval
    else:
        nan_value = np.nan

    # Division des données en ensembles d'entraînement et de test
    print("Division des données en ensembles d'entraînement et de test...")
    try:
        test_size_ratio = config.get('test_size_ratio', 0.8)
        train_df, nb_SessionTrain, test_df, nb_SessionTest = split_sessions(df_filtered, test_size=test_size_ratio,
                                                                            min_train_sessions=2,
                                                                            min_test_sessions=2)
    except ValueError as e:
        print(f"Erreur lors de la division des sessions : {e}")
        sys.exit(1)

    # Affichage du nombre de sessions dans chaque ensemble
    print(f"Nombre de session dans x_train  {nb_SessionTrain}  ")
    print(f"Nombre de session dans x_test  {nb_SessionTest}  ")

    # Création d'une copie des colonnes sélectionnées pour ne pas modifier la liste originale
    columns_to_use = selected_columns_manual.copy()

    # Création des versions "full" avec toutes les colonnes (avant filtrage des classes 99)
    X_train_full = train_df[:]
    y_train_full_label = train_df['class_binaire']
    X_test_full = test_df[:]
    y_test_full_label = test_df['class_binaire']

    # Sélection des données PnL (théoriques ou réelles selon la configuration)
    if (config['use_pnl_theoric']):
        df_pnl_data_train_full = train_df['trade_pnl_theoric']
        df_pnl_data_test_full = test_df['trade_pnl_theoric']
    else:
        df_pnl_data_train_full = train_df['trade_pnl']
        df_pnl_data_test_full = test_df['trade_pnl']
        # À vérifier - code d'arrêt si non théorique
        exit(98)

    # Création des versions filtrées (sans classe 99, qui représente les trades hors sessions sélectionnées)
    # Ensemble d'entraînement
    mask_train = y_train_full_label != 99
    X_train = X_train_full[mask_train]
    X_train = X_train[columns_to_use]  # Application de la sélection de colonnes
    y_train_label = y_train_full_label[mask_train]
    df_pnl_data_train = df_pnl_data_train_full[mask_train]

    # Ensemble de test
    mask_test = y_test_full_label != 99
    X_test = X_test_full[mask_test]
    X_test = X_test[columns_to_use]  # Application de la sélection de colonnes
    y_test_label = y_test_full_label[mask_test]
    df_pnl_data_test = df_pnl_data_test_full[mask_test]

    # Vérification de l'alignement entre les classes et les PnL
    y_train_label_aligned = y_train_full_label[mask_train]
    df_pnl_data_train_aligned = df_pnl_data_train_full[mask_train]
    check_original_alignment(y_train_label_aligned, df_pnl_data_train_aligned)

    # Test d'alignement des données d'entraînement (vérification des longueurs)
    if not (len(X_train) == len(y_train_label) == len(df_pnl_data_train)):
        raise ValueError(
            f"Erreur d'alignement: X_train ({len(X_train)} lignes), y_train_label ({len(y_train_label)} lignes), train_df ({len(train_df)} lignes), df_pnl_data_train ({len(df_pnl_data_train)} lignes)")
    else:
        print(f"Alignement vérifié: {len(X_train)} lignes pour tous les tableaux")

    # Test d'alignement des données de test (vérification des longueurs)
    if not (len(X_test) == len(y_test_label) == len(df_pnl_data_test)):
        raise ValueError(
            f"Erreur d'alignement: X_test ({len(X_test)} lignes), y_train_test ({len(y_test_label)} lignes), test_df ({len(test_df)} lignes), df_pnl_data_test ({len(df_pnl_data_test)} lignes)")
    else:
        print(f"Alignement vérifié: {len(X_test)} lignes pour tous les tableaux")

    # Vérification de la cohérence entre classes et PnL pour l'entraînement
    # y_train_label doit être 1 quand trade_pnl > 0 et 0 quand trade_pnl <= 0
    condition_respectee = ((y_train_label == 1) == (df_pnl_data_train > 0)).all()

    if condition_respectee:
        print("La condition est respectée pour tous les exemples")
    else:
        # Affichage des cas non conformes
        nb_non_conforme = ((y_train_label == 1) != (df_pnl_data_train > 0)).sum()
        print(f"La condition n'est pas respectée pour {nb_non_conforme} exemples")

        # Affichage de quelques exemples non conformes
        indices_non_conformes = ((y_train_label == 1) != (df_pnl_data_train['trade_pnl'] > 0))
        exemples_non_conformes = pd.DataFrame({
            'y_train_label': y_train_label[indices_non_conformes],
            'trade_pnl': df_pnl_data_train[indices_non_conformes]
        })
        print("\nVoici quelques exemples non conformes:")
        print(exemples_non_conformes.head())
        exit(56)  # Arrêt si incohérence détectée

    # Vérification similaire pour les données de test
    condition_respectee_test = ((y_test_label == 1) == (df_pnl_data_test > 0)).all()

    if condition_respectee_test:
        print("La condition est respectée pour tous les exemples de test")
    else:
        # Affichage des cas non conformes
        nb_non_conforme_test = ((y_test_label == 1) != (df_pnl_data_test['trade_pnl'] > 0)).sum()
        print(f"La condition n'est pas respectée pour {nb_non_conforme_test} exemples de test")

        # Affichage de quelques exemples non conformes
        indices_non_conformes_test = ((y_test_label == 1) != (df_pnl_data_test > 0))
        exemples_non_conformes_test = pd.DataFrame({
            'y_test_label': y_test_label[indices_non_conformes_test],
            'trade_pnl': df_pnl_data_test[indices_non_conformes_test]
        })
        print("\nVoici quelques exemples non conformes dans les données de test:")
        print(exemples_non_conformes_test.head())
        exit(57)  # Arrêt si incohérence détectée

    # ===== Gestion des colonnes temporelles pour le logging et la validation croisée =====
    # Ces colonnes sont nécessaires pour certains modes de validation croisée
    # et pour le logging temporel des prédictions, mais peuvent être absentes de X_train_full
    # si elles ne sont pas utilisées comme features

    # Liste des colonnes temporelles nécessaires
    temporal_columns = ['sc_timeStampOpening', 'sc_sessionStartEnd', 'sc_deltaTimestampOpening']

    for col in temporal_columns:
        if col not in X_train_full.columns:
            # Vérification de la présence de la colonne dans le DataFrame initial
            if col not in df_init_features.columns:
                raise ValueError(
                    f"La colonne '{col}' n'est pas présente dans df_init_features. Cette colonne est nécessaire pour les logs temporels et la validation croisée.")

            # Création d'une copie de X_train_full lors de la première modification
            if isinstance(X_train_full, pd.DataFrame) and not X_train_full.columns.empty:
                X_train_full = X_train_full.copy()

            # Ajout de la colonne en respectant l'indexation de X_train_full
            X_train_full[col] = df_init_features.loc[X_train_full.index, col]
            print(f"Colonne '{col}' ajoutée à X_train_full pour les logs temporels et la validation croisée")

    # ===== Vérification de la cohérence des distributions de classes =====
    # Compare les proportions de classes entre les différents ensembles

    def get_class_counts(y):
        """Retourne un tuple (count_0, count_1) pour un array/series donné"""
        if isinstance(y, pd.Series):
            return (y == 0).sum(), (y == 1).sum()
        return (y == 0).sum(), (y == 1).sum()

    # Vérification des ensembles complets (avec 99)
    counts_train_full = get_class_counts(y_train_full_label)
    counts_test_full = get_class_counts(y_test_full_label)

    # Vérification des ensembles réduits (sans 99)
    counts_train = get_class_counts(y_train_label)
    counts_test = get_class_counts(y_test_label)

    # Vérification de la cohérence des proportions
    if counts_train_full != counts_train or counts_test_full != counts_test:
        error_msg = f"""
    Incohérence détectée dans la distribution des classes:
    Train full    : {counts_train_full[0]} classe 0, {counts_train_full[1]} classe 1
    Train réduit  : {counts_train[0]} classe 0, {counts_train[1]} classe 1
    Test full     : {counts_test_full[0]} classe 0, {counts_test_full[1]} classe 1
    Test réduit   : {counts_test[0]} classe 0, {counts_test[1]} classe 1
    """
        raise ValueError(error_msg)

    # Vérification que les valeurs de X_test sont dans les bornes observées dans X_train
    try:
        check_lower_higher_bound_consistency(X_train, X_test, max_values_to_show=10)
        print("\n Vérification des bornes réussie sur x_train et x_test réussi")
    except ValueError as e:
        raise ValueError(f"Erreur: {e}")

    # Affichage des dimensions après filtrage manuel
    print("=" * 60)
    print(f"Shape de X_train après filtrage mannuel des features: {X_train.shape}")
    print(f"Shape de X_test  après filtrage mannuel des features: {X_test.shape}")
    print("=" * 60)

    # Liste des colonnes à vérifier

    # # Affichage des informations sur les features après exclusion manuelle
    # print(f"\nFeatures X_train_full après exclusion manuelle des features (short + 99)(a verivier AL)):")
    # compute_display_statistic(X=X_train_full, name="X_train_full",
    #                                config=config, compute_feature_stat=False)
    # print(f"Features X_train après exclusion manuelle des features (sur trades short après exclusion de 99):")
    # compute_display_statistic(X=X_train, name="X_train",
    #                                config=config, compute_feature_stat=False)

    X_train_cc = X_train.copy()
    y_train_label_cc = y_train_label.copy()
    X_test_cc = X_test.copy()
    y_test_label_cc = y_test_label.copy()
    df_pnl_data_train = df_pnl_data_train.copy()
    df_pnl_data_test = df_pnl_data_test.copy()
    # Appel de la fonction
    result ,scaler= filter_features(
        X_train=X_train,
        y_train_label=y_train_label,
        X_test=X_test,
        y_test_label=y_test_label,
        df_pnl_data_train=df_pnl_data_train,
        df_pnl_data_test=df_pnl_data_test,
        selected_columns_manual=selected_columns_manual,
        config=config,
        results_directory=results_directory,
        X_train_full=X_train_full
    )

    mask_train_noInfNan = result['mask_train_noInfNan']
    mask_test_noInfNan = result['mask_test_noInfNan']

    # Récupération des données nettoyées et des colonnes sélectionnées
    # Nettoyage cohérent des features, labels, et PNL avec les masques
    X_train_clean_noInfNan = X_train_cc[mask_train_noInfNan]
    y_train_label_clean_noInfNan = y_train_label_cc[mask_train_noInfNan]
    df_pnl_data_train_clean_noInfNan = df_pnl_data_train[mask_train_noInfNan]

    X_test_clean_noInfNan = X_test_cc[mask_test_noInfNan]
    y_test_label_clean_noInfNan = y_test_label_cc[mask_test_noInfNan]
    df_pnl_data_test_clean_noInfNan = df_pnl_data_test[mask_test_noInfNan]

    # selectionner les colonnes vif_stat
    # ajouter les highcorr_pca_colonms et leur contenu
    # memoriser et propager le non des highcorr_pca_colonms
    # dans le cv fairele split pour faire le pca sur les high corr et les supprimer de train.
    # les extrain de val et faire le transform dessus.
    # Faire le scaler sur train puis sur val avec le pca_
    # vif_stat_pca_result = compute_display_statistic(

    vif_stat_pca_result = result['vif_stat_pca_result']

    colomnsList_with_vif_stat_with_pca_ = vif_stat_pca_result['colomnsList_with_pca_vif_stat']
    pca_info = vif_stat_pca_result["pca_info"]
    high_corr_columns_used4pca = pca_info['high_corr_columns_used4pca']
    pca_object = pca_info['pca_object']

    print(
        f"\n📊 Colonnes finales utilisées (avec PCA + VIF + stats): {len(colomnsList_with_vif_stat_with_pca_)} colonnes")
    print(colomnsList_with_vif_stat_with_pca_)

    # 📤 Extraction des colonnes ne contenant pas 'pca_'
    colomnsList_with_vif_stat_without_pca_ = [
        col for col in colomnsList_with_vif_stat_with_pca_ if not col.startswith("pca_")
    ]

    # ✅ Affichage du résultat
    print(f"\n📊 Colonnes finales utilisées (sans les PCA) : {len(colomnsList_with_vif_stat_without_pca_)} colonnes")
    print(colomnsList_with_vif_stat_without_pca_)

    print(f"\n🔍 Colonnes utilisées pour le PCA (corrélées entre elles) : {len(high_corr_columns_used4pca)} colonnes")
    print(high_corr_columns_used4pca)

    # min_val = X_train_clean_noInfNan['bull_imbalance_low_3'].min()
    # max_val = X_train_clean_noInfNan['bull_imbalance_low_3'].max()
    #
    # print(f"Min: {min_val}, Max: {max_val}")
    # min_val = X_train['bull_imbalance_low_3'].min()
    # max_val = X_train['bull_imbalance_low_3'].max()
    #
    # print(f"Min: {min_val}, Max: {max_val}")

    def check_equality(a, b, name_a, name_b):
        if isinstance(a, pd.DataFrame) or isinstance(a, pd.Series):
            equal = a.equals(b)
        else:
            equal = np.array_equal(a, b)

        shape_a = a.shape if hasattr(a, 'shape') else 'N/A'
        shape_b = b.shape if hasattr(b, 'shape') else 'N/A'

        print(f"{name_a} (shape: {shape_a}) == {name_b} (shape: {shape_b}) ? {'✅ Oui' if equal else '❌ Non'}")

    check_equality(y_train_label_clean_noInfNan, y_train_label, 'y_train_label_clean_noInfNan', 'y_train_label')
    check_equality(X_train_clean_noInfNan, X_train, 'X_train_clean_noInfNan', 'X_train')
    check_equality(X_test_clean_noInfNan, X_test, 'X_test_clean_noInfNan', 'X_test')
    check_equality(y_test_label_clean_noInfNan, y_test_label, 'y_test_label_clean_noInfNan', 'y_test_label')

    # récupére les colonnes filtré après vif et stat et filtre le dataframe clean pour utiliser ces valeurs esuite
    # selected_columns = result['selected_columns']
    #
    # # Utilisation des colonnes sélectionnées
    # X_train_filtered = X_train_clean_noInfNan[selected_columns]
    # X_test_filtered = X_test_clean_noInfNan[selected_columns]

    # Affichage lisible
    print("Colonnes de X_train_clean_noInfNan :")
    print(list(X_train_clean_noInfNan.columns))

    # print("\nColonnes de X_test_clean_noInfNan :")
    # print(list(X_test_clean_noInfNan.columns))

    # Comparaison stricte : mêmes noms, même ordre
    if list(X_train_clean_noInfNan.columns) == list(X_test_clean_noInfNan.columns):
        print("\n✅ Les colonnes de X_train_clean_noInfNan et X_test_clean_noInfNan sont identiques (ordre et noms).")
    else:
        print("\n❌ Les colonnes sont différentes entre X_train_clean_noInfNan et X_test_clean_noInfNan.")

    def check_columns_not_empty_or_zero(df, df_name):
        empty_or_zero_cols = []
        for col in df.columns:
            values = df[col]
            # Vérifie si tout est NaN ou tout est 0
            if values.isna().all() or (values == 0).all():
                empty_or_zero_cols.append(col)

        if empty_or_zero_cols:
            raise ValueError(
                f"❌ Les colonnes suivantes de {df_name} sont vides ou constantes à 0 : {empty_or_zero_cols}")
        else:
            print(f"✅ Toutes les colonnes de {df_name} contiennent des valeurs valides (≠ 0 et ≠ NaN)")

    # Vérifications
    check_columns_not_empty_or_zero(X_train_clean_noInfNan, "X_train_clean_noInfNan")
    check_columns_not_empty_or_zero(X_test_clean_noInfNan, "X_test_clean_noInfNan")

    # Retour des ensembles de données préparés

    return (X_train_full, y_train_full_label, X_test_full, y_test_full_label,
            X_train_clean_noInfNan, y_train_label_clean_noInfNan, X_test_clean_noInfNan, y_test_label_clean_noInfNan,
            df_pnl_data_train_clean_noInfNan, df_pnl_data_test_clean_noInfNan,
            colomnsList_with_vif_stat_without_pca_, high_corr_columns_used4pca,
            nb_SessionTrain, nb_SessionTest, nan_value,scaler,pca_object)


def add_early_stopping_zone(ax, best_iteration, color='orange', alpha=0.2):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.axvspan(best_iteration, xmax, facecolor=color, alpha=alpha)
    ax.text(best_iteration + (xmax - best_iteration) / 2, ymax, 'Zone post early stopping',
            horizontalalignment='center', verticalalignment='top', fontsize=12, color='orange')


def plot_custom_metric_evolution_with_trade_info(model, evals_result, metric_name='custom_metric_PNL',
                                                 n_train_trades=None, n_test_trades=None, results_directory=None):
    if not evals_result or 'train' not in evals_result or 'test' not in evals_result:
        print("Résultats d'évaluation incomplets ou non disponibles.")
        return

    train_metric = evals_result['train'][metric_name]
    test_metric = evals_result['test'][metric_name]

    iterations = list(range(1, len(train_metric) + 1))
    best_test_iteration = np.argmax(test_metric)

    fig, ((ax1, ax2, ax5), (ax3, ax4, ax6)) = plt.subplots(2, 3, figsize=(24, 14))
    fig.suptitle(f'Entraînement du modèle final avec les paramètres optimaux (Optuna) :\n'
                 f'Évaluation du score {metric_name} sur l\'ensemble d\'entraînement (X_train) '
                 f'et un nouvel ensemble de test indépendant (X_test)', fontsize=12)

    def add_vertical_line_and_annotations(ax, is_train, is_normalized=False):
        ax.axvline(x=best_test_iteration, color='green', linestyle='--')
        y_pos = ax.get_ylim()[1] if is_train else ax.get_ylim()[0]
        score = train_metric[best_test_iteration] if is_train else test_metric[best_test_iteration]
        if is_normalized:
            score = (score - min(train_metric if is_train else test_metric)) / (
                    max(train_metric if is_train else test_metric) - min(
                train_metric if is_train else test_metric))
        ax.annotate(f'{"Train" if is_train else "Test"} Score: {score:.2f}',
                    (best_test_iteration, y_pos), xytext=(5, 5 if is_train else -5),
                    textcoords='offset points', ha='left', va='bottom' if is_train else 'top',
                    fontsize=8, bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))

    # X_train (Non Normalisé)
    ax1.plot(iterations, train_metric, label='Train', color='blue')
    ax1.set_title(f'X_train (Non Normalized)', fontsize=14)
    ax1.set_xlabel('Number of Iterations', fontsize=12)
    ax1.set_ylabel(f'{metric_name} Score', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.text(0.5, 0.1, f'Profit cumulés réalisé sur {n_train_trades} trades',
             horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes, fontsize=12,
             color='blue')
    add_vertical_line_and_annotations(ax1, is_train=True)
    add_early_stopping_zone(ax1, best_test_iteration)

    # X_test (Non Normalisé)
    ax2.plot(iterations, test_metric, label='Test', color='red')
    ax2.set_title(f'X_test (Non Normalized)', fontsize=14)
    ax2.set_xlabel('Number of Iterations', fontsize=12)
    ax2.set_ylabel(f'{metric_name} Score', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.text(0.5, 0.1, f'Profit cumulés réalisé sur {n_test_trades} trades',
             horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes, fontsize=12,
             color='red')
    add_vertical_line_and_annotations(ax2, is_train=False)
    add_early_stopping_zone(ax2, best_test_iteration)

    # Normalisation
    train_min, train_max = min(train_metric), max(train_metric)
    test_min, test_max = min(test_metric), max(test_metric)
    train_normalized = [(val - train_min) / (train_max - train_min) for val in train_metric]
    test_normalized = [(val - test_min) / (test_max - test_min) for val in test_metric]

    # X_train (Normalisé)
    ax3.plot(iterations, train_normalized, label='Train (Normalized)', color='blue')
    ax3.set_title(f'X_train (Normalized)', fontsize=14)
    ax3.set_xlabel('Number of Iterations', fontsize=12)
    ax3.set_ylabel(f'Normalized {metric_name} Score', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.text(0.5, 0.1,
             f'Profit cumulés réalisé sur {n_train_trades} trades\nNorm Ratio: [{train_min:.4f}, {train_max:.4f}]',
             horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes, fontsize=12,
             color='blue')
    add_vertical_line_and_annotations(ax3, is_train=True, is_normalized=True)
    add_early_stopping_zone(ax3, best_test_iteration)

    # X_test (Normalisé)
    ax4.plot(iterations, test_normalized, label='Test (Normalized)', color='red')
    ax4.set_title(f'X_test (Normalized)', fontsize=14)
    ax4.set_xlabel('Number of Iterations', fontsize=12)
    ax4.set_ylabel(f'Normalized {metric_name} Score', fontsize=12)
    ax4.legend(fontsize=10)
    ax4.grid(True, linestyle='--', alpha=0.7)
    ax4.text(0.5, 0.1,
             f'Profit cumulés réalisé sur {n_test_trades} trades\nNorm Ratio: [{test_min:.4f}, {test_max:.4f}]',
             horizontalalignment='center', verticalalignment='center', transform=ax4.transAxes, fontsize=12,
             color='red')
    add_vertical_line_and_annotations(ax4, is_train=False, is_normalized=True)
    add_early_stopping_zone(ax4, best_test_iteration)

    # X_test (Non Normalisé) jusqu'à best_test_iteration
    ax5.plot(iterations[:best_test_iteration + 1], test_metric[:best_test_iteration + 1], label='Test', color='red')
    ax5.set_title(f'X_test (Non Normalized) until best Test Score', fontsize=14)
    ax5.set_xlabel('Number of Iterations', fontsize=12)
    ax5.set_ylabel(f'{metric_name} Score', fontsize=12)
    ax5.legend(fontsize=10)
    ax5.grid(True, linestyle='--', alpha=0.7)
    add_vertical_line_and_annotations(ax5, is_train=False)

    # X_test (Normalisé) jusqu'à best_test_iteration
    ax6.plot(iterations[:best_test_iteration + 1], test_normalized[:best_test_iteration + 1],
             label='Test (Normalized)', color='red')
    ax6.set_title(f'X_test (Normalized) until best Test Score', fontsize=14)
    ax6.set_xlabel('Number of Iterations', fontsize=12)
    ax6.set_ylabel(f'Normalized {metric_name} Score', fontsize=12)
    ax6.legend(fontsize=10)
    ax6.grid(True, linestyle='--', alpha=0.7)
    add_vertical_line_and_annotations(ax6, is_train=False, is_normalized=True)
    plt.savefig(os.path.join(results_directory, f'Evolution of {metric_name} Score with Trade Information'),
                dpi=300,
                bbox_inches='tight')
    plt.tight_layout()

    plt.close()


def save_correlations_to_csv(corr_matrix, results_directory):
    csv_filename = "all_correlations.csv"
    csv_path = os.path.join(results_directory, csv_filename)

    # Initialisation
    seen_pairs = set()
    csv_data = []

    # Traitement de toutes les corrélations
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            feature_i = corr_matrix.columns[i]
            feature_j = corr_matrix.columns[j]
            corr_value = corr_matrix.iloc[i, j]

            # Créer une paire triée pour éviter les doublons
            pair = tuple(sorted([feature_i, feature_j]))

            # Si la paire n'est pas encore vue et ce n'est pas une auto-corrélation
            if pair not in seen_pairs and feature_i != feature_j:
                # Préparation données CSV
                interaction = f"{feature_i} <-> {feature_j}"
                absolute_corr = abs(corr_value)
                csv_data.append([
                    interaction,
                    f"{corr_value:.4f}",
                    f"{absolute_corr:.4f}"
                ])
                seen_pairs.add(pair)

    # Tri par corrélation absolue décroissante
    csv_data.sort(key=lambda x: float(x[2]), reverse=True)

    # Sauvegarde du CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(['Correlation_Pair', 'Correlation_Value', 'Absolute_Correlation'])
        writer.writerows(csv_data)

    print(f"\nFichier CSV sauvegardé: {csv_path}")
    return csv_path


def analyze_and_save_feature_correlations(X_train, results_directory, threshold=0.8):
    """
    Analyzes feature correlations in a given DataFrame, visualizes the correlation matrix,
    identifies pairs of highly correlated features, and saves the heatmap to a file.

    Parameters:
    - X_train (pd.DataFrame): The feature DataFrame.
    - results_directory (str): The directory where the correlation heatmap will be saved.
    - threshold (float): The correlation threshold for identifying highly correlated feature pairs.

    Returns:
    - high_corr_pairs (list of tuples): Sorted list of tuples with highly correlated feature pairs and their correlation values.
    """
    # Calculate the correlation matrix
    corr_matrix = X_train.corr()

    # Visualize the correlation matrix with smaller font size and two decimal precision
    plt.figure(figsize=(20, 20))
    sns.heatmap(corr_matrix, annot=True, fmt='.1f', cmap='coolwarm', annot_kws={"size": 4})
    plt.title('Feature Correlation Matrix', fontsize=10)

    # Ensure the results directory exists
    os.makedirs(results_directory, exist_ok=True)

    # Save the heatmap
    heatmap_path = os.path.join(results_directory, 'feature_correlation_matrix.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Identify pairs of highly correlated features
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                feature_i = corr_matrix.columns[i]
                feature_j = corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]
                high_corr_pairs.append((feature_i, feature_j, corr_value))

    # Sort the results by the absolute correlation value in descending order
    high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    # Display the sorted results in the console
    print(f"Highly correlated pairs (threshold={threshold}):")
    for feature_i, feature_j, corr_value in high_corr_pairs:
        print(f"{feature_i} <-> {feature_j}: {corr_value:.2f}")

    csv_path = save_correlations_to_csv(corr_matrix, results_directory)

    return high_corr_pairs


def analyze_shap_interactions(final_model, X_test, results_directory, dataset_name="test"):
    """
    Analyse les interactions SHAP et génère des graphiques ainsi qu'un CSV.

    Parameters:
    -----------
    final_model : xgboost.Booster
        Modèle XGBoost entraîné.
    X_test : pandas.DataFrame
        Données pour lesquelles les interactions SHAP doivent être calculées.
    results_directory : str
        Répertoire où sauvegarder les résultats.
    dataset_name : str, optional (default="test")
        Nom du dataset (par exemple, "train" ou "test") pour personnaliser les fichiers générés.

    Returns:
    --------
    None
    """
    # lxtest = lgb.Dataset(X_test) semble inutile
    shap_interaction_values = final_model.predict(
        X_test,
        pred_interactions=True,
        iteration_range=(0, final_model.best_iteration)  # C'est correct tel quel
    )
    # Calcul des valeurs d'interaction SHAP
    # shap_interaction_values = final_model.predict(xgb.DMatrix(X_test), pred_interactions=True)
    # Exclure le biais en supprimant la dernière ligne et la dernière colonne
    shap_interaction_values = shap_interaction_values[:, :-1, :-1]

    # Vérification de la compatibilité des dimensions
    print(f"Shape of shap_interaction_values ({dataset_name}):", shap_interaction_values.shape)
    print(f"Number of features in X_test ({dataset_name}):", len(X_test.columns))

    if shap_interaction_values.shape[1:] != (len(X_test.columns), len(X_test.columns)):
        print(
            "Erreur : Incompatibilité entre les dimensions des valeurs d'interaction SHAP et le nombre de features."
        )
        print(f"Dimensions des valeurs d'interaction SHAP : {shap_interaction_values.shape}")
        print(f"Nombre de features dans X_test ({dataset_name}): {len(X_test.columns)}")
        return

    # Calcul de la matrice d'interactions
    interaction_matrix = np.abs(shap_interaction_values).sum(axis=0)
    feature_names = X_test.columns
    interaction_df = pd.DataFrame(interaction_matrix, index=feature_names, columns=feature_names)

    # Masquer la diagonale (interactions d'une feature avec elle-même)
    np.fill_diagonal(interaction_df.values, 0)

    # Récupérer toutes les interactions non nulles
    all_interactions = interaction_df.unstack().sort_values(ascending=False)

    # Pour le graphique, on garde les top N
    N = 80
    top_interactions = all_interactions.head(N)

    # Visualisation des top interactions
    plt.figure(figsize=(24, 16))
    top_interactions.plot(kind='bar')
    plt.title(f"Top {N} Feature Interactions ({dataset_name})", fontsize=16)
    plt.xlabel("Feature Pairs", fontsize=12)
    plt.ylabel("Total Interaction Strength", fontsize=10)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(
        os.path.join(results_directory, f'top_feature_interactions_{dataset_name}.png'),
        dpi=300, bbox_inches='tight'
    )
    plt.close()

    # Calculer la somme totale des interactions pour la normalisation
    total_interaction_value = sum(value for (f1, f2), value in all_interactions.items() if f1 != f2)

    # Initialisation
    seen_pairs = set()
    csv_filename = f"all_interactions_{dataset_name}.csv"
    csv_path = os.path.join(results_directory, csv_filename)

    # Préparation des données CSV avec normalisation
    csv_data = []

    for (f1, f2), value in all_interactions.items():
        pair = tuple(sorted([f1, f2]))
        if f1 != f2 and pair not in seen_pairs:
            percentage = 2 * (value / total_interaction_value) * 100
            if len(seen_pairs) < 40:
                print(f"{f1} <-> {f2}: {value:.4f} ({percentage:.2f}%)")
            interaction = f"{f1} <-> {f2}"
            csv_data.append([interaction, f"{value:.4f}", f"{percentage:.4f}"])
            seen_pairs.add(pair)

    csv_data.sort(key=lambda x: float(x[1]), reverse=True)

    # Sauvegarde du CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(['Interaction', 'Value', 'Percentage_Contribution'])
        writer.writerows(csv_data)

    print(f"\nLes interactions ({dataset_name}) ont été sauvegardées dans : {csv_path}")
    print(f"Nombre total d'interactions sauvegardées ({dataset_name}) : {len(csv_data)}")

    # Création de la heatmap
    top_features = interaction_df.sum().sort_values(ascending=False).head(N).index
    plt.figure(figsize=(26, 20))
    sns.heatmap(
        interaction_df.loc[top_features, top_features].round(0).astype(int),
        annot=True, cmap='coolwarm', fmt='d',
        annot_kws={'size': 7}, square=True, linewidths=0.5,
        cbar_kws={'shrink': .8}
    )
    plt.title(f"SHAP Interaction Values Heatmap ({dataset_name})", fontsize=16)
    plt.tight_layout()
    plt.xticks(rotation=90, ha='center')
    plt.yticks(rotation=0)
    plt.savefig(
        os.path.join(results_directory, f'feature_interaction_heatmap_{dataset_name}.png'),
        dpi=300, bbox_inches='tight'
    )
    plt.close()

    print(f"Heatmap des interactions sauvegardée sous 'feature_interaction_heatmap_{dataset_name}.png'")


def best_modellastFold_analyse(X_test=None, y_test_label=None,
                               bestResult_dict=None, results_directory=None, config=None):
    best_modellastFold_string = bestResult_dict['best_modellastFold_string']
    best_modellastFold_params = bestResult_dict['best_modellastFold_params']
    # Recharger le modèle
    import lightgbm as lgb
    best_modellastFold = lgb.Booster(model_str=best_modellastFold_string)
    # Assigner les paramètres récupérés au modèle
    if best_modellastFold_params:
        best_modellastFold.params.update(best_modellastFold_params)

    other_params_best = bestResult_dict["other_params"]
    y_pred_proba_afterSig, pred_proba_log_odds, predictions_converted, (
        tn_xtest, fp_xtest, fn_xtest, tp_xtest), y_test_label_converted = predict_and_compute_metrics_XgbOrLightGbm(
        model=best_modellastFold,
        X_data=X_test,
        y_true=y_test_label,
        best_iteration=best_modellastFold.best_iteration,
        threshold=other_params_best['threshold'],
        config=config
    )

    reporting_model_performance(pred_proba_log_odds, tp_xtest, fp_xtest, fn_xtest, tn_xtest, config)

    """
    print_notification('###### DEBUT: ANALYSE DES DEPENDENCES SHAP DU MOBEL FINAL (ENTRAINEMENT) ##########',
                       color="blue")
    importance_df, shap_comparison, shap_values_train, shap_values_test, resulat_test_shap_feature_importance = main_shap_analysis(
                        cv_model, X_train, y_train_label_, X_test, y_test_label_,
                        save_dir=os.path.join(results_directory, 'shap_dependencies_results_reTrainModel'))
    print_notification('###### FIN: ANALYSE DES DEPENDENCES SHAP DU MOBEL FINAL (ENTRAINEMENT) ##########',
                       color="blue")
    """

from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from scipy.special import expit  # Sigmoid = expit(log_odds)


import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from sklearn.isotonic import IsotonicRegression
from scipy.special import expit



def log_stats(name, data, bins=30):
    print(f"\n📊 Statistiques pour {name}:")
    print(f"  Min       : {np.min(data):.6f}")
    print(f"  Max       : {np.max(data):.6f}")
    print(f"  Moyenne   : {np.mean(data):.6f}")
    print(f"  Écart-type: {np.std(data):.6f}")

    counts, edges = np.histogram(data, bins=bins)
    for i in range(len(counts)):
        print(f"  Bin {i+1} ({edges[i]:.3f} - {edges[i+1]:.3f}) : {counts[i]}")
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from sklearn.isotonic import IsotonicRegression
from scipy.special import expit


def compare_calibration_methods(model, X_valid, y_valid, config=None, other_params=None, score_bestIdx=0, save_dir=None,
                                display=True):
    """
    Calibre un modèle avec différentes méthodes et affiche les résultats
    sur des graphiques séparés pour une meilleure lisibilité.

    Args:
        model: Le modèle à calibrer
        X_valid: Les données de validation (features)
        y_valid: Les labels de validation (cibles)
        config: Configuration du modèle
        other_params: Paramètres additionnels
        score_bestIdx: Indice de la meilleure itération
        save_dir (str, optional): Répertoire où sauvegarder la figure. Si None, la figure n'est pas sauvegardée.
        display (bool, optional): Si True, affiche la figure. Par défaut True.

    Returns:
        dict: Dictionnaire contenant les probabilités et modèles de calibration
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from sklearn.metrics import brier_score_loss
    from sklearn.calibration import calibration_curve

    if config['model_type'] == modelType.XGB:
        import xgboost as xgb
        X_data = xgb.DMatrix(X_valid)
    else:
        X_data = X_valid  # LightGBM accepte le format natif

    # Appel de la fonction commune pour récupérer les probas et log-odds
    proba_raw, log_odds, _, _, _ = predict_and_compute_metrics_XgbOrLightGbm(
        model=model,
        X_data=X_data,
        y_true=y_valid,
        best_iteration=score_bestIdx + 1, threshold=other_params['threshold'],
        config=config
    )

    # 2) Platt scaling (sigmoid calibration) implemented manually
    from sklearn.linear_model import LogisticRegression
    platt_model = LogisticRegression(C=1.0)
    platt_model.fit(log_odds.reshape(-1, 1), y_valid)
    proba_platt = platt_model.predict_proba(log_odds.reshape(-1, 1))[:, 1]

    # 3) Isotonic Regression applied directly
    from sklearn.isotonic import IsotonicRegression
    iso_model = IsotonicRegression(out_of_bounds='clip')
    iso_model.fit(proba_raw, y_valid)
    proba_iso = iso_model.predict(proba_raw)

    # 4) Beta Calibration
    from functools import partial
    beta_model = BetaCalibrationCustom(method="ab")  # "ab", "abm", or "am"
    beta_model.fit(proba_raw, y_valid)
    proba_beta = beta_model.predict(proba_raw)

    # Brier scores
    brier_raw = brier_score_loss(y_valid, proba_raw)
    brier_platt = brier_score_loss(y_valid, proba_platt)
    brier_iso = brier_score_loss(y_valid, proba_iso)
    brier_beta = brier_score_loss(y_valid, proba_beta)

    print("\n🎯 Brier scores :")
    print(f"  - Avant calibration  : {brier_raw:.5f}")
    print(f"  - Platt scaling      : {brier_platt:.5f}")
    print(f"  - Isotonic           : {brier_iso:.5f}")
    print(f"  - Beta calibration   : {brier_beta:.5f}")

    # Log stats
    log_stats("Log-odds", log_odds, bins=30)
    log_stats("Proba brute (sigmoidée)", proba_raw, bins=30)
    log_stats("Platt scaling", proba_platt, bins=30)
    log_stats("Isotonic", proba_iso, bins=30)
    log_stats("Beta calibration", proba_beta, bins=30)

    # Création d'une figure adaptée à un écran large (3440x1440)
    fig = plt.figure(figsize=(34, 18))  # Ajusté pour écran large

    # Configuration des sous-figures en grille
    gs = plt.GridSpec(3, 4, figure=fig)  # Grille 3x4 pour 12 graphiques

    # 1. Distribution des log-odds
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(log_odds, bins=30, color='slateblue', alpha=0.7)
    ax1.set_title("Distribution log-odds", fontsize=14)
    ax1.set_xlabel("Valeur", fontsize=12)
    ax1.set_ylabel("Fréquence", fontsize=12)
    ax1.grid(True)

    # 2. Distribution des probas brutes (1 seule fois)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(proba_raw, bins=30, color='steelblue', alpha=0.7)
    ax2.set_title("Proba brute (sigmoid)", fontsize=14)
    ax2.set_xlabel("Probabilité", fontsize=12)
    ax2.set_ylabel("Fréquence", fontsize=12)
    ax2.grid(True)

    # 3. Distribution probabilités Platt
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(proba_platt, bins=30, color='darkorange', alpha=0.7)
    ax3.set_title("Distribution Platt scaling", fontsize=14)
    ax3.set_xlabel("Probabilité", fontsize=12)
    ax3.set_ylabel("Fréquence", fontsize=12)
    ax3.grid(True)

    # 4. Distribution probabilités Isotonic
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.hist(proba_iso, bins=30, color='forestgreen', alpha=0.7)
    ax4.set_title("Distribution Isotonic", fontsize=14)
    ax4.set_xlabel("Probabilité", fontsize=12)
    ax4.set_ylabel("Fréquence", fontsize=12)
    ax4.grid(True)

    # 5. Distribution probabilités Beta
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.hist(proba_beta, bins=30, color='crimson', alpha=0.7)
    ax5.set_title("Distribution Beta calibration", fontsize=14)
    ax5.set_xlabel("Probabilité", fontsize=12)
    ax5.set_ylabel("Fréquence", fontsize=12)
    ax5.grid(True)

    # Créer un grid de points x pour les courbes de calibration
    x_grid = np.linspace(0, 1, 100)

    # Platt scaling curve
    platt_a, platt_b = platt_model.coef_[0][0], platt_model.intercept_[0]
    platt_curve = 1 / (1 + np.exp(-(platt_a * x_grid + platt_b)))

    # Isotonic regression curve
    iso_curve = iso_model.predict(x_grid)

    # Beta calibration curve
    beta_curve = beta_model.predict(x_grid)

    # 6. Courbe de calibration Platt
    ax6 = fig.add_subplot(gs[1, 1])
    ax6.plot([0, 1], [0, 1], '--', color='gray', label='Parfaite calibration')
    ax6.plot(x_grid, platt_curve, color='darkorange', label=f'Platt (Brier: {brier_platt:.4f})')
    ax6.set_title("Courbe de calibration - Platt", fontsize=14)
    ax6.set_xlabel("Probabilité brute", fontsize=12)
    ax6.set_ylabel("Probabilité calibrée", fontsize=12)
    ax6.legend()
    ax6.grid(True)

    # 7. Courbe de calibration Isotonic
    ax7 = fig.add_subplot(gs[1, 2])
    ax7.plot([0, 1], [0, 1], '--', color='gray', label='Parfaite calibration')
    ax7.plot(x_grid, iso_curve, color='forestgreen', label=f'Isotonic (Brier: {brier_iso:.4f})')
    ax7.set_title("Courbe de calibration - Isotonic", fontsize=14)
    ax7.set_xlabel("Probabilité brute", fontsize=12)
    ax7.set_ylabel("Probabilité calibrée", fontsize=12)
    ax7.legend()
    ax7.grid(True)

    # 8. Courbe de calibration Beta
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.plot([0, 1], [0, 1], '--', color='gray', label='Parfaite calibration')
    ax8.plot(x_grid, beta_curve, color='crimson', label=f'Beta (Brier: {brier_beta:.4f})')
    ax8.set_title("Courbe de calibration - Beta", fontsize=14)
    ax8.set_xlabel("Probabilité brute", fontsize=12)
    ax8.set_ylabel("Probabilité calibrée", fontsize=12)
    ax8.legend()
    ax8.grid(True)

    # Reliability diagrams (calibration curves) - Un par méthode
    prob_true_raw, prob_pred_raw = calibration_curve(y_valid, proba_raw, n_bins=20)
    prob_true_platt, prob_pred_platt = calibration_curve(y_valid, proba_platt, n_bins=20)
    prob_true_iso, prob_pred_iso = calibration_curve(y_valid, proba_iso, n_bins=20)
    prob_true_beta, prob_pred_beta = calibration_curve(y_valid, proba_beta, n_bins=20)

    # 9. Reliability Diagram - Brute
    ax9 = fig.add_subplot(gs[2, 0])
    ax9.plot([0, 1], [0, 1], '--', color='gray')
    ax9.plot(prob_pred_raw, prob_true_raw, 's-', color='steelblue', label='Brute')
    ax9.set_title('Reliability Diagram - Brute', fontsize=14)
    ax9.set_xlabel('Proba moyenne prédite par bin', fontsize=12)
    ax9.set_ylabel('Fraction de positifs', fontsize=12)
    ax9.legend()
    ax9.grid(True)

    # 10. Reliability Diagram - Platt
    ax10 = fig.add_subplot(gs[2, 1])
    ax10.plot([0, 1], [0, 1], '--', color='gray')
    ax10.plot(prob_pred_platt, prob_true_platt, 'o-', color='darkorange', label='Platt')
    ax10.set_title('Reliability Diagram - Platt', fontsize=14)
    ax10.set_xlabel('Proba moyenne prédite par bin', fontsize=12)
    ax10.set_ylabel('Fraction de positifs', fontsize=12)
    ax10.legend()
    ax10.grid(True)

    # 11. Reliability Diagram - Isotonic
    ax11 = fig.add_subplot(gs[2, 2])
    ax11.plot([0, 1], [0, 1], '--', color='gray')
    ax11.plot(prob_pred_iso, prob_true_iso, '^-', color='forestgreen', label='Isotonic')
    ax11.set_title('Reliability Diagram - Isotonic', fontsize=14)
    ax11.set_xlabel('Proba moyenne prédite par bin', fontsize=12)
    ax11.set_ylabel('Fraction de positifs', fontsize=12)
    ax11.legend()
    ax11.grid(True)

    # 12. Reliability Diagram - Beta
    ax12 = fig.add_subplot(gs[2, 3])
    ax12.plot([0, 1], [0, 1], '--', color='gray')
    ax12.plot(prob_pred_beta, prob_true_beta, 'v-', color='crimson', label='Beta')
    ax12.set_title('Reliability Diagram - Beta', fontsize=14)
    ax12.set_xlabel('Proba moyenne prédite par bin', fontsize=12)
    ax12.set_ylabel('Fraction de positifs', fontsize=12)
    ax12.legend()
    ax12.grid(True)

    # Ajout d'un titre global
    fig.suptitle('Comparaison des méthodes de calibration', fontsize=18)

    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Ajustement pour le titre global

    # Sauvegarde de la figure si un répertoire est spécifié
    if save_dir:
        # Création du répertoire s'il n'existe pas
        os.makedirs(save_dir, exist_ok=True)

        # Création du nom de fichier
        filename = f"calibration_comparison.png"
        filepath = os.path.join(save_dir, filename)

        # Sauvegarde de la figure avec une bonne résolution
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"\n📥 Figure sauvegardée : {filepath}")

        # Sauvegarde des métriques dans un fichier texte
        metrics_filename = f"calibration_metrics.txt"
        metrics_filepath = os.path.join(save_dir, metrics_filename)

        with open(metrics_filepath, 'w') as f:
            f.write(" Brier scores :\n")
            f.write(f"  - Avant calibration  : {brier_raw:.5f}\n")
            f.write(f"  - Platt scaling      : {brier_platt:.5f}\n")
            f.write(f"  - Isotonic           : {brier_iso:.5f}\n")
            f.write(f"  - Beta calibration   : {brier_beta:.5f}\n")

        print(f"📝 Métriques sauvegardées : {metrics_filepath}")

    # Affichage de la figure selon le paramètre display
    if display:
        plt.show()

    return {
        "proba_raw": proba_raw,
        "proba_platt": proba_platt,
        "proba_iso": proba_iso,
        "proba_beta": proba_beta,
        "log_odds": log_odds,
        "brier_scores": {
            "raw": brier_raw,
            "platt": brier_platt,
            "isotonic": brier_iso,
            "beta": brier_beta
        },
        "calibrators": {
            "platt_model": platt_model,
            "iso_model": iso_model,
            "beta_model": beta_model
        }
    }

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from scipy.special import expit

def calibrate_and_plot_dual(
    log_odds,
    y_true,
    device="cpu",
    n_bins=10
):
    """
    Calibre les probabilités avec Platt et Isotonic, compare les courbes de calibration,
    affiche les Brier scores et les distributions, et donne le nombre de points par bin.

    Returns:
        dict: dictionnaire contenant les proba calibrées et les scores.
    """

    # GPU → CPU si nécessaire
    if device != "cpu":
        import cupy as cp
        log_odds = cp.asnumpy(log_odds)
        y_true = cp.asnumpy(y_true)

    # Probabilités brutes
    proba_raw = expit(log_odds)

    # Calibration : Platt
    calibrator_platt = LogisticRegression(solver='lbfgs')
    calibrator_platt.fit(proba_raw.reshape(-1, 1), y_true)
    proba_cal_platt = calibrator_platt.predict_proba(proba_raw.reshape(-1, 1))[:, 1]

    # Calibration : Isotonic
    calibrator_iso = IsotonicRegression(out_of_bounds='clip')
    calibrator_iso.fit(proba_raw, y_true)
    proba_cal_iso = calibrator_iso.predict(proba_raw)

    # Brier scores
    brier_raw = brier_score_loss(y_true, proba_raw)
    brier_platt = brier_score_loss(y_true, proba_cal_platt)
    brier_iso = brier_score_loss(y_true, proba_cal_iso)

    # Courbes de calibration
    raw_true, raw_pred = calibration_curve(y_true, proba_raw, n_bins=n_bins)
    platt_true, platt_pred = calibration_curve(y_true, proba_cal_platt, n_bins=n_bins)
    iso_true, iso_pred = calibration_curve(y_true, proba_cal_iso, n_bins=n_bins)

    # 📊 Nombre de points par bin (Platt)
    print(f"\n📌 Répartition des points par bin (Platt - {n_bins} bins) :")
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(proba_cal_platt, bins) - 1
    for i in range(n_bins):
        count = np.sum(bin_ids == i)
        print(f"Bin {i + 1} ({bins[i]:.2f} - {bins[i + 1]:.2f}) : {count} points")

    # 📊 Nombre de points par bin (Isotonic)
    print(f"\n📌 Répartition des points par bin (Isotonic - {n_bins} bins) :")
    bin_ids_iso = np.digitize(proba_cal_iso, bins) - 1
    for i in range(n_bins):
        count = np.sum(bin_ids_iso == i)
        print(f"Bin {i + 1} ({bins[i]:.2f} - {bins[i + 1]:.2f}) : {count} points")

    # 📈 Courbe de calibration
    plt.figure(figsize=(10, 6))
    plt.plot(raw_pred, raw_true, marker='o', label=f'Avant calibration\nBrier={brier_raw:.4f}')
    plt.plot(platt_pred, platt_true, marker='s', label=f'Platt scaling\nBrier={brier_platt:.4f}')
    plt.plot(iso_pred, iso_true, marker='^', label=f'Isotonic regression\nBrier={brier_iso:.4f}')
    plt.plot([0, 1], [0, 1], '--', color='gray', label='Calibration parfaite')
    plt.title(f"Courbe de calibration - Platt vs Isotonic")
    plt.xlabel("Probabilité prédite")
    plt.ylabel("Fréquence réelle (observée)")
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 📉 Histogrammes de distribution
    plt.figure(figsize=(10, 5))
    plt.hist(proba_raw, bins=30, alpha=0.5, label='Avant calibration')
    plt.hist(proba_cal_platt, bins=30, alpha=0.6, label='Platt scaling')
    plt.hist(proba_cal_iso, bins=30, alpha=0.6, label='Isotonic regression')
    plt.title("Distribution des probabilités (avant / après calibration)")
    plt.xlabel("Probabilité")
    plt.ylabel("Nombre d'occurrences")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return {
        "proba_raw": proba_raw,
        "proba_cal_platt": proba_cal_platt,
        "proba_cal_iso": proba_cal_iso,
        "brier_raw": brier_raw,
        "brier_platt": brier_platt,
        "brier_iso": brier_iso
    }

def reTrain_finalModel_analyse(
        X_train=None, X_train_full=None, X_test=None, X_test_full=None,
        y_train_label_=None, y_test_label_=None, y_pnl_data_train=None, y_pnl_data_test=None,
        nb_SessionTest=None, nan_value=None, feature_names=None,
        config=None, weight_param=None,other_params=None, bestResult_dict=None, is_log_enabled=False,
        scaler_init_dataSet=None,pca_object_init_dataSet=None):

    print_notification('###### DEBUT: ENTRAINEMENT MODELE FINAL ##########', color="blue")
    results_directory = config.get('results_directory', None)

    # Vérification que bestResult_dict contient les bonnes clés
    results_directory = config.get('results_directory', None)
    if results_directory == None:
        exit(25)
    if not bestResult_dict:
        raise ValueError("bestResult_dict ne peut pas être None")
    params_optuna = bestResult_dict["params_optuna"]
    other_params_best = bestResult_dict["other_params"]

    custom_objective_lossFct = config.get('custom_objective_lossFct', None)
    print(custom_objective_lossFct)
    if (custom_objective_lossFct == None):
        exit(13)

    start_time_train, end_time_train, num_sessions_train = get_val_cv_time_range(X_train_full, X_train)
    print(
        f"\nPériode d'entraînement : du {timestamp_to_date_utc(start_time_train)} au {timestamp_to_date_utc(end_time_train)}")
    print(f"Nombre de sessions entraînement : {num_sessions_train}")

    start_time_test, end_time_test, num_sessions_test = get_val_cv_time_range(X_test_full, X_test)
    print(f"Période de test : du {timestamp_to_date_utc(start_time_test)} au {timestamp_to_date_utc(end_time_test)}")
    print(f"Nombre de sessions test : {num_sessions_test}\n")

    model = config['model_type']
    # Sélection du processor de fold selon le framework
    print(f"Model demandé : {model}")
    try:
        fold_processor = select_fold_processor(model)
    except ValueError as e:
        print(f"❌ Erreur lors de la sélection du processor : {e}")
        exit(1)
    print(fold_processor)
    # Mise à jour des métriques et statistiques
    all_fold_stats = {}

    data_4cv={}

    train_pos=None
    val_pos=None
    if (True):
        data_4cv["X_train"] = X_train.copy()
        data_4cv["y_train_label"] = y_train_label_.copy()
        data_4cv["y_pnl_data_train"] = y_pnl_data_train.copy()


        # data_4cv["X_test"] = X_test.copy()
        # data_4cv["y_test_label"] = y_test_label_.copy()
        # data_4cv["y_pnl_data_test"] = y_pnl_data_test.copy()

        data_4cv["X_test"] =None
        data_4cv["y_test_label"] =None
        data_4cv["y_pnl_data_test"] =None
    else:
        # Déterminer l'indice de coupure pour 70%
        split_index = int(0.7 * len(X_train))

        # Split sans mélange (ordre préservé)
        data_4cv["X_train"] = X_train.iloc[:split_index].copy()
        data_4cv["X_test"] = X_train.iloc[split_index:].copy()

        data_4cv["y_train_label"] = y_train_label_.iloc[:split_index].copy()
        data_4cv["y_test_label"] = y_train_label_.iloc[split_index:].copy()

        data_4cv["y_pnl_data_train"] = y_pnl_data_train.iloc[:split_index].copy()
        data_4cv["y_pnl_data_test"] = y_pnl_data_train.iloc[split_index:].copy()

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("Les colonnes de X_train ne correspondent pas à celles de X_test (ordre ou contenu différent)")

        # Création du répertoire si nécessaire
    save_dir = os.path.join(results_directory, 'optuna_results')
    os.makedirs(save_dir, exist_ok=True)

    model_by_fold = bestResult_dict["model_by_fold"]
    X_val_by_fold = bestResult_dict["X_val_by_fold"]

    if model_by_fold is None:
        raise ValueError("❌ ERREUR : model_by_fold est None")
    if X_val_by_fold is None:
        raise ValueError("❌ ERREUR : X_val_by_fold est None")

    best_thresh_fold = bestResult_dict["best_thresh_fold"]
    best_iteration_fold = bestResult_dict["best_iteration_fold"]

    print(f"best_thresh_fold type: {type(best_thresh_fold)}")
    print(f"best_thresh_fold value: {best_thresh_fold}")
    print(f"best_iteration_fold type: {type(best_iteration_fold)}")
    print(f"best_iteration_fold value: {best_iteration_fold}")
    best_iteration_fold = np.array(best_iteration_fold)
    print(f"🎯 Entraînement final → num_boost_round = {int(np.median(best_iteration_fold))} "
          f"(médiane des best_iteration des {len(best_iteration_fold)} folds)")
    other_params['num_boost_round_finalTrain'] = int(np.median(best_iteration_fold))
    final_threshold_median = float(np.median(best_thresh_fold))
    print(f"🎯 Seuil médian retenu après CV : {other_params['threshold']:.4f} "
          f"(sur {len(best_thresh_fold)} folds)")

    # Traitement du fold avec le processor approprié
    try:
        print("➡️ Appel de fold_processor avec les paramètres suivants :")
        # print(f"train_pos: {train_pos}")
        # print(f"val_pos: {val_pos}")
        print(f"params_optuna: {params_optuna}")
        print(f"data keys: {list(data_4cv.keys())}")
        print(f"config keys: {list(config.keys())}")
        print(f"fold_num: {1}")

        fold_results= fold_processor(
            train_pos=train_pos,
            val_pos=val_pos,
            params_optuna=params_optuna,
            other_params=other_params_best,
            data_4cv=data_4cv,
            is_log_enabled=is_log_enabled,
            config=config,
            fold_num=1,
            log_evaluation=1)

        scaler_cv_train_test=fold_results["scaler"]
        pca_model_cv_train_test=fold_results["pca_model"]
        x_train_vifStatPCA_trainFinal_noCV=fold_results["x_train_vifStatPCA_trainFinal_noCV"]
        Y_train_trainFinal_noCV=fold_results["Y_train_trainFinal_noCV"]
        final_model=fold_results['current_model']
        eval_metrics=fold_results["eval_metrics"]
        print(fold_results["evals_result"])

        # ------------------------------------------------------------------
        # >> Vérification PnL sur le train final
        # ------------------------------------------------------------------

        # 0) Alias rapides ───────────────────────────────────────────────
        X_train_final = x_train_vifStatPCA_trainFinal_noCV  # features
        y_train_final = Y_train_trainFinal_noCV  # vraies classes
        scaler = scaler_cv_train_test  # scaler
        model_final = final_model  # booster
        profit_tp = 175
        loss_fp = -227
        penalty_fn = other_params.get('penalty_per_fn', 0.0)  # 0 par défaut
        num_round = other_params['num_boost_round_finalTrain']  # ← nombre d'arbres

        calibrator = fold_results.get('calibrator')  # None si pas de calibration

        print( X_train_final.columns.tolist())
        # Exemple d'utilisation pour l'entraînement
        train_results = evaluate_model(
            X_data=X_train_final,
            y_data=y_train_final,
            model=model_final,
            scaler=scaler,
            calibrator=calibrator,
            num_round=num_round,
            config=config,
            dataset_name="TRAIN",
            profit_tp=profit_tp,
            loss_fp=loss_fp,
            penalty_fn=penalty_fn
        )

        # Enregistrer le seuil sweepé trouvé sur l'entraînement
        best_thresh_train_full_sweep = train_results['best_threshold_sweep']

        data_4cv["X_test"] = X_test.copy()
        data_4cv["y_test_label"] = y_test_label_.copy()
        data_4cv["y_pnl_data_test"] = y_pnl_data_test.copy()

        # Exemple d'utilisation pour le test (en réutilisant le seuil de l'entraînement)
        test_results = evaluate_model(
            X_data=data_4cv["X_test"],
            y_data=data_4cv["y_test_label"],
            model=model_final,
            scaler=scaler_cv_train_test,
            calibrator=calibrator,
            num_round=num_round,
            config=config,
            best_thresh_sweep=best_thresh_train_full_sweep,  # Utiliser le seuil trouvé sur l'entraînement
            final_threshold_median=final_threshold_median,
            dataset_name="TEST",
            profit_tp=profit_tp,
            loss_fp=loss_fp,
            penalty_fn=penalty_fn
        )

        exit(32)


        # 1) Scaling ─────────────────────────────────────────────────────
        if scaler is not None:
            X_scaled = (pd.DataFrame(scaler.transform(X_train_final.values),
                                     columns=X_train_final.columns,
                                     index=X_train_final.index)
                        if isinstance(X_train_final, pd.DataFrame)
                        else scaler.transform(X_train_final))
        else:
            X_scaled = X_train_final

        import xgboost as xgb
        dtrain_final = xgb.DMatrix(X_scaled)


        # 2) Log-odds → proba (tous les arbres) ─────────────────────────
        log_odds_train_full = model_final.predict(
            dtrain_final,
            iteration_range=(0, num_round),
            output_margin=True
        )


        proba_train_full = (calibrator.predict(log_odds_train_full.reshape(-1, 1)).ravel()
                      if calibrator is not None
                      else 1.0 / (1.0 + np.exp(-log_odds_train_full)))

        # 3) Sweep PnL global sur toutes les proba ────────────────────────
        best_thresh_train_full_sweep, _, pnl_curve_full = sweep_threshold_pnl(
            y_train_final, proba_train_full,
            profit_tp=profit_tp, loss_fp=loss_fp,
            penalty_fn=penalty_fn, n_steps=200
        )

        # ➡️ On ne modifie pas params_optuna si on veut comparer — mieux séparer
        # Pour la prod réelle : tu peux décider si tu écrases ou pas le seuil.
        # final_threshold = best_thresh_train_full  (optionnel)

        # 4) Prédiction avec le seuil sweep final ─────────────────────────
        y_pred_sweep = (proba_train_full >= best_thresh_train_full_sweep).astype(int)
        tn, fp, fn, tp = compute_confusion_matrix_cpu(y_train_final, y_pred_sweep, config)

        pnl_sweep = tp * profit_tp + fp * loss_fp + fn * penalty_fn
        trades = tp + fp
        samples = len(y_train_final)
        winrate = 100 * tp / trades if trades else 0.0
        trade_part = 100 * trades / samples if samples else 0.0

        print(f"✅ Vérif entraînement final model → TP: {tp} | FP: {fp} | TN: {tn} | FN: {fn} | "
              f"Winrate: {winrate:.2f}% | PnL: {pnl_sweep:,.0f} | %Trades: {trade_part:.2f}%")
        print(f"🔑 Seuil sweepé sur train complet = {best_thresh_train_full_sweep:.4f} "
              f"| Max PnL observé (sweep) = {pnl_curve_full.max():,.0f} €")
        # 5) Comparaison avec seuil médian CV (final_threshold) ─────────────
        y_pred_median = (proba_train_full >= final_threshold_median).astype(int)
        tn_median, fp_median, fn_median, tp_median = compute_confusion_matrix_cpu(y_train_final, y_pred_median, config)

        pnl_median = tp_median * profit_tp + fp_median * loss_fp + fn_median * penalty_fn
        trades_median = tp_median + fp_median
        samples = len(y_train_final)
        winrate_median = 100 * tp_median / trades_median if trades_median else 0.0
        trade_part_median = 100 * trades_median / samples if samples else 0.0

        print(f"✅ Vérif seuil médian CV → TP: {tp_median} | FP: {fp_median} | TN: {tn_median} | FN: {fn_median} | "
              f"Winrate: {winrate_median:.2f}% | PnL: {pnl_median:,.0f} | %Trades: {trade_part_median:.2f}%")
        print(f"🔑 Seuil médian CV utilisé = {final_threshold_median:.4f}")

        # val_metrics = fold_results["eval_metrics"]
        # tp = val_metrics['tp']
        # fp = val_metrics['fp']
        # tn = val_metrics['tn']
        # fn = val_metrics['fn']
        # # Calculs
        #
        # nb_trades=tp+fp
        # total_samples = tp + fp+tn+fn
        # trade_part = (nb_trades / total_samples) * 100 if total_samples > 0 else 0
        # winrate = (tp / nb_trades) * 100 if nb_trades > 0 else 0
        # pnl = (tp * 175) + (fp * -227)
        # # Affichage
        # print(
        #     f"✅ Vérif entraînement final model → TP: {tp} | FP: {fp} | TN: {tn} | FN: {fn} | Winrate: {winrate:.2f}% | PnL: {pnl:.2f} | %Trades: {trade_part:.2f}%")
        print("Keys in fold_results:", fold_results.keys())
        print("Keys in eval_metrics:", fold_results["eval_metrics"].keys())
        # val_pred_proba_log_odds=fold_results["eval_metrics"]["val_pred_proba_log_odds"]
        # y_val_cv=fold_results["eval_metrics"]["y_val_cv"]
        #
        # val_score_bestIdx=fold_results["val_score_bestIdx"]
        # print("val_score_bestIdx: ",val_score_bestIdx)
        # evals_result = fold_results["evals_result"] #valid_names=['train', 'eval'],
        plot_train_eval_performance(fold_results,save_dir=save_dir, mode="train_only")

        #print("val_pred_proba_log_odds: ",val_pred_proba_log_odds)
    except Exception as e:
        print("❌ Erreur dans process_cv_fold_xgboost :")
        print("Type:", type(e).__name__)
        print("Message:", str(e))
        traceback.print_exc()  # Affiche la stack complète 🔥
        raise  # Optionnel : fait remonter l’erreur au niveau supérieur

    data_4cv["X_test"] = X_test.copy()
    data_4cv["y_test_label"] = y_test_label_.copy()
    data_4cv["y_pnl_data_test"] = y_pnl_data_test.copy()
    X_val_cv_scaled_np = scaler_cv_train_test.transform(data_4cv["X_test"].values)
    X_val_cv_scaled = pd.DataFrame(X_val_cv_scaled_np, columns=data_4cv["X_test"].columns, index=data_4cv["X_test"].index)
    # wrapped_model = BoosterWrapper(final_model, model_type='lgbm')  # ou 'lgbm' si LightGBM

    # log_odds = wrapped_model.predict( data_4cv["X_test"], output_margin=True)
    # proba_raw = expit(log_odds)  # Or 1/(1+np.exp(-log_odds))
    model_type = config['model_type']
    if model_type == modelType.XGB:
        import xgboost as xgb
        dData = xgb.DMatrix(X_val_cv_scaled)
    else:
        dData=X_val_cv_scaled
    # y_test_predProba, test_pred_proba_log_odds, test_pred, (tn_test, fp_test, fn_test, tp_test), Y_test_cv = \
    #     predict_and_compute_metrics_XgbOrLightGbm(model=final_model, X_data=dData, y_true= data_4cv["y_test_label"],
    #                                               best_iteration=val_score_bestIdx+1, threshold=other_params['threshold'],
    #                                               config=config)
    #
    # total_profit, tp, fp = calculate_profitBased(
    #     y_true_class_binaire=data_4cv["y_test_label"],
    #     y_pred_threshold=test_pred,
    #     y_pnl_data_array =data_4cv["y_pnl_data_test"] ,
    #     other_params=other_params,
    #     config=config
    # )
    # 5. Affichage
    print(f"final_model: tp_train: {tp_test} | fp_test: {fp_test} | tn_test: {tn_test} | fn_test: {fn_test} {total_profit} {tp} {fp}")

    res = sweep_threshold_pnl_real(
        model=final_model,
        X=X_val_cv_scaled,
        y_true_class_binaire=data_4cv["y_test_label"],
        y_pnl_data=data_4cv["y_pnl_data_test"],
        config=config,
        val_score_bestIdx=val_score_bestIdx,
        other_params=other_params,
        thresholds=np.arange(0.45, 0.61, 0.005)
    )

    # Affiche les 5 meilleurs seuils selon pnl_total
    for r in res:
        print(f"Seuil: {r['threshold']:.3f} | TP={r['TP']}, FP={r['FP']}, FN={r['FN']}, TN={r['TN']} "
              f"| TP_gains={r['TP_gains']:.2f}, FP_losses={r['FP_losses']:.2f}, "
              f"FN_missed={r['FN_missed']:.2f}, PnL={r['pnl_total']:.2f}")

    if config['model_type'] == modelType.XGB:
        results = compare_calibration_methods(
            final_model,
            X_val_cv_scaled,
            y_val_cv,
            config=config,
            score_bestIdx=val_score_bestIdx,
            other_params=other_params , # ← AJOUT OBLIGATOIRE
        save_dir=save_dir
        )
    elif config['model_type'] == modelType.LGBM:
        results = compare_calibration_methods(
            final_model,
            X_val_cv_scaled,
            y_val_cv,
            config=config,
            score_bestIdx=val_score_bestIdx,
            other_params=other_params,  # ← AJOUT ICI AUSSI
        save_dir=save_dir

        )


    data_4cv=other_params['data_4cv']
    cv=other_params["cv"]
    nb_pca = config.get('nb_pca', 1)

    for fold_num, (train_pos, val_pos) in enumerate(cv.split(X_train)):
        # Validation des indices
        validate_fold_indices(train_pos, val_pos)
        (X_train_cv, X_train_cv_pd, Y_train_cv, X_val_cv, X_val_cv_pd, y_val_cv,
         y_pnl_data_train_cv, y_pnl_data_val_cv) = prepare_dataSplit_cv_train_val(
            config, data_4cv, train_pos, val_pos)

        if list(X_train.columns) != list(X_val_cv.columns):
            raise ValueError(
                "Les colonnes de X_val ne correspondent pas à celles de X_train (ordre ou contenu différent)")

        # 1. Appliquer le scaling aux données de validation
        X_val_cv_scaled_np = scaler_cv_train_test.transform(X_val_cv.values)
        X_val_cv_scaled = pd.DataFrame(X_val_cv_scaled_np, columns=X_val_cv.columns, index=X_val_cv.index)



        if (config['nb_pca']>0):
            high_corr_columns_used4pca = other_params['high_corr_columns_used4pca']
            columns_without_pca = other_params['colomnsList_with_vif_stat_without_pca_']
            # 2. Appliquer la PCA aux données de validation scalées
            pca_val = pca_model_cv_train_test.transform(X_val_cv[high_corr_columns_used4pca])
            pca_cols = [f'pca_{i + 1}' for i in range(nb_pca)]
            print("nb_pca: ",nb_pca)
            print("Shape of pca_val:", pca_val.shape)
            print(pca_cols)
            print("Type of X_val_cv_scaled:", type(X_val_cv_scaled))
            print("Est-ce un DataFrame?", isinstance(X_val_cv_scaled, pd.DataFrame))
            for i, col_name in enumerate(pca_cols):
                X_val_cv_scaled[col_name] = pca_val[:, i]

            X_val_cv_scaled.drop(columns=high_corr_columns_used4pca, inplace=True)
            print("high_corr_columns_used4pca: ",high_corr_columns_used4pca)
        # Juste les noms de colonnes
        print("\n===== DÉMARRAGE DE LA PREDICTION =====")
        print(f"X_val_cv   : {len(X_val_cv_scaled.columns)} colonnes -> {list(X_val_cv_scaled.columns)}")
        print("========================================================================\n")

        start_time_train, end_time_train, _ = get_val_cv_time_range(X_full=X_train_full, X=X_train_cv_pd)
        print("Train dates: de", timestamp_to_date_utc_(start_time_train), "à", timestamp_to_date_utc_(end_time_train))
        start_time_val, end_time_val, _ = get_val_cv_time_range(X_full=X_train_full, X=X_val_cv_pd)
        print("Validation dates: de", timestamp_to_date_utc_(start_time_val), "à", timestamp_to_date_utc_(end_time_val))
        # Prédictions et métriques
        #sample_weights_train, sample_weights_val = compute_sample_weights(Y_train_cv, y_val_cv)
        if (config['model_type'] == modelType.XGB):
            dval = xgb.DMatrix(X_val_cv_scaled, label=y_val_cv, #weight=sample_weights_val
                               )
            val_pred_proba, val_pred_proba_log_odds, val_pred, (tn_val, fp_val, fn_val, tp_val), y_val_cv = \
                predict_and_compute_metrics_XgbOrLightGbm(model=final_model, X_data=dval, y_true=y_val_cv,
                                            best_iteration=final_model.best_iteration,
                                            threshold=other_params['threshold'],
                                            config=config)
        elif (config['model_type'] == modelType.LGBM):
            val_pred_proba, val_pred_proba_log_odds, val_pred, (tn_val, fp_val, fn_val, tp_val), y_val_cv = \
                predict_and_compute_metrics_XgbOrLightGbm(model=final_model, X_data=X_val_cv_scaled, y_true=y_val_cv,
                                            best_iteration=final_model.best_iteration,
                                            threshold=other_params['threshold'],
                                            config=config)
        elif (config['model_type'] == modelType.XGBRF or config['model_type'] == modelType.RF):
            val_pred_proba, val_pred_proba_log_odds, val_pred, tn_val, fp_val, fn_val, tp_val, y_val_converted = predict_and_compute_metrics_RF(
                model=final_model,
                X_data=X_val_cv_scaled,
                y_true=y_val_cv,
                threshold=other_params['threshold'],
                config=config
            )
        # elif (modelType.RF):
        # elif (modelType.RF):
        # elif (modelType.XGBRF):
        # elisf(modelType.SVC):




        # Nombre total d'échantillons
        total_samples = tn_val + fp_val + fn_val + tp_val

        # Nombre de trades pris (prédictions positives)
        total_trades = tp_val + fp_val

        # Winrate (proportion de trades gagnants parmi tous les trades pris)
        winrate = tp_val / total_trades if total_trades > 0 else 0
        winrate_percent = winrate * 100

        # Part des trades pris parmi tous les échantillons
        trade_rate = total_trades / total_samples if total_samples > 0 else 0
        trade_rate_percent = trade_rate * 100

        # Affichage des résultats
        print(f"Nombre total d'échantillons: {total_samples}")
        print(f"Nombre total de trades: {total_trades}")
        print(f"Winrate: {winrate_percent:.2f}%")
        print(f"Part des trades pris: {trade_rate_percent:.2f}%")





    for i, (model, X_val) in enumerate(zip(model_by_fold, X_val_by_fold)):
        status = f"Fold {i} : "
        if model is None:
            status += "❌ modèle manquant | "
        else:
            status += "✅ modèle OK | "

        if X_val is None:
            status += "⚠️ X_val = None"
        elif isinstance(X_val, pd.DataFrame):
            status += f"✅ X_val = {X_val.shape[0]}x{X_val.shape[1]}"
        else:
            status += f"❌ X_val = {type(X_val)}"

        print(status)

    # 0)  Entraînements déjà faits → list[Booster] + list[DataFrame]
    shap_raw = build_shap_table(model_by_fold, X_val_by_fold)

    # 1)  Ajout des métriques
    metrics = add_stability_metrics(shap_raw, top_n=20, add_levene=False)

    # 2)  Sélection (exemple : importance ≥1e‑3, cv ≤0.25, freq_top20 ≥3)
    features_ok, df_selected = select_features(
        metrics,
        min_mean=1e-3,
        max_cv=0.25,
        min_freq_topN=3
    )

    print(f"{len(features_ok)} features retenues →", features_ok)

    # 3)  Export complet + colonne « kept »
    metrics["kept"] = metrics.index.isin(features_ok)
    export_stability_csv(metrics, save_dir=save_dir)


    main_shap_analysis(final_model, X_train=x_train_vifStatPCA_trainFinal_noCV, y_train_label=y_train_label_, X_test=None, y_test_label=None,
                           save_dir=save_dir)

    exit(74)
    # config.update({
    #   'y_pnl_data_train_cv': y_train_label_,
    #   'y_pnl_data_val_cv_OrTest': y_test_label_
    # })
    results = train_and_evaluate_lightgbm_model(
        X_train_cv=X_train,
        X_val_cv=X_test,
        y_train_cv=y_train_label_,
        y_val_cv=y_test_label_,
        y_pnl_data_train_cv=y_pnl_data_train,
        y_pnl_data_val_cv_OrTest=y_pnl_data_test,
        params=params_optuna,
        other_params=other_params_best,
        config=config,
        fold_num=1,
        fold_stats_current=None,
        train_pos=None,
        val_pos=None,
        log_evaluation=5
    )

    final_model = results['current_model']
    evals_result = results['evals_result']
    # Affichage des résultats clés
    print(f"Sur X_Test :")
    print(f"  - Meilleur nombre d'itérations : {final_model.best_iteration}")
    print(f"  - Meilleur score : {final_model.best_score}")
    print(
        f"        - soit PNL {results['eval_metrics']['val_bestVal_custom_metric_pnl']} avec best_titeration de {results['eval_metrics']['best_iteration']} iérations | pour {results['eval_metrics']['total_samples']} samples ")
    print(
        f"        - Rappel sur train: PNL de {results['train_metrics']['train_bestVal_custom_metric_pnl']} | pour {results['train_metrics']['total_samples']}  samples")
    # Création du nom de fichier avec timestamp
    current_time = datetime.now()
    timestamp = current_time.strftime("%y%m%d_%H_%M_%S")  # Format: YYMMDD_HH_MM_SS



    # Sauvegarde du modèle
    model_file = os.path.join(save_dir, f"final_training_model_{timestamp}.json")
    final_model.save_model(model_file)
    print(f"\nModèle sauvegardé: {model_file}")

    # Utilisation de la fonction
    plot_custom_metric_evolution_with_trade_info(final_model, evals_result, n_train_trades=len(X_train),
                                                 n_test_trades=len(X_test), results_directory=results_directory)

    print_notification('###### FIN: ENTRAINEMENT MODELE FINAL ##########', color="blue")

    # Prédiction et évaluation
    print_notification('###### DEBUT: GENERATION PREDICTION AVEC MOBEL FINAL (TEST) ##########', color="blue")

    # Prédictions sur les ensembles d'entraînement et de test
    """
    y_test_predProba = final_model.predict(dtest)
    y_train_predProba = final_model.predict(dtrain)

    # Convertir les prédictions en Cupy et appliquer la transformation sigmoïde
    y_train_predProba = cp.asarray(y_train_predProba)
    y_train_predProba = sigmoidCustom(y_train_predProba)

    y_test_predProba = cp.asarray(y_test_predProba)
    y_test_predProba = sigmoidCustom(y_test_predProba)

    # Appliquer le seuil pour obtenir des classes binaires
    y_test_pred = (y_test_predProba > optimal_optuna_threshold).astype(int)
    y_train_pred = (y_train_predProba > optimal_optuna_threshold).astype(int)

    # Calcul des métriques avec compute_confusion_matrix_cupy
    y_test_gpu = cp.asarray(y_test_label)  # Convertir y_test en Cupy
    tn, fp_conf, fn, tp_conf = compute_confusion_matrix_cupy(y_test_gpu, y_test_pred)

    # Calcul du PnL avec xgb_calculate_profitBased_gpu
    pnl, tp_gpu, fp_gpu = xgb_calculate_profitBased_gpu(
        y_true_gpu=y_test_gpu,
        y_pred_threshold_gpu=y_test_pred,
        other_params=other_params
    )

    # Validation des résultats entre compute_confusion_matrix_cupy et xgb_calculate_profitBased_gpu
    if tp_conf != tp_gpu or fp_conf != fp_gpu:
        raise ValueError(
            f"Incohérence dans les calculs des métriques :\n"
            f"TP (Confusion): {tp_conf}, TP (PnL GPU): {tp_gpu}\n"
            f"FP (Confusion): {fp_conf}, FP (PnL GPU): {fp_gpu}"
        )

    # Affichage des résultats finaux
    print("Prediction sur X_test avec modele finale :")
    print(f"   -Confusion Matrix: TP: {tp_conf}, FP: {fp_conf}, FN: {fn}, TN: {tn}")
    print(f"   -PnL sur X_test : {pnl}")
    # exit(888)
    # Vérification des prédictions après transformation
    min_val = cp.min(y_test_predProba).item()
    max_val = cp.max(y_test_predProba).item()
    # print(f"Plage de valeurs après transformation sigmoïde : [{min_val:.4f}, {max_val:.4f}]")

    print(f"Plage de valeurs : [{min_val:.4f}, {max_val:.4f}]")

    # Vérifier si les valeurs sont dans l'intervalle [0, 1]
    if min_val < 0 or max_val > 1:
        print("ERREUR : Les prédictions ne sont pas dans l'intervalle [0, 1] attendu pour une classification binaire.")
        print("Vous devez appliquer une transformation (comme sigmoid) aux prédictions.")
        print("Exemple : y_pred_proba_afterSig = sigmoidCustom(final_model.predict(dtest))")
    else:
        print("Les prédictions sont dans l'intervalle [0, 1] attendu pour une classification binaire.")
    """
    y_pred_proba_afterSig, pred_proba_log_odds, predictions_converted, (
    tn_xtest, fp_xtest, fn_xtest, tp_xtest), y_test_label_converted = predict_and_compute_metrics_XgbOrLightGbm(
        model=final_model,
        X_data=X_test,
        y_true=y_test_label_,
        best_iteration=final_model.best_iteration,
        threshold=other_params_best['threshold'],
        config=config
    )

    reporting_model_performance(pred_proba_log_odds, tp_xtest, fp_xtest, fn_xtest, tn_xtest, config)

    print_notification('###### FIN: GENERATION PREDICTION AVEC MOBEL FINAL (TEST) ##########', color="blue")

    print_notification('###### DEBUT: ANALYSE DE LA DISTRIBUTION DES PROBABILITÉS PRÉDITES sur (XTEST) ##########',
                       color="blue")

    ###### DEBUT: ANALYSE DE LA DISTRIBUTION DES PROBABILITÉS PRÉDITES sur XTEST ##########

    # Pour la courbe de calibration et l'histogramme
    plot_calibrationCurve_distrib(y_test_label_converted, y_pred_proba_afterSig,
                                  optimal_optuna_threshold=optimal_optuna_threshold,
                                  num_sessions=nb_SessionTest, results_directory=results_directory, config=config)

    # Pour le graphique des taux FP/TP par feature

    import warnings
    y_train_predProba = results['y_train_predProba']

    if 'deltaTimestampOpeningSession1min' in X_test.columns:
        plot_fp_tp_rates(X=X_train, y_label=y_train_label_, y_pred_proba_afterSig=y_train_predProba,
                         feature_deltaTime_name='deltaTimestampOpeningSession1min',
                         optimal_optuna_threshold=optimal_optuna_threshold,
                         dataset_name="train", results_directory=results_directory)
        plot_fp_tp_rates(X=X_test, y_label=y_test_label_, y_pred_proba_afterSig=y_pred_proba_afterSig,
                         feature_deltaTime_name='deltaTimestampOpeningSession1min',
                         optimal_optuna_threshold=optimal_optuna_threshold,
                         dataset_name="test", results_directory=results_directory)


    else:
        warnings.warn(
            "La colonne 'deltaTimestampOpeningSession1min' n'est pas présente dans le jeu de test - Graphique non généré",
            UserWarning)

    print("\nDistribution des probabilités prédites sur XTest:")
    print(f"seuil: {optimal_optuna_threshold}")
    print(f"Min : {y_pred_proba_afterSig.min():.4f}")
    print(f"Max : {y_pred_proba_afterSig.max():.4f}")
    print(f"Moyenne : {y_pred_proba_afterSig.mean():.4f}")
    print(f"Médiane : {np.median(y_pred_proba_afterSig):.4f}")

    # Compter le nombre de prédictions dans différentes plages de probabilité
    # Définir le pas pour les intervalles en dessous de optimal_optuna_threshold
    step_below = 0.1  # Vous pouvez ajuster ce pas selon vos besoins

    # Créer les intervalles en dessous de optimal_optuna_threshold
    ranges_below = np.arange(0, optimal_optuna_threshold, step_below)
    ranges_below = np.append(ranges_below, optimal_optuna_threshold)

    # Définir le pas pour les intervalles au-dessus de optimal_optuna_threshold
    step_above = 0.02  # Taille des intervalles souhaitée au-dessus du seuil

    # Calculer le prochain multiple de step_above au-dessus de optimal_optuna_threshold
    next_multiple = np.ceil(optimal_optuna_threshold / step_above) * step_above

    # Créer les intervalles au-dessus de optimal_optuna_threshold
    ranges_above = np.arange(next_multiple, 1.0001, step_above)

    # Combiner les intervalles
    ranges = np.concatenate((ranges_below, ranges_above))
    ranges = np.unique(ranges)  # Supprimer les doublons et trier

    # Maintenant, vous pouvez utiliser ces ranges pour votre histogramme
    hist, _ = np.histogram(y_pred_proba_afterSig, bins=ranges)

    # Convertir les tableaux CuPy en NumPy si nécessaire
    if config['device_'] == 'cuda':
        import cupy as cp
        # Conversion si nécessaire
        y_test_predProba_np = cp.asnumpy(y_pred_proba_afterSig) if isinstance(y_pred_proba_afterSig,
                                                                              cp.ndarray) else y_pred_proba_afterSig
        y_test_label_np = cp.asnumpy(y_test_label_) if isinstance(y_test_label_, cp.ndarray) else y_test_label_
    else:
        # Sur CPU, y_pred_proba_afterSig et y_test_label_ sont déjà en NumPy
        y_test_predProba_np = y_pred_proba_afterSig
        y_test_label_np = y_test_label_

    print("\nDistribution des probabilités prédites avec TP et FP sur XTest:")
    cum_tp = 0
    cum_fp = 0
    for i in range(len(ranges) - 1):
        mask = (y_test_predProba_np >= ranges[i]) & (y_test_predProba_np < ranges[i + 1])
        predictions_in_range = y_test_predProba_np[mask]
        true_values_in_range = y_test_label_np[mask]
        tp = np.sum((predictions_in_range >= optimal_optuna_threshold) & (true_values_in_range == 1))
        fp = np.sum((predictions_in_range >= optimal_optuna_threshold) & (true_values_in_range == 0))
        total_trades_val = tp + fp
        win_rate = tp / total_trades_val * 100 if total_trades_val > 0 else 0
        cum_tp = cum_tp + tp
        cum_fp = cum_fp + fp
        print(
            f"Probabilité {ranges[i]:.2f} - {ranges[i + 1]:.2f} : {hist[i]} prédictions, TP: {tp}, FP: {fp}, Winrate: {win_rate:.2f}%")

    total_trades_cum = cum_tp + cum_fp
    Winrate = cum_tp / total_trades_cum * 100 if total_trades_cum > 0 else 0
    print(f"=> Test final: X_test avec model final optimisé : TP: {cum_tp}, FP: {cum_fp}, Winrate: {Winrate:.2f}%")
    # On prend la valeur 'max' pour chaque paramètre
    profit_value = float(config['profit_per_tp'])
    loss_value = float(config['loss_per_fp'])

    print(f"  - PNL : {cum_tp * profit_value + cum_fp * loss_value:.2f}")

    print("Statistiques de y_pred_proba:")
    print(f"Nombre d'éléments: {len(y_pred_proba_afterSig)}")
    print(f"Min: {np.min(y_pred_proba_afterSig)}")
    print(f"Max: {np.max(y_pred_proba_afterSig)}")
    print(f"Valeurs uniques: {np.unique(y_pred_proba_afterSig)}")
    print(f"Y a-t-il des NaN?: {np.isnan(y_pred_proba_afterSig).any()}")

    # Définissez min_precision si vous voulez l'utiliser, sinon laissez-le à None
    min_precision = None  # ou une valeur comme 0.7 si vous voulez l'utiliser

    # Création de la figure avec trois sous-graphiques côte à côte
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 8))

    # Sous-graphique 1 : Courbe ROC

    # Convertir les tableaux CuPy en NumPy
    # y_test_label_np = cp.asnumpy(y_test_label) if isinstance(y_test_label, cp.ndarray) else y_test_label
    # y_test_predProba_np = cp.asnumpy(y_pred_proba_afterSig) if isinstance(y_pred_proba_afterSig, cp.ndarray) else y_pred_proba_afterSig

    # Calculer la courbe ROC et le score AUC
    fpr, tpr, _ = roc_curve(y_test_label_np, y_test_predProba_np)
    auc_score = roc_auc_score(y_test_label_np, y_test_predProba_np)

    ax1.plot(fpr, tpr, color='blue', linestyle='-', linewidth=2, label=f'ROC Curve (AUC = {auc_score:.2f})')
    ax1.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=2)
    ax1.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.grid(True)
    ax1.legend(loc='lower right', fontsize=10)

    # Sous-graphique 2 : Distribution des probabilités prédites
    bins = np.linspace(y_pred_proba_afterSig.min(), y_pred_proba_afterSig.max(), 100)

    # Assurez-vous que y_pred_proba_afterSig est en NumPy

    # Conversion de y_pred_proba_afterSig en NumPy
    #    y_test_predProba_np = cp.asnumpy(y_pred_proba_afterSig) if isinstance(y_pred_proba_afterSig, cp.ndarray) else y_pred_proba_afterSig

    # Assurez-vous que optimal_optuna_threshold est un scalaire Python
    optimal_optuna_threshold = float(optimal_optuna_threshold)

    # Créez les masques pour les valeurs au-dessus et en dessous du seuil
    mask_below = y_test_predProba_np <= optimal_optuna_threshold
    mask_above = y_test_predProba_np > optimal_optuna_threshold

    # Créez bins comme un tableau NumPy
    bins = np.linspace(np.min(y_test_predProba_np), np.max(y_test_predProba_np), 100)

    # Utilisez ces masques avec y_test_predProba_np pour l'histogramme
    ax2.hist(y_test_predProba_np[mask_below], bins=bins, color='orange',
             label=f'Prédictions ≤ {optimal_optuna_threshold:.4f}', alpha=0.7)
    ax2.hist(y_test_predProba_np[mask_above], bins=bins, color='blue',
             label=f'Prédictions > {optimal_optuna_threshold:.4f}', alpha=0.7)

    ax2.axvline(x=optimal_optuna_threshold, color='r', linestyle='--',
                label=f'Seuil de décision ({optimal_optuna_threshold:.4f})')
    ax2.set_title('Proportion de prédictions négatives (fonction du choix du seuil) sur XTest', fontsize=14,
                  fontweight='bold')
    ax2.set_xlabel('Proportion de prédictions négatives (fonction du choix du seuil)', fontsize=12)
    ax2.set_ylabel('Nombre de prédictions', fontsize=12)

    # Ajout des annotations pour les comptes

    # Convertir y_pred_proba_afterSig en NumPy si c'est un tableau CuPy
    #    y_test_predProba_np = cp.asnumpy(y_pred_proba_afterSig) if isinstance(y_pred_proba_afterSig, cp.ndarray) else y_pred_proba_afterSig

    # Utiliser la version NumPy pour les calculs
    num_below = np.sum(y_test_predProba_np <= optimal_optuna_threshold)
    num_above = np.sum(y_test_predProba_np > optimal_optuna_threshold)

    ax2.text(0.05, 0.95, f'Count ≤ {optimal_optuna_threshold:.4f}: {num_below}', color='orange',
             transform=ax2.transAxes,
             va='top')
    ax2.text(0.05, 0.90, f'Count > {optimal_optuna_threshold:.4f}: {num_above}', color='blue', transform=ax2.transAxes,
             va='top')

    ax2.legend(fontsize=10)

    def to_numpy(arr):
        """
        Convertit différents types de tableaux en tableau NumPy.
        La conversion dépend du device configuré (CUDA ou CPU).

        Args:
            arr: Un tableau qui peut être de type CuPy, NumPy ou autre type convertible

        Returns:
            numpy.ndarray: Le tableau converti en format NumPy
        """
        if config['device_'] == 'cuda':
            import cupy as cp
            if isinstance(arr, cp.ndarray):
                return arr.get()

        # Si c'est déjà un tableau NumPy, le retourner tel quel
        if isinstance(arr, np.ndarray):
            return arr

        # Pour tout autre type, essayer de le convertir en tableau NumPy
        return np.array(arr)

    # Convertir y_test_label et y_pred_proba_afterSig en tableaux NumPy
    y_test_label_np = to_numpy(y_test_label_)
    y_test_predProba_np = to_numpy(y_pred_proba_afterSig)

    # Sous-graphique 3 : Courbe Precision-Recall
    precision, recall, _ = precision_recall_curve(y_test_label_np, y_test_predProba_np)
    ax3.plot(recall, precision, color='green', marker='.', linestyle='-', linewidth=2)
    ax3.set_title('Courbe Precision-Recall', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Recall (Taux de TP)', fontsize=12)
    ax3.set_ylabel('Precision (1 - Taux de FP)', fontsize=12)
    ax3.grid(True)

    # Ajout de la ligne de précision minimale si définie
    if min_precision is not None:
        ax3.axhline(y=min_precision, color='r', linestyle='--', label=f'Précision minimale ({min_precision:.2f})')
        ax3.legend(fontsize=10)

    # Ajustement de la mise en page
    plt.tight_layout()

    # Sauvegarde et affichage du graphique
    plt.savefig(os.path.join(results_directory, 'roc_distribution_precision_recall_combined.png'), dpi=300,
                bbox_inches='tight')

    # Afficher ou fermer la figure selon l'entrée de l'utilisateur

    plt.close()  # Fermer après l'affichage ou sans affichage

    analyze_thresholds(
        y_test_label_,
        y_pred_proba_afterSig,
        other_params_best,
        thresholds=np.arange(0.0, 1.01, 0.001),
        min_winrate=config['config_constraint_winrates_val_by_fold'],
        min_trades=config['config_constraint_min_trades_threshold_by_Fold']
    )

    print_notification('###### FIN: ANALYSE DE LA DISTRIBUTION DES PROBABILITÉS PRÉDITES sur (XTEST) ##########',
                       color="blue")

    print_notification('###### DEBUT: ANALYSE DES DEPENDENCES SHAP DU MOBEL FINAL (ENTRAINEMENT) ##########',
                       color="blue")
    importance_df, shap_comparison, shap_values_train, shap_values_test, resulat_test_shap_feature_importance = main_shap_analysis(
        final_model, X_train, y_train_label_, X_test, y_test_label_,
        save_dir=os.path.join(results_directory, 'shap_dependencies_results_reTrainModel'))
    print_notification('###### FIN: ANALYSE DES DEPENDENCES SHAP DU MOBEL FINAL (ENTRAINEMENT) ##########',
                       color="blue")

    # print_notification('###### DEBUT: ANALYSE DE L\'IMPACT DES VALEURS NaN DU MOBEL FINAL (ENTRAINEMENT) ##########',
    #                    color="blue")

    # Appeler la fonction d'analyse
    # analyze_nan_impact(model=final_model, X_train=X_train, feature_names=feature_names,
    #                  shap_values=shap_values_train, nan_value=nan_value,
    #                  save_dir=os.path.join(results_directory, 'nan_analysis_results'))

    # print_notification('###### FIN: ANALYSE DE L\'IMPACT DES VALEURS NaN DU MOBEL FINAL (ENTRAINEMENT) ##########',
    #                  color="blue")

    ###### DEBUT: ANALYSE DES ERREURS ##########
    print_notification('###### DEBUT: ANALYSE DES ERREURS ##########', color="blue")
    # Analyse des erreurs

    results_df, error_df = analyze_errors(X_test, y_test_label_, predictions_converted, y_pred_proba_afterSig,
                                          feature_names,
                                          save_dir=os.path.join(results_directory, 'analyse_error'),
                                          top_features=resulat_test_shap_feature_importance['top_10_features'])

    print_notification('###### FIN: ANALYSE DES ERREURS ##########', color="blue")
    ###### FIN: ANALYSE DES ERREURS ##########

    ###### DEBUT: ANALYSE DES ERREURS LES PLUS CONFIANTES ##########
    print_notification('###### DEBUT: ANALYSE DES ERREURS LES PLUS CONFIANTES ##########', color="blue")

    # Exemple d'utilisation :
    analyze_predictions_by_range(X_test, y_pred_proba_afterSig, shap_values_test, prob_min=0.5, prob_max=1.00,
                                 top_n_features=20, output_dir=results_directory)

    feature_importance = np.abs(shap_values_test).mean(axis=0)
    feature_importance_df = pd.DataFrame({
        'feature': X_test.columns,
        'importance': feature_importance
    })

    # 1. Identifier les erreurs les plus confiantes
    errors = results_df[results_df['true_label'] != results_df['predicted_label']]
    confident_errors = errors.sort_values('prediction_probability', ascending=False)

    # 2. Récupérer les features importantes à partir de l'analyse SHAP
    important_features = feature_importance_df['feature'].head(20).tolist()

    print("Visualisation des erreurs confiantes:")
    plot_confident_errors(
        shap_values_test,
        confident_errors=confident_errors,
        X_test=X_test,
        feature_names=feature_names,
        n=5, results_directory=results_directory)
    # plot_confident_errors(xgb_classifier, confident_errors, X_test, X_test.columns,explainer_Test)

    # Exécution des analyses
    print("\nAnalyse des erreurs confiantes:")

    analyze_confident_errors(shap_values_test, confident_errors=confident_errors, X_test=X_test,
                             feature_names=feature_names, important_features=important_features, n=5)
    correct_predictions = results_df[results_df['true_label'] == results_df['predicted_label']]
    print("\nComparaison des erreurs vs prédictions correctes:")
    compare_errors_vs_correct(confident_errors.head(30), correct_predictions, X_test, important_features,
                              results_directory)
    print("\nAnalyse SHAP terminée. Les visualisations ont été sauvegardées.")
    print("\nAnalyse terminée. Les visualisations ont été sauvegardées.")
    print_notification('###### FIN: ANALYSE DES ERREURS LES PLUS CONFIANTES ##########', color="blue")
    ###### FIN: ANALYSE DES ERREURS LES PLUS CONFIANTES ##########

    ###### DEBUT: CALCUL DES CORRELACTION ##########

    # Call the function
    high_corr_pairs = analyze_and_save_feature_correlations(X_train, results_directory, threshold=0.75)
    # Si high_corr_pairs est une liste de tuples (feature_i, feature_j, corr_value)
    ###### FIN: CALCUL DES CORRELACTION ##########

    ###### DEBUT: CALCUL DES VALEURS D'INTERACTION SHAP ##########
    print_notification("###### DEBUT: CALCUL DES VALEURS D'INTERACTION SHAP ##########", color="blue")
    analyze_shap_interactions(final_model, X_train, results_directory,
                              dataset_name='X_train')  ## utile pour filtrer les données
    analyze_shap_interactions(final_model, X_test, results_directory,
                              dataset_name='X_test')  ## utile pour verfiier la pertinance, stabilité et donc l'overfitting (variable qui bouge etc)

    print_notification("###### FIN: CALCUL DES VALEURS D'INTERACTION SHAP ##########", color="blue")
    ###### FIN: CALCUL DES VALEURS D'INTERACTION SHAP ##########

    return True


from sklearn.metrics import make_scorer
import numpy as np

"""
def custom_profit_scorer(y_true, y_pred_proba, other_params=None, normalize=False):

    if other_params is None:
        other_params = {}

    CHECK_THRESHOLD = 0.55555555
    threshold = other_params.get('threshold', CHECK_THRESHOLD)
    if threshold == CHECK_THRESHOLD:
        raise ValueError("Invalid threshold value detected")
    # Suppression de l'application de la sigmoïde
    # y_pred_proba = 1 / (1 + np.exp(-y_pred_proba))
    # y_pred_proba = np.clip(y_pred_proba, 0.0, 1.0)

    # Vérification des valeurs
    min_val = np.min(y_pred_proba)
    max_val = np.max(y_pred_proba)
    if min_val < 0 or max_val > 1:
        # return float('-inf')  # Retourne une valeur très basse en cas de problème
        raise ValueError(f"Probabilities out of bounds: min={min_val}, max={max_val}")

    # Conversion en prédictions binaires
    y_pred = (y_pred_proba > threshold).astype(int)

    # Calcul des métriques
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    # Calcul du profit
    profit_per_tp = other_params.get('profit_per_tp', 1.0)
    loss_per_fp = other_params.get('loss_per_fp', -1.1)
    penalty_per_fn = other_params.get('penalty_per_fn', 0)

    total_profit = (tp * profit_per_tp +
                    fp * loss_per_fp +
                    fn * penalty_per_fn)

    if normalize:
        total_trades = tp + fp
        if total_trades > 0:
            return total_profit / total_trades
        return 0.0

    return total_profit
"""

import numpy as np
from numba import jit


@jit(nopython=True)
def create_mask_numba(timestamps, periods_starts, periods_ends):
    """
    Crée un masque avec Numba pour les périodes sélectionnées
    """
    mask = np.zeros(len(timestamps), dtype=np.bool_)
    for i in range(len(timestamps)):
        for j in range(len(periods_starts)):
            if periods_starts[j] <= timestamps[i] < periods_ends[j]:
                mask[i] = True
                break
    return mask


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

"""
ANALYSE DES DONNÉES DE TRADING
Format timestamp spécifique : 0-1380
0-239.999 : 22h-23h59.999
240-1380 : 00h-20h59.999
"""


def analyze_trade_distribution(df_original, df_filtered, time_periods_dict):
    """
    Analyse améliorée de la distribution temporelle des trades
    """
    # Configuration du style
    plt.style.use('default')

    # Création de la figure
    fig = plt.figure(figsize=(20, 12), facecolor='white')
    gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=0.3)

    def prepare_data(data_df):
        df_copy = data_df.copy()
        # Regroupement par intervalles de 10 minutes
        df_copy['time_bin'] = (df_copy['deltaTimestampOpeningSession1min'] // 10).astype(int) * 10
        df_grouped = df_copy.groupby('time_bin')['class_binaire'].value_counts().unstack(fill_value=0)

        # S'assurer que l'index est complet de 0 à 1380 par pas de 10
        full_index = pd.Index(range(0, 1381, 10))
        df_grouped = df_grouped.reindex(full_index, fill_value=0)

        # Extraction des succès et échecs
        successes = df_grouped.get(1, pd.Series(0, index=df_grouped.index))
        failures = df_grouped.get(0, pd.Series(0, index=df_grouped.index))

        return df_grouped, successes, failures

    def format_time_label(minutes):
        """Convertit les minutes depuis 22:00 en format heure:minute"""
        total_minutes = (minutes + 22 * 60) % (24 * 60)
        hour = total_minutes // 60
        minute = total_minutes % 60
        return f"{int(hour):02d}:{int(minute):02d}"

    # Préparation des données pour les deux graphiques
    df_trades_original = df_original[df_original['class_binaire'].isin([0, 1])].copy()
    df_grouped_original, successes_original, failures_original = prepare_data(df_trades_original)

    df_trades_filtered = df_filtered[df_filtered['class_binaire'].isin([0, 1])].copy()
    df_grouped_filtered, successes_filtered, failures_filtered = prepare_data(df_trades_filtered)

    # Positions x pour les barres
    x_positions = range(len(df_grouped_original))
    width = 0.8

    # 1. Premier graphique - Tous les trades
    ax1 = fig.add_subplot(gs[0])
    ax1.bar(x_positions, successes_original, width, color='#2ecc71', alpha=0.7)  # Supprimé label='Succès'
    ax1.bar(x_positions, failures_original, width, bottom=successes_original, color='#e74c3c',
            alpha=0.7)  # Supprimé label='Échecs'
    ax1.set_title('Distribution horaire des trades', fontsize=14, pad=15)
    ax1.set_ylabel('Nombre de trades', fontsize=12)
    # Supprimé : ax1.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none', fontsize=12)

    # 2. Deuxième graphique - Trades filtrés
    ax2 = fig.add_subplot(gs[1])
    ax2.bar(x_positions, successes_filtered, width, color='#2ecc71', alpha=0.7)  # Supprimé label='Succès'
    ax2.bar(x_positions, failures_filtered, width, bottom=successes_filtered, color='#e74c3c',
            alpha=0.7)  # Supprimé label='Échecs'
    ax2.set_title('Distribution des trades - Périodes sélectionnées uniquement', fontsize=14, pad=15)
    ax2.set_ylabel('Nombre de trades', fontsize=12)
    # Supprimé : ax2.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none', fontsize=12)

    # Configuration commune des axes
    for ax in [ax1, ax2]:
        ax.grid(True, linestyle='--', alpha=0.5)

        # Labels d'axe toutes les 30 minutes (3 bins de 10 minutes)
        xticks_pos = list(range(0, len(df_grouped_original), 3))
        xticks_labels = [format_time_label(df_grouped_original.index[pos]) for pos in xticks_pos]
        ax.set_xticks(xticks_pos)
        ax.set_xticklabels(xticks_labels, rotation=90, fontsize=8)
        ax.set_xlim(-0.5, len(df_grouped_original) - 0.5)

        # Ligne de minuit
        # midnight_idx = 120 // 10  # 120 minutes après 22h, divisé par la taille du bin (10 minutes)
        # ax.axvline(x=midnight_idx, color='red', linestyle='--', alpha=0.7)

        ymin, ymax = ax.get_ylim()

        # Coloration des zones (toutes les périodes, qu'elles soient sélectionnées ou non)
        for name, info in time_periods_dict.items():
            # Les indices correspondent aux bins de 10 minutes
            start_idx = info['start'] // 10
            end_idx = info['end'] // 10

            color = f"#{format(hash(name) % 0xffffff, '06x')}"
            alpha = 0.45 if info['selected'] else 0.25  # Alpha différent selon 'selected'

            # Coloration de la zone
            ax.axvspan(start_idx, end_idx, alpha=alpha, color=color)

            # Label centré sur la zone
            center = start_idx + (end_idx - start_idx) / 2
            ax.text(center, ymax * 0.9,
                    f"{name}\n{format_time_label(info['start'])}-{format_time_label(info['end'])}",
                    rotation=90, va='center', ha='center', fontsize=8, color='black')

    # Calcul des statistiques
    total_success = successes_filtered.sum()
    total_failure = failures_filtered.sum()
    total_all = total_success + total_failure
    # Calcul des statistiques globales (avant sélection)
    total_success_original = successes_original.sum()
    total_failure_original = failures_original.sum()
    total_all_original = total_success_original + total_failure_original
    global_win_rate_original = (total_success_original / total_all_original * 100) if total_all_original > 0 else 0

    # Texte des statistiques
    if total_all > 0:
        stats = []
        global_win_rate = (total_success / total_all * 100)

        for name, info in time_periods_dict.items():
            if info['selected']:
                start_idx = info['start'] // 10
                end_idx = info['end'] // 10

                period_successes = successes_filtered.iloc[start_idx:end_idx].sum()
                period_failures = failures_filtered.iloc[start_idx:end_idx].sum()
                period_total = period_successes + period_failures

                if period_total > 0:
                    period_win_rate = (period_successes / period_total * 100)
                    stats.append(
                        f"{name} ({format_time_label(info['start'])}-{format_time_label(info['end'])}):\n"
                        f"  Trades: {period_total} ({period_total / total_all * 100:.1f}%)\n"
                        f"  Win Rate: {period_win_rate:.1f}%"
                    )

        stats_text = (
                f"Statistiques Globales avant sélection:\n"
                f" Total Trades: {total_all_original}\n"
                f" Succès: {total_success_original} ({total_success_original / total_all_original * 100:.1f}%)\n"
                f" Échecs: {total_failure_original} ({total_failure_original / total_all_original * 100:.1f}%)\n"
                f" Win Rate Global: {global_win_rate_original:.1f}%\n\n"
                f"Selection : statistiques Globales:\n"
                f" Total Trades: {total_all}\n"
                f" Succès: {total_success} ({total_success / total_all * 100:.1f}%)\n"
                f" Échecs: {total_failure} ({total_failure / total_all * 100:.1f}%)\n"
                f" Win Rate Global: {global_win_rate:.1f}%\n\n"
                + "\n\n".join(stats)
        )
    else:
        stats_text = "Aucun trade dans la période sélectionnée"

    # Affichage des statistiques
    plt.figtext(1.02, 0.6, stats_text, fontsize=12,
                bbox=dict(facecolor='white', edgecolor='gray', alpha=0.9, pad=10))

    plt.suptitle('Analyse de la Distribution des Trades sur 24H', fontsize=16, y=0.97)
    plt.subplots_adjust(left=0.1, right=0.85, top=0.9, bottom=0.2, hspace=0.4)

    # Retourner à la fois la figure et les statistiques
    stats = {
        'total_trades': total_all,
        'successes': total_success,
        'failures': total_failure,
        'win_rate': global_win_rate,
        'hourly_stats': {
            'timestamps': df_grouped_filtered.index.tolist(),
            'volumes': (successes_filtered + failures_filtered).tolist()
        }
    }

    return fig, stats


import warnings


def print_warning(message):
    print(f"\033[91m/!\ Warning: {message}\033[0m")


def sessions_selection(df, selected_sessions=None, CUSTOM_SESSIONS_=None, results_directory=None):
    """
    Met class_binaire à 99 pour les sessions NON sélectionnées et analyse la distribution.
    """
    if results_directory is None:
        print_warning("Le répertoire 'results_directory' est None")

    # Comptage AVANT filtrage
    original_success_count = len(df[df['class_binaire'] == 1])
    original_failure_count = len(df[df['class_binaire'] == 0])
    original_total_count = original_success_count + original_failure_count
    original_win_rate = (original_success_count / original_total_count * 100) if original_total_count > 0 else 0

    print("\n" + "=" * 50)
    print("STATISTIQUES AVANT FILTRAGE PAR SESSIONS")
    print("=" * 50)
    print(f"Total trades dans le dataset original: {original_total_count}")
    print(f"Trades réussis (1): {original_success_count}")
    print(f"Trades échoués (0): {original_failure_count}")
    print(f"Win Rate: {original_win_rate:.2f}%")
    print("=" * 50 + "\n")

    df_filtered = df.copy()
    timestamps = df['deltaTimestampOpening'].values
    print(f"Dimensions du DataFrame original: {df_filtered.shape}")

    if CUSTOM_SESSIONS_ is not None:
        # Sélection des périodes activées
        selected_periods = [
            (info['start'], info['end'])
            for info in CUSTOM_SESSIONS_.values()
            if info['selected']
        ]

        periods_starts = np.array([start for start, end in selected_periods], dtype=np.float64)
        periods_ends = np.array([end for start, end in selected_periods], dtype=np.float64)

        # Logging et calcul des statistiques par période
        cum_total_count = 0
        print("\nUtilisation des périodes du dictionnaire:")

        for name, info in CUSTOM_SESSIONS_.items():
            status = "activée" if info['selected'] else "désactivée"
            start_time = format_time(info['start'])
            end_time = format_time(info['end'])
            print(f"- {name}: {info['start']} à {info['end']} ({start_time} - {end_time}) ({status})")

            if info['selected']:
                # Calcul des statistiques pour cette période
                mask_period = (timestamps >= info['start']) & (timestamps < info['end'])
                period_data = df[mask_period]

                success_count = len(period_data[period_data['class_binaire'] == 1])
                failure_count = len(period_data[period_data['class_binaire'] == 0])
                total_count = success_count + failure_count
                win_rate = (success_count / total_count * 100) if total_count > 0 else 0
                cum_total_count += total_count

                print(f"  Statistiques pour {name}:")
                print(f"  - Minutes depuis 22h: {info['start']} - {info['end']}")
                print(f"  - Heures: {start_time} - {end_time}")
                print(f"  - Trades réussis: {success_count}")
                print(f"  - Trades échoués: {failure_count}")
                print(f"  - Total trades: {total_count}")
                print(f"  - Win Rate: {win_rate:.2f}%\n")

    elif CUSTOM_SESSIONS_ is not None:
        valid_sections = [s for s in CUSTOM_SESSIONS_ if s['name'] in selected_sessions]
        periods_starts = np.array([s['start'] for s in valid_sections], dtype=np.float64)
        periods_ends = np.array([s['end'] for s in valid_sections], dtype=np.float64)

        print("\nPériodes sélectionnées:")
        for section in valid_sections:
            start_time = format_time(section['start'])
            end_time = format_time(section['end'])
            print(f"- {section['name']}: {section['start']} à {section['end']} ({start_time} - {end_time})")
    else:
        return df_filtered

    print(f"\ncum_total_count des sessions = {cum_total_count}")

    # Créer le masque avec Numba et appliquer le filtrage
    mask_selected = create_mask_numba(timestamps, periods_starts, periods_ends)
    df_filtered.loc[~mask_selected, 'class_binaire'] = 99

    # Après avoir appliqué le masque de filtrage
    print("\n" + "=" * 50)
    print("STATISTIQUES APRÈS FILTRAGE PAR SESSIONS")
    print("=" * 50)
    success_filtered = len(df_filtered[df_filtered['class_binaire'] == 1])
    failure_filtered = len(df_filtered[df_filtered['class_binaire'] == 0])
    filtered_99 = len(df_filtered[df_filtered['class_binaire'] == 99])
    total_filtered = success_filtered + failure_filtered
    filtered_win_rate = (success_filtered / total_filtered * 100) if total_filtered > 0 else 0

    print(
        f"Trades réussis (1): {success_filtered} (avant: {original_success_count}, différence: {success_filtered - original_success_count})")
    print(
        f"Trades échoués (0): {failure_filtered} (avant: {original_failure_count}, différence: {failure_filtered - original_failure_count})")
    print(f"Trades filtrés (99): {filtered_99}")
    print(f"Total trades après filtrage: {total_filtered} (avant: {original_total_count})")
    print(f"Win Rate après filtrage: {filtered_win_rate:.2f}% (avant: {original_win_rate:.2f}%)")
    print(f"Pourcentage de données conservées: {(total_filtered / original_total_count * 100):.2f}%")
    print("=" * 50)

    # Vérification de cohérence
    if total_filtered != cum_total_count:
        print("\nATTENTION: Incohérence détectée!")
        print(f"Total des trades filtrés ({total_filtered}) != ")
        print(f"Somme des trades par période ({cum_total_count})")
        exit(111)
    # Analyser et visualiser la distribution
    fig, stats = analyze_trade_distribution(df, df_filtered, CUSTOM_SESSIONS_)

    print("\nRésumé des statistiques:")
    print(f"Total trades analysés: {stats['total_trades']}")
    print(f"Win Rate global: {stats['win_rate']:.1f}%")
    if results_directory is not None:
        plt.savefig(os.path.join(results_directory, 'initial_dataset_and_select_sessions.png'), dpi=300,
                    bbox_inches='tight')

    plt.show(block=False)
    plt.pause(2)  # Pause de 2 secondes
    print(df_filtered.shape)
    return df_filtered


def format_time(minutes):
    """Convertit les minutes depuis 22:00 en format heure:minute"""
    total_minutes = (minutes + 22 * 60) % (24 * 60)
    hour = total_minutes // 60
    minute = total_minutes % 60
    return f"{int(hour):02d}:{int(minute):02d}"


def calculate_global_ecart(
        tp_train_list, fp_train_list,
        tp_val_list, fp_val_list,
        use_trade_weighting=False
):
    """
    Calcule l'écart global entre les ratios de winrate (validation / train) sur plusieurs folds.

    Args:
        tp_train_list (list): Liste des true positives (TP) pour les données d'entraînement sur chaque fold.
        fp_train_list (list): Liste des false positives (FP) pour les données d'entraînement sur chaque fold.
        tp_val_list (list): Liste des true positives (TP) pour les données de validation sur chaque fold.
        fp_val_list (list): Liste des false positives (FP) pour les données de validation sur chaque fold.
        use_trade_weighting (bool): Si True, pondère l'écart global par le nombre de trades dans chaque fold.

    Returns:
        float: L'écart global entre les ratios validation/train, pondéré ou non.
    """
    ecarts = []  # Liste des écarts par fold
    poids = []  # Liste des poids associés à chaque fold (nombre de trades)

    # Calcul des écarts et des poids pour chaque fold
    for i in range(len(tp_train_list)):
        # Calcul du nombre total de trades (train et validation)
        train_trades = tp_train_list[i] + fp_train_list[i]
        val_trades = tp_val_list[i] + fp_val_list[i]

        # Éviter la division par zéro en remplaçant par une très petite valeur
        train_trades = max(train_trades, 1e-8)
        val_trades = max(val_trades, 1e-8)

        # Calcul des winrates
        train_winrate = tp_train_list[i] / train_trades
        val_winrate = tp_val_list[i] / val_trades

        # Calcul du ratio validation/train et de l'écart par rapport à 1
        ratio = val_winrate / max(train_winrate, 1e-8)
        ecart = abs(ratio - 1.0)

        # Ajouter l'écart et le poids (nombre de trades en validation) aux listes
        ecarts.append(ecart)
        poids.append(val_trades)  # On peut utiliser une autre définition de poids si nécessaire

    # Si aucun fold n'est présent, retourner un écart infini
    if len(ecarts) == 0:
        return float('inf')

    if use_trade_weighting:
        # Moyenne pondérée par le nombre de trades
        poids_total = sum(poids)
        if poids_total > 0:
            ecart_global = sum(e * p for e, p in zip(ecarts, poids)) / poids_total
        else:
            # Si tous les poids sont nuls, retourner un écart infini
            ecart_global = float('inf')
    else:
        # Moyenne simple des écarts
        ecart_global = sum(ecarts) / len(ecarts)

    return ecart_global


def calculate_normalized_pnl_objectives(trial,
                                        tp_train_list, fp_train_list,
                                        tp_val_list, fp_val_list,
                                        scores_train_list, scores_val_list, winrates_val_by_fold,
                                        brier_val_list=None,
                                        relative_brier_val_list=None,
                                        config=None,
                                        use_trade_weighting_for_ecart=True
                                        ):
    import math
    import numpy as np

    lengths = [len(tp_train_list), len(fp_train_list),
               len(tp_val_list), len(fp_val_list),
               len(scores_train_list), len(scores_val_list)]
    if not all(length == lengths[0] for length in lengths) or not lengths[0]:
        print("ERREUR: Listes incohérentes ou vides.")
        return {
            'pnl_norm_objective': float('-inf'),
            'ecart_train_val': float('inf')
        }

    fold_metrics = []
    for i in range(len(scores_train_list)):
        train_trades = tp_train_list[i] + fp_train_list[i]
        val_trades = tp_val_list[i] + fp_val_list[i]
        train_trades = max(train_trades, 1e-8)
        val_trades = max(val_trades, 1e-8)

        fold_metrics.append({
            'val_pnl': scores_val_list[i],
            'n_trades_val': val_trades
        })

    total_pnl = sum(m['val_pnl'] for m in fold_metrics)
    avg_pnl = total_pnl / len(fold_metrics)

    # On récupère l'écart global de winrate (train vs val)
    ecart_global = calculate_global_ecart(
        tp_train_list, fp_train_list,
        tp_val_list, fp_val_list,
        use_trade_weighting=use_trade_weighting_for_ecart
    )

    # Récupération des flags depuis la config
    use_imbalance_penalty = config.get('use_imbalance_penalty', False) if config else False
    use_std_penalty = config.get('use_std_penalty', False) if config else False
    use_brier_penalty = config.get('use_brier_penalty', False) if config else False
    use_winrate_std_penalty = config.get('use_winrate_std_penalty', False) if config else False
    use_negative_pnl_penalty = config.get('use_negative_pnl_penalty', False) if config else False

    # 🔴 Le nouveau flag pour pénaliser l’écart Train/Val
    use_winrate_ecart_penalty = config.get('use_winrate_ecart_penalty', False) if config else False

    penalty = 1.0
    imbalance_penalty = 1.0
    std_penalty = 1.0
    brier_penalty = 1.0
    winrate_std_penalty = 1.0
    negative_pnl_penalty = 1.0
    winrate_train_val_ecart_penalty = 1.0

    # ------------------------ PÉNALITÉS DÉJÀ EN PLACE ------------------------
    if use_imbalance_penalty:
        n_trades_val_list = [m['n_trades_val'] for m in fold_metrics]
        max_trades = max(n_trades_val_list)
        min_trades = max(min(n_trades_val_list), 1e-8)
        imbalance_penalty = 1 + math.log(max_trades / min_trades)
        penalty *= imbalance_penalty

    if use_std_penalty:
        n_trades_val_list = [m['n_trades_val'] for m in fold_metrics]
        std_trades = np.std(n_trades_val_list)
        mean_trades = np.mean(n_trades_val_list)
        cv_trades = std_trades / (mean_trades + 1e-8)
        std_penalty = 1 + cv_trades
        penalty *= std_penalty

    if use_winrate_std_penalty and winrates_val_by_fold is not None and len(winrates_val_by_fold) > 0:
        # Convertir si string avec '%'
        winrates_decimal = []
        for wr in winrates_val_by_fold:
            if isinstance(wr, str) and '%' in wr:
                winrates_decimal.append(float(wr.strip('%')) / 100)
            else:
                winrates_decimal.append(float(wr))

        std_winrate = np.std(winrates_decimal)
        mean_winrate = np.mean(winrates_decimal)
        cv_winrate = std_winrate / (mean_winrate + 1e-8)
        winrate_std_penalty = 1 + cv_winrate * 2  # Facteur multiplicatif ajustable
        penalty *= winrate_std_penalty

    if use_brier_penalty and relative_brier_val_list is not None and len(relative_brier_val_list) > 0:
        if len(relative_brier_val_list) == len(fold_metrics):
            mean_relative_brier = np.mean(relative_brier_val_list)
            brier_penalty = 1 + mean_relative_brier
            penalty *= brier_penalty
        else:
            raise ValueError(
                f"Liste de Brier scores relatifs incohérente: {len(relative_brier_val_list)}"
                f" scores pour {len(fold_metrics)} folds"
            )

    if use_negative_pnl_penalty and scores_val_list is not None and len(scores_val_list) > 0:
        try:
            min_pnl_val = min(scores_val_list)
            nb_neg = sum(1 for s in scores_val_list if s < 0)
            if nb_neg > 0:
                # log pour atténuer l’effet des grosses pertes
                print("log pour atténuer l’effet des grosses pertes")
                scaled_penalty = math.log(1 + abs(min_pnl_val) / 1000)
                negative_pnl_penalty = 1 + nb_neg + scaled_penalty
                penalty *= negative_pnl_penalty
        except Exception as e:
            print(f"⚠️ Erreur lors du traitement des scores_val_list : {e}")
            pnl_norm_objective = float('-inf')  # fallback très négatif si erreur
            return {
                'pnl_norm_objective': pnl_norm_objective,
                'ecart_train_val': float('inf'),
            }

    # ------------------ NOUVELLE PÉNALITÉ SUR L’ÉCART TRAIN/VAL ------------------
    # si le flag est activé, on pénalise le fait que l'écart (ecart_global) soit élevé
    if use_winrate_ecart_penalty:
        # Par exemple, multiplier la pénalité par (1 + factor * ecart_global).
        # Choisissez la constante d'échelle (ici 2.0) selon l'importance que vous donnez à la pénalisation.
        factor = config.get('winrate_ecart_factor', 10.0)  # Valeur par défaut
        winrate_train_val_ecart_penalty = 1 + factor * ecart_global
        penalty *= winrate_train_val_ecart_penalty

    # ------------------------ CALCUL DE L’OBJECTIF FINAL ------------------------
    pnl_norm_objective = avg_pnl / penalty
    ecart_train_val = ecart_global

    # Attributs enregistrés pour analyse ultérieure
    trial.set_user_attr('pnl_norm_objective', pnl_norm_objective)
    trial.set_user_attr('avg_pnl', avg_pnl)
    trial.set_user_attr('imbalance_penalty', imbalance_penalty)
    trial.set_user_attr('std_penalty', std_penalty)
    trial.set_user_attr('brier_penalty', brier_penalty)
    trial.set_user_attr('winrate_std_penalty', winrate_std_penalty)
    trial.set_user_attr('negative_pnl_penalty', negative_pnl_penalty)
    trial.set_user_attr('winrate_train_val_ecart_penalty', winrate_train_val_ecart_penalty)
    trial.set_user_attr('ecart_train_val', ecart_train_val)

    if relative_brier_val_list is not None and len(relative_brier_val_list) > 0:
        trial.set_user_attr('mean_relative_brier', np.mean(relative_brier_val_list))
    if brier_val_list is not None and len(brier_val_list) > 0:
        trial.set_user_attr('mean_brier', np.mean(brier_val_list))

    # ✅ LOG condensé : valeurs actuelles vs meilleur trial (si dispo)
    print("\n--- Résumé Trial Actuel vs Meilleur Trial ---")
    try:
        best = trial.study.best_trial  # Erreur si aucun trial terminé
        print(f"pnl_norm_objective : {pnl_norm_objective:.4f} | best : {best.user_attrs.get('pnl_norm_objective', 'n/a'):.4f}")
        print(f"avg_pnl            : {avg_pnl:.4f} | best : {best.user_attrs.get('avg_pnl', 'n/a'):.4f}")
        print(f"penalty total      : {penalty:.4f} | best : "
              f"{best.user_attrs.get('imbalance_penalty', 1.0) * best.user_attrs.get('std_penalty', 1.0) * best.user_attrs.get('brier_penalty', 1.0) * best.user_attrs.get('winrate_std_penalty', 1.0) * best.user_attrs.get('negative_pnl_penalty', 1.0) * best.user_attrs.get('winrate_train_val_ecart_penalty', 1.0):.4f}")
        print(f"imbalance_penalty  : {imbalance_penalty:.4f} | best : {best.user_attrs.get('imbalance_penalty', 'n/a'):.4f}")
        print(f"std_penalty        : {std_penalty:.4f} | best : {best.user_attrs.get('std_penalty', 'n/a'):.4f}")
        print(f"brier_penalty      : {brier_penalty:.4f} | best : {best.user_attrs.get('brier_penalty', 'n/a'):.4f}")
        print(f"winrate_std_penalty: {winrate_std_penalty:.4f} | best : {best.user_attrs.get('winrate_std_penalty', 'n/a'):.4f}")
        print(f"negative_pnl_penalty: {negative_pnl_penalty:.4f} | best : {best.user_attrs.get('negative_pnl_penalty', 'n/a'):.4f}")
        print(f"winrate_train_val_ecart_penalty: {winrate_train_val_ecart_penalty:.4f} | best : {best.user_attrs.get('winrate_train_val_ecart_penalty', 'n/a'):.4f}")

        if relative_brier_val_list:
            print(f"mean_relative_brier: {np.mean(relative_brier_val_list):.4f} | best : {best.user_attrs.get('mean_relative_brier', 'n/a'):.4f}")
        print(f"ecart_train_val    : {ecart_train_val:.4f} | best : {best.user_attrs.get('ecart_train_val', 'n/a'):.4f}")
    except ValueError:
        print("⚠️ Aucun trial terminé, affichage du trial en cours uniquement.")
        print(f"pnl_norm_objective : {pnl_norm_objective:.4f}")
        print(f"avg_pnl            : {avg_pnl:.4f}")
        print(f"penalty total      : {penalty:.4f}")
        print(f"imbalance_penalty  : {imbalance_penalty:.4f}")
        print(f"std_penalty        : {std_penalty:.4f}")
        print(f"brier_penalty      : {brier_penalty:.4f}")
        print(f"winrate_std_penalty: {winrate_std_penalty:.4f}")
        print(f"negative_pnl_penalty: {negative_pnl_penalty:.4f}")
        print(f"winrate_train_val_ecart_penalty: {winrate_train_val_ecart_penalty:.4f}")
        if relative_brier_val_list is not None and len(relative_brier_val_list) > 0:
            print(f"mean_relative_brier: {np.mean(relative_brier_val_list):.4f}")
        print(f"ecart_train_val    : {ecart_train_val:.4f}")
    print("------------------------------------------------")

    return {
        'pnl_norm_objective': pnl_norm_objective,
        'ecart_train_val': ecart_train_val,
        'raw_metrics': {
            'avg_pnl': avg_pnl,
            'ecart_global': ecart_global,
            'imbalance_penalty': imbalance_penalty,
            'std_penalty': std_penalty,
            'brier_penalty': brier_penalty,
            'winrate_std_penalty': winrate_std_penalty,
            'winrate_train_val_ecart_penalty': winrate_train_val_ecart_penalty
        }
    }

def normalize_to_range(value, old_min, old_max, new_min=0, new_max=1):
    """
    Normalise une valeur d'une plage à une autre
    """
    if old_max == old_min:
        return new_min

    value = max(min(value, old_max), old_min)
    normalized = (value - old_min) / (old_max - old_min)
    scaled = normalized * (new_max - new_min) + new_min

    return scaled


def process_RFE_filteringg(params=None, other_params=None, selected_columns=None, n_features_to_select=None,
                           X_train=None, y_train_label=None):
    # Configuration de RFECV
    # Configuration de RFECV
    params4RFECV = params.copy()
    params4RFECV['device'] = 'cpu'
    params4RFECV['tree_method'] = 'auto'
    # Ajouter early_stopping_rounds dans les paramètres
    params4RFECV['enable_categorical'] = False
    # Configuration des paramètres (reste inchangé)
    w_p = other_params['w_p']
    w_n = other_params['w_n']

    def custom_profit_scorer(y_true, y_pred_proba, other_params, normalize=False, config=None, custom=None):
        """
        Calcule le profit en fonction des prédictions de probabilités.

        Args:
            y_true: Labels réels
            y_pred_proba: Probabilités prédites (utilise la colonne 1 pour les prédictions positives)
            other_params: Dictionnaire contenant threshold, profit_per_tp, etc.
            normalize: Si True, normalise le profit par le nombre d'échantillons
        """
        # S'assurer que y_pred_proba est un tableau 2D
        if len(y_pred_proba.shape) == 1:
            y_pred_proba = y_pred_proba.reshape(-1, 1)
            y_pred_proba = np.column_stack((1 - y_pred_proba, y_pred_proba))

        # Prendre les probabilités de la classe positive
        positive_probs = y_pred_proba[:, 1]

        # Convertir en prédictions binaires selon le seuil
        y_pred = (positive_probs >= other_params['threshold']).astype(int)

        # Calculer les vrais positifs et faux positifs
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))

        # Calculer le profit
        profit = (tp * config['profit_per_tp']) - (fp * config['loss_per_fp'])

        if normalize:
            profit = profit / len(y_true)

        return profit
        # Vérifier que other_params est défini

    if other_params is None:
        raise ValueError("other_params ne peut pas être None")

    required_keys = ['profit_per_tp', 'loss_per_fp', 'threshold']
    missing_keys = [key for key in required_keys if key not in other_params]
    if missing_keys:
        raise ValueError(f"other_params doit contenir les clés suivantes: {missing_keys}")

    # Define the custom gradient and hessian functions
    def weighted_logistic_gradient_cpu_4classifier(predt: np.ndarray, y: np.ndarray, w_p: float,
                                                   w_n: float) -> np.ndarray:
        """Calculate the gradient for weighted logistic loss (CPU)."""
        predt = 1.0 / (1.0 + np.exp(-predt))  # Sigmoid function
        weights = np.where(y == 1, w_p, w_n)
        grad = weights * (predt - y)
        return grad

    def weighted_logistic_hessian_cpu_4classifier(predt: np.ndarray, y: np.ndarray, w_p: float,
                                                  w_n: float) -> np.ndarray:
        """Calculate the hessian for weighted logistic loss (CPU)."""
        predt = 1.0 / (1.0 + np.exp(-predt))
        weights = np.where(y == 1, w_p, w_n)
        hess = weights * predt * (1.0 - predt)
        return hess

    def weighted_logistic_hessian_cpu_4classifier(predt: np.ndarray, y: np.ndarray, w_p: float,
                                                  w_n: float) -> np.ndarray:
        """Calculate the hessian for weighted logistic loss (CPU)."""
        predt = 1.0 / (1.0 + np.exp(-predt))  # Sigmoid function
        weights = np.where(y == 1, w_p, w_n)
        hess = weights * predt * (1.0 - predt)
        return hess

    def weighted_logistic_obj_cpu_4classifier(y_true: np.ndarray, y_pred: np.ndarray, w_p: float, w_n: float) -> \
            Tuple[
                np.ndarray, np.ndarray]:
        """Custom objective function for weighted logistic loss."""
        predt = y_pred
        y = y_true
        grad = weighted_logistic_gradient_cpu_4classifier(predt, y, w_p, w_n)
        hess = weighted_logistic_hessian_cpu_4classifier(predt, y, w_p, w_n)
        return grad, hess

    # Create a partial function with fixed weights
    obj_function = partial(
        weighted_logistic_obj_cpu_4classifier,
        w_p=w_p,
        w_n=w_n
    )

    params4RFECV['objective'] = obj_function
    params4RFECV['disable_default_eval_metric'] = 1
    # Création du scorer personnalisé
    custom_scorer_partial = partial(
        custom_profit_scorer,
        other_params=other_params,
        normalize=False
    )

    custom_scorer = make_scorer(
        custom_scorer_partial,
        response_method="predict_proba",
        greater_is_better=True
    )

    # XGBoost wrapper class
    class XGBWrapper(xgb.XGBClassifier):
        """Wrapper for XGBoost with early stopping integrated into the constructor."""

        def fit(self, X, y, **kwargs):
            eval_set = kwargs.pop('eval_set', None)
            return super().fit(
                X, y,
                eval_set=eval_set,
                verbose=False,
                **kwargs
            )

    # Configuration et exécution de RFECV
    model4RFECV = XGBWrapper(**params4RFECV, n_jobs=1)
    cv_inner = KFold(n_splits=5, shuffle=False)

    rfecv = RFECV(
        estimator=model4RFECV,
        cv=cv_inner,
        step=1,
        scoring=custom_scorer,
        n_jobs=-1,  # Utilisation de tous les cœurs pour accélérer le processus
        min_features_to_select=5  # Nombre minimal de caractéristiques à sélectionner
    )

    def get_optimal_n_features(cv_results):
        """Retourne le nombre optimal de caractéristiques basé sur les scores moyens et ajustés."""
        mean_scores = cv_results['mean_test_score']
        std_scores = cv_results['std_test_score']

        # Calculer le score ajusté (mean - std)
        adjusted_scores = mean_scores - 2 * std_scores

        # Option 1 : Choisir le nombre de caractéristiques avec le score moyen maximal
        optimal_n_features_mean = np.argmax(mean_scores) + 1
        print(f"Nombre optimal de features basé sur le meilleur score moyen : {optimal_n_features_mean}")
        print(f"Score moyen optimal : {mean_scores.max():.2f}")

        # Option 2 : Choisir le nombre de caractéristiques avec le meilleur score ajusté (mean - std)
        optimal_n_features_adjusted = np.argmax(adjusted_scores) + 1
        print(f"Nombre optimal de features basé sur mean-std : {optimal_n_features_adjusted}")
        print(f"Score ajusté optimal : {adjusted_scores.max():.2f}")

        # Créer un DataFrame pour visualiser tous les scores
        scores_df = pd.DataFrame({
            'n_features': range(1, len(mean_scores) + 1),
            'mean_score': mean_scores,
            'std_score': std_scores,
            'adjusted_score': adjusted_scores
        })

        # Afficher le tableau des scores
        print("\nScores détaillés :")
        print(scores_df)

        # Retourner le nombre optimal basé sur le score ajusté
        return optimal_n_features_adjusted

    # Fit RFECV
    rfecv.fit(X_train, y_train_label)

    print('model4RF2CV done')

    # Obtenir le nombre optimal de caractéristiques
    optimal_n_features = get_optimal_n_features(rfecv.cv_results_)
    if (optimal_n_features < n_features_to_select):
        try:
            def get_top_n_features(selected_columns, n_features_to_select, X_train):
                if len(selected_columns) < n_features_to_select:
                    raise ValueError(
                        f"La liste selected_columns contient seulement {len(selected_columns)} features, "
                        f"alors que {n_features_to_select} sont demandées.")

                # Créer un masque pour les features sélectionnées
                feature_mask = X_train.columns.isin(selected_columns[:n_features_to_select])

                # Obtenir les indices triés
                sorted_indices = np.where(feature_mask)[0]

                # Obtenir les noms des caractéristiques correspondantes
                selected_feature_names = X_train.columns[sorted_indices]

                return selected_feature_names

            selected_feature_names = get_top_n_features(selected_columns, n_features_to_select, X_train)
        except ValueError as e:
            print(f"Erreur: {str(e)}")
    else:
        # Obtenir les rangs des caractéristiques
        feature_rankings = rfecv.ranking_

        # Obtenir les indices des caractéristiques triés par rang
        sorted_indices = np.argsort(feature_rankings)

        # Sélectionner les indices des 'optimal_n_features' meilleures caractéristiques
        optimal_indices = sorted_indices[:optimal_n_features]

        # Obtenir les noms des caractéristiques correspondantes
        selected_feature_names = X_train.columns[optimal_indices]

    # Afficher les caractéristiques sélectionnées
    print(f"Caractéristiques correspondant à optimal_n_features ({selected_feature_names}):")
    print(selected_feature_names.tolist())

    print(f"Nombre de features sélectionnées: {len(selected_feature_names)}")
    print("\nFeatures sélectionnées:")
    for i, feature in enumerate(selected_feature_names, 1):
        print(f"{i}. {feature}")

    # Affichage des scores de cross-validation
    print("\nScores de cross-validation par nombre de features:")
    """
    cv_scores = pd.DataFrame({
        'n_features': range(1, len(rfecv.cv_results_['mean_test_score']) + 1),
        'mean_score': rfecv.cv_results_['mean_test_score'],
        'std_score': rfecv.cv_results_['std_test_score']
    })
    print(cv_scores)
    """
    # Log des résultats
    return selected_feature_names;


import traceback


def create_custom_importance_plot(shap_df, dataset_name, save_dir):
    """
    Crée un graphique personnalisé basé sur les valeurs SHAP du CSV
    avec code couleur rouge/bleu selon le signe
    """
    plt.figure(figsize=(12, 8))

    # Prendre les 20 premières features pour la lisibilité
    top_n = 20
    df_plot = shap_df.head(top_n).iloc[::-1]  # Inverse l'ordre pour l'affichage de bas en haut

    # Création des barres
    y_pos = np.arange(len(df_plot))
    bars = plt.barh(y_pos, df_plot['importance'].abs(),
                    color=[('red' if x < 0 else 'blue') for x in df_plot['importance']])

    # Personnalisation du graphique
    plt.yticks(y_pos, df_plot['feature'])
    plt.xlabel('mean(|SHAP value|) (average impact on model output magnitude)')
    plt.title(f'SHAP Feature Importance - {dataset_name}')

    # Ajout des valeurs sur les barres
    for i, bar in enumerate(bars):
        width = bar.get_width()
        value = df_plot['importance'].iloc[i]
        plt.text(width, bar.get_y() + bar.get_height() / 2,
                 f'{value:.2e}',
                 ha='left', va='center', fontsize=8)

    # Ajustements de la mise en page
    plt.tight_layout()

    # Sauvegarde
    plt.savefig(os.path.join(save_dir, f'shap_importance_{dataset_name}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


def add_session_id(df, time_periods_dict):
    """
    Ajoute une colonne session_ID basée sur deltaTimestampOpeningSession1min
    """
    # Créer une colonne session_ID avec valeur par défaut None
    df['session_type_index'] = -1

    # Pour chaque période dans le dictionnaire
    for session_name, info in time_periods_dict.items():
        # Créer un masque pour les timestamps dans cette période
        mask = (df['deltaTimestampOpening'] >= info['start']) & \
               (df['deltaTimestampOpening'] < info['end'])
        # Assigner le session_type_index aux lignes correspondantes
        df.loc[mask, 'session_type_index'] = info['session_type_index']

    return df


def manage_rfe_selection(X_train, y_train_label, config, trial, params, weight_param, other_params):
    """
    Gère la sélection des features avec RFE selon la configuration.

    Args:
        X_train (pd.DataFrame): DataFrame contenant les features d'entraînement
        y_train_label: Labels d'entraînement
        config (dict): Configuration contenant les paramètres
        trial (optuna.Trial): Trial Optuna pour l'optimisation
        params (dict): Paramètres du modèle
        weight_param: Paramètres de pondération
        other_params (dict): Dictionnaire des métriques

    Returns:
        tuple: (X_train modifié, liste des noms des features sélectionnées)
    """
    # Récupération du paramètre RFE de la config avec valeur par défaut NO_RFE
    use_of_rfe_in_optuna = config.get('use_of_rfe_in_optuna', rfe_param.NO_RFE)

    # Liste qui contiendra les noms des features sélectionnées
    selected_feature_names = []

    if use_of_rfe_in_optuna != rfe_param.NO_RFE:
        # Log de l'activation du RFE
        print(f"\n------- RFE activé:")

        # Application du RFE pour la sélection des features
        X_train, selected_feature_names = select_features_ifRfe(
            X_train=X_train,
            y_train_label=y_train_label,
            trial=trial,
            config=config,
            params=params,
            weight_param=weight_param,
            other_params=other_params,
            use_of_rfe_in_optuna=use_of_rfe_in_optuna
        )

        # Log du nombre de features sélectionnées
        print(f"              - Features sélectionnées avec rfe: {len(selected_feature_names)}")
    else:
        # Si RFE non activé, on garde toutes les features
        selected_feature_names = X_train.columns.tolist()

    return X_train, selected_feature_names


def select_features_ifRfe(X_train, y_train_label, trial, config, params, other_params, use_of_rfe_in_optuna=None,
                          selected_columns=None):
    """Sélection des features avec RFE"""
    if use_of_rfe_in_optuna == rfe_param.RFE_WITH_OPTUNA:
        n_features_to_select = trial.suggest_int("n_features_to_select", 1, X_train.shape[1])
    elif use_of_rfe_in_optuna == rfe_param.RFE_AUTO:
        n_features_to_select = config.get('min_features_if_RFE_AUTO', 5)

    selected_feature_names = process_RFE_filteringg(params, other_params,
                                                    selected_columns, n_features_to_select,
                                                    X_train, y_train_label)
    X_train_selected = X_train[selected_feature_names]

    return X_train_selected, selected_feature_names


def manage_rfe_selection(X_train, y_train_label, config, trial, params, other_params):
    """
    Gère la sélection des features avec RFE selon la configuration.

    Args:
        X_train (pd.DataFrame): DataFrame contenant les features d'entraînement
        y_train_label: Labels d'entraînement
        config (dict): Configuration contenant les paramètres
        trial (optuna.Trial): Trial Optuna pour l'optimisation
        params (dict): Paramètres du modèle
        weight_param: Paramètres de pondération
        other_params (dict): Dictionnaire des métriques

    Returns:
        tuple: (X_train modifié, liste des noms des features sélectionnées)
    """
    # Récupération du paramètre RFE de la config avec valeur par défaut NO_RFE
    use_of_rfe_in_optuna = config.get('use_of_rfe_in_optuna', rfe_param.NO_RFE)

    # Liste qui contiendra les noms des features sélectionnées
    selected_feature_names = []

    if use_of_rfe_in_optuna != rfe_param.NO_RFE:
        # Log de l'activation du RFE
        print(f"\n------- RFE activé:")

        # Application du RFE pour la sélection des features
        X_train, selected_feature_names = select_features_ifRfe(
            X_train=X_train,
            y_train_label=y_train_label,
            trial=trial,
            config=config,
            params=params,
            other_params=other_params,
            use_of_rfe_in_optuna=use_of_rfe_in_optuna
        )

        # Log du nombre de features sélectionnées
        print(f"              - Features sélectionnées avec rfe: {len(selected_feature_names)}")
    else:
        # Si RFE non activé, on garde toutes les features
        selected_feature_names = X_train.columns.tolist()

    return X_train, selected_feature_names


import numpy as np
import xgboost as xgb
from typing import Dict, List, Any, Tuple


def compute_confusion_matrix_cupy(y_true_gpu, y_pred_gpu):
    """Calcule la matrice de confusion sur GPU"""
    # Conversion en CuPy si nécessaire
    if not isinstance(y_true_gpu, cp.ndarray):
        y_true_gpu = cp.asarray(y_true_gpu, dtype=cp.int32)
    if not isinstance(y_pred_gpu, cp.ndarray):
        y_pred_gpu = cp.asarray(y_pred_gpu, dtype=cp.int32)

    # Calculs sur GPU
    tp = cp.sum((y_true_gpu == 1) & (y_pred_gpu == 1))
    fp = cp.sum((y_true_gpu == 0) & (y_pred_gpu == 1))
    tn = cp.sum((y_true_gpu == 0) & (y_pred_gpu == 0))
    fn = cp.sum((y_true_gpu == 1) & (y_pred_gpu == 0))

    return tn, fp, fn, tp


def calculate_fold_stats(labels: object, set_name: object, config: object) -> object:
    """Calcule les statistiques du fold sur GPU ou CPU"""
    if config['device_'] == 'cuda':
        if not isinstance(labels, cp.ndarray):
            labels = cp.asarray(labels)
        xp = cp
    else:
        if not isinstance(labels, np.ndarray):
            labels = np.asarray(labels)
        xp = np

    decisions = xp.sum(labels != 99)
    success = xp.sum(labels == 1)
    failures = xp.sum(labels == 0)
    success_rate = xp.where(decisions > 0, success / decisions, xp.float32(0.0))

    return {
        f"{set_name}_n_trades": decisions,
        f"{set_name}_n_class_1": success,
        f"{set_name}_n_class_0": failures,
        f"{set_name}_class_ratio": success_rate,
        f"{set_name}_success_rate": success_rate
    }


def calculate_winrate_gpu(tp_val, fp_val):
    """Calcule le winrate de manière sûre sur GPU"""
    if not isinstance(tp_val, cp.ndarray):
        tp_val = cp.asarray(tp_val, dtype=cp.float32)
    if not isinstance(fp_val, cp.ndarray):
        fp_val = cp.asarray(fp_val, dtype=cp.float32)

    # Calcul sur GPU
    tp_fp_sum = tp_val + fp_val
    mask = tp_fp_sum != 0
    winrate = cp.where(mask, tp_val / tp_fp_sum, cp.float32(0.0))

    return winrate


def setup_other_params(trial, weight_param, config, other_params=None):
    """Configure le dictionnaire des métriques en fonction des paramètres Optuna"""

    if other_params is None:
        other_params = {}

    # Seuil de décision
    threshold_value = trial.suggest_float(
        'threshold',
        weight_param['threshold']['min'],
        weight_param['threshold']['max']
    )
    other_params['threshold'] = threshold_value

    # Pondération des TP et TN
    other_params['w_p'] = trial.suggest_float(
        'w_p', weight_param['w_p']['min'], weight_param['w_p']['max']
    )
    other_params['w_n'] = trial.suggest_float(
        'w_n', weight_param['w_n']['min'], weight_param['w_n']['max']
    )

    # Pénalité des faux négatifs (si activé)
    other_params['penalty_per_fn'] = trial.suggest_float(
        'penalty_per_fn',
        weight_param['penalty_per_fn']['min'],
        weight_param['penalty_per_fn']['max']
    )

    # Nombre d’itérations pour les modèles boosting
    if config['model_type'] in [modelType.XGB, modelType.LGBM]:
        other_params['num_boost_round'] = trial.suggest_int(
            'num_boost_round',
            weight_param['num_boost_round']['min'],
            weight_param['num_boost_round']['max']
        )

    return other_params


def setup_model_params_optuna(trial, config, random_state_seed_):
    """Configure les paramètres du modèle pour l'entraînement"""
    # Initialisation des variables
    params = None
    num_boost_round = None
    device = config.get('device_', None)
    model_type = config.get('model_type', None)
    boosting_type = config.get('boosting_type', None)
    # Vérification que toutes les variables sont définies
    if device is None or model_type is None or boosting_type is None:
        raise ValueError("Les paramètres suivants sont requis : 'device_', 'model_type', 'boosting_type'")

    modele_param_optuna_range = get_model_param_range(model_type)

    if model_type == modelType.XGB:
        params = {
            'max_depth': trial.suggest_int('max_depth',
                                           modele_param_optuna_range['max_depth']['min'],
                                           modele_param_optuna_range['max_depth']['max']),
            'learning_rate': trial.suggest_float('learning_rate',
                                                 modele_param_optuna_range['learning_rate']['min'],
                                                 modele_param_optuna_range['learning_rate']['max'],
                                                 log=modele_param_optuna_range['learning_rate'].get('log', False)),
            'min_child_weight': trial.suggest_int('min_child_weight',
                                                  modele_param_optuna_range['min_child_weight']['min'],
                                                  modele_param_optuna_range['min_child_weight']['max']),
            'subsample': trial.suggest_float('subsample',
                                             modele_param_optuna_range['subsample']['min'],
                                             modele_param_optuna_range['subsample']['max']),
            'colsample_bytree': trial.suggest_float('colsample_bytree',
                                                    modele_param_optuna_range['colsample_bytree']['min'],
                                                    modele_param_optuna_range['colsample_bytree']['max']),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel',
                                                     modele_param_optuna_range['colsample_bylevel']['min'],
                                                     modele_param_optuna_range['colsample_bylevel']['max']),
            'colsample_bynode': trial.suggest_float('colsample_bynode',
                                                    modele_param_optuna_range['colsample_bynode']['min'],
                                                    modele_param_optuna_range['colsample_bynode']['max']),
            'min_split_loss': trial.suggest_float('min_split_loss',
                                                  modele_param_optuna_range['min_split_loss']['min'],
                                                  modele_param_optuna_range['min_split_loss']['max']),
            'reg_alpha': trial.suggest_float('reg_alpha',
                                             modele_param_optuna_range['reg_alpha']['min'],
                                             modele_param_optuna_range['reg_alpha']['max'],
                                             log=modele_param_optuna_range['reg_alpha'].get('log', False)),
            'reg_lambda': trial.suggest_float('reg_lambda',
                                              modele_param_optuna_range['reg_lambda']['min'],
                                              modele_param_optuna_range['reg_lambda']['max'],
                                              log=modele_param_optuna_range['reg_lambda'].get('log', False)),
            'random_state': random_state_seed_,
            'tree_method': 'hist',
            'device': device,
            'booster': boosting_type,
        }
        # num_boost_round = trial.suggest_int('num_boost_round',
        #                                   modele_param_optuna_range['num_boost_round']['min'],
        #                                  modele_param_optuna_range['num_boost_round']['max'])

    elif model_type == modelType.LGBM:
        params = {
            # Configuration du nombre de feuilles par arbre
            'num_leaves': trial.suggest_int(
                'num_leaves',
                modele_param_optuna_range['num_leaves']['min'],
                modele_param_optuna_range['num_leaves']['max']
            ),

            # Configuration du taux d'apprentissage
            'learning_rate': trial.suggest_float(
                'learning_rate',
                modele_param_optuna_range['learning_rate']['min'],
                modele_param_optuna_range['learning_rate']['max'],
                log=modele_param_optuna_range['learning_rate'].get('log', False)
            ),

            # Configuration du nombre minimum d'observations par feuille
            'min_child_samples': trial.suggest_int(
                'min_child_samples',
                modele_param_optuna_range['min_child_samples']['min'],
                modele_param_optuna_range['min_child_samples']['max']
            ),

            # Configuration de la fraction des features par arbre
            'feature_fraction': trial.suggest_float(
                'feature_fraction',
                modele_param_optuna_range['feature_fraction']['min'],
                modele_param_optuna_range['feature_fraction']['max']
            ),

            # Configuration de la fraction des features par niveau
            'feature_fraction_bynode': trial.suggest_float(
                'feature_fraction_bynode',
                modele_param_optuna_range['feature_fraction_bynode']['min'],
                modele_param_optuna_range['feature_fraction_bynode']['max']
            ),

            # Configuration du gain minimum pour split
            'min_split_gain': trial.suggest_float(
                'min_split_gain',
                modele_param_optuna_range['min_split_gain']['min'],
                modele_param_optuna_range['min_split_gain']['max']
            ),

            # Configuration de la régularisation L1
            'lambda_l1': trial.suggest_float(
                'lambda_l1',
                modele_param_optuna_range['lambda_l1']['min'],
                modele_param_optuna_range['lambda_l1']['max'],
                log=modele_param_optuna_range['lambda_l1'].get('log', False)
            ),

            # Configuration de la régularisation L2
            'lambda_l2': trial.suggest_float(
                'lambda_l2',
                modele_param_optuna_range['lambda_l2']['min'],
                modele_param_optuna_range['lambda_l2']['max'],
                log=modele_param_optuna_range['lambda_l2'].get('log', False)
            ),

            # Configuration de la fraction des données pour le bagging
            'bagging_fraction': trial.suggest_float(
                'bagging_fraction',
                modele_param_optuna_range['bagging_fraction']['min'],
                modele_param_optuna_range['bagging_fraction']['max']
            ),

            # Configuration de la fréquence du bagging
            'bagging_freq': trial.suggest_int(
                'bagging_freq',
                modele_param_optuna_range['bagging_freq']['min'],
                modele_param_optuna_range['bagging_freq']['max']
            ),

            # Nouveau: Configuration de la profondeur maximale de l'arbre
            'max_depth': trial.suggest_int(
                'max_depth',
                modele_param_optuna_range.get('max_depth', {}).get('min', 2),
                modele_param_optuna_range.get('max_depth', {}).get('max', 5)
            ),

            # Nouveau: Configuration du lissage des chemins
            'path_smooth': trial.suggest_float(
                'path_smooth',
                modele_param_optuna_range.get('path_smooth', {}).get('min', 0),
                modele_param_optuna_range.get('path_smooth', {}).get('max', 3)
            ) if 'path_smooth' in modele_param_optuna_range else 0,

            # Nouveau: Configuration du dropout pour mode dart (si applicable)
            'drop_rate': trial.suggest_float(
                'drop_rate',
                modele_param_optuna_range.get('drop_rate', {}).get('min', 0.01),
                modele_param_optuna_range.get('drop_rate', {}).get('max', 0.5)
            ) if boosting_type == 'dart' and 'drop_rate' in modele_param_optuna_range else 0.1,

            'skip_drop': trial.suggest_float(
                'skip_drop',
                modele_param_optuna_range.get('skip_drop', {}).get('min', 0.1),
                modele_param_optuna_range.get('skip_drop', {}).get('max', 0.8)
            ) if boosting_type == 'dart' else 0.5,

            'max_drop': trial.suggest_int(
                'max_drop',
                modele_param_optuna_range.get('max_drop', {}).get('min', 5),
                modele_param_optuna_range.get('max_drop', {}).get('max', 50)
            ) if boosting_type == 'dart' else 10,

            # 🔧 Paramètres fixes
            'min_data_in_bin': 3,
            'max_bin': 255,
            'seed': random_state_seed_,
            'device_type': device,
            'metric': 'None',
            'objective': 'binary',
            'verbose': -1,
            'boost_from_average': True,
            'extra_trees': True,
            'early_stopping_rounds': modele_param_optuna_range.get('early_stopping_rounds', 100),
            'boosting_type': boosting_type,

        }
    elif model_type == modelType.XGBRF:
        params = {
            # Nombre total d'arbres
            'n_estimators': trial.suggest_int(
                'n_estimators',
                modele_param_optuna_range['n_estimators']['min'],
                modele_param_optuna_range['n_estimators']['max']
            ),

            # Profondeur maximale des arbres
            'max_depth': trial.suggest_int(
                'max_depth',
                modele_param_optuna_range['max_depth']['min'],
                modele_param_optuna_range['max_depth']['max']
            ),

            # Échantillonnage des lignes (bootstrap)
            'subsample': trial.suggest_float(
                'subsample',
                modele_param_optuna_range['subsample']['min'],
                modele_param_optuna_range['subsample']['max']
            ),

            # Échantillonnage des colonnes par nœud
            'colsample_bynode': trial.suggest_float(
                'colsample_bynode',
                modele_param_optuna_range['colsample_bynode']['min'],
                modele_param_optuna_range['colsample_bynode']['max']
            ),

            # Poids minimum par feuille
            'min_child_weight': trial.suggest_int(
                'min_child_weight',
                modele_param_optuna_range['min_child_weight']['min'],
                modele_param_optuna_range['min_child_weight']['max']
            ),

            # Perte minimale pour un split
            'gamma': trial.suggest_float(
                'gamma',
                modele_param_optuna_range['gamma']['min'],
                modele_param_optuna_range['gamma']['max']
            ),

            # Régularisation L1
            'reg_alpha': trial.suggest_float(
                'reg_alpha',
                modele_param_optuna_range['reg_alpha']['min'],
                modele_param_optuna_range['reg_alpha']['max'],
                log=modele_param_optuna_range['reg_alpha'].get('log', False)
            ),

            # Régularisation L2
            'reg_lambda': trial.suggest_float(
                'reg_lambda',
                modele_param_optuna_range['reg_lambda']['min'],
                modele_param_optuna_range['reg_lambda']['max'],
                log=modele_param_optuna_range['reg_lambda'].get('log', False)
            ),

            # Paramètres fixes
            'random_state': random_state_seed_,
            'n_jobs': -1,
            'verbosity': 0
        }

        # Ajout des paramètres fixes spécifiques à XGBRFClassifier
        # Learning rate DOIT être 1.0 pour simuler un vrai Random Forest
        params['learning_rate'] = 1.0

        # Activation GPU si configuré
        if config.get('gpu_4RF') == 'gpu_4RF':
            params['tree_method'] = 'gpu_hist'
    elif model_type == modelType.RF:
        params = {
            'n_estimators': trial.suggest_int(
                'n_estimators',
                modele_param_optuna_range['n_estimators']['min'],
                modele_param_optuna_range['n_estimators']['max']
            ),
            'max_depth': trial.suggest_int(
                'max_depth',
                modele_param_optuna_range['max_depth']['min'],
                modele_param_optuna_range['max_depth']['max']
            ),
            'min_samples_split': trial.suggest_int(
                'min_samples_split',
                modele_param_optuna_range['min_samples_split']['min'],
                modele_param_optuna_range['min_samples_split']['max']
            ),
            'min_samples_leaf': trial.suggest_int(
                'min_samples_leaf',
                modele_param_optuna_range['min_samples_leaf']['min'],
                modele_param_optuna_range['min_samples_leaf']['max']
            ),
            'max_features': trial.suggest_float(
                'max_features',
                modele_param_optuna_range['max_features']['min'],
                modele_param_optuna_range['max_features']['max']
            ),
            'max_leaf_nodes': trial.suggest_int(
                'max_leaf_nodes',
                modele_param_optuna_range['max_leaf_nodes']['min'],
                modele_param_optuna_range['max_leaf_nodes']['max']
            ),
            'ccp_alpha': trial.suggest_float(
                'ccp_alpha',
                modele_param_optuna_range['ccp_alpha']['min'],
                modele_param_optuna_range['ccp_alpha']['max'],
                log=modele_param_optuna_range['ccp_alpha'].get('log', False)
            ),
            'min_impurity_decrease': trial.suggest_float(
                'min_impurity_decrease',
                modele_param_optuna_range['min_impurity_decrease']['min'],
                modele_param_optuna_range['min_impurity_decrease']['max'],
                log=modele_param_optuna_range['min_impurity_decrease'].get('log', False)
            ),
            'bootstrap': False,
            'criterion': 'entropy',
            'random_state': random_state_seed_,
            'n_jobs': -1,
            'verbose': 0,
            'warm_start': False,
            'class_weight': None
        }



    elif model_type == modelType.SVC:
        params = {
            # Paramètre de régularisation - contrôle le compromis entre la marge et les erreurs
            'C': trial.suggest_float(

                'C',
                modele_param_optuna_range['C']['min'],
                modele_param_optuna_range['C']['max'],
                log=modele_param_optuna_range['C'].get('log', True)
            ),
            # Paramètre gamma pour les noyaux 'rbf', 'poly' et 'sigmoid'
            'gamma': trial.suggest_float(
                'gamma',
                modele_param_optuna_range['gamma']['min'],
                modele_param_optuna_range['gamma']['max'],
                log=modele_param_optuna_range['gamma'].get('log', True)
            ),
            # Choix du noyau - crucial pour la flexibilité du modèle
            'kernel': trial.suggest_categorical(
                'kernel',
                modele_param_optuna_range['kernel']
            ),
            # Degré du polynôme pour le noyau 'poly'
            'degree': trial.suggest_int(
                'degree',
                modele_param_optuna_range['degree']['min'],
                modele_param_optuna_range['degree']['max']
            ),
            # Coefficient pour les noyaux 'poly' et 'sigmoid'
            'coef0': trial.suggest_float(
                'coef0',
                modele_param_optuna_range['coef0']['min'],
                modele_param_optuna_range['coef0']['max']
            ),
            # Tolérance pour le critère d'arrêt - impacte la précision des frontières
            'tol': trial.suggest_float(
                'tol',
                modele_param_optuna_range['tol']['min'],
                modele_param_optuna_range['tol']['max'],
                log=modele_param_optuna_range['tol'].get('log', True)
            ),
            # Shrinking heuristic - peut accélérer l'entraînement
            'shrinking': trial.suggest_categorical(
                'shrinking',
                modele_param_optuna_range['shrinking']
            ),
            # Stratégie de gestion des classes déséquilibrées
            'class_weight': trial.suggest_categorical(
                'class_weight',
                modele_param_optuna_range['class_weight']
            ),
            # Paramètres fixes
            'random_state': random_state_seed_,
            'verbose': 0,  # Réduire les logs
            'probability': True,  # Pour permettre la calibration des probabilités si nécessaire
        }
    elif model_type == modelType.CATBOOST:
        params = {
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'verbose': False,
            'early_stopping_rounds': 50,
            'task_type': 'GPU' if device == 'cuda' else 'CPU',
            'boosting_type': 'Ordered',
            'has_time': True,
            'random_seed': random_state_seed_,
            'thread_count': -1,
            'fold_permutation_block_size': 10
        }

        # Ajout des paramètres optimisés
        for param_name, param_range in modele_param_optuna_range.items():
            if 'values' in param_range:
                params[param_name] = trial.suggest_categorical(param_name, param_range['values'])
            elif 'log' in param_range:
                params[param_name] = trial.suggest_float(param_name, param_range['min'],
                                                         param_range['max'], log=True)
            else:
                if isinstance(param_range['min'], int) and isinstance(param_range['max'], int):
                    params[param_name] = trial.suggest_int(param_name, param_range['min'],
                                                           param_range['max'])
                else:
                    params[param_name] = trial.suggest_float(param_name, param_range['min'],
                                                             param_range['max'])

        # num_boost_round = params.pop('iterations', None)  # Pour CatBoost, iterations = num_boost_round

    # Vérification finale
    if params is None:
        raise ValueError(f"Type de modèle non supporté ou paramètres non initialisés: {model_type}")

    return params


def validate_fold_indices(train_pos, val_pos):
    """
    Valide les indices des folds pour assurer leur intégrité.

    Args:
        train_pos (np.array): Indices d'entraînement
        val_pos (np.array): Indices de validation

    Raises:
        ValueError: Si les indices ne sont pas valides
    """
    # Conversion en entiers
    train_pos = train_pos.astype(int)
    val_pos = val_pos.astype(int)

    # Vérification des ensembles vides
    if len(train_pos) == 0 or len(val_pos) == 0:
        raise ValueError("⚠️ Ensemble vide détecté")

    # Vérifier que les indices sont strictement croissants
    if not all(train_pos[i] < train_pos[i + 1] for i in range(len(train_pos) - 1)):
        raise ValueError("Les indices d'entraînement ne sont pas strictement croissants")

    if not all(val_pos[i] < val_pos[i + 1] for i in range(len(val_pos) - 1)):
        raise ValueError("Les indices de validation ne sont pas strictement croissants")

    # Vérifier que validation vient après entraînement
    if not max(train_pos) < min(val_pos):
        raise ValueError("Les indices de validation doivent être postérieurs aux indices d'entraînement")


def prepare_fold_data(X_train, y_train_label, train_pos, val_pos):
    """
    Prépare les données pour un fold spécifique.

    Args:
        X_train (pd.DataFrame): Données d'entraînement complètes
        y_train_label (np.array/pd.Series): Labels
        train_pos (np.array): Indices d'entraînement
        val_pos (np.array): Indices de validation

    Returns:
        dict: Données préparées pour le fold
    """
    # Préparation des labels selon le type
    y_train_fold = y_train_label.iloc[train_pos] if isinstance(y_train_label, pd.Series) \
        else y_train_label[train_pos]
    y_val_fold = y_train_label.iloc[val_pos] if isinstance(y_train_label, pd.Series) \
        else y_train_label[val_pos]

    fold_data = {
        'train_indices': train_pos,
        'val_indices': val_pos,
        'y_train': y_train_fold,
        'y_val': y_val_fold,
        'X_train': X_train.iloc[train_pos] if isinstance(X_train, pd.DataFrame) else X_train[train_pos],
        'X_val': X_train.iloc[val_pos] if isinstance(X_train, pd.DataFrame) else X_train[val_pos]
    }

    # Calculer les statistiques de distribution
    train_dist = np.unique(y_train_fold, return_counts=True)
    val_dist = np.unique(y_val_fold, return_counts=True)

    fold_data['distributions'] = {
        'train': dict(zip(train_dist[0], train_dist[1])),
        'val': dict(zip(val_dist[0], val_dist[1]))
    }

    return fold_data


def calculate_final_results(metrics_dict, arrays, all_fold_stats, nb_split_tscv, config):
    """
    Calcule les résultats finaux de la validation croisée.

    Args:
        metrics_dict (dict): Métriques accumulées
        arrays (dict): Arrays GPU des métriques par fold
        all_fold_stats (dict): Statistiques par fold
        nb_split_tscv (int): Nombre de folds
    """
    try:
        # Conversion des métriques en numpy
        final_metrics = {key: convert_metrics_to_numpy_safe(value, config)
                         for key, value in metrics_dict.items()
                         if key.startswith('total_')}



        if config['device_'] == 'cuda':
            mean_val_score = float(cp.mean(arrays['scores_val']).get())
            std_val_score = float(cp.std(arrays['scores_val']).get())
        else:
            mean_val_score = float(np.mean(arrays['scores_val']))
            std_val_score = float(np.std(arrays['scores_val']))

        results = {

            # Entraînement
            'winrates_train_by_fold': arrays['winrates_train'],
            'nb_trades_train_by_fold': arrays['nb_trades_train'],
            'nb_samples_train_by_fold': arrays['nb_samples_train'],
            'scores_train_by_fold': arrays['scores_train'],
            'tp_train_by_fold': arrays['tp_train'],
            'fp_train_by_fold': arrays['fp_train'],
            'class0_raw_data_train_by_fold': arrays['class0_raw_data_train_by_fold'],
            'class1_raw_data_train_by_fold': arrays['class1_raw_data_train_by_fold'],
            'winrate_raw_data_train_by_fold': arrays['winrate_raw_data_train_by_fold'],
            'train_pred_proba_log_odds': arrays['train_pred_proba_log_odds'],
            'train_trades_samples_perct': arrays['train_trades_samples_perct'],
            'train_bestVal_custom_metric_pnl': arrays['train_bestVal_custom_metric_pnl'],

            # Validation
            'winrates_val_by_fold': arrays['winrates_val'],
            'nb_trades_val_by_fold': arrays['nb_trades_val'],
            'nb_samples_val_by_fold': arrays['nb_samples_val'],
            'tp_val_by_fold': arrays['tp_val'],
            'fp_val_by_fold': arrays['fp_val'],
            'X_val_by_fold': arrays['X_val_by_fold'],
            'model_by_fold': arrays['model_by_fold'],

            'best_thresh_fold': arrays['best_thresh_fold'],
            'best_iteration_fold': arrays['best_iteration_fold'],
            'calibrator': arrays['calibrator'],

            'scores_val_by_fold': arrays['scores_val'],
            'class0_raw_data_val_by_fold': arrays['class0_raw_data_val_by_fold'],
            'class1_raw_data_val_by_fold': arrays['class1_raw_data_val_by_fold'],
            'winrate_raw_data_val_by_fold': arrays['winrate_raw_data_val_by_fold'],
            'val_pred_proba_log_odds': arrays['val_pred_proba_log_odds'],
            'val_trades_samples_perct': arrays['val_trades_samples_perct'],
            'val_bestVal_custom_metric_pnl': arrays['val_bestVal_custom_metric_pnl'],

            'brier_val':arrays['brier_val'],
            'relative_brier_val':arrays['relative_brier_val'],



            'mean_val_score': mean_val_score,
            'std_val_score': std_val_score,
            'perctDiff_winrateRatio_train_val': arrays['perctDiff_winrateRatio_train_val'],
            'perctDiff_ratioTradeSample_train_val': arrays['perctDiff_ratioTradeSample_train_val'],
            'metrics': final_metrics,
            'fold_stats': all_fold_stats,
        }

        return results

    except Exception as e:
        print(f"Erreur dans calculate_final_results: {str(e)}")
        raise


def handle_exception(e):
    """
    Gère les exceptions de manière uniforme.

    Args:
        e (Exception): L'exception à traiter
    """
    print(f"\nErreur détaillée dans la validation croisée:")
    print(f"Type d'erreur: {type(e).__name__}")
    print(f"Message: {str(e)}")

    import traceback
    traceback.print_exc()


def cleanup_gpu_memory(data):
    """
    Nettoie la mémoire GPU.

    Args:
        data_gpu (dict): Dictionnaire contenant les données GPU à nettoyer
    """
    print("\nNettoyage de la mémoire GPU")

    # Nettoyage des données GPU
    if isinstance(data, dict):
        for key in data:
            if isinstance(data[key], cp.ndarray):
                data[key] = None

    # Libération des blocs de mémoire
    cp.get_default_memory_pool().free_all_blocks()


def convert_metrics_to_numpy_safe(value, config):
    """Convertit en toute sécurité les métriques GPU/CPU en numpy"""
    try:
        if config['device_'] == 'cuda':
            if isinstance(value, cp.ndarray):
                return value.get()
            if isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], cp.ndarray):
                return [v.get() if isinstance(v, cp.ndarray) else v for v in value]
        return value
    except Exception as e:
        print(f"Erreur lors de la conversion: {str(e)}")
        return value


def update_metrics_and_arrays(metrics_dict, arrays, fold_results, fold_num, all_fold_stats, config):
    """
    Met à jour les métriques et les arrays GPU avec les résultats du fold courant.

    Args:
        metrics_dict (dict): Dictionnaire des métriques à mettre à jour
        arrays (dict): Dictionnaire des arrays GPU
        fold_results (dict): Résultats du fold courant
        fold_num (int): Numéro du fold
        all_fold_stats (dict): Statistiques de tous les folds
    """
    try:
        # Mise à jour des métriques du dictionnaire
        metrics_dict = update_fold_metrics(
            metrics_dict,
            fold_results['eval_metrics'],
            fold_results['train_metrics'],
            fold_num, config
        )

        # Stockage des statistiques du fold
        all_fold_stats[fold_num] = fold_results['fold_stats']

        # Mise à jour des arrays GPU
        # Entraînement
        arrays['winrates_train'][fold_num] = fold_results['fold_stats']['train_winrate']
        arrays['nb_trades_train'][fold_num] = fold_results['fold_stats']['train_trades']
        arrays['nb_samples_train'][fold_num] = fold_results['fold_stats']['train_samples']
        arrays['train_pred_proba_log_odds'][fold_num] = fold_results['fold_stats']['train_pred_proba_log_odds']
        arrays['train_trades_samples_perct'][fold_num] = fold_results['fold_stats']['train_trades_samples_perct']
        arrays['tp_train'][fold_num] = fold_results['train_metrics']['tp']
        arrays['fp_train'][fold_num] = fold_results['train_metrics']['fp']

        arrays['train_bestVal_custom_metric_pnl'][fold_num] = fold_results['train_metrics'][
            'train_bestVal_custom_metric_pnl']

        arrays['class0_raw_data_train_by_fold'][fold_num] = fold_results['fold_raw_data']['distributions']['train'].get(
            0, 0)
        arrays['class1_raw_data_train_by_fold'][fold_num] = fold_results['fold_raw_data']['distributions']['train'].get(
            1, 0)

        class0_train = arrays['class0_raw_data_train_by_fold'][fold_num]
        class1_train = arrays['class1_raw_data_train_by_fold'][fold_num]

        # Calcul du winrate pour l’entraînement pour les raw data
        if (class0_train + class1_train) > 0:
            winrate_train = class1_train / (class0_train + class1_train)
        else:
            winrate_train = 0  # Gestion de la division par 0

        arrays['winrate_raw_data_train_by_fold'][fold_num] = winrate_train

        # Validation
        arrays['winrates_val'][fold_num] = fold_results['fold_stats']['val_winrate']
        arrays['nb_trades_val'][fold_num] = fold_results['fold_stats']['val_trades']
        arrays['nb_samples_val'][fold_num] = fold_results['fold_stats']['val_samples']
        arrays['val_pred_proba_log_odds'][fold_num] = fold_results['fold_stats']['val_pred_proba_log_odds']
        arrays['val_trades_samples_perct'][fold_num] = fold_results['fold_stats']['val_trades_samples_perct']
        arrays['tp_val'][fold_num] = fold_results['eval_metrics']['tp']
        arrays['fp_val'][fold_num] = fold_results['eval_metrics']['fp']
        arrays['X_val_by_fold'][fold_num] = fold_results['eval_metrics']['X_val_cv']

        arrays['model_by_fold'][fold_num] = fold_results['current_model']

        if config["calibration_method"]!="no_sweet_no_calib":
            arrays['best_thresh_fold'][fold_num] = fold_results['best_thresh_fold']
            arrays['best_iteration_fold'][fold_num] = fold_results['best_iteration_fold']

            arrays['calibrator'][fold_num] = fold_results['calibrator']


        arrays['val_bestVal_custom_metric_pnl'][fold_num] = fold_results['eval_metrics'][
            'val_bestVal_custom_metric_pnl']
        arrays['class0_raw_data_val_by_fold'][fold_num] = fold_results['fold_raw_data']['distributions']['val'].get(0,
                                                                                                                    0)
        arrays['class1_raw_data_val_by_fold'][fold_num] = fold_results['fold_raw_data']['distributions']['val'].get(1,
                                                                                                                    0)

        arrays['brier_val'][fold_num] = fold_results["eval_metrics"]["brier"]
        arrays['relative_brier_val'][fold_num] = fold_results["eval_metrics"]["relative_brier"]

        class0_val = arrays['class0_raw_data_val_by_fold'][fold_num]
        class1_val = arrays['class1_raw_data_val_by_fold'][fold_num]

        arrays['perctDiff_winrateRatio_train_val'][fold_num] = fold_results['fold_stats'][
            'perctDiff_winrateRatio_train_val']
        arrays['perctDiff_ratioTradeSample_train_val'][fold_num] = fold_results['fold_stats'][
            'perctDiff_ratioTradeSample_train_val']
        # Calcul du winrate pour la validation pour les raw data
        if (class0_val + class1_val) > 0:
            winrate_val = class1_val / (class0_val + class1_val)
        else:
            winrate_val = 0  # Gestion de la division par 0

        arrays['winrate_raw_data_val_by_fold'][fold_num] = winrate_val

        # Vérification optionnelle des mises à jour
        """
        verify_updates = False  # Mettre à True pour le débogage
        if verify_updates:
            arrays_to_check = {
                'winrates': arrays['winrates'],
                'nb_trades': arrays['nb_trades'],
                'tp_val': arrays['tp_val'],
                'scores_val': arrays['scores_val']
            }
            for name, arr in arrays_to_check.items():
                print(f"{name}: sum={cp.sum(arr)}, mean={cp.mean(arr)}")
        """
        return metrics_dict

    except Exception as e:
        print(f"Erreur dans update_metrics_and_arrays: {str(e)}")
        raise


# Fonction pour calculer et récupérer les métriques
def get_raw_metrics(cv, X_train, X_train_full, y_train_label, config, data, nb_split_tscv, is_log_enabled,
                    df_init_candles):
    """
    Calcul des métriques ou récupération des métriques depuis le cache global.
    """
    if "raw_metrics_byFold" in raw_metrics_cache:
        return raw_metrics_cache["raw_metrics_byFold"]

    print("#############################get_raw_metrics########################################## ")

    raw_metrics_byFold = []
    for fold_num, (train_pos, val_pos) in enumerate(cv.split(X_train)):
        fold_raw_data = prepare_fold_data(X_train, y_train_label, train_pos, val_pos)

        X_train_cv, X_train_cv_pd, Y_train_cv, X_val_cv, X_val_cv_pd, y_val_cv, y_pnl_data_train_cv, y_pnl_data_val_cv, \
            = prepare_dataSplit_cv_train_val(
            config, data, train_pos, val_pos)

        raw_metrics = compute_raw_train_dist(
            X_train_full=X_train_full,
            X_train_cv_pd=X_train_cv_pd,
            X_val_cv_pd=X_val_cv_pd,
            fold_num=fold_num,
            nb_split_tscv=nb_split_tscv,
            fold_raw_data=fold_raw_data,
            is_log_enabled=is_log_enabled,
            df_init_candles=df_init_candles
        )
        raw_metrics_byFold.append(raw_metrics)

    # Stocker dans le cache
    raw_metrics_cache["raw_metrics_byFold"] = raw_metrics_byFold
    return raw_metrics_byFold


raw_metrics_cache = {}


def run_cross_validation(X_train=None, X_train_full=None, y_train_label=None, y_pnl_data_train=None,
                         df_init_candles=None, trial=None, params_optuna=None,
                         other_params=None, cv=None, nb_split_tscv=None,
                         model=None, is_log_enabled=False, config=None, **kwargs):
    """
    Validation croisée unifiée pour XGBoost et CatBoost.

    Args:
        framework (str): 'xgboost' ou 'lightgbm'
        kwargs: Paramètres spécifiques au framework (num_boost_round pour XGBoost, etc.)
    """
    try:
        high_corr_columns_used4pca = other_params['high_corr_columns_used4pca']
        colomnsList_with_vif_stat_without_pca_ = other_params['colomnsList_with_vif_stat_without_pca_']
        print(f"\n📊 Nombre total de features (colonnes VIF et stats + (optionnel) colonnes choisies pour PCA si activé) : {len(X_train.columns)}")
        print(f"   → {list(X_train.columns)}\n")

        print(
            f"🧩 Nombre de features HORS PCA (colomnsList_with_vif_stat_without_pca_) : {len(colomnsList_with_vif_stat_without_pca_)}")
        print(f"   → {colomnsList_with_vif_stat_without_pca_}\n")

        print(
            f"  et nombre de features utilsé pour le PCA (high_corr_columns_used4pca) : {len(high_corr_columns_used4pca)}")
        print(f"   → {high_corr_columns_used4pca}\n")

        # Vérification finale
        declared_cols = set(high_corr_columns_used4pca).union(colomnsList_with_vif_stat_without_pca_)
        print(f"✅ Total features déclaré = {len(declared_cols)} colonnes (HORS PCA + AVEC COL FOR PCA)")

        # Vérifications communes
        # validate_inputs(X_train, y_train_label)

        # Initialisation des métriques et arrays
        metrics_dict = initialize_metrics_dict(nb_split_tscv, config)

        # Préparation données GPU - Interface commune
        data_4cv = prepare_data(X_train, y_train_label, y_pnl_data_train=y_pnl_data_train, config=config)
        other_params['data_4cv']=data_4cv
        # Sélection du processor de fold selon le framework
        fold_processor = select_fold_processor(model)
        arrays = initialize_arrays(nb_split_tscv, config, len(X_train))
        # Mise à jour des métriques et statistiques
        all_fold_stats = {}

        # Utilisation d'une variable globale

        raw_metrics_byFold = get_raw_metrics(cv, X_train, X_train_full, y_train_label, config, data_4cv, nb_split_tscv,
                                             is_log_enabled, df_init_candles)

        for fold_num, (train_pos, val_pos) in enumerate(cv.split(X_train)):
            validate_fold_indices(train_pos, val_pos)

            # Préparation des données du fold
            fold_raw_data = prepare_fold_data(X_train, y_train_label, train_pos, val_pos)

            # Log des informations du fold
            # if is_log_enabled:
            #   log_fold_info(fold_num, nb_split_tscv, X_train_full, fold_data)

            # Traitement du fold avec le processor approprié
            fold_results = fold_processor(
                train_pos=train_pos,
                val_pos=val_pos,
                params_optuna=params_optuna,
                data_4cv=data_4cv,
                other_params=other_params,
                is_log_enabled=is_log_enabled,
                config=config,
                fold_num=fold_num,
            )
            fold_results['fold_raw_data']=fold_raw_data
            update_metrics_and_arrays(metrics_dict, arrays, fold_results, fold_num, all_fold_stats, config)
        # Calcul des résultats finaux
        results = calculate_final_results(
            metrics_dict, arrays, all_fold_stats, nb_split_tscv, config)

        # model_lastFold = fold_results['current_model']
        # Sauvegarder l'état du modèle sous forme de chaîne
        #  model_lastFold_string = model_lastFold.model_to_string()
        # trial.set_user_attr('model_lastFold_string', model_lastFold_string)

        # Sauvegarder les paramètres séparément
        #        model_lastFold_params = model_lastFold.params
        #       trial.set_user_attr('model_lastFold_params', model_lastFold_params)

        #       print("results[model_lastFold]: ",model_lastFold)
        #        print("results['model_lastFold]: ",model_lastFold.params)

        # if config['device_'] == 'cuda':
        #     print(f"\nMémoire GPU finale: {cp.get_default_memory_pool().used_bytes() / 1024 ** 2:.2f} MB")
        # else:
        #     print("\nMode CPU - pas de tracking mémoire GPU")

        return results, raw_metrics_byFold

    except Exception as e:
        handle_exception(e)
        raise
    finally:
        if config['device_'] != 'cpu' and data_4cv is not None:
            cleanup_gpu_memory(data_4cv)


def prepare_data(X_train=None, y_train_label=None, y_pnl_data_train=None, config=None):
    """Prépare les données pour l'entraînement sur CPU ou GPU.

    Args:
        X_train: DataFrame contenant les features
        y_train_label: Series ou array contenant les labels
        config: dict contenant la configuration avec la clé 'device_'

    Returns:
        dict: Dictionnaire contenant les données préparées avec les clés
             'X_train_no99_fullRange' et 'y_train_no99_fullRange'
    """
    # print(type(X_train))

    if config['device_'] == 'cpu':
        X_processed = X_train  # .values
        y_processed = y_train_label  # .values
        trade_pnl_data_processed = y_pnl_data_train  # .values

    else:
        X_processed = cp.asarray(X_train.values, dtype=cp.float32)
        # Vérifie le type de y_train_label avant conversion
        y_values = y_train_label.values if isinstance(y_train_label, pd.Series) else y_train_label
        y_processed = cp.asarray(y_values, dtype=cp.int32)
        y_valuesPnl = y_pnl_data_train.values if isinstance(y_pnl_data_train, pd.Series) else y_pnl_data_train
        trade_pnl_data_processed = cp.asarray(y_valuesPnl, dtype=cp.int32)

    # print(type(X_processed))
    # print("OOOOOO")
    return {
        'X_train_no99_fullRange': X_processed,
        'y_train_no99_fullRange': y_processed,
        'X_train_no99_fullRange_pd': X_train,
        'y_train_no99_fullRange_pd': y_train_label,
        'y_pnl_data_train_no99_fullRange': trade_pnl_data_processed,
        'y_train_trade_pnl_no99_fullRange_pd': y_pnl_data_train,
    }




def apply_scalingAndOrPCA(other_params, config, X_train_cv, X_val_cv):
    """
    Applique le scaling et éventuellement la PCA sur les données déjà splitées.

    Args:
        X_train_cv (DataFrame): Données d'entraînement (features uniquement)
        X_val_cv (DataFrame): Données de validation (features uniquement)
        other_params (dict): Doit contenir les clés :
            - 'high_corr_columns_used4pca'
            - 'colomnsList_with_vif_stat_without_pca_'
        config (dict): Configuration globale du modèle

    Returns:
        tuple: (X_train_cv, X_val_cv) après scaling + PCA
    """

    # 🔍 Récupération des colonnes
    high_corr_columns_used4pca = other_params['high_corr_columns_used4pca']
    columns_without_pca = other_params['colomnsList_with_vif_stat_without_pca_']

    all_train_cols = set(X_train_cv.columns)
    declared_cols = set(high_corr_columns_used4pca).union(columns_without_pca)
    if all_train_cols != declared_cols:
        raise ValueError(
            "❌ Les colonnes déclarées (PCA + hors PCA) ne correspondent pas exactement "
            "aux colonnes de X_train_cv. Vérifier la config."
        )

    # ⚙️ Scaling
    chosen_scaler = config.get('scaler_choice', scalerChoice.SCALER_ROBUST)
    if chosen_scaler != scalerChoice.SCALER_DISABLE:
      #  print(f"\n⚙️  Scaler activé : {chosen_scaler}")
        X_train_cv, X_val_cv, scaler,scaler_params_ret = apply_scaling(
            X_train_cv,
            X_val_cv,
            save_path=None,
            chosen_scaler=chosen_scaler
        )
    else:
        print("\n⏭️  Scaler désactivé (SCALER_DISABLE)")

    # 🎯 PCA (si applicable)
    if high_corr_columns_used4pca:
        nb_pca = config.get('nb_pca', 1)
        from sklearn.decomposition import PCA
        pca_model = PCA(n_components=nb_pca)

        pca_model.fit(X_train_cv[high_corr_columns_used4pca])
        pca_train = pca_model.transform(X_train_cv[high_corr_columns_used4pca])
        pca_val = pca_model.transform(X_val_cv[high_corr_columns_used4pca])
        pca_cols = [f'pca_{i + 1}' for i in range(nb_pca)]

        for i, col_name in enumerate(pca_cols):
            X_train_cv[col_name] = pca_train[:, i]
            X_val_cv[col_name] = pca_val[:, i]

        X_train_cv.drop(columns=high_corr_columns_used4pca, inplace=True)
        X_val_cv.drop(columns=high_corr_columns_used4pca, inplace=True)
    else:
        pca_model=None
        #print("\n⏭️  Aucune colonne spécifiée pour la PCA (liste vide)")

    # print(X_train_cv.head(20))
    # print(X_val_cv.head(20))
    return X_train_cv, X_val_cv,scaler,pca_model


def select_fold_processor(model: None):
    """Sélectionne le processor approprié selon le framework"""
    processors = {
        modelType.XGB: process_cv_fold_xgboost,
        modelType.LGBM: process_cv_fold_lightgbm,
        modelType.RF: process_cv_fold_randomforest_model,
        modelType.XGBRF: process_cv_fold_randomforest_model,
        modelType.SVC: process_cv_fold_svc_model,
        # modelType.CATBOOST: process_cv_fold_catboost
    }
    if model not in processors:
        raise ValueError(f"Framework non supporté: {model}")
    return processors[model]

def process_cv_fold_xgboost(
    fold_num=0,
    train_pos=None,
    val_pos=None,
    params_optuna=None,
    data_4cv=None,
    other_params=None,
    is_log_enabled=False,
    config=None,
    log_evaluation=0
):
    #print("______ process_cv_fold_xgboost______")
    try:
        # Vérif que data est dispo
        if data_4cv is None:
            raise ValueError("❌ 'data' est None dans process_cv_fold_xgboost")

        # 1️⃣ Split des données train/val
        if train_pos is None and val_pos is None:
            # ➕ Données déjà préparées (pas de cross-validation)
            required_keys = [
                "X_train", "X_test",
                "y_train_label", "y_test_label",
                "y_pnl_data_train", "y_pnl_data_test"
            ]
            for key in required_keys:
                if key not in data_4cv:
                    raise KeyError(f"❌ Clé manquante dans data : {key}")

            X_train_cv = data_4cv["X_train"]
            X_val_cv = data_4cv["X_test"]
            Y_train_cv = data_4cv["y_train_label"]
            y_val_cv = data_4cv["y_test_label"]
            y_pnl_data_train_cv = data_4cv["y_pnl_data_train"]
            y_pnl_data_val_cv = data_4cv["y_pnl_data_test"]
        else:
            # 📆 Split custom basé sur train_pos/val_pos (CV temporelle)
            (X_train_cv,X_train_cv_pd,Y_train_cv,X_val_cv,X_val_cv_pd,y_val_cv,
             y_pnl_data_train_cv,y_pnl_data_val_cv) = prepare_dataSplit_cv_train_val(
                 config, data_4cv, train_pos, val_pos)

        # 2️⃣ Scaling/PCA
        X_train_cv, X_val_cv,scaler,pca_model= apply_scalingAndOrPCA(other_params, config, X_train_cv, X_val_cv)

        # 3️⃣ Stats
        fold_stats_current = {
            **calculate_fold_stats(Y_train_cv, "train", config),
            **calculate_fold_stats(y_val_cv, "val", config)
        }

        # 4️⃣ Entraînement & évaluation
        fold_results = train_and_evaluate_xgb_model(
            X_train_cv=X_train_cv,
            X_val_cv=X_val_cv,
            Y_train_cv=Y_train_cv,
            y_val_cv=y_val_cv,
            y_pnl_data_train_cv=y_pnl_data_train_cv,
            y_pnl_data_val_cv_OrTest=y_pnl_data_val_cv,
            params_optuna=params_optuna,
            other_params=other_params,
            config=config,
            fold_num=fold_num,
            fold_stats_current=fold_stats_current,
            train_pos=train_pos,
            val_pos=val_pos,
            log_evaluation=log_evaluation,
        )

        if (val_pos is None or train_pos is None): #pas de cv on travail sur un df entier
            fold_results["x_train_vifStatPCA_trainFinal_noCV"]=X_train_cv
            fold_results["Y_train_trainFinal_noCV"]=Y_train_cv

            fold_results["scaler"]=scaler
            fold_results["pca_model"]=pca_model

        return fold_results

    except Exception as e:
        print("\n❌ Erreur dans process_cv_fold_xgboost :")
        print(f"Type: {type(e).__name__}")
        print(f"Message: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
def process_cv_fold_lightgbm(
    fold_num=0,
    train_pos=None,
    val_pos=None,
    params_optuna=None,
    data_4cv=None,
    other_params=None,
    is_log_enabled=False,
    config=None,
    log_evaluation=0
):
   # print("______ process_cv_fold_lightgbm______")
    try:
        # Vérif que data est dispo
        if data_4cv is None:
            raise ValueError("❌ 'data' est None dans process_cv_fold_lightgbm")

        # 1️⃣ Split des données train/val
        if train_pos is None and val_pos is None:
            # ➕ Données déjà préparées (pas de cross-validation)
            required_keys = [
                "X_train", "X_test",
                "y_train_label", "y_test_label",
                "y_pnl_data_train", "y_pnl_data_test"
            ]
            for key in required_keys:
                if key not in data_4cv:
                    raise KeyError(f"❌ Clé manquante dans data : {key}")

            X_train_cv = data_4cv["X_train"]
            X_val_cv = data_4cv["X_test"]
            Y_train_cv = data_4cv["y_train_label"]
            y_val_cv = data_4cv["y_test_label"]
            y_pnl_data_train_cv = data_4cv["y_pnl_data_train"]
            y_pnl_data_val_cv = data_4cv["y_pnl_data_test"]
        else:
            # 📆 Split custom basé sur train_pos/val_pos (CV temporelle)
            (X_train_cv,X_train_cv_pd,Y_train_cv,X_val_cv,X_val_cv_pd,y_val_cv,
             y_pnl_data_train_cv,y_pnl_data_val_cv) = prepare_dataSplit_cv_train_val(
                 config, data_4cv, train_pos, val_pos)

        # 2️⃣ Scaling/PCA
        X_train_cv, X_val_cv, scaler, pca_model = apply_scalingAndOrPCA(other_params, config, X_train_cv, X_val_cv)

        # # Juste les noms de colonnes pour débogage
        # print("\n===== DÉMARRAGE DE LA FONCTION : train_and_evaluate_lightgbm_model =====")
        # print(f"X_train_cv : {len(X_train_cv.columns)} colonnes -> {list(X_train_cv.columns)}")
        # print(f"X_val_cv   : {len(X_val_cv.columns)} colonnes -> {list(X_val_cv.columns)}")
        # print("========================================================================\n")

        # 3️⃣ Stats
        fold_stats_current = {
            **calculate_fold_stats(Y_train_cv, "train", config),
            **calculate_fold_stats(y_val_cv, "val", config)
        }

        # 4️⃣ Entraînement & évaluation
        fold_results = train_and_evaluate_lightgbm_model(
            X_train_cv=X_train_cv,
            X_val_cv=X_val_cv,
            y_train_cv=Y_train_cv,
            y_val_cv=y_val_cv,
            y_pnl_data_train_cv=y_pnl_data_train_cv,
            y_pnl_data_val_cv_OrTest=y_pnl_data_val_cv,
            params_optuna=params_optuna,
            other_params=other_params,
            config=config,
            fold_num=fold_num,
            fold_stats_current=fold_stats_current,
            train_pos=train_pos,
            val_pos=val_pos,
            log_evaluation=log_evaluation,
        )

        if (val_pos is None or train_pos is None): #pas de cv on travail sur un df entier
            fold_results["x_train_vifStatPCA_trainFinal_noCV"]=X_train_cv
            fold_results["scaler"]=scaler
            fold_results["pca_model"]=pca_model


        return fold_results

    except Exception as e:
        print("\n❌ Erreur dans process_cv_fold_lightgbm :")
        print(f"Type: {type(e).__name__}")
        print(f"Message: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def process_cv_fold_randomforest_model(
    fold_num=0,
    train_pos=None,
    val_pos=None,
    params_optuna=None,
    data_4cv=None,
    other_params=None,
    is_log_enabled=False,
    config=None,
    log_evaluation=0
):
   # print("______ process_cv_fold_randomforest_model______")
    try:
        # Vérif que data est dispo
        if data_4cv is None:
            raise ValueError("❌ 'data' est None dans process_cv_fold_randomforest_model")

        # 1️⃣ Split des données train/val
        if train_pos is None and val_pos is None:
            # ➕ Données déjà préparées (pas de cross-validation)
            required_keys = [
                "X_train", "X_test",
                "y_train_label", "y_test_label",
                "y_pnl_data_train", "y_pnl_data_test"
            ]
            for key in required_keys:
                if key not in data_4cv:
                    raise KeyError(f"❌ Clé manquante dans data : {key}")

            X_train_cv = data_4cv["X_train"]
            X_val_cv = data_4cv["X_test"]
            Y_train_cv = data_4cv["y_train_label"]
            y_val_cv = data_4cv["y_test_label"]
            y_pnl_data_train_cv = data_4cv["y_pnl_data_train"]
            y_pnl_data_val_cv = data_4cv["y_pnl_data_test"]
        else:
            # 📆 Split custom basé sur train_pos/val_pos (CV temporelle)
            (X_train_cv,X_train_cv_pd,Y_train_cv,X_val_cv,X_val_cv_pd,y_val_cv,
             y_pnl_data_train_cv,y_pnl_data_val_cv) = prepare_dataSplit_cv_train_val(
                 config, data_4cv, train_pos, val_pos)

        # 2️⃣ Scaling/PCA
        X_train_cv, X_val_cv, scaler, pca_model = apply_scalingAndOrPCA(other_params, config, X_train_cv, X_val_cv)

        # 3️⃣ Stats
        fold_stats_current = {
            **calculate_fold_stats(Y_train_cv, "train", config),
            **calculate_fold_stats(y_val_cv, "val", config)
        }

        # 4️⃣ Entraînement & évaluation
        fold_results = train_and_evaluate_randomforest_model(
            X_train_cv=X_train_cv,
            X_val_cv=X_val_cv,
            y_train_cv=Y_train_cv,
            y_val_cv=y_val_cv,
            y_pnl_data_train_cv=y_pnl_data_train_cv,
            y_pnl_data_val_cv_OrTest=y_pnl_data_val_cv,
            params_optuna=params_optuna,
            other_params=other_params,
            config=config,
            fold_num=fold_num,
            fold_stats_current=fold_stats_current,
            train_pos=train_pos,
            val_pos=val_pos,
            log_evaluation=log_evaluation,
        )

        if (val_pos is None or train_pos is None): #pas de cv on travail sur un df entier
            fold_results["x_train_vifStatPCA_trainFinal_noCV"]=X_train_cv
            fold_results["scaler"]=scaler
            fold_results["pca_model"]=pca_model

        return fold_results

    except Exception as e:
        print("\n❌ Erreur dans process_cv_fold_randomforest_model :")
        print(f"Type: {type(e).__name__}")
        print(f"Message: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def process_cv_fold_svc_model(
    fold_num=0,
    train_pos=None,
    val_pos=None,
    params_optuna=None,
    data_4cv=None,
    other_params=None,
    is_log_enabled=False,
    config=None,
    log_evaluation=0
):
   # print("______ process_cv_fold_svc_model______")
    try:
        # Vérif que data est dispo
        if data_4cv is None:
            raise ValueError("❌ 'data' est None dans process_cv_fold_svc_model")

        # 1️⃣ Split des données train/val
        if train_pos is None and val_pos is None:
            # ➕ Données déjà préparées (pas de cross-validation)
            required_keys = [
                "X_train", "X_test",
                "y_train_label", "y_test_label",
                "y_pnl_data_train", "y_pnl_data_test"
            ]
            for key in required_keys:
                if key not in data_4cv:
                    raise KeyError(f"❌ Clé manquante dans data : {key}")

            X_train_cv = data_4cv["X_train"]
            X_val_cv = data_4cv["X_test"]
            Y_train_cv = data_4cv["y_train_label"]
            y_val_cv = data_4cv["y_test_label"]
            y_pnl_data_train_cv = data_4cv["y_pnl_data_train"]
            y_pnl_data_val_cv = data_4cv["y_pnl_data_test"]
        else:
            # 📆 Split custom basé sur train_pos/val_pos (CV temporelle)
            (X_train_cv,X_train_cv_pd,Y_train_cv,X_val_cv,X_val_cv_pd,y_val_cv,
             y_pnl_data_train_cv,y_pnl_data_val_cv) = prepare_dataSplit_cv_train_val(
                 config, data_4cv, train_pos, val_pos)

        # 2️⃣ Scaling/PCA
        X_train_cv, X_val_cv, scaler, pca_model = apply_scalingAndOrPCA(other_params, config, X_train_cv, X_val_cv)

        # 3️⃣ Stats
        fold_stats_current = {
            **calculate_fold_stats(Y_train_cv, "train", config),
            **calculate_fold_stats(y_val_cv, "val", config)
        }

        # 4️⃣ Entraînement & évaluation
        fold_results = train_and_evaluate_svc_model(
            X_train_cv=X_train_cv,
            X_val_cv=X_val_cv,
            y_train_cv=Y_train_cv,
            y_val_cv=y_val_cv,
            y_pnl_data_train_cv=y_pnl_data_train_cv,
            y_pnl_data_val_cv_OrTest=y_pnl_data_val_cv,
            params_optuna=params_optuna,
            config=config,
            fold_num=fold_num,
            fold_stats_current=fold_stats_current,
            train_pos=train_pos,
            val_pos=val_pos,
            log_evaluation=log_evaluation,
        )

        if (val_pos is None or train_pos is None): #pas de cv on travail sur un df entier
            fold_results["x_train_vifStatPCA_trainFinal_noCV"]=X_train_cv
            fold_results["scaler"]=scaler
            fold_results["pca_model"]=pca_model

        return fold_results

    except Exception as e:
        print("\n❌ Erreur dans process_cv_fold_svc_model :")
        print(f"Type: {type(e).__name__}")
        print(f"Message: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def validate_inputs(X_train, y_train_label):
    """Validation commune des entrées"""
    print(f"Shape X_train: {X_train.shape}")
    print(f"Shape y_train_label: {y_train_label.shape}")
    print(f"Index y_train_label: {y_train_label.index.min()} à {y_train_label.index.max()}")


def initialize_arrays(nb_split_tscv, config, len):
    """
    Initialise les tableaux nécessaires pour stocker les métriques
    de la cross-validation, en GPU ou CPU suivant config['device_'].

    Args:
        nb_split_tscv (int): Nombre de splits de la CV (K-Fold, TimeSeriesSplit, etc.).
        config (dict): Configuration, contenant notamment 'device_' (='cuda' ou 'cpu').

    Returns:
        dict: Dictionnaire contenant les tableaux de zéros pour chaque métrique.
              Ex. 'winrates', 'nb_trades', 'scores_train', etc.
    """
    # Choix de la bibliothèque en fonction du device
    if config['device_'] == 'cuda':
        import cupy as cp
        xp = cp
        int_dtype = cp.int32
    else:
        import numpy as np
        xp = np
        int_dtype = np.int32

    # Création des tableaux de zéros
    return {
        # Validation
        'winrates_val': xp.zeros(nb_split_tscv, dtype=xp.float32),
        'nb_trades_val': xp.zeros(nb_split_tscv, dtype=int_dtype),
        'nb_samples_val': xp.zeros(nb_split_tscv, dtype=int_dtype),
        'tp_val': xp.zeros(nb_split_tscv, dtype=xp.float32),
        'fp_val': xp.zeros(nb_split_tscv, dtype=xp.float32),
        'X_val_by_fold' : [None] * nb_split_tscv,
        'scores_val': xp.zeros(nb_split_tscv, dtype=xp.float32),
        'class0_raw_data_val_by_fold': xp.zeros(nb_split_tscv, dtype=xp.float32),
        'class1_raw_data_val_by_fold': xp.zeros(nb_split_tscv, dtype=xp.float32),
        'winrate_raw_data_val_by_fold': xp.zeros(nb_split_tscv, dtype=xp.float32),
        'val_pred_proba_log_odds': [None] * nb_split_tscv,
        'val_trades_samples_perct': xp.zeros(nb_split_tscv, dtype=xp.float32),
        'val_bestVal_custom_metric_pnl': xp.zeros(nb_split_tscv, dtype=xp.float32),
        'brier_val':xp.zeros(nb_split_tscv, dtype=xp.float32),
        'relative_brier_val':xp.zeros(nb_split_tscv, dtype=xp.float32),

        'model_by_fold':[None] * nb_split_tscv,

        'best_thresh_fold': [np.nan] * nb_split_tscv,
        'best_iteration_fold': [np.nan] * nb_split_tscv,
        'calibrator': [None] * nb_split_tscv,

        # Entraînement
        'winrates_train': xp.zeros(nb_split_tscv, dtype=xp.float32),
        'nb_trades_train': xp.zeros(nb_split_tscv, dtype=int_dtype),
        'nb_samples_train': xp.zeros(nb_split_tscv, dtype=int_dtype),
        'tp_train': xp.zeros(nb_split_tscv, dtype=xp.float32),
        'fp_train': xp.zeros(nb_split_tscv, dtype=xp.float32),
        'scores_train': xp.zeros(nb_split_tscv, dtype=xp.float32),
        'class0_raw_data_train_by_fold': xp.zeros(nb_split_tscv, dtype=xp.float32),
        'class1_raw_data_train_by_fold': xp.zeros(nb_split_tscv, dtype=xp.float32),
        'winrate_raw_data_train_by_fold': xp.zeros(nb_split_tscv, dtype=xp.float32),
        'train_pred_proba_log_odds': [None] * nb_split_tscv,
        'train_trades_samples_perct': xp.zeros(nb_split_tscv, dtype=xp.float32),
        'train_bestVal_custom_metric_pnl': xp.zeros(nb_split_tscv, dtype=xp.float32),

        'perctDiff_winrateRatio_train_val': xp.zeros(nb_split_tscv, dtype=xp.float32),
        'perctDiff_ratioTradeSample_train_val': xp.zeros(nb_split_tscv, dtype=xp.float32)
    }


def initialize_metrics_dict(nb_folds, config):
    """Initialise toutes les métriques nécessaires sur GPU ou CPU."""
    # Choisir la bibliothèque (NumPy ou CuPy)
    xp = cp if config['device_'] == 'cuda' else np

    return {
        # Métriques par fold - Validation
        'tp_val_by_fold': xp.zeros(nb_folds, dtype=xp.float32),
        'fp_val_by_fold': xp.zeros(nb_folds, dtype=xp.float32),
        'tn_val_by_fold': xp.zeros(nb_folds, dtype=xp.float32),
        'fn_val_by_fold': xp.zeros(nb_folds, dtype=xp.float32),
        'winrates_val_by_fold': xp.zeros(nb_folds, dtype=xp.float32),
        'nb_trades_val_by_fold': xp.zeros(nb_folds, dtype=xp.float32),
        'nb_samples_val_by_fold': xp.zeros(nb_folds, dtype=xp.float32),
        'class0_raw_data_val_by_fold': xp.array(nb_folds, dtype=xp.float32),
        'class1_raw_data_val_by_fold': xp.array(nb_folds, dtype=xp.float32),
        'winrate_raw_data_val_by_fold': xp.array(nb_folds, dtype=xp.float32),

        # Métriques par fold - Entraînement
        'tp_train_by_fold': xp.zeros(nb_folds, dtype=xp.float32),
        'fp_train_by_fold': xp.zeros(nb_folds, dtype=xp.float32),
        'tn_train_by_fold': xp.zeros(nb_folds, dtype=xp.float32),
        'fn_train_by_fold': xp.zeros(nb_folds, dtype=xp.float32),
        'winrates_train_by_fold': xp.zeros(nb_folds, dtype=xp.float32),
        'nb_trades_train_by_fold': xp.zeros(nb_folds, dtype=xp.float32),
        'nb_samples_train_by_fold': xp.zeros(nb_folds, dtype=xp.float32),
        'class0_raw_data_train_by_fold': xp.array(nb_folds, dtype=xp.float32),
        'class1_raw_data_train_by_fold': xp.array(nb_folds, dtype=xp.float32),
        'winrate_raw_data_train_by_fold': xp.array(nb_folds, dtype=xp.float32),

        # Totaux validation
        'total_tp_val': xp.array(0, dtype=xp.float32),
        'total_fp_val': xp.array(0, dtype=xp.float32),
        'total_tn_val': xp.array(0, dtype=xp.float32),
        'total_fn_val': xp.array(0, dtype=xp.float32),

        # Totaux entraînement
        'total_tp_train': xp.array(0, dtype=xp.float32),
        'total_fp_train': xp.array(0, dtype=xp.float32),
        'total_tn_train': xp.array(0, dtype=xp.float32),
        'total_fn_train': xp.array(0, dtype=xp.float32)
    }


def update_fold_metrics(metrics_dict, eval_metrics, train_metrics, fold_idx, config):
    """
    Met à jour toutes les métriques pour un fold sur GPU/CPU avec gestion des erreurs.

    Args:
        metrics_dict (dict): Dictionnaire contenant les métriques globales.
        eval_metrics (dict): Métriques de validation (tp, fp, tn, fn).
        train_metrics (dict): Métriques d'entraînement (tp, fp, tn, fn).
        fold_idx (int): Index du fold courant.
        config (dict): Configuration contenant l'information sur le device (cpu ou cuda).

    Returns:
        dict: Dictionnaire mis à jour avec les métriques du fold.
    """
    try:
        # Déterminer la bibliothèque (NumPy ou CuPy)
        xp = cp if config['device_'] == 'cuda' else np

        # Validation
        try:
            val_tp = xp.asarray(eval_metrics['tp'], dtype=xp.float32)
            val_fp = xp.asarray(eval_metrics['fp'], dtype=xp.float32)
            val_tn = xp.asarray(eval_metrics['tn'], dtype=xp.float32)
            val_fn = xp.asarray(eval_metrics['fn'], dtype=xp.float32)

            # metrics_dict['tp_val_by_fold'][fold_idx] = val_tp
            # metrics_dict['fp_val_by_fold'][fold_idx] = val_fp
            # metrics_dict['tn_val_by_fold'][fold_idx] = val_tn
            # metrics_dict['fn_val_by_fold'][fold_idx] = val_fn

            # total_trades = val_tp + val_fp
            # winrate = compute_winrate_safe(val_tp, total_trades, config)

            # metrics_dict['winrates_val_by_fold'][fold_idx] = winrate
            # metrics_dict['nb_trades_val_by_fold'][fold_idx] = total_trades
            # metrics_dict['nb_samples_val_by_fold'][fold_idx] = total_trades+val_tn+val_fn

            metrics_dict['total_tp_val'] += val_tp
            metrics_dict['total_fp_val'] += val_fp
            metrics_dict['total_tn_val'] += val_tn
            metrics_dict['total_fn_val'] += val_fn


        except KeyError as e:
            raise ValueError(f"Clé manquante dans eval_metrics: {e}")
        except Exception as e:
            raise RuntimeError(f"Erreur lors de la mise à jour des métriques de validation: {e}")

        # Entraînement
        try:
            train_tp = xp.asarray(train_metrics['tp'], dtype=xp.float32)
            train_fp = xp.asarray(train_metrics['fp'], dtype=xp.float32)
            train_tn = xp.asarray(train_metrics['tn'], dtype=xp.float32)
            train_fn = xp.asarray(train_metrics['fn'], dtype=xp.float32)

            # metrics_dict['tp_train_by_fold'][fold_idx] = train_tp
            # metrics_dict['fp_train_by_fold'][fold_idx] = train_fp
            # metrics_dict['tn_train_by_fold'][fold_idx] = train_tn
            # metrics_dict['fn_train_by_fold'][fold_idx] = train_fn

            metrics_dict['total_tp_train'] += train_tp
            metrics_dict['total_fp_train'] += train_fp
            metrics_dict['total_tn_train'] += train_tn
            metrics_dict['total_fn_train'] += train_fn
        except KeyError as e:
            raise ValueError(f"Clé manquante dans train_metrics: {e}")
        except Exception as e:
            raise RuntimeError(f"Erreur lors de la mise à jour des métriques d'entraînement: {e}")

        return metrics_dict

    except Exception as e:
        print(f"Erreur générale dans update_fold_metrics: {e}")
        raise


def report_trial_optuna(trial, best_trial, rfe_param, modele_param_optuna_range, selected_columns,
                        results_directory, config, bestResult_dict):
    best_trial_with_2_obj = best_trial.number + 1
    # Récupération des valeurs depuis trial.user_attrs
    total_tp_val = trial.user_attrs['total_tp_val']
    total_fp_val = trial.user_attrs['total_fp_val']
    total_tn_val = trial.user_attrs['total_tn_val']
    total_fn_val = trial.user_attrs['total_fn_val']
    weight_param = trial.user_attrs['weight_param']
    nb_split_tscv = trial.user_attrs['nb_split_tscv']
    mean_cv_score = trial.user_attrs['mean_cv_score']
    std_dev_score = trial.user_attrs['std_dev_score']
    std_penalty_factor = trial.user_attrs['std_penalty_factor']
    score_adjustedStd_val = trial.user_attrs['score_adjustedStd_val']
    train_pnl_perTrades = trial.user_attrs['train_pnl_perTrades']
    val_pnl_perTrades = trial.user_attrs['val_pnl_perTrades']
    pnl_perTrade_diff = trial.user_attrs['pnl_perTrade_diff']
    total_samples_val = trial.user_attrs['total_samples_val']
    n_trials_optuna = trial.user_attrs['n_trials_optuna']
    total_samples_val = trial.user_attrs['total_samples_val']
    cummulative_pnl_val = trial.user_attrs['cummulative_pnl_val']
    scores_ens_val_list = trial.user_attrs['scores_ens_val_list']
    tp_fp_diff_val = trial.user_attrs['tp_fp_diff_val']
    tp_fp_percentage_val = trial.user_attrs['tp_fp_percentage_val']
    win_rate = trial.user_attrs['win_rate_val']
    selected_feature_names = trial.user_attrs['selected_feature_names']
    rfe_param_value = trial.user_attrs['use_of_rfe_in_optuna']

    trial.set_user_attr('win_rate', win_rate)
    weight_split = trial.user_attrs['weight_split']
    nb_split_weight = trial.user_attrs['nb_split_weight']
    pnl_norm_objective = trial.user_attrs['pnl_norm_objective']
    # ecart_train_val = trial.user_attrs['ecart_train_val']
    # print(f"   ##Essai actuel: ")
    # print(
    #   f"    =>Objective 1, pnl_norm_objective : {pnl_norm_objective} avec weight_split {weight_split} nb_split_weight {nb_split_weight}")

    # print(
    #   f"     -score_adjustedStd_val : {score_adjustedStd_val:.2f} avec Moyenne des pnl des {nb_split_tscv} iterations : {mean_cv_score:.2f}, std_dev_score : {std_dev_score}, std_penalty_factor={std_penalty_factor}")
    # print(f"    =>Objective 2, pnl per trade: train {train_pnl_perTrades} // Val {val_pnl_perTrades} "
    #     f"donc diff val-train PNL per trade {pnl_perTrade_diff}\n"
    #     f"     ecart_train_val:{ecart_train_val}")

    if (rfe_param_value != rfe_param.NO_RFE):
        print(
            f"    =>Nombre de features sélectionnées par RFECVCV: {len(selected_feature_names)}, Noms des features: {', '.join(selected_feature_names)}")
    """
    print(f"    =>Principal métrique pour l'essai en cours :")
    print(f"     -Nombre de: TP (True Positives) : {total_tp_val}, FP (False Positives) : {total_fp_val}, "
          f"TN (True Negative) : {total_tn_val}, FN (False Negative) : {total_fn_val},")
    print(f"     -Pourcentage Winrate           : {win_rate:.2f}%")
    print(f"     -Pourcentage de trades(fp+tp)  : {tp_fp_percentage_val:.2f}%")
    print(f"     -Différence (TP - FP)          : {tp_fp_diff_val}")
    print(
        f"     -PNL                           : {cummulative_pnl_val}, original: (scores_ens_val_list: {scores_ens_val_list})")
    print(
        f"     -Nombre de d'échantillons      : {total_tp_val + total_fp_val + total_tn_val + total_fn_val} dont {total_tp_val + total_fp_val} trade pris")
    """
    if total_samples_val > 0:
        tp_percentage = (total_tp_val / total_samples_val) * 100
    else:
        tp_percentage = 0
    total_trades_val = total_tp_val + total_fp_val
    win_rate = total_tp_val / total_trades_val * 100 if total_trades_val > 0 else 0
    result_dict_trialOptuna = {
        "cummulative_pnl": cummulative_pnl_val,
        "win_rate_percentage": round(win_rate, 2),
        "scores_ens_val_list": scores_ens_val_list,
        "score_adjustedStd_val": score_adjustedStd_val,
        "std_dev_score": std_dev_score,
        "tp_fp_diff_val": tp_fp_diff_val,
        "total_trades_val": total_trades_val,
        "tp_percentage": round(tp_percentage, 3),
        "total_tp_val": total_tp_val,
        "total_fp_val": total_fp_val,
        "total_tn_val": total_tn_val,
        "total_fn_val": total_fn_val,
        "current_trial_number": trial.number + 1,
        "best_trial_with_2_Obj": best_trial_with_2_obj
    }

    def convert_to_serializable(obj):
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def convert_to_serializable_config(obj):
        """Convert non-serializable objects to a format suitable for JSON."""
        if isinstance(obj, model_custom_objective):
            return str(obj)  # or obj.name or obj.value depending on the enum or custom class
        try:
            json.dumps(obj)  # Try to serialize it
            return obj  # If no error, return the object itself
        except (TypeError, ValueError):
            return str(obj)  # If it's not serializable, convert it to string

    import time
    from contextlib import contextmanager

    def safe_file_replace(temp_filename, target_filename, max_retries=5, delay=1):
        """
        Tente de remplacer un fichier de manière sécurisée avec plusieurs essais.
        """
        for attempt in range(max_retries):
            try:
                # Tenter de supprimer le fichier cible s'il existe
                if os.path.exists(target_filename):
                    os.remove(target_filename)
                    time.sleep(0.1)  # Petit délai après la suppression

                # Tenter le remplacement
                os.rename(temp_filename, target_filename)
                return True
            except (PermissionError, OSError) as e:
                if attempt < max_retries - 1:
                    print(f"Tentative {attempt + 1} échouée: {str(e)}")
                    time.sleep(delay)
                else:
                    print(f"Échec final après {max_retries} tentatives: {str(e)}")
                    # Tenter une copie comme solution de repli
                    try:
                        shutil.copy2(temp_filename, target_filename)
                        os.remove(temp_filename)
                        return True
                    except Exception as copy_error:
                        print(f"Échec de la copie de secours: {str(copy_error)}")
                        return False

    def save_lightgbm_model(best_trial, model, save_dir):
        """
        Sauvegarde un modèle LightGBM dans un fichier.
        Si le fichier final_model.json existe déjà, il est supprimé avant d'être recréé.

        :param model: Modèle LightGBM à sauvegarder.
        :param save_dir: Répertoire de sauvegarde.
        :return: Chemin complet du fichier sauvegardé.
        """
        best_trial_number = best_trial.number + 1
        # Créer le répertoire si nécessaire
        os.makedirs(save_dir, exist_ok=True)

        # Nom du fichier
        model_file_path = os.path.join(save_dir, f"best_modellastFold_{best_trial_number}.json")

        # Supprimer le fichier existant si nécessaire
        if os.path.exists(model_file_path):
            try:
                os.remove(model_file_path)
                print(f"Fichier existant supprimé : {model_file_path}")
            except Exception as e:
                print(f"Erreur lors de la suppression du fichier existant : {e}")
                raise

        # Sauvegarde du modèle
        try:
            model.save_model(model_file_path)
            print(f"Modèle sauvegardé avec succès : {model_file_path}")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde du modèle : {e}")
            raise

        return model_file_path

    def save_trial_results(trial, result_dict_trialOptuna, config=None,
                           modele_param_optuna_range=None, weight_param=None, selected_columns=None,
                           save_dir="optuna_results",
                           result_file="optuna_results.json"):
        global _first_call_save_r_trialesults

        try:
            # Création du répertoire si nécessaire
            os.makedirs(save_dir, exist_ok=True)

            # Nettoyage au premier appel
            if _first_call_save_r_trialesults:
                if os.path.exists(save_dir):
                    for filename in os.listdir(save_dir):
                        file_path = os.path.join(save_dir, filename)
                        try:
                            if os.path.isfile(file_path):
                                os.remove(file_path)
                            elif os.path.isdir(file_path):
                                shutil.rmtree(file_path)
                        except Exception as e:
                            print(f'Attention: Échec suppression {file_path}: {e}')
                _first_call_save_r_trialesults = False

            result_file_path = os.path.join(save_dir, result_file)

            # Chargement des résultats existants
            results_data = {}
            if os.path.exists(result_file_path) and os.path.getsize(result_file_path) > 0:
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        with open(result_file_path, 'r') as f:
                            results_data = json.load(f)
                        break
                    except json.JSONDecodeError as e:
                        if attempt < max_retries - 1:
                            time.sleep(1)
                            continue
                        print(f"Erreur lecture résultats: {e}")
                        backup_file = f"{result_file_path}.bak_{int(time.time())}"
                        shutil.copy2(result_file_path, backup_file)
                        print(f"Backup créé: {backup_file}")
                    except Exception as e:
                        print(f"Erreur inattendue: {e}")
                        break

            # Mise à jour des données
            if 'selected_columns' not in results_data:
                results_data['selected_columns'] = selected_columns
            if 'config' not in results_data:
                results_data['config'] = {k: convert_to_serializable_config(v) for k, v in config.items()}
            if 'weight_param' not in results_data:
                results_data['weight_param'] = weight_param
            if 'modele_param_optuna_range' not in results_data:
                results_data['modele_param_optuna_range'] = modele_param_optuna_range

            # Ajout des résultats du nouveau trial
            results_data[f"trial_{trial.number + 1}"] = {
                "best_result": {k: convert_to_serializable(v) for k, v in result_dict_trialOptuna.items()},
                "params": {k: convert_to_serializable(v) for k, v in trial.params.items()},
                "other_params": {k: convert_to_serializable(v) for k, v in trial.user_attrs["other_params"].items()}
            }

            # Écriture sécurisée avec nom unique
            temp_filename = os.path.join(save_dir, f'temp_results_{int(time.time())}_{os.getpid()}.json')
            try:
                with open(temp_filename, 'w') as tf:
                    json.dump(results_data, tf, indent=4)

                # Remplacement sécurisé
                if not safe_file_replace(temp_filename, result_file_path):
                    raise Exception("Échec du remplacement du fichier")

            except Exception as e:
                print(f"Erreur lors de la sauvegarde: {e}")
                # Tentative de sauvegarde de secours
                backup_path = os.path.join(save_dir, f'backup_results_{int(time.time())}.json')
                try:
                    with open(backup_path, 'w') as bf:
                        json.dump(results_data, bf, indent=4)
                    print(f"Sauvegarde de secours créée: {backup_path}")
                except Exception as backup_error:
                    print(f"Échec de la sauvegarde de secours: {backup_error}")

            finally:
                # Nettoyage des fichiers temporaires
                if os.path.exists(temp_filename):
                    try:
                        os.remove(temp_filename)
                    except:
                        pass

        except Exception as e:
            print(f"Erreur globale dans save_trial_results: {e}")
            raise

    # print(f"   Config: {config}")

    # Appel de la fonction save_trial_results
    save_dir = os.path.join(results_directory, 'optuna_results')
    save_trial_results(
        trial,
        result_dict_trialOptuna,
        config=config,
        modele_param_optuna_range=modele_param_optuna_range, selected_columns=selected_columns,
        weight_param=weight_param,
        save_dir=save_dir,  # 'optuna_results' should be a string
        result_file="optuna_results.json"
    )

    # best_modellastFold_string =bestResult_dict['best_modellastFold_string']
    # best_modellastFold_params =bestResult_dict['best_modellastFold_params']

    # Recharger le modèle
    import lightgbm as lgb
    # best_modellastFold = lgb.Booster(model_str=best_modellastFold_string)

    # Assigner les paramètres récupérés au modèle
    # if best_modellastFold_params:
    #   best_modellastFold.params.update(best_modellastFold_params)

    # Vérification
    # print("report_trial_optuna best_modellastFold]: ", best_modellastFold)
    # print("report_trial_optuna best_modellastFold.params: ", best_modellastFold.params)

    # save_lightgbm_model(best_trial,best_modellastFold, save_dir)

    print(f"####{trial.number + 1}/{n_trials_optuna} Optuna results and model saved successfully.####")


def validate_and_update_params(trial_params, other_params, params_optuna):
    """
    Parcourt trial_params, met à jour les clés dans other_params et params_optuna,
    et affiche les résultats en colonnes. Affiche en vert les clés mises à jour et en violet celles non mises à jour,
    en distinguant les deux objets (other_params et params_optuna).

    Args:
        trial_params (dict): Dictionnaire contenant les paramètres du trial (Optuna).
        other_params (dict): Dictionnaire des poids du modèle.
        params_optuna (dict): Dictionnaire des paramètres du modèle.
    """
    from termcolor import colored

    # # Afficher les valeurs initiales
    # print("--- Avant mise à jour ---")
    # print("other_params:", other_params)
    # print("params_optuna:", params_optuna)

    updated_in_model_weight = []
    updated_in_params_optuna = []

    # Mettre à jour les valeurs et suivre les clés mises à jour
    for key, value in trial_params.items():
        if key in other_params:
            other_params[key] = value
            updated_in_model_weight.append(key)
        elif key in params_optuna:
            params_optuna[key] = value
            updated_in_params_optuna.append(key)

    # Vérifier les clés non mises à jour
    missing_in_model_weight = set(other_params.keys()) - set(updated_in_model_weight)
    missing_in_params_optuna = set(params_optuna.keys()) - set(updated_in_params_optuna)

    # Affichage des résultats pour other_params
    """
    print("\n--- Résultats pour other_params ---")
    print(f"{'Clé':<20}{'Statut':<20}")
    print("-" * 40)
    for key in other_params.keys():
        if key in updated_in_model_weight:
            status = colored("Mis à jour", "green")
        else:
            status = colored("Non mis à jour", "magenta")
        print(f"{key:<20}{status:<20}")

    # Affichage des résultats pour params_optuna
    print("\n--- Résultats pour params_optuna ---")
    print(f"{'Clé':<20}{'Statut':<20}")
    print("-" * 40)
    for key in params_optuna.keys():
        if key in updated_in_params_optuna:
            status = colored("Mis à jour", "green")
        else:
            status = colored("Non mis à jour", "magenta")
        print(f"{key:<20}{status:<20}")

    # Afficher les valeurs finales
    print("\n--- Après mise à jour ---")
    print("other_params:", other_params)
    print("params_optuna:", params_optuna)
    """


def callback_optuna(study, trial, optuna, study_optuna, rfe_param, config, results_directory):
    """
    Callback function for Optuna to monitor progress and log the best trial.
    """
    is_newBestTrial = False

    current_trial = trial.number
    print(f"\n {Fore.CYAN}Optuna Callback Current Trial {current_trial + 1}:{Style.RESET_ALL}")
    optuna_objective_type_value = trial.user_attrs.get('optuna_objective_type', optuna_doubleMetrics.DISABLE)
    cv_method = trial.user_attrs.get('cv_method', None)

    weightPareto_pnl_val = config.get('weightPareto_pnl_val', 0.6)
    weightPareto_pnl_diff = config.get('weightPareto_pnl_diff', 0.4)

    # Return early if the trial is not complete
    if trial.state != optuna.trial.TrialState.COMPLETE:
        return

    previous_completed_count = study.user_attrs.get('previous_completed_count', 0)
    current_completed_count = previous_completed_count + 1  # Puisque le trial actuel est complété
    study.set_user_attr('previous_completed_count', current_completed_count)

    if not current_completed_count:
        study.set_user_attr('bestResult_dict', {
            "best_optunaTrial_number": None,
            "best_pnl_val": None,
            "best_pnl_perTrade_diff": None,
            "best_params": None
        })
        print("No completed trials found.")
        return
    contraints_reached = True
    # Check if the optimization is single-objective or multi-objective
    if optuna_objective_type_value != optuna_doubleMetrics.DISABLE:
        completed_trials = study.get_trials(states=[optuna.trial.TrialState.COMPLETE])

        # Multi-objective optimization: custom selection of the best trial
        pareto_trials = study.best_trials
        weights = np.array([weightPareto_pnl_val, weightPareto_pnl_diff])

        if (optuna_objective_type_value == optuna_doubleMetrics.USE_DIST_TO_IDEAL):
            # Calculate distances for each Pareto front trial
            values_0 = np.array([t.values[0] for t in pareto_trials])
            values_1 = np.array([t.values[1] for t in pareto_trials])

            min_pnl, max_pnl = values_0.min(), values_0.max()
            min_pnl_diff, max_pnl_diff = values_1.min(), values_1.max()

            pnl_range = max_pnl - min_pnl or 1
            pnl_diff_range = max_pnl_diff - min_pnl_diff or 1

            pnl_normalized = (max_pnl - values_0) / pnl_range
            pnl_diff_normalized = (values_1 - min_pnl_diff) / pnl_diff_range

            distances = np.sqrt(
                (weightPareto_pnl_val * pnl_normalized) ** 2 +
                (weightPareto_pnl_diff * pnl_diff_normalized) ** 2
            )
            best_trial = pareto_trials[np.argmin(distances)]
        elif (optuna_objective_type_value == optuna_doubleMetrics.USE_WEIGHTED_AVG):
            # Weighted average calculation for custom selection
            df = pd.DataFrame({
                'PnL_val': [-t.values[0] for t in completed_trials],
                'pnl_perTrade_diff': [t.values[1] for t in completed_trials]
            })

            for col in df.columns:
                min_val = df[col].min()
                max_val = df[col].max()
                range_val = max_val - min_val or 1
                df[f'{col}_normalized'] = (df[col] - min_val) / range_val

            weighted_avg = df[['PnL_val_normalized', 'pnl_perTrade_diff_normalized']].dot(weights)
            best_trial = completed_trials[weighted_avg.idxmin()]
    else:  # (optuna_doubleMetrics ) nous utilisons le best trial d'optuna directement ici
        # Single-objective optimization: use best_trial directly
        try:
            best_trial = study.best_trial

            # Récupérer le numéro du best trial précédent depuis les user attributes
            previous_best_trial_number = study.user_attrs.get("previous_best_trial_number")

            # Comparer avec le best trial actuel
            if previous_best_trial_number != best_trial.number:
                print(f"Le meilleur essai a changé !")
                print(f"Ancien : {previous_best_trial_number}, Nouveau : {best_trial.number}")
                is_newBestTrial = True
                # Mettre à jour l'attribut utilisateur
                study.set_user_attr("previous_best_trial_number", best_trial.number)

        except ValueError as e:
            if "No feasible trials are completed yet" in str(e):
                print("no feasible trials are completed yet")
                # Utiliser le trial actuel pour l'affichage
                best_trial = trial
                contraints_reached = False
            else:
                raise e

    other_params = best_trial.user_attrs.get('other_params', None)
    params_optuna = best_trial.user_attrs.get('params_optuna', None)

    validate_and_update_params(best_trial.params, other_params, params_optuna)
    # print("other_params", other_params)
    # print("params_optuna", params_optuna)
    # Create the result dictionary

    # Récupérer le modèle sauvegardé
    # best_modellastFold_string = trial.user_attrs.get('model_lastFold_string', None)
    # best_modellastFold_params = trial.user_attrs.get('model_lastFold_params', None)

    # Recharger le modèle
    import lightgbm as lgb
    # best_modellastFold = lgb.Booster(model_str=best_modellastFold_string)

    # Assigner les paramètres récupérés au modèle
    # if best_modellastFold_params:
    #   best_modellastFold.params.update(best_modellastFold_params)

    # Vérification
    # print("in callback_optuna best_model best_modellastFold]: ", best_modellastFold)
    # print("in callback_optuna best_model best_modellastFold.params: ", best_modellastFold.params)

    bestResult_dict = {
        #   "best_modellastFold_string":best_modellastFold_string,
        # "best_modellastFold_params": best_modellastFold_params,
        "params_optuna": params_optuna,
        "other_params": other_params,
        "best_optunaTrial_number": best_trial.number + 1,
        "best_pnl_val": best_trial.user_attrs.get('best_pnl_val', None),
        "best_pnl_perTrade_diff": best_trial.user_attrs.get('pnl_perTrade_diff', None),
        "best_params": best_trial.params,
        'selected_feature_names': best_trial.user_attrs.get('selected_feature_names', None),
        'use_of_rfe_in_optuna': best_trial.user_attrs.get('use_of_rfe_in_optuna', None),
        'perctDiff_winrateRatio_train_val': best_trial.user_attrs.get('perctDiff_winrateRatio_train_val', None),
        'ecart_train_val': best_trial.user_attrs.get('ecart_train_val', None),
        # for pareto analysis. We compute a specific metric including trade ratio for weithing the impact
        'pnl_norm_objective': best_trial.values[0] if len(best_trial.values) > 0 else None,
        'model_by_fold': best_trial.user_attrs.get('model_by_fold', None),
        'X_val_by_fold': best_trial.user_attrs.get('X_val_by_fold', None),

        'best_thresh_fold': best_trial.user_attrs.get('best_thresh_fold', None),
        'best_iteration_fold': best_trial.user_attrs.get('best_iteration_fold', None),

        # 'ecart_train_val': best_trial.values[1] if len(best_trial.values) > 1 else None
    }
    study.set_user_attr('bestResult_dict', bestResult_dict)
    if bestResult_dict["perctDiff_winrateRatio_train_val"] is None:
        raise ValueError("perctDiff_winrateRatio_train_val ne peut pas être None dans bestResult_dict")
    winrates_val_by_fold_ = best_trial.user_attrs.get('winrates_val_by_fold', None)
    winrates_val_by_fold_formatted_ = [f"{x:.3f}" for x in winrates_val_by_fold_] if winrates_val_by_fold_ else None

    nb_trades_val_by_fold_ = best_trial.user_attrs.get('nb_trades_val_by_fold', None)
    nb_samples_val_by_fold_ = best_trial.user_attrs.get('nb_samples_val_by_fold', None)
    scores_ens_val_list_ = best_trial.user_attrs.get('scores_ens_val_list', None)
    scores_ens_val_list_formatted_ = [f"{x:.2f}" for x in scores_ens_val_list_] if scores_ens_val_list_ else None

    brier_val_list_ = best_trial.user_attrs.get('brier_val_list', None)
    brier_val_list_formatted_ = [f"{x:.3f}" for x in brier_val_list_] if brier_val_list_ else None

    best_thresh_fold_ = best_trial.user_attrs.get('best_thresh_fold', None)
    best_thresh_fold_formatted_ = [f"{x:.3f}" for x in best_thresh_fold_] if best_thresh_fold_ else None

    best_iteration_fold_ = best_trial.user_attrs.get('best_iteration_fold', None)
    best_iteration_fold_formatted_ = [f"{x:.3f}" for x in best_iteration_fold_] if best_iteration_fold_ else None

    # Ajout pour l'entraînement

    nb_trades_train_by_fold_ = best_trial.user_attrs.get('nb_trades_train_by_fold', None)
    nb_samples_train_by_fold_ = best_trial.user_attrs.get('nb_samples_train_by_fold', None)
    scores_ens_train_list_ = best_trial.user_attrs.get('scores_ens_train_list', None)
    scores_ens_train_list_formatted_ = [f"{x:.2f}" for x in scores_ens_train_list_] if scores_ens_train_list_ else None

    # Récupération des données pour train
    class0_raw_train = best_trial.user_attrs.get('class0_raw_data_train_by_fold', None)
    class1_raw_train = best_trial.user_attrs.get('class1_raw_data_train_by_fold', None)

    # Récupération des données pour val
    class0_raw_val = best_trial.user_attrs.get('class0_raw_data_val_by_fold', None)
    class1_raw_val = best_trial.user_attrs.get('class1_raw_data_val_by_fold', None)

    # Gestion des ratios inversés pour train
    # Gestion des ratios pour train
    if class0_raw_train is not None and class1_raw_train is not None:
        # Conversion en tableaux NumPy si nécessaire
        class0_raw_train = np.array(class0_raw_train)
        class1_raw_train = np.array(class1_raw_train)

        # Calcul du ratio élément par élément
        ratio_raw_class_1_0_train = np.where(
            class0_raw_train != 0,  # Condition pour éviter la division par zéro
            class1_raw_train / class0_raw_train,  # Division normale
            0  # Si division par zéro, le ratio est défini comme 0
        )
    else:
        ratio_raw_class_1_0_train = None  # Cas où les données sont manquantes

    # Gestion des ratios pour val
    if class0_raw_val is not None and class1_raw_val is not None:
        # Conversion en tableaux NumPy si nécessaire
        class0_raw_val = np.array(class0_raw_val)
        class1_raw_val = np.array(class1_raw_val)

        # Calcul du ratio élément par élément
        ratio_raw_class_1_0_val = np.where(
            class0_raw_val != 0,  # Condition pour éviter la division par zéro
            class1_raw_val / class0_raw_val,  # Division normale
            0  # Si division par zéro, le ratio est défini comme 0
        )
    else:
        ratio_raw_class_1_0_val = None  # Cas où les données sont manquantes
    # Récupération des données pour train
    winrate_train_list = best_trial.user_attrs.get('winrate_raw_data_train_by_fold', None)
    winrate_train_list_formatted = [f"{x:.2f}" for x in winrate_train_list] if winrate_train_list else None

    # Récupération des données pour val
    winrate_val_list = best_trial.user_attrs.get('winrate_raw_data_val_by_fold', None)
    winrate_val_list_formatted = [f"{x:.2f}" for x in winrate_val_list] if winrate_val_list else None
    raw_metrics_byFold = best_trial.user_attrs.get('raw_metrics_byFold', None),
    if raw_metrics_byFold is None:
        raise ValueError("raw_metrics_byFold is None")

    # Log additional metrics if available
    metrics = {
        # Validation metrics
        'total_tp_val': best_trial.user_attrs.get('total_tp_val', None),
        'total_fp_val': best_trial.user_attrs.get('total_fp_val', None),
        'total_tn_val': best_trial.user_attrs.get('total_tn_val', None),
        'total_fn_val': best_trial.user_attrs.get('total_fn_val', None),
        'cummulative_pnl_val': best_trial.user_attrs.get('cummulative_pnl_val', None),
        'tp_fp_diff_val': best_trial.user_attrs.get('tp_fp_diff_val', None),
        'tp_fp_percentage_val': best_trial.user_attrs.get('tp_fp_percentage_val', None),
        'win_rate_val': best_trial.user_attrs.get('win_rate_val', None),
        'scores_ens_val_list': best_trial.user_attrs.get('scores_ens_val_list', None),
        'winrates_val_by_fold_formatted': winrates_val_by_fold_formatted_,
        'nb_trades_val_by_fold': nb_trades_val_by_fold_,
        'nb_samples_val_by_fold': nb_samples_val_by_fold_,
        'scores_ens_val_list_formatted': scores_ens_val_list_formatted_,
        'brier_val_list_formatted':brier_val_list_formatted_,
        "best_thresh_fold_formatted_":best_thresh_fold_formatted_,
        "best_iteration_fold_formatted_":best_iteration_fold_formatted_,
    'class0_raw_data_val_by_fold': best_trial.user_attrs.get('class0_raw_data_val_by_fold', None),
        'class1_raw_data_val_by_fold': best_trial.user_attrs.get('class1_raw_data_val_by_fold', None),
        'val_trades_samples_perct': best_trial.user_attrs.get('val_trades_samples_perct', None),
        'winrates_val_by_fold': best_trial.user_attrs.get('winrates_val_by_fold', None),

        'winrate_raw_data_val_by_fold': best_trial.user_attrs.get('winrate_raw_data_val_by_fold', None),
        'ratio_raw_class_1_0_val ': ratio_raw_class_1_0_val,
        'winrate_val_list_formatted': winrate_val_list_formatted,
        'val_pred_proba_log_odds': best_trial.user_attrs.get('val_pred_proba_log_odds', None),

        # Training metrics
        'total_tp_train': best_trial.user_attrs.get('total_tp_train', None),  # Ajouté pour l'entraînement
        'total_fp_train': best_trial.user_attrs.get('total_fp_train', None),  # Ajouté pour l'entraînement
        'total_tn_train': best_trial.user_attrs.get('total_tn_train', None),  # Ajouté pour l'entraînement
        'total_fn_train': best_trial.user_attrs.get('total_fn_train', None),  # Ajouté pour l'entraînement
        'cummulative_pnl_train': best_trial.user_attrs.get('cummulative_pnl_train', None),  # Ajouté pour l'entraînement
        'tp_fp_diff_train': best_trial.user_attrs.get('tp_fp_diff_train', None),  # Ajouté pour l'entraînement
        'tp_fp_percentage_train': best_trial.user_attrs.get('tp_fp_percentage_train', None),
        'win_rate_train': best_trial.user_attrs.get('win_rate_train', None),  # Ajouté pour l'entraînement
        'scores_ens_train_list': best_trial.user_attrs.get('scores_ens_train_list', None),  # Ajouté pour l'entraînement

        'nb_trades_train_by_fold': nb_trades_train_by_fold_,  # Ajouté pour l'entraînement
        'nb_samples_train_by_fold': nb_samples_train_by_fold_,  # Ajouté pour l'entraînement
        'train_trades_samples_perct': best_trial.user_attrs.get('train_trades_samples_perct', None),
        'scores_ens_train_list_formatted': scores_ens_train_list_formatted_,
        'winrates_train_by_fold': best_trial.user_attrs.get('winrates_train_by_fold', None),

        'class0_raw_data_train_by_fold': best_trial.user_attrs.get('class0_raw_data_train_by_fold', None),
        'class1_raw_data_train_by_fold': best_trial.user_attrs.get('class1_raw_data_train_by_fold', None),
        'ratio_raw_class_1_0_train': ratio_raw_class_1_0_train,
        'winrate_train_list_formatted': winrate_train_list_formatted,
        'train_pred_proba_log_odds': best_trial.user_attrs.get('train_pred_proba_log_odds', None),

        # Other
        'weight_split': best_trial.user_attrs.get('weight_split', None),
        'nb_split_weight': best_trial.user_attrs.get('nb_split_weight', None),
        'perctDiff_ratioTradeSample_train_val': best_trial.user_attrs.get('perctDiff_ratioTradeSample_train_val', None)

    }
    # scores_ens_val_list_formatted = [f"{x:.2f}" for x in metrics['scores_ens_val_list']]
    # Vérification des valeurs None
    keys_with_none = [key for key, value in metrics.items() if value is None]

    if keys_with_none:
        raise ValueError(f"Les clés suivantes ont une valeur 'None' : {keys_with_none}")

    # Calcul des métriques
    selected_feature_names = best_trial.user_attrs.get('selected_feature_names', None)
    # Rapport
    modele_param_optuna_range = get_model_param_range(config['model_type'])

    method_names_pareto = {
        optuna_doubleMetrics.DISABLE: "DISABLE",
        optuna_doubleMetrics.USE_DIST_TO_IDEAL: "double avec USE_DIST_TO_IDEAL",
        optuna_doubleMetrics.USE_WEIGHTED_AVG: "double avec USE_WEIGHTED_AVG"
    }

    method_name_pareto = method_names_pareto.get(optuna_objective_type_value, "UNKNOWN")

    method_names_cv = {
        cv_config.TIME_SERIE_SPLIT: "TIME_SERIE_SPLIT",
        cv_config.TIME_SERIE_SPLIT_NON_ANCHORED_AFTER_PREVVAL: "TIME_SERIE_SPLIT_NON_ANCHORED",
        cv_config.TIME_SERIE_SPLIT_NON_ANCHORED_AFTER_PREVTRAIN: "TIME_SERIE_SPLIT_NON_ANCHORED_AFTER_PREVTRAIN",
        cv_config.TIMESERIES_SPLIT_BY_ID: 'TIMESERIES_SPLIT_BY_ID',
        cv_config.K_FOLD: "K_FOLD",
        cv_config.K_FOLD_SHUFFLE: "K_FOLD_SHUFFLE",
        cv_config.TIME_SERIE_SPLIT_NON_ANCHORED_ROLLING: "TIME_SERIE_SPLIT_NON_ANCHORED_ROLLING"

    }

    train_pred_proba_log_odds = metrics['train_pred_proba_log_odds']
    val_pred_proba_log_odds = metrics['val_pred_proba_log_odds']

    # Vérification que les deux listes ne sont pas None et ont le même nombre de folds
    if train_pred_proba_log_odds is not None and val_pred_proba_log_odds is not None:
        if len(train_pred_proba_log_odds) == len(val_pred_proba_log_odds):
            # print(f"Nombre de folds: {len(train_pred_proba_log_odds)}")

            # Initialiser des listes pour stocker les minima et maxima de chaque fold pour train et val
            train_min_values = []
            train_max_values = []
            val_min_values = []
            val_max_values = []

            # Initialiser des listes pour stocker les probabilités minimales et maximales
            train_prob_min = []
            train_prob_max = []
            val_prob_min = []
            val_prob_max = []

            # Parcourir les folds
            for i in range(len(train_pred_proba_log_odds)):
                train_fold_values = train_pred_proba_log_odds[i]
                val_fold_values = val_pred_proba_log_odds[i]

                # Convertir les valeurs en tableaux NumPy
                train_fold_array = np.array(train_fold_values)
                val_fold_array = np.array(val_fold_values)

                # Calculer les minima et maxima pour les log-odds et arrondir à 2 décimales
                train_min_values.append(round(np.min(train_fold_array), 2))
                train_max_values.append(round(np.max(train_fold_array), 2))
                val_min_values.append(round(np.min(val_fold_array), 2))
                val_max_values.append(round(np.max(val_fold_array), 2))

                # Calculer les probabilités en utilisant sigmoid
                train_probs = sigmoidCustom_cpu(train_fold_array)
                val_probs = sigmoidCustom_cpu(val_fold_array)

                # Calculer les minima et maxima pour les probabilités
                train_prob_min.append(np.round(np.min(train_probs), 2))
                train_prob_max.append(np.round(np.max(train_probs), 2))
                val_prob_min.append(np.round(np.min(val_probs), 2))
                val_prob_max.append(np.round(np.max(val_probs), 2))

        else:
            print("Le nombre de folds dans train et val ne correspond pas.")
            exit(99)
    else:
        print("train_pred_proba_log_odds ou val_pred_proba_log_odds est None ou n'est pas disponible.")

    method_name_cv = method_names_cv.get(cv_method, "UNKNOWN")
    use_imbalance_penalty = best_trial.user_attrs.get('use_imbalance_penalty', False)
    print(f"\n   {Fore.BLUE}##Meilleur essai jusqu'à present: {bestResult_dict['best_optunaTrial_number']}, "
          f"Methode=> Optuna: '{method_name_pareto} | CV: '{method_name_cv} |  Objective loss Fct : {config.get('custom_objective_lossFct', None)} | \n"
          f"use_imbalance_penalty: {'Activé' if use_imbalance_penalty else 'Désactivé'}' |"
          f"scaler_choice: {config.get('scaler_choice', 0)} | model:{config.get('model_type', 0)} |"
          f"auto_filtering_mode: {config.get('auto_filtering_mode', 0)} ##{Style.RESET_ALL}")
    if config['use_optuna_constraints_func'] == True:

        # nb_trades_val_by_fold_list = trial.user_attrs.get('nb_trades_val_by_fold', [float('inf')])
        # winrates_val_by_fold = trial.user_attrs.get('winrates_val_by_fold', None)
        contraints_list = calculate_constraints_optuna(trial=trial, config=config)
        # constraints_reached_check = all(c == 0 for c in contraints_list)
        # print("contraints_list: ",contraints_list)
        # if constraints_reached_check != contraints_reached:
        #   print(contraints_list)
        #  raise ValueError(
        #     f"Incohérence détectée : constraints_reached_check = {constraints_reached_check}, contraints_reached = {contraints_reached}")

        # Vérifier si la contrainte trades/samples est activée
        trades_samples_constraint_active = config.get('config_constraint_min_trades_samples_perct',
                                                      float('inf')) != float('inf')

        if contraints_reached:
            constraint_message = (Fore.GREEN + f"\u2713 All constraints respected:\n"
                                               f"    - ratioWinrate_train_val <= {config['config_constraint_ratioWinrate_train_val']} (max: {max(best_trial.user_attrs.get('perctDiff_winrateRatio_train_val', []))}) - {best_trial.user_attrs.get('perctDiff_winrateRatio_train_val', None)})\n"
                                               f"    - min_trades >= {config['config_constraint_min_trades_threshold_by_Fold']} (min: {min(best_trial.user_attrs.get('nb_trades_val_by_fold', []))}) - {best_trial.user_attrs.get('nb_trades_val_by_fold', None)})\n"
                                               f"    - winrate >= {config['config_constraint_winrates_val_by_fold']} (min: {min(best_trial.user_attrs.get('winrates_val_by_fold', []))}) - {best_trial.user_attrs.get('winrates_val_by_fold', None)})\n")

            # Ajouter le message pour la contrainte trades/samples si activée
            if trades_samples_constraint_active:
                val_trades_samples_perct = best_trial.user_attrs.get('val_trades_samples_perct', [])
                if val_trades_samples_perct:
                    min_perct = min(val_trades_samples_perct)
                    constraint_message += f"    - trades/samples ratio >= {config.get('config_constraint_min_trades_samples_perct')}% (min: {min_perct:.2f}%) - {val_trades_samples_perct})\n"

            print(constraint_message + Style.RESET_ALL)

        else:
            print(Fore.RED + "\u2717 Some constraints not respected:" + Style.RESET_ALL)
            # Application pour chaque contrainte
            # 1. Contrainte ratio Winrate train/val

            print(format_constraint_message(
                contraints_list[0] > 0,
                config,
                trial,
                "max ratioWinrate_train_val %",
                "config_constraint_ratioWinrate_train_val",
                "perctDiff_winrateRatio_train_val",
                check_type='max'
            ))

            # 2. Contrainte nombre minimum de trades
            print(format_constraint_message(
                contraints_list[1] > 0,
                config,
                trial,
                "val min_trades",
                "config_constraint_min_trades_threshold_by_Fold",
                "nb_trades_val_by_fold",
                check_type='min'
            ))

            # 3. Contrainte winrate minimum
            print(format_constraint_message(
                contraints_list[2] > 0,
                config,
                trial,
                "val min winrate %",
                "config_constraint_winrates_val_by_fold",
                "winrates_val_by_fold",
                check_type='min'
            ))

            # 4. Contrainte écart-type des trades (si présente)
            if 'config_constraint_max_std_trades' in config and config['config_constraint_max_std_trades'] != float(
                    'inf'):
                if len(contraints_list) > 3:
                    print(format_constraint_message(
                        contraints_list[3] > 0,
                        config,
                        trial,
                        "val max std trades",
                        "config_constraint_max_std_trades",
                        "nb_trades_val_by_fold",
                        check_type='std'
                    ))

            # 5. Contrainte ratio trades/samples (si activée)
            if trades_samples_constraint_active:
                constraint_index = 4
                if 'config_constraint_max_std_trades' in config and config['config_constraint_max_std_trades'] != float(
                        'inf'):
                    constraint_index = 4
                else:
                    constraint_index = 3

                if len(contraints_list) > constraint_index:
                    print(format_constraint_message(
                        contraints_list[constraint_index] > 0,
                        config,
                        trial,
                        "val min trades/samples ratio %",
                        "config_constraint_min_trades_samples_perct",
                        "val_trades_samples_perct",
                        check_type='min'
                    ))
    # print(f" perctDiff_winrateRatio_train_val: {best_trial.user_attrs.get('perctDiff_winrateRatio_train_val', None)}")
    # print(f" perctDiff_ratioTradeSample_train_val: {best_trial.user_attrs.get('perctDiff_ratioTradeSample_train_val', None)}")

    print(f"    =>Objective 1: pnl_norm_objective -> {bestResult_dict['pnl_norm_objective']:.4f} "
          f"avec weight_split: {metrics['weight_split']} nb_split_weight {metrics['nb_split_weight']}")
    if optuna_objective_type_value != optuna_doubleMetrics.DISABLE:
        print(
            f"    =>Objective 2: score différence par trade (train - val) -> {bestResult_dict['best_pnl_perTrade_diff']:.4f}\n"
            f"      score ecart_train_val   -> {bestResult_dict['perctDiff_winrateRatio_train_val']}")
    if (bestResult_dict['use_of_rfe_in_optuna'] != rfe_param.NO_RFE):
        print(
            f"    =>Nombre de features sélectionnées par RFECVCV: {len(bestResult_dict['selected_feature_names'])}, Noms des features: {', '.join(bestResult_dict['selected_feature_names'])}")

    print(f"Principal métrique pour le meilleur essai:")
    print(f" *Rappel des distributions initiales des classe 0 et 1 par fold : ")
    print(f"     -NB Class0 train                : {metrics['class0_raw_data_train_by_fold']}")
    print(f"     -NB Class1 train                : {metrics['class1_raw_data_train_by_fold']}")
    print(f"     -% Winrate initiale sur train   : {metrics['winrate_train_list_formatted']}")
    # print(f"     -Ratio class 1/0  train        : {metrics['ratio_raw_class_1_0_train']}")
    print(f"\n     -NB Class0 val                : {metrics['class0_raw_data_val_by_fold']}")
    print(f"     -NB Class1 val                  : {metrics['class1_raw_data_val_by_fold']}")
    print(f"     -% Winrate initiale sur val     : {metrics['winrate_val_list_formatted']}")
    # print(f"     -Ratio class 1/0  val              : {metrics['ratio_raw_class_1_0_val']:.2f}%")
    metrics_list = raw_metrics_byFold[0]
    print(f" *Ensemble Train: ")
    print(f"     -Fold Start: {[metrics['train_metrics']['start_time_train'] for metrics in metrics_list]}")
    print(f"     -Fold End  : {[metrics['train_metrics']['end_time_train'] for metrics in metrics_list]}")
    print(f"     -Slope     : {[metrics['train_metrics']['slope_cv_train'] for metrics in metrics_list]}")
    print(f"     -r2        : {[metrics['train_metrics']['r2_slope_cv_train'] for metrics in metrics_list]}")
    print(f"     -Counter Mv: {[metrics['train_metrics']['counter_moves_train'] for metrics in metrics_list]}")

    print(f"     -Train Log Odds Min: {train_min_values} => Probabilité: {train_prob_min}")
    print(f"     -Train Log Odds Max: {train_max_values} => Probabilité: {train_prob_max}")

    print(f"     -Nombre de: TP: {metrics['total_tp_train']}, FP: {metrics['total_fp_train']}, "
          f"TN: {metrics['total_tn_train']}, FN: {metrics['total_fn_train']}")
    print(f"     -% Winrate                     : {metrics['win_rate_train']:.2f}%")
    print(f"     -% de trades (tp+fp)           : {metrics['tp_fp_percentage_train']:.2f}%")
    print(f"     -Différence (TP - FP)          : {metrics['tp_fp_diff_train']}")
    print(
        f"     -PNL                           : {metrics['cummulative_pnl_train']},             Split: {metrics['scores_ens_train_list_formatted']}")
    print(
        f"                                                    Winrate train %: {[f'{p:.2f}%' for p in metrics['winrates_train_by_fold']]}")
    print(
        f"                                                    Nb trades train :{metrics['nb_trades_train_by_fold']}")
    print(
        f"                                                    Nb samples train :{metrics['nb_samples_train_by_fold']}")
    print(
        f"                                             trades/samples train % :{[f'{p:.2f}%' for p in metrics['train_trades_samples_perct']]}")
    print(f" *Ensemble Validation: ")
    print(f"     -Fold Start: {[metrics['val_metrics']['start_time_val'] for metrics in metrics_list]}")
    print(f"     -Fold End  : {[metrics['val_metrics']['end_time_val'] for metrics in metrics_list]}")
    print(f"     -Slope     : {[metrics['val_metrics']['slope_cv_val'] for metrics in metrics_list]}")
    print(f"     -r2        : {[metrics['val_metrics']['r2_slope_cv_val'] for metrics in metrics_list]}")
    print(f"     -Counter Mv: {[metrics['val_metrics']['counter_moves_val'] for metrics in metrics_list]}")

    print(f"     -Val Log Odds Min: {val_min_values} => Probabilités: {val_prob_min}")
    print(f"     -Val Log Odds Max: {val_max_values} => Probabilités: {val_prob_max}")
    print(f"     -Nombre de: TP: {metrics['total_tp_val']}, FP: {metrics['total_fp_val']}, "
          f"TN: {metrics['total_tn_val']}, FN: {metrics['total_fn_val']}")
    print(f"     -% Winrate                     : {metrics['win_rate_val']:.2f}%")
    print(f"     -% de trades (tp+fp)           : {metrics['tp_fp_percentage_val']:.2f}%")
    print(f"     -Différence (TP - FP)          : {metrics['tp_fp_diff_val']}")
    print(
        f"     -PNL                           : {metrics['cummulative_pnl_val']},             Split: {metrics['scores_ens_val_list_formatted']}")
    print(
        f"                                                     Winrate val %: {[f'{p:.2f}%' for p in metrics['winrates_val_by_fold']]}")

    print(
        f"                                                    Nb trades val :{metrics['nb_trades_val_by_fold']}")
    print(
        f"                                                    Nb samples val :{metrics['nb_samples_val_by_fold']}")
    print(
        f"                                                    trades/samples val % :{[f'{p:.2f}%' for p in metrics['val_trades_samples_perct']]}")
    print(
        f"                                                    Brier Score :{metrics['brier_val_list_formatted']}")
    print(
        f"     -Nombre d'échantillons           : {sum([metrics['total_tp_val'], metrics['total_fp_val'], metrics['total_tn_val'], metrics['total_fn_val']])} "
        f"dont {metrics['total_tp_val'] + metrics['total_fp_val']} trades pris")
    if sum(metrics['scores_ens_val_list']) != 0 and metrics['cummulative_pnl_val'] == 0:
        # raise ValueError(
        #   f"Erreur: La somme des métriques ({sum(metrics['scores_ens_val_list'])}) est différente de 0, mais cumulative_pnl_val est égal à 0.")
        warnings.warn(
            f"Attention: La somme des métriques ({sum(metrics['scores_ens_val_list'])}) "
            f"est différente de 0, mais cumulative_pnl_val est égal à 0.",
            UserWarning)
    else:
        print(
            f"Vérification réussie: Somme des métriques = {metrics['scores_ens_val_list']}, cumulative_pnl_val = {metrics['scores_ens_val_list']}")

    # Calcul du pourcentage avec gestion de la division par 0

    print(
        f"     -Nombre d'échantillons           : {sum([metrics['total_tp_train'], metrics['total_fp_train'], metrics['total_tn_train'], metrics['total_fn_train']])} "
        f"dont {metrics['total_tp_train'] + metrics['total_fp_train']} trades pris")
    print(f"\n    =>Hyperparamètres du meilleur score trouvé à date: {bestResult_dict['best_params']}")

    if is_newBestTrial == True:
        report_trial_optuna(trial, best_trial, rfe_param, modele_param_optuna_range, selected_feature_names,
                            results_directory, config, bestResult_dict)

    study_optuna.set_user_attr('bestResult_dict', bestResult_dict)

    # Vérification de l'option use_pnl_theoric
    # if metrics['cummulative_pnl_val'] > 8000 and contraints_reached:
    #    raise ValueError("INFO : cummulative_pnl_val > 12000")


@njit
def detect_naked_pocs_per_session(
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        pocs: np.ndarray,
        delta_timestamps: np.ndarray,
        session_start_idx: int,
        session_end_idx: int
) -> tuple[np.ndarray, np.ndarray]:
    n = session_end_idx - session_start_idx
    dist_above = np.full(n, np.nan, dtype=np.float64)
    dist_below = np.full(n, np.nan, dtype=np.float64)

    for idx in range(n):
        current_idx = session_start_idx + idx
        current_close = closes[current_idx]
        closest_above = np.inf
        closest_below = -np.inf
        has_above = False
        has_below = False

        for past_idx in range(current_idx - 1, session_start_idx - 1, -1):
            past_poc = pocs[past_idx]
            is_naked = True

            for check_idx in range(past_idx + 1, current_idx + 1):
                if lows[check_idx] <= past_poc <= highs[check_idx]:
                    is_naked = False
                    break

            if is_naked:
                if past_poc > current_close:
                    dist = past_poc - current_close
                    if dist < abs(closest_above - current_close):
                        closest_above = past_poc
                        has_above = True
                elif past_poc < current_close:
                    dist = current_close - past_poc
                    if dist < abs(current_close - closest_below):
                        closest_below = past_poc
                        has_below = True

        if has_above:
            dist_above[idx] = closest_above - current_close
        if has_below:
            dist_below[idx] = current_close - closest_below

    return dist_above, dist_below


@njit(parallel=True)
def detect_naked_pocs_parallel(
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        pocs: np.ndarray,
        delta_timestamps: np.ndarray,
        session_starts: np.ndarray,
        session_ends: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    n = len(closes)
    dist_above = np.full(n, np.nan, dtype=np.float64)
    dist_below = np.full(n, np.nan, dtype=np.float64)

    num_sessions = len(session_starts)

    for s in prange(num_sessions):
        session_start_idx = session_starts[s]
        session_end_idx = session_ends[s]

        # Appeler la fonction pour traiter une session
        session_dist_above, session_dist_below = detect_naked_pocs_per_session(
            closes, highs, lows, pocs, delta_timestamps,
            session_start_idx, session_end_idx
        )

        # Copier les résultats dans les tableaux principaux
        dist_above[session_start_idx:session_end_idx] = session_dist_above
        dist_below[session_start_idx:session_end_idx] = session_dist_below

    return dist_above, dist_below


def calculate_naked_poc_distances(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    start_time = time.time()
    print(f"Start calculate_naked_poc_distances")

    closes = df['close'].values.astype(np.float64)
    highs = df['high'].values.astype(np.float64)
    lows = df['low'].values.astype(np.float64)
    pocs = df['pocPrice'].values.astype(np.float64)
    delta_timestamps = df['deltaTimestampOpening'].values.astype(np.float64)

    # Identifier les indices de début et de fin de chaque session
    session_start_idxs = df[df['sc_sessionStartEnd'] == 10].index.values.astype(np.int64)
    session_end_idxs = np.append(session_start_idxs[1:], len(df))

    if len(session_start_idxs) == 0:
        raise ValueError("No sessions found with sc_sessionStartEnd=10")

    dist_above, dist_below = detect_naked_pocs_parallel(
        closes, highs, lows, pocs, delta_timestamps,
        session_start_idxs, session_end_idxs
    )

    dist_above = -np.abs(dist_above)
    total_time = time.time() - start_time
    print(f"Total time (calculate_naked_poc_distances): {total_time:.2f} seconds")
    return pd.Series(dist_above, index=df.index), pd.Series(dist_below, index=df.index)


import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError

import pandas as pd
import numpy as np
from functools import reduce
from numpy.linalg import LinAlgError

from numpy.linalg import LinAlgError


def fast_vif(df):
    """
    Calcul rapide du VIF en utilisant l'inversion de la matrice de corrélation.

    Args:
        df (pd.DataFrame): DataFrame contenant uniquement les colonnes numériques

    Returns:
        pd.DataFrame: DataFrame contenant les features et leurs valeurs VIF correspondantes
    """
    df_numeric = df.select_dtypes(include=[np.number])
    vif_data = pd.DataFrame()
    vif_data["Feature"] = df_numeric.columns

    try:
        X = df_numeric.values
        corr_matrix = np.corrcoef(X, rowvar=False)  # Matrice de corrélation
        inv_corr_matrix = np.linalg.pinv(corr_matrix)

        vif_data["VIF"] = np.diag(inv_corr_matrix)  # Extraction des VIFs depuis la diagonale
    except LinAlgError:
        vif_data["VIF"] = np.nan  # En cas de problème d'inversion de matrice

    return vif_data


def compute_vif_fast(df, threshold):
    """
    Processus rapide de filtrage des features basé sur le VIF, avec gestion des labels et erreurs cohérente.

    Args:
        df (pd.DataFrame): DataFrame contenant uniquement les features numériques.
        threshold (float): Seuil VIF au-delà duquel une feature est considérée comme trop colinéaire.

    Returns:
        pd.DataFrame: DataFrame contenant l'historique des VIF et le statut final des features.
    """
    df_numeric = df.select_dtypes(include=[np.number]).copy()
    if df_numeric.empty:
        print("⚠️ Aucune colonne numérique dans le dataset. Retour d'un DataFrame vide.")
        return pd.DataFrame({'Feature': df.columns, 'VIF1': 'n.a', 'StatusVif': 'Non calculé'})

    vif_dfs = []
    removed_features = []
    retained_features = df_numeric.columns.tolist()
    iteration_count = 1

    with np.errstate(divide='ignore', invalid='ignore'):  # Supprime les warnings division par zéro
        while True:
            try:
                vif_df = fast_vif(df_numeric)
                vif_df.columns = ['Feature', f'VIF{iteration_count}']
                vif_df[f'VIF{iteration_count}'] = pd.to_numeric(vif_df[f'VIF{iteration_count}'], errors='coerce')
                vif_dfs.append(vif_df)

                max_vif = vif_df[f'VIF{iteration_count}'].max()
                if pd.isna(max_vif) or max_vif <= threshold:
                    break

                # Identifier la feature à supprimer
                max_feature = vif_df.loc[vif_df[f'VIF{iteration_count}'].idxmax(), 'Feature']
                print(f"[Itération {iteration_count}] Max VIF = {max_vif:.2e}, on retire {max_feature}")

                # Suppression de la feature ayant le plus haut VIF
                df_numeric.drop(columns=[max_feature], inplace=True)
                removed_features.append(max_feature)
                retained_features.remove(max_feature)

                iteration_count += 1
            except Exception as e:
                print(f"⚠️ Erreur lors du calcul du VIF à l'itération {iteration_count}: {str(e)}")
                break

    # Si aucun calcul VIF n'a été effectué avec succès
    if not vif_dfs:
        vif_full = pd.DataFrame({'Feature': df.columns, 'VIF1': 'n.a', 'StatusVif': 'Non calculé'})
        return vif_full

    # Construire un DataFrame avec l'historique des VIF
    vif_full = reduce(lambda left, right: pd.merge(left, right, on='Feature', how='outer'), vif_dfs)

    # Assurer que toutes les features originales sont dans le résultat
    missing_features = set(df.columns) - set(vif_full['Feature'])
    if missing_features:
        missing_df = pd.DataFrame({'Feature': list(missing_features)})
        for col in vif_full.columns:
            if col != 'Feature':
                missing_df[col] = 'n.a'
        vif_full = pd.concat([vif_full, missing_df], ignore_index=True)

    # Définir le statut de chaque feature
    vif_full['StatusVif'] = vif_full['Feature'].apply(
        lambda x: 'Conservé' if x in retained_features else 'Non conservé'
    )

    # Nettoyage des valeurs manquantes dans l'historique des VIF
    vif_history_columns = [col for col in vif_full.columns if col.startswith('VIF')]
    for col in vif_history_columns:
        vif_full[col] = vif_full[col].astype(object)

    for feature in vif_full['Feature']:
        feature_mask = vif_full['Feature'] == feature
        for idx, col in enumerate(vif_history_columns):
            if pd.isna(vif_full.loc[feature_mask, col]).values[0]:
                vif_full.loc[feature_mask, col] = 'n.a'

    # Réorganiser les colonnes pour un affichage cohérent
    vif_columns_sorted = sorted(vif_history_columns, key=lambda x: int(x[3:]), reverse=True)
    vif_full = vif_full[['Feature'] + vif_columns_sorted + ['StatusVif']]

    # Appliquer un format scientifique aux colonnes VIF
    for col in vif_history_columns:
        vif_full[col] = vif_full[col].apply(
            lambda x: f"{x:.2e}" if isinstance(x, (int, float)) and not pd.isna(x) else x)

    # Compter les features conservées et supprimées
    features_conservees = sum(vif_full['StatusVif'] == 'Conservé')
    features_supprimees = sum(vif_full['StatusVif'] == 'Non conservé')
    total_features = features_conservees + features_supprimees

    print(f"Nombre de features conservées: {features_conservees}")
    print(f"Nombre de features supprimées: {features_supprimees}")
    print(f"Nombre total de features analysées: {total_features}\n")

    return vif_full

def calculate_vif(df):
    """
    Calcule le Variance Inflation Factor (VIF) pour chaque feature d'un DataFrame.
    Utilisé comme première méthode de filtrage des features.

    Args:
        df (pd.DataFrame): DataFrame contenant uniquement les colonnes numériques à analyser

    Returns:
        pd.DataFrame: DataFrame contenant les features et leurs valeurs VIF correspondantes
    """
    df_numeric = df.select_dtypes(include=[np.number])

    # Supprimer les colonnes constantes qui causeraient une division par zéro dans le calcul VIF
    df_numeric = df_numeric.loc[:, df_numeric.apply(pd.Series.nunique) != 1]

    vif_data = pd.DataFrame()
    vif_data["Feature"] = df_numeric.columns

    vif_values = []
    for i in range(df_numeric.shape[1]):
        try:
            vif = variance_inflation_factor(df_numeric.values, i)
        except (np.linalg.LinAlgError, ValueError):
            vif = np.nan
        vif_values.append(vif)

    vif_data["VIF"] = vif_values
    return vif_data


def compute_vif_and_remove_multicollinearity(df, threshold):
    """
    Processus itératif de filtrage des features basé sur le VIF.
    Supprime une à une les features ayant le VIF le plus élevé jusqu'à ce que
    toutes les features restantes aient un VIF inférieur au seuil.

    Args:
        df (pd.DataFrame): DataFrame d'entrée
        threshold (float): Seuil VIF au-delà duquel une feature est considérée comme trop colinéaire

    Returns:
        pd.DataFrame: DataFrame contenant l'historique des valeurs VIF et le statut final des features
    """
    # Sélectionner uniquement les colonnes numériques
    X_train_bis = df.select_dtypes(include=[np.number]).copy()

    # Nettoyer les valeurs infinies et NaN
    X_train_bis = X_train_bis.replace([np.inf, -np.inf], np.nan)

    # Remplacer les NaN par la moyenne de chaque colonne
    for col in X_train_bis.columns:
        if X_train_bis[col].isna().any():
            col_mean = X_train_bis[col].mean()
            # Si la moyenne est NaN (tous les éléments sont NaN), utiliser 0
            if pd.isna(col_mean):
                X_train_bis = X_train_bis.drop(columns=[col])
                print(f"Colonne supprimée car toutes valeurs sont NaN: {col}")
            else:
                X_train_bis[col] = X_train_bis[col].fillna(col_mean)

    vif_dfs = []
    iteration_count = 1  # Initialisation du compteur
    removed_features = []
    retained_features = X_train_bis.columns.tolist()

    # Boucle itérative avec gestion des avertissements
    with np.errstate(divide='ignore', invalid='ignore'):
        while True:
            try:
                vif_df = calculate_vif(X_train_bis)
                vif_df.columns = ['Feature', f'VIF{iteration_count}']
                vif_df[f'VIF{iteration_count}'] = pd.to_numeric(vif_df[f'VIF{iteration_count}'], errors='coerce')
                vif_dfs.append(vif_df)

                max_vif = vif_df[f'VIF{iteration_count}'].max()
                if pd.isna(max_vif) or max_vif <= threshold:
                    break
                else:
                    max_feature = vif_df.loc[vif_df[f'VIF{iteration_count}'].idxmax(), 'Feature']
                    print(
                        f"[Itération {iteration_count}] Max VIF = {max_vif}, on retire {max_feature}")  # Ajout du print
                    X_train_bis = X_train_bis.drop(columns=[max_feature])
                    removed_features.append(max_feature)
                    retained_features.remove(max_feature)
                    iteration_count += 1
            except Exception as e:
                print(f"Erreur lors du calcul VIF: {str(e)}")
                break

    # Si aucun calcul VIF n'a été effectué avec succès
    if not vif_dfs:
        vif_full = pd.DataFrame({'Feature': df.columns})
        vif_full[f'VIF1'] = 'n.a'
        vif_full['Status'] = 'Non calculé'
        return vif_full

    # Construction du DataFrame de résultat avec l'historique VIF
    vif_full = reduce(lambda left, right: pd.merge(left, right, on='Feature', how='outer'), vif_dfs)

    # Assurer que toutes les features originales sont dans le résultat
    missing_features = set(df.columns) - set(vif_full['Feature'])
    if missing_features:
        missing_df = pd.DataFrame({'Feature': list(missing_features)})
        for col in vif_full.columns:
            if col != 'Feature':
                missing_df[col] = 'n.a'
        vif_full = pd.concat([vif_full, missing_df], ignore_index=True)

    # Définir le statut
    vif_full['Status'] = vif_full['Feature'].apply(
        lambda x: 'Conservé' if x in retained_features else
        'Non conservé' if x in removed_features else
        'Non calculé'
    )

    # Gestion des valeurs manquantes dans l'historique VIF
    vif_columns = [col for col in vif_full.columns if col.startswith('VIF')]
    for col in vif_columns:
        vif_full[col] = vif_full[col].astype(object)

    for feature in vif_full['Feature']:
        feature_mask = vif_full['Feature'] == feature
        for idx, col in enumerate(vif_columns):
            if pd.isna(vif_full.loc[feature_mask, col]).values[0]:
                vif_full.loc[feature_mask, col] = 'n.a'

    # Réorganisation des colonnes
    vif_columns_sorted = sorted(vif_columns, key=lambda x: int(x[3:]), reverse=True)
    vif_full = vif_full[['Feature'] + vif_columns_sorted + ['Status']]

    return vif_full


def compute_and_keep_correled(merged_X_Y):
    """
    Compute linear (Pearson) and nonlinear (Spearman) correlation with the target variable.

    Parameters:
    -----------
    merged_X_Y : pandas.DataFrame
        Input DataFrame containing features and the target variable.

    Returns:
    --------
    corr_df : pandas.DataFrame
        DataFrame with columns 'Feature', 'lr_target', 'nlr_target' containing
        the linear and nonlinear correlation with the target variable.
    """
    # Compute linear and nonlinear correlations
    linear_correlation_target = merged_X_Y.corr()["target"] * 100
    linear_correlation_target.name = "lr_target"

    nonlinear_correlation_target = merged_X_Y.corr(method="spearman")["target"] * 100
    nonlinear_correlation_target.name = "nlr_target"

    # Combine into a DataFrame
    corr_df = pd.DataFrame({
        'Feature': linear_correlation_target.index,
        'lr_target': linear_correlation_target.values,
        'nlr_target': nonlinear_correlation_target.values
    })

    # Exclude the target row
    corr_df = corr_df[corr_df['Feature'] != 'target']

    return corr_df


from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

###############################################################################
# Fonction mRMR - Minimum Redundancy Maximum Relevance
###############################################################################
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from tqdm import tqdm
import time


def compute_mRMR_filtering(X, Y, config):
    """
    Implémentation simple de la méthode mRMR (Minimum Redundancy Maximum Relevance).

    Args:
        X (pd.DataFrame): DataFrame des features (numériques ou encodées).
        Y (pd.Series): Série cible (peut être catégorielle ou continue).
        config (dict):
            - mrmr_n_features (int) : Nombre maximum de features à sélectionner.
            - mrmr_score_threshold (float) : Seuil minimum pour le score mRMR
                                             (pertinence - redondance).
            - mi_method (str) : "classif" ou "regression" pour forcer le type
                                de calcul de l'information mutuelle.
              Si non spécifié, la fonction détermine automatiquement le mode
              en fonction de la nature de Y.
            - verbose (bool) : Si True, affiche la progression du traitement.

    Returns:
        selected (list): Liste des features retenues par la méthode mRMR.
        mrmr_scores (dict): Dictionnaire {feature: score_mRMR_final}.
                            Note : Certaines features peuvent rester à 0.0
                            si elles n'ont jamais été évaluées ou sélectionnées.
    """
    import numpy as np
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    from tqdm import tqdm
    import time

    # Paramètres par défaut
    n_features_to_select = config.get("mrmr_n_features", X.shape[1])
    score_threshold = config.get("mrmr_score_threshold", -np.inf)
    verbose = True

    if verbose:
        print(f"🔍 Démarrage de la sélection mRMR (max {n_features_to_select} features)")
        start_time = time.time()

    # 1) Détermination automatique du type de mutual_info (classif ou regression)
    if "mi_method" in config:
        mi_method = config["mi_method"]
    else:
        # Détermination automatique si la cible est plutôt discrète/catégorielle
        if (Y.nunique() <= 20 or Y.dtype == 'object' or
                Y.dtype == 'category' or np.issubdtype(Y.dtype, np.integer)):
            mi_method = "classif"
        else:
            mi_method = "regression"

    if verbose:
        print(f"✓ Méthode d'information mutuelle : {mi_method}")

    # 2) Calcul de la pertinence : MI(feature; cible)
    if verbose:
        print("📊 Calcul de la pertinence (MI entre features et cible)...")

    if mi_method == "classif":
        relevance = mutual_info_classif(X, Y, random_state=0)
    else:
        relevance = mutual_info_regression(X, Y, random_state=0)

    features = X.columns.tolist()
    relevance_dict = {f: mi for f, mi in zip(features, relevance)}

    if verbose:
        print(f"✓ Pertinence calculée pour {len(features)} features")

    from joblib import Parallel, delayed

    # Fonction pour calculer une seule valeur d'information mutuelle
    from joblib import Parallel, delayed
    from tqdm import tqdm
    import numpy as np
    import os

    # Fonction pour calculer une seule valeur d'information mutuelle
    def compute_mi_pair(i, j, X_values):
        xi = X_values[:, i].reshape(-1, 1)
        xj = X_values[:, j]
        return (i, j, mutual_info_regression(xi, xj, random_state=0)[0])

    # 3) Calcul de la matrice de redondance : MI(feature_i; feature_j)
    n = len(features)
    mi_matrix = np.zeros((n, n))

    # Convertir le DataFrame en numpy array
    X_values = X.values

    # Générer toutes les paires à calculer
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    if verbose:
        print(f"🔄 Calcul parallèle de la matrice de redondance ({n}x{n})...")
        print(f"Total: {len(pairs)} paires à calculer")

    # Définir un niveau de verbose plus bas pour joblib
    joblib_verbose = 1  # Un niveau bas mais qui montre quand même quelque chose

    # Exécuter les calculs en parallèle
    results = Parallel(n_jobs=-1, verbose=joblib_verbose if verbose else 0)(
        delayed(compute_mi_pair)(i, j, X_values) for i, j in pairs
    )

    # Remplir la matrice avec les résultats
    for i, j, mi_val in results:
        mi_matrix[i, j] = mi_val
        mi_matrix[j, i] = mi_val  # Symétrique

    # 4) Sélection mRMR itérative
    selected = []
    remaining = features.copy()

    # Dictionnaire pour stocker le score mRMR de chaque feature
    mrmr_scores = {f: 0.0 for f in features}

    if verbose:
        print("🔎 Début de la sélection itérative des features...")
        selection_progress = tqdm(total=min(n_features_to_select, len(features)),
                                  desc="Features sélectionnées", disable=not verbose)

    # -- 4.1) Sélection initiale : la feature la plus pertinente
    best_feature = max(remaining, key=lambda f: relevance_dict[f])
    init_score = relevance_dict[best_feature]  # score = pertinence, redondance=0 (aucun feature sélectionné avant)

    # Si même la feature la plus pertinente est en dessous du seuil, on s'arrête
    if init_score < score_threshold:
        if verbose:
            print(f"⚠️ Arrêt: le score initial ({init_score:.4f}) est inférieur au seuil ({score_threshold})")
        return [], mrmr_scores

    selected.append(best_feature)
    remaining.remove(best_feature)

    # On enregistre son score mRMR initial (qui est égal à la pertinence à ce stade)
    mrmr_scores[best_feature] = init_score

    if verbose:
        selection_progress.update(1)
        print(f"  → Première feature sélectionnée: {best_feature} (score: {init_score:.4f})")

    # -- 4.2) Boucle de sélection itérative
    while remaining and len(selected) < n_features_to_select:
        best_score = -np.inf
        best_candidate = None

        # Pour chaque feature restante, on calcule le score = pertinence - redondance_moyenne
        for f in remaining:
            rel = relevance_dict[f]
            idx_f = features.index(f)
            # Moyenne des MI entre f et les features déjà sélectionnées
            redundancy = np.mean([mi_matrix[idx_f, features.index(s)] for s in selected])
            mrmr_score = rel - redundancy

            # Mise à jour du meilleur candidat
            if mrmr_score > best_score:
                best_score = mrmr_score
                best_candidate = f

        # Vérification du score par rapport au seuil
        if best_candidate is None or best_score < score_threshold:
            if verbose:
                print(f"⚠️ Arrêt: aucune feature ne dépasse le seuil ({score_threshold})")
            break

        # Sélection de la feature best_candidate
        selected.append(best_candidate)
        remaining.remove(best_candidate)
        mrmr_scores[best_candidate] = best_score

        if verbose:
            selection_progress.update(1)
            if len(selected) % 5 == 0 or len(selected) == n_features_to_select:
                print(f"  → {len(selected)}/{n_features_to_select} features sélectionnées")

    if verbose:
        selection_progress.close()
        elapsed_time = time.time() - start_time
        print(f"✅ Sélection terminée: {len(selected)} features retenues en {elapsed_time:.2f} secondes")
        # Afficher les 5 meilleures features et leurs scores
        top_features = sorted([(f, s) for f, s in mrmr_scores.items() if s > 0],
                              key=lambda x: x[1], reverse=True)[:5]
        print("📊 Top 5 features sélectionnées:")
        for i, (feature, score) in enumerate(top_features, 1):
            print(f"  {i}. {feature}: {score:.4f}")

    # Retourne la liste des features retenues et le dictionnaire de scores
    return selected, mrmr_scores


###############################################################################
# Calcule les scores de Fisher (ANOVA F-Test)
###############################################################################
from sklearn.feature_selection import f_classif


def fisher_score_feature_selection(X: pd.DataFrame, Y: pd.Series, name: str, config: dict) -> pd.DataFrame:
    """
    Calcule les scores de Fisher (ANOVA F-Test) pour classer les features selon leur importance
    à prédire la cible Y.

    Args:
        X (pd.DataFrame): DataFrame des features à analyser.
        Y (pd.Series): Série cible à prédire (classification).
        name (str): Nom du dataset ou contexte (pour affichage/log).
        config (dict): Configuration avec les clés suivantes:
            - "fisher_top_n_features" (int, optionnel): Nombre maximal de features à retourner.
              Si None, retourne toutes les features.

    Returns:
        pd.DataFrame: DataFrame avec les colonnes suivantes :
            - "Feature": nom des features.
            - "Fisher_Score": le score Fisher associé à chaque feature.
            - "p-value": la p-value correspondante, indiquant la significativité statistique.
            Trié par Fisher Score décroissant.
    """
    fisher_top_n = config.get("fisher_top_n_features", None)

    # Calcul du Fisher Score (ANOVA F-test)
    fisher_scores, p_values = f_classif(X, Y)

    # DataFrame des résultats
    fisher_df = pd.DataFrame({
        'Feature': X.columns,
        'Fisher_Score': fisher_scores,
        'p-value': p_values
    })

    # Trier par Fisher Score décroissant
    fisher_df.sort_values(by='Fisher_Score', ascending=False, inplace=True)

    # Sélection des top_n si spécifié
    if fisher_top_n is not None:
        fisher_df = fisher_df.head(fisher_top_n)

    fisher_df.reset_index(drop=True, inplace=True)

    # print(f"\nFisher Score Feature Selection [{name}]:")
    # print(fisher_df)

    return fisher_df


import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


def calculate_vif_optimized(df):
    """
    Calcule le VIF en gérant les erreurs NaN et inf.

    Args:
        df (pd.DataFrame): DataFrame contenant uniquement les colonnes numériques

    Returns:
        pd.DataFrame: DataFrame contenant les features et leurs VIF
    """
    df_numeric = df.select_dtypes(include=[np.number]).copy()

    vif_data = pd.DataFrame()
    vif_data["Feature"] = df_numeric.columns
    vif_values = []

    for i in range(df_numeric.shape[1]):
        try:
            vif = variance_inflation_factor(df_numeric.values, i)
        except Exception as e:
            print(f"⚠️ Erreur lors du calcul du VIF pour {df_numeric.columns[i]} : {str(e)}")
            vif = np.nan  # On met NaN au lieu d'arrêter le programme
        vif_values.append(vif)

    vif_data["VIF"] = vif_values
    return vif_data


def compute_vif_optimized(df, threshold):
    """
    Processus itératif de filtrage des features basé sur le VIF (Version robuste).

    Args:
        df (pd.DataFrame): DataFrame d'entrée
        threshold (float): Seuil VIF pour élimination des features

    Returns:
        pd.DataFrame: DataFrame contenant l'historique des VIF et le statut des features
    """
    df_numeric = df.select_dtypes(include=[np.number]).copy()
    vif_dfs = []
    removed_features = []

    iteration_count = 1
    while True:
        try:
            vif_df = calculate_vif_optimized(df_numeric)
        except Exception as e:
            print(f"❌ Erreur critique lors du calcul VIF à l'itération {iteration_count} : {str(e)}")
            break

        vif_df.columns = ['Feature', f'VIF{iteration_count}']
        vif_dfs.append(vif_df)

        max_vif = vif_df[f'VIF{iteration_count}'].max()

        # Vérification des NaN : Si tous les VIF sont NaN, on arrête
        if vif_df[f'VIF{iteration_count}'].isna().all():
            print(f"🚨 Tous les VIF sont NaN à l'itération {iteration_count}, arrêt du processus.")
            break

        # Si le VIF max est NaN ou en dessous du seuil, on stoppe l'itération
        if pd.isna(max_vif) or max_vif <= threshold:
            print(f"✅ Seuil atteint : max_vif={max_vif}, arrêt du processus.")
            break

        # Suppression de la feature avec le VIF max
        max_feature = vif_df.loc[vif_df[f'VIF{iteration_count}'].idxmax(), 'Feature']
        print(f"[Itération {iteration_count}] Max VIF = {max_vif}, on retire {max_feature}")

        df_numeric.drop(columns=[max_feature], inplace=True)
        removed_features.append(max_feature)
        iteration_count += 1

    return pd.concat(vif_dfs, axis=1)


def apply_pca_on_scaledData_and_high_corr_features(X, feature_list, corr_threshold=0.6, n_components=2,
                                    plot_correlation=True, pca_prefix="diffPriceClosePoc_PCA_"):
    """
    Vérifie la présence des colonnes, identifie les variables fortement corrélées et applique la PCA.

    Paramètres :
    - X : DataFrame pandas contenant les données
    - feature_list : Liste des colonnes sur lesquelles appliquer la PCA
    - corr_threshold : Seuil de corrélation au-dessus duquel on applique la PCA (par défaut 0.6)
    - n_components : Nombre de composantes PCA à conserver (par défaut 2)
    - plot_correlation : Booléen, si True affiche la heatmap de corrélation
    - pca_prefix : Préfixe pour nommer les nouvelles features PCA (par défaut "diffPriceClosePoc_PCA_")

    Retourne :
    - X modifié avec les nouvelles features PCA
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.decomposition import PCA
    # Vérifier la présence des colonnes dans X
    missing_columns = [col for col in feature_list if col not in X.columns]
    if missing_columns:
        raise ValueError(f"❌ Colonnes manquantes pour la PCA : {missing_columns}")

    # 1️⃣ Calculer la matrice de corrélation
    corr_matrix = X[feature_list].corr(
        #    method='spearman'
    )

    # Affichage de la heatmap de corrélation si demandé
    if plot_correlation:
        # Supposons que corr_matrix est votre matrice de corrélation
        plt.figure(figsize=(12, 10))  # Augmenter la taille de la figure

        # Créer la heatmap avec des ajustements pour les labels
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")

        # Rotation des labels sur l'axe x et ajustement de l'espacement
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        # Ajuster les marges pour éviter la troncation
        plt.tight_layout()

        plt.title("Matrice de Corrélation des Variables")
        plt.show()

    # 2️⃣ Identifier les colonnes fortement corrélées
    high_corr_columns_used4pca = set()

    for i in range(len(feature_list)):
        for j in range(i + 1, len(feature_list)):  # On évite les doublons et la diagonale
            if abs(corr_matrix.iloc[i, j]) >= corr_threshold:
                high_corr_columns_used4pca.add(feature_list[i])
                high_corr_columns_used4pca.add(feature_list[j])

    high_corr_columns_used4pca = sorted(high_corr_columns_used4pca)  # Trier pour l'affichage

    if not high_corr_columns_used4pca:
        print(f"✅ Aucune variable fortement corrélée (> {corr_threshold}), la PCA n'est pas appliquée.")
        return X

    print(f"✅ Variables retenues pour la PCA (corrélation ≥ {corr_threshold}) : {high_corr_columns_used4pca}")

    # 3️⃣ Appliquer la PCA sur ces variables
    # Appliquer la PCA avec le bon nombre de composantes
    pca = PCA(n_components=n_components)
    # Correction ici : utiliser les données de X pour les colonnes à forte corrélation
    X_pca_transformed = pca.fit_transform(X[high_corr_columns_used4pca])

    # Ajouter les nouvelles features PCA à X avec le préfixe personnalisé
    for i in range(pca.n_components_):
        X[f'{pca_prefix}{i + 1}'] = X_pca_transformed[:, i]
        print(f'{pca_prefix}{i + 1} generated')

    essential_columns = {'deltaTimestampOpeningSession1min'}  # Liste des colonnes à ne PAS supprimer
    high_corr_columns_used4pca = [col for col in high_corr_columns_used4pca if col not in essential_columns]

    # Optionnel : Supprimer les anciennes colonnes après la PCA
    X.drop(columns=high_corr_columns_used4pca, inplace=True)

    return X, high_corr_columns_used4pca, pca  # Retourne X modifié


def calculate_statistical_power_job(X, y, feature_list=None,
                                    alpha=0.05, target_power=0.8, n_simulations_monte=20000,
                                    sample_fraction=0.8, verbose=True,
                                    method_powerAnaly='both', n_jobs=-1):
    """
    Calcule la puissance statistique analytique et/ou par simulation Monte Carlo.

    Comportement selon la méthode choisie :
    - 'analytical' : Calcule la puissance analytique uniquement pour les distributions normales (t-test).
                     Pour les distributions non normales ou petits échantillons (Mann-Whitney U),
                     aucune puissance analytique n'est calculée.
    - 'montecarlo' : Calcule la puissance par simulation Monte Carlo en réalisant plusieurs sous-échantillonnages
                     des données originales. Pour chaque sous-échantillon, la fonction détermine s'il faut utiliser
                     un test t (pour les distributions normales avec grands échantillons) ou un test de Mann-Whitney U
                     (pour les distributions non normales ou petits échantillons). Cette décision est prise
                     indépendamment pour chaque simulation et n'est pas influencée par les tests effectués
                     sur les données complètes.
    - 'both' : Combine les deux approches ci-dessus.

    Parameters:
    -----------
    X : pandas.DataFrame
        DataFrame contenant les variables indépendantes
    y : pandas.Series
        Variable cible binaire (0/1)
    feature_list : list, optional
        Liste des caractéristiques à analyser. Si None, toutes les colonnes de X sont utilisées.
    alpha : float, default=0.05
        Seuil de significativité
    target_power : float, default=0.8
        Puissance statistique cible pour le calcul de la taille d'échantillon requise
    n_simulations_monte : int, default=20000
        Nombre de simulations Monte Carlo à exécuter
    sample_fraction : float, default=0.8
        Fraction de l'échantillon à utiliser dans chaque simulation Monte Carlo
    verbose : bool, default=True
        Si True, affiche des informations sur la progression et les tests
    method_powerAnaly : str, default='both'
        Méthode de calcul de la puissance : 'analytical', 'montecarlo', ou 'both'
    n_jobs : int, default=-1
        Nombre de jobs parallèles pour la simulation Monte Carlo. -1 utilise tous les processeurs disponibles.

    Returns:
    --------
    pandas.DataFrame
        DataFrame contenant les résultats pour chaque caractéristique
    """
    import numpy as np
    import pandas as pd
    from scipy.stats import normaltest, mannwhitneyu, ttest_ind, levene
    from statsmodels.stats.power import TTestIndPower
    from joblib import Parallel, delayed

    def run_single_simulation_auto(group0, group1, sample_fraction, alpha):
        """
        Exécute une seule simulation Monte Carlo avec le test approprié.
        Retourne 1 si le test est significatif, 0 sinon.
        """
        # Sous-échantillonnage des groupes
        n0 = max(1, int(len(group0) * sample_fraction))
        n1 = max(1, int(len(group1) * sample_fraction))

        sample0 = np.random.choice(group0, size=n0, replace=True)
        sample1 = np.random.choice(group1, size=n1, replace=True)

        # Choix automatique du test approprié
        if n0 > 20 and n1 > 20:
            # Pour les grands échantillons, vérifier la normalité
            try:
                norm_test_0 = normaltest(sample0).pvalue
                norm_test_1 = normaltest(sample1).pvalue
                var_test_p = levene(sample0, sample1).pvalue

                if norm_test_0 > 0.05 and norm_test_1 > 0.05:
                    # Distributions normales → t-test
                    _, p_value = ttest_ind(sample0, sample1, equal_var=(var_test_p > 0.05))
                else:
                    # Distributions non normales → Mann-Whitney U
                    _, p_value = mannwhitneyu(sample0, sample1, alternative='two-sided')
            except:
                # En cas d'erreur, utiliser Mann-Whitney par défaut
                _, p_value = mannwhitneyu(sample0, sample1, alternative='two-sided')
        else:
            # Petits échantillons → Mann-Whitney par défaut
            _, p_value = mannwhitneyu(sample0, sample1, alternative='two-sided')

        # Retourne 1 si test significatif, 0 sinon
        return 1 if p_value < alpha else 0

    # Validation de la méthode d'analyse
    valid_methods = ['both', 'analytical', 'montecarlo']
    if method_powerAnaly not in valid_methods:
        raise ValueError(f"La méthode '{method_powerAnaly}' n'est pas valide. Options: {', '.join(valid_methods)}")

    # Initialisation de la liste des caractéristiques
    if feature_list is None:
        feature_list = X.columns.tolist()
    else:
        feature_list = [f for f in feature_list if f in X.columns]

    # Suppression des caractéristiques constantes
    constant_features = [col for col in feature_list if X[col].nunique() <= 1]
    if constant_features and verbose:
        print(f"⚠️ {len(constant_features)} features constantes retirées: {constant_features}")

    feature_list = [f for f in feature_list if f not in constant_features]

    # Initialisation des résultats
    results = []
    power_analysis = TTestIndPower()
    total_features = len(feature_list)

    # Analyser chaque caractéristique
    for i, feature in enumerate(feature_list):
        if verbose and i % max(1, total_features // 10) == 0:
            print(f"Traitement: {i + 1}/{total_features} features ({((i + 1) / total_features) * 100:.1f}%)")

        # Préparation des données
        X_feature = X[feature].copy()
        mask = X_feature.notna() & y.notna()
        X_filtered = X_feature[mask].values
        y_filtered = y[mask].values

        group0 = X_filtered[y_filtered == 0]
        group1 = X_filtered[y_filtered == 1]

        # Vérifier si les groupes sont assez grands
        if len(group0) <= 1 or len(group1) <= 1:
            if verbose:
                print(
                    f"⚠️ Feature '{feature}' ignorée : groupes trop petits (group0={len(group0)}, group1={len(group1)})")
            continue

        # Calcul de la taille d'effet
        mean_diff = np.mean(group1) - np.mean(group0)
        pooled_std = np.sqrt(((len(group0) - 1) * np.std(group0, ddof=1) ** 2 +
                              (len(group1) - 1) * np.std(group1, ddof=1) ** 2) /
                             (len(group0) + len(group1) - 2))

        if pooled_std == 0:
            if verbose:
                print(f"⚠️ Feature '{feature}' ignorée : écart-type commun nul")
            continue

        effect_size = mean_diff / pooled_std

        # Initialisation des variables pour les résultats
        power_analytical = None
        required_n = None
        p_value = None
        test_type = None
        normality_status = None

        # Analyse statistique et calcul de la puissance analytique
        if method_powerAnaly in ['both', 'analytical']:
            if len(group0) > 20 and len(group1) > 20:
                # Test de normalité pour les grands échantillons
                norm_test_0 = normaltest(group0).pvalue
                norm_test_1 = normaltest(group1).pvalue
                var_test_p = levene(group0, group1).pvalue

                if norm_test_0 > 0.05 and norm_test_1 > 0.05:
                    # Les deux distributions sont normales → test t
                    test_type = "t-test"
                    normality_status = "normales"
                    analytical_possible = True

                    t_stat, p_value = ttest_ind(group0, group1, equal_var=(var_test_p > 0.05))
                    equal_var_status = "homogènes" if var_test_p > 0.05 else "hétérogènes"

                    # Calcul de la puissance analytique
                    power_analytical = power_analysis.power(
                        effect_size=abs(effect_size),
                        nobs1=len(group0),
                        alpha=alpha,
                        ratio=len(group1) / len(group0)
                    )

                    # Calcul de la taille d'échantillon requise
                    try:
                        required_n = power_analysis.solve_power(
                            effect_size=abs(effect_size),
                            power=target_power,
                            alpha=alpha,
                            ratio=len(group1) / len(group0)
                        )
                    except (ValueError, RuntimeError):
                        required_n = np.nan
                else:
                    # Au moins une distribution non normale → Mann-Whitney U
                    test_type = "Mann-Whitney U"
                    normality_status = "non normales"
                    analytical_possible = False

                    t_stat, p_value = mannwhitneyu(group0, group1, alternative='two-sided')
                    power_analytical = None  # Pas de formule analytique pour Mann-Whitney
            else:
                # Petit échantillon → Mann-Whitney par défaut
                test_type = "Mann-Whitney U"
                normality_status = "échantillons trop petits pour test de normalité"
                analytical_possible = False

                t_stat, p_value = mannwhitneyu(group0, group1, alternative='two-sided')
                power_analytical = None  # Pas de formule analytique pour Mann-Whitney

            if verbose:
                if analytical_possible:
                    print(f"# {normality_status.capitalize()} → {test_type} (utilisé pour le calcul analytique)")
                else:
                    print(f"# {normality_status.capitalize()} → {test_type} (pas de calcul analytique possible)")
        else:
            # Si uniquement Monte Carlo, tester quand même pour information
            if len(group0) > 20 and len(group1) > 20:
                norm_test_0 = normaltest(group0).pvalue
                norm_test_1 = normaltest(group1).pvalue
                var_test_p = levene(group0, group1).pvalue

                if norm_test_0 > 0.05 and norm_test_1 > 0.05:
                    test_type = "t-test"
                    normality_status = "normales"
                    t_stat, p_value = ttest_ind(group0, group1, equal_var=(var_test_p > 0.05))
                else:
                    test_type = "Mann-Whitney U"
                    normality_status = "non normales"
                    t_stat, p_value = mannwhitneyu(group0, group1, alternative='two-sided')
            else:
                test_type = "Mann-Whitney U"
                normality_status = "échantillons trop petits pour test de normalité"
                t_stat, p_value = mannwhitneyu(group0, group1, alternative='two-sided')

            if verbose:
                test_info = f"# {normality_status.capitalize()} → {test_type} (non utilisé pour Monte Carlo)"

        # Simulation Monte Carlo
        power_monte_carlo = None
        if method_powerAnaly in ['both', 'montecarlo']:
            if verbose:
                if 'test_info' in locals():
                    print(f"{test_info} | Exécution de {n_simulations_monte} simulations Monte Carlo...")
                else:
                    print(f"# Exécution de {n_simulations_monte} simulations Monte Carlo...")

            results_parallel = Parallel(n_jobs=n_jobs)(
                delayed(run_single_simulation_auto)(group0, group1, sample_fraction, alpha)
                for _ in range(n_simulations_monte)
            )
            power_monte_carlo = np.mean(results_parallel)

            if verbose:
                print(f"# Puissance Monte Carlo calculée : {power_monte_carlo:.4f}")

        # Stockage des résultats
        result_row = {
            'Feature': feature,
            'Sample_Size': len(X_filtered),
            'Group0_Size': len(group0),
            'Group1_Size': len(group1),
            'Effect_Size': effect_size,
            'P_Value': p_value,
            'Test_Type': test_type,
            'Normality_Status': normality_status,
            'Required_N': np.ceil(required_n) if required_n is not None else np.nan,
            'Power_Analytical': power_analytical,
            'Power_MonteCarlo': power_monte_carlo,
        }

        results.append(result_row)

    # Résumé final
    if verbose:
        non_normal_count = sum(1 for r in results if r['Normality_Status'] == 'non normales')
        normal_count = sum(1 for r in results if r['Normality_Status'] == 'normales')
        small_sample_count = sum(
            1 for r in results if r['Normality_Status'] == 'échantillons trop petits pour test de normalité')

        print(f"\nRésumé des analyses :")
        print(f"- Nombre total de caractéristiques analysées : {len(results)}")
        print(f"- Distributions normales : {normal_count}")
        print(f"- Distributions non normales : {non_normal_count}")
        print(f"- Échantillons trop petits pour test de normalité : {small_sample_count}")

        if method_powerAnaly in ['both', 'analytical']:
            analytical_computed = sum(1 for r in results if r['Power_Analytical'] is not None)
            print(f"- Puissance analytique calculée pour {analytical_computed} caractéristiques")

        if method_powerAnaly in ['both', 'montecarlo']:
            print(f"- Puissance Monte Carlo calculée pour {len(results)} caractéristiques")

    return pd.DataFrame(results)


###############################################################################
# Fonction Gestion de la selection des features
###############################################################################
def compute_display_statistic(X=None, Y=None, name="Dataset",
                              config=None,
                              compute_feature_stat=False,
                              is_compute_vif=False):
    """
    Analyse complète des features d'un DataFrame avec options de filtrage.

    Cette fonction combine :
    1. L'analyse des valeurs manquantes et nulles
    2. (Optionnel) Le calcul des statistiques de features : Corrélation, MI, mRMR, Fisher
    3. L'affichage final avec ou sans statut de filtrage

    Args:
        X (pd.DataFrame): DataFrame des features
        Y (pd.Series): Série cible
        name (str): Nom du dataset pour l'affichage
        config (dict): Paramètres divers :
            - vif_threshold
            - corr_threshold, mi_threshold
            - mrmr_n_features, mrmr_score_threshold
            - fisher_score_threshold, fisher_pvalue_threshold, fisher_top_n_features
            - excluded_columns_principal, excluded_columns_tradeDirection,
              excluded_columns_CorrCol, excluded_columns_category
        compute_feature_stat (bool): Si True, calcule CORR + MI + mRMR + Fisher.
                                     Sinon, aucun calcul (Mode NO_FILTERING).
        is_compute_vif (bool): Si True, calcule le VIF avant toute autre analyse.

    Returns:
        list: Liste des colonnes conservées (si filtrage) ou None si pas de filtrage.
    """
    if compute_feature_stat:
        import seaborn as sns
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        if config['nb_pca'] > 0:
            # Liste des colonnes à analyser pour la PCA
            pca_columns = [
                'overlap_ratio_VA_11P_21P', 'poc_diff_11P_21P', 'poc_diff_ratio_6P_11P',
                'poc_diff_6P_21P', 'poc_diff_ratio_11P_21P', 'poc_diff_6P_16P', 'ratio_delta_vol_VA11P',
                'overlap_ratio_VA_6P_11P', 'poc_diff_6P_11P'
                #  ,'ratio_vol_VolCont_ZoneA_xTicksContZone','ratio_vol_VolCont_ZoneB_xTicksContZone','ratio_vol_VolCont_ZoneC_xTicksContZone',
                #           'ratio_delta_VolCont_ZoneA_xTicksContZone','ratio_delta_VolCont_ZoneB_xTicksContZone','ratio_delta_VolCont_ZoneC_xTicksContZone',
            ]
            # pca_columns = list(X.columns)
            # pca_columns=[]
            # Pour les mettre dans pca_columns
            if pca_columns:
                pca_columns = pca_columns
                X, high_corr_columns_used4pca, pca_object = apply_pca_on_scaledData_and_high_corr_features(X, pca_columns,
                                                                                            corr_threshold=0.75,
                                                                                            n_components=2,
                                                                                            plot_correlation=False,
                                                                                            pca_prefix="pca_")

    # Chargement des paramètres
    vif_threshold = config.get('vif_threshold', 0)
    retained_only_vif = config.get('retained_only_vif', 0)
    method_powerAnaly = config.get('method_powerAnaly', 0)
    n_simulations_monte = config.get('n_simulations_monte', 0)
    powAnaly_threshold = config.get('powAnaly_threshold', 0.8)
    corr_threshold = config.get('corr_threshold', 0)
    mi_threshold = config.get('mi_threshold', 0)

    # Calculer les min et max pour chaque colonne de X_non_sc (l'original non transformé)
    min_values = X.min()
    max_values = X.max()

    # Fonction pour formater de manière sécurisée les valeurs
    def format_range(min_val, max_val):
        try:
            # Essayer de formater comme nombre
            return f"[{float(min_val):.2f}, {float(max_val):.2f}]"
        except (ValueError, TypeError):
            # Si ce n'est pas un nombre, retourner comme chaîne
            return f"[{min_val}, {max_val}]"

    # Créer une liste de chaînes formatées "[min, max]" pour chaque feature
    ranges = [format_range(min_values[col], max_values[col]) for col in X.columns]

    # # Vérifier les valeurs min et max spécifiquement pour cette colonne
    # print("Type de deltaTimestampOpeningSession1min:", type(X['deltaTimestampOpeningSession1min'].iloc[0]))
    # print("Min:", X['deltaTimestampOpeningSession1min'].min())
    # print("Max:", X['deltaTimestampOpeningSession1min'].max())

    # 1) Création du DataFrame d'analyse basique
    analysis_df = pd.DataFrame()
    analysis_df['Feature'] = X.columns
    if is_compute_vif:
        analysis_df['Range_scaled'] = ranges
    else:
        analysis_df['Range'] = ranges

    analysis_df['NaN Count'] = X.isna().sum().values
    analysis_df['NaN%'] = (X.isna().mean() * 100).values
    analysis_df['Zeros%'] = ((X == 0).mean() * 100).values
    analysis_df['NaN+Zeros%'] = analysis_df['NaN%'] + analysis_df['Zeros%']
    analysis_df['non NaN'] = len(X) - analysis_df['NaN Count']
    analysis_df['non NaN%'] = 100 - analysis_df['NaN%']

    # for col in X.columns:
    #     print(f"🔍 {col}: Min={X[col].min()}, Max={X[col].max()}")
    has_status = False
    retained_columns = None
    vif_full = None

    # 2) Calcul du VIF (uniquement si demandé)
    if is_compute_vif:
        try:
            print('----compute vif')
            with np.errstate(divide='ignore', invalid='ignore'):
                def remove_zero_variance_features(df):
                    df_numeric = df.select_dtypes(include=[np.number])
                    # On calcule la variance de chaque colonne
                    variances = df_numeric.var()
                    # On repère les colonnes dont la variance est (strictement) égale à zéro
                    zero_var_cols = variances[variances == 0.0].index
                    # On les retire du DataFrame original
                    if len(zero_var_cols) > 0:
                        print(f"Suppression des colonnes à variance nulle : {list(zero_var_cols)}")
                        df = df.drop(columns=zero_var_cols, errors='ignore')
                    return df

                # Exemple d'utilisation :
                vif_full = compute_vif_fast(X, vif_threshold)
                analysis_df = pd.merge(analysis_df, vif_full, on='Feature', how='left')
                if 'StatusVif' in analysis_df.columns:
                    print("is_compute_vif= true ✅ Colonne StatusVif correctement ajoutée à analysis_df")
                else:
                    print("is_compute_vif= true ❌ Colonne StatusVif non présente dans analysis_df après fusion")
                # Colonnes conservées selon le statut "Conservé"
                retained_columns = vif_full[vif_full['StatusVif'] == 'Conservé']['Feature'].tolist()
                if retained_columns:
                    if retained_only_vif:
                        X_afterVIF = X[retained_columns]

                    else:
                        X_afterVIF = X  # permet d'afficher toutes ls valeurs meme les redonndantes
                    X_afterVIF = X[retained_columns]

                else:
                    print("⚠️ Aucune colonne conservée après VIF. Utilisation de X d'origine.")
                    X_afterVIF = X.copy()  # Pour éviter un DataFrame vide
                    exit(35)

        except Exception as e:
            print(f"Erreur lors du calcul VIF: {str(e)}")
            # On considère ici qu'on ne peut pas continuer le filtrage
            exit(36)
    else:
        X_afterVIF = X.copy()

    if X_afterVIF.shape[1] == 0:
        raise ValueError("🚨 Aucune colonne n'a été conservée après le filtrage VIF. Impossible de continuer.")

    # 3) Si compute_feature_stat=True, on calcule TOUTES les stats (Corr, MI, mRMR, Fisher)
    if compute_feature_stat:
        # 3.0) Analyse de puissance statistique
        power_results = calculate_statistical_power_job(X_afterVIF, Y, target_power=powAnaly_threshold,
                                                        n_simulations_monte=n_simulations_monte,
                                                        sample_fraction=0.8, verbose=True,
                                                        method_powerAnaly=method_powerAnaly)

        # Sélectionner les colonnes à garder selon la méthode
        cols_to_keep = ['Feature']

        # Créer une copie pour éviter WarningSettingWithCopyWarning
        power_results_subset = power_results[['Feature']].copy()

        if method_powerAnaly == "analytical":
            # Ajouter la colonne de puissance analytique renommée en 'powStat'
            power_results_subset['powStat'] = power_results['Power_Analytical']
            print("\nMéthode analytique utilisée")
        elif method_powerAnaly == "montecarlo":
            # Ajouter la colonne de puissance Monte Carlo renommée en 'powStat'
            power_results_subset['powStat'] = power_results['Power_MonteCarlo']
            print("\nMéthode Monte Carlo utilisée")
        elif method_powerAnaly == "both":
            # Ajouter la colonne de puissance Monte Carlo renommée en 'powStat'
            power_results_subset['powStat'] = power_results['Power_MonteCarlo']
            print("\nMéthode combinée utilisée (valeur Monte Carlo conservée)")
        else:
            print(f"Méthode invalide: {method_powerAnaly}")
            exit(23)

        # Fusionner les résultats avec analysis_df
        analysis_df = pd.merge(analysis_df, power_results_subset, on='Feature', how='left')

        # Afficher les colonnes 'Feature' et 'powStat' pour vérification

        # 3.1) Corrélation
        merged_dfVIF_target = pd.concat([X_afterVIF, Y], axis=1)
        merged_dfVIF_target.columns = list(X_afterVIF.columns) + ['target']
        corr_df = compute_and_keep_correled(merged_dfVIF_target)
        analysis_df = pd.merge(analysis_df, corr_df, on='Feature', how='left')

        # 3.2) Mutual Information (MI)
        if Y.nunique() <= 20 or Y.dtype == 'object' or Y.dtype == 'category' or np.issubdtype(Y.dtype, np.integer):
            mi_scores = mutual_info_classif(X_afterVIF, Y)
        else:
            mi_scores = mutual_info_regression(X_afterVIF, Y)

        mi_df = pd.DataFrame({'Feature': X_afterVIF.columns, 'mi': mi_scores})
        analysis_df = pd.merge(analysis_df, mi_df, on='Feature', how='left')

        # 3.3) mRMR
        selected, mrmr_scores = compute_mRMR_filtering(X_afterVIF, Y, config)
        analysis_df['mrmr_score'] = analysis_df['Feature'].map(mrmr_scores)

        # 3.4) Fisher
        fisher_df = fisher_score_feature_selection(X_afterVIF, Y, name, config)
        # Renommer si nécessaire pour correspondre aux colonnes attendues
        fisher_df = fisher_df.rename(columns={
            'Fisher_Score': 'Fisher_Score',
            'p-value': 'p-value'
        })
        analysis_df = pd.merge(analysis_df, fisher_df, on='Feature', how='left')

        # On peut décider d'un 'Status' final → Par exemple, on conserve toutes les colonnes
        # ou bien on applique un filtrage selon un critère global.
        # analysis_df['Status'] = True
        retained_columns = X_afterVIF.columns.tolist()

    else:
        # 4) Mode NO_FILTERING : aucun calcul
        print("✋ compute_feature_stat=False → Aucune statistique (CORR, MI, mRMR, Fisher) n'est calculée.")
        # analysis_df['Status'] = True  # tout est "conservé"
        retained_columns = None

    ############################################################################
    # Préparation des colonnes pour l'affichage
    ############################################################################
    # Fixed section of the compute_display_statistic function
    # Replace the section where headers are defined

    # Préparation des colonnes pour l'affichage
    ############################################################################
    # Préparation des colonnes pour l'affichage
    if is_compute_vif:
        base_columns = ['Feature', 'Range_scaled']
    else:
        base_columns = ['Feature', 'Range']

    # if has_status:
    #     base_columns.append('Status')

    # Création de is_vif et is_stat
    if is_compute_vif:
        conserved_features = set(vif_full[vif_full['StatusVif'] == 'Conservé']['Feature'])
        analysis_df['is_vif'] = analysis_df['Feature'].apply(lambda x: x in conserved_features)
        df_vif_true = analysis_df[analysis_df['is_vif'] == True]
        # print(df_vif_true[['Feature', 'is_vif']])

    if compute_feature_stat:
        analysis_df['is_stat'] = compute_feature_stat

    # Only add 'is_vif' if we are actually doing a VIF analysis
    if is_compute_vif:
        base_columns.append('is_vif')

    # Only add 'is_stat' if we are actually computing stats
    if compute_feature_stat:
        base_columns.append('is_stat')

    # On check s'il existe une colonne 'powStat'
    power_column = []
    if 'powStat' in analysis_df.columns:
        power_column = ['powStat']
    # On check s'il existe des colonnes de corrélation
    correlation_columns = []
    if 'lr_target' in analysis_df.columns and 'nlr_target' in analysis_df.columns:
        correlation_columns = ['lr_target', 'nlr_target']

    # On check s'il existe une colonne 'mi' (info mutuelle)
    mi_column = []
    if 'mi' in analysis_df.columns:
        mi_column = ['mi']

    # On check s'il existe une colonne 'mrmr_score'
    mrmr_col = []
    if 'mrmr_score' in analysis_df.columns:
        mrmr_col = ['mrmr_score']

    # Colonnes Fisher (si présentes)
    fisher_cols = []
    if 'Fisher_Score' in analysis_df.columns:
        fisher_cols.append('Fisher_Score')
    if 'p-value' in analysis_df.columns:
        fisher_cols.append('p-value')

    # Colonnes NaN
    nan_columns = ['NaN Count', 'NaN%', 'Zeros%', 'NaN+Zeros%', 'non NaN', 'non NaN%']

    # Colonnes VIF (si présentes)
    vif_cols = [col for col in analysis_df.columns if col.startswith('VIF')]

    # Construction de la liste finale des colonnes à afficher - NOUVEL ORDRE
    # Nouvel ordre: Feature, (Status), Corr, MI, mRMR, Fisher, NaN+stats, VIF
    analysis_df_columns = base_columns + power_column + correlation_columns + mi_column + mrmr_col + fisher_cols + nan_columns + vif_cols

    # Vérification pour éviter KeyError si certaines colonnes n'existent pas
    analysis_df_columns = [col for col in analysis_df_columns if col in analysis_df.columns]

    analysis_df = analysis_df[analysis_df_columns]

    # Instead of building headers piecewise again, just copy them directly:
    headers = analysis_df_columns.copy()
    # headers += vif_cols

    #####################
    # Définition des formats d'affichage
    #####################
    base_format = ["{:<56}"]  # Pour 'Feature'
    base_format.append("{:<15}")  # Range
    # if has_status:
    #     base_format.append("{:<6}")  # Pour 'Status'

    # Ensuite pour is_vif et is_stat
    if is_compute_vif:
        base_format.append("{:<6}")  # is_vif
        base_format.append("{:<6}")  # is_stat

    # power_column
    power_format = ["{:12}"] if power_column else []

    # Corrélation
    correlation_format = ["{:<9}", "{:<12}"] if correlation_columns else []
    # MI
    mi_format = ["{:^6}"] if mi_column else []
    # mrmr
    mrmr_format = ["{:>10}"] if mrmr_col else []
    # Fisher
    fisher_formats = ["{:>12}", "{:<9}"] if len(fisher_cols) > 0 else []
    # NaN
    nan_format = [
        "{:>8}",  # NaN Count
        "{:>8}",  # NaN%
        "{:>8}",  # Zeros%
        "{:>12}",  # NaN+Zeros%
        "{:>12}",  # Val non NaN
        "{:>12}",  # Val non NaN%
    ]
    # VIF
    vif_formats = ["{:>12}"] * len(vif_cols)

    full_format = (
            base_format
            + power_format
            + correlation_format
            + mi_format
            + mrmr_format
            + fisher_formats
            + nan_format
            + vif_formats
    )

    # Impression de l'entête (UNE SEULE FOIS)
    header_line = ''
    for header, fmt in zip(headers, full_format):
        header_line += fmt.format(header) + ' '
    print(header_line.strip())
    print("-" * len(header_line))

    # Codes couleur ANSI (optionnel)
    RED = '\033[91m'
    BLUE = '\033[94m'
    ORANGE = '\033[93m'
    YELLOW = '\033[33m'
    RESET = '\033[0m'

    # Récupération des colonnes exclues éventuelles (pour colorer les lignes)
    excluded_columns_principal = config.get('excluded_columns_principal', [])
    excluded_columns_tradeDirection = config.get('excluded_columns_tradeDirection', [])
    excluded_columns_CorrCol = config.get('excluded_columns_CorrCol', [])
    excluded_columns_category = config.get('excluded_columns_category', [])

    # 13) Créer la colonne `is_vif` : True si la feature est "Conservé" par VIF, False sinon
    # Création de la colonne is_vif indépendamment de StatusVif
    if is_compute_vif:
        # Alternative: se baser sur le résultat de vif_full directement
        conserved_features = set(vif_full[vif_full['StatusVif'] == 'Conservé']['Feature'].tolist())
        taille = len(conserved_features)
        print(f"La taille de l'ensemble est : {taille}")
        print("feature au statut VIF conservé: ",conserved_features)
        analysis_df['is_vif'] = analysis_df['Feature'].apply(lambda x: x in conserved_features)

    # S'assurer que is_vif est incluse dans l'affichage
    if 'is_vif' in analysis_df.columns:
        # Si is_vif existe mais n'est pas dans base_columns, l'ajouter
        if 'is_vif' not in base_columns:
            print("base_columns.append('is_vif') !!!!!!!!!!")
            base_columns.append('is_vif')
            exit(78)

    # IMPORTANT: Inclure StatusVif avant de filtrer
    # status_vif_col = ['StatusVif'] if 'StatusVif' in analysis_df.columns else []
    analysis_df_columns = base_columns + power_column + correlation_columns + mi_column + mrmr_col + fisher_cols + nan_columns + vif_cols  # + status_vif_col

    # Filtrer les colonnes UNE SEULE FOIS
    analysis_df_columns = [col for col in analysis_df_columns if col in analysis_df.columns]
    analysis_df = analysis_df[analysis_df_columns]

    # 14) Créer la colonne `is_stat` : True si la feature valide au moins un seuil statistique
    if (compute_feature_stat):
        analysis_df["is_stat"] = False  # Valeur par défaut

    # Récupération des seuils depuis config (ou valeurs par défaut)
    mrmr_thresh = config.get("mrmr_score_threshold", -np.inf)
    mi_thresh = config.get("mi_threshold", 0)
    fisher_thresh = config.get("fisher_score_threshold", 0)
    pval_thresh = config.get("fisher_pvalue_threshold", 0.05)

    # Affichons tous les seuils pour vérification
    print(
        f"\nSeuils utilisés: corr={corr_threshold}, mrmr={mrmr_thresh}, mi={mi_thresh}, fisher={fisher_thresh}, pval={pval_thresh}")

    def passes_stat_thresholds(row):
        """
        Évalue si une ligne (feature) passe au moins un des seuils statistiques,
        indépendamment du statut VIF.
        """

        # Gardons une trace des tests effectués
        conditions_tested = []

        # Corrélations (lr_target et nlr_target)
        if "lr_target" in row and isinstance(row["lr_target"], (int, float)):
            lr_val = abs(row["lr_target"])
            conditions_tested.append(f"lr_target={lr_val}, seuil={corr_threshold}, résultat={lr_val > corr_threshold}")
            if lr_val > corr_threshold:
                # print(f"Condition passée: lr_target={lr_val} > {corr_threshold}")
                return True

        if "nlr_target" in row and isinstance(row["nlr_target"], (int, float)):
            nlr_val = abs(row["nlr_target"])
            conditions_tested.append(
                f"nlr_target={nlr_val}, seuil={corr_threshold}, résultat={nlr_val > corr_threshold}")
            if nlr_val > corr_threshold:
                # print(f"Condition passée: nlr_target={nlr_val} > {corr_threshold}")
                return True

        # mrmr_score
        if "mrmr_score" in row and isinstance(row["mrmr_score"], (int, float)):
            mrmr_val = row["mrmr_score"]
            conditions_tested.append(f"mrmr_score={mrmr_val}, seuil={mrmr_thresh}, résultat={mrmr_val > mrmr_thresh}")
            if mrmr_val > mrmr_thresh:
                # print(f"Condition passée: mrmr_score={mrmr_val} > {mrmr_thresh}")
                return True

        # mi
        if "mi" in row and isinstance(row["mi"], (int, float)):
            mi_val = row["mi"]
            conditions_tested.append(f"mi={mi_val}, seuil={mi_thresh}, résultat={mi_val > mi_thresh}")
            if mi_val > mi_thresh:
                # print(f"Condition passée: mi={mi_val} > {mi_thresh}")
                return True

        # Fisher_Score
        if "Fisher_Score" in row and isinstance(row["Fisher_Score"], (int, float)) and not pd.isna(row["Fisher_Score"]):
            fisher_val = row["Fisher_Score"]
            conditions_tested.append(
                f"Fisher_Score={fisher_val}, seuil={fisher_thresh}, résultat={fisher_val > fisher_thresh}")
            if fisher_val > fisher_thresh:
                # print(f"Condition passée: Fisher_Score={fisher_val} > {fisher_thresh}")
                return True

        # p-value
        if "p-value" in row and isinstance(row["p-value"], (int, float)) and not pd.isna(row["p-value"]):
            pval = row["p-value"]
            conditions_tested.append(f"p-value={pval}, seuil={pval_thresh}, résultat={pval <= pval_thresh}")
            if pval <= pval_thresh:
                # print(f"Condition passée: p-value={pval} <= {pval_thresh}")
                return True

        # Si aucune condition n'est passée, affichons toutes les vérifications effectuées
        # print(f"Aucune condition passée pour cette ligne. Tests effectués: {conditions_tested}")
        # print(
        #    f"Valeurs de la ligne: {dict((k, v) for k, v in row.items() if k in ['lr_target', 'nlr_target', 'mrmr_score', 'mi', 'Fisher_Score', 'p-value'])}")
        return False

    if (compute_feature_stat):
        analysis_df["is_stat"] = analysis_df.apply(passes_stat_thresholds, axis=1)

    ############# fin de is_vif et is_stat #################

    ############# debut du tri par groupe #################

    ############# debut du tri par groupe #################

    def group_order(row):
        vif = row["is_vif"]  # True/False
        stat = row["is_stat"]  # True/False

        # Groupe 1 : non retenu VIF + ne valide pas stat
        if (not vif) and (not stat):
            return 1
        # Groupe 2 : non retenu VIF + valide stat
        elif (not vif) and stat:
            return 2
        # Groupe 3 : retenu VIF + ne valide pas stat
        elif vif and (not stat):
            return 3
        # Groupe 4 : retenu VIF + valide stat
        else:
            return 4

    if is_compute_vif:
        # Appliquer group_order UNE SEULE FOIS
        analysis_df["group_order"] = analysis_df.apply(group_order, axis=1)

        # Effectuer UN SEUL tri qui prend en compte les deux critères
        if 'powStat' in analysis_df.columns:
            analysis_df = analysis_df.sort_values(
                by=["group_order", "powStat"],
                ascending=[True, True]
            )
        else:
            # Si powStat n'existe pas, trier uniquement par groupe
            analysis_df = analysis_df.sort_values(by="group_order", ascending=True)
    else:
        # Si is_compute_vif est False, trier uniquement par powStat si disponible
        if 'powStat' in analysis_df.columns:
            analysis_df = analysis_df.sort_values(by='powStat', ascending=True)

    ############# fin du tri par groupe #################

    # SUPPRIMER OU COMMENTER ces lignes qui recalculent et trient une deuxième fois
    # if is_compute_vif:
    #     analysis_df = analysis_df.sort_values(by="group_order", ascending=True)

    # Vérification qu'elles existent
    analysis_df_columns = [c for c in analysis_df_columns if c in analysis_df.columns]
    analysis_df = analysis_df[analysis_df_columns]

    features_data = []

    for i in range(len(analysis_df)):
        row = analysis_df.iloc[i]
        output_values = []
        output_colors = []
        if is_compute_vif:

            if row['is_vif'] and row['is_stat'] and row['p-value'] <= config["fisher_pvalue_threshold"]:
                features_data.append({
                    'Feature': row['Feature'],
                    'is_vif': row['is_vif'],
                    'is_stat': row['is_stat'],
                    'powStat': row['powStat'],
                    'lr_target': row['lr_target'],
                    'nlr_target': row['nlr_target'],
                    'mi': row['mi'],
                    'mrmr_score': row['mrmr_score'],
                    'Fisher_Score': row['Fisher_Score'],
                    'p-value': row['p-value']
                })

        # (1) Feature
        output_values.append(row['Feature'])
        output_colors.append('')
        if is_compute_vif:
            output_values.append(row['Range_scaled'])
        else:
            output_values.append(row['Range'])

        output_colors.append('')

        # 🔥 Vérification stricte AVANT impression de la ligne
        # if len(output_values) != len(full_format):
        #     print("🚨 ERREUR : Décalage entre output_values et full_format !")
        #     print(f"➡ Nombre d'éléments dans output_values : {len(output_values)}")
        #     print(f"➡ Nombre d'éléments attendus (full_format) : {len(full_format)}")
        #     print(f"❌ Différence = {len(full_format) - len(output_values)}")
        #     print(f"👉 Contenu de output_values : {output_values}\n")
        #     print(f"👉 Contenu de full_format  : {full_format}\n")
        #
        #     raise ValueError("Décalage entre output_values et full_format. Vérifiez l'ajout des colonnes.")

        # (2) Status (si présent)
        # if has_status:
        #     # True/False => on ne colorie pas ici
        #     status_str = 'True' if bool(row['Status']) else 'False'
        #     output_values.append(status_str)
        #     output_colors.append('')

        ######################################################
        # (3) Gérer la colonne 'is_vif' si elle existe
        ######################################################
        if 'is_vif' in analysis_df.columns:

            vif_val = bool(row['is_vif'])
            if is_compute_vif:
                # Couleur si True/False
                output_values.append(f"{BLUE}True{RESET}" if vif_val else f"{RED}False{RESET}")
            else:
                # On affiche juste la valeur brute "True" / "False"
                output_values.append(str(vif_val))

            output_colors.append('')  # ou la couleur de fond, etc.

        ######################################################
        # (4) Gérer la colonne 'is_stat' si elle existe
        ######################################################
        if 'is_stat' in analysis_df.columns:
            stat_val = bool(row['is_stat'])
            if compute_feature_stat:
                output_values.append(f"{BLUE}True{RESET}" if stat_val else f"{RED}False{RESET}")
            else:
                output_values.append(str(stat_val))

            output_colors.append('')

        # 4.0) Power stat
        if 'powStat' in analysis_df.columns:  # Changez 'powAnaly' en 'powStat'
            powStat = row.get('powStat', 'n.a')  # Changez également ici
            if powStat != 'n.a' and not pd.isna(powStat):
                power_format = f"{powStat:.2f}"
                output_values.append(power_format)
                output_colors.append(BLUE if powStat > powAnaly_threshold else '')
            elif 'is_vif' in analysis_df.columns:

                output_values.append('n.a')
                output_colors.append('')

        # 4) Corrélation (lr_target et nlr_target), s'ils existent
        if 'lr_target' in analysis_df.columns and 'nlr_target' in analysis_df.columns:
            lr_val = row.get('lr_target', 'n.a')
            nlr_val = row.get('nlr_target', 'n.a')

            # lr_target
            if lr_val != 'n.a' and not pd.isna(lr_val):
                # On formate en nombre à 2 décimales
                lr_formatted = f"{lr_val:.2f}"
                output_values.append(lr_formatted)
                # Exemple : colorier en BLEU si la corrélation dépasse un seuil
                output_colors.append(BLUE if abs(lr_val) > corr_threshold else '')
            elif 'is_vif' in analysis_df.columns:
                output_values.append('n.a')
                output_colors.append('')

            # nlr_target
            if nlr_val != 'n.a' and not pd.isna(nlr_val):
                nlr_formatted = f"{nlr_val:.2f}"
                output_values.append(nlr_formatted)
                output_colors.append(BLUE if abs(nlr_val) > corr_threshold else '')
            elif 'is_vif' in analysis_df.columns:
                output_values.append('n.a')
                output_colors.append('')

        # 5) MI (Mutual Information)
        if 'mi' in analysis_df.columns:
            mi_val = row.get('mi', 'n.a')
            if mi_val != 'n.a' and not pd.isna(mi_val):
                mi_formatted = f"{mi_val:.3f}"
                output_values.append(mi_formatted)
                output_colors.append(BLUE if mi_val > mi_threshold else '')
            elif 'is_vif' in analysis_df.columns:
                output_values.append('n.a')
                output_colors.append('')

        # 6) mrmr_score
        if 'mrmr_score' in analysis_df.columns:
            mrmr_val = row.get('mrmr_score', 'n.a')
            if mrmr_val != 'n.a' and not pd.isna(mrmr_val):
                # Format 4 décimales
                output_values.append(f"{mrmr_val:.4f}")
                # Colorer en BLEU si la valeur > threshold
                mrmr_score_threshold = config.get('mrmr_score_threshold', -np.inf)
                output_colors.append(BLUE if mrmr_val > mrmr_score_threshold else '')
            elif 'is_vif' in analysis_df.columns:
                output_values.append('n.a')
                output_colors.append('')

        # ───────────────────────────────────────────────────────────
        # 7) Fisher columns : Fisher_Score et p-value
        #    On veut qu'ils apparaissent JUSTE après mrmr_score
        # ───────────────────────────────────────────────────────────
        if 'Fisher_Score' in analysis_df.columns:
            fisher_score = row.get('Fisher_Score', 'n.a')
            if fisher_score != 'n.a' and not pd.isna(fisher_score):
                # 4 décimales pour le fisher score
                output_values.append(f"{fisher_score:.3f}")
                fisher_score_threshold = config.get('fisher_score_threshold', 0.0)
                # Coloration en BLEU si le fisher_score dépasse un seuil
                output_colors.append(BLUE if fisher_score > fisher_score_threshold else '')
            elif 'is_vif' in analysis_df.columns:
                output_values.append('n.a')
                output_colors.append('')
        if 'p-value' in analysis_df.columns:
            p_value = row.get('p-value', 'n.a')
            if p_value != 'n.a' and not pd.isna(p_value):
                output_values.append(f"{p_value:.3f}")
                # Colorer si p-value < 0.05, par exemple
                p_value_threshold = config.get('fisher_pvalue_threshold', 0.05)
                output_colors.append(BLUE if p_value <= p_value_threshold else '')
            elif 'is_vif' in analysis_df.columns:
                output_values.append('n.a')
                output_colors.append('')

        # 8) Statistiques NaN + Zeros
        nan_count = row['NaN Count']
        nan_percentage = row['NaN%']
        zeros_percentage = row['Zeros%']
        total_percentage = row['NaN+Zeros%']
        non_nan_count = row['non NaN']
        non_nan_percentage = row['non NaN%']

        output_values.extend([
            nan_count,  # Utilisez les vraies valeurs au lieu de 1, 2, 3, 4...
            f"{nan_percentage:.2f}",
            f"{zeros_percentage:.2f}",
            f"{total_percentage:.2f}",
            non_nan_count,
            f"{non_nan_percentage:.2f}"
        ])
        output_colors.extend([''] * 6)

        # 9) VIF columns
        if is_compute_vif and vif_cols:  # Vérifie d'abord si on doit calculer VIF

            vif_values = []
            for col_vif in vif_cols:
                val = row.get(col_vif, 'n.a')  # Récupérer la valeur VIF

                # Vérification si c'est une chaîne ou un nombre
                if isinstance(val, str) and val == 'n.a':
                    vif_values.append('n.a')
                elif not pd.isna(val):
                    if isinstance(val, (int, float)):
                        # Format float sur 2 décimales
                        vif_values.append(f"{val:.2f}")
                    else:
                        # Valeur déjà en notation scientifique
                        vif_values.append(val)
                else:
                    vif_values.append('n.a')

            # Ajout des VIF dans output_values et la coloration possible
            for idx_vif, vif_val in enumerate(vif_values):
                output_values.append(vif_val)

                # Exemple de coloration
                if idx_vif == 0 and vif_val not in ['n.a', 'NaN']:
                    try:
                        if float(vif_val) > vif_threshold:
                            output_colors.append(BLUE)
                        else:
                            output_colors.append('')
                    except ValueError:
                        output_colors.append('')
                else:
                    output_colors.append('')
        else:
            # Si VIF n'est pas calculé, ne rien ajouter pour éviter les colonnes vides
            output_values.append('n.a')
            output_colors.append('')

        # 10) Couleur de ligne selon exclusions
        row_color = ''
        # # Si la feature est Status = False
        # if has_status and not row['Status']:
        #     row_color = RED

        feature_name = row['Feature']
        if feature_name in excluded_columns_principal:
            row_color = RED
        elif feature_name in excluded_columns_CorrCol:
            row_color = ORANGE
        elif feature_name in excluded_columns_tradeDirection:
            row_color = BLUE
        elif feature_name in excluded_columns_category:
            row_color = YELLOW

        # 11) Application de la couleur et formatage final
        formatted_values = []
        for val, fmt, color_code in zip(output_values, full_format, output_colors):
            tmp = fmt.format(val)
            if color_code:
                tmp = f"{color_code}{tmp}{RESET}"
            elif row_color:
                tmp = f"{row_color}{tmp}{RESET}"
            formatted_values.append(tmp)

        # 12) Impression de la ligne résultante
        print(' '.join(formatted_values))
        # --- Fin de la boucle for idx, row in analysis_df.iterrows() ---

        # # Retourne la liste des colonnes retenues si un filtrage a été appliqué
    # if auto_filtering_mode in [AutoFilteringOptions.ENABLE_VIF_CORR_MI, AutoFilteringOptions.ENABLE_MRMR,
    #                            AutoFilteringOptions.ENABLE_FISHER]:
    # Impression de l'entête (UNE SEULE FOIS)
    header_line = ''
    for header, fmt in zip(headers, full_format):
        header_line += fmt.format(header) + ' '
    print("-" * len(header_line))
    print(header_line.strip())
    # if compute_feature_stat ==True:
    #     return retained_columns

    # À la fin de la fonction compute_display_statistic, remplacez le bloc de retour par ceci:
    if compute_feature_stat == True:
        # Récupérer les colonnes PCA qui ont été générées
        pca_columns_created = [col for col in X.columns if col.startswith('pca_')]

        # Créer une structure pca_info
        pca_info = {
            'applied': len(pca_columns_created) > 0,
            'pca_columns_created': pca_columns_created,
            'high_corr_columns_used4pca': high_corr_columns_used4pca if 'high_corr_columns_used4pca' in locals() else [],
            'pca_object': pca_object if 'pca_object' in locals() else None  # Stocker l'objet PCA
        }
        df_feature_vifAndStat_selected = pd.DataFrame(features_data)
        df_feature_vifAndStat_selected = df_feature_vifAndStat_selected.drop_duplicates(subset='Feature', keep='first')
        print(df_feature_vifAndStat_selected)
        print(f"Nombre de feature slectionnées avec VIF et Stat: {len(df_feature_vifAndStat_selected)}")

        # Extract the feature names from the DataFrame
        selected_features = df_feature_vifAndStat_selected['Feature'].tolist()

        print(f"Features Vif, Stat avec PCA_ : \n X[selected_features]: \n {X[selected_features]} \nde longeur {len(X[selected_features])}")
        global x_train_vifStatPCA_initDataSet_DEBUG_global
        x_train_vifStatPCA_initDataSet_DEBUG_global=X[selected_features]
        # Retourner un dictionnaire contenant les colonnes retenues et les informations sur la PCA

        return {
            # 'df_feature_vifAndStat_selected': df_feature_vifAndStat_selected,
            'colomnsList_with_pca_vif_stat': selected_features,
            # Renvoyer X qui contient maintenant les colonnes sélectionnées
            'pca_info': pca_info
        }

    return None


def check_distribution_coherence(X_train, X_test, p_threshold=0.01):
    """
    Vérifie la cohérence des distributions entre X_train et X_test via un test de Kolmogorov-Smirnov.
    Pour chaque variable, si la p-value est très faible, cela indique que les distributions diffèrent significativement.

    Args:
        X_train (pd.DataFrame): Données d'entraînement.
        X_test (pd.DataFrame): Données de test.
        p_threshold (float): Seuil de p-value pour considérer qu'il y a un écart significatif.

    Returns:
        dict: dictionnaire contenant les features qui diffèrent significativement.
    """
    from scipy.stats import ks_2samp

    differing_features = {}
    for col in X_train.columns:
        # On ignore les colonnes non numériques
        if not np.issubdtype(X_train[col].dtype, np.number):
            continue

        # Test KS (Kolmogorov-Smirnov)
        statistic, p_value = ks_2samp(X_train[col], X_test[col])
        if p_value < p_threshold:
            differing_features[col] = {'statistic': statistic, 'p_value': p_value}
    return differing_features


def check_value_ranges(X_train, X_test):
    """
    Vérifie si X_test contient des valeurs hors des bornes (min, max) observées dans X_train.
    Retourne un dict avec la liste des features et le nombre de valeurs hors bornes.
    """
    out_of_bounds = {}
    for col in X_train.columns:
        if not np.issubdtype(X_train[col].dtype, np.number):
            continue
        train_min, train_max = X_train[col].min(), X_train[col].max()
        test_values = X_test[col]
        below_min_mask = test_values < train_min
        above_max_mask = test_values > train_max

        below_min = below_min_mask.sum()
        above_max = above_max_mask.sum()

        if below_min > 0 or above_max > 0:
            out_of_bounds[col] = {
                'below_min_count': int(below_min),
                'above_max_count': int(above_max),
                'train_min': float(train_min),
                'train_max': float(train_max),
                'below_min_values': test_values[below_min_mask].tolist(),  # Ajout des valeurs spécifiques
                'above_max_values': test_values[above_max_mask].tolist()  # Ajout des valeurs spécifiques
            }
    return out_of_bounds

def apply_scaling(X_train, X_test=None, save_path=None, chosen_scaler=None):
    """
    Applique la normalisation sur les données d'entraînement et de test.
    Compatible avec DataFrame et numpy array.

    Args:
        X_train: Données d'entraînement à normaliser (DataFrame ou array)
        X_test: Données de test à normaliser (peut être None)
        save_path: Chemin optionnel pour sauvegarder les paramètres du scaler
        chosen_scaler: Type de scaler à utiliser

    Returns:
        tuple: (X_train_scaled, X_test_scaled or None, scaler, scaler_params)
    """
    import pandas as pd
    from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler

    # Détection du type d'entrée
    is_dataframe = isinstance(X_train, pd.DataFrame)

    if is_dataframe:
        columns = X_train.columns
        train_index = X_train.index
        test_index = X_test.index if X_test is not None else None

    if chosen_scaler == scalerChoice.SCALER_DISABLE:
        return X_train, X_test, None, None

    if chosen_scaler == scalerChoice.SCALER_ROBUST:
        scaler = RobustScaler()
        scaler_name = "RobustScaler"
    elif chosen_scaler == scalerChoice.SCALER_STANDARD:
        scaler = StandardScaler()
        scaler_name = "StandardScaler"
    elif chosen_scaler == scalerChoice.SCALER_MINMAX:
        scaler = MinMaxScaler()
        scaler_name = "MinMaxScaler"
    elif chosen_scaler == scalerChoice.SCALER_MAXABS:
        scaler = MaxAbsScaler()
        scaler_name = "MaxAbsScaler"
    else:
        exit(98)

    X_train_np = X_train.values if is_dataframe else X_train
    X_test_np = X_test.values if is_dataframe and X_test is not None else X_test

    X_train_scaled_np = scaler.fit_transform(X_train_np)
    X_test_scaled_np = scaler.transform(X_test_np) if X_test is not None else None

    if is_dataframe:
        X_train_scaled = pd.DataFrame(X_train_scaled_np, columns=columns, index=train_index)
        X_test_scaled = pd.DataFrame(X_test_scaled_np, columns=columns, index=test_index) if X_test is not None else None

        scaler_params = {'scaler_type': scaler_name, 'features': columns.tolist()}

        if chosen_scaler == scalerChoice.SCALER_ROBUST:
            scaler_params.update({
                'center': dict(zip(columns, scaler.center_)),
                'scale': dict(zip(columns, scaler.scale_))
            })
        elif chosen_scaler == scalerChoice.SCALER_MINMAX:
            scaler_params.update({
                'min': dict(zip(columns, scaler.min_)),
                'scale': dict(zip(columns, scaler.scale_))
            })
        elif chosen_scaler == scalerChoice.SCALER_MAXABS:
            scaler_params.update({
                'scale': dict(zip(columns, scaler.scale_)),
                'max_abs': dict(zip(columns, scaler.max_abs_))
            })
        else:
            scaler_params.update({
                'mean': dict(zip(columns, scaler.mean_)),
                'scale': dict(zip(columns, scaler.scale_))
            })
    else:
        X_train_scaled = X_train_scaled_np
        X_test_scaled = X_test_scaled_np
        scaler_params = {'scaler_type': scaler_name}

        if chosen_scaler == scalerChoice.SCALER_ROBUST:
            scaler_params.update({'center': scaler.center_.tolist(), 'scale': scaler.scale_.tolist()})
        elif chosen_scaler == scalerChoice.SCALER_MINMAX:
            scaler_params.update({'min': scaler.min_.tolist(), 'scale': scaler.scale_.tolist()})
        elif chosen_scaler == scalerChoice.SCALER_MAXABS:
            scaler_params.update({'scale': scaler.scale_.tolist(), 'max_abs': scaler.max_abs_.tolist()})
        else:
            scaler_params.update({'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist()})

    if save_path is not None and is_dataframe:
        import os
        import json
        filename = f"{scaler_name.lower()}_params.json"
        full_path = os.path.join(save_path, filename)
        os.makedirs(save_path, exist_ok=True)
        with open(full_path, 'w') as f:
            json.dump(scaler_params, f, indent=4)
        print(f"\nParamètres sauvegardés dans: {full_path}")

    return X_train_scaled, X_test_scaled, scaler, scaler_params




def save_features_with_sessions(df, custom_sections, file_path):
    # Sauvegarder le DataFrame normalement
    print(f"in save_features_with_sessions: {file_path}")
    df.to_csv(file_path, sep=';', index=False, encoding='iso-8859-1')

    # Ajouter CUSTOM_SECTIONS à la fin avec un séparateur spécial
    with open(file_path, 'a', encoding='iso-8859-1') as f:
        f.write('\n###CUSTOM_SECTIONS_START###\n')
        for section, values in custom_sections.items():
            f.write(f"{section};{values['start']};{values['end']};{values['session_type_index']};"
                    f"{values['selected']};{values['description']}\n")
        f.write('###CUSTOM_SECTIONS_END###\n')


from io import StringIO
import os
import pandas as pd

def load_features_and_sections(file_path: str, chunk_size: int = 100_000):
    """
    Charge le fichier CSV « features » et renvoie :
        • features_df : DataFrame avec uniquement la partie "données"
        • custom_sections : dict des sections personnalisées
    La partie "données" s’arrête juste avant ###CUSTOM_SECTIONS_START###.
    """
    try:
        # ─── 0) Vérifications préliminaires ──────────────────────────────
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        print(f"[load] → Fichier trouvé : {file_path}  "
              f"({os.path.getsize(file_path)/(1024**2):.2f} MB)")

        # ─── 1) Première passe : repérer les marqueurs ──────────────────
        start_idx, end_idx = None, None
        with open(file_path, "r", encoding="iso-8859-1") as f:
            for i, line in enumerate(f):
                if "###CUSTOM_SECTIONS_START###" in line:
                    start_idx = i          # ligne du marqueur
                elif "###CUSTOM_SECTIONS_END###" in line:
                    end_idx = i
                    break

        # Aucune section perso ? → on lit tout et on s’en va
        if start_idx is None:
            print("[load] Aucun marqueur trouvé → lecture complète")
            features_df = pd.read_csv(
                file_path, sep=";", encoding="iso-8859-1", low_memory=False
            ).dropna(how="all")             # nettoyage lignes vides
            return features_df, {}

        print(f"[load] Marqueurs trouvés : start={start_idx}, end={end_idx}")

        # ─── 2) Lecture de la partie « données » ───────────────────────────────
        data_rows = max(start_idx - 1, 0)  # <── ❶ on s’arrête AVANT le marqueur
        chunks = []
        for chunk in pd.read_csv(
                file_path,
                sep=";",
                encoding="iso-8859-1",
                nrows=data_rows,
                chunksize=chunk_size,
                low_memory=False):
            chunks.append(chunk)

        features_df = pd.concat(chunks, ignore_index=True)

        # Filtre sécurité : on ne garde que les vraies lignes de données
        if "session_id" in features_df.columns:
            features_df = features_df[
                pd.to_numeric(features_df["session_id"], errors="coerce").notna()
            ].copy()

        # Si, pour une raison ou une autre, il resterait des lignes 100 % NaN
        features_df = features_df.dropna(how="all")

        print(f"[load] Données chargées : {len(features_df):,} lignes")

        # ─── 3) Lecture / parsing des sections perso ────────────────────
        custom_sections = {}
        with open(file_path, "r", encoding="iso-8859-1") as f:
            for _ in range(start_idx + 1):   # on saute jusqu’après START
                next(f)
            for line in f:
                if "###CUSTOM_SECTIONS_END###" in line:
                    break
                if not line.strip():
                    continue
                parts = line.strip().split(";")
                if len(parts) < 6:
                    continue  # ligne mal formée
                section, start, end, type_idx, selected, *desc = parts
                custom_sections[section] = {
                    "start": int(start),
                    "end": int(end),
                    "session_type_index": int(type_idx),
                    "selected": selected.lower() == "true",
                    "description": ";".join(desc) if desc else "",
                }
        print(f"[load] Sections perso : {len(custom_sections)} trouvées")

        # ─── 4) Conversion sécurisée de deltaTimestampOpening ───────────
        if "deltaTimestampOpening" in features_df.columns:
            features_df["deltaTimestampOpening"] = (
                pd.to_numeric(features_df["deltaTimestampOpening"],
                              errors="coerce")
                  .fillna(0)
                  .astype(int)
            )

        return features_df, custom_sections

    except Exception as e:
        print(f"[load] ERREUR : {e}")
        raise




from sklearn.model_selection import BaseCrossValidator


class CustomSessionTimeSeriesSplit_byID(BaseCrossValidator):

    def __init__(self, session_ids, n_splits=5):
        self.session_ids = np.array(session_ids)
        self.n_splits = n_splits
        # Trier les sessions uniques pour garantir l'ordre chronologique
        self.unique_sessions = np.sort(np.unique(self.session_ids))

    def split(self, X, y=None, groups=None):
        n_sessions = len(self.unique_sessions)
        print("____CustomSessionTimeSeriesSplit_byID____")
        if self.n_splits >= n_sessions:
            raise ValueError(
                f"Le nombre de splits ({self.n_splits}) doit être inférieur au nombre de sessions uniques ({n_sessions}).")

        # Calculer la taille minimale pour chaque fold
        min_sessions_per_fold = n_sessions // (self.n_splits + 1)

        for fold in range(self.n_splits):
            # Calculer les indices pour ce fold
            train_end = min_sessions_per_fold * (fold + 1)
            val_start = train_end
            val_end = val_start + min_sessions_per_fold

            # Sélectionner les sessions
            train_sessions = self.unique_sessions[:train_end]
            val_sessions = self.unique_sessions[val_start:val_end]

            # Obtenir les indices correspondants
            train_mask = np.isin(self.session_ids, train_sessions)
            val_mask = np.isin(self.session_ids, val_sessions)

            train_indices = np.where(train_mask)[0]
            val_indices = np.where(val_mask)[0]
            """
            # Vérifications et logs
            print(f"\nFold {fold + 1}:")
            print(f"Sessions train: {len(train_sessions)} ({train_sessions[0]} à {train_sessions[-1]})")
            print(f"Sessions val: {len(val_sessions)} ({val_sessions[0]} à {val_sessions[-1]})")
            print(f"Indices train: {len(train_indices)}")
            print(f"Indices val: {len(val_indices)}")
            """
            yield train_indices, val_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


class nonAnchore_dWalkForwardCV_afterPrevTrain(BaseCrossValidator):
    """
    Validateur walk-forward 'non ancré' (nonAnchored) qui calcule
    automatiquement la taille de l'entrainement (train_size).

    - Nombre de splits (folds) : n_splits
    - Ratio pour la validation : val_ratio (<= 1)

    Caractéristiques :
    ----------------
    1. Le train du fold suivant démarre là où s'arrête le train du fold précédent
       (pas de recouvrement du train).
    2. La validation est immédiatement après le train.
    3. train_size est calculé à partir de la formule :
         floor(n / ((1 + val_ratio) * n_splits))
    4. val_size = round(train_size * val_ratio).
    5. Si on ne peut pas générer l'ensemble des n_splits demandés
       parce qu'on atteint la fin du dataset, on arrête.
    """

    def __init__(self, n_splits, val_ratio=0.5):
        super().__init__()
        assert 0 < val_ratio <= 1, "val_ratio doit être compris entre 0 et 1."

        self.n_splits = n_splits
        self.val_ratio = val_ratio

        # train_size sera calculé dans .split() en fonction de la taille de X

    def get_n_splits(self, X, y=None, groups=None):
        """
        On retourne simplement le nombre de splits souhaité,
        même si en pratique on risque d'en produire moins
        si le dataset est trop petit.
        """
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n = len(X)

        train_size = int(n / (self.n_splits + self.val_ratio))
        val_size = int(train_size * self.val_ratio)

        # Ajuster pour l'espace restant
        total_used = self.n_splits * (train_size + val_size)
        leftover = n - total_used
        if leftover > 0:
            last_train_extra = leftover

        start = 0
        for fold in range(self.n_splits):
            current_train_size = train_size
            """
            if fold == self.n_splits - 1:  # dernier fold
                current_train_size += last_train_extra
            """
            end_train = start + current_train_size
            end_val = end_train + val_size

            if end_val > n:  # Sécurité
                break

            train_indices = np.arange(start, end_train)
            val_indices = np.arange(end_train, end_val)
            yield train_indices, val_indices

            start += current_train_size


class nonAnchore_dWalkForwardCV_afterPrevVal(BaseCrossValidator):
    """
    Validation croisée Non-Anchored Walk-Forward avec ratio stable.

    - Le ratio r = val_size / train_size est maintenu constant à partir du second fold.
    - Le leftover est ajouté au premier train pour utiliser au mieux toutes les données.

    """

    def __init__(self, n_splits, r=1.0):
        self.n_splits = n_splits
        self.r = r

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        N = len(X)
        nb_split_tscv = self.n_splits
        r = self.r

        train_size = int(N // (nb_split_tscv * (1 + r)))
        val_size = int(train_size * r)
        required_size = nb_split_tscv * (train_size + val_size)
        leftover = N - required_size

        # Ajout du leftover au premier train
        train_size_first = train_size + leftover

        # Fold 1
        current_index = 0
        train_indices = np.arange(current_index, current_index + train_size_first)
        val_indices = np.arange(current_index + train_size_first,
                                current_index + train_size_first + val_size)

        yield train_indices, val_indices

        # Folds suivants

        # Comportement original : train commence après la validation précédente
        current_index = current_index + train_size_first + val_size

        # Folds suivants
        for i in range(1, nb_split_tscv):
            train_indices = np.arange(current_index, current_index + train_size)
            val_indices = np.arange(current_index + train_size,
                                    current_index + train_size + val_size)

            yield train_indices, val_indices

            current_index += train_size + val_size


import numpy as np
from math import floor

from sklearn.model_selection import BaseCrossValidator

class nonAnchoreRolling_dWalkForwardCV_afterPrevTrain(BaseCrossValidator):
    """
    Validateur temporel personnalisé «non ancré» (non anchored) à fenêtres
    glissantes (rolling) pour la validation croisée.

    Caractéristiques principales :
    -----------------------------
    1. Le nombre de splits (n_splits) est donné.
    2. La proportion de la fenêtre de validation est val_ratio (0 < val_ratio <= 1).
    3. Le recouvrement entre deux trains successifs est de train_overlap_ratio (r),
       0 <= r < 1.
    4. train_size est déterminé automatiquement afin de pouvoir générer au moins
       n_splits splits. On utilise la formule théorique :

         train_size <= n / [1 + val_ratio + (n_splits - 1)*(1 - r)]

       puis on ajuste au besoin en décrémentant train_size si nécessaire.
    5. val_size = round(train_size * val_ratio).
    6. La fenêtre de train_(i+1) commence après un 'step' = floor(train_size*(1-r))
       échantillons supplémentaires par rapport au début de train_i,
       assurant le recouvrement partiel.

    Paramètres
    ----------
    n_splits : int
        Nombre de splits (ou folds) souhaités.

    val_ratio : float
        Proportion de la taille du train (entre 0 et 1] utilisée comme fenêtre de validation.

    train_overlap_ratio : float
        Proportion de recouvrement entre deux fenêtres d'entraînement successives
        (entre 0 inclus et 1 exclus).

    Exemples
    --------
    >>> validator = nonAnchoreRolling_dWalkForwardCV_afterPrevTrain(
    ...     n_splits=5,
    ...     val_ratio=0.66,
    ...     train_overlap_ratio=0.5
    ... )
    >>> for train_idx, val_idx in validator.split(np.arange(6000)):
    ...     print(train_idx[0], train_idx[-1], val_idx[0], val_idx[-1])
    """

    def __init__(self, n_splits, val_ratio=0.5, train_overlap_ratio=0.5):
        super().__init__()
        assert 0 < val_ratio <= 1, "val_ratio doit être dans l'intervalle (0, 1]."
        assert 0 <= train_overlap_ratio < 1, "train_overlap_ratio doit être dans [0, 1)."
        assert n_splits >= 1, "n_splits doit être >= 1."

        self.n_splits = n_splits
        self.val_ratio = val_ratio
        self.train_overlap_ratio = train_overlap_ratio

        self.train_size_ = None  # sera calculé dans .split()

    def get_n_splits(self, X=None, y=None, groups=None):
        """
        Retourne le nombre (souhaité) de splits.
        (En pratique, on lèvera une exception si on ne peut pas
        générer exactement n_splits splits complets.)
        """
        return self.n_splits

    def _max_splits_with(self, n, train_size, val_ratio, overlap_ratio):
        """
        Calcule combien de splits complets on peut générer pour un
        train_size donné.
        """
        val_size = round(train_size * val_ratio)
        step = int(train_size * (1 - overlap_ratio))

        # Si step = 0, on ne pourra pas avancer la fenêtre
        if step < 1:
            return 0

        # On applique la formule : floor((n - T - V) / step) + 1
        total_needed_per_fold = train_size + val_size
        leftover = n - total_needed_per_fold
        if leftover < 0:
            return 0
        return (leftover // step) + 1

    def split(self, X, y=None, groups=None):
        """
        Génère les indices (train, val) pour chaque split.
        """
        n = len(X)
        r = self.train_overlap_ratio

        # 1) Estimation analytique de train_size
        #    T <= n / [1 + val_ratio + (n_splits - 1)*(1 - r)]
        denom = (1 + self.val_ratio + (self.n_splits - 1) * (1 - r))
        if denom <= 0:
            raise ValueError("Paramètres invalides, la formule de train_size est non définie.")

        T_est = int(n // denom)  # division entière "basique"

        # 2) Ajustement par décrémentation si besoin jusqu'à satisfaire n_splits
        train_size = T_est
        while train_size > 0:
            possible = self._max_splits_with(n, train_size, self.val_ratio, r)
            if possible >= self.n_splits:
                break
            train_size -= 1

        if train_size <= 0:
            raise ValueError(
                f"Impossible de générer {self.n_splits} splits. "
                f"Essayez de réduire n_splits ou val_ratio."
            )

        self.train_size_ = train_size
        val_size = round(train_size * self.val_ratio)
        step = int(train_size * (1 - r))

        start = 0
        for i in range(self.n_splits):
            end_train = start + train_size
            end_val = end_train + val_size

            # On vérifie que la fenêtre de validation ne dépasse pas n
            if end_val > n:
                # On préfère "casser" la boucle si plus de place pour ce split
                break

            # Indices du train et validation
            train_indices = np.arange(start, end_train)
            val_indices = np.arange(end_train, end_val)
            yield train_indices, val_indices

            # Décalage pour le split suivant
            start += step


def convert_metrics_to_numpy(metrics_dict):
    """
    Convertit de manière sûre les métriques GPU en arrays NumPy

    Args:
        metrics_dict: Dictionnaire contenant les métriques CuPy
    Returns:
        Dict avec les mêmes métriques en NumPy
    """
    numpy_metrics = {}
    for key, value in metrics_dict.items():
        if isinstance(value, cp.ndarray):
            numpy_metrics[key] = cp.asnumpy(value)
        else:
            numpy_metrics[key] = value
    return numpy_metrics


def calculate_time_difference(start_date_str, end_date_str):
    date_format = "%Y-%m-%d %H:%M:%S"
    start_date = datetime.strptime(start_date_str, date_format)
    end_date = datetime.strptime(end_date_str, date_format)
    diff = relativedelta(end_date, start_date)
    return diff


def setup_cv_method(df_init_features=None, X_train=None, y_train_label=None, cv_method=None, nb_split_tscv=None,
                    config=None):
    # D'abord, vérifier si la colonne existe dans df_init_features
    if 'timeStampOpening' not in df_init_features.columns:
        raise ValueError("La colonne 'timeStampOpening' est absente de df_init_features.")

    """Configure la méthode de validation croisée"""
    if cv_method == cv_config.TIME_SERIE_SPLIT:
        return TimeSeriesSplit(n_splits=nb_split_tscv)
    elif cv_method == cv_config.TIME_SERIE_SPLIT_NON_ANCHORED_AFTER_PREVVAL:
        val_ratio = config.get('non_acnhored_val_ratio', 1)

        cv = nonAnchore_dWalkForwardCV_afterPrevVal(n_splits=nb_split_tscv, r=val_ratio)
        return cv
    elif cv_method == cv_config.TIME_SERIE_SPLIT_NON_ANCHORED_ROLLING:
        val_ratio = config.get('non_acnhored_val_ratio', 1)
        train_overlap_ratio = config.get('non_acnhoredRolling_train_overlap_ratio', 1)

        cv = nonAnchoreRolling_dWalkForwardCV_afterPrevTrain(
            n_splits=nb_split_tscv,
            val_ratio=val_ratio,
            train_overlap_ratio=train_overlap_ratio
        )
        return cv
    elif cv_method == cv_config.TIMESERIES_SPLIT_BY_ID:
        # Ensuite, appliquer la logique en fonction de cv_method
        # On garde une trace des colonnes originales
        X_train_ = X_train.copy()

        original_columns = set(X_train_.columns)

        # Ajout des colonnes nécessaires si elles n'existent pas déjà
        if 'timeStampOpening' not in X_train_.columns:
            X_train_['timeStampOpening'] = df_init_features.loc[X_train_.index, 'timeStampOpening']

        if 'session_type_index' not in X_train_.columns:
            X_train_['session_type_index'] = df_init_features.loc[X_train_.index, 'session_type_index']

        # Vérification des valeurs
        columns_to_check = ['timeStampOpening']
        comparison_results = X_train_[columns_to_check].eq(df_init_features.loc[X_train_.index, columns_to_check])
        discrepancies = comparison_results[~comparison_results.all(axis=1)]

        if not discrepancies.empty:
            print("\nDétails des divergences :")
            print(discrepancies)
            raise ValueError("divergence df_init_features X_train_")

        # Création d'une colonne temporaire session_type_index_utc
        X_train_['session_type_index_utc'] = timestamp_to_date_utc(X_train_['timeStampOpening']).str[:10] + '_' + \
                                             X_train_[
                                                 'session_type_index'].astype(str)

        session_ids = X_train_['session_type_index_utc'].values

        # Suppression de la colonne temporaire

        cv = CustomSessionTimeSeriesSplit_byID(session_ids=session_ids, n_splits=nb_split_tscv)

        return cv
    elif cv_method == cv_config.K_FOLD:
        return KFold(n_splits=nb_split_tscv, shuffle=False)
    elif cv_method == cv_config.K_FOLD_SHUFFLE:
        return KFold(n_splits=nb_split_tscv, shuffle=True, random_state=42)
    else:
        raise ValueError(f"Unknown cv_method: {cv_method}")


def check_variable(value, name):
    """
    Vérifie si une variable contient des valeurs invalides.

    Args:
        value: La valeur à vérifier (peut être une liste ou une valeur scalaire).
        name: Le nom de la variable (pour les messages d'erreur).

    Raises:
        ValueError: Si la variable contient des valeurs invalides.
    """
    if value is None:
        raise ValueError(f"La variable '{name}' est None et doit être définie.")
    if isinstance(value, (list, tuple)):
        if any(v is None or v == float('inf') or v == float('-inf') for v in value):
            raise ValueError(f"La variable '{name}' contient des valeurs invalides: {value}")
    else:
        if value == float('inf') or value == float('-inf'):
            raise ValueError(f"La variable '{name}' est infinie: {value}")
        if value is None:
            raise ValueError(f"La variable '{name}' est None et doit être définie.")


def calculate_constraints_optuna(trial=None, config=None, debug=False):
    """
    Calcule les contraintes pour un modèle d'optimisation en utilisant trial et config.
    Si un seuil est 'inf', on n'ajoute pas la contrainte dans la liste (on ne la vérifie pas).
    """
    use_cuda = (config is not None and config.get('device_') == 'cuda')
    if use_cuda:
        import cupy as cp
        xp = cp
    else:
        import numpy as np
        xp = np

    # Récup. des attributs dans le trial
    perctDiff_winrateRatio_train_val = trial.user_attrs.get('perctDiff_winrateRatio_train_val', float('inf'))
    nb_trades_val_by_fold_list = trial.user_attrs.get('nb_trades_val_by_fold', [float('inf')])
    winrates_val_by_fold = trial.user_attrs.get('winrates_val_by_fold', None)
    val_trades_samples_perct = trial.user_attrs.get('val_trades_samples_perct', None)


    # Vérif
    def validate_variable(var, name):
        if var is None or (isinstance(var, (list, tuple)) and any(v is None or v == float('inf') for v in var)):
            raise ValueError(f"La variable '{name}' contient des valeurs invalides: {var}")

    validate_variable(perctDiff_winrateRatio_train_val, "perctDiff_winrateRatio_train_val")
    validate_variable(nb_trades_val_by_fold_list, "nb_trades_val_by_fold_list")
    validate_variable(winrates_val_by_fold, "winrates_val_by_fold")
    validate_variable(val_trades_samples_perct, "val_trades_samples_perct")


    # Récupération des seuils
    cfg_ratio = config.get('config_constraint_ratioWinrate_train_val', float('inf'))
    cfg_min_trades = config.get('config_constraint_min_trades_threshold_by_Fold', float('inf'))
    cfg_min_winrates = config.get('config_constraint_winrates_val_by_fold', float('inf'))
    cfg_max_std_trades = config.get('config_constraint_max_std_trades', float('inf'))
    cfg_min_trades_samples_perct = config.get('config_constraint_min_trades_samples_perct', float('inf'))


    # Fonctions utilitaires
    def to_float(x):
        if hasattr(x, 'get'):  # cupy
            return float(x.get().item())
        if hasattr(x, 'item'):  # numpy
            return float(x.item())
        return float(x)

    def safe_max(array_like):
        return max(array_like) if array_like else float('inf')

    def safe_min(array_like):
        return min(array_like) if array_like else float('inf')

    constraints = []

    # --- 1) Contrainte de ratio Winrate train/val ---
    # Si le seuil n'est pas inf, on l'applique
    if cfg_ratio != float('inf'):
        val_ = safe_max(perctDiff_winrateRatio_train_val)
        c = xp.maximum(0, val_ - cfg_ratio)
        constraints.append(to_float(c))

    # --- 2) Contrainte min trades par fold ---
    if cfg_min_trades != float('inf'):
        min_trades = safe_min(nb_trades_val_by_fold_list)
        c = xp.maximum(0, cfg_min_trades - min_trades)
        constraints.append(to_float(c))

    # --- 3) Contrainte min winrate par fold ---
    if cfg_min_winrates != float('inf'):
        min_winrate = safe_min(winrates_val_by_fold)
        c = xp.maximum(0, cfg_min_winrates - min_winrate)
        constraints.append(to_float(c))

    # --- 4) Contrainte sur std trades ---
    if cfg_max_std_trades != float('inf'):
        arr_trades = xp.array(nb_trades_val_by_fold_list)
        std_trades = xp.std(arr_trades)
        c = xp.maximum(0, std_trades - cfg_max_std_trades)
        constraints.append(to_float(c))

    # --- 5) Contrainte surtrades_samples_perct  ---
    if cfg_min_trades_samples_perct != float('inf'):
        min_trades_samples_perct = safe_min(val_trades_samples_perct)
        c = xp.maximum(0, cfg_min_trades_samples_perct - min_trades_samples_perct)
        constraints.append(to_float(c))



    if debug:
        print("\n--- DEBUG: Contraintes Calculées ---")
        print(f"Liste finale des contraintes (longueur {len(constraints)}) : {constraints}")
        print("------------------------------------\n")

    return constraints


def analyze_thresholds(
        y_test,
        y_pred_proba_afterSig,
        other_params,
        thresholds=np.arange(0.0, 1.01, 0.001),
        min_winrate=55,
        min_trades=50
):
    """
    Analyze threshold impact on metrics with support for pandas Series and cupy arrays.

    Args:
        y_test: True labels (0 or 1) - can be numpy array, pandas Series, or cupy array
        y_pred_proba_afterSig: Predicted probabilities - can be numpy array, pandas Series, or cupy array
        other_params (dict): Parameters for profit calculation
        thresholds (np.ndarray): Thresholds to analyze (default: 0.0 to 1.0, step 0.01)
        min_winrate (float): Minimum required win rate (%)
        min_trades (int): Minimum required number of trades
    """
    # Convert inputs to numpy arrays if they're pandas Series or cupy arrays
    if hasattr(y_test, 'values'):
        y_test = y_test.values
    if hasattr(y_pred_proba_afterSig, 'values'):
        y_pred_proba_afterSig = y_pred_proba_afterSig.values

    # Convert cupy arrays to numpy if needed
    if hasattr(y_test, 'get'):
        y_test = y_test.get()
    if hasattr(y_pred_proba_afterSig, 'get'):
        y_pred_proba_afterSig = y_pred_proba_afterSig.get()

    # Initialize metric lists
    f1_scores = []
    profits = []
    winrates = []
    trade_counts = []

    # Calculate metrics for each threshold
    for threshold in thresholds:
        # Convert probabilities to binary predictions
        y_pred_threshold = (y_pred_proba_afterSig > threshold).astype(int)

        # Calculate metrics using numpy operations
        true_positive_mask = (y_test == 1) & (y_pred_threshold == 1)
        false_positive_mask = (y_test == 0) & (y_pred_threshold == 1)

        tp = np.sum(true_positive_mask)
        fp = np.sum(false_positive_mask)
        trades = tp + fp

        # Calculate winrate
        winrate = (tp / trades * 100) if trades > 0 else 0

        # Calculate F1-score
        f1 = f1_score(y_test, y_pred_threshold)

        # Calculate profit
        profit, tp_profit, fp_profit = xgb_calculate_profitBased_cpu(y_test, y_pred_threshold, other_params)

        # Store results
        f1_scores.append(f1)
        profits.append(profit)
        winrates.append(winrate)
        trade_counts.append(trades)

    # Find valid thresholds
    valid_thresholds = [
        (threshold, profit, winrate, trades)
        for threshold, profit, winrate, trades in zip(thresholds, profits, winrates, trade_counts)
        # if winrate >= min_winrate and trades >= min_trades
    ]

    # Display results
    print("\nThresholds meeting constraints:")
    for threshold, profit, winrate, trades in valid_thresholds:
        print(f"Threshold: {threshold:.3f}, Profit: {profit:.3f}, "
              f"Win Rate: {winrate:.3f}%, Trades: {trades}")

    # Visualization


""""
    plt.figure(figsize=(12, 6))

    plt.plot(thresholds, f1_scores, label='F1-score', color='blue')
    plt.plot(thresholds, profits, label='Profit', color='green')
    plt.plot(thresholds, winrates, label='Win Rate (%)', linestyle='--', color='orange')
    plt.plot(thresholds, trade_counts, label='Trade Count', linestyle=':', color='purple')

    # Mark valid thresholds
    for threshold, profit, winrate, trades in valid_thresholds:
        plt.scatter(threshold, profit, color='red',
                    label='Valid Threshold' if threshold == valid_thresholds[0][0] else "")

    # Add optimal threshold line
    if valid_thresholds:
        plt.axvline(x=valid_thresholds[0][0], color='red', linestyle='--',
                    label=f'Optimal Threshold: {valid_thresholds[0][0]:.2f}')

    plt.xlabel('Threshold')
    plt.ylabel('Values')
    plt.legend()
    plt.title('Threshold Impact on F1-score, Profit, Win Rate and Trade Count')
    plt.grid(True)
    plt.show()
"""


def remove_nan_inf(X=None, y=None, df_pnl_data=None, dataset_name=""):
    """
    Supprime les lignes contenant des NaN ou Inf dans X et les lignes correspondantes dans y et df_pnl_data.

    Args:
        X: DataFrame des features
        y: Series des labels
        df_pnl_data: Series des données PnL (optionnel)
        dataset_name: Nom du dataset pour l'affichage des statistiques

    Returns:
        tuple: (X_clean, y_clean, df_pnl_data_clean, mask)
    """
    initial_count = len(X)
    mask_noInfNan = ~X.replace([np.inf, -np.inf], np.nan).isna().any(axis=1)
    removed_indices = X[~mask_noInfNan].index

    X_clean_noInfNan = X[mask_noInfNan]
    y_clean_noInfNan = y[mask_noInfNan]

    # Nettoyage du df_pnl_data si fourni
    df_pnl_data_clean = None
    if df_pnl_data is not None:
        df_pnl_data_clean_noInfNan = df_pnl_data[mask_noInfNan]

    lines_removed = len(removed_indices)
    percentage_removed = (lines_removed / initial_count) * 100

    print(f"\nStatistiques {dataset_name}:")
    print(f"Nombre initial de trades : {initial_count}")
    print(f"Nombre de trades supprimés : {lines_removed}")
    print(f"Pourcentage de trades supprimés : {percentage_removed:.2f}%")

    if df_pnl_data is not None:
        return X_clean_noInfNan, y_clean_noInfNan, df_pnl_data_clean_noInfNan, mask_noInfNan
    else:
        return X_clean_noInfNan, y_clean_noInfNan, mask_noInfNan


def process_cv_results(cv_results, config, ENV=None, study=None):
    """
    Gère la conversion CPU/GPU des résultats, le nettoyage mémoire
    et l'arrêt éventuel de l'optimisation selon l'environnement.

    Args:
        cv_results (dict): Résultats de cross-validation.
        config (dict): Configuration contenant le type de device ('cpu' ou 'cuda').
        ENV (str): Environnement d'exécution ('pycharm', 'colab', etc.).
        study (optuna.Study, optional): Étude Optuna pour stopper l'optimisation si nécessaire.

    Returns:
        dict: Un dictionnaire avec les clés suivantes :
            - winrates_val_by_fold, nb_trades_val_by_fold, scores_train_by_fold,
              tp_train_by_fold, fp_train_by_fold, tp_val_by_fold, fp_val_by_fold,
              scores_val_by_fold (convertis en NumPy arrays)
            - fold_stats (stats par fold, converties)
            - metrics (totaux 'total_tp_val', 'total_fp_val', etc., convertis en float)
    """
    import platform, os
    import numpy as np

    # --- 1) Gérer l'arrêt de l'optimisation selon l'environnement ---
    if ENV == 'pycharm':
        system = platform.system()
        if system != "Darwin":  # Windows/Linux
            try:
                import keyboard
                if keyboard.is_pressed('²'):
                    if study is not None:
                        study.stop()
            except Exception as e:
                print(f"[process_cv_results] Erreur de gestion clavier : {e}")

    elif ENV == 'colab':
        if os.path.exists('stop_optimization.txt'):
            if study is not None:
                study.stop()

    # --- 2) Choisir la bibliothèque adéquate (NumPy ou CuPy) ---
    if config['device_'] == 'cuda':
        import cupy as cp
        xp = cp
    else:
        xp = np

    # --- 3) Fonction interne de conversion d'un array GPU -> CPU ou d'une liste -> np.array ---
    def to_numpy_if_needed(array_like):
        # Si GPU (CuPy), on convertit en NumPy
        if config['device_'] == 'cuda' and isinstance(array_like, xp.ndarray):
            return xp.asnumpy(array_like)

        # Si c'est une liste Python avec des sous-éléments, convertir récursivement
        elif isinstance(array_like, list):
            # Convertir chaque élément en tableau NumPy s'il ne l'est pas déjà
            return [to_numpy_if_needed(item) for item in array_like]

        # Si c'est un tableau NumPy ou une liste simple, tenter de convertir en tableau NumPy
        elif isinstance(array_like, (np.ndarray, list)):
            try:
                return np.array(array_like, dtype=object)  # Préserver les structures hétérogènes
            except ValueError:
                # Si les dimensions ne sont pas compatibles, retourner chaque élément séparément
                return [np.array(item) if isinstance(item, (list, np.ndarray)) else item for item in array_like]

        # Si c'est un scalaire ou tout autre type
        return array_like

    # --- 4) Convertir les métriques par fold ---
    winrates_val_by_fold = to_numpy_if_needed(cv_results['winrates_val_by_fold'])
    nb_trades_val_by_fold = to_numpy_if_needed(cv_results['nb_trades_val_by_fold'])
    nb_samples_val_by_fold = to_numpy_if_needed(cv_results['nb_samples_val_by_fold'])
    tp_val_by_fold = to_numpy_if_needed(cv_results['tp_val_by_fold'])
    fp_val_by_fold = to_numpy_if_needed(cv_results['fp_val_by_fold'])
    val_pred_proba_log_odds = to_numpy_if_needed(cv_results['val_pred_proba_log_odds'])
    class0_raw_data_val_by_fold = to_numpy_if_needed(cv_results['class0_raw_data_val_by_fold'])
    class1_raw_data_val_by_fold = to_numpy_if_needed(cv_results['class1_raw_data_val_by_fold'])
    winrate_raw_data_val_by_fold = to_numpy_if_needed(cv_results['winrate_raw_data_val_by_fold'])
    val_trades_samples_perct = to_numpy_if_needed(cv_results['val_trades_samples_perct'])
    val_bestVal_custom_metric_pnl = to_numpy_if_needed(cv_results['val_bestVal_custom_metric_pnl'])

    winrates_train_by_fold = to_numpy_if_needed(cv_results['winrates_train_by_fold'])
    nb_trades_train_by_fold = to_numpy_if_needed(cv_results['nb_trades_train_by_fold'])
    nb_samples_train_by_fold = to_numpy_if_needed(cv_results['nb_samples_train_by_fold'])
    tp_train_by_fold = to_numpy_if_needed(cv_results['tp_train_by_fold'])
    fp_train_by_fold = to_numpy_if_needed(cv_results['fp_train_by_fold'])
    scores_train_by_fold = to_numpy_if_needed(cv_results['scores_train_by_fold'])
    train_pred_proba_log_odds = to_numpy_if_needed(cv_results['train_pred_proba_log_odds'])
    class0_raw_data_train_by_fold = to_numpy_if_needed(cv_results['class0_raw_data_train_by_fold'])
    class1_raw_data_train_by_fold = to_numpy_if_needed(cv_results['class1_raw_data_train_by_fold'])
    winrate_raw_data_train_by_fold = to_numpy_if_needed(cv_results['winrate_raw_data_train_by_fold'])
    train_trades_samples_perct = to_numpy_if_needed(cv_results['train_trades_samples_perct'])
    train_bestVal_custom_metric_pnl = to_numpy_if_needed(cv_results['train_bestVal_custom_metric_pnl'])

    perctDiff_winrateRatio_train_val = to_numpy_if_needed(cv_results['perctDiff_winrateRatio_train_val'])
    perctDiff_ratioTradeSample_train_val = to_numpy_if_needed(cv_results['perctDiff_ratioTradeSample_train_val'])

    perctDiff_winrateRatio_train_val = to_numpy_if_needed(cv_results['perctDiff_winrateRatio_train_val'])
    perctDiff_ratioTradeSample_train_val = to_numpy_if_needed(cv_results['perctDiff_ratioTradeSample_train_val'])

    brier_val = to_numpy_if_needed(cv_results['brier_val'])
    relative_brier_val = to_numpy_if_needed(cv_results['relative_brier_val'])
    best_thresh_fold = to_numpy_if_needed(cv_results['best_thresh_fold'])
    best_iteration_fold = to_numpy_if_needed(cv_results['best_iteration_fold'])

    # --- 5) Conversion des fold_stats ---
    # On convertit chaque champ Cupy -> float si nécessaire
    fold_stats = {}
    if config['device_'] == 'cuda':
        import cupy as cp
        for fold_num, stats in cv_results['fold_stats'].items():
            fold_stats[fold_num] = {}
            for key, value in stats.items():
                if isinstance(value, cp.ndarray):
                    fold_stats[fold_num][key] = [float(v) for v in
                                                 cp.asnumpy(value).flatten()]  # Conversion pour les tableaux CuPy
                elif isinstance(value, list):
                    fold_stats[fold_num][key] = [float(v) for v in value]  # Conversion pour les listes
                else:
                    fold_stats[fold_num][key] = float(value) if isinstance(value, (int, float)) else value
        # Nettoyage mémoire GPU
        cp.get_default_memory_pool().free_all_blocks()
    else:
        for fold_num, stats in cv_results['fold_stats'].items():
            fold_stats[fold_num] = {}
            for key, value in stats.items():
                if isinstance(value, np.ndarray):
                    fold_stats[fold_num][key] = [float(v) for v in
                                                 value.flatten()]  # Conversion pour les tableaux NumPy
                elif isinstance(value, list):
                    fold_stats[fold_num][key] = [float(v) for v in value]  # Conversion pour les listes
                else:
                    fold_stats[fold_num][key] = float(value) if isinstance(value, (int, float)) else value

    # --- 6) Conversion des métriques totales (validation & entraînement) ---
    #    Exemple : total_tp_val, total_fp_val, etc.
    metrics_converted = {}
    for k, v in cv_results['metrics'].items():
        metrics_converted[k] = float(v)  # On force en float

    return {
        'winrates_val_by_fold': winrates_val_by_fold,
        'nb_trades_val_by_fold': nb_trades_val_by_fold,
        'nb_samples_val_by_fold': nb_samples_val_by_fold,
        'tp_val_by_fold': tp_val_by_fold,
        'fp_val_by_fold': fp_val_by_fold,
        'val_pred_proba_log_odds': val_pred_proba_log_odds,
        'class0_raw_data_val_by_fold': class0_raw_data_val_by_fold,
        'class1_raw_data_val_by_fold': class1_raw_data_val_by_fold,
        'winrate_raw_data_val_by_fold': winrate_raw_data_val_by_fold,
        'val_trades_samples_perct': val_trades_samples_perct,
        'val_bestVal_custom_metric_pnl': val_bestVal_custom_metric_pnl,

        'winrates_train_by_fold': winrates_train_by_fold,
        'nb_trades_train_by_fold': nb_trades_train_by_fold,
        'nb_samples_train_by_fold': nb_samples_train_by_fold,
        'tp_train_by_fold': tp_train_by_fold,
        'fp_train_by_fold': fp_train_by_fold,
        'scores_train_by_fold': scores_train_by_fold,
        'train_pred_proba_log_odds': train_pred_proba_log_odds,
        'train_bestVal_custom_metric_pnl': train_bestVal_custom_metric_pnl,

        'best_thresh_fold': best_thresh_fold,
        'best_iteration_fold':best_iteration_fold,

        'class0_raw_data_train_by_fold': class0_raw_data_train_by_fold,
        'class1_raw_data_train_by_fold': class1_raw_data_train_by_fold,
        'winrate_raw_data_train_by_fold': winrate_raw_data_train_by_fold,
        'train_trades_samples_perct': train_trades_samples_perct,

        'brier_val':brier_val,
        'relative_brier_val': relative_brier_val,

        'perctDiff_winrateRatio_train_val': perctDiff_winrateRatio_train_val,
        'perctDiff_ratioTradeSample_train_val': perctDiff_ratioTradeSample_train_val,
        'fold_stats': fold_stats,
        'metrics': metrics_converted
    }


def reporting_model_performance(pred_proba_log_odds, tp, fp, fn, tn, config):
    """
    Fonction pour rapporter les performances d'un modèle.

    Args:
        pred_proba_log_odds (np.array): Log-odds prédites par le modèle.
        tp (int): Nombre de vrais positifs.
        fp (int): Nombre de faux positifs.
        fn (int): Nombre de faux négatifs.
        tn (int): Nombre de vrais négatifs.
        config (dict): Configuration contenant les paramètres de profit et de perte.

    Returns:
        None: Affiche les résultats directement.
    """
    # Calcul du PnL
    # pnl = tp * config['profit_per_tp'] + fp * config['loss_per_fp']

    print(f"tp:{tp} | tp:{fp}")
    # pnl=tp * config['profit_per_tp'] + fp * config['loss_per_fp']

    # Calcul des statistiques pour log-odds
    log_odds_min = np.min(pred_proba_log_odds)
    log_odds_max = np.max(pred_proba_log_odds)

    # Transformation en probabilités avec sigmoidCustom_cpu
    proba_min = sigmoidCustom_cpu(log_odds_min)
    proba_max = sigmoidCustom_cpu(log_odds_max)

    # Affichage des résultats
    print(f"log-odds => min: {log_odds_min:.2f}, max: {log_odds_max:.2f} | "
          f"proba => min: {proba_min:.2f}, max: {proba_max:.2f}")
    print("Prediction sur X_test sur modèle final suite au ré-entrainement du modèle de la CV:")
    total_samples = tp + fp + fn + tn
    total_trades = tp + fp
    print(f"   - Trades testés : {total_trades} trades pour {total_samples} échantillons")
    print(f"   - Confusion Matrix : TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
    # print(f"   - PnL sur X_test : {pnl:.2f}")
    trade_percentage = (total_trades / total_samples) * 100
    print(f"   - % (trades testés / échantillons testés) : {trade_percentage:.2f}%")

    # Fonction principale optimisée avec Numba
    from numba import njit


import numpy as np
from numba import njit
# If you want a typed dictionary in pure nopython mode:
from numba.typed import Dict
from numba import types


@njit
def remplace_0_nan_reg_slope_p_2d(session_starts, reg_slopes_2d, windows):
    n, num_windows = reg_slopes_2d.shape
    out = np.copy(reg_slopes_2d)

    # Indices de début de session
    session_indices = []
    for i in range(n):
        if session_starts[i]:
            session_indices.append(i)

    for col in range(num_windows):
        w = windows[col]
        for start_idx in session_indices:
            end_idx = min(start_idx + w + 1, n)
            # Parcours de la fenêtre, remplacer juste les 0
            for row in range(start_idx, end_idx):
                if out[row, col] == 0:
                    out[row, col] = np.nan

    return out


import numpy as np
import pandas as pd


def process_reg_slope_replacement(df, session_starts, windows_list, reg_feature_prefix="reg_slope_"):
    """
    - Extrait dynamiquement les colonnes `reg_slope_X` depuis df en utilisant un préfixe personnalisé.
    - Applique la fonction `remplace_0_nan_reg_slope_p_2d`.
    - Convertit les résultats en DataFrame avec les noms de colonnes d'origine.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame contenant les colonnes `reg_slope_{w}` et `sc_sessionStartEnd`.
    session_starts : np.ndarray
        Tableau booléen indiquant les débuts de session.
    windows_list : list
        Liste des tailles de fenêtre utilisées pour la suppression.
    reg_slope_prefix : str, optional (default = "reg_slope_")
        Préfixe utilisé pour récupérer les colonnes de régression.

    Returns:
    --------
    df_results : pd.DataFrame
        DataFrame contenant les valeurs mises à jour (remplacement des 0 par NaN).
    """
    # Avant d'utiliser les colonnes
    if "sc_reg_slope_30P" in df.columns and "sc_reg_slope_30P_2" not in df.columns:
        df = df.rename(columns={"sc_reg_slope_30P": "sc_reg_slope_30P_2"})
    # Extraction dynamique des colonnes basées sur le préfixe donné
    reg_slopes_2d = np.column_stack([
        df[f"{reg_feature_prefix}{w}P_2"].values for w in windows_list
    ]).astype(np.float64)

    # Appliquer la fonction pour remplacer les 0 par NaN
    results_2d = remplace_0_nan_reg_slope_p_2d(session_starts, reg_slopes_2d, windows_list)

    # Convertir en DataFrame avec les noms d'origine
    df_results = pd.DataFrame(results_2d, columns=[f"{reg_feature_prefix}{w}P_2" for w in windows_list])

    return df_results


import numpy as np
from numba import jit

import numpy as np




def check_lower_higher_bound_consistency(X_train, X_test, max_values_to_show=5):
    """
    Vérifie si les valeurs de X_test sont en dehors des bornes observées dans X_train.

    Args:
        X_train (pd.DataFrame): Les données d'entraînement.
        X_test (pd.DataFrame): Les données de test à vérifier.
        max_values_to_show (int): Nombre maximum de valeurs hors bornes à afficher.

    Returns:
        bool: True si un problème de cohérence des bornes est détecté, False sinon.

    Raises:
        ValueError: Si des valeurs dans X_test sont en dehors des bornes de X_train.
    """
    import pandas as pd
    import numpy as np

    # Convertir en DataFrames si nécessaire
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train)

    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test, columns=X_train.columns)

    is_coherence_ranges_problem = False

    # Calculer les bornes pour chaque feature dans X_train
    oob = {}
    for feature in X_train.columns:
        train_min = X_train[feature].min()
        train_max = X_train[feature].max()

        # Vérifier les valeurs en dessous du minimum
        below_min_mask = X_test[feature] < train_min
        below_min_count = below_min_mask.sum()

        # Vérifier les valeurs au-dessus du maximum
        above_max_mask = X_test[feature] > train_max
        above_max_count = above_max_mask.sum()

        # Si des valeurs sont hors bornes, ajouter les informations au dictionnaire
        if below_min_count > 0 or above_max_count > 0:
            oob[feature] = {
                'train_min': train_min,
                'train_max': train_max,
                'below_min_count': below_min_count,
                'below_min_values': X_test.loc[below_min_mask, feature].head(
                    max_values_to_show).tolist() if below_min_count > 0 else [],
                'above_max_count': above_max_count,
                'above_max_values': X_test.loc[above_max_mask, feature].head(
                    max_values_to_show).tolist() if above_max_count > 0 else []
            }

    # Générer les messages d'erreur et déterminer s'il y a un problème
    error_messages = []
    if oob:
        error_messages.append(
            "\nAvertissement : certaines features contiennent des valeurs en dehors des bornes observées dans X_train :")
        for feature, info in oob.items():
            feature_messages = [f"\nAnalyse de la feature '{feature}':"]
            feature_messages.append(f"- Intervalle observé dans X_train : [{info['train_min']}, {info['train_max']}]")

            if info['below_min_count'] > 0:
                feature_messages.append(
                    f"--> {info['below_min_count']} valeur(s) dans X_test sont inférieures au minimum de X_train.")
                feature_messages.append(f"    Valeurs en dessous du minimum : {info['below_min_values']}")
                is_coherence_ranges_problem = True
            else:
                feature_messages.append("--> Aucune valeur inférieure au minimum.")

            if info['above_max_count'] > 0:
                feature_messages.append(
                    f"--> {info['above_max_count']} valeur(s) dans X_test sont supérieures au maximum de X_train.")
                feature_messages.append(f"    Valeurs au-dessus du maximum : {info['above_max_values']}")
                is_coherence_ranges_problem = True
            else:
                feature_messages.append("--> Aucune valeur supérieure au maximum.")

            error_messages.extend(feature_messages)

    # Lever une exception si un problème est détecté
    # if is_coherence_ranges_problem:
    #   raise ValueError("\n".join(error_messages) + "\nUn problème de valeurs hors bornes ou de distribution détecté")

    return is_coherence_ranges_problem


def apply_data_feature_scaling(X_train, X_test, y_train_label, y_test_label,
                               mask_train=None, mask_test=None,
                               chosen_scaler=None, results_directory=None,
                               config=None):
    """
    Applique le scaling aux données et gère la réinsertion potentielle des valeurs NaN/Inf.

    Args:
        X_train (DataFrame): Données d'entraînement X
        X_test (DataFrame): Données de test X
        y_train_label (Series): Labels d'entraînement
        y_test_label (Series): Labels de test
        mask_train (ndarray, optional): Masque indiquant les valeurs valides dans X_train
        mask_test (ndarray, optional): Masque indiquant les valeurs valides dans X_test
        chosen_scaler (int): Type de scaler à utiliser
        results_directory (str): Répertoire pour sauvegarder le scaler
        config (dict): Configuration additionnelle

    Returns:
        tuple: (X_train, X_test, y_train_label, y_test_label, scaler)
    """
    import os
    import numpy as np

    if config is None:
        config = {}

    # print(f"\n-- Scaler {chosen_scaler} actif ---\n")

    # Sauvegarde des données originales pour réinsertion potentielle
    X_train_original = X_train.copy()
    X_test_original = X_test.copy()
    y_train_label_original = y_train_label.copy()
    y_test_label_original = y_test_label.copy()

    save_scaler_dir = os.path.join(results_directory, 'optuna_results')

    # Vérification de la cohérence des distributions
    if 'check_distribution_coherence' in globals():
        diff_features = check_distribution_coherence(X_train, X_test)
        if diff_features:
            print("Avertissement : certaines features ont des distributions très différentes entre X_train et X_test :")
            for f, stats in diff_features.items():
                print(f"Feature: {f}, KS-stat: {stats['statistic']:.3f}, p-value: {stats['p_value']:.3e}")

    # Application du scaling
    X_train_scaled, X_test_scaled, scaler,scaler_params_ret = apply_scaling(
        X_train,
        X_test,
        save_path=save_scaler_dir,
        chosen_scaler=chosen_scaler
    )

    # Réinsertion des valeurs NaN et Inf si demandé
    reinsert_nan_inf_afterScaling = config.get('reinsert_nan_inf_afterScaling', False)

    if reinsert_nan_inf_afterScaling and mask_train is not None and mask_test is not None:
        X_train_result = X_train_original.copy()
        X_test_result = X_test_original.copy()
        y_train_label_result = y_train_label_original
        y_test_label_result = y_test_label_original

        # Mise à jour uniquement des valeurs valides avec les données scalées
        X_train_result[mask_train] = X_train_scaled
        X_test_result[mask_test] = X_test_scaled

        print("\nRéinsertion des valeurs NaN et Inf effectuée")
        print(f"Train : {(~mask_train).sum()} lignes réinsérées")
        print(f"Test : {(~mask_test).sum()} lignes réinsérées")
    else:
        X_train_result = X_train_scaled
        X_test_result = X_test_scaled
        y_train_label_result = y_train_label
        y_test_label_result = y_test_label

    # Vérification des tailles après scaling
    if len(X_train_result) != len(y_train_label_result):
        raise ValueError(f"Mismatch des tailles après scaling: "
                         f"X_train ({len(X_train_result)}) et y_train_label ({len(y_train_label_result)})")
    if len(X_test_result) != len(y_test_label_result):
        raise ValueError(f"Mismatch des tailles après scaling: "
                         f"X_test ({len(X_test_result)}) et y_test_label ({len(y_test_label_result)})")

    return X_train_result, X_test_result, y_train_label_result, y_test_label_result, scaler


@njit
def _calculate_atr_numba(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        session_starts: np.ndarray,
        period: int,
        avg_type_code: int,
        fill_value: float
) -> np.ndarray:
    """
    Version compilée Numba pour calculer l'ATR par session, sans chevauchement.

    Paramètres
    ----------
    high, low, close : np.ndarray (1D, float)
        Tableaux des prix high, low, close.
    session_starts : np.ndarray (1D, bool ou int)
        Indique True/1 si c'est la première barre d'une session, False/0 sinon.
    period : int
        Période de calcul de l’ATR.
    avg_type_code : int
        0 = 'sma'
        1 = 'ema'
        2 = 'wma'
    fill_value : float
        Valeur pour les barres < period dans une session.

    Retour
    ------
    atr_array : np.ndarray (1D, float)
        Tableau ATR, même taille que high/low/close.
    """
    n = len(high)

    # Préparer un array résultat, initialisé à fill_value
    atr_array = np.full(n, fill_value, dtype=np.float64)

    # Calcul du TR vectorisé
    # On a besoin de close décalé (close[-1]) => on peut le faire "à la main"
    close_shifted = np.empty(n, dtype=np.float64)
    close_shifted[0] = close[0]
    for i in range(1, n):
        close_shifted[i] = close[i - 1]

    tr_values = np.empty(n, dtype=np.float64)
    for i in range(n):
        # high-low
        hl = high[i] - low[i]
        # |high - close(-1)|
        hc = abs(high[i] - close_shifted[i])
        # |low - close(-1)|
        lc = abs(low[i] - close_shifted[i])
        tr_values[i] = max(hl, hc, lc)

    # Variables pour SMA
    window_sum = 0.0
    # On stocke la dernière "period" TR dans un buffer (pour SMA/WMA)
    # Cela évite l'overhead d'une deque ou list en Python
    buffer_tr = np.zeros(period, dtype=np.float64)
    buf_idx = 0  # Position courante dans le buffer
    count_in_window = 0

    # Variables pour EMA
    ema_prev = 0.0
    alpha = 2.0 / (period + 1.0)

    # Variables pour WMA (on stocke TR dans le même buffer,
    # mais le calcul diffère car on a besoin de weights = 1..period)
    weights = np.zeros(period, dtype=np.float64)
    for w in range(period):
        weights[w] = w + 1

    # Indice où commence la session en cours
    session_start = 0

    for i in range(n):
        # Si on démarre la toute première barre OU on est sur un début de session
        if i == 0 or session_starts[i]:
            # Réinitialisation
            window_sum = 0.0
            buf_idx = 0
            count_in_window = 0
            ema_prev = 0.0
            session_start = i

            # La première barre d'une session => on force fill_value
            atr_array[i] = fill_value
            continue

        # Valeur TR courante
        tr_val = tr_values[i]
        # Nombre de barres déjà vues dans la session
        bars_in_session = i - session_start + 1

        if avg_type_code == 0:
            # --- SMA ---
            # Retirer l'ancienne valeur si on a déjà "period" dans le buffer
            if count_in_window == period:
                old_val = buffer_tr[buf_idx]
                window_sum -= old_val

            # Ajouter la nouvelle
            buffer_tr[buf_idx] = tr_val
            window_sum += tr_val

            # Avancer buf_idx
            buf_idx = (buf_idx + 1) % period

            if count_in_window < period:
                count_in_window += 1

            # On ne calcule vraiment la SMA que si on a "period" barres
            if count_in_window < period:
                atr_array[i] = fill_value
            else:
                atr_array[i] = window_sum / period

        elif avg_type_code == 1:
            # --- EMA ---
            if bars_in_session == 2:
                # Sur la 2ème barre de la session, on initialise
                ema_prev = tr_val

            # Si on n'a pas encore "period" barres, on sort fill_value
            if bars_in_session < period:
                atr_array[i] = fill_value
            else:
                ema_prev = alpha * tr_val + (1 - alpha) * ema_prev
                atr_array[i] = ema_prev

        else:
            # --- WMA ---
            # Retirer l'ancienne valeur si on a déjà "period" dans le buffer
            if count_in_window == period:
                # On ne maintient pas un "sum" car c'est plus complexe
                # pour WMA. On recalcule à chaque fois.
                pass

            # Ajouter la nouvelle
            buffer_tr[buf_idx] = tr_val
            buf_idx = (buf_idx + 1) % period

            if count_in_window < period:
                count_in_window += 1

            if count_in_window < period:
                atr_array[i] = fill_value
            else:
                # Calcul de la WMA : sum(weights * data_window) / sum(weights)
                wma_sum = 0.0
                # On va parcourir la fenêtre "buffer_tr" en ordre chronologique
                # Le plus récent est en buf_idx - 1
                # Mais la WMA n'a pas besoin d'une orientation temps -> on associe
                # weights[i] = i+1 (ou i=0 => w=1)
                # On doit reconstituer la fenêtre dans l'ordre.
                b_index = buf_idx
                for w in range(period):
                    # Avancer en boucle circulaire
                    b_index = (b_index - 1) % period
                    wma_sum += weights[period - 1 - w] * buffer_tr[b_index]

                wma_val = wma_sum / weights.sum()
                atr_array[i] = wma_val

    return atr_array


def calculate_atr(df, period=6, avg_type='sma', fill_value=0.0):
    """
    Calcule l'ATR (Average True Range) par session en utilisant Numba
    pour plus de rapidité. Réinitialise à chaque début de session.
    Les premières barres (< period) d'une session sont à fill_value.

    Paramètres
    ----------
    df : pd.DataFrame
        - Colonnes attendues :
            'high', 'low', 'close', 'sc_sessionStartEnd'
    period : int
        Fenêtre de calcul de l’ATR (défaut 6).
    avg_type : {'sma', 'ema', 'wma'}
        Type de moyenne pour lisser l'ATR.
    fill_value : float
        Valeur à utiliser quand on n'a pas assez de barres
        (typ. début de session).

    Retour
    ------
    pd.Series
        Série ATR, alignée sur l'index de df.
    """

    # Copie locale pour éviter d'écraser l'index original
    df_local = df.reset_index(drop=True)

    # Repérage des débuts de session (True/False)
    # Ici, on suppose: sc_sessionStartEnd == 10 => début de session
    session_starts = (df_local['sc_sessionStartEnd'].values == 10)

    # Convertir avg_type en code
    if avg_type == 'sma':
        avg_type_code = 0
    elif avg_type == 'ema':
        avg_type_code = 1
    elif avg_type == 'wma':
        avg_type_code = 2
    else:
        raise ValueError("avg_type doit être 'sma', 'ema' ou 'wma'.")

    # Appel de la fonction Numba
    atr_array = _calculate_atr_numba(
        df_local['sc_high'].values.astype(np.float64),
        df_local['sc_low'].values.astype(np.float64),
        df_local['sc_close'].values.astype(np.float64),
        session_starts.astype(np.bool_),
        period,
        avg_type_code,
        float(fill_value)
    )

    # Reconstruire une série sur l'index original
    atr_series = pd.Series(atr_array, index=df.index, name='atr')
    return atr_series




def calculate_imbalance(df, imbalance_type, position, location, max_denominator=1, equal_value=0):
    """
    Calcule l'imbalance en fonction du type, de la position et de l'emplacement.

    Parameters
    ----------
    df : DataFrame
        DataFrame contenant les données de volume bid et ask
    imbalance_type : str
        Type d'imbalance, soit 'bull' soit 'bear'
    position : int
        Position de l'imbalance (0, 1, 2, 3, etc.)
    location : str
        Emplacement de l'imbalance, soit 'high' soit 'low'
    max_denominator : float
        Valeur maximale du dénominateur pour filtrage (défaut 1)
        Si 0 <= dénominateur < max_denominator, retourne 0
    equal_value : float
        Valeur à retourner si le dénominateur est égal à max_denominator (défaut 0)
        Utile pour les cas où (dénominateur >= 1) & (dénominateur <= 1)

    Returns
    -------
    Series
        Série contenant les valeurs d'imbalance calculées
    """
    # Déterminer les colonnes à utiliser selon le type et l'emplacement
    if imbalance_type == 'bull':
        # Bull imbalance: askVol / bidVol (acheteurs/vendeurs)
        if location == 'high':
            # Pour bull_imbalance_high_x (exemple: bull_imbalance_high_2 = askVolHigh_2 / bidVolHigh_3)
            numerator_col = f'askVolHigh_{position}'
            denominator_col = f'bidVolHigh_{position + 1}'
        else:  # location == 'low'
            # Pour bull_imbalance_low_x (exemple: bull_imbalance_low_2 = askVolLow_2 / bidVolLow_1)
            numerator_col = f'askVolLow_{position}'
            denominator_col = f'bidVolLow_{position - 1}'
    else:  # imbalance_type == 'bear'
        # Bear imbalance: bidVol / askVol (vendeurs/acheteurs)
        if location == 'high':
            # Pour bear_imbalance_high_x (exemple: bear_imbalance_high_3 = bidVolHigh_3 / askVolHigh_2)
            numerator_col = f'bidVolHigh_{position}'
            denominator_col = f'askVolHigh_{position - 1}'
        else:  # location == 'low'
            # Pour bear_imbalance_low_x (exemple: bear_imbalance_low_2 = bidVolLow_2 / askVolLow_3)
            numerator_col = f'bidVolLow_{position}'
            denominator_col = f'askVolLow_{position + 1}'

    # Calculer l'imbalance avec gestion du cas où le dénominateur est trop petit
    result = np.where(
        (df[denominator_col] >= 0) & (df[denominator_col] < max_denominator),
        0,  # Si 0 <= dénominateur < max_denominator, retourne 0
        df[numerator_col] / df[denominator_col]  # Sinon, calcule le ratio
    )

    return result

import numpy as np
import pandas as pd
from numba import jit, prange

def setup_model_weight_optuna():
    return None


def calculate_atr_thresholds(datasets, percentiles=[0.25, 0.50, 0.75]):
    """Calcule les seuils ATR basés sur les percentiles des datasets combinés"""
    all_atr = np.concatenate([d['atr_recalc'].values for d in datasets])
    thresholds = [round(np.percentile(all_atr, p*100), 1) for p in percentiles]
    return thresholds

# ───────────────────── MASK BUILDERS ────────────────────────────




def diviser_fichier_par_sessions(df, directory_path, file_name, use_default_params=False, encoding_used='utf-8'):
    """
    Divise un fichier en plusieurs parties selon les sessions spécifiées par l'utilisateur.

    Args:
        df: DataFrame pandas contenant les données
        directory_path: Chemin du dossier où sauvegarder les fichiers
        file_name: Nom du fichier original
        use_default_params: Si True, utilise les paramètres par défaut sans demander à l'utilisateur
        encoding_used: Encodage utilisé pour lire le fichier original
    """
    print_notification("Début de l'analyse et de la division du fichier par sessions")

    # Vérifier et convertir les types de données problématiques
    print_notification("Vérification et correction des types de données...")

    # Afficher les premières lignes pour comprendre la structure
    if not use_default_params:
        print("Aperçu des 5 premières lignes :")
        for col in df.columns:
            print(f"Colonne '{col}' - Premiers éléments: {df[col].head().tolist()}")

    # Conversion des colonnes si nécessaire
    try:
        # Essayer de convertir les colonnes numériques
        if 'sc_sessionStartEnd' in df.columns:
            df['sc_sessionStartEnd'] = pd.to_numeric(df['sc_sessionStartEnd'], errors='coerce')
            print_notification(f"Colonne 'sc_sessionStartEnd' convertie en numérique")

        # Forcer les conversions des colonnes qui posent problème (0, 4, 5)
        # Noter les indices de colonnes selon le message d'erreur
        problematic_columns = [df.columns[0]]
        if len(df.columns) > 4:
            problematic_columns.append(df.columns[4])
        if len(df.columns) > 5:
            problematic_columns.append(df.columns[5])

        for col in problematic_columns:
            # Essayer de déterminer le type approprié pour chaque colonne
            if df[col].str.contains('\.').any():  # Contient des points décimaux
                df[col] = pd.to_numeric(df[col], errors='ignore')
                print_notification(f"Colonne '{col}' traitée comme numérique")
            else:
                try:
                    df[col] = pd.to_numeric(df[col], errors='ignore')
                    print_notification(f"Colonne '{col}' traitée comme numérique")
                except:
                    print_notification(f"Colonne '{col}' conservée comme chaîne de caractères")
    except Exception as e:
        print_notification(f"Erreur lors de la conversion des types: {e}")
        print_notification("Poursuite du traitement avec les types actuels")

    # Vérifier que les colonnes nécessaires existent
    colonnes_requises = ['sc_sessionStartEnd', 'sc_timeStampOpening']
    for colonne in colonnes_requises:
        if colonne not in df.columns:
            print_notification(f"Erreur : La colonne '{colonne}' est absente du fichier")
            print_notification(f"Colonnes disponibles : {', '.join(df.columns)}")
            exit(1278)
            return

    # Convertir la colonne timeStampOpening en datetime si ce n'est pas déjà fait
    if 'timeStampOpeningConvertedtoDate' not in df.columns:
        print_notification("Conversion de la colonne timeStampOpening en datetime")
        df['timeStampOpeningConvertedtoDate'] = pd.to_datetime(df['sc_timeStampOpening'], unit='s')

    # Vérifier l'intégrité des sessions (nombre de 10 = nombre de 20)
    nb_debuts = sum(df['sc_sessionStartEnd'] == 10)
    nb_fins = sum(df['sc_sessionStartEnd'] == 20)

    print_notification(f"Nombre de débuts de session (10) : {nb_debuts}")
    print_notification(f"Nombre de fins de session (20) : {nb_fins}")

    if nb_debuts != nb_fins:
        print_notification("ATTENTION : Le nombre de débuts de session ne correspond pas au nombre de fins de session")
        print_notification("Le traitement continue mais les résultats pourraient être incorrects")
    else:
        print_notification(f"Le fichier contient {nb_debuts} sessions complètes")

    # Identifier toutes les sessions
    sessions = []
    debut_session = None

    for index, row in df.iterrows():
        if row['sc_sessionStartEnd'] == 10:
            debut_session = index, row['timeStampOpeningConvertedtoDate']
        elif row['sc_sessionStartEnd'] == 20 and debut_session is not None:
            sessions.append((debut_session[0], index, debut_session[1], row['timeStampOpeningConvertedtoDate']))
            debut_session = None

    # Si la dernière session n'est pas complète, l'ignorer
    if debut_session is not None:
        print_notification("ATTENTION : La dernière session n'est pas complète et sera ignorée")

    # Afficher toutes les sessions identifiées seulement si on n'utilise pas les paramètres par défaut
    if not use_default_params:
        print_notification(f"\n{'#' * 5} LISTE DES SESSIONS DISPONIBLES ({len(sessions)}) {'#' * 5}")
        print(f"\n{'N°':^4}|{'DATE DÉBUT':^25}|{'DATE FIN':^25}|{'DURÉE':^15}")
        print("-" * 71)

        for i, (start_idx, end_idx, start_time, end_time) in enumerate(sessions):
            duree = end_time - start_time
            heures = duree.total_seconds() // 3600
            minutes = (duree.total_seconds() % 3600) // 60
            duree_str = f"{int(heures)}h {int(minutes)}m"
            print(
                f"{i + 1:^4}| {start_time.strftime('%Y-%m-%d %H:%M:%S')} | {end_time.strftime('%Y-%m-%d %H:%M:%S')} | {duree_str:^15}")

        print("-" * 71)

    # Configuration des paramètres par défaut
    default_nb_splits = 5
    default_values = [85, 186, 285,334]  # Valeurs par défaut pour les débuts des splits 2, 3 et 4 5

    # Utiliser les paramètres par défaut ou demander à l'utilisateur
    if use_default_params:
        nb_splits = default_nb_splits
        print_notification(f"Utilisation des paramètres par défaut : {nb_splits} splits")
        print_notification(
            f"Points de division par défaut : début, session {default_values[0]}, session {default_values[1]}, session {default_values[2]}")
    else:
        # Étape 2: Demander combien de splits à effectuer avec valeur par défaut (4)
        print_notification(f"\nNombre de splits suggéré par défaut : {default_nb_splits}")

        try:
            user_input = input(f"Combien de splits souhaitez-vous effectuer ? [Par défaut: {default_nb_splits}] : ")

            # Si l'utilisateur appuie simplement sur Entrée, utiliser la valeur par défaut
            if user_input.strip() == "":
                nb_splits = default_nb_splits
                print(f"→ Utilisation de la valeur par défaut : {default_nb_splits} splits")
            else:
                nb_splits = int(user_input)

            if nb_splits <= 0:
                print_notification("Le nombre de splits doit être positif. Utilisation de la valeur par défaut.")
                nb_splits = default_nb_splits
            if nb_splits > len(sessions):
                print_notification(f"Attention : Il n'y a que {len(sessions)} sessions disponibles.")
                print_notification("Le nombre de splits sera limité au nombre de sessions.")
                nb_splits = len(sessions)
        except ValueError:
            print_notification("Entrée invalide. Utilisation de la valeur par défaut.")
            nb_splits = default_nb_splits

    # Ajuster les valeurs par défaut selon le nombre de splits
    if nb_splits != default_nb_splits:
        # Pour un autre nombre de splits, calculer des valeurs réparties uniformément
        default_values = []
        for i in range(1, nb_splits):
            # Répartir les sessions uniformément entre les splits
            val = min(round(i * len(sessions) / nb_splits), len(sessions))
            if val > 1:  # Éviter de commencer un split à la session 1
                default_values.append(val)

    points_de_division = [0]  # Le premier split commence toujours au début du fichier

    if use_default_params:
        # Utiliser directement les valeurs par défaut
        for i in range(1, nb_splits):
            if i - 1 < len(default_values):
                session_num = default_values[i - 1]
                if session_num > len(sessions):
                    session_num = len(sessions)
                points_de_division.append(session_num - 1)  # -1 car les indices commencent à 0
                print_notification(f"Split {i + 1} commencera à la session {session_num}")
    else:
        print_notification("\nLe premier split commencera au début du fichier.")

        # Afficher les valeurs par défaut suggérées pour les splits
        print_notification("\nValeurs suggérées pour les débuts des splits:")
        for i in range(1, nb_splits):
            if i - 1 < len(default_values):
                default_val = default_values[i - 1]
                if default_val > len(sessions):
                    default_val = len(sessions)
                print_notification(f"- Split {i + 1} : session {default_val}")

        for i in range(1, nb_splits):
            while True:
                try:
                    # Proposer la valeur par défaut et permettre à l'utilisateur de la modifier
                    default_val = default_values[i - 1] if i - 1 < len(default_values) else min(
                        i * len(sessions) // nb_splits, len(sessions))
                    if default_val > len(sessions):
                        default_val = len(sessions)

                    print(f"\nConsultez la liste des sessions ci-dessus pour faire votre choix.")
                    user_input = input(
                        f"Entrez le numéro de la session qui commence le split {i + 1} (2-{len(sessions)}) [Par défaut: {default_val}] : ")

                    # Si l'utilisateur appuie simplement sur Entrée, utiliser la valeur par défaut
                    if user_input.strip() == "":
                        session_num = default_val
                        print(f"→ Utilisation de la valeur par défaut : {default_val}")
                    else:
                        session_num = int(user_input)

                    if 2 <= session_num <= len(sessions):
                        # On utilise session_num - 1 car les indices commencent à 0 dans la liste
                        session_debut = sessions[session_num - 1][2]  # Date de début de la session
                        print(
                            f"→ Split {i + 1} commencera à la session {session_num} (débutant le {session_debut.strftime('%Y-%m-%d %H:%M:%S')})")
                        points_de_division.append(session_num - 1)
                        break
                    else:
                        print(f"❌ Veuillez entrer un numéro entre 2 et {len(sessions)}")
                except (ValueError, IndexError):
                    print("❌ Veuillez entrer un nombre valide")

    # Afficher les points de division pour débogage
    print_notification("\n[DEBUG] Points de division avant ajout de la fin: " + str(points_de_division))

    # Ajouter un point supplémentaire pour la fin du fichier
    points_de_division.append(len(sessions))

    print_notification("[DEBUG] Points de division après ajout de la fin: " + str(points_de_division))
    print_notification(f"[DEBUG] Nombre de points de division: {len(points_de_division)}")
    print_notification(f"[DEBUG] Nombre total de sessions: {len(sessions)}")

    # Créer les splits avec débogage
    splits = []
    for i in range(len(points_de_division) - 1):
        debut_split = points_de_division[i]
        # Utiliser directement le point de division suivant sans soustraction
        fin_split = points_de_division[i + 1]

        print_notification(
            f"[DEBUG] Vérification split {i + 1}: début={debut_split}, fin={fin_split}, len(sessions)={len(sessions)}")

        if fin_split > len(sessions):
            fin_split = len(sessions)
            print_notification(f"[DEBUG] Ajustement fin_split à {fin_split} car hors limites")

        if debut_split >= len(sessions):
            print_notification(
                f"[DEBUG] AVERTISSEMENT: Le split {i + 1} est invalide (début: {debut_split + 1}, fin: {fin_split}) - début hors limites")
            continue  # Cette ligne saute la création du split

        if debut_split >= fin_split:
            print_notification(
                f"[DEBUG] AVERTISSEMENT: Le split {i + 1} est invalide (début: {debut_split + 1}, fin: {fin_split}) - début ≥ fin")
            continue  # Cette ligne saute la création du split

        # Première session du split
        premiere_session = sessions[debut_split]
        # Dernière session du split
        derniere_session = sessions[fin_split - 1]  # -1 car fin_split est l'indice APRÈS la dernière session du split

        # Indices dans le DataFrame
        debut_idx = premiere_session[0]  # Indice de début de la première session
        fin_idx = derniere_session[1]  # Indice de fin de la dernière session

        # Dates de début et de fin
        date_debut = premiere_session[2]
        date_fin = derniere_session[3]

        splits.append((i + 1, debut_idx, fin_idx, date_debut, date_fin, fin_split - debut_split))
        print_notification(
            f"[DEBUG] Split {i + 1} créé: début={debut_split + 1}, fin={fin_split}, sessions incluses={fin_split - debut_split}")

    # Afficher les splits pour information de manière détaillée
    print_notification(f"\n{'#' * 5} SPLITS QUI SERONT CRÉÉS ({len(splits)}) {'#' * 5}")

    if not use_default_params:
        print(f"\n{'SPLIT':^5}|{'DATE DÉBUT':^25}|{'DATE FIN':^25}|{'NOMBRE SESSIONS':^15}|{'SESSIONS INCLUSES':^20}")
        print("-" * 93)

        for i, (num_split, debut_idx, fin_idx, date_debut, date_fin, nb_sessions) in enumerate(splits):
            # Déterminer les sessions incluses dans ce split
            debut_session_num = points_de_division[i] + 1  # +1 car l'affichage des sessions commence à 1
            fin_session_num = points_de_division[
                i + 1]  # Pas besoin de -1 car point_de_division[i+1] est déjà l'indice + 1
            sessions_incluses = f"{debut_session_num} à {fin_session_num if fin_session_num < len(sessions) + 1 else len(sessions)}"

            print(
                f"{num_split:^5}| {date_debut.strftime('%Y-%m-%d %H:%M:%S')} | {date_fin.strftime('%Y-%m-%d %H:%M:%S')} | {nb_sessions:^15} | {sessions_incluses:^20}")
    else:
        # Version simplifiée pour le mode automatique
        for num_split, debut_idx, fin_idx, date_debut, date_fin, nb_sessions in splits:
            print_notification(
                f"Split {num_split}: {date_debut.strftime('%Y-%m-%d')} à {date_fin.strftime('%Y-%m-%d')} ({nb_sessions} sessions)")

    # Demander confirmation ou procéder automatiquement
    if use_default_params:
        print_notification(f"Mode automatique activé : création de {len(splits)} fichiers sans confirmation")
        confirmation_ok = True
    else:
        print("\n" + "=" * 50)
        print("CONFIRMATION DE CRÉATION DES FICHIERS")
        print("=" * 50)
        print(f"Vous êtes sur le point de créer {len(splits)} fichiers distincts")
        print(f"Les fichiers seront enregistrés dans le même dossier que le fichier source")

        # Exemple avec le format de date DDMMYYYY pour la démonstration
        if splits:
            exemple_date_debut = splits[0][3].strftime('%d%m%Y')
            exemple_date_fin = splits[0][4].strftime('%d%m%Y')
            print(
                f"Exemple de nom de fichier: {file_name.split('.')[0]}__split1_{exemple_date_debut}_{exemple_date_fin}.{file_name.split('.')[-1]}")
        print("=" * 50)

        confirmation = input("\nConfirmez-vous la création de ces splits ? (o/n) : ")
        confirmation_ok = confirmation.lower() in ['o', 'oui', 'y', 'yes']

    if not confirmation_ok:
        print_notification("Opération annulée par l'utilisateur.")
        return

    # Créer les fichiers de split
    for num_split, debut_idx, fin_idx, date_debut, date_fin, _ in splits:
        # Extraire les données pour ce split
        split_df = df.iloc[debut_idx:fin_idx + 1].reset_index(drop=True)

        # Créer le nom du fichier
        nom_fichier = file_name
        nom_base, extension = os.path.splitext(nom_fichier)
        dossier = directory_path

        # Formater les dates pour le nom de fichier au format DDMMYYYY (jour-mois-année)
        date_debut_str = date_debut.strftime('%d%m%Y')
        date_fin_str = date_fin.strftime('%d%m%Y')

        nouveau_fichier = f"{nom_base}__split{num_split}_{date_debut_str}_{date_fin_str}{extension}"
        chemin_sortie = os.path.join(dossier, nouveau_fichier)

        # Sauvegarder le split avec le même encodage que celui utilisé pour lire le fichier
        split_df.to_csv(chemin_sortie, sep=';', index=False, encoding=encoding_used)
        print_notification(f"Split {num_split} sauvegardé dans {chemin_sortie} ({len(split_df)} lignes)")

    print_notification("Division du fichier par sessions terminée avec succès")