from enum import Enum
import numpy as np
import platform
from termcolor import colored
from colorama import Fore, Style, init

if platform.system() != "Darwin":  # "Darwin" est le nom interne de macOS
    import cupy as cp
else:
    print("CuPy ne sera pas importé sur macOS.")

import pandas as pd
from dateutil.relativedelta import relativedelta
from datetime import datetime, timedelta
import numba as nb
import time


class cv_config(Enum):
    TIME_SERIE_SPLIT = 0
    TIME_SERIE_SPLIT_NON_ANCHORED_AFTER_PREVTRAIN = 1
    TIME_SERIE_SPLIT_NON_ANCHORED_ROLLING=2
    TIME_SERIE_SPLIT_NON_ANCHORED_AFTER_PREVVAL = 3
    TIMESERIES_SPLIT_BY_ID = 4
    K_FOLD = 5
    K_FOLD_SHUFFLE = 6

class modelType (Enum):
    XGB=0
    LGBM=1
    CATBOOST=2
    RF=3
    SVC=4
    XGBRF=5 #random forest using xgb pour faster computing
class optuna_doubleMetrics(Enum):
    DISABLE = 0
    USE_DIST_TO_IDEAL = 1
    USE_WEIGHTED_AVG = 2

class rfe_param(Enum):
    NO_RFE = 0
    RFE_WITH_OPTUNA = 1
    RFE_AUTO = 2


class model_custom_objective(Enum):
    LGB_CUSTOM_OBJECTIVE_PROFITBASED=0
    LGB_CUSTOM_OBJECTIVE_BINARY = 10
    LGB_CUSTOM_OBJECTIVE_CROSS_ENTROPY = 11
    LGB_CUSTOM_OBJECTIVE_CROSS_ENTROPY_LAMBDA = 12
    XGB_CUSTOM_OBJECTIVE_PROFITBASED = 20


class model_custom_metric(Enum):
    LGB_CUSTOM_METRIC_PNL=0
    XGB_METRIC_ROCAUC = 10
    XGB_METRIC_AUCPR = 11
    XGB_METRIC_F1 = 12
    XGB_METRIC_PRECISION = 13
    XGB_METRIC_RECALL = 14
    XGB_METRIC_MCC = 15
    XGB_METRIC_YOUDEN_J = 16
    XGB_METRIC_SHARPE_RATIO = 17
    XGB_CUSTOM_METRIC_PNL = 18

class scalerChoice(Enum):
    SCALER_DISABLE = 0
    SCALER_STANDARD = 1
    SCALER_ROBUST = 2
    SCALER_MINMAX = 3  # Nouveau : échelle [0,1]
    SCALER_MAXABS = 4  # Nouveau : échelle [-1,1]

class AutoFilteringOptions(Enum):
    DISPLAY_MODE_NOFILTERING = 0
    ENABLE_VIF_CORR_MI = 1
    ENABLE_MRMR = 2
    ENABLE_FISHER=3

###############################################################################
# Classe d'options de filtrage
###############################################################################



class ScalerMode(Enum):
    FIT_TRANSFORM = 0  # Pour l'entraînement : fit + transform
    TRANSFORM = 1
     # Pour le test : transform uniquement
def check_label_pnl_alignment(data):
    """
    Vérifie que dans data, pour chaque échantillon où y_train_no99_fullRange == 1,
    y_pnl_data_train_no99_fullRange > 0.
    """
    # Extraire les tableaux
    y_labels = data['y_train_no99_fullRange']
    y_pnl = data['y_pnl_data_train_no99_fullRange']

    # Vérifier la même longueur
    if len(y_labels) != len(y_pnl):
        raise ValueError(f"check_label_pnl_alignment Taille différente: y_labels={len(y_labels)}, y_pnl={len(y_pnl)}")

    # Créer un masque pour les labels == 1
    mask_positive = (y_labels == 1)

    # Vérifier que tous les PnL correspondants sont > 0
    if not np.all(y_pnl[mask_positive] > 0):
        # Récupérer les indices où la condition échoue
        bad_indices = np.where(y_pnl[mask_positive] <= 0)[0]
        raise ValueError(
            f"Pour y_train_no99_fullRange==1, certains PnL ne sont pas > 0. Indices relatifs (dans les positifs)={bad_indices}"
        )
    else:
        print("Vérification OK : tous les échantillons label=1 ont un PnL strictement > 0.")

import numpy as np
def prepare_dataSplit_cv_train_val(config, data_4cv, train_pos, val_pos):
    """
    Prépare les données d'entraînement et de validation pour la validation croisée,
    en restant au format pandas (DataFrame/Series) et en utilisant l'indexation par position (.iloc).

    Parameters
    ----------
    config : dict
        Dictionnaire de configuration contenant la clé 'device_' avec la valeur 'cpu' ou 'gpu'.
    data : dict
        Contient les clés suivantes (toutes au format pandas) :
            - 'X_train_no99_fullRange': DataFrame des features
            - 'X_train_no99_fullRange_pd': DataFrame identique ou similaire (optionnel)
            - 'y_train_no99_fullRange': Series de la target
            - 'y_pnl_data_train_no99_fullRange': Series du PnL
    train_pos : np.ndarray
        Positions (lignes) à prendre pour l'entraînement
    val_pos : np.ndarray
        Positions (lignes) à prendre pour la validation

    Returns
    -------
    X_train_cv : DataFrame (features train)
    X_train_cv_pd : DataFrame identique ou “brute” (si besoin) pour train
    Y_train_cv : Series (target train)
    X_val_cv : DataFrame (features val)
    X_val_cv_pd : DataFrame identique ou “brute” (si besoin) pour val
    y_val_cv : Series (target val)
    y_pnl_data_train_cv : Series (PnL train)
    y_pnl_data_val_cv : Series (PnL val)
    """

    # --- 1) Extraction par .iloc => indexation par position ---
    # Features
    X_train_cv = data_4cv['X_train_no99_fullRange'].iloc[train_pos]
    X_val_cv   = data_4cv['X_train_no99_fullRange'].iloc[val_pos]

    # (Optionnel) Les versions "pd" si elles sont différentes et qu'on en a besoin
    X_train_cv_pd = data_4cv.get('X_train_no99_fullRange_pd', X_train_cv).iloc[train_pos]
    X_val_cv_pd   = data_4cv.get('X_train_no99_fullRange_pd', X_val_cv).iloc[val_pos]

    # Targets
    Y_train_cv = data_4cv['y_train_no99_fullRange'].iloc[train_pos]
    y_val_cv   = data_4cv['y_train_no99_fullRange'].iloc[val_pos]

    # PnL
    y_pnl_data_train_cv = data_4cv['y_pnl_data_train_no99_fullRange'].iloc[train_pos]
    y_pnl_data_val_cv   = data_4cv['y_pnl_data_train_no99_fullRange'].iloc[val_pos]

    # --- 2) GPU/CPU (optionnel) ---
    # Si device != 'cpu' et qu'on veut vraiment manipuler du GPU, il faudrait convertir en arrays,
    # puis faire du cupy, ce qui signifie quitter le format DataFrame. Exemple ci-dessous en commentaire :
    """
    if config.get('device_', 'cpu') != 'cpu':
        import cupy as cp
        # Conversion DataFrame -> ndarray
        X_train_cv_arr = X_train_cv.to_numpy()
        X_val_cv_arr   = X_val_cv.to_numpy()
        # Cupy arrays
        X_train_cv_gpu = cp.asarray(X_train_cv_arr)
        X_val_cv_gpu   = cp.asarray(X_val_cv_arr)
        # Idem pour Y_train_cv, y_val_cv, etc.
        # --> Puis potentiellement reconvertir en DataFrame cuDF (si tu utilises cuDF),
        # ou juste garder en Cupy array.
    """

    return (
        X_train_cv,         # DataFrame (features) - train
        X_train_cv_pd,      # DataFrame "pd" (if needed) - train
        Y_train_cv,         # Series (target) - train
        X_val_cv,           # DataFrame (features) - val
        X_val_cv_pd,        # DataFrame "pd" - val
        y_val_cv,           # Series (target) - val
        y_pnl_data_train_cv,# Series (PnL) - train
        y_pnl_data_val_cv   # Series (PnL) - val
    )



def sigmoidCustom(x):
# Supposons que x est déjà un tableau CuPy
  return 1 / (1 + cp.exp(-x))


def print_notification(message, color=None):
    """
    Affiche un message avec un horodatage. Optionnellement, le message peut être affiché en couleur.

    Args:
    - message (str): Le message à afficher.
    - color (str, optionnel): La couleur du texte ('red', 'green', 'yellow', 'blue', etc.).
    """
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

    # Définir les couleurs selon le choix de l'utilisateur pour le message uniquement
    if color == 'red':
        color_code = Fore.RED
    elif color == 'green':
        color_code = Fore.GREEN
    elif color == 'yellow':
        color_code = Fore.YELLOW
    elif color == 'blue':
        color_code = Fore.BLUE
    else:
        color_code = ''  # Pas de couleur

    # Afficher le message avec le timestamp non coloré et le message coloré si nécessaire
    print(f"\n[{timestamp}] {color_code}{message}{Style.RESET_ALL}")




def load_data(file_path: str) -> pd.DataFrame:
    print_notification(f"Début du chargement des données de: \n "
                       f"{file_path}")
    df = pd.read_csv(file_path, sep=';', encoding='iso-8859-1')
    print(f"Colonnes chargées: {df.columns.tolist()}")
    print(f"Premières lignes:\n{df.head()}")
    print_notification("Données chargées avec succès")
    return df

def verify_alignment(y_true_class_binaire, y_pnl_data):
    """
    Vérifie l'alignement entre les classes et le signe des PnL théoriques:
    - Si y_true_class_binaire == 1, alors y_pnl_data doit être > 0
    - Si y_true_class_binaire == 0, alors y_pnl_data doit être < 0

    Lève une ValueError si la condition n'est pas respectée.
    """
    import numpy as np

    # Convertir en tableaux numpy si ce n'est pas déjà le cas
    y_true_array = np.array(y_true_class_binaire)
    y_pnl_array = np.array(y_pnl_data)

    if np.any(y_pnl_array == 0):
        zero_indices = np.where(y_pnl_array == 0)[0]
        raise ValueError(
            f"La variable y_pnl_array contient des valeurs exactement égales à 0 ce qui n'est pas possible {zero_indices}")

    # Vérifier l'alignement des classes positives (y_true == 1)
    misaligned_positives = np.where((y_true_array.astype(bool)) & (y_pnl_array < 0))[0]
    #print(y_true_array.shape)
    #print(y_pnl_array.shape)


    if len(misaligned_positives) > 0:
        raise ValueError(f"Désalignement détecté: y_true == 1 mais y_pnl_data < 0 aux indices {misaligned_positives}")

    # Vérifier l'alignement des classes négatives (y_true == 0)
    misaligned_negatives = np.where((~y_true_array.astype(bool)) & (y_pnl_array > 0))[0]

    if len(misaligned_negatives) > 0:
        raise ValueError(f"Désalignement détecté: y_true == 0 mais y_pnl_data > 0 aux indices {misaligned_negatives}")

    #print("Vérification réussie: Les classes 0 et 1 sont correctement alignées avec le signe des PnL théoriques.")
def calculate_profitBased(
    y_true_class_binaire,
    y_pred_threshold,
    y_pnl_data_array=None,
    other_params=None,
    config=None
):
    """
    Calcule les métriques de profit en utilisant la série PNL
    directement fournie en paramètre y_pnl_data_array.
    """
    # Pas besoin de y_pnl_data_train_cv ni y_pnl_data_val_cv_OrTest
    # dans ce mode "direct", on suppose que c'est déjà le bon vecteur

    if y_pnl_data_array is None:
        raise ValueError("Il faut fournir y_pnl_data_array pour calculer le profit.")

    y_true_class_binaire = np.asarray(y_true_class_binaire)
    y_pred = np.asarray(y_pred_threshold)
    y_pnl_data_array = np.asarray(y_pnl_data_array)

    # Vérifier l'alignement
    if not (len(y_true_class_binaire) == len(y_pred) == len(y_pnl_data_array)):
        raise ValueError(f"Mauvais alignement: {len(y_true_class_binaire)=}, {len(y_pred)=}, {len(y_pnl_data_array)=}")

    # Calcul masques
    tp_mask = (y_true_class_binaire == 1) & (y_pred == 1)
    fp_mask = (y_true_class_binaire == 0) & (y_pred == 1)
    fn_mask = (y_true_class_binaire == 1) & (y_pred == 0)

    # Comptage
    tp = np.sum(tp_mask)
    fp = np.sum(fp_mask)
    fn = np.sum(fn_mask)

    # Récupération des pénalités dans metric_dict/config
    penalty_per_fn = other_params.get('penalty_per_fn', config.get('penalty_per_fn', 0))

    # Calcul
    tp_profits = np.sum(np.maximum(0, y_pnl_data_array[tp_mask])) if tp > 0 else 0
    fp_losses = np.sum(np.minimum(0, y_pnl_data_array[fp_mask])) if fp > 0 else 0
    fn_penalty = fn * penalty_per_fn

    total_profit = tp_profits + fp_losses + fn_penalty
    # Récupération des PnL utilisés pour les TP et FP
    tp_pnls = y_pnl_data_array[tp_mask]
    fp_pnls = y_pnl_data_array[fp_mask]

    # Vérification que toutes les valeurs de TP sont égales à 175
    tp_unique_values = np.unique(tp_pnls)
    if not np.all(tp_unique_values == 175):
        print(f"❌ Incohérence : TP contient des valeurs autres que 175 : {tp_unique_values}")
        exit(100)
    # else:
    #     print("✅ Tous les TP ont une valeur de 175.")
    #     exit(101)


    # Vérification que toutes les valeurs de FP sont égales à -227
    fp_unique_values = np.unique(fp_pnls)
    if not np.all(fp_unique_values == -227):
        print(f"❌ Incohérence : FP contient des valeurs autres que -227 : {fp_unique_values}")
        exit(102)

    # else:
    #     print("✅ Tous les FP ont une valeur de -227.")

    return float(total_profit), int(tp), int(fp)

# Fonction sigmoid pour XGBoost
def sigmoidCustom_cpu(x):
    """Custom sigmoid function."""
    x = np.array(x, dtype=np.float64)  # Conversion explicite
    return 1 / (1 + np.exp(-x))



def predict_and_process(pred_proba_raw, threshold, config):
    """Applique la sigmoid et le seuillage sur les prédictions.

    Args:
        pred_proba: array de probabilités brutes
        threshold: seuil de classification
        config: dict contenant la configuration avec la clé 'device_'

    Returns:
        tuple: (probabilités après sigmoid, prédictions binaires)
    """
    if config['device_'] == 'cpu':
        pred_proba_afterSig = sigmoidCustom_cpu(pred_proba_raw)
        pred_proba_afterSig = np.clip(pred_proba_afterSig, 0.0, 1.0)
        pred = (pred_proba_afterSig > threshold).astype(np.int32)
    else:
        pred_proba_afterSig = sigmoidCustom(pred_proba_raw)
        pred_proba_afterSig = cp.clip(pred_proba_afterSig, 0.0, 1.0)
        pred = (pred_proba_afterSig > threshold).astype(cp.int32)
        exit(55)

    return pred_proba_afterSig, pred


def compute_confusion_matrix_cupy(y_true_gpu, y_pred_gpu):
    tp = cp.sum((y_true_gpu == 1) & (y_pred_gpu == 1))
    fp = cp.sum((y_true_gpu == 0) & (y_pred_gpu == 1))
    tn = cp.sum((y_true_gpu == 0) & (y_pred_gpu == 0))
    fn = cp.sum((y_true_gpu == 1) & (y_pred_gpu == 0))
    return tn, fp, fn, tp


def compute_confusion_matrix_cpu(y_true, y_pred, config):
    """Calcule la matrice de confusion.

    Args:
        y_true: Labels réels
        y_pred: Prédictions
        config: dict contenant la configuration avec la clé 'device_'

    Returns:
        tuple: (TN, FP, FN, TP) - éléments de la matrice de confusion
    """
    if config['device_'] == 'cpu':
        TP = np.sum((y_true == 1) & (y_pred == 1))
        TN = np.sum((y_true == 0) & (y_pred == 0))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))
    else:
        TP = cp.sum((y_true == 1) & (y_pred == 1))
        TN = cp.sum((y_true == 0) & (y_pred == 0))
        FP = cp.sum((y_true == 0) & (y_pred == 1))
        FN = cp.sum((y_true == 1) & (y_pred == 0))

    return TN, FP, FN, TP


def predict_and_compute_metrics_XgbOrLightGbm(model, X_data, y_true, best_iteration, threshold, config):
    """
    Effectue les prédictions et calcule les métriques de confusion pour un jeu de données.

    Args:
        model: Le modèle entraîné
        X_data: Les features d'entrée
        y_true: Les labels réels
        best_iteration: Le meilleur nombre d'itérations du modèle
        threshold: Le seuil de classification
        config: Dictionnaire de configuration

    Returns:
        tuple: (pred_proba, predictions, (tn, fp, fn, tp), y_true_converted)
    """
    # Prédictions
    model_type = config['model_type']
    if model_type == modelType.XGB:
        import xgboost as xgb
       # dData = xgb.DMatrix(X_data)
        pred_proba_log_odds = model.predict(X_data, iteration_range=(0, best_iteration), output_margin=True)
        #print("pred_proba_log_odds ",pred_proba_log_odds)
    elif model_type == modelType.LGBM:
        pred_proba_log_odds = model.predict(X_data, num_iteration=model.best_iteration, raw_score=True)
    #print(f"pred_proba_log_oddsMin: {np.min(pred_proba_log_odds)}, Max: {np.max(pred_proba_log_odds)}")

    # Conversion GPU si nécessaire
    if config['device_'] != 'cpu':
        import cupy as cp
        pred_proba_log_odds = cp.asarray(pred_proba_log_odds, dtype=cp.float32)

    # Application du seuil et processing
    pred_proba_afterSig, predictions = predict_and_process(pred_proba_log_odds, threshold, config)

    # Conversion des données pour la matrice de confusion
    if config['device_'] != 'cpu':
        import cupy as cp
        y_true_converted = cp.asarray(y_true) if isinstance(y_true, (np.ndarray, pd.Series)) else y_true
        predictions_converted = cp.asarray(predictions) if isinstance(predictions,
                                                                      (np.ndarray, pd.Series)) else predictions
    else:
        y_true_converted = y_true
        predictions_converted = predictions

    # Calcul de la matrice de confusion
    tn, fp, fn, tp = compute_confusion_matrix_cpu(y_true_converted, predictions_converted, config)

    return pred_proba_afterSig,pred_proba_log_odds, predictions_converted, (tn, fp, fn, tp), y_true_converted


def predict_and_compute_metrics_RF(
        model,
        X_data,
        y_true,
        threshold=0.5,
        config=None
):
    """
    Effectue les prédictions et calcule les métriques sur un ensemble de données
    pour un modèle Random Forest ou XGBRFClassifier.
    Prend en charge la conversion CPU/GPU selon la configuration.

    Parameters:
    -----------
    model : modèle entraîné (RandomForestClassifier ou XGBRFClassifier)
    X_data : features de l'ensemble de données
    y_true : labels réels de l'ensemble de données
    threshold : seuil de décision (défaut: 0.5)
    config : dictionnaire de configuration avec clé 'device_' (défaut: None)

    Returns:
    --------
    tuple :
        - pred_proba : probabilités prédites
        - pred_proba_log_odds : log-odds des probabilités
        - predictions_converted : prédictions binaires converties (CPU ou GPU)
        - tn : true negative
        - fp : false positive
        - fn : false negative
        - tp : true positive
        - y_true_converted : valeurs réelles converties (CPU ou GPU)
    """
    import numpy as np
    from sklearn.metrics import confusion_matrix
    import pandas as pd

    # Valeur par défaut pour config
    if config is None:
        config = {'device_': 'cpu'}

    # Calcul des probabilités et log-odds
    pred_proba = model.predict_proba(X_data)[:, 1]
    pred_proba_log_odds = np.log(pred_proba / (1 - pred_proba + 1e-10))
    predictions = (pred_proba >= threshold).astype(int)

    # Conversion des données pour la matrice de confusion
    if config['device_'] != 'cpu':
        import cupy as cp
        y_true_converted = cp.asarray(y_true) if isinstance(y_true, (np.ndarray, pd.Series)) else y_true
        predictions_converted = cp.asarray(predictions) if isinstance(predictions,
                                                                      (np.ndarray, pd.Series)) else predictions
    else:
        y_true_converted = y_true
        predictions_converted = predictions

    # Calcul de la matrice de confusion
    tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()

    return pred_proba, pred_proba_log_odds, predictions_converted, tn, fp, fn, tp, y_true_converted

def timestamp_to_date_utc_(timestamp):
    try:
        date_format = "%Y-%m-%d %H:%M:%S"
        if isinstance(timestamp, pd.Series):
            return timestamp.apply(lambda x: time.strftime(date_format, time.gmtime(x)))
        return time.strftime(date_format, time.gmtime(timestamp))
    except Exception as e:
        print(f"Erreur lors de la conversion timestamp->date: {str(e)}")
        return None

def compute_raw_train_dist(X_train_full, X_train_cv_pd, X_val_cv_pd, fold_num, nb_split_tscv, fold_raw_data
                           ,is_log_enabled,df_init_candles):
    """
    Log les métriques de validation croisée au format standardisé.

    Format:
    === Fold N/Total ===
    === Ranges ===
    X_train_full index | Train pos range | Val pos range
    === TRAIN ===
    Distribution (Avant optimisation): stats
    Métriques (après optimisation): métriques
    === VALIDATION ===
    Période: dates et durée
    Distribution (Avant optimisation): stats
    Métriques (après optimisation): métriques

    Args:
        X_train_full (pd.DataFrame): Données d'entraînement complètes
        X_train_cv (pd.DataFrame): Données d'entraînement du fold
        X_val_cv_pd (pd.DataFrame): Données de validation du fold
        val_pos (array-like): Positions des données de validation
        Y_train_cv (cp.ndarray): Labels d'entraînement
        y_val_cv (cp.ndarray): Labels de validation
        tp_train, fp_train, tn_train, fn_train (int): Métriques d'entraînement
        tp_val, fp_val, tn_val, fn_val (int): Métriques de validation
        config (dict): Configuration
        fold_num (int): Numéro du fold courant
        nb_split_tscv (int): Nombre total de folds
        fold_data (dict): Données du fold

    Returns:
        dict: Dictionnaire contenant les métriques calculées
    """
    try:
        # 1. Conversion des timestamps en dates UTC




        # Distribution avant optimisation
        train_dist = fold_raw_data['distributions']['train']
        total_samples_train = sum(train_dist.values())
        class1_train = train_dist.get(1, 0)
        class0_train = train_dist.get(0, 0)
        winrate_train = (class1_train / total_samples_train * 100) if total_samples_train > 0 else 0
        ratio_train = class1_train / class0_train if class0_train > 0 else float('inf')
        train_pos = fold_raw_data['train_indices']
        val_pos = fold_raw_data['val_indices']
        start_time_train, end_time_train, _ = get_val_cv_time_range(X_full=X_train_full, X=X_train_cv_pd)
        time_diff_train = calculate_time_difference(
            timestamp_to_date_utc_(start_time_train),
            timestamp_to_date_utc_(end_time_train)
        )



        # Période
        start_time_val, end_time_val, _ = get_val_cv_time_range(X_full=X_train_full, X=X_val_cv_pd)
        time_diff_val = calculate_time_difference(
            timestamp_to_date_utc_(start_time_val),
            timestamp_to_date_utc_(end_time_val)
        )

        # Distribution avant optimisation
        val_dist = fold_raw_data['distributions']['val']
        total_samples_val = sum(val_dist.values())
        class1_val = val_dist.get(1, 0)
        class0_val = val_dist.get(0, 0)
        winrate_val = (class1_val / total_samples_val * 100) if total_samples_val > 0 else 0
        ratio_val = class1_val / class0_val if class0_val > 0 else float('inf')


        # indice consécutif pour les données d'entraînement sur le fold
        train_start = X_train_cv_pd.index[0]  # Premier indice
        train_end = X_train_cv_pd.index[-1]  # Dernier indice
        close_cv_train = df_init_candles['close'].loc[train_start:train_end]
        high_cv_train = df_init_candles['high'].loc[train_start:train_end]
        low_cv_train = df_init_candles['low'].loc[train_start:train_end]

        #slope_cv_train = linear_regression_slope_market_trend(close_cv_train)
        slope_cv_train,r2_slope_cv_train,counter_moves_train=calculate_trend_strength(close_cv_train,high_cv_train,low_cv_train)

        # indice consécutif pour les données de validation sur le fold
        val_start = X_val_cv_pd.index[0]
        val_end = X_val_cv_pd.index[-1]
        close_cv_val = df_init_candles['close'].loc[val_start:val_end]
        high_cv_val = df_init_candles['high'].loc[val_start:val_end]
        low_cv_val = df_init_candles['low'].loc[val_start:val_end]

        #slope_cv_val = linear_regression_slope_market_trend(close_cv_val)
        slope_cv_val,r2_slope_cv_val,counter_moves_val=calculate_trend_strength(close_cv_val,high_cv_val,low_cv_val)


        # Mise à jour du dictionnaire des métriques
        raw_metrics={
            'train_metrics': {
                'total_samples_train': total_samples_train,
                'class1_train': class1_train,
                'class0_train': class0_train,
                'winrate': winrate_train,
                'ratio_train': ratio_train,
                'start_time_train': timestamp_to_date_utc_(start_time_train),
                'end_time_train': timestamp_to_date_utc_(end_time_train),
                'slope_cv_train':slope_cv_train,
                'r2_slope_cv_train': r2_slope_cv_train,
                'counter_moves_train': counter_moves_train,
            },
            'val_metrics': {
                'total': total_samples_val,
                'class1_val': class1_val,
                'class0_val': class0_val,
                'winrate': winrate_val,
                'ratio_val': ratio_val,
                'start_time_val': timestamp_to_date_utc_(start_time_val),
                'end_time_val': timestamp_to_date_utc_(end_time_val),
                'slope_cv_val': slope_cv_val,
                'r2_slope_cv_val':r2_slope_cv_val,
                'counter_moves_val': counter_moves_val
            }
        }
        if is_log_enabled:
            # === Fold et Ranges ===
            print(f"\n============ Fold {fold_num + 1}/{nb_split_tscv} ============")
            print("=== Ranges ===")

            print(f"X_train_full index: {X_train_full.index.min()}-{X_train_full.index.max()} | "
                  f"Train pos range: {min(train_pos)}-{max(train_pos)} | "
                  f"Val pos range: {min(val_pos)}-{max(val_pos)}")

            # === TRAIN ===
            print("=== TRAIN ===")
            # Période

            print(f"Période: Du {timestamp_to_date_utc_(start_time_train)} au {timestamp_to_date_utc_(end_time_train)} "
                  f"(Durée: {time_diff_train.months} mois, {time_diff_train.days} jours)")



            print("Indices d'entraînement :")
            print(f"De {train_start} à {train_end}")
            print("Valeurs des closes correspondantes de df_init_candles pour l'entraînement :")
            vals_close_train = close_cv_train.tolist()
            print(f"[{', '.join(map(str, vals_close_train[:10]))} ... {', '.join(map(str, vals_close_train[-10:]))}]")
            print(f"Pente de la régression linéaire pour l'entraînement: {slope_cv_train}\n")

            print("Distribution (Avant optimisation):")
            print(
                f"- Total: {total_samples_train} échantillons | class1: {class1_train} | class0: {class0_train}")
            print(f"- Winrate: {winrate_train:.2f}%")
            print(f"- Ratio réussis/échoués: {ratio_train:.2f}")

            # Métriques après optimisation
            print("Métriques (après optimisation):")


            # === VALIDATION ===
            print("\n=== VALIDATION ===")
            print(f"Période: Du {timestamp_to_date_utc_(start_time_val)} au {timestamp_to_date_utc_(end_time_val)} "
                  f"(Durée: {time_diff_val.months} mois, {time_diff_val.days} jours)")
            # Pour les données de validation

            print("Indices de validation :")
            print(f"De {val_start} à {val_end}")
            print("Valeurs des closes correspondantes de df_init_candles pour la validation :")
            vals_close_val = close_cv_val.tolist()
            print(f"[{', '.join(map(str, vals_close_val[:10]))} ... {', '.join(map(str, vals_close_val[-10:]))}]")
            print(f"Pente de la régression linéaire pour l'entraînement: {slope_cv_val}\n")

            print("Distribution (Avant optimisation):")
            print(
                f"- Total: {total_samples_val} échantillons | Trades réussis: {class1_val} | Trades échoués: {class0_val}")
            print(f"- Winrate: {winrate_val:.2f}%")
            print(f"- Ratio réussis/échoués: {ratio_val:.2f}")

        return raw_metrics
    except Exception as e:
        print(f"#########################################")
        print(f"Erreur dans compute_raw_train_dist: {str(e)}")
        print(f"#########################################")
        return None

def calculate_time_difference(start_date_str, end_date_str):
    date_format = "%Y-%m-%d %H:%M:%S"
    start_date = datetime.strptime(start_date_str, date_format)
    end_date = datetime.strptime(end_date_str, date_format)
    diff = relativedelta(end_date, start_date)
    return diff


def compute_balanced_weights_gpu(y):
    """Version GPU de compute_sample_weight('balanced')"""
    unique_classes, class_counts = cp.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(unique_classes)

    # Calcul des poids comme sklearn
    weights = n_samples / (n_classes * class_counts)
    samples_weights = weights[y.astype(cp.int32)]

    return samples_weights



def get_val_cv_time_range(X_full, X, index_val=None):
    """
    Obtient la plage temporelle en utilisant les index originaux.

    Args:
        X_full (pd.DataFrame): DataFrame complet avec toutes les colonnes incluant 'timeStampOpening'
        X (pd.DataFrame): DataFrame potentiellement réduit mais avec les mêmes indices que X_full
        index_val (array-like, optional): Indices de validation. Si None, utilise tout le DataFrame

    Returns:
        tuple: (start_time, end_time, num_sessions)
            - start_time (int): Timestamp du début de la période
            - end_time (int): Timestamp de la fin de la période
            - num_sessions (int): Nombre de sessions dans cette période

    Raises:
        ValueError: Si les DataFrames sont vides ou si 'timeStampOpening' n'existe pas
        KeyError: Si les index ne correspondent pas entre X_full et X
    """
    # Vérifications préliminaires
    """
    if X_full.empty:
        raise ValueError("X_full ne peut pas etre vide")

    if X.empty:
        raise ValueError("X ne peut pas etre vide")

    """
    if 'timeStampOpening' not in X_full.columns:
        raise ValueError("La colonne 'timeStampOpening' doit être présente dans X_full")

    try:
        # Si index_val est None, utiliser tous les indices de X
        if index_val is None:
            original_indices = X.index
        else:
            # Vérifier que index_val est valide
            if len(index_val) == 0:
                raise ValueError("index_val ne peut pas être vide")
            # Récupérer les index originaux correspondant aux indices de validation
            print(index_val)
            original_indices = X.index[index_val]

        # Vérifier que les indices existent dans X_full
        if not all(idx in X_full.index for idx in [original_indices[0], original_indices[-1]]):
            raise KeyError("Certains indices ne sont pas présents dans X_full")

        # Utiliser ces index originaux pour accéder à X_full
        start_time = X_full.loc[original_indices[0], 'timeStampOpening']
        end_time = X_full.loc[original_indices[-1], 'timeStampOpening']

        # Vérifier l'ordre chronologique
        if end_time < start_time:
            start_time, end_time = end_time, start_time

        # Extraire les données en utilisant les index originaux
        df_extracted = X_full.loc[original_indices[0]:original_indices[-1]]

        # Vérifier que df_extracted n'est pas vide
        if df_extracted.empty:
            raise ValueError("Aucune donnée n'a été extraite pour la période spécifiée")

        num_sessions = calculate_and_display_sessions(df_extracted)[0]

        return start_time, end_time, num_sessions

    except Exception as e:
        raise ValueError(f"Erreur lors du traitement des données temporelles: {str(e)}")

def calculate_and_display_sessions(df):
    session_start_end = df['SessionStartEnd'].astype(np.int32).values
    delta_timestamp = df['deltaTimestampOpening'].astype(np.float64).values
    ts_startSection = df['timeStampOpening'].head(1).values[0]
    ts_endSection = df['timeStampOpening'].tail(1).values[0]

    date_startSection = timestamp_to_date_utc(ts_startSection)
    date_endSection = timestamp_to_date_utc(ts_endSection)

    complete_sessions, total_minutes, residual_minutes = calculate_session_duration(session_start_end, delta_timestamp)

    session_duration_hours = 23
    session_duration_minutes = session_duration_hours * 60

    residual_sessions = residual_minutes / session_duration_minutes
    total_sessions = complete_sessions + residual_sessions

    # print(f"Nombre de sessions complètes : {complete_sessions}")
    # print(f"Minutes résiduelles : {residual_minutes:.2f}")
    # print(f"Équivalent en sessions des minutes résiduelles : {residual_sessions:.2f}")
    # print(f"Nombre total de sessions (complètes + résiduelles) : {total_sessions:.2f}")

    return total_sessions, date_startSection, date_endSection

@nb.njit
def calculate_session_duration(session_start_end, delta_timestamp):
    total_minutes = 0
    residual_minutes = 0
    session_start = -1
    complete_sessions = 0
    in_session = False

    # Gérer le cas où le fichier ne commence pas par un 10
    if session_start_end[0] != 10:
        first_20_index = np.where(session_start_end == 20)[0][0]
        residual_minutes += delta_timestamp[first_20_index] - delta_timestamp[0]

    for i in range(len(session_start_end)):
        if session_start_end[i] == 10:
            session_start = i
            in_session = True
        elif session_start_end[i] == 20:
            if in_session:
                # Session complète
                total_minutes += delta_timestamp[i] - delta_timestamp[session_start]
                complete_sessions += 1
                in_session = False
            else:
                # 20 sans 10 précédent, ne devrait pas arriver mais gérons le cas
                residual_minutes += delta_timestamp[i] - delta_timestamp[session_start if session_start != -1 else 0]

    # Gérer le cas où le fichier se termine au milieu d'une session
    if in_session:
        residual_minutes += delta_timestamp[-1] - delta_timestamp[session_start]
    elif session_start_end[-1] != 20:
        # Si la dernière valeur n'est pas 20 et qu'on n'est pas dans une session,
        # ajoutons le temps depuis le dernier 20 jusqu'à la fin
        last_20_index = np.where(session_start_end == 20)[0][-1]
        residual_minutes += delta_timestamp[-1] - delta_timestamp[last_20_index]

    return complete_sessions, total_minutes, residual_minutes



def timestamp_to_date_utc(timestamp):
    date_format = "%Y-%m-%d %H:%M:%S"
    if isinstance(timestamp, pd.Series):
        return timestamp.apply(lambda x: time.strftime(date_format, time.gmtime(x)))
    else:
        return time.strftime(date_format, time.gmtime(timestamp))


def compute_winrate_safe(tp, total_trades, config):
    """
    Calcule le winrate de manière sécurisée en pourcentage.

    Args:
        tp: Nombre de True Positives (trades gagnants)
        total_trades: Nombre total de trades
        config: Dictionnaire de configuration avec la clé 'device_'

    Returns:
        array: Winrate en pourcentage (0-100) avec gestion des divisions par zéro
    """
    # Sélectionne la bibliothèque appropriée selon le device
    xp = cp if config['device_'] == 'cuda' else np

    # Crée un masque pour identifier les positions où total_trades n'est pas zéro
    mask = total_trades != 0

    # Initialise le tableau de résultats avec des zéros
    result = xp.zeros_like(total_trades, dtype=xp.float32)

    # Calcule le winrate en pourcentage uniquement pour les positions valides
    # Multiplie par 100 pour convertir en pourcentage
    result[mask] = (tp[mask] / total_trades[mask]) * 100

    return result


def compute_winrate_ratio_difference(winrate_train, winrate_val, config):
    """
    Calcule l'écart relatif entre deux winrates sous forme de ratio en pourcentage.

    Args:
        winrate_train: Winrate de l'ensemble d'entraînement (déjà calculé)
        winrate_val: Winrate de l'ensemble de validation (déjà calculé)
        config: Configuration avec le type de device (CPU/GPU)

    Returns:
        array: Écart relatif en pourcentage avec gestion des cas spéciaux
    """
    # Sélectionne la bibliothèque selon le device
    xp = cp if config['device_'] == 'cuda' else np

    # Initialise le résultat avec des zéros
    result = xp.zeros_like(winrate_train, dtype=xp.float32)

    # Crée un masque pour les positions où winrate_train n'est pas zéro
    mask = winrate_train != 0

    # Calcule le ratio uniquement pour les positions valides
    result[mask] = (winrate_train[mask] - winrate_val[mask]) / winrate_train[mask] * 100

    return result

def log_fold_info(fold_num, nb_split_tscv, X_train_full, fold_data):
    """
    Affiche les informations détaillées sur le fold courant.

    Args:
        fold_num (int): Numéro du fold actuel
        nb_split_tscv (int): Nombre total de folds
        X_train_full (pd.DataFrame): Données d'entraînement complètes
        fold_data (dict): Données du fold préparées
    """


def calculate_profit_ratio(tp, fp, tp_fp_sum, profit_per_tp, loss_per_fp):
    """
    Calcule le ratio profit par trade avec gestion de la division par zéro

    Args:
        tp: Nombre de True Positives
        fp: Nombre de False Positives
        tp_fp_sum: Somme des trades (TP + FP)
        profit_per_tp: Profit par trade gagnant
        loss_per_fp: Perte par trade perdant

    Returns:
        float: Ratio de profit par trade, 0 si aucun trade
    """
    if tp_fp_sum == 0:
        return 0.0

    return (tp * profit_per_tp + fp * loss_per_fp) / tp_fp_sum


import numpy as np


def calculate_ratio_difference(ratio_train, ratio_val, config):
    """
    Calcule l'écart relatif entre les ratios train et validation
    avec gestion de la division par zéro et compatibilité GPU/CPU

    Args:
        ratio_train: Ratio sur l'ensemble d'entraînement (float)
        ratio_val: Ratio sur l'ensemble de validation (float)
        config: Dictionnaire de configuration contenant la clé 'device_'

    Returns:
        float: Écart relatif en pourcentage
    """
    # Sélection de la bibliothèque appropriée (CuPy ou NumPy)
    xp = cp if config['device_'] == 'cuda' else np

    # Conversion des valeurs en type compatible avec le device
    ratio_train = xp.array(ratio_train, dtype=xp.float32)
    ratio_val = xp.array(ratio_val, dtype=xp.float32)

    # Gestion du cas où ratio_train est 0
    if xp.equal(ratio_train, 0):
        return 0.0 if xp.equal(ratio_val, 0) else 100.0

    # Calcul de l'écart relatif
    result = ((ratio_train - ratio_val) / ratio_train) * 100

    # Arrondir à 2 décimales
    return float(xp.round(result, decimals=2))

def format_constraint_message(is_violated, config, trial, constraint_name, config_key, attribute_key, check_type):
    """
    Formate un message de contrainte de manière uniforme avec les valeurs minimales ou maximales.

    Args:
        is_violated (bool): Indique si la contrainte est violée.
        config (dict): Dictionnaire de configuration contenant les seuils.
        trial (optuna.Trial): Essai Optuna en cours.
        constraint_name (str): Nom de la contrainte pour l'affichage.
        config_key (str): Clé pour accéder au seuil dans config.
        attribute_key (str): Clé pour accéder aux valeurs dans trial.user_attrs.
        check_type (str): Type de comparaison, 'max' ou 'min'.
    """
    # Récupération des valeurs avec gestion des cas vides
    values = trial.user_attrs.get(attribute_key, [])
    threshold = config.get(config_key, 0)

    # Détermine la valeur à afficher (min ou max selon check_type)
    if check_type == 'max':
        value_to_compare = max(values) if values else 0
        comparison = ">" if is_violated else "<="
    elif check_type == 'min':
        value_to_compare = min(values) if values else 0
        comparison = "<" if is_violated else ">="
    else:
        raise ValueError("check_type must be 'max' or 'min'.")

    # Détermine le style du message
    color = Fore.RED if is_violated else Fore.GREEN
    symbol = "-" if is_violated else "\u2713"

    # Formate les valeurs de la liste pour l'affichage
    formatted_values = [f"{x:.2f}" for x in values] if values else []

    # Construction du message
    if is_violated:
        return (
            color +
            f"    {symbol} {constraint_name} {comparison} {threshold} " +
            f"({check_type}: {value_to_compare:.2f}, values: {formatted_values})" +
            Style.RESET_ALL
        )
    else:
        return (
            color +
            f"    {symbol} {constraint_name} {comparison} {threshold}" +
            Style.RESET_ALL
        )

from sklearn.linear_model import LinearRegression

def linear_regression_slope_market_trend(series):
    X = np.arange(len(series)).reshape(-1, 1)
    y = series.values.reshape(-1, 1)
    model = LinearRegression().fit(X, y)
    slope = model.coef_[0][0]
    return slope

def calculate_trend_strength(close=None, high=None, low=None):
    """
    Calcule un score de force de tendance entre 0 et 100
    en combinant plusieurs métriques
    """
    try:
        # 1. Régression linéaire pour la direction et la régularité
        X = np.arange(len(close)).reshape(-1, 1)
        y = close.values.reshape(-1, 1)
        reg = LinearRegression().fit(X, y)

        # Force de la pente (normalisée)
        slope = reg.coef_[0][0]
        r2 = reg.score(X, y)  # R² pour mesurer la linéarité
        # 3. Analyse améliorée des contre-tendances avec high/low
        if slope > 0:  # Tendance haussière
            # Un counter move significatif est quand :
            # - Le low descend sous le low précédent
            # - ET/OU le high n'arrive pas à dépasser le high précédent
            low_breaks = (low+1.5 < low.shift(1)).sum()
            high_fails = (high < high.shift(1)).sum()
            counter_moves = (low_breaks ) / ( len(close))
        else:  # Tendance baissière
            # Un counter move significatif est quand :
            # - Le high monte au-dessus du high précédent
            # - ET/OU le low n'arrive pas à descendre sous le low précédent
            high_breaks = (high-1.5 > high.shift(1)).sum()
            low_fails = (low > low.shift(1)).sum()
            counter_moves = (high_breaks ) / ( len(close))

        # On pourrait aussi ajouter l'amplitude
        if slope > 0:
            counter_amplitude = abs((low - low.shift(1)) / low.shift(1))[low < low.shift(1)].mean()
        else:
            counter_amplitude = abs((high - high.shift(1)) / high.shift(1))[high > high.shift(1)].mean()




        """
        # 2. Calcul des drawdowns pour évaluer les pullbacks
        price_series = pd.Series(series)
        rolling_max = price_series.expanding().max()
        drawdowns = (price_series - rolling_max) / rolling_max * 100
        max_drawdown = abs(drawdowns.min())
    
        
    
        # 4. Calcul du score composite
        trend_score = (
                abs(slope) * 40 +  # Force de la direction (40% du score)
                r2 * 30 +  # Régularité (30% du score)
                (1 - counter_moves) * 20 +  # Continuité (20% du score)
                (1 - max_drawdown / 100) * 10  # Résistance aux pullbacks (10% du score)
        )
    
        return {
            'trend_score': trend_score,
            'slope': slope,
            'r2': r2,
            'max_drawdown': max_drawdown,
            'counter_moves_ratio': counter_moves
        }
        """



        return slope,r2,counter_moves
    except Exception as e:
        print(f"Error in calculate_trend_strength: {str(e)}")
        raise e  # On relance l'exception capturée


def apply_slope_with_session_check(data, window):
    result = pd.Series(index=data.index, dtype=float)

    for idx in result.index:
        historical_data = data.loc[:idx]
        last_session_start = historical_data[::-1]['SessionStartEnd'].eq(10).idxmax()
        bars_since_session_start = len(historical_data.loc[last_session_start:idx])

        if bars_since_session_start >= window:
            series = data.loc[:idx, 'close'].tail(window)
            result[idx] = linear_regression_slope_market_trend(series)
        else:
            result[idx] = np.nan

    return result


def compute_sample_weights(Y_train, y_val):
    """Calculer les poids pour l'entraînement et la validation."""
    N0 = np.sum(Y_train == 0)
    N1 = np.sum(Y_train == 1)
    N = len(Y_train)
    w_0 = N / N0
    w_1 = N / N1
    sample_weights_train = np.where(Y_train == 1, w_1, w_0)
    sample_weights_val = np.ones(len(y_val))
    return sample_weights_train, sample_weights_val


def get_best_iteration(evals_result, custom_metric_eval):
    """
    Extraction de la meilleure itération en fonction du custom metric.
    Retourne : (best_iteration, best_idx, val_best, train_best)
    """
    if custom_metric_eval in (model_custom_metric.XGB_CUSTOM_METRIC_PNL,
                              model_custom_metric.LGB_CUSTOM_METRIC_PNL):
        eval_scores = evals_result['eval']['custom_metric_PNL']
        train_scores = evals_result['train']['custom_metric_PNL']
    else:
        eval_scores = evals_result['eval']['auc']
        train_scores = evals_result['train']['auc']
        raise ValueError("Fonction metric non reconnue, impossible de continuer.")

    val_best = max(eval_scores)
    best_idx = eval_scores.index(val_best)
    best_iteration = best_idx + 1
    train_best = train_scores[best_idx]
    return best_iteration, best_idx, val_best, train_best


def compile_fold_stats(fold_num, best_iteration, train_pred_proba_log_odds, train_metrics, winrate_train,
                       tp_fp_sum_train, tp_fp_tn_fn_sum_train, train_best, train_pos, train_trades_samples_perct,
                       val_pred_proba_log_odds, val_metrics, winrate_val, tp_fp_sum_val, tp_fp_tn_fn_sum_val,
                       val_best, val_pos, val_trades_samples_perct, ratio_difference,
                       perctDiff_ratioTradeSample_train_val):
    """Compilation des statistiques du fold dans un dictionnaire commun."""
    return {
        'fold_num': fold_num,
        'best_iteration': best_iteration,
        'train_pred_proba_log_odds': train_pred_proba_log_odds,
        'train_metrics': train_metrics,
        'train_winrate': winrate_train,
        'train_trades': tp_fp_sum_train,
        'train_samples': tp_fp_tn_fn_sum_train,
        'train_bestVal_custom_metric_pnl': train_best,
        'train_size': len(train_pos) if train_pos is not None else None,
        'train_trades_samples_perct': train_trades_samples_perct,
        'val_pred_proba_log_odds': val_pred_proba_log_odds,
        'val_metrics': val_metrics,
        'val_winrate': winrate_val,
        'val_trades': tp_fp_sum_val,
        'val_samples': tp_fp_tn_fn_sum_val,
        'val_bestVal_custom_metric_pnl': val_best,
        'val_size': len(val_pos) if val_pos is not None else None,
        'val_trades_samples_perct': val_trades_samples_perct,
        'perctDiff_winrateRatio_train_val': ratio_difference,
        'perctDiff_ratioTradeSample_train_val': perctDiff_ratioTradeSample_train_val
    }


def compile_debug_info(model_weight_optuna, config, val_pred_proba, y_train_predProba):
    """Construction du dictionnaire de debug."""
    xp = np if config['device_'] != 'cuda' else cp
    return {
        'threshold_used': model_weight_optuna['threshold'],
        'pred_proba_ranges': {
            'val': {
                'min': float(xp.min(val_pred_proba)),
                'max': float(xp.max(val_pred_proba))
            },
            'train': {
                'min': float(xp.min(y_train_predProba)),
                'max': float(xp.max(y_train_predProba))
            }
        }
    }


def are_scalers_partially_identical(scaler1, scaler2, columns1, columns2):
    """
    Vérifie si deux objets scaler sont identiques pour les colonnes communes

    Args:
        scaler1: Premier scaler à comparer
        scaler2: Second scaler à comparer
        columns1: Liste des noms de colonnes pour scaler1
        columns2: Liste des noms de colonnes pour scaler2

    Returns:
        bool: True si les paramètres sont identiques pour les colonnes communes
    """
    # Vérifie d'abord qu'ils sont du même type
    if type(scaler1) != type(scaler2):
        print("Les scalers sont de types différents")
        return False

    # Trouver les colonnes communes
    common_columns = set(columns1) & set(columns2)
    if not common_columns:
        print("Aucune colonne commune entre les deux scalers")
        return False

    print(f"Nombre de colonnes communes: {len(common_columns)}")

    # Obtenir les indices des colonnes communes
    indices1 = [columns1.index(col) for col in common_columns]
    indices2 = [columns2.index(col) for col in common_columns]

    # Comparaison des attributs selon le type de scaler
    if hasattr(scaler1, 'mean_'):
        # Pour StandardScaler
        means_equal = np.allclose(scaler1.mean_[indices1], scaler2.mean_[indices2])
        scales_equal = np.allclose(scaler1.scale_[indices1], scaler2.scale_[indices2])
        print(f"Moyennes identiques pour colonnes communes: {means_equal}")
        print(f"Écarts-types identiques pour colonnes communes: {scales_equal}")
        return means_equal and scales_equal

    elif hasattr(scaler1, 'center_'):
        # Pour RobustScaler
        centers_equal = np.allclose(scaler1.center_[indices1], scaler2.center_[indices2])
        scales_equal = np.allclose(scaler1.scale_[indices1], scaler2.scale_[indices2])
        print(f"Médianes identiques pour colonnes communes: {centers_equal}")
        print(f"IQR identiques pour colonnes communes: {scales_equal}")
        return centers_equal and scales_equal

    elif hasattr(scaler1, 'min_'):
        # Pour MinMaxScaler
        mins_equal = np.allclose(scaler1.min_[indices1], scaler2.min_[indices2])
        scales_equal = np.allclose(scaler1.scale_[indices1], scaler2.scale_[indices2])
        print(f"Minimums identiques pour colonnes communes: {mins_equal}")
        print(f"Échelles identiques pour colonnes communes: {scales_equal}")
        return mins_equal and scales_equal

    else:
        # Pour les autres types de scalers
        print("Type de scaler non géré par cette fonction")
        return False







def are_pca_identical(pca1, pca2):
    """Vérifie si deux objets PCA sont identiques"""
    # Vérifier que le nombre de composantes est le même
    if pca1.n_components != pca2.n_components:
        print("Nombre de composantes différent")
        return False

    # Vérifier que les moyennes sont identiques
    means_equal = np.allclose(pca1.mean_, pca2.mean_)

    # Pour les composantes, attention car le signe peut être inversé
    # On compare les valeurs absolues des produits scalaires
    components_similar = True
    for i in range(pca1.n_components):
        # Calculer le produit scalaire normalisé (cosinus de l'angle entre les vecteurs)
        dot_product = np.abs(np.dot(pca1.components_[i], pca2.components_[i]) /
                             (np.linalg.norm(pca1.components_[i]) * np.linalg.norm(pca2.components_[i])))
        if dot_product < 0.99:  # Si l'angle est significatif (cosinus < 0.99)
            components_similar = False
            print(f"Composante {i + 1} différente (cosinus: {dot_product})")

    # Vérifier que les variances expliquées sont similaires
    variance_equal = np.allclose(pca1.explained_variance_ratio_, pca2.explained_variance_ratio_)

    print(f"Moyennes identiques: {means_equal}")
    print(f"Composantes similaires: {components_similar}")
    print(f"Variances expliquées identiques: {variance_equal}")

    return means_equal and components_similar and variance_equal


import pandas as pd
import numpy as np

import pandas as pd
import numpy as np


def compare_dataframes(df1, df2, name1, name2, tolerance=1e-6):
    # Comparaison des noms de colonnes avec plus de détails
    print("\nDiagnostic détaillé des colonnes:")

    # Convertir les noms de colonnes en listes pour préserver l'ordre
    cols1 = list(df1.columns)
    cols2 = list(df2.columns)

    # Afficher les colonnes avec leur représentation brute
    print(f"\nColonnes de {name1} ({len(cols1)}):")
    for i, col in enumerate(cols1):
        print(f"  {i}: '{col}' (type: {type(col)}, repr: {repr(col)})")

    print(f"\nColonnes de {name2} ({len(cols2)}):")
    for i, col in enumerate(cols2):
        print(f"  {i}: '{col}' (type: {type(col)}, repr: {repr(col)})")

    # Vérifier les colonnes communes et comparer leurs valeurs
    common_cols = set([str(c).strip() for c in cols1]) & set([str(c).strip() for c in cols2])
    print(f"\nNombre de colonnes communes (ignorant casse/espaces): {len(common_cols)}")

    # Comparer les valeurs des colonnes qui semblent identiques
    print("\nComparaison des valeurs pour les colonnes qui semblent correspondre:")

    for col1 in cols1:
        # Chercher une colonne équivalente dans df2
        col1_norm = str(col1).strip().lower()
        for col2 in cols2:
            col2_norm = str(col2).strip().lower()
            if col1_norm == col2_norm:
                print(f"\nComparaison: '{col1}' et '{col2}'")

                # Vérifier si les données sont identiques
                if df1[col1].equals(df2[col2]):
                    print("  ✅ Valeurs identiques")
                else:
                    print("  ❌ Valeurs différentes")

                    # Si numériques, calculer les différences
                    if pd.api.types.is_numeric_dtype(df1[col1]) and pd.api.types.is_numeric_dtype(df2[col2]):
                        diff = np.abs(df1[col1] - df2[col2])
                        max_diff = diff.max()
                        print(f"  Différence max: {max_diff}")

                        # Afficher quelques exemples de différences
                        if max_diff > 0:
                            diff_indices = np.where(diff > 0)[0][:3]  # Prendre jusqu'à 3 exemples
                            print("  Exemples de différences:")
                            for idx in diff_indices:
                                print(f"    Index {idx}: {df1[col1].iloc[idx]} vs {df2[col2].iloc[idx]}")
                break


import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit


class BetaCalibrationCustom:
    """
    Implémentation simplifiée de Beta Calibration (Kull et al. 2017).
    Calibre des probabilités p -> f(p).

    On modélise: f(p) = (alpha_1 * p^(beta_1 - 1) * (1 - p)^(beta_2 - 1)) /
                       (alpha_1 * p^(beta_1 - 1) * (1 - p)^(beta_2 - 1) +
                        alpha_2 * p^(beta_3 - 1) * (1 - p)^(beta_4 - 1))

    Hypothèse: alpha_1 > 0, alpha_2 > 0, etc.
    """

    def __init__(self, method="ab"):
        """
        Choisir la forme de Beta calibration:
        - "ab": full parameters
        - "abm": mid-level (fewer parameters)
        - "am": alternative version
        """
        self.method = method
        self.params_ = None

    def _negative_log_likelihood(self, params, p, y):
        # Selon la méthode, on associe les params
        if self.method == "ab":  # 4 paramètres
            a, b, c, d = params
        elif self.method == "abm":  # 3 paramètres
            a, b, c = params
            d = 1.0
        elif self.method == "am":  # 2 paramètres
            a, b = params
            c, d = 1.0, 1.0
        else:
            raise ValueError("Unknown method")

        # f(p) = numerator / (numerator + denominator)
        # numerator = p^(a) * (1-p)^(b)
        # denominator = p^(c) * (1-p)^(d)
        # => on ajoute un petit epsilon pour éviter log(0)
        eps = 1e-12
        num = np.power(p, a) * np.power((1 - p), b) + eps
        den = np.power(p, c) * np.power((1 - p), d) + eps
        f = num / (num + den)

        # log-likelihood
        ll = y * np.log(f + eps) + (1 - y) * np.log(1 - f + eps)
        return -np.sum(ll)

    def fit(self, p, y):
        """
        p: probabilités non calibrées
        y: labels (0 ou 1)
        """
        p = np.clip(p, 1e-8, 1 - 1e-8)
        y = np.array(y, dtype=float)

        # Init des params
        if self.method == "ab":
            init_params = [1.0, 1.0, 1.0, 1.0]  # a, b, c, d
        elif self.method == "abm":
            init_params = [1.0, 1.0, 1.0]  # a, b, c
        elif self.method == "am":
            init_params = [1.0, 1.0]  # a, b
        else:
            raise ValueError("Unknown method")

        # On minimise la -log-likelihood
        res = minimize(self._negative_log_likelihood, init_params, args=(p, y),
                       method='L-BFGS-B', bounds=None)

        self.params_ = res.x
        return self

    def predict(self, p):
        """
        Calibre les probabilités p en f(p).
        """
        if self.params_ is None:
            raise RuntimeError("BetaCalibrationCustom is not fitted!")

        p = np.clip(p, 1e-8, 1 - 1e-8)

        # Récup params
        if self.method == "ab":
            a, b, c, d = self.params_
        elif self.method == "abm":
            a, b, c = self.params_
            d = 1.0
        elif self.method == "am":
            a, b = self.params_
            c, d = 1.0, 1.0
        else:
            raise ValueError("Unknown method")

        # f(p)
        eps = 1e-12
        num = np.power(p, a) * np.power(1 - p, b) + eps
        den = np.power(p, c) * np.power(1 - p, d) + eps
        return num / (num + den)


import numpy as np

import numpy as np

def log_stats(name, data, bins=30):
    """
    Affiche min, max, moyenne, std + répartition sur 'bins' intervalles.
    """
    data = np.array(data)
    print(f"\n📊 Statistiques pour {name}:")
    print(f"  Min       : {np.min(data):.6f}")
    print(f"  Max       : {np.max(data):.6f}")
    print(f"  Moyenne   : {np.mean(data):.6f}")
    print(f"  Écart-type: {np.std(data):.6f}")

    counts, edges = np.histogram(data, bins=bins)
    for i in range(len(counts)):
        print(f"  Bin {i+1} ({edges[i]:.3f} - {edges[i+1]:.3f}) : {counts[i]}")
import numpy as np
import xgboost as xgb
import lightgbm as lgb

class BoosterWrapper:
    def __init__(self, booster, model_type=None):
        self.booster = booster

        if model_type is not None:
            self.model_type = model_type
        elif isinstance(booster, lgb.Booster) or "lightgbm.sklearn" in str(type(booster)).lower():
            self.model_type = 'lgbm'
        elif isinstance(booster, xgb.Booster):
            self.model_type = 'xgb'
        else:
            raise ValueError(f"Impossible de détecter le type du booster : {type(booster)}")

    def fit(self, X, y):
        # Stub : on considère le booster déjà entraîné
        return self

    def predict(self, X, output_margin=False):
        if self.model_type == 'lgbm':
            if "DMatrix" in str(type(X)):
                raise TypeError("LightGBM cannot predict on an XGBoost DMatrix.")
            num_iteration = getattr(self.booster, "best_iteration", None)
            if output_margin:
                return self.booster.predict(X, raw_score=True, num_iteration=num_iteration)
            else:
                probas = self.booster.predict(X, num_iteration=num_iteration)
                return np.round(probas).astype(int)

        else:
            if not isinstance(X, xgb.DMatrix):
                X = xgb.DMatrix(X)
            ntree_limit = getattr(self.booster, "best_iteration", None)
            if ntree_limit is not None:
                ntree_limit += 1
            if output_margin:
                return self.booster.predict(X, output_margin=True, ntree_limit=ntree_limit)
            else:
                probas = self.booster.predict(X, ntree_limit=ntree_limit)
                return np.round(probas).astype(int)

    def predict_proba(self, X):
        if self.model_type == 'lgbm':
            if "DMatrix" in str(type(X)):
                raise TypeError("LightGBM cannot predict on an XGBoost DMatrix.")
            num_iteration = getattr(self.booster, "best_iteration", None)
            raw_preds = self.booster.predict(X, num_iteration=num_iteration)
        else:
            if not isinstance(X, xgb.DMatrix):
                X = xgb.DMatrix(X)
            ntree_limit = getattr(self.booster, "best_iteration", None)
            if ntree_limit is not None:
                ntree_limit += 1
            raw_preds = self.booster.predict(X, ntree_limit=ntree_limit)

        if len(raw_preds.shape) == 1:
            probs = np.zeros((len(raw_preds), 2))
            probs[:, 1] = raw_preds
            probs[:, 0] = 1 - raw_preds
            return probs
        return raw_preds

    def get_params(self, deep=True):
        return {"booster": self.booster, "model_type": self.model_type}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


def plot_train_eval_performance(fold_results, save_dir=None, display=True, mode="trainAndVal"):
    """
    Visualise les résultats d'entraînement et de validation d'un modèle LightGBM
    avec mise en évidence de la meilleure itération.

    Args:
        fold_results (dict): Dictionnaire contenant les résultats d'entraînement et de validation
                            avec une clé 'evals_result' et éventuellement 'best_iteration'
        save_dir (str, optional): Répertoire où sauvegarder la figure. Si None, la figure n'est pas sauvegardée.
        display (bool, optional): Si True, affiche la figure en plus de l'enregistrer. Par défaut True.
        mode (str, optional): Mode d'affichage: "train_only" pour afficher uniquement les données d'entrainement,
                             "trainAndVal" pour afficher les données d'entrainement et de validation. Par défaut "trainAndVal".

    Returns:
        matplotlib.figure.Figure: L'objet figure créé
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    # Récupération des résultats d'évaluation
    evals_result = fold_results["evals_result"]

    # Déterminer les métriques disponibles
    train_metrics = list(evals_result['train'].keys())
    eval_metrics = list(evals_result['eval'].keys()) if 'eval' in evals_result else []

    # Récupération de la meilleure itération si disponible
    best_iteration = fold_results.get("best_iteration", None)
    if best_iteration is None and 'best_iteration' in fold_results:
        best_iteration = fold_results['best_iteration']

    # Si toujours pas disponible, essayer de la déterminer à partir de la métrique principale
    if best_iteration is None and len(eval_metrics) > 0:
        primary_metric = eval_metrics[0]  # Supposer que la première métrique est la principale
        eval_values = evals_result['eval'][primary_metric]
        # Pour les métriques où plus grand est meilleur (comme PnL)
        best_iteration = np.argmax(eval_values) + 1  # +1 car les itérations commencent à 1

    # Création de la figure en fonction du mode
    if mode == "train_only":
        fig, ax_train = plt.subplots(figsize=(12, 8))
        axs = [ax_train]
    else:  # mode == "trainAndVal"
        fig, axs = plt.subplots(1, 2, figsize=(20, 8))
        ax_train = axs[0]

    # Couleurs pour les différentes métriques
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']

    # Plot des données d'entraînement
    for i, metric in enumerate(train_metrics):
        values = evals_result['train'][metric]
        iterations = range(1, len(values) + 1)
        ax_train.plot(iterations, values, label=f'{metric}', color=colors[i % len(colors)])

        # Marquer la meilleure itération
        if best_iteration and best_iteration <= len(values):
            best_value = values[best_iteration - 1]
            ax_train.scatter(best_iteration, best_value, color=colors[i % len(colors)], s=100, marker='o',
                             edgecolor='black', zorder=10)
            ax_train.axvline(x=best_iteration, color='gray', linestyle='--', alpha=0.5)
            ax_train.text(best_iteration, best_value, f'  Itération {best_iteration}\n  Valeur: {best_value:.4f}',
                          verticalalignment='center')

    ax_train.set_title('Performance sur les données d\'entraînement', fontsize=14)
    ax_train.set_xlabel('Itérations', fontsize=12)
    ax_train.set_ylabel('Valeur métrique', fontsize=12)
    ax_train.grid(True, alpha=0.3)
    ax_train.legend(loc='best')

    # Plot des données de validation (seulement si mode trainAndVal)
    if mode == "trainAndVal" and eval_metrics:
        ax_eval = axs[1]
        for i, metric in enumerate(eval_metrics):
            values = evals_result['eval'][metric]
            iterations = range(1, len(values) + 1)
            ax_eval.plot(iterations, values, label=f'{metric}', color=colors[i % len(colors)])

            # Marquer la meilleure itération
            if best_iteration and best_iteration <= len(values):
                best_value = values[best_iteration - 1]
                ax_eval.scatter(best_iteration, best_value, color=colors[i % len(colors)], s=100, marker='o',
                                edgecolor='black', zorder=10)
                ax_eval.axvline(x=best_iteration, color='gray', linestyle='--', alpha=0.5)
                ax_eval.text(best_iteration, best_value, f'  Itération {best_iteration}\n  Valeur: {best_value:.4f}',
                             verticalalignment='center')

        ax_eval.set_title('Performance sur les données de validation', fontsize=14)
        ax_eval.set_xlabel('Itérations', fontsize=12)
        ax_eval.set_ylabel('Valeur métrique', fontsize=12)
        ax_eval.grid(True, alpha=0.3)
        ax_eval.legend(loc='best')

    # Ajout d'un titre global
    if mode == "train_only":
        titre_base = "Évolution des métriques"
    else:
        titre_base = "Évolution des métriques d'entraînement et de validation"

    if best_iteration:
        plt.suptitle(f'{titre_base}\nMeilleure itération: {best_iteration}', fontsize=16)
    else:
        plt.suptitle(titre_base, fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Ajustement pour le titre global

    # Sauvegarde de la figure si un répertoire est spécifié
    if save_dir:
        # Création du répertoire s'il n'existe pas
        os.makedirs(save_dir, exist_ok=True)

        # Création du nom de fichier
        mode_suffix = "_train_only" if mode == "train_only" else ""
        iter_suffix = f"_iter_{best_iteration}" if best_iteration else ""
        filename = f"model_performance{mode_suffix}{iter_suffix}.png"
        filepath = os.path.join(save_dir, filename)

        # Sauvegarde de la figure
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"\n📥 Figure sauvegardée : {filepath}")

    # Affichage de la figure selon le paramètre display
    if display:
        plt.show()

    # Afficher un résumé des performances à la meilleure itération
    if best_iteration:
        print(f"\n📊 Performances à la meilleure itération ({best_iteration}):")
        print("-" * 50)
        print("Métriques d'entraînement:")
        for metric in train_metrics:
            if best_iteration <= len(evals_result['train'][metric]):
                value = evals_result['train'][metric][best_iteration - 1]
                print(f"  - {metric}: {value:.6f}")

        if mode == "trainAndVal" and eval_metrics:
            print("\nMétriques de validation:")
            for metric in eval_metrics:
                if best_iteration <= len(evals_result['eval'][metric]):
                    value = evals_result['eval'][metric][best_iteration - 1]
                    print(f"  - {metric}: {value:.6f}")

        # Sauvegarde des métriques dans un fichier texte si un répertoire est spécifié
        if save_dir:
            mode_suffix = "_train_only" if mode == "train_only" else ""
            metrics_filename = f"model_metrics{mode_suffix}_iter_{best_iteration}.txt"
            metrics_filepath = os.path.join(save_dir, metrics_filename)

            with open(metrics_filepath, 'w') as f:
                f.write(f"Performances à la meilleure itération ({best_iteration}):\n")
                f.write("-" * 50 + "\n")
                f.write("Métriques d'entraînement:\n")
                for metric in train_metrics:
                    if best_iteration <= len(evals_result['train'][metric]):
                        value = evals_result['train'][metric][best_iteration - 1]
                        f.write(f"  - {metric}: {value:.6f}\n")

                if mode == "trainAndVal" and eval_metrics:
                    f.write("\nMétriques de validation:\n")
                    for metric in eval_metrics:
                        if best_iteration <= len(evals_result['eval'][metric]):
                            value = evals_result['eval'][metric][best_iteration - 1]
                            f.write(f"  - {metric}: {value:.6f}\n")

            print(f"📝 Métriques sauvegardées : {metrics_filepath}")

    return fig

import numpy as np
def sweep_threshold_pnl_real(
    model,
    X,
    y_true_class_binaire,
    y_pnl_data,
    config,
    val_score_bestIdx,
    other_params,
    thresholds=np.linspace(0.45, 0.60, num=31)
):
    """
    Balaye plusieurs seuils et calcule le profit via calculate_profitBased à chaque seuil.

    Retourne pour chaque seuil : TP, FP, FN, TN, TP_gains, FP_losses, FN_missed, total_profit
    """


    model_type = config['model_type']
    if model_type == modelType.XGB:
        import xgboost as xgb
        dData = xgb.DMatrix(X)
    else:
        dData = X

    # Obtenir les probabilités de classe 1 avec le pipeline custom
    y_pred_proba, _, _, _, _ = predict_and_compute_metrics_XgbOrLightGbm(
        model=model,
        X_data=dData,
        y_true=y_true_class_binaire,
        best_iteration=val_score_bestIdx + 1,
        threshold=other_params['threshold'],
        config=config
    )

    # Initialisation
    y_true_class_binaire = np.array(y_true_class_binaire)
    y_pnl_array = np.array(y_pnl_data)
    probas = np.array(y_pred_proba)

    results = []
    for threshold in thresholds:
        y_pred = (probas >= threshold).astype(int)

        # Masques pour FP, TP, FN, TN (toujours utiles pour info)
        tp_mask = (y_true_class_binaire == 1) & (y_pred == 1)
        fp_mask = (y_true_class_binaire == 0) & (y_pred == 1)
        fn_mask = (y_true_class_binaire == 1) & (y_pred == 0)
        tn_mask = (y_true_class_binaire == 0) & (y_pred == 0)

        # Calcul via ta fonction métier calculate_profitBased


        total_profit, tp, fp = calculate_profitBased(
            y_true_class_binaire=y_true_class_binaire,
            y_pred_threshold=y_pred,
            y_pnl_data_array=y_pnl_array,
            other_params=other_params,
            config=config
        )

        # Statistiques complémentaires
        fn = np.sum(fn_mask)
        tn = np.sum(tn_mask)
        tp_gains = np.sum(y_pnl_array[tp_mask])
        fp_losses = np.sum(y_pnl_array[fp_mask])
        fn_missed = np.sum(y_pnl_array[fn_mask])

        results.append({
            "threshold": round(threshold, 5),
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "TN": int(tn),
            "TP_gains": float(tp_gains),
            "FP_losses": float(fp_losses),
            "FN_missed": float(fn_missed),
            "pnl_total": float(total_profit)  # basé sur ta fonction métier
        })

    #results.sort(key=lambda x: x["pnl_total"], reverse=True)
    return results


# ----------------------------------------------
# 0.  Imports
# ----------------------------------------------
import os
import numpy as np
import pandas as pd
import shap
from typing import List, Sequence, Tuple, Optional

# ----------------------------------------------
# 1.  SHAP par fold
# ----------------------------------------------
def get_fold_shap_mean_abs(model, X_fold: pd.DataFrame) -> pd.Series:
    """
    Renvoie la moyenne absolue des valeurs SHAP par feature pour un fold donné.
    Gère classification binaire (liste shap_vals[0]/[1]) et régression.
    """
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_fold)

    if isinstance(shap_vals, Sequence):  # classification
        # on prend la classe "positive" (index 1) s'il y a 2 matrices
        shap_vals = shap_vals[1] if len(shap_vals) == 2 else shap_vals[0]

    return pd.Series(
        np.abs(shap_vals).mean(axis=0),
        index=X_fold.columns, name="mean_abs_shap"
    )

# ----------------------------------------------
# 2.  Table d’importance SHAP fold × feature
# ----------------------------------------------
def build_shap_table(
    models_by_fold: List, X_val_by_fold: List[pd.DataFrame]
) -> pd.DataFrame:
    """
    Concatène les importances SHAP par fold → DataFrame (features × folds).
    """
    shap_cols = []
    for i, (m, Xv) in enumerate(zip(models_by_fold, X_val_by_fold)):
        s = get_fold_shap_mean_abs(m, Xv)
        s.name = f"fold_{i}"
        shap_cols.append(s)

    return pd.concat(shap_cols, axis=1).fillna(0)

# ----------------------------------------------
# 3.  Calcul des métriques de stabilité
#     (CV, rang moyen, freq_top_N, Levene p‑value optionnelle)
# ----------------------------------------------
def add_stability_metrics(
    df: pd.DataFrame,
    top_n: int = 20,
    add_levene: bool = False
) -> pd.DataFrame:
    """
    Ajoute colonnes :
      - mean_SHAP, std_SHAP, cv_SHAP
      - mean_rank (rang moyen sur chaque fold, plus petit = meilleur)
      - freq_topN  (nb de fois où la feature est dans le top N de son fold)
      - levene_p   (homogénéité de variance entre folds, facultatif)
    """
    metric_df = df.copy()

    # statistiques de base
    metric_df["mean_SHAP"] = metric_df.mean(axis=1)
    metric_df["std_SHAP"] = metric_df.std(axis=1)
    metric_df["cv_SHAP"] = metric_df["std_SHAP"] / (metric_df["mean_SHAP"] + 1e-12)

    # stabilité de rang
    ranks = df.rank(axis=0, ascending=False)            # rang 1 = plus important
    metric_df["mean_rank"] = ranks.mean(axis=1)

    # fréquence d’apparition dans le top N
    metric_df["freq_topN"] = (ranks <= top_n).sum(axis=1)

    # homogénéité de variance (option)
    if add_levene:
        from scipy.stats import levene
        levene_vals = []
        for feat in df.index:
            stat, pval = levene(*[df.loc[feat, c].values
                                  for c in df.columns if c.startswith("fold_")])
            levene_vals.append(pval)
        metric_df["levene_p"] = levene_vals

    return metric_df

# ----------------------------------------------
# 4.  Sélection finale des features
# ----------------------------------------------
def select_features(
    metrics_df: pd.DataFrame,
    *,
    min_mean: float = 0.0,
    max_cv: float = 0.30,
    max_mean_rank: Optional[float] = None,
    min_freq_topN: Optional[int] = None,
    levene_alpha: Optional[float] = None,
    top_k: Optional[int] = None
) -> Tuple[list, pd.DataFrame]:
    """
    Filtre les features selon plusieurs critères.
    Retourne (liste_features_gardées, sous‑DataFrame filtré).
    """
    keep = (metrics_df["mean_SHAP"] >= min_mean) & (metrics_df["cv_SHAP"] <= max_cv)

    if max_mean_rank is not None:
        keep &= metrics_df["mean_rank"] <= max_mean_rank
    if min_freq_topN is not None:
        keep &= metrics_df["freq_topN"] >= min_freq_topN
    if (levene_alpha is not None) and ("levene_p" in metrics_df.columns):
        keep &= metrics_df["levene_p"] >= levene_alpha

    filtered = metrics_df[keep].sort_values(["cv_SHAP", "mean_SHAP"])

    if top_k:
        filtered = filtered.head(top_k)

    return filtered.index.tolist(), filtered

# ----------------------------------------------
# 5.  Export CSV séparateur « ; »
# ----------------------------------------------
def export_stability_csv(
    metrics_df: pd.DataFrame,
    save_dir: str,
    filename: str = "shap_stability_byfold_report.csv"
):
    """
    Sauvegarde le tableau complet avec ; comme séparateur et index=feature.
    """
    path = os.path.join(save_dir, filename)
    metrics_df.to_csv(path, sep=";")
    print(f"[+] Fichier sauvegardé → {path}")


# streamlit run C:\Users\aulac\OneDrive\Documents\Trading\PyCharmProject\MLStrategy\stats_sc\main_shap_byFold.py


# =============================================================================
# 📘 Mode d’emploi : Filtres SHAP dans l’interface Streamlit
# =============================================================================
#
# 🎯 Objectif : Sélectionner les features SHAP les plus importantes et stables
#
# -----------------------------------------------------------------------------
# 1. mean_SHAP : importance moyenne (float)
# -----------------------------------------------------------------------------
# ➤ Définition :
#     Moyenne des valeurs SHAP absolues sur tous les folds
#     mean_SHAP = moyenne(|SHAP| sur tous les folds)
#
# ➤ Filtrage :
#     Seuil minimal ajustable avec le slider Streamlit
#
# -----------------------------------------------------------------------------
# 2. cv_SHAP : instabilité relative (float)
# -----------------------------------------------------------------------------
# ➤ Définition :
#     Coefficient de variation = std_SHAP / mean_SHAP
#     (permet de rejeter les features instables d’un fold à l’autre)
#
# ➤ Filtrage :
#     Seuil maximal de cv_SHAP ajustable
#
# -----------------------------------------------------------------------------
# 3. Top-K : nombre de features conservées à la fin (int)
# -----------------------------------------------------------------------------
# ➤ Définition :
#     Une fois les filtres appliqués, on garde uniquement les Top-K
#     features selon leur mean_SHAP
#
# ➤ Filtrage :
#     Via un nombre limité dans le champ Top-K (0 = illimité)
#
# -----------------------------------------------------------------------------
# 4. freq_topN : fréquence d'apparition dans les meilleurs (int)
# -----------------------------------------------------------------------------
# ➤ Définition :
#     Nombre de folds dans lesquels une feature est classée
#     dans le Top-N des plus importantes
#
#     freq_topN = nb de fois où rank(feature) <= N
#
# ➤ Paramètre Top-N :
#     Défini par défaut à 20, ou passé via la fonction add_stability_metrics()
#
# ➤ Filtrage :
#     Optionnel : filtrer uniquement les features avec freq_topN >= seuil
#
# -----------------------------------------------------------------------------
# ✅ Exemple :
#     - 6 folds
#     - Top-N = 20
#     - freq_topN = 4  ➜ la feature est dans le top 20 dans 4 folds sur 6
#
# -----------------------------------------------------------------------------
# 🧠 Résumé des filtres utilisés :
# -----------------------------------------------------------------------------
#     - mean_SHAP >= seuil
#     - cv_SHAP <= seuil
#     - freq_topN >= min si activé
#     - top_k : limite de features finales
# =============================================================================
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

# ------------------------------------------------------------
# 1. Calibration  (Isotonic | Platt-sigmoid | none)
# ------------------------------------------------------------
def calibrate_probabilities(log_odds: np.ndarray,
                            y_true: np.ndarray,
                            method: str = "isotonic"):
    """
    Calibre les log-odds XGB et renvoie
    (proba_calibrées, objet_calibrator | None)
    """
    if method == "none":
        return 1.0 / (1.0 + np.exp(-log_odds)), None

    if method == "sigmoid":
        calibrator = LogisticRegression(solver="lbfgs")
    elif method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip")
    else:
        raise ValueError("method doit être 'none', 'sigmoid' ou 'isotonic'")

    calibrator.fit(log_odds.reshape(-1, 1), y_true)
    proba_cal = calibrator.predict(log_odds.reshape(-1, 1))
    return proba_cal, calibrator


# ------------------------------------------------------------
# 2. Sweep de seuil orienté PnL
# ------------------------------------------------------------
def sweep_threshold_pnl(y_true: np.ndarray,
                        proba: np.ndarray,
                        profit_tp: float = 175,
                        loss_fp: float = -227,
                        penalty_fn: float = 0.0,
                        n_steps: int = 200):
    lo, hi = proba.min(), proba.max()
    lo, hi = max(lo, .02), min(hi, .98)         # on évite les extrêmes
    thresholds = np.linspace(lo, hi, n_steps)

    pnl_curve = []
    for t in thresholds:
        pred = (proba >= t).astype(int)
        tp = np.sum((pred == 1) & (y_true == 1))
        fp = np.sum((pred == 1) & (y_true == 0))
        fn = np.sum((pred == 0) & (y_true == 1))
        pnl_curve.append(tp * profit_tp + fp * loss_fp + fn * penalty_fn)

    pnl_curve = np.asarray(pnl_curve)
    best_idx = pnl_curve.argmax()
    return thresholds[best_idx], thresholds, pnl_curve


# ------------------------------------------------------------
# 3. Agrégation des seuils par fold
# ------------------------------------------------------------
def aggregate_fold_thresholds(thresholds_list, mode: str = "median") -> float:
    return float(np.mean(thresholds_list) if mode == "mean"
                 else np.median(thresholds_list))


def evaluate_model(X_data, y_data, model, scaler=None, calibrator=None,
                   num_round=None, config=None, best_thresh_sweep=None,
                   final_threshold_median=None, dataset_name="",
                   profit_tp=175, loss_fp=-227, penalty_fn=0.0, n_steps=200):
    """
    Fonction générique pour évaluer un modèle sur des données (train ou test)

    Args:
        X_data: Features à évaluer
        y_data: Labels réels
        model: Modèle XGBoost entraîné
        scaler: Scaler pour normaliser les données
        calibrator: Calibrateur de probabilités (ou None)
        num_round: Nombre d'arbres à utiliser
        config: Configuration pour la matrice de confusion
        best_thresh_sweep: Seuil sweepé pré-calculé (optionnel)
        final_threshold_median: Seuil médian CV (optionnel)
        dataset_name: Nom du jeu de données ("TRAIN" ou "TEST")
        profit_tp: Profit par vrai positif
        loss_fp: Perte par faux positif
        penalty_fn: Pénalité par faux négatif
        n_steps: Nombre d'étapes pour le sweep

    Returns:
        Dict contenant les métriques d'évaluation
    """
    import numpy as np
    import pandas as pd
    import xgboost as xgb

    results = {}
    print("eval model for ",dataset_name)
    # 1) Scaling des données si nécessaire
    if scaler is not None:
        X_scaled = (pd.DataFrame(scaler.transform(X_data.values),
                                 columns=X_data.columns,
                                 index=X_data.index)
                    if isinstance(X_data, pd.DataFrame)
                    else scaler.transform(X_data))
    else:
        X_scaled = X_data

    # 2) Préparer la DMatrix
    dmatrix = xgb.DMatrix(X_scaled)
    # 3) Log-odds → proba (tous les arbres)
    log_odds = model.predict(
        dmatrix,
        iteration_range=(0, num_round),
        output_margin=True
    )

    proba = (calibrator.predict(log_odds.reshape(-1, 1)).ravel()
             if calibrator is not None
             else 1.0 / (1.0 + np.exp(-log_odds)))

    results['probabilities'] = proba

    # 4) Sweep PnL global si demandé
    if best_thresh_sweep is None:
        best_thresh_sweep, _, pnl_curve = sweep_threshold_pnl(
            y_data, proba,
            profit_tp=profit_tp, loss_fp=loss_fp,
            penalty_fn=penalty_fn, n_steps=n_steps
        )
        results['best_threshold_sweep'] = best_thresh_sweep
        results['pnl_curve'] = pnl_curve

        print(f"🔍 Meilleur seuil théorique sur {dataset_name} = {best_thresh_sweep:.4f} "
              f"| Max PnL observé (sweep) = {pnl_curve.max():,.0f} €")
    else:
        results['best_threshold_sweep'] = best_thresh_sweep

    # 5) Prédiction avec le seuil sweep
    y_pred_sweep = (proba >= best_thresh_sweep).astype(int)
    tn, fp, fn, tp = compute_confusion_matrix_cpu(y_data, y_pred_sweep, config)

    pnl_sweep = tp * profit_tp + fp * loss_fp + fn * penalty_fn
    trades = tp + fp
    samples = len(y_data)
    winrate = 100 * tp / trades if trades else 0.0
    trade_part = 100 * trades / samples if samples else 0.0

    results['sweep_metrics'] = {
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'pnl': pnl_sweep,
        'winrate': winrate,
        'trade_part': trade_part,
        'trades': trades
    }

    print(f"📊 Évaluation {dataset_name} (seuil sweepé) → TP: {tp} | FP: {fp} | TN: {tn} | FN: {fn} | "
          f"Winrate: {winrate:.2f}% | PnL: {pnl_sweep:,.0f} | %Trades: {trade_part:.2f}%")

    # 6) Prédiction avec le seuil médian CV si fourni
    if final_threshold_median is not None:
        y_pred_median = (proba >= final_threshold_median).astype(int)
        tn_median, fp_median, fn_median, tp_median = compute_confusion_matrix_cpu(y_data, y_pred_median, config)

        pnl_median = tp_median * profit_tp + fp_median * loss_fp + fn_median * penalty_fn
        trades_median = tp_median + fp_median
        winrate_median = 100 * tp_median / trades_median if trades_median else 0.0
        trade_part_median = 100 * trades_median / samples if samples else 0.0

        results['median_metrics'] = {
            'tp': tp_median,
            'fp': fp_median,
            'tn': tn_median,
            'fn': fn_median,
            'pnl': pnl_median,
            'winrate': winrate_median,
            'trade_part': trade_part_median,
            'trades': trades_median
        }

        print(
            f"📊 Évaluation {dataset_name} (seuil médian CV) → TP: {tp_median} | FP: {fp_median} | TN: {tn_median} | FN: {fn_median} | "
            f"Winrate: {winrate_median:.2f}% | PnL: {pnl_median:,.0f} | %Trades: {trade_part_median:.2f}%")

    return results