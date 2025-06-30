import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from scipy.stats import spearmanr, ttest_ind
from statsmodels.stats.power import TTestIndPower
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import os
import warnings
from numba import jit
from func_standard import (print_notification,
                           check_gpu_availability,
                           optuna_doubleMetrics,
                           callback_optuna,
                           calculate_weighted_adjusted_score_custom,
                           scalerChoice,
                           reTrain_finalModel_analyse, init_dataSet,best_modellastFold_analyse,
                           calculate_normalized_pnl_objectives,
                           run_cross_validation,
                           setup_model_params_optuna, setup_model_weight_optuna, cv_config,
                           load_features_and_sections, manage_rfe_selection,
                           setup_cv_method,
                           calculate_constraints_optuna, add_session_id, process_cv_results)
from stats_sc.standard_stat_sc import *

# Ignorer les avertissements pour un affichage plus propre
warnings.filterwarnings('ignore')




@jit(nopython=True)
def analyser_et_filtrer_sessions_numba(session_start_end, timestamps, duree_normale, seuil_anormal):
    """
    Analyse et filtre les sessions en fonction de leur durée.
    Version optimisée avec Numba.

    Args:
        session_start_end: Tableau des marqueurs de début (10) et fin (20) de session
        timestamps: Tableau des timestamps
        duree_normale: Durée normale d'une session en minutes
        seuil_anormal: Seuil en dessous duquel une session est considérée anormale

    Returns:
        Tuple contenant diverses statistiques sur les sessions
    """
    sessions = []
    sessions_normales = []
    sessions_anormales = 0
    sessions_normales_count = 0
    sessions_superieures = 0
    duree_minimale = np.iinfo(np.int64).max
    duree_maximale = 0
    session_plus_courte = None
    session_plus_longue = None
    session_start = None

    for i in range(len(session_start_end)):
        if session_start_end[i] == 10:
            session_start = timestamps[i]
        elif session_start_end[i] == 20 and session_start is not None:
            session_end = timestamps[i]
            duree = (session_end - session_start) // 60  # Convertir de secondes en minutes

            # Classifier la session
            if duree < seuil_anormal * duree_normale:
                sessions_anormales += 1
            elif duree > duree_normale:
                sessions_superieures += 1
            else:
                sessions_normales_count += 1
                sessions_normales.append((session_start, session_end))

            sessions.append((session_start, session_end, duree))

            # Mettre à jour la session la plus courte et la plus longue
            if duree < duree_minimale:
                duree_minimale = duree
                session_plus_courte = (session_start, session_end, duree)
            if duree > duree_maximale:
                duree_maximale = duree
                session_plus_longue = (session_start, session_end, duree)

            session_start = None

    return sessions, sessions_normales, sessions_anormales, sessions_normales_count, sessions_superieures, session_plus_courte, session_plus_longue


def analyser_et_sauvegarder_sessions(df, duree_normale=1380, seuil_anormal=0.95, taille_lot=100000,
                                     fichier_original=None, sessions_a_sauvegarder=None):
    """
    Analyse et sauvegarde les sessions normales d'un DataFrame.

    Args:
        df: DataFrame contenant les données
        duree_normale: Durée normale d'une session en minutes
        seuil_anormal: Seuil en dessous duquel une session est considérée anormale
        taille_lot: Taille des lots pour le traitement
        fichier_original: Chemin du fichier original
        sessions_a_sauvegarder: Nombre de sessions à sauvegarder

    Returns:
        Tuple contenant diverses statistiques sur les sessions
    """
    print_notification("Début de l'analyse et de la sauvegarde des sessions")

    if 'SessionStartEnd' not in df.columns or 'timeStampOpeningConvertedtoDate' not in df.columns:
        raise ValueError(
            "Les colonnes 'SessionStartEnd' et 'timeStampOpeningConvertedtoDate' doivent être présentes dans le DataFrame.")

    if not np.all(np.isin(df['SessionStartEnd'], [10, 15, 20])):
        raise ValueError("La colonne 'SessionStartEnd' contient des valeurs autres que 10, 15 ou 20.")

    if df['SessionStartEnd'].value_counts()[10] != df['SessionStartEnd'].value_counts()[20]:
        print("Le nombre de débuts de session (10) ne correspond pas au nombre de fins de session (20).")

    print_notification("Préparation des données pour l'analyse")
    timestamps = df['timeStampOpeningConvertedtoDate'].astype(np.int64).values // 10 ** 9
    session_start_end = df['SessionStartEnd'].values

    all_sessions = []
    all_sessions_normales = []
    sessions_anormales = 0
    sessions_normales = 0
    sessions_superieures = 0
    session_plus_courte = None
    session_plus_longue = None

    print_notification("Début de l'analyse des sessions par lots")
    for i in range(0, len(df), taille_lot):
        lot_session_start_end = session_start_end[i:i + taille_lot]
        lot_timestamps = timestamps[i:i + taille_lot]

        results = analyser_et_filtrer_sessions_numba(lot_session_start_end, lot_timestamps, duree_normale,
                                                     seuil_anormal)

        all_sessions.extend(results[0])
        all_sessions_normales.extend(results[1])
        sessions_anormales += results[2]
        sessions_normales += results[3]
        sessions_superieures += results[4]

        if results[5] and (session_plus_courte is None or results[5][2] < session_plus_courte[2]):
            session_plus_courte = results[5]
        if results[6] and (session_plus_longue is None or results[6][2] > session_plus_longue[2]):
            session_plus_longue = results[6]

        print_notification(f"Lot traité : {i + 1} à {min(i + taille_lot, len(df))} sur {len(df)}")

    print_notification("Conversion des timestamps")
    all_sessions = [(pd.Timestamp(start, unit='s'), pd.Timestamp(end, unit='s'), pd.Timedelta(minutes=int(duree))) for
                    start, end, duree in all_sessions]
    if session_plus_courte:
        session_plus_courte = (
            pd.Timestamp(session_plus_courte[0], unit='s'), pd.Timestamp(session_plus_courte[1], unit='s'),
            pd.Timedelta(minutes=int(session_plus_courte[2])))
    if session_plus_longue:
        session_plus_longue = (
            pd.Timestamp(session_plus_longue[0], unit='s'), pd.Timestamp(session_plus_longue[1], unit='s'),
            pd.Timedelta(minutes=int(session_plus_longue[2])))

    # Sauvegarder les sessions normales si un fichier original est spécifié
    if fichier_original:
        print_notification("Début de la sauvegarde des sessions normales")
        df_sessions_normales = pd.DataFrame()

        # Trier les sessions normales par ordre chronologique décroissant
        all_sessions_normales.sort(key=lambda x: x[1], reverse=True)

        # Si un nombre spécifique de sessions est demandé, ne prendre que les dernières sessions
        if sessions_a_sauvegarder and sessions_a_sauvegarder < len(all_sessions_normales):
            print_notification(f"Sauvegarde des {sessions_a_sauvegarder} dernières sessions normales")
            all_sessions_normales = all_sessions_normales[:sessions_a_sauvegarder]
        else:
            print_notification("Sauvegarde de toutes les sessions normales")

        # Inverser l'ordre pour traiter du plus ancien au plus récent
        all_sessions_normales.reverse()

        for i, (start, end) in enumerate(all_sessions_normales):
            start_ts = pd.Timestamp(start, unit='s')
            end_ts = pd.Timestamp(end, unit='s')
            session_data = df[
                (df['timeStampOpeningConvertedtoDate'] >= start_ts) & (df['timeStampOpeningConvertedtoDate'] <= end_ts)]
            df_sessions_normales = pd.concat([df_sessions_normales, session_data])
            if (i + 1) % 100 == 0:
                print_notification(f"Sessions normales traitées : {i + 1} sur {len(all_sessions_normales)}")

        df_sessions_normales = df_sessions_normales.reset_index(drop=True)

        # Modify the part where the new filename is created
        nom_fichier = os.path.basename(fichier_original)
        nom_fichier, extension = os.path.splitext(nom_fichier)
        dossier = os.path.dirname(fichier_original)

        fileStep3 = os.path.basename(fichier_original).replace("Step2", "Step3")
        fileStep3 = os.path.splitext(fileStep3)[0]
        if sessions_a_sauvegarder:
            nouveau_fichier = f"{fileStep3}_extractOnly{sessions_a_sauvegarder}LastFullSession{extension}"
        else:
            nouveau_fichier = f"{fileStep3}_extractOnlyFullSession{extension}"

        # Combine the directory path with the new filename
        nouveau_fichier_complet = os.path.join(dossier, nouveau_fichier)

        print_notification(f"Sauvegarde du fichier : {nouveau_fichier_complet}")
        df_sessions_normales.to_csv(nouveau_fichier_complet, sep=';', index=False)

        print_notification(f"Les sessions normales ont été sauvegardées dans le fichier : {nouveau_fichier}")
        print_notification(f"Nombre de lignes dans le nouveau fichier : {len(df_sessions_normales)}")

    print_notification("Fin de l'analyse et de la sauvegarde des sessions")
    return all_sessions, sessions_anormales, sessions_normales, sessions_superieures, session_plus_courte, session_plus_longue



def extract_session_data(df, session_ids):
    """
    Extrait les données pour un ensemble spécifique de sessions.

    Args:
        df (pd.DataFrame): DataFrame avec la colonne session_id
        session_ids (list): Liste des IDs de session à extraire

    Returns:
        pd.DataFrame: Sous-ensemble du DataFrame contenant uniquement les sessions spécifiées
    """
    return df[df['session_id'].isin(session_ids)]


def compute_mRMR_scores(X, Y, config):
    """
    Calcule les scores mRMR pour toutes les features sans filtrage.

    Args:
        X (pd.DataFrame): DataFrame des features (numériques ou encodées).
        Y (pd.Series): Série cible (peut être catégorielle ou continue).
        config (dict):
            - mi_method (str) : "classif" ou "regression" pour forcer le type
                                de calcul de l'information mutuelle.
            - verbose (bool) : Si True, affiche la progression du traitement.

    Returns:
        mrmr_scores (dict): Dictionnaire {feature: score_mRMR} pour toutes les features.
        relevance_dict (dict): Dictionnaire {feature: relevance} (information mutuelle avec la cible).
        redundancy_dict (dict): Dictionnaire {feature: redundancy_moyenne} (redondance moyenne avec autres features).
    """
    # Paramètres par défaut
    verbose = config.get("verbose", True)

    if verbose:
        print(f"🔍 Démarrage du calcul des scores mRMR pour {X.shape[1]} features")
        start_time = time.time()

    # 1) Détermination automatique du type de mutual_info
    if "mi_method" in config:
        mi_method = config["mi_method"]
    else:
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

    # 3) Calcul de la matrice de redondance : MI(feature_i; feature_j)
    n = len(features)
    mi_matrix = np.zeros((n, n))

    if verbose:
        print(f"🔄 Calcul de la matrice de redondance ({n}x{n})...")
        total_pairs = n * (n - 1) // 2
        pbar = tqdm(total=total_pairs, desc="Calcul MI", disable=not verbose)

    for i in range(n):
        for j in range(i + 1, n):
            xi = X.iloc[:, i].values.reshape(-1, 1)
            xj = X.iloc[:, j].values
            mi_val = mutual_info_regression(xi, xj, random_state=0)[0]
            mi_matrix[i, j] = mi_val
            mi_matrix[j, i] = mi_val

            if verbose:
                pbar.update(1)

    if verbose:
        pbar.close()
        print(f"✓ Matrice de redondance calculée")

    # 4) Calcul des scores mRMR pour toutes les features
    mrmr_scores = {}
    redundancy_dict = {}

    if verbose:
        print("🧮 Calcul des scores mRMR pour toutes les features...")
        calc_progress = tqdm(total=len(features), desc="Calcul des scores", disable=not verbose)

    for i, feature in enumerate(features):
        # Pour chaque feature, calculer sa redondance moyenne avec toutes les autres features
        redundancy = np.mean([mi_matrix[i, j] for j in range(n) if j != i])
        redundancy_dict[feature] = redundancy

        # Score mRMR = pertinence - redondance moyenne
        mrmr_score = relevance_dict[feature] - redundancy
        mrmr_scores[feature] = mrmr_score

        if verbose:
            calc_progress.update(1)

    if verbose:
        calc_progress.close()
        elapsed_time = time.time() - start_time
        print(f"✅ Calcul terminé en {elapsed_time:.2f} secondes")

        # Afficher les 5 meilleures features selon le score mRMR
        top_features = sorted(mrmr_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        print("📊 Top 5 features par score mRMR:")
        for i, (feature, score) in enumerate(top_features, 1):
            print(
                f"  {i}. {feature}: {score:.4f} (pertinence: {relevance_dict[feature]:.4f}, redondance: {redundancy_dict[feature]:.4f})")

    # Retourne les dictionnaires de scores, pertinence et redondance
    return mrmr_scores, relevance_dict, redundancy_dict

from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from scipy.stats import spearmanr
from statsmodels.stats.power import TTestIndPower

def analyze_rolling_windows(df, target_column, feature_columns, window_size=5, step_size=1, mi_method=None):
    """
    Analyse la stabilité temporelle des variables à travers des fenêtres glissantes de sessions.

    Args:
        df (pd.DataFrame): DataFrame contenant les données de trading avec une colonne 'session_id'
        target_column (str): Nom de la colonne cible (variable à prédire)
        feature_columns (list): Liste des noms des colonnes de features à analyser
        window_size (int): Nombre de sessions par fenêtre d'analyse
        step_size (int): Nombre de sessions de décalage entre deux fenêtres consécutives
        mi_method (str, optional): "classif" ou "regression" pour l'information mutuelle.
                                   Si None, détermination automatique.

    Returns:
        pd.DataFrame: DataFrame contenant les métriques pour chaque feature et chaque fenêtre
        dict: Statistiques récapitulatives pour chaque feature
        pd.DataFrame: DataFrame avec les features classées par stabilité
    """
    print_notification(f"Début de l'analyse par fenêtres glissantes (taille={window_size}, pas={step_size})")
    start_time = time.time()

    if 'session_id' not in df.columns:
        raise ValueError("La colonne 'session_id' est requise. Veuillez utiliser preprocess_sessions() d'abord.")

    # 1. Identification des sessions uniques et création d'une liste ordonnée
    unique_sessions = df['session_id'].unique()
    unique_sessions.sort()  # Assurons-nous que les sessions sont dans l'ordre chronologique

    if len(unique_sessions) < window_size:
        raise ValueError(
            f"Nombre insuffisant de sessions ({len(unique_sessions)}) "
            f"pour une fenêtre de taille {window_size}"
        )

    # 2. Détermination du type de l'information mutuelle si non spécifié
    if mi_method is None:
        y_sample = df[target_column]
        if (
            y_sample.nunique() <= 20
            or y_sample.dtype == 'object'
            or y_sample.dtype == 'category'
            or np.issubdtype(y_sample.dtype, np.integer)
        ):
            mi_method = "classif"
        else:
            mi_method = "regression"

    print(f"✓ Méthode d'information mutuelle : {mi_method}")

    # 3. Création des fenêtres glissantes
    windows = []
    for i in range(0, len(unique_sessions) - window_size + 1, step_size):
        windows.append(unique_sessions[i : i + window_size])

    print(f"✓ {len(windows)} fenêtres d'analyse créées")

    # 4. Prétraitement : suppression des lignes avec NaN ou Inf
    print("🔄 Prétraitement des données...")

    # On copie seulement les colonnes utiles pour économiser la mémoire
    df_clean = df[['session_id', target_column] + feature_columns].copy()

    # Filtrer les lignes avec NaN ou Inf dans les features ou la cible
    mask_valid = ~df_clean[feature_columns + [target_column]].isna().any(axis=1)
    mask_valid &= ~np.isinf(df_clean[feature_columns + [target_column]]).any(axis=1)

    # Filtrer pour garder uniquement les valeurs binaires pour la cible (si classification)
    if mi_method == "classif":
        mask_valid &= df_clean[target_column].isin([0, 1])

    df_clean = df_clean[mask_valid]
    print(
        f"✓ {df_clean.shape[0]}/{df.shape[0]} lignes conservées après filtrage "
        f"({df_clean.shape[0] / df.shape[0] * 100:.1f}%)"
    )

    # 5. Préparation des structures pour stocker les résultats
    results = []

    # 6. Analyse pour chaque fenêtre
    print("📊 Calcul des métriques pour chaque fenêtre...")
    for window_idx, window_sessions in enumerate(tqdm(windows, desc="Fenêtres analysées")):
        # Sélectionner les données pour cette fenêtre
        window_data = df_clean[df_clean['session_id'].isin(window_sessions)].copy()

        # Appliquer un scaler spécifique à chaque fenêtre (pour éviter le data leakage)
        scaler = StandardScaler()
        window_data[feature_columns] = scaler.fit_transform(window_data[feature_columns])

        # Pour chaque feature, calculer les métriques
        for feature in feature_columns:
            feature_vals = window_data[feature].values.reshape(-1, 1)
            target_vals = window_data[target_column].values

            # Si trop peu de données dans la fenêtre
            if len(feature_vals) < 10:
                results.append({
                    'window_idx': window_idx,
                    'window_start': min(window_sessions),
                    'window_end': max(window_sessions),
                    'feature': feature,
                    'mi_score': np.nan,
                    'spearman_corr': np.nan,
                    'spearman_pvalue': np.nan,
                    'power_statistic': np.nan,
                    'sample_size': len(feature_vals),
                    'valid_data_percentage': (
                        len(feature_vals)
                        / df[df['session_id'].isin(window_sessions)].shape[0]
                        * 100
                    )
                })
                continue

            # 1. Information Mutuelle
            if mi_method == "classif":
                mi_score = mutual_info_classif(feature_vals, target_vals, random_state=0)[0]
            else:
                mi_score = mutual_info_regression(feature_vals, target_vals, random_state=0)[0]

            # 2. Corrélation de Spearman
            spearman_corr, spearman_pvalue = spearmanr(feature_vals.flatten(), target_vals)

            # 3. Puissance statistique (TTestIndPower)
            median_y = np.median(target_vals)
            group1 = feature_vals[target_vals <= median_y].flatten()
            group2 = feature_vals[target_vals > median_y].flatten()

            if len(group1) == 0 or len(group2) == 0:
                power_stat = np.nan
            else:
                try:
                    # Calcul de la taille d'effet (Cohen's d)
                    effect_size = (
                        (np.mean(group1) - np.mean(group2))
                        / np.sqrt(
                            ((len(group1) - 1) * np.var(group1)
                             + (len(group2) - 1) * np.var(group2))
                            / (len(group1) + len(group2) - 2)
                        )
                    )
                    power_analysis = TTestIndPower()
                    power_stat = power_analysis.solve_power(
                        effect_size=abs(effect_size),
                        nobs1=len(group1),
                        ratio=len(group2) / len(group1),
                        alpha=0.05,
                        alternative='two-sided'
                    )
                except:
                    power_stat = np.nan

            # Enregistrer les résultats
            results.append({
                'window_idx': window_idx,
                'window_start': min(window_sessions),
                'window_end': max(window_sessions),
                'feature': feature,
                'mi_score': mi_score,
                'spearman_corr': spearman_corr,
                'spearman_pvalue': spearman_pvalue,
                'power_statistic': power_stat,
                'sample_size': len(feature_vals),
                'valid_data_percentage': (
                    len(feature_vals)
                    / df[df['session_id'].isin(window_sessions)].shape[0]
                    * 100
                )
            })

    # 7. Conversion des résultats en DataFrame
    results_df = pd.DataFrame(results)

    # 8. Calcul des statistiques récapitulatives pour chaque feature
    summary_stats = {}
    for feature in feature_columns:
        feature_data = results_df[results_df['feature'] == feature]
        if feature_data.empty:
            continue

        mi_data = feature_data['mi_score'].dropna()
        spearman_data = feature_data['spearman_corr'].dropna()
        power_data = feature_data['power_statistic'].dropna()

        summary_stats[feature] = {
            # Information Mutuelle
            'mi_mean': mi_data.mean() if not mi_data.empty else np.nan,
            'mi_std': mi_data.std() if not mi_data.empty else np.nan,
            'mi_stability_ratio': (
                mi_data.mean() / mi_data.std()
            ) if not mi_data.empty and mi_data.std() != 0 else np.nan,
            'mi_min': mi_data.min() if not mi_data.empty else np.nan,
            'mi_max': mi_data.max() if not mi_data.empty else np.nan,

            # Corrélation de Spearman
            'spearman_mean': spearman_data.mean() if not spearman_data.empty else np.nan,
            'spearman_std': spearman_data.std() if not spearman_data.empty else np.nan,
            'spearman_stability_ratio': (
                spearman_data.mean() / spearman_data.std()
            ) if not spearman_data.empty and spearman_data.std() != 0 else np.nan,
            'spearman_min': spearman_data.min() if not spearman_data.empty else np.nan,
            'spearman_max': spearman_data.max() if not spearman_data.empty else np.nan,
            'spearman_sign_consistency': np.mean(
                np.sign(spearman_data) == np.sign(spearman_data.mean())
            ) if not spearman_data.empty else np.nan,

            # Puissance statistique
            'power_mean': power_data.mean() if not power_data.empty else np.nan,
            'power_std': power_data.std() if not power_data.empty else np.nan,
            'power_min': power_data.min() if not power_data.empty else np.nan,
            'power_max': power_data.max() if not power_data.empty else np.nan,
            'power_above_threshold': np.mean(power_data >= 0.8) if not power_data.empty else np.nan,

            # Métriques générales
            'windows_with_valid_data': feature_data.dropna(
                subset=['mi_score', 'spearman_corr', 'power_statistic']
            ).shape[0],
            'total_windows': feature_data.shape[0],
            'data_quality': (
                feature_data.dropna(subset=['mi_score', 'spearman_corr', 'power_statistic']).shape[0]
                / feature_data.shape[0]
            ) if feature_data.shape[0] > 0 else np.nan,
        }

    summary_df = pd.DataFrame.from_dict(summary_stats, orient='index')

    # 9. Tri par stabilité décroissante
    def rank_features(df):
        ranking_df = df.copy()
        ranking_df['mi_rank'] = ranking_df['mi_stability_ratio'].rank(ascending=False, na_option='bottom')
        ranking_df['spearman_rank'] = ranking_df['spearman_stability_ratio'].rank(ascending=False, na_option='bottom')
        ranking_df['power_rank'] = ranking_df['power_mean'].rank(ascending=False, na_option='bottom')
        ranking_df['avg_rank'] = (
            ranking_df['mi_rank'] + ranking_df['spearman_rank'] + ranking_df['power_rank']
        ) / 3
        return ranking_df.sort_values('avg_rank')

    ranked_summary = rank_features(summary_df)

    elapsed_time = time.time() - start_time
    print(f"✅ Analyse terminée en {elapsed_time:.2f} secondes")

    # 10. Affichage du top 5 features
    print("📊 Top 5 features les plus stables :")
    top_features = ranked_summary.head(5).index
    for i, feature in enumerate(top_features, 1):
        stats = summary_stats[feature]
        print(f"  {i}. {feature}:")
        print(
            f"     MI: {stats['mi_mean']:.4f} ± {stats['mi_std']:.4f} "
            f"(ratio: {stats.get('mi_stability_ratio', 'N/A'):.2f})"
        )
        print(
            f"     Spearman: {stats['spearman_mean']:.4f} ± {stats['spearman_std']:.4f} "
            f"(ratio: {stats.get('spearman_stability_ratio', 'N/A'):.2f})"
        )
        print(f"     Puissance: {stats['power_mean']:.4f} ± {stats['power_std']:.4f}")

    return results_df, summary_stats, ranked_summary




def plot_feature_stability(results_df, feature_name, save_path=None):
    """
    Crée des visualisations de la stabilité d'une feature à travers les fenêtres.

    Args:
        results_df (pd.DataFrame): DataFrame contenant les résultats de l'analyse
        feature_name (str): Nom de la feature à visualiser
        save_path (str, optional): Chemin pour sauvegarder la figure. Si None, la figure est affichée
    """
    feature_data = results_df[results_df['feature'] == feature_name].copy()

    if feature_data.empty:
        print(f"Aucune donnée disponible pour la feature {feature_name}")
        return

    # Créer des labels pour les fenêtres
    feature_data['window_label'] = feature_data['window_idx'].astype(str)

    # Créer une figure avec 3 sous-graphiques
    fig, axs = plt.subplots(3, 1, figsize=(12, 15))

    # 1. Information Mutuelle
    axs[0].plot(feature_data['window_idx'], feature_data['mi_score'], marker='o', linestyle='-', color='blue')
    axs[0].set_title(f'Information Mutuelle pour {feature_name}')
    axs[0].set_xlabel('Indice de fenêtre')
    axs[0].set_ylabel('Score MI')
    axs[0].grid(True, linestyle='--', alpha=0.7)

    # Ajouter une ligne horizontale pour la moyenne
    mean_mi = feature_data['mi_score'].mean()
    axs[0].axhline(y=mean_mi, color='r', linestyle='--', label=f'Moyenne: {mean_mi:.4f}')
    axs[0].legend()

    # 2. Corrélation de Spearman
    axs[1].plot(feature_data['window_idx'], feature_data['spearman_corr'], marker='o', linestyle='-', color='green')
    axs[1].set_title(f'Corrélation de Spearman pour {feature_name}')
    axs[1].set_xlabel('Indice de fenêtre')
    axs[1].set_ylabel('Coefficient de corrélation')
    axs[1].grid(True, linestyle='--', alpha=0.7)

    # Ajouter des lignes horizontales pour la moyenne et zéro
    mean_spearman = feature_data['spearman_corr'].mean()
    axs[1].axhline(y=mean_spearman, color='r', linestyle='--', label=f'Moyenne: {mean_spearman:.4f}')
    axs[1].axhline(y=0, color='k', linestyle='-', alpha=0.3)

    # Colorer les points en fonction de la significativité
    significant = feature_data['spearman_pvalue'] < 0.05
    if significant.any():
        axs[1].scatter(feature_data.loc[significant, 'window_idx'],
                       feature_data.loc[significant, 'spearman_corr'],
                       color='darkgreen', s=100, zorder=5, label='p < 0.05')
    axs[1].legend()

    # 3. Puissance statistique
    axs[2].plot(feature_data['window_idx'], feature_data['power_statistic'], marker='o', linestyle='-', color='purple')
    axs[2].set_title(f'Puissance statistique pour {feature_name}')
    axs[2].set_xlabel('Indice de fenêtre')
    axs[2].set_ylabel('Puissance')
    axs[2].grid(True, linestyle='--', alpha=0.7)

    # Ajouter une ligne horizontale pour le seuil de puissance de 0.8
    axs[2].axhline(y=0.8, color='r', linestyle='--', label='Seuil: 0.8')
    axs[2].axhline(y=0.6, color='orange', linestyle='--', label='Seuil: 0.6')

    # Colorer les points en fonction du niveau de puissance
    high_power = feature_data['power_statistic'] >= 0.8
    medium_power = (feature_data['power_statistic'] >= 0.6) & (feature_data['power_statistic'] < 0.8)

    if high_power.any():
        axs[2].scatter(feature_data.loc[high_power, 'window_idx'],
                       feature_data.loc[high_power, 'power_statistic'],
                       color='darkgreen', s=100, zorder=5, label='≥ 0.8')
    if medium_power.any():
        axs[2].scatter(feature_data.loc[medium_power, 'window_idx'],
                       feature_data.loc[medium_power, 'power_statistic'],
                       color='orange', s=100, zorder=5, label='0.6 - 0.8')
    axs[2].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure sauvegardée à {save_path}")
    else:
        plt.show()


def calculate_temporal_stability_score(summary_df, weights=None):
    """
    Calcule un score global de stabilité temporelle pour chaque feature.

    Args:
        summary_df (pd.DataFrame): DataFrame contenant les statistiques récapitulatives
        weights (dict, optional): Dictionnaire de poids pour chaque composante.
                                  Par défaut: {'mi': 0.3, 'spearman': 0.3, 'power': 0.2, 'consistency': 0.2}

    Returns:
        pd.DataFrame: DataFrame avec le score de stabilité pour chaque feature
    """
    if weights is None:
        weights = {'mi': 0.3, 'spearman': 0.3, 'power': 0.2, 'consistency': 0.2}

    # Copier le DataFrame pour éviter de modifier l'original
    result_df = summary_df.copy()

    # Normaliser les ratios de stabilité pour l'information mutuelle et Spearman
    for col in ['mi_stability_ratio', 'spearman_stability_ratio']:
        if col in result_df.columns:
            max_val = result_df[col].max()
            if max_val > 0:
                result_df[f'{col}_norm'] = result_df[col] / max_val
            else:
                result_df[f'{col}_norm'] = 0
        else:
            result_df[f'{col}_norm'] = 0

    # Normaliser la puissance moyenne
    if 'power_mean' in result_df.columns:
        result_df['power_mean_norm'] = result_df['power_mean'] / 1.0  # Déjà entre 0 et 1
    else:
        result_df['power_mean_norm'] = 0

    # Normaliser la cohérence de signe pour Spearman
    if 'spearman_sign_consistency' in result_df.columns:
        result_df['consistency_norm'] = result_df['spearman_sign_consistency']
    else:
        result_df['consistency_norm'] = 0

    # Calculer le score global de stabilité
    result_df['temporal_stability_score'] = (
            weights['mi'] * result_df['mi_stability_ratio_norm'] +
            weights['spearman'] * result_df['spearman_stability_ratio_norm'] +
            weights['power'] * result_df['power_mean_norm'] +
            weights['consistency'] * result_df['consistency_norm']
    )

    # Normaliser le score final entre 0 et 100
    max_score = result_df['temporal_stability_score'].max()
    if max_score > 0:
        result_df['temporal_stability_score'] = 100 * result_df['temporal_stability_score'] / max_score

    # Tri par score de stabilité décroissant
    result_df = result_df.sort_values('temporal_stability_score', ascending=False)

    return result_df[
        ['temporal_stability_score'] + [col for col in result_df.columns if col != 'temporal_stability_score']]


def prepare_csv_file(file_path, date_column=None):
    """
    Prépare un fichier CSV pour l'analyse en convertissant les colonnes temporelles.

    Args:
        file_path (str): Chemin vers le fichier CSV à traiter
        date_column (str, optional): Nom de la colonne de timestamp à convertir en datetime

    Returns:
        pd.DataFrame: DataFrame préparé pour l'analyse
    """
    print_notification(f"Préparation du fichier CSV: {file_path}")

    # Déterminer le séparateur
    with open(file_path, 'r') as f:
        first_line = f.readline()
        if ';' in first_line:
            sep = ';'
        else:
            sep = ','

    # Charger le fichier
    df, CUSTOM_SESSIONS = load_features_and_sections(file_path)
    print(f"Fichier chargé: {len(df)} lignes, {len(df.columns)} colonnes")

    # Convertir la colonne temporelle si spécifiée
    if date_column and date_column in df.columns:
        if df[date_column].dtype == 'int64':
            print(f"Conversion de la colonne {date_column} de timestamp Unix en datetime")
            df[date_column + 'ConvertedtoDate'] = pd.to_datetime(df[date_column], unit='s')
        else:
            try:
                print(f"Tentative de conversion de la colonne {date_column} en datetime")
                df[date_column + 'ConvertedtoDate'] = pd.to_datetime(df[date_column])
            except:
                print(f"Impossible de convertir la colonne {date_column} en datetime")

    return df


def run_stability_analysis(file_path, target_column, feature_columns, window_size=5, step_size=1,
                           output_dir=None, date_column='timeStampOpening'):
    """
    Exécute une analyse complète de la stabilité temporelle des features.

    Args:
        file_path (str): Chemin vers le fichier de données
        target_column (str): Nom de la colonne cible
        feature_columns (list): Liste des colonnes de features à analyser
        window_size (int): Taille de la fenêtre en nombre de sessions
        step_size (int): Pas entre les fenêtres
        output_dir (str, optional): Dossier où sauvegarder les résultats
        date_column (str): Nom de la colonne contenant les timestamps

    Returns:
        tuple: (results_df, summary_stats, ranked_summary, stability_scores)
    """
    print_notification(f"Analyse de stabilité temporelle pour {len(feature_columns)} features")
    print(f"Fichier: {file_path}")
    print(f"Cible: {target_column}")
    print(f"Fenêtre: {window_size} sessions, pas: {step_size}")

    # 1. Charger et préparer les données
    df = prepare_csv_file(file_path, date_column=date_column)

    # 2. Prétraiter les sessions
    df_with_sessions = preprocess_sessions_with_date(df)

    # 3. Vérifier que les colonnes existent
    missing_columns = [col for col in feature_columns + [target_column] if col not in df_with_sessions.columns]
    if missing_columns:
        raise ValueError(f"Colonnes manquantes: {missing_columns}")

    # 4. Lancer l'analyse par fenêtres glissantes
    results_df, summary_stats, ranked_summary = analyze_rolling_windows(
        df_with_sessions,
        target_column,
        feature_columns,
        window_size=window_size,
        step_size=step_size
    )

    # 5. Calculer le score de stabilité temporelle
    stability_scores = calculate_temporal_stability_score(ranked_summary)

    # 6. Afficher le classement des features
    print_notification("Classement des features par stabilité temporelle")
    print(stability_scores[['temporal_stability_score', 'mi_mean', 'spearman_mean', 'power_mean']].head(10))

    # 7. Générer les visualisations si un dossier de sortie est spécifié
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # Sauvegarder les résultats en CSV
        results_file = os.path.join(output_dir, 'stability_results.csv')
        results_df.to_csv(results_file, index=False)

        summary_file = os.path.join(output_dir, 'stability_summary.csv')
        stability_scores.to_csv(summary_file)

        # Générer les graphiques pour toutes les features
        print_notification("Génération des graphiques")
        for feature in tqdm(feature_columns, desc="Graphiques"):
            output_file = os.path.join(output_dir, f"{feature}_stability.png")
            plot_feature_stability(results_df, feature, save_path=output_file)

    return results_df, summary_stats, ranked_summary, stability_scores


# Exemple d'utilisation:
if __name__ == "__main__":
    # Définir le chemin du fichier à analyser
    DIRECTORY_PATH = r"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra chart\xTickReversal\simu\5_0_5TP_1SL\merge_I1_I2"
    FILE_NAME_ = "Step5_5_0_5TP_1SL_150924_280225_bugFixTradeResult_extractOnlyFullSession_OnlyShort_feat_winsorized.csv"
    file_path = os.path.join(DIRECTORY_PATH, FILE_NAME_)

    # Définir la variable cible et les features à analyser
    target_column = "class_binaire"  # Remplacer par votre variable cible

    # Liste des features à analyser
    feature_columns = [
        'ratio_delta_vol_VA16P',
        'diffLowPrice_0_1',
        'cumDOM_AskBid_pullStack_avgDiff_ratio',
        'ratio_volRevMove_volImpulsMove',
        'VolPocVolRevesalXContRatio','ratio_deltaRevMoveExtrem_volRevMoveExtrem',
        'diffVolDelta_2_2Ratio'
    ]

    # Définir le dossier de sortie pour les résultats
    output_dir = os.path.join(DIRECTORY_PATH, "stability_analysis")

    # Exécuter l'analyse
    results_df, summary_stats, ranked_summary, stability_scores = run_stability_analysis(
        file_path,
        target_column,
        feature_columns,
        window_size=50,
        step_size=2,
        output_dir=output_dir
    )

    print_notification("Analyse terminée")