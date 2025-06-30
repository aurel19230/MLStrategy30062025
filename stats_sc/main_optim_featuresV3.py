import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
from sklearn.metrics import roc_auc_score

import os

from func_standard import print_notification, load_data, calculate_naked_poc_distances, CUSTOM_SESSIONS, \
    save_features_with_sessions,remplace_0_nan_reg_slope_p_2d,process_reg_slope_replacement
file_name = "Step4_version2_170924_110325_bugFixTradeResult1_extractOnlyFullSession_OnlyShort.csv"
#file_name = "Step4_5_0_5TP_1SL_150924_280225_bugFixTradeResult_extractOnlyFullSession_OnlyShort.csv"
# Chemin du répertoire
directory_path =  r"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra chart\xTickReversal\simu\5_0_5TP_1SL\version1"
directory_path =  r"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra chart\xTickReversal\simu\5_0_5TP_1SL\version2\merge"

# Construction du chemin complet du fichier
file_path = os.path.join(directory_path, file_name)
import keyboard  # Assurez-vous d'installer cette bibliothèque avec: pip install keyboard

import keyboard  # Assurez-vous d'installer cette bibliothèque avec: pip install keyboard
from pynput import keyboard as pynput_keyboard  # Alternative si keyboard pose problème

# Variable globale pour contrôler l'arrêt de l'optimisation
STOP_OPTIMIZATION = False

def callback_optuna_stop(study, trial):
    global STOP_OPTIMIZATION
    if STOP_OPTIMIZATION:
        print("Callback triggered: stopping the study.")
        study.stop()

def on_press(key):
    global STOP_OPTIMIZATION
    try:
        if key.char == '²':
            print("Stop signal received: stopping the study.")
            STOP_OPTIMIZATION = True
    except AttributeError:
        pass
REPLACE_NAN = False
REPLACED_NANVALUE_BY = 90000.54789
REPLACED_NANVALUE_BY_INDEX = 1
if REPLACE_NAN:
    print(
        f"\nINFO : Implémenter dans le code => les valeurs NaN seront remplacées par {REPLACED_NANVALUE_BY} et un index")
else:
    print(
        f"\nINFO : Implémenter dans le code => les valeurs NaN ne seront pas remplacées par une valeur choisie par l'utilisateur mais laissé à NAN")

# Configuration
CONFIG = {
    'FILE_PATH': file_path,
}

df = load_data(CONFIG['FILE_PATH'])

df_filtered = df[df['class_binaire'].isin([0, 1])].copy()

# Nouvelles features - Order Flow:
# Imbalances haussières
df_filtered['bull_imbalance_low_2'] = np.where(
    df_filtered['bidVolLow_1'] == 0,
    -6,
    np.where(
        (df_filtered['bidVolLow_1'] >= 1) & (df_filtered['bidVolLow_1'] <= 2),
        -2,
        df_filtered['askVolLow_2'] / df_filtered['bidVolLow_1']
    )
)

target_y = df_filtered['class_binaire'].copy()  # Série de la cible

# Fonction pour calculer l'information value (IV)
def calculate_iv(feature, target):
    df = pd.DataFrame({'feature': feature, 'target': target})
    cross_tab = pd.crosstab(df['feature'], df['target'])
    cross_tab = cross_tab + 0.5  # Éviter division par zéro

    print("\nCross-tabulation:")
    print(cross_tab)  # Debug

    # Vérifier si la classe 1 est présente
    if 1 not in cross_tab.columns or 0 not in cross_tab.columns:
        print("Warning: Missing class in cross-tab. Returning IV = 0.")
        return 0  # Retourne un IV nul si une classe est absente

    prop_event = cross_tab[1] / cross_tab[1].sum()
    prop_non_event = cross_tab[0] / cross_tab[0].sum()

    woe = np.log(prop_event / prop_non_event)
    iv = (prop_event - prop_non_event) * woe

    return iv.sum()

def calculate_power(feature, target):
    # Vérification si la feature contient bien les classes attendues
    unique_values = set(feature)
    if len(unique_values) < 2:
        print("Warning: Feature has insufficient unique values. Returning power = 0.")
        return 0  # Évite les erreurs dues à des variables constantes

    # Test du Chi2
    contingency = pd.crosstab(feature, target)
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        print("Warning: Contingency table does not contain enough classes. Returning power = 0.")
        return 0

    chi2, p_value, _, _ = stats.chi2_contingency(contingency)

    # Information Value (IV)
    iv = calculate_iv(feature, target)

    # Nouveau score combiné sans AUC
    # Exemple: 50% IV et 50% -log10(p_value)
    # Ajuste les pondérations à ta convenance
    score = 1 * iv + 0* (- np.log10(p_value + 1e-10))

    return score


from sklearn.metrics import normalized_mutual_info_score
def evaluate_feature(feature, target):
    # Calculer IV
    iv = calculate_iv(feature, target)

    # Calculer MI normalisée
    mi = normalized_mutual_info_score(feature, target)

    # Calculer la significativité statistique
    contingency = pd.crosstab(feature, target)
    chi2, p_value, _, _ = chi2_contingency(contingency)

    # Score combiné (exemple de pondération)
    score = (0.65 * iv) + (0.35 * mi) + (0 * (-np.log10(p_value + 1e-10)))

    return {
        'iv': iv,
        'mi': mi,
        'p_value': p_value,
        'combined_score': score
    }
# Choose which imbalance feature to optimize BEFORE running Optuna
# You can change this variable to select different features
SELECTED_FEATURE = 'bull_imbalance_low_2'  # Change this to any of your 12 features

# Map the selected feature to the appropriate volume columns
if SELECTED_FEATURE == 'bull_imbalance_low_1':
    bid_vol_col = 'bidVolLow'
    ask_vol_col = 'askVolLow_1'
    is_bull = True
elif SELECTED_FEATURE == 'bull_imbalance_low_2':
    bid_vol_col = 'bidVolLow_1'
    ask_vol_col = 'askVolLow_2'
    is_bull = True
elif SELECTED_FEATURE == 'bull_imbalance_low_3':
    bid_vol_col = 'bidVolLow_2'
    ask_vol_col = 'askVolLow_3'
    is_bull = True
elif SELECTED_FEATURE == 'bull_imbalance_high_0':
    bid_vol_col = 'bidVolHigh_1'
    ask_vol_col = 'askVolHigh'
    is_bull = True
elif SELECTED_FEATURE == 'bull_imbalance_high_1':
    bid_vol_col = 'bidVolHigh_2'
    ask_vol_col = 'askVolHigh_1'
    is_bull = True
elif SELECTED_FEATURE == 'bull_imbalance_high_2':
    bid_vol_col = 'bidVolHigh_3'
    ask_vol_col = 'askVolHigh_2'
    is_bull = True
# Bearish low features
elif SELECTED_FEATURE == 'bear_imbalance_low_0':
    ask_vol_col = 'askVolLow_1'
    bid_vol_col = 'bidVolLow'
    is_bull = False
elif SELECTED_FEATURE == 'bear_imbalance_low_1':
    ask_vol_col = 'askVolLow_2'
    bid_vol_col = 'bidVolLow_1'
    is_bull = False
elif SELECTED_FEATURE == 'bear_imbalance_low_2':
    ask_vol_col = 'askVolLow_3'
    bid_vol_col = 'bidVolLow_2'
    is_bull = False
# Adding bearish high features
elif SELECTED_FEATURE == 'bear_imbalance_high_1':
    ask_vol_col = 'askVolHigh'
    bid_vol_col = 'bidVolHigh_1'
    is_bull = False
elif SELECTED_FEATURE == 'bear_imbalance_high_2':
    ask_vol_col = 'askVolHigh_1'
    bid_vol_col = 'bidVolHigh_2'
    is_bull = False
elif SELECTED_FEATURE == 'bear_imbalance_high_3':
    ask_vol_col = 'askVolHigh_2'
    bid_vol_col = 'bidVolHigh_3'
    is_bull = False
else:
    raise ValueError(f"Unknown feature: {SELECTED_FEATURE}")

# Print confirmation of which feature is being optimized
print(f"Optimizing feature: {SELECTED_FEATURE}")
if is_bull:
    print(f"Using bid volume column: {bid_vol_col}")
    print(f"Using ask volume column: {ask_vol_col}")
else:
    print(f"Using ask volume column: {ask_vol_col}")
    print(f"Using bid volume column: {bid_vol_col}")
print(f"Feature type: {'Bullish' if is_bull else 'Bearish'} imbalance")
# Constants for special values
Imb_Div0 = -6
Imb_zone = -2
from sklearn.metrics import mutual_info_score
from scipy.stats import chi2_contingency
# Define your objective function using the selected columns
# [Tout votre code d'importation et de préparation des données reste identique]

# Ajouter ces lignes juste avant la définition de la fonction objective
# Définir des constantes globales pour les contraintes
MIN_BIN_SPREAD = 0.06  # Écart minimum entre bin3 et bin4 (5%)
MAX_bin_3_win_rate = 0.48  # Maximum pour bin3 (doit être négatif)
MIN_bin_4_win_rate = 0.52  # Minimum pour bin4 (doit être positif)
MIN_BIN_SIZE = 0.06  # Taille minimale pour un bin individuel (3%)


# Function to display and optionally modify constraints
def show_constraints():
    print("\n=== CONTRAINTES ACTUELLES ===")
    print(f"Écart minimum entre bins: {MIN_BIN_SPREAD:.4f}")
    print(f"Win rate maximale bin3: {MAX_bin_3_win_rate:.4f}")
    print(f"Win rate minimale bin4: {MIN_bin_4_win_rate:.4f}")
    print(f"Taille minimale de bin: {MIN_BIN_SIZE:.4f}")


# Afficher les contraintes initiales
show_constraints()


# Define your objective function using the selected columns
def objective(trial):
    # 1. Espace de recherche ajusté pour les seuils de volume
    yy = trial.suggest_int('yy', 0, 20)
    z = trial.suggest_int('z', yy, 20)

    # 2. Ajustement des plages pour c, d, e, f
    c = trial.suggest_float('c', 0.9, 1.5)

    d_min = max(c + 0.1, 1.3)
    d_max = 1.7
    d = trial.suggest_float('d', d_min, d_max)

    e_min = max(d + 0.1, 1.6)
    e_max = 2.2
    e = trial.suggest_float('e', e_min, e_max)

    f_min = max(e + 5, 20)
    f_max = 40
    f = trial.suggest_float('f', f_min, f_max)

    # 3. Construction de la feature avec les seuils optimisés
    if is_bull:
        imbalance_raw = np.where(
            df_filtered[bid_vol_col] < yy,
            Imb_Div0,
            np.where(
                (df_filtered[bid_vol_col] >= yy) & (df_filtered[bid_vol_col] <= z),
                Imb_zone,
                df_filtered[ask_vol_col] / df_filtered[bid_vol_col]
            )
        )
    else:
        imbalance_raw = np.where(
            df_filtered[ask_vol_col] < yy,
            Imb_Div0,
            np.where(
                (df_filtered[ask_vol_col] >= yy) & (df_filtered[ask_vol_col] <= z),
                Imb_zone,
                df_filtered[bid_vol_col] / df_filtered[ask_vol_col]
            )
        )

    # 4. Création des bins
    bins = [-np.inf, -5, 0, c, d, e, f, np.inf]
    if not all(bins[i] < bins[i + 1] for i in range(len(bins) - 1)):
        return -np.inf


    imbalance_binned = pd.cut(imbalance_raw, bins=bins, labels=False)

    K = pd.Series(imbalance_binned).value_counts().shape[0]
    # On laisse Optuna choisir i, j parmi [0..K-1], i < j
    i = trial.suggest_int('bin_index_i', 0, K-2)
    j = trial.suggest_int('bin_index_j', i+1, K-1)
    i=3
    j=4
    # 5. Calculer les taux de gain par bin
    try:
        win_rates = pd.crosstab(imbalance_binned, target_y, normalize='index')
        win_rates = win_rates[1]
    except Exception as e:
        return -np.inf

    # 6. Obtenir les win rates des bins 3 et 4
    try:
        bin_3_win_rate = win_rates[i]
        bin_4_win_rate = win_rates[j]
    except KeyError:
        return -np.inf

    bin_spread = bin_4_win_rate - bin_3_win_rate

    # 7. Calculer les tailles des bins
    bin_counts = pd.Series(imbalance_binned).value_counts(normalize=True)
    bin_3_pct = bin_counts.get(i, 0)
    bin_4_pct = bin_counts.get(j, 0)

    # 8. UTILISER LES VARIABLES GLOBALES POUR LES CONTRAINTES

    # Contrainte 1: Différence minimum entre les bins
    if bin_spread < MIN_BIN_SPREAD:
        return -np.inf

    # Contrainte 2: Exiger un bin3 négatif
    if bin_3_win_rate > MAX_bin_3_win_rate:
        return -np.inf

    # Contrainte 3: Exiger un bin4 positif
    if bin_4_win_rate < MIN_bin_4_win_rate:
        return -np.inf

    # Contrainte 4: Tailles minimales des bins
    if bin_3_pct < MIN_BIN_SIZE or bin_4_pct < MIN_BIN_SIZE:
        return -np.inf

    # 9. Score et bonus
    normalized_spread = bin_spread * 100
    bin_size_score = (bin_3_pct + bin_4_pct) * 15

    combined_score = (0.8 * normalized_spread) + (0.2 * bin_size_score)

    if bin_spread >= 0.15:
        combined_score *= 1.15
    elif bin_spread >= 0.12:
        combined_score *= 1.1

    # 10. Stockage pour analyse
    trial.set_user_attr('bin_3_win_rate', float(bin_3_win_rate))
    trial.set_user_attr('bin_4_win_rate', float(bin_4_win_rate))
    trial.set_user_attr('bin_3_pct', float(bin_3_pct))
    trial.set_user_attr('bin_4_pct', float(bin_4_pct))
    trial.set_user_attr('bin_spread', float(bin_spread))
    trial.set_user_attr('combined_score', float(combined_score))
    trial.set_user_attr('c_val', float(c))
    trial.set_user_attr('d_val', float(d))
    trial.set_user_attr('e_val', float(e))
    trial.set_user_attr('f_val', float(f))
    trial.set_user_attr('bin_i', i)
    trial.set_user_attr('bin_j', j)

    # 11. Logs
    if trial.number % 50 == 0:
        print(f"Trial {trial.number}: Spread={bin_spread:.4f}, Bin3={bin_3_pct:.2%}, Bin4={bin_4_pct:.2%}")
        print(f"  Win rates: Bin3={bin_3_win_rate:.4f}, Bin4={bin_4_win_rate:.4f}")
        print(f"  Paramètres: yy={yy}, z={z}, c={c:.3f}, d={d:.3f}, e={e:.3f}, f={f:.1f}")
        print(f"  Score: {combined_score:.2f}")

    return combined_score

# Define a function to modify constraints
def modify_constraints(min_spread=None, max_bin3=None, min_bin4=None, min_size=None):
    global MIN_BIN_SPREAD, MAX_bin_3_win_rate, MIN_bin_4_win_rate, MIN_BIN_SIZE

    if min_spread is not None:
        MIN_BIN_SPREAD = min_spread
    if max_bin3 is not None:
        MAX_bin_3_win_rate = max_bin3
    if min_bin4 is not None:
        MIN_bin_4_win_rate = min_bin4
    if min_size is not None:
        MIN_BIN_SIZE = min_size

    show_constraints()
    print("Contraintes modifiées!")
def calculate_constraints_optuna(trial):  # Ajout de "trial" ici
    bin_3_win_rate = trial.user_attrs.get('bin_3_win_rate', 0)
    bin_4_win_rate = trial.user_attrs.get('bin_4_win_rate', 0)
    bin_spread    = trial.user_attrs.get('bin_spread', 0)
    bin_3_pct      = trial.user_attrs.get('bin_3_pct', 0)
    bin_4_pct      = trial.user_attrs.get('bin_4_pct', 0)

    # Contraintes :
    # 1. bin_spread >= MIN_BIN_SPREAD
    c1 = MIN_BIN_SPREAD - bin_spread
    # 2. bin_3_win_rate <= MAX_bin_3_win_rate
    c2 = bin_3_win_rate - MAX_bin_3_win_rate
    # 3. bin_4_win_rate >= MIN_bin_4_win_rate
    c3 = MIN_bin_4_win_rate - bin_4_win_rate
    # 4. bin_3_pct >= MIN_BIN_SIZE
    c4 = MIN_BIN_SIZE - bin_3_pct
    # 5. bin_4_pct >= MIN_BIN_SIZE
    c5 = MIN_BIN_SIZE - bin_4_pct

    return [c1, c2, c3, c4, c5]

def create_constraints_func():
    return calculate_constraints_optuna  # Retourne directement la fonction


    return constraints_func
# Callback pour suivre le meilleur essai
def print_best_trial_callback(study, trial):
    # N'afficher que périodiquement
    if trial.number % 100 == 0:
        try:
            best_trial = study.best_trial
            print("\n📊 Meilleur Trial jusqu'à présent:")
            print(f"  Trial ID: {best_trial.number}")
            print(f"  Score: {best_trial.value:.4f}")
            print(f"  Bin 3: {best_trial.user_attrs.get('bin_3_win_rate', 'N/A'):.4f} "
                  f"({best_trial.user_attrs.get('bin_3_pct', 'N/A'):.2%})")
            print(f"  Bin 4: {best_trial.user_attrs.get('bin_4_win_rate', 'N/A'):.4f} "
                  f"({best_trial.user_attrs.get('bin_4_pct', 'N/A'):.2%})")
            print(f"  Écart: {best_trial.user_attrs.get('bin_spread', 'N/A'):.4f}")
            print(f"  Seuils: c={best_trial.user_attrs.get('c_val', 'N/A'):.3f}, "
                  f"d={best_trial.user_attrs.get('d_val', 'N/A'):.3f}, "
                  f"e={best_trial.user_attrs.get('e_val', 'N/A'):.3f}, "
                  f"f={best_trial.user_attrs.get('f_val', 'N/A'):.1f}")
            print(f"  Paramètres: yy={best_trial.params['yy']}, z={best_trial.params['z']}")
        except ValueError:
            print("\nPas encore de meilleur essai trouvé...")


# Si aucun essai valide n'est trouvé, vous pouvez ajuster les contraintes
# Exemple:
# modify_constraints(min_spread=0.03, max_bin3=0.49, min_bin4=0.51, min_size=0.02)

# [Le reste de votre code pour créer et exécuter l'optimisation]
sampler = optuna.samplers.TPESampler(
    seed=41,
    constraints_func=create_constraints_func()
)

study = optuna.create_study(
    direction="maximize",
    sampler=sampler,
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
)

# Démarrer le listener de clavier avant l'optimisation
listener = pynput_keyboard.Listener(on_press=on_press)
listener.start()

# Lancer l'optimisation avec les callbacks
study.optimize(
    objective,
    n_trials=30000,
    callbacks=[callback_optuna_stop, print_best_trial_callback]  # Ajout du callback de suivi
)

# Arrêter proprement le listener après l'optimisation
listener.stop()
listener.join()

# Traiter les résultats avec gestion d'erreur
# In the objective function where you set attributes:

# Then in the final analysis:
try:
    # Meilleurs paramètres
    best_params = study.best_params
    best_i = study.best_trial.user_attrs.get('bin_i')
    best_j = study.best_trial.user_attrs.get('bin_j')

    print("\n=== RÉSULTATS FINAUX ===")
    print("Meilleurs paramètres trouvés:")
    print(f"y = {best_params['yy']}, z = {best_params['z']}")
    print(f"Indices de bins optimaux: i = {best_i}, j = {best_j}")
    print(f"Seuils de bins: c = {best_params['c']}, d = {best_params['d']}, e = {best_params['e']}"
          f", f = {best_params['f']}")

    # [Le reste de votre code pour appliquer les meilleurs paramètres]
    # Appliquer les meilleurs paramètres au dataset final
    best_z = best_params['z']
    best_yy = best_params['yy']
    best_a, best_b = -5, 0
    best_c, best_d, best_e = best_params['c'], best_params['d'], best_params['e']
    best_f = best_params['f']

    # Constants for special values - use the SAME values as in the objective function
    Imb_Div0 = -6
    Imb_zone = -2

    # Create the raw feature using exactly the same logic as in the objective function
    if is_bull:
        imbalance_raw = np.where(
            df_filtered[bid_vol_col] < best_params['yy'],
            Imb_Div0,
            np.where(
                (df_filtered[bid_vol_col] >= best_params['yy']) & (df_filtered[bid_vol_col] <= best_params['z']),
                Imb_zone,
                df_filtered[ask_vol_col] / df_filtered[bid_vol_col]
            )
        )
    else:
        imbalance_raw = np.where(
            df_filtered[ask_vol_col] < best_params['yy'],
            Imb_Div0,
            np.where(
                (df_filtered[ask_vol_col] >= best_params['yy']) & (df_filtered[ask_vol_col] <= best_params['z']),
                Imb_zone,
                df_filtered[bid_vol_col] / df_filtered[ask_vol_col]
            )
        )

    # Use the exact same binning thresholds as in the objective function
    best_bins = [-np.inf, -5, 0, best_params['c'], best_params['d'], best_params['e'], best_params['f'], np.inf]
    feature_name = f"{SELECTED_FEATURE}_optimized"
    df_filtered.loc[:, feature_name] = pd.cut(imbalance_raw, bins=best_bins, labels=False)

    # Analysis with the new feature²
    bin_analysis = pd.crosstab(df_filtered[feature_name], target_y, normalize='index')
    bin_counts = df_filtered[feature_name].value_counts().sort_index()

    analysis_df = pd.DataFrame({
        'Bin': range(len(bin_counts)),
        'Count': bin_counts.values,
        'Percentage': bin_counts.values / len(df_filtered) * 100,
        'Target_Rate': bin_analysis[1]
    })

    print(f"\nAnalyse détaillée des bins optimisés pour {SELECTED_FEATURE}:")
    print(analysis_df)

except ValueError as e:
    print(f"\n⚠️ ERREUR: {e}")
    print("Aucun essai valide n'a été trouvé. Essayez de relâcher les contraintes avec:")
    print("  modify_constraints(min_spread=0.03, max_bin3=0.49, min_bin4=0.51, min_size=0.02)")
    print("\n=== RÉSULTATS FINAUX ===")
    print("Meilleurs paramètres trouvés:")

    # En cas d'échec, vous pouvez relâcher les contraintes et réessayer automatiquement
    # Décommenter les lignes suivantes pour activer cette fonctionnalité:
    #
    # print("\nRelâchement automatique des contraintes et nouvel essai...")
    # modify_constraints(min_spread=0.03, max_bin3=0.49, min_bin4=0.51, min_size=0.02)
    # study = optuna.create_study(direction="maximize", sampler=sampler)
    # study.optimize(objective, n_trials=1000, callbacks=[callback_optuna_stop, print_best_trial_callback])