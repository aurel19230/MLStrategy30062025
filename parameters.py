from definition import *
import os
import numpy as np

def get_path():
    from func_standard import detect_environment
    FILE_NAME_ = "Step5_4_0_6TP_1SL_080919_141024_extractOnlyFullSession_OnlyShort_feat_winsorized_oldclean.csv"
    FILE_NAME_ = "Step5_4_0_6TP_1SL_080919_141024_extractOnlyFullSession_OnlyShort_feat_winsorized.csv"
    # FILE_NAME_ = "Step5_4_0_6TP_1SL_080919_141024_extractOnlyFullSession_OnlyShort_feat_winsorizedScaledWithNanVal.csv"
    # FILE_NAME_ = "Step5_4_0_5TP_1SL_newBB_080919_281124_extractOnlyFullSession_OnlyShort_feat_winsorized_MorningasieEurope.csv"
    FILE_NAME_ = "Step5_4_0_5TP_1SL_newBB_080919_281124_extractOnlyFullSession_OnlyShort_feat_winsorized_MorningasieEurope.csv"
    FILE_NAME_ = "Step5_4_0_5TP_1SL_newBB_080919_281124_extractOnlyFullSession_OnlyShort_feat_winsorized_MorningasieEurope.csv"
    FILE_NAME_ = "Step5_4_0_5TP_1SL_newBB_080919_281124_extractOnly220LastFullSession_OnlyShort_feat_winsorized.csv"
    FILE_NAME_ = "Step5_4_0_5TP_1SL_newBB_080919_281124_extractOnly220LastFullSession_OnlyShort_feat_winsorized.csv"
    FILE_NAME_ = "Step5_4_0_5TP_1SL_newBB_080919_281124_extractOnly900LastFullSession_OnlyShort_feat_winsorized_MorningasieEurope.csv"
    FILE_NAME_ = "Step5_version2_170924_110325_bugFixTradeResult1_extractOnlyFullSession_OnlyShort_feat.csv"
    FILE_NAME_ = "Step5_version2_170924_110325_bugFixTradeResult1_extractOnlyFullSession_OnlyShort_feat_winsorized.csv"
    #FILE_NAME_ = "Step5_version2_100325_260325_bugFixTradeResult1_extractOnlyFullSession_OnlyShort_feat_winsorized.csv"

    ENV = detect_environment()
    if ENV == 'pycharm':
        if platform.system() != "Darwin":
            base_results_path = r"C:/Users/aulac/OneDrive/Documents/Trading/PyCharmProject/MLStrategy/data_preprocessing/results_optim/"
            DIRECTORY_PATH = r"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra chart\xTickReversal\simu\5_0_5TP_1SL\version2\merge\extend"
            DIRECTORY_PATH = r"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra chart\xTickReversal\simu\5_0_5TP_1SL\version2\merge"

        else:
            base_results_path = "/Users/aurelienlachaud/Documents/trading_local/data_preprocessing/results_optim/"
            DIRECTORY_PATH ="/Users/aurelienlachaud/Documents/trading_local/"
            DIRECTORY_PATH = "/Users/aurelienlachaud/Library/CloudStorage/OneDrive-Personal/Documents/Trading/VisualStudioProject/Sierra chart/xTickReversal/simu/4_0_5TP_1SL_newBB/merge"
    else:  # collab
        DIRECTORY_PATH =r"/content/drive/MyDrive/testFile/"
        base_results_path = r"/content/drive/MyDrive/Colab_Notebooks/xtickReversal/results_optim/"
    FILE_PATH = os.path.join(DIRECTORY_PATH, FILE_NAME_)

    return DIRECTORY_PATH,FILE_PATH,base_results_path

def get_model_param_range(model_type):
    """Retourne la configuration des paramètres selon le type de modèle"""
    if model_type == modelType.XGB:
        return {
            # Complexité de l’arbre
            'max_depth': {'min': 2, 'max': 5},  # on ouvre 1 à 2 niveaux de + pour capter + de patterns
            'min_child_weight': {'min': 5, 'max': 25},  # on relâche la contrainte pour laisser passer des splits rares
            'min_split_loss': {'min': 0.8, 'max': 5},  # (= gamma) 0 autorisé, 4 ≈ seuil raisonnable

            # Taux d’apprentissage
            'learning_rate': {'min': 0.006, 'max': 0.07, 'log': True},

            # Sous‑échantillonnage
            'subsample': {'min': 0.4, 'max': 0.85},
            'colsample_bytree': {'min': 0.55, 'max': 0.9},
            'colsample_bylevel': {'min': 0.55, 'max': 0.9},
            'colsample_bynode': {'min': 0.55, 'max': 0.9},

            # Régularisation
            'reg_alpha': {'min': 0.8, 'max': 5, 'log': True},
            'reg_lambda': {'min': 0.8, 'max': 5, 'log': True},
        }


    elif model_type == modelType.LGBM:
        return {
            'num_leaves': {'min': 3, 'max':18},  # Au lieu de 12, pour limiter la complexité
            'learning_rate': {'min': 0.01, 'max': 0.09, 'log': True},  # Plafond ramené à 0.02
            'min_child_samples': {'min': 49, 'max': 200},  # On autorise jusqu’à 120
            'bagging_fraction': {'min': 0.4, 'max': 0.7},  # Mini remonté à 0.4
            'feature_fraction': {'min': 0.3, 'max': 0.7},
            'feature_fraction_bynode': {'min': 0.5, 'max': 0.7},
            'min_split_gain': {'min': 1, 'max': 3.5},
            'lambda_l1': {'min': 1.5, 'max': 10.0, 'log': True},
            'lambda_l2': {'min': 1.0, 'max': 10.0, 'log': True},
            'bagging_freq': {'min': 1, 'max': 10},  # On évite le bagging à chaque itération
            'max_depth': {'min': 2, 'max': 5},  # On limite le max à 4
            'path_smooth': {'min': 2, 'max': 3.2},
            'drop_rate': {'min': 0.05, 'max': 0.3},  # taux de drop par itération
            'skip_drop': {'min': 0.1, 'max': 0.8},  # proba de ne pas drop
            'max_drop': {'min': 5, 'max': 50},  # max d’arbres droppés
        }
    elif model_type == modelType.XGBRF:
        return {
            # Nombre total d'arbres pour Random Forest
            'n_estimators': {'min': 400, 'max': 2000},

            # Profondeur maximale des arbres
            'max_depth': {'min': 2, 'max': 5},

            # Échantillonnage des lignes (bootstrap)
            'subsample': {'min': 0.5, 'max': 0.9},

            # Échantillonnage des colonnes par nœud (équivalent à max_features)
            'colsample_bynode': {'min': 0.6, 'max': 0.9},

            # Poids minimum par feuille
            'min_child_weight': {'min': 30, 'max': 100},

            # Perte minimale pour un split
            'gamma': {'min': 0.5, 'max': 3},

            # Régularisation L1
            'reg_alpha': {'min': 0.1, 'max': 1.3, 'log': True},

            # Régularisation L2
            'reg_lambda': {'min': 2, 'max': 5.5, 'log': True},

            # Paramètres fixes qui seront appliqués ailleurs dans le code
            # learning_rate=1.0 (obligatoire pour RF)
            # tree_method='gpu_hist' (pour GPU)
        }
    elif model_type == modelType.RF:
        return {
            # Équivalent à bagging_freq et bagging_fraction dans LGBM
            'n_estimators': {'min': 400, 'max': 1200},  # Réduit le max pour plus d'efficacité

            # Influence la complexité comme num_leaves dans LGBM
            'max_depth': {'min': 3, 'max': 11},  # Augmentation pour permettre des arbres plus complexes

            # Comparable au min_split_gain dans LGBM
            'min_samples_split': {'min': 2, 'max': 60},  # Ajusté pour permettre plus de splits

            # Similaire à min_child_samples dans LGBM
            'min_samples_leaf': {'min': 45, 'max': 100},  # Ajusté pour correspondre à la plage LGBM

            # Similaire à feature_fraction dans LGBM
            'max_features': {'min': 0.6, 'max': 0.95},  # Aligné sur feature_fraction de LGBM

            # Influence la complexité comme num_leaves dans LGBM
            'max_leaf_nodes': {'min': 30, 'max': 120},  # Aligné sur num_leaves de LGBM

            # Ajouté pour contrôler la régularisation comme lambda_l1/l2 dans LGBM
            'ccp_alpha': {'min': 0.01, 'max': 1, 'log': True},

            # Ajouté pour contrôler les splits comme min_split_gain dans LGBM
            'min_impurity_decrease': {'min': 0.001, 'max': 1, 'log': True},

            # Bootstrap - garde True pour le bagging (similaire au concept dans LGBM)
            'bootstrap': [True],
        }
    elif model_type == modelType.SVC:
        return {
            # Paramètre de régularisation - équivalent inverse de lambda_l1/l2 dans LGBM
            # Des valeurs plus élevées = moins de régularisation = log odds plus étendus
            'C': {'min': 0.01, 'max': 5, 'log': True},

            # Paramètre gamma contrôle la complexité locale - similaire à num_leaves/max_depth
            # Valeurs plus élevées = frontières plus complexes = log odds plus différenciés
            'gamma': {'min': 0.001, 'max': 5, 'log': True},

            # Degré du polynôme pour le noyau 'poly' - impact sur complexité
            'degree': {'min': 2, 'max': 4},  # Réduit légèrement pour éviter surapprentissage

            # Coefficient pour les noyaux 'poly' et 'sigmoid' - influence la flexibilité
            'coef0': {'min': 0.1, 'max': 5.0},  # Plage plus adaptée aux données financières

            # Gestion des classes déséquilibrées - important pour les signaux de trading
            'class_weight': ['balanced', None],

            # Tolérance pour convergence - impacte la précision des frontières
            'tol': {'min': 0.5e-5, 'max': 1e-2, 'log': True},  # Décommenté car important

            # Noyau - crucial pour la flexibilité du modèle
            'kernel': ['rbf', 'poly','sigmoid'],  # Ajouté explicitement pour varier la complexité

            # Shrinking heuristic - accélère l'entraînement et peut influencer les décisions limites
            'shrinking': [True, False],  # Décommenté car peut impacter performance
        }
    # elif model_type == modelType.CATBOOST:
    #     return {
    #         # Équivalent à XGB num_boost_round : Nombre total d'arbres
    #         'iterations': {'min': 500, 'max': 1500},
    #
    #         # Équivalent à XGB max_depth : Profondeur maximale de chaque arbre
    #         'depth': {'min': 6, 'max': 15},
    #
    #         # Équivalent à XGB learning_rate : Taux d'apprentissage
    #         'learning_rate': {'min': 0.007, 'max': 0.06, 'log': True},
    #
    #         # Paramètres de sous-échantillonnage
    #         # Équivalent à XGB min_child_weight : Nombre minimum d'échantillons dans une feuille
    #         'min_child_samples': {'min': 10, 'max': 50},
    #
    #         # Équivalent à XGB subsample : Fraction des observations pour chaque arbre
    #         'subsample': {'min': 0.6, 'max': 0.8},
    #
    #         # Équivalent à XGB colsample_bytree : Fraction des features pour chaque arbre
    #         'colsample_ratio': {'min': 0.6, 'max': 0.9},
    #
    #         # Paramètres de régularisation
    #         # Équivalent à XGB reg_lambda : Régularisation L2
    #         'l2_leaf_reg': {'min': 2.0, 'max': 15.0, 'log': True},
    #
    #         # Spécifique à CatBoost : Force de la randomisation dans la construction des arbres
    #         'random_strength': {'min': 0.05, 'max': 0.5, 'log': True},
    #
    #         # Spécifique à CatBoost : Contrôle l'intensité du bagging bayésien
    #         'bagging_temperature': {'min': 0.2, 'max': 0.8},
    #
    #         # Paramètres temporels
    #         # Spécifique à CatBoost : Taille des blocs pour la permutation des données
    #         'fold_permutation_block': {'min': 10, 'max': 50},
    #
    #         # Spécifique à CatBoost : Nombre d'itérations pour l'estimation des valeurs des feuilles
    #         'leaf_estimation_iterations': {'min': 5, 'max': 15},
    #
    #         # Paramètres catégoriels
    #         # Spécifique à CatBoost : Méthode d'estimation des valeurs des feuilles
    #         'leaf_estimation_method': {'values': ['Newton', 'Gradient']},
    #
    #         # Spécifique à CatBoost : Stratégie de croissance des arbres
    #         'grow_policy': {'values': ['SymmetricTree', 'Depthwise', 'Lossguide']},
    #
    #         # Spécifique à CatBoost : Type de bootstrap utilisé
    #         'bootstrap_type': {'values': ['Bayesian', 'Bernoulli', 'MVS']},
    #
    #         # Paramètres supplémentaires
    #         # Paramètres d'arrêt précoce
    #         'od_type': {'values': ['IncToDec', 'Iter']},
    #         'od_wait': {'min': 20, 'max': 50},
    #
    #         # Paramètres de gestion des variables catégorielles
    #         'max_ctr_complexity': {'min': 1, 'max': 4},
    #         'ctr_leaf_count_limit': {'min': 10, 'max': 100}
    #     }

def get_weight_param():

    weight_param = {
        # Nombre d'itérations de boosting (équivalent à num_boost_round dans XGB)
        'num_boost_round': {'min': 50,'max': 900},
        'threshold': {'min': 0.5, 'max': 0.5},
        'w_p': {'min': 1, 'max': 1},
        'w_n': {'min': 1.1, 'max': 2}, # 1 1

        'penalty_per_fn': {'min': 0, 'max': 0},
        'weight_split': {'min': 0.65, 'max': 0.65},
        'nb_split_weight': {'min': 0, 'max': 0},  # si 0, pas d'utilisation de weight_split
        'std_penalty_factor': {'min': 0, 'max': 0}
    }
    return weight_param

def get_config():
    # Configuration
    config = {
        'boosting_type':'gbtree', # gbtree: gbdt rf dart ||  xgb: gbtree gbtree gblinear
        'target_directory': "",
        'device_': 'cpu', #gpu_4RF
        'n_trials_optuna': 100000,
        'test_size_ratio': 0.10,
        'nanvalue_to_newval_': np.nan,
        'random_state_seed': 35,
        'early_stopping_rounds': 60,
        #'profit_per_tp':1.25,
        #'loss_per_fp': -1.25,
        # 'use_shapeImportance_file': r"C:\Users\aulac\OneDrive\Documents\Trading\PyCharmProject\MLStrategy\data_preprocessing\shap_dependencies_results\shap_values_Training_Set.csv",
        'results_directory': "",

        'enable_script_CUSTOM_SESSIONS':False,
        'weightPareto_pnl_val': 0.65,
        'weightPareto_pnl_diff': 0.35,
        'use_of_rfe_in_optuna': rfe_param.NO_RFE,
        'min_features_if_RFE_AUTO': 3,
        'optuna_objective_type': optuna_doubleMetrics.DISABLE,  # USE_DIST_TO_IDEAL, DISABLE
        #hard penealties via constraints
        'use_optuna_constraints_func': True,
        'config_constraint_min_trades_threshold_by_Fold': 15,
        'config_constraint_ratioWinrate_train_val': 25,
        'config_constraint_winrates_val_by_fold': 51,
        'config_constraint_min_trades_samples_perct': 8,#np.inf,
        'config_constraint_max_std_trades':np.inf, #active (std max autorisée entre folds) ex 0.15 sinon np.inf
        #soft penality via score
        'use_imbalance_penalty': False, #pour prendre en compte les différence de trade entre les folds
        'use_std_penalty': True,  # pour prendre en compte les différence de trade entre les folds
        'use_winrate_std_penalty': False,
        "use_negative_pnl_penalty":True,
        "use_winrate_ecart_penalty": True,
        'use_brier_penalty': False,
        "use_penalty_only_bestTrial":False,

        "calibration_method": "none",  # none | sigmoid | isotonic #"no_sweet_no_calib"
        "threshold_sweep_steps": 200,
        'is_log_enabled': False,
        'remove_inf_nan_afterFeaturesSelections': True,
       # 'compute_feature_stat': AutoFilteringOptions.ENABLE_VIF_CORR_MI , #ENABLE_MRMR #DISPLAY_MODE_NOFILTERING ENABLE_VIF_CORR_MI ENABLE_FISHER
        'compute_vif':True,
        'retained_only_vif': True,

        'method_powerAnaly': "montecarlo",#['both', 'analytical', 'montecarlo']
        'n_simulations_monte':1000,
        'powAnaly_threshold':0.5,
        'vif_threshold': 7,
        'corr_threshold': 2.8,
        'mi_threshold': 0.01,
        "mrmr_score_threshold": 0.0085,  # par défaut -np.inf ( quand commenté) , ce qui signifie pas de seuil

        "fisher_score_threshold": 4.5,  # exemple de seuil Fisher (optionnel)
        "fisher_pvalue_threshold": 0.15,  # seuil classique p-value (optionnel)
        'use_pnl_theoric':True,
        'nb_pca':0,#0 desactivate pca computing
        'scaler_choice': scalerChoice.SCALER_ROBUST,  # ou  ou SCALER_DISABLE SCALER_ROBUST SCALER_STANDARD SCALER_ROBUST SCALER_STANDARD SCALER_MINMAX SCALER_MAXABS
        'nb_split_tscv_': 5 , #le meillleur a été optenu avec 6 et les ancien aparam
        'non_acnhored_val_ratio': 0.5,
            # La taille de chaque fenêtre de validation est fixée à 70% de la taille de la fenêtre d'entraînement.
            # Autrement dit, pour chaque split, val_size = round(train_size * 0.7)
        'non_acnhoredRolling_train_overlap_ratio': 0.65,
            # Chaque fenêtre d'entraînement se décale dans le temps en conservant un recouvrement de 50% avec la précédente.
            # Par exemple, si train_size = 1000, alors la fenêtre suivante commence 500 lignes après la précédente (décalage = 1000 * (1 - 0.5))
        'cv_method': cv_config.TIME_SERIE_SPLIT_NON_ANCHORED_ROLLING,
        # TIME_SERIE_SPLIT_NON_ANCHORED_ROLLING_AFTER_PREVTRAIN TIME_SERIE_SPLIT_NON_ANCHORED_AFTER_PREVTRAIN, TIME_SERIE_SPLIT_NON_ANCHORED_ROLLING
        # TIME_SERIE_SPLIT_NON_ANCHORED_AFTER_PREVTRAIN TIME_SERIE_SPLIT TIME_SERIE_SPLIT_NON_ANCHORED_ROLLING
        # cv_config.K_FOLD, #,  TIME_SERIE_SPLIT TIMESERIES_SPLIT_BY_ID TIME_SERIE_SPLIT_NON_ANCHORED_AFTER_PREVVAL
        #'reinsert_nan_inf_afterScaling':False, ne fonctionne pas à date
        'svc_probability':False, #calibration interne des probabilité avec cv intterne
        #'svc_kernel':'poly', #['rbf', 'poly', 'sigmoid'],
        'model_type': modelType.XGB, #LGBM LGBM SVC RF XGBRF XGB
        'custom_objective_lossFct': model_custom_objective.XGB_CUSTOM_OBJECTIVE_PROFITBASED  ,#LGB_CUSTOM_OBJECTIVE_PROFITBASED XGB_CUSTOM_OBJECTIVE_PROFITBASED LGB_CUSTOM_OBJECTIVE_CROSS_ENTROPY LGB_CUSTOM_OBJECTIVE_PROFITBASED
        'custom_metric_eval': model_custom_metric.XGB_CUSTOM_METRIC_PNL, #LGB_CUSTOM_METRIC_PNL XGB_CUSTOM_METRIC_PNL
         #'custom_objective_lossFct': xgb_metric.LGB_CUSTOM_METRIC_PNL, LGB_CUSTOM_METRIC_PNL

    }
    return config

