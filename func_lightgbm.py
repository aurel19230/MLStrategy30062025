import lightgbm as lgb

from definition import *
from sklearn.metrics import brier_score_loss


# Fonction objective personnalisée pour LightGBM
def lgb_weighted_logistic_objective(w_p: float, w_n: float):
    def weighted_logistic_obj(preds, train_data):
        y_true = train_data.get_label()
        preds = 1.0 / (1.0 + np.exp(-preds))  # Application de la sigmoid

        # Calcul du gradient et de la hessienne avec les poids
        weights = np.where(y_true == 1, w_p, w_n)
        grad = (preds - y_true) * weights
        hess = preds * (1.0 - preds) * weights

        return grad, hess

    return weighted_logistic_obj


# Fonction auxiliaire pour le calcul du profit (équivalent à xgb_calculate_profitBased_gpu)
"""
def lgb_calculate_profitBased(y_true=None, y_pred_threshold=None, metric_dict=None,config=None):
    """
   # Calcule les métriques de profit pour LightGBM
"""
    # Calcul des métriques de base
    tp = np.sum((y_true == 1) & (y_pred_threshold == 1))
    fp = np.sum((y_true == 0) & (y_pred_threshold == 1))
    fn = np.sum((y_true == 1) & (y_pred_threshold == 0))

    # Récupération des paramètres de profit/perte
    profit_per_tp = config.get('profit_per_tp', 11111)
    loss_per_fp = config.get('loss_per_fp', 11111)
    penalty_per_fn = metric_dict.get('penalty_per_fn', 11111)
    if (penalty_per_fn == 11111 or profit_per_tp == 11111 or loss_per_fp == 11111):
        raise ValueError(
            f"Paramètres invalides : penalty_per_fn={penalty_per_fn}, profit_per_tp={profit_per_tp}, loss_per_fp={loss_per_fp}")

    # Calcul du profit total incluant les pénalités FN
    total_profit = (tp * profit_per_tp) + (fp * loss_per_fp) + (fn * penalty_per_fn)

    return float(total_profit), int(tp), int(fp)
"""
def lgb_custom_metric_pnl_factory(
    config=None,
    other_params=None,
):
    """
    Retourne une fonction de métrique LightGBM
    qui accède à config, metric_dict, etc.
    """

    def custom_metric_pnl(preds, train_data):
        """
        Métrique LightGBM pour le profit,
        utilisant config, metric_dict, etc.
        """
        # Récupération du vecteur de labels
        y_true = train_data.get_label()

        # Récupération de la série PnL attachée au dataset
        y_pnl_data = train_data.pnl_data  # On suppose que vous avez fait ltrain.pnl_data = y_pnl_data_train_cv

        # Application de la sigmoid
        preds_sigmoid = 1.0 / (1.0 + np.exp(-preds))
        preds_sigmoid = np.clip(preds_sigmoid, 0.0, 1.0)

        # Récupération d’un paramètre (ex. threshold) dans metric_dict
        threshold = other_params.get('threshold', 0.55555555)
        y_pred_threshold = (preds_sigmoid > threshold).astype(np.int32)

        # Calcul du profit, etc.
        total_profit, tp, fp = calculate_profitBased(
            y_true_class_binaire=y_true,
            y_pred_threshold=y_pred_threshold,
            y_pnl_data_array=y_pnl_data,  # <-- n'existe pas dans la signature
            other_params=other_params,
            config=config
        )
        #print(preds)
        total_profit=(tp * 175) + (fp * -227)
        #print(f"✅ Profit total: {total_profit} | TP: {tp} | FP: {fp}")

        # Troisième élément = True -> "plus c'est grand, mieux c'est"
        return 'custom_metric_PNL', float(total_profit), True

    return custom_metric_pnl





def train_and_evaluate_lightgbm_model(
        X_train_cv=None,
        X_val_cv=None,
        y_train_cv=None,
        y_val_cv=None,
        y_pnl_data_train_cv=None,
        y_pnl_data_val_cv_OrTest=None,
        params_optuna=None,
        other_params=None,
        config=None,
        fold_num=0,
        fold_raw_data=None,
        fold_stats_current=None,
        train_pos=None,
        val_pos=None,
        log_evaluation=0,
):
    ltrain = lgb.Dataset(X_train_cv, label=y_train_cv)
    # on attache y_pnl_data_train_cv dans ltrain
    ltrain.pnl_data = y_pnl_data_train_cv



    lval = lgb.Dataset(X_val_cv, label=y_val_cv)
    # on attache y_pnl_data_val_cv_OrTest dans lval
    lval.pnl_data = y_pnl_data_val_cv_OrTest

    # Configuration des paramètres selon l'objectif
    custom_objective_lossFct = config.get('custom_objective_lossFct', 13)
    if custom_objective_lossFct == model_custom_objective.LGB_CUSTOM_OBJECTIVE_PROFITBASED:
        params_optuna.update({
            'objective': lgb_weighted_logistic_objective(other_params['w_p'], other_params['w_n']),
            'metric': None,
            'device_type': 'cpu'
        })
    elif custom_objective_lossFct == model_custom_objective.LGB_CUSTOM_OBJECTIVE_BINARY:
        params_optuna.update({'objective': 'binary', 'metric': None, 'device_type': 'cpu'})
    elif custom_objective_lossFct == model_custom_objective.LGB_CUSTOM_OBJECTIVE_CROSS_ENTROPY:
        params_optuna.update({'objective': 'cross_entropy', 'metric': None, 'device_type': 'cpu'})
    elif custom_objective_lossFct == model_custom_objective.LGB_CUSTOM_OBJECTIVE_CROSS_ENTROPY_LAMBDA:
        params_optuna.update({'objective': 'cross_entropy_lambda', 'metric': None, 'device_type': 'cpu'})
    else:
        raise ValueError("Choisir une fonction objective / Fonction objective non reconnue")

    if config.get('custom_metric_eval', 13) == model_custom_metric.LGB_CUSTOM_METRIC_PNL:
        custom_metric_function = lgb_custom_metric_pnl_factory(config=config, other_params=other_params)

    else:
        params_optuna.update({'metric': ['auc', 'binary_logloss']})

    params_optuna['early_stopping_rounds'] = config.get('early_stopping_rounds', 13)
    params_optuna['verbose'] = -1
    if 'boosting_type' not in params_optuna:
        raise ValueError("Le paramètre 'boosting_type' n'est pas spécifié dans params_optuna")
    else:
        print(f"Boosting type configuré : {params_optuna['boosting_type']}")
        print(f"num_boost_round : {other_params['num_boost_round']}")

    # Si le boosting_type est dart, on retire tous les paramètres liés à l'early stopping
    if params_optuna['boosting_type'] == 'dart':
        # Paramètres d'early stopping à enlever s'ils existent
        early_stopping_params = ['early_stopping_round', 'early_stopping_rounds', 'early_stopping']

        # Créer une copie du dictionnaire pour éviter de modifier l'original pendant l'itération
        params_copy = params_optuna.copy()

        # Retirer les paramètres d'early stopping
        for param in early_stopping_params:
            if param in params_copy:
                del params_copy[param]
                #print(f"Mode dart détecté: paramètre '{param}' retiré")
    else:
        # Utiliser les paramètres sans modification
        params_copy = params_optuna

    evals_result = {}
    # 3) Appeler l'entraînement
    current_model = lgb.train(
        params=params_optuna,
        train_set=ltrain,
        num_boost_round=other_params['num_boost_round'],
        valid_sets=[ltrain, lval],
        valid_names=['train', 'eval'],
        feval=custom_metric_function,
        callbacks=[
            lgb.record_evaluation(evals_result),
            lgb.log_evaluation(period=log_evaluation)  # Affiche les logs toutes les 10 itérations
        ],
    )
    # Extraction de la meilleure itération
    best_iteration, best_idx, val_best, train_best = get_best_iteration(evals_result,
                                                                        config.get('custom_metric_eval', 13))


    #print(evals_result)
    # Prédictions et métriques
    val_pred_proba, val_pred_proba_log_odds, val_pred, (tn_val, fp_val, fn_val, tp_val), y_val_cv = \
        predict_and_compute_metrics_XgbOrLightGbm(model=current_model, X_data=X_val_cv, y_true=y_val_cv,
                                    best_iteration=best_iteration, threshold=other_params['threshold'],
                                    config=config)

    y_train_predProba, train_pred_proba_log_odds, train_pred, (tn_train, fp_train, fn_train, tp_train), Y_train_cv = \
        predict_and_compute_metrics_XgbOrLightGbm(model=current_model, X_data=X_train_cv, y_true=y_train_cv,
                                    best_iteration=best_iteration, threshold=other_params['threshold'],
                                    config=config)

    print("📊 Dernières prédictions val_pred_proba_log_odds :", val_pred_proba_log_odds[-5:])

    # Calcul du PnL attendu
    pnl_recalcule_val = (tp_val * 175) + (fp_val * -227)
    pnl_recalcule_train = (tp_train * 175) + (fp_train * -227)


    # Affichage des valeurs
    # for i, score in enumerate(evals_result['eval']['custom_metric_PNL']):
    #     print(f"[{i}] Profit = {score}")

    print("train_and_evaluate_lightgbm_model", other_params['threshold'])

    print(f"best_idx={best_idx} val_best : {val_best} ")
    print(f"PnL calculé : {pnl_recalcule_val} | Vérif : TP = {tp_val}, FP = {fp_val}")

    # Vérification
    if val_best == pnl_recalcule_val:
        print("✅ val_best est bien égal au PnL calculé.")
    else:
        print("❌ Mismatch : val_best est différent du PnL calculé.")

    # 📉 Brier Score brut
    brier = brier_score_loss(y_val_cv, val_pred_proba)
    # print("brier score: ",brier)
    # 📊 Brier Score baseline
    baseline_pred = np.full_like(y_val_cv, y_val_cv.mean(), dtype=np.float64)
    baseline_brier = brier_score_loss(y_val_cv, baseline_pred)
    relative_brier = brier / baseline_brier if baseline_brier > 0 else brier

    # Construction des métriques
    val_metrics = {
        'tp': tp_val,
        'fp': fp_val,
        'tn': tn_val,
        'fn': fn_val,
        'total_samples': len(y_val_cv),
        'val_bestVal_custom_metric_pnl': pnl_recalcule_val,
        'best_iteration': best_iteration,
        'brier':brier,
        'relative_brier':relative_brier,
        'y_val_cv':y_val_cv,
        'val_pred_proba_log_odds':val_pred_proba_log_odds,
    }
    train_metrics = {
        'tp': tp_train,
        'fp': fp_train,
        'tn': tn_train,
        'fn': fn_train,
        'total_samples': len(Y_train_cv),
        'train_bestVal_custom_metric_pnl': train_best
    }
    #print(
    #   f"Val metrics: tp={val_metrics['tp']}, fp={val_metrics['fp']}, tn={val_metrics['tn']}, fn={val_metrics['fn']}, samples={val_metrics['total_samples']}, best_pnl={val_metrics['val_bestVal_custom_metric_pnl']:.4f}, best_iter={val_metrics['best_iteration']}")
    #print(
    #   f"Train metrics: tp={train_metrics['tp']}, fp={train_metrics['fp']}, tn={train_metrics['tn']}, fn={train_metrics['fn']}, samples={train_metrics['total_samples']}, best_pnl={train_metrics['train_bestVal_custom_metric_pnl']:.4f}")
    tp_fp_tn_fn_sum_val = tp_val + fp_val + tn_val + fn_val
    tp_fp_tn_fn_sum_train = tp_train + fp_train + tn_train + fn_train
    tp_fp_sum_val = tp_val + fp_val
    tp_fp_sum_train = tp_train + fp_train

    train_trades_samples_perct = round(tp_fp_sum_train / tp_fp_tn_fn_sum_train * 100,
                                       2) if tp_fp_tn_fn_sum_train else 0.00
    val_trades_samples_perct = round(tp_fp_sum_val / tp_fp_tn_fn_sum_val * 100, 2) if tp_fp_tn_fn_sum_val else 0.00
    perctDiff_ratioTradeSample_train_val = abs(
        calculate_ratio_difference(train_trades_samples_perct, val_trades_samples_perct, config))
    winrate_train = compute_winrate_safe(tp_train, tp_fp_sum_train, config)
    winrate_val = compute_winrate_safe(tp_val, tp_fp_sum_val, config)
    ratio_difference = abs(calculate_ratio_difference(winrate_train, winrate_val, config))

    # Compilation des statistiques du fold
    fold_stats = compile_fold_stats(
        fold_num, best_iteration, train_pred_proba_log_odds, train_metrics, winrate_train,
        tp_fp_sum_train, tp_fp_tn_fn_sum_train, train_best, train_pos, train_trades_samples_perct,
        val_pred_proba_log_odds, val_metrics, winrate_val, tp_fp_sum_val, tp_fp_tn_fn_sum_val,
        val_best, val_pos, val_trades_samples_perct, ratio_difference, perctDiff_ratioTradeSample_train_val
    )
    if fold_stats_current is not None:
        fold_stats.update(fold_stats_current)

    debug_info = compile_debug_info(other_params, config, val_pred_proba, y_train_predProba)

    return {
        'current_model': current_model,
        'fold_raw_data': fold_raw_data,
        'y_train_predProba': y_train_predProba,
        'eval_metrics': val_metrics,
        'train_metrics': train_metrics,
        'fold_stats': fold_stats,
        'evals_result': evals_result,
        'best_iteration': best_iteration,
        'val_score_bestIdx': best_idx,
        'debug_info': debug_info,
    }

