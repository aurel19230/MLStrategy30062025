import lightgbm as lgb

from definition import *

from sklearn.metrics import brier_score_loss
import warnings
import logging

# Configure your logger
logger = logging.getLogger(__name__)
def train_and_evaluate_randomforest_model(
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
        fold_stats_current=None,
        train_pos=None,
        val_pos=None,
        log_evaluation=0,
):
    """
    Entraîne et évalue soit un RandomForest sklearn, soit XGBRFClassifier (XGBoost),
    selon use_xgbrf=True/False.
    """
    import numpy as np
    import time
    from sklearn.metrics import confusion_matrix
    model_type = config['model_type']

    # -- 0) Fonction interne pour charger le bon classifieur ------------------
    if model_type == modelType.XGBRF:
        from xgboost import XGBRFClassifier
        # par défaut, pour simuler un "RF" XGBoost, on s'assure learning_rate=1.0
        # et on peut utiliser tree_method='gpu_hist' si on veut le GPU
        # cf. doc : https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBRFClassifier
        ModelClass = XGBRFClassifier
    elif model_type == modelType.RF:
        from sklearn.ensemble import RandomForestClassifier
        ModelClass = RandomForestClassifier
    else:
        raise ValueError(f"Type de modèle non pris en charge: {model_type}")

    # -- 1) Mesure du temps total de la fonction -----------------------------

    # -- 2) Calcul des poids, si besoin --------------------------------------
    sample_weights_train, sample_weights_val = compute_sample_weights(y_train_cv, y_val_cv)

    # -- 3) Entraînement du modèle -------------------------------------------
    current_model = ModelClass(**params_optuna)
    # Vérifier que sample_weight est supporté (XGBRFClassifier le supporte aussi)
    import warnings
    with warnings.catch_warnings(record=True) as caught_warnings:
        current_model.fit(X_train_cv, y_train_cv, sample_weight=sample_weights_train)
        if caught_warnings:
            for warning in caught_warnings:
                # Log the warning
                print(f"Warning during model fitting: {warning.message}")
                # Or use a proper logger
                # logger.warning(f"Warning during model fitting: {warning.message}")

    # -- 4-6) Prédictions et métriques sur Val et Train à l'aide de predict_and_compute_metrics_RF
    threshold = other_params.get('threshold', 0.5)

    # Validation
    val_pred_proba, val_pred_proba_log_odds, val_pred, tn_val, fp_val, fn_val, tp_val, y_val_converted = predict_and_compute_metrics_RF(
        model=current_model,
        X_data=X_val_cv,
        y_true=y_val_cv,
        threshold=threshold,
        config=config
    )

    # Train
    y_train_predProba, train_pred_proba_log_odds, train_pred, tn_train, fp_train, fn_train, tp_train, y_train_converted = predict_and_compute_metrics_RF(
        model=current_model,
        X_data=X_train_cv,
        y_true=y_train_cv,
        threshold=threshold,
        config=config
    )

    # -- 7) Calcul PNL sur Validation ---------------------------------------
    val_pnl = 0
    if y_pnl_data_val_cv_OrTest is not None:
        tp_mask_val = (y_val_cv == 1) & (val_pred == 1)
        fp_mask_val = (y_val_cv == 0) & (val_pred == 1)
        val_pnl = np.sum(y_pnl_data_val_cv_OrTest[tp_mask_val]) + np.sum(y_pnl_data_val_cv_OrTest[fp_mask_val])

    # -- 8) Calcul PNL sur Train --------------------------------------------
    train_pnl = 0
    if y_pnl_data_train_cv is not None:
        tp_mask_train = (y_train_cv == 1) & (train_pred == 1)
        fp_mask_train = (y_train_cv == 0) & (train_pred == 1)
        train_pnl = np.sum(y_pnl_data_train_cv[tp_mask_train]) + np.sum(y_pnl_data_train_cv[fp_mask_train])

    # -- 9) Best iteration (pas vraiment utile pour RF, mais conservé) ------
    best_iteration = None
    best_idx = None

    # 📉 Brier Score brut
    brier = brier_score_loss(y_val_cv, val_pred_proba)

    # 📊 Brier Score baseline
    baseline_pred = np.full_like(y_val_cv, y_val_cv.mean(), dtype=np.float64)
    baseline_brier = brier_score_loss(y_val_cv, baseline_pred)
    relative_brier = brier / baseline_brier if baseline_brier > 0 else brier

    # -- 10) Construit les dictionnaires de métriques -----------------------
    start_time_metrics_build = time.time()
    val_metrics = {
        'tp': tp_val,
        'fp': fp_val,
        'tn': tn_val,
        'fn': fn_val,
        'total_samples': len(y_val_cv),
        'val_bestVal_custom_metric_pnl': val_pnl,
        'best_iteration': best_iteration,
        'brier': brier,
        'relative_brier': relative_brier,
    }

    train_metrics = {
        'tp': tp_train,
        'fp': fp_train,
        'tn': tn_train,
        'fn': fn_train,
        'total_samples': len(y_train_cv),
        'train_bestVal_custom_metric_pnl': train_pnl
    }

    # Quelques stats globales
    tp_fp_tn_fn_sum_val = tp_val + fp_val + tn_val + fn_val
    tp_fp_tn_fn_sum_train = tp_train + fp_train + tn_train + fn_train
    tp_fp_sum_val = tp_val + fp_val
    tp_fp_sum_train = tp_train + fp_train

    train_trades_samples_perct = round(tp_fp_sum_train / tp_fp_tn_fn_sum_train * 100,
                                       2) if tp_fp_tn_fn_sum_train else 0.00
    val_trades_samples_perct = round(tp_fp_sum_val / tp_fp_tn_fn_sum_val * 100, 2) if tp_fp_tn_fn_sum_val else 0.00

    perctDiff_ratioTradeSample_train_val = abs(
        calculate_ratio_difference(train_trades_samples_perct, val_trades_samples_perct, config)
    )
    winrate_train = compute_winrate_safe(tp_train, tp_fp_sum_train, config)
    winrate_val = compute_winrate_safe(tp_val, tp_fp_sum_val, config)
    ratio_difference = abs(calculate_ratio_difference(winrate_train, winrate_val, config))

    fold_stats = compile_fold_stats(
        fold_num, best_iteration, train_pred_proba_log_odds, train_metrics, winrate_train,
        tp_fp_sum_train, tp_fp_tn_fn_sum_train, train_pnl, train_pos, train_trades_samples_perct,
        val_pred_proba_log_odds, val_metrics, winrate_val, tp_fp_sum_val, tp_fp_tn_fn_sum_val,
        val_pnl, val_pos, val_trades_samples_perct, ratio_difference, perctDiff_ratioTradeSample_train_val
    )

    if fold_stats_current is not None:
        fold_stats.update(fold_stats_current)

    # -- 11) Temps total ----------------------------------------------------

    # -- 12) Feature importances -------------------------------------------
    # (XGBRFClassifier possède .feature_importances_, tout comme sklearn)
    # importances = current_model.feature_importances_
    # feature_importances_dict = dict(zip(range(X_train_cv.shape[1]), importances))

    # -- 13) Debug info ----------------------------------------------------
    debug_info = {
        'model_params': params_optuna,
        'threshold': threshold,
      #  'feature_importances': feature_importances_dict,
    }

    return {
        'current_model': current_model,
        'y_train_predProba': y_train_predProba,
        'eval_metrics': val_metrics,
        'train_metrics': train_metrics,
        'fold_stats': fold_stats,
        'evals_result': None,
        'best_iteration': best_iteration,
        'val_score_bestIdx': best_idx,
        'debug_info': debug_info,
    }
