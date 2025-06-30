import numpy as np
import xgboost as xgb
import warnings
from definition import *
from typing import Tuple
import logging
from sklearn.metrics import brier_score_loss

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.expand_frame_repr', False)
# Fonctions GPU mises à jour







# Fonction pour vérifier la disponibilité du GPU

def xgb_calculate_profitBased_gpu(y_true_gpu, y_pred_threshold_gpu, metric_dict):
    """
    Calcule les métriques de profit directement sur GPU sans conversions inutiles

    Args:
        y_true_gpu: cp.ndarray - Labels déjà sur GPU
        y_pred_threshold_gpu: cp.ndarray - Prédictions déjà sur GPU
        metric_dict: dict - Dictionnaire des paramètres de métrique
    """
    # Vérification que les entrées sont bien sur GPU
    if not isinstance(y_true_gpu, cp.ndarray):
        raise TypeError("y_true_gpu doit être un tableau CuPy")
    if not isinstance(y_pred_threshold_gpu, cp.ndarray):
        raise TypeError("y_pred_threshold_gpu doit être un tableau CuPy")

    # Calcul des métriques de base
    tp = cp.sum((y_true_gpu == 1) & (y_pred_threshold_gpu == 1))
    fp = cp.sum((y_true_gpu == 0) & (y_pred_threshold_gpu == 1))
    fn = cp.sum((y_true_gpu == 1) & (y_pred_threshold_gpu == 0))

    # Récupération des paramètres de profit/perte
    profit_per_tp = metric_dict.get('profit_per_tp', 1.0)
    loss_per_fp = metric_dict.get('loss_per_fp', -1.1)
    penalty_per_fn = metric_dict.get('penalty_per_fn', -0.1)

    # Calcul du profit total incluant les pénalités FN
    total_profit = (tp * profit_per_tp) + (fp * loss_per_fp) + (fn * penalty_per_fn)

    return float(total_profit), int(tp), int(fp)


def xgb_custom_metric_Profit(predt: np.ndarray, dtrain: xgb.DMatrix, metric_dict, normalize: bool = False) -> Tuple[
    str, float]:
    """Fonction commune pour calculer les métriques de profit"""
    # Conversion des données en GPU une seule fois
    y_true_gpu = cp.asarray(dtrain.get_label())
    predt_gpu = cp.asarray(predt)

    # Application de la sigmoid et normalisation
    predt_gpu = sigmoidCustom(predt_gpu)
    predt_gpu = cp.clip(predt_gpu, 0.0, 1.0)

    # Vérification des prédictions
    mean_pred = cp.mean(predt_gpu).item()
    std_pred = cp.std(predt_gpu).item()
    min_val = cp.min(predt_gpu).item()
    max_val = cp.max(predt_gpu).item()

    if min_val < 0 or max_val > 1:
        logging.warning(f"Prédictions hors intervalle [0, 1]: [{min_val:.4f}, {max_val:.4f}]")
        return "custom_metric_ProfitBased", float('-inf')

    # Application du seuil directement sur GPU
    threshold = metric_dict.get('threshold', 0.55555555)
    y_pred_threshold_gpu = (predt_gpu > threshold).astype(cp.int32)

    # Calcul du profit et des métriques
    total_profit, tp, fp = xgb_calculate_profitBased_gpu(y_true_gpu, y_pred_threshold_gpu, metric_dict)

    # Normalisation si demandée
    if normalize:
        total_trades = tp + fp
        final_profit = total_profit / total_trades if total_trades > 0 else total_profit
        metric_name = 'custom_metric_ProfitBased_norm'
    else:
        final_profit = total_profit
        metric_name = 'custom_metric_ProfitBased'

    return metric_name, float(final_profit)

# Création des deux fonctions spécifiques à partir de la fonction commune
def xgb_custom_metric_ProfitBased_gpu(predt: np.ndarray, dtrain: xgb.DMatrix, metric_dict) -> Tuple[str, float]:
    """Version GPU de la métrique de profit"""
    return xgb_custom_metric_Profit(predt, dtrain, metric_dict, normalize=False)


def xgb_calculate_profitBased_cpu(y_true, y_pred_threshold, metric_dict):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred_threshold)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    profit_per_tp = metric_dict.get('profit_per_tp', 1.0)
    loss_per_fp = metric_dict.get('loss_per_fp', -1.1)
    penalty_per_fn = metric_dict.get('penalty_per_fn', -0.1)
    total_profit = (tp * profit_per_tp) + (fp * loss_per_fp) + (fn * penalty_per_fn)
    return float(total_profit), int(tp), int(fp)

""""""
def xgb_custom_metric_Profit_cpu(predt: np.ndarray, dtrain: xgb.DMatrix, metric_dict, normalize: bool = False) -> Tuple[str, float]:
    y_true = dtrain.get_label()
    CHECK_THRESHOLD = 0.55555555
    threshold = metric_dict.get('threshold', CHECK_THRESHOLD)

    predt = np.array(predt)
    predt = sigmoidCustom_cpu(predt)
    predt = np.clip(predt, 0.0, 1.0)

    mean_pred = np.mean(predt)
    std_pred = np.std(predt)
    min_val = np.min(predt)
    max_val = np.max(predt)

    if min_val < 0 or max_val > 1:
        logging.warning(f"Les prédictions sont hors de l'intervalle [0, 1]: [{min_val:.4f}, {max_val:.4f}]")
        exit(12)

    y_pred_threshold = (predt > threshold).astype(int)

    total_profit, tp, fp = xgb_calculate_profitBased_cpu(y_true, y_pred_threshold, metric_dict)

    if normalize:
        total_trades_val = tp + fp
        if total_trades_val > 0:
            final_profit = total_profit / total_trades_val
        else:
            final_profit = 0.0
        metric_name = 'custom_metric_ProfitBased_norm'
    else:
        final_profit = total_profit
        metric_name = 'custom_metric_ProfitBased'

    return metric_name, float(final_profit)


# Métrique personnalisée pour XGBoost
# Métrique personnalisée pour XGBoost
def xgb_custom_metric_PNL(metric_dict=None, config=None):
    def profit_metric(preds, dset):
        # y_true
        y_true = dset.get_label()
        # y_pnl_data correspondant *à ce dataset*
        y_pnl_data_array = dset.pnl_data

        # Sigmoid + clip
        preds_sigmoid = 1.0 / (1.0 + np.exp(-preds))
        preds_sigmoid = np.clip(preds_sigmoid, 0.0, 1.0)

        # Threshold
        threshold = metric_dict.get('threshold', 0.55555555)
        y_pred_threshold = (preds_sigmoid > threshold).astype(int)

        # Calcul du PnL
        total_profit, tp, fp = calculate_profitBased(
            y_true_class_binaire=y_true,
            y_pred_threshold=y_pred_threshold,
            y_pnl_data_array=y_pnl_data_array,  # pas un param global !
            other_params=metric_dict,
            config=config
        )

        # ...
        return "custom_metric_PNL", float(total_profit)

    return profit_metric



# Fonction objective XGBoost - Version ajustée
def xgb_create_weighted_logistic_obj_cpu(w_p: float, w_n: float):

    def weighted_logistic_obj(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
        y = dtrain.get_label()
        #print(f"weighted_logistic_obj y:{y} \ predt:{predt} \ w_p:{w_p} \ w_n:{w_n}")
        predt = sigmoidCustom_cpu(predt)

        # print(f"{w_p} {w_n}")

        # Calcul du gradient et de la hessienne avec les poids
        weights = np.where(y == 1, w_p, w_n)
        # print(f"{w_p} {w_n} => {weights}")
        grad = weights * (predt - y)
        hess = weights * predt * (1.0 - predt)
        # 👉 Correction ici : plus de .get()
        #print("Moyenne hessienne =", np.mean(hess))
        return grad, hess

    return weighted_logistic_obj
#
# def train_and_evaluate_xgb_model(
#         X_train_cv=None,
#         X_val_cv=None,
#         Y_train_cv=None,
#         y_val_cv=None,
#         y_pnl_data_train_cv=None,
#         y_pnl_data_val_cv_OrTest=None,
#         params_optuna=None,
#         other_params=None,
#         config=None,
#         fold_num=0,
#         fold_stats_current=None,
#         train_pos=None,
#         val_pos=None,
#         log_evaluation=0,
# ):
#     import numpy as np
#     import xgboost as xgb
#
#     # Calcul des poids
#     sample_weights_train, sample_weights_val = compute_sample_weights(Y_train_cv, y_val_cv)
#
#     # Logs sur les PnL
#     y = y_pnl_data_val_cv_OrTest
#     total = len(y)
#     nb_pos = np.sum(y > 0)
#     nb_neg = np.sum(y < 0)
#     mean_pos = np.mean(y[y > 0]) if nb_pos > 0 else 0
#     mean_neg = np.mean(y[y < 0]) if nb_neg > 0 else 0
#     # print(f"Total y_pnl_data_val_cv_OrTest: {total:,} | Pos: {nb_pos:,} (mean: {mean_pos:.4f}) | Neg: {nb_neg:,} (mean: {mean_neg:.4f})")
#
#     dtrain = xgb.DMatrix(X_train_cv, label=Y_train_cv, weight=sample_weights_train)
#     dtrain.pnl_data = y_pnl_data_train_cv  # <--- Aligné sur TRAIN (1540 lignes)
#
#     dval = xgb.DMatrix(X_val_cv, label=y_val_cv, weight=sample_weights_val)
#     dval.pnl_data = y_pnl_data_val_cv_OrTest  # <--- Aligné sur VAL (616 lignes)
#
#     # Objectif personnalisé
#     custom_objective_lossFct = config.get('custom_objective_lossFct', 13)
#     obj_function = None
#     if custom_objective_lossFct == model_custom_objective.XGB_CUSTOM_OBJECTIVE_PROFITBASED:
#         obj_function = xgb_create_weighted_logistic_obj_cpu(other_params['w_p'], other_params['w_n'])
#         params_optuna['disable_default_eval_metric'] = 1
#
#     # Métrique personnalisée
#     if config.get('custom_metric_eval', 13) == model_custom_metric.XGB_CUSTOM_METRIC_PNL:
#         custom_metric = xgb_custom_metric_PNL(
#             metric_dict=other_params,  # <= threshold, etc.
#             config=config
#         )
#     else:
#         custom_metric = None
#         params_optuna.update({'eval_metric': ['auc', 'logloss']})
#
#     #params_optuna['booster'] = "gbtree"  # ou "dart", "gblinear", etc.
#
#     # Entraînement
#     evals_result = {}
#     current_model = xgb.train(
#         params=params_optuna,
#         dtrain=dtrain,
#         num_boost_round=other_params['num_boost_round'],
#         evals=[(dtrain, 'train'), (dval, 'eval')],
#         obj=obj_function,
#         custom_metric=custom_metric,
#         early_stopping_rounds=config.get('early_stopping_rounds', 13),
#         verbose_eval=log_evaluation,
#         evals_result=evals_result,
#         maximize=True
#     )
#
#     # Best itération
#     best_iteration, best_idx, val_best, train_best = get_best_iteration(
#         evals_result, config.get('custom_metric_eval', 13)
#     )
#
#     # Prédictions sur validation
#     val_pred_proba, val_pred_proba_log_odds, val_pred, (tn_val, fp_val, fn_val, tp_val), y_val_cv = \
#         predict_and_compute_metrics_XgbOrLightGbm(
#             model=current_model,
#             X_data=dval,
#             y_true=y_val_cv,
#             best_iteration=best_iteration,
#             threshold=other_params['threshold'],
#             config=config
#         )
#
#     # print(f"[XGB Fold {fold_num}] Validation Confusion: TN={tn_val:,} | FP={fp_val:,} | FN={fn_val:,} | TP={tp_val:,}")
#
#     # Prédictions sur train
#     y_train_predProba, train_pred_proba_log_odds, train_pred, (tn_train, fp_train, fn_train, tp_train), Y_train_cv = \
#         predict_and_compute_metrics_XgbOrLightGbm(
#             model=current_model,
#             X_data=dtrain,
#             y_true=Y_train_cv,
#             best_iteration=best_iteration,
#             threshold=other_params['threshold'],
#             config=config
#         )
#     # 📉 Brier Score brut
#     brier = brier_score_loss(y_val_cv, val_pred_proba)
#
#     # 📊 Brier Score baseline
#     baseline_pred = np.full_like(y_val_cv, y_val_cv.mean(), dtype=np.float64)
#     baseline_brier = brier_score_loss(y_val_cv, baseline_pred)
#     relative_brier = brier / baseline_brier if baseline_brier > 0 else brier
#
#     pnl_recalcul = (tp_val * 175) + (fp_val * -227)
#
#     # if abs(pnl_recalcul - val_best) < 1e-6:
#     #     print(f"✅ PnL match : val_best = {val_best}, recalculé = {pnl_recalcul}")
#     # else:
#     #     print(f"❌ PnL mismatch : val_best = {val_best}, recalculé = {pnl_recalcul}")
#
#     # Construction des métriques
#     val_metrics = {
#         'tp': tp_val,
#         'fp': fp_val,
#         'tn': tn_val,
#         'fn': fn_val,
#         'total_samples': len(y_val_cv),
#         'val_bestVal_custom_metric_pnl': val_best,
#         'best_iteration': best_iteration,
#         'brier':brier,
#         'relative_brier':relative_brier,
#         'X_val_cv':X_val_cv,
#         'y_val_cv': y_val_cv,
#         'val_pred_proba_log_odds': val_pred_proba_log_odds,
#     }
#     train_metrics = {
#         'tp': tp_train,
#         'fp': fp_train,
#         'tn': tn_train,
#         'fn': fn_train,
#         'total_samples': len(Y_train_cv),
#         'train_bestVal_custom_metric_pnl': train_best
#     }
#
#     # Calculs complémentaires (winrate, trades, pourcentages, etc.)
#     tp_fp_tn_fn_sum_val = tp_val + fp_val + tn_val + fn_val
#     tp_fp_tn_fn_sum_train = tp_train + fp_train + tn_train + fn_train
#     tp_fp_sum_val = tp_val + fp_val
#     tp_fp_sum_train = tp_train + fp_train
#
#     train_trades_samples_perct = round(tp_fp_sum_train / tp_fp_tn_fn_sum_train * 100,
#                                        2) if tp_fp_tn_fn_sum_train else 0.00
#     val_trades_samples_perct = round(tp_fp_sum_val / tp_fp_tn_fn_sum_val * 100, 2) if tp_fp_tn_fn_sum_val else 0.00
#     perctDiff_ratioTradeSample_train_val = abs(
#         calculate_ratio_difference(train_trades_samples_perct, val_trades_samples_perct, config))
#     winrate_train = compute_winrate_safe(tp_train, tp_fp_sum_train, config)
#     winrate_val = compute_winrate_safe(tp_val, tp_fp_sum_val, config)
#     ratio_difference = abs(calculate_ratio_difference(winrate_train, winrate_val, config))
#
#     # Compilation des statistiques du fold
#     fold_stats = compile_fold_stats(
#         fold_num, best_iteration, train_pred_proba_log_odds, train_metrics, winrate_train,
#         tp_fp_sum_train, tp_fp_tn_fn_sum_train, train_best, train_pos, train_trades_samples_perct,
#         val_pred_proba_log_odds, val_metrics, winrate_val, tp_fp_sum_val, tp_fp_tn_fn_sum_val,
#         val_best, val_pos, val_trades_samples_perct, ratio_difference, perctDiff_ratioTradeSample_train_val
#     )
#     if fold_stats_current is not None:
#         fold_stats.update(fold_stats_current)
#
#     debug_info = compile_debug_info(other_params, config, val_pred_proba, y_train_predProba)
#
#     # Retour de l'ensemble
#     return {
#         'current_model': current_model,
#         'y_train_predProba': y_train_predProba,
#         'eval_metrics': val_metrics,
#         'train_metrics': train_metrics,
#         'fold_stats': fold_stats,
#         'evals_result': evals_result,
#         'best_iteration': best_iteration,
#         'val_score_bestIdx': best_idx,
#         'debug_info': debug_info,
#     }
#

def train_and_evaluate_xgb_model(
        X_train_cv=None,
        X_val_cv=None,
        Y_train_cv=None,
        y_val_cv=None,
        y_pnl_data_train_cv=None,
        y_pnl_data_val_cv_OrTest=None,
        params_optuna=None,
        other_params=None,
        config=None,
        fold_num: int = 0,
        fold_stats_current: dict | None = None,
        train_pos=None,
        val_pos=None,
        log_evaluation: int = 0,
):
    """
    Entraîne XGBoost sur un fold (ou tout le train pour l'entraînement final).
    - Si X_val_cv / y_val_cv sont None → mode entraînement final (pas de cv).
    - Calibre (option), sweep le seuil, renvoie toutes les métriques.
    """
    import numpy as np
    import xgboost as xgb
    from sklearn.metrics import brier_score_loss

    calibration_method    = config.get("calibration_method", "isotonic").lower()
    threshold_sweep_steps = config.get("threshold_sweep_steps", 200)
    profit_per_tp             = other_params.get("profit_per_tp", 175)
    loss_per_fp               = other_params.get("loss_per_fp", -227)
    penalty_per_fn            = other_params.get("penalty_per_fn", 0.0)

    # ─────────────── DMatrix ──────────────────────────────────────
    dtrain = xgb.DMatrix(X_train_cv, label=Y_train_cv)
    dtrain.pnl_data = y_pnl_data_train_cv

    if X_val_cv is not None and y_val_cv is not None:
        dval = xgb.DMatrix(X_val_cv, label=y_val_cv)
        dval.pnl_data = y_pnl_data_val_cv_OrTest
    else:
        dval = None

    # ─────────── Objectif / métrique custom ──────────────────────
    obj_fn, custom_metric = None, None
    if config.get('custom_objective_lossFct', 13) == model_custom_objective.XGB_CUSTOM_OBJECTIVE_PROFITBASED:
        obj_fn = xgb_create_weighted_logistic_obj_cpu(other_params['w_p'], other_params['w_n'])
        params_optuna['disable_default_eval_metric'] = 1

    if config.get('custom_metric_eval', 13) == model_custom_metric.XGB_CUSTOM_METRIC_PNL:
        custom_metric = xgb_custom_metric_PNL(metric_dict=other_params, config=config)
        params_optuna['disable_default_eval_metric'] = 1
    else:
        params_optuna.update({'eval_metric': ['auc', 'logloss']})

    # ───────────── Entraînement XGBoost ───────────────────────────
    evals_result = {}
    evals = [(dtrain, 'train')]
    if dval is not None:
        evals.append((dval, 'eval'))
        early_stop = config.get('early_stopping_rounds', 13)
        num_round  = other_params['num_boost_round']

    else:  # mode entraînement final
        early_stop = None
        num_round  = other_params['num_boost_round_finalTrain']
        print("num_boost_round_finalTrain: ",num_round)
        print("threshold: ", other_params["threshold"])

    model = xgb.train(
        params=params_optuna,
        dtrain=dtrain,
        num_boost_round=num_round,
        evals=evals,
        obj=obj_fn,
        custom_metric=custom_metric,
        early_stopping_rounds=early_stop,
        verbose_eval=log_evaluation,
        evals_result=evals_result,
        maximize=True
    )

    # ─────────── Si pas de validation, retourner early ───────────
    if dval is None:
        return {
            'current_model': model,
            'evals_result' : evals_result,
            'best_iteration_fold': None,
            'best_thresh_fold'  : None,
            'calibrator'        : None,
            'fold_stats'        : {},
            'train_metrics'     : {},
            'eval_metrics'      : {},
            'y_train_predProba' : None,
            'fold_raw_data'     : {},
        }

    # ───────────── Best iteration & borne sûre ──────────────────────
    best_it, best_idx, _, _ = get_best_iteration(
        evals_result, config.get('custom_metric_eval', 13)
    )
    best_it = (best_it if best_it is not None
               else getattr(model, "best_iteration", None)
                    or getattr(model, "best_ntree_limit", 0) - 1)

    # total d'arbres, cross-version
    n_boost = getattr(model, "num_boost_round", None)
    if n_boost is None:
        n_boost = getattr(model, "num_boosted_rounds", None)
        if n_boost is not None and callable(n_boost):
            n_boost = n_boost()  # Call the method to get the value
    if n_boost is None:  # fallback absolu
        n_boost = len(model.get_dump())

    end_tree = min(best_it, n_boost - 1)  # borne sûre

    # ═════ 1) Calibration + Sweep sur validation ════════════════════
    val_log_odds = model.predict(dval, iteration_range=(0, end_tree + 1), output_margin=True)

    if calibration_method == "none":
        val_proba_cal = 1.0 / (1.0 + np.exp(-val_log_odds))
        calibrator = None
    else:
        val_proba_cal, calibrator = calibrate_probabilities(
            val_log_odds, y_val_cv, method=calibration_method
        )

    best_thresh_fold, thresh_grid, pnl_curve = sweep_threshold_pnl(
        y_val_cv, val_proba_cal,
        profit_tp=profit_per_tp, loss_fp=loss_per_fp,
        penalty_fn=penalty_per_fn, n_steps=threshold_sweep_steps
    )

    # ═════ 2) Prédictions + confusion matrix ════════════════════════
    def _predict_and_conf(bst, dmat, y_true, thr):
        log_odds = bst.predict(dmat, iteration_range=(0, end_tree + 1), output_margin=True)
        proba    = 1.0 / (1.0 + np.exp(-log_odds))
        preds    = (proba >= thr).astype(int)
        tn, fp, fn, tp = compute_confusion_matrix_cpu(y_true, preds, config)
        return proba, log_odds, preds, (tn, fp, fn, tp)

    val_proba, val_log_odds2, _, (tn_val, fp_val, fn_val, tp_val) = \
        _predict_and_conf(model, dval, y_val_cv, best_thresh_fold)
    tr_proba, tr_log_odds,  _, (tn_tr, fp_tr, fn_tr, tp_tr) = \
        _predict_and_conf(model, dtrain, Y_train_cv, best_thresh_fold)

    # ═════ 3) PnL recalculé (seuil sweep) ════════════════════════════
    pnl_val = tp_val * profit_per_tp + fp_val * loss_per_fp + fn_val * penalty_per_fn
    pnl_tr  = tp_tr  * profit_per_tp + fp_tr  * loss_per_fp + fn_tr  * penalty_per_fn
    print(f"[Fold {fold_num}] Max PnL sweep = {pnl_curve.max():.0f} | "
          f"PnL recalculé = {pnl_val:.0f} | Seuil optimal = {best_thresh_fold:.4f}")

    # ═════ 4) Brier score ═══════════════════════════════════════════
    brier      = brier_score_loss(y_val_cv, val_proba_cal)
    rel_brier  = brier / brier_score_loss(y_val_cv, np.full_like(y_val_cv, y_val_cv.mean()))

    # ═════ 5) Métriques dict ════════════════════════════════════════
    val_metrics = {
        'tp': tp_val, 'fp': fp_val, 'tn': tn_val, 'fn': fn_val,
        'total_samples': len(y_val_cv),
        'val_bestVal_custom_metric_pnl': pnl_val,
        'best_iteration': best_it,
        'brier': brier, 'relative_brier': rel_brier,
        'calibration_method': calibration_method,
        'best_thresh_fold': best_thresh_fold,
        'threshold_grid': thresh_grid,
        'pnl_curve': pnl_curve,
        'val_proba_calibrated': val_proba_cal,
        'X_val_cv': X_val_cv,
        'y_val_cv': y_val_cv,

        'val_pred_proba_log_odds': val_log_odds2  # Add this line

    }
    train_metrics = {
        'tp': tp_tr, 'fp': fp_tr, 'tn': tn_tr, 'fn': fn_tr,
        'total_samples': len(Y_train_cv),
        'train_bestVal_custom_metric_pnl': pnl_tr
    }

    # ═════ 6) Stats supplémentaires (win-rate, %) ═══════════════════
    winrate_tr  = compute_winrate_safe(tp_tr, tp_tr + fp_tr, config)
    winrate_val = compute_winrate_safe(tp_val, tp_val + fp_val, config)
    fold_stats = compile_fold_stats(
        fold_num, best_it,
        tr_log_odds, train_metrics, winrate_tr,
        tp_tr + fp_tr, len(Y_train_cv), pnl_tr, train_pos,
        (tp_tr + fp_tr) / len(Y_train_cv) * 100,
        val_log_odds2, val_metrics, winrate_val,
        tp_val + fp_val, len(y_val_cv), pnl_val, val_pos,
        (tp_val + fp_val) / len(y_val_cv) * 100,
        abs(winrate_tr - winrate_val),
        abs((tp_tr + fp_tr) / len(Y_train_cv) * 100 - (tp_val + fp_val) / len(y_val_cv) * 100)
    )
    if fold_stats_current:
        fold_stats.update(fold_stats_current)

    fold_raw_data = {
        'distributions': {
            'train': {0: int(np.sum(Y_train_cv == 0)), 1: int(np.sum(Y_train_cv == 1))},
            'val'  : {0: int(np.sum(y_val_cv == 0)),   1: int(np.sum(y_val_cv == 1))}
        }
    }

    # ═════ 7) Retour ════════════════════════════════════════════════
    return {
        'current_model'   : model,
        'y_train_predProba': tr_proba,
        'eval_metrics'    : val_metrics,
        'train_metrics'   : train_metrics,
        'fold_stats'      : fold_stats,
        'evals_result'    : evals_result,
        'best_iteration_fold'  : best_it,
        'val_score_bestIdx': best_idx,
        'debug_info'      : compile_debug_info(other_params, config, val_proba, tr_proba),
        'best_thresh_fold': best_thresh_fold,
        'calibrator'      : calibrator,
        'fold_raw_data'   : fold_raw_data,
    }
