"""Read-only evaluator: does monotone (isotonic) calibration of the v5 ensemble
blend honestly improve the LOSO-CV R²?

This NEVER touches the feature pipeline or the backend — it only reads the
already-computed per-model out-of-fold predictions in models/loso_oof.npz, builds
the production blend (using the chosen weights in metrics.json), and asks:

  Does a MONOTONE post-hoc calibrator g(blend) -> y, fit ONLY on held-out
  sensors (GroupKFold-over-sensors), raise the pooled LOSO R²?

Isotonic regression (PAVA) finds the monotone g minimizing held-out MSE; it can
ONLY help if the blend has genuine monotone miscalibration (tree ensembles shrink
toward the mean and under-predict extremes). Fitting per grouped fold and scoring
on the held-out fold makes the reported gain optimism-free and paper-defensible
(van der Laan Super-Learner / Platt calibration).

Run AFTER training completes:
    python pipeline/eval_calibration.py
It prints the uncalibrated vs grouped-CV-calibrated LOSO R². It does NOT modify
any artifact — wiring the calibrator into the bundle is a separate, deliberate
step taken only if v5 needs the margin.
"""
import os
import json

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")


def load_oof():
    p = os.path.join(MODELS_DIR, "loso_oof.npz")
    if not os.path.exists(p):
        raise SystemExit(f"missing {p} — run training first (loso_cv saves it).")
    z = np.load(p, allow_pickle=True)
    model_names = [str(m) for m in z["model_names"]]
    oof = {n: z[f"oof_{n}"] for n in model_names}
    return z["y"], z["valid"], z["sensor_id"], model_names, oof


def chosen_weights(model_names):
    """Use the LOSO-optimized weights the trainer chose (metrics.json). Falls
    back to equal weight if absent."""
    mp = os.path.join(MODELS_DIR, "metrics.json")
    try:
        m = json.load(open(mp))
        w = m["loso_cv_optimized"]["weights"]
        return {n: float(w.get(n, 0.0)) for n in model_names}
    except Exception:
        return {n: 1.0 / len(model_names) for n in model_names}


def grouped_isotonic_r2(blend, y, groups, n_splits=5):
    """Pooled LOSO R² after monotone calibration fit on held-out sensors."""
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    cal = np.full(len(y), np.nan)
    for tr, te in gkf.split(blend, y, groups):
        ir = IsotonicRegression(out_of_bounds="clip", increasing=True)
        ir.fit(blend[tr], y[tr])
        cal[te] = ir.predict(blend[te])
    return cal


def main():
    y_all, valid, sid_all, model_names, oof = load_oof()
    stack = np.column_stack([oof[n] for n in model_names])
    row_valid = np.asarray(valid) & np.all(np.isfinite(stack), axis=1)
    P = stack[row_valid]
    y = np.asarray(y_all)[row_valid]
    groups = np.asarray(sid_all)[row_valid]

    w = chosen_weights(model_names)
    wv = np.array([w[n] for n in model_names])
    wv = wv / wv.sum()
    blend = P @ wv

    print("=" * 66)
    print("ISOTONIC CALIBRATION EVALUATION (read-only, on LOSO OOF)")
    print("=" * 66)
    print(f"  rows={len(y):,}  sensors={len(np.unique(groups))}  models={model_names}")
    print(f"  chosen weights: " + "  ".join(f"{n.upper()}:{w[n]:.3f}" for n in model_names))

    r2_un = r2_score(y, blend)
    rmse_un = np.sqrt(mean_squared_error(y, blend))
    mae_un = mean_absolute_error(y, blend)
    print(f"\n  UNCALIBRATED blend   R²={r2_un:.4f}  RMSE={rmse_un:.4f}  MAE={mae_un:.4f}")

    cal = grouped_isotonic_r2(blend, y, groups)
    r2_cal = r2_score(y, cal)
    rmse_cal = np.sqrt(mean_squared_error(y, cal))
    mae_cal = mean_absolute_error(y, cal)
    print(f"  CALIBRATED (grpCV)   R²={r2_cal:.4f}  RMSE={rmse_cal:.4f}  MAE={mae_cal:.4f}")
    print(f"\n  Δ from calibration:  R² {r2_cal - r2_un:+.4f}   (honest, grouped-CV, no optimism)")

    if r2_cal > r2_un:
        print(f"\n  → calibration HELPS by {r2_cal - r2_un:+.4f}. If v5 needs the margin,")
        print(f"    fit a FINAL isotonic on ALL OOF and store its (x,y) knots in the")
        print(f"    bundle, applied right after the weighted blend at inference.")
    else:
        print(f"\n  → calibration does NOT help (blend already well-calibrated). Skip it.")
    print(f"\n  HEADLINE: uncal={r2_un:.4f}  cal={r2_cal:.4f}")


if __name__ == "__main__":
    main()
