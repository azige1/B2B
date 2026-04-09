import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime

import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)
VENDOR_DIR = os.path.join(PROJECT_ROOT, ".vendor_tree_backends")
if os.path.isdir(VENDOR_DIR):
    if VENDOR_DIR in sys.path:
        sys.path.remove(VENDOR_DIR)
    sys.path.insert(0, VENDOR_DIR)

from src.analysis.phase_eval_utils import evaluate_context_csv, evaluate_context_frame
from src.models.tabular_hurdle import TabularHurdleModel


OUT_DIR = os.path.join(PROJECT_ROOT, "reports", "phase6f_tree_family_compare")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models_phase6f_tree_family_compare")
TIMING_LOG = os.path.join(OUT_DIR, "phase6f_timing_log.csv")
STAGEA_JSON = os.path.join(OUT_DIR, "phase6f_stagea_smoke.json")
STAGE_STATE_JSON = os.path.join(OUT_DIR, "phase6f_stage_state.json")
SMOKE_TABLE = os.path.join(OUT_DIR, "phase6f_single_anchor_smoke_table.csv")
RAW_ANCHOR_TABLE = os.path.join(OUT_DIR, "phase6f_raw_anchor_table.csv")
CALIBRATED_ANCHOR_TABLE = os.path.join(OUT_DIR, "phase6f_calibrated_anchor_table.csv")

PHASE54_PHASE_DIR = os.path.join(PROJECT_ROOT, "reports", "phase5_4", "phase5")
PHASE55_DIR = os.path.join(PROJECT_ROOT, "reports", "phase5_5")
PHASE57_DIR = os.path.join(PROJECT_ROOT, "reports", "phase5_7")

ANCHORS = ["2025-09-01", "2025-10-01", "2025-11-01", "2025-12-01"]
SMOKE_ANCHOR = "2025-12-01"
SEED = "2026"
VAL_MODE = os.environ.get("PHASE6F_VAL_MODE", "single_anchor").strip().lower()
RUN_SUMMARY = os.environ.get("PHASE6F_RUN_SUMMARY", "1") == "1"

BASELINE_EXP = "p527_lstm_l3_v5_lite_s2027"
LIGHTGBM_RAW_KEY = "lightgbm_raw_g025"
LIGHTGBM_MAINLINE_KEY = "lightgbm_sep098_oct093"
LIGHTGBM_RAW_EXP_TEMPLATE = "p57a_{anchor_tag}_covact_lr005_l63_s2026_hard_g025"
FEATURE_SET = "cov_activity"
GATE_MODE = "hard"
QTY_GATE = "0.25"

BACKEND_SPECS = {
    "catboost": {
        "candidate_key_raw": "catboost_raw_g025",
        "candidate_key_calibrated": "catboost_sep098_oct093",
        "classifier_params": {
            "depth": 6,
            "learning_rate": 0.05,
            "iterations": 400,
            "l2_leaf_reg": 3.0,
            "loss_function": "Logloss",
            "allow_writing_files": False,
        },
        "regressor_params": {
            "depth": 6,
            "learning_rate": 0.05,
            "iterations": 400,
            "l2_leaf_reg": 3.0,
            "loss_function": "RMSE",
            "allow_writing_files": False,
        },
    },
    "xgboost": {
        "candidate_key_raw": "xgboost_raw_g025",
        "candidate_key_calibrated": "xgboost_sep098_oct093",
        "classifier_params": {
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_estimators": 400,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
        },
        "regressor_params": {
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_estimators": 400,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
        },
    },
}

CALIBRATION_SCALES = {
    "2025-09-01": 0.98,
    "2025-10-01": 0.93,
    "2025-11-01": 1.00,
    "2025-12-01": 1.00,
}


def parse_backend_env_list(name, fallback=None):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return set(fallback or [])
    return {token.strip().lower() for token in raw.split(",") if token.strip()}


FORCE_RAW_BACKENDS = parse_backend_env_list("PHASE6F_FORCE_RAW_BACKENDS")
FORCE_STAGE_C_BACKENDS = parse_backend_env_list("PHASE6F_FORCE_STAGE_C_BACKENDS", FORCE_RAW_BACKENDS)
FORCE_STAGE_D_BACKENDS = parse_backend_env_list("PHASE6F_FORCE_STAGE_D_BACKENDS", FORCE_STAGE_C_BACKENDS)


def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(VENDOR_DIR, exist_ok=True)


def init_timing_log():
    if os.path.exists(TIMING_LOG):
        return
    with open(TIMING_LOG, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "stage",
                "backend",
                "anchor_date",
                "exp_id",
                "start_time",
                "end_time",
                "elapsed_min",
                "status",
                "note",
            ]
        )


def append_timing(stage, backend, anchor_date, exp_id, t0, t1, status, note=""):
    with open(TIMING_LOG, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                stage,
                backend,
                anchor_date,
                exp_id,
                datetime.fromtimestamp(t0).strftime("%Y-%m-%d %H:%M:%S"),
                datetime.fromtimestamp(t1).strftime("%Y-%m-%d %H:%M:%S"),
                round((t1 - t0) / 60.0, 3),
                status,
                note,
            ]
        )


def anchor_tag(anchor_date):
    return anchor_date.replace("-", "")


def report_dir(anchor_date):
    path = os.path.join(OUT_DIR, anchor_tag(anchor_date), "phase5")
    os.makedirs(path, exist_ok=True)
    return path


def model_dir(anchor_date, backend):
    path = os.path.join(MODEL_DIR, backend, anchor_tag(anchor_date))
    os.makedirs(path, exist_ok=True)
    return path


def phase57_raw_context_path(anchor_date):
    exp_id = LIGHTGBM_RAW_EXP_TEMPLATE.format(anchor_tag=anchor_tag(anchor_date))
    return exp_id, os.path.join(
        PHASE57_DIR,
        anchor_tag(anchor_date),
        "phase5",
        f"eval_context_{exp_id}.csv",
    )


def phase55_context_path(anchor_date, base_exp):
    return os.path.join(
        PHASE55_DIR,
        anchor_tag(anchor_date),
        "phase5",
        f"eval_context_p55_{anchor_tag(anchor_date)}_{base_exp}_s2026.csv",
    )


def phase54_dec_baseline_path():
    return os.path.join(PHASE54_PHASE_DIR, f"eval_context_p54_{BASELINE_EXP}_s2026.csv")


def phase6f_exp_id(anchor_date, backend):
    return f"p6f_{anchor_tag(anchor_date)}_{backend}_covact_g025_s{SEED}"


def phase6f_context_path(anchor_date, backend):
    exp_id = phase6f_exp_id(anchor_date, backend)
    base = report_dir(anchor_date)
    candidates = [
        os.path.join(base, f"eval_context_{exp_id}.csv"),
        os.path.join(base, "phase5", f"eval_context_{exp_id}.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def resolve_phase6f_context_path(anchor_date, backend):
    exp_id = phase6f_exp_id(anchor_date, backend)
    base = report_dir(anchor_date)
    candidates = [
        os.path.join(base, f"eval_context_{exp_id}.csv"),
        os.path.join(base, "phase5", f"eval_context_{exp_id}.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def candidate_dirs(version, tags, meta_name, train_name):
    for tag in tags:
        if tag is None:
            processed_dir = os.path.join(PROJECT_ROOT, "data", f"processed_{version}")
            artifacts_dir = os.path.join(PROJECT_ROOT, "data", f"artifacts_{version}")
        else:
            processed_dir = os.path.join(PROJECT_ROOT, "data", f"processed_{version}_{tag}")
            artifacts_dir = os.path.join(PROJECT_ROOT, "data", f"artifacts_{version}_{tag}")
        meta_path = os.path.join(artifacts_dir, meta_name)
        train_path = os.path.join(processed_dir, train_name)
        if os.path.exists(meta_path) and os.path.exists(train_path):
            return processed_dir, artifacts_dir
    return None, None


def ensure_assets(anchor_date):
    suffix = anchor_tag(anchor_date)
    v5_tags = [
        f"p57a_{suffix}_v5_lite",
        f"p56_{suffix}_v5_lite",
        f"p55_{suffix}_v5_lite",
        None,
    ]
    v5_processed, v5_artifacts = candidate_dirs("v5_lite", v5_tags, "meta_v5_lite.json", "X_train_dyn.bin")
    if not (v5_processed and v5_artifacts):
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["FEATURE_SPLIT_DATE"] = anchor_date
        env["FEATURE_OUTPUT_TAG"] = f"p6f_{suffix}_v5_lite"
        env["FEATURE_VAL_MODE"] = VAL_MODE
        rc = subprocess.run(
            [sys.executable, os.path.join(PROJECT_ROOT, "src", "features", "build_features_v5_lite_sku.py")],
            cwd=PROJECT_ROOT,
            env=env,
        ).returncode
        if rc != 0:
            raise RuntimeError(f"Failed to build v5_lite assets for {anchor_date}")

    v6_tags = [
        f"p57a_{suffix}_v6_event",
        f"p56_{suffix}_v6_event",
        f"p55_{suffix}_v6_event",
        None,
    ]
    processed_dir, artifacts_dir = candidate_dirs("v6_event", v6_tags, "meta_v6_event.json", "X_train.npy")
    if processed_dir and artifacts_dir:
        return processed_dir, artifacts_dir

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["FEATURE_SPLIT_DATE"] = anchor_date
    env["FEATURE_OUTPUT_TAG"] = f"p6f_{suffix}_v6_event"
    env["FEATURE_VAL_MODE"] = VAL_MODE
    rc = subprocess.run(
        [sys.executable, os.path.join(PROJECT_ROOT, "src", "features", "build_features_v6_event_sku.py")],
        cwd=PROJECT_ROOT,
        env=env,
    ).returncode
    if rc != 0:
        raise RuntimeError(f"Failed to build v6_event assets for {anchor_date}")

    return (
        os.path.join(PROJECT_ROOT, "data", f"processed_v6_event_p6f_{suffix}_v6_event"),
        os.path.join(PROJECT_ROOT, "data", f"artifacts_v6_event_p6f_{suffix}_v6_event"),
    )


def run_cmd(args, stage, backend, anchor_date="", exp_id="", env=None, note=""):
    t0 = time.time()
    rc = subprocess.run(args, cwd=PROJECT_ROOT, env=env).returncode
    t1 = time.time()
    append_timing(stage, backend, anchor_date, exp_id, t0, t1, "success" if rc == 0 else "failed", note)
    return rc


def with_vendor_pythonpath(env=None):
    out = dict(env or os.environ.copy())
    current = out.get("PYTHONPATH", "").strip()
    out["PYTHONPATH"] = VENDOR_DIR if not current else VENDOR_DIR + os.pathsep + current
    return out


def stage_a_smoke():
    files_to_check = [
        os.path.join(PROJECT_ROOT, "src", "models", "tabular_hurdle.py"),
        os.path.join(PROJECT_ROOT, "src", "train", "train_tabular_v6.py"),
        os.path.join(PROJECT_ROOT, "run_phase6f_tree_family_compare.py"),
        os.path.join(PROJECT_ROOT, "src", "analysis", "summarize_phase6f_tree_family_compare_results.py"),
    ]
    syntax_ok = run_cmd([sys.executable, "-m", "py_compile", *files_to_check], "stage_a_syntax", "all", note="py_compile") == 0

    import_checks = {}
    for backend in BACKEND_SPECS:
        rc = run_cmd(
            [sys.executable, "-c", f"import {backend}"],
            "stage_a_import",
            backend,
            env=with_vendor_pythonpath(),
            note=f"import {backend}",
        )
        import_checks[backend] = rc == 0

    backend_smoke = {}
    for backend in BACKEND_SPECS:
        ok = False
        error = ""
        if import_checks[backend]:
            try:
                rng = np.random.default_rng(2026)
                x_train = rng.normal(size=(128, 12)).astype(np.float32)
                y_cls = (x_train[:, 0] + x_train[:, 1] > 0).astype(int)
                y_reg = np.log1p(np.clip((x_train[:, 0] * 3.0 + x_train[:, 2] * 2.0 + 3.0) * y_cls, a_min=0.0, a_max=None)).astype(np.float32)
                x_val = rng.normal(size=(48, 12)).astype(np.float32)
                y_val_cls = (x_val[:, 0] + x_val[:, 1] > 0).astype(int)
                y_val_reg = np.log1p(np.clip((x_val[:, 0] * 3.0 + x_val[:, 2] * 2.0 + 3.0) * y_val_cls, a_min=0.0, a_max=None)).astype(np.float32)
                cls_weights = np.where(y_cls > 0, 2.0, 1.0).astype(np.float32)
                reg_weights = np.ones_like(y_reg, dtype=np.float32)
                model = TabularHurdleModel(
                    random_state=2026,
                    backend=backend,
                    classifier_params=BACKEND_SPECS[backend]["classifier_params"],
                    regressor_params=BACKEND_SPECS[backend]["regressor_params"],
                )
                model.fit(
                    x_train,
                    y_cls,
                    y_reg,
                    cls_sample_weight=cls_weights,
                    reg_sample_weight=reg_weights,
                    x_val=x_val,
                    y_val_cls=y_val_cls,
                    y_val_reg=y_val_reg,
                )
                pred = model.predict_quantity(x_val, gate_mode="hard", gate_threshold=0.25)
                if pred["qty"].shape[0] != x_val.shape[0]:
                    raise RuntimeError("predict_quantity row count mismatch")
                with tempfile.TemporaryDirectory(prefix=f"phase6f_{backend}_", dir=OUT_DIR) as tmpdir:
                    cls_path = os.path.join(tmpdir, "cls.pkl")
                    reg_path = os.path.join(tmpdir, "reg.pkl")
                    meta_path = os.path.join(tmpdir, "meta.json")
                    model.save(cls_path, reg_path, meta_path, extra_meta={"smoke": True})
                    loaded, _ = TabularHurdleModel.load(cls_path, reg_path, meta_path)
                    loaded_pred = loaded.predict_quantity(x_val, gate_mode="hard", gate_threshold=0.25)
                    if loaded_pred["qty"].shape[0] != x_val.shape[0]:
                        raise RuntimeError("loaded predict_quantity row count mismatch")
                ok = True
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
        backend_smoke[backend] = {"ok": ok, "error": error}

    payload = {
        "syntax_ok": syntax_ok,
        "imports": import_checks,
        "backend_smoke": backend_smoke,
    }
    with open(STAGEA_JSON, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return payload


def build_candidate_row_from_context(path, candidate_key, backend, stage_name, anchor_date):
    row = evaluate_context_csv(path)
    row["candidate_key"] = candidate_key
    row["backend"] = backend
    row["stage_name"] = stage_name
    row["anchor_date"] = anchor_date
    row["context_path"] = path
    return row


def run_backend_job(anchor_date, backend):
    exp_id = phase6f_exp_id(anchor_date, backend)
    context_path = resolve_phase6f_context_path(anchor_date, backend)
    if context_path is not None:
        return exp_id, context_path

    processed_dir, artifacts_dir = ensure_assets(anchor_date)
    env = with_vendor_pythonpath()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env["EXP_ID"] = exp_id
    env["EXP_VERSION"] = "v6_event"
    env["EXP_FEATURE_SET"] = FEATURE_SET
    env["EXP_GATE_MODE"] = GATE_MODE
    env["EXP_QTY_GATE"] = QTY_GATE
    env["EXP_SEED"] = SEED
    env["EXP_TREE_BACKEND"] = backend
    env["EXP_PROCESSED_DIR"] = processed_dir
    env["EXP_ARTIFACTS_DIR"] = artifacts_dir
    env["EXP_MODEL_DIR"] = model_dir(anchor_date, backend)
    env["EXP_REPORT_DIR"] = report_dir(anchor_date)
    env["EXP_TREE_CLS_PARAMS_JSON"] = json.dumps(BACKEND_SPECS[backend]["classifier_params"])
    env["EXP_TREE_REG_PARAMS_JSON"] = json.dumps(BACKEND_SPECS[backend]["regressor_params"])

    if run_cmd([sys.executable, os.path.join(PROJECT_ROOT, "src", "train", "train_tabular_v6.py")], "train", backend, anchor_date, exp_id, env=env) != 0:
        raise RuntimeError(f"Training failed for {exp_id}")
    if run_cmd([sys.executable, os.path.join(PROJECT_ROOT, "evaluate_tabular.py")], "eval_tabular", backend, anchor_date, exp_id, env=env) != 0:
        raise RuntimeError(f"evaluate_tabular failed for {exp_id}")
    if run_cmd([sys.executable, os.path.join(PROJECT_ROOT, "evaluate_agg.py")], "eval_agg", backend, anchor_date, exp_id, env=env) != 0:
        raise RuntimeError(f"evaluate_agg failed for {exp_id}")
    context_path = resolve_phase6f_context_path(anchor_date, backend)
    if context_path is None:
        raise FileNotFoundError(
            f"Missing eval context after evaluation for {exp_id} under {report_dir(anchor_date)}"
        )
    return exp_id, context_path


def smoke_gate(candidate_row, lgbm_row):
    primary_ok = (
        0.90 <= float(candidate_row["global_ratio"]) <= 1.10
        and float(candidate_row["4_25_under_wape"]) <= float(lgbm_row["4_25_under_wape"]) + 0.01
        and float(candidate_row["4_25_sku_p50"]) >= float(lgbm_row["4_25_sku_p50"]) - 0.01
        and float(candidate_row["ice_4_25_sku_p50"]) >= float(lgbm_row["ice_4_25_sku_p50"]) - 0.01
    )
    improvement_ok = (
        float(candidate_row["blockbuster_under_wape"]) <= float(lgbm_row["blockbuster_under_wape"]) - 0.02
        or float(candidate_row["blockbuster_sku_p50"]) >= float(lgbm_row["blockbuster_sku_p50"]) + 0.03
        or float(candidate_row["top20_true_volume_capture"]) >= float(lgbm_row["top20_true_volume_capture"]) + 0.02
        or float(candidate_row["rank_corr_positive_skus"]) >= float(lgbm_row["rank_corr_positive_skus"]) + 0.02
    )
    return primary_ok and improvement_ok


def summarize_rows(anchor_rows):
    metric_cols = [
        "global_ratio",
        "global_wmape",
        "4_25_under_wape",
        "4_25_sku_p50",
        "ice_4_25_sku_p50",
        "blockbuster_under_wape",
        "blockbuster_sku_p50",
        "top20_true_volume_capture",
        "rank_corr_positive_skus",
    ]
    agg = {col: (col, "mean") for col in metric_cols}
    if "anchor_pass" in anchor_rows.columns:
        agg["anchor_passes"] = ("anchor_pass", "sum")
    return anchor_rows.groupby(["candidate_key", "backend", "stage_name"], as_index=False).agg(**agg)


def stage_b_single_anchor(stagea):
    exp_id, lgbm_path = phase57_raw_context_path(SMOKE_ANCHOR)
    lgbm_row = build_candidate_row_from_context(lgbm_path, LIGHTGBM_RAW_KEY, "lightgbm", "single_anchor_smoke", SMOKE_ANCHOR)
    lgbm_row["smoke_gate_pass"] = True
    lgbm_row["forced_to_stage_c"] = False
    lgbm_row["advance_to_stage_c"] = True
    lgbm_row["smoke_ok"] = True

    rows = [lgbm_row]
    gate_passed = []
    advanced = []
    for backend in BACKEND_SPECS:
        smoke_ok = bool(stagea["imports"].get(backend)) and bool(stagea["backend_smoke"].get(backend, {}).get("ok"))
        if not smoke_ok:
            row = {key: np.nan for key in lgbm_row.keys()}
            row.update(
                {
                    "candidate_key": BACKEND_SPECS[backend]["candidate_key_raw"],
                    "backend": backend,
                    "stage_name": "single_anchor_smoke",
                    "anchor_date": SMOKE_ANCHOR,
                    "context_path": "",
                    "smoke_gate_pass": False,
                    "forced_to_stage_c": False,
                    "advance_to_stage_c": False,
                    "smoke_ok": False,
                    "smoke_error": stagea["backend_smoke"].get(backend, {}).get("error", "backend smoke failed"),
                }
            )
            rows.append(row)
            continue

        _, context_path = run_backend_job(SMOKE_ANCHOR, backend)
        row = build_candidate_row_from_context(context_path, BACKEND_SPECS[backend]["candidate_key_raw"], backend, "single_anchor_smoke", SMOKE_ANCHOR)
        row["smoke_ok"] = True
        row["smoke_gate_pass"] = smoke_gate(row, lgbm_row)
        row["forced_to_stage_c"] = backend in FORCE_STAGE_C_BACKENDS and not row["smoke_gate_pass"]
        row["advance_to_stage_c"] = row["smoke_gate_pass"] or row["forced_to_stage_c"]
        row["smoke_error"] = ""
        if row["smoke_gate_pass"]:
            gate_passed.append(backend)
        if row["advance_to_stage_c"]:
            advanced.append(backend)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(SMOKE_TABLE, index=False, encoding="utf-8-sig")
    return df, gate_passed, advanced


def raw_stage_gate(candidate_row, lgbm_row):
    primary_ok = (
        0.90 <= float(candidate_row["global_ratio"]) <= 1.10
        and float(candidate_row["4_25_under_wape"]) <= float(lgbm_row["4_25_under_wape"]) + 0.01
        and float(candidate_row["4_25_sku_p50"]) >= float(lgbm_row["4_25_sku_p50"]) - 0.01
        and float(candidate_row["ice_4_25_sku_p50"]) >= float(lgbm_row["ice_4_25_sku_p50"]) - 0.01
    )
    improvement_ok = (
        float(candidate_row["blockbuster_under_wape"]) <= float(lgbm_row["blockbuster_under_wape"]) - 0.02
        or float(candidate_row["blockbuster_sku_p50"]) >= float(lgbm_row["blockbuster_sku_p50"]) + 0.03
        or float(candidate_row["top20_true_volume_capture"]) >= float(lgbm_row["top20_true_volume_capture"]) + 0.02
        or float(candidate_row["rank_corr_positive_skus"]) >= float(lgbm_row["rank_corr_positive_skus"]) + 0.02
    )
    return primary_ok and improvement_ok


def stage_c_raw_compare(stage_c_backends):
    rows = []
    for anchor_date in ANCHORS:
        _, lgbm_path = phase57_raw_context_path(anchor_date)
        rows.append(build_candidate_row_from_context(lgbm_path, LIGHTGBM_RAW_KEY, "lightgbm", "raw_4anchor_compare", anchor_date))

    for backend in stage_c_backends:
        for anchor_date in ANCHORS:
            _, context_path = run_backend_job(anchor_date, backend)
            rows.append(build_candidate_row_from_context(context_path, BACKEND_SPECS[backend]["candidate_key_raw"], backend, "raw_4anchor_compare", anchor_date))

    df = pd.DataFrame(rows)
    df.to_csv(RAW_ANCHOR_TABLE, index=False, encoding="utf-8-sig")

    summary = summarize_rows(df)
    lgbm_summary = summary[summary["candidate_key"] == LIGHTGBM_RAW_KEY]
    if lgbm_summary.empty:
        return df, summary, []
    lgbm_row = lgbm_summary.iloc[0]

    gate_passed = []
    advanced = []
    for backend in stage_c_backends:
        candidate_key = BACKEND_SPECS[backend]["candidate_key_raw"]
        sub = summary[summary["candidate_key"] == candidate_key]
        if sub.empty:
            continue
        gate_pass = raw_stage_gate(sub.iloc[0], lgbm_row)
        if gate_pass:
            gate_passed.append(backend)
        if gate_pass or backend in FORCE_STAGE_D_BACKENDS:
            advanced.append(backend)
    return df, summary, gate_passed, advanced


def load_baseline_rows():
    rows = []
    for anchor_date in ANCHORS:
        path = phase54_dec_baseline_path() if anchor_date == "2025-12-01" else phase55_context_path(anchor_date, BASELINE_EXP)
        rows.append(build_candidate_row_from_context(path, "p527", "sequence", "baseline", anchor_date))
    return pd.DataFrame(rows)


def apply_sep098_scaling(df, anchor_date):
    out = df.copy()
    scale = CALIBRATION_SCALES[anchor_date]
    for col in ("ai_pred_qty_open", "ai_pred_qty", "ai_pred_positive_qty"):
        out[col] = out[col].astype(float)
    if scale != 1.0:
        for col in ("ai_pred_qty_open", "ai_pred_qty", "ai_pred_positive_qty"):
            out[col] = out[col] * scale
    out["ai_pred_positive_qty"] = (out["ai_pred_qty"].astype(float) > 0).astype(int)
    out["abs_error"] = (out["ai_pred_qty"].astype(float) - out["true_replenish_qty"].astype(float)).abs()
    return out


def add_anchor_pass(df, baseline_df):
    merged = df.merge(
        baseline_df[["anchor_date", "4_25_sku_p50", "4_25_wmape_like", "ice_4_25_sku_p50"]],
        on="anchor_date",
        how="left",
        suffixes=("", "_baseline"),
    )
    merged["anchor_pass"] = (
        (merged["4_25_sku_p50"] > merged["4_25_sku_p50_baseline"])
        & (merged["4_25_wmape_like"] <= merged["4_25_wmape_like_baseline"])
        & (merged["ice_4_25_sku_p50"] > merged["ice_4_25_sku_p50_baseline"])
        & merged["global_ratio"].between(0.90, 1.10)
    )
    return merged


def stage_d_calibrated_compare(raw_passed_backends):
    baseline_df = load_baseline_rows()
    rows = []
    for anchor_date in ANCHORS:
        _, lgbm_path = phase57_raw_context_path(anchor_date)
        raw_df = pd.read_csv(lgbm_path)
        adj = apply_sep098_scaling(raw_df, anchor_date)
        row = evaluate_context_frame(adj, f"{LIGHTGBM_RAW_EXP_TEMPLATE.format(anchor_tag=anchor_tag(anchor_date))}_sep098_oct093")
        row["candidate_key"] = LIGHTGBM_MAINLINE_KEY
        row["backend"] = "lightgbm"
        row["stage_name"] = "calibrated_final_compare"
        row["anchor_date"] = anchor_date
        row["context_path"] = lgbm_path
        rows.append(row)

    for backend in raw_passed_backends:
        for anchor_date in ANCHORS:
            raw_path = phase6f_context_path(anchor_date, backend)
            raw_df = pd.read_csv(raw_path)
            adj = apply_sep098_scaling(raw_df, anchor_date)
            row = evaluate_context_frame(adj, f"{phase6f_exp_id(anchor_date, backend)}_sep098_oct093")
            row["candidate_key"] = BACKEND_SPECS[backend]["candidate_key_calibrated"]
            row["backend"] = backend
            row["stage_name"] = "calibrated_final_compare"
            row["anchor_date"] = anchor_date
            row["context_path"] = raw_path
            rows.append(row)

    df = pd.DataFrame(rows)
    df = add_anchor_pass(df, baseline_df)
    df.to_csv(CALIBRATED_ANCHOR_TABLE, index=False, encoding="utf-8-sig")
    return df


def main():
    ensure_dirs()
    init_timing_log()

    stagea = stage_a_smoke()
    smoke_df, smoke_gate_passed_backends, stage_c_backends = stage_b_single_anchor(stagea)
    raw_anchor_df, raw_summary_df, raw_gate_passed_backends, stage_d_backends = stage_c_raw_compare(stage_c_backends)
    raw_summary_df["raw_stage_pass"] = raw_summary_df["backend"].map(
        lambda backend: backend == "lightgbm" or backend in raw_gate_passed_backends
    )
    raw_summary_df["forced_to_stage_d"] = raw_summary_df["backend"].map(
        lambda backend: backend != "lightgbm" and backend in stage_d_backends and backend not in raw_gate_passed_backends
    )
    raw_summary_df["advance_to_stage_d"] = raw_summary_df["backend"].map(
        lambda backend: backend == "lightgbm" or backend in stage_d_backends
    )
    calibrated_anchor_df = stage_d_calibrated_compare(stage_d_backends)

    stage_state = {
        "current_mainline": "sep098_oct093",
        "stagea": stagea,
        "force_stage_c_backends": sorted(FORCE_STAGE_C_BACKENDS),
        "force_stage_d_backends": sorted(FORCE_STAGE_D_BACKENDS),
        "smoke_gate_passed_backends": smoke_gate_passed_backends,
        "stage_c_backends": stage_c_backends,
        "raw_gate_passed_backends": raw_gate_passed_backends,
        "stage_d_backends": stage_d_backends,
        "raw_summary": raw_summary_df.to_dict(orient="records"),
        "generated_files": {
            "stagea_json": STAGEA_JSON,
            "smoke_table": SMOKE_TABLE,
            "raw_anchor_table": RAW_ANCHOR_TABLE,
            "calibrated_anchor_table": CALIBRATED_ANCHOR_TABLE,
        },
    }
    with open(STAGE_STATE_JSON, "w", encoding="utf-8") as fh:
        json.dump(stage_state, fh, ensure_ascii=False, indent=2)

    if RUN_SUMMARY:
        run_cmd([sys.executable, os.path.join(PROJECT_ROOT, "src", "analysis", "summarize_phase6f_tree_family_compare_results.py")], "summary", "all", note="phase6f summary")

    print(f"[OK] stagea -> {STAGEA_JSON}")
    print(f"[OK] smoke table -> {SMOKE_TABLE}")
    print(f"[OK] raw anchor table -> {RAW_ANCHOR_TABLE}")
    print(f"[OK] calibrated anchor table -> {CALIBRATED_ANCHOR_TABLE}")
    print(f"[OK] stage state -> {STAGE_STATE_JSON}")


if __name__ == "__main__":
    main()
