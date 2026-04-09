import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(PROJECT_ROOT, "src", "analysis"))

from phase_eval_utils import evaluate_context_frame, numeric_cols_for_rounding


BASE_REPORT_ROOT = os.path.join(PROJECT_ROOT, "reports", "phase7_tail_allocation_optimization")
BASE_MODEL_ROOT = os.path.join(PROJECT_ROOT, "models_phase7_tail_allocation_optimization")
PHASE57_DIR = os.path.join(PROJECT_ROOT, "reports", "phase5_7")
PHASE55_DIR = os.path.join(PROJECT_ROOT, "reports", "phase5_5")
PHASE54_PHASE_DIR = os.path.join(PROJECT_ROOT, "reports", "phase5_4", "phase5")
PHASE6_MAINLINE = os.path.join(PROJECT_ROOT, "reports", "phase6", "phase6_frozen_mainline.json")

ANCHORS = ["2025-09-01", "2025-10-01", "2025-11-01", "2025-12-01"]
SEEDS = ["2026", "2027", "2028"]
TREE_BACKEND = "lightgbm"
VAL_MODE = "single_anchor"
BASELINE_EXP = "p527_lstm_l3_v5_lite_s2027"
RAW_TEMPLATE = "p57a_{anchor_tag}_covact_lr005_l63_s2026_hard_g025"
CAL_SCALES = {"2025-09-01": 0.98, "2025-10-01": 0.93, "2025-11-01": 1.00, "2025-12-01": 1.00}

VARIANTS = {
    "qfo_plus": {"feature_set": "cov_activity_qfo", "track": "cov_activity_qfo", "required_cols": ["qty_first_order_bucket"]},
    "tail_peak": {"feature_set": "cov_activity_tail", "track": "cov_activity_tail", "required_cols": ["count_repl_gt25_30", "count_future_gt25_30"]},
    "style_category_priors": {
        "feature_set": "cov_activity_priors",
        "track": "cov_activity_priors",
        "required_cols": ["style_sum_repl_30", "category_sum_repl_30"],
    },
    "tail_full": {
        "feature_set": "cov_activity_tail_full",
        "track": "cov_activity_tail_full",
        "required_cols": ["qty_first_order_bucket", "count_repl_gt25_30", "style_sum_repl_30"],
    },
}

S0_JOB = {"variant_key": "qfo_plus", "anchor_date": "2025-12-01", "seed": "2026", "lr": "0.05", "num_leaves": "63", "qty_gate": "0.25", "n_estimators": "400"}
S1_PARAMS = {"lr": "0.05", "num_leaves": "63", "qty_gate": "0.25", "n_estimators": "400"}
S2_LRS = ["0.03", "0.05"]
S2_LEAVES = ["31", "63"]
S2_GATES = ["0.23", "0.25", "0.27"]
S2_N_ESTIMATORS = "800"
S3_TOP_K = 6


def now_id():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def anchor_tag(anchor_date):
    return anchor_date.replace("-", "")


def tok(value):
    return str(value).replace(".", "")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def read_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def write_text(path, text):
    with open(path, "w", encoding="utf-8-sig") as fh:
        fh.write(text)


def apply_calibration(df, anchor_date):
    out = df.copy()
    scale = float(CAL_SCALES.get(anchor_date, 1.0))
    if scale != 1.0:
        for col in ("ai_pred_qty_open", "ai_pred_qty", "ai_pred_positive_qty"):
            out[col] = out[col].astype(float) * scale
    out["ai_pred_positive_qty"] = (out["ai_pred_qty"].astype(float) > 0).astype(int)
    out["abs_error"] = (out["ai_pred_qty"].astype(float) - out["true_replenish_qty"].astype(float)).abs()
    return out


class Runner:
    def __init__(self, run_id):
        self.run_id = run_id
        self.report_dir = ensure_dir(os.path.join(BASE_REPORT_ROOT, f"overnight_{run_id}"))
        self.model_dir = ensure_dir(os.path.join(BASE_MODEL_ROOT, f"overnight_{run_id}"))
        self.jobs_dir = ensure_dir(os.path.join(self.report_dir, "jobs"))
        self.timing_log = os.path.join(self.report_dir, "timing_log.csv")
        self.mainline = read_json(PHASE6_MAINLINE)
        self._init_timing()
        write_json(
            os.path.join(self.report_dir, "run_manifest.json"),
            {
                "run_id": self.run_id,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "baseline_mainline": self.mainline["mainline_candidate"],
                "anchors": ANCHORS,
                "seeds": SEEDS,
                "report_dir": self.report_dir,
                "model_dir": self.model_dir,
            },
        )

    def _init_timing(self):
        if os.path.exists(self.timing_log):
            return
        with open(self.timing_log, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["stage", "job_key", "anchor_date", "variant_key", "seed", "lr", "num_leaves", "qty_gate", "n_estimators", "step", "start_time", "end_time", "elapsed_min", "status", "note"])

    def stage_status(self, stage_name, status, **extra):
        payload = {"stage": stage_name, "status": status, "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        payload.update(extra)
        write_json(os.path.join(self.report_dir, f"{stage_name}_status.json"), payload)

    def append_timing(self, job, step, t0, t1, status, note=""):
        with open(self.timing_log, "a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    job["stage"],
                    job["job_key"],
                    job["anchor_date"],
                    job["variant_key"],
                    job["seed"],
                    job["lr"],
                    job["num_leaves"],
                    job["qty_gate"],
                    job["n_estimators"],
                    step,
                    datetime.fromtimestamp(t0).strftime("%Y-%m-%d %H:%M:%S"),
                    datetime.fromtimestamp(t1).strftime("%Y-%m-%d %H:%M:%S"),
                    round((t1 - t0) / 60.0, 3),
                    status,
                    note,
                ]
            )

    def report_anchor_dir(self, stage_name, anchor_date):
        return ensure_dir(os.path.join(self.report_dir, stage_name, anchor_tag(anchor_date)))

    def model_paths(self, exp_id):
        return (
            os.path.join(self.model_dir, f"{exp_id}_cls.pkl"),
            os.path.join(self.model_dir, f"{exp_id}_reg.pkl"),
            os.path.join(self.model_dir, f"{exp_id}_meta.json"),
        )

    def copy_model_artifacts(self, src_exp, dst_exp):
        src_cls, src_reg, src_meta = self.model_paths(src_exp)
        dst_cls, dst_reg, dst_meta = self.model_paths(dst_exp)
        shutil.copy(src_cls, dst_cls)
        shutil.copy(src_reg, dst_reg)
        shutil.copy(src_meta, dst_meta)

    def env(self, extra):
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        env.update(extra)
        return env

    def phase57_context_path(self, anchor_date):
        exp_id = RAW_TEMPLATE.format(anchor_tag=anchor_tag(anchor_date))
        return os.path.join(PHASE57_DIR, anchor_tag(anchor_date), "phase5", f"eval_context_{exp_id}.csv")

    def phase55_context_path(self, anchor_date, exp_name):
        return os.path.join(PHASE55_DIR, anchor_tag(anchor_date), "phase5", f"eval_context_p55_{anchor_tag(anchor_date)}_{exp_name}_s2026.csv")

    def phase54_context_path(self):
        return os.path.join(PHASE54_PHASE_DIR, f"eval_context_p54_{BASELINE_EXP}_s2026.csv")

    def assets_ready(self, processed_dir, artifacts_dir, anchor_date):
        meta_path = os.path.join(artifacts_dir, "meta_v6_event.json")
        x_train = os.path.join(processed_dir, "X_train.npy")
        if not (os.path.exists(meta_path) and os.path.exists(x_train)):
            return False
        try:
            meta = read_json(meta_path)
        except Exception:
            return False
        groups = meta.get("feature_groups", {})
        return (
            meta.get("split_date") == anchor_date
            and bool(meta.get("qfo_cols"))
            and bool(meta.get("tail_cols"))
            and bool(meta.get("prior_cols"))
            and "qfo" in groups
            and "tail" in groups
            and "priors" in groups
        )

    def ensure_assets(self, anchor_date):
        suffix = anchor_tag(anchor_date)
        v5_tag = f"p7b_{suffix}_v5_lite"
        v5_processed = os.path.join(PROJECT_ROOT, "data", f"processed_v5_lite_{v5_tag}")
        v5_artifacts = os.path.join(PROJECT_ROOT, "data", f"artifacts_v5_lite_{v5_tag}")
        if not (os.path.exists(os.path.join(v5_processed, "X_train_dyn.bin")) and os.path.exists(os.path.join(v5_artifacts, "meta_v5_lite.json"))):
            rc = subprocess.run(
                [sys.executable, os.path.join(PROJECT_ROOT, "src", "features", "build_features_v5_lite_sku.py")],
                cwd=PROJECT_ROOT,
                env=self.env({"FEATURE_SPLIT_DATE": anchor_date, "FEATURE_OUTPUT_TAG": v5_tag, "FEATURE_VAL_MODE": VAL_MODE}),
            ).returncode
            if rc != 0:
                raise RuntimeError(f"build v5_lite failed for {anchor_date}")

        v6_tag = f"p7b_{suffix}_v6_event"
        processed_dir = os.path.join(PROJECT_ROOT, "data", f"processed_v6_event_{v6_tag}")
        artifacts_dir = os.path.join(PROJECT_ROOT, "data", f"artifacts_v6_event_{v6_tag}")
        if self.assets_ready(processed_dir, artifacts_dir, anchor_date):
            return processed_dir, artifacts_dir
        rc = subprocess.run(
            [sys.executable, os.path.join(PROJECT_ROOT, "src", "features", "build_features_v6_event_sku.py")],
            cwd=PROJECT_ROOT,
            env=self.env({"FEATURE_SPLIT_DATE": anchor_date, "FEATURE_OUTPUT_TAG": v6_tag, "FEATURE_VAL_MODE": VAL_MODE}),
        ).returncode
        if rc != 0:
            raise RuntimeError(f"build v6_event failed for {anchor_date}")
        return processed_dir, artifacts_dir

    def make_job(self, stage_name, variant_key, anchor_date, seed, lr, num_leaves, qty_gate, n_estimators):
        cfg = f"{variant_key}_lr{tok(lr)}_l{num_leaves}_g{tok(qty_gate)}_n{n_estimators}_s{seed}"
        base_exp = f"p7ov_{self.run_id}_{stage_name}_{anchor_tag(anchor_date)}_{cfg}"
        eval_exp = f"{base_exp}_hard_g{tok(qty_gate)}"
        report_anchor_dir = self.report_anchor_dir(stage_name, anchor_date)
        return {
            "stage": stage_name,
            "variant_key": variant_key,
            "feature_set": VARIANTS[variant_key]["feature_set"],
            "track": VARIANTS[variant_key]["track"],
            "required_cols": VARIANTS[variant_key]["required_cols"],
            "anchor_date": anchor_date,
            "seed": str(seed),
            "lr": str(lr),
            "num_leaves": str(num_leaves),
            "qty_gate": str(qty_gate),
            "n_estimators": str(n_estimators),
            "candidate_key": cfg,
            "job_key": f"{stage_name}_{anchor_tag(anchor_date)}_{cfg}",
            "base_exp": base_exp,
            "eval_exp": eval_exp,
            "report_dir": report_anchor_dir,
            "context_csv": os.path.join(report_anchor_dir, "phase5", f"eval_context_{eval_exp}.csv"),
            "eval_txt": os.path.join(report_anchor_dir, f"eval_{eval_exp}.txt"),
            "agg_txt": os.path.join(report_anchor_dir, f"agg_{eval_exp}.txt"),
        }

    def validate_meta(self, job):
        _, _, meta_path = self.model_paths(job["base_exp"])
        if not os.path.exists(meta_path):
            return False, "missing_meta"
        try:
            meta = read_json(meta_path)
        except Exception:
            return False, "bad_meta"
        cols = meta.get("selected_feature_cols", [])
        if meta.get("feature_set") != job["feature_set"]:
            return False, "feature_set_mismatch"
        for col in job["required_cols"]:
            if col not in cols:
                return False, f"missing_required_col:{col}"
        return True, "ok"

    def validate_context(self, job):
        if not os.path.exists(job["context_csv"]):
            return False, "missing_context"
        try:
            df = pd.read_csv(job["context_csv"], usecols=["anchor_date"])
        except Exception:
            return False, "bad_context"
        anchors = sorted(df["anchor_date"].astype(str).unique().tolist())
        if anchors != [job["anchor_date"]]:
            return False, f"anchor_mismatch:{anchors}"
        return True, "ok"

    def job_manifest_path(self, job):
        return os.path.join(self.jobs_dir, f"{job['job_key']}.json")

    def write_job_manifest(self, job, status, note="", processed_dir=None, artifacts_dir=None):
        payload = dict(job)
        payload["status"] = status
        payload["note"] = note
        payload["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if processed_dir:
            payload["processed_dir"] = processed_dir
        if artifacts_dir:
            payload["artifacts_dir"] = artifacts_dir
        _, _, meta_path = self.model_paths(job["base_exp"])
        payload["model_meta_path"] = meta_path
        write_json(self.job_manifest_path(job), payload)

    def run_job(self, job):
        meta_ok, meta_note = self.validate_meta(job)
        ctx_ok, ctx_note = self.validate_context(job)
        if meta_ok and ctx_ok and os.path.exists(job["eval_txt"]) and os.path.exists(job["agg_txt"]):
            self.write_job_manifest(job, "completed_cached", note=json.dumps({"meta": meta_note, "context": ctx_note}, ensure_ascii=False))
            return

        processed_dir, artifacts_dir = self.ensure_assets(job["anchor_date"])
        cls_path, reg_path, _ = self.model_paths(job["base_exp"])
        if not (os.path.exists(cls_path) and os.path.exists(reg_path) and meta_ok):
            env = self.env(
                {
                    "EXP_ID": job["base_exp"],
                    "EXP_VERSION": "v6_event",
                    "EXP_FEATURE_SET": job["feature_set"],
                    "EXP_GATE_MODE": "hard",
                    "EXP_QTY_GATE": job["qty_gate"],
                    "EXP_SEED": job["seed"],
                    "EXP_TREE_BACKEND": TREE_BACKEND,
                    "EXP_PROCESSED_DIR": processed_dir,
                    "EXP_ARTIFACTS_DIR": artifacts_dir,
                    "EXP_MODEL_DIR": self.model_dir,
                    "EXP_REPORT_DIR": job["report_dir"],
                    "EXP_LGBM_LR": job["lr"],
                    "EXP_LGBM_NUM_LEAVES": job["num_leaves"],
                    "EXP_LGBM_CLS_CHILD": "20",
                    "EXP_LGBM_REG_CHILD": "20",
                    "EXP_LGBM_N_ESTIMATORS": job["n_estimators"],
                    "EXP_LGBM_SUBSAMPLE": "0.80",
                    "EXP_LGBM_COLSAMPLE": "0.80",
                }
            )
            t0 = time.time()
            rc = subprocess.run([sys.executable, os.path.join(PROJECT_ROOT, "src", "train", "train_tabular_v6.py")], cwd=PROJECT_ROOT, env=env).returncode
            t1 = time.time()
            self.append_timing(job, "train", t0, t1, "success" if rc == 0 else "failed")
            if rc != 0:
                self.write_job_manifest(job, "train_failed", processed_dir=processed_dir, artifacts_dir=artifacts_dir)
                raise RuntimeError(f"train failed: {job['job_key']}")

        self.copy_model_artifacts(job["base_exp"], job["eval_exp"])
        env = self.env(
            {
                "EXP_ID": job["eval_exp"],
                "EXP_VERSION": "v6_event",
                "EXP_FEATURE_SET": job["feature_set"],
                "EXP_GATE_MODE": "hard",
                "EXP_QTY_GATE": job["qty_gate"],
                "EXP_SEED": job["seed"],
                "EXP_TREE_BACKEND": TREE_BACKEND,
                "EXP_PROCESSED_DIR": processed_dir,
                "EXP_ARTIFACTS_DIR": artifacts_dir,
                "EXP_MODEL_DIR": self.model_dir,
                "EXP_REPORT_DIR": job["report_dir"],
            }
        )
        t0 = time.time()
        rc = subprocess.run([sys.executable, os.path.join(PROJECT_ROOT, "evaluate_tabular.py")], cwd=PROJECT_ROOT, env=env).returncode
        if rc == 0:
            rc = subprocess.run([sys.executable, os.path.join(PROJECT_ROOT, "evaluate_agg.py")], cwd=PROJECT_ROOT, env=env).returncode
        t1 = time.time()
        self.append_timing(job, "eval", t0, t1, "success" if rc == 0 else "failed")
        if rc != 0:
            self.write_job_manifest(job, "eval_failed", processed_dir=processed_dir, artifacts_dir=artifacts_dir)
            raise RuntimeError(f"eval failed: {job['job_key']}")
        self.write_job_manifest(job, "completed", processed_dir=processed_dir, artifacts_dir=artifacts_dir)

    def build_s0_jobs(self):
        return [self.make_job("stage_s0", **S0_JOB)]

    def build_s1_jobs(self):
        jobs = []
        for variant_key in VARIANTS:
            for anchor_date in ANCHORS:
                jobs.append(self.make_job("stage_s1", variant_key, anchor_date, "2026", **S1_PARAMS))
        return jobs

    def build_s2_jobs(self):
        jobs = []
        for variant_key in VARIANTS:
            for anchor_date in ANCHORS:
                for lr in S2_LRS:
                    for leaves in S2_LEAVES:
                        for gate in S2_GATES:
                            jobs.append(self.make_job("stage_s2", variant_key, anchor_date, "2026", lr, leaves, gate, S2_N_ESTIMATORS))
        return jobs

    def build_s3_jobs(self, s2_cal_summary):
        candidates = s2_cal_summary.loc[~s2_cal_summary["candidate_key"].isin(["lightgbm_sep098_oct093"])].copy()
        candidates = candidates.sort_values(
            ["replacement_gate_like", "blockbuster_under_wape", "blockbuster_sku_p50", "top20_true_volume_capture", "rank_corr_positive_skus"],
            ascending=[False, True, False, False, False],
        ).head(S3_TOP_K)
        jobs = []
        for _, row in candidates.iterrows():
            for seed in SEEDS:
                for anchor_date in ANCHORS:
                    jobs.append(
                        self.make_job(
                            "stage_s3",
                            row["variant_key"],
                            anchor_date,
                            seed,
                            f"{float(row['lr']):.2f}",
                            str(int(row["num_leaves"])),
                            f"{float(row['qty_gate']):.2f}",
                            str(int(row["n_estimators"])),
                        )
                    )
        return jobs

    def build_baseline_rows(self):
        rows = []
        for anchor_date in ANCHORS:
            if anchor_date == "2025-12-01":
                df = pd.read_csv(self.phase54_context_path())
                exp_id = f"p54_{BASELINE_EXP}_s2026"
            else:
                df = pd.read_csv(self.phase55_context_path(anchor_date, BASELINE_EXP))
                exp_id = f"p55_{anchor_tag(anchor_date)}_{BASELINE_EXP}_s2026"
            row = evaluate_context_frame(df, exp_id)
            row["anchor_date"] = anchor_date
            row["candidate_key"] = "p527"
            row["track"] = "sequence_baseline"
            rows.append(row)
        return pd.DataFrame(rows)

    def build_current_raw_rows(self):
        rows = []
        for anchor_date in ANCHORS:
            df = pd.read_csv(self.phase57_context_path(anchor_date))
            row = evaluate_context_frame(df, f"{RAW_TEMPLATE.format(anchor_tag=anchor_tag(anchor_date))}_raw")
            row["anchor_date"] = anchor_date
            row["candidate_key"] = "lightgbm_raw_g025"
            row["track"] = "current_raw_mainline"
            rows.append(row)
        return pd.DataFrame(rows)

    def build_current_cal_rows(self):
        rows = []
        for anchor_date in ANCHORS:
            df = apply_calibration(pd.read_csv(self.phase57_context_path(anchor_date)), anchor_date)
            row = evaluate_context_frame(df, f"{RAW_TEMPLATE.format(anchor_tag=anchor_tag(anchor_date))}_sep098_oct093")
            row["anchor_date"] = anchor_date
            row["candidate_key"] = "lightgbm_sep098_oct093"
            row["track"] = "current_calibrated_mainline"
            rows.append(row)
        return pd.DataFrame(rows)

    def build_candidate_rows(self, jobs, calibrated=False):
        rows = []
        for job in jobs:
            if not os.path.exists(job["context_csv"]):
                continue
            df = pd.read_csv(job["context_csv"])
            if calibrated:
                df = apply_calibration(df, job["anchor_date"])
            row = evaluate_context_frame(df, job["eval_exp"])
            row["anchor_date"] = job["anchor_date"]
            row["candidate_key"] = job["candidate_key"]
            row["variant_key"] = job["variant_key"]
            row["track"] = job["track"]
            row["seed"] = int(job["seed"])
            row["lr"] = float(job["lr"])
            row["num_leaves"] = int(job["num_leaves"])
            row["qty_gate"] = float(job["qty_gate"])
            row["n_estimators"] = int(job["n_estimators"])
            rows.append(row)
        return pd.DataFrame(rows)

    def add_anchor_pass(self, anchor_df, baseline_df):
        merged = anchor_df.merge(
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

    def summarize_candidates(self, df, include_anchor_pass=False):
        df = df.copy()
        if "variant_key" not in df.columns:
            df["variant_key"] = df["candidate_key"]
        else:
            df["variant_key"] = df["variant_key"].fillna(df["candidate_key"])
        for col, default in [("seed", -1), ("lr", -1.0), ("num_leaves", -1), ("qty_gate", -1.0), ("n_estimators", -1)]:
            if col not in df.columns:
                df[col] = default
            else:
                df[col] = df[col].fillna(default)
        metric_cols = ["global_ratio", "global_wmape", "4_25_under_wape", "4_25_sku_p50", "ice_4_25_sku_p50", "blockbuster_under_wape", "blockbuster_sku_p50", "top20_true_volume_capture", "rank_corr_positive_skus", "1_3_ratio"]
        group_cols = ["candidate_key", "track", "variant_key", "seed", "lr", "num_leaves", "qty_gate", "n_estimators"]
        agg = {col: (col, "mean") for col in metric_cols}
        if include_anchor_pass:
            agg["anchor_passes"] = ("anchor_pass", "sum")
        return df.groupby(group_cols, as_index=False).agg(**agg)

    def add_raw_gate(self, raw_summary):
        base = raw_summary.loc[raw_summary["candidate_key"] == "lightgbm_raw_g025"].iloc[0]
        out = raw_summary.copy()
        out["tail_gain_like"] = (
            (out["blockbuster_under_wape"] <= float(base["blockbuster_under_wape"]) - 0.02)
            | (out["blockbuster_sku_p50"] >= float(base["blockbuster_sku_p50"]) + 0.03)
        )
        out["allocation_gain_like"] = (
            (out["top20_true_volume_capture"] >= float(base["top20_true_volume_capture"]) + 0.02)
            | (out["rank_corr_positive_skus"] >= float(base["rank_corr_positive_skus"]) + 0.02)
        )
        out["raw_stage_pass"] = (
            out["global_ratio"].between(0.90, 1.10)
            & (out["4_25_under_wape"] <= float(base["4_25_under_wape"]) + 0.01)
            & (out["4_25_sku_p50"] >= float(base["4_25_sku_p50"]) - 0.01)
            & (out["ice_4_25_sku_p50"] >= float(base["ice_4_25_sku_p50"]) - 0.01)
            & (out["tail_gain_like"] | out["allocation_gain_like"])
        )
        return out

    def add_cal_gates(self, cal_summary):
        base = cal_summary.loc[cal_summary["candidate_key"] == "lightgbm_sep098_oct093"].iloc[0]
        out = cal_summary.copy()
        out["tail_gain_like"] = (
            (out["blockbuster_under_wape"] <= float(base["blockbuster_under_wape"]) - 0.02)
            | (out["blockbuster_sku_p50"] >= float(base["blockbuster_sku_p50"]) + 0.03)
        )
        out["allocation_gain_like"] = (
            (out["top20_true_volume_capture"] >= float(base["top20_true_volume_capture"]) + 0.02)
            | (out["rank_corr_positive_skus"] >= float(base["rank_corr_positive_skus"]) + 0.02)
        )
        out["potential_gate_like"] = (
            (out["anchor_passes"] >= 4)
            & (out["4_25_under_wape"] <= float(base["4_25_under_wape"]) + 0.015)
            & (out["4_25_sku_p50"] >= float(base["4_25_sku_p50"]) - 0.015)
            & (out["ice_4_25_sku_p50"] >= float(base["ice_4_25_sku_p50"]) - 0.015)
            & out["global_ratio"].between(0.90, 1.10)
            & (out["tail_gain_like"] | out["allocation_gain_like"])
            & (out["1_3_ratio"] <= 1.45)
        )
        out["replacement_gate_like"] = (
            (out["anchor_passes"] >= 4)
            & (out["4_25_under_wape"] <= float(base["4_25_under_wape"]) + 0.01)
            & (out["4_25_sku_p50"] >= float(base["4_25_sku_p50"]) - 0.01)
            & (out["ice_4_25_sku_p50"] >= float(base["ice_4_25_sku_p50"]) - 0.01)
            & out["global_ratio"].between(0.90, 1.10)
            & out["tail_gain_like"]
            & out["allocation_gain_like"]
            & (out["1_3_ratio"] <= 1.45)
        )
        return out

    def stage_outputs(self, stage_name, jobs):
        baseline = self.build_baseline_rows()
        raw_anchor = pd.concat([self.build_current_raw_rows(), self.build_candidate_rows(jobs, calibrated=False)], ignore_index=True, sort=False)
        cal_anchor = pd.concat([self.build_current_cal_rows(), self.build_candidate_rows(jobs, calibrated=True)], ignore_index=True, sort=False)
        cal_anchor = self.add_anchor_pass(cal_anchor, baseline)

        raw_anchor.to_csv(os.path.join(self.report_dir, f"{stage_name}_raw_anchor_table.csv"), index=False, encoding="utf-8-sig")
        cal_anchor.to_csv(os.path.join(self.report_dir, f"{stage_name}_calibrated_anchor_table.csv"), index=False, encoding="utf-8-sig")

        raw_summary = self.add_raw_gate(self.summarize_candidates(raw_anchor, include_anchor_pass=False))
        cal_summary = self.add_cal_gates(self.summarize_candidates(cal_anchor, include_anchor_pass=True))

        raw_disp = raw_summary.copy()
        cal_disp = cal_summary.copy()
        for frame in (raw_disp, cal_disp):
            for col in numeric_cols_for_rounding(frame):
                if col in frame.columns:
                    frame[col] = frame[col].map(lambda v: "" if pd.isna(v) else f"{float(v):.4f}")
        lines = [f"# {stage_name.upper()} Summary", "", f"- baseline_mainline: `{self.mainline['mainline_candidate']}`", "", "## Raw Compare", "", "| candidate_key | track | raw_stage_pass | tail_gain_like | allocation_gain_like | global_ratio | 4_25_under_wape | 4_25_sku_p50 | ice_4_25_sku_p50 | blockbuster_under_wape | blockbuster_sku_p50 | top20_true_volume_capture | rank_corr_positive_skus |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for _, row in raw_disp.iterrows():
            lines.append(f"| {row['candidate_key']} | {row['track']} | {row.get('raw_stage_pass', '')} | {row.get('tail_gain_like', '')} | {row.get('allocation_gain_like', '')} | {row['global_ratio']} | {row['4_25_under_wape']} | {row['4_25_sku_p50']} | {row['ice_4_25_sku_p50']} | {row['blockbuster_under_wape']} | {row['blockbuster_sku_p50']} | {row['top20_true_volume_capture']} | {row['rank_corr_positive_skus']} |")
        lines.extend(["", "## Calibrated Compare", "", "| candidate_key | track | replacement_gate_like | potential_gate_like | anchor_passes | global_ratio | 4_25_under_wape | 4_25_sku_p50 | ice_4_25_sku_p50 | blockbuster_under_wape | blockbuster_sku_p50 | top20_true_volume_capture | rank_corr_positive_skus | 1_3_ratio |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"])
        for _, row in cal_disp.iterrows():
            ap = row.get("anchor_passes", "")
            if ap != "":
                ap = int(ap)
            lines.append(f"| {row['candidate_key']} | {row['track']} | {row.get('replacement_gate_like', '')} | {row.get('potential_gate_like', '')} | {ap} | {row['global_ratio']} | {row['4_25_under_wape']} | {row['4_25_sku_p50']} | {row['ice_4_25_sku_p50']} | {row['blockbuster_under_wape']} | {row['blockbuster_sku_p50']} | {row['top20_true_volume_capture']} | {row['rank_corr_positive_skus']} | {row['1_3_ratio']} |")
        write_text(os.path.join(self.report_dir, f"{stage_name}_summary.md"), "\n".join(lines))
        return raw_summary, cal_summary

    def final_outputs(self, winner_action, reason, final_summary):
        disp = final_summary.copy()
        for col in numeric_cols_for_rounding(disp):
            if col in disp.columns:
                disp[col] = disp[col].map(lambda v: "" if pd.isna(v) else f"{float(v):.4f}")
        lines = ["# Phase7 Overnight Summary", "", f"- run_id: `{self.run_id}`", f"- baseline_mainline: `{self.mainline['mainline_candidate']}`", f"- winner_action: `{winner_action}`", f"- reason_or_selected: `{reason}`", "", "## Final Calibrated Leaderboard", "", "| candidate_key | track | replacement_gate_like | potential_gate_like | anchor_passes | 4_25_under_wape | 4_25_sku_p50 | ice_4_25_sku_p50 | blockbuster_under_wape | blockbuster_sku_p50 | top20_true_volume_capture | rank_corr_positive_skus | 1_3_ratio |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for _, row in disp.sort_values(["replacement_gate_like", "blockbuster_under_wape", "blockbuster_sku_p50", "top20_true_volume_capture", "rank_corr_positive_skus"], ascending=[False, True, False, False, False]).iterrows():
            ap = row.get("anchor_passes", "")
            if ap != "":
                ap = int(ap)
            lines.append(f"| {row['candidate_key']} | {row['track']} | {row.get('replacement_gate_like', '')} | {row.get('potential_gate_like', '')} | {ap} | {row['4_25_under_wape']} | {row['4_25_sku_p50']} | {row['ice_4_25_sku_p50']} | {row['blockbuster_under_wape']} | {row['blockbuster_sku_p50']} | {row['top20_true_volume_capture']} | {row['rank_corr_positive_skus']} | {row['1_3_ratio']} |")
        write_text(os.path.join(self.report_dir, "overnight_summary.md"), "\n".join(lines))
        write_json(os.path.join(self.report_dir, "overnight_winner.json"), {"run_id": self.run_id, "winner_action": winner_action, "reason_or_selected": reason, "baseline_mainline": self.mainline["mainline_candidate"], "final_calibrated_summary": final_summary.to_dict(orient="records")})

    def run_stage(self, stage_name, jobs):
        self.stage_status(stage_name, "running", jobs_total=len(jobs), jobs_completed=0, jobs_failed=0)
        done = 0
        for idx, job in enumerate(jobs, start=1):
            self.stage_status(stage_name, "running", jobs_total=len(jobs), jobs_completed=done, current_job=job["job_key"], job_index=idx)
            self.run_job(job)
            done += 1
        self.stage_status(stage_name, "completed", jobs_total=len(jobs), jobs_completed=done, jobs_failed=0)

    def run_smoke(self):
        jobs = self.build_s0_jobs()
        self.run_stage("stage_s0", jobs)
        _, artifacts_dir = self.ensure_assets("2025-12-01")
        meta = read_json(os.path.join(artifacts_dir, "meta_v6_event.json"))
        if meta.get("split_date") != "2025-12-01":
            raise RuntimeError("smoke split_date mismatch")
        for key in ("qfo_cols", "tail_cols", "prior_cols"):
            if not meta.get(key):
                raise RuntimeError(f"smoke missing {key}")
        for key in ("qfo", "tail", "priors"):
            if key not in meta.get("feature_groups", {}):
                raise RuntimeError(f"smoke missing feature group {key}")
        meta_ok, meta_note = self.validate_meta(jobs[0])
        if not meta_ok:
            raise RuntimeError(f"smoke meta invalid: {meta_note}")
        ctx_ok, ctx_note = self.validate_context(jobs[0])
        if not ctx_ok:
            raise RuntimeError(f"smoke context invalid: {ctx_note}")
        self.stage_outputs("stage_s0", jobs)
        return jobs

    def run_full(self):
        self.run_smoke()
        s1_jobs = self.build_s1_jobs()
        self.run_stage("stage_s1", s1_jobs)
        s1_raw, s1_cal = self.stage_outputs("stage_s1", s1_jobs)
        if not bool(s1_raw.loc[~s1_raw["candidate_key"].isin(["lightgbm_raw_g025"]), "raw_stage_pass"].fillna(False).any()):
            self.final_outputs("freeze_current_tree_mainline", "stage_s1_all_weak", s1_cal)
            return
        s2_jobs = self.build_s2_jobs()
        self.run_stage("stage_s2", s2_jobs)
        _, s2_cal = self.stage_outputs("stage_s2", s2_jobs)
        if not bool(s2_cal.loc[~s2_cal["candidate_key"].isin(["lightgbm_sep098_oct093"]), "potential_gate_like"].fillna(False).any()):
            self.final_outputs("freeze_current_tree_mainline", "stage_s2_no_promising_candidate", s2_cal)
            return
        s3_jobs = self.build_s3_jobs(s2_cal)
        self.run_stage("stage_s3", s3_jobs)
        _, s3_cal = self.stage_outputs("stage_s3", s3_jobs)
        promoted = s3_cal.loc[(~s3_cal["candidate_key"].isin(["lightgbm_sep098_oct093"])) & (s3_cal["replacement_gate_like"].fillna(False))].copy()
        if promoted.empty:
            self.final_outputs("freeze_current_tree_mainline", "stage_s3_no_replacement", s3_cal)
            return
        promoted = promoted.sort_values(["blockbuster_under_wape", "blockbuster_sku_p50", "top20_true_volume_capture", "rank_corr_positive_skus"], ascending=[True, False, False, False])
        self.final_outputs("promote_phase7_candidate", str(promoted.iloc[0]["candidate_key"]), s3_cal)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--run-id", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    run_id = args.run_id.strip() or now_id()
    runner = Runner(run_id)
    print(f"[Phase7 Overnight] mode={args.mode} run_id={run_id}")
    print(f"[Phase7 Overnight] report_dir={runner.report_dir}")
    print(f"[Phase7 Overnight] model_dir={runner.model_dir}")
    if args.mode == "smoke":
        runner.run_smoke()
        print("[OK] smoke passed")
        return
    runner.run_full()
    print("[DONE] overnight chain completed")


if __name__ == "__main__":
    main()
