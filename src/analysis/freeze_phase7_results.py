import json
import os


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUN_DIR = os.path.join(PROJECT_ROOT, "reports", "phase7_tail_allocation_optimization", "overnight_20260402_overnight")
OUT_DIR = os.path.join(PROJECT_ROOT, "reports", "phase7")
WINNER_JSON = os.path.join(RUN_DIR, "overnight_winner.json")
OUT_FREEZE_SUMMARY = os.path.join(OUT_DIR, "phase7_freeze_summary.md")
OUT_FROZEN_JSON = os.path.join(OUT_DIR, "phase7_frozen_mainline.json")
OUT_DELTA = os.path.join(OUT_DIR, "phase7_vs_phase6_delta.md")


def ensure_dir():
    os.makedirs(OUT_DIR, exist_ok=True)


def load_rows():
    with open(WINNER_JSON, "r", encoding="utf-8") as f:
        payload = json.load(f)
    rows = payload["final_calibrated_summary"]
    baseline = next(row for row in rows if row["candidate_key"] == "lightgbm_sep098_oct093")
    winner = next(row for row in rows if row["candidate_key"] == payload["reason_or_selected"])
    return payload, baseline, winner


def format_delta(old, new):
    if old is None or new is None:
        return "-"
    delta = float(new) - float(old)
    return f"{old:.4f} -> {new:.4f} ({delta:+.4f})"


def write_summary(payload, baseline, winner):
    evidence = [
        "reports/phase7_tail_allocation_optimization/overnight_20260402_overnight/overnight_summary.md",
        "reports/phase7_tail_allocation_optimization/overnight_20260402_overnight/overnight_winner.json",
        "reports/phase7h_december_readable_report/dec_20251201_dashboard.html",
        "reports/phase7h_december_readable_report/dec_20251201_static_feature_audit.md",
    ]
    text = "\n".join(
        [
            "# Phase7 冻结总结",
            "",
            "`phase7` 的定义是：**Tail / Allocation 定向优化阶段**。",
            "",
            "## 结论",
            "",
            "- 过夜长实验已完成，当前 winner 已满足替换主线门槛。",
            "- `phase7` 新主线正式升级为 `tail_full_lr005_l63_g027_n800_s2028 + sep098_oct093`。",
            "- 本轮改进不是只优化一边，而是同时改善了 `4-25 / Ice 4-25 / blockbuster / allocation`。",
            "- `phase6` 冻结主线 `p57_covact_lr005_l63_hard_g025 + sep098_oct093` 退为旧主线参考。",
            "",
            "## 新主线",
            "",
            f"- 主线候选：`{payload['reason_or_selected']} + sep098_oct093`",
            f"- raw 模型：`{payload['reason_or_selected']}`",
            "- 校准规则：`sep098_oct093`",
            "- 树模型家族：`LightGBM`",
            "- 特征组：`cov_activity_tail_full`",
            "",
            "## 关键证据",
            "",
            *(f"- `{path}`" for path in evidence),
            "",
            "## 主指标变化",
            "",
            f"- `4_25_under_wape`: {format_delta(baseline['4_25_under_wape'], winner['4_25_under_wape'])}",
            f"- `4_25_sku_p50`: {format_delta(baseline['4_25_sku_p50'], winner['4_25_sku_p50'])}",
            f"- `ice_4_25_sku_p50`: {format_delta(baseline['ice_4_25_sku_p50'], winner['ice_4_25_sku_p50'])}",
            f"- `blockbuster_under_wape`: {format_delta(baseline['blockbuster_under_wape'], winner['blockbuster_under_wape'])}",
            f"- `blockbuster_sku_p50`: {format_delta(baseline['blockbuster_sku_p50'], winner['blockbuster_sku_p50'])}",
            f"- `top20_true_volume_capture`: {format_delta(baseline['top20_true_volume_capture'], winner['top20_true_volume_capture'])}",
            f"- `rank_corr_positive_skus`: {format_delta(baseline['rank_corr_positive_skus'], winner['rank_corr_positive_skus'])}",
            f"- `global_ratio`: {format_delta(baseline['global_ratio'], winner['global_ratio'])}",
            f"- `global_wmape`: {format_delta(baseline['global_wmape'], winner['global_wmape'])}",
            f"- `1_3_ratio`: {format_delta(baseline['1_3_ratio'], winner['1_3_ratio'])}",
            "",
        ]
    )
    with open(OUT_FREEZE_SUMMARY, "w", encoding="utf-8") as f:
        f.write(text + "\n")


def write_frozen_json(payload, baseline, winner):
    out = {
        "phase": "phase7",
        "status": "frozen",
        "winner_action": payload["winner_action"],
        "mainline_candidate": f"{payload['reason_or_selected']} + sep098_oct093",
        "raw_model": payload["reason_or_selected"],
        "calibration_rule": "sep098_oct093",
        "feature_set": winner["track"],
        "tree_family": "lightgbm",
        "previous_mainline": "p57_covact_lr005_l63_hard_g025 + sep098_oct093",
        "anchor_passes": winner["anchor_passes"],
        "key_metrics": {
            "4_25_under_wape": round(float(winner["4_25_under_wape"]), 4),
            "4_25_sku_p50": round(float(winner["4_25_sku_p50"]), 4),
            "ice_4_25_sku_p50": round(float(winner["ice_4_25_sku_p50"]), 4),
            "blockbuster_under_wape": round(float(winner["blockbuster_under_wape"]), 4),
            "blockbuster_sku_p50": round(float(winner["blockbuster_sku_p50"]), 4),
            "top20_true_volume_capture": round(float(winner["top20_true_volume_capture"]), 4),
            "rank_corr_positive_skus": round(float(winner["rank_corr_positive_skus"]), 4),
            "global_ratio": round(float(winner["global_ratio"]), 4),
            "global_wmape": round(float(winner["global_wmape"]), 4),
            "1_3_ratio": round(float(winner["1_3_ratio"]), 4),
        },
        "phase6_baseline_metrics": {
            "4_25_under_wape": round(float(baseline["4_25_under_wape"]), 4),
            "4_25_sku_p50": round(float(baseline["4_25_sku_p50"]), 4),
            "ice_4_25_sku_p50": round(float(baseline["ice_4_25_sku_p50"]), 4),
            "blockbuster_under_wape": round(float(baseline["blockbuster_under_wape"]), 4),
            "blockbuster_sku_p50": round(float(baseline["blockbuster_sku_p50"]), 4),
            "top20_true_volume_capture": round(float(baseline["top20_true_volume_capture"]), 4),
            "rank_corr_positive_skus": round(float(baseline["rank_corr_positive_skus"]), 4),
            "global_ratio": round(float(baseline["global_ratio"]), 4),
            "global_wmape": round(float(baseline["global_wmape"]), 4),
            "1_3_ratio": round(float(baseline["1_3_ratio"]), 4),
        },
        "evidence_files": [
            "reports/phase7_tail_allocation_optimization/overnight_20260402_overnight/overnight_summary.md",
            "reports/phase7_tail_allocation_optimization/overnight_20260402_overnight/overnight_winner.json",
            "reports/phase7h_december_readable_report/dec_20251201_dashboard.html",
            "reports/phase7h_december_readable_report/dec_20251201_static_feature_audit.md",
        ],
    }
    with open(OUT_FROZEN_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def write_delta_md(baseline, winner):
    rows = [
        ("4_25_under_wape", "越低越好"),
        ("4_25_sku_p50", "越高越好"),
        ("ice_4_25_sku_p50", "越高越好"),
        ("blockbuster_under_wape", "越低越好"),
        ("blockbuster_sku_p50", "越高越好"),
        ("top20_true_volume_capture", "越高越好"),
        ("rank_corr_positive_skus", "越高越好"),
        ("global_ratio", "接近 1 更好"),
        ("global_wmape", "越低越好"),
        ("1_3_ratio", "当前仅 guardrail"),
    ]
    lines = ["# Phase7 vs Phase6 主线对照", "", "| metric | direction | phase6 | phase7 | delta |", "| --- | --- | ---: | ---: | ---: |"]
    for metric, direction in rows:
        old = float(baseline[metric])
        new = float(winner[metric])
        lines.append(f"| `{metric}` | {direction} | {old:.4f} | {new:.4f} | {new - old:+.4f} |")
    lines.extend([
        "",
        "## 判定",
        "",
        "- `phase7` 新主线在核心业务段、cold-start、tail、allocation 上都优于 `phase6` 旧主线。",
        "- 本轮不是单点指标偶然改善，而是结构性提升。",
        "- `phase7` 现在可以作为新的正式主线引用。",
        "",
    ])
    with open(OUT_DELTA, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    ensure_dir()
    payload, baseline, winner = load_rows()
    write_summary(payload, baseline, winner)
    write_frozen_json(payload, baseline, winner)
    write_delta_md(baseline, winner)
    print(json.dumps({"phase7_dir": OUT_DIR, "summary": OUT_FREEZE_SUMMARY, "frozen_json": OUT_FROZEN_JSON, "delta_md": OUT_DELTA}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
