"""
check_phase5_code.py — Phase 5 代码与数据预检脚本
运行: python check_phase5_code.py
"""
import ast
import json
import os
import sys


if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def banner(text):
    print(f"\n{'=' * 68}")
    print(f"  {text}")
    print(f"{'=' * 68}")


def check_syntax(rel_path):
    full = os.path.join(PROJECT_ROOT, rel_path)
    if not os.path.exists(full):
        print(f"  [MISSING] {rel_path}")
        return False
    try:
        with open(full, "r", encoding="utf-8") as f:
            src = f.read()
        ast.parse(src)
        print(f"  [OK]      {rel_path}")
        return True
    except SyntaxError as e:
        print(f"  [SYNTAX]  {rel_path}: line {e.lineno} - {e.msg}")
        return False
    except Exception as e:
        print(f"  [READERR] {rel_path}: {e}")
        return False


def check_contains(rel_path, symbols):
    full = os.path.join(PROJECT_ROOT, rel_path)
    if not os.path.exists(full):
        print(f"  [MISSING] {rel_path}")
        return False
    with open(full, "r", encoding="utf-8") as f:
        src = f.read()

    ok = True
    print(f"  [{rel_path}]")
    for sym, desc in symbols.items():
        found = sym in src
        status = "[OK]" if found else "[MISS]"
        print(f"    {status:<7} {desc}")
        ok = ok and found
    return ok


def check_path(rel_path, label, required=True):
    full = os.path.join(PROJECT_ROOT, rel_path)
    exists = os.path.exists(full)
    if exists:
        if os.path.isdir(full):
            print(f"  [OK]      {label}: {rel_path}/")
        else:
            size_mb = os.path.getsize(full) / 1024**2
            print(f"  [OK]      {label}: {rel_path} ({size_mb:.1f} MB)")
        return True
    if required:
        print(f"  [MISSING] {label}: {rel_path}")
        return False
    print(f"  [INFO]    {label}: {rel_path} (运行时自动创建)")
    return True


def load_meta(rel_path):
    full = os.path.join(PROJECT_ROOT, rel_path)
    with open(full, "r", encoding="utf-8") as f:
        return json.load(f)


def smoke_shape(version, lookback, dyn_feat_dim):
    from src.train.dataset import create_lazy_dataloaders

    processed_dir = os.path.join(PROJECT_ROOT, "data", f"processed_{version}")
    train_loader, val_loader = create_lazy_dataloaders(
        processed_dir=processed_dir,
        batch_size=2,
        num_workers=0,
        lookback=lookback,
        dyn_feat_dim=dyn_feat_dim,
    )

    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    train_x_dyn, train_x_static, train_y_cls, train_y_reg = train_batch
    val_x_dyn, val_x_static, val_y_cls, val_y_reg = val_batch

    checks = [
        (tuple(train_x_dyn.shape) == (2, lookback, dyn_feat_dim), f"{version} train x_dyn == (2, {lookback}, {dyn_feat_dim})"),
        (tuple(val_x_dyn.shape) == (2, lookback, dyn_feat_dim), f"{version} val x_dyn == (2, {lookback}, {dyn_feat_dim})"),
        (train_x_static.shape[1] == 13, f"{version} train x_static dim == 13"),
        (val_x_static.shape[1] == 13, f"{version} val x_static dim == 13"),
        (tuple(train_y_cls.shape) == (2, 1), f"{version} train y_cls == (2, 1)"),
        (tuple(train_y_reg.shape) == (2, 1), f"{version} train y_reg == (2, 1)"),
    ]

    ok = True
    for passed, desc in checks:
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {status:<7} {desc}")
        ok = ok and passed
    return ok


def main():
    overall_ok = True

    banner("1. 语法检查")
    files_to_check = [
        "src/train/dataset.py",
        "src/train/run_training_v2.py",
        "evaluate.py",
        "run_phase5_experiments.py",
        "check_phase5_code.py",
    ]
    for rel in files_to_check:
        overall_ok = check_syntax(rel) and overall_ok

    banner("2. 关键实现检查")
    overall_ok = check_contains("src/train/dataset.py", {
        "lookback=None": "Dataset 接口支持显式 lookback",
        "expected_dyn_floats": "显式契约会校验动态张量大小",
        "lookback=lookback": "DataLoader 将 lookback 传入 Dataset",
        "persistent_workers": "DataLoader 启用持久 worker",
        "prefetch_factor": "DataLoader 启用 prefetch",
    }) and overall_ok
    overall_ok = check_contains("src/train/run_training_v2.py", {
        "lookback = meta.get('lookback', 90)": "训练入口从 meta 读取 lookback",
        "dyn_feat_dim = meta.get('dyn_feat_dim', 7)": "训练入口从 meta 读取 dyn_feat_dim",
        "lookback=lookback": "训练入口将 lookback 传给 DataLoader",
        "EXP_WORKERS": "训练入口支持 worker 注入",
    }) and overall_ok
    overall_ok = check_contains("evaluate.py", {
        "EXP_VERSION = os.environ.get('EXP_VERSION', 'v3').lower()": "评估入口支持 EXP_VERSION 路由",
        "MODEL_DIR     = os.path.join(PROJECT_ROOT, 'models_v5')": "V5 评估指向 models_v5",
        "lookback=meta.get('lookback', 90)": "评估入口将 lookback 传给 DataLoader",
        "dyn_feat_dim=meta.get('dyn_feat_dim', 7)": "评估入口将 dyn_feat_dim 传给 DataLoader",
        "EXP_EVAL_WORKERS": "评估入口支持 worker 注入",
    }) and overall_ok
    overall_ok = check_contains("run_phase5_experiments.py", {
        "def get_model_dir(version):": "Runner 有统一模型目录映射",
        "models_v2": "V3 映射到 models_v2",
        "models_v5": "V5 映射到 models_v5",
        "def run_preflight():": "Runner 有正式 preflight",
        "\"train_error\"": "训练失败会终止流水线",
        "\"eval_error\"": "评估失败会终止流水线",
        "\"agg_error\"": "聚合评估失败会终止流水线",
        "e62_bilstm_l3_v5_d04_ls": "Runner 使用当前 12 组实验矩阵",
        "PHASE5_WORKERS": "Runner 统一注入 workers",
        "PHASE5_PATIENCE": "Runner patience 可配置",
    }) and overall_ok

    banner("3. 数据与目录检查")
    paths = [
        ("data/processed_v3/X_train_dyn.bin", "V3 train dyn", True),
        ("data/processed_v3/X_val_dyn.bin", "V3 val dyn", True),
        ("data/artifacts_v3/meta_v2.json", "V3 meta", True),
        ("data/processed_v5/X_train_dyn.bin", "V5 train dyn", True),
        ("data/processed_v5/X_val_dyn.bin", "V5 val dyn", True),
        ("data/artifacts_v5/meta_v5.json", "V5 meta", True),
        ("models_v2", "V3 model dir", False),
        ("models_v5", "V5 model dir", False),
        ("reports/phase5", "Phase5 report dir", False),
        ("reports/phase5_timing_log.csv", "Phase5 timing log", False),
    ]
    for rel, label, required in paths:
        overall_ok = check_path(rel, label, required=required) and overall_ok

    banner("4. Meta 契约")
    try:
        meta_v3 = load_meta("data/artifacts_v3/meta_v2.json")
        print(
            f"  [OK]      V3 meta: lookback={meta_v3.get('lookback', 90)}, "
            f"dyn_feat_dim={meta_v3.get('dyn_feat_dim', 7)}, static_dim={meta_v3.get('static_dim')}"
        )
    except Exception as e:
        print(f"  [FAIL]    V3 meta 读取失败: {e}")
        overall_ok = False
        meta_v3 = None

    try:
        meta_v5 = load_meta("data/artifacts_v5/meta_v5.json")
        print(
            f"  [OK]      V5 meta: lookback={meta_v5.get('lookback', 90)}, "
            f"dyn_feat_dim={meta_v5.get('dyn_feat_dim', 7)}, static_dim={meta_v5.get('static_dim')}"
        )
    except Exception as e:
        print(f"  [FAIL]    V5 meta 读取失败: {e}")
        overall_ok = False
        meta_v5 = None

    banner("5. Shape Smoke Check")
    if meta_v3 is not None:
        try:
            overall_ok = smoke_shape("v3", meta_v3.get("lookback", 90), meta_v3.get("dyn_feat_dim", 7)) and overall_ok
        except Exception as e:
            print(f"  [FAIL]    V3 shape smoke check 失败: {e}")
            overall_ok = False
    if meta_v5 is not None:
        try:
            overall_ok = smoke_shape("v5", meta_v5.get("lookback", 90), meta_v5.get("dyn_feat_dim", 7)) and overall_ok
        except Exception as e:
            print(f"  [FAIL]    V5 shape smoke check 失败: {e}")
            overall_ok = False

    banner("6. 结论")
    if overall_ok:
        print("  [READY]   Phase 5 可启动。")
        print("  [NEXT]    先运行 Group A，再审阅结果后进入 Group B/C。")
        sys.exit(0)

    print("  [BLOCKED] Phase 5 仍有阻塞项，暂不建议启动。")
    sys.exit(1)


if __name__ == "__main__":
    main()
