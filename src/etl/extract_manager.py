# 📁 文件位置: src/etl/extract_manager.py
import oracledb
import pandas as pd
import os
import datetime
import warnings
import time

warnings.filterwarnings('ignore')

# ================= 配置 =================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
DATA_WAREHOUSE_DIR = os.path.join(PROJECT_ROOT, "data_warehouse")
TODAY_STR = datetime.datetime.now().strftime("%Y%m%d")

DB_CONFIG = {
    "user": "IRSRead",
    "password": "IrS_REaD$2025",
    "dsn": "47.104.90.96:1521/RFZS"
}

# ================= 任务清单 =================
TASKS = [
    # --- 基础维表 ---
    {
        "table": "BOSNDS3.V_IRS_PRODUCT",
        "folder": "dim_product",
        "file_name": "product_info",
        "strategy": "overwrite",
        [cite_start]"desc": "商品基础信息 [cite: 5]"
    },
    {
        "table": "BOSNDS3.V_IRS_STORE",
        "folder": "dim_store",
        "file_name": "store_info",
        "strategy": "overwrite",
        [cite_start]"desc": "门店基础信息 [cite: 9]"
    },

    # --- 状态快照 ---
    {
        "table": "BOSNDS3.V_IRS_STORAGE",
        "folder": "snapshot_inventory",
        "file_name": "storage_stock",
        "strategy": "snapshot",
        [cite_start]"desc": "库存快照 [cite: 13]"
    },
    {
        "table": "BOSNDS3.V_IRS_B2BSTORAGE",
        "folder": "snapshot_inventory",
        "file_name": "b2b_stock",
        "strategy": "snapshot",
        [cite_start]"desc": "B2B预售库存 [cite: 21]"
    },

    # --- 交易流水 (增量) ---
    {
        "table": "BOSNDS3.V_IRS_ORDER",
        "folder": "fact_orders",
        "file_name": "order_history",
        "strategy": "append",
        [cite_start]"desc": "订单流水(增量) [cite: 17]",
        "time_col": "modifieddate"
    },
    {
        "table": "BOSNDS3.V_IRS_PREORDER",
        "folder": "fact_orders",
        "file_name": "b2b_preorder",
        "strategy": "append",
        [cite_start]"desc": "B2B预售订单(增量) [cite: 25]",
        "time_col": "modifieddate"
    },
    {
        "table": "BOSNDS3.V_IRS_EVENT",
        "folder": "fact_events",
        "file_name": "user_events",
        "strategy": "append",
        [cite_start]"desc": "用户行为埋点(增量) [cite: 33]",
        "time_col": "Creationdate"
    },
    
    # --- 计算型大表 (重点修改处) ---
    
    # ❌ [已禁用] 表7：因计算量大且未物理化，暂时跳过以免超时
    # {
    #     "table": "BOSNDS3.V_IRS_PRO_DATA", 
    #     "folder": "snapshot_metrics",
    #     "file_name": "pro_data_calc",
    #     "strategy": "snapshot", 
    #     [cite_start]"desc": "商品计算指标 [cite: 29]"
    # },

    # ✅ [已启用] 表10：商户画像，下载速度快，包含关键特征
    {
        "table": "BOSNDS3.V_IRS_CUS_PROFILE", 
        "folder": "snapshot_metrics",
        "file_name": "customer_profile",
        "strategy": "snapshot",
        [cite_start]"desc": "商户画像指标 [cite: 47]"
    }
]

# ================= 执行逻辑 =================
def run_pipeline():
    print(f"🚀 [ETL] 开始执行数据同步 (Plan B: 跳过表7)")
    if not os.path.exists(DATA_WAREHOUSE_DIR):
        os.makedirs(DATA_WAREHOUSE_DIR)

    try:
        oracledb.init_oracle_client(lib_dir="/root/instantclient_19_8")
    except:
        pass

    try:
        conn = oracledb.connect(**DB_CONFIG)
        print("✅ 数据库连接成功\n")

        for task in TASKS:
            full_path = os.path.join(DATA_WAREHOUSE_DIR, task['folder'])
            if not os.path.exists(full_path): os.makedirs(full_path)
            
            fname = f"{task['file_name']}_latest.csv" if task['strategy'] == 'overwrite' else f"{TODAY_STR}_{task['file_name']}.csv"
            save_path = os.path.join(full_path, fname)
            
            print(f"👉 正在处理: {task['desc']} ...")
            
            # 构建 SQL
            if task['strategy'] == 'append' and 'time_col' in task:
                sql = f"SELECT * FROM {task['table']} WHERE {task['time_col']} >= TRUNC(SYSDATE)"
            else:
                sql = f"SELECT * FROM {task['table']}" # 全量读取
            
            try:
                df = pd.read_sql(sql, conn)
                if not df.empty:
                    df.to_csv(save_path, index=False, encoding='utf-8-sig')
                    print(f"   ✅ 完成 | 行数: {len(df)} | 已保存")
                else:
                    print(f"   ⚠️ 无数据 (行数为0)")
            except Exception as e:
                print(f"   ❌ 失败: {e}")

        conn.close()
        print("\n🏁 ETL 任务结束")

    except Exception as e:
        print(f"❌ 严重错误: {e}")

if __name__ == "__main__":
    run_pipeline()