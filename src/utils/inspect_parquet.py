import pandas as pd
import os
import glob

# ================= 配置 =================
# 指向存放 parquet 文件的目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data_warehouse", "fact_orders")

def inspect_parquet():
    print(f"🔍 开始检查目录: {DATA_DIR}")
    
    # 查找所有 parquet 文件
    parquet_files = glob.glob(os.path.join(DATA_DIR, "*.parquet"))
    
    if not parquet_files:
        print("❌ 未找到任何 .parquet 文件！请检查路径或文件后缀。")
        return

    print(f"✅ 发现 {len(parquet_files)} 个 Parquet 文件。")
    print("-" * 50)

    # 为了避免刷屏，只检查第一个文件（通常结构是一样的）
    # 或者你可以循环检查所有
    target_file = parquet_files[0]
    print(f"📄 正在深入分析文件: {os.path.basename(target_file)}")
    
    try:
        # 读取数据
        df = pd.read_parquet(target_file)
        
        # 1. 基础信息
        print(f"\n[1] 数据概览:")
        print(f"    - 行数: {len(df)}")
        print(f"    - 列数: {len(df.columns)}")
        print(f"    - 字段列表: {list(df.columns)}")
        
        # 2. 尝试识别关键日期列
        # 常见的日期字段名猜测
        date_candidates = ['order_date', 'date', 'modifieddate', 'billdate', 'createddate', '日期', '单据日期']
        date_col = None
        for col in df.columns:
            if col.lower() in date_candidates:
                date_col = col
                break
        
        if date_col:
            # 转换日期看看范围
            try:
                dates = pd.to_datetime(df[date_col])
                print(f"\n[2] 时间范围 ({date_col}):")
                print(f"    - 最早: {dates.min()}")
                print(f"    - 最晚: {dates.max()}")
            except:
                print(f"    ⚠️ 找到列 {date_col} 但无法解析为日期")
        else:
            print(f"\n[2] ⚠️ 未自动识别到明显的日期列，请人工核对字段列表！")

        # 3. 检查核心指标 (Qty)
        # 我们 Step 2 需要的映射: qtyso, qtyspo, qtyout, qtypur
        print(f"\n[3] 关键指标字段检查:")
        expected_keywords = ['qty', '数量', 'so', 'spo', 'out']
        found_keywords = [col for col in df.columns if any(k in col.lower() for k in expected_keywords)]
        print(f"    - 包含 'qty' 或 '数量' 的列: {found_keywords}")

        # 4. 数据预览
        print(f"\n[4] 前 3 行数据预览:")
        pd.set_option('display.max_columns', None) # 显示所有列
        print(df.head(3))

    except Exception as e:
        print(f"❌ 读取失败: {e}")
        print("💡 提示: 缺少依赖？请尝试运行 pip install pyarrow")

if __name__ == "__main__":
    inspect_parquet()