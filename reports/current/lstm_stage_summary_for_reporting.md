# LSTM 阶段汇报总结

## 1. 这一阶段解决了什么问题

LSTM 阶段的核心目标是：基于历史 90 天序列，预测未来 30 天的 SKU 补货需求，并形成一条可以复现、可以批量实验、可以用于业务验证的 sequence 主线。

这一阶段最终完成了 4 件事：

1. 把预测粒度从早期的 `(buyer_id, sku_id, date)` 收口到 `(sku_id, date)`。
2. 把双塔 LSTM 训练、验证、推理链路完整跑通。
3. 系统比较了多种 sequence 架构、特征版本和 seed 稳定性，冻结出最佳 LSTM 基线。
4. 明确验证了 sequence 路线的上限，并为后续 tree 路线提供了稳定对照基线。

## 2. 为什么会有 V2.0 聚合升级

最早的 V1.8 系统是在 `(buyer_id, sku_id)` 的微观原子维度上预测未来 30 天需求。这个思路的问题是：样本极度稀疏，绝大多数时空节点都是真实零值，导致模型很容易学成“保守、均值化”的输出。

项目里对这件事的正式总结在：

- `reports/v1.8_demo/V2.0_Aggregation_Strategy.md`

这个升级的逻辑是：

- 业务上真正需要的不是“某个买手某天可能买几件”，而是“某个 SKU 在未来 30 天全国总共该补多少”。
- 统计上，把 `(buyer_id, sku_id, date)` 聚合到 `(sku_id, date)` 后，序列更连续、噪声更小，模型更容易学习真正的波动结构。

文档里给出的关键数字是：

- 在活跃订单宽表上，原子级正样本占比大约只有 `4.17%`
- 聚合到 `(sku_id, date)` 后，有效正样本占比提高到 `10.37%`

这就是为什么 v2.0 不是简单换模型，而是先换建模粒度。

## 3. LSTM 阶段的模型结构

这一阶段的主模型是双塔 sequence 模型，代码入口在：

- `src/train/run_training_v2.py`
- `src/models/enhanced_model.py`

结构可以概括为：

- 左塔：动态时序塔
  - 输入过去 `90` 天动态特征
  - 用 LSTM / GRU / BiLSTM / Attention 变体编码时间序列
- 右塔：静态特征塔
  - 对 `sku_id / style_id / category / season / band / month` 等做 embedding
  - 数值静态特征如 `qty_first_order / price_tag` 单独处理
- 顶部：双头输出
  - 一个头做是否补货的分类
  - 一个头做补货量回归

这是一个标准的 two-tower + multi-task sequence 框架。

## 4. 特征版本是怎么演进的

LSTM 阶段主要经历了这些特征版本：

- `v3`
  - 早期基础版 sequence 特征
  - 核心是 90 天 lookback 的日频动态序列 + SKU 静态信息
- `v3_filtered`
  - 在 `v3` 基础上做 universe 过滤和样本清洗
  - 目标是减少明显无效样本对 sequence 的干扰
- `v5`
  - 扩展动态特征，加入更强的订单流和库存相关信号
- `v5_lite`
  - 作为最终 sequence 主线的轻量主版本
  - 在性能和训练成本之间做了更平衡的折中
- `v5_lite_cov`
  - 想继续增强 coverage / pooling / attention
  - 最终没有打赢 `v5_lite`

相关脚本在：

- `src/features/build_features_v2_sku.py`
- `src/features/build_features_v3_filtered_sku.py`
- `src/features/build_features_v5_sku.py`
- `src/features/build_features_v5_lite_sku.py`
- `src/features/build_features_v5_lite_cov_sku.py`

## 5. LSTM 阶段的关键实验路径

### 5.1 Phase2：先做架构搜索

这一阶段主要是问：sequence 路线里，哪种 RNN 架构最合适。

结果在：

- `reports/phase2/phase2_summary.csv`

核心结果：

| 实验 | 架构 | 最佳 F1 | ROC-AUC | 大盘 Ratio |
| --- | --- | ---: | ---: | ---: |
| `e11_GRU` | GRU | 0.3737 | 0.6551 | 1.23 |
| `e12_BiLSTM` | BiLSTM | 0.4791 | 0.7708 | 1.16 |
| `e13_Attn` | Attention | 0.4288 | 0.7267 | 1.19 |
| `e21_h128` | LSTM h128 | 0.4934 | 0.7989 | 0.99 |
| `e22_h384` | LSTM h384 | 0.5291 | 0.8447 | 1.19 |
| `e23_l1` | LSTM l1 | 0.4354 | 0.6693 | 1.18 |
| `e24_l3` | LSTM l3 | 0.5722 | 0.8861 | 1.16 |
| `e25_d05` | LSTM dropout=0.5 | 0.5316 | 0.8217 | 1.05 |

结论：

- 纯架构层面，`LSTM l3` 是这一阶段最强的 sequence 主结构。
- GRU、BiLSTM、Attention 都试过，但没有形成稳定替代。

### 5.2 Phase4：试过周频路线，但放弃

周频实验的日志还保留在：

- `reports/history_w4a_look_8.csv`
- `reports/history_w4a_look_12.csv`
- `reports/history_w4a_look_16.csv`
- `reports/history_w4a_look_24.csv`
- `reports/history_w4d_bilstm.csv`

从这些日志看，周频路线的 `val_f1` 基本在 `0.33 ~ 0.39` 之间，明显弱于后来的日频 sequence 主线。

结论：

- 周频思路试过
- 但对这个任务来说，日频信息损失太大
- 所以后来明确放弃 weekly 线

### 5.3 Phase5.1：做 sequence 架构与特征重检

这一阶段比较的是：

- `p511 ~ p517`
- LSTM / BiLSTM
- `v3_ref / v5 / v5_lite`
- 学习率变体

结果在：

- `reports/phase5_1/phase5_1_summary_recheck.csv`

这一轮的要点：

- `p516_lstm_l3_v3_ref` 在 AUC/F1 上不差，但业务指标不够稳
- `p511_lstm_l3_v5_lite` 和 `p512_bilstm_l3_v5_lite` 是更接近业务目标的版本
- 低学习率变体没有带来收益

### 5.4 Phase5.2：做 seed 稳定性比较，冻结最佳 LSTM 参考线

这一阶段真正决定了 sequence 最终主线。

结果在：

- `reports/phase5_2/phase5_2_decision_table.md`
- `reports/phase5_2/phase5_2_decision_table.csv`

比较对象：

- `v3_filtered`：`p521 / p522 / p525 / p526`
- `v5_lite`：`p523 / p524 / p527 / p528`

关键结论：

- 最终冻结的最佳 LSTM 是：
  - `p527_lstm_l3_v5_lite_s2027`

它在单锚点决策表上的表现是：

- `auc = 0.8627`
- `f1 = 0.5882`
- `global_ratio = 0.9361`
- `global_wmape = 1.2703`
- `4_25_ratio = 0.3735`
- `4_25_wmape_like = 0.6827`
- `4_25_sku_p50 = 0.3335`
- `ice_4_25_sku_p50 = 0.1658`
- `blockbuster_sku_p50 = 0.1002`

这里有两个重要结论：

1. `v5_lite` 明显优于 `v3_filtered`
2. LSTM 对 seed 比较敏感

例如同结构的 `p523` 和 `p527` 差异就很明显：

- `p523 global_wmape = 1.6584`
- `p527 global_wmape = 1.2703`

所以后面 sequence 线一定要讲 seed 稳定性问题。

### 5.5 Phase5.3：继续给 sequence 加复杂结构，但没有赢

这一阶段 sequence 又试了：

- `p537_lstm_pool_v5_lite_cov`
- `p539_attn_v5_lite_cov`

同时和 tree 线做正面对比。

结果在：

- `reports/phase5_3/phase5_3_decision_table.md`
- `reports/phase5_3/phase5_3_decision_table.csv`

sequence 这两条线的结果：

- `p537`
  - `4_25_sku_p50 = 0.2386`
  - `ice_4_25_sku_p50 = 0.1069`
  - `elapsed_min = 45.8`
- `p539`
  - `4_25_sku_p50 = 0.2217`
  - `ice_4_25_sku_p50 = 0.0950`
  - `elapsed_min = 41.25`

结论：

- pooling 和 attention 都试过
- 不仅没超过 `p527`
- 反而业务指标更差，训练更慢

于是 sequence 线到这里基本见顶。

### 5.6 Phase5.4：正式把 LSTM 冻结成 sequence baseline

最终 sequence 线的正式定位在：

- `reports/phase5_4/phase5_4_base_summary.csv`
- `reports/phase6e_tree_validation_pack/phase6e_tree_validation_pack_table.csv`

四锚点均值下，最佳 LSTM 基线 `p527` 的正式指标是：

- `global_wmape = 1.4735`
- `4_25_ratio = 0.2645`
- `4_25_wmape_like = 0.7484`
- `4_25_sku_p50 = 0.2329`
- `4_25_under_wape = 0.7419`
- `ice_4_25_sku_p50 = 0.0865`
- `blockbuster_sku_p50 = 0.1000`
- `blockbuster_under_wape = 0.8649`
- `top20_true_volume_capture = 0.3868`
- `rank_corr_positive_skus = 0.2669`

这就是后来所有 tree 路线对比的 sequence 基线。

## 6. LSTM 阶段最后得出的结论

### 6.1 LSTM 阶段的正面贡献

这一阶段不是失败，而是完成了非常关键的几件事：

- 明确了业务任务的正式建模粒度：`(sku, date)`
- 把 90 天 lookback / 30 天 forecast 的 sequence 预测链路跑通
- 建立了统一的 sequence 特征资产与训练入口
- 建立了正式业务指标体系，尤其是：
  - `4-25`
  - `Ice 4-25`
  - `blockbuster`
  - `allocation`
- 冻结了一个稳定可引用的 sequence baseline：`p527`

如果没有这阶段，后面的 tree 路线无法被严格证明“更好”。

### 6.2 LSTM 阶段的能力边界

LSTM 最终没有继续作为主线，核心原因有 4 个：

1. **主需求段不够强**
   - `4_25_sku_p50 = 0.2329`
   - `4_25_under_wape = 0.7419`
   - 说明主需求段明显偏保守、少补严重

2. **冷启动弱**
   - `ice_4_25_sku_p50 = 0.0865`
   - 对冷启动 SKU 的支持明显不够

3. **大货和分配能力弱**
   - `blockbuster_sku_p50 = 0.1000`
   - `top20_true_volume_capture = 0.3868`
   - `rank_corr_positive_skus = 0.2669`

4. **训练成本高、seed 敏感**
   - 最佳 LSTM 单次训练在 `30~40` 分钟量级
   - 同时期 tree 模型通常不到 `1` 分钟
   - 同结构不同 seed 的结果波动明显

所以最终不是说 LSTM 没价值，而是：

- 它足够作为 baseline
- 但已经不适合作为继续投入的主线

## 7. 一句话总结给汇报时用

可以直接这样讲：

> LSTM 阶段的核心成果，不是把最终最优模型做出来，而是把整个 30 天补货预测任务从早期的买手级稀疏建模，收口到 SKU 级聚合预测，建立了一条完整可复现的 sequence 基线。我们系统比较了 LSTM、BiLSTM、GRU、Attention、weekly、不同特征版本和不同 seed，最终冻结出 `p527_lstm_l3_v5_lite_s2027` 作为 sequence 最佳版本。这个版本已经能够提供可用预测，但在 `4-25`、冷启动、`>25` 和 allocation 上存在明显上限，因此后续才转向 tree 路线。 

## 8. 汇报时建议强调的顺序

1. 为什么要从 `(buyer, sku, date)` 升级到 `(sku, date)`
2. LSTM 阶段做了哪些系统性工作
3. sequence 内部做过哪些架构和特征比较
4. 最终冻结的最佳 LSTM 是什么
5. 为什么它是“合格 baseline”，但不是最终主线
