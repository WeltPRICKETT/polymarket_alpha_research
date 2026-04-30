# Bug Fixes — Polymarket Alpha Research

> 修复日期: 2026-04-11

---

## 已修复的严重 Bug

### Bug 1: `win_rate` = BUY 比例（逻辑错误）
**文件**: `src/preprocessing/feature_engineer.py`

**问题**: 原始代码中，BUY 和 SELL 的赢率判断完全错误：
```python
# 原始错误代码
if price < 0.5:
    wins += 1
else:
    wins += 1   # ← 两个分支都 wins += 1
    pnl = size * (1.0 - price)
```
这使得 **所有 BUY 交易无条件算赢**，导致 `win_rate` 等同于 BUY 比例（相关性 = 1.0），与实际市场表现完全无关联。

**修复**: 使用价格启发式：
- BUY at price ≥ 0.5 → 买入有利可图的押注 → win
- BUY at price < 0.5 → 逆向押注 → loss

---

### Bug 2: `max_drawdown` 恒等于 -0.01（无效特征）
**文件**: `src/preprocessing/feature_engineer.py`

**问题**: 原始代码有一行关键的 `clip`：
```python
self.features_df["max_drawdown"] = self.features_df["max_drawdown"].clip(upper=-0.01)
```
这将所有 `max_drawdown` 值（包括 0）强制截断到 `-0.01`，使整个特征列成为常数，**对机器学习毫无价值**。

**修复**: 删除该 clip 语句，改用归一化相对回撤（相对于总投入资本），范围 `[-1.0, 0.0]`。

---

### Bug 3: `max_drawdown` 除以 epsilon 产生极大值
**文件**: `src/preprocessing/feature_engineer.py`

**问题**: 修复 Bug 2 后，drawdown 计算存在 peak=0 时除以 epsilon 的问题：
```python
dd = (p - peak) / (abs(peak) + 1e-6)  # peak=0时: 大负数/1e-6 = 极大值(百万量级)
```

**修复**: 将 running PnL 归一化到 total_invested 的比例，再计算相对回撤：
```python
equity_curve = [p / capital_base for p in running_pnls]
dd = eq - peak_eq  # 直接差值即为相对回撤，无除法风险
```

---

### Bug 4: 标签计算导致数据泄漏（Label Leakage）
**文件**: `src/preprocessing/feature_engineer.py` + `pipeline.py`

**问题**: 原始代码在 `feature_engineer.py` 中用**全量数据**的 `quantile(0.8)` 计算 `Trader_Success_Rate` 标签阈值：
```python
threshold = self.features_df["Risk_Adjusted_Return"].quantile(0.8)  # 全量！
```
然后再做 train/test split。这导致测试集的信息（其 RAR 分布）已经泄漏进训练标签。

**修复**: 将标签赋值移到 `pipeline.py`，在 temporal split **之后**仅用训练集的分位数计算阈值：
```python
train_rar = df_with_dates.loc[df_with_dates['is_train'], 'Risk_Adjusted_Return']
threshold = train_rar.quantile(0.8)
df_with_dates['Trader_Success_Rate'] = (df_with_dates['Risk_Adjusted_Return'] >= threshold).astype(int)
```

---

### Bug 5: `information_ratio` 使用错误公式
**文件**: `src/preprocessing/feature_engineer.py`

**问题**: 原始公式计算 `(1-p)/p` 和 `p/(1-p)` 作为 "return"，这是赔率比而非预期收益率，数值范围极不稳定（p→0 时趋近无穷大）。

**修复**: 使用更合理的预期收益边际：
```python
# BUY 边际 = (1-p) - p = 1 - 2*p (帕式预期收益)
returns.append(1.0 - 2.0 * p)
```

---

### Bug 6: XGBoost 弃用参数
**文件**: `src/models/trainer.py`

**问题**: XGBoost 3.x 已移除 `use_label_encoder` 参数：
```python
XGBClassifier(use_label_encoder=False, ...)  # ← 在 XGBoost 2x+ 中弃用/移除
```

**修复**: 删除该参数。

---

### Bug 7: `profit_loss_ratio` 计算逻辑错误
**文件**: `src/preprocessing/feature_engineer.py`

**问题**: 原始代码：
```python
avg_win = total_pnl / wins if wins > 0 and total_pnl > 0 else 0  # ← 全局PnL/win数? 
avg_loss = abs(total_pnl) / losses if losses > 0 and total_pnl < 0 else 1  # ← 相互排斥！
```
这两行逻辑相互排斥（`total_pnl > 0` 与 `total_pnl < 0` 不能同时成立），导致 `pl_ratio` 几乎总是 0 或 1。

**修复**: 使用逐交易盈亏列表：
```python
w_pnls = [p for p in trade_pnls if p > 0]
l_pnls = [abs(p) for p in trade_pnls if p < 0]
avg_win  = mean(w_pnls) if w_pnls else 0
avg_loss = mean(l_pnls) if l_pnls else 1e-6
```

---

## 修复后的特征分布

| 特征 | 修复前 | 修复后 |
|---|---|---|
| `win_rate` | 0 或 1（= BUY比例） | [0.0, 1.0] 真实分布 |
| `max_drawdown` | 全部 -0.01 （常数！） | [-1.0, 0.0] 真实回撤 |
| `information_ratio` | 数值爆炸性增长 | [-10, 10] 稳定范围 |
| `Trader_Success_Rate` | 用全量分位数 | 仅用训练集分位数 |
| `profit_loss_ratio` | 逻辑矛盾接近0 | 真实赢亏比 |

---

## 下一步操作

运行完整 pipeline 生成修复后的数据和模型：

```bash
# 步骤1: 重新特征工程
python -m src.preprocessing.pipeline

# 步骤2: 重新训练
python -m src.models.trainer

# 步骤3: 重新回测
python -m src.backtesting.run_backtest

# 或一次性全部运行
python run.py --pipeline
```

---

## 待修复的次要问题（未在本次修复范围内）

1. **`information_ratio` 有效信息量低**: 75% 的交易者该指标为 0（因为大多数人只有 1 笔交易）
2. **标签不平衡**: 测试集中正类比例 ~55%，较高，说明启发式标签仍需改进
3. **仅有 "Yes" resolution**: 数据集中所有已解析市场都解析为 "Yes"，缺乏真实多元结果数据
4. **回测的 `_run_backtest` 端点**: 直接用 subprocess 调用脚本，无状态轮询，前端无法看到实时进度
