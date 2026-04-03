# A1 实验报告：训练目标对照实验

**日期**：2026-04-03  
**状态**：完成  
**核心发现**：Predictive 目标的 SINDy R² 比 Reconstruction 提升 64%，验证了"重建目标破坏动力学可恢复性"的假设。Contrastive 目标的规划距离最优但 SINDy 最差，揭示了动力学可恢复性与控制性能的分离。

---

## 实验配置

| 参数 | 值 |
|------|-----|
| 物理系统 | 20 段 Cosserat 杆触手 |
| 数据集 | 1000 轨迹 × 200 步 = 200,000 转换对 |
| Backbone | TentacleEncoder(140→256→128→64) + TentaclePredictor(144→256→256→64) |
| Latent 维度 | 64 |
| 训练 | 50 epochs, batch=256, lr=1e-3, cosine annealing, grad clip=1.0 |
| 随机种子 | 42（三个 variant 完全相同） |
| 硬件 | NVIDIA RTX 4060 (8 GB) |

### 唯一区别：训练目标

| Variant | 损失函数 | 防坍缩机制 |
|---------|---------|-----------|
| **Reconstruction** | MSE(z_pred, z_t1) + MSE(s_recon, s_t) | Decoder 重建 |
| **Predictive** | MSE(z_pred, z_t1) + 0.1 × VICReg(z_t) | VICReg 正则 |
| **Contrastive** | MSE(z_pred, z_t1) + InfoNCE(z_t, z_t1, z_neg) | 对比学习 |

---

## 训练收敛

### Prediction Loss（val_pred，公平对比指标）

| Epoch | Reconstruction | Predictive | Contrastive |
|-------|---------------|------------|-------------|
| 0 | 0.0178 | 0.0002 | 0.0005 |
| 10 | 0.0083 | 0.0013 | 0.0001 |
| 25 | 0.0066 | 0.0009 | 0.0000 |
| 49 | 0.0066 | 0.0002 | 0.0000 |

关键观察：Predictive 和 Contrastive 的 prediction loss 比 Reconstruction 低 1-2 个数量级。这说明 **reconstruction 目标迫使 encoder 在"保留状态信息"和"组织可预测的 latent"之间做权衡**，而 predictive/contrastive 目标可以全力优化 latent 的时序结构。

---

## 核心结果

| 指标 | Reconstruction | Predictive | Contrastive | 最优 |
|------|---------------|------------|-------------|------|
| **SINDy R²** ↑ | 0.293 | **0.479** | 0.256 | Predictive |
| **Effective Rank** | 22.2 | 18.2 | 27.3 | — |
| **Planning Dist** ↓ | 773.7 | 907.5 | **337.5** | Contrastive |
| Curvature Probe R² | **0.599** | 0.595 | 0.501 | Reconstruction |
| Velocity Probe R² | **0.841** | 0.805 | 0.730 | Reconstruction |
| Tip Position Probe R² | 0.343 | 0.335 | **0.351** | Contrastive |

---

## 跨 Variant CKA 矩阵

```
                     Reconstruction  Predictive  Contrastive
Reconstruction            1.000       0.001       0.001
Predictive                0.001       1.000       0.001
Contrastive               0.001       0.001       1.000
```

**三种训练目标学到了完全不同的 latent 结构**（CKA ≈ 0），这意味着训练目标是决定 latent 组织方式的主要因素，而非数据本身。

---

## 分析

### 发现 1：Predictive 目标显著提升动力学可恢复性

SINDy R² 从 Reconstruction 的 0.293 提升到 Predictive 的 0.479（+64%），且 effective rank 从 22.2 降到 18.2。这支持核心假设：

> **Reconstruction 目标迫使 latent 保留高维度的空间信息（高 effective rank），导致动力学规律分散在多个维度中，SINDy 难以稀疏拟合。Predictive 目标允许 encoder 压缩无关信息，latent 演化更集中、更规则。**

### 发现 2：动力学可恢复性 ≠ 控制性能

Contrastive 的 SINDy R² 最低（0.256），但 Planning Distance 最优（337.5 vs 773.7）。这揭示了一个重要的分离：

- **SINDy R²** 衡量的是"latent 演化能否被简单方程描述"
- **Planning Distance** 衡量的是"latent 能否指导有效控制"

Contrastive 学习产生的 latent（高 effective rank=27.3）虽然对 SINDy 不友好，但其时序邻域结构使得基于 latent 梯度的规划器能更好地生成有效动作。

### 发现 3：Reconstruction 是物理探测的最优目标

Reconstruction 在 curvature R²（0.599）和 velocity R²（0.841）上均为最高。这符合预期：decoder 必须从 latent 重建完整状态，因此物理量必然被编码。但这种"什么都保留"的策略代价是动力学结构的退化。

### 发现 4：无公共结构（CKA ≈ 0）

与 Phase 1（相同目标不同种子 CKA=0.944）形成鲜明对比。**训练目标的选择比随机初始化更强烈地决定了 latent 的组织方式。**

---

## SINDy 方程（Predictive variant, R²=0.479）

```
x0' =  0.644 x0 + 1.722 x1 - 1.148 x2 - 1.164 x3 + nonlinear terms
x1' = -5.522 x1 + 48.850 x1² - 16.918 x0·x1 + ...
x2' = -2.169 x1 - 0.891 x2 + 29.909 x1² + ...
x3' = -0.224 x3 + ...
```

与 Reconstruction variant 相比，Predictive 的 SINDy 方程呈现更清晰的线性衰减项（如 x1' 中的 -5.522 x1），暗示 latent 动力学更接近线性系统加非线性修正的形式。

---

## 论文主张

基于 A1 实验结果，论文 claim 可升级为：

> Training objective is a key determinant of dynamical recoverability in learned latent spaces: reconstruction objectives induce high-rank mixed representations that preserve spatial information but degrade dynamical structure (SINDy R² = 0.29), while predictive objectives yield lower-rank, more dynamically coherent representations (SINDy R² = 0.48, +64%). However, dynamical recoverability and control performance are not equivalent — temporal contrastive objectives produce the best control performance despite the lowest SINDy fit.

---

## 局限性

1. **单一物理系统**：仅在触手域验证，需要在热传导等其他系统上重复
2. **SINDy 候选库限制**：仅使用 2 阶多项式，更复杂的动力学可能需要更丰富的库
3. **规划器未优化**：Planning distance 受规划器质量影响，更好的规划算法可能改变排序
4. **单次种子**：seed=42 的单次实验，未做多种子统计

---

## 后续方向

1. **A1+**：在 Predictive latent 上接 Relatum（取代 Reconstruction），看 Phase 4 全系统是否进一步提升
2. **混合目标**：λ_recon 从 0→1 连续扫描，绘制 SINDy R² vs Probe R² 的 Pareto 前沿
3. **多种子统计**：3 个种子 × 3 个 variant = 9 次训练，确认结论稳健性
