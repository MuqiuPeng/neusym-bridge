# NoRelatum 消融实验报告

**日期**：2026-04-04  
**状态**：完成 -- Scenario C（Relatum 贡献极小）  
**核心发现**：Contrastive encoder + Interface Layer 是性能提升的主要来源（337.5 -> 96.1, -71.5%），Relatum 推理层在 Contrastive 配置下几乎不触发（96.1 -> 97.1, +1.0%）。

---

## 实验目的

分解 Contrastive Full System (97.8) 的性能贡献，填入缺失的消融点：

```
Contrastive Pure       337.5  (无 Interface, 无 Relatum)
Contrastive NoRelatum   ???   (有 Interface, 无 Relatum)  <-- 本实验
Contrastive Full        97.8  (有 Interface, 有 Relatum)
```

---

## 实验设计

**NoRelatum Planner**：使用 Interface Layer 的 confidence 直接做硬阈值判断（>0.5 即触发保守策略），不经过 Relatum 的 Noisy-OR 合并、坍缩阈值判断和 Provenance 推导。

5 种配置在相同 100 个任务上对比：

| 配置 | Encoder | Interface | Relatum |
|------|---------|-----------|---------|
| recon_pure | Reconstruction | N | N |
| recon_full | Reconstruction | Y | Y |
| contrastive_pure | Contrastive | N | N |
| contrastive_norelatum | Contrastive | Y | N |
| contrastive_full | Contrastive | Y | Y |

---

## 结果

### 完整消融表

| 配置 | Avg Distance | Interface | Relatum |
|------|-------------|-----------|---------|
| recon_pure | 773.6 | N | N |
| contrastive_pure | 337.5 | N | N |
| **recon_full** | **244.8** | **Y** | **Y** |
| **contrastive_norelatum** | **96.1** | **Y** | **N** |
| **contrastive_full** | **97.1** | **Y** | **Y** |

### 贡献分解（Contrastive 路径）

| 组件 | From | To | Delta | 比例 |
|------|------|----|-------|------|
| **Encoder** | 773.6 (recon_pure) | 337.5 (contr_pure) | -436.1 | **-56.4%** |
| **Interface** | 337.5 (contr_pure) | 96.1 (contr_norelatum) | -241.4 | **-71.5%** |
| **Relatum** | 96.1 (contr_norelatum) | 97.1 (contr_full) | +1.0 | **+1.0%** |
| **Total** | 773.6 (recon_pure) | 97.1 (contr_full) | -676.5 | **-87.4%** |

---

## 分析

### 发现 1：Interface Layer 是主要贡献者

Interface Layer 将 Contrastive Pure 的距离从 337.5 降至 96.1（-71.5%），这是三个组件中贡献最大的。Interface 的作用是将 latent 空间的连续值转换为离散的风险信号，直接指导保守/正常策略切换。

### 发现 2：Relatum 推理层在 Contrastive 配置下不工作

Contrastive NoRelatum (96.1) 和 Contrastive Full (97.1) 几乎无差异。原因：

1. **Interface 的硬阈值已经足够好**：NoRelatum 直接对每个 predicate 做 >0.5 阈值判断，这在 Contrastive latent 上已经产生了准确的风险检测
2. **Relatum 的 structural_risk 规则要求 3 个 predicate 同时满足**（conjunction），这比 NoRelatum 的 "任一 predicate" 条件更严格，导致触发频率更低
3. **Noisy-OR 合并和坍缩阈值 0.6 增加了延迟**，在某些情况下反而错过了最佳介入时机

### 发现 3：Relatum 对 Reconstruction 仍有价值

Reconstruction 路径：recon_pure (773.6) -> recon_full (244.8) = -68.4%。这里 Interface + Relatum 的联合贡献显著。对比 Contrastive 路径中 Interface 单独就能达到 -71.5%，说明 **Contrastive encoder 产生的 latent 使 Interface 更有效，减少了对 Relatum 推理的依赖**。

### 发现 4：Encoder 选择是基础

所有配置中，Contrastive encoder 始终优于 Reconstruction encoder（同一配置下）。Encoder 质量是整个系统性能的地基。

---

## 对论文的影响

### 原始 claim（需修正）

> "The symbolic layer's contribution is amplified by a better latent space"

### 修正后的 claim

> The neuro-symbolic bridge's performance is primarily driven by the encoder (Contrastive, -56%) and the interface layer (-71%), while the Relatum reasoning layer provides negligible additional benefit when paired with a high-quality encoder (+1%). This suggests that the interface layer's direct perception-to-action mapping subsumes Relatum's logical inference when the underlying latent space is sufficiently well-organized. Relatum retains value for reconstruction-based encoders (-68% combined) and for providing interpretable failure explanations, but is not the primary performance driver.

### 架构层面的洞察

```
性能贡献排序：
1. Interface Layer (直接感知 -> 策略切换)   -71.5%  <-- 关键组件
2. Encoder choice (Contrastive vs Recon)     -56.4%  <-- 基础质量
3. Relatum (逻辑推理 + 坍缩)                 +1.0%  <-- 可解释性工具

Relatum 的真正价值不在于性能提升，而在于：
- 提供结构化的失败诊断（explanation rate）
- 对低质量 encoder 的补偿作用
- 安全约束的可审计性
```

---

## 全项目最终状态

```
Phase 1 (Common structure):     PASS  CKA=0.944
Phase 2 (Causal information):   PASS  3/4 interventions
Phase 3 (Collapse mechanism):   PASS  6/6
Phase 4 (End-to-end):           PASS  6/6

A1 (Training objective):
  Predictive SINDy R² = 0.479 (+64% vs Reconstruction)
  Contrastive planning distance = 337.5 (best pure)

Contrastive Full System:        dist = 97.1  (-87.4% vs recon_pure)
NoRelatum Ablation:             Interface = -71.5%, Relatum = +1.0%

Optimal config: Contrastive Encoder + Interface Layer (Relatum optional)
```
