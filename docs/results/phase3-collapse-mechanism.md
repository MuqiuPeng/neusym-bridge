# Phase 3 实验报告：坍缩机制验证

**日期**：2026-04-03  
**状态**：修复后通过（20 个测试全通过，verdict 6/6 聚合检查通过）  
**首次运行**：16/20 通过，4 个失败（Scenario B 撤回逻辑 bug）  
**修复后**：20/20 通过  
**结论**：坍缩机制的核心逻辑经修复后验证通过。初版实现存在撤回传播 bug，暴露了 Provenance 依赖追踪的设计缺陷。

---

## 实验配置

| 参数 | 值 |
|------|-----|
| 实现语言 | Python（镜像 Relatum 概念，**未修改 Relatum Rust 代码**） |
| 规则集 | 2 条规则（heat_concentration, structural_risk） |
| 谓词来源 | Phase 2 干预实验的 3 个因果谓词 |
| 置信度合并 | Noisy-OR: P = 1 - Π(1 - p_i) |
| 坍缩阈值 | 0.85（默认） |
| 撤回阈值 | 0.30（低于此值触发撤回） |

## 测试矩阵

共 20 个 pytest 测试，分布如下：

| 组别 | 测试数 | 首次运行 | 修复后 |
|------|--------|---------|--------|
| Noisy-OR 单元测试 | 6 | 6/6 通过 | 6/6 通过 |
| Scenario A（正常坍缩） | 5 | 5/5 通过 | 5/5 通过 |
| Scenario B（矛盾撤回） | 3 | **0/3 通过** | 3/3 通过 |
| Scenario C（主动查询） | 4 | 4/4 通过 | 4/4 通过 |
| Integration（完整循环） | 2 | **1/2 通过** | 2/2 通过 |
| **总计** | **20** | **16/20** | **20/20** |

Verdict 的 6 个聚合检查项是从 20 个测试中提取的高层判定，不是独立的 6 个测试。

---

## 首次运行失败记录

### 4 个失败的测试

1. **`test_contradiction_triggers_retraction`** — 注入低置信度 `temperature_dominant(node_3)=0.12` 后，`heat_concentration(node_3)` 仍为 COLLAPSED
2. **`test_unrelated_facts_preserved`** — 同上原因导致 node_3 未撤回
3. **`test_full_cycle`** (Integration) — 矛盾阶段未触发撤回
4. **`test_multiple_nodes_independent`** (Integration) — 同上

### 根因

`assert_probabilistic()` 的撤回检查逻辑只检查 `fact_id in self.collapsed_facts`。但观测事实（如 `temperature_dominant(node_3)`）本身不在 `collapsed_facts` 中——它是作为前提被消费的，坍缩的是推导结论（`heat_concentration`）。因此矛盾证据注入时，撤回条件永远不触发。

**本质问题**：初版实现没有区分"直接坍缩的事实"和"作为坍缩前提的观测事实"，导致对观测事实的矛盾更新被忽略。

### 修复

在 `assert_probabilistic()` 中，当 `confidence < retraction_threshold` 时，不仅检查 `fact_id` 本身是否在 `collapsed_facts`，还通过 `_find_dependents(fact_id)` 查找所有依赖该观测的已坍缩结论。如果存在，触发沿 Provenance 链的撤回。

```python
# 修复前：只检查自身
if fact_id in self.collapsed_facts:
    if confidence < self.retraction_threshold:
        self._retract_with_provenance(fact_id)

# 修复后：检查所有依赖链
if confidence < self.retraction_threshold:
    dependents = self._find_dependents(fact_id)
    affected_collapsed = [d for d in dependents if d in self.collapsed_facts]
    if fact_id in self.collapsed_facts:
        affected_collapsed.append(fact_id)
    if affected_collapsed:
        self._retract_with_provenance(fact_id)
```

---

## Scenario A: 正常坍缩

**结果：PASS**

### 坍缩过程
```
t=0: assert temperature_dominant(node_3) = 0.91
t=1: assert temperature_global(node_3)   = 0.87
t=2: assert temperature_spatial(node_3)  = 0.83
     → update_closure fires heat_concentration_rule
     → Noisy-OR(0.91, 0.87, 0.83) = 0.998 > 0.85
     → ✓ heat_concentration(node_3) COLLAPSED (conf=0.971)
     → cascade: structural_risk_rule fires
     → ✓ structural_risk(node_3) COLLAPSED
```

### 验证项
| 检查 | 结果 |
|------|------|
| 三证据触发坍缩 | ✓ |
| 推导级联（structural_risk 自动推出） | ✓ |
| 坍缩置信度 > 0.95 | ✓ (0.971) |
| Provenance 链完整（可追溯到 3 个观测） | ✓ |
| 部分证据（2/3）不触发坍缩 | ✓ |

---

## Scenario B: 证据冲突撤回

**结果：PASS**

### 撤回过程
```
初始状态: structural_risk(node_3) = COLLAPSED
          structural_risk(node_7) = COLLAPSED (独立)

注入: temperature_dominant(node_3) = 0.12 (< 0.30 撤回阈值)
      → 检测到矛盾
      → Provenance 追踪: temperature_dominant → heat_concentration → structural_risk
      → ↩ 撤回: structural_risk(node_3)
      → ↩ 撤回: heat_concentration(node_3)
      → node_7 不受影响
```

### 验证项
| 检查 | 结果 |
|------|------|
| 矛盾证据触发撤回 | ✓ |
| 撤回沿 Provenance 链传播 | ✓ |
| 无关节点（node_7）保留 | ✓ |
| 非依赖观测事实保留（temperature_global） | ✓ |

---

## Scenario C: 主动查询

**结果：PASS**

### 查询过程
```
已知: temperature_dominant(node_5) = 0.89
      temperature_global(node_5) = 0.84
缺失: temperature_spatial(node_5)

→ find_missing_premises() 返回:
  [QueryRequest(predicate='temperature_spatial', args=('node_5',),
                reason='Needed to derive heat_concentration(node_5)')]

响应: temperature_spatial(node_5) = 0.81
→ update_closure fires rule
→ ✓ heat_concentration(node_5) COLLAPSED
```

### 验证项
| 检查 | 结果 |
|------|------|
| 正确检测缺失前提 | ✓ |
| 查询响应后触发坍缩 | ✓ |
| 低置信度（0.40, 0.35, 0.30）不坍缩 | ✓ (Noisy-OR=0.727 < 0.85) |

---

## Integration: 完整循环

**结果：PASS**

```
Step 1 (建立):  3 观测 → collapse → structural_risk = True     ✓
Step 2 (推翻):  矛盾证据 → retract → structural_risk = False   ✓
Step 3 (重建):  新观测(0.78) → re-collapse → structural_risk = True  ✓
```

重建后置信度 0.992（Noisy-OR of 0.78, 0.87, 0.83），略低于初始的 0.998 但仍远超阈值。

---

## Phase 3 Verdict

### Verdict 聚合检查（6 项，从 20 个测试中提取）

| 检查项 | 首次运行 | 修复后 |
|--------|---------|--------|
| Scenario A: 正常坍缩 | PASS | **PASS** |
| Scenario A: Provenance 完整 | PASS | **PASS** |
| Scenario B: 矛盾触发撤回 | **FAIL** | **PASS** |
| Scenario B: 最小化撤回 | **FAIL** | **PASS** |
| Scenario C: 主动查询正确 | PASS | **PASS** |
| Integration: 完整循环无崩溃 | **FAIL** | **PASS** |

**首次运行：3/6 通过。修复撤回传播 bug 后：6/6 通过。**

### 诚实评估

- Scenario A（正常坍缩）和 Scenario C（主动查询）**首次实现即通过**，逻辑正确
- Scenario B（矛盾撤回）**首次实现失败**，暴露了 Provenance 依赖追踪的设计缺陷
- 修复是针对性的（3 行代码变更），不是重写，说明核心架构是对的，但边界条件处理有遗漏
- **Relatum Rust 侧未做任何修改**，Phase 3 完全在 Python 侧验证逻辑

---

## 实现细节

### Noisy-OR 置信度合并
- 适用于独立证据源的概率融合
- 三个 0.91/0.87/0.83 的证据合并为 0.998
- 三个 0.40/0.35/0.30 的证据合并为 0.727（不触发坍缩）
- 性质：永远 ≤ 1.0，单调递增，0 不贡献

### 最小化撤回算法
1. 从被推翻的事实出发，BFS 遍历 Provenance 图
2. 只撤回直接或间接依赖该事实的结论
3. 独立推导链不受影响
4. 撤回后可立即接受新证据并重建

### 主动查询机制
1. 遍历所有规则和已知事实的 grounding
2. 对每个 grounding 检查哪些前提缺失
3. 返回按紧迫性排序的查询请求
4. 外部响应后自动触发增量闭包

---

## Phase 4 输入

```python
phase4_inputs = {
    "relatum_interface": "RelatumInterface v1.0 (Python)",
    "predicates": ["temperature_dominant", "temperature_global", "temperature_spatial"],
    "rules": "rules/heat_rules.pl",
    "collapse_threshold": 0.85,
    "retraction_threshold": 0.30,
    "confidence_combiner": "noisy_or",
    "verified_scenarios": ["normal_collapse", "contradiction_retraction", "active_query"],
}
```
