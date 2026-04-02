# Phase 3 实验报告：坍缩机制验证

**日期**：2026-04-03  
**状态**：PASS（6/6 项通过）  
**结论**：概率事实经推导坍缩为确定性知识的机制完全验证，包括正常坍缩、矛盾撤回和主动查询。

---

## 实验配置

| 参数 | 值 |
|------|-----|
| 实现语言 | Python（镜像 Relatum 概念） |
| 规则集 | 2 条规则（heat_concentration, structural_risk） |
| 谓词来源 | Phase 2 干预实验的 3 个因果谓词 |
| 置信度合并 | Noisy-OR: P = 1 - Π(1 - p_i) |
| 坍缩阈值 | 0.85（默认） |
| 撤回阈值 | 0.30（低于此值触发撤回） |

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

| 检查项 | 结果 |
|--------|------|
| Scenario A: 正常坍缩 | **PASS** |
| Scenario A: Provenance 完整 | **PASS** |
| Scenario B: 矛盾触发撤回 | **PASS** |
| Scenario B: 最小化撤回 | **PASS** |
| Scenario C: 主动查询正确 | **PASS** |
| Integration: 完整循环无崩溃 | **PASS** |

**6/6 全部通过 → 坍缩机制验证通过**

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
