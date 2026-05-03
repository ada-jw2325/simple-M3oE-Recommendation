# 轻量化重训练评估汇总

- 说明: 本次结果来自 CPU + 小样本快速重训练，验证流程可复现。

## Combined 集排名（按 AVG_AUC）

| Rank | Model | AVG_AUC | CTR_AUC | CVR_AUC_POST_CLICK |
|---|---|---:|---:|---:|
| 1 | M3OE | 0.5853 | 0.5970 | 0.5736 |
| 2 | MMOE | 0.5550 | 0.5639 | 0.5461 |
| 3 | ESMM | 0.5543 | 0.5654 | 0.5432 |
| 4 | MTL | 0.5266 | 0.5286 | 0.5245 |
| 5 | PLE | 0.5124 | 0.5101 | 0.5147 |

## 分模型分数据集指标

### ESMM

| Split | CTR_AUC | CVR_AUC_POST_CLICK | AVG_AUC |
|---|---:|---:|---:|
| Online | 0.6210 | 0.5471 | 0.5840 |
| Random | 0.5517 | 0.5003 | 0.5260 |
| Combined | 0.5654 | 0.5432 | 0.5543 |

### MTL

| Split | CTR_AUC | CVR_AUC_POST_CLICK | AVG_AUC |
|---|---:|---:|---:|
| Online | 0.6252 | 0.5560 | 0.5906 |
| Random | 0.5365 | 0.5158 | 0.5262 |
| Combined | 0.5286 | 0.5245 | 0.5266 |

### MMOE

| Split | CTR_AUC | CVR_AUC_POST_CLICK | AVG_AUC |
|---|---:|---:|---:|
| Online | 0.6307 | 0.5320 | 0.5813 |
| Random | 0.5471 | 0.5075 | 0.5273 |
| Combined | 0.5639 | 0.5461 | 0.5550 |

### PLE

| Split | CTR_AUC | CVR_AUC_POST_CLICK | AVG_AUC |
|---|---:|---:|---:|
| Online | 0.6247 | 0.5388 | 0.5818 |
| Random | 0.5385 | 0.5103 | 0.5244 |
| Combined | 0.5101 | 0.5147 | 0.5124 |

### M3OE

| Split | CTR_AUC | CVR_AUC_POST_CLICK | AVG_AUC |
|---|---:|---:|---:|
| Online | 0.6606 | 0.5819 | 0.6212 |
| Random | 0.5520 | 0.4720 | 0.5120 |
| Combined | 0.5970 | 0.5736 | 0.5853 |

