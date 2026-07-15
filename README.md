# 简历信息抽取（QLoRA 微调）+ 岗位匹配

用 QLoRA 对 Qwen2.5-1.5B-Instruct 做 4-bit 量化微调，把中文简历纯文本解析成规范 JSON（姓名 / 学历 / 技能 / 工作年限 / 各段经历）；再用一个不需要训练的规则模块，拿这份 JSON 和岗位 JD 算出一个可解释的匹配分。

全流程在单张 8GB 消费级显卡（RTX 5060 Laptop）上完成训练和部署，训练数据全部是合成的，不涉及真实个人信息。

## 思路

- 抽取这一步输入输出范式固定、结果可量化，适合用模型学，所以用 QLoRA 微调；匹配打分缺标注、也很难客观评估，硬上模型反而不稳，就用规则做，好处是可解释。
- 训练数据用 DeepSeek 批量生成「简历文本 + 标准 JSON」配对，再让小模型去学（数据蒸馏）。
- 8GB 单卡能训，靠的是 4-bit 量化 + LoRA + 梯度检查点 + paged 优化器，显存峰值大概 7GB。
- 效果在同一测试集上做了微调前后对比，列表字段平均 F1 从 68.1% 到 79.5%，JSON 可解析率从 96.6% 到 100%。

## 效果对比

验证集 29 条，"微调前"指原始 Qwen 直接 prompt。

| 指标 | 微调前 | 微调后 | 提升 |
|---|---|---|---|
| JSON 可解析率 | 96.6% | 100.0% | +3.4pp |
| 姓名准确率 | 93.1% | 96.6% | +3.4pp |
| 工作年限准确率 | 79.3% | 89.7% | +10.3pp |
| 技能 F1 | 76.8% | 85.6% | +8.8pp |
| 教育经历 F1 | 64.9% | 90.0% | +25.1pp |
| 工作经历 F1 | 62.5% | 63.0% | +0.5pp |
| 列表字段平均 F1 | 68.1% | 79.5% | +11.5pp |

完整结果在 `outputs/eval_result.json`。教育经历提升最大；工作经历几乎没动——公司/职位的表述太杂、噪声多，是下一步要改的地方。

## 超参数消融

以当前配置（r=8、QKVO、lr=2e-4）为中心做控制变量消融：每组只改一个维度，固定种子 42、同一份 train/val、同样 3 个 epoch。共 6 组，完整数据在 `outputs/ablation_results.json`（`run_ablation.py` 可复现）。

| 配置 | 改动维度 | 可训练参数 | 训练时长 | 可解析率 | 技能 F1 | 教育 F1 | 经历 F1 | 平均 F1 |
|---|---|---|---|---|---|---|---|---|
| baseline（r=8, QKVO, 2e-4） | — | 2.2M | 4.7min | 100% | 84.7% | 92.5% | 58.7% | 78.6% |
| r=4（alpha=8） | 秩 | 1.1M | 4.5min | 100% | 85.2% | 87.5% | 60.9% | 77.9% |
| r=16（alpha=32） | 秩 | 4.4M | 4.7min | 100% | 85.7% | 92.5% | 63.0% | 80.4% |
| QKVO+MLP | 挂载层 | 9.2M | 5.8min | 100% | 85.7% | 92.5% | 70.3% | 82.9% |
| lr=1e-4 | 学习率 | 2.2M | 4.7min | 96.6% | 83.6% | 79.5% | 63.6% | 75.6% |
| lr=5e-4 | 学习率 | 2.2M | 4.6min | 100% | 84.2% | 92.5% | 60.9% | 79.2% |

几个结论：

- **秩 r**：4→8→16 平均 F1 单调上升（77.9%→78.6%→80.4%），说明 r=8 还没到这个任务的容量拐点，但每翻倍一次收益只有 1~2 个点，边际递减。
- **挂载层是收益最大的维度**：加上 MLP 层（gate/up/down_proj）平均 F1 +4.2pp，而且提升几乎全部来自最短板的工作经历字段（58.7%→70.3%，+11.6pp）。代价是可训练参数从 2.2M 到 9.2M、训练时间 +23%，显存几乎没变——在这个任务上明确值得。
- **学习率**：1e-4 最差（75.6%，且可解析率掉到 96.6%——3 个 epoch 内没收敛到位，连格式服从都不稳）；2e-4 和 5e-4 接近（78.6% vs 79.2%），说明 2e-4~5e-4 都在稳定区间，1e-4 对 LoRA 偏小。
- **噪声提示**：验证集只有 29 条、单种子，1 个点以内的差异不足为凭（固定种子重训的 baseline 是 78.6%，和最初跑的 79.5% 差约 1 个点，就是这个量级的波动）；上面 MLP 的 +4.2pp 和 lr=1e-4 的 -3pp 超出了噪声范围，结论可信。
- 表中"峰值显存"按 torch 实际分配的张量显存统计约 2.4~2.5GB；任务管理器/nvidia-smi 看到的 ~7GB 还包含 CUDA 上下文和缓存池，两个口径不同。

## 分布外（OOD）评估

train/val 同源（同一个生成器 + 同一套模板），分数偏乐观——乐观多少，用一个分布外验证集来量化：40 条，换生成模型（deepseek-chat → deepseek-reasoner），并要求模拟真实简历的"脏"格式（口语化流水账、字段缺失、时间线断档、公司缩写、技能夹在经历描述里、错别字、时间格式混乱）。数据在 `data/resume_ood.jsonl`（全合成，无隐私），完整结果在 `outputs/ood_eval_result.json`。

同一个微调后模型在两个集上的对比：

| 指标 | 同源 val（29 条） | OOD（40 条） | 差值 |
|---|---|---|---|
| JSON 可解析率 | 100% | 100% | 0 |
| 姓名准确率 | 96.6% | 97.5% | +0.9pp |
| 工作年限准确率 | 89.7% | 47.5% | **-42.2pp** |
| 技能 F1 | 85.6% | 67.4% | -18.2pp |
| 教育经历 F1 | 90.0% | 78.1% | -11.9pp |
| 工作经历 F1 | 63.0% | 77.7% | +14.6pp |
| 列表字段平均 F1 | 79.5% | 74.4% | **-5.2pp** |

- 平均 F1 掉 5.2 个点——这就是"同源评估偏乐观"的量级。格式服从（JSON 可解析率）没有退化，说明微调学到的输出范式对分布偏移是稳的，退化的是内容抽取。
- 最大崩塌是工作年限（-42pp）：OOD 文本的时间线断档、"至今"、"干了三年"式模糊表述让年限推算大面积失效。技能掉 18 个点主要因为技能藏在经历描述句里。
- 工作经历 F1 反而更高，是标注口径差异：OOD 集要求公司名照抄文本写法（含缩写），归一化难度反而低——横向对比要按字段解释，不能只看均值。
- 注意 OOD 集仍是合成数据，只是分布外；真实简历上的表现大概率还要更低，脱敏真实简历测试集仍是改进项。

## DPO 偏好对齐

SFT 只教模型"什么是对的"，没教"什么是错的"。在 SFT adapter 之上补了一轮 DPO：

- **偏好对**（`gen_dpo_data.py`）：chosen=标准答案；rejected=原始基座零样本的**真实错误输出**（270 条 train 里基座只有 4 条全对，取 200 条"可解析但内容错"的），存 `data/resume_dpo.jsonl`。
- **训练**（`train_dpo.py`）：trl `DPOTrainer`，beta=0.1、lr=1e-5、2 epoch，约 5.5 分钟。8GB 显存的关键：`ref_model=None`（对 PEFT 模型用 `disable_adapter()` 当 reference，一份权重双角色）+ `precompute_ref_log_probs=True` + `max_length` 按实测最长样本收紧到 800。不这么做的话，双份 logits（词表 15 万）会把显存顶爆、静默溢出到共享内存，每步慢两个数量级（实测 390s/步 → 优化后 7s/步）。
- **结果**（`eval_dpo.py`，`outputs/dpo_eval_result.json`）：**收益有限**。同源 val 平均 F1 79.5%→80.4%（+0.9pp，在 29 条验证集的噪声边缘）；OOD 上 74.4%→74.5%（持平）；最想改善的工作经历字段没有变化。
- **原因分析**：rejected 来自基座，但 DPO 施加在 SFT 之上——SFT 已经修掉了基座的那些错误模式（markdown 包裹、null 值、字段错位），偏好对教的是模型已经不犯的错（训练时 reward accuracy 轻松到 100% 也是这个信号）。改进方向：用 SFT 模型自己的 on-policy 错误（如经历字段的脑补样本）构造偏好对。

负结果照实记录：DPO 全流程验证跑通，在这个任务上当前构造方式下收益有限。

## 推理性能基准

29 条 val 真实 prompt、贪心解码、前 2 条热身不计入（`bench_infer.py`，结果在 `outputs/bench_infer_result.json`）：

| 指标 | 4bit + adapter | merged bf16 |
|---|---|---|
| 模型显存 | **1.08 GB** | 2.88 GB |
| 首 token 延迟（均值 / P95） | 102 / 118 ms | **61 / 80 ms** |
| 端到端延迟（均值 / P95） | 12.7 / 17.3 s | **6.0 / 8.5 s** |
| 解码吞吐 | 12.2 tok/s | **25.8 tok/s** |

结论：合并后 bf16 吞吐是 4-bit 的 2.1 倍（4-bit 每层反量化的开销在解码阶段占大头），代价是显存 2.7 倍。训练/开发用 4-bit 省显存，延迟敏感的部署合并后跑 bf16。

## 错误分析驱动的数据修复（工作经历字段专项）

微调后工作经历 F1 长期停在 63%，逐条分析 val 集 12 条错例（`analyze_exp_errors.py`，错例在 `outputs/exp_error_cases.json`）后发现，**相当一部分"错"是标注噪声**：生成训练数据时 DeepSeek 会自行补全公司后缀（原文"恒达实业"标成"恒达实业有限公司"）、编造原文没有的公司名和任职时间——schema 校验只查格式，查不出这种内容不忠实。

修复（`relabel_exp.py`）：按严格"忠实原文"规则（公司名照抄原文写法、原文没有的信息一律"未提及"、晋升取最终职位）重标 experience 字段，train 改动 210/270 条、val 改动 21/29 条；train/val 切分保持不变。重训得 `outputs/resume_adapter_v2`。

**效应分离**（`eval_exp_fix.py`，结果在 `outputs/exp_fix_result.json`）——把提升拆成两部分：

| 阶段 | 经历 F1 | 平均 F1 |
|---|---|---|
| ① 旧模型 × 旧标注 | 63.0% | 79.5% |
| ② 旧模型 × 新标注（评估修正 +2.2pp） | 65.2% | 80.3% |
| ③ 新模型 × 新标注（真实提升 +10.3pp） | **75.6%** | **83.3%** |

关键交叉验证：**OOD 集标注未动**，新 adapter 在 OOD 上经历 F1 77.6%→83.1%（+5.5pp）、平均 F1 74.4%→76.9%——证明是真泛化提升，不是评估口径变化。整个过程没有调任何超参，纯数据侧修复。

小退步照实记：OOD 工作年限准确率 47.5%→37.5%（约 4 条样本），可能与 duration 统一"未提及"后模型对年限的推断变保守有关。

## 架构

```
简历文本 ──▶ [QLoRA 微调的 Qwen2.5-1.5B] ──▶ 简历 JSON ──┐
                                                          ├──▶ [规则匹配模块] ──▶ 匹配分 + 逐项理由
岗位 JD  ──▶ [原始 Qwen 基座 (关闭 adapter)] ──▶ JD JSON ──┘
```

- 抽取：4-bit 基座 + LoRA adapter，只对答案 JSON 算 loss（completion-only）。
- JD 解析：用 `disable_adapter()` 临时关掉 LoRA 跑零样本，一个模型干两件事，省显存。
- 匹配打分：技能重合率 60% + 年限达标 20% + 学历达标 20%，凑成 0~100 分。

## 技术栈

- Python 3.12 / PyTorch 2.11（cu128，支持 50 系显卡）
- transformers、peft、trl、bitsandbytes、accelerate、datasets
- 基座模型 Qwen2.5-1.5B-Instruct（从 modelscope 下载）
- 造数据用 DeepSeek API（OpenAI 兼容接口）

## 目录结构

```
resume_qlora/
├── check_env.py            # 环境验收（torch/CUDA/4-bit 冒烟测试）
├── download_model.py       # 下载基座模型
├── make_ner_data.py        # 合成 NER 数据（先跑通范式）
├── train_ner_qlora.py      # NER QLoRA 训练（范式验证）
├── infer_ner.py            # NER 推理
├── gen_resume_data.py      # DeepSeek 蒸馏生成简历数据
├── clean_resume_data.py    # 数据清洗（学历规范化）
├── schema.py               # 简历 JSON schema + 指令 + 校验（三处共享）
├── train_resume_qlora.py   # 简历抽取 QLoRA 训练
├── evaluate.py             # 微调前后量化对比
├── run_ablation.py         # 超参数消融（r / 挂载层 / 学习率，6 组控制变量）
├── gen_ood_data.py         # 生成分布外验证集（换生成模型 + 脏格式模板）
├── eval_ood.py             # 同源 val vs OOD 对比评估
├── gen_dpo_data.py         # 构造 DPO 偏好对（基座真实错误输出做 rejected）
├── train_dpo.py            # DPO 训练（SFT adapter 之上继续对齐）
├── eval_dpo.py             # SFT vs SFT+DPO 双集对比
├── bench_infer.py          # 推理性能基准（4bit+adapter vs merged bf16）
├── analyze_exp_errors.py   # 工作经历字段错误分析（导出错例）
├── relabel_exp.py          # 修复 experience 标注噪声（忠实原文重标）
├── eval_exp_fix.py         # 效应分离评估（评估修正 vs 真实提升）
├── matcher.py              # 岗位匹配模块（规则）
├── cli_demo.py             # 命令行 demo
├── requirements.txt
├── .env                    # API Key（gitignore，不提交）
├── data/                   # 合成训练/验证数据（已入库，全合成无隐私）
├── models/                 # 基座模型（gitignore，用 download_model.py 下载）
└── outputs/                # 最终 LoRA adapter + 评估结果（已入库；体积大的 checkpoint 不入库）
```

## 复现步骤

仓库里带了训练好的 LoRA adapter（`outputs/{ner,resume}_adapter/`，各约 4MB）。只想看效果的话，装好依赖、下载基座模型后直接跳到第 6/7 步（评估 / Demo）就行，不用重新训练。

```bash
# 0. 建虚拟环境
python -m venv .venv
.venv\Scripts\activate

# 1. 装依赖（torch 要单独从 cu128 源装，否则认不出 50 系卡）
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# 2. 验收环境
python check_env.py

# 3. 下载基座模型
python download_model.py

# 4. 造数据（需在 .env 填 DEEPSEEK_API_KEY）
python gen_resume_data.py --n 300
python clean_resume_data.py

# 5. 微调
python train_resume_qlora.py

# 6. 评估（微调前 vs 微调后）
python evaluate.py

# 6.5 可选: 超参数消融（6 组训练+评估, 约 40 分钟, 支持断点续跑）
python run_ablation.py

# 6.6 可选: OOD 评估（生成分布外集需要 API Key; 仓库已带 data/resume_ood.jsonl, 可直接评估）
python gen_ood_data.py --n 40
python eval_ood.py

# 6.7 可选: DPO 偏好对齐（仓库已带 data/resume_dpo.jsonl 和 outputs/dpo_adapter）
python gen_dpo_data.py
python train_dpo.py
python eval_dpo.py

# 7. Demo
python cli_demo.py                                  # 交互模式
python cli_demo.py --resume r.txt --jd jd.txt       # 文件模式
```

## Demo 示例

输入简历 + JD，输出结构化 JSON 和匹配理由：

```
===== 结构化简历 JSON =====
{ "name": "王强", "education": [{"school":"江城大学","degree":"硕士",...}],
  "skills": ["Python","Java","MySQL","Redis","分布式系统"], "work_years": 5, ... }

===== 匹配结果 =====
匹配分: 85.0 / 100
分项: 技能 75.0 | 年限 100.0 | 学历 100.0
理由:
  - 技能: 命中 3/4（命中: Python, MySQL, Redis）（缺失: 消息队列）
  - 年限: 要求 3 年, 简历 5 年 -> 达标
  - 学历: 要求 本科, 简历最高 硕士 -> 达标
```

## 已知局限

- 1.5B 模型偶尔会脑补：抽错日期，或给缺失字段补内容（比如把"两段经历"扩成具体公司名）。
- train/val 都来自 DeepSeek 生成，同分布，分数偏乐观。乐观程度已定量化：分布外集上平均 F1 掉 5.2pp、年限准确率掉 42pp（见"分布外评估"一节）；但 OOD 集仍是合成数据，真实简历上还会更低。
- JD 解析是基座零样本，没专门微调，稳定性不如简历抽取；学历"X 及以上"偶尔判错。
- 工作经历字段曾长期停在 63%，错误分析后定位为标注噪声 + 数据修复解决（现 75.6%，见"错误分析驱动的数据修复"一节）；`outputs/resume_adapter_v2` 是当前效果最好的 adapter。
- 重标后 OOD 的工作年限准确率有小幅退步（-10pp，约 4 条），duration 标注口径变化的副作用，待观察。
