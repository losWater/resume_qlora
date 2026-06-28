# 简历信息抽取（QLoRA 微调）+ 岗位匹配

用 **QLoRA** 对 **Qwen2.5-1.5B-Instruct** 做 4-bit 量化微调，把**中文简历纯文本**解析成**规范 JSON**（姓名 / 学历 / 技能 / 工作年限 / 各段经历）；再用一个**不需要训练的规则模块**，拿这份 JSON 与岗位 JD 算出可解释的匹配分。

> 全流程在单张 **8GB 消费级显卡（RTX 5060 Laptop）** 上完成训练与部署，使用**全合成数据**（无真实个人信息，合规）。

---

## ✨ 核心亮点

- **该用大模型的用大模型，不该用的用规则**：抽取（输入输出范式固定、可量化）用 QLoRA 微调；匹配打分（缺标注、难客观评估）用规则——可解释、稳定。
- **数据蒸馏**：用强模型（DeepSeek）批量生成「简历文本 + 标准 JSON」配对数据，微调小模型去学。
- **8GB 单卡可训**：4-bit 量化 + LoRA + 梯度检查点 + paged 优化器，显存峰值 ~7GB。
- **量化验证有效**：同一测试集上微调前后对比，列表字段平均 F1 **68.1% → 79.5%**，JSON 可解析率 **96.6% → 100%**。

---

## 📊 效果对比（验证集 29 条，微调前 = 原始 Qwen 直接 prompt）

| 指标 | 微调前 | 微调后 | 提升 |
|---|---|---|---|
| JSON 可解析率 | 96.6% | **100.0%** | +3.4pp |
| 姓名准确率 | 93.1% | **96.6%** | +3.4pp |
| 工作年限准确率 | 79.3% | **89.7%** | +10.3pp |
| 技能 F1 | 76.8% | **85.6%** | +8.8pp |
| 教育经历 F1 | 64.9% | **90.0%** | +25.1pp |
| 工作经历 F1 | 62.5% | 63.0% | +0.5pp |
| **列表字段平均 F1** | 68.1% | **79.5%** | **+11.5pp** |

> 结果文件：`outputs/eval_result.json`。教育经历提升最大；工作经历提升小（公司/职位表述多样、描述噪声多，是下一轮迭代方向）。

---

## 🏗 架构

```
简历文本 ──▶ [QLoRA 微调的 Qwen2.5-1.5B] ──▶ 简历 JSON ──┐
                                                          ├──▶ [规则匹配模块] ──▶ 匹配分 + 逐项理由
岗位 JD  ──▶ [原始 Qwen 基座 (关闭 adapter)] ──▶ JD JSON ──┘
```

- **抽取（✅ 微调）**：4-bit 基座 + LoRA adapter，只对答案 JSON 计算 loss（completion-only）。
- **JD 解析（⚙️ 基座零样本）**：`disable_adapter()` 临时关掉 LoRA，一个模型双角色，省显存。
- **匹配打分（⚙️ 规则）**：技能重合率 60% + 年限达标 20% + 学历达标 20% → 0~100 分。

---

## 🧰 技术栈

- Python 3.12 / PyTorch 2.11 (cu128, 支持 Blackwell 50 系)
- transformers · peft · trl · bitsandbytes · accelerate · datasets
- 基座：Qwen2.5-1.5B-Instruct（modelscope 下载）
- 造数据：DeepSeek API（OpenAI 兼容）

---

## 📁 目录结构

```
resume_qlora/
├── check_env.py            # M1 环境验收（torch/CUDA/4-bit 冒烟测试）
├── download_model.py       # 下载基座模型
├── make_ner_data.py        # M2 合成 NER 数据（跑通范式用）
├── train_ner_qlora.py      # M2 NER QLoRA 训练（范式验证）
├── infer_ner.py            # M2 NER 推理
├── gen_resume_data.py      # M3 DeepSeek 蒸馏生成简历数据
├── clean_resume_data.py    # M3 数据清洗（学历规范化）
├── schema.py               # 简历 JSON schema + 指令 + 校验（三处共享）
├── train_resume_qlora.py   # M4 简历抽取 QLoRA 训练 ★
├── evaluate.py             # M5 微调前后量化对比 ★
├── matcher.py              # M6 岗位匹配模块（规则）★
├── cli_demo.py             # M7 命令行 demo ★
├── requirements.txt
├── .env                    # API Key（已 gitignore，不提交）
├── data/                   # 合成训练/验证数据（已入库，全合成无隐私）
├── models/                 # 基座模型（gitignore，用 download_model.py 下载）
└── outputs/                # 最终 LoRA adapter + 评估结果（已入库；体积大的 checkpoint 不入库）
```

---

## 🚀 复现步骤

> 仓库已附带训练好的 LoRA adapter（`outputs/{ner,resume}_adapter/`，各约 4MB）。只想体验效果可在装好依赖、下载基座模型后**直接跳到第 6/7 步**（评估 / Demo），无需重新训练。

```bash
# 0. 建虚拟环境（用纯净的母 Python）
python -m venv .venv
.venv\Scripts\activate

# 1. 装依赖（torch 必须单独从 cu128 源装，否则认不出 50 系卡）
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

# 7. Demo
python cli_demo.py                                  # 交互模式
python cli_demo.py --resume r.txt --jd jd.txt       # 文件模式
```

---

## 🎬 Demo 示例

输入简历 + JD，输出结构化 JSON 与匹配理由：

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

---

## ⚠️ 已知局限（诚实记录）

- **小模型偶有脑补**：1.5B 模型偶尔抽错日期、或为缺失字段补全内容（如把"两段经历"扩成具体公司名）。
- **同分布评估**：train/val 同源于 DeepSeek 生成，分数偏乐观；更严格应在真实简历或不同来源上测泛化。
- **JD 解析用基座零样本**，稳定性弱于简历抽取（JD 无专门微调）；学历"X 及以上"偶有误判。
- **工作经历字段**提升有限，是后续重点优化方向（增样本 / 细化标注 / 约束解码）。

---

## 📝 简历描述句

> 基于 QLoRA 对 Qwen2.5-1.5B 进行 4-bit 量化微调，实现中文简历结构化解析（非结构化文本 → JSON），列表字段平均 F1 从 68% 提升至 80%、JSON 可解析率提升至 100%；并构建可解释的岗位匹配评分模块。全流程在单张 8GB 消费级显卡上完成训练与部署，训练数据由强模型蒸馏合成。
