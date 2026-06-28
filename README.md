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
- train/val 都来自 DeepSeek 生成，同分布，分数偏乐观；要更严格得拿真实简历或别的来源测泛化。
- JD 解析是基座零样本，没专门微调，稳定性不如简历抽取；学历"X 及以上"偶尔判错。
- 工作经历这个字段提升有限，是后面要重点优化的（加样本 / 细化标注 / 约束解码）。
