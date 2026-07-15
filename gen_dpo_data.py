"""
M10 第一步: 构造 DPO 偏好对。
prompt   = 指令 + train 集简历文本
chosen   = 标准答案 JSON
rejected = 原始基座 (零样本) 的真实错误输出——凡与 gold 不一致即为 rejected,
           优先保留"可解析但内容错"的样本 (比"完全不可解析"学习信号更强)。

为什么用真实错误而不是人工构造错误: DPO 压制的应该是模型真的会犯的错
(脑补字段/格式错/漏抽), 人工编的错误分布和模型的错误分布对不上。

运行:  .venv\\Scripts\\python.exe gen_dpo_data.py
产物:  data/resume_dpo.jsonl (入库, 全合成无隐私)
"""
import sys, os, json
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from schema import INSTRUCTION
from evaluate import generate, try_parse_json

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "models", "Qwen2.5-1.5B-Instruct")
TRAIN_FILE = os.path.join(HERE, "data", "resume_train.jsonl")
OUT_FILE = os.path.join(HERE, "data", "resume_dpo.jsonl")
MAX_PAIRS = 200


def main():
    rows = [json.loads(l) for l in open(TRAIN_FILE, encoding="utf-8")]
    print(f"train 集 {len(rows)} 条, 用原始基座零样本推理收集错误输出 ...")

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb, dtype=torch.bfloat16, device_map="cuda",
    )
    base.eval()

    pairs_wrong, pairs_unparse = [], []   # 可解析但内容错 / 不可解析
    n_correct = 0
    for i, r in enumerate(rows):
        gold = r["label_json"]
        raw = generate(base, tok, r["resume_text"])
        pred = try_parse_json(raw)
        if pred == gold:
            n_correct += 1
        else:
            pair = {
                "prompt": INSTRUCTION + r["resume_text"],
                "chosen": json.dumps(gold, ensure_ascii=False),
                "rejected": raw,
                "rejected_type": "unparseable" if pred is None else "parseable_but_wrong",
            }
            (pairs_unparse if pred is None else pairs_wrong).append(pair)
        if (i + 1) % 20 == 0:
            print(f"  进度 {i+1}/{len(rows)} | 内容错 {len(pairs_wrong)} "
                  f"| 不可解析 {len(pairs_unparse)} | 完全正确 {n_correct}")

    # 优先"可解析但内容错", 不够再补"不可解析"
    pairs = pairs_wrong[:MAX_PAIRS]
    if len(pairs) < MAX_PAIRS:
        pairs += pairs_unparse[:MAX_PAIRS - len(pairs)]

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\n完成: 偏好对 {len(pairs)} 条 -> {OUT_FILE}")
    print(f"  可解析但内容错: {len(pairs_wrong)} (采用 {min(len(pairs_wrong), MAX_PAIRS)})")
    print(f"  不可解析:       {len(pairs_unparse)} (采用 {len(pairs) - min(len(pairs_wrong), MAX_PAIRS)})")
    print(f"  基座完全答对:   {n_correct}")


if __name__ == "__main__":
    main()
