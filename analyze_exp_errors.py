"""
P1: 工作经历字段专项 —— 第一步: 错误分析。
用微调后模型 (SFT adapter) 在 val 集推理, 把 experience 字段有错的样本导出:
每条含 简历原文 / gold / pred / 差异明细 (漏抽 miss、多抽 extra、字段级对不上),
供人工归类错误模式 (公司名表述不一致 / 职位改写 / 脑补 / 漏抽 ...)。

运行:  .venv\\Scripts\\python.exe analyze_exp_errors.py
产物:  outputs/exp_error_cases.json (机器可读) + 控制台摘要
"""
import sys, os, json
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from evaluate import generate, try_parse_json, exp_set, norm

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "models", "Qwen2.5-1.5B-Instruct")
ADAPTER_DIR = os.path.join(HERE, "outputs", "resume_adapter")
VAL_FILE = os.path.join(HERE, "data", "resume_val.jsonl")
OUT_FILE = os.path.join(HERE, "outputs", "exp_error_cases.json")


def main():
    rows = [json.loads(l) for l in open(VAL_FILE, encoding="utf-8")]
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
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    model.eval()

    cases = []
    for i, r in enumerate(rows):
        gold = r["label_json"]
        raw = generate(model, tok, r["resume_text"])
        pred = try_parse_json(raw)
        if pred is None:
            cases.append({"idx": i, "type": "unparseable", "raw": raw[:500],
                          "resume_text": r["resume_text"]})
            continue
        g, p = exp_set(gold), exp_set(pred)
        if g != p:
            cases.append({
                "idx": i,
                "type": "exp_mismatch",
                "resume_text": r["resume_text"],
                "gold_experience": gold.get("experience", []),
                "pred_experience": pred.get("experience", []),
                "missed": sorted([list(x) for x in (g - p)]),   # gold 有 pred 没有
                "extra": sorted([list(x) for x in (p - g)]),    # pred 有 gold 没有
            })
        if (i + 1) % 10 == 0:
            print(f"  进度 {i+1}/{len(rows)} | 目前错例 {len(cases)}")

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump({"n_val": len(rows), "n_error": len(cases), "cases": cases},
                  f, ensure_ascii=False, indent=2)

    print(f"\nval {len(rows)} 条中 experience 有错 {len(cases)} 条 -> {OUT_FILE}")
    for c in cases:
        if c["type"] == "exp_mismatch":
            print(f"\n--- 样本 {c['idx']} ---")
            print("  漏抽:", c["missed"])
            print("  多抽:", c["extra"])


if __name__ == "__main__":
    main()
