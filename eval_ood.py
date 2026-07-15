"""
M9: 同源 val vs 分布外 OOD 对比评估。
用同一个微调后模型 (outputs/resume_adapter), 分别在:
  - data/resume_val.jsonl   (29 条, 与训练数据同源: deepseek-chat + 干净模板)
  - data/resume_ood.jsonl   (deepseek-reasoner + "脏"模板, 分布外)
上跑同一套指标 (口径与 evaluate.py 完全一致), 差值就是"同源评估乐观了多少"。

运行:  .venv\\Scripts\\python.exe eval_ood.py
产物:  outputs/ood_eval_result.json
"""
import sys, os, json
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from evaluate import evaluate

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "models", "Qwen2.5-1.5B-Instruct")
ADAPTER_DIR = os.path.join(HERE, "outputs", "resume_adapter")
VAL_FILE = os.path.join(HERE, "data", "resume_val.jsonl")
OOD_FILE = os.path.join(HERE, "data", "resume_ood.jsonl")
OUT_FILE = os.path.join(HERE, "outputs", "ood_eval_result.json")


def main():
    val_rows = [json.loads(l) for l in open(VAL_FILE, encoding="utf-8")]
    ood_rows = [json.loads(l) for l in open(OOD_FILE, encoding="utf-8")]

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

    in_dist = evaluate(model, tok, val_rows, f"同源 val ({len(val_rows)} 条)")
    ood = evaluate(model, tok, ood_rows, f"OOD ({len(ood_rows)} 条)")

    labels = [
        ("JSON 可解析率", "json_parse_rate"),
        ("姓名准确率", "name_acc"),
        ("工作年限准确率", "work_years_acc"),
        ("技能 F1", "skills_f1"),
        ("教育经历 F1", "education_f1"),
        ("工作经历 F1", "experience_f1"),
        ("列表字段平均 F1", "avg_list_f1"),
    ]
    print("\n" + "=" * 60)
    print("同源 val vs 分布外 OOD (同一个微调后模型)")
    print("=" * 60)
    print(f"{'指标':<16}{'同源':>10}{'OOD':>10}{'差值':>12}")
    print("-" * 50)
    table = []
    for name, key in labels:
        a, b = in_dist[key], ood[key]
        print(f"{name:<16}{a*100:>9.1f}%{b*100:>9.1f}%{(b-a)*100:>+10.1f}pp")
        table.append({"metric": name, "in_dist": round(a, 4), "ood": round(b, 4),
                      "delta_pp": round((b - a) * 100, 1)})

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "adapter": "outputs/resume_adapter (SFT, r=8 QKVO lr=2e-4)",
            "in_dist_file": "data/resume_val.jsonl",
            "ood_file": "data/resume_ood.jsonl",
            "n_in_dist": len(val_rows), "n_ood": len(ood_rows),
            "ood_construction": "换生成模型(deepseek-reasoner) + 脏格式模板(口语化/字段缺失/断档/缩写/技能内嵌/错别字/时间格式混乱)",
            "in_dist": in_dist, "ood": ood, "table": table,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {OUT_FILE}")


if __name__ == "__main__":
    main()
