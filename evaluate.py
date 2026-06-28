"""
M5: 微调前 vs 微调后 量化对比评估。
在验证集上, 对比 [原始 Qwen2.5-1.5B 直接 prompt] vs [QLoRA 微调后] 的抽取效果。

指标:
- JSON 可解析率: 输出能被 json.loads 成功解析的比例
- name / work_years: 精确匹配准确率
- skills / education / experience: 集合 P/R/F1
- 综合 F1: 三个列表字段 micro-F1 的平均

运行:  .venv\\Scripts\\python.exe evaluate.py
"""
import sys, json, re
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from schema import INSTRUCTION

MODEL_PATH = r"D:\cv_view\resume_qlora\models\Qwen2.5-1.5B-Instruct"
ADAPTER_DIR = r"D:\cv_view\resume_qlora\outputs\resume_adapter"
VAL_FILE = r"D:\cv_view\resume_qlora\data\resume_val.jsonl"


# ---------------- 解析与指标 ----------------
def try_parse_json(text):
    """尽量把模型输出解析成 dict; 失败返回 None。"""
    try:
        return json.loads(text)
    except Exception:
        pass
    # 兜底: 抓第一个 {...} 块
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def norm(s):
    return str(s).strip().lower()


def prf(pred_set, gold_set):
    """集合 P/R/F1。"""
    if not gold_set and not pred_set:
        return 1.0, 1.0, 1.0
    tp = len(pred_set & gold_set)
    p = tp / len(pred_set) if pred_set else 0.0
    r = tp / len(gold_set) if gold_set else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def skills_set(obj):
    return {norm(x) for x in obj.get("skills", []) if isinstance(x, str)}


def edu_set(obj):
    out = set()
    for e in obj.get("education", []):
        if isinstance(e, dict):
            out.add((norm(e.get("school", "")), norm(e.get("degree", "")), norm(e.get("major", ""))))
    return out


def exp_set(obj):
    out = set()
    for e in obj.get("experience", []):
        if isinstance(e, dict):
            out.add((norm(e.get("company", "")), norm(e.get("title", ""))))
    return out


# ---------------- 推理 ----------------
def build_prompt(tok, resume_text):
    msgs = [{"role": "user", "content": INSTRUCTION + resume_text}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def generate(model, tok, resume_text):
    text = build_prompt(tok, resume_text)
    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=512, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(gen, skip_special_tokens=True).strip()


def evaluate(model, tok, rows, tag):
    n = len(rows)
    parse_ok = 0
    name_ok = 0
    wy_ok = 0
    # 累积 micro PRF 的 tp/pred/gold
    agg = {"skills": [0, 0, 0], "edu": [0, 0, 0], "exp": [0, 0, 0]}  # tp, npred, ngold

    def add(key, pred_set, gold_set):
        agg[key][0] += len(pred_set & gold_set)
        agg[key][1] += len(pred_set)
        agg[key][2] += len(gold_set)

    print(f"\n--- 评估 [{tag}] 共 {n} 条 ---")
    for i, r in enumerate(rows):
        gold = r["label_json"]
        raw = generate(model, tok, r["resume_text"])
        pred = try_parse_json(raw)
        if pred is None:
            print(f"  [{i+1}/{n}] JSON 解析失败")
            continue
        parse_ok += 1
        if norm(pred.get("name", "")) == norm(gold.get("name", "")):
            name_ok += 1
        try:
            if int(pred.get("work_years", -999)) == int(gold.get("work_years", -1)):
                wy_ok += 1
        except Exception:
            pass
        add("skills", skills_set(pred), skills_set(gold))
        add("edu", edu_set(pred), edu_set(gold))
        add("exp", exp_set(pred), exp_set(gold))
        if (i + 1) % 10 == 0:
            print(f"  进度 {i+1}/{n}")

    def micro_f1(key):
        tp, npred, ngold = agg[key]
        p = tp / npred if npred else 0.0
        r = tp / ngold if ngold else 0.0
        return 2 * p * r / (p + r) if (p + r) else 0.0

    res = {
        "json_parse_rate": parse_ok / n,
        "name_acc": name_ok / n,
        "work_years_acc": wy_ok / n,
        "skills_f1": micro_f1("skills"),
        "education_f1": micro_f1("edu"),
        "experience_f1": micro_f1("exp"),
    }
    res["avg_list_f1"] = (res["skills_f1"] + res["education_f1"] + res["experience_f1"]) / 3
    return res


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
    base.eval()

    # 微调前: 原始基座
    before = evaluate(base, tok, rows, "微调前 (原始 Qwen)")

    # 微调后: 挂上 adapter
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    model.eval()
    after = evaluate(model, tok, rows, "微调后 (QLoRA)")

    # ---------------- 对比表 ----------------
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
    print("微调前 vs 微调后 对比")
    print("=" * 60)
    print(f"{'指标':<16}{'微调前':>10}{'微调后':>10}{'提升':>12}")
    print("-" * 50)
    table = []
    for name, key in labels:
        b, a = before[key], after[key]
        delta = a - b
        print(f"{name:<16}{b*100:>9.1f}%{a*100:>9.1f}%{delta*100:>+10.1f}pp")
        table.append({"metric": name, "before": round(b, 4), "after": round(a, 4)})

    with open(r"D:\cv_view\resume_qlora\outputs\eval_result.json", "w", encoding="utf-8") as f:
        json.dump({"before": before, "after": after, "table": table}, f, ensure_ascii=False, indent=2)
    print("\n结果已保存: outputs\\eval_result.json")


if __name__ == "__main__":
    main()
