"""
M3 附加: 清洗蒸馏数据的噪声, 再重新切分 train/val。
当前主要噪声: 学历字段不规范 (学士/金融硕士/...)，统一到 大专/本科/硕士/博士。

读: data/resume_all.jsonl
写: data/resume_train.jsonl, data/resume_val.jsonl (覆盖)

运行:  .venv\\Scripts\\python.exe clean_resume_data.py
"""
import sys, os, json, random
from collections import Counter
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from schema import validate_label

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")

STD_DEGREES = ["大专", "本科", "硕士", "博士"]


def normalize_degree(d: str) -> str:
    """把五花八门的学历值映射到 4 个标准值; 无法识别的原样返回。"""
    if not isinstance(d, str):
        return d
    s = d.strip()
    if "博" in s:                       # 博士 / 博士后
        return "博士"
    if "硕" in s or "研究生" in s:       # 硕士 / 金融硕士 / 专业硕士
        return "硕士"
    if "专" in s:                       # 大专 / 专科 / 高职
        return "大专"
    if "本科" in s or "学士" in s:       # 本科 / 学士
        return "本科"
    return s


def clean_row(r: dict) -> dict:
    lbl = r["label_json"]
    for e in lbl.get("education", []):
        if "degree" in e:
            e["degree"] = normalize_degree(e["degree"])
    # work_years 兜底成 int
    wy = lbl.get("work_years")
    if isinstance(wy, float):
        lbl["work_years"] = int(wy)
    # 文本去首尾空白
    r["resume_text"] = r["resume_text"].strip()
    return r


def main():
    all_path = os.path.join(DATA_DIR, "resume_all.jsonl")
    rows = [json.loads(l) for l in open(all_path, encoding="utf-8")]
    print(f"读入 {len(rows)} 条")

    before = Counter(e.get("degree", "?") for r in rows
                     for e in r["label_json"]["education"])
    print("清洗前学历分布:", dict(before))

    cleaned, dropped = [], 0
    for r in rows:
        r = clean_row(r)
        ok, why = validate_label(r["label_json"])
        if ok:
            cleaned.append(r)
        else:
            dropped += 1

    after = Counter(e.get("degree", "?") for r in cleaned
                    for e in r["label_json"]["education"])
    print("清洗后学历分布:", dict(after))
    print(f"通过校验 {len(cleaned)} 条, 丢弃 {dropped} 条")

    random.Random(42).shuffle(cleaned)
    n_val = max(1, len(cleaned) // 10)
    val, train = cleaned[:n_val], cleaned[n_val:]

    for path, data in [(os.path.join(DATA_DIR, "resume_all.jsonl"), cleaned),
                       (os.path.join(DATA_DIR, "resume_train.jsonl"), train),
                       (os.path.join(DATA_DIR, "resume_val.jsonl"), val)]:
        with open(path, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"已写出: train {len(train)} / val {len(val)} (resume_all.jsonl 也已更新为清洗版)")


if __name__ == "__main__":
    main()
