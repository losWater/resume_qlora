"""
P1 经历专项 第二步: 修复 experience 标注噪声。

错误分析 (outputs/exp_error_cases.json) 发现的标注问题:
  - gold 自行补全公司后缀 ("恒达实业" -> "恒达实业有限公司")
  - gold 编造原文没有的公司名 / 任职时间
  - 职位表述不忠实原文 (改写/同义替换)
本脚本让 DeepSeek 按严格"忠实原文"规则重新校对每条的 experience 字段,
**其他字段一律不动, train/val 切分与行序保持原样** (保证新旧指标可比)。

运行:  .venv\\Scripts\\python.exe relabel_exp.py
改写:  data/resume_train.jsonl / data/resume_val.jsonl (原位置覆盖, 旧版在 git 历史里)
产物:  outputs/relabel_report.json (改动统计)
"""
import sys, os, json, time
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from dotenv import load_dotenv
from openai import OpenAI
from schema import EXP_FIELDS

HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(HERE, ".env"))
API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip()

FILES = ["data/resume_train.jsonl", "data/resume_val.jsonl"]

SYS = "你是训练数据标注校对助手。你的任务是让标注严格忠实于原文，绝不添加原文没有的信息。"

TMPL = """下面是一份简历原文和它现有的 experience 标注。现有标注可能存在"编造原文没有的信息"的噪声，请按以下规则逐条校对并输出修正后的 experience。

【规则（严格执行）】
1. company：必须照抄原文中出现的写法（包括缩写、简称），禁止补全后缀（原文写"恒达实业"就不能写"恒达实业有限公司"）；原文完全没提公司名的，填"未提及"。
2. title：用原文的表述；原文是"从A做到B"式晋升的，取最终职位 B；原文只说"带团队"等而无明确职位名的，填"未提及"。
3. duration：仅当原文明确给出该段经历的时间时才填，且照抄原文写法（"8年"就写"8年"，"2019.3-2020.12"就照抄）；原文没给的填"未提及"，禁止推算或编造年份。
4. desc：概括原文中该段经历的职责描述，只能用原文提到的内容；原文没有描述的填"未提及"。
5. 经历条目以原文为准：原文提到几段就写几段（包括实习、校内项目），顺序按原文；现有标注里编造的条目要删掉，遗漏的条目要补上。
6. 每个条目必须包含 company/title/duration/desc 四个键，值都是字符串。

【简历原文】
{text}

【现有 experience 标注】
{exp}

只输出一个 JSON 对象：{{"experience": [...]}}，不要任何解释。"""


def relabel_one(client, row):
    user = TMPL.format(text=row["resume_text"],
                       exp=json.dumps(row["label_json"].get("experience", []),
                                      ensure_ascii=False))
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": SYS},
                          {"role": "user", "content": user}],
                response_format={"type": "json_object"},
                temperature=0,          # 校对任务, 不要发散
                max_tokens=1500,
            )
            data = json.loads(resp.choices[0].message.content)
            exp = data.get("experience")
            if not isinstance(exp, list):
                raise ValueError("缺 experience 数组")
            for e in exp:
                if not isinstance(e, dict) or not all(k in e for k in EXP_FIELDS):
                    raise ValueError("条目字段不全")
                for k in EXP_FIELDS:
                    if not isinstance(e[k], str):
                        e[k] = str(e[k])
            return exp
        except Exception as ex:
            if attempt == 2:
                return None
            time.sleep(1.5 * (attempt + 1))


def main():
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    report = {"model": MODEL, "files": {}}

    for rel in FILES:
        path = os.path.join(HERE, rel)
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        print(f"\n校对 {rel} ({len(rows)} 条) ...")
        changed, failed = 0, 0
        results = {}
        with ThreadPoolExecutor(max_workers=6) as ex:
            futs = {ex.submit(relabel_one, client, r): i for i, r in enumerate(rows)}
            done = 0
            for fut in as_completed(futs):
                i = futs[fut]
                results[i] = fut.result()
                done += 1
                if done % 30 == 0 or done == len(rows):
                    print(f"  进度 {done}/{len(rows)}")

        for i, r in enumerate(rows):
            new_exp = results.get(i)
            if new_exp is None:
                failed += 1          # 校对失败的保持原标注
                continue
            if new_exp != r["label_json"].get("experience", []):
                changed += 1
            r["label_json"]["experience"] = new_exp

        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        report["files"][rel] = {"n": len(rows), "changed": changed, "failed": failed}
        print(f"  {rel}: 改动 {changed}/{len(rows)} 条, 校对失败(保留原标注) {failed}")

    with open(os.path.join(HERE, "outputs", "relabel_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\n完成。改动统计已存 outputs/relabel_report.json")
    print("注意: resume_all.jsonl 未动 (它只是切分前的原始汇总, 训练/评估不用它)。")


if __name__ == "__main__":
    main()
