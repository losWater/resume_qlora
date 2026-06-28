"""
M7: CLI Demo —— 简历结构化抽取 + 岗位匹配 的命令行演示。

两种用法:
1) 交互模式 (推荐演示):    .venv\\Scripts\\python.exe cli_demo.py
   粘贴简历文本, 输入空行结束 -> 输出 JSON; 再可选粘贴 JD -> 输出匹配分。
2) 文件模式:               .venv\\Scripts\\python.exe cli_demo.py --resume r.txt [--jd jd.txt]

复用 matcher.py 的解析与打分逻辑。
"""
import sys, json, argparse
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from matcher import ResumeJDParser, match_score


def read_block(prompt):
    """读多行直到空行; 返回拼接文本 (去尾空白)。"""
    print(prompt)
    print("（粘贴完成后，输入一个空行结束；直接空行=跳过）")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def show_resume(resume):
    if not resume:
        print("⚠ 简历解析失败（未得到合法 JSON）")
        return
    print("\n===== 结构化简历 JSON =====")
    print(json.dumps(resume, ensure_ascii=False, indent=2))


def show_match(resume, jd):
    if not jd:
        print("⚠ JD 解析失败（未得到合法 JSON）")
        return
    print("\n===== JD 要求 =====")
    print(json.dumps(jd, ensure_ascii=False, indent=2))
    res = match_score(resume, jd)
    print("\n===== 匹配结果 =====")
    print(f"匹配分: {res['score']} / 100")
    s = res["sub_scores"]
    print(f"分项: 技能 {s['skills']} | 年限 {s['years']} | 学历 {s['degree']}")
    print("理由:")
    for r in res["reasons"]:
        print("  -", r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", help="简历文本文件路径")
    ap.add_argument("--jd", help="岗位 JD 文本文件路径")
    args = ap.parse_args()

    print("加载模型中（基座 + LoRA adapter，首次约需十几秒）...")
    parser = ResumeJDParser()
    print("加载完成。\n")

    if args.resume:
        # 文件模式
        resume_text = open(args.resume, encoding="utf-8").read().strip()
        resume = parser.parse_resume(resume_text)
        show_resume(resume)
        if args.jd and resume:
            jd_text = open(args.jd, encoding="utf-8").read().strip()
            jd = parser.parse_jd(jd_text)
            show_match(resume, jd)
        return

    # 交互模式 (可循环)
    while True:
        resume_text = read_block("\n请粘贴【简历文本】:")
        if not resume_text:
            print("未输入简历，退出。")
            break
        resume = parser.parse_resume(resume_text)
        show_resume(resume)

        if resume:
            jd_text = read_block("\n（可选）请粘贴【岗位 JD】做匹配:")
            if jd_text:
                jd = parser.parse_jd(jd_text)
                show_match(resume, jd)

        again = input("\n再来一条？(y/N): ").strip().lower()
        if again != "y":
            print("结束。")
            break


if __name__ == "__main__":
    main()
