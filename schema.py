"""
简历抽取项目的共享定义: JSON Schema + 指令 + 校验函数。
训练(M4)、推理、评估(M5) 都从这里 import, 保证三处一致。

目标 JSON Schema (文档默认 5 字段):
{
  "name": "张三",
  "education": [
    {"school": "某大学", "degree": "本科", "major": "计算机科学与技术", "year": "2018-2022"}
  ],
  "skills": ["Python", "机器学习", "SQL"],
  "work_years": 3,
  "experience": [
    {"company": "某公司", "title": "算法工程师", "duration": "2022-2025", "desc": "负责..."}
  ]
}
"""

# 给被微调模型用的指令 (输入端)
INSTRUCTION = (
    "你是简历解析助手。请把下面的简历文本抽取成指定 JSON 格式，字段如下：\n"
    "- name: 姓名(字符串)\n"
    "- education: 教育经历(数组)，每项含 school/degree/major/year\n"
    "- skills: 技能(字符串数组)\n"
    "- work_years: 工作年限(整数)\n"
    "- experience: 工作经历(数组)，每项含 company/title/duration/desc\n"
    "只输出 JSON，不要多余解释。\n简历："
)

# 顶层字段及类型
TOP_FIELDS = {
    "name": str,
    "education": list,
    "skills": list,
    "work_years": int,
    "experience": list,
}
EDU_FIELDS = ["school", "degree", "major", "year"]
EXP_FIELDS = ["company", "title", "duration", "desc"]


def validate_label(obj) -> tuple[bool, str]:
    """校验一个 label_json 是否符合 schema。返回 (是否合法, 原因)。"""
    if not isinstance(obj, dict):
        return False, "顶层不是对象"
    for f, t in TOP_FIELDS.items():
        if f not in obj:
            return False, f"缺字段 {f}"
        # work_years 允许是 int 或能转 int 的数字
        if f == "work_years":
            if not isinstance(obj[f], (int, float)):
                return False, "work_years 不是数字"
        elif not isinstance(obj[f], t):
            return False, f"{f} 类型应为 {t.__name__}"
    for e in obj["education"]:
        if not isinstance(e, dict) or not all(k in e for k in EDU_FIELDS):
            return False, "education 项字段不全"
    for e in obj["experience"]:
        if not isinstance(e, dict) or not all(k in e for k in EXP_FIELDS):
            return False, "experience 项字段不全"
    if not all(isinstance(s, str) for s in obj["skills"]):
        return False, "skills 应为字符串数组"
    return True, "ok"
