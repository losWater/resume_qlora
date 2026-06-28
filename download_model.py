"""
下载基座模型 Qwen2.5-1.5B-Instruct (用 modelscope, 国内快且稳)
存到项目内的 models/ 目录 (已被 .gitignore 忽略)。
运行:  .venv\\Scripts\\python.exe download_model.py
"""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from modelscope import snapshot_download

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
LOCAL_DIR = r"D:\cv_view\resume_qlora\models\Qwen2.5-1.5B-Instruct"

print(f"开始下载 {MODEL_ID} -> {LOCAL_DIR}")
path = snapshot_download(MODEL_ID, local_dir=LOCAL_DIR)
print(f"下载完成，模型路径: {path}")
