# 运行这个脚本来检查环境
import sys
import subprocess

print("Python环境检查:")
print(f"Python路径: {sys.executable}")
print(f"Python版本: {sys.version}")

# 检查已安装的包
try:
    subprocess.run([sys.executable, "-m", "pip", "list"], check=True)
except:
    print("无法获取已安装包列表")