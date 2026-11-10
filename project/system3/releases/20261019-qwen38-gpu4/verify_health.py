import sys
from pathlib import Path

print("=== 正在自检海事协同研判智能体服务环境 ===")

# 1. 检查 Python 版本
py_ver = sys.version_info
print(f"1. Python 版本: {py_ver.major}.{py_ver.minor}.{py_ver.micro} (要求 >= 3.11)")
if py_ver < (3, 11):
    print("   [错误] Python 版本过低，请使用 Python 3.11 或更高版本！")
    sys.exit(1)

# 2. 检查核心数据库
db_file = Path("storage/maritime.sqlite3")
print(f"2. 检查主数据库: {db_file}")
if not db_file.exists():
    print("   [错误] 未找到主数据库 storage/maritime.sqlite3！")
    sys.exit(1)
print(f"   -> 数据库就绪 (文件大小: {db_file.stat().st_size / (1024*1024):.1f} MB)")

# 3. 检查依赖包导入
print("3. 检查核心模块导入...")
try:
    import fastapi
    import uvicorn
    import langgraph
    import langchain_openai
    import numpy
    print("   -> 核心依赖包 (fastapi, langgraph, langchain_openai, numpy) 导入正常")
except ImportError as e:
    print(f"   [警告] 依赖包缺失: {e}，请运行: pip install -r requirements.txt")
    sys.exit(1)

# 4. 检查智能体模块
print("4. 检查智能体核心引擎...")
try:
    from maritime_agent.config import Settings
    from maritime_agent.storage import SQLiteRepository
    from maritime_agent.coordinated_agent import CoordinatedAgent
    from maritime_agent.colregs import solve_continuous_segment_cpa
    from maritime_agent.anchor_physics import assess_anchor_stability
    print("   -> 智能体核心引擎与流体力学/连续CPA模块全部就绪！")
except Exception as e:
    print(f"   [错误] 智能体模块加载失败: {e}")
    sys.exit(1)

print("\n=== 环境自检全部通过！系统已具备在服务器上直接启动运行的全部条件 ===")
print("请运行: ./start_server.sh (Linux) 或 start_server.bat (Windows) 启动系统。")
