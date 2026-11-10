import sys
import io
import json
import re
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, "src")
from langchain_core.messages import AIMessage, ToolMessage
from maritime_agent.sqlite_repository import SQLiteRepository
from maritime_agent.coordinated_agent import CoordinatedAgent

class RoutingModel:
    def __init__(self, roles):
        self.roles = roles
        self.calls = []

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        prompt = str(messages[0].content)
        self.calls.append(prompt)
        if "海事问题协调员" in prompt:
            return AIMessage(content=json.dumps({"roles": self.roles, "subquestions": {}}, ensure_ascii=False))
        if "海事综合分析员" in prompt:
            ids = list(dict.fromkeys(re.findall(r"【([^】]+)】", str(messages[-1].content))))
            return AIMessage(content="综合核查：" + " ".join(f"【{item}】" for item in ids))
        return AIMessage(content="核查通过")

def main():
    repo = SQLiteRepository(Path("storage/maritime.sqlite3"), dataset_id="houston-beryl-2024")
    model = RoutingModel(["ais", "remote_sensing"])
    agent = CoordinatedAgent(repo, model, max_tool_rounds=3, memory_recent_turns=10)

    question = (
        "请针对 2024-01-06（UTC）挪威 Haugesund (hau) 海区，对目标船舶 MMSI 258041000 执行全维度的时空态势调查与多源情报交叉核验，要求完成以下深度任务：\n"
        "1. 【高精度航迹与动力学】：调取该船全天的逐报文离散 AIS 观测点，核算其数据链路完整性（包括有效报文数、超过 15 分钟的观测缺口数），并分析全天航速（SOG）极大值与均值分布；\n"
        "2. 【工况与走锚机理排查】：结合船舶运动学特征与流体力学锚泊机理，严格排查其在 2024-01-06 是否存在走锚失控、非受控漂移或异常违规停留迹象；\n"
        "3. 【COLREGs 会遇与极限逼近解算】：扫描该海区当天全天时空快照，解算目标船舶与他船在切平面下的相对速度矢量矩阵，找出全天物理距离最近时刻、对应船舶 MMSI、极限最小物理间距（米与海里），并按照 COLREGs 1972 国际避碰规则核定会遇态势与双方避碰责任义务；\n"
        "4. 【多源模态资产与环境比对】：核查该海区当日已接入的多源情报资产覆盖情况（涵盖 AIS、海洋气象水文与卫星遥感影像切片），确认各模态有效观测时间窗口是否能够时空对齐；\n"
        "5. 【严苛防幻觉与证据审计】：全篇报告必须区分物理测量事实与派生推论，所有结论必须严格绑定底层可核查证据编号【evidence_id】，禁止任何虚构。"
    )

    print("=== 开始端到端运行单船全维度复合调查长任务 ===")
    res = agent.run_coordinated(question, history=[], summary="")
    
    answer = res.get("answer", "")
    thought = res.get("thought", "")
    evidence = res.get("evidence", [])
    warnings = res.get("warnings", [])

    print("\n==================== [研判报告 ANSWER] ====================")
    print(answer)

    print("\n==================== [思考过程 THOUGHT] ====================")
    print(thought)

    print(f"\n==================== [证据数量]: {len(evidence)} 项 ====================")
    for e in evidence:
        if isinstance(e, dict) and e.get("evidence_id"):
            print(" -", e["evidence_id"])

    print(f"\n==================== [警告信息]: {warnings} ====================")

    # 严苛物理断言核验
    assert "3,236" in answer or "3236" in answer, "缺少有效报文数 3236"
    assert "0 个" in answer or "0个" in answer or "缺口为 0" in answer, "缺少缺口数 0"
    assert "11.0" in answer, "缺少航速极大值 11.0 节"
    assert "20.4" in answer, "缺少极限最小物理逼近距离 20.4 米"
    assert "133.1" in answer or "133" in answer, "缺少遥感空间间距 133 米"
    assert "1.0 秒" in answer or "1.0秒" in answer or "1 秒" in answer, "缺少遥感时间差 1 秒"
    assert "accepted" in answer, "缺少遥感判定决策 accepted"
    assert "未能完成" not in answer, "回答包含消极废话'未能完成'"
    assert "预算已用尽" not in answer, "回答包含消极废话'预算已用尽'"
    assert "无法核验" not in answer, "回答包含消极废话'无法核验'"
    assert "不能给数" not in answer, "回答包含消极废话'不能给数'"

    print("\n🎉 端到端单船全维度复合调查核验 100% 完美通过！")

if __name__ == "__main__":
    main()
