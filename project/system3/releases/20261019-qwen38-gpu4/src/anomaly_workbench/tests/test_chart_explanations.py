import io
import json
import sys
from types import SimpleNamespace

import pytest

from anomaly_workbench import chart_explanations as charts


@pytest.fixture(autouse=True)
def clear_cache():
    charts._CACHE.clear()
    yield
    charts._CACHE.clear()


@pytest.fixture
def evidence():
    return (
        {"summary": {"candidate_windows": 10, "field_valid_ratio": .0135},
         "timeline": [{"candidate_count": 3}, {"candidate_count": 7}],
         "type_counts": {"deviation": 5, "speed_drop": 2, "observation_glitch": 4, "congestion": 12},
         "scatter": [{"field_residual_mps": 2.5, "e_kin_mps": .2}, {"field_residual_mps": None, "e_kin_mps": 1}]},
        {"sog_kn": [8.5, 8, 3.1], "gate_weights": [1, .2, 1], "time_seconds": [0, 60, 120],
         "motion_observations": {"time_seconds": [0, 60, 120], "source_sog": [8.5, 8, 3.1],
                                 "source_cog": [359, 1, 5], "sog_reported_mask": [True, True, True],
                                 "cog_reported_mask": [True, True, True]},
         "diagnostics": {"pg_probability": [.1, .3, .9]},
         "field_residual_mps": 6.413,
         "maritime_policy": {"neighbor_supported": True, "initial_sog_kn": 8.5, "late_sog_kn": 3.1}},
    )


def settings(monkeypatch, base="http://127.0.0.1:8002/v1"):
    value = SimpleNamespace(model_base_url=base, model_name="qwen3.8-27b", model_api_key="REDACTED")
    monkeypatch.setitem(sys.modules, "maritime_agent.config", SimpleNamespace(Settings=SimpleNamespace(from_env=lambda: value)))


def model_transport(monkeypatch, answer):
    requests = []

    def opened(request, timeout):
        requests.append((request, timeout))
        result = {"choices": [{"message": {"content": json.dumps(answer, ensure_ascii=False)}}]}
        return io.BytesIO(json.dumps(result).encode())

    monkeypatch.setattr(charts.urllib.request, "build_opener", lambda *args: SimpleNamespace(open=opened))
    return requests


def test_real_chart_facts_do_not_invent_neighbors_or_congestion(evidence):
    scan, item = evidence
    facts, reference = charts.chart_evidence(scan, item)
    assert facts["quality"]["low_confidence_points"] == 1
    assert facts["timeline"]["windows"] == 2
    assert facts["timeline"]["field_valid_percent"] == 1.4
    assert facts["types"]["dominant_behavior"] == "方向变化线索"
    assert facts["types"]["regional_congestion_windows"] == 12
    assert facts["residual"]["plotted_windows"] == 1
    assert facts["residual"]["unplotted_windows"] == 1
    assert "不能据散度直接判断拥堵" in reference["timeline"]
    assert "不能将各项相加当作船数" in reference["types"]
    assert "6.41米/秒" in reference["residual"]
    assert "平均相对运动差异" in reference["residual"] and "而非未来方向" in reference["residual"]


def test_local_model_reads_only_bounded_facts_and_returns_five_readings(monkeypatch, evidence):
    settings(monkeypatch)
    scan, item = evidence
    scan["dataset"] = {"name": "DO_NOT_SEND_UNTRUSTED_DATASET_INSTRUCTION"}
    item["review_history"] = [{"note": "DO_NOT_SEND_PRIVATE_NOTE"}]
    item["latlon"] = [[30, -88]] * 10000
    _, reference = charts.chart_evidence(scan, item)
    requests = model_transport(monkeypatch, reference)
    result = charts.generate_chart_explanations(scan, item)
    assert result["source"] == "local_llm"
    assert set(result["explanations"]) == set(charts.PURPOSES)
    assert all(set(v) == {"purpose", "reading"} for v in result["explanations"].values())
    request, timeout = requests[0]
    assert request.full_url == "http://127.0.0.1:8002/v1/chat/completions"
    assert timeout <= 25
    payload = json.loads(request.data)
    assert payload["chat_template_kwargs"]["enable_thinking"] is False
    assert payload["response_format"] == {"type": "json_object"}
    assert "DO_NOT_SEND" not in payload["messages"][1]["content"]
    assert len(request.data) < 20000


@pytest.mark.parametrize("base", ["https://model.remote.example/v1", "file:///tmp/model", "http://localhost:8002/v1?redirect=external", "http://user:pass@127.0.0.1/v1"])
def test_external_or_credentialed_model_never_receives_evidence(monkeypatch, evidence, base):
    settings(monkeypatch, base)
    monkeypatch.setattr(charts.urllib.request, "build_opener", lambda *a: pytest.fail("must not make request"))
    result = charts.generate_chart_explanations(*evidence)
    assert result["source"] == "template_fallback"
    assert "规则说明" in result["notices"][0]


def test_model_failure_is_explicit_fallback(monkeypatch, evidence):
    monkeypatch.setattr(charts, "_ask_model", lambda *args: (_ for _ in ()).throw(TimeoutError()))
    result = charts.generate_chart_explanations(*evidence)
    assert result["source"] == "template_fallback"
    assert "本地模型暂未完成" in result["notices"][0]
    assert "前段航速为8.50节" in result["explanations"]["motion"]["reading"]


@pytest.mark.parametrize("replacement", ["当前航速为54321节，已经异常。", "当前船的SOG需要核查。", "当前船的异常概率很高。"])
def test_model_cannot_add_numbers_jargon_or_probability_claims(monkeypatch, evidence, replacement):
    settings(monkeypatch)
    _, reference = charts.chart_evidence(*evidence)
    reference["motion"] = replacement
    model_transport(monkeypatch, reference)
    result = charts.generate_chart_explanations(*evidence)
    assert result["source"] == "template_fallback"
    assert replacement not in result["explanations"]["motion"]["reading"]


@pytest.mark.parametrize('boundary', ['非提前预警概率', '不是异常概率', '不代表海水流速'])
def test_explicit_negation_of_unsupported_claim_is_valid(monkeypatch, evidence, boundary):
    settings(monkeypatch)
    _, reference = charts.chart_evidence(*evidence)
    reference['quality'] = '此图仅作模型变化提示，' + boundary + '，仍需结合真实轨迹核查。'
    model_transport(monkeypatch, reference)
    result = charts.generate_chart_explanations(*evidence)
    assert result['source'] == 'local_llm'
    assert boundary in result['explanations']['quality']['reading']


def test_sparse_field_does_not_expose_stale_residual(evidence):
    scan, item = evidence
    item["maritime_policy"]["neighbor_supported"] = False
    facts, reference = charts.chart_evidence(scan, item)
    assert facts["residual"]["selected_difference_mps"] is None
    assert "6.41" not in reference["residual"]
    assert "未确认" in reference["residual"] and "不填零" in reference["residual"]


def test_fact_cache_is_bounded_and_returns_independent_objects(monkeypatch, evidence):
    calls = []

    def answer(facts, reference):
        calls.append(facts)
        return reference

    monkeypatch.setattr(charts, "_ask_model", answer)
    first = charts.generate_chart_explanations(*evidence)
    second = charts.generate_chart_explanations(*evidence)
    second["explanations"]["motion"]["reading"] = "changed only returned copy"
    third = charts.generate_chart_explanations(*evidence)
    assert first["cached"] is False and third["cached"] is True
    assert len(calls) == 1 and third["explanations"]["motion"]["reading"] != second["explanations"]["motion"]["reading"]
    for n in range(charts._MAX_CACHE + 2):
        evidence[1]["maritime_policy"]["initial_sog_kn"] = n
        charts.generate_chart_explanations(*evidence)
    assert len(charts._CACHE) == charts._MAX_CACHE


def test_model_busy_does_not_queue_unbounded_calls(evidence):
    assert charts._MODEL_SLOT.acquire(blocking=False)
    try:
        result = charts.generate_chart_explanations(*evidence)
        assert result["source"] == "template_fallback" and "其他图表" in result["notices"][0]
        assert not charts._CACHE and not charts._INFLIGHT
    finally:
        charts._MODEL_SLOT.release()


def test_redirect_to_another_destination_is_blocked():
    with pytest.raises(ValueError, match="不接受重定向"):
        charts._NoRedirect().redirect_request(None, None, 302, "redirect", {}, "https://external.example")


def test_missing_data_are_not_filled_as_normal_or_zero():
    facts, reference = charts.chart_evidence({}, {})
    assert facts["motion"]["speed_min_kn"] is None
    assert facts["residual"]["selected_difference_mps"] is None
    assert facts['timeline']['field_valid_percent'] is None
    assert "缺少有效原始航速采样点" in reference["motion"]
    assert "未提供有效观测可信程度" in reference["quality"]


def test_number_validation_handles_chinese_adjacent_numbers():
    assert charts._number_set("航速为54321节，可信度0.30。") == {54321, charts.Decimal("0.3")}


def test_chart_facts_use_original_reports_and_keep_missing_fields(evidence):
    scan, item = evidence
    item['sog_kn'] = [99, 99, 99]
    item['motion_observations']['source_sog'][1] = None
    item['motion_observations']['cog_reported_mask'][2] = False
    facts, reference = charts.chart_evidence(scan, item)
    assert facts['motion']['speed_max_kn'] == 8.5
    assert facts['motion']['missing_sampled_speed_points'] == 1
    assert facts['motion']['missing_sampled_course_points'] == 1
    assert facts['quality']['sampled_points_with_missing_report_fields'] == 2
    assert '缺失不补值' in reference['motion']
    assert '航速、航向字段' in reference['quality']


def test_original_report_alignment_never_interpolates():
    item = {'time_seconds': [0, 10.0005, 19, 30],
            'motion_observations': {'time_seconds': [0, 10, 20, 30],
                                    'source_sog': [0, 5, 6, 7], 'source_cog': [0, 359, 1, 2],
                                    'sog_reported_mask': [True, True, True, False],
                                    'cog_reported_mask': [True, True, True, True]}}
    result = charts._original_series(item)
    assert result['sog_kn'] == [0, 5, None, None]
    assert result['cog_deg'] == [0, 359, None, 2]


def test_brief_counting_keeps_time_vessel_and_quality_units_separate():
    from anomaly_workbench.prompts import template_brief
    scan = {'dataset': {'name': 'Mobile'},
            'summary': {'window_count': 360, 'unique_mmsi': 236, 'vessels_in_window': 236, 'quality_count': 8},
            'type_counts': {'observation_glitch': 29}}
    result = template_brief(scan, {'id': 'selected', 'review_level': 'clue'})
    assert '时间窗 360 个，唯一船舶 236 艘' in result
    assert '去重船舶 8 艘、船舶窗口 29 个' in result
    assert '236 个窗口' not in result and '8 个窗口' not in result
    assert '一般运动线索，未满足重点核查资格' in result


def test_brief_missing_counts_stay_missing():
    from anomaly_workbench.prompts import template_brief
    result = template_brief({}, {})
    assert '时间窗 未提供 个，唯一船舶 未提供 艘' in result
    assert '去重船舶 未提供 艘、船舶窗口 未提供 个' in result
