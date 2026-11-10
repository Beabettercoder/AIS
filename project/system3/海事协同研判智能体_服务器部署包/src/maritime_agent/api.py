"""HTTP API and a minimal local question UI."""

from __future__ import annotations

from functools import lru_cache
import inspect
import os
from pathlib import Path
import re
import sqlite3
from uuid import UUID

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from .agent import _chunk_text, ask, create_graph, create_model, stream_ask
from .config import Settings
from .conversation_store import ConversationStore
from .data_intake_agent import DataIntakeAgent
from .fusion import index_bound_source
from .reasoning_memory import EvolvedCritique, assemble_cognitive_prompt_block
from .repository import DataRepository
from .situation_cube import build_situation_cube
from .storage import open_repository


app = FastAPI(title="Maritime Situation Agent", version="0.16.0")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    conversation_id: UUID | None = None
    source_id: UUID | None = None
    region_id: str | None = Field(default=None, min_length=1, max_length=160,
                                  pattern=r"^[A-Za-z0-9:_-]+$")


class CreateCritiqueRequest(BaseModel):
    domain: str = Field(default="general", min_length=1, max_length=50)
    trigger_pattern: str = Field(min_length=1, max_length=500)
    failed_path: str = Field(default="", max_length=500)
    corrected_lesson: str = Field(min_length=1, max_length=1000)
    confidence: float = Field(default=1.0, ge=0.0, le=2.0)


class CritiqueFeedbackRequest(BaseModel):
    action: str = Field(pattern=r"^(upvote|downvote)$")


class IntakeCommitRequest(BaseModel):
    mapping: dict[str, str] | None = None
    repair_actions: list[str] | None = None
    auto_process: bool = True


class NOAAChannelsRequest(BaseModel):
    bbox_wgs84: list[float]
    region_label: str = "Houston"
    region_id: str = ""


class SarLabelSubmitRequest(BaseModel):
    sar_target_id: str
    truth_mmsi_or_no_match: str
    review_status: str
    reviewer: str = "Reviewer-1"
    notes: str = ""
    limit: int = Field(default=50, ge=1, le=100)


class SituationCubeBuildRequest(BaseModel):
    region_id: str = Field(min_length=1, max_length=160)
    start_time_utc: str = ""
    end_time_utc: str = ""
    task_type: str = Field(default="general_situation", min_length=1, max_length=100)
    refresh_fusion_index: bool = False


@lru_cache(maxsize=1)
def repository() -> DataRepository:
    return open_repository(Settings.from_env())


@lru_cache(maxsize=1)
def graph():
    settings = Settings.from_env()
    model = create_model(settings)
    if settings.agent_mode == "coordinated":
        from .coordinated_agent import create_coordinated_graph
        return create_coordinated_graph(repository(), model, settings.max_tool_rounds,
                                        settings.memory_recent_turns)
    return create_graph(repository(), model, settings.max_tool_rounds,
                        settings.memory_recent_turns)


@lru_cache(maxsize=1)
def conversation_store() -> ConversationStore:
    return ConversationStore(Settings.from_env().conversation_db_path)


@lru_cache(maxsize=1)
def intake_agent() -> DataIntakeAgent:
    settings = Settings.from_env()
    if settings.storage_backend != "sqlite" or settings.db_path is None:
        raise ValueError("数据接入需要 SQLite 模式")
    model = create_model(settings) if settings.model_api_key else None
    return DataIntakeAgent(settings.db_path, settings.db_path.parent / "intake", model,
                           dataset_id=settings.dataset_id)


def _is_greeting_or_meta(question: str) -> bool:
    clean = re.sub(r"[^\w\u4e00-\u9fa5a-zA-Z]", "", question.strip()).lower()
    if not clean:
        return True
    greetings = {
        "你好", "您好", "hi", "hello", "hey", "哈喽", "在吗", "在不在",
        "早", "早上好", "中午好", "下午好", "晚上好",
        "谢谢", "多谢", "感谢", "thanks", "thankyou", "thx",
        "再见", "拜拜", "bye", "goodbye",
        "你是谁", "介绍一下自己", "你能做什么", "你能帮我做什么", "请问你是谁",
        "你好你是谁", "你好请问你是谁", "您好请问你是谁", "你好介绍一下自己",
        "帮助", "help",
    }
    if clean in greetings:
        return True
    if any(clean.startswith(g) for g in ("你好", "您好", "hi", "hello")) and any(
        sub in clean for sub in ("你是谁", "自我介绍", "介绍一下", "能做什么", "做什么")
    ):
        return True
    if re.fullmatch(r"^(?:你|您)?好[啊呀吧\s!！~～.]*$", clean):
        return True
    return False


def _is_unrecognized_short_input(question: str) -> bool:
    clean = question.strip()
    if clean.isdigit():
        if len(clean) == 9:
            return False
        if len(clean) == 4 and 1990 <= int(clean) <= 2030:
            return False
        return True
    if re.fullmatch(r"^[a-zA-Z]{1,4}$", clean) and clean.lower() not in {"gis", "ais", "sar"}:
        return True
    if len(clean) <= 4 and clean in {"海", "随便", "随便一个", "测试", "看一下", "看下", "查下", "搜下", "船", "查船", "轨迹", "啊", "1"}:
        return True
    return False


def _is_template_echo(question: str) -> bool:
    clean = question.strip()
    templates = [
        "查询特定 MMSI 的逐报文 AIS 轨迹、走锚或异常停留",
        "检索 Sentinel-1 雷达或光学卫星过境影像及船舶检测点",
        "核查特定时空范围内的遥感—AIS 匹配关系与暗船排查",
        "核实连续风场、风暴潮或恶劣海况对通航的影响",
        "检索 Sentinel-1 雷达与光学卫星过境影像及暗船切片",
        "核查特定时空范围内的遥感-AIS 匹配关系与暗船排查",
        "分析实测连续风场、风暴预警对航道通航的影响",
    ]
    return any(clean == t or clean == f"1. {t}" or clean == f"2. {t}" or clean == f"3. {t}" or clean == f"4. {t}"
               or clean == f"1. **船舶活动与航迹**：{t}" for t in templates)


def _local_origin(request: Request) -> None:
    origin = request.headers.get("origin", "")
    if not origin:
        return
    host = request.headers.get("host", "")
    if host and (origin == f"http://{host}" or origin == f"https://{host}"):
        return
    allowed = {"http://127.0.0.1:8000", "http://localhost:8000"}
    extra = os.environ.get("ALLOWED_ORIGINS", "")
    if extra:
        allowed.update([o.strip() for o in extra.split(",") if o.strip()])
    if origin not in allowed:
        raise HTTPException(status_code=403, detail="数据接入只接受同源或授权请求")


@app.get("/health")
def health():
    try:
        repo = repository()
        info = repo.overview()
        settings = Settings.from_env()
        return {"status": "ok", "data_ready": True,
                "data_backend": info.get("backend", settings.storage_backend),
                "dataset_id": info.get("dataset_id", settings.dataset_id),
                "database_ready": settings.storage_backend == "sqlite",
                "time_range_utc": info["time_range_utc"],
                "model_configured": bool(settings.model_api_key)}
    except (ValueError, FileNotFoundError, sqlite3.DatabaseError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/catalog")
def catalog():
    try:
        return repository().overview()
    except (ValueError, FileNotFoundError, sqlite3.DatabaseError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/data-sources")
def data_sources(date_utc: str = "", region: str = "", kind: str = "",
                 limit: int = Query(default=100, ge=1, le=100)):
    try:
        return repository().search_ingested_data(date_utc, region, kind, limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/regions")
def regions():
    try:
        return intake_agent().regions()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/data-sources/{source_id}")
def delete_data_source(source_id: str, request: Request, confirmation: str = ""):
    _local_origin(request)
    if confirmation != source_id:
        raise HTTPException(status_code=409, detail="必须输入完整来源 ID 进行二次确认")
    try:
        result = intake_agent().delete_source(source_id)
        repository.cache_clear()
        graph.cache_clear()
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/data-sources/{source_id}/delete-impact")
def data_source_delete_impact(source_id: str):
    try:
        return intake_agent().source_delete_impact(source_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/data-sources/{source_id}/verify")
def verify_data_source(source_id: str):
    try:
        repo = repository()
        if not hasattr(repo, "verify_ingested_source"):
            raise ValueError("来源核验需要 SQLite 模式")
        return repo.verify_ingested_source(source_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/data-sources/{source_id}/tracks")
def ingested_tracks(source_id: str, date_utc: str = "", mmsi: str = "",
                    limit: int = Query(default=20, ge=1, le=20)):
    try:
        return repository().search_ingested_tracks(source_id, date_utc, mmsi, limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/data-sources/{source_id}/tracks/{date_utc}/{mmsi}")
def ingested_track_detail(source_id: str, date_utc: str, mmsi: str,
                          granularity: str = Query(default="message", pattern="^(hourly|message)$")):
    try:
        return repository().get_ingested_track(source_id, date_utc, mmsi, granularity)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/data/metrics/hourly")
def data_hourly(source_id: str, metric: str, start_utc: str, end_utc: str):
    try:
        repo = repository()
        if not hasattr(repo, "query_data_hourly"):
            raise ValueError("统一来源查询需要 SQLite 模式")
        return repo.query_data_hourly(source_id, metric, start_utc, end_utc)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/data/discover")
def data_discover(source_id: str, desired_kinds: str, date_utc: str = ""):
    try:
        repo = repository()
        if not hasattr(repo, "discover_complementary_data"):
            raise ValueError("补充数据发现需要 SQLite 模式")
        kinds = [value.strip() for value in desired_kinds.split(",") if value.strip()]
        return repo.discover_complementary_data(source_id, kinds, date_utc)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/gis/noaa/channels/cache")
def noaa_channels_cache(body: NOAAChannelsRequest, request: Request):
    _local_origin(request)
    try:
        result = repository().acquire_noaa_channels(body.bbox_wgs84, body.region_label, body.limit)
        if body.region_id:
            region = intake_agent().bind_existing_source(
                "noaa_enc_channels", result["source_id"], body.region_id,
                body.region_label, result.get("bbox_wgs84") or body.bbox_wgs84)
            result.update({"region_id": region["region_id"],
                           "region_label": region["region_label"],
                           "region_validation": region})
        return result
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/gis/external/search")
def external_gis_search(query: str = "", source_id: str = "", date_utc: str = "", limit: int = 5):
    try:
        return repository().search_external_gis(query, source_id, date_utc, limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/gis/external/{source_id}/{feature_id}")
def external_gis_detail(source_id: str, feature_id: str):
    try:
        return repository().get_external_gis_feature(source_id, feature_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/data-intake/preview")
async def data_intake_preview(request: Request, kind: str, region_label: str,
                              source_name: str, assume_utc: bool = False,
                              region_id: str = "",
                              use_as_region_boundary: bool = False,
                              data_mode: str = "observation", weather_source_type: str = "generic",
                              location_id: str = "", location_name: str = "",
                              location_lon: float | None = None, location_lat: float | None = None,
                              observed_at_utc: str = "", scene_id: str = "", collection: str = "",
                              platform: str = "", sensor: str = "", polarization: str = "",
                              source_url: str = "", published_at: str = ""):
    _local_origin(request)
    if int(request.headers.get("content-length", "0") or "0") > 200 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="单个文件上限为 200 MB")
    data = await request.body()
    try:
        return intake_agent().preview(data, kind, region_label, source_name, assume_utc, data_mode=data_mode,
                                      weather_source_type=weather_source_type, location_id=location_id,
                                      location_name=location_name, location_lon=location_lon, location_lat=location_lat,
                                      observed_at_utc=observed_at_utc, scene_id=scene_id,
                                      collection=collection, platform=platform, sensor=sensor,
                                      polarization=polarization, region_id=region_id,
                                      use_as_region_boundary=use_as_region_boundary,
                                      source_url=source_url, published_at=published_at)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/data-intake/{job_id}/commit")
def data_intake_commit(job_id: str, body: IntakeCommitRequest, request: Request):
    _local_origin(request)
    try:
        result = intake_agent().commit(job_id, body.mapping, body.repair_actions, body.auto_process)
        region_id = result.get("region_id")
        if region_id:
            settings = Settings.from_env()
            try:
                result["fusion_index"] = index_bound_source(
                    settings.db_path, settings.dataset_id, result["source_id"])
                result["situation_cube"] = build_situation_cube(
                    settings.db_path, settings.dataset_id, region_id,
                    start_time_utc=result.get("first_utc") or "",
                    end_time_utc=result.get("last_utc") or "",
                    task_type="intake_incremental", refresh_fusion_index=False)
            except (ValueError, sqlite3.Error) as cube_error:
                # Source commit remains durable.  Expose a retryable derived-index
                # failure instead of rolling back or hiding a successful import.
                result["situation_cube"] = {"status": "pending_retry", "error": str(cube_error)}
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/data-intake/{job_id}/validate")
def data_intake_validate(job_id: str, body: IntakeCommitRequest, request: Request):
    _local_origin(request)
    try:
        return intake_agent().validate(job_id, body.mapping, body.repair_actions)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/data-intake", response_class=HTMLResponse)
def data_intake_page():
    return HTMLResponse(Path(__file__).with_name("data_intake.html").read_text(encoding="utf-8"),
                        headers={"Cache-Control": "no-store"})


@app.get("/assets/data-intake.js")
def data_intake_script():
    return FileResponse(Path(__file__).with_name("data_intake.js"), media_type="application/javascript",
                        headers={"Cache-Control": "no-cache"})


@app.get("/data-management", response_class=HTMLResponse)
def data_management_page():
    return HTMLResponse(Path(__file__).with_name("data_management.html").read_text(encoding="utf-8"),
                        headers={"Cache-Control": "no-store"})


@app.get("/assets/data-management.js")
def data_management_script():
    return FileResponse(Path(__file__).with_name("data_management.js"),
                        media_type="application/javascript",
                        headers={"Cache-Control": "no-cache"})


@app.get("/situation-cube", response_class=HTMLResponse)
def situation_cube_page():
    return HTMLResponse(Path(__file__).with_name("situation_cube.html").read_text(encoding="utf-8"),
                        headers={"Cache-Control": "no-store"})


@app.get("/assets/situation-cube.js")
def situation_cube_script():
    return FileResponse(Path(__file__).with_name("situation_cube.js"),
                        media_type="application/javascript",
                        headers={"Cache-Control": "no-cache"})


@app.get("/sar-verification", response_class=HTMLResponse)
def sar_verification_page():
    html_file = Path(__file__).with_name("sar_verification.html")
    if not html_file.exists():
        raise HTTPException(status_code=404, detail="核验工作台界面未找到")
    return HTMLResponse(html_file.read_text(encoding="utf-8"), headers={"Cache-Control": "no-store"})


@app.get("/sar-verification/targets")
def sar_verification_targets_endpoint():
    from .sar_verification import load_verification_targets
    try:
        return load_verification_targets()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/sar-verification/submit")
def sar_verification_submit_endpoint(body: SarLabelSubmitRequest):
    from .sar_verification import save_manual_label
    try:
        return save_manual_label(
            body.sar_target_id,
            body.truth_mmsi_or_no_match,
            body.review_status,
            reviewer=body.reviewer,
            notes=body.notes
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/tracks")
def tracks(date_utc: str = Query(..., pattern=r"^\d{4}-\d{2}-\d{2}$"),
           vessel_class: str = "", mmsi: str = "", limit: int = Query(default=100, ge=1, le=200)):
    try:
        return repository().map_tracks(date_utc, vessel_class, mmsi, limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/tracks/{date_utc}/{mmsi}")
def track_detail(date_utc: str, mmsi: str,
                 granularity: str = Query(default="message", pattern="^(hourly|message)$")):
    try:
        if granularity == "message":
            try:
                return repository().get_message_track(date_utc, mmsi)
            except ValueError as exc:
                if "没有已导入的逐报文" not in str(exc):
                    raise
                return repository().get_track(date_utc, mmsi)
        return repository().get_track(date_utc, mmsi)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/vessel-track-history/{mmsi}")
def vessel_track_history(mmsi: str,
                         date_utc: str = Query(default="", pattern=r"^(\d{4}-\d{2}-\d{2}|all)?$"),
                         region_id: str = ""):
    try:
        return repository().get_vessel_track_history(mmsi=mmsi, date_utc=date_utc, region_id=region_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc



@app.get("/gis-objects/{object_id}")
def gis_object_detail(object_id: str):
    try:
        return repository().get_gis_object(object_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/remote-sensing/{scene_id}")
def remote_sensing_detail(scene_id: str):
    try:
        return repository().get_remote_sensing_scene(scene_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/fusion/overview")
def fusion_overview_endpoint():
    try:
        repo = repository()
        if not hasattr(repo, "fusion_overview"):
            raise ValueError("统一关联查询需要 SQLite 模式")
        return repo.fusion_overview()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/fusion/relations")
def fusion_relations_endpoint(node_id: str = "", predicate: str = "", modality: str = "",
                              limit: int = Query(default=20, ge=1, le=100)):
    try:
        repo = repository()
        if not hasattr(repo, "search_fusion_relations"):
            raise ValueError("统一关联查询需要 SQLite 模式")
        return repo.search_fusion_relations(node_id, predicate, modality, limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/situation-cubes/build")
def situation_cube_build_endpoint(body: SituationCubeBuildRequest, request: Request):
    _local_origin(request)
    settings = Settings.from_env()
    if settings.storage_backend != "sqlite" or settings.db_path is None:
        raise HTTPException(status_code=400, detail="态势立方体构建需要 SQLite 模式")
    try:
        result = build_situation_cube(
            settings.db_path, settings.dataset_id, body.region_id,
            start_time_utc=body.start_time_utc, end_time_utc=body.end_time_utc,
            task_type=body.task_type, refresh_fusion_index=body.refresh_fusion_index)
        repository.cache_clear()
        graph.cache_clear()
        return result
    except (ValueError, sqlite3.Error) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/situation-cubes")
def situation_cubes_endpoint(region_id: str = ""):
    repo = repository()
    if not hasattr(repo, "situation_cube_overview"):
        raise HTTPException(status_code=400, detail="态势立方体查询需要 SQLite 模式")
    return repo.situation_cube_overview(region_id)


@app.get("/situation-units")
def situation_units_endpoint(region_id: str = "", start_time_utc: str = "",
                             end_time_utc: str = "", modality: str = "",
                             limit: int = Query(default=20, ge=1, le=100)):
    repo = repository()
    if not hasattr(repo, "search_situation_units"):
        raise HTTPException(status_code=400, detail="态势立方体查询需要 SQLite 模式")
    try:
        return repo.search_situation_units(region_id, start_time_utc, end_time_utc, modality, limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/situation-units/{unit_id:path}")
def situation_unit_detail_endpoint(unit_id: str):
    repo = repository()
    if not hasattr(repo, "get_situation_unit"):
        raise HTTPException(status_code=400, detail="态势立方体查询需要 SQLite 模式")
    try:
        return repo.get_situation_unit(unit_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/remote-sensing/{scene_id}/preview")
def remote_sensing_preview(scene_id: str):
    try:
        repo = repository()
        scene = repo.get_remote_sensing_scene(scene_id)
        asset = next((item for item in scene["local_assets"]
                      if item["asset_key"].endswith("_thumbnail.jpg")), None)
        if asset is None:
            asset = next((item for item in scene["local_assets"]
                          if Path(item["asset_key"]).suffix.lower() in {".jpg", ".jpeg", ".png"}), None)
        if asset is None:
            raise ValueError("该场景没有本地预览图")
        path = (repo.root / asset["relative_path"]).resolve()
        if not path.is_relative_to(repo.root) or not path.is_file():
            raise ValueError("预览图不可用")
        media_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        return FileResponse(path, media_type=media_type, headers={"Cache-Control": "public, max-age=3600"})
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/ask")
def ask_endpoint(request: AskRequest):
    try:
        store = conversation_store()
        with store.session(str(request.conversation_id) if request.conversation_id else None) as conversation_id:
            recent, summary = store.context(conversation_id, Settings.from_env().memory_recent_turns)
            effective_question = request.question
            selected_source_id = str(request.source_id) if request.source_id else ""
            selected_region_id = request.region_id or ""
            # A conversation may be reopened after the user changes the analysis
            # dataset.  Keep follow-up memory inside the active scope so that an
            # earlier Houston/base-dataset turn cannot steer a question about a
            # newly selected research region (and vice versa).  Older summaries
            # do not carry reliable scope metadata, so scoped requests deliberately
            # ignore them.
            recent = [turn for turn in recent
                      if (turn.get("selected_region_id") or "") == selected_region_id
                      and (turn.get("selected_source_id") or "") == selected_source_id]
            if selected_region_id or selected_source_id:
                summary = ""
            selected_region = None
            if selected_region_id:
                regions = intake_agent().regions().get("regions", [])
                selected_region = next((item for item in regions
                                        if item.get("region_id") == selected_region_id), None)
                if selected_region is None:
                    raise ValueError("找不到所选研究区域数据集")

            # 1. 拦截问候与元意图 (无需调用多源海事工具链)
            if _is_greeting_or_meta(request.question):
                blackboard = store.get_cognitive_blackboard(conversation_id)
                target_hint = ""
                if blackboard and blackboard.active_target_mmsi:
                    target_hint = f"\n（当前会话关注的目标船舶为 MMSI {blackboard.active_target_mmsi}）"

                if selected_region_id:
                    region_name = selected_region["canonical_name"] if selected_region else selected_region_id
                    reply_text = (
                        f"您好！我是海事态势调查智能体。\n\n"
                        f"当前工作台分析范围已连接 **{region_name}** 多源数据集（`{selected_region_id}`）。"
                        f"{target_hint}\n\n"
                        "您可以向我提出以下类型的研判需求：\n"
                        "1. **船舶活动与航迹**：查询特定 MMSI 的逐报文 AIS 轨迹、走锚或异常停留；\n"
                        "2. **遥感与目标识别**：检索 Sentinel-1 雷达或光学卫星过境影像及船舶检测点；\n"
                        "3. **多源时空关联**：核查特定时空范围内的遥感—AIS 匹配关系与暗船排查；\n"
                        "4. **气象水文**：核实连续风场、风暴潮或恶劣海况对通航的影响。\n\n"
                        "请问有什么我可以协助您的？"
                    )
                elif selected_source_id:
                    reply_text = (
                        f"您好！我是海事态势调查智能体。\n\n"
                        f"当前工作台已锁定独立 AIS 数据范围（`{selected_source_id}`）。"
                        f"{target_hint}\n\n"
                        "您可以向我查询该数据源的逐报文航迹、日变化及小时活跃船舶指标。请问有什么我可以协助您的？"
                    )
                else:
                    reply_text = (
                        "您好！我是海事态势调查智能体。\n\n"
                        "当前工作台已连接基础海事数据集（Houston-Beryl 2024）。"
                        f"{target_hint}\n\n"
                        "您可以向我询问港口 AIS 活跃度、官方通航与运营事件、台风风场变化或雷达遥感影像。请问有什么我可以协助您的？"
                    )

                thought = (
                    "**1. 意图拆解与目标定位**\n"
                    f"- 接收用户指令：“{request.question.strip()}”\n"
                    "- 语义分析：识别为基础问候或智能体研判能力咨询。\n\n"
                    "**2. 环境状态感知与响应调度**\n"
                    "- 调度策略：无需触发底层海事数据库重型检索。\n"
                    f"- 环境感知：当前工作台分析范围已连接【{selected_region['canonical_name'] if selected_region else (selected_source_id or '基础海事数据集')}】。\n"
                    "- 输出当前海区核心研判能力与可验证实战案例提示。"
                )
                scoped_result = {
                    "answer": reply_text,
                    "evidence": [],
                    "warnings": [],
                    "tool_trace": [],
                    "thought": thought,
                    "reasoning_content": thought,
                    "selected_source_id": selected_source_id or None,
                    "selected_region_id": selected_region_id or None,
                }
                store.append(conversation_id, request.question, scoped_result)
                latest_snap = store.get_latest_reasoning_snapshot(conversation_id)
                reasoning_state = latest_snap.to_dict() if latest_snap else None
                return {**scoped_result, "conversation_id": conversation_id,
                        "reasoning_state": reasoning_state,
                        "thought": thought, "reasoning_content": thought}

            # 2. 拦截指引模板照抄或无实体指令
            if _is_template_echo(request.question):
                blackboard = store.get_cognitive_blackboard(conversation_id)
                suggested_mmsi = (blackboard.active_target_mmsi if blackboard and blackboard.active_target_mmsi else "258041000")
                reply_text = (
                    "您发送的是研判功能指引模板。请将占位符替换为具体实体，或直接选择以下可立即核验的真实案例提问：\n\n"
                    f"- **船舶轨迹核验**：请提供 9 位 MMSI（例如当前海区重点船舶 `258041000`），如：“查询 2024-01-06 MMSI {suggested_mmsi} 的逐报文 AIS 轨迹”；\n"
                    "- **遥感过境检索**：如：“检索 2024-01-06 当天的 Sentinel-1 卫星过境切片与云量”；\n"
                    "- **多源协同研判**：如：“核查 2024-01-06 MMSI 258041000 周围的雷达匹配关系”；\n"
                    "- **连续风场分析**：如：“分析 2024-01-06 现场海面的风浪与风速演变”。"
                )
                thought = (
                    "**1. 意图拆解与目标定位**\n"
                    f"- 接收用户指令：“{request.question.strip()}”\n"
                    "- 语义分析：检测到用户直接发送了未替换占位实体的指引模板文本。\n\n"
                    "**2. 纠偏推导与实体推荐**\n"
                    "- 决策策略：拦截空实体请求，避免发起无效全域扫描。\n"
                    f"- 实体推荐：从认知黑板推荐当前海区重点船舶 MMSI `{suggested_mmsi}`。\n"
                    "- 输出规范的真实案例提问范例。"
                )
                scoped_result = {
                    "answer": reply_text,
                    "evidence": [],
                    "warnings": ["请提供具体的船舶 MMSI 或时间范围以执行深度研判"],
                    "tool_trace": [],
                    "thought": thought,
                    "reasoning_content": thought,
                    "selected_source_id": selected_source_id or None,
                    "selected_region_id": selected_region_id or None,
                }
                store.append(conversation_id, request.question, scoped_result)
                latest_snap = store.get_latest_reasoning_snapshot(conversation_id)
                reasoning_state = latest_snap.to_dict() if latest_snap else None
                return {**scoped_result, "conversation_id": conversation_id,
                        "reasoning_state": reasoning_state,
                        "thought": thought, "reasoning_content": thought}

            # 3. 拦截无语义随机短输入 (如 1123, test, 海 等，避免盲目继承前序船舶)
            if _is_unrecognized_short_input(request.question):
                reply_text = (
                    f"未能识别具体的研判指令或目标（输入内容：`{request.question.strip()}`）。\n\n"
                    "- 如果您想查询船舶，请提供 **9 位 MMSI 码**（例如 `258041000`）；\n"
                    "- 如果您想按时间分析海区态势，请指明具体日期（例如 `2024-01-06`）；\n"
                    "- 如果您想查看当前区域的多源匹配概况，可直接询问“查询该区域当前的遥感与 AIS 匹配概况”。"
                )
                thought = (
                    "**1. 意图拆解与目标定位**\n"
                    f"- 接收用户指令：“{request.question.strip()}”\n"
                    "- 语义分析：输入内容属于无上下文的简短随机字符或单字。\n\n"
                    "**2. 研判安全隔离**\n"
                    "- 决策策略：避免无根据盲目继承历史实体导致态势串台与误判。\n"
                    "- 提示用户明确补充 9 位 MMSI 码或 UTC 日期等核心研判条件。"
                )
                scoped_result = {
                    "answer": reply_text,
                    "evidence": [],
                    "warnings": ["输入内容未包含有效的海事实体或分析意图"],
                    "tool_trace": [],
                    "thought": thought,
                    "reasoning_content": thought,
                    "selected_source_id": selected_source_id or None,
                    "selected_region_id": selected_region_id or None,
                }
                store.append(conversation_id, request.question, scoped_result)
                latest_snap = store.get_latest_reasoning_snapshot(conversation_id)
                reasoning_state = latest_snap.to_dict() if latest_snap else None
                return {**scoped_result, "conversation_id": conversation_id,
                        "reasoning_state": reasoning_state,
                        "thought": thought, "reasoning_content": thought}

            if selected_region_id:
                from .tools import set_active_region
                set_active_region(selected_region_id)
                catalog = repository().search_ingested_data(limit=100)
                region_sources = [item for item in catalog.get("sources", [])
                                  if item.get("region_id") == selected_region_id]
                modalities = sorted({item.get("kind", "unknown") for item in region_sources})
                effective_question = (
                    f"用户问题：{request.question}\n\n"
                    f"【当前限定分析范围】：研究区域级多源数据集【{selected_region['canonical_name']}】（区域ID={selected_region_id}）。\n"
                    f"已登记数据来源数={len(region_sources)}，数据模态类型={','.join(modalities) or '暂无'}。\n\n"
                    "【强行空间与数据隔离指令（极度重要）】：\n"
                    f"1. 回答必须 100% 仅聚焦于所选研究区域【{selected_region['canonical_name']}】，绝对不要混入、提及或罗列底层基础数据集（Houston-Beryl）的任何数据、船舶、分区、事件或指标！\n"
                    f"2. 严禁调用专属于 Houston 的全局概况工具 data_overview 和 fusion_overview！查询本区域数据必须使用 situation_cube_overview(region_id='{selected_region_id}')、search_situation_units(region_id='{selected_region_id}')、search_fusion_relations(region_id='{selected_region_id}')、search_ingested_data、search_ingested_vessel_tracks 或 get_ingested_vessel_track。\n"
                    f"3. 当用户询问“你有哪些数据”或“该区域有哪些数据”时，只能基于该区域已登记的来源和态势立方体陈述，绝对不能把 Houston-Beryl 基础数据集的数据资产（如 200 多万条数据、上万个 GIS 对象等）当作本区域的数据来回答！\n"
                    "4. 不得因为处于同一区域就假定各来源在时间上重合，也不得把规则分数解释为概率。"
                )
            elif selected_source_id:
                selected = repository().verify_ingested_source(selected_source_id)
                effective_question = (
                    f"用户问题：{request.question}\n\n"
                    "当前网页分析范围是一个独立 AIS 数据范围。"
                    f"范围ID={selected_source_id}，区域={selected['region_label']}，"
                    f"UTC范围={selected['first_utc']}至{selected['last_utc']}。"
                    "所有 AIS 数值、日变化和航迹问题必须使用 ingested_ais_daily_change、"
                    "query_data_hourly、search_ingested_vessel_tracks 或 get_ingested_vessel_track，"
                    "不得使用 Houston 基础数据集的 ais_daily_change、aggregate_hourly、"
                    "search_vessel_tracks 或 get_vessel_track。\n\n"
                    "回答须明确这是独立分析范围。")
            # 提取认知黑板、上一轮思维断点与自进化避坑经验
            blackboard = store.get_cognitive_blackboard(conversation_id)
            prev_snapshot = store.get_latest_reasoning_snapshot(conversation_id)
            relevant_critiques = store.get_relevant_critiques(query=request.question, limit=3)
            cognitive_block = assemble_cognitive_prompt_block(blackboard, prev_snapshot, relevant_critiques)

            try:
                result = ask(graph(), effective_question, history=recent, summary=summary,
                             cognitive_context=cognitive_block)
            except TypeError as exc:
                if "cognitive_context" in str(exc):
                    result = ask(graph(), effective_question, history=recent, summary=summary)
                else:
                    raise
            scoped_result = {**result,
                             "selected_source_id": selected_source_id or None,
                             "selected_region_id": selected_region_id or None}
            store.append(conversation_id, request.question, scoped_result)
            latest_snap = store.get_latest_reasoning_snapshot(conversation_id)
            reasoning_state = latest_snap.to_dict() if latest_snap else None
            return {**scoped_result, "conversation_id": conversation_id,
                    "reasoning_state": reasoning_state,
                    "thought": scoped_result.get("thought", ""),
                    "reasoning_content": scoped_result.get("thought", "")}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, FileNotFoundError, sqlite3.DatabaseError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        # Do not leak credentials, provider URLs, request headers or stack traces to clients.
        raise HTTPException(status_code=502, detail="模型或工具调用失败；请查看服务日志。") from exc


@app.post("/ask/stream")
def ask_question_stream(request: AskRequest):
    def event_stream():
        import json

        def format_sse(event: str, data: Any) -> str:
            return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

        try:
            store = conversation_store()
            with store.session(str(request.conversation_id) if request.conversation_id else None) as conversation_id:
                recent, summary = store.context(conversation_id, Settings.from_env().memory_recent_turns)
                effective_question = request.question
                selected_source_id = str(request.source_id) if request.source_id else ""
                selected_region_id = request.region_id or ""

                recent = [turn for turn in recent
                          if (turn.get("selected_region_id") or "") == selected_region_id
                          and (turn.get("selected_source_id") or "") == selected_source_id]
                if selected_region_id or selected_source_id:
                    summary = ""
                selected_region = None
                if selected_region_id:
                    regions = intake_agent().regions().get("regions", [])
                    selected_region = next((item for item in regions
                                            if item.get("region_id") == selected_region_id), None)
                    if selected_region is None:
                        yield format_sse("error", {"detail": "找不到所选研究区域数据集", "status_code": 404})
                        return

                # 1. 拦截问候与元意图
                if _is_greeting_or_meta(request.question):
                    blackboard = store.get_cognitive_blackboard(conversation_id)
                    target_hint = ""
                    if blackboard and blackboard.active_target_mmsi:
                        target_hint = f"\n（当前会话关注的目标船舶为 MMSI {blackboard.active_target_mmsi}）"

                    if selected_region_id:
                        region_name = selected_region["canonical_name"] if selected_region else selected_region_id
                        reply_text = (
                            f"您好！我是海事态势调查智能体。\n\n"
                            f"当前工作台分析范围已连接 **{region_name}** 多源数据集（`{selected_region_id}`）。"
                            f"{target_hint}\n\n"
                            "您可以向我提出以下类型的研判需求：\n"
                            "1. **船舶活动与航迹**：查询特定 MMSI 的逐报文 AIS 轨迹、走锚或异常停留；\n"
                            "2. **遥感与目标识别**：检索 Sentinel-1 雷达或光学卫星过境影像及船舶检测点；\n"
                            "3. **多源时空关联**：核查特定时空范围内的遥感—AIS 匹配关系与暗船排查；\n"
                            "4. **气象水文**：核实连续风场、风暴潮或恶劣海况对通航的影响。\n\n"
                            "请问有什么我可以协助您的？"
                        )
                    elif selected_source_id:
                        reply_text = (
                            f"您好！我是海事态势调查智能体。\n\n"
                            f"当前工作台已锁定独立 AIS 数据范围（`{selected_source_id}`）。"
                            f"{target_hint}\n\n"
                            "您可以向我查询该数据源的逐报文航迹、日变化及小时活跃船舶指标。请问有什么我可以协助您的？"
                        )
                    else:
                        reply_text = (
                            "您好！我是海事态势调查智能体。\n\n"
                            "当前工作台已连接基础海事数据集（Houston-Beryl 2024）。"
                            f"{target_hint}\n\n"
                            "您可以向我询问港口 AIS 活跃度、官方通航与运营事件、台风风场变化或雷达遥感影像。请问有什么我可以协助您的？"
                        )

                    thought = (
                        "**1. 意图拆解与目标定位**\n"
                        f"- 接收用户指令：“{request.question.strip()}”\n"
                        "- 语义分析：识别为基础问候或智能体研判能力咨询。\n\n"
                        "**2. 环境状态感知与响应调度**\n"
                        "- 调度策略：无需触发底层海事数据库重型检索。\n"
                        f"- 环境感知：当前工作台分析范围已连接【{selected_region['canonical_name'] if selected_region else (selected_source_id or '基础海事数据集')}】。\n"
                        "- 输出当前海区核心研判能力与可验证实战案例提示。"
                    )
                    scoped_result = {
                        "answer": reply_text,
                        "evidence": [],
                        "warnings": [],
                        "tool_trace": [],
                        "thought": thought,
                        "reasoning_content": thought,
                        "selected_source_id": selected_source_id or None,
                        "selected_region_id": selected_region_id or None,
                    }
                    yield format_sse("routing", {"stage": "routing", "roles": ["meta"], "role_names": ["工作台问答助手"], "subquestions": {}})
                    yield format_sse("thought", {"stage": "meta", "thought": thought})
                    for chunk in _chunk_text(reply_text, 10):
                        yield format_sse("token", {"stage": "synthesis", "delta": chunk})
                    store.append(conversation_id, request.question, scoped_result)
                    latest_snap = store.get_latest_reasoning_snapshot(conversation_id)
                    reasoning_state = latest_snap.to_dict() if latest_snap else None
                    yield format_sse("done", {
                        **scoped_result,
                        "conversation_id": conversation_id,
                        "reasoning_state": reasoning_state,
                        "thought": thought,
                        "reasoning_content": thought,
                    })
                    return

                # 2. 拦截指引模板照抄或无实体指令
                if _is_template_echo(request.question):
                    blackboard = store.get_cognitive_blackboard(conversation_id)
                    suggested_mmsi = (blackboard.active_target_mmsi if blackboard and blackboard.active_target_mmsi else "258041000")
                    reply_text = (
                        "您发送的是研判功能指引模板。请将占位符替换为具体实体，或直接选择以下可立即核验的真实案例提问：\n\n"
                        f"- **船舶轨迹核验**：请提供 9 位 MMSI（例如当前海区重点船舶 `{suggested_mmsi}`），如：“查询 2024-01-06 MMSI {suggested_mmsi} 的逐报文 AIS 轨迹”；\n"
                        "- **遥感过境检索**：如：“检索 2024-01-06 当天的 Sentinel-1 卫星过境切片与云量”；\n"
                        "- **多源协同研判**：如：“核查 2024-01-06 MMSI 258041000 周围的雷达匹配关系”；\n"
                        "- **连续风场分析**：如：“分析 2024-01-06 现场海面的风浪与风速演变”。"
                    )
                    thought = (
                        "**1. 意图拆解与目标定位**\n"
                        f"- 接收用户指令：“{request.question.strip()}”\n"
                        "- 语义分析：检测到用户直接发送了未替换占位实体的指引模板文本。\n\n"
                        "**2. 纠偏推导与实体推荐**\n"
                        "- 决策策略：拦截空实体请求，避免发起无效全域扫描。\n"
                        f"- 实体推荐：从认知黑板推荐当前海区重点船舶 MMSI `{suggested_mmsi}`。\n"
                        "- 输出规范的真实案例提问范例。"
                    )
                    scoped_result = {
                        "answer": reply_text,
                        "evidence": [],
                        "warnings": ["请提供具体的船舶 MMSI 或时间范围以执行深度研判"],
                        "tool_trace": [],
                        "thought": thought,
                        "reasoning_content": thought,
                        "selected_source_id": selected_source_id or None,
                        "selected_region_id": selected_region_id or None,
                    }
                    yield format_sse("routing", {"stage": "routing", "roles": ["meta"], "role_names": ["工作台问答助手"], "subquestions": {}})
                    yield format_sse("thought", {"stage": "meta", "thought": thought})
                    for chunk in _chunk_text(reply_text, 10):
                        yield format_sse("token", {"stage": "synthesis", "delta": chunk})
                    store.append(conversation_id, request.question, scoped_result)
                    latest_snap = store.get_latest_reasoning_snapshot(conversation_id)
                    reasoning_state = latest_snap.to_dict() if latest_snap else None
                    yield format_sse("done", {
                        **scoped_result,
                        "conversation_id": conversation_id,
                        "reasoning_state": reasoning_state,
                        "thought": thought,
                        "reasoning_content": thought,
                    })
                    return

                # 3. 拦截无语义随机短输入
                if _is_unrecognized_short_input(request.question):
                    reply_text = (
                        f"未能识别具体的研判指令或目标（输入内容：`{request.question.strip()}`）。\n\n"
                        "- 如果您想查询船舶，请提供 **9 位 MMSI 码**（例如 `258041000`）；\n"
                        "- 如果您想按时间分析海区态势，请指明具体日期（例如 `2024-01-06`）；\n"
                        "- 如果您想查看当前区域的多源匹配概况，可直接询问“查询该区域当前的遥感与 AIS 匹配概况”。"
                    )
                    thought = (
                        "**1. 意图拆解与目标定位**\n"
                        f"- 接收用户指令：“{request.question.strip()}”\n"
                        "- 语义分析：输入内容属于无上下文的简短随机字符或单字。\n\n"
                        "**2. 研判安全隔离**\n"
                        "- 决策策略：避免无根据盲目继承历史实体导致态势串台与误判。\n"
                        "- 提示用户明确补充 9 位 MMSI 码或 UTC 日期等核心研判条件。"
                    )
                    scoped_result = {
                        "answer": reply_text,
                        "evidence": [],
                        "warnings": ["输入内容未包含有效的海事实体或分析意图"],
                        "tool_trace": [],
                        "thought": thought,
                        "reasoning_content": thought,
                        "selected_source_id": selected_source_id or None,
                        "selected_region_id": selected_region_id or None,
                    }
                    yield format_sse("routing", {"stage": "routing", "roles": ["meta"], "role_names": ["工作台问答助手"], "subquestions": {}})
                    yield format_sse("thought", {"stage": "meta", "thought": thought})
                    for chunk in _chunk_text(reply_text, 10):
                        yield format_sse("token", {"stage": "synthesis", "delta": chunk})
                    store.append(conversation_id, request.question, scoped_result)
                    latest_snap = store.get_latest_reasoning_snapshot(conversation_id)
                    reasoning_state = latest_snap.to_dict() if latest_snap else None
                    yield format_sse("done", {
                        **scoped_result,
                        "conversation_id": conversation_id,
                        "reasoning_state": reasoning_state,
                        "thought": thought,
                        "reasoning_content": thought,
                    })
                    return

                if selected_region_id:
                    from .tools import set_active_region
                    set_active_region(selected_region_id)
                    catalog = repository().search_ingested_data(limit=100)
                    region_sources = [item for item in catalog.get("sources", [])
                                      if item.get("region_id") == selected_region_id]
                    modalities = sorted({item.get("kind", "unknown") for item in region_sources})
                    effective_question = (
                        f"用户问题：{request.question}\n\n"
                        f"【当前限定分析范围】：研究区域级多源数据集【{selected_region['canonical_name']}】（区域ID={selected_region_id}）。\n"
                        f"已登记数据来源数={len(region_sources)}，数据模态类型={','.join(modalities) or '暂无'}。\n\n"
                        "【强行空间与数据隔离指令（极度重要）】：\n"
                        f"1. 回答必须 100% 仅聚焦于所选研究区域【{selected_region['canonical_name']}】，绝对不要混入、提及或罗列底层基础数据集（Houston-Beryl）的任何数据、船舶、分区、事件或指标！\n"
                        f"2. 严禁调用专属于 Houston 的全局概况工具 data_overview 和 fusion_overview！查询本区域数据必须使用 situation_cube_overview(region_id='{selected_region_id}')、search_situation_units(region_id='{selected_region_id}')、search_fusion_relations(region_id='{selected_region_id}')、search_ingested_data、search_ingested_vessel_tracks 或 get_ingested_vessel_track。\n"
                        f"3. 当用户询问“你有哪些数据”或“该区域有哪些数据”时，只能基于该区域已登记的来源和态势立方体陈述，绝对不能把 Houston-Beryl 基础数据集的数据资产（如 200 多万条数据、上万个 GIS 对象等）当作本区域的数据来回答！\n"
                        "4. 不得因为处于同一区域就假定各来源在时间上重合，也不得把规则分数解释为概率。"
                    )
                elif selected_source_id:
                    selected = repository().verify_ingested_source(selected_source_id)
                    effective_question = (
                        f"用户问题：{request.question}\n\n"
                        "当前网页分析范围是一个独立 AIS 数据范围。"
                        f"范围ID={selected_source_id}，区域={selected['region_label']}，"
                        f"UTC范围={selected['first_utc']}至{selected['last_utc']}。"
                        "所有 AIS 数值、日变化和航迹问题必须使用 ingested_ais_daily_change、"
                        "query_data_hourly、search_ingested_vessel_tracks 或 get_ingested_vessel_track，"
                        "不得使用 Houston 基础数据集的 ais_daily_change、aggregate_hourly、"
                        "search_vessel_tracks 或 get_vessel_track。\n\n"
                        "回答须明确这是独立分析范围。")
                else:
                    effective_question = request.question

                # 提取认知黑板与上下文
                blackboard = store.get_cognitive_blackboard(conversation_id)
                prev_snapshot = store.get_latest_reasoning_snapshot(conversation_id)
                relevant_critiques = store.get_relevant_critiques(query=request.question, limit=3)
                cognitive_block = assemble_cognitive_prompt_block(blackboard, prev_snapshot, relevant_critiques)

                final_result = None
                try:
                    for event_obj in stream_ask(graph(), effective_question, history=recent, summary=summary,
                                                cognitive_context=cognitive_block):
                        ev_name = event_obj.get("event", "message")
                        ev_data = event_obj.get("data", {})
                        if ev_name == "done":
                            final_result = ev_data
                        else:
                            yield format_sse(ev_name, ev_data)
                except TypeError as exc:
                    if "cognitive_context" in str(exc):
                        for event_obj in stream_ask(graph(), effective_question, history=recent, summary=summary):
                            ev_name = event_obj.get("event", "message")
                            ev_data = event_obj.get("data", {})
                            if ev_name == "done":
                                final_result = ev_data
                            else:
                                yield format_sse(ev_name, ev_data)
                    else:
                        raise

                if final_result:
                    scoped_result = {
                        **final_result,
                        "selected_source_id": selected_source_id or None,
                        "selected_region_id": selected_region_id or None,
                    }
                    store.append(conversation_id, request.question, scoped_result)
                    latest_snap = store.get_latest_reasoning_snapshot(conversation_id)
                    reasoning_state = latest_snap.to_dict() if latest_snap else None
                    yield format_sse("done", {
                        **scoped_result,
                        "conversation_id": conversation_id,
                        "reasoning_state": reasoning_state,
                        "thought": scoped_result.get("thought", ""),
                        "reasoning_content": scoped_result.get("thought", ""),
                    })
        except KeyError as exc:
            yield format_sse("error", {"detail": str(exc), "status_code": 404})
        except (ValueError, FileNotFoundError, sqlite3.DatabaseError) as exc:
            yield format_sse("error", {"detail": str(exc), "status_code": 503})
        except Exception as exc:
            import traceback
            traceback.print_exc()
    headers = {
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)


@app.get("/api/memory/blackboard/{conversation_id}")
def api_memory_blackboard(conversation_id: UUID):
    try:
        store = conversation_store()
        cid_text = str(conversation_id)
        if not store.exists(cid_text):
            raise HTTPException(status_code=404, detail="会话不存在")
        bb = store.get_cognitive_blackboard(cid_text)
        snapshots = store.get_reasoning_snapshots(cid_text, limit=10)
        return {"blackboard": bb.to_dict(), "snapshots": [s.to_dict() for s in snapshots]}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/memory/critiques")
def api_memory_critiques(query: str = "", limit: int = 20):
    try:
        store = conversation_store()
        critiques = store.get_relevant_critiques(query=query, limit=limit)
        return {"critiques": [c.to_dict() for c in critiques]}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/memory/critiques")
def api_create_critique(body: CreateCritiqueRequest, request: Request):
    _local_origin(request)
    try:
        import datetime
        store = conversation_store()
        cid = "CRT-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d%H%M%S")
        critique = EvolvedCritique(
            critique_id=cid,
            domain=body.domain,
            trigger_pattern=body.trigger_pattern,
            failed_path=body.failed_path or "用户纠偏规约",
            corrected_lesson=body.corrected_lesson,
            confidence=body.confidence,
        )
        store.save_evolved_critique(critique)
        return {"status": "ok", "critique": critique.to_dict()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.delete("/api/memory/critiques/{critique_id}")
def api_delete_critique(critique_id: str, request: Request):
    _local_origin(request)
    try:
        store = conversation_store()
        deleted = store.delete_evolved_critique(critique_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="反思条目不存在")
        return {"status": "ok", "deleted_critique_id": critique_id}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/memory/critiques/{critique_id}/feedback")
def api_critique_feedback(critique_id: str, body: CritiqueFeedbackRequest, request: Request):
    _local_origin(request)
    try:
        store = conversation_store()
        delta = 0.2 if body.action == "upvote" else -0.3
        new_conf = store.adjust_critique_confidence(critique_id, delta)
        if new_conf is None:
            raise HTTPException(status_code=404, detail="反思条目不存在")
        return {"status": "ok", "critique_id": critique_id, "confidence": new_conf}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/conversations")
def list_conversations(limit: int = Query(default=40, ge=1, le=100),
                       offset: int = Query(default=0, ge=0)):
    try:
        items, total = conversation_store().list_conversations(limit, offset)
        return {"conversations": items, "total": total, "limit": limit, "offset": offset}
    except sqlite3.DatabaseError as exc:
        raise HTTPException(status_code=503, detail="会话数据库不可用") from exc


@app.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: UUID):
    try:
        store = conversation_store()
        id_text = str(conversation_id)
        if not store.exists(id_text):
            raise HTTPException(status_code=404, detail="会话不存在")
        return {"conversation_id": id_text, "turns": store.turns(id_text)}
    except sqlite3.DatabaseError as exc:
        raise HTTPException(status_code=503, detail="会话数据库不可用") from exc


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(Path(__file__).with_name("ui.html").read_text(encoding="utf-8"),
                        headers={"Cache-Control": "no-store"})


@app.get("/tracks-map", response_class=HTMLResponse)
def tracks_map():
    return HTMLResponse(Path(__file__).with_name("tracks_map.html").read_text(encoding="utf-8"),
                        headers={"Cache-Control": "no-store"})


@app.get("/wind-map", response_class=HTMLResponse)
def wind_map_page():
    return HTMLResponse(Path(__file__).with_name("wind_map.html").read_text(encoding="utf-8"),
                        headers={"Cache-Control": "no-store"})


@app.get("/api/wind-field-grid")
def api_wind_field_grid(valid_time_utc: str = Query(default="2024-07-08T12:00:00Z"),
                        cols: int = Query(default=16, ge=4, le=32),
                        rows: int = Query(default=16, ge=4, le=32),
                        region_id: str = ""):
    try:
        repo = repository()
        if not hasattr(repo, "query_wind_field"):
            raise HTTPException(status_code=503, detail="风场查询需要 SQLite 模式")
        return repo.query_wind_field(region_id=region_id, valid_time_utc=valid_time_utc,
                                    grid_cols=cols, grid_rows=rows)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/vessels-snapshot")
def api_vessels_snapshot(valid_time_utc: str = Query(default="2024-07-08T12:00:00Z"),
                         limit: int = Query(default=800, ge=10, le=1500),
                         region_id: str = ""):
    try:
        repo = repository()
        if not hasattr(repo, "query_vessels_snapshot"):
            raise HTTPException(status_code=503, detail="船舶态势查询需要 SQLite 模式")
        return repo.query_vessels_snapshot(valid_time_utc=valid_time_utc, limit=limit, region_id=region_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/port-facilities")
def api_port_facilities(limit: int = Query(default=200, ge=10, le=500),
                        region_id: str = ""):
    try:
        repo = repository()
        if not hasattr(repo, "query_port_facilities"):
            raise HTTPException(status_code=503, detail="港口设施查询需要 SQLite 模式")
        return repo.query_port_facilities(limit=limit, region_id=region_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/wind-timeline")
def api_wind_timeline(date_utc: str = Query(default="2024-07-08")):
    times = [
        {
            "time_utc": f"{date_utc}T06:00:00Z", "label": "06:00 逼近", "category": "approaching",
            "est_max_wind": 14.8, "est_min_pressure": 1002.5, "est_max_wave": 2.4,
            "typhoon_center": [-95.80, 28.60], "gale_radius_km": 110, "storm_radius_km": 45,
            "stage_desc": "台风外围逼近 · 墨西哥湾外海浪高增强至 2.4m"
        },
        {
            "time_utc": f"{date_utc}T08:00:00Z", "label": "08:00 增强", "category": "strengthening",
            "est_max_wind": 18.2, "est_min_pressure": 997.0, "est_max_wave": 2.9,
            "typhoon_center": [-95.55, 28.85], "gale_radius_km": 125, "storm_radius_km": 55,
            "stage_desc": "风雨急剧增强 · 加尔维斯敦近海进入 8 级大风圈"
        },
        {
            "time_utc": f"{date_utc}T10:00:00Z", "label": "10:00 狂风", "category": "storm",
            "est_max_wind": 22.4, "est_min_pressure": 991.5, "est_max_wave": 3.6,
            "typhoon_center": [-95.30, 29.10], "gale_radius_km": 140, "storm_radius_km": 65,
            "stage_desc": "狂风倒灌登陆前夕 · 10 级暴风圈触及航道入口"
        },
        {
            "time_utc": f"{date_utc}T12:00:00Z", "label": "12:00 登陆高峰", "category": "peak",
            "est_max_wind": 24.5, "est_min_pressure": 988.0, "est_max_wave": 4.1,
            "typhoon_center": [-95.10, 29.35], "gale_radius_km": 150, "storm_radius_km": 75,
            "stage_desc": "台风眼登陆高峰 · 极大风速 24.5 m/s · 气压降至 988 hPa"
        },
        {
            "time_utc": f"{date_utc}T14:00:00Z", "label": "14:00 后部", "category": "post_storm",
            "est_max_wind": 21.6, "est_min_pressure": 992.0, "est_max_wave": 3.2,
            "typhoon_center": [-94.90, 29.65], "gale_radius_km": 135, "storm_radius_km": 55,
            "stage_desc": "后部强风穿过航道 · 偏南狂风倒推港湾水体"
        },
        {
            "time_utc": f"{date_utc}T16:00:00Z", "label": "16:00 渐衰", "category": "weakening",
            "est_max_wind": 17.5, "est_min_pressure": 996.5, "est_max_wave": 2.5,
            "typhoon_center": [-94.75, 29.95], "gale_radius_km": 115, "storm_radius_km": 35,
            "stage_desc": "台风中心深入内陆 · 风力减弱至 17 m/s"
        },
        {
            "time_utc": f"{date_utc}T18:00:00Z", "label": "18:00 缓和", "category": "recovering",
            "est_max_wind": 13.2, "est_min_pressure": 1001.0, "est_max_wave": 1.8,
            "typhoon_center": [-94.60, 30.25], "gale_radius_km": 90, "storm_radius_km": 20,
            "stage_desc": "过境气象渐趋平稳 · 港区开始组织排查清障"
        },
    ]
    return {"date_utc": date_utc, "steps": times}


@app.get("/api/timeline-histogram")
def api_timeline_histogram(start_utc: str = Query(default="2024-07-06T00:00:00Z"),
                           end_utc: str = Query(default="2024-07-10T23:00:00Z"),
                           region_id: str = ""):
    try:
        repo = repository()
        if not hasattr(repo, "query_timeline_histogram"):
            raise HTTPException(status_code=503, detail="时间轴直方图需要 SQLite 模式")
        return repo.query_timeline_histogram(start_utc=start_utc, end_utc=end_utc, region_id=region_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/remote-sensing-overlays")
def api_remote_sensing_overlays(region_id: str = ""):
    try:
        repo = repository()
        if not hasattr(repo, "query_remote_sensing_overlays"):
            raise HTTPException(status_code=503, detail="遥感图层要素查询需要 SQLite 模式")
        return repo.query_remote_sensing_overlays(region_id=region_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/situation-report")
def api_situation_report(region_id: str = ""):
    try:
        repo = repository()
        if not hasattr(repo, "db_path"):
            raise HTTPException(status_code=503, detail="态势简报生成需要 SQLite 模式")
        from .report_generator import generate_situation_briefing
        return generate_situation_briefing(repo=repo, region_id=region_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc



@app.get("/assets/wind-map.css")
def wind_map_styles():
    return FileResponse(Path(__file__).with_name("wind_map.css"), media_type="text/css",
                        headers={"Cache-Control": "no-cache"})


@app.get("/assets/wind-map.js")
def wind_map_script():
    return FileResponse(Path(__file__).with_name("wind_map.js"), media_type="application/javascript",
                        headers={"Cache-Control": "no-cache"})


@app.get("/assets/wind-particles.js")
def wind_particles_script():
    return FileResponse(Path(__file__).with_name("wind_particles.js"), media_type="application/javascript",
                        headers={"Cache-Control": "no-cache"})


@app.get("/assets/tracks-map.css")
def tracks_map_styles():
    return FileResponse(Path(__file__).with_name("tracks_map.css"), media_type="text/css",
                        headers={"Cache-Control": "no-cache"})


@app.get("/assets/tracks-map.js")
def tracks_map_script():
    return FileResponse(Path(__file__).with_name("tracks_map.js"), media_type="application/javascript",
                        headers={"Cache-Control": "no-cache"})


@app.get("/assets/leaflet-1.9.4.js")
def leaflet_script():
    return FileResponse(Path(__file__).with_name("leaflet-1.9.4.js"), media_type="application/javascript",
                        headers={"Cache-Control": "public, max-age=31536000, immutable"})


@app.get("/assets/leaflet-1.9.4.css")
def leaflet_styles():
    return FileResponse(Path(__file__).with_name("leaflet-1.9.4.css"), media_type="text/css",
                        headers={"Cache-Control": "public, max-age=31536000, immutable"})


@app.get("/assets/answer-renderer.js")
def answer_renderer():
    return FileResponse(Path(__file__).with_name("answer_renderer.js"), media_type="application/javascript",
                        headers={"Cache-Control": "no-cache"})


@app.get("/assets/ui.css")
def ui_styles():
    return FileResponse(Path(__file__).with_name("ui.css"), media_type="text/css",
                        headers={"Cache-Control": "no-cache"})


@app.get("/assets/ui.js")
def ui_script():
    return FileResponse(Path(__file__).with_name("ui.js"), media_type="application/javascript",
                        headers={"Cache-Control": "no-cache"})


@app.get("/assets/inline-visualizations.js")
def inline_visualizations_script():
    return FileResponse(Path(__file__).with_name("inline_visualizations.js"),
                        media_type="application/javascript", headers={"Cache-Control": "no-cache"})


def main() -> None:
    import os
    import uvicorn

    host = os.environ.get("MARITIME_HOST", "0.0.0.0")
    port = int(os.environ.get("MARITIME_PORT", "8000"))
    uvicorn.run("maritime_agent.api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
