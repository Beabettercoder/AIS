"""Strict, column-aware datetime parsing for heterogeneous intake files."""

from __future__ import annotations

import re
from datetime import date, datetime, timezone
from typing import Any, Iterable


_SEPARATED_DATE = re.compile(
    r"^\s*(\d{1,4})\s*[/.-]\s*(\d{1,2})\s*[/.-]\s*(\d{1,4})(?:\D|$)"
)
_INTEGER = re.compile(r"^[+-]?\d+$")


def infer_date_order(values: Iterable[Any]) -> dict[str, Any]:
    """Infer YMD/DMY/MDY from unambiguous values across one source column."""
    evidence = {"YMD": 0, "DMY": 0, "MDY": 0}
    separated = 0
    for value in values:
        text = str(value or "").strip()
        match = _SEPARATED_DATE.match(text)
        if not match:
            continue
        separated += 1
        first, second, third = (int(part) for part in match.groups())
        if first >= 1000:
            evidence["YMD"] += 1
        elif third >= 1000:
            if first > 12 >= second:
                evidence["DMY"] += 1
            elif second > 12 >= first:
                evidence["MDY"] += 1
    supported = [order for order, count in evidence.items() if count]
    if len(supported) == 1:
        return {"date_order": supported[0], "status": "inferred",
                "evidence_counts": evidence, "separated_samples": separated}
    if len(supported) > 1:
        return {"date_order": None, "status": "conflicting",
                "evidence_counts": evidence, "separated_samples": separated}
    return {"date_order": None, "status": "ambiguous" if separated else "not_applicable",
            "evidence_counts": evidence, "separated_samples": separated}


def _fast_parse_separated(text: str, date_order: str | None = None) -> datetime | None:
    parts = text.split()
    if len(parts) != 2:
        return None
    d_part, t_part = parts[0], parts[1]
    for sep in ("/", "-", "."):
        if sep in d_part:
            d_tokens = d_part.split(sep)
            break
    else:
        return None
    if len(d_tokens) != 3:
        return None
    try:
        n1, n2, n3 = int(d_tokens[0]), int(d_tokens[1]), int(d_tokens[2])
    except ValueError:
        return None
    if n1 >= 1000:
        year, month, day = n1, n2, n3
    elif n3 >= 1000:
        year = n3
        if date_order == "DMY" or (n1 > 12 >= n2):
            day, month = n1, n2
        elif date_order == "MDY" or (n2 > 12 >= n1):
            month, day = n1, n2
        else:
            month, day = n1, n2
    else:
        return None
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None
    t_tokens = t_part.split(":")
    if len(t_tokens) < 2:
        return None
    try:
        hour = int(t_tokens[0])
        minute = int(t_tokens[1])
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            return None
        if len(t_tokens) >= 3:
            s_str = t_tokens[2]
            if "." in s_str:
                s_base, s_micro = s_str.split(".", 1)
                second = int(s_base)
                microsecond = int(s_micro[:6].ljust(6, "0"))
            else:
                second = int(s_str)
                microsecond = 0
        else:
            second = 0
            microsecond = 0
        if not (0 <= second <= 59):
            return None
        return datetime(year, month, day, hour, minute, second, microsecond)
    except Exception:
        return None


def _candidate(value: Any, date_order: str | None = None) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    text = str(value or "").strip()
    if not text:
        raise ValueError("时间为空")
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        pass

    fast_dt = _fast_parse_separated(text, date_order)
    if fast_dt is not None:
        if 1900 <= fast_dt.year <= 2100:
            return fast_dt
        raise ValueError("时间年份超出 1900 至 2100")

    if _INTEGER.fullmatch(text) and len(text.lstrip("+-")) in {10, 13}:
        number = int(text)
        seconds = number / 1000 if len(text.lstrip("+-")) == 13 else number
        try:
            parsed = datetime.fromtimestamp(seconds, timezone.utc)
        except (OverflowError, OSError, ValueError) as exc:
            raise ValueError("Unix 时间戳超出有效范围") from exc
        if 1900 <= parsed.year <= 2100:
            return parsed
        raise ValueError("Unix 时间戳年份超出 1900 至 2100")

    try:
        import dateparser
    except ImportError as exc:
        raise ValueError("服务器缺少通用时间解析组件 dateparser") from exc
    settings: dict[str, Any] = {
        "STRICT_PARSING": True,
        "REQUIRE_PARTS": ["year", "month", "day"],
        "PARSERS": ["custom-formats", "absolute-time"],
    }
    if date_order:
        settings.update({"DATE_ORDER": date_order, "PREFER_LOCALE_DATE_ORDER": False})
    normalized_text = (text.replace("年", "-").replace("月", "-").replace("日", " ")
                       .replace("时", ":").replace("分", ":").replace("秒", "").strip())
    parsed = dateparser.parse(normalized_text, settings=settings)
    if parsed is None:
        raise ValueError("时间不能解析为完整日期和时间")
    if not 1900 <= parsed.year <= 2100:
        raise ValueError("时间年份超出 1900 至 2100")
    return parsed


def parse_utc(value: Any, assume_utc: bool, date_order: str | None = None) -> str:
    parsed = _candidate(value, date_order)
    if parsed.tzinfo is None:
        if not assume_utc:
            raise ValueError("原始时间无时区；必须明确选择按 UTC 解释")
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def is_datetime_like(value: Any) -> bool:
    try:
        _candidate(value)
        return True
    except ValueError:
        return False
