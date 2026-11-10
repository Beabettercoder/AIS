"""Bind on-demand diagnostics to the observations stored by their scan."""
import hashlib
import json


def evidence_signature(motion_observations, neighbor_frames):
    if not isinstance(motion_observations, dict) or not isinstance(neighbor_frames, list):
        return None
    value={'motion_observations':motion_observations,'neighbor_frames':neighbor_frames}
    try:
        canonical=json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
    except (TypeError,ValueError):
        return None
    return hashlib.sha256(canonical.encode()).hexdigest()


def unavailable_diagnostics(item, reason):
    return {'pg_probability':[],'raw_probability':[],
            'gate_weights':item.get('gate_weights',[]),'time_seconds':item.get('time_seconds',[]),
            'method':reason,'notices':[reason],'source':'saved_observations_only'}
