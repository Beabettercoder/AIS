"""Development-only LOO feature construction. Not a completed experiment."""
import numpy as np

def build_field_inputs(core, field, arm="F1"):
    """Return physical arrays. Caller must fit normalization using training only."""
    for key in ("window_id", "mmsi", "mask", "time_s"):
        if not np.array_equal(core[key], field[key]):
            raise ValueError("Mismatched paired inputs: " + key)
    if arm not in ("F0", "F1"):
        raise ValueError("Invalid field variant")
    mask = np.asarray(core["mask"], dtype=bool)
    fit = np.asarray(field["field_common_valid"], dtype=bool) & mask
    x = np.asarray(field["base__" + arm], dtype=np.float64)
    gated = np.asarray(field["gated7_legacy"], dtype=np.float64)
    # Use target velocity after causal gating, not the raw-target residual.
    residual = gated[..., 2:4] - x[..., 7:9]
    d = np.asarray(field[arm + "__divergence_per_s"], dtype=np.float64)
    c = np.asarray(field[arm + "__curl_per_s"], dtype=np.float64)
    n = np.asarray(field["plain_member_count"], dtype=np.float64)
    rho = np.where(fit, np.clip(n / 4.0, 0.0, 1.0), 0.0)
    reference = np.where(fit[..., None], x[..., 7:9], 0.0)
    feat = np.where(fit[..., None], np.concatenate((residual, d[..., None], c[..., None]), axis=-1), 0.0)
    speed = np.linalg.norm(reference, axis=-1)
    times = np.asarray(core["time_s"], dtype=np.float64)
    diff = np.zeros_like(speed)
    dt = times[:, 1:] - times[:, :-1]
    ok = fit[:, 1:] & fit[:, :-1] & (dt > 0)
    np.divide(np.abs(speed[:, 1:] - speed[:, :-1]), dt, out=diff[:, 1:], where=ok)
    state = np.stack((speed, diff, np.where(fit, d, 0.0), np.where(fit, c, 0.0)), axis=-1)
    for name, a in (("target", gated), ("field", feat), ("rho", rho), ("state", state)):
        if not np.isfinite(a).all():
            raise ValueError("Nonfinite " + name)
    return {"self": gated.astype(np.float32), "field": feat.astype(np.float32),
            "rho": rho.astype(np.float32), "state": state.astype(np.float32), "valid": fit}
