"""Causal forecasting features. Units: metres, seconds, m/s; no future labels here."""
from __future__ import annotations
import hashlib
import numpy as np

KNOT = 1852.0 / 3600.0
SELF_NAMES = ['x_from_first_m', 'y_from_first_m', 'vx_mps', 'vy_mps',
              'speed_mps', 'sin_cog', 'cos_cog', 'elapsed_s', 'dt_s']
FIELD_NAMES = ['background_vx', 'background_vy', 'residual_vx', 'residual_vy',
               'divergence_per_s', 'curl_per_s', 'support', 'field_mode']
VIEWS = ['base', 'target_jump', 'target_drift', 'target_speed', 'neighbor_stress']

def seeded(text):
    return np.random.default_rng(int(hashlib.sha256(str(text).encode()).hexdigest()[:16], 16))

def project(lon, lat, bbox):
    lon0, lat0 = (bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2
    return np.stack([(np.asarray(lon)-lon0)*111195*np.cos(np.deg2rad(lat0)),
                     (np.asarray(lat)-lat0)*111195], axis=-1)

def vectors(sog, cog):
    angle = np.deg2rad(cog)
    return np.asarray(sog)[..., None]*KNOT*np.stack([np.sin(angle), np.cos(angle)], -1)

def compatibility(t, p, v):
    """Current step uses only current and immediately preceding observation."""
    w = np.ones(len(t)); valid = np.zeros(len(t), bool); err = np.zeros(len(t))
    dt = np.diff(t)
    ok = (dt > 0) & (dt <= 300) & np.isfinite(np.diff(p, axis=0)).all(1)
    ok &= np.isfinite(v[1:]).all(1) & np.isfinite(v[:-1]).all(1)
    e = np.linalg.norm(np.diff(p, axis=0)/np.maximum(dt[:, None], 1e-9)
                       - .5*(v[1:]+v[:-1]), axis=1)
    valid[1:] = ok; err[1:] = np.where(ok, e, 0)
    w[1:] = np.where(ok, np.exp(-np.maximum(e-2.5, 0)**2/8), 1)
    return w, valid, err

def clip_vector(v, cap):
    return v * min(1., cap/max(float(np.linalg.norm(v)), 1e-9))

def causal_gate(t, p, v, w, valid, cap):
    """Historical causal branches, reconstructed increments, not scaled cumulative x/y."""
    out = np.zeros_like(p, dtype=float); branches = np.zeros(len(t), np.int8)
    trusted_history = []
    for k in range(1, len(t)):
        dt = t[k]-t[k-1]; dp = p[k]-p[k-1]; wt = w[k]
        step = wt*dp
        if valid[k] and wt >= .3:
            step += (1-wt)*.5*(v[k]+v[k-1])*dt
            branches[k] = 1
        elif valid[k]:
            if len(trusted_history) >= 3:
                ref = np.median(np.array(trusted_history[-5:]), axis=0); posv = dp/dt
                repv = .5*(v[k]+v[k-1])
                dpos, drep = np.linalg.norm(posv-ref), np.linalg.norm(repv-ref)
                if dpos < 2.5 and dpos < .5*drep:
                    step = dp; branches[k] = 2
                elif drep < 2.5 and drep < .5*dpos:
                    step += (1-wt)*clip_vector(repv, cap)*dt; branches[k] = 3
                else:
                    step += (1-wt)*clip_vector(ref, cap)*dt; branches[k] = 4
            else:
                branches[k] = 5
        out[k] = out[k-1]+step
        if valid[k] and w[k] >= .7 and np.isfinite(v[k]).all():
            trusted_history.append(v[k].copy())
    gated_v = v.copy()
    for k in range(1, len(t)):
        if valid[k] and w[k] < .2:
            gated_v[k] = gated_v[k-1]
    return out, gated_v, branches

def self_features(t, rel_p, v, cog=None):
    speed = np.linalg.norm(v, axis=1)
    direction = v/np.maximum(speed[:, None], 1e-9)
    if cog is not None:
        direction[speed < 1e-9] = np.stack([np.sin(np.deg2rad(cog)),
                                          np.cos(np.deg2rad(cog))], -1)[speed < 1e-9]
    return np.column_stack([rel_p, v, speed, direction, t-t[0], np.r_[0, np.diff(t)]]).astype('f4')

def fit_field(delta_p, v, distance, w, target_v, radius_m=8*1852., sigma_m=4*1852.):
    """LOO affine velocity field; zero-order kernel fallback keeps sparse context."""
    n = len(v); f = np.zeros(8, dtype='f4')
    if not n:
        return f, False
    weights = np.exp(-distance**2/(2*sigma_m**2))*w
    positive = weights > 1e-12
    if not positive.any() or not np.isfinite(weights).all():
        return f, False
    weights = weights/weights.sum()
    bg = np.sum(weights[:, None]*v, 0); divergence = curl = 0.; mode = 1
    X = np.column_stack([np.ones(n), delta_p/radius_m])
    gram = X.T@(weights[:, None]*X)
    if positive.sum() >= 4 and np.linalg.cond(gram) <= 1e8:
        B = np.linalg.solve(gram+.001*np.diag([0.,1.,1.]), X.T@(weights[:, None]*v))
        bg = B[0]; divergence = (B[1,0]+B[2,1])/radius_m
        curl = (B[1,1]-B[2,0])/radius_m; mode = 2
    # Support belongs to the observations actually contributing to this estimator.
    rho = min(float(positive.sum())/4, 1.)
    f[:] = [*bg, *(target_v-bg), divergence, curl, rho, mode]
    return f, True

class Background:
    """Time-sorted source arrays. Every lookup is backward-only and excludes target."""
    def __init__(self, table, bbox):
        order = np.argsort(table.timestamp.to_numpy(), kind='stable')
        self.t = table.timestamp.to_numpy(dtype=float)[order]
        self.ids = table.mmsi.to_numpy(dtype='i8')[order]
        self.p = project(table.lon.to_numpy()[order], table.lat.to_numpy()[order], bbox)
        self.sog = table.sog.to_numpy(dtype=float)[order]
        self.cog = table.cog.to_numpy(dtype=float)[order]
        self.v = vectors(self.sog, self.cog)
        lon, lat = table.lon.to_numpy()[order], table.lat.to_numpy()[order]
        self.valid_raw = (np.isfinite(self.t) & np.isfinite(self.p).all(1)
            & np.isfinite(self.sog) & np.isfinite(self.cog) & (np.abs(lon)<=180)
            & (np.abs(lat)<=90) & (self.sog>=0) & (self.sog<102.2)
            & (self.cog>=0) & (self.cog<360))
        self.prev = np.full(len(order), -1, dtype='i8')
        vessel_order = np.argsort(self.ids, kind='stable')
        ii, jj = vessel_order[1:], vessel_order[:-1]
        same = self.ids[ii] == self.ids[jj]
        self.prev[ii[same]] = jj[same]

    def visible(self, indices, stress):
        p, v = self.p[indices].copy(), self.v[indices].copy()
        if stress is not None:
            start, end, factors, offsets = stress
            for k, idx in enumerate(indices):
                member = int(self.ids[idx])
                if start <= self.t[idx] <= end and member in factors:
                    scale = min(factors[member], 102.1/max(self.sog[idx], 1e-6))
                    v[k] *= scale
                    p[k] += offsets[member]
        return p, v

    def stress_spec(self, hist_t, target, excluded, key):
        lo, hi = np.searchsorted(self.t, [hist_t[0]-300, hist_t[-1]], side='right')
        ids = np.unique(self.ids[lo:hi]); factors, offsets = {}, {}
        for mid in ids:
            if int(mid) == target or int(mid) in excluded:
                continue
            rng = seeded(f'{key}/neighbor/{mid}')
            if rng.random() < .5:
                factors[int(mid)] = rng.uniform(3.,5.)
                angle = rng.uniform(0, 2*np.pi)
                offsets[int(mid)] = 600*np.array([np.sin(angle),np.cos(angle)])
        return (hist_t[len(hist_t)//2], hist_t[-1], factors, offsets)

    def lookup(self, qt, target_p, target_cog, target_id, excluded, stress, cfg):
        lo = np.searchsorted(self.t, qt-cfg['background_age_s'], side='left')
        hi = np.searchsorted(self.t, qt, side='right')
        indices = np.arange(lo, hi)
        if not len(indices):
            return np.zeros((0,2)), np.zeros((0,2)), np.zeros(0), np.zeros(0), np.zeros(0,dtype='i8'), np.zeros(0)
        _, reverse_indices = np.unique(self.ids[indices][::-1], return_index=True)
        indices = indices[len(indices)-1-reverse_indices]
        allowed = np.array([int(i) != target_id and int(i) not in excluded for i in self.ids[indices]])
        indices = indices[allowed]
        p, v = self.visible(indices, stress)
        speed = np.linalg.norm(v, axis=1)
        dp = p-target_p; distance = np.linalg.norm(dp, axis=1)
        heading = np.rad2deg(np.arctan2(v[:,0], v[:,1])) % 360
        dh = np.abs((heading-target_cog+180)%360-180)
        good = self.valid_raw[indices] & np.isfinite(dp).all(1) & np.isfinite(v).all(1)
        good &= (distance <= cfg['radius_nm']*1852) & (speed >= cfg['min_neighbor_kn']*KNOT)
        good &= (dh < cfg['heading_deg'])
        indices, p, v, dp, distance = [a[good] for a in [indices,p,v,dp,distance]]
        take = np.argsort(distance, kind='stable')[:cfg['top_k']]
        indices, p, v, dp, distance = [a[take] for a in [indices,p,v,dp,distance]]
        w = np.zeros(len(indices))
        if len(indices):
            prev = self.prev[indices]; known = prev >= 0
            known &= self.valid_raw[np.maximum(prev,0)]
            pp, pv = self.visible(np.maximum(prev,0), stress)
            dt = self.t[indices]-self.t[np.maximum(prev,0)]
            known &= (dt > 0) & (dt <= 300)
            err = np.linalg.norm((p-pp)/np.maximum(dt[:,None],1e-9)-.5*(v+pv),axis=1)
            known &= np.isfinite(err)
            w[known] = np.exp(-np.maximum(err[known]-2.5,0)**2/8)
        return dp, v, distance, w, self.ids[indices], self.t[indices]

def make_view(t, p, sog, cog, target_id, background, excluded, key, view, cap, cfg):
    """Accept HISTORY ONLY. Fixed projection is external, never a clean future anchor."""
    p, sog = p.copy(), sog.copy()
    rng = seeded(key+'/target'); angle = rng.uniform(0,2*np.pi)
    direction = np.array([np.sin(angle),np.cos(angle)])
    if view == 'target_jump':
        p[-1] += 1000*direction
    elif view == 'target_drift':
        p[-4:] += np.linspace(0,600,4)[:,None]*direction
    elif view == 'target_speed':
        sog[-2:] = np.minimum(sog[-2:]*4,102.1)
    v = vectors(sog,cog); w, valid, err = compatibility(t,p,v)
    gp, gv, branches = causal_gate(t,p,v,w,valid,cap)
    raw = self_features(t,p-p[0],v,cog)
    gated = self_features(t,gp,gv,cog)
    stress = background.stress_spec(t,target_id,excluded,key) if view == 'neighbor_stress' else None
    f0, f1 = [], []; valid0, valid1 = [], []; members = set(); max_neighbor_time = -np.inf
    for step, qt in enumerate(t):
        dp, nv, dist, nw, mids, nt = background.lookup(qt,p[step],cog[step],target_id,excluded,stress,cfg)
        a, va = fit_field(dp,nv,dist,np.ones(len(nv)),gv[step],cfg['radius_nm']*1852,cfg['sigma_nm']*1852)
        b, vb = fit_field(dp,nv,dist,nw,gv[step],cfg['radius_nm']*1852,cfg['sigma_nm']*1852)
        assert not len(nt) or nt.max() <= qt
        f0.append(a);f1.append(b);valid0.append(va);valid1.append(vb)
        members.update(int(x) for x in mids)
        if len(nt): max_neighbor_time = max(max_neighbor_time,float(nt.max()))
    out = dict(raw=raw,gated=gated,f0=np.array(f0),f1=np.array(f1),
               f0_valid=np.array(valid0),f1_valid=np.array(valid1),
               anchor_raw=p[-1].astype('f8'),anchor_gate=(p[0]+gp[-1]).astype('f8'),
               vel_raw=v[-1].astype('f4'),vel_gate=gv[-1].astype('f4'),
               target_weight=w.astype('f4'),target_weight_valid=valid,gate_branch=branches)
    audit = dict(background_mmsi=sorted(members),max_neighbor_time=None if not np.isfinite(max_neighbor_time) else max_neighbor_time,
                 history_cutoff=float(t[-1]))
    return out,audit
