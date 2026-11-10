"""Independent construction, normalization and forward formulas for auditing."""
from pathlib import Path
import json
import numpy as np
import torch
from torch.nn import functional as F
from audit_helpers import FIELDS, BOOL_FIELDS, as_bool, require, sha

ARM_SOURCES={'self_quality':'raw_meta','mean_context':'mean_meta',
             'affine_context':'affine_meta','reliable_context':'reliable_raw'}


def derive(source):
    mask=np.asarray(source['mask'],bool)
    common=(mask & source['context_valid'].astype(bool) & source['plain_affine_valid'].astype(bool)
            & source['plain_context_valid'].astype(bool))
    result={}
    for arm,key in ARM_SOURCES.items():
        x=np.asarray(source[key],np.float32).copy()
        require(x.shape==mask.shape+(16,),f'{arm}: feature shape')
        require(np.isfinite(x).all() and np.all(x[~mask]==0),f'{arm}: finite padding')
        x[:,:,12]=common.astype(np.float32);x[:,:,13]=common.astype(np.float32)
        result[arm]=x
    ref=result['self_quality']
    for arm,x in result.items():
        for sl in (slice(0,7),slice(9,14)):
            require(np.array_equal(x[:,:,sl],ref[:,:,sl]),f'{arm}: non-context slots differ')
    require(np.all(ref[:,:,[7,8,14,15]]==0),'Self quality contains neighbor field')
    return result,common


def load_inputs(parent,protocol):
    chunks={s:[] for s in ('train','val')};proof=[]
    for s in ('train','val'):
        for r in protocol['regions']:
            path=Path(parent)/'data'/r/f'{s}.npz'
            before=sha(path)
            with np.load(path,allow_pickle=False) as z:
                source={k:z[k] for k in ('mask','context_valid','plain_affine_valid','plain_context_valid','raw16')+tuple(ARM_SOURCES.values())}
                arms,common=derive(source)
                d={key:z[key] for key in FIELDS}
                for key in BOOL_FIELDS:d[key]=np.asarray([as_bool(v) for v in d[key]],bool)
                d['mask']=source['mask'].astype(bool);d['raw16']=source['raw16'].astype(np.float32)
                d['context_valid']=source['context_valid'].astype(bool);d['common_support']=common;d['features']=arms
                n=len(d['y']);mask=d['mask']
                require(n==protocol['samples'][f'{s}_per_region'],f'{path}: sample count')
                require(mask.shape==(n,20) and np.all(mask.sum(1)>0),f'{path}: mask')
                require(np.array_equal(mask,np.arange(20)[None,:]<mask.sum(1)[:,None]),f'{path}: non-prefix mask')
                require(np.array_equal(d['raw16'][:,:,:7],arms['self_quality'][:,:,:7]),f'{path}: raw channel mismatch')
                require(np.all(d['raw16'][:,:,7:]==0),f'{path}: nonzero baseline filler')
                require(np.all(d['region'].astype(str)==r),f'{path}: wrong region')
                chunks[s].append(d)
            require(sha(path)==before,f'{path}: source changed')
            proof.append(dict(split=s,region=r,path=str(path),sha256=before,samples=n,
                              valid_steps=int(mask.sum()),trusted_valid_steps=int((mask&d['context_valid']).sum()),
                              common_support_steps=int(common.sum()),zero_support_windows=int((common.sum(1)==0).sum())))
    output={}
    for s,parts in chunks.items():
        output[s]={k:np.concatenate([x[k] for x in parts]) for k in FIELDS+('mask','raw16','context_valid','common_support')}
        output[s]['features']={arm:np.concatenate([x['features'][arm] for x in parts]) for arm in ARM_SOURCES}
        for key in ('window_id','base_window_id'):
            require(len(set(output[s][key].astype(str)))==len(output[s]['y']),f'{s}: repeated {key}')
    for key in ('window_id','base_window_id','mmsi'):
        require(not(set(output['train'][key].astype(str)) & set(output['val'][key].astype(str))),f'Cross split {key}')
    return output,proof


def fit_adapter_scaler(x,support):
    selected=np.asarray(x,np.float64)[support]
    require(len(selected)>0,'No supported training observations')
    return dict(mean=selected.mean(0).astype(np.float32),
                std=np.maximum(selected.std(0),1e-6).astype(np.float32),valid_steps=int(len(selected)))


def normalized_adapter(x,support,scaler):
    out=((np.asarray(x,np.float32)-scaler['mean'])/scaler['std']).astype(np.float32)
    out[~support]=0
    require(np.isfinite(out).all(),'Nonfinite adapter normalization')
    return out


def normalized_baseline(x,mask,mean,std):
    out=((np.asarray(x,np.float32)-mean)/std).astype(np.float32);out[~mask]=0
    require(np.isfinite(out).all(),'Nonfinite baseline normalization')
    return out


def reference_pool(h,mask):
    w=mask.unsqueeze(-1)
    return torch.cat(((h*w).sum(1)/w.sum(1).clamp_min(1),h.masked_fill(~w,float('-inf')).max(1).values),dim=1)


def get_adapter_tensors(state):
    require(len(state)==4,'Adapter must have two Linear layers only')
    by_shape={tuple(v.shape):(k,v) for k,v in state.items()}
    expected=((32,16),(32,),(64,32),(64,))
    require(set(by_shape)==set(expected),'Unexpected adapter tensor dimensions')
    return [by_shape[k][1] for k in expected]


def reference_delta(x,mask,support,state):
    w1,b1,w2,b2=get_adapter_tensors(state)
    values=F.linear(F.gelu(F.linear(x,w1,b1)),w2,b2)
    weight=(mask & support).unsqueeze(-1)
    return (values*weight).sum(1)/mask.sum(1).clamp_min(1).unsqueeze(-1)


@torch.inference_mode()
def reference_baseline(model,x,mask,batch=128):
    model.eval();hs=[];logits=[]
    for lo in range(0,len(x),batch):
        xx=torch.from_numpy(x[lo:lo+batch]);mm=torch.from_numpy(mask[lo:lo+batch])
        h=reference_pool(model.encode(xx,mm),mm);v=model.head(h).squeeze(-1)
        hs.append(h.numpy());logits.append(v.numpy())
    h=np.concatenate(hs);logit=np.concatenate(logits)
    return h,logit,torch.sigmoid(torch.from_numpy(logit)).numpy()


@torch.inference_mode()
def reference_predictions(head,h,x,mask,support,state,batch=128):
    head.eval();scores=[];deltas=[]
    for lo in range(0,len(h),batch):
        sl=slice(lo,lo+batch)
        delta=reference_delta(torch.from_numpy(x[sl]),torch.from_numpy(mask[sl]),torch.from_numpy(support[sl]),state)
        v=head(torch.from_numpy(h[sl])+delta).squeeze(-1)
        scores.append(torch.sigmoid(v).numpy());deltas.append(delta.numpy())
    return np.concatenate(scores),np.concatenate(deltas)


def availability(data):
    answer={}
    for s,d in data.items():
        mask=d['mask'];common=d['common_support'];trusted=mask&d['context_valid'];n=int(mask.sum())
        answer[s]=dict(windows=len(mask),valid_steps=n,trusted_valid_steps=int(trusted.sum()),
                       common_support_steps=int(common.sum()),
                       trusted_fraction=float(trusted.sum()/n),common_fraction=float(common.sum()/n),
                       trusted_steps_excluded_by_intersection=int((trusted&~common).sum()),
                       zero_common_support_windows=int((common.sum(1)==0).sum()))
    return answer
