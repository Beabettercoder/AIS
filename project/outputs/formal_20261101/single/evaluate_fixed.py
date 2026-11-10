"""Evaluate frozen 25th-epoch single vessel models on previously exposed test."""
import sys,json,hashlib,argparse
from pathlib import Path
import numpy as np,torch
from sklearn.metrics import average_precision_score,roc_auc_score
P=Path('/mnt/sda/WangZijia/AIS/outputs/formal_20261101/single');ROOT=Path('/mnt/sda/WangZijia/AIS/outputs/framework_validation_v1_20261028');PREP=Path('/mnt/sda/WangZijia/AIS/outputs/eight_backbone_preparation_20261030')
sys.path.insert(0,str(P));import train_single as exp

def sh(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def quality(y,score):
 n=len(y);pos=int((y>0).sum());neg=n-pos
 if pos==0 or neg==0:return {'n':n,'positive':pos,'negative':neg,'AP':None,'AUROC':None}
 return {'n':n,'positive':pos,'negative':neg,'AP':float(average_precision_score(y,score)),'AUROC':float(roc_auc_score(y,score))}
def load_test():
 with np.load(ROOT/'data/individual/test.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
 with np.load(ROOT/'extra_data/field_test.npz',allow_pickle=False) as z:e={k:z[k] for k in z.files}
 with np.load(PREP/'rho_test.npz',allow_pickle=False) as z:rho=z['rho']
 f0=exp.build_field_inputs(d,e,'F0');f1=exp.build_field_inputs(d,e,'F1');assert np.array_equal(f0['valid'],f1['valid']) and np.array_equal(rho>0,f0['valid'])
 cat={x:i for i,x in enumerate(['turn_back','deviation','stall','sinuous','speed_drop'],1)};yc=np.array([cat.get(str(x),0) for x in d['behavior_type']],np.int64);assert np.array_equal(yc>0,d['y']>0)
 t=d['time_s'].copy();t-=t[:,:1];t[~d['mask']]=0
 v=dict(raw=d['base__C0'][...,:7],self=f0['self'],mask=d['mask'],times=t.astype(np.float32),y=d['y'].astype(np.float32),yc=yc,region=d['region'].astype(str),mmsi=d['mmsi'].astype(str),ids=d['window_id'].astype(str),f0=f0['field'],f1=f1['field'],state0=f0['state'],state1=f1['state'],rho=rho,valid=f0['valid'])
 return d,v

def main(name,seed,gpu):
 if gpu==4:raise ValueError('GPU4 reserved')
 torch.set_num_threads(2);p=P/'runs'/name/f'seed{seed}';lock=json.loads((P/'frozen_checkpoint_thresholds.json').read_text());out=p/'test';out.mkdir(exist_ok=False)
 pref=json.loads((P/'frozen_train_protocol.json').read_text());assert sh(P/'frozen_train_protocol.json')==lock['train_protocol_sha256']
 train=exp.load('train');val=exp.load('val');norm=exp.fit(train)
 for k in norm:np.testing.assert_array_equal(norm[k],np.array(pref['normalizer_arrays'][k]))
 raw,d=load_test();assert not (set(d['mmsi'])&set(train['mmsi'])) and not (set(d['mmsi'])&set(val['mmsi']))
 dev=torch.device('cuda:'+str(gpu));t=exp.convert(d,norm,dev); results={}
 for arm in ['raw','causal_bundle','f0_bundle','f1_bundle']:
  record=lock['arms'][f'{name}/{seed}/{arm}'];check=p/arm/'final.pt';assert sh(check)==record['checkpoint_sha256'];model=exp.Model(name).to(dev);model.load_state_dict(torch.load(check,map_location=dev,weights_only=True));model.eval();key='raw' if arm=='raw' else 'self';fld='f0' if arm=='f0_bundle' else 'f1';has=arm.startswith('f');score=[]
  with torch.no_grad():
   for i in range(0,len(d['y']),256):
    z=slice(i,i+256);score.extend(torch.sigmoid(model(t[key][z],t['mask'][z],t['times'][z],t[fld][z],t['rho'][z],has)[0]).cpu().numpy())
  score=np.array(score);assert np.isfinite(score).all();y=d['y'];r=d['region'];g=quality(y,score);regions={x:quality(y[r==x],score[r==x]) for x in sorted(set(r))};g['macro_AP']=float(np.mean([regions[x]['AP'] for x in regions if regions[x]['AP'] is not None]));g['regions']=regions
  g['operating_points']={}
  for b,v in record['thresholds'].items():
   threshold=v['threshold'];pred=score>threshold;g['operating_points'][b]={'threshold':threshold,'validation_FAR':v['val_FAR'],'actual_test_FAR':float(pred[y==0].mean()),'test_recall':float(pred[y>0].mean())}
  for col in ('noise_type','noise_scope','observation_noise','observation_noise_effective','target_noise_effective','neighbor_noise_effective'):
   if col in raw:
    v=raw[col].astype(str);g['strata_'+col]={x:quality(y[v==x],score[v==x]) for x in sorted(set(v))}
  for beh in ('turn_back','deviation','stall','sinuous','speed_drop'):
   selected=(y==0)|(raw['behavior_type'].astype(str)==beh);g.setdefault('behavior_vs_all_normal',{})[beh]=quality(y[selected],score[selected])
  file=out/(arm+'_scores.npz');np.savez_compressed(file,score=score,y=y,mmsi=d['mmsi'],region=r,window_id=d['ids'],noise_type=raw['noise_type'],noise_scope=raw['noise_scope'],observation_noise_effective=raw['observation_noise_effective'],behavior_type=raw['behavior_type']);g['prediction_sha256']=sh(file);g['checkpoint_sha256']=sh(check);results[arm]=g
 (out/'results.json').write_text(json.dumps({'name':name,'seed':seed,'dataset':'test previously exposed; not blind','threshold_lock_sha256':sh(P/'frozen_checkpoint_thresholds.json'),'metrics':results},indent=2));print('DONE',name,seed,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--seed',required=True,type=int);p.add_argument('--gpu',required=True,type=int);a=p.parse_args();main(a.name,a.seed,a.gpu)
