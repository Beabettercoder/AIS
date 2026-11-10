"""Small paired pipeline pilot; this is not the final multi-seed experiment.

Every architecture/input configuration receives the same frozen subset. All
normalization, checkpoint selection and threshold choice exclude test data.
No generator, field operator, PSA or consistency augmentation is implemented here.
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
from pathlib import Path
import argparse,copy,csv,hashlib,json,math,random,sys,time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from sklearn.metrics import (average_precision_score,roc_auc_score,
                             precision_recall_curve,confusion_matrix,
                             precision_recall_fscore_support)

FEATURES=('raw','plain_context','plain_affine_context','context')
BACKBONES=('tcn','gru','transformer')
SPLITS=('train','val','test')
DIMENSIONS={'raw':7,'plain_context':16,'plain_affine_context':16,'context':16}
REGIONS=('houston','ny','la','puget','great_belt','haugesund','kattegat','shanghai')

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def seed_all(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 torch.use_deterministic_algorithms(True)

def validate_mask(mask):
 mask=np.asarray(mask,bool)
 if mask.ndim!=2 or not np.all(mask.sum(1)>=1):raise ValueError('Nonempty N x T mask required')
 if not np.array_equal(mask,np.arange(mask.shape[1])[None,:]<mask.sum(1)[:,None]):
  raise ValueError('Pilot requires chronological right-padding masks')
 return mask

def binary_quota(total,positive_fraction=.4):
 positive=int(round(total*positive_fraction))
 if positive<1 or positive>=total:raise ValueError('Each pilot domain/split needs both labels')
 return total-positive,positive

def pick_subset(y,window_id,total,seed,region,split):
 """Stable, label-stratified metadata selection, never score-dependent."""
 y=np.asarray(y);ids=np.asarray(window_id,dtype=str)
 if not np.isin(y,[0,1]).all():raise ValueError('y must be binary')
 if len(ids)!=len(y) or len(set(ids))!=len(ids):raise ValueError('Unique window_id required')
 wanted=binary_quota(total)
 rng=np.random.default_rng(int.from_bytes(hashlib.sha256(f'{seed}:{region}:{split}:subset'.encode()).digest()[:8],'little'))
 selected=[]
 for cls,n in enumerate(wanted):
  ii=np.flatnonzero(y==cls);ii=ii[np.argsort(ids[ii],kind='stable')]
  if len(ii)<n:raise ValueError(f'Insufficient {region}/{split}/class{cls}: {len(ii)} < {n}')
  selected.extend(rng.permutation(ii)[:n].tolist())
 selected=np.array(selected,dtype=np.int64)
 return selected[np.argsort(ids[selected],kind='stable')]

def load_pilot(data,regions,sizes,seed,features):
 chunks={s:[] for s in SPLITS};proof=[];selection=[]
 for region in regions:
  for split in SPLITS:
   p=data/region/f'{split}.npz'
   with np.load(p,allow_pickle=False) as z:
    required={'y','mask','mmsi','window_id'}|set(features)
    missing=required-set(z.files)
    if missing:raise ValueError(f'{p}: missing {sorted(missing)}')
    original_y=z['y']
    if original_y.ndim!=1 or not np.isin(original_y,[0,1]).all():raise ValueError(f'{p}: y must be a binary vector')
    y=original_y.astype(np.int64);ids=z['window_id'].astype(str)
    idx=pick_subset(y,ids,sizes[split],seed,region,split)
    mask=validate_mask(z['mask'][idx]);n=len(idx)
    if mask.shape[1]!=20:raise ValueError('Expected 20-step source windows')
    mid=z['mmsi'][idx].astype(str)
    base=(z['base_window_id'][idx].astype(str) if 'base_window_id' in z.files else ids[idx])
    c={'y':y[idx],'mask':mask,'window_id':ids[idx],'base_window_id':base,
       'mmsi':mid,'region':np.repeat(region,n),'source_row':idx,'features':{}}
    for key in features:
     x=z[key][idx].astype(np.float32)
     if x.shape!=(n,20,DIMENSIONS[key]):raise ValueError(f'{p}: bad {key} shape {x.shape}')
     if not np.isfinite(x).all():raise ValueError(f'{p}: nonfinite {key}')
     if np.any(x[~mask]!=0):raise ValueError(f'{p}: nonzero padding in {key}')
     c['features'][key]=x
    for k in ['behavior_type','noise_type','observation_noise','group_normal_variation']:
     if k in z.files:
      v=z[k]
      if v.shape!=(len(y),):raise ValueError(f'{p}: unexpected {k} shape')
      c[k]=v[idx].astype(str)
   proof.append({'region':region,'split':split,'path':str(p),'sha256':sha(p),
                 'selected':n,'normal':int((c['y']==0).sum()),'anomaly':int((c['y']==1).sum()),
                 'selected_target_mmsi':len(set(mid))})
   for j in range(n):
    selection.append({'region':region,'split':split,'window_id':str(c['window_id'][j]),
                      'base_window_id':str(base[j]),'mmsi':str(mid[j]),'source_row':int(idx[j]),'y':int(c['y'][j])})
   chunks[split].append(c)
 output={}
 for split,parts in chunks.items():
  output[split]={k:np.concatenate([p[k] for p in parts]) for k in ['y','mask','window_id','base_window_id','mmsi','region','source_row']}
  output[split]['features']={k:np.concatenate([p['features'][k] for p in parts]) for k in features}
  for k in ['behavior_type','noise_type','observation_noise','group_normal_variation']:
   if all(k in p for p in parts):output[split][k]=np.concatenate([p[k] for p in parts])
 for a,b in [('train','val'),('train','test'),('val','test')]:
  for k in ['mmsi','window_id','base_window_id']:
   if set(output[a][k])&set(output[b][k]):raise ValueError(f'{k} crosses {a}/{b}')
 for split,c in output.items():
  if len(set(c['window_id']))!=len(c['window_id']):raise ValueError(f'Repeated window in {split}')
 return output,proof,selection

def fit_normalizer(x,mask):
 valid=np.asarray(x,dtype=np.float64)[validate_mask(mask)]
 return {'mean':valid.mean(0).astype(np.float32),
         'std':np.maximum(valid.std(0),1e-6).astype(np.float32),
         'valid_steps':int(len(valid))}

def normalized(x,mask,normalizer):
 result=(x-normalizer['mean'])/normalizer['std']
 result=np.asarray(result,dtype=np.float32);result[~mask]=0
 if not np.isfinite(result).all():raise FloatingPointError('Nonfinite normalized features')
 return result

def masked_pool(h,mask):
 if h.ndim!=3 or mask.shape!=h.shape[:2]:raise ValueError('Pooling dimensions')
 weight=mask.unsqueeze(-1)
 mean=(h*weight).sum(1)/weight.sum(1).clamp_min(1)
 maximum=h.masked_fill(~weight,float('-inf')).max(1).values
 return torch.cat((mean,maximum),-1)

class CausalResidual(nn.Module):
 def __init__(self,hidden,dilation,dropout):
  super().__init__();self.left=2*dilation
  self.c1=nn.Conv1d(hidden,hidden,3,dilation=dilation)
  self.c2=nn.Conv1d(hidden,hidden,3,dilation=dilation)
  self.drop=nn.Dropout(dropout);self.norm=nn.LayerNorm(hidden)
 def forward(self,x,mask):
  y=self.drop(F.gelu(self.c1(F.pad(x.transpose(1,2),(self.left,0)))))
  y=self.drop(F.gelu(self.c2(F.pad(y,(self.left,0))))).transpose(1,2)
  return self.norm(x+y)*mask.unsqueeze(-1)

class BackboneClassifier(nn.Module):
 def __init__(self,kind,input_dim,hidden=32,dropout=.1,max_steps=128):
  super().__init__();self.kind=kind;self.hidden=hidden
  if kind=='tcn':
   self.project=nn.Linear(input_dim,hidden)
   self.blocks=nn.ModuleList([CausalResidual(hidden,d,dropout) for d in [1,2,4,8]])
  elif kind=='gru':
   self.rnn=nn.GRU(input_dim,hidden,num_layers=2,batch_first=True,dropout=dropout)
  elif kind=='transformer':
   self.project=nn.Linear(input_dim,hidden)
   layer=nn.TransformerEncoderLayer(hidden,4,hidden*2,dropout=dropout,
                                    activation='gelu',batch_first=True,norm_first=True)
   self.encoder=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
   pos=torch.arange(max_steps).float().unsqueeze(1)
   scale=torch.exp(torch.arange(0,hidden,2).float()*(-math.log(10000.)/hidden))
   pe=torch.zeros(max_steps,hidden);pe[:,0::2]=torch.sin(pos*scale);pe[:,1::2]=torch.cos(pos*scale)
   self.register_buffer('position_encoding',pe)
  else:raise ValueError(kind)
  self.head=nn.Sequential(nn.Linear(hidden*2,hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden,1))
 def encode(self,x,mask):
  x=x*mask.unsqueeze(-1)
  if self.kind=='tcn':
   h=self.project(x)*mask.unsqueeze(-1)
   for layer in self.blocks:h=layer(h,mask)
  elif self.kind=='gru':
   packed=pack_padded_sequence(x,mask.sum(1).cpu(),batch_first=True,enforce_sorted=False)
   h,_=self.rnn(packed);h,_=pad_packed_sequence(h,batch_first=True,total_length=x.shape[1])
  else:
   h=self.project(x)+self.position_encoding[:x.shape[1]]
   h=self.encoder(h,src_key_padding_mask=~mask)
  return h
 def forward(self,x,mask):return self.head(masked_pool(self.encode(x,mask),mask)).squeeze(-1)

def choose_threshold(y_val,p_val):
 """Only validation arrays are accepted; highest threshold breaks F1 ties."""
 y=np.asarray(y_val);p=np.asarray(p_val)
 if not np.isin(y,[0,1]).all() or len(np.unique(y))!=2:raise ValueError('Both validation classes required')
 if len(y)!=len(p) or not np.isfinite(p).all():raise ValueError('Invalid validation predictions')
 precision,recall,thresholds=precision_recall_curve(y,p)
 f1=2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-15)
 best=np.max(f1);ties=np.flatnonzero(np.isclose(f1,best,rtol=0,atol=1e-12))
 return float(thresholds[ties[-1]]),float(best)

def metrics(y,p,threshold):
 y=np.asarray(y);p=np.asarray(p);pred=p>=threshold
 pr,re,f1,_=precision_recall_fscore_support(y,pred,average='binary',zero_division=0)
 tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
 return {'AP':float(average_precision_score(y,p)),'AUROC':float(roc_auc_score(y,p)),
         'F1':float(f1),'precision':float(pr),'recall':float(re),
         'confusion':{'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)},
         'samples':len(y),'positive_samples':int(y.sum()),'positive_fraction':float(y.mean())}

def region_metrics(y,p,regions,threshold):
 region=np.asarray(regions);per={r:metrics(y[region==r],p[region==r],threshold) for r in sorted(set(region))}
 macro={k:float(np.mean([v[k] for v in per.values()])) for k in ['AP','AUROC','F1','precision','recall']}
 return {'macro':macro,'pooled':metrics(y,p,threshold),'per_region':per}

@torch.inference_mode()
def predict(model,x,mask,batch_size):
 model.eval();scores=[]
 for lo in range(0,len(x),batch_size):scores.append(torch.sigmoid(model(x[lo:lo+batch_size],mask[lo:lo+batch_size])).cpu().numpy())
 return np.concatenate(scores)

def write_csv(path,rows,fields):
 with open(path,'w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def write_predictions(path,dataset,pred,threshold,kind,feature,seed):
 rows=[]
 for i,score in enumerate(pred):
  row={'backbone':kind,'feature_group':feature,'seed':seed,'region':str(dataset['region'][i]),
       'window_id':str(dataset['window_id'][i]),'base_window_id':str(dataset['base_window_id'][i]),
       'mmsi':str(dataset['mmsi'][i]),'y':int(dataset['y'][i]),'score':float(score),
       'threshold':threshold,'prediction':int(score>=threshold)}
  for k in ['behavior_type','noise_type','observation_noise','group_normal_variation']:
   row[k]=str(dataset[k][i]) if k in dataset else ''
  rows.append(row)
 write_csv(path,rows,list(rows[0]))

def train_one(kind,feature,datasets,normalizer,args,out):
 seed_all(args.seed);start=time.time();out.mkdir(parents=True,exist_ok=False)
 tensors={s:{'x':torch.from_numpy(normalized(d['features'][feature],d['mask'],normalizer)),
             'mask':torch.from_numpy(d['mask']),'y':torch.from_numpy(d['y'].astype(np.float32))} for s,d in datasets.items()}
 model=BackboneClassifier(kind,DIMENSIONS[feature],args.hidden,args.dropout)
 params=sum(p.numel() for p in model.parameters())
 optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
 rng=np.random.default_rng(args.seed+100);history=[];best=None;best_ap=float('-inf');best_epoch=None
 for epoch in range(1,args.epochs+1):
  model.train();order=rng.permutation(len(tensors['train']['x']));loss_sum=0.
  for lo in range(0,len(order),args.batch_size):
   ii=order[lo:lo+args.batch_size];tr=tensors['train']
   optimizer.zero_grad(set_to_none=True);logits=model(tr['x'][ii],tr['mask'][ii])
   loss=F.binary_cross_entropy_with_logits(logits,tr['y'][ii]);loss.backward()
   nn.utils.clip_grad_norm_(model.parameters(),args.clip_grad_norm);optimizer.step();loss_sum+=float(loss.detach())*len(ii)
  vp=predict(model,tensors['val']['x'],tensors['val']['mask'],args.batch_size)
  vals=region_metrics(datasets['val']['y'],vp,datasets['val']['region'],.5)
  va=vals['macro']['AP'];history.append({'epoch':epoch,'train_loss':loss_sum/len(order),'validation_macro_AP':va})
  if va>best_ap:
   best_ap=va;best_epoch=epoch;best=copy.deepcopy(model.state_dict())
  print(f'EPOCH {kind}/{feature} {epoch}/{args.epochs} loss={history[-1]["train_loss"]:.5f} val_macro_AP={va:.5f}',flush=True)
 model.load_state_dict(best);model.eval()
 vp=predict(model,tensors['val']['x'],tensors['val']['mask'],args.batch_size)
 threshold,best_f1=choose_threshold(datasets['val']['y'],vp)
 # The test prediction is computed only once, after all validation choices.
 tp=predict(model,tensors['test']['x'],tensors['test']['mask'],args.batch_size)
 result={'backbone':kind,'feature_group':feature,'input_channels':DIMENSIONS[feature],
         'parameters':params,'seed':args.seed,'epochs_budget':args.epochs,'selected_epoch':best_epoch,
         'checkpoint_rule':'maximum validation region-macro AP; earliest epoch on exact tie',
         'threshold':threshold,'threshold_rule':'maximum pooled validation F1; highest threshold on tie',
         'validation_selected_F1':best_f1,'validation':region_metrics(datasets['val']['y'],vp,datasets['val']['region'],threshold),
         'test':region_metrics(datasets['test']['y'],tp,datasets['test']['region'],threshold),
         'elapsed_seconds':time.time()-start,'history':history,
         'normalizer_fit':'only selected source-training valid steps',
         'augmentation':'none; PSA and consistency excluded from this paired mechanism pilot',
         'result_scope':'Single-seed pipeline pilot; final multi-seed experiments are separate.'}
 write_predictions(out/'validation_predictions.csv',datasets['val'],vp,threshold,kind,feature,args.seed)
 write_predictions(out/'test_predictions.csv',datasets['test'],tp,threshold,kind,feature,args.seed)
 torch.save({'state_dict':best,'backbone':kind,'input_channels':DIMENSIONS[feature],
             'hidden':args.hidden,'dropout':args.dropout,'normalizer_mean':torch.from_numpy(normalizer['mean']),
             'normalizer_std':torch.from_numpy(normalizer['std']),'threshold':threshold},out/'checkpoint.pt')
 result['output_hashes']={p.name:sha(p) for p in out.iterdir() if p.is_file()}
 (out/'metrics.json').write_text(json.dumps(result,indent=2))
 print('CONFIG_COMPLETE',kind,feature,json.dumps(result['test']['macro']),flush=True)
 return result

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
 ap.add_argument('--epochs',type=int,default=8);ap.add_argument('--threads',type=int,default=4);ap.add_argument('--seed',type=int,default=0)
 ap.add_argument('--hidden',type=int,default=32);ap.add_argument('--dropout',type=float,default=.1)
 ap.add_argument('--lr',type=float,default=.002);ap.add_argument('--weight-decay',type=float,default=.01)
 ap.add_argument('--clip-grad-norm',type=float,default=5.);ap.add_argument('--batch-size',type=int,default=128)
 ap.add_argument('--regions',nargs='+',default=list(REGIONS));ap.add_argument('--backbones',nargs='+',choices=BACKBONES,default=list(BACKBONES))
 ap.add_argument('--features',nargs='+',choices=FEATURES,default=list(FEATURES))
 ap.add_argument('--train-per-region',type=int,default=240);ap.add_argument('--val-per-region',type=int,default=80);ap.add_argument('--test-per-region',type=int,default=80)
 ap.add_argument('--prepare-only',action='store_true',help='Audit inputs and write frozen subset/normalizers without training')
 args=ap.parse_args()
 if not 1<=args.threads<=4:raise SystemExit('CPU threads must be in [1,4]')
 if args.hidden%4 or args.epochs<1:raise SystemExit('Hidden width must divide four and epochs positive')
 if args.out.exists() and any(args.out.iterdir()):raise SystemExit('Refusing to overwrite a nonempty pilot output; choose a fresh directory')
 args.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
 sizes={s:getattr(args,f'{s}_per_region') for s in SPLITS};data,proof,selection=load_pilot(args.data,args.regions,sizes,args.seed,args.features)
 subset=args.out/'pilot_subset.csv';write_csv(subset,selection,list(selection[0]))
 scalers={f:fit_normalizer(data['train']['features'][f],data['train']['mask']) for f in args.features}
 np.savez_compressed(args.out/'training_normalizers.npz',**{f'{f}_{k}':v[k] for f,v in scalers.items() for k in ['mean','std']})
 configuration={'stage':'paired_pipeline_pilot','data':str(args.data),'script_sha256':sha(Path(__file__)),
                'subset_sha256':sha(subset),'subset_policy':'same feature-independent 60/40 stratified source-window subset for all configurations',
                'normalizer_policy':'fit selected train valid steps only; force padding to zero after normalization',
                'architecture_definitions':{'tcn':'causal 4-block residual TCN; two kernel3 convolutions per block, dilations 1/2/4/8; per-step LayerNorm',
                  'gru':'two-layer unidirectional GRU with packed right-padded sequences',
                  'transformer':'two-layer encoder, sinusoidal positions, four heads, explicit padding mask'},
                'pooling':'masked mean concatenated with masked max','input_groups':args.features,
                'source_files':proof,'configuration':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                'versions':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,
                  'sklearn':__import__('sklearn').__version__},'device':'cpu',
                'capacity_scope':'The three 16-channel context groups have identical network parameter counts within each backbone. Raw7 changes the input layer; this is not a separately capacity-matched control.',
                'labels':'Controlled injection labels from supplied data; the harness does not alter labels.'}
 (args.out/'pilot_configuration.json').write_text(json.dumps(configuration,indent=2))
 if args.prepare_only:
  print(json.dumps({'status':'inputs_verified_no_training','out':str(args.out),'counts':{s:len(d['y']) for s,d in data.items()}},indent=2));return
 results=[]
 for kind in args.backbones:
  for feature in args.features:results.append(train_one(kind,feature,data,scalers[feature],args,args.out/f'{kind}__{feature}'))
 for kind in args.backbones:
  counts={r['parameters'] for r in results if r['backbone']==kind and r['input_channels']==16}
  if len(counts)>1:raise RuntimeError('Context group parameter counts differ')
 summary={'status':'completed_single_seed_pilot','scope':'12 paired pipeline configurations by default; not final multi-seed evidence',
          'same_subset_for_all_configs':True,'subset_sha256':sha(subset),'source_data_unchanged':all(sha(Path(p['path']))==p['sha256'] for p in proof),
          'results':[{k:r[k] for k in ['backbone','feature_group','input_channels','parameters','seed','selected_epoch','threshold','test','elapsed_seconds']} for r in results]}
 (args.out/'pilot_summary.json').write_text(json.dumps(summary,indent=2));print('PILOT_COMPLETE',str(args.out),flush=True)

if __name__=='__main__':main()
