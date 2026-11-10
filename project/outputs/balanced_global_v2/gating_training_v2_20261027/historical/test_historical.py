from pathlib import Path
import copy,json,tempfile,unittest
from unittest.mock import patch
import numpy as np
import torch
from torch.nn import functional as F
import historical_model as hm
import train_historical as tr

torch.set_num_threads(4)

def sample_batch(n=6,T=20,nf=9):
    torch.manual_seed(33)
    return dict(x=torch.randn(n,T,nf),mask=torch.ones(n,T,dtype=torch.bool),y=torch.tensor(([0.,1.]*(n//2))),
                label_class=torch.tensor(([0,1]*(n//2))),ep=torch.randn(n,T,4),rf=torch.rand(n,T),
                nb_seq=torch.randn(n,32,5,4),nb_xy=torch.randn(n,32,2)*1000,nb_mask=torch.ones(n,32,dtype=torch.bool),nb_w=torch.rand(n,32))

def reference_full_loss(model,b,cw):
    x=b['x'];args=(b['nb_seq'],b['nb_xy'],b['nb_mask'],b['nb_w'],b['rf'][:,-1])
    pb,pc,h,ctx=model(x,*args)
    positive=b['y']>0
    loss=F.binary_cross_entropy_with_logits(pb,b['y'])+.5*F.cross_entropy(pc[positive],b['label_class'][positive]-1,weight=cw)
    keep=(torch.rand(x.shape[0],x.shape[1])>.15).float();x2=x*keep[...,None]
    jj=torch.randint(0,x.shape[1],(len(x),));x2[torch.arange(len(x)),jj,0]+=2.
    _,_,h2,_=model(x2,*args);e=model.ctrl(b['ep'].mean(1));eps=.05+.2*(torch.tanh(e[:,0])+1)/2
    mu=h2.mean(0,keepdim=True).detach();sd=h2.std(0,keepdim=True).detach().clamp_min(1e-5)
    hp=mu+sd*(((h2-mu)/sd)*(1+eps[:,None])+.02*torch.tanh(e[:,1])[:,None])
    p1,p2=F.softmax(pc,-1),F.softmax(model.cls(hp),-1);middle=.5*(p1+p2)
    kl=lambda a,b:(a*(torch.log(a+1e-8)-torch.log(b+1e-8))).sum(-1)
    con=((.5*kl(p1,middle)+.5*kl(p2,middle))*keep.mean(1)).mean()
    loss=loss+.2*con+.3*F.binary_cross_entropy_with_logits(model.bin(hp).squeeze(-1),b['y'])
    hseq=model.enc.seq(x)
    return loss+.01*(hseq[:,:,2:]-2*hseq[:,:,1:-1]+hseq[:,:,:-2]).square().mean()

def synthetic_data(n,split):
    rng=np.random.default_rng(20 if split=='train' else 21);T=20;mask=np.ones((n,T),bool)
    y=np.arange(n)%2;yc=np.where(y,1+np.arange(n)%5,0)
    d=dict(mask=mask,y=y.astype(np.int64),label_class=yc.astype(np.int64),ep=rng.normal(size=(n,T,4)).astype('f4'),rf=np.zeros((n,T),'f4'),
           nb_seq=rng.normal(size=(n,32,5,4)).astype('f4'),nb_xy=np.zeros((n,32,2),'f4'),nb_mask=np.ones((n,32),bool),nb_w=np.ones((n,32),'f4'))
    for k in tr.META:
        if k in ['observation_noise','target_noise_effective','neighbor_noise_effective','observation_noise_effective']:d[k]=np.arange(n)<4
        elif k=='region':d[k]=np.repeat('houston',n)
        else:d[k]=np.array([f'{split}_{k}_{i}' for i in range(n)])
    d['features']={k:rng.normal(size=(n,T,next(c['nf'] for c in hm.ARMS.values() if c['key']==k))).astype('f4') for k in tr.INPUT_KEYS}
    return d

class HistoricalTests(unittest.TestCase):
    def test_exact_full_forward_and_parameters(self):
        b=sample_batch()
        for kind,wanted in [('tcn',71125),('gru',65237)]:
            torch.manual_seed(4);ref=hm.legacy.Net(kind,9,'ours')
            torch.manual_seed(4);model=hm.HistoricalModel(kind,'full35_compat')
            self.assertEqual(sum(p.numel() for p in model.parameters()),wanted)
            self.assertEqual(tr.state_sha(ref.state_dict()),tr.state_sha(model.core.state_dict()))
            out=ref(b['x'],b['nb_seq'],b['nb_xy'],b['nb_mask'],b['nb_w'],b['rf'][:,-1])
            actual=model(b['x'],b['mask'],b['nb_seq'],b['nb_xy'],b['nb_mask'],b['nb_w'],b['rf'])
            for x,y in zip(out,actual):self.assertTrue(torch.equal(x,y))
    def test_exact_compat_loss_and_gradients(self):
        b=sample_batch();cw=torch.ones(5)
        for kind in ['tcn','gru']:
            torch.manual_seed(5);ref=hm.legacy.Net(kind,9,'ours')
            torch.manual_seed(5);model=hm.HistoricalModel(kind,'full35_compat')
            torch.manual_seed(91);expected=reference_full_loss(ref,b,cw)
            torch.manual_seed(91);actual,logs=hm.training_loss(model,b,cw)
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            actual.backward();expected.backward()
            for (name,a),(name2,bp) in zip(model.core.named_parameters(),ref.named_parameters()):
                self.assertEqual(name,name2)
                if a.grad is None:self.assertIsNone(bp.grad)
                else:torch.testing.assert_close(a.grad,bp.grad,rtol=1e-6,atol=1e-7)
            self.assertGreater(float(model.core.gamma.grad.abs()),0)
            self.assertGreater(float(model.ctrl.weight.grad.norm()),0)
    def test_full_length_masked_equals_compat(self):
        b=sample_batch();cw=torch.ones(5)
        for kind in ['tcn','gru']:
            torch.manual_seed(9);a=hm.HistoricalModel(kind,'full35_compat')
            torch.manual_seed(9);m=hm.HistoricalModel(kind,'full35_masked')
            torch.manual_seed(81);la,_=hm.training_loss(a,b,cw)
            torch.manual_seed(81);lm,_=hm.training_loss(m,b,cw)
            self.assertTrue(torch.equal(la,lm))
    def test_masked_encoding_padding_and_rf_invariance(self):
        b=sample_batch();b['mask'][0,8:]=False;b['mask'][1,12:]=False
        for kind in ['tcn','gru']:
            model=hm.HistoricalModel(kind,'full35_masked').eval()
            a=model(b['x'],b['mask'],b['nb_seq'],b['nb_xy'],b['nb_mask'],b['nb_w'],b['rf'])[0]
            x=b['x'].clone();x[~b['mask']]=10000;rf=b['rf'].clone();rf[~b['mask']]=10000
            c=model(x,b['mask'],b['nb_seq'],b['nb_xy'],b['nb_mask'],b['nb_w'],rf)[0]
            self.assertTrue(torch.equal(a,c))
    def test_masked_augmentation_only_valid(self):
        x=torch.ones(10,20,9);mask=torch.arange(20)[None,:]<torch.arange(1,11)[:,None]
        for seed in range(5):
            torch.manual_seed(seed);a,ret,jj,keep=hm.augment_features(x,mask,True)
            self.assertTrue(bool(torch.all(jj<mask.sum(1))))
            self.assertTrue(bool(torch.all(a[~mask]==0)))
            self.assertTrue(torch.equal(ret,(keep*mask).sum(1)/mask.sum(1)))
    def test_historical_normalizer_includes_corrupt_train_only(self):
        d=synthetic_data(12,'train');x=d['features']['raw_legacy7'];x[:4]+=100
        s=tr.fit_normalizer(x,d['mask']);self.assertTrue(np.array_equal(s['mean'],x[d['mask']].mean(0,keepdims=True)))
        self.assertGreater(float(s['mean'][0,0]),25)
        mask=d['mask'].copy();mask[:,15:]=False
        self.assertTrue(np.all(tr.normalized(x,mask,s,True)[~mask]==0))
        self.assertFalse(np.all(tr.normalized(x,mask,s,False)[~mask]==0))
    def test_shuffle_matches_legacy_clean_only(self):
        ids=np.array([0,2,3,5,8]);orders,h,_=tr.make_orders(ids,3,4)
        np.random.seed(3);work=ids.copy()
        for got in orders:np.random.shuffle(work);self.assertTrue(np.array_equal(got,work))
        self.assertEqual(h,tr.make_orders(ids,3,4)[1])
    def test_loader_never_opens_test_and_checks_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'houston').mkdir()
            for split in ['train','val']:
                d=synthetic_data(12,split);features=d.pop('features');np.savez(root/'houston'/f'{split}.npz',**d,**features)
            original=np.load;opened=[]
            def safe(path,*a,**kw):
                self.assertIn(Path(path).name,['train.npz','val.npz']);opened.append(Path(path).name);return original(path,*a,**kw)
            with patch.object(tr.np,'load',side_effect=safe):d,p,rows=tr.load_data(root,regions=('houston',),expected={'train':12,'val':12})
            self.assertEqual(opened,['train.npz','val.npz']);self.assertEqual(len(rows),24)
    def test_tiny_training_fixed_final_and_best_saved(self):
        data={s:synthetic_data(12,s) for s in ['train','val']};stats=tr.fit_normalizer(data['train']['features']['raw_legacy7'],data['train']['mask'])
        training={'epochs':2,'batch_size':8,'learning_rate':.002,'weight_decay':.01}
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'run';r=tr.train_one('tcn','raw_legacy_compat',0,data,stats,training,torch.device('cpu'),out,'test-protocol','test-config')
            self.assertEqual(r['primary_epoch'],2);self.assertEqual(r['gradient_training_windows'],8)
            self.assertIn(r['best_epoch'],[1,2]);self.assertTrue((out/'checkpoint_final.pt').exists());self.assertTrue((out/'checkpoint_best.pt').exists())
            with np.load(out/'epoch_validation_predictions.npz') as z:self.assertEqual(z['scores'].shape,(2,12))

if __name__=='__main__':unittest.main()
