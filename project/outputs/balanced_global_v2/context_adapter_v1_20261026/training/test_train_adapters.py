import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch.nn import functional as F
import train_adapters as m


def fixture(n=12,region='houston',offset=0):
    rng=np.random.default_rng(42+offset)
    mask=np.zeros((n,20),bool)
    for i in range(n):mask[i,:8+i%9]=True
    raw=np.zeros((n,20,16),np.float32);raw[...,:7]=rng.normal(size=(n,20,7));raw[~mask]=0
    kin=rng.normal(size=(n,20,3)).astype(np.float32);kin[~mask]=0
    z=dict(raw16=raw,mask=mask,context_valid=mask.copy(),plain_affine_valid=mask.copy(),plain_context_valid=mask.copy(),
           y=np.arange(n,dtype=np.int64)%2)
    z['context_valid'][0]=False
    for source in m.ARM_SOURCES.values():
        x=raw.copy();x[...,9:12]=kin
        if source!='raw_meta':
            x[...,7:9]=rng.normal(size=(n,20,2));x[...,14:16]=rng.normal(size=(n,20,2))
        x[~mask]=0;z[source]=x
    for key in m.META:
        if key=='mmsi':z[key]=np.array([str(offset+i+100000000) for i in range(n)])
        elif key in ('window_id','base_window_id'):z[key]=np.array([f'w{offset+i}' for i in range(n)])
        elif key=='region':z[key]=np.repeat(region,n)
        elif key.endswith('_effective'):z[key]=np.repeat(False,n)
        else:z[key]=np.repeat('none',n)
    return z


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_common_support_and_exact_arm_difference(self):
        z=fixture();z['plain_affine_valid'][1,2]=False;z['plain_context_valid'][2,3]=False
        raw,mask,support,arms=m.construct_arms(z)
        self.assertFalse(support[0].any());self.assertFalse(support[1,2]);self.assertFalse(support[2,3])
        for x in arms.values():
            np.testing.assert_array_equal(x[...,:7],raw[...,:7])
            np.testing.assert_array_equal(x[...,12],support)
            np.testing.assert_array_equal(x[...,13],support)
        z['reliable_raw'][2,1,0]+=1
        with self.assertRaises(ValueError):m.construct_arms(z)

    def test_only_train_val_opened_and_identity_isolation(self):
        with tempfile.TemporaryDirectory() as temp:
            r=Path(temp);folder=r/'data/houston';folder.mkdir(parents=True)
            for split,offset in [('train',0),('val',100)]:np.savez(folder/(split+'.npz'),**fixture(offset=offset))
            (folder/'test.npz').write_bytes(b'forbidden fixture must never be opened')
            original=np.load;opened=[]
            def guarded(path,*a,**kw):
                opened.append(Path(path).name)
                self.assertNotEqual(Path(path).name,'test.npz')
                return original(path,*a,**kw)
            with patch.object(m.np,'load',guarded):
                d,_,_=m.load_data(r,regions=('houston',),counts={'train':12,'val':12})
            self.assertEqual(opened,['train.npz','val.npz'])
            self.assertFalse(set(d['train']['mmsi']) & set(d['val']['mmsi']))
            np.savez(folder/'val.npz',**fixture(offset=0))
            with self.assertRaises(ValueError):m.load_data(r,regions=('houston',),counts={'train':12,'val':12})

    def test_scaler_uses_supported_train_rows_and_clears_other_rows(self):
        z=fixture();_,_,support,arms=m.construct_arms(z);x=arms['mean_context']
        expected=x[support].astype(np.float64)
        changed=x.copy();changed[~support]=1e9
        a=m.fit_adapter_scaler(x,support);b=m.fit_adapter_scaler(changed,support)
        np.testing.assert_array_equal(a['mean'],b['mean']);np.testing.assert_allclose(a['mean'],expected.mean(0),rtol=1e-6,atol=1e-6)
        normalized=m.adapter_normalized(changed,support,a)
        self.assertTrue(np.all(normalized[~support]==0))
        with self.assertRaises(ValueError):m.fit_adapter_scaler(x,np.zeros_like(support))

    def test_zero_init_has_learnable_output_gradient_and_frozen_head(self):
        m.base.seed_all(1);adapter=m.ResidualAdapter();model=m.base.BackboneClassifier('tcn',16,32,.1).eval().requires_grad_(False)
        state=m.digest_state(model.state_dict());x=torch.randn(6,20,16);mask=torch.ones(6,20,dtype=torch.bool)
        h=torch.randn(6,64);delta=adapter(x,mask,mask)
        self.assertTrue(torch.equal(delta,torch.zeros_like(delta)))
        loss=F.binary_cross_entropy_with_logits(model.head(h+delta).squeeze(-1),torch.tensor([0.,1.,0.,1.,0.,1.]))
        loss.backward()
        self.assertGreater(float(adapter.last.weight.grad.norm()),0)
        self.assertEqual(float(adapter.first.weight.grad.norm()),0)
        self.assertEqual(state,m.digest_state(model.state_dict()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_unsupported_and_padding_cannot_contribute_delta(self):
        adapter=m.ResidualAdapter()
        with torch.no_grad():adapter.last.weight.fill_(.2);adapter.last.bias.fill_(.3)
        x=torch.randn(3,20,16);mask=torch.zeros(3,20,dtype=torch.bool);mask[:,:8]=True
        support=mask.clone();support[0]=False
        d1=adapter(x,mask,support)
        altered=x.clone();altered[~support]=10000
        d2=adapter(altered,mask,support)
        self.assertTrue(torch.equal(d1,d2));self.assertTrue(torch.equal(d1[0],torch.zeros(64)))

    def test_pairing_and_epoch_zero_tie(self):
        hashes=[]
        for _ in range(2):
            m.base.seed_all(2001);hashes.append(m.digest_state(m.ResidualAdapter().state_dict()))
        self.assertEqual(hashes[0],hashes[1])
        a=m.make_orders(37,1,12);b=m.make_orders(37,1,12)
        self.assertEqual(a[1:],b[1:]);self.assertNotEqual(a[1],m.make_orders(37,2,12)[1])
        self.assertEqual(m.select_epoch([dict(epoch=0,validation_macro_AP=.7),dict(epoch=1,validation_macro_AP=.7)]),0)

    def test_parent_cache_matches_direct_prediction(self):
        for backbone in ('tcn','gru','transformer'):
            z=fixture();raw,mask,support,arms=m.construct_arms(z)
            model=m.base.BackboneClassifier(backbone,16,32,.1).eval().requires_grad_(False)
            scaler=m.base.fit_normalizer(raw,mask)
            d={'raw':raw,'mask':mask};cache=m.encode_cache(model,d,scaler,4)
            direct=m.base.predict(model,torch.from_numpy(m.base.normalized(raw,mask,scaler)),torch.from_numpy(mask),4)
            np.testing.assert_array_equal(direct,cache['baseline_scores'])

    def test_synthetic_training_identity_selection_and_replay(self):
        with tempfile.TemporaryDirectory() as temp:
            data={}
            for split,offset in [('train',0),('val',100)]:
                z=fixture(n=16,offset=offset);raw,mask,support,arms=m.construct_arms(z)
                data[split]=dict(y=z['y'],raw=raw,mask=mask,common_support=support,trusted_support=mask,features=arms)
                data[split].update({k:z[k].astype(str) for k in m.META})
            model=m.base.BackboneClassifier('tcn',16,32,.1).eval().requires_grad_(False)
            rawscale=m.base.fit_normalizer(data['train']['raw'],data['train']['mask'])
            cache={s:m.encode_cache(model,d,rawscale,4) for s,d in data.items()}
            proof=dict(state_sha256=m.digest_state(model.state_dict()),checkpoint_sha256='synthetic',raw_normalizer_mean=rawscale['mean'].tolist(),raw_normalizer_std=rawscale['std'].tolist())
            train=dict(epochs=2,batch_size=4,learning_rate=.001,weight_decay=.01,gradient_clip_norm=5.)
            scale=m.fit_adapter_scaler(data['train']['features']['self_quality'],data['train']['common_support'])
            result=m.train_arm(model,cache,data,'self_quality',0,train,scale,Path(temp)/'run',proof,{},'synthetic','synthetic')
            self.assertEqual(len(result['history']),3)
            self.assertGreaterEqual(result['validation']['macro']['AP'],result['baseline_validation']['macro']['AP'])
            self.assertTrue(all(v==0 for k,v in result['identity_checks'].items() if 'error' in k))
            self.assertEqual(result['frozen_state_sha256_before'],result['frozen_state_sha256_after'])
            loaded=m.resume_run(Path(temp)/'run',result['run_spec']);self.assertEqual(loaded['selected_epoch'],result['selected_epoch'])
            with (Path(temp)/'run/history.json').open('a') as f:f.write(' ')
            with self.assertRaises(ValueError):m.resume_run(Path(temp)/'run',result['run_spec'])


if __name__=='__main__':unittest.main(verbosity=2)
