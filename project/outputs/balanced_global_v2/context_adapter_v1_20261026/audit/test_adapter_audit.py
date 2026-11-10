import unittest
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from audit_helpers import threshold_from_counts,metric_set,region_metrics,require
from independent_adapter_math import derive,fit_adapter_scaler,normalized_adapter,reference_delta,reference_pool
from audit_adapters import expected_initial_adapter,expected_orders,summarize


def source_fixture():
    rng=np.random.default_rng(3)
    mask=np.array([[1,1,1,0],[1,1,0,0]],bool)
    x=np.zeros((2,4,16),np.float32)
    x[:,:,:7]=rng.normal(size=(2,4,7));x[:,:,9:12]=rng.normal(size=(2,4,3));x[~mask]=0
    d=dict(mask=mask,context_valid=np.array([[1,1,1,0],[0,1,0,0]],bool),
           plain_affine_valid=np.array([[1,0,1,0],[1,1,0,0]],bool),
           plain_context_valid=mask.copy(),raw_meta=x.copy())
    for name,offset in [('mean_meta',1),('affine_meta',2),('reliable_raw',3)]:
        d[name]=x.copy();d[name][:,:,7:9]=offset;d[name][:,:,14:16]=offset if name!='mean_meta' else 0
        d[name][~mask]=0
    return d


class AuditTests(unittest.TestCase):
    def test_threshold_ties_agree_exhaustive(self):
        y=np.array([1,0,1,0,1,0]);p=np.array([.9,.9,.6,.6,.2,.1])
        choices=[]
        for t in sorted(set(p)):
            q=p>=t;tp=int((q&(y==1)).sum());fp=int((q&(y==0)).sum());fn=int((~q&(y==1)).sum())
            choices.append((2*tp/(2*tp+fp+fn),t))
        score,t=max(choices)
        got,gs=threshold_from_counts(y,p)
        self.assertEqual((got,gs),(t,score))

    def test_single_class_preserves_counts_without_ranking(self):
        r=metric_set(np.array([0,0]),np.array([.2,.9]),.5)
        self.assertIsNone(r['AP']);self.assertIsNone(r['AUROC']);self.assertEqual(r['FPR'],.5)
        macro=region_metrics(np.array([0,0,1,0]),np.array([.2,.9,.8,.1]),np.array(['a','a','b','b']),.5)
        self.assertEqual(macro['metric_contributors']['AP'],1)

    def test_derived_common_support_and_only_allowed_differences(self):
        d=source_fixture();arms,c=derive(d)
        self.assertTrue(np.array_equal(c,np.array([[1,0,1,0],[0,1,0,0]],bool)))
        for x in arms.values():
            np.testing.assert_array_equal(x[:,:,12],c)
            np.testing.assert_array_equal(x[:,:,13],c)
            np.testing.assert_array_equal(x[:,:,:7],arms['self_quality'][:,:,:7])
        self.assertTrue(np.all(arms['self_quality'][:,:,[7,8,14,15]]==0))

    def test_reject_shared_slot_tampering(self):
        d=source_fixture();d['mean_meta'][0,0,1]+=1
        with self.assertRaises(AssertionError):derive(d)

    def test_scaler_uses_supported_training_rows_only(self):
        x=np.array([[[1.,2.],[3.,6.],[1e9,-1e9]]],np.float32);support=np.array([[1,1,0]],bool)
        s=fit_adapter_scaler(x,support)
        np.testing.assert_array_equal(s['mean'],[2.,4.]);np.testing.assert_array_equal(s['std'],[1.,2.])
        self.assertEqual(s['valid_steps'],2)
        z=normalized_adapter(x,support,s);np.testing.assert_array_equal(z[0,2],[0,0])

    def test_delta_divides_total_valid_steps_and_bypasses_zero_support(self):
        state=expected_initial_adapter(0);state['last.bias']=torch.ones(64)
        x=torch.zeros(2,4,16);mask=torch.tensor([[1,1,1,0],[1,1,0,0]],dtype=torch.bool)
        support=torch.tensor([[1,0,0,0],[0,0,0,0]],dtype=torch.bool)
        delta=reference_delta(x,mask,support,state)
        torch.testing.assert_close(delta[0],torch.full((64,),1/3));self.assertTrue(torch.equal(delta[1],torch.zeros(64)))

    def test_zero_last_layer_still_receives_gradient(self):
        state={k:v.clone().requires_grad_() for k,v in expected_initial_adapter(1).items()}
        x=torch.randn(3,4,16);mask=torch.ones(3,4,dtype=torch.bool)
        delta=reference_delta(x,mask,mask,state)
        self.assertTrue(torch.equal(delta,torch.zeros_like(delta)))
        loss=F.binary_cross_entropy_with_logits(delta.sum(1),torch.zeros(3));loss.backward()
        self.assertGreater(float(state['last.bias'].grad.norm()),0)
        self.assertTrue(torch.equal(state['first.weight'].grad,torch.zeros_like(state['first.weight'])))

    def test_masked_pool_ignores_padding_and_exact_initialization_pairing(self):
        x=torch.tensor([[[1.,3.],[3.,1.],[1e9,1e9]]]);m=torch.tensor([[1,1,0]],dtype=torch.bool)
        torch.testing.assert_close(reference_pool(x,m),torch.tensor([[2.,2.,3.,3.]]))
        a=expected_initial_adapter(4);b=expected_initial_adapter(4)
        self.assertTrue(all(torch.equal(a[k],b[k]) for k in a));self.assertEqual(expected_orders(100,4,12),expected_orders(100,4,12))


if __name__=='__main__':unittest.main()
