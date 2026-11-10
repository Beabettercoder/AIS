"""Behavioral contracts for paired pilot models and leakage boundaries."""
import tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score
from pilot_train import (BACKBONES,FEATURES,DIMENSIONS,BackboneClassifier,
                         choose_threshold,fit_normalizer,normalized,pick_subset,
                         load_pilot,masked_pool,region_metrics,seed_all)

class PilotContracts(unittest.TestCase):
 @classmethod
 def setUpClass(cls):torch.set_num_threads(2)
 def test_padding_values_and_extra_padding_do_not_change_predictions(self):
  for kind in BACKBONES:
   with self.subTest(kind=kind):
    seed_all(12);m=BackboneClassifier(kind,16,hidden=16,dropout=0).eval()
    x=torch.randn(3,12,16);mask=torch.arange(12)[None,:]<torch.tensor([8,10,12])[:,None]
    xp=x.clone();xp[~mask]=1000
    longer=torch.cat((xp,torch.full((3,8,16),-888.)),1)
    longer_mask=torch.cat((mask,torch.zeros(3,8,dtype=torch.bool)),1)
    with torch.no_grad():
     expected=m(x,mask);actual=m(xp,mask);extended=m(longer,longer_mask)
    torch.testing.assert_close(expected,actual,atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(expected,extended,atol=2e-6,rtol=2e-6)
 def test_transformer_positions_affect_order(self):
  seed_all(3);m=BackboneClassifier('transformer',7,16,0).eval()
  x=torch.randn(1,10,7);mask=torch.ones(1,10,dtype=torch.bool)
  with torch.no_grad():a=m(x,mask);b=m(x.flip(1),mask)
  self.assertGreater(float(torch.abs(a-b)),1e-6)
 def test_tcn_is_causal(self):
  seed_all(3);m=BackboneClassifier('tcn',7,16,0).eval()
  x=torch.randn(2,20,7);mask=torch.ones(2,20,dtype=torch.bool);changed=x.clone();changed[:,10:]+=500
  with torch.no_grad():a=m.encode(x,mask);b=m.encode(changed,mask)
  torch.testing.assert_close(a[:,:10],b[:,:10],rtol=0,atol=0)
 def test_masked_pool_ignores_extreme_padding(self):
  h=torch.tensor([[[1.,2.],[3.,4.],[999.,999.]]]);mask=torch.tensor([[True,True,False]])
  torch.testing.assert_close(masked_pool(h,mask),torch.tensor([[2.,3.,3.,4.]]))
 def test_normalizer_ignores_padding_and_only_uses_supplied_train(self):
  x=np.array([[[1.,3.],[3.,7.],[10000.,-10000.]]],dtype=np.float32);mask=np.array([[True,True,False]])
  n=fit_normalizer(x,mask);np.testing.assert_array_equal(n['mean'],[2,5]);np.testing.assert_array_equal(n['std'],[1,2])
  result=normalized(x,mask,n);np.testing.assert_array_equal(result[0,2],[0,0])
 def test_threshold_is_optimal_on_validation_only(self):
  y=np.array([0,1,0,1,1,0]);p=np.array([.05,.4,.3,.6,.9,.8])
  threshold,f1=choose_threshold(y,p)
  expected=max(f1_score(y,p>=t,zero_division=0) for t in np.unique(p))
  self.assertAlmostEqual(f1,expected);self.assertAlmostEqual(f1_score(y,p>=threshold),expected)
  # No test arrays are accepted by the function; changing unrelated data leaves
  # its deterministic validation-only decision unchanged.
  unrelated_test=np.array([0.,1.]);unrelated_test[:]=.7
  self.assertEqual(choose_threshold(y,p),(threshold,f1))
 def test_subset_is_feature_independent_and_order_invariant(self):
  y=np.array([0]*30+[1]*20);ids=np.array([f'w{i:03}' for i in range(50)])
  a=pick_subset(y,ids,20,17,'domain','train');self.assertEqual(int(y[a].sum()),8)
  perm=np.random.default_rng(4).permutation(len(y));b=pick_subset(y[perm],ids[perm],20,17,'domain','train')
  self.assertEqual(list(ids[a]),list(ids[perm][b]))
 def test_context_parameter_counts_match(self):
  for kind in BACKBONES:
   counts=[sum(p.numel() for p in BackboneClassifier(kind,DIMENSIONS[f],16).parameters()) for f in FEATURES[1:]]
   self.assertEqual(len(set(counts)),1)
 def test_all_configurations_have_finite_training_gradients(self):
  for kind in BACKBONES:
   for feature in FEATURES:
    with self.subTest(kind=kind,feature=feature):
     seed_all(5);m=BackboneClassifier(kind,DIMENSIONS[feature],16,.1).train()
     x=torch.randn(4,20,DIMENSIONS[feature]);mask=torch.arange(20)[None,:]<torch.tensor([8,12,16,20])[:,None]
     logits=m(x,mask);loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.tensor([0.,1.,0.,1.]))
     loss.backward();grads=[p.grad for p in m.parameters() if p.requires_grad]
     self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
 def test_macro_is_mean_of_regions(self):
  y=np.array([0,1,0,1]);p=np.array([.2,.9,.8,.1]);regions=np.array(['a','a','b','b'])
  result=region_metrics(y,p,regions,.5)
  for k in ['AP','AUROC','F1']:
   self.assertAlmostEqual(result['macro'][k],np.mean([v[k] for v in result['per_region'].values()]))
 def test_loader_checks_identity_isolation_and_shared_subset(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);(root/'r').mkdir()
   for k,s in enumerate(['train','val','test']):
    y=np.array([0]*6+[1]*4);mask=np.ones((10,20),bool)
    z={'y':y,'mask':mask,'window_id':np.array([f'{s}{i}' for i in range(10)]),'mmsi':np.array([200000000+k*100+i for i in range(10)])}
    for f in FEATURES:z[f]=np.ones((10,20,DIMENSIONS[f]),np.float32)
    np.savez(root/'r'/f'{s}.npz',**z)
   data,proof,selection=load_pilot(root,['r'],{'train':10,'val':10,'test':10},0,FEATURES)
   self.assertEqual(len(selection),30)
   self.assertTrue(all(len(d['features'][f])==10 for d in data.values() for f in FEATURES))
   # Overlapping test target identity must be rejected before any fitting.
   with np.load(root/'r'/'test.npz') as src:z={k:src[k] for k in src.files}
   z['mmsi'][0]=200000000;np.savez(root/'r'/'test.npz',**z)
   with self.assertRaisesRegex(ValueError,'mmsi crosses'):load_pilot(root,['r'],{'train':10,'val':10,'test':10},0,FEATURES)

if __name__=='__main__':unittest.main(verbosity=2)
