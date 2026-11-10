import unittest
import numpy as np
from sklearn.metrics import precision_recall_curve
from audit_component_stage import threshold_from_counts,metric_set,region_metrics,distribution,summaries,as_bool,exploratory_support_strata,stratified_summaries

class AuditTests(unittest.TestCase):
    def test_threshold_independent_cumulative_counts_matches_reference(self):
        rng=np.random.default_rng(81)
        for _ in range(30):
            y=rng.integers(0,2,200);p=np.round(rng.random(200),2)
            precision,recall,t=precision_recall_curve(y,p)
            f=2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-15)
            i=np.flatnonzero(np.isclose(f,f.max(),rtol=0,atol=1e-12))[-1]
            threshold,best=threshold_from_counts(y,p)
            self.assertEqual(threshold,t[i]);self.assertAlmostEqual(best,f[i],places=14)

    def test_single_class_ranking_null_and_rates_present(self):
        r=metric_set(np.zeros(4,int),np.array([.1,.2,.3,.9]),.5)
        self.assertIsNone(r['AP']);self.assertIsNone(r['AUROC']);self.assertEqual(r['FPR'],.25)
        self.assertIsNone(r['recall']);self.assertEqual(r['ranking_metric_status'],'single_class_undefined')

    def test_empty_strata_no_fabricated_metrics(self):
        r=region_metrics(np.array([],int),np.array([]),np.array([],str),.5)
        self.assertIsNone(r['macro']['AP']);self.assertEqual(r['metric_contributors']['AP'],0)
        self.assertIsNone(r['pooled']['F1'])

    def test_region_macro_does_not_weight_large_regions(self):
        y=np.array([0,1,0,0,0,1]);p=np.array([.1,.9,.9,.8,.7,.1]);g=np.array(['a','a','b','b','b','b'])
        r=region_metrics(y,p,g,.5)
        self.assertAlmostEqual(r['macro']['AP'],.625)

    def test_sample_std_not_population(self):
        self.assertEqual(distribution([1.,2.,3.])['sample_std'],1.)
        self.assertIsNone(distribution([1.])['sample_std'])

    def test_interaction_uses_all_four_cells(self):
        names=['legacy_full','reliable_gate_target','reliable_gate_residual','reliable_raw']
        values=[.8,.7,.65,.6]
        result=[dict(backbone='tcn',feature_group=f,seed=0,validation={'macro':{m:v for m in ['AP','AUROC','F1','precision','recall']}}) for f,v in zip(names,values)]
        p={'backbones':['tcn'],'feature_names':names,'training':{'seeds':[0]},'predeclared_contrasts':[]}
        s=summaries(result,p)
        self.assertAlmostEqual(s['gating_2x2_interaction'][0]['AP_interaction']['mean'],.05)

    def test_boolean_strings_are_not_python_truthiness(self):
        self.assertFalse(as_bool('False'));self.assertFalse(as_bool('0'));self.assertTrue(as_bool('True'))
        with self.assertRaises(ValueError):as_bool('maybe')

    def test_support_bins_use_valid_steps_and_exact_half_boundary(self):
        val={'mask':np.array([[1,1,0,0],[1,1,1,1],[1,1,0,0]],bool),
             'context_valid':np.array([[0,0,1,1],[1,0,0,0],[1,0,0,0]],bool),
             'y':np.array([0,1,1]),'region':np.array(['a','a','a'])}
        s=exploratory_support_strata(val,np.array([.1,.8,.9]),.5)
        self.assertEqual([v['pooled']['samples'] for v in s['groups'].values()],[1,1,1])
        self.assertIsNone(s['groups']['at_least_half']['macro']['AP'])

    def test_stratified_seed_summary_preserves_unavailable_ranking(self):
        m=region_metrics(np.array([0,0]),np.array([.1,.9]),np.array(['a','a']),.5)
        runs=[dict(backbone='tcn',feature_group='raw16',seed=s,
                   validation_strata={'noise_type':{'none':m}},exploratory_support_strata={'groups':{}}) for s in range(3)]
        row=stratified_summaries(runs)['rows'][0]
        self.assertIsNone(row['AP']['mean']);self.assertEqual(row['AP']['n'],0)
        self.assertEqual(row['AP']['contributing_regions_per_seed'],[0,0,0])
        self.assertEqual(row['FPR']['mean'],.5)

if __name__=='__main__':unittest.main()
