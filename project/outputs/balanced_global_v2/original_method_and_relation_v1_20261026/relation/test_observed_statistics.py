import unittest
import numpy as np
from observed_statistics import score_observed,kinematic_weights

def fixture():
    t=np.arange(12,dtype=float)*60
    base=np.ones(12)*10;slow=base.copy();slow[6:]=7
    def pos(s):
        v=s*1852/3600
        y=np.r_[0,np.cumsum((v[1:]+v[:-1])/2*60)]
        return np.column_stack((np.zeros(12),y))
    target_s=np.stack([base,base,slow,slow]);neighbor_s=np.stack([np.tile(base,4),np.tile(slow,4),np.tile(base,4),np.tile(slow,4)])
    return dict(target_t=np.tile(t,(4,1)),target_pos=np.stack([pos(s) for s in target_s]),target_sog_kn=target_s,target_cog_deg=np.zeros((4,12)),
                neighbor_t=np.tile(np.tile(t,4),(4,1)),neighbor_pos=np.stack([np.concatenate([pos(s[j*12:(j+1)*12])+[j*100,0] for j in range(4)]) for s in neighbor_s]),
                neighbor_sog_kn=neighbor_s,neighbor_cog_deg=np.zeros((4,48)),member_offsets=np.arange(0,49,12))

class Tests(unittest.TestCase):
    def test_quartet_and_marginals(self):
        r=score_observed(fixture())
        np.testing.assert_allclose([x['relative_mean'] for x in r],[0,.3,.3,0],atol=1e-12)
        np.testing.assert_allclose([x['relative_reliable'] for x in r],[0,.3,.3,0],atol=1e-12)
        self.assertEqual(r[0]['target_only'],r[1]['target_only']);self.assertEqual(r[0]['neighbor_only'],r[2]['neighbor_only'])
    def test_future_neighbor_report_ignored(self):
        o=fixture();before=score_observed(o)
        for k in ['neighbor_t','neighbor_pos','neighbor_sog_kn','neighbor_cog_deg']:
            parts=[]
            for j in range(4):
                piece=o[k][:,12*j:12*(j+1)]
                if k=='neighbor_t':extra=np.ones((4,1))*10000
                elif k=='neighbor_pos':extra=np.ones((4,1,2))*1e8
                else:extra=np.ones((4,1))*99
                parts.append(np.concatenate((piece,extra),axis=1))
            o[k]=np.concatenate(parts,axis=1)
        o['member_offsets']=np.arange(0,53,13)
        after=score_observed(o)
        self.assertEqual(before,after)
    def test_oracle_keys_rejected(self):
        o=fixture();o['y']=np.array([0,1,1,0])
        with self.assertRaises(ValueError):score_observed(o)
    def test_position_teleport_does_not_change_report_speed_score(self):
        o=fixture();before=score_observed(o)
        o['neighbor_pos'][:,8,0]+=1000
        after=score_observed(o)
        for a,b in zip(before,after):self.assertEqual(a['relative_mean'],b['relative_mean'])
        self.assertLess(after[0]['member_weights'][0],before[0]['member_weights'][0])
    def test_invalid_predecessor_interrupts_kinematics(self):
        t=np.arange(5,dtype=float)*60;s=np.array([10.,10.,np.nan,10.,10.]);c=np.zeros(5);p=np.column_stack((np.zeros(5),t*10*1852/3600))
        w=kinematic_weights(t,p,s,c);np.testing.assert_array_equal(w,[0,1,0,0,1])

if __name__=='__main__':unittest.main()
