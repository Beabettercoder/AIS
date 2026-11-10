import unittest
import numpy as np
from audit_neighborhood import Source,kin,geometry,coverage_ok,NM

class Check(unittest.TestCase):
 def test_geometry_needs_four_and_two_spatial_directions(self):
  q=np.array([[0.,0.],[500.,0.],[0.,500.],[500.,500.]])
  self.assertFalse(geometry(q[:3],np.ones(3),8*NM)[0])
  self.assertTrue(geometry(q,np.ones(4),8*NM)[0])
  self.assertEqual(geometry(np.column_stack((np.arange(4)*500.,np.zeros(4))),np.ones(4),8*NM)[1],'degenerate_geometry')
 def test_boundary_depends_on_actual_radius(self):
  p=np.array([4000.,0.]);box=[-10000,-10000,10000,10000]
  self.assertFalse(coverage_ok(p,box,8*NM));self.assertTrue(coverage_ok(p,box,3*NM))
  self.assertTrue(coverage_ok(p,None,8*NM))
 def test_kinematic_corruption_has_two_step_effect(self):
  t=np.array([0.,70.,140.,210.]);s=np.ones(4)*10;c=np.ones(4)*90
  p=np.column_stack((t*10*1852/3600,np.zeros(4)));q=p.copy();q[1,0]+=1000
  self.assertTrue(np.allclose(kin(t,p,s,c)[-1][1:],1))
  w=kin(t,q,s,c)[-1];self.assertLess(w[1],.3);self.assertLess(w[2],.3);self.assertEqual(w[3],1)
 def test_latest_is_causal_unique_and_allowed(self):
  z=Source.__new__(Source);z.ids=np.array(['A','A','B','C']);z.t=np.array([0.,100.,90.,101.]);z.time_order=np.argsort(z.t);z.sorted_t=z.t[z.time_order]
  ii=z.latest(100,'B',{'A','B','C'});self.assertEqual(ii.tolist(),[1])
 def test_overlay_replays_recorded_teleport(self):
  z=Source.__new__(Source);z.ids=np.repeat('A',4);z.t=np.array([0.,70.,140.,210.]);z.p=np.column_stack((z.t*10*1852/3600,np.zeros(4)));z.s=np.ones(4)*10;z.c=np.ones(4)*90;z.ship={'A':np.arange(4)}
  z.good,z.valid,z.e,z.w=kin(z.t,z.p,z.s,z.c);z.lookup={('A',float(t)):i for i,t in enumerate(z.t)}
  md={'decision_time_s':210.,'neighbor_observation_noise':{'changes':[{'effective':True,'mmsi':'A','noise':'teleport','affected_time_s':[70.],'parameters':{'report_index':1,'time_s':70.,'offset_m':1000.,'angle_deg':90.}}]}}
  over=z.overlay(md);self.assertAlmostEqual(over[1][0][0],z.p[1,0]+1000);self.assertLess(over[2][-1],.3);self.assertEqual(z.p[1,0],70*10*1852/3600)
 def test_overlay_does_not_bridge_invalid_source_report(self):
  z=Source.__new__(Source);z.ids=np.repeat('A',5);z.t=np.arange(5)*70.;z.p=np.column_stack((z.t*10*1852/3600,np.zeros(5)));z.s=np.ones(5)*10;z.c=np.ones(5)*90;z.s[2]=np.nan;z.c[2]=np.nan;z.ship={'A':np.arange(5)}
  z.good,z.valid,z.e,z.w=kin(z.t,z.p,z.s,z.c);z.lookup={('A',float(t)):i for i,t in enumerate(z.t)}
  md={'decision_time_s':280.,'neighbor_observation_noise':{'changes':[{'effective':True,'mmsi':'A','noise':'teleport','affected_time_s':[70.],'parameters':{'report_index':1,'time_s':70.,'offset_m':1000.,'angle_deg':90.}}]}}
  over=z.overlay(md);self.assertFalse(over[3][3]);self.assertEqual(over[3][-1],0.);self.assertEqual(over[3][-2],0.)

if __name__=='__main__':unittest.main()
