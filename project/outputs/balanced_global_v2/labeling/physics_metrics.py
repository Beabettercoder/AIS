"""Report distinct, dimensionally correct physics quantities on actual AIS times."""
import numpy as np
KNOT=1852/3600
def physics_metrics(t,pos,sog_kn,cog_deg,max_gap_s=300,min_direction_speed_kn=.5):
 t=np.asarray(t,float);p=np.asarray(pos,float);s=np.asarray(sog_kn,float);c=np.deg2rad(np.asarray(cog_deg,float))
 if t.ndim!=1 or p.shape!=(len(t),2) or s.shape!=t.shape or c.shape!=t.shape or np.any(np.diff(t)<=0):raise ValueError('Invalid observed sequence')
 dt=np.diff(t);v=KNOT*s[:,None]*np.column_stack((np.sin(c),np.cos(c)))
 wrap=(np.diff(c)+np.pi)%(2*np.pi)-np.pi
 interval=(dt>0)&(dt<=max_gap_s)
 direction=interval&(s[1:]>=min_direction_speed_kn)&(s[:-1]>=min_direction_speed_kn)
 err=np.diff(p,axis=0)/dt[:,None]-(v[1:]+v[:-1])/2
 return {'pair_interval_valid':interval,'pair_direction_valid':direction,'dt_s':dt,
  'kinematic_innovation_vector_mps':err,'e_kin_mps':np.linalg.norm(err,axis=1),
  'delta_velocity_mps':np.linalg.norm(np.diff(v,axis=0),axis=1),
  'longitudinal_acceleration_mps2':np.diff(s)*KNOT/dt,
  'course_rate_deg_s':np.rad2deg(wrap)/dt,
  'lateral_acceleration_mps2':(s[1:]+s[:-1])/2*KNOT*np.abs(wrap)/dt,
  'metric_semantics':'Velocity increments, acceleration, course rate and position/velocity innovation are separate quantities. Course rate is not directly measured hull yaw rate.'}
