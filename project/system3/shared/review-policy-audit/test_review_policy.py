import math
import numpy as np
import pytest

from anomaly_workbench.review_policy import assess_review_evidence


def record(speed, xy=None, times=None, heading=None, kind='speed_drop'):
    speed = np.asarray(speed, float)
    times = np.arange(len(speed))*60. if times is None else np.asarray(times, float)
    heading = np.zeros(len(speed)) if heading is None else np.asarray(heading, float)
    if xy is None:
        distance = .5*(speed[1:]+speed[:-1])*.514444*np.diff(times)
        xy = np.c_[np.zeros(len(speed)), np.r_[0., np.cumsum(distance)]]
    return {'is_candidate':True, 'type':kind, 'score':.87,
            'maritime_policy':{'initial_sog_kn':float(np.median(speed[:3])), 'supported_types':[kind],
                               'neighbor_supported':False, 'scoring_neighbor_count':0,'valid_local_steps':0},
            'motion_observations':{'sog_kn':speed.tolist(), 'cog_deg':heading.tolist(),
                                   'time_seconds':times.tolist(), 'xy_m':np.asarray(xy).tolist(),
                                   'gate_weights':np.ones(len(speed)).tolist()}}


def test_sustained_speed_drop_uses_stable_segments_not_single_point():
    r = record([10,10,10,10,1,1,1,1,1])
    result = assess_review_evidence(r)
    assert result['eligible'] and result['review_type']=='stall'
    change = result['evidence']['deceleration']
    assert change['before_median_kn']==10 and change['after_median_kn']==1
    assert change['transition_seconds']==60 and change['low_speed_seconds']>=90
    assert r['score']==.87


def test_a_speed_spike_does_not_supply_a_stable_underway_baseline():
    r = record([1,1,1,10,1,1,1,1,1])
    assert not assess_review_evidence(r)['eligible']


def test_gradual_arrival_is_only_a_movement_clue_without_independent_context():
    r = record([10,10,10,9,8,7,6,5,4,3,2,1,1,1],kind='stall')
    assert not assess_review_evidence(r)['eligible']


def test_brief_drop_without_90_second_low_speed_state_is_not_focused():
    r = record([10,10,10,1,1,10,10,10])
    assert not assess_review_evidence(r)['eligible']


def test_low_speed_vessel_does_not_qualify_by_large_relative_drop():
    r = record([3,3,3,3,.1,.1,.1,.1])
    assert not assess_review_evidence(r)['eligible']


def test_position_jump_cannot_connect_opposite_moving_legs():
    r = record([10]*8, heading=[0,0,0,0,180,180,180,180],kind='turn_back')
    r['motion_observations']['xy_m'][4]=[100000,100000]
    r['motion_observations']['gate_weights'][4]=0
    assert not assess_review_evidence(r)['eligible']


def test_actual_stable_direction_reversal_with_only_two_points_per_leg():
    times = [0,100,200,300,400]
    xy = [[0,0],[0,500],[100,700],[100,200],[100,-300]]
    heading = [0,0,160,180,180]
    r = record([9.72]*5,xy=xy,times=times,heading=heading,kind='turn_back')
    result=assess_review_evidence(r)
    assert result['eligible'] and result['review_type']=='turn_back'
    assert result['evidence']['turnback_stable_angle_deg']>=150


def test_wide_normal_course_change_alone_is_not_focused():
    xy = np.array([[0,0],[0,300],[50,600],[150,850],[350,1000],[600,1100]],float)
    times=np.arange(len(xy))*90
    step=np.linalg.norm(np.diff(xy,axis=0),axis=1)/90/.514444
    speed=np.r_[step[0],step]
    heading=np.r_[0,np.rad2deg(np.arctan2(np.diff(xy[:,0]),np.diff(xy[:,1])))%360]
    r=record(speed,xy=xy,times=times,heading=heading,kind='deviation')
    assert not assess_review_evidence(r)['eligible']


def test_sinuous_requires_large_alternating_geometry_not_reported_heading_noise():
    r=record([8]*12,heading=[0,80,350,60,340,70,350,50,340,60,350,0],kind='sinuous')
    assert not assess_review_evidence(r)['eligible']


def test_real_alternating_motion_qualifies_without_neighbors():
    xy=np.array([[0,0],[150,150],[-50,350],[150,550],[-50,750],[150,950],[-50,1150]],float)
    times=np.arange(len(xy))*60.
    speed=np.full(len(xy),9.)
    headings=np.r_[45,np.rad2deg(np.arctan2(np.diff(xy[:,0]),np.diff(xy[:,1])))%360]
    r=record(speed,xy=xy,times=times,heading=headings,kind='sinuous')
    result=assess_review_evidence(r)
    assert result['eligible'] and result['review_type']=='sinuous'
    assert result['evidence']['sinuous']['alternating_turns']>=3


def test_missing_original_observations_cannot_gain_priority_from_model_score():
    r={'is_candidate':True,'type':'stall','model_raw_score':1.,'maritime_policy':{'initial_sog_kn':8}}
    assert not assess_review_evidence(r)['eligible']


def test_generic_deviation_does_not_gain_focus_from_local_reference_alone():
    r=record([8]*12,kind='deviation')
    r['field_residual_mps']=5
    r['maritime_policy'].update(neighbor_supported=True,scoring_neighbor_count=4,valid_local_steps=8)
    assert not assess_review_evidence(r)['eligible']


def test_local_relationship_requires_all_support_conditions():
    r=record([8]*12,kind='group_inconsistency')
    r['field_residual_mps']=2.5
    r['maritime_policy'].update(neighbor_supported=True,scoring_neighbor_count=3,valid_local_steps=3)
    assert assess_review_evidence(r)['eligible']
    r['maritime_policy']['valid_local_steps']=2
    assert not assess_review_evidence(r)['eligible']
    r['maritime_policy']['valid_local_steps']=3
    r['field_residual_mps']=1.9
    assert not assess_review_evidence(r)['eligible']


@pytest.mark.parametrize('field,value',[('stationary',True),('is_stationary',True)])
def test_stationary_gate_overrides_strong_looking_input(field,value):
    r=record([10,10,10,10,1,1,1,1,1])
    r['maritime_policy'][field]=value
    assert not assess_review_evidence(r)['eligible']


def test_first_stage_motion_candidate_flag_has_precedence():
    r=record([10,10,10,10,1,1,1,1,1])
    r['motion_candidate']=False
    assert not assess_review_evidence(r)['eligible']


def test_long_observation_gap_does_not_supply_sudden_change():
    r=record([10,10,10,10,1,1,1,1,1],times=[0,60,120,180,600,660,720,780,840])
    assert not assess_review_evidence(r)['eligible']
