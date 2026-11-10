"""Exact-time, leave-one-vessel-out neighbour geometry and legacy compatibility.

Decoded point fixtures isolate neighbourhood mathematics from SQLite I/O.
They are unit-test fixtures, never business/demo inputs.
"""
from __future__ import annotations

import math
from pathlib import Path
from tempfile import TemporaryDirectory
import sqlite3
import unittest
from unittest.mock import patch

import numpy as np

from anomaly_workbench import sqlite_stream as stream


BASE = 1723766400.0  # 2024-08-16 00:00:00 UTC, aligned to a minute.
OWN = '111111111'
OTHER = '222222222'


def point(vessel, second, *, x=0., y=0., cog=90., sog=5., weight=1., reported=True):
    return dict(mmsi=vessel, timestamp=BASE+second, x=x, y=y, cog=cog,
                sog=sog, weight=weight, reported=reported)


def decoded(points):
    rows=sorted(points,key=lambda p:(p['timestamp'],p['mmsi']))
    ts=np.array([p['timestamp'] for p in rows],float)
    ids=np.array([p['mmsi'] for p in rows])
    xy=np.array([[p['x'],p['y']] for p in rows],float)
    sog=np.array([p['sog'] for p in rows],float)
    cog=np.array([p['cog'] for p in rows],float)
    radians=np.deg2rad(cog)
    velocity=np.column_stack((sog*.514444*np.sin(radians),sog*.514444*np.cos(radians)))
    return dict(timestamp=ts,mmsi=ids,xy=xy,velocity=velocity,sog=sog,cog=cog,
                w=np.array([p['weight'] for p in rows],float),
                reported=np.array([p['reported'] for p in rows],bool),
                e_kin=np.zeros(len(rows)),heading=cog.copy(),origin=[11.,55.],
                lat=55.+np.rad2deg(xy[:,1]/stream.EARTH),
                lon=11.+np.rad2deg(xy[:,0]/(stream.EARTH*math.cos(math.radians(55.)))),
                groups={v:np.flatnonzero(ids==v) for v in np.unique(ids)},source_rows=len(rows))


def build(points, *, end=700, revision=2):
    data=decoded(points)
    with patch.object(stream,'read_points',return_value=data):
        window=stream.build_window({'id':'unit-neighbours','name':'unit'},BASE,BASE+end,
                                   neighbor_revision=revision)
    return window


def own_meta(window):
    return next(m for m in window['metadata'] if m['mmsi']==OWN)


def members(frame):
    return {n['mmsi']:n for n in frame['neighbors']}


class NeighborGeometryTests(unittest.TestCase):
    def test_same_minute_report_visible_only_after_its_actual_time(self):
        window=build([point(OWN,29),point(OWN,59,x=75),
                      point(OTHER,30,x=500)])
        frames=own_meta(window)['neighbor_frames']
        self.assertEqual(frames[0]['count'],0)
        self.assertEqual(frames[1]['count'],1)
        neighbour=members(frames[1])[OTHER]
        self.assertEqual(neighbour['report_epoch'],BASE+30)
        self.assertEqual(neighbour['age_seconds'],29)
        self.assertEqual(frames[1]['snapshot_utc'],stream.utc(BASE+59))

    def test_latest_report_is_distinct_per_vessel_and_never_from_future(self):
        window=build([point(OWN,29),point(OWN,59,x=75),
                      point(OTHER,30,x=500),point(OTHER,50,x=600),
                      point(OTHER,70,x=700)])
        last=own_meta(window)['neighbor_frames'][-1]
        self.assertEqual(last['count'],1)
        self.assertEqual(members(last)[OTHER]['report_epoch'],BASE+50)
        self.assertTrue(all(n['report_epoch']<=last['time_epoch'] for n in last['neighbors']))
        self.assertNotIn(OWN,members(last))

    def test_staleness_is_measured_at_actual_target_utc_including_300_seconds(self):
        window=build([point(OWN,0),point(OWN,599,x=100),
                      point(OWN,600,x=101),point(OWN,601,x=102),
                      point(OTHER,300,x=500)],end=602)
        frames=own_meta(window)['neighbor_frames']
        self.assertEqual([f['count'] for f in frames],[0,1,1,0])
        self.assertEqual(members(frames[1])[OTHER]['age_seconds'],299)
        self.assertEqual(members(frames[2])[OTHER]['age_seconds'],300)

    def test_heading_wrap_accepts_359_and_1_degrees(self):
        wrong='333333333'
        window=build([point(OWN,0,cog=359),point(OWN,60,cog=359),
                      point(OTHER,0,x=400,cog=1),point(OTHER,60,x=450,cog=1),
                      point(wrong,0,y=400,cog=90),point(wrong,60,y=450,cog=90)])
        last=own_meta(window)['neighbor_frames'][-1]
        self.assertIn(OTHER,members(last))
        self.assertNotIn(wrong,members(last))
        self.assertNotIn(OWN,members(last))

    def test_low_speed_zero_course_uses_recent_reliable_moving_reference(self):
        window=build([point(OWN,0,cog=90,sog=5),point(OWN,120,cog=0,sog=0),
                      point(OTHER,60,x=500,cog=90),point(OTHER,120,x=550,cog=90)])
        last=own_meta(window)['neighbor_frames'][-1]
        self.assertIn(OTHER,members(last))
        self.assertEqual(last['direction_reference_utc'],stream.utc(BASE))
        self.assertIn('可靠在航',last['direction_basis'])

    def test_low_speed_without_reliable_reference_does_not_invent_same_direction(self):
        window=build([point(OWN,0,cog=0,sog=0),point(OWN,60,cog=0,sog=0),
                      point(OTHER,0,x=500,cog=0),point(OTHER,60,x=550,cog=0)])
        last=own_meta(window)['neighbor_frames'][-1]
        self.assertEqual(last['count'],0)
        self.assertGreater(last['nearby_count'],0)
        self.assertIsNone(last['direction_reference_utc'])
        self.assertIn('不可判定',last['direction_basis'])

    def test_reliable_direction_reference_expires_after_five_minutes(self):
        window=build([point(OWN,0,cog=90,sog=5),
                      point(OWN,300,cog=0,sog=0),point(OWN,301,cog=0,sog=0),
                      point(OTHER,299,x=500,cog=90),point(OTHER,301,x=510,cog=90)],end=302)
        frames=own_meta(window)['neighbor_frames']
        self.assertEqual(frames[1]['count'],1)
        self.assertEqual(frames[1]['direction_reference_utc'],stream.utc(BASE))
        self.assertEqual(frames[2]['count'],0)
        self.assertIsNone(frames[2]['direction_reference_utc'])

    def test_early_neighbours_survive_in_union_when_final_frame_has_none(self):
        window=build([point(OWN,0),point(OWN,60,x=50),point(OWN,120,x=100),
                      point(OWN,480,x=200),point(OTHER,0,x=500),point(OTHER,60,x=550)],end=600)
        meta=own_meta(window);frames=meta['neighbor_frames']
        self.assertGreater(frames[0]['count'],0)
        self.assertEqual(frames[-1]['count'],0)
        self.assertEqual(meta['neighbor_tracks'],[])  # Legacy last-frame geometry.
        catalog={n['mmsi']:n for n in meta['neighbor_track_catalog']}
        self.assertIn(OTHER,catalog)
        self.assertEqual(catalog[OTHER]['timestamps'],[BASE,BASE+60])
        self.assertEqual(len(catalog[OTHER]['latlon']),2)
        own_index=next(i for i,m in enumerate(window['metadata']) if m['mmsi']==OWN)
        self.assertEqual(len(frames),int(window['arrays']['mask'][own_index].sum()))

    def test_untrusted_out_of_range_and_wrong_direction_neighbours_are_not_added(self):
        untrusted='333333333';too_far='444444444';missing='555555555';slow='666666666'
        points=[point(OWN,0),point(OWN,60,x=25),point(OTHER,0,x=500),point(OTHER,60,x=525),
                point(untrusted,0,x=600,weight=.2),point(untrusted,60,x=625,weight=.2),
                point(too_far,0,x=stream.LEGACY_RADIUS+100),point(too_far,60,x=stream.LEGACY_RADIUS+125),
                point(missing,0,x=700,reported=False),point(missing,60,x=725,reported=False),
                point(slow,0,x=800,sog=1.9),point(slow,60,x=825,sog=1.9)]
        last=own_meta(build(points))['neighbor_frames'][-1]
        self.assertEqual(set(members(last)),{OTHER})
        self.assertLessEqual(members(last)[OTHER]['distance_nm'],8.)

    def test_revision_one_retains_minute_snapshot_for_saved_scan_numbers(self):
        points=[point(OWN,29),point(OWN,59,x=75),point(OTHER,30,x=500)]
        old=build(points,revision=1);new=build(points,revision=2)
        old_meta=own_meta(old);new_meta=own_meta(new)
        self.assertEqual([f['count'] for f in old_meta['neighbor_frames']],[0,0])
        self.assertEqual(new_meta['neighbor_frames'][-1]['count'],1)
        self.assertEqual(old_meta['neighbor_frames'][-1]['snapshot_utc'],stream.utc(BASE))
        self.assertFalse(old['arrays']['nb_mask'][0].any())

    def test_fresh_prior_window_report_keeps_marker_without_fabricating_track(self):
        # A vessel can be present from a fresh preceding report while having no
        # trajectory samples inside the current window. Its actual marker must
        # remain available; an empty trail must not hide it or invent a line.
        window=build([point(OTHER,-299,x=500),point(OWN,0),point(OWN,1,x=5)],end=2)
        meta=own_meta(window);frames=meta['neighbor_frames']
        self.assertEqual([f['count'] for f in frames],[1,1])
        self.assertEqual(members(frames[-1])[OTHER]['age_seconds'],300)
        self.assertEqual(len(members(frames[-1])[OTHER]['position']),2)
        trail=next(n for n in meta['neighbor_track_catalog'] if n['mmsi']==OTHER)
        self.assertEqual(trail['latlon'],[])
        self.assertEqual(trail['timestamps'],[])

    def test_revision_three_radius_includes_3_49_and_excludes_3_51_nm(self):
        far='333333333'
        points=[point(OWN,0),point(OWN,60),
                point(OTHER,0,x=3.49*1852),point(OTHER,60,x=3.49*1852),
                point(far,0,x=3.51*1852),point(far,60,x=3.51*1852)]
        latest=own_meta(build(points,revision=3))['neighbor_frames'][-1]
        self.assertEqual(set(members(latest)),{OTHER})
        self.assertEqual(latest['radius_nm'],3.5)
        legacy=own_meta(build(points,revision=2))['neighbor_frames'][-1]
        self.assertIn(far,members(legacy))
        self.assertEqual(legacy['radius_nm'],8.)

    def test_revision_three_gaussian_sigma_and_combined_weights(self):
        farther='333333333';third='444444444'
        points=[point(OWN,0),point(OWN,60),
                point(OTHER,0,x=1852),point(OTHER,60,x=1852),
                point(farther,0,x=2*1852,weight=.5),point(farther,60,x=2*1852,weight=.5),
                point(third,0,y=1852),point(third,60,y=1852)]
        w=build(points,revision=3);latest=own_meta(w)['neighbor_frames'][-1];ns=members(latest)
        near_weight=math.exp(-1/(2*1.5**2));far_weight=math.exp(-4/(2*1.5**2))
        self.assertAlmostEqual(ns[OTHER]['distance_weight'],near_weight)
        self.assertAlmostEqual(ns[farther]['distance_weight'],far_weight)
        self.assertAlmostEqual(ns[farther]['distance_weight']/ns[OTHER]['distance_weight'],math.exp(-3/(2*1.5**2)))
        self.assertAlmostEqual(ns[farther]['combined_weight'],.5*far_weight)
        slots=w['arrays']['nb_mask'][0]
        # The graph edge weights follow the same quality × distance rule.
        np.testing.assert_allclose(np.sort(w['arrays']['nb_w'][0,slots]),
                                   np.sort([near_weight,near_weight,.5*far_weight]),rtol=1e-6,atol=1e-7)

    def test_revision_three_accepts_three_noncollinear_neighbors_legacy_requires_four(self):
        points=[point(OWN,0),point(OWN,60)]
        for vessel,x,y in [(OTHER,1852,0),('333333333',0,1852),('444444444',1852,1852)]:
            points += [point(vessel,0,x=x,y=y),point(vessel,60,x=x,y=y)]
        current=build(points,revision=3);legacy=build(points,revision=2)
        self.assertEqual(own_meta(current)['neighbor_frames'][-1]['minimum_neighbors'],3)
        self.assertEqual(own_meta(current)['neighbor_frames'][-1]['count'],3)
        self.assertTrue(current['arrays']['field_valid'][0,-1 if current['arrays']['mask'][0,-1] else 1])
        self.assertFalse(legacy['arrays']['field_valid'][0].any())
        self.assertAlmostEqual(float(current['arrays']['r_field'][0,1]),0.,places=5)

    def test_three_collinear_neighbors_still_fail_rank_check(self):
        points=[point(OWN,0),point(OWN,60)]
        for number,vessel in enumerate([OTHER,'333333333','444444444'],1):
            points += [point(vessel,0,x=number*1852),point(vessel,60,x=number*1852)]
        w=build(points,revision=3)
        self.assertEqual(own_meta(w)['neighbor_frames'][-1]['count'],3)
        self.assertFalse(w['arrays']['field_valid'][0].any())

    def test_motion_metadata_uses_all_original_observations_before_sampling(self):
        points=[point(OWN,k*20,x=k*15,sog=3. if k==1 else 0.) for k in range(40)]
        w=build(points,end=800,revision=3);meta=own_meta(w);motion=meta['motion_observations']
        self.assertEqual(len(meta['sog_kn']),20)
        self.assertEqual(len(motion['sog_kn']),40)
        self.assertEqual(meta['source_sog_max_kn'],3.)
        self.assertEqual(meta['source_sog_min_kn'],0.)
        self.assertEqual(meta['source_sog_valid_count'],40)
        self.assertEqual(motion['time_seconds'],[k*20 for k in range(40)])
        self.assertEqual(motion['sog_kn'][1],3.)

    def test_missing_cog_never_overwrites_observed_stationary_sog(self):
        from anomaly_workbench.tests.test_sqlite_pipeline import make_db
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'source.sqlite';start=make_db(path)
            db=sqlite3.connect(path)
            db.execute("UPDATE ingested_ais_positions SET sog_knots=0,cog_degrees=NULL WHERE mmsi='219000001'")
            db.commit();db.close()
            entry=stream.catalog(path,True)['datasets'][0]
            data=stream.read_points(entry,start,start+1200,path)
            rows=data['groups']['219000001']
            self.assertTrue((data['sog'][rows]==0).all())
            self.assertTrue((data['source_sog'][rows]==0).all())
            self.assertTrue(data['sog_reported'][rows].all())
            self.assertFalse(data['cog_reported'][rows].any())
            window=stream.build_window(entry,start,start+1200,path,neighbor_revision=3)
            meta=next(m for m in window['metadata'] if m['mmsi']=='219000001')
            motion=meta['motion_observations']
            self.assertEqual(meta['source_sog_max_kn'],0.)
            self.assertEqual(meta['motion_state'],'stationary_source_max_below_2kn')
            self.assertTrue(all(v==0 for v in motion['sog_kn']))
            self.assertTrue(all(v is None for v in motion['cog_deg']))
            self.assertTrue(all(motion['sog_reported_mask']))
            self.assertFalse(any(motion['reported_mask']))
            # Missing COG is not a physical teleport confidence of 0.25.
            self.assertTrue(all(v==1 for v in motion['gate_weights']))
            self.assertTrue(all(v is None for v in motion['e_kin_mps']))
            legacy=stream.read_points(entry,start,start+1200,path,preserve_report_fields=False)
            self.assertTrue((legacy['sog'][rows][1:]>0).any())

    def test_missing_sog_derivation_preserves_a_valid_reported_course(self):
        from anomaly_workbench.tests.test_sqlite_pipeline import make_db
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'source.sqlite';start=make_db(path)
            db=sqlite3.connect(path)
            db.execute("UPDATE ingested_ais_positions SET sog_knots=NULL,cog_degrees=123 WHERE mmsi='219000001'")
            db.commit();db.close()
            entry=stream.catalog(path,True)['datasets'][0]
            data=stream.read_points(entry,start,start+1200,path);rows=data['groups']['219000001']
            self.assertTrue((data['cog'][rows]==123).all())
            self.assertTrue(data['cog_reported'][rows].all())
            self.assertFalse(data['sog_reported'][rows].any())
            window=stream.build_window(entry,start,start+1200,path,neighbor_revision=3)
            meta=next(m for m in window['metadata'] if m['mmsi']=='219000001')
            self.assertTrue(all(v is None for v in meta['motion_observations']['sog_kn']))
            self.assertIsNone(meta['source_sog_max_kn'])
            self.assertEqual(meta['motion_state'],'unknown_missing_reported_sog')

    def test_revision_three_kpis_use_only_original_reported_sog(self):
        from anomaly_workbench.tests.test_sqlite_pipeline import make_db
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'source.sqlite';start=make_db(path)
            db=sqlite3.connect(path)
            db.execute("UPDATE ingested_ais_positions SET sog_knots=0,cog_degrees=NULL WHERE mmsi='219000001'")
            db.commit();db.close()
            entry=stream.catalog(path,True)['datasets'][0]
            current=stream.build_window(entry,start,start+1200,path,neighbor_revision=3)
            self.assertAlmostEqual(current['mean_sog_kn'],25/6)
            self.assertEqual(current['moving_vessels'],5)
            self.assertEqual(current['source_sog_missing_count'],0)
            self.assertEqual(current['speed_statistics_basis'],'original_valid_reported_sog')
            # With no reported SOG, derived GPS velocity is never silently
            # promoted into an observed speed or a confirmed underway count.
            db=sqlite3.connect(path);db.execute('UPDATE ingested_ais_positions SET sog_knots=NULL');db.commit();db.close()
            unknown=stream.build_window(entry,start,start+1200,path,neighbor_revision=3)
            self.assertIsNone(unknown['mean_sog_kn'])
            self.assertEqual(unknown['moving_vessels'],0)
            self.assertEqual(unknown['source_sog_missing_count'],120)


if __name__=='__main__':
    unittest.main()
