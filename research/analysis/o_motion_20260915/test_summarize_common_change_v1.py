"""Small hand-count unit tests; these fixtures are NOT research observations.

No model/real predictions, GPU, generated score files, or bootstrap-based
scientific claim. Ratios are checked against arithmetic written independently
of the implementation; scene retention and pairing are checked separately.
"""
import copy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize_common_change_v1 as summary
from common_occupancy_change_metrics_v1 import POSITIVE_GROUPS, SCHEMA


def hand_metric(token, scene, tp, fn, fp=0, grid=301):
    """A static GT/prediction pair at all five times, described by counts.

    All positive GT support belongs to the single fixed test speed group.
    Both false-negative and false-positive state transitions are realizable:
    static FN gives GT 11 -> predicted 00; static FP gives GT 00 -> 11.
    Arbitrary speed attribution here tests the ledger only, not box kinematics.
    """
    tn = grid-tp-fn-fp
    if min(tp,fn,fp,tn) < 0:
        raise ValueError('Bad test fixture')
    hist = [[tn,fp],[fn,tp]]
    transition = [[tn,0,0,fp],[0,0,0,0],[0,0,0,0],[fn,0,0,tp]]
    horizons = []
    for h in range(5):
        groups = {k: dict(GT=0,TP=0,FN=0) for k in POSITIVE_GROUPS}
        groups['speed_gt_0.5_le_5'] = dict(GT=tp+fn,TP=tp,FN=fn)
        horizons.append(dict(horizon_index=h,occupancy=dict(confusion=copy.deepcopy(hist)),
            global_negative=dict(GT=tn+fp,FP=fp,TN=tn),
            motion_positive_attribution=None if h == 0 else dict(groups=groups)))
    return dict(schema=SCHEMA,identity=dict(sample_token=token,scene_token=scene),
        horizons=horizons,
        transitions=[dict(horizon_index=h,confusion=copy.deepcopy(transition),
            domain=dict(grid_voxels=grid,both_valid=grid,t0_valid_h_ignored=0,
                t0_ignored_h_valid=0,both_ignored=0,t0_valid=grid,h_valid=grid,
                t0_foreground_excluded=0,h_foreground_excluded=0)) for h in range(1,5)],
        t0_boundary=dict(prediction_full_binary_sha256='same-models-'+token,
                         gt_valid_mask_sha256='all-valid-'+token))


def row(token, scene, tp, fn, arms=('O','A','Z'), fp=0):
    m = hand_metric(token,scene,tp,fn,fp=fp)
    return dict(sample_token=token,scene_token=scene,
                metrics_by_arm={a:copy.deepcopy(m) for a in arms})


class AggregationTests(unittest.TestCase):
    def test_sum_before_ratio_is_99_percent_not_75_percent(self):
        # Sample one: 1/2; sample two: 98/98. Pooled: 99/100.
        rows = [row('one','s1',1,1), row('two','s2',98,0)]
        result = summary.summarize(rows,bootstrap_repetitions=32,seed=11)
        self.assertEqual(result['values']['O']['future_GMO_IoU_percent'],99.)
        self.assertNotEqual(result['values']['O']['future_GMO_IoU_percent'],75.)
        self.assertEqual(result['values']['O']['future_speed_gt_0.5_recall_percent'],99.)
        self.assertEqual(result['pooled_counts']['O']['occupancy_5x2x2'][1],[[502,0],[1,99]])

    def test_unequal_scene_anchor_counts_retained_and_identical_CI_exact_zero(self):
        # Scene s1 contains TWO anchors (1/2 and 98/98); s2 contains ONE (0/200).
        # Full-data pool is 99/300 = 33%, not scene-mean (99%+0%)/2 = 49.5%.
        rows=[row('one','s1',1,1),row('two','s1',98,0),row('three','s2',0,200)]
        arms,scenes,sums=summary.collect(rows)
        self.assertEqual(scenes,['s1','s2'])
        scene_occ,_,_=summary.unpack(sums['O'])
        np.testing.assert_array_equal(scene_occ[0,1],[[502,0],[1,99]])
        np.testing.assert_array_equal(scene_occ[1,1],[[101,0],[200,0]])
        result=summary.summarize(rows,bootstrap_repetitions=256,seed=11)
        self.assertEqual(result['samples'],3);self.assertEqual(result['scenes'],2)
        self.assertAlmostEqual(result['values']['O']['future_GMO_IoU_percent'],33.)
        self.assertNotEqual(result['values']['O']['future_GMO_IoU_percent'],49.5)
        for name in ('A-minus-O','Z-minus-O','A-minus-Z'):
            for key in ('future_GMO_IoU_percent','future_FP','future_FN',
                        'future_speed_gt_0.5_recall_percent'):
                c=result['comparisons'][name][key]
                self.assertEqual((c['difference'],c['lower95'],c['upper95']),(0.,0.,0.))
                self.assertEqual(c['finite_bootstrap_repetitions'],256)
                self.assertEqual(c['undefined_bootstrap_repetitions'],0)
            # These test scenes have no arrival support or predictions.
            # Identical 0/0 is undefined, not a fabricated perfect score/CI.
            c=result['comparisons'][name]['future_01_arrival_IoU_percent']
            self.assertIsNone(c['difference']);self.assertIsNone(c['lower95']);self.assertIsNone(c['upper95'])
            self.assertEqual(c['undefined_bootstrap_repetitions'],256)

    def test_actual_bootstrap_keeps_shared_scene_weights_for_nonidentical_arms(self):
        # All true positives per scene are equal (100). O recalls 0%; A recalls
        # 100% in s1 and 20% in s2. Scene bootstrap outcomes: 20,60,100 pp.
        rows=[row('one','s1',0,100,arms=('O','A')),row('two','s2',0,100,arms=('O','A'))]
        rows[0]['metrics_by_arm']['A']=hand_metric('one','s1',100,0)
        rows[1]['metrics_by_arm']['A']=hand_metric('two','s2',20,80)
        for r in rows:
            # Common initial prediction really is all empty for BOTH arms.
            # A alone improves future recall, so its GT11 hits are predicted
            # 01 transitions, not 11. Do not fake a common hash for two
            # different t0 confusion matrices in this positive test fixture.
            candidate=r['metrics_by_arm']['A']
            candidate['horizons'][0]=copy.deepcopy(r['metrics_by_arm']['O']['horizons'][0])
            for t in candidate['transitions']:
                tp=t['confusion'][3][3]
                t['confusion'][3][3]=0
                t['confusion'][3][1]=tp
        result=summary.summarize(rows,bootstrap_repetitions=1024,seed=11)
        c=result['comparisons']['A-minus-O']['future_GMO_IoU_percent']
        self.assertEqual(c['difference'],60.)
        self.assertEqual(c['lower95'],20.);self.assertEqual(c['upper95'],100.)

    def test_original_GT_and_transition_GT_support_mismatch_rejected(self):
        r=row('one','s1',1,1)
        r['metrics_by_arm']['A']=hand_metric('one','s1',1,2)
        with self.assertRaisesRegex(ValueError,'Original GT denominator'):
            summary.collect([r])
        r=row('one','s1',1,1)
        # Keep total joint domain but change GT truth-state marginal.
        a=r['metrics_by_arm']['A']['transitions'][0]['confusion']
        a[0][0]-=1;a[3][3]+=1
        with self.assertRaisesRegex(ValueError,'Transition GT denominator'):
            summary.collect([r])

    def test_group_GT_TP_FN_wholeGT_and_global_negative_conservation(self):
        r=row('one','s1',1,1,fp=3)
        vector=summary.count_vector(r['metrics_by_arm']['O'],r)
        occ,_,groups=summary.unpack(vector)
        np.testing.assert_array_equal(groups[:,:,0].sum(1),[2,2,2,2])
        np.testing.assert_array_equal(groups[:,:,1].sum(1),[1,1,1,1])
        np.testing.assert_array_equal(groups[:,:,2].sum(1),occ[1:,1,0])
        for mutation,message in [('bad_sum','GT != TP'),('extra_GT','partition GT'),
                                 ('extra_TP','TP attribution'),('negative_duplicate','Global negatives')]:
            broken=copy.deepcopy(r);m=broken['metrics_by_arm']['O']
            g=m['horizons'][1]['motion_positive_attribution']['groups']['speed_gt_0.5_le_5']
            if mutation=='bad_sum':g['FN']+=1
            elif mutation=='extra_GT':g['GT']+=1;g['FN']+=1
            elif mutation=='extra_TP':g['TP']+=1;g['FN']-=1
            else:m['horizons'][1]['global_negative']['FP']+=1
            with self.subTest(mutation=mutation),self.assertRaisesRegex(ValueError,message):
                summary.count_vector(m,broken)

    def test_motion_support_t0_and_domain_changes_rejected(self):
        for mutation,message in [('group','Speed group support'),('t0','Common t0 prediction'),
                                 ('mask','Transition masks'),('count','Joint-domain count')]:
            r=row('one','s1',1,1);m=r['metrics_by_arm']['A']
            if mutation=='group':
                g=m['horizons'][1]['motion_positive_attribution']['groups']
                g['speed_le_0.1']=g['speed_gt_0.5_le_5'];g['speed_gt_0.5_le_5']=dict(GT=0,TP=0,FN=0)
            elif mutation=='t0':m['t0_boundary']['prediction_full_binary_sha256']='different'
            elif mutation=='mask':
                m['transitions'][0]['domain']['grid_voxels']+=1
                m['transitions'][0]['domain']['both_ignored']+=1
            else:m['transitions'][0]['domain']['both_valid']+=1
            with self.subTest(mutation=mutation),self.assertRaisesRegex(ValueError,message):
                summary.collect([r])


if __name__=='__main__':
    unittest.main()
