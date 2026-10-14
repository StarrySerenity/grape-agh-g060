import unittest,copy
from real_predict import predict_case,compare_metrics

def balls():return [dict(id=str(i),center=[(i%2)*2.,(i//2%2)*2.,(i//4)*2.],radius=.4,shape='sphere') for i in range(8)]
def pairs():
    result=[]
    for i in range(6):
        b=balls();late=copy.deepcopy(b)
        for m in late:m['radius']*=1.2;m['center'][1]*=1.2
        result.append(dict(identity=f'GROW_{i:02d}',group='GROW',status='ready',days=40,early=dict(models=b,bunch_id=f'GROW_{i:02d}'),late=dict(models=late)))
    return result

class GrowthTests(unittest.TestCase):
    def test_own_future_cannot_change_original_prediction(self):
        rows=pairs();sample=rows[0]['early'];before=copy.deepcopy(sample)
        first=predict_case(sample,rows,'GROW',conditional=False)
        rows[0]['late']['models'][0]['radius']=100
        second=predict_case(sample,rows,'GROW',conditional=False)
        self.assertEqual(first['forecast']['prediction'],second['forecast']['prediction'])
        self.assertEqual(first['grown']['models'],second['grown']['models'])
        self.assertNotIn('GROW_00',first['forecast']['training_identities'])
        self.assertEqual(sample,before)

    def test_future_repair_requires_explicit_conditional_mode_and_flags_survive(self):
        sample=dict(models=balls(),bunch_id='GROW_00',uses_future_reference=True)
        with self.assertRaises(ValueError):predict_case(sample,pairs(),'GROW',conditional=False)
        result=predict_case(sample,pairs(),'GROW',conditional=True)
        self.assertTrue(result['input_uses_future_reference'])
        self.assertFalse(result['independent_early_prediction'])
        self.assertTrue(result['forecast']['uses_future_reference'])
        self.assertTrue(all(m['uses_future_reference'] for m in result['grown']['models']))
        self.assertTrue(sample['uses_future_reference'])

    def test_metric_errors_are_dimensionless_and_include_all_fields(self):
        stats=compare_metrics(balls(),balls())
        self.assertEqual(len(stats['absolute_errors']),4)
        self.assertTrue(all(v==0 for v in stats['absolute_errors']))

if __name__=='__main__':unittest.main()
