import unittest,copy
import numpy as np
from demo_core import pack_spheres, predict_young

def balls():
    return [dict(id=f'B{i}',center=[i%2*1.8,i//2%2*1.8,i//4*1.8],radius=1.,shape='sphere') for i in range(8)]

class DemoTests(unittest.TestCase):
    def test_packing_preserves_identity_radius_and_input(self):
        a=balls();before=copy.deepcopy(a);result=pack_spheres(a)
        self.assertEqual(a,before)
        self.assertEqual([m['id'] for m in result['models']],[m['id'] for m in a])
        self.assertEqual([m['radius'] for m in result['models']],[m['radius'] for m in a])
        p=np.array([m['center'] for m in result['models']]);d=np.linalg.norm(p[:,None]-p[None],axis=2);np.fill_diagonal(d,np.inf)
        self.assertGreaterEqual(float(d.min()),1.999)

    def test_future_assisted_input_rejected_before_prediction(self):
        with self.assertRaises(ValueError): predict_young(dict(models=balls(),uses_future_reference=True),[],group='GROW')

    def test_full_output_keeps_ids_and_uses_only_other_pair_data(self):
        a=balls();pairs=[]
        for i in range(5):
            late=copy.deepcopy(a)
            for m in late:
                m['radius']=1.1
                m['center'][1]*=1.5
                m['center'][2]*=.8
            pairs.append(dict(identity=f'TRAIN_{i}',group='GROW',status='ready',days=40,early=dict(models=copy.deepcopy(a)),late=dict(models=late)))
        result=predict_young(dict(models=a,uses_future_reference=False),pairs,group='GROW')
        self.assertFalse(result['forecast']['uses_future_reference'])
        self.assertEqual(len(result['grown']['models']),8)
        self.assertEqual(result['forecast']['training_identities'],[f'TRAIN_{i}' for i in range(5)])
        self.assertTrue(all(m['shape']=='sphere' for m in result['grown']['models']))

if __name__=='__main__':unittest.main()
