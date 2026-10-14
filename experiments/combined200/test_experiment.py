import unittest
import tempfile,hashlib
from pathlib import Path
from experiment import validate_identity_splits,validate_real_sources,real_statistics
from summarize_verified import check_sources

class ProtocolTests(unittest.TestCase):
    def test_changed_training_input_still_blocks_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'train.py';path.write_text('changed',encoding='utf-8')
            with self.assertRaises(ValueError):check_sources({str(path):'oldhash'},Path(tmp)/'build_report.py')

    def test_reporting_only_change_is_reported_without_rewriting_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'build_report.py';path.write_text('new report',encoding='utf-8')
            expected={str(path):'oldhash'}
            unchanged,changed=check_sources(expected,path)
            self.assertEqual(unchanged,0)
            self.assertEqual(changed[0]['expected_sha256'],'oldhash')
            self.assertEqual(changed[0]['actual_sha256'],hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(expected[str(path)],'oldhash')

    def test_duplicate_geometry_across_train_and_test_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_identity_splits({'training':['SYN_000'],'development':['SYN_001'],
                'previous_test_regression':['SYN_002'],'fresh_evaluation':['SYN_000']})

    def test_same_pixels_in_gallery_and_query_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_real_sources([dict(image_sha256='same')],[dict(image_sha256='same')])

    def test_wrong_accept_and_abstention_remain_in_denominator(self):
        rows=[dict(expected_in_database=True,expected_identity='A',ranking=dict(status='ranked',candidates=[dict(identity='B')]),
            decision=dict(top1='B',accepted=True,predicted_identity='B')),
            dict(expected_in_database=True,expected_identity='A',ranking=dict(status='insufficient',candidates=[]),
            decision=dict(top1=None,accepted=False,predicted_identity=None))]
        stats=real_statistics(rows)
        self.assertEqual(stats['queries'],2)
        self.assertEqual(stats['known_correct_accepted'],0)
        self.assertEqual(stats['known_wrong_accepted'],1)
        self.assertEqual(stats['known_rejected'],1)
        self.assertEqual(stats['insufficient_queries'],1)

if __name__=='__main__':unittest.main()
