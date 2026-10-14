import pytest


def test_structure_metrics_count_failed_queries_in_denominator():
    from updates import structure_scores
    scores=structure_scores({'records':[{'rank':1,'ap':.8},{'rank':3,'ap':.2},{'rank':None,'ap':0}]})
    assert scores['queries']==3
    assert scores['top1']==pytest.approx(1/3)
    assert scores['top3']==pytest.approx(2/3)


def test_growth_metrics_reject_test_identity_in_training():
    from updates import growth_scores
    with pytest.raises(ValueError,match='identity'):
        growth_scores({'folds':[{'identity':'A','train_ids':['A'],'prediction':[2],'reference':[1],'baseline':[0]}]})


def test_growth_mae_is_computed_from_fold_predictions():
    from updates import growth_scores
    value=growth_scores({'folds':[{'identity':'A','train_ids':['B'],'prediction':[3,4],'reference':[1,3],'baseline':[1,1]}]})
    assert value['model_mae']==[2.,1.]
    assert value['baseline_mae']==[0.,2.]
    assert value['better_than_mean']==[False,True]


def test_frozen_checkpoint_and_gallery_reproduce_failure_rank():
    pytest.importorskip('torch')
    from updates import structure_case,read
    for key in ['real_only','mixed']:
        actual=structure_case(0,key)
        expected=read('structure_training_20261014/results.json')['models'][key]['regression']['records'][0]
        assert actual['rank']==expected['rank']
        assert actual['candidates'][0]['identity']==expected['candidates'][0]['identity']
        assert actual['dimensions']==128 and actual['encoding_recomputed']
