from DVC.neptune_dvc import NeptuneDVC
from airflow.decorators import task


from classifier import MLPClassifier

from nn_modules import Linear, ReLU

import numpy as np

from sklearn.datasets import make_blobs, make_moons

@task()
def init_dvc(experiment_name: str) -> NeptuneDVC:
    print(f'init dvc {experiment_name}')
    return NeptuneDVC(experiment_name)


@task(multiple_outputs=True)
def get_moons_data(dvc: NeptuneDVC):
    x, y = make_moons(400, noise=0.075)
    dvc['train_data/X'] = x
    dvc['train_data/y'] = y
    
    x_test, y_test = make_moons(400, noise=0.075)
    dvc['test_data/X'] = x_test
    dvc['test_data/y'] = y_test
    
    return {
        'x': x,
        'y': y,
        'x_test': x_test,
        'y_test': y_test,
    }
    

@task(multiple_outputs=True)
def get_blobs_data(dvc: NeptuneDVC):
    x, y = make_blobs(400, 2, centers=[[0, 0], [2.5, 2.5], [-2.5, 3]])
    dvc['train_data/X'] = x
    dvc['train_data/y'] = y
    
    x_test, y_test = make_blobs(
        400,
        2,
        centers=[[0, 0], [2.5, 2.5], [-2.5, 3]],
    )
    dvc['test_data/X'] = x_test
    dvc['test_data/y'] = y_test
    
    return {
        'x': x,
        'y': y,
        'x_test': x_test,
        'y_test': y_test,
    }


@task()
def test_synth_moons(dvc: NeptuneDVC, data_dict) -> bool:
    x, y = data_dict['x'], data_dict['y']
    x_test, y_test = data_dict['x_test'], data_dict['y_test']

    best_acc = 0
    for _ in range(25):
        p = MLPClassifier(
            modules=[Linear(x.shape[1], 64), ReLU(), Linear(64, 2)],
            epochs=100,
        )

        p.fit(x, y)
        score = np.mean(p.predict(x_test) == y_test)
        dvc['train/scores'].log(score)
        if score > best_acc:
            best_acc = score
            dvc['best_model'] = p

    dvc['best_score'] = best_acc
    dvc.commit()
    return best_acc >= 0.85


@task()
def test_sanity_blobs(dvc: NeptuneDVC, data_dict) -> bool:
    x, y = data_dict['x'], data_dict['y']
    x_test, y_test = data_dict['x_test'], data_dict['y_test']
    
    
    best_acc = 0
    for _ in range(5):
        p = MLPClassifier(
            modules=[
                Linear(x.shape[1], 64),
                ReLU(),
                Linear(64, 64),
                ReLU(),
                Linear(64, 3),
            ],
            epochs=150,
        )

        p.fit(x, y)
        score = np.mean(p.predict(x_test) == y_test)
        dvc['train/scores'].log(score)
        if score > best_acc:
            best_acc = score
            dvc['best_model'] = p
            
    dvc['best_score'] = best_acc
    dvc.commit()
    
    return best_acc >= 0.85

@task()
def test_results(moons_result: bool, blobs_result: bool):
    if moons_result and blobs_result:
        print('Sanity test passed')
    else:
        print('Sanity test failed')