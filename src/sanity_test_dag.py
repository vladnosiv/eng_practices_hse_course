from airflow.decorators import dag
import pendulum
from sanity_tests import *

@dag(
    schedule=None,
    start_date=pendulum.datetime(2022, 12, 22, tz="UTC"),
    catchup=False,
    tags=["sanity_test"],
)
def sanity_tests_dag():    
    dvc_moons = init_dvc('moons')
    dvc_blobs = init_dvc('blobs')
    
    moons_data = get_moons_data(dvc_moons)
    blobs_data = get_blobs_data(dvc_blobs)
    
    moons_result = test_synth_moons(dvc_moons, moons_data)
    blobs_result = test_sanity_blobs(dvc_blobs, blobs_data)
    
    test_results(moons_result, blobs_result)
    

if __name__ == '__main__':
    sanity_tests_dag()