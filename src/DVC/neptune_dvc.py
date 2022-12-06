import neptune.new as neptune
import os
from typing import Any
from .dvc import DVC

class NeptuneDVC(DVC):
    def __init__(self, experiment_name: str = ""):
        self.run = neptune.init_run(
            project="vladnosiv/eng-practices-hse",
            api_token=os.environ["NEPTUNE_TOKEN"],
        )
        self.path_prefix = experiment_name
                
    def __getitem__(self, key: str):
        return self.run[f'{self.path_prefix}/{key}']
    
    def __setitem__(self, key: str, item: Any):
        self.run[f'{self.path_prefix}/{key}'] = item
        
    def commit(self):
        self.run.stop()
