from abc import ABC
from typing import Any

class DVC(ABC):
    def __getitem__(self, key: str):
        pass
    
    def __setitem__(self, key: str, item: Any):
        pass
    
    def commit():
        pass
