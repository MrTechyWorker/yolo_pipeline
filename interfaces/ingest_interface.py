from abc import ABC, abstractmethod

class IIngets_data(ABC):
    
    @abstractmethod
    def get_coco_data(self, path: str): ...

    @abstractmethod
    def get_id_to_label(self): ...