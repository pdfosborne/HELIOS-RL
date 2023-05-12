from abc import ABC, abstractmethod
from chess import Board


class SimilarityMeasureAbstract(ABC):
    @abstractmethod
    def similarity(self, *args, **kwargs) -> str:
        pass


class SimilarityMeasure(SimilarityMeasureAbstract):
    def similarity(self, ) -> float:
        pass
    
   

