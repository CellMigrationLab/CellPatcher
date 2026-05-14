from abc import ABC, abstractmethod

from cellpatcher.types import PatchRegion


class BasePatchStrategy(ABC):
    mode: str

    @abstractmethod
    def generate_regions(self, *, image_shape, labels=None, image_id=None) -> list[PatchRegion]:
        ...
