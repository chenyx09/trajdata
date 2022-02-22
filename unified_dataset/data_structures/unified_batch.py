import torch

from typing import List
from torch.utils.data._utils.collate import default_collate

from .unified_batch_element import UnifiedBatchElement


class UnifiedBatch:
    """A batch of data.
    """
    def __init__(self, nums: torch.Tensor) -> None:
        self.nums = nums


def unified_collate(batch_elems: List[UnifiedBatchElement]) -> UnifiedBatch:
    return UnifiedBatch(nums=default_collate([batch_elem.num for batch_elem in batch_elems]))
