import torch

from typing import List
from torch.utils.data._utils.collate import default_collate

from .batch_element import AgentBatchElement, SceneBatchElement


class AgentBatch:
    """A batch of agent-centric data.
    """
    def __init__(self, nums: torch.Tensor) -> None:
        self.nums = nums

    @classmethod
    def collate_fn(cls, batch_elems: List[AgentBatchElement]):
        return cls(nums=[7])


class SceneBatch:
    """A batch of scene-centric data.
    """
    def __init__(self, nums: torch.Tensor) -> None:
        self.nums = nums

    @classmethod
    def collate_fn(cls, batch_elems: List[SceneBatchElement]):
        return cls(nums=default_collate([batch_elem.history_sec_at_most for batch_elem in batch_elems]))