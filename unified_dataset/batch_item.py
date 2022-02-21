from typing import List
from torch.utils.data._utils.collate import default_collate


class SingleDatum:
    def __init__(self, num: int) -> None:
        self.num = num


def unified_collate(batch_elems: List[SingleDatum]):
    return default_collate([batch_elem.num for batch_elem in batch_elems])
