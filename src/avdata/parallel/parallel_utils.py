from multiprocessing import Pool
from typing import Callable, List, Optional

from tqdm import tqdm


def parallel_apply(
    element_fn: Callable,
    element_list: List,
    num_workers: int,
    desc: Optional[str] = None,
    disable: bool = False,
) -> List:
    with Pool(processes=num_workers) as pool:
        return list(
            tqdm(
                pool.imap_unordered(element_fn, element_list),
                desc=desc,
                total=len(element_list),
                disable=disable,
            )
        )
