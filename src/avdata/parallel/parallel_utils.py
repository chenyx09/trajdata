from multiprocessing import Pool
from typing import Callable, Iterable, List, Optional

from tqdm import tqdm


def parallel_apply(
    element_fn: Callable,
    element_list: Iterable,
    num_workers: int,
    desc: Optional[str] = None,
    disable: bool = False,
) -> List:
    return list(parallel_iapply(element_fn, element_list, num_workers, desc, disable))


def parallel_iapply(
    element_fn: Callable,
    element_list: Iterable,
    num_workers: int,
    desc: Optional[str] = None,
    disable: bool = False,
) -> Iterable:
    with Pool(processes=num_workers) as pool:
        for fn_output in tqdm(
            pool.imap(element_fn, element_list),
            desc=desc,
            total=len(element_list),
            disable=disable,
        ):
            yield fn_output
