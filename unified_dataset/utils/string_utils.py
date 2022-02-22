from typing import List, Tuple


def pretty_string_tuples(tuple_list: List[Tuple], delimiter: str = '-') -> List[str]:
    return [delimiter.join(tupl) for tupl in tuple_list]
