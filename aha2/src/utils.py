from typing import TypeVar, Dict, List, overload

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')

@overload
def remove_nones(d: Dict[K, V]) -> Dict[K, V]: ...

@overload
def remove_nones(d: List[T]) -> List[T]: ...

def remove_nones(d: Dict[K, V] | List[T]) -> Dict[K, V] | List[T]:
    if isinstance(d, dict):
        return {k: v for k, v in d.items() if v is not None}
    return [v for v in d if v is not None]
