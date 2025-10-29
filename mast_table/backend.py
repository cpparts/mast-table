from abc import ABC, abstractmethod
from astropy.table import Table
from astroquery.mast import Observations


class MastTableBackend(ABC):
    def __init__(self, table=None):
        self.table = table
        self.context = None

    @abstractmethod
    def load(self, *args, **kwargs):
        pass

    def filter(self, **criteria):
        for key, val in criteria.items():
            self.table = self.table[self.table[key] == val]

class FilesetBackend(MastTableBackend):
    def __init__(self, table=None):
        super().__init__(table)
        self.context = "fileset"

    def load(self, obs_id, **kwargs):
        pass

class ProductsBackend(MastTableBackend):
    def __init__(self, table=None):
        super().__init__(table)
        self.context = "products"

    def load(self, obs_id, **kwargs):
        pass

_BACKEND_MAP = {
    "fileset": FilesetBackend,
    "products": ProductsBackend,
}

def get_backend(name):
    name = name.lower()
    if name not in _BACKEND_MAP:
        raise ValueError(f"Unknown backend '{name}'. Valid options: {list(_BACKEND_MAP)}")
    return _BACKEND_MAP[name]
