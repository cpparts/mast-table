from typing import List
import numpy as np
from astropy.table import Table
import math


def py_type(dtype):
    kind = dtype.kind.lower()
    if kind in "iu":
        return int
    elif kind == "f":
        return float
    elif kind == "b":
        return bool
    return dtype


def table_py_types(table):
    return {
        column: py_type(dtype)
        for column, (dtype, _) in table.dtype.fields.items()
    }


def use_table_column_names(table) -> List[str]:
    """Return a list of column names from a table."""
    return table.colnames


def table_value_count(table, column, limit: int):
    return Table(
        data=[{'value': str(group[column][0]), 'count': len(group[column])}
              for i, group in enumerate(table.group_by(column).groups)
              if i < limit]
        )


def table_filter_values(table, column, values, invert=False):
    filter = np.isin(
        table[column].astype(str),
        np.asarray(values).astype(str),
    )

    if invert:
        filter = ~filter

    return filter


def table_range(table, column):
    return table[column].min().tolist(), table[column].max().tolist()


def slide_or_select(
    table,
    column,
):
    if not np.issubdtype(table[column].dtype, np.number):
        return "select"
    else:
        nunique_values = len(list(
            np.unique(table[column].astype(str))
        ))
        if nunique_values > 10:
            return "slider"
        else:
            return "select"


def step_size(
    vmin,
    vmax
):
    rng = vmax-vmin
    step = rng/5 if rng <= 1 else 0.1
    decimals = max(0, int(-math.floor(math.log10(step)))) if step < 1 else 0
    return round(step, decimals)
