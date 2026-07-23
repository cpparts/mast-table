import numpy as np
from astropy.table import Table, join
import math
import warnings


def num_py_type(table, col):
    """Return kind of numerical type for column of a table."""
    dtype = table[col].dtype
    kind = dtype.kind
    if kind in "iu":
        return int
    elif kind == "f":
        return float
    else:
        warnings.warn(
            f"Column {col} of datatype {dtype} not supported for Slider",
            UserWarning
        )


def table_value_count(table, column, limit: int):
    """Return values and counts for column of a table."""
    values, counts = np.unique(table[column], return_counts=True)

    return Table(
        data=[{
            'value': str(value),
            'count': count
            } for value, count in zip(values[:limit], counts[:limit])]
    )


def table_filter_values(table, column, values, invert=False):
    """Return filter for column of a table."""
    col = table[column]
    filter = np.zeros(len(table), dtype=bool)

    if "--" in values and hasattr(col, "mask"):
        filter |= col.mask

    real_values = np.array([v for v in values if v != "--"])
    column_data = col.data
    dtype = column_data.dtype

    if len(real_values):
        real_value_in_column_data = np.isin(column_data, real_values.astype(dtype))
        if hasattr(col, "mask"):
            filter |= (~col.mask) & real_value_in_column_data
        else:
            filter |= real_value_in_column_data

    if invert:
        filter = ~filter

    return filter


def table_range(table, column):
    """Return range for column of a table."""
    return table[column].min().tolist(), table[column].max().tolist()


def slide_or_select(
    table,
    column,
):
    """Return type of filter to render for column of a table."""
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
    """Return stepsize for range of values."""
    rng = vmax-vmin
    step = rng/5 if rng <= 1 else 0.1
    decimals = max(0, int(-math.floor(math.log10(step)))) if step < 1 else 0
    return round(step, decimals)


def build_select_items(col):
    """Return unique values for column and whether column is full masked."""
    unmasked = col
    fully_masked = False

    if hasattr(col, "mask"):
        mask = col.mask
        unmasked = col[~mask]
        if np.sum(mask)/len(col) == 1:
            fully_masked = True

    vals = np.asarray(unmasked).astype(str)
    items = np.unique(vals).tolist()
    return items, fully_masked


def build_select_filter_preview(
    table,
    column,
    max_unique=100,
    table_filtered=None,
):
    """
    Determines the items and value_counts for a given column used to generate
    filter preview information for columns that use `select` logic.

    Parameters
    ----------
    - `table`: The astropy Table to filter.
    - `column`: The column to filter on.
    - `max_unique`: The maximum number of unique values to show in the dropdown.
    - `table_filtered`: The filtered astropy Table.

    Returns
    -------
    - `items`: List of dictionaries containing information for each unique value.
    - `value_counts`: The astropy Table of value counts for each unique value.
    """
    value_counts = table_value_count(table, column, limit=max_unique + 1)
    value_counts.rename_column('count', 'count_max')

    if table_filtered:
        value_counts_filtered = table_value_count(
            table_filtered, column, limit=max_unique + 1
        )

        if len(value_counts_filtered):
            value_counts = join(
                value_counts,
                value_counts_filtered,
                join_type="left",
                keys="value"
            )
    else:
        value_counts['count'] = value_counts['count_max']

    if "count" not in value_counts.colnames:
        value_counts["count"] = 0

    value_counts = value_counts.filled(0)
    value_counts["exists"] = value_counts["count"] > 0
    value_counts.sort('value')

    items = [
        {
            # calling tolist() casts to python types rather than np types
            "value": row['value'].tolist(),
            "text": str(row['value'].tolist()),
            "count": row['count'].tolist(),
            "count_max": row['count_max'].tolist(),
        } for row in value_counts
    ]

    return items, value_counts
