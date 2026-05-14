import operator
from typing import Any, List, Optional, cast, Callable

import numpy as np
from IPython.display import display

import solara
from solara.components.cross_filter import Select
import reacton.ipyvuetify as v

from astropy.table import Table, join
from mast_table.mast_table import MastTable, col_unique_row_index


def py_type(dtype):
    kind = dtype.kind.lower()
    if kind in "iu":
        return int
    elif kind == "f":
        return float
    elif kind == "b":
        return bool
    else:
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
    filter = np.isin(table[column], values)
    if invert:
        filter = ~filter
    return filter


def table_range(table, column):
    return table[column].min().tolist(), table[column].max().tolist()


@solara.component
def CrossFilterSelect(
    table: Table,
    column: str,
    max_unique: int = 100,
    multiple: bool = False,
    invert=False,
    configurable=True,
    classes: List[str] = [],
):
    """A Select widget that will cross filter an astropy Table.

    ## Arguments

    - `table`: The astropy Table to filter.
    - `column`: The column to filter on.
    - `max_unique`: The maximum number of unique values to show in the dropdown.
    - `multiple`: Whether to allow multiple values to be selected.
    - `invert`: Whether to invert the selection.
    - `configurable`: Whether to show the configuration button.
    - `classes`: Additional CSS classes to add to the main widget.

    """
    filter, set_filter = solara.use_cross_filter(id(table), "filter-dropdown")
    filter_values, set_filter_values = solara.use_state([])
    column, set_column = solara.use_state_or_update(column)
    invert, set_invert = solara.use_state_or_update(invert)
    multiple, set_multiple = solara.use_state_or_update(multiple)

    if filter is not None:
        table_filtered = table[filter]
    else:
        table_filtered = table

    value_counts = table_value_count(table, column, limit=max_unique + 1)
    value_counts.rename_column('count', 'count_max')

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
        value_counts['count'] = 0
        value_counts['exists'] = False
        value_counts['count_max'] = False

    value_counts = value_counts.filled(0)
    value_counts["exists"] = value_counts["count"] > 0
    value_counts.sort('value')

    columns = use_table_column_names(table)

    def set_values_and_filter(values):
        if values is None:
            set_filter_values([])
            return

        if multiple:
            set_filter_values([value["value"] for value in values])
        else:
            set_filter_values([values["value"]])

    def reset():
        set_filter_values([])

    solara.use_memo(reset, dependencies=[column])

    def update_filter():
        if len(filter_values) == 0:
            set_filter(None)
        else:
            filter = table_filter_values(
                table, column, filter_values,
                invert=invert
            )
            set_filter(filter)

    solara.use_memo(update_filter, dependencies=[filter_values, invert])

    items = [
        {
            # calling tolist() casts to python types rather than np types
            "value": row['value'].tolist(),
            "text": str(row['value'].tolist()),
            "count": row['count'].tolist(),
            "count_max": row['count_max'].tolist(),
        } for row in value_counts
    ]
    value: Any = None
    if not multiple:
        value = {"value": filter_values[0]} if len(filter_values) > 0 else None
    else:
        value = [{"value": k} for k in filter_values]

    # TODO: reacton bug, we cannot add this under any component context manager
    # this gives an error, probably because the button is added twice
    with v.Btn(v_on="x.on", icon=True) as btn:
        v.Icon(children=["mdi-settings"])
    with solara.VBox(classes=classes) as main:
        with solara.HBox(align_items="baseline"):
            with solara.Row():
                v.Select(
                    v_model=column,
                    items=columns,
                    on_v_model=set_column,
                    label="Show rows conditioned on column:"
                )

                label = (
                    f"{column} = " if not invert else f"{column} != "
                )
                Select.element(
                    value=value,
                    items=items,
                    on_value=set_values_and_filter,
                    label=label,
                    clearable=True,
                    return_object=True,
                    multiple=multiple,
                    filtered=filter is not None,
                    count=len(table_filtered),
                    messages=(
                        f"Too many unique values, will only show the first {max_unique}"
                        if len(value_counts) > max_unique else ""
                    ),
                    class_="solara-cross-filter-select",
                )
                if configurable:
                    v_slots = [{"name": "activator", "variable": "x", "children": btn}]
                    with v.Menu(v_slots=v_slots, close_on_content_click=False):
                        with v.Sheet():
                            with v.Container(py_0=True, px_3=True, ma_0=True):
                                with v.Row():
                                    with v.Col():
                                        v.Switch(
                                            v_model=invert,
                                            on_v_model=set_invert,
                                            label="Invert filter"
                                        )
                                        v.Switch(
                                            v_model=multiple,
                                            on_v_model=set_multiple,
                                            label="Select multiple"
                                        )

    return main


@solara.component
def CrossFilterReport(table, classes: List[str] = []):
    """Shows a report of the current cross filter state.

    Shows number of rows filtered, and the total number of rows.

    See [use_cross_filter](/documentation/api/hooks/use_cross_filter)
    for more information about how to use cross filtering.

    ## Arguments

    - `df`: The table where the filter is applied to.
    - `classes`: Additional CSS classes to add to the main widget.

    """
    filter, set_filter = solara.use_cross_filter(id(table), "summary")
    table_filtered = table
    filtered = False
    if filter is not None:
        filtered = True
        table_filtered = table[filter]
    progress = len(table_filtered) / len(table) * 100
    with solara.VBox(classes=classes) as main:
        with solara.HBox(align_items="center"):
            icon = "mdi-filter"
            v.Icon(
                children=[icon],
                style_="opacity: 0.1" if not filtered else ""
            )
            if filtered:
                summary = f"{len(table_filtered):,} / {len(table):,}"
            else:
                summary = f"{len(table_filtered):,}"
            v.Html(tag="h3", children=[summary], style_="display: inline")
        # always add a progress bar to make sure the layout is the same
        if filtered:
            v.ProgressLinear(value=progress).key("visible")
        else:
            v.ProgressLinear(value=0, style_="visibility: hidden").key("hidden")

    return main


@solara.component
def CrossFilterSlider(
    table,
    column: str,
    invert=False,
    enable: bool = True,
    mode: str = ">=",
    configurable=True,
):
    """A Slider widget that will cross filter an astropy Table.

    See [use_cross_filter](/documentation/api/hooks/use_cross_filter)
    for more information about how to use cross filtering.

    ## Arguments

    - `table`: The astropy Table to filter.
    - `column`: The column to filter on.
    - `invert`: If True, the filter will be inverted.
    - `enable`: If False, the filter will be disabled.
    - `mode`: The mode to use for filtering. Can be one of `==`, `>=`, `<=`, `>`, `<`.
    - `configurable`: Whether to show a configuration button.

    """
    filter, set_filter = solara.use_cross_filter(id(table), "filter-slider")
    filter_value, set_filter_value = solara.use_state(cast(Optional[int], None))
    column, set_column = solara.use_state_or_update(column)
    invert, set_invert = solara.use_state_or_update(invert)
    enable, set_enable = solara.use_state_or_update(enable)
    mode, set_mode = solara.use_state_or_update(mode)

    vmin, vmax = table_range(table, column)

    columns = use_table_column_names(table)
    py_types = table_py_types(table)
    columns_numeric = [c for c in columns if py_types[c] in [int, float]]

    def reset():
        set_filter_value(vmin)

    solara.use_memo(reset, dependencies=[column])

    def update_filter():
        if not enable or filter_value is None:
            set_filter(None)
        else:
            operator_map = {
                "==": operator.eq,
                ">=": operator.ge,
                "<=": operator.le,
                ">": operator.gt,
                "<": operator.lt,
                "!=": operator.ne,
            }
            filter = operator_map[mode](table[column], filter_value)
            if invert:
                filter = ~filter
            set_filter(filter)

    solara.use_memo(update_filter, dependencies=[filter_value, invert, enable, mode])

    # TODO: reacton bug, see CrossFilterSelect
    with v.Btn(v_on="x.on", icon=True) as btn:
        v.Icon(children=["mdi-settings"])

    with solara.VBox() as main:
        with solara.Div(style="max-width: 600px;"), solara.HBox(align_items="center"):
            label = f"Show {column} {mode} " if not invert else f"Drop {column} {mode} "

            if issubclass(py_types[column], (int, np.integer)):
                solara.SliderInt(
                    label=label,
                    value=filter_value,
                    min=vmin,
                    max=vmax,
                    on_value=set_filter_value,
                    disabled=not enable,
                    thumb_label='always',
                    tick_labels='end_points'
                )
                if filter_value is not None:
                    solara.Text(f"{filter_value:,}")
            elif issubclass(py_types[column], (float, np.floating)):
                solara.SliderFloat(
                    label=label,
                    value=filter_value,
                    min=vmin, max=vmax,
                    on_value=set_filter_value,
                    disabled=not enable,
                    thumb_label='always',
                    tick_labels='end_points'
                )
                if filter_value is not None:
                    solara.Text(f"{filter_value:,}")
            else:
                solara.Warning(f"{py_types[column]} not supported for Slider")

            if configurable:
                v_slots = [{"name": "activator", "variable": "x", "children": btn}]
                with v.Menu(v_slots=v_slots, close_on_content_click=False):
                    with v.Sheet():
                        with v.Container(py_0=True, px_3=True, ma_0=True):
                            with v.Row():
                                with v.Col():
                                    columns_numeric = [
                                        c for c in columns if py_types[c] in [int, float]
                                    ]
                                    v.Select(
                                        v_model=column,
                                        items=columns_numeric,
                                        on_v_model=set_column,
                                        label="Choose column"
                                    )
                                    v.Switch(
                                        v_model=invert,
                                        on_v_model=set_invert,
                                        label="Invert filter"
                                    )
                                    v.Switch(
                                        v_model=enable,
                                        on_v_model=set_enable,
                                        label="Enable filter"
                                    )

                                    with solara.ToggleButtonsSingle(value=mode, on_value=set_mode):
                                        solara.Button(
                                            icon_name="mdi-code-equal",
                                            icon=True,
                                            value="=="
                                        )
                                        solara.Button(
                                            icon_name="mdi-code-not-equal",
                                            icon=True,
                                            value="!="
                                        )
                                        solara.Button(
                                            icon_name="mdi-code-less-than",
                                            icon=True,
                                            value="<"
                                        )
                                        solara.Button(
                                            icon_name="mdi-code-less-than-or-equal",
                                            icon=True,
                                            value="<="
                                        )
                                        solara.Button(
                                            icon_name="mdi-code-greater-than",
                                            icon=True,
                                            value=">"
                                        )
                                        solara.Button(
                                            icon_name="mdi-code-greater-than-or-equal",
                                            icon=True,
                                            value=">="
                                        )

    return main


@solara.component
def SelectableTable(
    table,
    items_per_page: int = 10,
    on_selected_indices: Optional[Callable[[List[int]], None]] = None,
    **kwargs
):
    """An ipyvuetify DataTable with checkbox selection.

    Displays a paginated table with selectable rows.  Reports the
    indices (into *table*) of the currently selected rows.
    """
    page, set_page = solara.use_state(1)
    selected, set_selected = solara.use_state([])

    # Build vuetify column headers from the table
    def handle_input(msg):
        new_entries = [item[col_unique_row_index] for item in msg['new']]
        set_selected(new_entries)
        if on_selected_indices is not None and len(msg['new']):
            indices = [item[col_unique_row_index] for item in msg['new']]
            on_selected_indices(indices)

    mast_table = MastTable(
        table,
        item_key=col_unique_row_index,
        items_per_page=items_per_page,
        page=page,
        on_page=set_page,
        **kwargs
    )
    mast_table.selected_rows = [
        item for item in mast_table.items
        if item[col_unique_row_index] in selected
    ]
    mast_table.observe(handle_input, 'selected_rows')
    display(mast_table)
    solara.Info(f"{selected=}")


@solara.component
def CrossFilterSelectableTable(
    table,
    items_per_page: int = 10
):
    """A selectable table that participates in cross-filtering.

    * Incoming cross-filters from other components narrow which rows
      are shown.
    * When the user checks rows, a filter is set so that *other*
      cross-filter consumers only see the selected rows.
    """
    filter, set_filter = solara.use_cross_filter(id(table), "selectable-table")

    # Apply incoming cross-filter
    if filter is not None:
        table_filtered = table[filter]
    else:
        table_filtered = table

    def on_selected_indices(indices: List[int]):
        if not indices:
            # Nothing selected → don't restrict other components
            set_filter(None)
        else:
            # Build a boolean mask over the *original* table
            mask = np.isin(table[col_unique_row_index], indices)
            set_filter(mask)

    solara.Info(f"Showing {len(table_filtered)} of {len(table)} rows")
    SelectableTable(
        table_filtered,
        on_selected_indices=on_selected_indices,
        items_per_page=items_per_page
    )


@solara.component
def CrossFilterMastTable(observations):
    solara.provide_cross_filter()
    with solara.Column():
        CrossFilterSelect(observations, "optical_element")
        CrossFilterSlider(observations, "visit", mode='>=')
        CrossFilterSelectableTable(observations)
