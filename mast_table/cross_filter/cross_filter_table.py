import operator
from typing import List, Optional, Callable

import functools
import uuid

import numpy as np
from IPython.display import display

import solara
from solara.components.cross_filter import Select
import reacton.ipyvuetify as v

from astropy.table import Table, join
from mast_table.mast_table import MastTable, serialize, col_unique_row_index
from mast_table.cross_filter.utils import (
    table_py_types, table_value_count, table_filter_values,
    table_range, slide_or_select, step_size, build_select_items,
)


@solara.component
def FilterModeButtons(
    mode,
    set_mode,
):
    with solara.ToggleButtonsSingle(
        value=mode,
        on_value=set_mode,
    ):
        solara.Button(
            icon_name="mdi-code-equal",
            icon=True,
            value="==",
        )
        solara.Button(
            icon_name="mdi-code-not-equal",
            icon=True,
            value="!=",
        )
        solara.Button(
            icon_name="mdi-code-less-than",
            icon=True,
            value="<",
        )
        solara.Button(
            icon_name="mdi-code-less-than-or-equal",
            icon=True,
            value="<=",
        )
        solara.Button(
            icon_name="mdi-code-greater-than",
            icon=True,
            value=">",
        )
        solara.Button(
            icon_name="mdi-code-greater-than-or-equal",
            icon=True,
            value=">=",
        )


@solara.component
def CrossFilterSelect(
    table: Table,
    column: str,
    filter_id: str,
    set_mask: Callable,
    initial_values=None,
    max_unique: int = 100,
    multiple: bool = False,
    invert: bool = False,
    configurable: bool = True,
    classes: List[str] = [],
):
    """A Select widget that will cross filter an astropy Table.

    ## Arguments

    - `table`: The astropy Table to filter.
    - `column`: The column to filter on.
    - `filter_id`: The unique filter instance ID.
    - `set_mask`: Callback for updating filter's mask.
    - `initial_values`: The initial values to set as selected.
    - `max_unique`: The maximum number of unique values to show in the dropdown.
    - `multiple`: Whether to allow multiple values to be selected.
    - `invert`: Whether to invert the selection.
    - `configurable`: Whether to show the configuration button.
    - `classes`: Additional CSS classes to add to the main widget.

    """
    if initial_values is None:
        initial_values = []
    elif isinstance(initial_values, str):
        initial_values = [initial_values]

    filter_values, set_filter_values = solara.use_state(initial_values or [])
    solara.use_effect(
        lambda: set_filter_values(initial_values or []),
        [initial_values],
    )
    invert, set_invert = solara.use_state_or_update(invert)
    multiple, set_multiple = solara.use_state_or_update(multiple)

    def clear_not_multiple():
        if not multiple and len(filter_values) > 1:
            set_filter_values([filter_values[0]])

    solara.use_effect(
        clear_not_multiple,
        [multiple]
    )

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
        if (
            len(filter_values) == 0 or
            set(filter_values) >= set(np.unique(table[column].astype(str)))
        ):
            set_mask(filter_id, None)
            return

        mask = table_filter_values(table, column, filter_values, invert=invert)
        set_mask(filter_id, mask)

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
    value = (
        [{"value": v} for v in filter_values]
        if multiple
        else ({"value": filter_values[0]} if filter_values else None)
    )

    # TODO: reacton bug, we cannot add this under any component context manager
    # this gives an error, probably because the button is added twice
    with v.Btn(v_on="x.on", icon=True) as btn:
        v.Icon(children=["mdi-settings"])
    with solara.VBox(classes=classes) as main:
        with solara.HBox(align_items="baseline"):
            with solara.Row(style={
                "flex-wrap": "wrap",
                "align-items": "center",
                "gap": "12px",
            }):
                solara.Markdown(f"**{column}**")

                # creating settings menu
                if configurable:
                    v_slots = [{"name": "activator", "variable": "x", "children": btn}]
                    with solara.Div(
                        style={
                            "display": "flex",
                            "flex-wrap": "wrap",
                            "gap": "8px",
                            "margin-left": "auto",
                        }
                    ):
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

                # creating selection dropdown
                label = (
                    "Condition = " if not invert else "Condition != "
                )
                Select.element(
                    value=value,
                    items=items,
                    on_value=set_values_and_filter,
                    label=label,
                    clearable=False,
                    return_object=True,
                    multiple=multiple,
                    filtered=len(filter_values) > 0,
                    count=len(table_filtered),
                    messages=(
                        f"Too many unique values, will only show the first {max_unique}"
                        if len(value_counts) > max_unique else ""
                    ),
                    class_="solara-cross-filter-select",
                )
    return main


@solara.component
def CrossFilterSlider(
    table,
    column: str,
    filter_id: str,
    set_mask: Callable,
    initial_value=None,
    invert=False,
    mode: str = ">=",
    configurable=True,
):
    """A Slider widget that will cross filter an astropy Table.

    See [use_cross_filter](/documentation/api/hooks/use_cross_filter)
    for more information about how to use cross filtering.

    ## Arguments

    - `table`: The astropy Table to filter.
    - `column`: The column to filter on.
    - `filter_id`: The unique filter instance ID.
    - `set_mask`: Callback for updating filter's mask.
    - `initial_value`: The initial value to set for the slider.
    - `invert`: If True, the filter will be inverted.
    - `mode`: The mode to use for filtering. Can be one of `==`, `>=`, `<=`, `>`, `<`.
    - `configurable`: Whether to show a configuration button.

    """
    filter_value, set_filter_value = solara.use_state(initial_value)
    solara.use_effect(
        lambda: set_filter_value(initial_value),
        [initial_value],
    )
    invert, set_invert = solara.use_state_or_update(invert)
    mode, set_mode = solara.use_state_or_update(mode)

    vmin, vmax = table_range(table, column)

    py_types = table_py_types(table)

    def reset():
        if initial_value is not None:
            set_filter_value(initial_value)
        else:
            set_filter_value(vmin)

    solara.use_memo(reset, dependencies=[column])

    def update_filter():
        filter = None
        if filter_value:
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
        set_mask(filter_id, filter)

    solara.use_memo(update_filter, dependencies=[filter_value, invert, mode])

    # TODO: reacton bug, see CrossFilterSelect
    with v.Btn(v_on="x.on", icon=True) as btn:
        v.Icon(children=["mdi-settings"])

    with solara.VBox() as main:
        label = f"Condition {mode} " if not invert else f"Drop condition {mode} "
        if filter_value is not None:
            label = label + f"{filter_value}"
        with solara.Div(style={"width": "100%"}):
            with solara.Row(
                style={
                    "flex-wrap": "wrap",
                    "align-items": "center",
                    "gap": "12px",
                }
            ):
                solara.Markdown(f"**{column}**")

                # creating settings menu
                if configurable:
                    with solara.Div(
                        style={
                            "display": "flex",
                            "flex-wrap": "wrap",
                            "gap": "8px",
                            "margin-left": "auto",
                        }
                    ):
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

                                            FilterModeButtons(
                                                mode=mode,
                                                set_mode=set_mode,
                                            )
                solara.Markdown(label)
            slider_args = {
                "label": "",
                "value": filter_value,
                "min": vmin,
                "max": vmax,
                "step": step_size(vmin, vmax),
                "on_value": set_filter_value,
                "thumb_label": True,
                "tick_labels": False,
            }
            # creating slider
            if issubclass(py_types[column], (int, np.integer)):
                solara.SliderInt(*slider_args)
            elif issubclass(py_types[column], (float, np.floating)):
                solara.SliderFloat(*slider_args)
            else:
                solara.Warning(f"{py_types[column]} not supported for Slider")

    return main


@solara.component
def SelectableTable(
    table,
    items_per_page: int = 10,
    on_selected_indices: Optional[Callable[[List[int]], None]] = None,
    drawer_open: bool = True,
    set_drawer_open=None,
    **kwargs
):
    """An ipyvuetify DataTable with checkbox selection.

    Displays a paginated table with selectable rows.  Reports the
    indices (into *table*) of the currently selected rows.
    """
    selected, set_selected = solara.use_state([])

    # Build vuetify column headers from the table
    def handle_input(msg):
        new_entries = [item[col_unique_row_index] for item in msg['new']]
        set_selected(new_entries)
        if on_selected_indices is not None and len(msg['new']):
            indices = [item[col_unique_row_index] for item in msg['new']]
            on_selected_indices(indices)

    def func():
        def on_change(change):
            set_drawer_open(change["new"])

        mt = MastTable(
            table,
            item_key=col_unique_row_index,
            items_per_page=items_per_page,
            filter_tray_open=drawer_open
        )
        mt.observe(on_change, 'filter_tray_open')
        return mt

    mast_table = solara.use_memo(
        func,
        [],
    )

    mast_table.selected_rows = [
        item for item in mast_table.items
        if item[col_unique_row_index] in selected
    ]
    mast_table.observe(handle_input, 'selected_rows')

    solara.use_effect(
        lambda: setattr(mast_table, "items", serialize(table)),
        [table]
    )

    display(mast_table)


@solara.component
def CrossFilterMastTable(observations):
    """A selectable table that participates in cross-filtering.

    * Incoming cross-filters from other components narrow which rows
      are shown.
    * When the user checks rows, a filter is set so that *other*
      cross-filter consumers only see the selected rows.
    * Conditions are set and tracked in a popout window.
    """
    solara.provide_cross_filter()

    pending_column, set_pending_column = solara.use_state(
        observations.colnames[0]
    )
    pending_value, set_pending_value = solara.use_state("")
    pending_mode, set_pending_mode = solara.use_state(">=")
    pending_reducer, set_pending_reducer = solara.use_state("AND")
    filter_masks, set_filter_masks = solara.use_state({})
    filters, set_filters = solara.use_state([])
    drawer_open, set_drawer_open = solara.use_state(True)

    def add_filter():
        new_filters = filters + [
            {
                "id": str(uuid.uuid4()),
                "column": pending_column,
                "value": pending_value,
                "mode": pending_mode
            }
        ]
        set_filters(new_filters)

        default_column = observations.colnames[0]
        set_pending_column(default_column)

        set_pending_mode(">=")

        opt = slide_or_select(observations, default_column)
        if opt == "slider":
            vmin, _ = table_range(observations, default_column)
            set_pending_value(vmin)
        else:
            set_pending_value("")

    def remove_filter(filter_id):
        set_filters([f for f in filters if f["id"] != filter_id])

        updated = dict(filter_masks)
        updated.pop(filter_id, None)
        set_filter_masks(updated)

    def set_mask(filter_id, mask):
        updated = dict(filter_masks)

        if mask is None:
            updated.pop(filter_id, None)
        else:
            updated[filter_id] = mask

        set_filter_masks(updated)

    active_masks = [
        mask for mask in filter_masks.values()
        if mask is not None
    ]

    if not active_masks:
        combined_mask = None
    elif pending_reducer == "AND":
        combined_mask = functools.reduce(
            operator.and_,
            active_masks
        )
    else:
        combined_mask = functools.reduce(
            operator.or_,
            active_masks
        )

    solara.lab.theme.themes.light.primary = "#00627e"

    with solara.Column(
        style={
            "overflow-y": "auto",
        }
    ):
        with solara.Row():
            # creating popout conditions panel
            if drawer_open:
                with solara.Card(
                    style="""
                    width: 320px;
                    flex-shrink: 0;
                    overflow-y: auto;
                    """
                ):
                    with solara.Row():
                        solara.Markdown("##Active conditions")
                        solara.Style(
                            """
                            .custom-toggle .v-btn {
                                background-color: transparent# !important;
                                color: #00627e !important;
                            }

                            .custom-toggle .v-btn.v-item--active {
                                background-color: #00627e !important;
                                color: white !important;
                            }
                            """
                        )

                        solara.ToggleButtonsSingle(
                            value=pending_reducer,
                            values=["AND", "OR"],
                            on_value=set_pending_reducer,
                            classes=["custom-toggle"],
                        )

                    # creating slide/select for each active condition
                    for i, f in enumerate(filters):
                        with solara.Row(style={"width": "100%"}):
                            with solara.Card(
                                style={
                                    "border": "2px solid #00627e",
                                    "box-shadow": "none",
                                }
                            ):
                                opt = slide_or_select(observations, f["column"])
                                initial_val = f["value"] if f.get("value") is not None else None
                                if opt == "slider":
                                    CrossFilterSlider(
                                        observations,
                                        f["column"],
                                        filter_id=f["id"],
                                        set_mask=set_mask,
                                        mode=f["mode"],
                                        initial_value=initial_val,
                                    )
                                else:
                                    CrossFilterSelect(
                                        observations,
                                        f["column"],
                                        filter_id=f["id"],
                                        set_mask=set_mask,
                                        initial_values=initial_val,
                                    )
                                with solara.Row(justify="end"):
                                    solara.Button(
                                        icon_name="mdi-close",
                                        on_click=lambda id=f["id"]: remove_filter(id),
                                        style={"background-color": "#00627e", "color": "white"}
                                    )
                    if not len(filters):
                        solara.Markdown("No active conditions")

                    # creating add condition section
                    solara.Markdown("##Add condition")
                    v.Select(
                        label="Column",
                        items=observations.colnames,
                        v_model=pending_column,
                        on_v_model=set_pending_column,
                    )

                    opt = slide_or_select(observations, pending_column)
                    fully_masked = False

                    # creating slide/select based on column user selects
                    if opt == "slider":
                        with solara.Row(
                            style={
                                "align-items": "center",
                                "gap": "8px",
                                "flex-wrap": "wrap",
                            }
                        ):
                            solara.Markdown("Operator")

                            FilterModeButtons(
                                mode=pending_mode,
                                set_mode=set_pending_mode,
                            )

                        vmin, vmax = table_range(observations, pending_column)

                        py_types = table_py_types(observations)

                        if pending_value in ("", None):
                            pending_value = vmin

                        label = f"Condition {pending_mode} {pending_value}"
                        solara.Markdown(label)

                        if issubclass(py_types[pending_column], (int, np.integer)):
                            solara.SliderInt(
                                label="",
                                value=int(pending_value),
                                min=int(vmin),
                                max=int(vmax),
                                step=step_size(vmin, vmax),
                                on_value=set_pending_value,
                                thumb_label=False,
                                tick_labels=False,
                            )

                        elif issubclass(py_types[pending_column], (float, np.floating)):
                            solara.SliderFloat(
                                label="",
                                value=float(pending_value),
                                min=float(vmin),
                                max=float(vmax),
                                step=step_size(vmin, vmax),
                                on_value=set_pending_value,
                                thumb_label=False,
                                tick_labels=False,
                            )

                    else:
                        unique_values, fully_masked = build_select_items(
                            observations[pending_column]
                        )

                        v.Select(
                            label="Value",
                            items=unique_values,
                            v_model=pending_value,
                            on_v_model=set_pending_value,
                        )

                    with solara.Row(justify="end"):
                        solara.Button(
                            label="Apply condition",
                            icon_name="mdi-plus",
                            on_click=lambda *args: add_filter(),
                            disabled=fully_masked,
                            style={"background-color": "#00627e", "color": "white"}
                        )
                    if fully_masked:
                        with solara.Row(justify="end"):
                            solara.Markdown("(Column fully masked)")

            with solara.Column(style="flex: 1; overflow: auto; min-height: 0"):
                filtered_table = (
                    observations[combined_mask]
                    if combined_mask is not None
                    else observations
                )
                SelectableTable(
                    filtered_table,
                    drawer_open=drawer_open,
                    set_drawer_open=set_drawer_open
                )
