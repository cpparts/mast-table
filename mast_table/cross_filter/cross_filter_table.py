import operator
from typing import List, Optional, Callable

import functools
import uuid

import numpy as np
from IPython.display import display

import solara
from solara.components.cross_filter import Select
import reacton.ipyvuetify as v

from astropy.table import Table
from mast_table.base import MastTable, serialize, col_unique_row_index
from mast_table.cross_filter.utils import (
    table_py_types, table_filter_values, table_range,
    slide_or_select, step_size, build_select_items,
    build_select_filter_preview,
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
def SettingsMenu(
    btn,
    invert,
    set_invert,
    mode=None,
    set_mode=None,
    multiple=None,
    set_multiple=None
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

                        if mode:
                            FilterModeButtons(
                                mode=mode,
                                set_mode=set_mode,
                            )

                        if multiple is not None:
                            v.Switch(
                                v_model=multiple,
                                on_v_model=set_multiple,
                                label="Select multiple"
                            )


@solara.component
def RemoveConditionButton(on_remove, filter_id):
    solara.Button(
        icon_name="mdi-close",
        on_click=lambda: on_remove(filter_id),
        style={"background-color": "#00627e", "color": "white"},
        classes=["close-button"],
    )
    solara.Style(
        """
        .close-button {
            min-width: 10px !important;
            width: 30px !important;
            height: 30px !important;
            padding: 0 !important;
        }
        """
    )


@solara.component
def CrossFilterSelect(
    table: Table,
    column: str,
    filter_id: str,
    set_mask: Callable,
    table_filtered: Table,
    initial_values=None,
    on_remove=None,
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
    - `on_remove`: Callback to remove this filter from parent filter list.
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

    filter_values, set_filter_values = solara.use_state(initial_values)
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

    items, value_counts = build_select_filter_preview(
        table,
        column,
        max_unique=max_unique,
        table_filtered=table_filtered
    )

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
        if hasattr(table[column], 'mask'):
            unmasked_values_as_strings = (
                list(table[column].data[~table[column].mask].astype(str)) +
                ['--']  # masked value
            )
        else:
            unmasked_values_as_strings = table[column].astype(str)
        if (
            len(filter_values) == 0 or
            (not invert and set(filter_values).issuperset(unmasked_values_as_strings))
        ):
            set_mask(filter_id, None)
            return

        mask = table_filter_values(table, column, filter_values, invert=invert)
        set_mask(filter_id, mask)

    solara.use_memo(update_filter, dependencies=[filter_values, invert])

    value = (
        [{"value": v} for v in filter_values]
        if multiple
        else ({"value": filter_values[0]} if filter_values else None)
    )

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

                with solara.Div(
                    style={
                        "display": "flex",
                        "flex-wrap": "wrap",
                        "gap": "8px",
                        "margin-left": "auto",
                    }
                ):
                    # creating settings menu
                    if configurable:
                        SettingsMenu(
                            btn,
                            invert,
                            set_invert,
                            multiple=multiple,
                            set_multiple=set_multiple
                        )

                    if on_remove is not None:
                        RemoveConditionButton(on_remove, filter_id)

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
    on_remove=None,
    invert: bool = False,
    mode: str = ">=",
    configurable: bool = True,
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
    - `on_remove`: Callback to remove this filter from parent filter list.
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

    with v.Btn(v_on="x.on", icon=True) as btn:
        v.Icon(children=["mdi-settings"])

    with solara.VBox() as main:
        label = f"Condition {mode} " if not invert else f"Drop condition {mode} "
        if filter_value is not None:
            label = label + f"{filter_value}"
        with solara.Row(
            style={
                "flex-wrap": "wrap",
                "align-items": "center",
                "gap": "12px",
            }
        ):
            solara.Markdown(f"**{column}**")

            with solara.Div(
                style={
                    "display": "flex",
                    "flex-wrap": "wrap",
                    "gap": "8px",
                    "margin-left": "auto",
                }
            ):

                # creating settings menu
                if configurable:
                    SettingsMenu(
                        btn,
                        invert,
                        set_invert,
                        mode=mode,
                        set_mode=set_mode
                    )

                if on_remove is not None:
                    RemoveConditionButton(on_remove, filter_id)

        solara.Markdown(label, style={"color": "#6c6c6c", "font-size": "0.85em"})
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
            solara.SliderInt(**slider_args)
        elif issubclass(py_types[column], (float, np.floating)):
            solara.SliderFloat(**slider_args)
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
            filter_tray_open=drawer_open,
            **kwargs
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
def CrossFilterMastTable(table, **kwargs):
    """A selectable table that participates in cross-filtering.

    * Incoming cross-filters from other components narrow which rows
      are shown.
    * When the user checks rows, a filter is set so that *other*
      cross-filter consumers only see the selected rows.
    * Conditions are set and tracked in a popout window.
    """
    solara.provide_cross_filter()

    pending_column, set_pending_column = solara.use_state(
        table.colnames[0]
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

        default_column = table.colnames[0]
        set_pending_column(default_column)

        set_pending_mode(">=")

        opt = slide_or_select(table, default_column)
        if opt == "slider":
            vmin, _ = table_range(table, default_column)
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
                    width: 320px !important;
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
                                    "width": "100%",
                                }
                            ):
                                opt = slide_or_select(table, f["column"])
                                initial_val = f["value"] if f.get("value") is not None else None
                                if opt == "slider":
                                    CrossFilterSlider(
                                        table,
                                        f["column"],
                                        filter_id=f["id"],
                                        set_mask=set_mask,
                                        mode=f["mode"],
                                        initial_value=initial_val,
                                        on_remove=remove_filter
                                    )
                                else:
                                    other_masks = [
                                        mask
                                        for fid, mask in filter_masks.items()
                                        if fid != f["id"] and mask is not None
                                    ]

                                    if not other_masks:
                                        table_filtered = table
                                    elif pending_reducer == "AND":
                                        table_filtered = table[
                                            functools.reduce(
                                                operator.and_,
                                                other_masks
                                            )
                                        ]
                                    else:
                                        table_filtered = table[
                                            functools.reduce(
                                                operator.or_,
                                                other_masks
                                            )
                                        ]

                                    CrossFilterSelect(
                                        table,
                                        f["column"],
                                        filter_id=f["id"],
                                        set_mask=set_mask,
                                        initial_values=initial_val,
                                        on_remove=remove_filter,
                                        table_filtered=table_filtered
                                    )

                    if not len(filters):
                        solara.Markdown("No active conditions")

                    # creating add condition section
                    solara.Markdown("##Add condition")

                    column_names = table.colnames
                    if col_unique_row_index in column_names:
                        # never give the internal unique column as an option
                        column_names.remove(col_unique_row_index)

                    v.Select(
                        label="Column",
                        items=column_names,
                        v_model=pending_column,
                        on_v_model=set_pending_column,
                    )

                    opt = slide_or_select(table, pending_column)
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

                        vmin, vmax = table_range(table, pending_column)

                        py_types = table_py_types(table)

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
                            table[pending_column]
                        )

                        table_filtered = table[combined_mask] if combined_mask is not None else None

                        max_unique = 100

                        items, value_counts = build_select_filter_preview(
                            table,
                            pending_column,
                            max_unique=max_unique,
                            table_filtered=table_filtered,
                        )

                        value = (
                            {"value": unique_values[0]} if unique_values else None
                        )

                        def set_pending_select_value(selection):
                            if selection is None:
                                set_pending_value("")
                            else:
                                set_pending_value(selection["value"])

                        Select.element(
                            value=value,
                            items=items,
                            on_value=set_pending_select_value,
                            label="Value",
                            clearable=False,
                            return_object=True,
                            multiple=False,
                            filtered=pending_value is not None,
                            count=len(table_filtered) if table_filtered is not None else len(table),
                            messages=(
                                f"Too many unique values, will only show the first {max_unique}"
                                if len(value_counts) > max_unique else ""
                            ),
                            class_="solara-cross-filter-select",
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
                    table[combined_mask]
                    if combined_mask is not None
                    else table
                )
                SelectableTable(
                    filtered_table,
                    drawer_open=drawer_open,
                    set_drawer_open=set_drawer_open,
                    **kwargs
                )
