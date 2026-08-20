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
from mast_table.base import BaseMastTable, serialize, col_unique_row_index
from mast_table.cross_filter_utils import (
    operator_map, num_py_type, table_filter_values, table_range,
    slide_or_select, step_size, build_select_items,
    build_select_filter_preview,
)


@solara.component
def FilterModeButtons(
    mode,
    set_mode,
    dense=False
):
    """Comparison operator selector for slider widgets.

    Parameters
    ----------
    - `mode`: The astropy Table to filter.
    - `set_mode`: Callback for updating filter's mask.
    - `dense`: Boolean to condense button options.

    """
    with solara.ToggleButtonsSingle(
        value=mode,
        on_value=set_mode,
        dense=dense
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
    invert,
    set_invert,
    multiple=None,
    set_multiple=None
):
    """A menu for widgets that manages options for mode, inversion, and multiple.

    Parameters
    ----------
    - `invert`: Whether to invert the selection.
    - `set_invert`: Callback for updating filter's inversion.
    - `multiple`: Whether to allow multiple values to be selected.
    - `set_multiple`: Callback for updating filter's ability to select multiple values.

    """
    with v.Container(
        fluid=True,
        class_="pa-0 ma-0",
    ):
        v.Switch(
            v_model=invert,
            on_v_model=set_invert,
            label="Invert filter",
            hide_details=True,
            dense=True,
        )

        if multiple is not None:
            v.Switch(
                v_model=multiple,
                on_v_model=set_multiple,
                label="Select multiple",
                hide_details=True,
                dense=True,
            )


@solara.component
def RemoveConditionButton(
    filter_id: str,
    on_remove=None,
):
    """Button for removal of filter.

    Parameters
    ----------
    - `filter_id`: The unique filter instance ID.
    - `on_remove`: Callback to remove this filter from parent filter list.

    """
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
    set_filter_mode: Callable,
    table_filtered: Table,
    initial_values=None,
    max_unique: int = 100,
    multiple: bool = False,
    invert: bool = False,
    mode: str = "==",
    configurable: bool = True,
    classes: List[str] = [],
):
    """A Select widget that will cross filter an astropy Table.

    Parameters
    ----------
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

    filter_values, set_filter_values = solara.use_state(initial_values)
    solara.use_effect(
        lambda: set_filter_values(initial_values or []),
        [initial_values],
    )
    invert, set_invert = solara.use_state_or_update(invert)
    multiple, set_multiple = solara.use_state_or_update(multiple)

    def update_mode(new_invert):
        set_invert(new_invert)
        set_filter_mode("!=" if new_invert else "==")

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

    with solara.VBox(classes=classes) as main:
        with solara.Column():
            if len(items) < 5:
                # set styling for compact checkboxes
                solara.Style(
                    """
                    .compact-checkboxes .v-input {
                        margin-bottom: 0px !important;
                        margin-top: 0px !important;
                    }

                    .compact-checkboxes .v-input__control {
                        min-height: 24px !important;
                    }

                    .compact-checkboxes .v-input__slot {
                        margin: 0 !important;
                        min-height: 24px !important;
                    }

                    .compact-checkboxes .v-input--selection-controls {
                        margin-top: 0 !important;
                        margin-bottom: 0 !important;
                        padding-top: 0 !important;
                        padding-bottom: 0 !important;
                    }
                    """
                )

                with v.Container(
                    fluid=True,
                    class_="pa-0 ma-0 compact-checkboxes",
                ):
                    for opt in items:
                        checked = opt["value"] in filter_values

                        def toggle_value(checked, value=opt["value"]):
                            if checked:
                                set_filter_values(filter_values + [value])
                            else:
                                set_filter_values(
                                    [v for v in filter_values if v != value]
                                )

                        solara.Checkbox(
                            value=checked,
                            on_value=toggle_value,
                            label=opt["text"],
                        )

                    with solara.Row(
                        style={
                            "align-items": "center",
                            "justify-content": "space-between",
                            "width": "100%",
                            "padding": 0,
                        }
                    ):
                        solara.Button(
                            "Select All",
                            on_click=lambda: set_filter_values(
                                [item["value"] for item in items]
                            ),
                            text=True,
                            style={"background-color": "#00627e", "color": "white"}
                        )

                        solara.Button(
                            "Clear All",
                            on_click=lambda: set_filter_values([]),
                            text=True,
                            style={"background-color": "#00627e", "color": "white"}
                        )

            else:
                # creating selection dropdown
                Select.element(
                    value=value,
                    items=items,
                    on_value=set_values_and_filter,
                    label="",
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

                # creating settings menu
                if configurable:
                    SettingsMenu(
                        invert,
                        update_mode,
                        multiple=multiple,
                        set_multiple=set_multiple
                    )

    return main


@solara.component
def CrossFilterSlider(
    table,
    column: str,
    filter_id: str,
    set_mask: Callable,
    set_filter_mode: Callable,
    initial_value=None,
    mode: str = ">=",
    configurable: bool = True,
):
    """A Slider widget that will cross filter an astropy Table.

    See [use_cross_filter](/documentation/api/hooks/use_cross_filter)
    for more information about how to use cross filtering.

    Parameters
    ----------
    - `table`: The astropy Table to filter.
    - `column`: The column to filter on.
    - `filter_id`: The unique filter instance ID.
    - `set_mask`: Callback for updating filter's mask.
    - `set_filter_mode`: Callback for updating filter's mode.
    - `initial_value`: The initial value to set for the slider.
    - `mode`: The mode to use for filtering. Can be one of `==`, `>=`, `<=`, `>`, `<`.
    - `configurable`: Whether to show a configuration button.

    """
    filter_value, set_filter_value = solara.use_state(initial_value)
    solara.use_effect(
        lambda: set_filter_value(initial_value),
        [initial_value],
    )
    mode, set_mode = solara.use_state_or_update(mode)

    def update_mode(new_mode):
        set_mode(new_mode)
        set_filter_mode(new_mode)

    vmin, vmax = table_range(table, column)

    py_type = num_py_type(table, column)

    def reset():
        if initial_value is not None:
            set_filter_value(initial_value)
        else:
            set_filter_value(vmin)

    solara.use_memo(reset, dependencies=[column])

    def update_filter():
        filter = None
        if filter_value:
            filter = operator_map[mode](table[column], filter_value)
        set_mask(filter_id, filter)

    solara.use_memo(update_filter, dependencies=[filter_value, mode])

    with solara.VBox() as main:
        label = f"Condition {mode} "
        if filter_value is not None:
            label = label + f"{filter_value}"

        solara.Style(
            """
            .crossfilter-slider .v-slider__thumb::before {
                display: none !important;
            }

            .crossfilter-slider .v-input--is-focused .v-slider__thumb::before {
                display: none !important;
            }
            """
        )

        input_args = {
            "label": None,
            "continuous_update": False,
            "style": {
                "max-width": "50px",
                "padding-top": "5px",
                "padding-bottom": "5px"
            },
        }

        slider_args = {
            "label": "",
            "min": vmin,
            "max": vmax,
            "step": step_size(vmin, vmax),
            "thumb_label": False,
            "tick_labels": False,
        }

        # creating slider
        with solara.Row(
            style={"alignItems": "end"},
            classes=["crossfilter-slider"]
        ):
            if issubclass(py_type, (int, np.integer)):
                solara.InputInt(
                    value=filter_value,
                    on_value=set_filter_value,
                    **input_args
                )
                solara.SliderInt(
                    value=filter_value,
                    on_value=set_filter_value,
                    **slider_args
                )
            elif issubclass(py_type, (float, np.floating)):
                solara.InputFloat(
                    value=filter_value,
                    on_value=set_filter_value,
                    **input_args
                )
                solara.SliderFloat(
                    value=filter_value,
                    on_value=set_filter_value,
                    **slider_args
                )

        # creating settings menu
        if configurable:
            FilterModeButtons(
                mode=mode,
                set_mode=update_mode,
                dense=True,
            )

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

    Parameters
    ----------
    table : `~astropy.table.Table`
        A table to load.

    items_per_page : int (optional, default is 10)
        Number of items to render on each page.

    on_selected_indices : callable (optional, default is `None)
        Callback on selected indices.

    drawer_open : bool (optional, default is `True`)
        If `True`, the CrossFilterMenu sidepanel is initialized
        open.

    set_drawer_open: callable (optional, default is `None)
        Callback to open CrossFilterMenu sidepanel.

    **kwargs
        Remaining keyword arguments are passed to MastTable.

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

        mt = BaseMastTable(
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
def MastTable(table, **kwargs):
    """A selectable table that participates in cross-filtering.

    * Incoming cross-filters from other components narrow which rows
      are shown.
    * When the user checks rows, a filter is set so that *other*
      cross-filter consumers only see the selected rows.
    * Conditions are set and tracked in a popout window.

    Parameters
    ----------
    table : `~astropy.table.Table`
        A table to load.

    **kwargs
        Keyword arguments are passed to SelectableTable.

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

    def add_filter(opt):
        new_filters = filters + [
            {
                "id": str(uuid.uuid4()),
                "column": pending_column,
                "value": pending_value,
                "mode": pending_mode if opt == "slider" else "==",
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
            unique_values, _ = build_select_items(
                table[default_column]
            )
            set_pending_value(unique_values[0] if unique_values else "")

    def remove_filter(filter_id):
        set_filters([f for f in filters if f["id"] != filter_id])

        updated = dict(filter_masks)
        updated.pop(filter_id, None)
        set_filter_masks(updated)

    def update_filter_mode(filter_id, new_mode):
        set_filters(
            [
                {**f, "mode": new_mode} if f["id"] == filter_id else f
                for f in filters
            ]
        )

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
    expanded_ids, set_expanded_ids = solara.use_state(set())

    with solara.Column(
        style={
            "overflow-y": "auto",
        }
    ):
        with solara.Row():
            # creating popout conditions panel
            with solara.Card(
                style={
                    "display": "block" if drawer_open else "none",
                    "width": "320px",
                    "flex-shrink": "0",
                    "min-height": "420px",
                    "max-height": "550px",
                    "overflow-y": "auto",
                }
            ):
                with solara.Row(
                    style={
                        "align-items": "center",
                        "justify-content": "space-between",
                        "width": "100%",
                        "padding": 0,
                    }
                ):
                    solara.Markdown("##Active conditions")
                    if len(filters) > 1:
                        solara.Style(
                            """
                            .custom-toggle .v-btn {
                                background-color: transparent# !important;
                                color: #00627e !important;
                                height: 40px !important;
                                width: 50px !important;
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

                        solara.Style(
                            """
                            .reset-button {
                                min-width: 0px !important;
                                width: 40px !important;
                                height: 40px !important;
                                padding: 0 !important;
                            }
                            """
                        )

                        with solara.Tooltip("Remove all active filters"):
                            solara.Button(
                                icon_name="mdi-refresh",
                                on_click=lambda *args: (set_filters([]), set_filter_masks({})),
                                style={"background-color": "#00627e", "color": "white"},
                                classes=["reset-button"]
                            )

                # creating slide/select for each active condition
                for i, f in enumerate(filters):
                    with solara.Row(style={"width": "100%"}):
                        solara.Style(
                            """
                            .filter-card .v-card {
                                padding: 0 !important;
                            }

                            .filter-card .v-card__text {
                                padding: 4px 8px !important;
                            }

                            .filter-card .v-card__actions {
                                padding: 0px 8px !important;
                            }
                            """
                        )

                        with solara.Card(
                            classes=["filter-card"],
                            style={
                                "border": "2px solid #00627e",
                                "box-shadow": "none",
                                "width": "100%",
                            }
                        ):
                            opt = slide_or_select(table, f["column"])
                            initial_val = f["value"] if f.get("value") is not None else None

                            is_expanded = f["id"] in expanded_ids

                            def toggle(filter_id=f["id"]):
                                ids = set(expanded_ids)
                                if filter_id in ids:
                                    ids.remove(filter_id)
                                else:
                                    ids.add(filter_id)
                                set_expanded_ids(ids)

                            with solara.Row(
                                style={
                                    "align-items": "center",
                                    "justify-content": "space-between",
                                    "width": "100%",
                                    "padding": 0,
                                }
                            ):
                                solara.Style(
                                    """
                                    .v-btn.filter-column {
                                        text-transform: none !important;
                                        font-family: inherit !important;
                                        font-size: inherit !important;
                                        font-weight: bold !important;
                                        letter-spacing: normal !important;
                                    }
                                    """
                                )

                                label = f'{f["column"]} {f["mode"]}'

                                solara.Button(
                                    label=label,
                                    on_click=toggle,
                                    icon_name=(
                                        "mdi-chevron-up" if is_expanded
                                        else "mdi-chevron-down"
                                    ),
                                    text=True,
                                    classes=["filter-column"],
                                    style={
                                        "margin-left": "-8px",
                                        "justify-content": "flex-start",
                                        "text-align": "left",
                                        "padding-left": "8px",
                                        "padding-right": "8px",
                                        "min-width": "0",
                                        "flex-grow": "1",
                                    },
                                )

                                with solara.Div():
                                    RemoveConditionButton(f["id"], remove_filter)

                            with solara.Div(
                                style={
                                    "display": "block" if is_expanded else "none"
                                }
                            ):
                                filter_id = f["id"]

                                def on_mode_change(new_mode, filter_id=filter_id):
                                    update_filter_mode(filter_id, new_mode)

                                if opt == "slider":
                                    CrossFilterSlider(
                                        table,
                                        f["column"],
                                        filter_id=filter_id,
                                        set_mask=set_mask,
                                        set_filter_mode=on_mode_change,
                                        mode=f["mode"],
                                        initial_value=initial_val,
                                    )
                                else:
                                    other_masks = [
                                        mask
                                        for fid, mask in filter_masks.items()
                                        if fid != filter_id and mask is not None
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
                                        filter_id=filter_id,
                                        set_mask=set_mask,
                                        set_filter_mode=on_mode_change,
                                        initial_values=initial_val,
                                        table_filtered=table_filtered
                                    )

                if not len(filters):
                    solara.Markdown("No active conditions")

                # creating add condition section
                solara.Markdown("##Add condition")

                column_names = table.colnames
                column_names.sort(key=str.casefold)
                if col_unique_row_index in column_names:
                    # never give the internal unique column as an option
                    column_names.remove(col_unique_row_index)

                # handling pending val initialization when col changes
                def on_pending_column_change(column):
                    set_pending_column(column)

                    opt = slide_or_select(table, column)

                    if opt == "slider":
                        vmin, _ = table_range(table, column)
                        set_pending_value(vmin)
                    else:
                        unique_values, _ = build_select_items(table[column])
                        set_pending_value(
                            unique_values[0] if unique_values else ""
                        )

                v.Autocomplete(
                    label="Column",
                    items=column_names,
                    v_model=pending_column,
                    on_v_model=on_pending_column_change,
                )

                opt = slide_or_select(table, pending_column)
                fully_masked = False

                # set initial column pending value
                if pending_value in ("", None):
                    if opt == "slider":
                        vmin, vmax = table_range(table, pending_column)
                        pending_value = vmin
                    else:
                        unique_values, fully_masked = build_select_items(
                            table[pending_column]
                        )
                        pending_value = unique_values[0] if unique_values else ""

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

                    py_type = num_py_type(table, pending_column)

                    label = f"Condition {pending_mode} {pending_value}"
                    solara.Markdown(label)

                    table_filtered = table[
                        combined_mask
                    ] if combined_mask is not None else table
                    comparison = operator_map[pending_mode]
                    slider_mask = comparison(table_filtered[pending_column], pending_value)

                    if issubclass(py_type, (int, np.integer)):
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

                    elif issubclass(py_type, (float, np.floating)):
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

                    solara.Markdown(
                        (
                            f"<div style='font-size: 12px; text-align: right;'>"
                            f"{len(table_filtered[slider_mask])} of {len(table_filtered)} "
                            "after filtering</div>"
                        )
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
                        {"value": pending_value}
                        if pending_value not in ("", None)
                        else None
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
                        on_click=lambda *args: add_filter(opt),
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
