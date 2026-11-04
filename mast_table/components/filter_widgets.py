import functools
from typing import List, Callable

import reacton.ipyvuetify as v
import solara

from astropy.table import Table
from mast_table.utils import (
    operator_map, table_filter_values, table_range,
    step_size, build_select_filter_preview,
)

from mast_table.components.cross_filter_select import Select
from mast_table.components.cross_filter_input import Input


__all__ = [
    'CrossFilterSelect',
    'CrossFilterSlider',
    'FilterModeButtons',
    'SettingsMenu',
    'RemoveFilterButton',
]


def preserve_docstring(func):
    # when a function is decorated with `solara.component`,
    # the function's docstring is not passed through to the
    # wrapped function unless you also use this decorator.

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


@preserve_docstring
@solara.component
def FilterModeButtons(
    mode,
    set_mode,
    dense=False
):
    """
    Comparison operator selector for slider widgets.

    Parameters
    ----------
    mode : str
        The mode to use for filtering. Can be one of `==`, `>=`, `<=`, `>`, `<`.
    set_mode : callable
        Callback for updating filter's mask.
    dense : bool, optional
        Boolean to condense button options.

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


@preserve_docstring
@solara.component
def SettingsMenu(
    invert,
    set_invert,
    multiple=None,
    set_multiple=None
):
    """
    A menu for widgets that manages options for mode, inversion, and multiple.

    Parameters
    ----------
    invert : bool
        Whether to invert the selection.
    set_invert : callable
        Callback for updating filter's inversion.
    multiple : bool, optional
        Whether to allow multiple values to be selected.
    set_multiple : callable, optional
        Callback for updating filter's ability to select multiple values.

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
            density="compact",
            color="primary"
        )

        if multiple is not None:
            v.Switch(
                v_model=multiple,
                on_v_model=set_multiple,
                label="Select multiple",
                hide_details=True,
                density="compact",
                color="primary"
            )


@preserve_docstring
@solara.component
def RemoveFilterButton(
    filter_id: str,
    on_remove=None,
):
    """
    Button for removal of filter.

    Parameters
    ----------
    filter_id : str
        The unique filter instance ID.
    on_remove : callable, optional
        Callback to remove this filter from parent filter list.

    """
    solara.Button(
        icon_name="mdi-close",
        on_click=lambda: on_remove(filter_id),
        style={"background-color": "light-dark(#00617e, #013b4d)", "color": "white"},
        classes=["close-button"],
    )
    solara.Style(
        """
        .close-button {
            min-width: 40px;
            width: 40px;
            height: 30px;
            padding: 0;
        }
        """
    )


@preserve_docstring
@solara.component
def CrossFilterSelect(
    table: Table,
    column: str,
    filter_id: str,
    set_mask: Callable,
    set_filter_mode: Callable,
    filtered_table: Table,
    initial_values=None,
    max_unique: int = 100,
    multiple: bool = False,
    invert: bool = False,
    configurable: bool = True,
    classes: List[str] = [],
):
    """
    A Select widget that will cross filter an `astropy.table.Table`.

    Parameters
    ----------
    table : `astropy.table.Table`
        The astropy Table to filter.
    column : str
        The column to filter on.
    filter_id : str
        The unique filter instance ID.
    set_mask : callable
        Callback for updating filter's mask.
    set_filter_mode : callable
        Callback for updating filter mode
    filtered_table : `astropy.table.Table`
        Table after filtering.
    initial_values : list, optional
        The initial values to set as selected. Default is None.
    max_unique : int, optional
        The maximum number of unique values to show in the dropdown. Default is 100.
    multiple : bool, optional
        Whether to allow multiple values to be selected. Default is False.
    invert : bool, optional
        Whether to invert the selection. Default is False.
    configurable : bool, optional
        Whether to show the configuration button. Default is True.
    classes : list, optional
        Additional CSS classes to add to the main widget.

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
        filtered_table=filtered_table
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

    if multiple:
        value = [item for item in items if item["value"] in filter_values]
    else:
        if filter_values:
            value = next(
                (item for item in items if item["value"] == filter_values[0]),
                None,
            )
        else:
            value = None

    with solara.VBox(classes=classes) as main:
        with solara.Column():
            if len(items) < 5:
                # set styling for compact checkboxes
                solara.Style(
                    """
                    .compact-checkboxes .v-selection-control {
                        min-height: 24px;
                        padding: 0;
                    }

                    .compact-checkboxes .v-selection-control__wrapper {
                        height: 24px;
                    }

                    .compact-checkboxes .v-label {
                        margin: 0;
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

                        v.Checkbox(
                            v_model=checked,
                            on_v_model=toggle_value,
                            label=opt["text"],
                            density="compact",
                            hide_details=True,
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
                            style={
                                "background-color": "light-dark(#00617e, #013b4d)",
                                "color": "white"
                            }
                        )

                        solara.Button(
                            "Clear All",
                            on_click=lambda: set_filter_values([]),
                            text=True,
                            style={
                                "background-color": "light-dark(#00617e, #013b4d)",
                                "color": "white"
                            }
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
                    count=len(filtered_table),
                    messages=(
                        f"Too many unique values, will only show the first {max_unique}"
                        if len(value_counts) > max_unique else ""
                    ),
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


@preserve_docstring
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
    """
    A Slider widget that will cross filter an astropy Table.

    See `solara.use_cross_filter` for more information about how to use cross filtering.

    Parameters
    ----------
    table : `astropy.table.Table`
        The astropy Table to filter.
    column : str
        The column to filter on.
    filter_id : str
        The unique filter instance ID.
    set_mask : callable
        Callback for updating filter's mask.
    set_filter_mode : callable
        Callback for updating filter's mode.
    initial_value : float, optional
        The initial value to set for the slider.
    mode : str, optional
        The mode to use for filtering. Can be one of `==`, `>=`, `<=`, `>`, `<`.
    configurable : bool, optional
        Whether to show a configuration button. Default is True.

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

    def reset():
        if initial_value is not None:
            set_filter_value(initial_value)
        else:
            set_filter_value(vmin)

    solara.use_memo(reset, dependencies=[column])

    def update_filter():
        filter = None
        if filter_value is not None:
            filter = operator_map[mode](table[column], filter_value)
        set_mask(filter_id, filter)

    solara.use_memo(update_filter, dependencies=[filter_value, mode])

    with solara.VBox() as main:
        solara.Style(
            """
            .crossfilter-slider .v-slider {
                transform: translateY(10px);
            }
            """
        )

        slider_args = {
            "label": "",
            "min": vmin,
            "max": vmax,
            "step": step_size(vmin, vmax),
            "thumb_label": False,
            "tick_labels": False,
            "color": "primary",
        }

        # creating slider
        with solara.Row(
            style={"alignItems": "center"},
            classes=["crossfilter-slider"]
        ):
            Input.element(
                value=filter_value,
                min=vmin,
                max=vmax,
                step=step_size(vmin, vmax),
                on_value=set_filter_value,
            )

            v.Slider(
                v_model=filter_value,
                on_v_model=set_filter_value,
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
