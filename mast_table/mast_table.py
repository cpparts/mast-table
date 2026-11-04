import functools
import operator
import uuid
import warnings

from typing import List, Optional, Callable
import numpy as np

from IPython.display import display
import ipywidgets as widgets
from ipypopout import PopoutButton
from sidecar import Sidecar
import solara
import reacton.ipyvuetify as v

from mast_table.base import BaseMastTable, col_unique_row_index
from mast_table.components.filter_widgets import preserve_docstring
from mast_table.components import (
    Alert, Input, Select, RemoveFilterButton, CrossFilterSlider,
    CrossFilterSelect, FilterModeButtons
)
from mast_table.utils import (
    operator_map, table_range,
    slide_or_select, step_size, build_select_items,
    build_select_filter_preview,
)


# register loaded table widgets as they're initialized
_table_widgets = []

__all__ = [
    'MastTable',
    'get_current_table',
    'SelectableTable',
    'MastTableView',
]


class MastTable:
    """
    A selectable table with row filtering.
    """

    def __init__(self, table, **kwargs):
        """
        Parameters
        ----------
        table : `~astropy.table.Table`
            A table to load.

        **kwargs
            Keyword arguments are passed to `~mast_table.base.BaseMastTable`.
        """
        _table_widgets.append(self)

        self._mast_table_source = table
        self.widget = BaseMastTable(
            table,
            item_key=col_unique_row_index,
            **kwargs,
        )
        self.sidecar = kwargs.get("sidecar", None)
        self.popout_button = None

        self.filter_mask = np.ones(len(self.table), dtype=bool)
        self._selected_mask = np.zeros(len(self.table), dtype=bool)
        self._updating_selection = False

        self._selection_listeners = []
        self._filter_listeners = []

        self.widget.observe(self._on_selection_change, names="selected_rows")

    def __getattr__(self, name):
        return getattr(self.widget, name)

    def add_selection_listener(self, callback):
        self._selection_listeners.append(callback)

    def add_filter_listener(self, callback):
        self._filter_listeners.append(callback)

    def remove_selection_listener(self, callback):
        if callback in self._selection_listeners:
            self._selection_listeners.remove(callback)

    def remove_filter_listener(self, callback):
        if callback in self._filter_listeners:
            self._filter_listeners.remove(callback)

    def _notify_listeners(self, listeners, mask):
        for callback in listeners:
            callback(mask.copy())

    def _validate_mask(self, mask, length):
        mask = np.asarray(mask)
        if mask.dtype.kind != 'b':
            raise TypeError("Invalid mask provided. Must be a boolean array.")
        if mask.shape != (length,):
            raise ValueError("Mask length must match the table.")
        return mask

    def _on_selection_change(self, change):
        if self._updating_selection:
            return

        mask = np.zeros(len(self.table), dtype=bool)
        mask[np.asarray(change["new"], dtype=int)] = True

        self._selected_mask = mask
        self._notify_listeners(self._selection_listeners, mask)

    def _set_filter_mask(self, mask):
        self.filter_mask = mask
        self._update_selection()
        self._notify_listeners(self._filter_listeners, self.filter_mask)

    @property
    def selected_mask(self):
        return self._selected_mask.copy()

    @selected_mask.setter
    def selected_mask(self, mask):
        self.set_selected_mask(mask)

    def set_selected_mask(self, mask, notify=True):
        self._selected_mask = self._validate_mask(mask, len(self.table))
        self._update_selection()

        if notify:
            self._notify_listeners(
                self._selection_listeners,
                self._selected_mask,
            )

    def _update_selection(self):
        mask = self._selected_mask & self.filter_mask
        self._updating_selection = True
        try:
            self.widget.selected_rows = [
                str(i) for i in np.flatnonzero(mask)
            ]
        finally:
            self._updating_selection = False

    @property
    def selected_rows(self):
        return self.widget.selected_rows

    @selected_rows.setter
    def selected_rows(self, value):
        if value and isinstance(value[0], dict):
            value = [item[col_unique_row_index] for item in value]

        mask = np.zeros(len(self.table), dtype=bool)
        mask[np.asarray(value, dtype=int)] = True
        self.selected_mask = mask

    @property
    def selected_rows_table(self):
        return self.table[self._selected_mask]

    @property
    def items(self):
        return self.widget.items

    @items.setter
    def items(self, value):
        self.widget.items = value

    def _set_popout_button(self, popout_button):
        self.popout_button = popout_button

    def _ipython_display_(self):
        display(
            MastTableView(
                self.table,
                base_mast_table=self.widget,
                on_popout_button=self._set_popout_button,
                on_filter_mask=self._set_filter_mask,
            )
        )

    def show(self, loc="inline", title="mast-table", ref=None):
        """
        Display MastTable.

        Parameters
        ----------
        loc : str, optional. Default "inline"
            Location to display the MastTable. Supported locations are:

            - "inline": Displays the MastTable inline in a notebook.
            - "sidecar": Displays the MastTable in a separate JupyterLab window from the
              notebook, with location decided by 'anchor'. Anchor options available from
              ``jupyterlab-sidecar`` are: {'split-right', 'split-left', 'split-top',
              'split-bottom', 'tab-before', 'tab-after', 'right'}.
              Example ``loc`` for a sidecar at the bottom is ``loc='sidecar:split-bottom'``.

            - "popout": Display the MastTable in a detached display. By default, a new
              window will open (browser popup permissions required). Anchor options are
              ``popout:window`` (default, opens MastTable in a new, detached popout) or
              ``popout:tab`` (opens MastTable in a new, detached tab in your browser).

        title : str, optional. Default "mast-table"
            The title of the sidecar tab, only applicable to a "sidecar" display.

        ref : sidecar.Sidecar [1]_ or app, optional. Default None.
            An existing reference sidecar or app with ``.sidecar`` attribute. Opens the new sidecar
            display with its anchor defined relative to the ref sidecar. Only applicable to a
            "sidecar" display.

        References
        ----------
        .. [1] `JupyterLab Sidecar
               <https://github.com/jupyter-widgets/jupyterlab-sidecar#sidecar>`_
        """
        valid_anchors = [
            "split-right", "split-left", "split-top", "split-bottom",
            "tab-before", "tab-after", "right", None
        ]

        if type(loc) is not str:
            raise TypeError("Invalid loc type provided. Must be a string.")

        if loc == "inline":
            self._ipython_display_()
        elif loc.startswith("sidecar"):
            anchor = None if loc == "sidecar" else loc.split(":")[-1]

            if anchor not in valid_anchors:
                raise ValueError(
                    "Invalid anchor provided. Must be one of the available from "
                    "``jupyterlab-sidecar``. Valid anchors are: "
                    f"{valid_anchors[:-1]}")

            if ref and not isinstance(ref, Sidecar):
                try:
                    ref = ref.sidecar
                except AttributeError:
                    raise ValueError(
                        "Invalid ref provided. Must be a `Sidecar` instance or an "
                        "app with `.sidecar` attribute."
                    )

                if ref is None or not isinstance(ref, Sidecar):
                    raise ValueError(
                        "Invalid ref provided. App must have an initialized `.sidecar` "
                        "attribute."
                    )

            sc = Sidecar(title=title, anchor=anchor, ref=ref)
            with sc:
                self._ipython_display_()
            self.sidecar = sc

        elif loc.startswith("popout"):
            if self.popout_button is None:
                self._hidden_output = widgets.Output()
                self._hidden_output.layout.display = "none"

                with self._hidden_output:
                    self._ipython_display_()

            anchor = None if loc == "popout" else loc.split(":")[-1]
            if anchor in (None, "window"):
                self.popout_button.open_window()
            elif anchor == "tab":
                self.popout_button.open_tab()
            else:
                raise ValueError(
                    "Invalid anchor provided. Must be 'window' or 'tab'."
                )
        else:
            raise ValueError("Invalid loc provided. Must be 'inline', 'sidecar', or 'popout'.")

    def select(self, arg=None, column=None):
        """
        Select rows in MastTable based on Boolean mask, row indices, or column values.

        Parameters
        ----------
        arg : array of {bool, int}, int, list, slice, float, str
            If a Boolean mask, selects rows that are `True`.
            When ``column = None``, selects the rows by specified index value(s) or slices.
            When column is defined, selects the column values that match arg.

        column : str, optional (default None)
            The column corresponding to specific values you want to select on. Selects
            rows from MastTable whose values for ``column`` match the value(s) specified
            by ``arg``.

        """

        if arg is None:
            selection_mask = np.zeros(len(self.table), dtype=bool)

        elif isinstance(arg, np.ndarray):
            if arg.dtype.kind == 'b':
                selection_mask = arg
            elif arg.dtype.kind == 'i':
                selection_mask = np.zeros(len(self.table), dtype=bool)
                selection_mask[arg] = True
            else:
                raise TypeError(
                    "Invalid arg type provided. Array must be a Boolean mask or an "
                    "array of indices."
                )

        elif column is not None:
            if column not in self.table.colnames:
                raise ValueError(
                    f"Invalid column name provided. ``{column}`` does not exist in the table."
                )
            if not (np.isscalar(arg) or isinstance(arg, list)):
                raise TypeError(
                    "Invalid arg type provided. Must be int, list, float, or str."
                )

            values = np.atleast_1d(arg)
            selection_mask = np.isin(self.table[column], values)

        else:
            if isinstance(arg, int):
                idxs = [arg]
            elif isinstance(arg, list):
                idxs = arg
            elif isinstance(arg, slice):
                idxs = range(*arg.indices(len(self.table)))
            else:
                raise TypeError(
                    "Invalid arg type provided. Must be int, list, or slice."
                )

            selection_mask = np.zeros(len(self.table), dtype=bool)
            selection_mask[idxs] = True

        self.selected_mask = selection_mask


def get_current_table():
    """
    Return the last instantiated table widget, warns user
    if none exist.
    """
    if _table_widgets:
        return _table_widgets[-1]
    else:
        warnings.warn(
            "No `MastTable` exists.", UserWarning
        )


@preserve_docstring
@solara.component
def SelectableTable(
    table,
    base_mast_table,
    on_selected_indices: Optional[Callable[[List[int]], None]] = None,
    drawer_open: bool = False,
    set_drawer_open=None
):
    """An `ipyvuetify.DataTable` with checkbox selection.

    Displays a paginated table with selectable rows.  Reports the
    indices (into *table*) of the currently selected rows.

    Parameters
    ----------
    table : `~astropy.table.Table`
        A table to load.

    base_mast_table : `~mast_table.base.BaseMastTable`
        BaseMastTable widget to display.

    on_selected_indices : callable (optional, default is `None`)
        Callback on selected indices.

    drawer_open : bool (optional, default is `True`)
        If `True`, the CrossFilterMenu sidepanel is initialized
        open.

    set_drawer_open: callable (optional, default is `None`)
        Callback to open CrossFilterMenu sidepanel.

    """
    _, set_selected = solara.use_state(list(base_mast_table.selected_rows))

    # Build vuetify column headers from the table
    def handle_input(msg):
        new_entries = msg["new"]
        set_selected(new_entries)

        if on_selected_indices is not None:
            indices = [int(index) for index in new_entries]
            on_selected_indices(indices)

    def on_change(change):
        set_drawer_open(change["new"])

    base_mast_table.filter_tray_open = drawer_open

    def observe_widget(base_mast_table, on_change, handle_input):
        base_mast_table.observe(on_change, "filter_tray_open")
        base_mast_table.observe(handle_input, "selected_rows")

    solara.use_effect(
        lambda: observe_widget(
            base_mast_table,
            on_change,
            handle_input,
        ),
        [base_mast_table],
    )

    # update BaseMastTable items and selections on filter changes
    def update_items(table):
        base_mast_table.update_items(table)
        valid_idxs = set(
            np.asarray(table[col_unique_row_index]).astype(str)
        )
        base_mast_table.selected_rows = [
            key for key in base_mast_table.selected_rows
            if key in valid_idxs
        ]

    solara.use_effect(
        lambda: update_items(table),
        [table],
    )

    display(base_mast_table)


@preserve_docstring
@solara.component
def MastTableView(
    table,
    base_mast_table,
    on_popout_button=None,
    on_filter_mask=None,
):
    """
    Displays selectable table with cross-filtering.

    Parameters
    ----------
    table : `~astropy.table.Table`
        A table to load.

    base_mast_table : `~mast_table.base.BaseMastTable`
        BaseMastTable widget to display.

    on_popout_button: callable, optional
        Callback to expose popout button. Default is None.

    on_filter_mask : callable (optional, default is `None`)
        Callback to expose `filter_mask` from cross-filtering.
        Default is None

    """
    solara.provide_cross_filter()

    pending_reducer, set_pending_reducer = solara.use_state("AND")
    filter_masks, set_filter_masks = solara.use_state({})
    filters, set_filters = solara.use_state([])
    drawer_open, set_drawer_open = solara.use_state(False)

    # get defaults for "add filter", establish pending accordingly
    default_column = table.colnames[0]
    default_opt = slide_or_select(table, default_column)
    if default_opt == "slider":
        default_value, _ = table_range(table, default_column)
    else:
        unique_values, _ = build_select_items(table[default_column])
        default_value = unique_values[0] if unique_values else ""
    default_mode = ">=" if default_opt == "slider" else "=="

    pending, set_pending = solara.use_state({
        "column": default_column,
        "value": default_value,
        "mode": default_mode,
    })
    pending_column = pending["column"]
    pending_value = pending["value"]
    pending_mode = pending["mode"]

    def set_pending_value(value):
        set_pending({
            **pending,
            "value": value,
        })

    def set_pending_mode(mode):
        set_pending({
            **pending,
            "mode": mode,
        })

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

        set_pending({
            "column": default_column,
            "value": default_value,
            "mode": default_mode,
        })

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
        filter_mask = np.ones(
            len(table),
            dtype=bool
        )
    elif pending_reducer == "AND":
        filter_mask = functools.reduce(
            operator.and_,
            active_masks
        )
    else:
        filter_mask = functools.reduce(
            operator.or_,
            active_masks
        )

    solara.use_effect(
        lambda: (
            on_filter_mask(filter_mask)
            if on_filter_mask is not None
            else None
        ),
        [filter_mask, on_filter_mask],
    )

    solara.lab.theme.themes.light.primary = "#00617e"
    solara.lab.theme.themes.dark.primary = "#013b4d"
    expanded_ids, set_expanded_ids = solara.use_state(set())
    target_model_id = solara.use_reactive("")

    with solara.Column(
        classes=["mast-table-view"],
        style={
            "width": "100%",
            "min-width": "0",
        }
    ) as mast_table_view:
        solara.Style(
            """
            body.jupyter-widgets-popout-container .mast-table-view {
                width: 100vw;
                max-width: 100vw;
            }
            """
        )
        with solara.Row():
            # creating popout filters panel
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
                    solara.Markdown(
                        "##Active filters",
                        style={
                            "margin": "0",
                            "padding": "0",
                            "line-height": "1",
                        }
                    )
                    if len(filters) > 1:
                        solara.Style(
                            """
                            .custom-toggle {
                                display: flex;
                                flex: 0 0 auto;
                                width: 100px;
                                min-width: 100px;
                                overflow: hidden;
                                margin: 0;
                                padding: 0;
                            }
                            .custom-toggle .v-btn {
                                background-color: #F2F2F2;
                                color: light-dark(#00617e, #013b4d);
                                height: 40px;
                                width: 50px;
                                min-width: 50px;
                                max-width: 50px;
                                flex: 0 0 50px;
                                margin: 0;
                                padding: 0;
                            }
                            .custom-toggle .v-btn--active {
                                background-color: light-dark(#00617e, #013b4d);
                                color: white;
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
                                min-width: 0px;
                                width: 40px;
                                height: 40px;
                                padding: 0;
                                transform: translateY(-4px);
                            }
                            """
                        )

                        with solara.Tooltip("Remove all active filters"):
                            solara.Button(
                                icon_name="mdi-refresh",
                                on_click=lambda *args: (set_filters([]), set_filter_masks({})),
                                style={
                                    "background-color": "light-dark(#00617e, #013b4d)",
                                    "color": "white"
                                },
                                classes=["reset-button"]
                            )

                # creating slide/select for each active filter
                for f in filters:
                    with solara.Row(style={"width": "100%"}):
                        solara.Style(
                            """
                            .filter-card .v-card {
                                padding: 0;
                            }
                            .filter-card .v-card-text {
                                padding: 4px 8px;
                            }
                            .filter-card .v-card-actions {
                                padding: 0px 8px;
                            }
                            """
                        )

                        with solara.Card(
                            classes=["filter-card"],
                            style={
                                "border": "2px solid light-dark(#00617e, #013b4d)",
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
                                        text-transform: none;
                                        font-family: inherit;
                                        font-size: inherit;
                                        font-weight: bold;
                                        letter-spacing: normal;
                                    }
                                    .filter-column {
                                        min-width: 0;
                                        flex: 1 1 auto;
                                    }
                                    .filter-column .v-btn__content {
                                        min-width: 0;
                                        max-width: 100%;
                                        overflow-x: auto;
                                        overflow-y: hidden;
                                        white-space: nowrap;
                                        display: block;
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
                                    RemoveFilterButton(f["id"], remove_filter)

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
                                        filtered_table = table
                                    elif pending_reducer == "AND":
                                        filtered_table = table[
                                            functools.reduce(
                                                operator.and_,
                                                other_masks
                                            )
                                        ]
                                    else:
                                        filtered_table = table[
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
                                        filtered_table=filtered_table
                                    )

                if not len(filters):
                    solara.Markdown("No active filters")

                # creating add filter section
                solara.Markdown("##Add filter")

                column_names = table.colnames
                column_names.sort(key=str.casefold)
                if col_unique_row_index in column_names:
                    # never give the internal unique column as an option
                    column_names.remove(col_unique_row_index)

                # handling pending val initialization when col changes
                def on_pending_column_change(column):
                    opt = slide_or_select(table, column)

                    if opt == "slider":
                        value, _ = table_range(table, column)
                    else:
                        unique_values, _ = build_select_items(table[column])
                        value = unique_values[0] if unique_values else ""

                    set_pending({
                        "column": column,
                        "value": value,
                        "mode": ">=" if opt == "slider" else "==",
                    })

                v.Autocomplete(
                    label="Column",
                    items=column_names,
                    v_model=pending_column,
                    on_v_model=on_pending_column_change,
                    density="compact",
                    bg_color="transparent",
                    item_color="light-dark(#00617e, #013b4d)"
                )

                opt = slide_or_select(table, pending_column)
                fully_masked = False

                filtered_table = table[filter_mask]

                # creating slide/select based on column user selects
                if opt == "slider":
                    with solara.Row(
                        style={
                            "align-items": "center",
                            "gap": "8px",
                            "flex-wrap": "wrap",
                        }
                    ):
                        solara.Text(
                            "Operator",
                            style={"margin": "0", "padding": "0"}
                        )

                        FilterModeButtons(
                            mode=pending_mode,
                            set_mode=set_pending_mode,
                        )

                    vmin, vmax = table_range(table, pending_column)

                    solara.Text(
                        f"{pending_column} {pending_mode}",
                        style={"margin": "0", "padding": "0"},
                    )

                    comparison = operator_map[pending_mode]
                    slider_mask = comparison(filtered_table[pending_column], pending_value)

                    slider_args = {
                        "label": "",
                        "min": vmin,
                        "max": vmax,
                        "step": step_size(vmin, vmax),
                        "thumb_label": False,
                        "tick_labels": False,
                        "color": "primary",
                    }

                    with solara.Row(
                        style={
                            "display": "flex",
                            "alignItems": "bottom"
                        },
                        classes=["crossfilter-slider"]
                    ):
                        Input.element(
                            value=pending_value,
                            on_value=set_pending_value,
                            **slider_args,
                        )

                        v.Slider(
                            v_model=pending_value,
                            on_v_model=set_pending_value,
                            **slider_args
                        )

                    with solara.Row(
                        style={
                            "width": "100%",
                            "justify-content": "flex-end",
                            "margin": "0",
                            "padding": "0",
                        }
                    ):
                        solara.Text(
                            (
                                f"{len(filtered_table[slider_mask])} of "
                                f"{len(filtered_table)} after filtering"
                            ),
                            style={
                                "font-size": "12px",
                                "margin": "0",
                                "padding": "0",
                            },
                        )
                else:
                    unique_values, fully_masked = build_select_items(
                        table[pending_column]
                    )

                    max_unique = 100

                    items, value_counts = build_select_filter_preview(
                        table,
                        pending_column,
                        max_unique=max_unique,
                        filtered_table=filtered_table,
                    )

                    if pending_value not in ("", None):
                        value = next(
                            (item for item in items if item["value"] == pending_value),
                            None,
                        )
                    else:
                        value = None

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
                        count=len(filtered_table),
                    )

                    if len(value_counts) > max_unique:
                        Alert.element(
                            text=(
                                f"{pending_column} has {len(value_counts)} unique values. "
                                f"Showing the first {max_unique}."
                            ),
                            icon="mdi-information"
                        )

                with solara.Row(justify="end"):
                    solara.Button(
                        label="Apply filter",
                        icon_name="mdi-plus",
                        on_click=lambda *args: add_filter(opt),
                        disabled=fully_masked,
                        style={"background-color": "light-dark(#00617e, #013b4d)", "color": "white"}
                    )
                if fully_masked:
                    with solara.Row(justify="end"):
                        solara.Markdown("(Column fully masked)")

            with solara.Column(style="flex: 1; overflow: auto; min-height: 0"):
                filtered_table = table[filter_mask]
                SelectableTable(
                    filtered_table,
                    base_mast_table,
                    drawer_open=drawer_open,
                    set_drawer_open=set_drawer_open,
                )
                # Popout button overlays the BaseMastTable
                if target_model_id.value:
                    solara.Style(
                        """
                        .popout-button {
                            position: relative;
                        }

                        .popout-button::after {
                            content: "Display in a new window";
                            position: absolute;
                            top: calc(100% + 6px);
                            right: 0;
                            padding: 6px 10px;
                            background-color: #424242;
                            color: white;
                            border-radius: 4px;
                            font-size: 14px;
                            white-space: nowrap;
                            opacity: 0;
                            visibility: hidden;
                            pointer-events: none;
                            z-index: 10000;
                        }

                        .popout-button:hover::after {
                            opacity: 1;
                            visibility: visible;
                        }
                        """
                    )
                    with solara.Column(
                        classes=["popout-button"],
                        style={
                            "position": "absolute",
                            "top": "25px",
                            "right": "10px",
                            "z-index": "1000",
                        }
                    ):
                        PopoutButton.element(
                            target_model_id=target_model_id.value,
                            window_features="popup,width=1200,height=600",
                        )

    def set_popout():
        mast_table_widget = solara.get_widget(mast_table_view)
        target_model_id.set(mast_table_widget._model_id)

        popout_button = PopoutButton(
            target=mast_table_widget,
            window_features="popup,width=1200,height=600",
        )

        if on_popout_button is not None:
            on_popout_button(popout_button)

    solara.use_effect(set_popout)
