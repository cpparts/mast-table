"""Demo: Cross-filtered table with selectable rows.

Shows how to use solara's cross-filter system with a custom ipyvuetify
DataTable that has checkbox row selection. The selected rows feed back
into the cross-filter so other widgets react to the selection.

Run with:
    solara run nogit/cross_filter_demo.py
"""

from typing import Callable, Dict, List

import numpy as np
import pandas as pd
import plotly
import reacton.ipyvuetify as v

import solara

df = plotly.data.gapminder()


@solara.component
def SelectableDataFrame(
    df: pd.DataFrame,
    items_per_page: int = 10,
    selected_indices: List[int] = None,
    set_selected_indices: Callable[[List[int]], None] = None,
):
    """An ipyvuetify DataTable with checkbox selection.

    Displays a paginated table with selectable rows.  Reports the
    indices (into *df*) of the currently selected rows.
    """
    page, set_page = solara.use_state(1)

    # Build vuetify column headers from the dataframe
    headers = [{"text": col, "value": col} for col in df.columns]

    # Build the items list – include the original df index so we can
    # map selections back even after cross-filtering has sliced the df.
    items = []
    index_to_item = {}

    for idx, row in df.iterrows():
        item: Dict = {"_solara_index": idx}
        for col in df.columns:
            val = row[col]
            # vuetify needs JSON-friendly values
            if isinstance(val, (int, float, str, bool)):
                item[col] = val
            else:
                item[col] = str(val)
        items.append(item)
        index_to_item[idx] = item

    selected_items = []
    if selected_indices is not None:
        for idx in selected_indices:
            if idx in index_to_item:
                selected_items.append(index_to_item[idx])

    def handle_input(new_selected):
        if set_selected_indices:
            indices = [item["_solara_index"] for item in new_selected]
            set_selected_indices(indices)

    v.DataTable(
        headers=headers,
        items=items,
        item_key="_solara_index",
        v_model=selected_items,
        on_v_model=handle_input,
        show_select=True,
        items_per_page=items_per_page,
        page=page,
        on_page=set_page,
        dense=True,
        style_="min-width: 1200px;",
    )


@solara.component
def CrossFilterSelectableDataFrame(
    df: pd.DataFrame,
    items_per_page: int = 10,
):
    """A selectable table that participates in cross-filtering.

    * Incoming cross-filters from other components narrow which rows
      are shown.
    * When the user checks rows, a filter is set so that *other*
      cross-filter consumers only see the selected rows.
    """
    filter, set_filter = solara.use_cross_filter(id(df), "selectable-table")

    # Apply incoming cross-filter
    if filter is not None:
        dff = df[filter]
    else:
        dff = df

    def on_selected_indices(indices: List[int]):
        if not indices:
            # Nothing selected → don't restrict other components
            set_filter(None)
        else:
            # Build a boolean mask over the *original* df
            mask = df.index.isin(indices)
            set_filter(mask)

    solara.Info(f"Showing {len(dff)} of {len(df)} rows")
    SelectableDataFrame(dff, items_per_page=items_per_page, on_selected_indices=on_selected_indices)


@solara.component
def CrossFilterInspector(df: pd.DataFrame, name: str = "inspector"):
    """Shows the raw cross-filter state for debugging / education.

    Displays the combined boolean mask that this component receives
    from all other cross-filter participants.
    """
    filter, _set_filter = solara.use_cross_filter(id(df), name)

    if filter is None:
        summary = "No active filters — all rows pass."
        mask_repr = "None"
        details = ""
    else:
        mask = np.asarray(filter)
        n_true = int(mask.sum())
        n_total = len(mask)
        n_false = n_total - n_true

        summary = f"**{n_true}** of **{n_total}** rows pass ({n_false} filtered out)"

        # Show a compact visual: ✓ / ✗ for the first/last few values
        max_show = 400
        if n_total <= max_show:
            bits = "".join("✓" if b else "·" for b in mask)
        else:
            head = "".join("✓" if b else "·" for b in mask[:200])
            tail = "".join("✓" if b else "·" for b in mask[-200:])
            bits = f"{head} … {tail}"

        mask_repr = f"`{bits}`"

        # Which indices pass?
        passing = np.where(mask)[0]
        if len(passing) <= 20:
            idx_str = ", ".join(str(i) for i in passing)
        else:
            idx_str = ", ".join(str(i) for i in passing[:10]) + f" … ({len(passing)} total)"
        details = f"Passing indices: `[{idx_str}]`"

    with v.ExpansionPanels(v_model=0, flat=True):
        with v.ExpansionPanel():
            with v.ExpansionPanelHeader():
                solara.Text("🔍 Cross-Filter Inspector")
            with v.ExpansionPanelContent():
                solara.Markdown(summary)
                solara.Markdown(f"**Mask:** {mask_repr}")
                if details:
                    solara.Markdown(details)
                solara.Markdown(
                    f"**Type:** `{type(filter).__module__}.{type(filter).__name__}`"
                    if filter is not None
                    else "**Type:** `NoneType`"
                )


@solara.component
def Page():
    solara.provide_cross_filter()

    with solara.Column():
        solara.Markdown("## Cross-Filter Demo with Selectable Rows")
        solara.Markdown(
            "Select rows with the checkboxes to filter the other widgets. "
            "Use the dropdown to filter what shows up in the table."
        )

        with solara.Row():
            solara.CrossFilterSelect(df, "continent")
            solara.CrossFilterSlider(df, "year")
        with solara.Row():
            with solara.Column():
                solara.CrossFilterReport(df, classes=["py-2"])
            with solara.Column():
                CrossFilterSelectableDataFrame(df)
        with solara.Row():
            CrossFilterInspector(df)
