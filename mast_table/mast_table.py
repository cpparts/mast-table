
import os
import warnings

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

from mast_table import validate
from astroquery.mast import MastMissions
from mast_table.app import SelectableDataFrame, CrossFilterInspector
import solara
import pandas as pd

__all__ = [
    'MastTable',
    'get_current_table',
]

# register loaded table widgets as they're initialized
_table_widgets = dict()


def serialize(table):
    """
    Convert an astropy table to a list of dictionaries
    containing each column as a list of pure Python objects.
    """
    return [
        {
            k: v.tolist()
            for k, v in dict(row).items()
        } for row in table
    ]


known_unique_mast_table_cols = [
    'fileSetName',          # data products from astroquery.mast.MastMissions
    'source_id',            # Gaia
    'MatchID',              # Hubble Source Catalog
    'objID',                # PanSTARRS,
    'product_key',          # list_products queries
    'obs_id',               # astroquery.mast.Observations,
    'sci_data_set_name',    # HST
]


class MastTable():
    """
    Table widget for observation queries from Mission MAST.
    """

    def __init__(self, table, app=None, update_viewport=True, unique_column=None, **kwargs):
        """
        Parameters
        ----------
        table : `~astropy.table.Table`
            A table to load.

        app : `~mast_aladin.app.MastAladin`
            An instance of the ``MastAladin`` app.

        update_viewport : bool (optional, default is `True`)
            If `True`, set the `~mast_aladin.app.MastAladin`
            viewport center to the position of the item in the
            first row of the table on load.

        unique_column : str (optional, default is `None`)
            A column which contains unique values in each row.

            If no `unique_column` is given, ``MastTable`` will look for a
            column known to have unique values for each row in common MAST
            observation queries. If no known `unique_column` is found,
            search through the table to find a column with unique rows.

            For tables with many rows, unique column searches are inefficient
            and a warning will be raised..
        """

        super().__init__(**kwargs)

        self.table = table
        self.app = app

        self.items = serialize(table)
        self.mission = validate.detect_mission_or_products(table)
        columns = table.colnames
        self.column_descriptions = validate.get_column_descriptions(self.mission)

        self._set_item_key(columns, unique_column)

        self.headers_avail = list(columns)

        # by default, remove the `s_region`` column
        # from the visible columns in the widget:
        if 's_region' in columns:
            columns.remove('s_region')

        self.headers_visible = columns
        self.selected_indices = solara.reactive([])

        _table_widgets[len(_table_widgets)] = self

        if update_viewport and self.app is not None:
            ra_dec_colnames = dict(
                hst=['sci_ra', 'sci_dec'],
                roman=['ra_ref', 'dec_ref'],
                jwst=['targ_ra', 'targ_def'],
            )
            ra_column, dec_column = ra_dec_colnames[self.mission]

            center_coord = SkyCoord(
                ra=table[ra_column][0] * u.deg,
                dec=table[dec_column][0] * u.deg,
                unit=u.deg
            )

            # change the coordinate frame to match the coordinates in the MAST table:
            self.app.target = f"{center_coord.ra.degree} {center_coord.dec.degree}"

    def _set_item_key(self, table_columns, item_key, n_rows_slow=10e6):
        """
        `item_key` should be set to the name of a table column that contains
        unique values in each row, which can be used for selection.

        If no `unique_column` is given at construction, look for a column known to have
        unique values for each row in MAST catalog and observation queries. If no known
        columns are found, search through the table to find a column with unique rows.

        Unique row searches are inefficient for tables with more than `n_rows_slow` rows,
        and a warning will be raised. The default `n_rows_slow = 10e6` takes about 100
        milliseconds per table column.
        """
        if item_key is None:
            # check for known unique columns:
            for column in known_unique_mast_table_cols:
                if column in table_columns:
                    self.item_key = column
                    break

            # warn the user if unique row search will be inefficient:
            if len(self.table) > n_rows_slow:
                warnings.warn(
                    "No `unique_column` was given, so all columns will be checked "
                    f"for unique entries. This table has {len(self.table)} rows, so "
                    "the search for unique rows may be slow. To avoid this in the future,"
                    "use the `unique_column` keyword argument when calling "
                    "`MastAladin.load_table`.", UserWarning
                )

            # search for columns with unique rows:
            for column in table_columns:
                n_unique_values = np.unique(self.table[column]).size
                if n_unique_values == len(self.table):
                    self.item_key = column
                    break
            else:
                raise ValueError(
                    "No `unique_column` specified, and no unique columns were found."
                )

        elif item_key in table_columns:
            self.item_key = item_key

        else:
            raise ValueError(
                f"item_key '{item_key}' not found in table columns: {table_columns}"
            )

    @property
    def selected_rows_table(self):
        """
        `~astropy.table.Table` of only the selected rows.
        """
        df = self.table.to_pandas()
        return Table.from_pandas(df.loc[self.selected_indices.value])

    def open_selected_rows_in_jdaviz(self):
        from jdaviz import Imviz
        from jdaviz.configs.imviz.helper import _current_app as viz

        selected_df = self.selected_rows_table

        if viz is None:
            viz = Imviz()

        with viz.batch_load():
            for filename in selected_df['filename']:
                _download_from_mast(filename)
                viz.load(filename)

        orientation = viz.plugins['Orientation']
        orientation.align_by = 'WCS'
        orientation.set_north_up_east_left()

        plot_options = viz.plugins['Plot Options']
        if len(plot_options.layer.choices) > 1:
            for layer in plot_options.layer.choices:
                plot_options.layer = layer
                plot_options.image_color_mode = 'Color'

            plot_options.apply_RGB_presets()

        return viz

    def open_selected_rows_in_aladin(self):
        from mast_aladin.app import gca

        selected_df = self.selected_rows_table

        mal = gca()

        for filename in selected_df['filename']:
            _download_from_mast(filename)
            mal.delayed_add_fits(filename)

        return mal


@solara.component
def MastTableCrossFiltered(
    mast_table: MastTable,
    df: pd.DataFrame,
    items_per_page: int = 10,
):
    filter, set_filter = solara.use_cross_filter(id(df), "mast-table")

    selected_indices = mast_table.selected_indices.value

    solara.use_effect(
        lambda: (
            set_filter(None)
            if not selected_indices
            else set_filter(df.index.isin(selected_indices))
        ),
        [tuple(selected_indices)],
    )

    # Apply incoming filters
    if filter is not None:
        dff = df[filter]
    else:
        dff = df

    solara.Info(f"Showing {len(dff)} of {len(df)} rows")

    SelectableDataFrame(
        dff,
        items_per_page=items_per_page,
        selected_indices=selected_indices,
        set_selected_indices=lambda indices: mast_table.selected_indices.set(indices),
    )

    if mast_table.mission == "list_products":
        with solara.Row():
            solara.Button(
                "Open in Aladin",
                disabled=(len(selected_indices) == 0),
                on_click=lambda: mast_table.open_selected_rows_in_aladin(),
            )
            solara.Button(
                "Open in Jdaviz",
                disabled=(len(selected_indices) == 0),
                on_click=lambda: mast_table.open_selected_rows_in_jdaviz(),
            )


@solara.component
def MastTablePage(mast_table: MastTable):
    solara.provide_cross_filter()

    # by default, remove the `s_region`` column
    # from the visible columns in the widget:
    df = mast_table.table.to_pandas()[mast_table.headers_visible]

    with solara.Column():
        solara.Markdown("## MAST Table Viewer (Cross-Filtered)")
        solara.Markdown(
            "Select rows with the checkboxes to filter the other widgets. "
            "Use the dropdown to filter what shows up in the table."
        )

        if mast_table.mission == "list_products":
            with solara.Row():
                solara.CrossFilterSelect(df, "file_suffix")
        else:
            with solara.Row():
                solara.CrossFilterSelect(df, "productLevel")

        with solara.Row(style={"width": "100%"}):
            with solara.Column(
                style={
                    "max-width": "50px",
                }
            ):
                solara.CrossFilterReport(df)
            with solara.Column(
                style={
                    "min-width": "0",
                }
            ):
                with solara.Div(style={
                    "overflow-x": "auto",
                    "width": "100%",
                }):
                    with solara.Div(style={
                        "min-width": "max-content",
                    }):
                        MastTableCrossFiltered(mast_table, df)

        with solara.Row():
            CrossFilterInspector(df)


def _download_from_mast(product_file_name):
    if os.path.exists(product_file_name):
        # support load from cache without query to MM
        return

    # temporarily support JWST and HST until Roman is also available:
    if product_file_name.startswith('jw'):
        mission = 'jwst'
    elif product_file_name.startswith('r'):
        mission = 'roman'
    else:
        mission = 'hst'

    MastMissions(mission=mission).download_file(product_file_name)


def get_current_table():
    """
    Return the last instantiated table widget, create a new
    one if none exist.
    """
    if len(_table_widgets):
        latest_table_index = list(_table_widgets.keys())[-1]
        return _table_widgets[latest_table_index]

    return MastTable()
