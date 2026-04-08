
import os
import warnings

from traitlets import List, Unicode, Bool, Int, Any, observe
from ipypopout import PopoutButton
from ipyvuetify import VuetifyTemplate
from ipywidgets.widgets import widget_serialization

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

from mast_table import validate
from astroquery.mast import MastMissions
import jdaviz

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
    'fileSetName',  # data products from astroquery.mast.MastMissions
    'source_id',    # Gaia
    'MatchID',      # Hubble Source Catalog
    'objID',        # PanSTARRS,
    'product_key',  # list_products queries
    'obs_id',       # astroquery.mast.Observations,
    'sci_data_set_name',  # HST
]


class MastTable(VuetifyTemplate):
    """
    Table widget for observation queries from Mission MAST.
    """
    template_file = __file__, "mast_table.vue"

    items = List().tag(sync=True)
    headers_visible = List().tag(sync=True)
    headers_avail = List().tag(sync=True)
    show_if_empty = Bool(True).tag(sync=True)
    show_rowselect = Bool(True).tag(sync=True)
    selected_rows = List().tag(sync=True)
    column_descriptions = List().tag(sync=True)
    multiselect = Bool(True).tag(sync=True)
    items_per_page = Int(5).tag(sync=True)
    show_tooltips = Bool(False).tag(sync=True)
    menu_open = Bool(False).tag(sync=True)
    clear_btn_lbl = Unicode('Clear Table').tag(sync=True)
    popout_button = Any().tag(sync=True, **widget_serialization)
    enable_load_in_app = Bool(False).tag(sync=True)
    mission = Unicode().tag(sync=True)

    # item_key is a column of the table with unique values
    # for each row, enabling selection of the row by lookup
    item_key = Unicode().tag(sync=True)

    table = None
    row_select_callbacks = []

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
        self.popout_button = PopoutButton(self)

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

    @observe('selected_rows')
    def _on_row_selection(self, msg={}):
        for func in self.row_select_callbacks:
            func(msg)

    @property
    def selected_rows_table(self):
        """
        `~astropy.table.Table` of only the selected rows.
        """
        return Table(self.selected_rows)

    def vue_open_selected_rows_in_jdaviz(self, *args):

        if jdaviz.gca() is None:
            jdaviz.new_app()

        viz = jdaviz.gca()

        #image_viewers = viz.app.get_viewers_of_cls('ImvizImageView')
        #if image_viewers:
        #    glue_viewer = image_viewers[0]
        #else:
        #    raise ValueError(
        #        "No compatible viewers available in jdaviz. You must "
        #        "load data with a world coordinate system before using ImvizSyncAdapter."
        #    )

        with viz.batch_load():
            for filename in self.selected_rows_table['filename']:
                _download_from_mast(filename)
                viz.load(filename, format="Image")

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

    def vue_open_selected_rows_in_aladin(self, *args):
        from mast_aladin.app import gca

        mal = gca()

        for filename in self.selected_rows_table['filename']:
            _download_from_mast(filename)
            mal.delayed_add_fits(filename)

        return mal

    @observe('mission')
    def _on_mission_update(self, msg={}):
        self.enable_load_in_app = msg['new'] == 'list_products'


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
