
import os
import re
import warnings

from traitlets import List, Unicode, Bool, Int, Dict, Any, observe
from ipypopout import PopoutButton
from ipyvuetify import VuetifyTemplate
from ipywidgets.widgets import widget_serialization

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

from mast_table import validate
from astroquery.mast import MastMissions

__all__ = [
    'MastTable',
    'get_current_table',
]

col_unique_row_index = '_unique_row_index'

# register loaded table widgets as they're initialized
_table_widgets = dict()


mission_mast_ra_dec_colnames = dict(
    hst=['sci_ra', 'sci_dec'],
    roman=['ra', 'dec'],
    jwst=['targ_ra', 'targ_def'],
)


def _format_value(value, fmt):
    """
    Apply an astropy ``Column.info.format`` spec to a single ``value``.

    Supports new-style format specs (e.g. ``'.3f'``), printf-style specs
    (e.g. ``'%.3f'``), and callables. Falls back to the raw value if the
    format cannot be applied. NaN values are returned as an empty string
    so the UI shows a blank cell instead of ``'nan'``.
    """
    if fmt is None:
        return value
    try:
        if isinstance(value, float) and np.isnan(value):
            return ''
    except (TypeError, ValueError):
        pass
    if callable(fmt):
        try:
            return fmt(value)
        except Exception:
            return value
    try:
        return format(value, fmt)
    except (ValueError, TypeError):
        pass
    try:
        return fmt % value
    except (ValueError, TypeError):
        return value


def _json_safe(value):
    """
    Convert a single cell ``value`` to a JSON-safe representation for the
    frontend, with no per-column precision logic. Per-column precision is
    handled separately via :func:`_format_value` driven by
    ``Column.info.format``.
    """
    if isinstance(value, SkyCoord):
        return value.to_string('hmsdms', precision=4)
    if isinstance(value, u.Quantity):
        if value.isscalar and np.isnan(value.value):
            return ''
        if value.isscalar:
            return f"{value.value} {value.unit.to_string()}"
        return {"value": value.value.tolist(), "unit": str(value.unit)}
    if isinstance(value, np.floating):
        v = float(value)
        return '' if np.isnan(v) else v
    if isinstance(value, float) and np.isnan(value):
        return ''
    if isinstance(value, np.bool_):
        return bool(value)
    if hasattr(value, 'tolist'):
        try:
            return value.tolist()
        except (TypeError, ValueError):
            pass
    return value


def serialize(table):
    """
    Convert an astropy table to a list of dictionaries of JSON-safe values.

    Per-column print precision is taken from each column's
    ``Column.info.format`` attribute, so users can configure precision via
    standard astropy machinery, e.g.::

        table['flux'].info.format = '.3e'
        table['ra'].info.format = '%.5f'
    """
    column_names = table.colnames

    def _replace_nan(value):
        if 'nan' in value:
            value = ''
        return value

    def nan_to_empty_str(column):
        nans_found = re.findall('nan', ''.join(column))
        if len(nans_found):
            column = [_replace_nan(row) for row in column]
        return column

    formatted_rows = list(zip(*[
        nan_to_empty_str(
            table[col].pformat(show_name=False, show_unit=False)
        )
        for col in column_names
    ]))

    serialized = [
        {name: f"{val}" for name, val in zip(column_names, row)}
        for row in formatted_rows
    ]
    return serialized


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
    mission = Unicode(allow_none=True).tag(sync=True)
    filter_tray_open = Bool(True).tag(sync=True)

    # Server-side pagination traitlets
    server_pagination = Bool(True).tag(sync=True)
    server_items_length = Int(0).tag(sync=True)
    table_options = Dict({}).tag(sync=True)

    # item_key is a column of the table with unique values
    # for each row, enabling selection of the row by lookup
    item_key = Unicode().tag(sync=True)

    table = None
    row_select_callbacks = []

    def __init__(
            self,
            table,
            app=None,
            update_viewport=True,
            unique_column=None,
            ra_column=None,
            dec_column=None,
            **kwargs):
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
            and a warning will be raised.

        ra_column : str (optional, default is `None`)
            Column name for the right ascension in degrees.

        dec_column : str (optional, default is `None`)
            Column name for the declination in degrees.

        **kwargs
            Remaining keyword arguments are passed to
            ``ipyvuetify.VuetifyTemplate``.
        """

        # initialize the row cache, so the ``table_options`` observer is safe to fire
        # if that traitlet is passed in via ``kwargs``.
        self._all_items = []

        super().__init__(**kwargs)

        self.popout_button = PopoutButton(self)
        self.table = table
        self.table[col_unique_row_index] = np.arange(len(table))
        self.app = app

        if not self.table_options:
            self.table_options = {'page': 1, 'itemsPerPage': self.items_per_page}

        self._all_items = serialize(self.table)
        self.server_items_length = len(self._all_items)
        self._push_current_page()
        columns = self.table.colnames

        self._set_item_key(columns, unique_column)

        self.headers_avail = [
            column for column in columns if column != col_unique_row_index
        ]

        # by default, remove the `s_region`` column
        # from the visible columns in the widget:
        self.headers_visible = [
            column for column in self.headers_avail
            if column != 's_region'
        ]

        _table_widgets[len(_table_widgets)] = self

        if mission := validate.detect_mission_or_products(table):
            self.column_descriptions = validate.get_column_descriptions(mission)

            # if the user hasn't defined the ra/dec columns, use
            # the expectated Mission Mast names for this mission:
            if ra_column is None and dec_column is None:
                ra_column, dec_column = mission_mast_ra_dec_colnames[mission]

        # if the ra/dec columns are available in the table:
        if (
                ra_column in columns and
                dec_column in columns and
                update_viewport and
                self.app is not None):

            # use the first sky coordinate as a reference for centering the viewer.
            # an alternative would be to use e.g. mean(RA), though means would return an
            # unhelpful coordinate in the case where observations span the meridian or poles.
            reference_coord = SkyCoord(
                ra=self.table[ra_column][0] * u.deg,
                dec=self.table[dec_column][0] * u.deg,
                unit=u.deg
            )

            # set the center of the viewer on the reference coord:
            self.app.target = f"{reference_coord.ra.degree} {reference_coord.dec.degree}"

    @observe('table_options')
    def _table_options_changed(self, msg):
        if not self.server_pagination or not self._all_items:
            return
        self._push_current_page()

    def _push_current_page(self):
        """Push only the current page slice of ``_all_items`` to ``items``."""
        if not self.server_pagination:
            self.items = list(self._all_items)
            return
        opts = self.table_options or {}
        page = opts.get('page', 1)
        per_page = opts.get('itemsPerPage', self.items_per_page)
        if per_page == -1:
            self.items = list(self._all_items)
            return
        start = (page - 1) * per_page
        end = start + per_page
        self.items = self._all_items[start:end]

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
        import jdaviz as jd

        viz = jd.gca()

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
