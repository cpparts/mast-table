from mast_table import MastTable
from mast_table.mast_table import serialize
import numpy as np
import astropy.units as u
from astropy.table import Table


def test_mast_table_init(mast_observation_table):
    mast_table = MastTable(mast_observation_table)

    # check that astropy table is stored on the widget
    assert 's_region' in mast_table.table.colnames

    # check that s_region col exists in available widget columns
    assert 's_region' in mast_table.headers_avail

    # check that s_region col isn't visible by default
    assert 's_region' not in mast_table.headers_visible

    # the MAST observation query has a ArchiveFileID column,
    # which should be chosen as the default item_key:
    assert mast_table.item_key == 'ArchiveFileID'


def test_server_side_pagination(mast_observation_table):
    # fixture has 5 rows
    n_rows = len(mast_observation_table)
    mast_table = MastTable(mast_observation_table, items_per_page=2)

    # full row cache holds every row, while only the first page is pushed to the UI
    assert mast_table.server_pagination is True
    assert mast_table.server_items_length == n_rows
    assert len(mast_table._all_items) == n_rows
    assert len(mast_table.items) == min(2, n_rows)

    # simulate the frontend updating page/itemsPerPage and verify the slice updates
    mast_table.table_options = {'page': 2, 'itemsPerPage': 2}
    assert len(mast_table.items) == min(2, max(0, n_rows - 2))
    assert mast_table.items[0] == mast_table._all_items[2]

    # itemsPerPage = -1 means "show all"
    mast_table.table_options = {'page': 1, 'itemsPerPage': -1}
    assert len(mast_table.items) == n_rows


def test_server_side_pagination_disabled(mast_observation_table):
    mast_table = MastTable(mast_observation_table, server_pagination=False)
    # verify the kwarg actually reached the traitlet
    assert mast_table.server_pagination is False
    # with server pagination disabled, the full table is pushed to the UI
    assert len(mast_table.items) == len(mast_observation_table)


def test_serialize_respects_column_format():
    """``Column.info.format`` should drive per-column print precision."""
    t = Table({
        'ra': [12.345678, 98.765432],
        'flux': [1.2345e-15, 6.7890e-14],
        'name': ['a', 'b'],
    })
    t['ra'].info.format = '.3f'
    t['flux'].info.format = '%.2e'

    rows = serialize(t)
    assert rows[0]['ra'] == '12.346'
    assert rows[1]['ra'] == '98.765'
    assert rows[0]['flux'] == '1.23e-15'
    # columns without a format should pass through to plain Python types
    assert rows[0]['name'] == 'a'

    # NaN cells should render as an empty string when a format is set
    t2 = Table({'x': [np.nan, 1.0]})
    t2['x'].info.format = '.3f'
    rows2 = serialize(t2)
    assert rows2[0]['x'] == ''
    assert rows2[1]['x'] == '1.000'

    # Quantity columns respect the format spec (units appear in header, not cells)
    t3 = Table({'wave': [500.123, 600.456] * u.nm})
    t3['wave'].info.format = '.1f'
    rows3 = serialize(t3)
    assert rows3[0]['wave'] == '500.1'
    assert rows3[1]['wave'] == '600.5'


def test_column_format_propagates_to_widget(mast_observation_table):
    """Setting ``Column.info.format`` before constructing the widget should
    show up in the cached items pushed to the UI."""
    col = next(
        (c for c in mast_observation_table.colnames
         if mast_observation_table[c].dtype.kind == 'f'),
        None,
    )
    assert col is not None
    mast_observation_table[col].info.format = '.2f'
    mast_table = MastTable(mast_observation_table)
    # every cached row should have the formatted (string) value for this column
    for row in mast_table._all_items:
        value = row[col]
        # NaNs become '' otherwise we expect a formatted string with 2 decimals
        assert value == '' or (isinstance(value, str) and value.count('.') == 1
                               and len(value.split('.')[1]) == 2)
