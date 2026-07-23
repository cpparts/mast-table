import numpy as np
from astropy.table import Table
import pytest
import random
import mast_table.cross_filter.utils as utils


@pytest.fixture
def test_table():
    return Table(
        {
            "test1": ["a", "a", "a", "b", "b", "c"],
            "test2": ["a", "a", "a", "b", "b", "c"],
            "test3": ["a", "a", "a", "b", "b", "c"],
        },
        masked=True
    )


def test_num_py_type():
    test_table = Table(
        {
            "test1": [2, 1, 3, 4, 6, 5],
            "test2": [0.01, 0.03, 11.3, 0.005, 10.001, 5.01],
            "test3": [True, True, False, True, False, False],
            "test4": ["a", "a", "a", "b", "b", "c"],
        }
    )
    py_type = utils.num_py_type(test_table, "test1")
    assert issubclass(py_type, (int, np.integer))

    py_type = utils.num_py_type(test_table, "test2")
    issubclass(py_type, (float, np.floating))

    with pytest.warns(UserWarning, match="not supported for Slider"):
        utils.num_py_type(test_table, "test3")

    with pytest.warns(UserWarning, match="not supported for Slider"):
        utils.num_py_type(test_table, "test4")


def test_table_value_count(test_table):
    partial_mask = np.array([False, False, True, False, True, True])
    full_mask = np.array([True for x in range(len(test_table))])
    test_table["test2"].mask = partial_mask
    test_table["test3"].mask = full_mask

    # getting counts for unmasked
    val_counts = utils.table_value_count(test_table, "test1", 3)
    assert isinstance(val_counts, Table)
    assert val_counts.colnames == ["value", "count"]
    assert (val_counts["value"] == ["a", "b", "c"]).all()
    assert (val_counts["count"] == [3, 2, 1]).all()

    # checking limit arg
    val_counts = utils.table_value_count(test_table, "test1", 1)
    assert isinstance(val_counts, Table)
    assert val_counts.colnames == ["value", "count"]
    assert val_counts["value"] == ["a"]
    assert val_counts["count"] == 3

    # getting counts for partially masked
    val_counts = utils.table_value_count(test_table, "test2", 3)
    assert isinstance(val_counts, Table)
    assert val_counts.colnames == ["value", "count"]
    assert (val_counts["value"] == ["--", "a", "b"]).all()
    assert (val_counts["count"] == [3, 2, 1]).all()

    # getting counts for fully masked
    val_counts = utils.table_value_count(test_table, "test3", 3)
    assert isinstance(val_counts, Table)
    assert val_counts.colnames == ["value", "count"]
    assert val_counts["value"] == "--"
    assert val_counts["count"] == 6


def test_table_filter_values(test_table):
    test_values = ["a", "b", "c"]

    # masking "test2" partially and "test3" fully
    partial_mask = np.array([False, False, True, False, True, True])
    full_mask = np.array([True for x in range(len(test_table))])
    test_table["test2"].mask = partial_mask
    test_table["test3"].mask = full_mask

    # checking unmasked (as-is and inverted)
    filter = utils.table_filter_values(
        test_table, "test1", test_values
    )
    assert isinstance(filter, np.ndarray)
    assert np.all(filter)

    filter_inverted = utils.table_filter_values(
        test_table, "test1", test_values, invert=True
    )
    assert isinstance(filter_inverted, np.ndarray)
    assert np.all(~filter_inverted)

    # checking partially masked (as-is and inverted)
    filter = utils.table_filter_values(
        test_table, "test2", test_values
    )
    assert isinstance(filter, np.ndarray)
    assert (filter == ~partial_mask).all()

    filter_inverted = utils.table_filter_values(
        test_table, "test2", test_values, invert=True
    )
    assert isinstance(filter_inverted, np.ndarray)
    assert (filter_inverted == partial_mask).all()

    # checking fully masked (as-is and inverted)
    filter = utils.table_filter_values(
        test_table, "test3", test_values
    )
    assert isinstance(filter, np.ndarray)
    assert (filter == ~full_mask).all()

    filter_inverted = utils.table_filter_values(
        test_table, "test3", test_values, invert=True
    )
    assert isinstance(filter_inverted, np.ndarray)
    assert (filter_inverted == full_mask).all()


def test_table_range():
    test_table = Table(
        {
            "test1": [2, 1, 3, 4, 6, 5],
            "test2": [0.01, 0.03, 11.3, 0.005, 10.001, 5.01],
        }
    )

    # integers
    test_range = utils.table_range(test_table, "test1")
    assert isinstance(test_range, tuple)
    assert test_range == (1, 6)

    # floats
    test_range = utils.table_range(test_table, "test2")
    assert isinstance(test_range, tuple)
    assert test_range == (0.005, 11.3)


def test_slide_or_select():
    test_table = Table(
        {
            "test1": ["test" for x in range(12)],
            "test2": [random.randint(1, 8) for x in range(12)],
            "test3": random.sample(range(1, 101), 12),
        }
    )

    # non-number select
    non_num = utils.slide_or_select(test_table, "test1")
    assert isinstance(non_num, str)
    assert non_num == "select"

    # number select
    num_sel = utils.slide_or_select(test_table, "test2")
    assert isinstance(num_sel, str)
    assert num_sel == "select"

    # number slider
    num_slide = utils.slide_or_select(test_table, "test3")
    assert isinstance(num_slide, str)
    assert num_slide == "slider"


@pytest.mark.parametrize(
    "vmin, vmax, expected",
    [
        (0, 10, 0.1),
        (0, 1, 0.2),
        (0, 0.4, 0.08),
        (0, 0.1, 0.02),
        (0, 0.01, 0.002),
        (-0.5, 0.5, 0.2),
    ],
)
def test_step_size(vmin, vmax, expected):
    step_size = utils.step_size(vmin, vmax)
    assert step_size == expected


def test_build_select_items(test_table):
    # masking "test2" partially and "test3" fully
    partial_mask = np.array([False, False, True, False, True, True])
    full_mask = np.array([True for x in range(len(test_table))])
    test_table["test2"].mask = partial_mask
    test_table["test3"].mask = full_mask

    # checking unmasked
    unique_values, fully_masked = utils.build_select_items(test_table["test1"])
    assert unique_values == ["a", "b", "c"]
    assert not fully_masked

    # checking partially masked
    unique_values, fully_masked = utils.build_select_items(test_table["test2"])
    assert unique_values == ["a", "b"]
    assert not fully_masked

    # checking fully masked
    unique_values, fully_masked = utils.build_select_items(test_table["test3"])
    assert unique_values == []
    assert fully_masked


def test_build_select_filter_preview():
    test_table = Table(
        {
            "test1": ["a", "a", "a", "b", "b", "c"],
        },
    )

    # unfiltered table
    items, value_counts = utils.build_select_filter_preview(test_table, "test1")
    assert isinstance(items, list)
    assert isinstance(value_counts, Table)

    assert items == [
        {'value': 'a', 'text': 'a', 'count': 3, 'count_max': 3},
        {'value': 'b', 'text': 'b', 'count': 2, 'count_max': 2},
        {'value': 'c', 'text': 'c', 'count': 1, 'count_max': 1}
    ]
    assert np.all(
        value_counts == Table(
            {
                "value": ["a", "b", "c"],
                "count_max": [3, 2, 1],
                "count": [3, 2, 1],
                "exists": [True, True, True]
            }
        )
    )

    # checking max_unique
    items, value_counts = utils.build_select_filter_preview(test_table, "test1", max_unique=0)
    assert isinstance(items, list)
    assert isinstance(value_counts, Table)

    assert items == [
        {'value': 'a', 'text': 'a', 'count': 3, 'count_max': 3},
    ]
    assert np.all(
        value_counts == Table(
            {
                "value": ["a"],
                "count_max": [3],
                "count": [3],
                "exists": [True]
            }
        )
    )

    # filtered table
    mask = test_table["test1"] == "a"
    test_table_filtered = test_table[mask]
    items, value_counts = utils.build_select_filter_preview(
        test_table, "test1", table_filtered=test_table_filtered
    )
    assert isinstance(items, list)
    assert isinstance(value_counts, Table)

    assert items == [
        {'value': 'a', 'text': 'a', 'count': 3, 'count_max': 3},
        {'value': 'b', 'text': 'b', 'count': 0, 'count_max': 2},
        {'value': 'c', 'text': 'c', 'count': 0, 'count_max': 1}
    ]
    assert np.all(
        value_counts == Table(
            {
                "value": ["a", "b", "c"],
                "count_max": [3, 2, 1],
                "count": [3, 0, 0],
                "exists": [True, False, False]
            }
        )
    )
