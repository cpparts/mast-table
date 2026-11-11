import pytest
import numpy as np

from mast_table import MastTable


def test_set_selected_rows(mast_observation_table):
    mast_table = MastTable(mast_observation_table)

    # confirm intial masks filter out nothing and
    # select nothing
    assert all(mast_table.filter_mask)
    assert not all(mast_table._selected_mask)

    # confirm selecting with .selected_rows
    mast_table.selected_rows = mast_table.items[0:1]
    assert mast_table.selected_rows_table == mast_observation_table[0]
    assert mast_table.selected_rows == ["0"]
    assert mast_table._selected_mask[0]

    # confirm resetting with .selected_rows
    mast_table.selected_rows = []
    assert mast_table.selected_rows == []
    assert not all(mast_table._selected_mask)
    assert len(mast_table.selected_rows_table) == 0


def test_set_selected_mask(mast_observation_table):
    mast_table = MastTable(mast_observation_table)

    # confirm intial masks filter out nothing and
    # select nothing
    assert all(mast_table.filter_mask)
    assert not all(mast_table._selected_mask)

    # confirm selecting with .selected_mask
    mast_table.selected_mask = mast_observation_table["observtn"] == 2
    assert all(mast_table.selected_rows_table == mast_observation_table[0:3])
    assert mast_table.selected_rows == ["0", "1", "2"]
    assert all(mast_table._selected_mask[0:3])


def test_select(mast_observation_table):
    mast_table = MastTable(mast_observation_table)

    # confirm intial masks filter out nothing and
    # select nothing
    assert all(mast_table.filter_mask)
    assert not all(mast_table._selected_mask)

    # by ind
    mast_table.select(2)
    assert mast_table.selected_rows_table == mast_observation_table[2]
    assert mast_table.selected_rows == ["2"]
    assert mast_table._selected_mask[2]

    mast_table.select([1, 3])
    assert all(mast_table.selected_rows_table == mast_observation_table[[1, 3]])
    assert mast_table.selected_rows == ["1", "3"]
    assert all(mast_table._selected_mask[[1, 3]])

    # by slice
    mast_table.select(slice(0, 2))
    assert all(mast_table.selected_rows_table == mast_observation_table[0:2])
    assert mast_table.selected_rows == ["0", "1"]
    assert all(mast_table._selected_mask[0:2])

    # by column and condition
    mast_table.select(3, column="observtn")
    assert all(mast_table.selected_rows_table == mast_observation_table[3:])
    assert mast_table.selected_rows == ["3", "4"]
    assert all(mast_table._selected_mask[3:])

    # by mask
    test_mask = mast_observation_table["targ_ra"] > 245.93
    mast_table.select(test_mask)
    assert all(mast_table.selected_rows_table == mast_observation_table[0:3])
    assert mast_table.selected_rows == ["0", "1", "2"]
    assert all(mast_table._selected_mask[0:3])

    # by array of inds
    test_array = np.array([1, 2, 3])
    mast_table.select(test_array)
    assert all(mast_table.selected_rows_table == mast_observation_table[1:4])
    assert mast_table.selected_rows == ["1", "2", "3"]
    assert all(mast_table._selected_mask[1:4])

    # test reset selections with no args
    mast_table.select()
    assert mast_table.selected_rows == []
    assert not all(mast_table._selected_mask)
    assert len(mast_table.selected_rows_table) == 0


def test_invalid_select(mast_observation_table):
    mast_table = MastTable(mast_observation_table)

    # invalid arg type
    with pytest.raises(TypeError, match="Invalid arg type provided. Must be int, list, or slice."):
        mast_table.select("fake")

    # invalid array
    with pytest.raises(TypeError, match="Array must be a Boolean mask or"):
        mast_table.select(np.ones(5))

    # non-existent column
    with pytest.raises(ValueError, match="Invalid column name provided."):
        mast_table.select(5, column="fake")

    # column and invalid arg type
    with pytest.raises(TypeError, match="Invalid arg type provided."):
        mast_table.select(slice(1, 3), column="observtn")


def test_validate_mask(mast_observation_table):
    mast_table = MastTable(mast_observation_table)

    # non-array input
    with pytest.raises(TypeError, match="Invalid mask provided."):
        mast_table._validate_mask("fake", 5)

    # invalid array values
    with pytest.raises(TypeError, match="Invalid mask provided."):
        mast_table._validate_mask(np.ones(5), 5)

    # invalid shape
    with pytest.raises(ValueError, match="Mask length must match the table."):
        mast_table._validate_mask(np.ones(4, dtype=bool), 5)
