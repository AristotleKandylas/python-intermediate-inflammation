"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
import pytest

from inflammation.models import daily_mean, daily_max, daily_min

# def test_daily_mean_zeros():
#     """Test that mean function works for an array of zeros."""
    

#     test_input = np.array([[0, 0],
#                            [0, 0],
#                            [0, 0]])
#     test_result = np.array([0, 0])

#     # Need to use Numpy testing functions to compare arrays
#     npt.assert_array_equal(daily_mean(test_input), test_result)


# def test_daily_mean_integers():
#     """Test that mean function works for an array of positive integers."""

#     test_input = np.array([[1, 2],
#                            [3, 4],
#                            [5, 6]])
#     test_result = np.array([3, 4])

#     # Need to use Numpy testing functions to compare arrays
#     npt.assert_array_equal(daily_mean(test_input), test_result)

####parameterized tests for daily_mean function
@pytest.mark.parametrize(
    "test, expected",
    [
        ([ [0, 0], [0, 0], [0, 0] ], [0, 0]),
        ([ [1, 2], [3, 4], [5, 6] ], [3, 4]),
    ])
def test_daily_mean(test, expected):
    """Test mean function works for array of zeroes and positive integers."""
    npt.assert_array_equal(daily_mean(np.array(test)), np.array(expected))


# def test_daily_max_integers():
#     """Test that max function works for an array of positive integers."""
    
#     test_input = np.array([[1, 2],
#                            [3, 4],
#                            [5, 6]])
#     test_result = np.array([5, 6])
#     # Need to use Numpy testing functions to compare arrays
#     npt.assert_array_equal(daily_max(test_input), test_result)

# def test_daily_max_negatives():
#     """Test that max function works for an array of negative integers."""

#     test_input = np.array([[-1, -2],
#                            [-3, -4],
#                            [-5, -6]])
#     test_result = np.array([-1, -2])

#     # Need to use Numpy testing functions to compare arrays
#     npt.assert_array_equal(daily_max(test_input), test_result)
@pytest.mark.parametrize(
    "test, expected",
    [
        ( [ [1, 2], [3, 4], [5, 6] ], [5, 6] ),
        ( [ [-1, -2], [-3, -4], [-5, -6] ], [-1, -2] ),
    ])
def test_daily_max(test, expected):
    """Test max function works for array of positive and negative integers."""
    npt.assert_array_equal(daily_max(np.array(test)), np.array(expected))

# def test_daily_min_mixed():
#     """Test that min function works for an array of mixed positive and negative integers."""

#     test_input = np.array([[1, -2],
#                            [-3, 4],
#                            [5, -6]])
#     test_result = np.array([-3, -6])

#     # Need to use Numpy testing functions to compare arrays
#     npt.assert_array_equal(daily_min(test_input), test_result)

# def test_daily_min_zeros():
#     """Test that min function works for an array of zeros."""
    
#     test_input = np.array([[0, 0],
#                            [0, 0],
#                            [0, 0]])
#     test_result = np.array([0, 0])

#     # Need to use Numpy testing functions to compare arrays
#     npt.assert_array_equal(daily_min(test_input), test_result)
@pytest.mark.parametrize(
    "test, expected",
    [
        ( [ [1, -2], [-3, 4], [5, -6] ], [-3, -6] ),
        ( [ [0, 0], [0, 0], [0, 0] ], [0, 0] ),
    ])
def test_daily_min(test, expected):
    """Test min function works for array of mixed positive/negative integers and zeros."""
    npt.assert_array_equal(daily_min(np.array(test)), np.array(expected))

def test_daily_min_string():
    """Test for TypeError when passing strings"""

    with pytest.raises(TypeError):
        error_expected = daily_min([['Hello', 'there'], ['General', 'Kenobi']])
