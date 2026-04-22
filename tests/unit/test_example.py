"""
A test module that tests your example module.

Some people prefer to write tests in a test file for each function or
method/ class. Others prefer to write tests for each module. That decision
is up to you. This test example provides a single test for the example.py
module.
"""

from pyospackage_howard.example import add_numbers
import pytest

def test_add_numbers():
    """
    Test that add_numbers works as expected.

    A single line docstring for tests is generally sufficient.
    """
    out = add_numbers(1, 2)
    expected_out = 3
    assert  out == expected_out, f"Expected {expected_out} but got {out}"

@pytest.mark.parametrize("a, b", [
    (1, 2),
    (1.5, 2.5),
    (0, -3.14),
])
def test_number_type(a, b):
    """Validate that inputs are numeric."""
    assert isinstance(a, (int, float)), f"First argument must be numeric, got {type(a).__name__}"
    assert isinstance(b, (int, float)), f"Second argument must be numeric, got {type(b).__name__}"

def test_add_zeroes():
    out = add_numbers(1, 0)
    expected_out = 1
    assert  out == expected_out, f"Expected {expected_out} but got {out}"

    out = add_numbers(0, 2)
    expected_out = 2
    assert  out == expected_out, f"Expected {expected_out} but got {out}"

def test_add_negatives():
    out = add_numbers(1, -2)
    expected_out = -1
    assert  out == expected_out, f"Expected {expected_out} but got {out}"

    out = add_numbers(-1, 2)
    expected_out = 1
    assert  out == expected_out, f"Expected {expected_out} but got {out}"
