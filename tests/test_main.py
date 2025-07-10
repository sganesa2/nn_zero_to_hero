import pytest
from src.micrograd.utils import add

@pytest.mark.parametrize(
    "inp1, inp2, expected",
    [
        (1,2,3)
    ])
def test_addition(inp1:int, inp2:int, expected:int)->None:
    assert add(inp1, inp2) == expected