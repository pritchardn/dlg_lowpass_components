import pytest

from dlg_lowpass_components import LPSignalGenerator

given = pytest.mark.parametrize


def test_myApp_class():
    assert LPSignalGenerator("a", "a").run() == "Hello from LPSignalGenerator"


def test_myData_class():
    assert MyDataDROP("a", "a").getIO() == "Hello from MyDataDROP"


def test_myData_dataURL():
    assert MyDataDROP("a", "a").dataURL == "Hello from the dataURL method"
