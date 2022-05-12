"""
dlg_lowpass_components component module.

This is the module of dlg_lowpass_components containing DALiuGE data components.
Here you put your main data classes and objects.

Typically a component project will contain multiple components and will
then result in a single EAGLE palette.

Be creative! do whatever you need to do!
"""
import base64
import logging
import os
import pickle

from dlg.drop import AbstractDROP
from dlg.meta import (
    dlg_batch_input,
    dlg_batch_output,
    dlg_bool_param,
    dlg_component,
    dlg_float_param,
    dlg_int_param,
    dlg_streaming_input,
    dlg_string_param,
)

logger = logging.getLogger(__name__)

##
# @brief MyData
# @details Template app for demonstration only!
# Replace the documentation with whatever you want/need to show in the DALiuGE
# workflow editor. The dataclass parameter should contain the relative Pythonpath
# to import MyApp.
#
# @par EAGLE_START
# @param category DataDrop
# @param[in] param/appclass Drop Class/dlg_lowpass_components.MyData/String/readonly/
#     \~English Import direction for application class
# @param[in] param/dummy Dummy parameter/ /String/readwrite/
#     \~English Dummy modifyable parameter
# @param[in] port/dummy Dummy in/float/
#     \~English Dummy producer port
# @param[out] port/dummy Dummy out/float/
#     \~English Dummy consumer port
# @par EAGLE_END

# Data components usually directly inhert from the AbstractDROP class. Please
# refer to the Developer Guide for more information.


class MyDataDROP(AbstractDROP):
    """
    A dummy dataDROP that points to nothing.
    """

    def initialize(self, **kwargs):
        pass

    def getIO(self):
        return f"Hello from {__class__.__name__}"

    @property
    def dataURL(self):
        return f"Hello from the dataURL method"
