"""
https://github.com/hoffstadt/DearPyGui/issues/844
"""

from dearpygui.core import *
import numpy as np

from dearpygui import core as dpgc
from dearpygui import simple as dpgs

with dpgs.window("my window"):
    add_plot("Line Plot##demo", x_axis_name="x", y_axis_name="y", height=400)
    add_line_series(
        "Line Plot##demo", "111",
        np.arange(10, dtype=np.float64), np.arange(10, dtype=np.int32))

    # uncomment below code ... fails for np.int64
    # add_line_series(
    #     "Line Plot##demo", "222",
    #     np.zeros(10), np.arange(10, dtype=np.int64))

    # uncomment below code ... fails for np.float32
    # uncomment below lines to get error
    # add_line_series(
    #     "Line Plot##demo", "333",
    #     np.zeros(10, dtype=np.float32), np.arange(10))

dpgc.start_dearpygui()

