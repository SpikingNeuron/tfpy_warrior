from dearpygui import core as dpgc
from dearpygui import simple as dpgs

with dpgs.window("my window"):
    with dpgs.managed_columns("columns", columns=2):
        # dpgc.set_managed_column_width(item="columns", column=0, width=0.2)
        # dpgc.set_managed_column_width(item="columns", column=1, width=0.8)
        dpgc.add_text("Hi I am DearPyGui")
        dpgc.add_text("I dont know Groot ...")
        dpgc.add_text("Hi I am DearPyGui")
        dpgc.add_text("I dont know Groot ...")

dpgc.start_dearpygui()
