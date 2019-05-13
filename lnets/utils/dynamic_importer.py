import os
import sys
import importlib


def dynamic_import(filepath, class_name):
    tail, head = os.path.split(filepath)
    sys.path.append(tail)
    module = importlib.import_module(".".join(head.split(".")[:-1]))
    sys.path.remove(tail)

    my_class = getattr(module, class_name)

    return my_class
