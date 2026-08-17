import pybind11
from setuptools import setup, Extension

ext_modules = [
    Extension(
        "graph_segmenter_cpp", 
        ["graph_segmenter.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=["-O3", "-std=c++17", "-march=native"],
    ),
]

setup(
    name="graph_segmenter_cpp",
    ext_modules=ext_modules,
)