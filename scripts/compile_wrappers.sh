#!/bin/bash

# Compile cpp subsampling
cd ../utils/cpp_subsampling
python3 setup.py build_ext --inplace
cd ..

