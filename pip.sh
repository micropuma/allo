# install python packages under the allo conda environment
python3 -m pip install -r $MLIR_AIE_ROOT_DIR/python/requirements.txt
HOST_MLIR_PYTHON_PACKAGE_PREFIX=aie python3 -m pip install -r $MLIR_AIE_ROOT_DIR/python/requirements_extras.txt
