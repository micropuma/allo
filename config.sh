#!/bin/bash
source /usr/xilinx/Environment_Bash/AIE.sh         # 配置AIE环境
export LLVM_BUILD_DIR=/home/douliyang/large/mlir-workspace/allo/externals/llvm-project/build
export MLIR_AIE_ROOT_DIR=/home/douliyang/large/mlir-workspace/allo/mlir-aie

export PATH=$MLIR_AIE_ROOT_DIR/install/bin:$PATH
export PYTHONPATH=$MLIR_AIE_ROOT_DIR/install/python:$PYTHONPATH
export MLIR_AIE_INSTALL_DIR=$MLIR_AIE_ROOT_DIR/install
export PEANO_INSTALL_DIR=/home/douliyang/app/llvm-aie/llvm-aie
