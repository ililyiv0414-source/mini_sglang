# Source this file in the remote Linux shell: source scripts/activate_sgl.sh
SGL_ROOT=/huhaifeng2/ljl/sgl
source "$SGL_ROOT/.venv/bin/activate"
export CUDA_HOME="$SGL_ROOT/toolchains/cuda-12.4"
export CC="$SGL_ROOT/toolchains/gcc/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$SGL_ROOT/toolchains/gcc/bin/x86_64-conda-linux-gnu-g++"
export PATH="$CUDA_HOME/bin:$PATH"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
