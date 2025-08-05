# Compile PTX (run from project root)
nvcc --ptx -O3 --fmad=true -o src/core/hardware/cuda/mev_kernel.ptx src/core/hardware/cuda/mev_kernel.cu