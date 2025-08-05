#[cfg(feature = "cuda")]
fn compile_kernel() {
    nvcc::compile("src/core/hardware/cuda/mev_kernel.cu").unwrap();
}