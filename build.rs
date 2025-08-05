fn main() {
    #[cfg(feature = "cuda")]
    {
        println!("cargo:rerun-if-changed=src/core/hardware/cuda/mev_kernel.cu");
        
        let output = std::process::Command::new("nvcc")
            .args(&[
                "--ptx",
                "-O3",
                "--fmad=true",
                "--use_fast_math",
                "--gpu-architecture=sm_80", // Target Ampere (adjust for your GPUs)
                "-o", "src/core/hardware/cuda/mev_kernel.ptx",
                "src/core/hardware/cuda/mev_kernel.cu"
            ])
            .output()
            .expect("Failed to execute nvcc");

        if !output.status.success() {
            panic!(
                "CUDA kernel compilation failed:\n{}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }

        println!("cargo:rustc-env=PTX_PATH=src/core/hardware/cuda/mev_kernel.ptx");
    }
}