# AKT26 GPU tile Cholesky starter

This folder contains a native FP32/FP64 blocked Cholesky benchmark for CUDA.jl.
It is a starting point for the challenge, not a measured result: run it on the
remote GPU and keep only rows that pass the accuracy check.

## Files

- `chol_gpu.jl`: blocked factorization, GPU SPD input generation, and residual check.
- `smoke_test.jl`: small FP32/FP64 end-to-end GPU check.
- `bench_one.jl`: runs exactly one precision and one matrix size per job.
- `bench.jl`: optional multi-size sweep; do not use it for the per-test Slurm jobs.
- `plot_results.py`: annotated performance plot with theoretical and sustained peaks.
- `run_smoke.slurm`: five-minute system smoke test.
- `run_final.slurm`: five-minute benchmark script for one precision/size pair.

## Setup on the cluster

Copy these files into `/home/khawarizmi/team1234`. Julia was not available in
the initial login-shell check, so first load a Julia module if the cluster
provides one. Otherwise, install a compatible Julia release in your home
directory. With Julia available, install/precompile CUDA.jl on the login node:

```bash
cd /home/khawarizmi/team1234
julia --project=. -e 'using Pkg; Pkg.instantiate(); using CUDA; CUDA.precompile_runtime()'
```

Run that once on the login node while package downloads are available; it
creates the manifest and warms the CUDA runtime. Verify CUDA.jl detects the
compute-node GPU before using the benchmark.

## Run and plot

Submit the smoke test as one short Slurm job from the project directory:

```bash
mkdir -p results
sbatch run_smoke.slurm
```

Then submit every benchmark point as its own job. For example:

```bash
sbatch --export=ALL,N=1000,PRECISION=Float32 run_final.slurm
sbatch --export=ALL,N=1000,PRECISION=Float64 run_final.slurm
sbatch --export=ALL,N=1500,PRECISION=Float32 run_final.slurm
sbatch --export=ALL,N=1500,PRECISION=Float64 run_final.slurm
```

Continue in 500-element steps through N=7000. Each job tests only one
precision/size pair, runs three factorization timings, and writes a uniquely
named CSV under `results/`. Matrix creation, warm-up, and the N=203 accuracy
check are outside the timed region. Do not submit a multi-size sweep as one
five-minute job.

Create the graph where Python and Matplotlib are available. Supply FP32/FP64
theoretical peaks computed from the actual GPU specifications/clocks; do not
copy estimated values from a draft plan:

```bash
python3 plot_results.py results/bench_*.csv \
  --fp32-peak FP32_PEAK_GFLOPS \
  --fp64-peak FP64_PEAK_GFLOPS \
  --gpu "GPU model" \
  --output results/performance.png
```

The sustained-peak references are calculated from the maximum measured rate
in the CSV. The theoretical peak values are arguments because they must be
verified for the actual GPU and operating clocks.
