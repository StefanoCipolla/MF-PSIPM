# PS-IPM V2 latest bundle

This bundle contains the latest spectral stable-column PS-IPM V2 and its two
benchmark drivers as of 2026-08-12.

## Files

- `psipm_spectral_preconditioner.py`: V2 solver and command-line entry point.
- `psipm_stable_preconditioner.py`: stable active-column preconditioner used by
  V2.
- `psipm_active_pool.py`: shared PS-IPM engine used by both solver layers.
- `batch_test_spectral_preconditioner.py`: process-isolated comparison of V2,
  HiGHS IPX, and HiPO on every supported LP in a selected folder.
- `batch_test_vlsi_spectral.py`: process-isolated comparison on the Bonn Josef
  and Erhard DIMACS network-flow files.
- `environment.yml`: reproducible Conda environment.
- `install_hipo.sh`: matching `highspy-extras` installer and verifier.

Keep the three `psipm_*.py` files in the same directory. V2 imports the other
two modules by filename.

## Environment

```bash
conda env create --file environment.yml
conda activate psipm-benchmark
```

The environment includes HiPO. If HiPO needs to be repaired after a package
upgrade, use the active environment's interpreter:

```bash
PYTHON="$(command -v python)" ./install_hipo.sh
```

## Run V2 directly

```bash
python psipm_spectral_preconditioner.py MODEL.mps \
  --time-limit 2000 --pc 3 --printlevel 1
```

Add `--presolve` to enable HiGHS presolve.

## Benchmark a folder of LP files

```bash
python batch_test_spectral_preconditioner.py /path/to/models \
  --csv comparison.csv --time-limit 2000 --timeout-margin 120 \
  --threads 1 --presolve off --repeats 1 --order rotate --overwrite
```

Use `--choose-folder` instead of a path on a machine with a graphical display.
Supported inputs include MPS, compressed MPS, LP, QPS, and native OR-Library
`rail*.gz` files. Each solver run uses a fresh process; total time includes
read, conversion, setup, solve, and validation.

## Benchmark Josef and Erhard

Put `Josef_FlowGraph_1.net` and `Erhard_FlowGraph_1.net` in the selected folder,
then run:

```bash
python batch_test_vlsi_spectral.py /scratch/sc9c23/vlsi \
  --models all --csv vlsi_comparison.csv \
  --time-limit 4000 --timeout-margin 600 --threads 1 --overwrite
```

For Josef only:

```bash
python batch_test_vlsi_spectral.py /scratch/sc9c23/vlsi \
  --models Josef --methods psipm-ssp highs-ipx highs-hipo \
  --objective-scaling auto --ground-network auto \
  --csv josef_comparison.csv --time-limit 4000 --overwrite
```

Network grounding removes one redundant incidence row per connected component
and validates the returned iterate against the original ungrounded equations.
