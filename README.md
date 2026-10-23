# SU(2) Lattice Gauge Theory

A C++17 implementation of **SU(2) lattice gauge theory** using Monte Carlo methods. This project studies the finite-temperature deconfinement transition on a 4D lattice, featuring OpenMP parallelization, exact heat bath sampling, and finite-size scaling analysis.

## Features

- **Physics & Algorithms:**
  - SU(2) gauge field implementation using unit quaternions.
  - Exact Kennedy-Pendleton heat bath algorithm for SU(2) link updates.
  - Observables: Plaquette, Polyakov loop, connected susceptibility ($\chi_L$), and Binder cumulant ($U_4$).
  - Binned jackknife statistical error analysis handling non-linear cross-correlations.
  - Hot (random) and cold (identity) start initializations.
- **Performance:**
  - Race-free checkerboard (even/odd) lattice decomposition.
  - OpenMP parallelization for multi-core scaling.
  - Robust thread-safe RNG using `std::mt19937` seeded via `splitmix64` to prevent state overlap.
  - Automated strong-scaling benchmarks.

## Prerequisites

To build and run the simulation, you will need:
- A C++ compiler with C++17 support (e.g., `g++` or `clang++`).
- OpenMP (usually bundled with modern compilers).
- Python 3.x with `matplotlib`, `numpy`, and `pandas` (for analysis scripts).

## Building and Running

Compile the simulation using the provided Makefile:
```bash
make
```

## Structure

```text
.
├── src/
│   ├── benchmarks.hpp
│   ├── kernels.hpp
│   ├── lattice.hpp
│   ├── main.cpp
│   ├── observables.hpp
│   ├── statistics.hpp
│   └── su2.hpp
│
├── analysis/
│   ├── plot_cold.py
│   └── plot_hot.py
│
├── results/
│   ├── raw/
│   ├── processed/
│   └── figures/
│
├── docs/
│   └── LQCD.pdf
│
├── Makefile
└── README.md
```
