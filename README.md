# SU(2) Lattice Gauge Theory

A C++ implementation of **SU(2) lattice gauge theory** using Monte Carlo methods. The project studies the finite-temperature deconfinement transition and includes OpenMP parallelization and basic performance benchmarking.

## Features

- SU(2) lattice gauge field implementation
- Monte Carlo heatbath updates
- Checkerboard lattice updates
- Polyakov loop measurement
- Connected susceptibility
- Binder cumulant
- Hot and cold start simulations
- Jackknife statistical analysis
- OpenMP parallelization
- Strong-scaling benchmarks

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
