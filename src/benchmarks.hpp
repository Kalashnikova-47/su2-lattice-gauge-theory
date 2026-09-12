#pragma once

#include "lattice.hpp"
#include "kernels.hpp"
#include "statistics.hpp"

#include <iostream>
#include <fstream>
#include <vector>
#include <omp.h>

// ---- Parallel scaling benchmark ----

// Runs SWEEPS sweeps at a fixed β and records wall time for each thread count.
// Uses a fresh lattice and RNG pool so results are independent of the main scan.

inline void run_benchmark() {
    // Fixed parameters for a representative test case
    const int LS = 12; 
    const int LT = 4; 
    const int SWEEPS = 500; 
    const double beta = 2.30;
    
    Lattice lat(LS, LS, LS, LT); 
    lat.cold_start();
    
    int max_threads = omp_get_max_threads();
    std::vector<std::mt19937> rngs(max_threads);
    for(int i = 0; i < max_threads; i++) {
        rngs[i].seed(splitmix64_seed(123456ULL, i));
    }

    std::ofstream bfile("benchmark.csv");
    bfile << "Threads,Time,Speedup,Efficiency\n";
    
    double time_1_thread = 0.0;
    // Testing powers of 2 or specific thread counts
    std::vector<int> thread_counts = {1, 2, 4, 8, 12, 16}; 

    std::cout << "\n>>> Starting Performance Benchmark...\n";

    for(int t : thread_counts) {
        if (t > max_threads) continue;
        
        omp_set_num_threads(t);
        double start_time = omp_get_wtime();
        
        for(int i = 0; i < SWEEPS; i++) {
            heatbath_sweep_parallel(lat, beta, rngs);
        }
        
        double time_taken = omp_get_wtime() - start_time;
        if (t == 1) time_1_thread = time_taken;
        
        double speedup = time_1_thread / time_taken;
        double efficiency = (speedup / t) * 100.0;

        bfile << t << "," << time_taken << "," << speedup << "," << efficiency << "\n";
        
        std::cout << "  Threads: " << t << " | Time: " << time_taken 
                  << "s | Efficiency: " << efficiency << "%" << std::endl;
    }
    bfile.close();
}