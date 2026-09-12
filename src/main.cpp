// ================================================================
//  3+1D SU(2) Lattice Gauge Theory — finite-size scaling study
//  
//  Algorithm:      Kennedy-Pendleton exact heat bath
//  Parallelism:    OpenMP checkerboard (even/odd site decomposition)
//                  with splitmix64-seeded per-thread RNGs        
//  Error:          binned jackknife 
//  Output:         su2_data_L{8,12,16}.csv  +  benchmark.csv
//
//  Compile:  $ make && ./su2_sim
//                        or
//            $ g++ -O3 -fopenmp -std=c++17 main.cpp -o su2_sim -lm   
//  Run:      OMP_NUM_THREADS=N ./su2
// ================================================================

#include "su2.hpp"
#include "lattice.hpp"
#include "kernels.hpp"
#include "observables.hpp"
#include "statistics.hpp"
#include "benchmarks.hpp" 

#include <iostream>
#include <fstream>
#include <iomanip>
#include <omp.h>

// ---- Main ----
int main() {
    // 1. Simulation Hyperparameters
    const int LT = 4;
    const int N_THERM = 1000;
    const int N_MEAS = 500;
    const int N_SKIP = 5;
    const int JK_BIN_SIZE = 20;
    
    const double B_START = 2.10;
    const double B_END = 2.60;
    const int N_BETA = 51;
    double db = (B_END - B_START) / (N_BETA - 1);

    // 2. Parallel RNG Setup
    int max_threads = omp_get_max_threads();
    std::vector<std::mt19937> rngs_cold(max_threads);
    std::vector<std::mt19937> rngs_hot(max_threads);

    
    for(int i=0; i<max_threads; i++) {
        rngs_cold[i].seed(splitmix64_seed(123456789ULL, i));
        rngs_hot[i].seed(splitmix64_seed(987654321ULL, i)); // different base seed - a must to prevet correlation
    }

    // 3. Finite-size scaling
    std::vector<int> volumes = {8, 12, 16};

    for (int LS : volumes) {
        std::string fname_cold = "su2_cold_L" + std::to_string(LS) + ".csv";
        std::string fname_hot  = "su2_hot_L"  + std::to_string(LS) + ".csv";

        std::ofstream file_cold(fname_cold);
        std::ofstream file_hot(fname_hot);

        file_cold << "beta,P,P_err,L,L_err,chi,chi_err,U4,U4_err\n";
        file_hot  << "beta,P,P_err,L,L_err,chi,chi_err,U4,U4_err\n";

        std::cout << "\nStarting Scan on " << LS << "^3 x " << LT << "...\n";
        
        for(int ib=0; ib<N_BETA; ib++){
            double beta = B_START + ib*db;

            // --- Cold start ---
            Lattice lat_c(LS,LS,LS,LT);
            lat_c.cold_start();

            // --- Hot start ---
            Lattice lat_h(LS,LS,LS,LT);
            lat_h.hot_start(rngs_hot[0]); // seed initial links

            // 4. Thermalization
            for(int sw=0; sw<N_THERM; sw++) {
                heatbath_sweep_parallel(lat_c, beta, rngs_cold);
                heatbath_sweep_parallel(lat_h, beta, rngs_hot);
            }

            std::vector<double> P_ts_c(N_MEAS), L_ts_c(N_MEAS);
            std::vector<double> P_ts_h(N_MEAS), L_ts_h(N_MEAS);

            // 5. Measuring
            for(int m=0; m<N_MEAS; m++){
                for(int sk=0; sk<N_SKIP; sk++) {
                    heatbath_sweep_parallel(lat_c, beta, rngs_cold);
                    heatbath_sweep_parallel(lat_h, beta, rngs_hot);
                }

                P_ts_c[m] = measure_plaquette(lat_c);
                L_ts_c[m] = measure_polyakov(lat_c);

                P_ts_h[m] = measure_plaquette(lat_h);
                L_ts_h[m] = measure_polyakov(lat_h);
            }

            // 6. Analysis
            Res P_c = jackknife(P_ts_c, JK_BIN_SIZE);
            Res L_c, chi_c, U4_c;
            jackknife_derived(L_ts_c, JK_BIN_SIZE, double(LS*LS*LS), L_c, chi_c, U4_c);

            Res P_h = jackknife(P_ts_h, JK_BIN_SIZE);
            Res L_h, chi_h, U4_h;
            jackknife_derived(L_ts_h, JK_BIN_SIZE, double(LS*LS*LS), L_h, chi_h, U4_h);

            // 7. Output
            file_cold << beta << "," << P_c.mean << "," << P_c.err << ","
                      << L_c.mean << "," << L_c.err << ","
                      << chi_c.mean << "," << chi_c.err << ","
                      << U4_c.mean << "," << U4_c.err << "\n";

            file_hot << beta << "," << P_h.mean << "," << P_h.err << ","
                     << L_h.mean << "," << L_h.err << ","
                     << chi_h.mean << "," << chi_h.err << ","
                     << U4_h.mean << "," << U4_h.err << "\n";

            std::cout << "\r  Beta: " << std::fixed << std::setprecision(3) << beta 
                      << " | " << std::setw(3) << int(100.0*(ib+1)/N_BETA)
                      << "% complete" << std::flush;
        }
    }

    std::cout << "\n\nRunning benchmark...\n";
    run_benchmark();
    return 0;
}


// Congratulations for reaching the end !!