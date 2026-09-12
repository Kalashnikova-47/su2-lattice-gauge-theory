#pragma once

#include "lattice.hpp"
#include "su2.hpp"
#include <omp.h>

// ---- Gauge update kernel ----

// Collects the 2(ND-1)=6 3-link paths ("staples") in touch with U_μ(s).
// The link action is  S_link = -(β/2) ReTr[U_μ(s)·K],  
// so K determines the local Boltzmann weight used by the heat bath.

inline SU2 compute_staple(const Lattice& lat, int s, int mu) {
    SU2 K(0,0,0,0);
    int s_pmu = lat.fwd[s][mu];
    for(int nu=0; nu<ND; nu++){
        if(nu==mu) continue;
        int s_pnu = lat.fwd[s][nu], s_mnu = lat.bwd[s][nu];
        // Forward staple:  U_ν(s+μ̂) · U_μ†(s+ν̂) · U_ν†(s)
        SU2 F = mul(lat.U(s_pmu,nu), mul(dag(lat.U(s_pnu,mu)), dag(lat.U(s,nu))));
        // Backward staple: U_ν†(s+μ̂-ν̂) · U_μ†(s-ν̂) · U_ν(s-ν̂)
        SU2 B = mul(dag(lat.U(lat.fwd[s_mnu][mu],nu)), mul(dag(lat.U(s_mnu,mu)), lat.U(s_mnu,nu)));
        K = add(K, add(F,B));
    }
    return K;
}

// Kennedy-Pendleton exact heat bath for SU(2)
// Samples U from P(U) ∝ exp((β/2) ReTr[U·K]) in three steps:
//   1. Decompose K = a·V,  a = ||K||,  V ∈ SU(2).
//   2. Sample scalar part w₀ from exp(k·w₀)·√(1-w₀²) via rejection.
//      (Exponential proposal; Jacobian acceptance with a single draw r₂
//       so that P(accept|w0) = √(1-w₀²).)
//   3. Sample (w₁,w₂,w₃) uniformly on the 2-sphere of radius √(1-w0²).
//   4. Reconstruct U = W·V†  where W = (w₁,w₂,w₃).

inline SU2 su2_heatbath(const SU2& K, double beta, std::mt19937& rng) {
    
    double a = std::sqrt(norm2(K));
    if(a < 1e-14) return random_su2(rng);  // degenerate staple → Haar-uniform
    
    SU2 V(K.q[0]/a, K.q[1]/a, K.q[2]/a, K.q[3]/a);
    double k = beta * a;                    // effective coupling for this link
    
    std::uniform_real_distribution<double> uni(0.0,1.0);
    if(k < 1e-8) return mul(random_su2(rng), dag(V));  // flat weight → Haar
    
    double exp_m2k = std::exp(-2.0*k);
    double w0;

    for(;;){
        // Draw from exp(k·x) on [-1,1] via inversion: r₁ ∈ [e^{-2k},1]
        w0 = 1.0 + std::log(exp_m2k + (1.0-exp_m2k)*uni(rng))/k;
        // Accept with probability sin(θ) = √(1-w₀²); one draw, then square.
        double r2 = uni(rng);
        if(r2*r2 <= std::max(0.0, 1.0-w0*w0)) break; 
    }

    double r = std::sqrt(std::max(0.0, 1.0-w0*w0));
    double phi = 2.0*M_PI*uni(rng), cth = 2.0*uni(rng)-1.0, sth = std::sqrt(std::max(0.0, 1.0-cth*cth));
    
    SU2 W(w0, r*sth*std::cos(phi), r*sth*std::sin(phi), r*cth);
    SU2 U = mul(W, dag(V)); reunitarize(U); 
    return U;
}

// One full lattice sweep using the checkerboard decomposition.
// The outer loop over μ means we update all spatial/temporal links of one
// direction before moving to the next. This is a Gauss-Seidel ordering
// (newer neighbour values are read as soon as they are available), which
// is a valid Gibbs sampler — detailed balance holds direction-by-direction.
// Each even-site update is race-free because even-site staples only read
// links sitting at odd-parity neighbours, which are not written in that update.

inline void heatbath_sweep_parallel(Lattice& lat, double beta, std::vector<std::mt19937>& thread_rngs) {
    for(int mu = 0; mu < ND; mu++) {

        #pragma omp parallel 
        {
            int tid = omp_get_thread_num();
            auto& rng = thread_rngs[tid];

            #pragma omp for 
            for(size_t i = 0; i < lat.even_sites.size(); i++) {
                int s = lat.even_sites[i];
                lat.U(s, mu) = su2_heatbath(compute_staple(lat, s, mu), beta, rng);
            }
        }
        
        #pragma omp parallel 
        {
            int tid = omp_get_thread_num();
            auto& rng = thread_rngs[tid];

            #pragma omp for 
            for(size_t i = 0; i < lat.odd_sites.size(); i++) {
                int s = lat.odd_sites[i];
                lat.U(s, mu) = su2_heatbath(compute_staple(lat, s, mu), beta, rng);
            }
        }
    }
}


