#pragma once

#include <vector>
#include <cmath>
#include <cstdint>


// ---- Jackknife estimators ----
// Binned jackknife for a linear observable.
// N measurements → N_b = N/B bins of size B. Jackknife error:
//   σ_JK = √((N_b-1)/N_b · Σ_k (x̄_JK^k - x̄)²)
// Choose B > 2τ_int to ensure bin means are approximately decorrelated.

struct Res { double mean, err; };

inline Res jackknife(const std::vector<double>& data, int B) {
    
    int N = data.size(), Nb = N/B;
    if(Nb <= 1) return {0,0};
    
    double total_sum = 0;
    std::vector<double> bin_sums(Nb, 0.0);
    for(int i=0; i<Nb*B; i++) {
        total_sum += data[i];
        bin_sums[i/B] += data[i];
    }
    double mean = total_sum / (Nb*B);
    
    double err_sq = 0;
    for(int i=0; i<Nb; i++) {
        double jk_val = (total_sum - bin_sums[i]) / ((Nb-1)*B);
        err_sq += (jk_val - mean)*(jk_val - mean);
    }
    return {mean, std::sqrt((Nb-1)*err_sq/Nb)};
}

// Joint jackknife for three nonlinear Polyakov observables.
//   χL = V₃·(<L̄²> - <|L̄|>²)  — susceptibility; peaks at β_c
//   U4  = 1 - <L̄⁴>/(3<L̄²>²)  — Binder cumulant; sigmoid 0 to 2/3,
//    crosses at U4* ≈ 0.465 for different Ls (3D Ising universality)
//

inline void jackknife_derived(const std::vector<double>& Lb, int B, double vol3, Res& L_abs, Res& chi, Res& U4) {
    int N = Lb.size(), Nb = N/B;
    
    double sum_absL=0, sum_L2=0, sum_L4=0;
    std::vector<double> b_absL(Nb,0), b_L2(Nb,0), b_L4(Nb,0);

    for(int i=0; i<Nb*B; i++) {
        double aL = std::abs(Lb[i]), L2 = Lb[i]*Lb[i], L4 = L2*L2;
        sum_absL += aL; sum_L2 += L2; sum_L4 += L4;
        b_absL[i/B] += aL; b_L2[i/B] += L2; b_L4[i/B] += L4;
    }
    
    double m_absL = sum_absL/(Nb*B), m_L2 = sum_L2/(Nb*B), m_L4 = sum_L4/(Nb*B);
    
    L_abs.mean = m_absL; 
    chi.mean = vol3 * (m_L2 - m_absL * m_absL);
    U4.mean = (m_L2 > 1e-16) ? 1.0 - m_L4/(3.0*m_L2*m_L2) : 0.0;

    double sq_absL=0, sq_chi=0, sq_U4=0;

    for(int i=0; i<Nb; i++) {
        double jk_absL = (sum_absL - b_absL[i]) / ((Nb-1)*B);
        double jk_L2   = (sum_L2 - b_L2[i]) / ((Nb-1)*B);
        double jk_L4   = (sum_L4 - b_L4[i]) / ((Nb-1)*B);
        
        double jk_chi = vol3 * (jk_L2 - jk_absL * jk_absL);
        double jk_u4  = (jk_L2 > 1e-16) ? 1.0 - jk_L4/(3.0*jk_L2*jk_L2) : 0.0;
        
        sq_absL += (jk_absL - L_abs.mean)*(jk_absL - L_abs.mean);
        sq_chi  += (jk_chi - chi.mean)*(jk_chi - chi.mean);
        sq_U4   += (jk_u4 - U4.mean)*(jk_u4 - U4.mean);
    }
    
    L_abs.err = std::sqrt((Nb-1)*sq_absL/Nb);
    chi.err = std::sqrt((Nb-1)*sq_chi/Nb);
    U4.err = std::sqrt((Nb-1)*sq_U4/Nb);
}
