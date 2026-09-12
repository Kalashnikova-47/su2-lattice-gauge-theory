#pragma once

//#include "lattice.hpp"


// ---- Observables ----
// Average plaquette <P> = (1/N_plaq) Σ (1/2) ReTr[U_μν].
// For SU(2), (1/2) ReTr[U] = q[0] ∈ [-1,1].
// Ranges from 0 (strong coupling) to 1 (continuum limit).

inline double measure_plaquette(const Lattice& lat) {
    double sum=0.0; long n=0;
    for(int s=0; s<lat.vol; s++) {
        for(int mu=0; mu<ND-1; mu++) {
            for(int nu=mu+1; nu<ND; nu++) {
                SU2 P=mul(lat.U(s,mu),mul(lat.U(lat.fwd[s][mu],nu),
                      mul(dag(lat.U(lat.fwd[s][nu],mu)),dag(lat.U(s,nu)))));
                sum+=P.q[0]; n++;
            }
        }
    }
    return sum/double(n);
}

// Spatially averaged Polyakov loop L̄ = (1/V₃) Σ_{x,y,z} (1/2)Tr[∏_t U₃(x,y,z,t)].
// Order parameter for deconfinement: L̄ → 0 (confined), L̄ ≠ 0 (deconfined).
// Returns the signed spatial mean; take |L̄| per config before averaging.

inline double measure_polyakov(const Lattice& lat) {
    double sum=0.0;
    for(int x=0; x<lat.L[0]; x++) for(int y=0; y<lat.L[1]; y++) for(int z=0; z<lat.L[2]; z++){
        SU2 loop = lat.U(lat.site_idx(x,y,z,0), 3);
        for(int t=1; t<lat.L[3]; t++) loop = mul(loop, lat.U(lat.site_idx(x,y,z,t), 3));
        sum += loop.q[0];
    }
    return sum / double(lat.L[0]*lat.L[1]*lat.L[2]);
}
