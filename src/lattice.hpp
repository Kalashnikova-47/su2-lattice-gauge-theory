#pragma once 
    
#include <vector>
#include <array>
#include <random>
#include "su2.hpp"

static constexpr int ND = 4;

// --- Lattice ---
// Flat 4D lattice: sites indexed as s = x + Lx(y + Ly(z + Lz·t)).
// Neighbour tables are precomputed so the inner loops has no modulos.
// even_sites / odd_sites partition by coordinate parity
// making even-site and odd-site updates independent within each pass.

struct Lattice {
    int L[ND]; int vol;
    std::vector<std::array<int,ND>> fwd, bwd;
    std::vector<SU2> links;
    std::vector<int> even_sites, odd_sites; 

    Lattice(int Lx, int Ly, int Lz, int Lt) {
        L[0]=Lx; L[1]=Ly; L[2]=Lz; L[3]=Lt; vol=Lx*Ly*Lz*Lt;
        links.resize(vol*ND); 
        even_sites.reserve(vol/2); odd_sites.reserve(vol/2);
        build_tables();
    }
    inline int site_idx(int x, int y, int z, int t) const { return x+L[0]*(y+L[1]*(z+L[2]*t)); }
    
    void build_tables() {
        fwd.resize(vol); bwd.resize(vol);
        for(int s=0; s<vol; s++){
            int c[ND]; int v=s;
            // Extracting coordinates using modulo arithmetic
            c[0]=v%L[0]; v/=L[0]; c[1]=v%L[1]; v/=L[1];
            c[2]=v%L[2]; c[3]=v/L[2];
            
            if ((c[0]+c[1]+c[2]+c[3]) % 2 == 0) even_sites.push_back(s);
            else odd_sites.push_back(s);

            for(int mu=0; mu<ND; mu++){
                int cf[ND], cb[ND];
                for(int i=0; i<ND; i++){ cf[i]=cb[i]=c[i]; }
                cf[mu]=(c[mu]+1)%L[mu]; cb[mu]=(c[mu]-1+L[mu])%L[mu];
                fwd[s][mu]=site_idx(cf[0],cf[1],cf[2],cf[3]);
                bwd[s][mu]=site_idx(cb[0],cb[1],cb[2],cb[3]);
            }
        }
    }
    inline SU2& U(int s, int mu) { return links[s*ND+mu]; }
    inline const SU2& U(int s, int mu) const { return links[s*ND+mu];}

    void cold_start(){ for(auto&u:links){u.q[0]=1;u.q[1]=u.q[2]=u.q[3]=0;} }   // start with identity 
    void hot_start(std::mt19937& rng){ for(auto& u:links) u=random_su2(rng); } // start wih random SU(2)
};
