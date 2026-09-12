#pragma once

#include <cmath>
#include <random>


// ---- SU(2) group elements represented as unit quaternions ----
// U = q[0]·I + i·(q[1]·σ₁ + q[2]·σ₂ + q[3]·σ₃),  Σq²=1
// Dagger flips the sign of the vector part.
// mul() is the quaternion product derived from the Pauli algebra.


struct SU2 {
    double q[4]; 
    SU2() { q[0]=1; q[1]=q[2]=q[3]=0; }
    SU2(double a, double b, double c, double d) { q[0]=a; q[1]=b; q[2]=c; q[3]=d; }
};


inline SU2 dag(const SU2& U) { return {U.q[0], -U.q[1], -U.q[2], -U.q[3]}; }
inline SU2 mul(const SU2& A, const SU2& B) {
    return {
        A.q[0]*B.q[0] - A.q[1]*B.q[1] - A.q[2]*B.q[2] - A.q[3]*B.q[3],
        A.q[0]*B.q[1] + B.q[0]*A.q[1] + A.q[2]*B.q[3] - A.q[3]*B.q[2],
        A.q[0]*B.q[2] + B.q[0]*A.q[2] + A.q[3]*B.q[1] - A.q[1]*B.q[3],
        A.q[0]*B.q[3] + B.q[0]*A.q[3] + A.q[1]*B.q[2] - A.q[2]*B.q[1]
    };
}
inline SU2 add(const SU2& A, const SU2& B) { return {A.q[0]+B.q[0], A.q[1]+B.q[1], A.q[2]+B.q[2], A.q[3]+B.q[3]}; }
inline double norm2(const SU2& U) { return U.q[0]*U.q[0] + U.q[1]*U.q[1] + U.q[2]*U.q[2] + U.q[3]*U.q[3]; }

inline void reunitarize(SU2& U) {
    double n = std::sqrt(norm2(U));
    if (n < 1e-28){ U=SU2(); return; }
    double inv = 1.0/n;
    U.q[0]*=inv; U.q[1]*=inv; U.q[2]*=inv; U.q[3]*=inv;
}

inline SU2 random_su2(std::mt19937& rng) {
    std::normal_distribution<double> g(0.0, 1.0);
    SU2 U(g(rng), g(rng), g(rng), g(rng));
    reunitarize(U); 
    return U;
}

// ----  RNG setup ----

// Maps a (base_seed, thread_id) pair to a well-separated 64-bit seed.
// Consecutive integers seed in MT19937 gives correlated initial states; 
// splitmix64 kills that correlation before MT is initialised.
// Refer https://rosettacode.org/wiki/Pseudo-random_numbers/Splitmix64

inline uint64_t splitmix64_seed(uint64_t base, int thread_id) {
    uint64_t z = base + (uint64_t)thread_id * 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}