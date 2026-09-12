// ================================================================
//  3+1D SU(2) Lattice Gauge Theory — finite-size scaling study
//  
//  Formulas to keep in mind - Wilson action:  S = β Σ_{x,μ<ν} [1 - (1/2) ReTr U_μν(x)]
//  Algorithm:      Kennedy-Pendleton exact heat bath
//  Parallelism:    OpenMP checkerboard (even/odd site decomposition)
//                  with splitmix64-seeded per-thread RNGs        
//  Error:          binned jackknife 
//  Output:         su2_data_L{8,12,16}.csv  +  benchmark.csv
//
//  Compile:  g++ -O3 -fopenmp -std=c++17 -o su2 su2.cpp -lm
//  Run:      OMP_NUM_THREADS=N ./su2
// ================================================================

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include <array>
#include <omp.h>


static constexpr int ND = 4;

// ---- SU(2) group elements stored as unit quaternions ----
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

SU2 random_su2(std::mt19937& rng) {
    std::normal_distribution<double> g(0.0, 1.0);
    SU2 U(g(rng), g(rng), g(rng), g(rng));
    reunitarize(U); 
    return U;
}

// ---- Lattice & RNG setup ----

// Maps a (base_seed, thread_id) pair to a well-separated 64-bit seed.
// Consecutive integers seed in MT19937 gives correlated initial states; 
// splitmix64 kills that correlation before MT is initialised.

uint64_t splitmix64_seed(uint64_t base, int thread_id) {
    uint64_t z = base + (uint64_t)thread_id * 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

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

// ---- Gauge update kernel ----

// Collects the 2(ND-1)=6 3-link paths ("staples") in touch with U_μ(s).
// The link action is  S_link = -(β/2) ReTr[U_μ(s)·K],  
// so K determines the local Boltzmann weight used by the heat bath.

SU2 compute_staple(const Lattice& lat, int s, int mu) {
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

SU2 su2_heatbath(const SU2& K, double beta, std::mt19937& rng) {
    
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
// NOTE: thread_rngs must be created once outside the sweep loop and passed
// by reference so each thread's state persists across calls.
void heatbath_sweep_parallel(Lattice& lat, double beta, std::vector<std::mt19937>& thread_rngs) {
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

// ---- Observables ----

// Average plaquette <P> = (1/N_plaq) Σ (1/2) ReTr[U_μν].
// For SU(2), (1/2) ReTr[U] = q[0] ∈ [-1,1].
// Ranges from 0 (strong coupling) to 1 (continuum limit).
double measure_plaquette(const Lattice& lat) {
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
double measure_polyakov(const Lattice& lat) {
    double sum=0.0;
    for(int x=0; x<lat.L[0]; x++) for(int y=0; y<lat.L[1]; y++) for(int z=0; z<lat.L[2]; z++){
        SU2 loop = lat.U(lat.site_idx(x,y,z,0), 3);
        for(int t=1; t<lat.L[3]; t++) loop = mul(loop, lat.U(lat.site_idx(x,y,z,t), 3));
        sum += loop.q[0];
    }
    return sum / double(lat.L[0]*lat.L[1]*lat.L[2]);
}

// ---- Jackknife estimators ----

struct Res { double mean, err; };

// Binned jackknife for a linear observable.
// N measurements → N_b = N/B bins of size B. Jackknife error:
//   σ_JK = √((N_b-1)/N_b · Σ_k (x̄_JK^k - x̄)²)
// Choose B ≳ 2τ_int to ensure bin means are approximately decorrelated.
Res jackknife(const std::vector<double>& data, int B) {
    
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
// All three use the SAME bin deletion so inter-observable correlations
// (e.g. between <L²> and <L⁴> in U4) are correctly propagated.
//
//   χ_L = V₃·(<L̄²> - <|L̄|>²)  — susceptibility; peaks at β_c
//   U4  = 1 - <L̄⁴>/(3<L̄²>²)   — Binder cumulant; sigmoid 0→2/3,
//         crosses at U4* ≈ 0.465 for different Ls (3D Ising universality)
//
// Note: χ_L is the VARIANCE form V₃·Var(|L̄|), NOT V₃·<L̄²>.
// The latter is monotone in β and does not peak — see Karsch (2002).
void jackknife_derived(const std::vector<double>& Lb, int B, double vol3, Res& L_abs, Res& chi, Res& U4) {
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

// ---- Parallel scaling benchmark ----

// Runs SWEEPS sweeps at a fixed β and records wall time for each thread count.
// Uses a fresh lattice and RNG pool so results are independent of the main scan.
void run_benchmark() {
    const int L = 12; const int Nt = 4; const int SWEEPS = 500; const double beta = 2.30;
    Lattice lat(L, L, L, Nt); 
    lat.cold_start();
    
    int max_threads = omp_get_max_threads();
    std::vector<std::mt19937> rngs(max_threads);
    for(int i=0; i<max_threads; i++) rngs[i].seed(splitmix64_seed(123456ULL, i));

    std::ofstream bfile("benchmark.csv");
    bfile << "Threads,Time,Speedup,Efficiency\n";
    
    double time_1_thread = 0.0;
    std::vector<int> thread_counts = {1, 2, 4, 6, 8, 10, 12, 14};

    for(int t : thread_counts) {
        if (t > max_threads) continue;
        omp_set_num_threads(t);
        double start_time = omp_get_wtime();
        for(int i = 0; i < SWEEPS; i++) heatbath_sweep_parallel(lat, beta, rngs);
        double time_taken = omp_get_wtime() - start_time;
        if (t == 1) time_1_thread = time_taken;
        
        double speedup = time_1_thread / time_taken;
        bfile << t << "," << time_taken << "," << speedup << "," << (speedup/t)*100.0 << "\n";
    }
}

// ---- Main driver ----

int main() {
    // Hardcoded thread count — replace with omp_get_max_threads() for portability.
    omp_set_num_threads(14); 
    const int LT=4;
    const int N_THERM=1000, N_MEAS=500, N_SKIP=5, JK_BIN_SIZE=20;
    const double BLO=2.10, BHI=2.60;  // narrow window centred on β_c ≈ 2.299
    const int NB=31; 
    double db=(BHI-BLO)/(NB-1);

    // One RNG pool shared across all volume scans.
    // The three volumes are not independently seeded, but they are statistically
    // independent because they run on separate lattice instances in sequence.
    // For perfectly reproducible per-volume results, seed inside the LS loop.
    int max_threads = omp_get_max_threads();
    std::vector<std::mt19937> rngs_cold(max_threads);
    for(int i=0; i<max_threads; i++) rngs_cold[i].seed(splitmix64_seed(123456789ULL, i));

    // Finite-size scaling: scan three volumes and look for Binder crossing at β_c.
    std::vector<int> volumes = {8, 12, 16};

    for (int LS : volumes) {
        std::string fname = "su2_data_L" + std::to_string(LS) + ".csv";
        std::ofstream file(fname);
        file << "beta,P_c,P_c_err,L_c,L_c_err,chi_c,chi_c_err,U4_c,U4_c_err\n";

        std::cout << "\nStarting Scan on " << LS << "^3 x " << LT << "...\n";
        
        for(int ib=0; ib<NB; ib++){
            double beta = BLO + ib*db;
            
            Lattice lat_c(LS,LS,LS,LT);
            lat_c.cold_start(); 
            for(int sw=0; sw<N_THERM; sw++) heatbath_sweep_parallel(lat_c, beta, rngs_cold);

            std::vector<double> P_ts_c(N_MEAS), L_ts_c(N_MEAS);
            for(int m=0; m<N_MEAS; m++){
                for(int sk=0; sk<N_SKIP; sk++) heatbath_sweep_parallel(lat_c, beta, rngs_cold);
                P_ts_c[m] = measure_plaquette(lat_c);
                L_ts_c[m] = measure_polyakov(lat_c);  // signed L̄, not |L̄|
            }

            Res P_c = jackknife(P_ts_c, JK_BIN_SIZE);
            Res L_c, chi_c, U4_c;
            jackknife_derived(L_ts_c, JK_BIN_SIZE, double(LS*LS*LS), L_c, chi_c, U4_c);

            file << beta << "," << P_c.mean << "," << P_c.err << "," << L_c.mean << "," << L_c.err << "," 
                 << chi_c.mean << "," << chi_c.err << "," << U4_c.mean << "," << U4_c.err << "\n";
                 
            std::cout << "\r  Beta: " << std::fixed << std::setprecision(3) << beta 
                      << " | " << std::setw(3) << int(100.0*(ib+1)/NB) << "% complete" << std::flush;
        }
    }
    std::cout << "\n\nRunning benchmark...\n";
    run_benchmark();
    return 0;
}