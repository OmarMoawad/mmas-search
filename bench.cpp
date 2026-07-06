// Honest sorted-array search benchmark.
// Compares, in ONE compiled language, on IN-RAM int64 arrays, with MIXED hit/miss queries:
//   stl_lb      : std::lower_bound (optimized branchless binary search) -- timing gold standard
//   binary      : textbook binary search (probe-counted)
//   interp      : interpolation search (probe-counted)
//   mmas_closed : 5-sample linear/power/exp detect + direct-inverse + capped IS residual
//   pgm_pla     : piecewise-linear (GreedyPLR, bounded-error) model + windowed binary search
// Reports ns/query (min over repeats), mean/max DATA probes, and correctness.

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <array>

using std::vector;
using i64 = long long;

static vector<i64> load_i64(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); return {}; }
    i64 n = 0; fread(&n, sizeof(i64), 1, f);
    vector<i64> v(n);
    if (n) fread(v.data(), sizeof(i64), n, f);
    fclose(f);
    return v;
}

// ---------------- counted searches ----------------
struct Res { i64 idx; i64 probes; };

static Res binary_counted_range(const vector<i64>& A, i64 v, i64 lo, i64 hi) {
    i64 probes = 0;
    while (lo <= hi) {
        i64 mid = lo + (hi - lo) / 2;
        probes++;
        if (A[mid] == v) return {mid, probes};
        if (A[mid] < v) lo = mid + 1; else hi = mid - 1;
    }
    return {-1, probes};
}

static Res binary_counted(const vector<i64>& A, i64 v) {
    i64 lo = 0, hi = (i64)A.size() - 1, probes = 0;
    while (lo <= hi) {
        i64 mid = lo + (hi - lo) / 2;
        probes++;
        if (A[mid] == v) return {mid, probes};
        if (A[mid] < v) lo = mid + 1; else hi = mid - 1;
    }
    return {-1, probes};
}

static Res interp_counted(const vector<i64>& A, i64 v) {
    i64 lo = 0, hi = (i64)A.size() - 1, probes = 0;
    while (lo <= hi && A[lo] <= v && v <= A[hi]) {
        if (A[hi] == A[lo]) { probes++; return {A[lo] == v ? lo : -1, probes}; }
        // 128-bit-safe interpolation
        long double frac = (long double)(v - A[lo]) / (long double)(A[hi] - A[lo]);
        i64 p = lo + (i64)(frac * (long double)(hi - lo));
        if (p < lo) p = lo; if (p > hi) p = hi;
        probes++;
        if (A[p] == v) return {p, probes};
        if (A[p] < v) lo = p + 1; else hi = p - 1;
    }
    return {-1, probes};
}

// IS restricted to [lo,hi] with a hard probe cap (binary fallback) -> O(log n) worst case
static Res is_bracket_capped(const vector<i64>& A, i64 v, i64 lo, i64 hi) {
    i64 probes = 0;
    if (lo > hi) return {-1, 0};
    i64 cap = 2 * ((i64)std::ceil(std::log2((double)(hi - lo + 2))) + 1);
    i64 is_used = 0;
    while (lo <= hi) {
        if (A[lo] > v || A[hi] < v) return {-1, probes};
        if (A[hi] == A[lo]) { probes++; return {A[lo] == v ? lo : -1, probes}; }
        i64 p;
        if (is_used < cap / 2) {
            long double frac = (long double)(v - A[lo]) / (long double)(A[hi] - A[lo]);
            p = lo + (i64)(frac * (long double)(hi - lo));
            if (p < lo) p = lo; if (p > hi) p = hi;
            is_used++;
        } else {
            p = lo + (hi - lo) / 2;
        }
        probes++;
        if (A[p] == v) return {p, probes};
        if (A[p] < v) lo = p + 1; else hi = p - 1;
    }
    return {-1, probes};
}

// ---------------- MMAS closed-form model ----------------
struct Model {
    int kind; // 0 binary, 1 linear, 2 power, 3 exponential
    double a, b, c, p, r, a0; // params per kind
};

static double relerr_max(const vector<double>& pred, const vector<double>& act) {
    double span = *std::max_element(act.begin(), act.end()) - *std::min_element(act.begin(), act.end());
    if (span == 0) return 1e18;
    double e = 0;
    for (size_t i = 0; i < pred.size(); i++) e = std::max(e, std::fabs(pred[i] - act[i]));
    return e / span;
}

static Model detect_closed(const vector<i64>& A) {
    Model m{0,0,0,0,0,0,0};
    i64 n = A.size();
    if (n < 5) return m;
    i64 pos[5] = {0, n/4, n/2, (3*n)/4, n-1};
    vector<double> xs(5), ys(5);
    for (int i = 0; i < 5; i++) { xs[i] = (double)pos[i]; ys[i] = (double)A[pos[i]]; }
    double a0 = ys[0];
    if (ys[4] <= ys[0]) return m;
    const double TH = 0.10;
    struct Cand { int kind; double err; Model mm; };
    vector<Cand> cands;

    // linear: y = a*x + b
    {
        double mx=0,my=0; for(int i=0;i<5;i++){mx+=xs[i];my+=ys[i];} mx/=5;my/=5;
        double cov=0,var=0; for(int i=0;i<5;i++){cov+=(xs[i]-mx)*(ys[i]-my);var+=(xs[i]-mx)*(xs[i]-mx);}
        if (var>0){ double a=cov/var,b=my-a*mx; if(a>0){ vector<double>pr(5); for(int i=0;i<5;i++)pr[i]=a*xs[i]+b;
            double e=relerr_max(pr,ys); if(e<TH){Model mm{1,a,b,0,0,0,0}; cands.push_back({1,e,mm});}}}
    }
    // power: y - a0 = c * x^p  (log-log, skip x=0)
    {
        vector<double> lx,ly; for(int i=0;i<5;i++){ if(xs[i]>0 && ys[i]-a0>0){lx.push_back(std::log(xs[i]));ly.push_back(std::log(ys[i]-a0));}}
        if(lx.size()>=3){ double mx=0,my=0; for(size_t i=0;i<lx.size();i++){mx+=lx[i];my+=ly[i];} mx/=lx.size();my/=lx.size();
            double cov=0,var=0; for(size_t i=0;i<lx.size();i++){cov+=(lx[i]-mx)*(ly[i]-my);var+=(lx[i]-mx)*(lx[i]-mx);}
            if(var>0){ double p=cov/var; if(p>=0.3&&p<=8){ double c=std::exp(my-p*mx);
                vector<double>pr(5),ac(5); for(int i=0;i<5;i++){pr[i]=c*std::pow(xs[i],p)+a0; ac[i]=ys[i];}
                double e=relerr_max(pr,ac); if(e<TH){Model mm{2,0,0,c,p,0,a0}; cands.push_back({2,e,mm});}}}}
    }
    // exponential: y = c * r^x
    {
        bool ok=true; for(int i=0;i<5;i++) if(ys[i]<=0) ok=false;
        if(ok){ vector<double> ly(5); for(int i=0;i<5;i++) ly[i]=std::log(ys[i]);
            double mx=0,my=0; for(int i=0;i<5;i++){mx+=xs[i];my+=ly[i];} mx/=5;my/=5;
            double cov=0,var=0; for(int i=0;i<5;i++){cov+=(xs[i]-mx)*(ly[i]-my);var+=(xs[i]-mx)*(xs[i]-mx);}
            if(var>0){ double lr=cov/var; if(lr*(xs[4]-xs[0])>=0.01){ double r=std::exp(lr),c=std::exp(my-lr*mx);
                vector<double>pr(5); for(int i=0;i<5;i++)pr[i]=c*std::pow(r,xs[i]);
                double e=relerr_max(pr,ys); if(e<TH){Model mm{3,0,0,c,0,r,0}; cands.push_back({3,e,mm});}}}}
    }
    if(cands.empty()) return m;
    std::sort(cands.begin(),cands.end(),[](const Cand&a,const Cand&b){return a.err<b.err;});
    return cands[0].mm;
}

static Res mmas_closed_search(const vector<i64>& A, i64 v, const Model& m) {
    i64 n = A.size();
    if (m.kind == 0 || v < A[0] || v > A[n-1]) return binary_counted(A, v);
    if (m.kind == 1) return is_bracket_capped(A, v, 0, n-1);
    i64 ipred;
    if (m.kind == 2) {
        double vt = (double)v - m.a0;
        if (vt <= 0) ipred = 0;
        else ipred = (i64)llround(std::pow(vt / m.c, 1.0 / m.p));
    } else { // exponential
        if (v <= 0 || m.c <= 0 || m.r <= 1) ipred = n/2;
        else ipred = (i64)llround(std::log((double)v / m.c) / std::log(m.r));
    }
    if (ipred < 0) ipred = 0; if (ipred > n-1) ipred = n-1;
    i64 probes = 1;
    if (A[ipred] == v) return {ipred, probes};
    Res r = (A[ipred] < v) ? is_bracket_capped(A, v, ipred+1, n-1)
                           : is_bracket_capped(A, v, 0, ipred-1);
    return {r.idx, probes + r.probes};
}

// ---------------- PGM-style piecewise linear (GreedyPLR, bounded error) ----------------
struct Seg { i64 x0; double slope; i64 y0; };
struct PLA { vector<Seg> segs; vector<i64> keys; i64 eps; };

static PLA build_pla(const vector<i64>& A, i64 eps) {
    PLA pla; pla.eps = eps;
    i64 n = A.size(), i = 0;
    while (i < n) {
        i64 x0 = A[i], y0 = i;
        double slo = -1e300, shi = 1e300;
        i64 j = i + 1;
        for (; j < n; j++) {
            double dx = (double)(A[j] - x0);
            double lo = ((double)(j - y0) - eps) / dx;
            double hi = ((double)(j - y0) + eps) / dx;
            double nlo = std::max(slo, lo), nhi = std::min(shi, hi);
            if (nlo > nhi) break;
            slo = nlo; shi = nhi;
        }
        double slope = (slo < -1e299 || shi > 1e299) ? 0.0 : 0.5 * (slo + shi);
        pla.segs.push_back({x0, slope, y0});
        pla.keys.push_back(x0);
        i = j;
    }
    return pla;
}

// returns {idx, data_probes}; model_probes reported via out param
static Res pla_search(const vector<i64>& A, i64 v, const PLA& pla, i64& model_probes) {
    i64 n = A.size();
    // locate segment: largest keys[s] <= v  (binary search over segment anchors)
    i64 lo = 0, hi = (i64)pla.keys.size() - 1, s = 0, mp = 0;
    while (lo <= hi) { i64 mid = lo + (hi-lo)/2; mp++; if (pla.keys[mid] <= v) { s = mid; lo = mid+1; } else hi = mid-1; }
    model_probes += mp;
    const Seg& sg = pla.segs[s];
    i64 pred = sg.y0 + (i64)llround(sg.slope * (double)(v - sg.x0));
    i64 w = pla.eps + 1;
    i64 a = pred - w, b = pred + w;
    if (a < 0) a = 0; if (b > n-1) b = n-1;
    return binary_counted_range(A, v, a, b);
}

// ---------------- benchmark harness ----------------
template <class F>
static double time_ns_per_query(const vector<i64>& Q, F&& fn, int repeats, volatile i64& sink) {
    double best = 1e300;
    for (int r = 0; r < repeats; r++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        i64 acc = 0;
        for (i64 v : Q) acc += fn(v);
        auto t1 = std::chrono::high_resolution_clock::now();
        sink += acc;
        double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / (double)Q.size();
        best = std::min(best, ns);
    }
    return best;
}

int main(int argc, char** argv) {
    std::string dir = argc > 1 ? argv[1] : "data";
    int repeats = argc > 2 ? atoi(argv[2]) : 5;
    i64 eps = argc > 3 ? atoll(argv[3]) : 32;

    FILE* mf = fopen((dir + "/manifest.txt").c_str(), "r");
    if (!mf) { fprintf(stderr, "no manifest in %s\n", dir.c_str()); return 1; }
    char name[256];
    volatile i64 sink = 0;

    printf("dataset,n,algo,ns_per_query,mean_data_probes,max_data_probes,mean_model_probes,errors,detected\n");
    while (fscanf(mf, "%255s", name) == 1) {
        std::string base = dir + "/" + name;
        vector<i64> A = load_i64(base + ".bin");
        vector<i64> Q = load_i64(base + ".qry");
        if (A.empty() || Q.empty()) continue;
        // Cap queries: uncapped interpolation search is O(n)/query on pathological
        // data (exponential, adversarial); 20k mixed queries give stable stats.
        const size_t MAXQ = 4000;
        if (Q.size() > MAXQ) Q.resize(MAXQ);
        i64 n = A.size();

        // ground truth found-count
        i64 hits = 0; for (i64 v : Q) if (std::binary_search(A.begin(), A.end(), v)) hits++;

        Model model = detect_closed(A);
        const char* mname = (model.kind==0?"binary":model.kind==1?"linear":model.kind==2?"power":"exponential");
        PLA pla = build_pla(A, eps);

        // correctness + probe stats (single pass)
        auto verify = [&](auto search) {
            i64 err=0, sump=0, maxp=0, summ=0;
            for (i64 v : Q) {
                i64 mp = 0; Res r = search(v, mp);
                bool truth = std::binary_search(A.begin(), A.end(), v);
                bool got = (r.idx >= 0);
                if (got != truth || (got && A[r.idx] != v)) err++;
                sump += r.probes; if (r.probes>maxp) maxp=r.probes; summ += mp;
            }
            return std::array<double,4>{ (double)err, (double)sump/Q.size(), (double)maxp, (double)summ/Q.size() };
        };

        auto st_bin  = verify([&](i64 v, i64& mp){ return binary_counted(A, v); });
        auto st_int  = verify([&](i64 v, i64& mp){ return interp_counted(A, v); });
        auto st_mma  = verify([&](i64 v, i64& mp){ return mmas_closed_search(A, v, model); });
        auto st_pla  = verify([&](i64 v, i64& mp){ return pla_search(A, v, pla, mp); });

        // timing (return index as i64 to keep work observable)
        double t_lb  = time_ns_per_query(Q, [&](i64 v){ auto it=std::lower_bound(A.begin(),A.end(),v); return (i64)(it!=A.end() && *it==v ? it-A.begin() : -1); }, repeats, sink);
        double t_bin = time_ns_per_query(Q, [&](i64 v){ return binary_counted(A, v).idx; }, repeats, sink);
        double t_int = time_ns_per_query(Q, [&](i64 v){ return interp_counted(A, v).idx; }, repeats, sink);
        double t_mma = time_ns_per_query(Q, [&](i64 v){ return mmas_closed_search(A, v, model).idx; }, repeats, sink);
        double t_pla = time_ns_per_query(Q, [&](i64 v){ i64 mp=0; return pla_search(A, v, pla, mp).idx; }, repeats, sink);

        printf("%s,%lld,stl_lb,%.2f,,,,,\n", name, n, t_lb);
        printf("%s,%lld,binary,%.2f,%.2f,%.0f,%.2f,%.0f,\n",      name,n,t_bin,st_bin[1],st_bin[2],st_bin[3],st_bin[0]);
        printf("%s,%lld,interp,%.2f,%.2f,%.0f,%.2f,%.0f,\n",      name,n,t_int,st_int[1],st_int[2],st_int[3],st_int[0]);
        printf("%s,%lld,mmas_closed,%.2f,%.2f,%.0f,%.2f,%.0f,%s\n",name,n,t_mma,st_mma[1],st_mma[2],st_mma[3],st_mma[0],mname);
        printf("%s,%lld,pgm_pla,%.2f,%.2f,%.0f,%.2f,%.0f,segs=%zu\n",name,n,t_pla,st_pla[1],st_pla[2],st_pla[3],st_pla[0],pla.segs.size());
        fflush(stdout);
        fprintf(stderr, "[%s] n=%lld hits=%lld/%zu detect=%s segs=%zu\n", name, n, hits, Q.size(), mname, pla.segs.size());
    }
    fclose(mf);
    fprintf(stderr, "sink=%lld\n", (i64)sink);
    return 0;
}

