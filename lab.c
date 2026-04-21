#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <string.h>
#include <x86intrin.h>
#include <immintrin.h>

#define FLOAT_BIAS 127
#define FLOAT_MANTISSA_LENGTH 23

#define X_MIN (-44.85347f)
#define X_MAX ( 38.53184f)

#define SHIFTER 12582912.0f /* 0x1.8p23 */

#define LOG2_10      3.32192809488736234787f

/* V1+ : split log2(10) */
#define LOG2_10_HI   3.32177734375f
#define LOG2_10_LO   0.00015075113736234787f

/* Taylor degree 10 for 2^r */
#define T10_C0  1.0f
#define T10_C1  0.6931471805599453f
#define T10_C2  0.2402265069591007f
#define T10_C3  0.0555041086648216f
#define T10_C4  0.0096181291076285f
#define T10_C5  0.0013333558146428f
#define T10_C6  0.0001540353039338f
#define T10_C7  0.0000152527338050f
#define T10_C8  0.0000013215486790f
#define T10_C9  0.0000001017177780f
#define T10_C10 0.0000000070526790f

/* Taylor degree 6 for 2^r */
#define T6_C0  1.0f
#define T6_C1  0.6931471805599453f
#define T6_C2  0.2402265069591007f
#define T6_C3  0.0555041086648216f
#define T6_C4  0.0096181291076285f
#define T6_C5  0.0013333558146428f
#define T6_C6  0.0001540353039338f

/* V2/V3/V4/V5: degree-5 hand-tuned / minimax-style fit for 2^r on [-0.5, 0.5] */
#define M5_C0  1.00000005f
#define M5_C1  0.69314720f
#define M5_C2  0.24022212f
#define M5_C3  0.05550341f
#define M5_C4  0.00967076f
#define M5_C5  0.00133953f

/* V6: tiny poly for 2^r on very small interval [-1/64, 1/64] */
#define S2_C0  1.0f
#define S2_C1  0.6931471805599453f
#define S2_C2  0.2402265069591007f

#define TABLE_BITS 5
#define TABLE_SIZE (1 << TABLE_BITS)
#define TABLE_SCALE ((float)TABLE_SIZE)
#define INV_TABLE_SCALE (1.0f / (float)TABLE_SIZE)

static float g_exp2_table[TABLE_SIZE];

typedef float (*scalar_fn_t)(float);

/* ---------- helpers ---------- */

static inline float F32(uint32_t x) {
    union {
        uint32_t u;
        float f;
    } v = { x };
    return v.f;
}

static inline uint32_t U32(float x) {
    union {
        float f;
        uint32_t u;
    } v = { x };
    return v.u;
}

static inline float pow2i_reconstruct(int32_t n) {
    if (n > 127) return INFINITY;

    if (n >= -126) {
        uint32_t bits = (uint32_t)(n + FLOAT_BIAS) << FLOAT_MANTISSA_LENGTH;
        return F32(bits);
    }

    if (n >= -149) {
        /* subnormal: 2^-149 corresponds to bit 0 */
        uint32_t bits = 1u << (uint32_t)(n + 149);
        return F32(bits);
    }

    return 0.0f;
}

static inline float ref_exp10f(float x) {
    return (float)powl(10.0L, (long double)x);
}

static inline int32_t float_to_ordered_int(float x) {
    union {
        float f;
        int32_t i;
    } v = { x };

    return (v.i < 0) ? (0x80000000 - v.i) : v.i;
}

static unsigned ulp_distance_float(float a, float b) {
    if (isnan(a) || isnan(b)) return UINT_MAX;
    if (isinf(a) || isinf(b)) return (a == b) ? 0u : UINT_MAX;

    int32_t ia = float_to_ordered_int(a);
    int32_t ib = float_to_ordered_int(b);

    long long diff = (long long)ia - (long long)ib;
    if (diff < 0) diff = -diff;

    return (unsigned)diff;
}

static inline float round_shifter(float x) {
    float s = copysignf(SHIFTER, x);
    float y = x + s;
    y = y - s;
    return y;
}

static inline uint64_t start_tsc(void) {
    return __rdtsc();
}

static inline uint64_t stop_tsc(void) {
    unsigned aux;
    return __rdtscp(&aux);
}

static void init_table(void) {
    for (int i = 0; i < TABLE_SIZE; ++i) {
        g_exp2_table[i] = exp2f((float)i * INV_TABLE_SCALE);
    }
}

/* ---------- V0 ---------- */

__attribute__((noinline))
float exp10_v0(float x) {
    if (isnan(x)) return NAN;
    if (isinf(x)) return x > 0 ? INFINITY : 0.0f;
    if (x > X_MAX) return INFINITY;
    if (x < X_MIN) return 0.0f;

    float y = x * LOG2_10;
    float n = roundf(y);
    float r = y - n;

    float poly = fmaf(T10_C10, r, T10_C9);
    poly = fmaf(poly, r, T10_C8);
    poly = fmaf(poly, r, T10_C7);
    poly = fmaf(poly, r, T10_C6);
    poly = fmaf(poly, r, T10_C5);
    poly = fmaf(poly, r, T10_C4);
    poly = fmaf(poly, r, T10_C3);
    poly = fmaf(poly, r, T10_C2);
    poly = fmaf(poly, r, T10_C1);
    poly = fmaf(poly, r, T10_C0);

    return pow2i_reconstruct((int32_t)n) * poly;
}

/* ---------- V1 ---------- */

__attribute__((noinline))
float exp10_v1(float x) {
    if (isnan(x)) return NAN;
    if (isinf(x)) return x > 0 ? INFINITY : 0.0f;
    if (x > X_MAX) return INFINITY;
    if (x < X_MIN) return 0.0f;

    float y = fmaf(x, LOG2_10_HI, x * LOG2_10_LO);
    float n = roundf(y);
    float r = y - n;

    float poly = fmaf(T6_C6, r, T6_C5);
    poly = fmaf(poly, r, T6_C4);
    poly = fmaf(poly, r, T6_C3);
    poly = fmaf(poly, r, T6_C2);
    poly = fmaf(poly, r, T6_C1);
    poly = fmaf(poly, r, T6_C0);

    return pow2i_reconstruct((int32_t)n) * poly;
}

/* ---------- V2 ---------- */

__attribute__((noinline))
float exp10_v2(float x) {
    if (isnan(x)) return NAN;
    if (isinf(x)) return x > 0 ? INFINITY : 0.0f;
    if (x > X_MAX) return INFINITY;
    if (x < X_MIN) return 0.0f;

    float y = fmaf(x, LOG2_10_HI, x * LOG2_10_LO);
    float n = roundf(y);
    float r = y - n;

    float poly = fmaf(M5_C5, r, M5_C4);
    poly = fmaf(poly, r, M5_C3);
    poly = fmaf(poly, r, M5_C2);
    poly = fmaf(poly, r, M5_C1);
    poly = fmaf(poly, r, M5_C0);

    return pow2i_reconstruct((int32_t)n) * poly;
}

/* ---------- V3 ---------- */

__attribute__((noinline))
float exp10_v3(float x) {
    if (isnan(x)) return NAN;
    if (isinf(x)) return x > 0 ? INFINITY : 0.0f;
    if (x > X_MAX) return INFINITY;
    if (x < X_MIN) return 0.0f;

    float y = fmaf(x, LOG2_10_HI, x * LOG2_10_LO);
    float n = roundf(y);
    float r = y - n;

    float r2 = r * r;
    float t54 = fmaf(M5_C5, r, M5_C4);
    float t32 = fmaf(M5_C3, r, M5_C2);
    float t10 = fmaf(M5_C1, r, M5_C0);
    float poly = fmaf(t54, r2, t32);
    poly = fmaf(poly, r2, t10);

    return pow2i_reconstruct((int32_t)n) * poly;
}

/* ---------- V4 ---------- */

__attribute__((noinline))
float exp10_v4(float x) {
    if (isnan(x)) return NAN;
    if (isinf(x)) return x > 0 ? INFINITY : 0.0f;
    if (x > X_MAX) return INFINITY;
    if (x < X_MIN) return 0.0f;

    float y = fmaf(x, LOG2_10_HI, x * LOG2_10_LO);
    float n = round_shifter(y);
    float r = y - n;

    float poly = fmaf(M5_C5, r, M5_C4);
    poly = fmaf(poly, r, M5_C3);
    poly = fmaf(poly, r, M5_C2);
    poly = fmaf(poly, r, M5_C1);
    poly = fmaf(poly, r, M5_C0);

    return pow2i_reconstruct((int32_t)n) * poly;
}

/* ---------- V5 ---------- */

__attribute__((noinline))
float exp10_v5(float x) {
    if (isnan(x)) return NAN;
    if (isinf(x)) return x > 0 ? INFINITY : 0.0f;
    if (x > X_MAX) return INFINITY;
    if (x < X_MIN) return 0.0f;

    float y = x * LOG2_10_HI + x * LOG2_10_LO;
    float n = round_shifter(y);
    float r = y - n;

    float poly = M5_C5 * r + M5_C4;
    poly = poly * r + M5_C3;
    poly = poly * r + M5_C2;
    poly = poly * r + M5_C1;
    poly = poly * r + M5_C0;

    return pow2i_reconstruct((int32_t)n) * poly;
}

/* ---------- V6 ---------- */

__attribute__((noinline))
float exp10_v6(float x) {
    if (isnan(x)) return NAN;
    if (isinf(x)) return x > 0 ? INFINITY : 0.0f;
    if (x > X_MAX) return INFINITY;
    if (x < X_MIN) return 0.0f;

    float y = fmaf(x, LOG2_10_HI, x * LOG2_10_LO);

    float kf = roundf(y * TABLE_SCALE);
    int32_t k = (int32_t)kf;

    int32_t n = (int32_t)floorf((float)k * INV_TABLE_SCALE);
    int32_t i = k - (n << TABLE_BITS);
    if (i < 0) {
        i += TABLE_SIZE;
        n -= 1;
    }

    float r = y - kf * INV_TABLE_SCALE;

    float poly = fmaf(S2_C2, r, S2_C1);
    poly = fmaf(poly, r, S2_C0);

    float base = pow2i_reconstruct(n);
    float table_val = g_exp2_table[i];

    return base * table_val * poly;
}

/* ---------- V7 AVX2 kernel ---------- */

#if defined(__AVX2__) && defined(__FMA__)

static inline __m256 poly_m5_avx2(__m256 r) {
    const __m256 c0 = _mm256_set1_ps(M5_C0);
    const __m256 c1 = _mm256_set1_ps(M5_C1);
    const __m256 c2 = _mm256_set1_ps(M5_C2);
    const __m256 c3 = _mm256_set1_ps(M5_C3);
    const __m256 c4 = _mm256_set1_ps(M5_C4);
    const __m256 c5 = _mm256_set1_ps(M5_C5);

    __m256 p = _mm256_fmadd_ps(c5, r, c4);
    p = _mm256_fmadd_ps(p, r, c3);
    p = _mm256_fmadd_ps(p, r, c2);
    p = _mm256_fmadd_ps(p, r, c1);
    p = _mm256_fmadd_ps(p, r, c0);
    return p;
}

void exp10_v7_avx2_kernel(const float *x, float *out, size_t n) {
    const __m256 c_log2_10 = _mm256_set1_ps(LOG2_10);
    const __m256i c_bias_i = _mm256_set1_epi32(FLOAT_BIAS);

    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);

        __m256 y = _mm256_mul_ps(vx, c_log2_10);
        __m256 nps = _mm256_round_ps(y, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256 r = _mm256_sub_ps(y, nps);

        __m256 poly = poly_m5_avx2(r);

        __m256i ni = _mm256_cvtps_epi32(nps);
        __m256i ei = _mm256_add_epi32(ni, c_bias_i);
        __m256i bits = _mm256_slli_epi32(ei, FLOAT_MANTISSA_LENGTH);
        __m256 two_pow_n = _mm256_castsi256_ps(bits);

        __m256 res = _mm256_mul_ps(two_pow_n, poly);
        _mm256_storeu_ps(out + i, res);
    }

    for (; i < n; ++i) {
        out[i] = exp10_v5(x[i]);
    }
}
#endif

/* ---------- benchmarking ---------- */

typedef struct {
    const char *version;
    const char *optimization;
    scalar_fn_t fn;
} row_desc_t;

typedef struct {
    unsigned max_ulp;
    double avg_ulp;
    float worst_x;
    double latency_cpe;
    double throughput_cpe;
    int skipped;
} row_result_t;

static row_result_t measure_scalar_row(scalar_fn_t fn, int accuracy_samples, int bench_n, int repeats) {
    row_result_t rr;
    memset(&rr, 0, sizeof(rr));

    /* accuracy */
    {
        unsigned max_ulp = 0;
        double sum_ulp = 0.0;
        unsigned valid = 0;
        float worst_x = 0.0f;

        for (int i = 0; i < accuracy_samples; ++i) {
            float t = (float)i / (float)(accuracy_samples - 1);
            float x = X_MIN + (X_MAX - X_MIN) * t;

            float my = fn(x);
            float ref = ref_exp10f(x);
            unsigned ulp = ulp_distance_float(my, ref);

            if (ulp == UINT_MAX) continue;

            sum_ulp += (double)ulp;
            valid++;

            if (ulp > max_ulp) {
                max_ulp = ulp;
                worst_x = x;
            }
        }

        rr.max_ulp = max_ulp;
        rr.avg_ulp = (valid > 0) ? (sum_ulp / (double)valid) : 0.0;
        rr.worst_x = worst_x;
    }

    /* latency: dependent chain */
    {
        double best = 1e100;

        static const float base_args[16] = {
            -3.0f, -2.5f, -2.0f, -1.5f,
            -1.0f, -0.5f,  0.0f,  0.5f,
             1.0f,  1.5f,  2.0f,  2.5f,
             3.0f, -2.2f,  1.3f, -0.8f
        };

        volatile float prev = 0.12345f;

        for (int warm = 0; warm < 10000; ++warm) {
            float arg = base_args[warm & 15] + 0x1p-20f * prev;
            prev = fn(arg);
        }

        for (int rep = 0; rep < repeats; ++rep) {
            uint64_t t0 = start_tsc();

            for (int i = 0; i < bench_n; ++i) {
                float arg = base_args[i & 15] + 0x1p-20f * prev;
                prev = fn(arg);
            }

            uint64_t t1 = stop_tsc();

            double cpe = (double)(t1 - t0) / (double)bench_n;
            if (cpe < best) best = cpe;
        }

        rr.latency_cpe = best;

        if (prev == -999.0f) {
            printf("ignore: %f\n", prev);
        }
    }

    /* throughput: independent calls, measured but not shown for scalar rows */
    {
        double best = 1e100;

        static float args[4] = {0.10f, 0.20f, 0.30f, 0.40f};
        volatile float r0 = 0.0f;
        volatile float r1 = 0.0f;
        volatile float r2 = 0.0f;
        volatile float r3 = 0.0f;

        for (int warm = 0; warm < 10000; ++warm) {
            r0 = fn(args[0]);
            r1 = fn(args[1]);
            r2 = fn(args[2]);
            r3 = fn(args[3]);
        }

        for (int rep = 0; rep < repeats; ++rep) {
            uint64_t t0 = start_tsc();

            for (int i = 0; i < bench_n; ++i) {
                r0 = fn(args[0]);
                r1 = fn(args[1]);
                r2 = fn(args[2]);
                r3 = fn(args[3]);
            }

            uint64_t t1 = stop_tsc();

            double cpe = (double)(t1 - t0) / (double)(4 * bench_n);
            if (cpe < best) best = cpe;
        }

        rr.throughput_cpe = best;

        if ((r0 + r1 + r2 + r3) == -999.0f) {
            printf("ignore: %f\n", r0);
        }
    }

    return rr;
}

#if defined(__AVX2__) && defined(__FMA__)
static row_result_t measure_v7_avx2(int bench_n, int repeats) {
    row_result_t rr;
    memset(&rr, 0, sizeof(rr));

    enum { N = 1 << 15 };
    static float in[N];
    static float out[N];

    for (int i = 0; i < N; ++i) {
        float t = (float)i / (float)(N - 1);
        in[i] = -5.0f + 10.0f * t;
    }

    /* accuracy: compare AVX2 output to scalar V5 */
    {
        exp10_v7_avx2_kernel(in, out, N);

        unsigned max_ulp = 0;
        double sum_ulp = 0.0;

        for (int i = 0; i < N; ++i) {
            float ref = exp10_v5(in[i]);
            unsigned ulp = ulp_distance_float(out[i], ref);
            if (ulp != UINT_MAX) {
                sum_ulp += (double)ulp;
                if (ulp > max_ulp) max_ulp = ulp;
            }
        }

        rr.max_ulp = max_ulp;
        rr.avg_ulp = sum_ulp / (double)N;
        rr.worst_x = 0.0f;
    }

    /* throughput only */
    {
        double best = 1e100;

        for (int warm = 0; warm < 50; ++warm) {
            exp10_v7_avx2_kernel(in, out, N);
        }

        for (int rep = 0; rep < repeats; ++rep) {
            uint64_t t0 = start_tsc();

            for (int k = 0; k < bench_n; ++k) {
                exp10_v7_avx2_kernel(in, out, N);
            }

            uint64_t t1 = stop_tsc();

            double total_elems = (double)bench_n * (double)N;
            double cpe = (double)(t1 - t0) / total_elems;
            if (cpe < best) best = cpe;
        }

        rr.throughput_cpe = best;
        rr.latency_cpe = 0.0;
    }

    return rr;
}
#endif

static void print_header(void) {
    printf("\n");
    printf("---------------------------------------------------------------------------------------------------------------\n");
    printf("| %-7s | %-54s | %-12s | %-12s | %-22s |\n",
           "version", "optimizations made", "throughput", "latency", "accuracy");
    printf("---------------------------------------------------------------------------------------------------------------\n");
}

static void print_row(const char *version, const char *opt, const row_result_t *rr, int has_latency, int has_throughput) {
    char thr[32], lat[32], acc[64];

    if (rr->skipped) {
        snprintf(thr, sizeof(thr), "skipped");
        snprintf(lat, sizeof(lat), "skipped");
        snprintf(acc, sizeof(acc), "skipped");
    } else {
        if (has_throughput) snprintf(thr, sizeof(thr), "%.2f cpe", rr->throughput_cpe);
        else snprintf(thr, sizeof(thr), "-");

        if (has_latency) snprintf(lat, sizeof(lat), "%.2f cpe", rr->latency_cpe);
        else snprintf(lat, sizeof(lat), "-");

        snprintf(acc, sizeof(acc), "max=%u, avg=%.2f", rr->max_ulp, rr->avg_ulp);
    }

    printf("| %-7s | %-54s | %-12s | %-12s | %-22s |\n",
           version, opt, thr, lat, acc);
}

static void print_footer(void) {
    printf("---------------------------------------------------------------------------------------------------------------\n");
    printf("\n");
}

int main(void) {
    init_table();

    const int ACCURACY_SAMPLES = 50000;
    const int BENCH_N = 20000;
    const int REPEATS = 7;

    row_desc_t rows[] = {
        {"V0", "nothing",                                exp10_v0},
        {"V1", "2-part LOG2_10 + shorter Taylor",        exp10_v1},
        {"V2", "degree-5 minimax-style coeffs",          exp10_v2},
        {"V3", "Estrin scheme",                          exp10_v3},
        {"V4", "shifter instead of roundf",              exp10_v4},
        {"V5", "no explicit fmaf in source",             exp10_v5},
        {"V6", "table of 2^(i/32) + tiny residual poly", exp10_v6},
    };

    const int row_count = (int)(sizeof(rows) / sizeof(rows[0]));

    print_header();
    fflush(stdout);

    for (int i = 0; i < row_count; ++i) {
        row_result_t rr = measure_scalar_row(rows[i].fn, ACCURACY_SAMPLES, BENCH_N, REPEATS);
        print_row(rows[i].version, rows[i].optimization, &rr, 1, 0);
        fflush(stdout);
    }

#if defined(__AVX2__) && defined(__FMA__)
    {
        row_result_t rr = measure_v7_avx2(200, REPEATS);
        print_row("V7", "AVX2 vector kernel (throughput-oriented)", &rr, 0, 1);
        fflush(stdout);
    }
#else
    {
        row_result_t rr;
        memset(&rr, 0, sizeof(rr));
        rr.skipped = 1;
        print_row("V7", "AVX2 vector kernel (not compiled / not supported)", &rr, 0, 1);
        fflush(stdout);
    }
#endif

    print_footer();

    printf("Notes:\n");
    printf("1) scalar rows report latency only\n");
    printf("2) vector row V7 reports throughput only\n");
    printf("3) accuracy is reported as max/avg ULP on %d grid points in [%.5f, %.5f]\n",
           ACCURACY_SAMPLES, X_MIN, X_MAX);
    printf("\n");

    return 0;
}