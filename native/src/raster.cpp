#include "visualizeit/raster.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__)
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
#include <immintrin.h>
#define VI_HAS_AVX512 1
#else
#define VI_HAS_AVX512 0
#endif

namespace {

constexpr float kNearZ = 1e-4f;
constexpr double kNearZD = 1e-4;
constexpr float kGuardMarginPx = 2.0f;
constexpr double kGuardMarginPxD = 2.0;
constexpr float kGuardBand = 1e6f;
constexpr int kMaxImageDim = 16384;
constexpr int kMaxVertices = 5000000;
constexpr int kMaxTriangles = 10000000;

struct ClipVert {
    float x;
    float y;
    float z;
    float u;
    float v;
};

struct ClipVertD {
    double x;
    double y;
    double z;
    double u;
    double v;
};

struct PreparedTri {
    float x0, y0;
    float x1, y1;
    float x2, y2;
    float q0, q1, q2;
    float uq0, uq1, uq2;
    float vq0, vq1, vq2;
    float area;
    float inv_area;
    float dq_dx;
    float duq_dx;
    float dvq_dx;
    int bx0, bx1;
    int by0, by1;
};

struct RasterScratch {
    std::vector<PreparedTri> tris;
    std::vector<uint8_t> hit;
    std::vector<float> u;
    std::vector<float> v;
};

thread_local RasterScratch tls_scratch;

inline float eval_edge(float xa, float ya, float xb, float yb, float sx, float sy) noexcept {
    return (xb - xa) * (sy - ya) - (yb - ya) * (sx - xa);
}

inline bool inside_tri(
    const PreparedTri& tri,
    float sx,
    float sy,
    float eps0,
    float eps1,
    float eps2
) noexcept {
    const float e0 = eval_edge(tri.x1, tri.y1, tri.x2, tri.y2, sx, sy);
    const float e1 = eval_edge(tri.x2, tri.y2, tri.x0, tri.y0, sx, sy);
    const float e2 = eval_edge(tri.x0, tri.y0, tri.x1, tri.y1, sx, sy);
    return (e0 >= -eps0) && (e1 >= -eps1) && (e2 >= -eps2);
}

inline bool narrow_edge_span(
    float xa,
    float ya,
    float xb,
    float yb,
    float sy,
    float eps,
    float& x_left,
    float& x_right
) noexcept {
    const float ax = ya - yb;
    const float cy = (xb - xa) * (sy - ya) + eps;
    if (std::fabs(ax) <= 1e-7f) {
        return cy >= 0.0f;
    }
    const float x_bound = xa - cy / ax;
    if (ax > 0.0f) {
        if (x_bound > x_left) {
            x_left = x_bound;
        }
    } else {
        if (x_bound < x_right) {
            x_right = x_bound;
        }
    }
    return x_left <= x_right;
}

inline int clip_near_plane(const ClipVertD* in_v, int in_n, ClipVertD* out_v) noexcept {
    if (in_n < 3) {
        return 0;
    }
    int out_n = 0;
    ClipVertD prev = in_v[in_n - 1];
    double d_prev = prev.z - kNearZD;
    bool in_prev = (d_prev >= 0.0);
    for (int i = 0; i < in_n; ++i) {
        const ClipVertD& curr = in_v[i];
        const double d_curr = curr.z - kNearZD;
        const bool in_curr = (d_curr >= 0.0);
        if (in_prev != in_curr) {
            const double denom = d_prev - d_curr;
            double t = 0.0;
            if (denom != 0.0) {
                const double raw_t = d_prev / denom;
                if (std::isfinite(raw_t)) {
                    t = std::clamp(raw_t, 0.0, 1.0);
                }
            }
            out_v[out_n++] = ClipVertD{
                prev.x + t * (curr.x - prev.x),
                prev.y + t * (curr.y - prev.y),
                kNearZD,
                prev.u + t * (curr.u - prev.u),
                prev.v + t * (curr.v - prev.v)
            };
        }
        if (in_curr) {
            out_v[out_n++] = curr;
        }
        prev = curr;
        d_prev = d_curr;
        in_prev = in_curr;
    }
    return out_n;
}

inline int clip_homogeneous_plane(
    const ClipVertD* in_v,
    int in_n,
    double nx,
    double ny,
    double nz,
    ClipVertD* out_v
) noexcept {
    if (in_n < 3) {
        return 0;
    }
    int out_n = 0;
    ClipVertD prev = in_v[in_n - 1];
    double d_prev = nx * prev.x + ny * prev.y + nz * prev.z;
    bool in_prev = (d_prev >= 0.0);
    for (int i = 0; i < in_n; ++i) {
        const ClipVertD& curr = in_v[i];
        const double d_curr = nx * curr.x + ny * curr.y + nz * curr.z;
        const bool in_curr = (d_curr >= 0.0);
        if (in_prev != in_curr) {
            const double denom = d_prev - d_curr;
            double t = 0.0;
            if (denom != 0.0) {
                const double raw_t = d_prev / denom;
                if (std::isfinite(raw_t)) {
                    t = std::clamp(raw_t, 0.0, 1.0);
                }
            }
            out_v[out_n++] = ClipVertD{
                prev.x + t * (curr.x - prev.x),
                prev.y + t * (curr.y - prev.y),
                std::max(kNearZD, prev.z + t * (curr.z - prev.z)),
                prev.u + t * (curr.u - prev.u),
                prev.v + t * (curr.v - prev.v)
            };
        }
        if (in_curr) {
            out_v[out_n++] = curr;
        }
        prev = curr;
        d_prev = d_curr;
        in_prev = in_curr;
    }
    return out_n;
}

inline void append_projected_screen_tri(
    float x0,
    float y0,
    float tq0,
    float u0,
    float v0,
    float x1,
    float y1,
    float tq1,
    float u1,
    float v1,
    float x2,
    float y2,
    float tq2,
    float u2,
    float v2,
    int frame_w,
    int frame_h,
    std::vector<PreparedTri>& out_tris,
    int& roi_x0,
    int& roi_y0,
    int& roi_x1,
    int& roi_y1
) {
    if (!std::isfinite(x0) || !std::isfinite(y0) ||
        !std::isfinite(x1) || !std::isfinite(y1) ||
        !std::isfinite(x2) || !std::isfinite(y2)) {
        return;
    }
    if (std::fabs(x0) > kGuardBand || std::fabs(y0) > kGuardBand ||
        std::fabs(x1) > kGuardBand || std::fabs(y1) > kGuardBand ||
        std::fabs(x2) > kGuardBand || std::fabs(y2) > kGuardBand) {
        return;
    }

    float area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0);
    if (!std::isfinite(area) || std::fabs(area) < 1e-8f) {
        return;
    }

    if (area < 0.0f) {
        std::swap(x1, x2);
        std::swap(y1, y2);
        std::swap(tq1, tq2);
        std::swap(u1, u2);
        std::swap(v1, v2);
        area = -area;
    }

    const float min_x = std::min({x0, x1, x2});
    const float max_x = std::max({x0, x1, x2});
    const float min_y = std::min({y0, y1, y2});
    const float max_y = std::max({y0, y1, y2});

    const int bx0 = std::max(0, static_cast<int>(std::floor(min_x - 0.5f)));
    const int bx1 = std::min(frame_w - 1, static_cast<int>(std::ceil(max_x - 0.5f)));
    const int by0 = std::max(0, static_cast<int>(std::floor(min_y - 0.5f)));
    const int by1 = std::min(frame_h - 1, static_cast<int>(std::ceil(max_y - 0.5f)));
    if (bx0 > bx1 || by0 > by1) {
        return;
    }

    const float inv_area = 1.0f / area;
    const float dl0_dx = (y1 - y2) * inv_area;
    const float dl1_dx = (y2 - y0) * inv_area;
    const float dl2_dx = -dl0_dx - dl1_dx;

    const float uq0 = u0 * tq0;
    const float uq1 = u1 * tq1;
    const float uq2 = u2 * tq2;
    const float vq0 = v0 * tq0;
    const float vq1 = v1 * tq1;
    const float vq2 = v2 * tq2;

    const float dq_dx = dl0_dx * tq0 + dl1_dx * tq1 + dl2_dx * tq2;
    const float duq_dx = dl0_dx * uq0 + dl1_dx * uq1 + dl2_dx * uq2;
    const float dvq_dx = dl0_dx * vq0 + dl1_dx * vq1 + dl2_dx * vq2;

    if (!std::isfinite(uq0) || !std::isfinite(uq1) || !std::isfinite(uq2) ||
        !std::isfinite(vq0) || !std::isfinite(vq1) || !std::isfinite(vq2) ||
        !std::isfinite(dq_dx) || !std::isfinite(duq_dx) || !std::isfinite(dvq_dx)) {
        return;
    }

    PreparedTri tri{};
    tri.x0 = x0;
    tri.y0 = y0;
    tri.x1 = x1;
    tri.y1 = y1;
    tri.x2 = x2;
    tri.y2 = y2;
    tri.q0 = tq0;
    tri.q1 = tq1;
    tri.q2 = tq2;
    tri.uq0 = uq0;
    tri.uq1 = uq1;
    tri.uq2 = uq2;
    tri.vq0 = vq0;
    tri.vq1 = vq1;
    tri.vq2 = vq2;
    tri.area = area;
    tri.inv_area = inv_area;
    tri.dq_dx = dq_dx;
    tri.duq_dx = duq_dx;
    tri.dvq_dx = dvq_dx;
    tri.bx0 = bx0;
    tri.bx1 = bx1;
    tri.by0 = by0;
    tri.by1 = by1;

    out_tris.push_back(tri);
    if (bx0 < roi_x0) roi_x0 = bx0;
    if (by0 < roi_y0) roi_y0 = by0;
    if (bx1 > roi_x1) roi_x1 = bx1;
    if (by1 > roi_y1) roi_y1 = by1;
}

template <int WrapRepeat, int BlendMode, bool HasMask>
int composite_roi_pass(
    const uint8_t* __restrict hit_buf,
    const float* __restrict u_buf,
    const float* __restrict v_buf,
    const uint8_t* __restrict texture_rgba,
    int tex_w,
    int tex_h,
    float opacity,
    float inv_L_base,
    uint8_t* __restrict frame_rgb_inout,
    uint8_t* __restrict mask_out,
    int frame_w,
    int frame_h,
    int roi_x0,
    int roi_y0,
    int roi_w,
    int roi_h
) noexcept {
    const float tex_max_x = static_cast<float>(tex_w - 1);
    const float tex_max_y = static_cast<float>(tex_h - 1);
    const float alpha_scale = opacity * (1.0f / 255.0f);
    constexpr float kMinAlpha = 0.5f / 255.0f;
    constexpr float kInkScale = 0.92f / 255.0f;
    const size_t total_frame_pixels = static_cast<size_t>(frame_w) * static_cast<size_t>(frame_h);

    int rendered_pixels = 0;

#if VI_HAS_AVX512
    const auto* __restrict tex32 = reinterpret_cast<const int32_t*>(texture_rgba);
    const __m512 v_zero = _mm512_setzero_ps();
    const __m512 v_one = _mm512_set1_ps(1.0f);
    const __m512 v_half = _mm512_set1_ps(0.5f);
    const __m512 v_255 = _mm512_set1_ps(255.0f);
    const __m512 v_pos_inf = _mm512_set1_ps(std::numeric_limits<float>::infinity());
    const __m512 v_tex_max_x = _mm512_set1_ps(tex_max_x);
    const __m512 v_tex_max_y = _mm512_set1_ps(tex_max_y);
    const __m512 v_alpha_scale = _mm512_set1_ps(alpha_scale);
    const __m512 v_min_alpha = _mm512_set1_ps(kMinAlpha);
    const __m512 v_decal_lo = _mm512_set1_ps(-1e-4f);
    const __m512 v_decal_hi = _mm512_set1_ps(1.0001f);
    const __m512 v_inv_L_base = _mm512_set1_ps(inv_L_base);
    const __m512 v_ink_scale = _mm512_set1_ps(kInkScale);
    const __m512 v_lum_r = _mm512_set1_ps(77.0f / 256.0f);
    const __m512 v_lum_g = _mm512_set1_ps(150.0f / 256.0f);
    const __m512 v_lum_b = _mm512_set1_ps(29.0f / 256.0f);
    const __m512 v_r_lo = _mm512_set1_ps(0.0f);
    const __m512 v_r_hi = _mm512_set1_ps(1.65f);
    const __m512 v_005 = _mm512_set1_ps(0.05f);
    const __m512 v_095 = _mm512_set1_ps(0.95f);
    const __m512 v_020 = _mm512_set1_ps(0.20f);
    const __m512 v_050 = _mm512_set1_ps(0.50f);

    const __m512i vi_zero = _mm512_setzero_si512();
    const __m512i vi_one = _mm512_set1_epi32(1);
    const __m512i vi_255 = _mm512_set1_epi32(255);
    const __m512i vi_tex_w = _mm512_set1_epi32(tex_w);
    const __m512i vi_tex_w_m1 = _mm512_set1_epi32(tex_w - 1);
    const __m512i vi_tex_h_m1 = _mm512_set1_epi32(tex_h - 1);
    const __m512i vi_rgb_byte_offsets = _mm512_setr_epi32(
        0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45
    );
    const __m128i v16_zero = _mm_setzero_si128();
    const __m128i v16_255 = _mm_set1_epi8(static_cast<char>(0xFF));
#endif

    for (int ry = 0; ry < roi_h; ++ry) {
        const size_t roi_row_base = static_cast<size_t>(ry) * static_cast<size_t>(roi_w);
        const size_t frame_row_base =
            static_cast<size_t>(roi_y0 + ry) * static_cast<size_t>(frame_w) +
            static_cast<size_t>(roi_x0);

        int rx = 0;

#if VI_HAS_AVX512
        for (; rx + 16 <= roi_w && (frame_row_base + static_cast<size_t>(rx) + 16u < total_frame_pixels); rx += 16) {
            const size_t r_idx = roi_row_base + static_cast<size_t>(rx);
            const __m128i h16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(hit_buf + r_idx));
            __mmask16 m = _mm_cmpneq_epi8_mask(h16, v16_zero);
            if (m == 0) {
                continue;
            }

            const __m512 u_raw = _mm512_maskz_loadu_ps(m, u_buf + r_idx);
            const __m512 v_raw = _mm512_maskz_loadu_ps(m, v_buf + r_idx);
            // Reject both NaN and +/-Inf on u and v: ordered & |val| < +Inf
            const __mmask16 m_ord =
                _mm512_cmp_ps_mask(u_raw, u_raw, _CMP_ORD_Q) &
                _mm512_cmp_ps_mask(v_raw, v_raw, _CMP_ORD_Q);
            const __mmask16 m_fin =
                m_ord &
                _mm512_cmp_ps_mask(_mm512_abs_ps(u_raw), v_pos_inf, _CMP_LT_OQ) &
                _mm512_cmp_ps_mask(_mm512_abs_ps(v_raw), v_pos_inf, _CMP_LT_OQ);
            m &= m_fin;
            if (m == 0) {
                continue;
            }
            const __m512 u = _mm512_maskz_mov_ps(m, u_raw);
            const __m512 v = _mm512_maskz_mov_ps(m, v_raw);
            __m512 us;
            __m512 vs;

            if constexpr (WrapRepeat == VI_WRAP_REPEAT) {
                us = _mm512_sub_ps(u, _mm512_floor_ps(u));
                vs = _mm512_sub_ps(v, _mm512_floor_ps(v));
                const __mmask16 m_uge1 = _mm512_cmp_ps_mask(us, v_one, _CMP_GE_OQ);
                const __mmask16 m_vge1 = _mm512_cmp_ps_mask(vs, v_one, _CMP_GE_OQ);
                us = _mm512_mask_sub_ps(us, m_uge1, us, v_one);
                vs = _mm512_mask_sub_ps(vs, m_vge1, vs, v_one);
                us = _mm512_min_ps(_mm512_max_ps(us, v_zero), v_one);
                vs = _mm512_min_ps(_mm512_max_ps(vs, v_zero), v_one);
            } else {
                const __mmask16 m_u_ok =
                    _mm512_cmp_ps_mask(u, v_decal_lo, _CMP_GE_OQ) &
                    _mm512_cmp_ps_mask(u, v_decal_hi, _CMP_LE_OQ);
                const __mmask16 m_v_ok =
                    _mm512_cmp_ps_mask(v, v_decal_lo, _CMP_GE_OQ) &
                    _mm512_cmp_ps_mask(v, v_decal_hi, _CMP_LE_OQ);
                m &= (m_u_ok & m_v_ok);
                if (m == 0) {
                    continue;
                }
                us = _mm512_min_ps(_mm512_max_ps(_mm512_maskz_mov_ps(m, u), v_zero), v_one);
                vs = _mm512_min_ps(_mm512_max_ps(_mm512_maskz_mov_ps(m, v), v_zero), v_one);
            }

            const __m512 tx = _mm512_mul_ps(us, v_tex_max_x);
            const __m512 ty = _mm512_mul_ps(vs, v_tex_max_y);
            const __m512i x0 = _mm512_min_epi32(_mm512_max_epi32(_mm512_cvttps_epi32(tx), vi_zero), vi_tex_w_m1);
            const __m512i y0 = _mm512_min_epi32(_mm512_max_epi32(_mm512_cvttps_epi32(ty), vi_zero), vi_tex_h_m1);
            const __m512i x1 = _mm512_min_epi32(_mm512_add_epi32(x0, vi_one), vi_tex_w_m1);
            const __m512i y1 = _mm512_min_epi32(_mm512_add_epi32(y0, vi_one), vi_tex_h_m1);

            const __m512 dx = _mm512_sub_ps(tx, _mm512_cvtepi32_ps(x0));
            const __m512 dy = _mm512_sub_ps(ty, _mm512_cvtepi32_ps(y0));
            const __m512 om_dx = _mm512_sub_ps(v_one, dx);
            const __m512 om_dy = _mm512_sub_ps(v_one, dy);

            const __m512 w00 = _mm512_mul_ps(om_dx, om_dy);
            const __m512 w10 = _mm512_mul_ps(dx, om_dy);
            const __m512 w01 = _mm512_mul_ps(om_dx, dy);
            const __m512 w11 = _mm512_mul_ps(dx, dy);

            const __m512i row0 = _mm512_mullo_epi32(y0, vi_tex_w);
            const __m512i row1 = _mm512_mullo_epi32(y1, vi_tex_w);
            const __m512i idx00 = _mm512_add_epi32(row0, x0);
            const __m512i idx10 = _mm512_add_epi32(row0, x1);
            const __m512i idx01 = _mm512_add_epi32(row1, x0);
            const __m512i idx11 = _mm512_add_epi32(row1, x1);

            const __m512i c00 = _mm512_mask_i32gather_epi32(vi_zero, m, idx00, tex32, 4);
            const __m512i c10 = _mm512_mask_i32gather_epi32(vi_zero, m, idx10, tex32, 4);
            const __m512i c01 = _mm512_mask_i32gather_epi32(vi_zero, m, idx01, tex32, 4);
            const __m512i c11 = _mm512_mask_i32gather_epi32(vi_zero, m, idx11, tex32, 4);

            const __m512 a00 = _mm512_cvtepi32_ps(_mm512_srli_epi32(c00, 24));
            const __m512 a10 = _mm512_cvtepi32_ps(_mm512_srli_epi32(c10, 24));
            const __m512 a01 = _mm512_cvtepi32_ps(_mm512_srli_epi32(c01, 24));
            const __m512 a11 = _mm512_cvtepi32_ps(_mm512_srli_epi32(c11, 24));

            const __m512 sa = _mm512_fmadd_ps(
                w00, a00,
                _mm512_fmadd_ps(w10, a10, _mm512_fmadd_ps(w01, a01, _mm512_mul_ps(w11, a11)))
            );
            const __m512 alpha = _mm512_mul_ps(sa, v_alpha_scale);
            m &= _mm512_cmp_ps_mask(alpha, v_min_alpha, _CMP_GT_OQ);
            if (m == 0) {
                continue;
            }

            auto interp_chan = [&](int shift) noexcept -> __m512 {
                const __m512 p00 = _mm512_cvtepi32_ps(_mm512_and_si512(_mm512_srli_epi32(c00, shift), vi_255));
                const __m512 p10 = _mm512_cvtepi32_ps(_mm512_and_si512(_mm512_srli_epi32(c10, shift), vi_255));
                const __m512 p01 = _mm512_cvtepi32_ps(_mm512_and_si512(_mm512_srli_epi32(c01, shift), vi_255));
                const __m512 p11 = _mm512_cvtepi32_ps(_mm512_and_si512(_mm512_srli_epi32(c11, shift), vi_255));
                return _mm512_fmadd_ps(
                    w00, p00,
                    _mm512_fmadd_ps(w10, p10, _mm512_fmadd_ps(w01, p01, _mm512_mul_ps(w11, p11)))
                );
            };

            const __m512 sr = interp_chan(0);
            const __m512 sg = interp_chan(8);
            const __m512 sb = interp_chan(16);

            const size_t f_base = frame_row_base + static_cast<size_t>(rx);
            uint8_t* __restrict dst_base = frame_rgb_inout + f_base * 3u;
            const __m512i dst_packed = _mm512_mask_i32gather_epi32(vi_zero, m, vi_rgb_byte_offsets, dst_base, 1);

            const __m512 dr = _mm512_cvtepi32_ps(_mm512_and_si512(dst_packed, vi_255));
            const __m512 dg = _mm512_cvtepi32_ps(_mm512_and_si512(_mm512_srli_epi32(dst_packed, 8), vi_255));
            const __m512 db = _mm512_cvtepi32_ps(_mm512_and_si512(_mm512_srli_epi32(dst_packed, 16), vi_255));

            __m512 br = sr;
            __m512 bg = sg;
            __m512 bb = sb;

            if constexpr (BlendMode == VI_BLEND_LUMINANCE) {
                const __m512 L_px = _mm512_fmadd_ps(
                    v_lum_r, dr,
                    _mm512_fmadd_ps(v_lum_g, dg, _mm512_mul_ps(v_lum_b, db))
                );
                const __m512 ratio = _mm512_min_ps(_mm512_max_ps(_mm512_mul_ps(L_px, v_inv_L_base), v_r_lo), v_r_hi);
                const __mmask16 m_hi = _mm512_cmp_ps_mask(ratio, v_one, _CMP_GT_OQ);

                const __m512 gain_lo = _mm512_fmadd_ps(v_095, ratio, v_005);
                const __m512 br_lo = _mm512_mul_ps(sr, gain_lo);
                const __m512 bg_lo = _mm512_mul_ps(sg, gain_lo);
                const __m512 bb_lo = _mm512_mul_ps(sb, gain_lo);

                if (m_hi != 0) {
                    const __m512 hi = _mm512_sub_ps(ratio, v_one);
                    const __m512 gain_hi = _mm512_fmadd_ps(v_050, hi, v_one);
                    const __m512 sheen = _mm512_mul_ps(v_020, hi);
                    const __m512 br_hi = _mm512_fmadd_ps(sr, gain_hi, _mm512_mul_ps(sheen, _mm512_sub_ps(v_255, sr)));
                    const __m512 bg_hi = _mm512_fmadd_ps(sg, gain_hi, _mm512_mul_ps(sheen, _mm512_sub_ps(v_255, sg)));
                    const __m512 bb_hi = _mm512_fmadd_ps(sb, gain_hi, _mm512_mul_ps(sheen, _mm512_sub_ps(v_255, sb)));
                    br = _mm512_mask_blend_ps(m_hi, br_lo, br_hi);
                    bg = _mm512_mask_blend_ps(m_hi, bg_lo, bg_hi);
                    bb = _mm512_mask_blend_ps(m_hi, bb_lo, bb_hi);
                } else {
                    br = br_lo;
                    bg = bg_lo;
                    bb = bb_lo;
                }
            } else if constexpr (BlendMode == VI_BLEND_INK) {
                br = _mm512_mul_ps(_mm512_mul_ps(dr, sr), v_ink_scale);
                bg = _mm512_mul_ps(_mm512_mul_ps(dg, sg), v_ink_scale);
                bb = _mm512_mul_ps(_mm512_mul_ps(db, sb), v_ink_scale);
            }

            const __m512 inv_a = _mm512_sub_ps(v_one, alpha);
            const __m512i out_r = _mm512_min_epi32(
                _mm512_max_epi32(_mm512_cvttps_epi32(_mm512_add_ps(_mm512_fmadd_ps(alpha, br, _mm512_mul_ps(inv_a, dr)), v_half)), vi_zero),
                vi_255
            );
            const __m512i out_g = _mm512_min_epi32(
                _mm512_max_epi32(_mm512_cvttps_epi32(_mm512_add_ps(_mm512_fmadd_ps(alpha, bg, _mm512_mul_ps(inv_a, dg)), v_half)), vi_zero),
                vi_255
            );
            const __m512i out_b = _mm512_min_epi32(
                _mm512_max_epi32(_mm512_cvttps_epi32(_mm512_add_ps(_mm512_fmadd_ps(alpha, bb, _mm512_mul_ps(inv_a, db)), v_half)), vi_zero),
                vi_255
            );

            const __m128i r16 = _mm512_maskz_cvtepi32_epi8(0xFFFFu, out_r);
            const __m128i g16 = _mm512_maskz_cvtepi32_epi8(0xFFFFu, out_g);
            const __m128i b16 = _mm512_maskz_cvtepi32_epi8(0xFFFFu, out_b);

            if constexpr (HasMask) {
                _mm_mask_storeu_epi8(mask_out + f_base, m, v16_255);
            }

#if defined(__AVX512VBMI__)
            if (m == static_cast<__mmask16>(0xFFFFu)) {
                alignas(64) static constexpr uint8_t kInterleaveRgb[64] = {
                    0, 16, 32,  1, 17, 33,  2, 18, 34,  3, 19, 35,
                    4, 20, 36,  5, 21, 37,  6, 22, 38,  7, 23, 39,
                    8, 24, 40,  9, 25, 41, 10, 26, 42, 11, 27, 43,
                   12, 28, 44, 13, 29, 45, 14, 30, 46, 15, 31, 47,
                    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                    0,  0,  0,  0
                };
                const __m512i vi_interleave = _mm512_load_si512(kInterleaveRgb);
                const __m512i planar = _mm512_inserti32x4(
                    _mm512_inserti32x4(_mm512_zextsi128_si512(r16), g16, 1),
                    b16,
                    2
                );
                const __m512i rgb48 = _mm512_permutexvar_epi8(vi_interleave, planar);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(dst_base + 0), _mm512_castsi512_si128(rgb48));
                _mm_storeu_si128(reinterpret_cast<__m128i*>(dst_base + 16), _mm512_maskz_extracti32x4_epi32(0xFu, rgb48, 1));
                _mm_storeu_si128(reinterpret_cast<__m128i*>(dst_base + 32), _mm512_maskz_extracti32x4_epi32(0xFu, rgb48, 2));
                rendered_pixels += 16;
                continue;
            }
#endif

            alignas(16) uint8_t r_bytes[16];
            alignas(16) uint8_t g_bytes[16];
            alignas(16) uint8_t b_bytes[16];
            _mm_store_si128(reinterpret_cast<__m128i*>(r_bytes), r16);
            _mm_store_si128(reinterpret_cast<__m128i*>(g_bytes), g16);
            _mm_store_si128(reinterpret_cast<__m128i*>(b_bytes), b16);

            uint32_t bits = static_cast<uint32_t>(m);
            rendered_pixels += __builtin_popcount(bits);
            while (bits != 0u) {
                const int lane = __builtin_ctz(bits);
                bits &= (bits - 1u);
                uint8_t* p = dst_base + static_cast<size_t>(lane) * 3u;
                p[0] = r_bytes[lane];
                p[1] = g_bytes[lane];
                p[2] = b_bytes[lane];
            }
        }
#endif

        // Exact scalar loop for tail pixels (or full row when AVX-512 is disabled).
        for (; rx < roi_w; ++rx) {
            const size_t r_idx = roi_row_base + static_cast<size_t>(rx);
            if (!hit_buf[r_idx]) {
                continue;
            }

            const float u = u_buf[r_idx];
            const float v = v_buf[r_idx];
            float us = 0.0f;
            float vs = 0.0f;

            if constexpr (WrapRepeat == VI_WRAP_REPEAT) {
                if (!std::isfinite(u) || !std::isfinite(v)) {
                    continue;
                }
                us = u - std::floor(u);
                vs = v - std::floor(v);
                if (!std::isfinite(us) || !std::isfinite(vs)) {
                    continue;
                }
                if (us >= 1.0f) us -= 1.0f;
                if (us < 0.0f) us = 0.0f;
                if (vs >= 1.0f) vs -= 1.0f;
                if (vs < 0.0f) vs = 0.0f;
            } else {
                if (!(u >= -1e-4f && u <= 1.0001f && v >= -1e-4f && v <= 1.0001f)) {
                    continue;
                }
                us = std::clamp(u, 0.0f, 1.0f);
                vs = std::clamp(v, 0.0f, 1.0f);
            }

            const float tx = us * tex_max_x;
            const float ty = vs * tex_max_y;
            const int x0 = std::clamp(static_cast<int>(tx), 0, tex_w - 1);
            const int y0 = std::clamp(static_cast<int>(ty), 0, tex_h - 1);
            const int x1 = (x0 + 1 < tex_w) ? (x0 + 1) : (tex_w - 1);
            const int y1 = (y0 + 1 < tex_h) ? (y0 + 1) : (tex_h - 1);
            const float dx = std::clamp(tx - static_cast<float>(x0), 0.0f, 1.0f);
            const float dy = std::clamp(ty - static_cast<float>(y0), 0.0f, 1.0f);

            const float w00 = (1.0f - dx) * (1.0f - dy);
            const float w10 = dx * (1.0f - dy);
            const float w01 = (1.0f - dx) * dy;
            const float w11 = dx * dy;

            const size_t row0 = static_cast<size_t>(y0) * static_cast<size_t>(tex_w);
            const size_t row1 = static_cast<size_t>(y1) * static_cast<size_t>(tex_w);
            const uint8_t* p00 = texture_rgba + (row0 + static_cast<size_t>(x0)) * 4u;
            const uint8_t* p10 = texture_rgba + (row0 + static_cast<size_t>(x1)) * 4u;
            const uint8_t* p01 = texture_rgba + (row1 + static_cast<size_t>(x0)) * 4u;
            const uint8_t* p11 = texture_rgba + (row1 + static_cast<size_t>(x1)) * 4u;

            const float sa = w00 * p00[3] + w10 * p10[3] + w01 * p01[3] + w11 * p11[3];
            const float alpha = sa * alpha_scale;
            if (alpha <= kMinAlpha) {
                continue;
            }

            const float sr = w00 * p00[0] + w10 * p10[0] + w01 * p01[0] + w11 * p11[0];
            const float sg = w00 * p00[1] + w10 * p10[1] + w01 * p01[1] + w11 * p11[1];
            const float sb = w00 * p00[2] + w10 * p10[2] + w01 * p01[2] + w11 * p11[2];

            const size_t f_idx = frame_row_base + static_cast<size_t>(rx);
            uint8_t* dst = frame_rgb_inout + f_idx * 3u;
            const float dr = static_cast<float>(dst[0]);
            const float dg = static_cast<float>(dst[1]);
            const float db = static_cast<float>(dst[2]);

            float br = sr;
            float bg = sg;
            float bb = sb;

            if constexpr (BlendMode == VI_BLEND_LUMINANCE) {
                const float L_px = (77.0f * dr + 150.0f * dg + 29.0f * db) * (1.0f / 256.0f);
                const float ratio = std::clamp(L_px * inv_L_base, 0.0f, 1.65f);
                if (ratio <= 1.0f) {
                    const float gain = 0.05f + 0.95f * ratio;
                    br = sr * gain;
                    bg = sg * gain;
                    bb = sb * gain;
                } else {
                    const float hi = ratio - 1.0f;
                    const float gain = 1.0f + 0.50f * hi;
                    const float sheen = 0.20f * hi;
                    br = sr * gain + sheen * (255.0f - sr);
                    bg = sg * gain + sheen * (255.0f - sg);
                    bb = sb * gain + sheen * (255.0f - sb);
                }
            } else if constexpr (BlendMode == VI_BLEND_INK) {
                br = dr * sr * kInkScale;
                bg = dg * sg * kInkScale;
                bb = db * sb * kInkScale;
            }

            const float inv_a = 1.0f - alpha;
            dst[0] = static_cast<uint8_t>(std::clamp(static_cast<int>(alpha * br + inv_a * dr + 0.5f), 0, 255));
            dst[1] = static_cast<uint8_t>(std::clamp(static_cast<int>(alpha * bg + inv_a * dg + 0.5f), 0, 255));
            dst[2] = static_cast<uint8_t>(std::clamp(static_cast<int>(alpha * bb + inv_a * db + 0.5f), 0, 255));
            if constexpr (HasMask) {
                mask_out[f_idx] = 255u;
            }
            ++rendered_pixels;
        }
    }

    return rendered_pixels;
}

template <int WrapRepeat, int BlendMode>
inline int dispatch_mask(
    const uint8_t* hit_buf,
    const float* u_buf,
    const float* v_buf,
    const uint8_t* texture_rgba,
    int tex_w,
    int tex_h,
    float opacity,
    float inv_L_base,
    uint8_t* frame_rgb_inout,
    uint8_t* mask_out,
    int frame_w,
    int frame_h,
    int roi_x0,
    int roi_y0,
    int roi_w,
    int roi_h
) noexcept {
    if (mask_out != nullptr) {
        return composite_roi_pass<WrapRepeat, BlendMode, true>(
            hit_buf, u_buf, v_buf, texture_rgba, tex_w, tex_h, opacity, inv_L_base,
            frame_rgb_inout, mask_out, frame_w, frame_h, roi_x0, roi_y0, roi_w, roi_h
        );
    }
    return composite_roi_pass<WrapRepeat, BlendMode, false>(
        hit_buf, u_buf, v_buf, texture_rgba, tex_w, tex_h, opacity, inv_L_base,
        frame_rgb_inout, nullptr, frame_w, frame_h, roi_x0, roi_y0, roi_w, roi_h
    );
}

} // namespace

int vi_rasterize_mesh_rgbd(
    const float* vertices_cam,
    int num_vertices,
    const int32_t* triangles,
    int num_triangles,
    const float* vertex_uvs,
    const uint8_t* texture_rgba,
    int tex_w,
    int tex_h,
    int wrap_repeat,
    float fx,
    float fy,
    float cx,
    float cy,
    float opacity,
    int blend_mode,
    uint8_t* frame_rgb_inout,
    float* depth_buf_inout,
    uint8_t* mask_out,
    int frame_w,
    int frame_h
) {
    // Step 1: Atomic up-front validation before mutating any buffer.
    if (!vertices_cam || !triangles || !vertex_uvs || !texture_rgba ||
        !frame_rgb_inout || !depth_buf_inout) {
        return VI_RASTER_ERR_NULL_PTR;
    }
    if (num_vertices <= 0 || num_triangles <= 0 ||
        tex_w <= 0 || tex_h <= 0 ||
        frame_w <= 0 || frame_h <= 0 ||
        frame_w > kMaxImageDim || frame_h > kMaxImageDim ||
        tex_w > kMaxImageDim || tex_h > kMaxImageDim ||
        num_vertices > kMaxVertices || num_triangles > kMaxTriangles) {
        return VI_RASTER_ERR_DIMENSIONS;
    }
    if (!std::isfinite(fx) || !std::isfinite(fy) ||
        !std::isfinite(cx) || !std::isfinite(cy) ||
        fx <= 0.0f || fy <= 0.0f) {
        return VI_RASTER_ERR_INTRINSICS;
    }
    if (!std::isfinite(opacity) || opacity < 0.0f || opacity > 1.0f ||
        blend_mode < VI_BLEND_NORMAL || blend_mode > VI_BLEND_INK ||
        (wrap_repeat != VI_WRAP_DECAL && wrap_repeat != VI_WRAP_REPEAT)) {
        return VI_RASTER_ERR_PARAMS;
    }

    const size_t total_indices = static_cast<size_t>(num_triangles) * 3u;
    for (size_t i = 0; i < total_indices; ++i) {
        const int32_t idx = triangles[i];
        if (idx < 0 || idx >= num_vertices) {
            return VI_RASTER_ERR_INDEX_OUT_OF_RANGE;
        }
    }

    const size_t total_xyz = static_cast<size_t>(num_vertices) * 3u;
    for (size_t i = 0; i < total_xyz; ++i) {
        if (!std::isfinite(vertices_cam[i])) {
            return VI_RASTER_ERR_NONFINITE_GEOMETRY;
        }
    }

    const size_t total_uv = static_cast<size_t>(num_vertices) * 2u;
    for (size_t i = 0; i < total_uv; ++i) {
        if (!std::isfinite(vertex_uvs[i])) {
            return VI_RASTER_ERR_NONFINITE_GEOMETRY;
        }
    }

    // Step 2: Fast interior setup + 5-plane double-precision Sutherland-Hodgman frustum clipping.
    auto& tris = tls_scratch.tris;
    tris.clear();
    if (tris.capacity() < static_cast<size_t>(num_triangles)) {
        tris.reserve(static_cast<size_t>(num_triangles));
    }

    int roi_x0 = frame_w;
    int roi_y0 = frame_h;
    int roi_x1 = -1;
    int roi_y1 = -1;

    const float min_fast_x = -kGuardMarginPx;
    const float max_fast_x = static_cast<float>(frame_w) + kGuardMarginPx;
    const float min_fast_y = -kGuardMarginPx;
    const float max_fast_y = static_cast<float>(frame_h) + kGuardMarginPx;

    const double fxd = static_cast<double>(fx);
    const double fyd = static_cast<double>(fy);
    const double cxd = static_cast<double>(cx);
    const double cyd = static_cast<double>(cy);
    const double fw_pad = static_cast<double>(frame_w) + kGuardMarginPxD;
    const double fh_pad = static_cast<double>(frame_h) + kGuardMarginPxD;

    for (int t = 0; t < num_triangles; ++t) {
        const int32_t i0 = triangles[static_cast<size_t>(t) * 3u + 0u];
        const int32_t i1 = triangles[static_cast<size_t>(t) * 3u + 1u];
        const int32_t i2 = triangles[static_cast<size_t>(t) * 3u + 2u];

        const ClipVert v0{
            vertices_cam[static_cast<size_t>(i0) * 3u + 0u],
            vertices_cam[static_cast<size_t>(i0) * 3u + 1u],
            vertices_cam[static_cast<size_t>(i0) * 3u + 2u],
            vertex_uvs[static_cast<size_t>(i0) * 2u + 0u],
            vertex_uvs[static_cast<size_t>(i0) * 2u + 1u]
        };
        const ClipVert v1{
            vertices_cam[static_cast<size_t>(i1) * 3u + 0u],
            vertices_cam[static_cast<size_t>(i1) * 3u + 1u],
            vertices_cam[static_cast<size_t>(i1) * 3u + 2u],
            vertex_uvs[static_cast<size_t>(i1) * 2u + 0u],
            vertex_uvs[static_cast<size_t>(i1) * 2u + 1u]
        };
        const ClipVert v2{
            vertices_cam[static_cast<size_t>(i2) * 3u + 0u],
            vertices_cam[static_cast<size_t>(i2) * 3u + 1u],
            vertices_cam[static_cast<size_t>(i2) * 3u + 2u],
            vertex_uvs[static_cast<size_t>(i2) * 2u + 0u],
            vertex_uvs[static_cast<size_t>(i2) * 2u + 1u]
        };

        if (v0.z < kNearZ && v1.z < kNearZ && v2.z < kNearZ) {
            continue;
        }

        const float q0_est = 1.0f / std::max(v0.z, kNearZ);
        const float q1_est = 1.0f / std::max(v1.z, kNearZ);
        const float q2_est = 1.0f / std::max(v2.z, kNearZ);
        if (!std::isfinite(v0.u * q0_est) || !std::isfinite(v0.v * q0_est) ||
            !std::isfinite(v1.u * q1_est) || !std::isfinite(v1.v * q1_est) ||
            !std::isfinite(v2.u * q2_est) || !std::isfinite(v2.v * q2_est)) {
            continue;
        }

        if (v0.z >= kNearZ && v1.z >= kNearZ && v2.z >= kNearZ) {
            const float q0 = 1.0f / v0.z;
            const float q1 = 1.0f / v1.z;
            const float q2 = 1.0f / v2.z;
            const float x0 = fx * v0.x * q0 + cx;
            const float y0 = fy * v0.y * q0 + cy;
            const float x1 = fx * v1.x * q1 + cx;
            const float y1 = fy * v1.y * q1 + cy;
            const float x2 = fx * v2.x * q2 + cx;
            const float y2 = fy * v2.y * q2 + cy;

            if (x0 >= min_fast_x && x0 <= max_fast_x && y0 >= min_fast_y && y0 <= max_fast_y &&
                x1 >= min_fast_x && x1 <= max_fast_x && y1 >= min_fast_y && y1 <= max_fast_y &&
                x2 >= min_fast_x && x2 <= max_fast_x && y2 >= min_fast_y && y2 <= max_fast_y) {
                append_projected_screen_tri(
                    x0, y0, q0, v0.u, v0.v,
                    x1, y1, q1, v1.u, v1.v,
                    x2, y2, q2, v2.u, v2.v,
                    frame_w, frame_h, tris, roi_x0, roi_y0, roi_x1, roi_y1
                );
                continue;
            }
        }

        ClipVertD buf_a[12] = {
            {v0.x, v0.y, v0.z, v0.u, v0.v},
            {v1.x, v1.y, v1.z, v1.u, v1.v},
            {v2.x, v2.y, v2.z, v2.u, v2.v}
        };
        ClipVertD buf_b[12];

        int n = clip_near_plane(buf_a, 3, buf_b);
        if (n < 3) continue;
        n = clip_homogeneous_plane(buf_b, n, fxd, 0.0, cxd + kGuardMarginPxD, buf_a);      // Left:   x_proj >= -2
        if (n < 3) continue;
        n = clip_homogeneous_plane(buf_a, n, -fxd, 0.0, fw_pad - cxd, buf_b);              // Right:  x_proj <= frame_w + 2
        if (n < 3) continue;
        n = clip_homogeneous_plane(buf_b, n, 0.0, fyd, cyd + kGuardMarginPxD, buf_a);      // Top:    y_proj >= -2
        if (n < 3) continue;
        n = clip_homogeneous_plane(buf_a, n, 0.0, -fyd, fh_pad - cyd, buf_b);              // Bottom: y_proj <= frame_h + 2
        if (n < 3) continue;

        const double q0_d = 1.0 / buf_b[0].z;
        const float px0 = static_cast<float>(fxd * buf_b[0].x * q0_d + cxd);
        const float py0 = static_cast<float>(fyd * buf_b[0].y * q0_d + cyd);
        const float pq0 = static_cast<float>(q0_d);
        const float pu0 = static_cast<float>(buf_b[0].u);
        const float pv0 = static_cast<float>(buf_b[0].v);

        for (int i = 1; i + 1 < n; ++i) {
            const double qi_d = 1.0 / buf_b[i].z;
            const double qj_d = 1.0 / buf_b[i + 1].z;
            append_projected_screen_tri(
                px0,
                py0,
                pq0,
                pu0,
                pv0,
                static_cast<float>(fxd * buf_b[i].x * qi_d + cxd),
                static_cast<float>(fyd * buf_b[i].y * qi_d + cyd),
                static_cast<float>(qi_d),
                static_cast<float>(buf_b[i].u),
                static_cast<float>(buf_b[i].v),
                static_cast<float>(fxd * buf_b[i + 1].x * qj_d + cxd),
                static_cast<float>(fyd * buf_b[i + 1].y * qj_d + cyd),
                static_cast<float>(qj_d),
                static_cast<float>(buf_b[i + 1].u),
                static_cast<float>(buf_b[i + 1].v),
                frame_w,
                frame_h,
                tris,
                roi_x0,
                roi_y0,
                roi_x1,
                roi_y1
            );
        }
    }

    if (tris.empty() || roi_x0 > roi_x1 || roi_y0 > roi_y1) {
        return 0;
    }

    // Step 3: Pass 1 — Vectorized scanline span z-buffer & perspective UV pass.
    const int roi_w = roi_x1 - roi_x0 + 1;
    const int roi_h = roi_y1 - roi_y0 + 1;
    const size_t roi_count = static_cast<size_t>(roi_w) * static_cast<size_t>(roi_h);

    if (tls_scratch.hit.size() < roi_count) {
        tls_scratch.hit.resize(roi_count);
    }
    std::memset(tls_scratch.hit.data(), 0, roi_count * sizeof(uint8_t));
    if (tls_scratch.u.size() < roi_count) {
        tls_scratch.u.resize(roi_count);
    }
    if (tls_scratch.v.size() < roi_count) {
        tls_scratch.v.resize(roi_count);
    }

#if VI_HAS_AVX512
    const __m512 v_lanes = _mm512_setr_ps(
        0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f,
        8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f
    );
    const __m512 v_one_p1 = _mm512_set1_ps(1.0f);
    const __m512 v_q_eps = _mm512_set1_ps(1e-12f);
    const __m128i v_hit_one = _mm_set1_epi8(1);
#endif

    for (const PreparedTri& tri : tris) {
        const float eps0 = 1e-4f * std::max(std::fabs(tri.x2 - tri.x1), std::fabs(tri.y2 - tri.y1));
        const float eps1 = 1e-4f * std::max(std::fabs(tri.x0 - tri.x2), std::fabs(tri.y0 - tri.y2));
        const float eps2 = 1e-4f * std::max(std::fabs(tri.x1 - tri.x0), std::fabs(tri.y1 - tri.y0));
#if VI_HAS_AVX512
        const __m512 v_dq = _mm512_set1_ps(tri.dq_dx);
        const __m512 v_duq = _mm512_set1_ps(tri.duq_dx);
        const __m512 v_dvq = _mm512_set1_ps(tri.dvq_dx);
#endif

        for (int py = tri.by0; py <= tri.by1; ++py) {
            const float sy = static_cast<float>(py) + 0.5f;
            float x_left = static_cast<float>(tri.bx0) + 0.5f;
            float x_right = static_cast<float>(tri.bx1) + 0.5f;

            if (!narrow_edge_span(tri.x1, tri.y1, tri.x2, tri.y2, sy, eps0, x_left, x_right)) {
                continue;
            }
            if (!narrow_edge_span(tri.x2, tri.y2, tri.x0, tri.y0, sy, eps1, x_left, x_right)) {
                continue;
            }
            if (!narrow_edge_span(tri.x0, tri.y0, tri.x1, tri.y1, sy, eps2, x_left, x_right)) {
                continue;
            }

            int px_start = std::max(tri.bx0, static_cast<int>(std::ceil(x_left - 0.5f)));
            int px_end = std::min(tri.bx1, static_cast<int>(std::floor(x_right - 0.5f)));

            while (px_start <= px_end && !inside_tri(tri, static_cast<float>(px_start) + 0.5f, sy, eps0, eps1, eps2)) {
                ++px_start;
            }
            while (px_end >= px_start && !inside_tri(tri, static_cast<float>(px_end) + 0.5f, sy, eps0, eps1, eps2)) {
                --px_end;
            }
            if (px_start > px_end) {
                continue;
            }

            const float sx0 = static_cast<float>(px_start) + 0.5f;
            const float l0 = eval_edge(tri.x1, tri.y1, tri.x2, tri.y2, sx0, sy) * tri.inv_area;
            const float l1 = eval_edge(tri.x2, tri.y2, tri.x0, tri.y0, sx0, sy) * tri.inv_area;
            const float l2 = 1.0f - l0 - l1;

            const float q_base = l0 * tri.q0 + l1 * tri.q1 + l2 * tri.q2;
            const float uq_base = l0 * tri.uq0 + l1 * tri.uq1 + l2 * tri.uq2;
            const float vq_base = l0 * tri.vq0 + l1 * tri.vq1 + l2 * tri.vq2;

            const int span = px_end - px_start + 1;
            float* __restrict depth_ptr =
                depth_buf_inout + static_cast<size_t>(py) * static_cast<size_t>(frame_w) + px_start;
            const size_t roi_offset =
                static_cast<size_t>(py - roi_y0) * static_cast<size_t>(roi_w) +
                static_cast<size_t>(px_start - roi_x0);
            uint8_t* __restrict hit_ptr = tls_scratch.hit.data() + roi_offset;
            float* __restrict u_ptr = tls_scratch.u.data() + roi_offset;
            float* __restrict v_ptr = tls_scratch.v.data() + roi_offset;

            int k = 0;
#if VI_HAS_AVX512
            const __m512 v_q_base = _mm512_set1_ps(q_base);
            const __m512 v_uq_base = _mm512_set1_ps(uq_base);
            const __m512 v_vq_base = _mm512_set1_ps(vq_base);
            for (; k < span; k += 16) {
                const int rem = span - k;
                const __mmask16 lane_m = (rem >= 16)
                    ? static_cast<__mmask16>(0xFFFFu)
                    : static_cast<__mmask16>((1u << rem) - 1u);
                const __m512 v_fk = _mm512_add_ps(_mm512_set1_ps(static_cast<float>(k)), v_lanes);
                const __m512 v_q = _mm512_fmadd_ps(v_dq, v_fk, v_q_base);
                const __mmask16 m_pos = lane_m & _mm512_cmp_ps_mask(v_q, v_q_eps, _CMP_GT_OQ);
                if (m_pos == 0) {
                    continue;
                }
                const __m512 v_z = _mm512_div_ps(v_one_p1, v_q);
                const __m512 v_cur_d = _mm512_maskz_loadu_ps(m_pos, depth_ptr + k);
                const __mmask16 m_vis = m_pos & _mm512_cmp_ps_mask(v_z, v_cur_d, _CMP_LT_OQ);
                if (m_vis != 0) {
                    _mm512_mask_storeu_ps(depth_ptr + k, m_vis, v_z);
                    const __m512 v_u = _mm512_mul_ps(_mm512_fmadd_ps(v_duq, v_fk, v_uq_base), v_z);
                    const __m512 v_v = _mm512_mul_ps(_mm512_fmadd_ps(v_dvq, v_fk, v_vq_base), v_z);
                    _mm512_mask_storeu_ps(u_ptr + k, m_vis, v_u);
                    _mm512_mask_storeu_ps(v_ptr + k, m_vis, v_v);
                    _mm_mask_storeu_epi8(hit_ptr + k, m_vis, v_hit_one);
                }
            }
#else
            const float dq_dx = tri.dq_dx;
            const float duq_dx = tri.duq_dx;
            const float dvq_dx = tri.dvq_dx;
            for (; k < span; ++k) {
                const float fk = static_cast<float>(k);
                const float q = q_base + dq_dx * fk;
                if (q > 1e-12f) {
                    const float z = 1.0f / q;
                    if (z < depth_ptr[k]) {
                        depth_ptr[k] = z;
                        hit_ptr[k] = 1u;
                        u_ptr[k] = (uq_base + duq_dx * fk) * z;
                        v_ptr[k] = (vq_base + dvq_dx * fk) * z;
                    }
                }
            }
#endif
        }
    }

    // Step 4: Pass 2 — Bilinear RGBA sampling & photorealistic blend over ROI.
    if (opacity <= (0.5f / 255.0f)) {
        return 0;
    }

    const uint8_t* __restrict hit_buf = tls_scratch.hit.data();
    const float* __restrict u_buf = tls_scratch.u.data();
    const float* __restrict v_buf = tls_scratch.v.data();

    float inv_L_base = 1.0f / 160.0f;
    if (blend_mode == VI_BLEND_LUMINANCE) {
        uint64_t lum256_sum = 0;
        size_t hit_count = 0;
        const size_t total_frame_pixels = static_cast<size_t>(frame_w) * static_cast<size_t>(frame_h);
#if VI_HAS_AVX512
        const __m512i vi_zero = _mm512_setzero_si512();
        const __m512i vi_255 = _mm512_set1_epi32(255);
        const __m512i vi_77 = _mm512_set1_epi32(77);
        const __m512i vi_150 = _mm512_set1_epi32(150);
        const __m512i vi_29 = _mm512_set1_epi32(29);
        const __m512i vi_rgb_byte_offsets = _mm512_setr_epi32(
            0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45
        );
        const __m128i v16_zero = _mm_setzero_si128();
#endif
        for (int ry = 0; ry < roi_h; ++ry) {
            const uint8_t* __restrict hit_row = hit_buf + static_cast<size_t>(ry) * static_cast<size_t>(roi_w);
            const size_t frame_row_base =
                static_cast<size_t>(roi_y0 + ry) * static_cast<size_t>(frame_w) +
                static_cast<size_t>(roi_x0);
            const uint8_t* __restrict rgb_row = frame_rgb_inout + frame_row_base * 3u;
            int rx = 0;
#if VI_HAS_AVX512
            for (; rx + 16 <= roi_w && (frame_row_base + static_cast<size_t>(rx) + 16u < total_frame_pixels); rx += 16) {
                const __m128i h16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(hit_row + rx));
                const __mmask16 m = _mm_cmpneq_epi8_mask(h16, v16_zero);
                if (m == 0) {
                    continue;
                }
                const __m512i packed = _mm512_mask_i32gather_epi32(
                    vi_zero, m, vi_rgb_byte_offsets, rgb_row + static_cast<size_t>(rx) * 3u, 1
                );
                const __m512i dr = _mm512_and_si512(packed, vi_255);
                const __m512i dg = _mm512_and_si512(_mm512_srli_epi32(packed, 8), vi_255);
                const __m512i db = _mm512_and_si512(_mm512_srli_epi32(packed, 16), vi_255);
                const __m512i lum = _mm512_add_epi32(
                    _mm512_mullo_epi32(dr, vi_77),
                    _mm512_add_epi32(_mm512_mullo_epi32(dg, vi_150), _mm512_mullo_epi32(db, vi_29))
                );
                lum256_sum += static_cast<uint64_t>(_mm512_mask_reduce_add_epi32(m, lum));
                hit_count += static_cast<size_t>(__builtin_popcount(static_cast<uint32_t>(m)));
            }
#endif
            for (; rx < roi_w; ++rx) {
                if (hit_row[rx]) {
                    const uint8_t* p = rgb_row + static_cast<size_t>(rx) * 3u;
                    lum256_sum += 77u * static_cast<uint64_t>(p[0]) +
                                  150u * static_cast<uint64_t>(p[1]) +
                                  29u * static_cast<uint64_t>(p[2]);
                    ++hit_count;
                }
            }
        }
        if (hit_count == 0) {
            return 0;
        }
        const float L_mean = static_cast<float>(lum256_sum) / (256.0f * static_cast<float>(hit_count));
        const float L_base = std::clamp(0.65f * L_mean + 0.35f * 160.0f, 48.0f, 220.0f);
        inv_L_base = 1.0f / L_base;
    }

    if (wrap_repeat == VI_WRAP_REPEAT) {
        if (blend_mode == VI_BLEND_NORMAL) {
            return dispatch_mask<VI_WRAP_REPEAT, VI_BLEND_NORMAL>(
                hit_buf, u_buf, v_buf, texture_rgba, tex_w, tex_h, opacity, inv_L_base,
                frame_rgb_inout, mask_out, frame_w, frame_h, roi_x0, roi_y0, roi_w, roi_h
            );
        }
        if (blend_mode == VI_BLEND_LUMINANCE) {
            return dispatch_mask<VI_WRAP_REPEAT, VI_BLEND_LUMINANCE>(
                hit_buf, u_buf, v_buf, texture_rgba, tex_w, tex_h, opacity, inv_L_base,
                frame_rgb_inout, mask_out, frame_w, frame_h, roi_x0, roi_y0, roi_w, roi_h
            );
        }
        return dispatch_mask<VI_WRAP_REPEAT, VI_BLEND_INK>(
            hit_buf, u_buf, v_buf, texture_rgba, tex_w, tex_h, opacity, inv_L_base,
            frame_rgb_inout, mask_out, frame_w, frame_h, roi_x0, roi_y0, roi_w, roi_h
        );
    }

    if (blend_mode == VI_BLEND_NORMAL) {
        return dispatch_mask<VI_WRAP_DECAL, VI_BLEND_NORMAL>(
            hit_buf, u_buf, v_buf, texture_rgba, tex_w, tex_h, opacity, inv_L_base,
            frame_rgb_inout, mask_out, frame_w, frame_h, roi_x0, roi_y0, roi_w, roi_h
        );
    }
    if (blend_mode == VI_BLEND_LUMINANCE) {
        return dispatch_mask<VI_WRAP_DECAL, VI_BLEND_LUMINANCE>(
            hit_buf, u_buf, v_buf, texture_rgba, tex_w, tex_h, opacity, inv_L_base,
            frame_rgb_inout, mask_out, frame_w, frame_h, roi_x0, roi_y0, roi_w, roi_h
        );
    }
    return dispatch_mask<VI_WRAP_DECAL, VI_BLEND_INK>(
        hit_buf, u_buf, v_buf, texture_rgba, tex_w, tex_h, opacity, inv_L_base,
        frame_rgb_inout, mask_out, frame_w, frame_h, roi_x0, roi_y0, roi_w, roi_h
    );
}
