#include "visualizeit/raster.h"
#include "visualizeit/pose.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

namespace {

#define REQUIRE(cond, msg)                                                        \
    do {                                                                          \
        if (!(cond)) {                                                            \
            std::cerr << "FAIL [" << __func__ << ":" << __LINE__ << "]: " << msg  \
                      << "\n";                                                    \
            return false;                                                         \
        }                                                                         \
    } while (0)

struct Vec3 {
    float x, y, z;
};

inline Vec3 sub3(Vec3 a, Vec3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

inline Vec3 cross3(Vec3 a, Vec3 b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

inline float dot3(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

bool ray_triangle_uv(
    Vec3 dir,
    Vec3 a,
    Vec3 b,
    Vec3 c,
    const float* uvs,
    float& out_u,
    float& out_v
) {
    const Vec3 e1 = sub3(b, a);
    const Vec3 e2 = sub3(c, a);
    const Vec3 p = cross3(dir, e2);
    const float det = dot3(e1, p);
    if (std::fabs(det) < 1e-12f) {
        return false;
    }
    const Vec3 off = {-a.x, -a.y, -a.z};
    const float u_bary = dot3(off, p) / det;
    const Vec3 q = cross3(off, e1);
    const float v_bary = dot3(dir, q) / det;
    const float t = dot3(e2, q) / det;
    const float w_bary = 1.0f - u_bary - v_bary;
    if (t <= 0.0f || u_bary < 1e-4f || v_bary < 1e-4f || w_bary < 1e-4f) {
        return false;
    }
    out_u = w_bary * uvs[0] + u_bary * uvs[2] + v_bary * uvs[4];
    out_v = w_bary * uvs[1] + u_bary * uvs[3] + v_bary * uvs[5];
    return true;
}

bool test_invalid_arguments() {
    float verts[9] = {-0.1f, -0.1f, 1.0f, 0.1f, -0.1f, 1.0f, 0.0f, 0.1f, 1.0f};
    int32_t tris[3] = {0, 1, 2};
    float uvs[6] = {0.0f, 0.0f, 1.0f, 0.0f, 0.5f, 1.0f};
    uint8_t tex[16] = {255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 0, 255};
    std::vector<uint8_t> rgb(64 * 64 * 3, 123);
    std::vector<float> depth(64 * 64, 99.0f);

    REQUIRE(vi_rasterize_mesh_rgbd(nullptr, 3, tris, 1, uvs, tex, 2, 2, 0, 50.f, 50.f, 32.f, 32.f, 1.f, 0, rgb.data(), depth.data(), nullptr, 64, 64) == VI_RASTER_ERR_NULL_PTR, "null vertices");
    REQUIRE(vi_rasterize_mesh_rgbd(verts, 3, nullptr, 1, uvs, tex, 2, 2, 0, 50.f, 50.f, 32.f, 32.f, 1.f, 0, rgb.data(), depth.data(), nullptr, 64, 64) == VI_RASTER_ERR_NULL_PTR, "null triangles");
    REQUIRE(vi_rasterize_mesh_rgbd(verts, 3, tris, 1, nullptr, tex, 2, 2, 0, 50.f, 50.f, 32.f, 32.f, 1.f, 0, rgb.data(), depth.data(), nullptr, 64, 64) == VI_RASTER_ERR_NULL_PTR, "null uvs");
    REQUIRE(vi_rasterize_mesh_rgbd(verts, 3, tris, 1, uvs, nullptr, 2, 2, 0, 50.f, 50.f, 32.f, 32.f, 1.f, 0, rgb.data(), depth.data(), nullptr, 64, 64) == VI_RASTER_ERR_NULL_PTR, "null tex");
    REQUIRE(vi_rasterize_mesh_rgbd(verts, 3, tris, 1, uvs, tex, 2, 2, 0, 50.f, 50.f, 32.f, 32.f, 1.f, 0, nullptr, depth.data(), nullptr, 64, 64) == VI_RASTER_ERR_NULL_PTR, "null frame");
    REQUIRE(vi_rasterize_mesh_rgbd(verts, 3, tris, 1, uvs, tex, 2, 2, 0, 50.f, 50.f, 32.f, 32.f, 1.f, 0, rgb.data(), nullptr, nullptr, 64, 64) == VI_RASTER_ERR_NULL_PTR, "null depth");

    REQUIRE(vi_rasterize_mesh_rgbd(verts, 0, tris, 1, uvs, tex, 2, 2, 0, 50.f, 50.f, 32.f, 32.f, 1.f, 0, rgb.data(), depth.data(), nullptr, 64, 64) == VI_RASTER_ERR_DIMENSIONS, "zero vertices");
    REQUIRE(vi_rasterize_mesh_rgbd(verts, 3, tris, 0, uvs, tex, 2, 2, 0, 50.f, 50.f, 32.f, 32.f, 1.f, 0, rgb.data(), depth.data(), nullptr, 64, 64) == VI_RASTER_ERR_DIMENSIONS, "zero triangles");
    REQUIRE(vi_rasterize_mesh_rgbd(verts, 3, tris, 1, uvs, tex, 0, 2, 0, 50.f, 50.f, 32.f, 32.f, 1.f, 0, rgb.data(), depth.data(), nullptr, 64, 64) == VI_RASTER_ERR_DIMENSIONS, "zero tex_w");
    REQUIRE(vi_rasterize_mesh_rgbd(verts, 3, tris, 1, uvs, tex, 2, 2, 0, 50.f, 50.f, 32.f, 32.f, 1.f, 0, rgb.data(), depth.data(), nullptr, 0, 64) == VI_RASTER_ERR_DIMENSIONS, "zero frame_w");

    const float nan_val = std::numeric_limits<float>::quiet_NaN();
    REQUIRE(vi_rasterize_mesh_rgbd(verts, 3, tris, 1, uvs, tex, 2, 2, 0, 0.f, 50.f, 32.f, 32.f, 1.f, 0, rgb.data(), depth.data(), nullptr, 64, 64) == VI_RASTER_ERR_INTRINSICS, "fx <= 0");
    REQUIRE(vi_rasterize_mesh_rgbd(verts, 3, tris, 1, uvs, tex, 2, 2, 0, 50.f, -1.f, 32.f, 32.f, 1.f, 0, rgb.data(), depth.data(), nullptr, 64, 64) == VI_RASTER_ERR_INTRINSICS, "fy <= 0");
    REQUIRE(vi_rasterize_mesh_rgbd(verts, 3, tris, 1, uvs, tex, 2, 2, 0, 50.f, 50.f, nan_val, 32.f, 1.f, 0, rgb.data(), depth.data(), nullptr, 64, 64) == VI_RASTER_ERR_INTRINSICS, "nan cx");

    REQUIRE(vi_rasterize_mesh_rgbd(verts, 3, tris, 1, uvs, tex, 2, 2, 0, 50.f, 50.f, 32.f, 32.f, -0.1f, 0, rgb.data(), depth.data(), nullptr, 64, 64) == VI_RASTER_ERR_PARAMS, "opacity < 0");
    REQUIRE(vi_rasterize_mesh_rgbd(verts, 3, tris, 1, uvs, tex, 2, 2, 0, 50.f, 50.f, 32.f, 32.f, 1.1f, 0, rgb.data(), depth.data(), nullptr, 64, 64) == VI_RASTER_ERR_PARAMS, "opacity > 1");
    REQUIRE(vi_rasterize_mesh_rgbd(verts, 3, tris, 1, uvs, tex, 2, 2, 0, 50.f, 50.f, 32.f, 32.f, nan_val, 0, rgb.data(), depth.data(), nullptr, 64, 64) == VI_RASTER_ERR_PARAMS, "nan opacity");
    REQUIRE(vi_rasterize_mesh_rgbd(verts, 3, tris, 1, uvs, tex, 2, 2, 0, 50.f, 50.f, 32.f, 32.f, 1.0f, -1, rgb.data(), depth.data(), nullptr, 64, 64) == VI_RASTER_ERR_PARAMS, "blend < 0");
    REQUIRE(vi_rasterize_mesh_rgbd(verts, 3, tris, 1, uvs, tex, 2, 2, 0, 50.f, 50.f, 32.f, 32.f, 1.0f, 3, rgb.data(), depth.data(), nullptr, 64, 64) == VI_RASTER_ERR_PARAMS, "blend > 2");
    REQUIRE(vi_rasterize_mesh_rgbd(verts, 3, tris, 1, uvs, tex, 2, 2, 2, 50.f, 50.f, 32.f, 32.f, 1.0f, 0, rgb.data(), depth.data(), nullptr, 64, 64) == VI_RASTER_ERR_PARAMS, "wrap_repeat = 2");

    int32_t bad_tris_neg[3] = {0, -1, 2};
    int32_t bad_tris_hi[3] = {0, 1, 3};
    REQUIRE(vi_rasterize_mesh_rgbd(verts, 3, bad_tris_neg, 1, uvs, tex, 2, 2, 0, 50.f, 50.f, 32.f, 32.f, 1.f, 0, rgb.data(), depth.data(), nullptr, 64, 64) == VI_RASTER_ERR_INDEX_OUT_OF_RANGE, "negative index");
    REQUIRE(vi_rasterize_mesh_rgbd(verts, 3, bad_tris_hi, 1, uvs, tex, 2, 2, 0, 50.f, 50.f, 32.f, 32.f, 1.f, 0, rgb.data(), depth.data(), nullptr, 64, 64) == VI_RASTER_ERR_INDEX_OUT_OF_RANGE, "oob index");

    for (uint8_t v : rgb) REQUIRE(v == 123, "rgb mutated on error");
    for (float d : depth) REQUIRE(d == 99.0f, "depth mutated on error");

    float nan_verts[9] = {0.f, 0.f, 1.f, nan_val, 0.f, 1.f, 0.f, 1.f, 1.f};
    float inf_uvs[6] = {0.f, 0.f, std::numeric_limits<float>::infinity(), 0.f, 0.5f, 1.f};
    REQUIRE(vi_rasterize_mesh_rgbd(nan_verts, 3, tris, 1, uvs, tex, 2, 2, 0, 50.f, 50.f, 32.f, 32.f, 1.f, 0, rgb.data(), depth.data(), nullptr, 64, 64) == VI_RASTER_ERR_NONFINITE_GEOMETRY, "nan vert");
    REQUIRE(vi_rasterize_mesh_rgbd(verts, 3, tris, 1, inf_uvs, tex, 2, 2, 0, 50.f, 50.f, 32.f, 32.f, 1.f, 0, rgb.data(), depth.data(), nullptr, 64, 64) == VI_RASTER_ERR_NONFINITE_GEOMETRY, "inf uv");

    return true;
}

bool test_zbuffer_order_and_winding_invariance() {
    const int W = 128, H = 128;
    // Texture: left half pure red (u in [0, 0.4]), right half pure blue (u in [0.6, 1.0])
    uint8_t tex[2 * 4 * 4] = {};
    for (int y = 0; y < 2; ++y) {
        for (int x = 0; x < 4; ++x) {
            uint8_t* p = tex + (y * 4 + x) * 4;
            if (x < 2) {
                p[0] = 240; p[1] = 20; p[2] = 20; p[3] = 255;
            } else {
                p[0] = 20; p[1] = 40; p[2] = 240; p[3] = 255;
            }
        }
    }

    // Front triangle at z = 1.0 (samples red half), back larger triangle at z = 2.0 (samples blue half)
    const float verts[18] = {
        -0.25f, -0.25f, 1.0f,  0.25f, -0.25f, 1.0f,  0.0f, 0.25f, 1.0f,
        -0.80f, -0.80f, 2.0f,  0.80f, -0.80f, 2.0f,  0.0f, 0.80f, 2.0f
    };
    const float uvs[12] = {
        0.1f, 0.2f,  0.2f, 0.2f,  0.15f, 0.8f,
        0.8f, 0.2f,  0.9f, 0.2f,  0.85f, 0.8f
    };

    const int32_t tris_fb[6] = {0, 1, 2, 3, 4, 5};
    const int32_t tris_bf[6] = {3, 4, 5, 0, 1, 2};
    const int32_t tris_cw[6] = {0, 2, 1, 3, 5, 4};

    auto run = [&](const int32_t* t_idx, std::vector<uint8_t>& rgb, std::vector<float>& zbuf) {
        rgb.assign(W * H * 3, 100);
        zbuf.assign(W * H, std::numeric_limits<float>::infinity());
        return vi_rasterize_mesh_rgbd(
            verts, 6, t_idx, 2, uvs, tex, 4, 2,
            VI_WRAP_DECAL, 120.f, 120.f, 64.f, 64.f,
            0.8f, VI_BLEND_NORMAL, rgb.data(), zbuf.data(), nullptr, W, H
        );
    };

    std::vector<uint8_t> rgb1, rgb2, rgb3;
    std::vector<float> z1, z2, z3;
    const int n1 = run(tris_fb, rgb1, z1);
    const int n2 = run(tris_bf, rgb2, z2);
    const int n3 = run(tris_cw, rgb3, z3);

    REQUIRE(n1 > 200 && n1 == n2 && n1 == n3, "rendered pixel counts must match across order & winding");
    REQUIRE(rgb1 == rgb2 && rgb1 == rgb3, "RGB output must be bit-for-bit order & winding invariant");
    REQUIRE(z1 == z2 && z1 == z3, "Z-buffer must be bit-for-bit order & winding invariant");

    // Center pixel (64, 60) is covered by front red triangle (0.8 * 240 + 0.2 * 100 = 212)
    const size_t c_idx = (60 * W + 64) * 3;
    REQUIRE(rgb1[c_idx + 0] == 212 && rgb1[c_idx + 2] == 36, "overlap center must show single-blended front red");
    return true;
}

bool test_perspective_correct_uv_vs_ray_oracle() {
    const int W = 160, H = 120;
    const int TW = 256, TH = 256;
    std::vector<uint8_t> tex(TW * TH * 4);
    for (int y = 0; y < TH; ++y) {
        for (int x = 0; x < TW; ++x) {
            uint8_t* p = tex.data() + (y * TW + x) * 4;
            p[0] = static_cast<uint8_t>(x);
            p[1] = static_cast<uint8_t>(y);
            p[2] = 128;
            p[3] = 255;
        }
    }

    const float verts[9] = {
        -0.35f, -0.25f, 0.8f,
         0.95f, -0.30f, 2.8f,
        -0.10f,  0.45f, 1.4f
    };
    const float uvs[6] = {0.05f, 0.10f, 0.95f, 0.20f, 0.30f, 0.90f};
    const int32_t tris[3] = {0, 1, 2};

    const float fx = 140.f, fy = 140.f, cx = 80.f, cy = 60.f;
    std::vector<uint8_t> rgb(W * H * 3, 0);
    std::vector<float> zbuf(W * H, std::numeric_limits<float>::infinity());
    std::vector<uint8_t> mask(W * H, 0);

    const int n = vi_rasterize_mesh_rgbd(
        verts, 3, tris, 1, uvs, tex.data(), TW, TH,
        VI_WRAP_DECAL, fx, fy, cx, cy,
        1.0f, VI_BLEND_NORMAL, rgb.data(), zbuf.data(), mask.data(), W, H
    );
    REQUIRE(n > 400, "tilted triangle must render interior pixels");

    const Vec3 va{verts[0], verts[1], verts[2]};
    const Vec3 vb{verts[3], verts[4], verts[5]};
    const Vec3 vc{verts[6], verts[7], verts[8]};

    int checked = 0;
    for (int py = 2; py < H - 2; ++py) {
        for (int px = 2; px < W - 2; ++px) {
            if (!mask[py * W + px]) continue;
            Vec3 ray{(px + 0.5f - cx) / fx, (py + 0.5f - cy) / fy, 1.0f};
            float ou = 0.f, ov = 0.f;
            if (!ray_triangle_uv(ray, va, vb, vc, uvs, ou, ov)) continue;
            const int exp_r = static_cast<int>(std::round(ou * 255.0f));
            const int exp_g = static_cast<int>(std::round(ov * 255.0f));
            const uint8_t* p = rgb.data() + (py * W + px) * 3;
            REQUIRE(std::abs(static_cast<int>(p[0]) - exp_r) <= 2, "perspective U mismatch vs Moller-Trumbore");
            REQUIRE(std::abs(static_cast<int>(p[1]) - exp_g) <= 2, "perspective V mismatch vs Moller-Trumbore");
            ++checked;
        }
    }
    REQUIRE(checked > 300, "sufficient interior oracle rays checked");
    return true;
}

bool test_shared_edge_quad_no_crack_or_seam() {
    const int W = 128, H = 128;
    const float verts[12] = {
        -0.3f, -0.3f, 1.0f,
         0.3f, -0.3f, 1.0f,
         0.3f,  0.3f, 1.0f,
        -0.3f,  0.3f, 1.0f
    };
    const float uvs[8] = {0.f, 0.f, 1.f, 0.f, 1.f, 1.f, 0.f, 1.f};
    const int32_t tris[6] = {0, 1, 2, 0, 2, 3};
    const uint8_t tex[16] = {
        200, 100, 50, 255,  200, 100, 50, 255,
        200, 100, 50, 255,  200, 100, 50, 255
    };

    std::vector<uint8_t> rgb(W * H * 3, 50);
    std::vector<float> zbuf(W * H, std::numeric_limits<float>::infinity());
    std::vector<uint8_t> mask(W * H, 0);

    const int n = vi_rasterize_mesh_rgbd(
        verts, 4, tris, 2, uvs, tex, 2, 2,
        VI_WRAP_DECAL, 100.f, 100.f, 64.f, 64.f,
        0.6f, VI_BLEND_NORMAL, rgb.data(), zbuf.data(), mask.data(), W, H
    );
    REQUIRE(n >= 3600, "quad pixel count");

    const uint8_t ref_r = rgb[(64 * W + 64) * 3 + 0];
    const uint8_t ref_g = rgb[(64 * W + 64) * 3 + 1];
    const uint8_t ref_b = rgb[(64 * W + 64) * 3 + 2];

    for (int py = 36; py <= 91; ++py) {
        for (int px = 36; px <= 91; ++px) {
            const size_t idx = py * W + px;
            REQUIRE(mask[idx] == 255, "shared diagonal crack detected");
            REQUIRE(rgb[idx * 3 + 0] == ref_r &&
                    rgb[idx * 3 + 1] == ref_g &&
                    rgb[idx * 3 + 2] == ref_b,
                    "shared diagonal double-blend seam detected");
        }
    }
    return true;
}

bool test_near_plane_clipping() {
    const int W = 96, H = 96;
    const uint8_t tex[16] = {
        255, 128, 64, 255,  255, 128, 64, 255,
        255, 128, 64, 255,  255, 128, 64, 255
    };
    const float uvs[6] = {0.f, 0.f, 1.f, 0.f, 0.5f, 1.f};
    const int32_t tris[3] = {0, 1, 2};

    // 1. Completely behind camera
    const float behind_verts[9] = {-0.2f, -0.2f, -0.5f, 0.2f, -0.2f, -0.5f, 0.0f, 0.2f, -0.5f};
    std::vector<uint8_t> rgb(W * H * 3, 77);
    std::vector<float> zbuf(W * H, 100.0f);
    REQUIRE(vi_rasterize_mesh_rgbd(
        behind_verts, 3, tris, 1, uvs, tex, 2, 2, 0,
        80.f, 80.f, 48.f, 48.f, 1.f, 0, rgb.data(), zbuf.data(), nullptr, W, H
    ) == 0, "all-behind triangle must render 0 pixels");

    // 2. 1 vertex behind, 2 in front
    const float clip1_verts[9] = {-0.2f, -0.2f, 1.0f, 0.2f, -0.2f, 1.0f, 0.0f, 0.1f, -0.4f};
    zbuf.assign(W * H, std::numeric_limits<float>::infinity());
    const int n1 = vi_rasterize_mesh_rgbd(
        clip1_verts, 3, tris, 1, uvs, tex, 2, 2, 0,
        80.f, 80.f, 48.f, 48.f, 1.f, 0, rgb.data(), zbuf.data(), nullptr, W, H
    );
    REQUIRE(n1 > 0, "1-behind straddling triangle must render > 0 pixels");
    for (float d : zbuf) {
        if (std::isfinite(d)) {
            REQUIRE(d >= 1e-4f, "clipped depth must satisfy near plane");
        }
    }

    // 3. 2 vertices behind, 1 in front
    const float clip2_verts[9] = {0.0f, 0.0f, 1.0f, -0.2f, -0.1f, -0.4f, 0.2f, -0.1f, -0.4f};
    zbuf.assign(W * H, std::numeric_limits<float>::infinity());
    const int n2 = vi_rasterize_mesh_rgbd(
        clip2_verts, 3, tris, 1, uvs, tex, 2, 2, 0,
        80.f, 80.f, 48.f, 48.f, 1.f, 0, rgb.data(), zbuf.data(), nullptr, W, H
    );
    REQUIRE(n2 > 0, "2-behind straddling triangle must render > 0 pixels");
    return true;
}

bool test_wrap_repeat_vs_decal() {
    const int W = 128, H = 128;
    const float verts[12] = {
        -0.4f, -0.4f, 1.0f,
         0.4f, -0.4f, 1.0f,
         0.4f,  0.4f, 1.0f,
        -0.4f,  0.4f, 1.0f
    };
    const float uvs[8] = {-0.5f, -0.5f, 1.5f, -0.5f, 1.5f, 1.5f, -0.5f, 1.5f};
    const int32_t tris[6] = {0, 1, 2, 0, 2, 3};
    const uint8_t tex[16] = {
        250, 50, 50, 255,  50, 250, 50, 255,
        50, 50, 250, 255,  250, 250, 50, 255
    };

    std::vector<uint8_t> rgb_decal(W * H * 3, 90);
    std::vector<float> z_decal(W * H, std::numeric_limits<float>::infinity());
    const int n_decal = vi_rasterize_mesh_rgbd(
        verts, 4, tris, 2, uvs, tex, 2, 2,
        VI_WRAP_DECAL, 100.f, 100.f, 64.f, 64.f,
        1.0f, VI_BLEND_NORMAL, rgb_decal.data(), z_decal.data(), nullptr, W, H
    );

    std::vector<uint8_t> rgb_rep(W * H * 3, 90);
    std::vector<float> z_rep(W * H, std::numeric_limits<float>::infinity());
    const int n_rep = vi_rasterize_mesh_rgbd(
        verts, 4, tris, 2, uvs, tex, 2, 2,
        VI_WRAP_REPEAT, 100.f, 100.f, 64.f, 64.f,
        1.0f, VI_BLEND_NORMAL, rgb_rep.data(), z_rep.data(), nullptr, W, H
    );

    REQUIRE(n_decal > 400, "decal central region rendered");
    REQUIRE(n_rep > 3 * n_decal, "repeat mode tiles full [-0.5, 1.5]^2 quad");
    // Outer quad pixel (30, 30) is outside [0,1]^2 decal, so decal leaves it at 90
    REQUIRE(rgb_decal[(30 * W + 30) * 3] == 90, "decal outside [0,1] untouched");
    REQUIRE(rgb_rep[(30 * W + 30) * 3] != 90, "repeat outside [0,1] textured");
    return true;
}

bool test_blend_modes() {
    const int W = 128, H = 64;
    const float verts[12] = {
        -0.5f, -0.2f, 1.0f,
         0.5f, -0.2f, 1.0f,
         0.5f,  0.2f, 1.0f,
        -0.5f,  0.2f, 1.0f
    };
    const float uvs[8] = {0.f, 0.f, 1.f, 0.f, 1.f, 1.f, 0.f, 1.f};
    const int32_t tris[6] = {0, 1, 2, 0, 2, 3};
    const uint8_t tex[16] = {
        220, 120, 40, 255,  220, 120, 40, 255,
        220, 120, 40, 255,  220, 120, 40, 255
    };

    auto make_split_bg = [&]() {
        std::vector<uint8_t> bg(W * H * 3);
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                const uint8_t v = (x < W / 2) ? 70 : 215;
                bg[(y * W + x) * 3 + 0] = v;
                bg[(y * W + x) * 3 + 1] = v;
                bg[(y * W + x) * 3 + 2] = v;
            }
        }
        return bg;
    };

    auto run_mode = [&](int mode) {
        auto bg = make_split_bg();
        std::vector<float> zbuf(W * H, std::numeric_limits<float>::infinity());
        vi_rasterize_mesh_rgbd(
            verts, 4, tris, 2, uvs, tex, 2, 2, 0,
            100.f, 100.f, 64.f, 32.f, 1.0f, mode,
            bg.data(), zbuf.data(), nullptr, W, H
        );
        return bg;
    };

    const auto out_norm = run_mode(VI_BLEND_NORMAL);
    const size_t l_idx = (32 * W + 40) * 3;
    const size_t r_idx = (32 * W + 88) * 3;
    REQUIRE(out_norm[l_idx] == 220 && out_norm[r_idx] == 220, "normal mode ignores background luminance");

    const auto out_lum = run_mode(VI_BLEND_LUMINANCE);
    REQUIRE(out_lum[r_idx + 0] > out_lum[l_idx + 0] &&
            out_lum[r_idx + 1] > out_lum[l_idx + 1],
            "luminance blend preserves scene brightness gradient");

    const auto out_ink = run_mode(VI_BLEND_INK);
    REQUIRE(out_ink[l_idx + 0] < 70 && out_ink[r_idx + 0] < 215 &&
            out_ink[r_idx + 0] > out_ink[l_idx + 0],
            "ink blend darkens substrate proportionally");
    return true;
}

bool test_mask_null_and_alpha_cutout() {
    const int W = 100, H = 100;
    const float verts[12] = {
        -0.3f, -0.3f, 1.0f,
         0.3f, -0.3f, 1.0f,
         0.3f,  0.3f, 1.0f,
        -0.3f,  0.3f, 1.0f
    };
    const float uvs[8] = {0.f, 0.f, 1.f, 0.f, 1.f, 1.f, 0.f, 1.f};
    const int32_t tris[6] = {0, 1, 2, 0, 2, 3};

    // 16x16 texture: left 8 cols alpha=0, right 8 cols alpha=255
    std::vector<uint8_t> tex(16 * 16 * 4, 0);
    for (int y = 0; y < 16; ++y) {
        for (int x = 8; x < 16; ++x) {
            uint8_t* p = tex.data() + (y * 16 + x) * 4;
            p[0] = 200; p[1] = 80; p[2] = 40; p[3] = 255;
        }
    }

    std::vector<uint8_t> rgb1(W * H * 3, 120), rgb2(W * H * 3, 120);
    std::vector<float> z1(W * H, std::numeric_limits<float>::infinity());
    std::vector<float> z2(W * H, std::numeric_limits<float>::infinity());
    std::vector<uint8_t> mask(W * H, 0);

    const int n1 = vi_rasterize_mesh_rgbd(
        verts, 4, tris, 2, uvs, tex.data(), 16, 16, 0,
        100.f, 100.f, 50.f, 50.f, 1.0f, VI_BLEND_NORMAL,
        rgb1.data(), z1.data(), nullptr, W, H
    );
    const int n2 = vi_rasterize_mesh_rgbd(
        verts, 4, tris, 2, uvs, tex.data(), 16, 16, 0,
        100.f, 100.f, 50.f, 50.f, 1.0f, VI_BLEND_NORMAL,
        rgb2.data(), z2.data(), mask.data(), W, H
    );

    REQUIRE(n1 == n2 && n1 > 1000, "mask_out nullptr vs non-null pixel count parity");
    REQUIRE(mask[50 * W + 30] == 0, "transparent half must have mask 0");
    REQUIRE(mask[50 * W + 70] == 255, "opaque half must have mask 255");
    return true;
}

bool test_pose_abi_coexistence() {
    REQUIRE(vi_pose_abi_version() == 1u, "vi_pose_abi_version must remain 1");
    return true;
}

bool test_hd_1280x720_performance_gate() {
    const int W = 1280, H = 720;
    const int grid_n = 32;
    const int num_verts = (grid_n + 1) * (grid_n + 1);
    const int num_tris = grid_n * grid_n * 2;

    std::vector<float> verts(num_verts * 3);
    std::vector<float> uvs(num_verts * 2);
    for (int gy = 0; gy <= grid_n; ++gy) {
        for (int gx = 0; gx <= grid_n; ++gx) {
            const float u = static_cast<float>(gx) / static_cast<float>(grid_n);
            const float v = static_cast<float>(gy) / static_cast<float>(grid_n);
            const float x = (u - 0.5f) * 0.78f;
            const float y = (v - 0.5f) * 0.46f;
            const float z = 1.0f + 0.08f * std::cos((u - 0.5f) * 3.14159f);
            const int idx = gy * (grid_n + 1) + gx;
            verts[idx * 3 + 0] = x;
            verts[idx * 3 + 1] = y;
            verts[idx * 3 + 2] = z;
            uvs[idx * 2 + 0] = u;
            uvs[idx * 2 + 1] = v;
        }
    }

    std::vector<int32_t> tris;
    tris.reserve(num_tris * 3);
    for (int gy = 0; gy < grid_n; ++gy) {
        for (int gx = 0; gx < grid_n; ++gx) {
            const int i0 = gy * (grid_n + 1) + gx;
            const int i1 = i0 + 1;
            const int i2 = i0 + (grid_n + 1) + 1;
            const int i3 = i0 + (grid_n + 1);
            tris.push_back(i0); tris.push_back(i1); tris.push_back(i2);
            tris.push_back(i0); tris.push_back(i2); tris.push_back(i3);
        }
    }

    const int TW = 512, TH = 512;
    std::vector<uint8_t> tex(TW * TH * 4);
    for (int i = 0; i < TW * TH; ++i) {
        tex[i * 4 + 0] = static_cast<uint8_t>((i * 7) & 255);
        tex[i * 4 + 1] = static_cast<uint8_t>((i * 13) & 255);
        tex[i * 4 + 2] = static_cast<uint8_t>((i * 29) & 255);
        tex[i * 4 + 3] = 255;
    }

    std::vector<uint8_t> base_rgb(W * H * 3);
    for (int i = 0; i < W * H * 3; ++i) {
        base_rgb[i] = static_cast<uint8_t>(110 + (i % 70));
    }
    std::vector<uint8_t> frame_rgb = base_rgb;
    std::vector<float> zbuf(W * H, std::numeric_limits<float>::infinity());

    // 5 warmup runs
    for (int i = 0; i < 5; ++i) {
        std::fill(zbuf.begin(), zbuf.end(), std::numeric_limits<float>::infinity());
        const int px = vi_rasterize_mesh_rgbd(
            verts.data(), num_verts, tris.data(), num_tris, uvs.data(),
            tex.data(), TW, TH, VI_WRAP_DECAL,
            1000.f, 1000.f, 640.f, 360.f, 0.9f, VI_BLEND_LUMINANCE,
            frame_rgb.data(), zbuf.data(), nullptr, W, H
        );
        REQUIRE(px > 200000, "HD mesh must cover substantial viewport area");
    }

    const int iters = 30;
    std::vector<double> samples_ms;
    samples_ms.reserve(iters);

    for (int i = 0; i < iters; ++i) {
        frame_rgb = base_rgb;
        std::fill(zbuf.begin(), zbuf.end(), std::numeric_limits<float>::infinity());
        const auto t0 = std::chrono::steady_clock::now();
        const int px = vi_rasterize_mesh_rgbd(
            verts.data(), num_verts, tris.data(), num_tris, uvs.data(),
            tex.data(), TW, TH, VI_WRAP_DECAL,
            1000.f, 1000.f, 640.f, 360.f, 0.9f, VI_BLEND_LUMINANCE,
            frame_rgb.data(), zbuf.data(), nullptr, W, H
        );
        const auto t1 = std::chrono::steady_clock::now();
        REQUIRE(px > 200000, "HD mesh rendered pixels");
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        samples_ms.push_back(ms);
    }

    std::sort(samples_ms.begin(), samples_ms.end());
    const double median_ms = samples_ms[iters / 2];
    const double mean_ms = std::accumulate(samples_ms.begin(), samples_ms.end(), 0.0) / iters;

    std::cout << "[PERF] 1280x720 2048-tri luminance raster: median=" << median_ms
              << " ms, mean=" << mean_ms << " ms\n";

    REQUIRE(median_ms <= 4.0, "1280x720 median raster latency exceeded 4.0 ms");
    REQUIRE(mean_ms <= 4.0, "1280x720 mean raster latency exceeded 4.0 ms");
    return true;
}

bool test_finite_uv_perspective_overflow_no_crash() {
    const uint8_t tex[4 * 4 * 4] = {
        200, 100, 50, 255,  200, 100, 50, 255,  200, 100, 50, 255,  200, 100, 50, 255,
        200, 100, 50, 255,  200, 100, 50, 255,  200, 100, 50, 255,  200, 100, 50, 255,
        200, 100, 50, 255,  200, 100, 50, 255,  200, 100, 50, 255,  200, 100, 50, 255,
        200, 100, 50, 255,  200, 100, 50, 255,  200, 100, 50, 255,  200, 100, 50, 255
    };
    const int32_t tris[3] = {0, 1, 2};

    // Case 1: 1e37f UVs with z = 0.001f on 17x17 (scalar tail + Pass 1 overflow)
    {
        const int W = 17, H = 17;
        const float verts[9] = {-0.5f, -0.5f, 0.001f, 0.5f, -0.5f, 0.001f, 0.0f, 0.5f, 0.001f};
        const float uvs[6] = {1e37f, 1e37f, -1e37f, 1e37f, 0.0f, -1e37f};
        for (int wrap : {VI_WRAP_DECAL, VI_WRAP_REPEAT}) {
            std::vector<uint8_t> rgb(W * H * 3, 77);
            std::vector<float> zbuf(W * H, std::numeric_limits<float>::infinity());
            const int px = vi_rasterize_mesh_rgbd(
                verts, 3, tris, 1, uvs, tex, 4, 4, wrap,
                10.f, 10.f, 8.f, 8.f, 1.0f, VI_BLEND_NORMAL,
                rgb.data(), zbuf.data(), nullptr, W, H
            );
            REQUIRE(px == 0, "1e37f UV overflow must render 0 pixels safely");
            for (uint8_t b : rgb) REQUIRE(b == 77, "rgb must remain untouched on UV overflow");
        }
    }

    // Case 2: 3e38f UVs with z = 0.5f on 33x33 (AVX-512 + scalar tail)
    {
        const int W = 33, H = 33;
        const float verts[9] = {-0.25f, -0.25f, 0.5f, 0.25f, -0.25f, 0.5f, 0.0f, 0.25f, 0.5f};
        const float uvs[6] = {3e38f, 3e38f, 3e38f, 3e38f, 3e38f, 3e38f};
        for (int wrap : {VI_WRAP_DECAL, VI_WRAP_REPEAT}) {
            std::vector<uint8_t> rgb(W * H * 3, 100);
            std::vector<float> zbuf(W * H, std::numeric_limits<float>::infinity());
            const int px = vi_rasterize_mesh_rgbd(
                verts, 3, tris, 1, uvs, tex, 4, 4, wrap,
                40.f, 40.f, 16.5f, 16.5f, 1.0f, VI_BLEND_LUMINANCE,
                rgb.data(), zbuf.data(), nullptr, W, H
            );
            REQUIRE(px == 0, "3e38f UV overflow must render 0 pixels safely");
            for (uint8_t b : rgb) REQUIRE(b == 100, "rgb must remain untouched on 3e38f UV overflow");
        }
    }

    // Case 3: Finite uq_i & duq_dx in setup, but uq * z overflows float32 to +Inf in Pass 1 -> Pass 2 guard
    {
        const int W = 35, H = 35;
        const float verts[9] = {-0.4f, -0.4f, 1.0f, 0.9f, -0.4f, 3.0f, 0.0f, 0.8f, 2.5f};
        const float uvs[6] = {2.0e38f, 2.0e38f, 2.0e38f, 2.0e38f, 2.0e38f, 2.0e38f};
        for (int wrap : {VI_WRAP_DECAL, VI_WRAP_REPEAT}) {
            std::vector<uint8_t> rgb(W * H * 3, 88);
            std::vector<float> zbuf(W * H, std::numeric_limits<float>::infinity());
            const int px = vi_rasterize_mesh_rgbd(
                verts, 3, tris, 1, uvs, tex, 4, 4, wrap,
                30.f, 30.f, 17.5f, 17.5f, 1.0f, VI_BLEND_NORMAL,
                rgb.data(), zbuf.data(), nullptr, W, H
            );
            REQUIRE(px >= 0, "Pass-2 Inf UV guard must not crash");
        }
    }
    return true;
}

bool test_hd_frustum_clipping_guard_band_and_zero_edge_bloat() {
    const int W = 1280, H = 720;
    const float fx = 1000.f, fy = 1000.f, cx = 640.f, cy = 360.f;
    const int32_t tris[3] = {0, 1, 2};
    const uint8_t solid_tex[16] = {
        240, 120, 60, 255,  240, 120, 60, 255,
        240, 120, 60, 255,  240, 120, 60, 255
    };
    const float uvs[6] = {0.1f, 0.1f, 0.9f, 0.1f, 0.5f, 0.9f};

    // 1. Off-axis HD tabletop straddling near plane (2 vertices in front at z=1.5, 1 behind at z=-0.5)
    {
        const float v_straddle[9] = {
            -0.4f, 0.20f,  1.5f,
             0.4f, 0.20f,  1.5f,
             0.0f, 0.20f, -0.5f
        };
        std::vector<uint8_t> rgb(W * H * 3, 0);
        std::vector<float> zbuf(W * H, std::numeric_limits<float>::infinity());
        const int px = vi_rasterize_mesh_rgbd(
            v_straddle, 3, tris, 1, uvs, solid_tex, 2, 2, VI_WRAP_REPEAT,
            fx, fy, cx, cy, 1.0f, VI_BLEND_NORMAL,
            rgb.data(), zbuf.data(), nullptr, W, H
        );
        REQUIRE(px > 140000 && px < 152000, "HD near-straddling tabletop must render ~146,756 pixels");
    }

    // 2. Off-axis HD straddle with 1 vertex in front, 2 behind, and z == 0.0 vertex
    {
        const float v_1front[9] = {0.0f, 0.20f, 1.5f, -0.5f, 0.20f, -0.5f, 0.5f, 0.20f, -0.5f};
        std::vector<uint8_t> rgb(W * H * 3, 0);
        std::vector<float> zbuf(W * H, std::numeric_limits<float>::infinity());
        const int px1 = vi_rasterize_mesh_rgbd(
            v_1front, 3, tris, 1, uvs, solid_tex, 2, 2, VI_WRAP_REPEAT,
            fx, fy, cx, cy, 1.0f, VI_BLEND_NORMAL,
            rgb.data(), zbuf.data(), nullptr, W, H
        );
        REQUIRE(px1 > 90000, "HD 1-front near-straddling triangle must render > 90,000 pixels");

        const float v_z0[9] = {-0.4f, 0.20f, 1.5f, 0.4f, 0.20f, 1.5f, 0.0f, 0.20f, 0.0f};
        std::fill(zbuf.begin(), zbuf.end(), std::numeric_limits<float>::infinity());
        const int px0 = vi_rasterize_mesh_rgbd(
            v_z0, 3, tris, 1, uvs, solid_tex, 2, 2, VI_WRAP_REPEAT,
            fx, fy, cx, cy, 1.0f, VI_BLEND_NORMAL,
            rgb.data(), zbuf.data(), nullptr, W, H
        );
        REQUIRE(px0 > 120000, "HD z=0 touching triangle must render > 120,000 pixels");
    }

    // 3. Huge 1e8 projected guard-band coordinate (all z >= 1e-4)
    {
        const float v_guard[9] = {
            -0.2f, -0.1f, 1.0f,
             0.2f, -0.1f, 1.0f,
             0.0f, 10.0f, 1e-4f
        };
        std::vector<uint8_t> rgb(W * H * 3, 0);
        std::vector<float> zbuf(W * H, std::numeric_limits<float>::infinity());
        const int px_g = vi_rasterize_mesh_rgbd(
            v_guard, 3, tris, 1, uvs, solid_tex, 2, 2, VI_WRAP_REPEAT,
            fx, fy, cx, cy, 1.0f, VI_BLEND_NORMAL,
            rgb.data(), zbuf.data(), nullptr, W, H
        );
        REQUIRE(px_g > 100000, "1e8 guard-band triangle must clip cleanly and render > 100,000 pixels");
    }

    // 4. Zero silhouette edge bloat when 3rd vertex projects to x = 800,000
    {
        // Edge (v0, v1) projects to (280, 200) -> (320, 400); at y = 300.5, exact edge x is 300.1
        const float v_bloat[9] = {
            (280.f - cx) / fx, (200.f - cy) / fy, 1.0f,
            (320.f - cx) / fx, (400.f - cy) / fy, 1.0f,
            (800000.f - cx) / fx, (300.f - cy) / fy, 1.0f
        };
        std::vector<uint8_t> rgb(W * H * 3, 0);
        std::vector<float> zbuf(W * H, std::numeric_limits<float>::infinity());
        std::vector<uint8_t> mask(W * H, 0);
        const int px_b = vi_rasterize_mesh_rgbd(
            v_bloat, 3, tris, 1, uvs, solid_tex, 2, 2, VI_WRAP_REPEAT,
            fx, fy, cx, cy, 1.0f, VI_BLEND_NORMAL,
            rgb.data(), zbuf.data(), mask.data(), W, H
        );
        REQUIRE(px_b > 50000, "bloat-probe triangle must render interior");
        int first_px = -1;
        for (int x = 0; x < W; ++x) {
            if (mask[300 * W + x]) {
                first_px = x;
                break;
            }
        }
        REQUIRE(first_px == 300, "first rendered pixel at row 300 must be exactly 300 (zero 8-px bloat)");
    }

    return true;
}

} // namespace

int main() {
    bool ok = true;
    ok = ok && test_invalid_arguments();
    ok = ok && test_zbuffer_order_and_winding_invariance();
    ok = ok && test_perspective_correct_uv_vs_ray_oracle();
    ok = ok && test_shared_edge_quad_no_crack_or_seam();
    ok = ok && test_near_plane_clipping();
    ok = ok && test_wrap_repeat_vs_decal();
    ok = ok && test_blend_modes();
    ok = ok && test_mask_null_and_alpha_cutout();
    ok = ok && test_pose_abi_coexistence();
    ok = ok && test_finite_uv_perspective_overflow_no_crash();
    ok = ok && test_hd_frustum_clipping_guard_band_and_zero_edge_bloat();
    ok = ok && test_hd_1280x720_performance_gate();

    if (!ok) {
        return 1;
    }
    std::cout << "PASS: all 12 rasterizer C++ unit & performance suites passed.\n";
    return 0;
}
