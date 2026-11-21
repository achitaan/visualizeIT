#ifndef VISUALIZEIT_RASTER_H
#define VISUALIZEIT_RASTER_H

#include <stdint.h>

#ifndef VI_API
#if defined(_WIN32) && defined(VI_SHARED)
#define VI_API __declspec(dllexport)
#else
#define VI_API
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum vi_raster_error {
    VI_RASTER_ERR_NULL_PTR           = -1,
    VI_RASTER_ERR_DIMENSIONS         = -2,
    VI_RASTER_ERR_INTRINSICS         = -3,
    VI_RASTER_ERR_PARAMS             = -4,
    VI_RASTER_ERR_INDEX_OUT_OF_RANGE = -5,
    VI_RASTER_ERR_NONFINITE_GEOMETRY = -6
};

enum vi_blend_mode {
    VI_BLEND_NORMAL    = 0,
    VI_BLEND_LUMINANCE = 1,
    VI_BLEND_INK       = 2
};

enum vi_wrap_mode {
    VI_WRAP_DECAL  = 0,
    VI_WRAP_REPEAT = 1
};

/* Perspective-correct z-buffered triangle mesh rasterizer and ROI compositor.
 *
 * Coordinate conventions:
 * - vertices_cam: camera-space XYZ (X right, Y down, Z forward; Z > 0 in front).
 * - Pinhole projection: x = fx * (X / Z) + cx, y = fy * (Y / Z) + cy, sampled at
 *   pixel centers (px + 0.5f, py + 0.5f).
 * - vertex_uvs: top-left image UV origin ((0,0) -> top-left texel [0,0],
 *   (1,1) -> bottom-right texel [tex_h-1, tex_w-1]).
 * - wrap_repeat: 1 = tile/repeat UV via u - floor(u); 0 = decal [0,1] (pixels
 *   outside [0,1] are transparent and leave frame_rgb_inout/mask_out untouched).
 * - mask_out: optional (may be NULL). When non-NULL, sets mask_out[idx] = 255
 *   for every pixel composited with effective alpha > 0 (leaving other entries
 *   of mask_out unchanged so callers may zero-init or accumulate across calls).
 *
 * Returns >= 0 (number of composited pixels with effective alpha > 0) on success,
 * or a negative vi_raster_error code on invalid arguments without mutating buffers.
 */
VI_API int vi_rasterize_mesh_rgbd(
    const float* vertices_cam,     /* (V * 3) camera-space XYZ (z > 0 in front) */
    int num_vertices,
    const int32_t* triangles,      /* (T * 3) vertex indices */
    int num_triangles,
    const float* vertex_uvs,       /* (V * 2) UV coordinates */
    const uint8_t* texture_rgba,   /* (tex_h * tex_w * 4) RGBA design texture */
    int tex_w,
    int tex_h,
    int wrap_repeat,               /* 1 = tile/repeat UV, 0 = clamp/decal [0,1] */
    float fx, float fy, float cx, float cy,
    float opacity,                 /* [0.0, 1.0] */
    int blend_mode,                /* 0 = normal, 1 = luminance_shading, 2 = ink */
    uint8_t* frame_rgb_inout,      /* (frame_h * frame_w * 3) RGB uint8 buffer mutated in place */
    float* depth_buf_inout,        /* (frame_h * frame_w) float32 z-buffer mutated in place */
    uint8_t* mask_out,             /* (frame_h * frame_w) uint8 rendered pixel mask (optional, may be NULL) */
    int frame_w,
    int frame_h
);

#ifdef __cplusplus
}
#endif

#endif /* VISUALIZEIT_RASTER_H */
