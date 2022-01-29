#include "../microfacet.h"

Spectrum eval_op::operator()(const DisneyMetal &bsdf) const {
    if (dot(vertex.geometry_normal, dir_in) < 0 ||
            dot(vertex.geometry_normal, dir_out) < 0) {
        // No light below the surface
        return make_zero_spectrum();
    }
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) < 0) {
        frame = -frame;
    }
    // init
    Vector3 half_vector = normalize(dir_in + dir_out);
    Real n_dot_h = dot(frame.n, half_vector);
    Real n_dot_in = dot(frame.n, dir_in);
    Real n_dot_out = dot(frame.n, dir_out);
    if (n_dot_out <= 0 || n_dot_h <= 0) {
        return make_zero_spectrum();
    }
    Spectrum base_color = eval(bsdf.base_color, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real roughness = eval(bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real anisotropic = eval(bsdf.anisotropic, vertex.uv, vertex.uv_screen_size, texture_pool);
    // Clamp roughness to avoid numerical issues.
    roughness = std::clamp(roughness, Real(0.01), Real(1));
    Spectrum F_m = schlick_fresnel(base_color, std::fabs(dot(half_vector, dir_out)));

    Vector3 local_half_vector = to_local(frame, half_vector);
    Real hlx = local_half_vector.x;
    Real hly = local_half_vector.y;
    Real hlz = local_half_vector.z;
    Real aspect = std::sqrt(1.0 - 0.9 * anisotropic);
    Real alpha_x = std::max(0.0001, roughness * roughness / aspect);
    Real alpha_y = std::max(0.0001, roughness * roughness * aspect);
    Real D_m = 1.0 / (c_PI * alpha_x * alpha_y * std::pow((hlx * hlx / (alpha_x * alpha_x) + hly * hly / (alpha_y * alpha_y) + hlz * hlz), 2.0));


    auto G = [&frame, alpha_x, alpha_y](const Vector3& w) {
        const Vector3 local_w = to_local(frame, w);
        Real wx = local_w.x;
        Real wy = local_w.y;
        Real wz = local_w.z;
        Real lambda = (std::sqrt(1 + ((wx * wx * alpha_x * alpha_x) + (wy * wy * alpha_y * alpha_y)) / (wz * wz)) - 1) * 0.5;
        return 1.0 / (1 + lambda);
    };

    Real G_m = G(dir_in) * G(dir_out);
    return F_m * D_m * G_m / (4 * std::fabs(n_dot_in));
}

Real pdf_sample_bsdf_op::operator()(const DisneyMetal &bsdf) const {
    if (dot(vertex.geometry_normal, dir_in) < 0 ||
            dot(vertex.geometry_normal, dir_out) < 0) {
        // No light below the surface
        return 0;
    }
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) < 0) {
        frame = -frame;
    }
    // init
    Vector3 half_vector = normalize(dir_in + dir_out);
    Real n_dot_h = dot(frame.n, half_vector);
    Real n_dot_in = dot(frame.n, dir_in);
    Real n_dot_out = dot(frame.n, dir_out);
    if (n_dot_out <= 0 || n_dot_h <= 0) {
        return 0;
    }
    Spectrum base_color = eval(bsdf.base_color, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real roughness = eval(bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real anisotropic = eval(bsdf.anisotropic, vertex.uv, vertex.uv_screen_size, texture_pool);
    // Clamp roughness to avoid numerical issues.
    roughness = std::clamp(roughness, Real(0.01), Real(1));

    Vector3 local_half_vector = to_local(frame, half_vector);
    Real hlx = local_half_vector.x;
    Real hly = local_half_vector.y;
    Real hlz = local_half_vector.z;
    Real aspect = std::sqrt(1.0 - 0.9 * anisotropic);
    Real alpha_x = std::max(0.0001, roughness * roughness / aspect);
    Real alpha_y = std::max(0.0001, roughness * roughness * aspect);
    Real D_m = 1.0 / (c_PI * alpha_x * alpha_y * std::pow((hlx * hlx / (alpha_x * alpha_x) + hly * hly / (alpha_y * alpha_y) + hlz * hlz), 2.0));

    auto G = [&frame, alpha_x, alpha_y](const Vector3& w) {
        const Vector3 local_w = to_local(frame, w);
        Real wx = local_w.x;
        Real wy = local_w.y;
        Real wz = local_w.z;
        Real lambda = (std::sqrt(1 + ((wx * wx * alpha_x * alpha_x) + (wy * wy * alpha_y * alpha_y)) / (wz * wz)) - 1) * 0.5;
        return 1.0 / (1 + lambda);
    };

    return D_m * G(dir_in) / (4 * std::fabs(n_dot_in));
}

std::optional<BSDFSampleRecord>
        sample_bsdf_op::operator()(const DisneyMetal &bsdf) const {
    if (dot(vertex.geometry_normal, dir_in) < 0) {
        // No light below the surface
        return {};
    }
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) < 0) {
        frame = -frame;
    }
    // Convert the incoming direction to local coordinates
    Vector3 local_dir_in = to_local(frame, dir_in);
    Real roughness = eval(
        bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real anisotropic = eval(bsdf.anisotropic, vertex.uv, vertex.uv_screen_size, texture_pool);
    // Clamp roughness to avoid numerical issues.
    roughness = std::clamp(roughness, Real(0.01), Real(1));
    Real aspect = std::sqrt(1.0 - 0.9 * anisotropic);
    Real alpha_x = std::max(0.0001, roughness * roughness / aspect);
    Real alpha_y = std::max(0.0001, roughness * roughness * aspect);
    Vector3 local_micro_normal =
        sample_visible_normals_anisotropic(local_dir_in, alpha_x, alpha_y, rnd_param_uv);

    // Transform the micro normal to world space
    Vector3 half_vector = to_world(frame, local_micro_normal);
    // Reflect over the world space normal
    Vector3 reflected = normalize(-dir_in + 2 * dot(dir_in, half_vector) * half_vector);
    return BSDFSampleRecord{
        reflected,
        Real(0) /* eta */, roughness /* roughness */
    };
}

TextureSpectrum get_texture_op::operator()(const DisneyMetal &bsdf) const {
    return bsdf.base_color;
}
