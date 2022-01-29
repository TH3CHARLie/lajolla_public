#include "../microfacet.h"

Spectrum eval_op::operator()(const DisneyClearcoat &bsdf) const {
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
    Vector3 local_half_vector = to_local(frame, half_vector);
    Real n_dot_h = dot(frame.n, half_vector);
    Real n_dot_in = dot(frame.n, dir_in);
    Real n_dot_out = dot(frame.n, dir_out);
    if (n_dot_out <= 0 || n_dot_h <= 0) {
        return make_zero_spectrum();
    }
    Real clearcoat_gloss = eval(bsdf.clearcoat_gloss, vertex.uv, vertex.uv_screen_size, texture_pool);
    auto R_0 = [](const Real eta) {
        return (eta - 1) * (eta - 1) / ((eta + 1) * (eta + 1));
    };
    Real F_c = schlick_fresnel(R_0(1.5), std::fabs(dot(half_vector, dir_out)));
    Real alpha_g = (1 - clearcoat_gloss) * 0.1 + clearcoat_gloss * 0.001;
    Real ag2 = alpha_g * alpha_g;
    Real D_c = (ag2 - 1) / (c_PI * std::log(ag2) * (1 + (ag2 - 1) * local_half_vector.z * local_half_vector.z));
    Real G_c = smith_masking_gtr2(to_local(frame, dir_in), std::sqrt(0.25)) * smith_masking_gtr2(to_local(frame, dir_out), std::sqrt(0.25));
    return make_const_spectrum(F_c * D_c * G_c * 0.25 / std::fabs(n_dot_in));
}

Real pdf_sample_bsdf_op::operator()(const DisneyClearcoat &bsdf) const {
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

    Vector3 half_vector = normalize(dir_in + dir_out);
    Vector3 local_half_vector = to_local(frame, half_vector);
    Real n_dot_h = dot(frame.n, half_vector);
    Real n_dot_out = dot(frame.n, dir_out);
    if (n_dot_out <= 0 || n_dot_h <= 0) {
        return 0;
    }
    Real clearcoat_gloss = eval(bsdf.clearcoat_gloss, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real alpha_g = (1 - clearcoat_gloss) * 0.1 + clearcoat_gloss * 0.001;
    Real ag2 = alpha_g * alpha_g;
    Real D_c = (ag2 - 1) / (c_PI * std::log(ag2) * (1 + (ag2 - 1) * local_half_vector.z * local_half_vector.z));
    return D_c * std::fabs(n_dot_h) / (4 * std::fabs(dot(half_vector, dir_out)));
}

std::optional<BSDFSampleRecord>
        sample_bsdf_op::operator()(const DisneyClearcoat &bsdf) const {
    if (dot(vertex.geometry_normal, dir_in) < 0) {
        // No light below the surface
        return {};
    }
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) < 0) {
        frame = -frame;
    }
    Real clearcoat_gloss = eval(bsdf.clearcoat_gloss, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real alpha_g = (1 - clearcoat_gloss) * 0.1 + clearcoat_gloss * 0.001;
    Real ag2 = alpha_g * alpha_g;
    Real cos_elevation = std::sqrt((1 - std::pow(ag2, 1 - rnd_param_uv.x)) / (1 - ag2));
    Real azimuth = 2 * c_PI * rnd_param_uv.y;
    Real sin_elevation = std::sqrt(1 - cos_elevation * cos_elevation);
    Vector3 local_micro_normal(sin_elevation * std::cos(azimuth), sin_elevation * std::sin(azimuth), cos_elevation);
    Vector3 half_vector = to_world(frame, local_micro_normal);
    // Reflect over the world space normal
    Vector3 reflected = normalize(-dir_in + 2 * dot(dir_in, half_vector) * half_vector);
    return BSDFSampleRecord{
        reflected,
        Real(0) /* eta */, Real(1) /* roughness */
    };
}

TextureSpectrum get_texture_op::operator()(const DisneyClearcoat &bsdf) const {
    return make_constant_spectrum_texture(make_zero_spectrum());
}
