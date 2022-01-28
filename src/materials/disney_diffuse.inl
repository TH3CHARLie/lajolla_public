Spectrum eval_op::operator()(const DisneyDiffuse &bsdf) const {
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
    Spectrum base_color = eval(bsdf.base_color, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real roughness = eval(bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real subsurface = eval(bsdf.subsurface, vertex.uv, vertex.uv_screen_size, texture_pool);
    Vector3 half_vector = normalize(dir_in + dir_out);

    // compute base diffuse BRDF
    auto factor = [&frame](const Real F_90, const Vector3& w) {
        return 1 + (F_90 - 1) * std::pow(1 - std::fabs(dot(frame.n, w)),  5.0);
    };
    const Real F_D_90 = 0.5 + 2 * roughness * dot(half_vector, dir_out) * dot(half_vector, dir_out);
    Spectrum f_base_diffuse = base_color / c_PI * factor(F_D_90, dir_in) * factor(F_D_90, dir_out) * std::fabs(dot(frame.n, dir_out));

    const Real F_SS_90 = roughness * dot(half_vector, dir_out) * dot(half_vector, dir_out);
    Spectrum f_subsurface = 1.25 * base_color / c_PI *
                (factor(F_SS_90, dir_in) * factor(F_SS_90, dir_out) * (1.0 / (std::fabs(dot(frame.n, dir_in)) + std::fabs(dot(frame.n, dir_out))) - 0.5) + 0.5)
                * std::fabs(dot(frame.n, dir_out));

    return (1 - subsurface) * f_base_diffuse + subsurface * f_subsurface;
}

Real pdf_sample_bsdf_op::operator()(const DisneyDiffuse &bsdf) const {
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

    return fmax(dot(frame.n, dir_out), Real(0)) / c_PI;
}

std::optional<BSDFSampleRecord> sample_bsdf_op::operator()(const DisneyDiffuse &bsdf) const {
    if (dot(vertex.geometry_normal, dir_in) < 0) {
        // No light below the surface
        return {};
    }
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) < 0) {
        frame = -frame;
    }

    return BSDFSampleRecord{
        to_world(frame, sample_cos_hemisphere(rnd_param_uv)),
        Real(0) /* eta */, Real(1) /* roughness */};
}

TextureSpectrum get_texture_op::operator()(const DisneyDiffuse &bsdf) const {
    return bsdf.base_color;
}
