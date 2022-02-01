#include "../microfacet.h"

Spectrum eval_op::operator()(const DisneyBSDF &bsdf) const {
    bool reflect = dot(vertex.geometry_normal, dir_in) *
                   dot(vertex.geometry_normal, dir_out) > 0;
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) * dot(vertex.geometry_normal, dir_in) < 0) {
        frame = -frame;
    }
    // If we are going into the surface, then we use normal eta
    // (internal/external), otherwise we use external/internal.
    Real eta = dot(vertex.geometry_normal, dir_in) > 0 ? bsdf.eta : 1 / bsdf.eta;

    // init
    Spectrum base_color = eval(bsdf.base_color, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real specular_transmission = eval(bsdf.specular_transmission , vertex.uv, vertex.uv_screen_size, texture_pool);
    Real metallic = eval(bsdf.metallic , vertex.uv, vertex.uv_screen_size, texture_pool);
    Real subsurface = eval(bsdf.subsurface , vertex.uv, vertex.uv_screen_size, texture_pool);
    Real specular = eval(bsdf.specular , vertex.uv, vertex.uv_screen_size, texture_pool);
    Real roughness = eval(bsdf.roughness , vertex.uv, vertex.uv_screen_size, texture_pool);
    Real specular_tint = eval(bsdf.specular_tint , vertex.uv, vertex.uv_screen_size, texture_pool);
    Real anisotropic = eval(bsdf.anisotropic , vertex.uv, vertex.uv_screen_size, texture_pool);
    Real sheen = eval(bsdf.sheen , vertex.uv, vertex.uv_screen_size, texture_pool);
    Real sheen_tint = eval(bsdf.sheen_tint , vertex.uv, vertex.uv_screen_size, texture_pool);
    Real clearcoat = eval(bsdf.clearcoat , vertex.uv, vertex.uv_screen_size, texture_pool);
    Real clearcoat_gloss = eval(bsdf.clearcoat_gloss , vertex.uv, vertex.uv_screen_size, texture_pool);

    Vector3 half_vector;
    if (reflect) {
        half_vector = normalize(dir_in + dir_out);
    } else {
        // "Generalized half-vector" from Walter et al.
        // See "Microfacet Models for Refraction through Rough Surfaces"
        half_vector = normalize(dir_in + dir_out * eta);
    }

    // Flip half-vector if it's below surface
    if (dot(half_vector, frame.n) < 0) {
        half_vector = -half_vector;
    }

    // Clamp roughness to avoid numerical issues.
    roughness = std::clamp(roughness, Real(0.01), Real(1));

    // helper constants
    Real n_dot_h = dot(frame.n, half_vector);
    Real n_dot_in = dot(frame.n, dir_in);
    Real n_dot_out = dot(frame.n, dir_out);
    Real h_dot_in = dot(half_vector, dir_in);
    Real h_dot_out = dot(half_vector, dir_out);
    Vector3 local_half_vector = to_local(frame, half_vector);
    Real hlx = local_half_vector.x;
    Real hly = local_half_vector.y;
    Real hlz = local_half_vector.z;
    Real aspect = std::sqrt(1.0 - 0.9 * anisotropic);
    Real alpha_x = std::max(0.0001, roughness * roughness / aspect);
    Real alpha_y = std::max(0.0001, roughness * roughness * aspect);

    // helper lambdas
    auto G = [&frame, alpha_x, alpha_y](const Vector3& w) {
        const Vector3 local_w = to_local(frame, w);
        Real wx = local_w.x;
        Real wy = local_w.y;
        Real wz = local_w.z;
        Real lambda = (std::sqrt(1 + ((wx * wx * alpha_x * alpha_x) + (wy * wy * alpha_y * alpha_y)) / (wz * wz)) - 1) * 0.5;
        return 1.0 / (1 + lambda);
    };
    auto R_0 = [](const Real eta) {
        return (eta - 1) * (eta - 1) / ((eta + 1) * (eta + 1));
    };

    // diffuse
    auto factor = [&frame](const Real F_90, const Real cosine) {
        return 1 + (F_90 - 1) * std::pow(1 - std::fabs(cosine),  5.0);
    };
    const Real F_D_90 = 0.5 + 2 * roughness * h_dot_out * h_dot_out;
    Spectrum f_base_diffuse = base_color / c_PI * factor(F_D_90, n_dot_in) * factor(F_D_90, n_dot_out) * std::fabs(n_dot_out);

    const Real F_SS_90 = roughness * dot(half_vector, dir_out) * dot(half_vector, dir_out);
    Spectrum f_subsurface = 1.25 * base_color / c_PI *
                (factor(F_SS_90, n_dot_in) * factor(F_SS_90, n_dot_out) * (1.0 / (std::fabs(n_dot_in) + std::fabs(n_dot_out)) - 0.5) + 0.5)
                * std::fabs(n_dot_out);

    Spectrum f_diffuse = (1 - subsurface) * f_base_diffuse + subsurface * f_subsurface;
    Real diffuse_weight = (1 - specular_transmission) * (1 - metallic);

    // sheen
    Spectrum C_tint = luminance(base_color) > 0 ? (base_color / luminance(base_color)) : Spectrum(1.0, 1.0, 1.0);
    Spectrum C_sheen = (1 - sheen_tint) + sheen_tint * C_tint;
    Spectrum f_sheen = C_sheen * std::pow(1 - std::fabs(h_dot_out), 5.0) * std::fabs(n_dot_out);
    Real sheen_weight = (1 - metallic) * sheen;

    // metal (modified version)
    Real D_m = 1.0 / (c_PI * alpha_x * alpha_y * std::pow((hlx * hlx / (alpha_x * alpha_x) + hly * hly / (alpha_y * alpha_y) + hlz * hlz), 2.0));
    Real G_m = G(dir_in) * G(dir_out);
    Spectrum C_0 = specular * R_0(eta) * (1 - metallic) * ((1 - specular_tint) + specular_tint * C_tint) + metallic * base_color;
    Spectrum F_m = schlick_fresnel(C_0, std::fabs(h_dot_out));
    Spectrum f_metal = F_m * D_m * G_m / (4 * std::fabs(n_dot_in));
    Real metal_weight = (1 - specular_transmission * (1 - metallic));

    // clearcoat
    Real F_c = schlick_fresnel(R_0(1.5), std::fabs(h_dot_out));
    Real alpha_g = (1 - clearcoat_gloss) * 0.1 + clearcoat_gloss * 0.001;
    Real ag2 = alpha_g * alpha_g;
    Real D_c = (ag2 - 1) / (c_PI * std::log(ag2) * (1 + (ag2 - 1) * local_half_vector.z * local_half_vector.z));
    Real G_c = smith_masking_gtr2(to_local(frame, dir_in), std::sqrt(0.25)) * smith_masking_gtr2(to_local(frame, dir_out), std::sqrt(0.25));
    Spectrum f_clearcoat = make_const_spectrum(F_c * D_c * G_c * 0.25 / std::fabs(n_dot_in));
    Real clearcoat_weight = 0.25 * clearcoat;

    // glass
    Real D_g = 1.0 / (c_PI * alpha_x * alpha_y * std::pow((hlx * hlx / (alpha_x * alpha_x) + hly * hly / (alpha_y * alpha_y) + hlz * hlz), 2.0));
    Real G_g = G(dir_in) * G(dir_out);
    Real F_g = fresnel_dielectric(h_dot_in, eta);
    Spectrum f_glass;
    if (reflect) {
        f_glass = base_color * (F_g * D_g * G_g) / (4 * fabs(n_dot_in));
    } else {
        Real sqrt_denom = h_dot_in + eta * h_dot_out;
        f_glass = sqrt(base_color) * ((1 - F_g) * D_g * G_g * fabs(h_dot_out * h_dot_in)) /
            (fabs(n_dot_in) * sqrt_denom * sqrt_denom);
    }
    Real glass_weight = (1 - metallic) * specular_transmission;

    if (dot(vertex.geometry_normal, dir_in) < 0) {
        f_diffuse = f_metal = f_clearcoat = f_sheen = make_zero_spectrum();
    }
    if (reflect) {
        return f_diffuse * diffuse_weight + f_sheen * sheen_weight + f_metal * metal_weight + f_clearcoat * clearcoat_weight + f_glass * glass_weight;
    } else {
        return f_glass * glass_weight;
    }
}

Real pdf_sample_bsdf_op::operator()(const DisneyBSDF &bsdf) const {
    bool reflect = dot(vertex.geometry_normal, dir_in) *
                   dot(vertex.geometry_normal, dir_out) > 0;
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) * dot(vertex.geometry_normal, dir_in) < 0) {
        frame = -frame;
    }
    // If we are going into the surface, then we use normal eta
    // (internal/external), otherwise we use external/internal.
    Real eta = dot(vertex.geometry_normal, dir_in) > 0 ? bsdf.eta : 1 / bsdf.eta;

    // init
    Spectrum base_color = eval(bsdf.base_color, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real specular_transmission = eval(bsdf.specular_transmission , vertex.uv, vertex.uv_screen_size, texture_pool);
    Real metallic = eval(bsdf.metallic , vertex.uv, vertex.uv_screen_size, texture_pool);
    Real subsurface = eval(bsdf.subsurface , vertex.uv, vertex.uv_screen_size, texture_pool);
    Real specular = eval(bsdf.specular , vertex.uv, vertex.uv_screen_size, texture_pool);
    Real roughness = eval(bsdf.roughness , vertex.uv, vertex.uv_screen_size, texture_pool);
    Real specular_tint = eval(bsdf.specular_tint , vertex.uv, vertex.uv_screen_size, texture_pool);
    Real anisotropic = eval(bsdf.anisotropic , vertex.uv, vertex.uv_screen_size, texture_pool);
    Real sheen = eval(bsdf.sheen , vertex.uv, vertex.uv_screen_size, texture_pool);
    Real sheen_tint = eval(bsdf.sheen_tint , vertex.uv, vertex.uv_screen_size, texture_pool);
    Real clearcoat = eval(bsdf.clearcoat , vertex.uv, vertex.uv_screen_size, texture_pool);
    Real clearcoat_gloss = eval(bsdf.clearcoat_gloss , vertex.uv, vertex.uv_screen_size, texture_pool);

    Vector3 half_vector;
    if (reflect) {
        half_vector = normalize(dir_in + dir_out);
    } else {
        // "Generalized half-vector" from Walter et al.
        // See "Microfacet Models for Refraction through Rough Surfaces"
        half_vector = normalize(dir_in + dir_out * eta);
    }

    // Flip half-vector if it's below surface
    if (dot(half_vector, frame.n) < 0) {
        half_vector = -half_vector;
    }

    // Clamp roughness to avoid numerical issues.
    roughness = std::clamp(roughness, Real(0.01), Real(1));

    // helper constants
    Real n_dot_h = dot(frame.n, half_vector);
    Real n_dot_in = dot(frame.n, dir_in);
    Real n_dot_out = dot(frame.n, dir_out);
    Real h_dot_in = dot(half_vector, dir_in);
    Real h_dot_out = dot(half_vector, dir_out);
    Vector3 local_half_vector = to_local(frame, half_vector);
    Real hlx = local_half_vector.x;
    Real hly = local_half_vector.y;
    Real hlz = local_half_vector.z;
    Real aspect = std::sqrt(1.0 - 0.9 * anisotropic);
    Real alpha_x = std::max(0.0001, roughness * roughness / aspect);
    Real alpha_y = std::max(0.0001, roughness * roughness * aspect);
    // helper lambdas
    auto G = [&frame, alpha_x, alpha_y](const Vector3& w) {
        const Vector3 local_w = to_local(frame, w);
        Real wx = local_w.x;
        Real wy = local_w.y;
        Real wz = local_w.z;
        Real lambda = (std::sqrt(1 + ((wx * wx * alpha_x * alpha_x) + (wy * wy * alpha_y * alpha_y)) / (wz * wz)) - 1) * 0.5;
        return 1.0 / (1 + lambda);
    };
    auto R_0 = [](const Real eta) {
        return (eta - 1) * (eta - 1) / ((eta + 1) * (eta + 1));
    };

    // diffuse
    Real diffuse_pdf = fmax(n_dot_out, Real(0)) / c_PI;
    Real diffuse_weight = (1 - specular_transmission) * (1 - metallic);

    // metal
    Real D_m = 1.0 / (c_PI * alpha_x * alpha_y * std::pow((hlx * hlx / (alpha_x * alpha_x) + hly * hly / (alpha_y * alpha_y) + hlz * hlz), 2.0));
    Real G_m = G(dir_in) * G(dir_out);
    Real metal_pdf = D_m * G(dir_in) / (4 * std::fabs(n_dot_in));
    Real metal_weight = (1 - specular_transmission * (1 - metallic));

    // clearcoat
    Real F_c = schlick_fresnel(R_0(1.5), std::fabs(h_dot_out));
    Real alpha_g = (1 - clearcoat_gloss) * 0.1 + clearcoat_gloss * 0.001;
    Real ag2 = alpha_g * alpha_g;
    Real D_c = (ag2 - 1) / (c_PI * std::log(ag2) * (1 + (ag2 - 1) * local_half_vector.z * local_half_vector.z));
    Real clearcoat_pdf =  D_c * std::fabs(n_dot_h) / (4 * std::fabs(dot(half_vector, dir_out)));
    Real clearcoat_weight = 0.25 * clearcoat;

    // glass
    Real F_g = fresnel_dielectric(h_dot_in, eta);
    Real D_g = 1.0 / (c_PI * alpha_x * alpha_y * std::pow((hlx * hlx / (alpha_x * alpha_x) + hly * hly / (alpha_y * alpha_y) + hlz * hlz), 2.0));
    Real G_in = G(dir_in);

    Real glass_pdf;
    if (reflect) {
        glass_pdf = (F_g * D_g * G_in) / (4 * fabs(dot(frame.n, dir_in)));
    } else {
        Real sqrt_denom = h_dot_in + eta * h_dot_out;
        Real dh_dout = h_dot_out / (sqrt_denom * sqrt_denom);
        glass_pdf = (1 - F_g) * D_g * G_in * fabs(dh_dout * h_dot_in / dot(frame.n, dir_in));
    }
    Real glass_weight = (1 - metallic) * specular_transmission;

    Real weight_sum = diffuse_weight + metal_weight + clearcoat_weight + glass_weight;
    diffuse_weight /= weight_sum;
    metal_weight /= weight_sum;
    clearcoat_weight /= weight_sum;
    glass_weight / weight_sum;

    if (dot(vertex.geometry_normal, dir_in) < 0) {
        diffuse_weight = metal_weight = clearcoat_weight = 0.0;
        glass_weight = glass_weight > 0 ? 1.0 : glass_weight;
    }
    if (reflect) {
        return diffuse_pdf * diffuse_weight + metal_pdf * metal_weight + clearcoat_pdf * clearcoat_weight + glass_pdf * glass_weight;
    } else {
        return glass_pdf * glass_weight;
    }
}

std::optional<BSDFSampleRecord>
        sample_bsdf_op::operator()(const DisneyBSDF &bsdf) const {
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) * dot(vertex.geometry_normal, dir_in) < 0) {
        frame = -frame;
    }
    // If we are going into the surface, then we use normal eta
    // (internal/external), otherwise we use external/internal.
    Real eta = dot(vertex.geometry_normal, dir_in) > 0 ? bsdf.eta : 1 / bsdf.eta;

    // init
    Spectrum base_color = eval(bsdf.base_color, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real specular_transmission = eval(bsdf.specular_transmission , vertex.uv, vertex.uv_screen_size, texture_pool);
    Real metallic = eval(bsdf.metallic , vertex.uv, vertex.uv_screen_size, texture_pool);
    Real subsurface = eval(bsdf.subsurface , vertex.uv, vertex.uv_screen_size, texture_pool);
    Real specular = eval(bsdf.specular , vertex.uv, vertex.uv_screen_size, texture_pool);
    Real roughness = eval(bsdf.roughness , vertex.uv, vertex.uv_screen_size, texture_pool);
    Real specular_tint = eval(bsdf.specular_tint , vertex.uv, vertex.uv_screen_size, texture_pool);
    Real anisotropic = eval(bsdf.anisotropic , vertex.uv, vertex.uv_screen_size, texture_pool);
    Real sheen = eval(bsdf.sheen , vertex.uv, vertex.uv_screen_size, texture_pool);
    Real sheen_tint = eval(bsdf.sheen_tint , vertex.uv, vertex.uv_screen_size, texture_pool);
    Real clearcoat = eval(bsdf.clearcoat , vertex.uv, vertex.uv_screen_size, texture_pool);
    Real clearcoat_gloss = eval(bsdf.clearcoat_gloss , vertex.uv, vertex.uv_screen_size, texture_pool);

    // Clamp roughness to avoid numerical issues.
    roughness = std::clamp(roughness, Real(0.01), Real(1));

    // Sample a micro normal and transform it to world space -- this is our half-vector.
    Real aspect = std::sqrt(1.0 - 0.9 * anisotropic);
    Real alpha_x = std::max(0.0001, roughness * roughness / aspect);
    Real alpha_y = std::max(0.0001, roughness * roughness * aspect);
    Vector3 local_dir_in = to_local(frame, dir_in);

    Real diffuse_weight = (1 - specular_transmission) * (1 - metallic);
    Real metal_weight = (1 - specular_transmission * (1 - metallic));
    Real clearcoat_weight = 0.25 * clearcoat;
    Real glass_weight = (1 - metallic) * specular_transmission;

    Real weight_sum = diffuse_weight + metal_weight + clearcoat_weight + glass_weight;
    diffuse_weight /= weight_sum;
    metal_weight /= weight_sum;
    clearcoat_weight /= weight_sum;
    glass_weight / weight_sum;

    if (dot(vertex.geometry_normal, dir_in) >= 0) {
        if (rnd_param_w <= diffuse_weight) {
            return BSDFSampleRecord{
                to_world(frame, sample_cos_hemisphere(rnd_param_uv)),
                Real(0) /* eta */, Real(1) /* roughness */};
        } else if (rnd_param_w <= (diffuse_weight + metal_weight)) {
            Vector3 local_micro_normal =
                sample_visible_normals_anisotropic(local_dir_in, alpha_x, alpha_y, rnd_param_uv);
            Vector3 half_vector = to_world(frame, local_micro_normal);
            Vector3 reflected = normalize(-dir_in + 2 * dot(dir_in, half_vector) * half_vector);
            return BSDFSampleRecord{
                reflected,
                Real(0) /* eta */, roughness /* roughness */
            };
        } else if (rnd_param_w <= (diffuse_weight + metal_weight + glass_weight)) {
                Vector3 local_micro_normal =
                    sample_visible_normals_anisotropic(local_dir_in, alpha_x, alpha_y, rnd_param_uv);

                Vector3 half_vector = to_world(frame, local_micro_normal);
                // Flip half-vector if it's below surface
                if (dot(half_vector, frame.n) < 0) {
                    half_vector = -half_vector;
                }

                // Now we need to decide whether to reflect or refract.
                // We do this using the Fresnel term.
                Real h_dot_in = dot(half_vector, dir_in);
                Real F = fresnel_dielectric(h_dot_in, eta);
                Real new_w = (rnd_param_w - diffuse_weight - metal_weight) / glass_weight;
                if (new_w <= F) {
                    // Reflection
                    Vector3 reflected = normalize(-dir_in + 2 * dot(dir_in, half_vector) * half_vector);
                    // set eta to 0 since we are not transmitting
                    return BSDFSampleRecord{reflected, Real(0) /* eta */, roughness};
                } else {
                    // Refraction
                    // https://en.wikipedia.org/wiki/Snell%27s_law#Vector_form
                    // (note that our eta is eta2 / eta1, and l = -dir_in)
                    Real h_dot_out_sq = 1 - (1 - h_dot_in * h_dot_in) / (eta * eta);
                    if (h_dot_out_sq <= 0) {
                        // Total internal reflection
                        // This shouldn't really happen, as F will be 1 in this case.
                        return {};
                    }
                    // flip half_vector if needed
                    if (h_dot_in < 0) {
                        half_vector = -half_vector;
                    }
                    Real h_dot_out= sqrt(h_dot_out_sq);
                    Vector3 refracted = -dir_in / eta + (fabs(h_dot_in) / eta - h_dot_out) * half_vector;
                    return BSDFSampleRecord{refracted, eta, roughness};
                }
        } else {
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
                    Real(0) /* eta */, roughness
                };
        }
    }
}

TextureSpectrum get_texture_op::operator()(const DisneyBSDF &bsdf) const {
    return bsdf.base_color;
}
