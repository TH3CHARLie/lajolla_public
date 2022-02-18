#pragma once

#include "scene.h"
#include <limits>

// The simplest volumetric renderer:
// single absorption only homogeneous volume
// only handle directly visible light sources
Spectrum vol_path_tracing_1(const Scene &scene,
                            int x, int y, /* pixel coordinates */
                            pcg32_state &rng) {
    int w = scene.camera.width, h = scene.camera.height;
    Vector2 screen_pos((x + next_pcg32_real<Real>(rng)) / w,
                       (y + next_pcg32_real<Real>(rng)) / h);
    Ray ray = sample_primary(scene.camera, screen_pos);
    RayDifferential ray_diff = RayDifferential{Real(0), Real(0)};
    std::optional<PathVertex> vertex_ = intersect(scene, ray, ray_diff);
    if (!vertex_) {
        // no env map
        return make_zero_spectrum();
    }
    PathVertex vertex = *vertex_;

    Medium medium = scene.media[vertex.exterior_medium_id];
    Real t = distance(ray.org, vertex.position);
    Spectrum sigma_a = get_sigma_a(medium, vertex.position);
    Spectrum transmittance = exp(-sigma_a * t);
    Spectrum Le = make_zero_spectrum();
    if (is_light(scene.shapes[vertex.shape_id])) {
        Le = emission(vertex, -ray.dir, scene);
    }
    return transmittance * Le;
}

// The second simplest volumetric renderer:
// single monochromatic homogeneous volume with single scattering,
// no need to handle surface lighting, only directly visible light source
Spectrum vol_path_tracing_2(const Scene &scene,
                            int x, int y, /* pixel coordinates */
                            pcg32_state &rng) {
    int w = scene.camera.width, h = scene.camera.height;
    Vector2 screen_pos((x + next_pcg32_real<Real>(rng)) / w,
                       (y + next_pcg32_real<Real>(rng)) / h);
    Ray ray = sample_primary(scene.camera, screen_pos);
    RayDifferential ray_diff = RayDifferential{Real(0), Real(0)};
    std::optional<PathVertex> vertex_ = intersect(scene, ray, ray_diff);
    Real u = next_pcg32_real<Real>(rng);
    Medium medium = scene.media[scene.camera.medium_id];
	Spectrum sigma_a = get_sigma_a(medium, make_zero_spectrum());
	Spectrum sigma_s = get_sigma_s(medium, make_zero_spectrum());

    Spectrum sigma_t = (sigma_a + sigma_s);
    // monochromatic assumption
    Real t = -std::log(1 - u) / sigma_t.x;
    Real t_hit = vertex_ ? distance(ray.org, vertex_->position) : std::numeric_limits<Real>::infinity();
    if (t < t_hit) {
        Real trans_pdf = std::exp(-sigma_t.x * t) * sigma_t.x;
        Spectrum transmittance = exp(-sigma_t * t);
        Vector3 p = ray.org + t * ray.dir;
        // sample_on_light, copied from path tracing
        Vector2 light_uv{next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng)};
        Real light_w = next_pcg32_real<Real>(rng);
        Real shape_w = next_pcg32_real<Real>(rng);
        int light_id = sample_light(scene, light_w);
        const Light &light = scene.lights[light_id];
        PointAndNormal point_on_light =
            sample_point_on_light(light, p, light_uv, shape_w, scene);
        // first term: compute rho
        Vector3 dir_in = -ray.dir;
        Vector3 dir_out = normalize(point_on_light.position - p);
        PhaseFunction phase_function = get_phase_function(medium);
        Spectrum rho = eval(phase_function, dir_in, dir_out);
        // second term: compute Le
        Spectrum Le = emission(light, -dir_out, Real(0), point_on_light, scene);
        // third term
        Real dist = distance(p, point_on_light.position);
        Real exp_term = std::exp(-sigma_t.x * dist);
        // forth term
        Real geometry_term = std::fabs(dot(-dir_out, point_on_light.normal)) / (dist * dist);
        // visibility term
        Ray visibility_ray{p, dir_out, get_shadow_epsilon(scene), (1 - get_shadow_epsilon(scene)) * dist};
        // std::optional<PathVertex> shadow_vertex_ = intersect(scene, visibility_ray, ray_diff);
        Real visibility = 1;
        if (occluded(scene, visibility_ray)) {
            visibility = 0;
        }
        Spectrum L_s1_estimate = rho * Le * exp_term * geometry_term * visibility;
        Real L_s1_pdf = light_pmf(scene, light_id) * pdf_point_on_light(light, point_on_light, p, scene);
        return transmittance / trans_pdf * sigma_s * L_s1_estimate / L_s1_pdf;
    } else {
		Real trans_pdf = std::exp(-sigma_t.x * t_hit);
		Spectrum transmittance = exp(-sigma_t * t_hit);
        Spectrum Le = make_zero_spectrum();
        if (is_light(scene.shapes[vertex_->shape_id])) {
            Le = emission(*vertex_, -ray.dir, scene);
        }
        return transmittance * Le / trans_pdf;
    }
}

// The third volumetric renderer (not so simple anymore): 
// multiple monochromatic homogeneous volumes with multiple scattering
// no need to handle surface lighting, only directly visible light source
Spectrum vol_path_tracing_3(const Scene &scene,
                            int x, int y, /* pixel coordinates */
                            pcg32_state &rng) {
    // Homework 2: implememt this!
    return make_zero_spectrum();
}

// The fourth volumetric renderer: 
// multiple monochromatic homogeneous volumes with multiple scattering
// with MIS between next event estimation and phase function sampling
// still no surface lighting
Spectrum vol_path_tracing_4(const Scene &scene,
                            int x, int y, /* pixel coordinates */
                            pcg32_state &rng) {
    // Homework 2: implememt this!
    return make_zero_spectrum();
}

// The fifth volumetric renderer: 
// multiple monochromatic homogeneous volumes with multiple scattering
// with MIS between next event estimation and phase function sampling
// with surface lighting
Spectrum vol_path_tracing_5(const Scene &scene,
                            int x, int y, /* pixel coordinates */
                            pcg32_state &rng) {
    // Homework 2: implememt this!
    return make_zero_spectrum();
}

// The final volumetric renderer: 
// multiple chromatic heterogeneous volumes with multiple scattering
// with MIS between next event estimation and phase function sampling
// with surface lighting
Spectrum vol_path_tracing(const Scene &scene,
                          int x, int y, /* pixel coordinates */
                          pcg32_state &rng) {
    // Homework 2: implememt this!
    return make_zero_spectrum();
}
