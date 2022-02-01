Answers:

1.1: From the rendering of the simple sphere, the lambertain BRDF becomes darker as we go from the center to the edge of the sphere.
     The base diffuse BRDF of the disney diffuse BRDF is similar to the lambertain BRDF, while the subsurface BRDF is more uniform than the
     previous two BRDFs, it shows a relatively strong intensity even at the edge.

1.2: The base diffuse BRDF is similar to the lambertain BRDF that fall at the edge while the subsurface one has a rise at the edge.
     This is because subsurface BRDF models the internal scattering of the material while the base diffuse BRDF follows the Fresnel reflection.
     When the lighting direction is aligned with the view direction, two BRDFs differ the most.

2.1: The disney metal BRDF is anisotropic (a slanted ellipse) while the roughplastic BRDF is isotropic (a dim circle).

2.2: As roughness increases, the material becomes less anisotropic. And the specularity is less-concentrated.

3.1: Clearcoat BRDF is isotropic and less specular than the metal BRDF using similar roughness/gloss value.

4.1: As eta increases, the material changes from transparent to opaque, metal-like, it feels like the thickness of the glass grows.

5.1: The rendered image is all black. Because the default light direction in this setting cause the out direction and the half_vector the same, leading to a zero term
     in the BRDF. If we change the light source position, then we start to see a distribution that peaks at the edge and gradually falls (depending on the specific light position)

6.1: see the submission/ folder for the image