use image::RgbaImage;

pub fn oklch_to_oklab(lch: glam::Vec3) -> glam::Vec3 {
    glam::vec3(lch.x, lch.y * lch.z.cos(), lch.y * lch.z.sin())
}

pub fn oklab_to_linear_rgb(lab: glam::Vec3) -> glam::Vec3 {
    const M1: glam::Mat3 = glam::Mat3::from_cols_array(&[
        1.000000000,
        1.000000000,
        1.000000000,
        0.396337777,
        -0.105561346,
        -0.089484178,
        0.215803757,
        -0.063854173,
        -1.291485548,
    ]);
    const M2: glam::Mat3 = glam::Mat3::from_cols_array(&[
        4.076724529,
        -1.268143773,
        -0.004111989,
        -3.307216883,
        2.609332323,
        -0.703476310,
        0.230759054,
        -0.341134429,
        1.706862569,
    ]);

    let lms = M1 * lab;
    return M2 * (lms * lms * lms);
}

pub fn linear_rgb_to_oklab(rgb: glam::Vec3) -> glam::Vec3 {
    const M1: glam::Mat3 = glam::Mat3::from_cols_array(&[
        0.4121656120,
        0.2118591070,
        0.0883097947,
        0.5362752080,
        0.6807189584,
        0.2818474174,
        0.0514575653,
        0.1074065790,
        0.6302613616,
    ]);

    const M2: glam::Mat3 = glam::Mat3::from_cols_array(&[
        0.2104542553,
        1.9779984951,
        0.0259040371,
        0.7936177850,
        -2.4285922050,
        0.7827717662,
        -0.0040720468,
        0.4505937099,
        -0.8086757660,
    ]);

    let lms = M1 * rgb;
    return M2 * (lms.signum() * lms.abs().powf(1.0 / 3.0));
}

const GOLDEN_ANGLE: f32 = 2.39996323; // radians (~137.5 degrees)

/// Generates a palette of N perceptually balanced, distinct colors.
/// Returns Linear sRGB.
pub fn generate_palette(n: usize) -> Vec<glam::Vec3> {
    let mut palette = Vec::with_capacity(n);

    for i in 0..n {
        // 1. Calculate Hue (Golden Angle)
        // This spreads colors around the cylinder evenly without resonance.
        let hue = i as f32 * GOLDEN_ANGLE;

        // 2. Calculate Lightness & Chroma
        // We vary these to maximize distinctness.
        // We oscillate Lightness so neighbors (i, i+1) have high contrast.
        // Range: 0.5 to 0.85 (avoiding near-black and blown-out white)
        let t = i as f32 / (n as f32);

        // This 'bias' shifts the lightness slightly over the whole set
        // to prevent patterns from repeating exactly.
        let lightness_bias = 0.60;
        let l = lightness_bias + 0.20 * (i as f32 * 0.5).sin();

        // Chroma: Keep it moderate. Too high = out of gamut.
        // 0.10 is pastel, 0.15 is vibrant, 0.20 is neon/risky.
        let c = 0.14;

        // 3. Convert Polar (LCh) -> Oklab (Lab)
        // a = C * cos(h), b = C * sin(h)
        let oklab = glam::vec3(l, c * hue.cos(), c * hue.sin());

        // 4. Convert Oklab -> Linear sRGB
        let mut rgb = oklab_to_linear_rgb(oklab);

        // 5. Gamut Mapping (Simple Chroma Reduction)
        // If the color is outside 0.0-1.0, we desaturate it until it fits.
        if !(rgb.min_element() >= 0.0 && rgb.max_element() <= 1.0) {
            rgb = fit_to_gamut(l, c, hue);
        }

        palette.push(rgb);
    }

    palette
}

/// Binary search to find the maximum Chroma that fits in sRGB
/// Preserves Hue and Lightness (Perceptually superior to clamping)
fn fit_to_gamut(l: f32, max_c: f32, h: f32) -> glam::Vec3 {
    let mut low = 0.0;
    let mut high = max_c;
    let mut best_rgb = glam::Vec3::ZERO;

    for _ in 0..5 {
        let mid_c = (low + high) * 0.5;
        let oklab = glam::vec3(l, mid_c * h.cos(), mid_c * h.sin());
        let rgb = oklab_to_linear_rgb(oklab);

        if rgb.min_element() >= 0.0 && rgb.max_element() <= 1.0 {
            best_rgb = rgb;
            low = mid_c;
        } else {
            high = mid_c;
        }
    }
    best_rgb
}

use palette::FromColor;
use rand::prelude::*;

pub fn generate_from_image(img: &image::RgbaImage) {
    let size = glam::uvec2(img.dimensions().0, img.dimensions().1);
    let mut rng = rand::rng();

    const SAMPLES: u32 = 100;
    for _ in 0..SAMPLES {
        let x = rng.next_u32() % size.x;
        let y = rng.next_u32() % size.y;
        let rgba = img.get_pixel(x, y).0;
        let srgb = palette::Srgb::from_components((rgba[0], rgba[1], rgba[2])).into_linear::<f32>();
        let oklab = palette::Oklab::from_color(srgb);
        dbg!(oklab);
    }
}
