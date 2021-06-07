use af;

pub fn clip_by_value(src: &af::Array<f32>, clip_min: f32, clip_max: f32) -> af::Array<f32> {
    let min_clipped = af::selectl(clip_min as f64, &af::lt(src, &clip_min, false), src);
    af::selectl(
        clip_max as f64,
        &af::gt(&min_clipped, &clip_max, false),
        &min_clipped,
    )
}

/// Take inputs and return mini batches
pub fn batch() {

}

