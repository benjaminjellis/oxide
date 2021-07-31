///
///
pub fn clip_by_value(src: &af::Array<f32>, clip_min: f32, clip_max: f32) -> af::Array<f32> {
    let min_clipped = af::selectl(clip_min as f64, &af::lt(src, &clip_min, false), src);
    af::selectl(
        clip_max as f64,
        &af::gt(&min_clipped, &clip_max, false),
        &min_clipped,
    )
}

/// Take inputs and return mini batches, input should be (x, y, target)
///  if n / bs != whole number the last mini batch is dropped
///
/// # Parameters
/// - arrays: tuple of x, targets
/// - bs: batch size
/// - shuffle: whether to shuffle the order of the mini batches
#[allow(clippy::type_complexity)]
pub fn batch(
    arrays: (&af::Array<f32>, &af::Array<f32>),
    bs: u64,
) -> (Vec<af::Array<f32>>, Vec<af::Array<f32>>) {
    let (x, t) = arrays;

    assert_eq!(
        x.dims()[0],
        t.dims()[0],
        "Expected equal numbers of observations for x and targets, got {} and {}",
        x.dims()[0],
        t.dims()[0],
    );

    // check that batch size is feasible
    assert!(
        bs <= x.dims()[0],
        "Batch size is smaller than the number of observations, can't \
    batch. Try making the batch size smaller"
    );
    let mut n: f32 = x.dims()[0] as f32 / bs as f32;

    if n.fract() != 0.0 {
        // not easily divisible
        n = n.floor();
    }

    let mut x_batched = Vec::new();
    let mut t_batched = Vec::new();

    // allocate x and y to x_batched and y_batched
    let mut first_index = 0u64;

    for _idx in 0..n as u64 {
        let last_index = first_index + bs - 1;
        let x_seq = &[
            af::Seq::new(first_index as u32, last_index as u32, 1u32),
            af::Seq::default(),
        ];

        let t_seq = &[af::Seq::new(first_index as u32, last_index as u32, 1u32)];

        let x_slice = af::index(&x, x_seq);
        let t_slice = af::index(&t, t_seq);

        x_batched.push(x_slice);
        t_batched.push(t_slice);
        first_index = last_index + 1;
    }

    (x_batched, t_batched)
}

///
pub fn get_val(array: &af::Array<f32>) -> f32 {
    fn to_vec<T: af::HasAfEnum + Default + Clone>(array: &af::Array<T>) -> Vec<T> {
        let mut vec = vec![T::default(); array.elements()];
        array.host(&mut vec);
        vec
    }
    to_vec(array)[0] as f32
}
