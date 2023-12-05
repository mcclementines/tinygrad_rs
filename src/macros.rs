//! src/macros.rs

#[macro_export]
macro_rules! data {
    ($($x:expr),+ $(,)?) => (
        Data::from_vec(vec![$($x),+])
    );
}

/// Initializes a Tensor::with_data() in a user friendly way
macro_rules! tensor {
    () => (Tensor::new(vec![1], 0.0))
}

/// Creates a TensorSlice
macro_rules! tslice {
    () => (unimplemented!())
}
