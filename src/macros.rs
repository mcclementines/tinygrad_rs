//! src/macros.rs

#[macro_export]
macro_rules! data {
    ($($x:expr),+ $(,)?) => (
        Data::from_vec(vec![$($x),+])
    );
}
