//! src/lib.rs

#[macro_use]
pub mod macros;

pub mod tensor;
pub use crate::tensor::Tensor;

pub mod data;
pub use crate::data::Data;
