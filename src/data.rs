//! src/data.rs

use std::fmt::Debug;
use std::iter::Sum;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::{cell::RefCell, rc::Rc};

/// Data object used in tinygrad_rs
///
#[derive(Clone, Debug)]
pub struct Data<
    T: Clone
        + Copy
        + Debug
        + Default
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Neg<Output = T>
        + Sum,
>(pub Rc<RefCell<T>>);

impl<T> Data<T>
where
    T: Clone
        + Copy
        + Debug
        + Default
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Neg<Output = T>
        + Sum,
{
    /// A `Data` object in the `tinygrad_rs` library.
    ///
    /// # Examples
    ///
    /// Creating a new `Data` instance:
    ///
    /// ```
    /// use tinygrad_rs::Data;
    ///
    /// // Create a new `Data` instance with an initial value
    /// let data = Data::new(1.0);
    ///
    /// // Access the value for verification or other operations
    /// let value = data.get();
    ///
    /// // Perform operations or transformations on `data` as needed
    /// // ...
    /// ```
    pub fn new(data: T) -> Data<T> {
        Data(Rc::new(RefCell::new(data)))
    }

    pub fn from_vec(data: Vec<T>) -> Vec<Data<T>> {
        data.iter().map(|x| Data::new(*x)).collect()
    }

    /// Retrieves a copy of the raw data value from a `Data` instance.
    ///
    /// This method returns the current value stored in a `Data` object without
    /// altering the original instance. The value is returned as a copy, ensuring
    /// that the `Data` instance remains unmodified.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use tinygrad_rs::Data;
    ///
    /// // Create a new Data instance with a specified value
    /// let data = Data::new(1.0);
    ///
    /// // Retrieve the value from the Data instance
    /// let value = data.get();
    ///
    /// // Check that the retrieved value matches the expected value
    /// assert_eq!(value, 1.0);
    /// ```
    pub fn get(&self) -> T {
        self.0.borrow().to_owned()
    }

    /// Sets the raw data value for a `Data` instance.
    ///
    /// This method allows updating the value contained within a `Data` object.
    /// The existing data is replaced with the new value provided as an argument.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use tinygrad_rs::Data;
    ///
    /// // Create a new Data instance with an initial value
    /// let mut data = Data::new(1.0);
    ///
    /// // Update the value of data
    /// data.set(2.0);
    ///
    /// // Check that the value has been updated correctly
    /// assert_eq!(data.get(), 2.0);
    /// ```
    pub fn set(&self, data: T) {
        *self.0.borrow_mut() = data;
    }
}

impl<T> Mul for Data<T>
where
    T: Clone
        + Copy
        + Debug
        + Default
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Neg<Output = T>
        + Sum,
{
    type Output = Data<T>;

    fn mul(self, rhs: Data<T>) -> Data<T> {
        Data::new(self.get() * rhs.get())
    }
}

impl<
        T: Clone
            + Copy
            + Debug
            + Default
            + Add<Output = T>
            + Sub<Output = T>
            + Mul<Output = T>
            + Div<Output = T>
            + Neg<Output = T>
            + Sum
            + PartialEq,
    > PartialEq for Data<T>
{
    fn eq(&self, other: &Self) -> bool {
        self.get() == other.get()
    }
}

impl<
        T: Clone
            + Copy
            + Debug
            + Default
            + Add<Output = T>
            + Sub<Output = T>
            + Mul<Output = T>
            + Div<Output = T>
            + Neg<Output = T>
            + Sum
            + PartialEq,
    > PartialEq<T> for Data<T>
{
    fn eq(&self, other: &T) -> bool {
        self.get() == *other
    }
}

impl PartialEq<Data<f32>> for f32 {
    fn eq(&self, other: &Data<f32>) -> bool {
        *self == other.get()
    }
}
