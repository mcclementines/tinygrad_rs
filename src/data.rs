//! src/data.rs

use std::fmt::Debug;
use std::ops::Mul;
use std::{cell::RefCell, rc::Rc};

/// Data object used in tinygrad_rs
///
#[derive(Clone, Debug)]
pub struct Data(pub Rc<RefCell<f64>>);

impl Data {
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
    pub fn new(data: f64) -> Data {
        Data(Rc::new(RefCell::new(data)))
    }

    pub fn from_vec(data: Vec<f64>) -> Vec<Data> {
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
    pub fn get(&self) -> f64 {
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
    pub fn set(&self, data: f64) {
        *self.0.borrow_mut() = data;
    }
}

impl Mul for Data {
    type Output = Data;

    fn mul(self, rhs: Data) -> Data {
        Data::new(self.get() * rhs.get())
    }
}

impl PartialEq for Data {
    fn eq(&self, other: &Self) -> bool {
        self.get() == other.get()
    }
}

impl PartialEq<f64> for Data {
    fn eq(&self, other: &f64) -> bool {
        self.get() == *other
    }
}

impl PartialEq<Data> for f64 {
    fn eq(&self, other: &Data) -> bool {
        *self == other.get()
    }
}
