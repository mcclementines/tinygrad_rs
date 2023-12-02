//! src/tensor.rs

use std::{cell::RefCell, rc::Rc, usize};

/// Represents a Tensor
///
/// # Example
///
/// ```
/// use tinygrad_rs::Tensor;
///
/// let mut tensor = Tensor::new(vec![2, 2]);
/// tensor.set(vec![0,0], 1.0);
/// tensor.set(vec![1,1], 4.0);
///
/// assert_eq!(tensor.get(vec![0,0]), 1.0);
/// assert_eq!(tensor.get(vec![1,1]), 4.0);
/// ```
pub struct Tensor(Rc<RefCell<TensorData>>);

struct TensorStorage(Rc<RefCell<Vec<f32>>>);

struct TensorData {
    data: TensorStorage,
    dim: Vec<usize>,
    strides: Vec<usize>,
}

impl Tensor {
    /// Create a new Tensor with specified dimensions. Values are initialized
    /// at 0.0
    ///
    /// # Example
    ///
    /// ```
    /// use tinygrad_rs::Tensor;
    ///
    /// let tensor = Tensor::new(vec![2,2]);
    /// ```
    pub fn new(dim: Vec<usize>) -> Tensor {
        let mut data = Vec::new();
        data.resize(dim.iter().product(), 0.0);

        let data = TensorStorage(Rc::new(RefCell::new(data)));

        let mut stride: usize = dim.iter().product();
        let strides = dim
            .iter()
            .map(|x| {
                stride /= x;
                stride
            })
            .collect();

        let tensor_data = TensorData { data, dim, strides };

        Tensor(Rc::new(RefCell::new(tensor_data)))
    }

    /// Constructs a new Tensor with specified dimensions and data.
    /// The size of data must be equivent to given dimensions
    ///
    /// # Example
    ///
    /// ```
    /// use tinygrad_rs::Tensor;
    ///
    /// let tensor = Tensor::with_data(vec![2, 1], vec![1.0, 2.0]);
    pub fn with_data(dim: Vec<usize>, data: Vec<f32>) -> Tensor {
        assert_eq!(
            dim.iter().product::<usize>(),
            data.len(),
            "Data does not fit specified dimensions"
        );

        let t = Tensor::new(dim);
        t.set_data(data);

        t
    }

    /// Get the Tensor value located at the specified index
    ///
    /// # Example
    ///
    /// ```
    /// use tinygrad_rs::Tensor;
    ///
    /// let tensor = Tensor::new(vec![2,1]);
    /// let value = tensor.get(vec![1,0]);
    ///
    /// assert_eq!(value, 0.0);
    pub fn get(&self, index: Vec<usize>) -> f32 {
        let stride = self.stride(index);

        return self.0.borrow().data.0.borrow()[stride];
    }

    /// Set the Tensor value located at the specified index
    ///
    /// # Example
    ///
    /// ```
    /// use tinygrad_rs::Tensor;
    ///
    /// let mut tensor = Tensor::new(vec![2,1]);
    /// tensor.set(vec![1,0], 1.0);
    ///
    /// assert_eq!(tensor.get(vec![1,0]), 1.0);
    pub fn set(&mut self, index: Vec<usize>, value: f32) {
        let stride = self.stride(index);

        self.0.borrow_mut().data.0.borrow_mut()[stride] = value;
    }

    pub fn shape(&self) -> Vec<usize> {
        self.get_dim()
    }

    fn stride(&self, index: Vec<usize>) -> usize {
        index
            .iter()
            .zip(self.get_strides())
            .map(|(i, s)| i * s)
            .sum()
    }

    fn get_data(&self) -> Rc<RefCell<Vec<f32>>> {
        unimplemented!()
    }

    fn set_data(&self, data: Vec<f32>) {
        *self.0.borrow_mut().data.0.borrow_mut() = data;
    }

    fn get_dim(&self) -> Vec<usize> {
        self.0.borrow().dim.clone()
    }

    fn get_strides(&self) -> Vec<usize> {
        self.0.borrow().strides.clone()
    }
}
