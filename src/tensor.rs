//! src/tensor.rs

use std::{cell::RefCell, rc::Rc, usize};

use crate::Data;

/// Represents a Tensor
///
/// # Example
///
/// ```
/// use tinygrad_rs::Tensor;
/// use tinygrad_rs::Data;
///
/// let mut tensor = Tensor::new(vec![2, 2], 0.0);
/// tensor.set(vec![0,0], Data::new(1.0));
/// tensor.set(vec![1,1], Data::new(4.0));
///
/// assert_eq!(tensor.get(vec![0,0]).item(), 1.0);
/// assert_eq!(tensor.get(vec![1,1]).item(), 4.0);
/// ```
pub struct Tensor<T: Clone + Copy>(Rc<RefCell<TensorData<T>>>);

struct TensorData<T: Clone + Copy> {
    data: Vec<Data<T>>,
    dim: Vec<usize>,
    strides: Vec<usize>,
    offset: usize
}

impl<T: Clone + Copy> Tensor<T> {
    /// Create a new Tensor with specified dimensions. Values are initialized
    /// at 0.0
    ///
    /// # Example
    ///
    /// ```
    /// use tinygrad_rs::Tensor;
    ///
    /// let tensor = Tensor::new(vec![2,2], 0.0);
    /// ```
    pub fn new(dim: Vec<usize>, initial_element: T) -> Tensor<T> {
        let mut data = Vec::new();
        data.resize(dim.iter().product(), Data::new(initial_element));
        
        let mut stride: usize = dim.iter().product();
        let strides = dim
            .iter()
            .map(|x| {
                stride /= x;
                stride
            })
            .collect();
        
        let offset = 0;

        let tensor_data = TensorData { data, dim, strides, offset };

        Tensor(Rc::new(RefCell::new(tensor_data)))
    }

    /// Constructs a new Tensor with specified dimensions and data.
    /// The size of data must be equivent to given dimensions
    ///
    /// # Example
    ///
    /// ```
    /// use tinygrad_rs::Tensor;
    /// use tinygrad_rs::Data;
    /// use tinygrad_rs::data;
    ///
    /// let tensor = Tensor::with_data(vec![2, 1], data![1.0, 2.0]);
    /// ```
    pub fn with_data(dim: Vec<usize>, data: Vec<Data<T>>) -> Tensor<T> {
        // TODO: Figure out a better way to handle tensor views.
        // Idea here is that, instead of creating a new Data vec
        // for every tensor, we just provide views onto the original
        // Data vec (Pytorch method?) enabled by strides, dim, and offset.
        assert!(
            dim.iter().product::<usize>() <= data.len(),
            "Data does not fit specified dimensions"
        );

        let tensor = Tensor::new(dim, data[0].get());
        tensor.set_data(data);

        tensor
    }

    /// Get the Tensor value located at the specified index
    ///
    /// # Example
    ///
    /// ```
    /// use tinygrad_rs::Tensor;
    ///
    /// let tensor = Tensor::new(vec![2,1], 0.0);
    /// let value = tensor.get(vec![1,0]).item();
    ///
    /// assert_eq!(value, 0.0);
    /// ```
    pub fn get(&self, index: Vec<usize>) -> Tensor<T> {
        // TODO: To be replaced when I have time, quick and dirty solution
        // # of dim == index coords
        assert_eq!(self.get_dim().len(), index.len(), "Provided index does not fit tensor");

        // index in range
        assert!(self.get_dim().iter().gt(index.iter()), "Provided index is out of range");

        let stride = self.stride(index);
        
        Tensor::with_data(vec![1], vec![self.0.borrow().data[stride].clone()])
    }
    
    /// Get the specified slice of the Tensor
    ///
    /// # Example
    ///
    /// ```
    /// use tinygrad_rs::Tensor;
    /// use tinygrad_rs::Data;
    /// use tinygrad_rs::data;
    ///
    /// let tensor = Tensor::with_data(vec![2,2,2], data![1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0]);
    /// let slice = tensor.get_slice(vec![4], vec![2], 0);
    ///
    /// assert_eq!(slice.get(vec![0]).item(), 1.0);
    /// assert_eq!(slice.get(vec![2]).item(), 5.0);
    /// ```
    pub fn get_slice(&self, dim: Vec<usize>, strides: Vec<usize>, offset: usize) -> Tensor<T> {
        let mut tensor = Tensor::with_data(dim, self.get_data());
        tensor.set_strides(strides);
        tensor.set_offset(offset);

        tensor
    }

    /// Set the Tensor value located at the specified index
    ///
    /// # Example
    ///
    /// ```
    /// use tinygrad_rs::Tensor;
    /// use tinygrad_rs::Data;
    ///
    /// let mut tensor = Tensor::new(vec![2,1], 0.0);
    /// tensor.set(vec![1,0], Data::new(1.0));
    ///
    /// assert_eq!(tensor.get(vec![1,0]).item(), 1.0);
    /// ```
    pub fn set(&mut self, index: Vec<usize>, value: Data<T>) {
        let stride = self.stride(index);

        self.0.borrow_mut().data[stride] = value;
    }
    
    /// Get the shape of the Tensor
    ///
    /// # Example
    ///
    /// ```
    /// use tinygrad_rs::Tensor;
    ///
    /// let mut tensor = Tensor::new(vec![2,2], 0.0);
    /// let shape = tensor.shape();
    ///
    /// assert_eq!(shape, vec![2,2]);
    /// ```
    pub fn shape(&self) -> Vec<usize> {
        self.get_dim()
    }
    
    /// Get the Data item of a scalar tensor.
    ///
    /// # Example
    ///
    /// ```
    /// use tinygrad_rs::Tensor;
    /// use tinygrad_rs::Data;
    ///
    /// let tensor = Tensor::with_data(vec![1], Data::from_vec(vec![2.0]));
    ///
    /// assert_eq!(tensor.item(), 2.0);
    /// ```
    pub fn item(&self) -> Data<T> {
        match self.0.borrow().data.len() == 1 {
            true => self.0.borrow().data[0].to_owned(),
            false => panic!("Tensor references multiple pieces of Data")
        }
    }

    fn stride(&self, index: Vec<usize>) -> usize {
        index
            .iter()
            .zip(self.get_strides())
            .map(|(i, s)| i * s)
            .sum::<usize>() + self.get_offset()
    }

    fn get_data(&self) -> Vec<Data<T>> {
        self.0.borrow().data.to_owned()
    }

    fn set_data(&self, data: Vec<Data<T>>) {
        self.0.borrow_mut().data = data;
    }

    fn get_dim(&self) -> Vec<usize> {
        self.0.borrow().dim.to_owned()
    }
    
    fn set_dim(&mut self, dim: Vec<usize>) {
        self.0.borrow_mut().dim = dim;
    }

    fn get_strides(&self) -> Vec<usize> {
        self.0.borrow().strides.to_owned()
    }
    
    fn set_strides(&mut self, strides: Vec<usize>) {
        self.0.borrow_mut().strides = strides;
    }

    fn get_offset(&self) -> usize {
        self.0.borrow().offset.to_owned()
    }

    fn set_offset(&mut self, offset: usize) {
        self.0.borrow_mut().offset = offset;
    }
}
