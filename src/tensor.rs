//! src/tensor.rs

use std::{cell::RefCell, rc::Rc, usize};

use rand::Rng;

use crate::Data;

/// Represents a Tensor
///
/// # Example
///
/// ```
/// use tinygrad_rs::Tensor;
/// use tinygrad_rs::Data;
///
/// let mut tensor = Tensor::new(vec![2, 2]);
/// tensor.set(vec![0,0], Data::new(1.0));
/// tensor.set(vec![1,1], Data::new(4.0));
///
/// assert_eq!(tensor.get(vec![0,0]).item(), 1.0);
/// assert_eq!(tensor.get(vec![1,1]).item(), 4.0);
/// ```
pub struct Tensor(Rc<RefCell<InnerTensor>>);

struct InnerTensor {
    data: Vec<Data>,
    dim: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,

    requires_grad: bool,
    backward: Box<dyn Fn()>,
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
        data.resize(dim.iter().product(), Data::new(Default::default()));

        let mut stride: usize = dim.iter().product();
        let strides: Vec<usize> = dim
            .iter()
            .map(|x| {
                stride /= x;
                stride
            })
            .collect();

        let offset = 0;
        let requires_grad = false;
        let backward = Box::new(|| {});

        let tensor_data = InnerTensor {
            data,
            dim,
            strides,
            offset,
            requires_grad,
            backward,
        };

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
    pub fn with_data(dim: Vec<usize>, data: Vec<Data>) -> Tensor {
        assert!(
            dim.iter().product::<usize>() <= data.len(),
            "Data does not fit specified dimensions"
        );

        let tensor = Tensor::new(dim);
        tensor.set_data(data);

        tensor
    }

    /// Constructs a tensor with a single value
    ///
    /// # Example
    ///
    /// ```
    /// use tinygrad_rs::Tensor;
    ///
    /// let tensor = Tensor::scalar(2.0);
    ///
    /// assert_eq!(tensor.item(), 2.0);
    /// ```
    pub fn scalar(data: f64) -> Tensor {
        Tensor::with_data(vec![1], data![data])
    }

    /// Constructs a new Tensor with specified dimensions and random data
    ///
    /// # Example
    ///
    /// ```
    /// use tinygrad_rs::Tensor;
    ///
    /// let tensor = Tensor::randn(vec![2,2]);
    /// ```
    pub fn randn(dim: Vec<usize>) -> Tensor {
        let mut rng = rand::thread_rng();

        let size = dim.iter().product();
        let mut data = Vec::with_capacity(size);
        data.resize(size, Data::new(rng.gen()));

        Tensor::with_data(dim, data)
    }

    /// Get the Tensor value located at the specified index
    ///
    /// # Example
    ///
    /// ```
    /// use tinygrad_rs::Tensor;
    ///
    /// let tensor = Tensor::new(vec![2,1]);
    /// let value = tensor.get(vec![1,0]).item();
    ///
    /// assert_eq!(value, 0.0);
    /// ```
    pub fn get(&self, index: Vec<usize>) -> Tensor {
        // TODO: To be replaced when I have time, quick and dirty solution
        // # of dim == index coords
        assert_eq!(
            self.get_dim().len(),
            index.len(),
            "Provided index does not fit tensor"
        );

        // index in range
        assert!(
            self.get_dim()
                .iter()
                .zip(index.iter())
                .all(|(&d, &i)| d > i),
            "Provided index is out of range"
        );

        let stride = self.stride(index);

        Tensor::with_data(vec![1], vec![self.0.borrow().data[stride].clone()])
    }

    /// Get the specified slice of the Tensor. TODO: Range Support
    ///
    /// # Example
    ///
    /// ```
    /// use tinygrad_rs::Tensor;
    /// use tinygrad_rs::Data;
    /// use tinygrad_rs::{data, tslice};
    ///
    /// let tensor = Tensor::with_data(vec![2,2,2], data![1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0]);
    /// let slice = tensor.get_slice(tslice![:, 0, :]);
    ///
    /// assert_eq!(slice.get(vec![0,0]).item(), 1.0);
    /// assert_eq!(slice.get(vec![1,1]).item(), 6.0);
    /// ```
    pub fn get_slice(&self, slice: Vec<isize>) -> Tensor {
        let mut dim: Vec<usize> = Vec::new();
        let mut strides: Vec<usize> = Vec::new();
        let mut offset: usize = 0;

        self.get_dim()
            .iter()
            .enumerate()
            .zip(slice.iter())
            .for_each(|((i, &d), s)| match s {
                -1 => {
                    dim.push(d);
                    strides.push(self.get_strides()[i]);
                }
                _ => {
                    offset += self.get_strides()[i] * *s as usize;
                }
            });

        let tensor = Tensor::with_data(dim, self.get_data());
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
    /// let tensor = Tensor::new(vec![2,1]);
    /// tensor.set(vec![1,0], Data::new(1.0));
    ///
    /// assert_eq!(tensor.get(vec![1,0]).item(), 1.0);
    /// ```
    pub fn set(&self, index: Vec<usize>, value: Data) {
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
    /// let tensor = Tensor::new(vec![2,2]);
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
    ///
    /// let tensor = Tensor::scalar(2.0);
    ///
    /// assert_eq!(tensor.item(), 2.0);
    /// ```
    pub fn item(&self) -> Data {
        match self.0.borrow().data.len() == 1 {
            true => self.0.borrow().data[0].to_owned(),
            false => panic!("Tensor references multiple pieces of Data"),
        }
    }

    /// Add two tensors together
    ///
    /// # Example
    ///
    /// ```
    /// use tinygrad_rs::Tensor;
    /// use tinygrad_rs::Data;
    /// use tinygrad_rs::data;
    ///
    /// let a = Tensor::with_data(vec![2,2], data![1.0, 2.0, 3.0, 4.0]);
    /// let b = Tensor::with_data(vec![2,2], data![2.0, 4.0, 8.0, 16.0]);
    /// let c = a.add(&b);
    ///
    /// // Previous tensors remain unchanged
    /// assert_eq!(a.get(vec![1,1]).item(), 4.0);
    /// assert_eq!(b.get(vec![1,1]).item(), 16.0);
    ///
    /// // Addition results in new tensor
    /// assert_eq!(c.get(vec![1,1]).item(), 20.0);
    /// ```
    pub fn add(&self, rhs: &Tensor) -> Tensor {
        assert!(
            self.shape() == rhs.shape() || (rhs.shape().len() == 1 && rhs.shape()[0] == 1),
            "Tensor shapes do not match or are not scalar"
        );

        let data: Vec<Data> = if rhs.shape().len() == 1 && rhs.shape()[0] == 1 {
            self.get_data()
                .iter()
                .map(|d| Data::new(d.get() + rhs.get(vec![0]).item().get()))
                .collect::<Vec<Data>>()
        } else {
            self.get_data()
                .iter()
                .zip(rhs.get_data())
                .map(|(l, r)| Data::new(l.get() + r.get()))
                .collect::<Vec<Data>>()
        };

        Tensor::with_data(self.get_dim(), data)
    }

    /// Subtract two tensors
    ///
    /// # Example
    ///
    /// ```
    /// use tinygrad_rs::Tensor;
    /// use tinygrad_rs::Data;
    /// use tinygrad_rs::data;
    ///
    /// let a = Tensor::with_data(vec![2,2], data![1.0, 2.0, 3.0, 4.0]);
    /// let b = Tensor::with_data(vec![2,2], data![2.0, 4.0, 8.0, 16.0]);
    /// let c = a.sub(&b);
    ///
    /// // Previous tensors remain unchanged
    /// assert_eq!(a.get(vec![1,1]).item(), 4.0);
    /// assert_eq!(b.get(vec![1,1]).item(), 16.0);
    ///
    /// // Addition results in new tensor
    /// assert_eq!(c.get(vec![1,1]).item(), -12.0);
    /// ```
    pub fn sub(&self, rhs: &Tensor) -> Tensor {
        self.add(&rhs.neg())
    }

    /// Return a tensor with the data negated. Results in a new tensor.
    /// use `Tensor.neg_()` to negate the tensor in place
    ///
    /// # Example
    ///
    /// ```
    /// use tinygrad_rs::Tensor;
    /// use tinygrad_rs::Data;
    /// use tinygrad_rs::data;
    ///
    /// let a = Tensor::with_data(vec![2,2], data![1.0, 2.0, 3.0, 4.0]);
    /// let c = a.neg();
    ///
    /// // Previous tensors remain unchanged
    /// assert_eq!(a.get(vec![1,1]).item(), 4.0);
    ///
    /// // Negation results in new tensor
    /// assert_eq!(c.get(vec![1,1]).item(), -4.0);
    /// ```
    pub fn neg(&self) -> Tensor {
        let data = self
            .get_data()
            .iter()
            .map(|d| Data::new(-d.get()))
            .collect();

        Tensor::with_data(self.get_dim(), data)
    }

    /// Return a tensor with the data negated. Results in a new tensor.
    /// use `Tensor.neg()` return a new tensor with the data negated
    ///
    /// # Example
    ///
    /// ```
    /// use tinygrad_rs::Tensor;
    /// use tinygrad_rs::Data;
    /// use tinygrad_rs::data;
    ///
    /// let a = Tensor::with_data(vec![2,2], data![1.0, 2.0, 3.0, 4.0]);
    /// a.neg_();
    ///
    /// assert_eq!(a.get(vec![1,1]).item(), -4.0);
    /// ```
    pub fn neg_(&self) {
        self.get_data().iter().for_each(|d| d.set(-d.get()));
    }

    /// Element-wise multiplication of tensors. Multiplying by a scalar is also
    /// possible (needs to be wrapped in a Tensor)
    ///
    /// # Example
    ///
    /// ```
    /// use tinygrad_rs::Tensor;
    /// use tinygrad_rs::Data;
    /// use tinygrad_rs::data;
    ///
    /// let a = Tensor::with_data(vec![2,2], data![1.0, 2.0, 3.0, 4.0]);
    /// let b = Tensor::with_data(vec![2,2], data![1.0, 2.0, 3.0, 4.0]);
    /// let scalar = Tensor::with_data(vec![1], data![2.0]);
    ///
    /// let ab = a.mul(&b);
    /// assert_eq!(ab.get(vec![1,1]).item(), 16.0);
    ///
    /// let ascalar = a.mul(&scalar);
    /// assert_eq!(ascalar.get(vec![1,1]).item(), 8.0);
    /// ```
    pub fn mul(&self, rhs: &Tensor) -> Tensor {
        assert!(
            self.shape() == rhs.shape() || (rhs.shape().len() == 1 && rhs.shape()[0] == 1),
            "Tensor shapes do not match or are not scalar"
        );

        let data: Vec<Data> = if self.shape() == rhs.shape() {
            self.get_data()
                .iter()
                .zip(rhs.get_data().iter())
                .map(|(x, y)| Data::new(x.get() * y.get()))
                .collect()
        } else {
            self.get_data()
                .iter()
                .map(|x| Data::new(x.get() * rhs.item().get()))
                .collect()
        };

        Tensor::with_data(self.shape(), data)
    }

    /// Matrix multiplication of tensors. Roughly implemented. A lot of
    /// conditions still need to be handled, as well as broadcasting
    ///
    /// # Example
    ///
    /// ```
    /// use tinygrad_rs::Tensor;
    /// use tinygrad_rs::Data;
    /// use tinygrad_rs::data;
    ///
    /// let a = Tensor::with_data(vec![2,2], data![1.0, 2.0, 3.0, 4.0]);
    /// let b = Tensor::with_data(vec![2,2], data![1.0, 2.0, 3.0, 4.0]);
    ///
    /// let ab = a.matmul(&b);
    /// assert_eq!(ab.get(vec![0,0]).item(), 7.0);
    /// assert_eq!(ab.get(vec![1,1]).item(), 22.0);
    /// ```
    pub fn matmul(&self, rhs: &Tensor) -> Tensor {
        assert!(!self.shape().is_empty() && !rhs.shape().is_empty());

        if self.shape().len() == 1 && self.shape() == rhs.shape() {
            let mut sum = Default::default();

            for i in 0..self.shape()[0] {
                sum += (self.get(vec![i]).item() * rhs.get(vec![i]).item()).get();
            }

            let data = vec![Data::new(sum)];
            let dim = vec![1];

            Tensor::with_data(dim, data)
        } else if rhs.shape().len() == 1 && *self.shape().last().unwrap() == rhs.shape()[0] {
            let data = self
                .get_data()
                .chunks(rhs.shape()[0])
                .flat_map(|c| {
                    Tensor::with_data(rhs.shape(), c.to_vec())
                        .matmul(rhs)
                        .get_data()
                })
                .collect();

            let mut dim = self.shape();
            dim.pop();

            Tensor::with_data(dim, data)
        } else if self.shape().len() == 1 && *rhs.shape().last().unwrap() == self.shape()[0] {
            // need broadcasting
            unimplemented!()
        } else if self.shape().len() == 2
            && rhs.shape().len() == 2
            && self.shape()[1] == rhs.shape()[0]
        {
            let mut data = Vec::new();

            for i in 0..self.shape()[1] {
                let a = self.get_slice(vec![i as isize, -1]);

                for j in 0..rhs.shape()[1] {
                    let b = rhs.get_slice(vec![-1, j as isize]);

                    data.extend(a.matmul(&b).get_data());
                }
            }

            let dim = vec![self.shape()[0], rhs.shape()[1]];

            Tensor::with_data(dim, data)
        } else if self.shape().len() > 2
            && rhs.shape().len() > 2
            && self.shape()[1] == rhs.shape()[0]
        {
            assert_eq!(
                self.shape()[0..self.shape().len() - 2],
                rhs.shape()[0..self.shape().len() - 2]
            );

            let mut dim = self.shape()[0..self.shape().len() - 2].to_vec();
            let mut data = Vec::new();
            let mat_shape = &self.shape()[self.shape().len() - 2..].to_vec();
            let mat_size: usize = mat_shape.iter().product();

            for i in 0..dim.iter().product::<usize>() {
                let imat = Tensor::with_data(
                    mat_shape.to_owned(),
                    self.get_data()[i * mat_size..mat_size * (i + 1)].to_vec(),
                )
                .matmul(&Tensor::with_data(
                    mat_shape.to_owned(),
                    rhs.get_data()[i * mat_size..mat_size * (i + 1)].to_vec(),
                ));

                data.extend(imat.get_data());
            }

            dim.extend(mat_shape);

            Tensor::with_data(dim, data)
        } else {
            panic!("cannot perform matmul on provided tensors");
        }
    }

    /// Batch matrix multiplication of tensors. Roughly implemented. Will
    /// not support broadcasting
    ///
    /// # Example
    ///
    /// ```
    /// use tinygrad_rs::Tensor;
    /// use tinygrad_rs::Data;
    /// use tinygrad_rs::data;
    ///
    /// let a = Tensor::with_data(vec![2,2,2], data![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]);
    /// let b = Tensor::with_data(vec![2,2,2], data![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]);
    ///
    /// let ab = a.bmm(&b);
    /// assert_eq!(ab.get(vec![0,0,0]).item(), 7.0);
    /// assert_eq!(ab.get(vec![1,1,1]).item(), 22.0);
    /// ```
    pub fn bmm(&self, rhs: &Tensor) -> Tensor {
        assert_eq!(
            self.shape()[0..self.shape().len() - 2],
            rhs.shape()[0..self.shape().len() - 2]
        );

        let mut dim = self.shape()[0..self.shape().len() - 2].to_vec();

        if dim.is_empty() {
            return self.matmul(rhs);
        }

        let mut data = Vec::new();
        let mat_shape = &self.shape()[self.shape().len() - 2..].to_vec();
        let mat_size: usize = mat_shape.iter().product();

        for i in 0..dim.iter().product::<usize>() {
            let imat = Tensor::with_data(
                mat_shape.to_owned(),
                self.get_data()[i * mat_size..mat_size * (i + 1)].to_vec(),
            )
            .matmul(&Tensor::with_data(
                mat_shape.to_owned(),
                rhs.get_data()[i * mat_size..mat_size * (i + 1)].to_vec(),
            ));

            data.extend(imat.get_data());
        }

        dim.extend(mat_shape);

        Tensor::with_data(dim, data)
    }

    /// Divide tensor by another tensor
    ///
    /// # Example
    ///
    /// ```
    /// use tinygrad_rs::Tensor;
    /// use tinygrad_rs::Data;
    /// use tinygrad_rs::data;
    ///
    /// let ta = Tensor::with_data(vec![2,2], data![2.0,2.0,2.0,2.0]);
    /// let tb = Tensor::with_data(vec![2,2], data![2.0,2.0,2.0,2.0]);
    /// let result = ta.div(&tb);
    ///
    /// assert_eq!(result.get(vec![1,1]).item(), 1.0);
    /// ```
    pub fn div(&self, rhs: &Tensor) -> Tensor {
        let denominator = rhs.pow(-1.0);

        self.mul(&denominator)
    }

    /// Raise data of tensor to specified power
    ///
    /// # Example
    ///
    /// ```
    /// use tinygrad_rs::Tensor;
    /// use tinygrad_rs::Data;
    /// use tinygrad_rs::data;
    ///
    /// let tensor = Tensor::with_data(vec![2,2], data![2.0,2.0,2.0,2.0]);
    /// let tensor_sq = tensor.pow(2.0);
    ///
    /// assert_eq!(tensor_sq.get(vec![1,1]).item(), 4.0);
    /// ```
    pub fn pow(&self, pow: f64) -> Tensor {
        let data = self
            .get_data()
            .iter()
            .map(|d| Data::new(d.get().powf(pow)))
            .collect();

        Tensor::with_data(self.shape(), data)
    }

    /// Compute the exponential of each element
    ///
    /// # Example
    ///
    /// ```
    /// use tinygrad_rs::Tensor;
    /// use tinygrad_rs::Data;
    /// use tinygrad_rs::data;
    ///
    /// let tensor = Tensor::with_data(vec![2,2], data![2.0,2.0,2.0,2.0]);
    /// let tensor_exp = tensor.exp();
    ///
    /// assert_eq!(tensor_exp.get(vec![1,1]).item(), 7.38905609893065);
    /// ```
    pub fn exp(&self) -> Tensor {
        let data = self
            .get_data()
            .iter()
            .map(|d| Data::new(d.get().exp()))
            .collect();

        Tensor::with_data(self.shape(), data)
    }

    /// Compute the tanh of each element
    ///
    /// # Example
    ///
    /// ```
    /// use tinygrad_rs::Tensor;
    /// use tinygrad_rs::Data;
    /// use tinygrad_rs::data;
    ///
    /// let tensor = Tensor::with_data(vec![2,2], data![2.0,2.0,2.0,2.0]);
    /// let tensor_tanh = tensor.tanh();
    ///
    /// assert_eq!(tensor_tanh.get(vec![1,1]).item(), 0.9640275800758169);
    /// ```
    pub fn tanh(&self) -> Tensor {
        let scalar1 = Tensor::with_data(vec![1], data![1.0]);
        let scalar2 = Tensor::with_data(vec![1], data![2.0]);

        let numerator = (self.mul(&scalar2)).exp().sub(&scalar1);
        let denominator = (self.mul(&scalar2)).exp().add(&scalar1);

        numerator.div(&denominator)
    }

    /// When set to true, autograd capabilities are enabled. Default
    /// value is false
    ///
    /// # Example
    ///
    /// ```
    /// use tinygrad_rs::Tensor;
    ///
    /// let mut tensor = Tensor::randn(vec![2,2,2]);
    /// tensor.requires_grad(true);
    /// ```
    pub fn requires_grad(&self, requires_grad: bool) {
        self.set_requires_grad(requires_grad);
    }

    /// Check if autograd is on. Can be set using `Tensor.requires_grad(bool)`
    ///
    /// # Example
    ///
    /// ```
    /// use tinygrad_rs::Tensor;
    ///
    /// let tensor = Tensor::randn(vec![2,2,2]);
    /// tensor.requires_grad(true);
    ///
    /// assert!(tensor.autograd_on());
    /// ```
    pub fn autograd_on(&self) -> bool {
        self.get_requires_grad()
    }

    /// Zero grad the graph starting from this tensor
    ///
    /// # Example
    ///
    /// ```
    /// use tinygrad_rs::Tensor;
    ///
    /// let tensor = Tensor::randn(vec![2,2,2]);
    /// tensor.zero_grad();
    ///
    /// assert_eq!(tensor.get(vec![0,0,0]).item().get_grad(), 0.0);
    /// ```
    pub fn zero_grad(&self) {
        self.get_data()
            .iter()
            .for_each(|x| x.set_grad(Default::default()));
    }

    /// Perform backward operation from this tensor
    ///
    /// # Example
    ///
    /// ```
    /// use tinygrad_rs::Tensor;
    ///
    /// let tensor = Tensor::randn(vec![2,2,2]);
    /// tensor.backward();
    /// ```
    pub fn backward(&self) {
        (self.0.borrow().backward)();
    }

    fn stride(&self, index: Vec<usize>) -> usize {
        index
            .iter()
            .zip(self.get_strides())
            .map(|(i, s)| i * s)
            .sum::<usize>()
            + self.get_offset()
    }

    fn get_data(&self) -> Vec<Data> {
        self.0.borrow().data.to_owned()
    }

    fn set_data(&self, data: Vec<Data>) {
        self.0.borrow_mut().data = data;
    }

    fn get_dim(&self) -> Vec<usize> {
        self.0.borrow().dim.to_owned()
    }

    fn get_strides(&self) -> Vec<usize> {
        self.0.borrow().strides.to_owned()
    }

    fn set_strides(&self, strides: Vec<usize>) {
        self.0.borrow_mut().strides = strides;
    }

    fn get_offset(&self) -> usize {
        self.0.borrow().offset.to_owned()
    }

    fn set_offset(&self, offset: usize) {
        self.0.borrow_mut().offset = offset;
    }

    fn get_requires_grad(&self) -> bool {
        self.0.borrow().requires_grad.to_owned()
    }

    fn set_requires_grad(&self, requires_grad: bool) {
        self.0.borrow_mut().requires_grad = requires_grad;
    }

    fn backward_(&self) {
        (self.0.borrow().backward)();
    }

    fn set_backward(&self, backward: Box<dyn Fn()>) {
        self.0.borrow_mut().backward = backward;
    }
}
