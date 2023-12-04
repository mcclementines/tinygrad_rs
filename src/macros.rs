//! src/macros.rs

macro_rules! data {
    ($e:literal) => {
        match e {
            _ if $e.to_string().parse::<f32>().is_ok() => Data::new(DataValue::Float($e)),
            _ if $e.to_string().parse::<f64>().is_ok() => Data::new(DataValue::Double($e)),
            _ => panic!("Unsupported type");
        }
    }
}
