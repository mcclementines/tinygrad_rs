//! src/macros.rs

#[macro_export]
macro_rules! data {
    ($($x:expr),+) => (
        Data::from_vec(vec![$($x),+])
    );
}

#[macro_export]
macro_rules! tslice {
    ($($x:tt),+) => {{
        let mut v = Vec::new();

        $(
            match stringify!($x) {
                ":" => v.push(-1),
                _ => match stringify!($x).parse::<isize>() {
                    Ok(i) => v.push(i),
                    Err(_) => panic!("tslice! macro expects integer literals or ':'"),
                }
            }
        )*

        v
    }};
}
