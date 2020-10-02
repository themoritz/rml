extern crate blas_src;

use cblas::*;
use ndarray::prelude::*;

fn nd_mult() {
    let a = array![[1.0, 4.0], [2.0, 5.0], [3.0, 6.0],];
    let b = array![
        [1.0, 5.0, 9.0],
        [2.0, 6.0, 10.0],
        [3.0, 7.0, 11.0],
        [4.0, 8.0, 12.0],
    ];
    let c = b.dot(&a);
}

fn blas_mult() {
    let (m, n, k) = (2, 4, 3);
    let a = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
    let b = vec![
        1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0,
    ];
    let mut c = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    unsafe {
        dgemm(
            Layout::ColumnMajor,
            Transpose::None,
            Transpose::None,
            m,
            n,
            k,
            1.0,
            &a,
            m,
            &b,
            k,
            0.0,
            &mut c,
            m,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nd_mult() {
        nd_mult()
    }
}

#[cfg(test)]
mod benchmarks {
    use super::*;

    extern crate test;
    use test::{black_box, Bencher};

    #[bench]
    fn b_blas_mult(b: &mut Bencher) {
        b.iter(|| black_box({ blas_mult() }));
    }

    #[bench]
    fn b_nd_mult(b: &mut Bencher) {
        b.iter(|| black_box({ nd_mult() }));
    }
}
