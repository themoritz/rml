use crate::ad::{Ix, Tape, T};

pub struct Net {
    tape: Tape,
    parameters: Vec<(Ix, Ix)>,
    input: Ix,
    expected: Ix,
    output: Ix,
}

impl Net {
    pub fn new(layers: &[usize]) -> Self {
        assert!(
            layers.len() >= 2,
            "Expected at least 2 layers, found {}",
            layers.len()
        );

        let mut parameters = Vec::with_capacity(layers.len() - 1);

        let mut t = Tape::init();
        let mut a = t.var("a0");
        let input = a;

        for layer in 1..layers.len() {
            let b = t.var(&format!("b{}", layer));
            let w = t.var(&format!("w{}", layer));
            let m = t.mult_mat(w, a);
            let z = t.add_vec(m, b);
            a = t.sigma(z);
            parameters.push((w, b));
        }

        let y = t.var("y");
        let loss = t.loss(y, a);

        t.compile();

        Self {
            tape: t,
            parameters,
            input,
            expected: y,
            output: loss,
        }
    }
}
