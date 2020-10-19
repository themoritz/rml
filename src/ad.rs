use petgraph::algo::toposort;
use petgraph::graph::{DiGraph, NodeIndex};
use std::collections::HashMap;

#[derive(Debug, Clone)]
enum Expr {
    Add { left: NodeIndex, right: NodeIndex },
    Mult { left: NodeIndex, right: NodeIndex },
    Var { name: String },
    Sin { sub: NodeIndex },
    Lit { value: f64 },
}

#[derive(Debug, Clone)]
struct Node {
    expr: Expr,
    z: f64,
    w: f64,
}

impl Node {
    fn new(expr: Expr) -> Self {
        Self {
            expr,
            z: 0.0,
            w: 0.0,
        }
    }
}

#[derive(Debug)]
pub struct Tape {
    graph: DiGraph<Node, ()>,
}

impl Tape {
    fn init() -> Self {
        Self {
            graph: DiGraph::new(),
        }
    }

    fn add(&mut self, left: NodeIndex, right: NodeIndex) -> NodeIndex {
        let new = self.graph.add_node(Node::new(Expr::Add { left, right }));
        self.graph.add_edge(left, new, ());
        self.graph.add_edge(right, new, ());
        new
    }

    fn mult(&mut self, left: NodeIndex, right: NodeIndex) -> NodeIndex {
        let new = self.graph.add_node(Node::new(Expr::Mult { left, right }));
        self.graph.add_edge(left, new, ());
        self.graph.add_edge(right, new, ());
        new
    }

    fn var(&mut self, name: &str) -> NodeIndex {
        self.graph.add_node(Node::new(Expr::Var {
            name: name.to_string(),
        }))
    }

    fn sin(&mut self, sub: NodeIndex) -> NodeIndex {
        let new = self.graph.add_node(Node::new(Expr::Sin { sub }));
        self.graph.add_edge(sub, new, ());
        new
    }

    fn lit(&mut self, value: f64) -> NodeIndex {
        self.graph.add_node(Node::new(Expr::Lit { value }))
    }

    fn node(&self, ix: NodeIndex) -> &Node {
        self.graph.node_weight(ix).unwrap()
    }

    pub fn eval(&mut self, var_values: &[(NodeIndex, f64)]) -> f64 {
        for (ix, value) in var_values {
            let node = self.graph.node_weight_mut(*ix).unwrap();
            node.z = *value;
        }
        let mut value = 0.0;
        for ix in toposort(&self.graph, None).unwrap() {
            value = match self.node(ix).expr {
                Expr::Add { left, right } => self.node(left).z + self.node(right).z,
                Expr::Mult { left, right } => self.node(left).z * self.node(right).z,
                Expr::Var { .. } => self.node(ix).z,
                Expr::Sin { sub } => self.node(sub).z.sin(),
                Expr::Lit { value } => value,
            };
            let node = self.graph.node_weight_mut(ix).unwrap();
            node.z = value;
        }
        value
    }

    pub fn grad(&mut self, output: NodeIndex) -> HashMap<String, f64> {
        for n in self.graph.node_weights_mut() {
            n.w = 0.0;
        }
        self.graph.node_weight_mut(output).unwrap().w = 1.0;
        self.graph.reverse();
        let mut result: HashMap<String, f64> = HashMap::new();
        for ix in toposort(&self.graph, None).unwrap() {
            let n = self.node(ix).clone();
            let w = n.w;
            match n.expr {
                Expr::Add { left, right } => {
                    self.graph.node_weight_mut(left).unwrap().w += w;
                    self.graph.node_weight_mut(right).unwrap().w += w;
                }
                Expr::Mult { left, right } => {
                    self.graph.node_weight_mut(left).unwrap().w += w * self.node(right).z;
                    self.graph.node_weight_mut(right).unwrap().w += w * self.node(left).z;
                }
                Expr::Var { name } => {
                    result.insert(name.clone(), n.w);
                }
                Expr::Sin { sub } => {
                    self.graph.node_weight_mut(sub).unwrap().w += w * self.node(sub).z.cos();
                }
                Expr::Lit { .. } => {}
            }
        }
        self.graph.reverse();
        result
    }
}

pub fn example() -> (Tape, f64, HashMap<String, f64>) {
    let mut t = Tape::init();
    let x1 = t.var("x1");
    let x2 = t.var("x2");
    let sin = t.sin(x1);
    let mult = t.mult(x1, x2);
    let lit = t.lit(1.0);
    let add = t.add(sin, mult);
    let y = t.add(add, lit);
    let result = t.eval(&[(x1, 1.0), (x2, 2.0)]);
    let grad = t.grad(y);
    (t, result, grad)
}

pub mod vec {
    use ndarray::prelude::*;
    use petgraph::algo::toposort;
    use petgraph::graph::{DiGraph, NodeIndex};
    use std::collections::HashMap;

    type T = ArrayD<f64>;

    #[derive(Debug, Clone)]
    enum Expr {
        LitVec { value: T },
        AddVec { left: NodeIndex, right: NodeIndex },
        MultMat { mat: NodeIndex, vec: NodeIndex },
        Sigma { vec: NodeIndex },
        Relu { vec: NodeIndex },
        Var { name: String },
        Loss { expected: T, actual: NodeIndex },
    }

    #[derive(Debug, Clone)]
    struct Node {
        expr: Expr,
        z: T,
        w: T,
    }

    impl Node {
        fn new(expr: Expr) -> Self {
            Self {
                expr,
                z: T::zeros(IxDyn(&[0])),
                w: T::zeros(IxDyn(&[0])),
            }
        }
    }

    fn sigma(x: &f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn sigma_deriv(x: &f64) -> f64 {
        let ex = (-x).exp();
        ex / (1.0 + ex) / (1.0 + ex)
    }

    fn relu(x: &f64) -> f64 {
        if *x > 0.0 {
            *x
        } else {
            0.0
        }
    }

    fn relu_deriv(x: &f64) -> f64 {
        if *x > 0.0 {
            1.0
        } else {
            0.0
        }
    }

    #[derive(Debug)]
    pub struct Tape {
        graph: DiGraph<Node, ()>,
    }

    impl Tape {
        fn init() -> Self {
            Self {
                graph: DiGraph::new(),
            }
        }

        fn lit(&mut self, value: T) -> NodeIndex {
            self.graph.add_node(Node::new(Expr::LitVec { value }))
        }

        fn add_vec(&mut self, left: NodeIndex, right: NodeIndex) -> NodeIndex {
            let new = self.graph.add_node(Node::new(Expr::AddVec { left, right }));
            self.graph.add_edge(left, new, ());
            self.graph.add_edge(right, new, ());
            new
        }

        fn mult_mat(&mut self, mat: NodeIndex, vec: NodeIndex) -> NodeIndex {
            let new = self.graph.add_node(Node::new(Expr::MultMat { mat, vec }));
            self.graph.add_edge(mat, new, ());
            self.graph.add_edge(vec, new, ());
            new
        }

        fn sigma(&mut self, vec: NodeIndex) -> NodeIndex {
            let new = self.graph.add_node(Node::new(Expr::Sigma { vec }));
            self.graph.add_edge(vec, new, ());
            new
        }

        fn relu(&mut self, vec: NodeIndex) -> NodeIndex {
            let new = self.graph.add_node(Node::new(Expr::Relu { vec }));
            self.graph.add_edge(vec, new, ());
            new
        }

        fn var(&mut self, name: &str) -> NodeIndex {
            self.graph.add_node(Node::new(Expr::Var {
                name: name.to_string(),
            }))
        }

        fn loss(&mut self, expected: T, actual: NodeIndex) -> NodeIndex {
            let new = self
                .graph
                .add_node(Node::new(Expr::Loss { expected, actual }));
            self.graph.add_edge(actual, new, ());
            new
        }

        fn node(&self, ix: &NodeIndex) -> &Node {
            self.graph.node_weight(*ix).unwrap()
        }

        fn eval(&mut self, var_values: Vec<(NodeIndex, T)>) -> T {
            for (ix, value) in var_values {
                self.graph.node_weight_mut(ix).unwrap().z = value;
            }
            let mut value = T::zeros(IxDyn(&[]));
            for ix in toposort(&self.graph, None).unwrap() {
                value = match &self.node(&ix).expr {
                    Expr::LitVec { value } => value.clone(),
                    Expr::AddVec { left, right } => &self.node(left).z + &self.node(right).z,
                    Expr::MultMat { mat, vec } => self
                        .node(mat)
                        .z
                        .clone()
                        .into_dimensionality::<Ix2>()
                        .unwrap()
                        .dot(
                            &self
                                .node(vec)
                                .z
                                .clone()
                                .into_dimensionality::<Ix1>()
                                .unwrap(),
                        )
                        .into_dyn(),
                    Expr::Sigma { vec } => self.node(vec).z.map(sigma),
                    Expr::Relu { vec } => self.node(vec).z.map(relu),
                    Expr::Var { .. } => self.node(&ix).z.clone(),
                    Expr::Loss { expected, actual } => arr0(
                        -0.5 * expected
                            .into_iter()
                            .zip(self.node(actual).z.into_iter())
                            .map(|(y, a)| (y - a) * (y - a))
                            .sum::<f64>(),
                    )
                    .into_dyn(),
                };
                let node = self.graph.node_weight_mut(ix).unwrap();
                node.z = value.clone();
            }
            value
        }

        fn grad(&mut self, output: NodeIndex) -> HashMap<String, T> {
            for n in self.graph.node_weights_mut() {
                n.w = T::zeros(n.z.shape());
            }
            let n = self.graph.node_weight_mut(output).unwrap();
            n.w = T::ones(n.z.shape());
            self.graph.reverse();
            let mut result: HashMap<String, T> = HashMap::new();
            for ix in toposort(&self.graph, None).unwrap() {
                let n = self.node(&ix).clone();
                let w = &n.w;
                match n.expr {
                    Expr::LitVec { .. } => {}
                    Expr::AddVec { left, right } => {
                        self.graph.node_weight_mut(left).unwrap().w += w;
                        self.graph.node_weight_mut(right).unwrap().w += w;
                    }
                    Expr::MultMat { mat, vec } => {
                        let deriv: T = self
                            .node(&mat)
                            .z
                            .t()
                            .into_dimensionality::<Ix2>()
                            .unwrap()
                            .dot(&w.clone().into_dimensionality::<Ix1>().unwrap())
                            .into_dyn();
                        self.graph.node_weight_mut(vec).unwrap().w += &deriv;

                        let deriv: T = outer_product(
                            w.clone().into_dimensionality::<Ix1>().unwrap(),
                            self.node(&vec)
                                .z
                                .clone()
                                .into_dimensionality::<Ix1>()
                                .unwrap(),
                        )
                        .into_dyn();
                        self.graph.node_weight_mut(mat).unwrap().w += &deriv;
                    }
                    Expr::Sigma { vec } => {
                        let deriv: T = w * &self.node(&vec).z.map(sigma_deriv);
                        self.graph.node_weight_mut(vec).unwrap().w += &deriv;
                    }
                    Expr::Relu { vec } => {
                        let deriv: T = w * &self.node(&vec).z.map(relu_deriv);
                        self.graph.node_weight_mut(vec).unwrap().w += &deriv;
                    }
                    Expr::Var { name } => {
                        result.insert(name.clone(), n.w);
                    }
                    Expr::Loss { expected, actual } => {
                        let deriv: T = w
                            .clone()
                            .into_dimensionality::<Ix0>()
                            .unwrap()
                            .into_scalar()
                            * (&expected - &self.node(&actual).z);
                        self.graph.node_weight_mut(actual).unwrap().w += &deriv;
                    }
                }
            }
            self.graph.reverse();
            result
        }
    }

    pub fn outer_product(a: Array1<f64>, b: Array1<f64>) -> Array2<f64> {
        let a_len = a.len();
        let b_len = b.len();
        a.into_shape((a_len, 1))
            .unwrap()
            .dot(&b.into_shape((1, b_len)).unwrap())
    }

    pub fn example() -> (Tape, T, HashMap<String, T>) {
        let mut t = Tape::init();
        let a0 = t.lit(array![1.0, 2.0, 3.0].into_dyn());
        let w1 = t.var("w1");
        let b1 = t.var("b1");
        let m1 = t.mult_mat(w1, a0);
        let z1 = t.add_vec(m1, b1);
        let a1 = t.relu(z1);
        let l = t.loss(array![6.0, 12.0].into_dyn(), a1);

        // let w2 = t.var("w2");
        // let b2 = t.var("b2");
        // let m2 = t.mult_mat(w2, a1);
        // let z2 = t.add_vec(m2, b2);
        // let a2 = t.sigma(z2);

        let result = t.eval(vec![
            (w1, array![[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]].into_dyn()),
            (b1, array![1.0, 1.0].into_dyn()),
        ]);
        let grad = t.grad(l);
        (t, result, grad)
    }
}
