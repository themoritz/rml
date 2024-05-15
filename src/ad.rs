use ndarray::prelude::*;
use petgraph::algo::toposort;
use petgraph::graph::{DiGraph, NodeIndex};

pub type T = ArrayD<f64>;

pub type Ix = NodeIndex;

#[derive(Debug, Clone)]
enum Expr {
    AddVec { left: Ix, right: Ix },
    MultMat { mat: Ix, vec: Ix },
    Sigma { vec: Ix },
    Relu { vec: Ix },
    Var { _name: String },
    Loss { expected: Ix, actual: Ix },
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
    order: Vec<Ix>,
}

impl Tape {
    pub fn init() -> Self {
        Self {
            graph: DiGraph::new(),
            order: vec![],
        }
    }

    /// Add two vectors together.
    pub fn add_vec(&mut self, left: Ix, right: Ix) -> Ix {
        let new = self.graph.add_node(Node::new(Expr::AddVec { left, right }));
        self.graph.add_edge(left, new, ());
        self.graph.add_edge(right, new, ());
        new
    }

    /// Multiply a vector by a matrix.
    pub fn mult_mat(&mut self, mat: Ix, vec: Ix) -> Ix {
        let new = self.graph.add_node(Node::new(Expr::MultMat { mat, vec }));
        self.graph.add_edge(mat, new, ());
        self.graph.add_edge(vec, new, ());
        new
    }

    /// Apply the sigma function to every element of the vector.
    pub fn sigma(&mut self, vec: Ix) -> Ix {
        let new = self.graph.add_node(Node::new(Expr::Sigma { vec }));
        self.graph.add_edge(vec, new, ());
        new
    }

    /// Apply the ReLU function to every element of the vector.
    pub fn relu(&mut self, vec: Ix) -> Ix {
        let new = self.graph.add_node(Node::new(Expr::Relu { vec }));
        self.graph.add_edge(vec, new, ());
        new
    }

    /// Introduce a free variable.
    pub fn var(&mut self, name: &str) -> Ix {
        self.graph.add_node(Node::new(Expr::Var {
            _name: name.to_string(),
        }))
    }

    /// Loss function
    pub fn loss(&mut self, expected: Ix, actual: Ix) -> Ix {
        let new = self
            .graph
            .add_node(Node::new(Expr::Loss { expected, actual }));
        self.graph.add_edge(actual, new, ());
        new
    }

    fn node(&self, ix: &Ix) -> &Node {
        self.graph.node_weight(*ix).unwrap()
    }

    pub fn set_val(&mut self, ix: Ix, val: T) {
        self.graph.node_weight_mut(ix).unwrap().z = val;
    }

    pub fn get_val(&self, ix: Ix) -> T {
        self.graph.node_weight(ix).unwrap().z.clone()
    }

    pub fn get_grad(&self, ix: Ix) -> T {
        self.graph.node_weight(ix).unwrap().w.clone()
    }

    /// Call after setting up the expression graph.
    pub fn compile(&mut self) {
        self.order = toposort(&self.graph, None).unwrap();
    }

    /// Forward evaluation.
    pub fn eval(&mut self) {
        for ix in &self.order {
            self.graph.node_weight_mut(*ix).unwrap().z = match &self.node(ix).expr {
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
                Expr::Var { .. } => self.node(ix).z.clone(),
                Expr::Loss { expected, actual } => arr0(
                    -0.5 * self
                        .node(expected)
                        .z
                        .into_iter()
                        .zip(self.node(actual).z.into_iter())
                        .map(|(y, a)| (y - a) * (y - a))
                        .sum::<f64>(),
                )
                .into_dyn(),
            };
        }
    }

    /// Backward propagation.
    pub fn grad(&mut self, output: Ix) {
        for n in self.graph.node_weights_mut() {
            n.w = T::zeros(n.z.shape());
        }
        let n = self.graph.node_weight_mut(output).unwrap();
        n.w = T::ones(n.z.shape());

        for ix in self.order.iter().rev() {
            let n = self.node(ix).clone();
            let w = &n.w;
            match n.expr {
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
                Expr::Var { .. } => {}
                Expr::Loss { expected, actual } => {
                    let deriv: T = w
                        .clone()
                        .into_dimensionality::<Ix0>()
                        .unwrap()
                        .into_scalar()
                        * (&self.node(&expected).z - &self.node(&actual).z);
                    self.graph.node_weight_mut(actual).unwrap().w += &deriv;
                }
            }
        }
    }
}

pub fn outer_product(a: Array1<f64>, b: Array1<f64>) -> Array2<f64> {
    let a_len = a.len();
    let b_len = b.len();
    a.into_shape((a_len, 1))
        .unwrap()
        .dot(&b.into_shape((1, b_len)).unwrap())
}

pub fn example() -> Tape {
    let mut t = Tape::init();

    let a0 = t.var("a0");
    let w1 = t.var("w1");
    let b1 = t.var("b1");
    let m1 = t.mult_mat(w1, a0);
    let z1 = t.add_vec(m1, b1);
    let a1 = t.relu(z1);
    let y = t.var("y");
    let l = t.loss(y, a1);

    t.compile();

    t.set_val(a0, array![1.0, 2.0, 3.0].into_dyn());
    t.set_val(w1, array![[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]].into_dyn());
    t.set_val(b1, array![1.0, 1.0].into_dyn());
    t.set_val(y, array![6.0, 12.0].into_dyn());

    t.eval();
    t.grad(l);

    t
}
