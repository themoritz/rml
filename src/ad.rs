use ndarray::prelude::*;
use petgraph::algo::toposort;
use petgraph::graph::{DiGraph, NodeIndex};

type T = ArrayD<f64>;

#[derive(Debug, Clone)]
enum Expr {
    AddVec { left: NodeIndex, right: NodeIndex },
    MultMat { mat: NodeIndex, vec: NodeIndex },
    Sigma { vec: NodeIndex },
    Relu { vec: NodeIndex },
    Var { name: String },
    Loss { expected: T, actual: NodeIndex },
}

#[derive(Debug, Clone)]
pub struct Node {
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
    pub graph: DiGraph<Node, ()>,
    order: Vec<NodeIndex>,
}

impl Tape {
    fn init() -> Self {
        Self {
            graph: DiGraph::new(),
            order: vec![],
        }
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

    fn set_val(&mut self, ix: NodeIndex, val: T) {
        self.graph.node_weight_mut(ix).unwrap().z = val;
    }

    fn get_val(&self, ix: NodeIndex) -> T {
        self.graph.node_weight(ix).unwrap().z.clone()
    }

    fn compile(&mut self) {
        self.order = toposort(&self.graph, None).unwrap();
    }

    fn eval(&mut self) {
        for ix in &self.order {
            self.graph.node_weight_mut(*ix).unwrap().z = match &self.node(&ix).expr {
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
        }
    }

    fn grad(&mut self, output: NodeIndex) {
        for n in self.graph.node_weights_mut() {
            n.w = T::zeros(n.z.shape());
        }
        let n = self.graph.node_weight_mut(output).unwrap();
        n.w = T::ones(n.z.shape());

        for ix in self.order.iter().rev() {
            let n = self.node(&ix).clone();
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
                        * (&expected - &self.node(&actual).z);
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
    let l = t.loss(array![6.0, 12.0].into_dyn(), a1);

    t.compile();

    t.set_val(a0, array![1.0, 2.0, 3.0].into_dyn());
    t.set_val(w1, array![[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]].into_dyn());
    t.set_val(b1, array![1.0, 1.0].into_dyn());

    t.eval();
    t.grad(l);

    t
}
