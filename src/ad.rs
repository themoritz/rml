use petgraph::algo::toposort;
use petgraph::graph::{DiGraph, NodeIndex};
use std::collections::HashMap;

#[derive(Debug)]
enum Expr {
    Add { left: NodeIndex, right: NodeIndex },
    Mult { left: NodeIndex, right: NodeIndex },
    Var { name: String },
    Sin { sub: NodeIndex },
    Lit { value: f64 },
}

#[derive(Debug)]
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

    fn node(&mut self, ix: NodeIndex) -> &Node {
        self.graph.node_weight(ix).unwrap()
    }

    pub fn eval(&mut self, var_values: &[(NodeIndex, f64)]) -> f64 {
        for (ix, value) in var_values {
            let node = self.graph.node_weight_mut(*ix).unwrap();
            node.z = *value;
        }
        let order = toposort(&self.graph, None).unwrap();
        let mut value = 0.0;
        for ix in order {
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
}

pub fn example() -> (Tape, f64) {
    let mut t = Tape::init();
    let x1 = t.var("x1");
    let x2 = t.var("x2");
    let sin = t.sin(x1);
    let mult = t.mult(x1, x2);
    t.add(sin, mult);
    let result = t.eval(&[(x1, 1.0), (x2, 1.0)]);
    (t, result)
}
