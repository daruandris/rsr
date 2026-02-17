use crate::ast::eval::evaluate_ast;
use crate::ast::format::format_ast;
use crate::ast::heuristic::simplify_ast;
use crate::ast::node::Node;
use crate::metrics::dataset::SimdDataset;
use crate::metrics::mse::calculate_mse;
use crate::optimization::nelder_mead::optimize_individual_constants;
use std::fmt;

#[derive(Clone)]
pub struct Individual {
    pub nodes: Vec<Node>,
    pub fitness: f64,
}

impl Individual {
    pub fn new(nodes: Vec<Node>) -> Self {
        Self { nodes, fitness: f64::MAX }
    }

    pub fn evaluate(&self, features: &[f64]) -> f64 {
        evaluate_ast(&self.nodes, features)
    }

    pub fn calculate_mse(&self, dataset: &SimdDataset) -> f64 {
        calculate_mse(&self.nodes, dataset)
    }

    pub fn get_constants(&self) -> Vec<f64> {
        self.nodes.iter().filter_map(|node| {
            if let Node::Constant(c) = node { Some(*c) } else { None }
        }).collect()
    }

    pub fn set_constants(&mut self, new_constants: &[f64]) {
        let mut const_idx = 0;
        for node in self.nodes.iter_mut() {
            if let Node::Constant(c) = node {
                if const_idx < new_constants.len() {
                    *c = new_constants[const_idx];
                    const_idx += 1;
                }
            }
        }
    }

    pub fn optimize_constants(&mut self, dataset: &SimdDataset, iterations: usize) {
        optimize_individual_constants(self, dataset, iterations);
    }

    pub fn simplify(&mut self) {
        if self.nodes.is_empty() { return; }
        self.nodes = simplify_ast(&self.nodes);
        self.fitness = f64::MAX;
    }

    pub fn get_subtree_bounds(&self, root_idx: usize) -> (usize, usize) {
        let mut needed = 1;
        let mut current_idx = root_idx;
        loop {
            needed = needed + self.nodes[current_idx].arity() - 1;
            if needed == 0 {
                return (current_idx, root_idx);
            }
            if current_idx == 0 {
                break;
            }
            current_idx -= 1;
        }
        (0, root_idx)
    }

    pub fn complexity(&self) -> usize{
        self.nodes.iter().map(|node| node.weight()).sum()
    }
}

impl fmt::Display for Individual {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", format_ast(&self.nodes))
    }
}