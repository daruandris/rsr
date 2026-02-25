use std::fmt;
use crate::data::dataset::Dataset;
use crate::eval::evaluator;
use crate::eval::scalar::Scalar;
use crate::expr::format::format_ast;
use crate::expr::node::Node;
use crate::expr::program::Program;
use crate::expr::simplify::simplify_ast;
use crate::optimize::optimize_individual_constants;

#[derive(Clone)]
pub struct Individual {
    pub nodes: Vec<Node>,
    pub fitness: f32,
    pub age: usize,
    pub program: Option<Program>,
    pub rank: u32,
    pub crowding_distance: f32,
}

impl Individual {
    pub fn new(nodes: Vec<Node>) -> Self {
        Self {
            nodes,
            fitness: f32::MAX,
            age: 0,
            program: None,
            rank: 0,
            crowding_distance: 0.0,
        }
    }

    pub fn compile(&mut self) {
        if self.program.is_none() {
            self.program = Some(Program::from_nodes(&self.nodes));
        }
    }

    pub fn calculate_mse(&mut self, dataset: &Dataset) -> f32 {
        if self.program.is_none() {
            self.compile();
        }

        if let Some(prog) = &self.program {
            evaluator::compute_mse(prog, dataset)
        } else {
            f32::MAX
        }
    }

    pub fn get_constants(&self) -> Vec<Scalar> {
        self.nodes
            .iter()
            .filter_map(|node| {
                if let Node::Constant(c, _) = node {
                    Some(*c)
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn set_constants(&mut self, new_constants: &[Scalar]) {
        let mut const_idx = 0;
        for node in self.nodes.iter_mut() {
            if let Node::Constant(c, _) = node {
                if const_idx < new_constants.len() {
                    *c = new_constants[const_idx];
                    const_idx += 1;
                }
            }
        }
        self.invalidate();
    }

    pub fn optimize_constants(&mut self, dataset: &Dataset, iterations: usize) {
        optimize_individual_constants(self, dataset, iterations);
        let threshold = 0.05;
        let mut constants = self.get_constants();
        for c in constants.iter_mut() {
            c.apply_threshold(threshold);
        }
        self.set_constants(&constants);
        self.simplify();
        self.compile();
    }

    pub fn simplify(&mut self) {
        if self.nodes.is_empty() {
            return;
        }
        self.nodes = simplify_ast(&self.nodes);
        self.invalidate();
    }

    pub fn invalidate(&mut self) {
        self.program = None;
        self.fitness = f32::MAX;
    }

    pub fn get_subtree_bounds(&self, root_idx: usize) -> (usize, usize) {
        let mut needed = 1;
        let mut current_idx = root_idx;
        loop {
            needed += self.nodes[current_idx].arity() as isize - 1;
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

    pub fn complexity(&self) -> usize {
        self.nodes.iter().map(|node| node.weight()).sum()
    }
}

impl fmt::Display for Individual {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", format_ast(&self.nodes))
    }
}