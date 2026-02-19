use crate::ast::node::Node;
use crate::ast::bytecode::CompiledExpr;
use crate::domain::Domain;
use crate::metrics::dataset::SimdDataset;
use crate::optimization::nelder_mead::optimize_individual_constants;
use crate::ast::format::format_ast;
use std::fmt;

#[derive(Clone)]
pub struct Individual<D: Domain> {
    pub nodes: Vec<Node<D>>,
    pub fitness: f32,
    pub age: usize,
    pub program: Option<CompiledExpr<D>>,
    pub rank: u32,
    pub crowding_distance: f32
}

impl<D: Domain> Individual<D> {
    pub fn new(nodes: Vec<Node<D>>) -> Self {
        Self { 
            nodes, 
            fitness: f32::MAX, 
            age: 0, 
            program: None,
            rank: 0,
            crowding_distance: 0.0 
        }
    }

    pub fn compile(&mut self) {
        if self.program.is_none() {
            self.program = Some(CompiledExpr::from_nodes(&self.nodes));
        }
    }

    pub fn calculate_mse(&mut self, dataset: &SimdDataset) -> f32 {
        if self.program.is_none() {
            self.compile();
        }

        if let Some(prog) = &self.program {
            // Javítás: Átadjuk a code és constants slice-okat
            D::compute_mse(&prog.code, &prog.constants, dataset)
        } else {
            f32::MAX
        }
    }

    pub fn get_constants(&self) -> Vec<D::ScalarValue> {
        self.nodes.iter().filter_map(|node| {
            if let Node::Constant(c) = node { Some(*c) } else { None }
        }).collect()
    }

    pub fn set_constants(&mut self, new_constants: &[D::ScalarValue]) {
        let mut const_idx = 0;
        for node in self.nodes.iter_mut() {
            if let Node::Constant(c) = node {
                if const_idx < new_constants.len() {
                    *c = new_constants[const_idx];
                    const_idx += 1;
                }
            }
        }
        self.invalidate();
    }

    pub fn optimize_constants(&mut self, dataset: &SimdDataset, iterations: usize) {
        optimize_individual_constants(self, dataset, iterations);
        self.compile();
    }

    pub fn simplify(&mut self) {
        if self.nodes.is_empty() { return; }
        self.nodes = D::simplify(&self.nodes);
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
            if current_idx == 0 { break; }
            current_idx -= 1;
        }
        (0, root_idx)
    }

    pub fn complexity(&self) -> usize {
        self.nodes.iter().map(|node| node.weight()).sum()
    }
}

impl<D: Domain> fmt::Display for Individual<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", format_ast(&self.nodes))
    }
}