use crate::engine::data::dataset::Dataset;
use crate::engine::eval::scalar::Scalar;
use crate::engine::expr::node::Node;
use crate::engine::search::individual::Individual;

mod cmaes;
mod lbfgs;
mod nelder_mead;

pub fn optimize_individual_constants(
    ind: &mut Individual,
    dataset: &Dataset,
    max_iterations: usize,
) {
    if ind.program.is_none() {
        ind.compile();
    }

    let program = match &ind.program {
        Some(p) => p,
        None => return,
    };

    let mut is_differentiable = true;
    let mut requires_cmaes = false;

    // Dinamikus lekérdezés a fában lévő műveletektől
    for node in &ind.nodes {
        if let Node::Operator(op) = node {
            if !op.is_differentiable() {
                is_differentiable = false;
            }
            if op.requires_cmaes() {
                requires_cmaes = true;
            }
        }
    }

    // A konstansok típusai is kikényszeríthetik a CMA-ES-t
    for c in &program.constants {
        if matches!(
            c,
            Scalar::Vec2(_) | Scalar::Vec3(_) | Scalar::Mat2(_) | Scalar::Mat3(_)
        ) {
            requires_cmaes = true;
            break;
        }
    }

    if is_differentiable {
        let lbfgs_iters = 15;
        lbfgs::run_lbfgs(ind, dataset, lbfgs_iters);
    } else if requires_cmaes {
        let cmaes_iters = (max_iterations / 16).max(2);
        cmaes::run_cma_es(ind, dataset, cmaes_iters);
    } else {
        nelder_mead::run_nelder_mead(ind, dataset, max_iterations);
    }
}

pub trait Parameterized {
    fn param_count(&self) -> usize;
    fn flatten_params(&self, buffer: &mut [f32]);
    fn unflatten_params(&mut self, buffer: &[f32]);
}