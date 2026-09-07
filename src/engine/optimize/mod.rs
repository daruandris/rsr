use crate::engine::data::dataset::Dataset;
use crate::engine::eval::scalar::Scalar;
use crate::engine::expr::node::Node;
use crate::engine::search::individual::Individual;
use crate::engine::search::config::LossFunctionType;

mod cmaes;
mod lbfgs;
mod nelder_mead;

pub fn optimize_individual_constants(
    ind: &mut Individual,
    dataset: &Dataset,
    max_iterations: usize,
    loss_type: LossFunctionType,
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

    if loss_type == LossFunctionType::TensorMseMat2 {
        is_differentiable = false;
        requires_cmaes = true;
    }

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
        lbfgs::run_lbfgs(ind, dataset, lbfgs_iters, loss_type);
    } else if requires_cmaes {
        let cmaes_iters = (max_iterations / 16).max(2);
        cmaes::run_cma_es(ind, dataset, cmaes_iters, loss_type);
    } else {
        nelder_mead::run_nelder_mead(ind, dataset, max_iterations, loss_type);
    }
}

pub trait Parameterized {
    fn param_count(&self) -> usize;
    fn flatten_params(&self, buffer: &mut [f32]);
    fn unflatten_params(&mut self, buffer: &[f32]);
}
