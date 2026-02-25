pub mod cmaes;
pub mod lbfgs;
pub mod nelder_mead;

use crate::data::dataset::Dataset;
use crate::eval::op::Op;
use crate::eval::scalar::Scalar;
use crate::expr::node::Node;
use crate::search::individual::Individual;

pub fn optimize_individual_constants(
    ind: &mut Individual, 
    dataset: &Dataset, 
    max_iterations: usize
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

    for node in &ind.nodes {
        if let Node::Operator(op) = node {
            if matches!(*op, Op::IfElseF | Op::InverseM2 | Op::InverseM3) {
                is_differentiable = false;
                break;
            }
        }
    }

    for c in &program.constants {
        if matches!(c, Scalar::Vec2(_) | Scalar::Vec3(_) | Scalar::Mat2(_) | Scalar::Mat3(_)) { 
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