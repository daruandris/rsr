pub mod nelder_mead;
pub mod cma_es;
pub mod lbfgs;

use crate::engine::individual::Individual;
use crate::metrics::dataset::SimdDataset;
use crate::domain::Domain;
use crate::domain::universal::{UniversalScalar, UniversalOp};
use crate::ast::node::Node;

pub fn optimize_individual_constants<D: Domain<ScalarValue = UniversalScalar, Operator = UniversalOp>>(
    ind: &mut Individual<D>, 
    dataset: &SimdDataset, 
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
            if matches!(*op, UniversalOp::IfElseF | UniversalOp::InverseM2 | UniversalOp::InverseM3) {
                is_differentiable = false;
                break;
            }
        }
    }

    for c in &program.constants {
        if matches!(c, UniversalScalar::Vec2(_) | UniversalScalar::Vec3(_) | UniversalScalar::Mat2(_) | UniversalScalar::Mat3(_)) { 
            requires_cmaes = true;
            break;
        }
    }

    if is_differentiable {
        let lbfgs_iters = 15;
        lbfgs::run_lbfgs(ind, dataset, lbfgs_iters);
    } else if requires_cmaes { 
        let cmaes_iters = (max_iterations / 16).max(2);
        cma_es::run_cma_es(ind, dataset, cmaes_iters);
    } else {
        nelder_mead::run_nelder_mead(ind, dataset, max_iterations);
    }
}