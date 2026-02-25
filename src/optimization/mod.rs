pub mod nelder_mead;
pub mod cma_es;
pub mod lbfgs; // <-- ÚJ modulunk!

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

    // --- ZERO-COST ROUTER ---
    let mut is_differentiable = true;
    let mut requires_cmaes = false;

    // 1. Vizsgáljuk a fát a differenciálhatóság miatt
    for node in &ind.nodes {
        if let Node::Operator(op) = node {
            // Ide jöhetnek később a nem differenciálható operátorok (pl. Step, Sign, Abs)
            if matches!(*op, UniversalOp::IfElseF | UniversalOp::InverseM2 | UniversalOp::InverseM3) {
                is_differentiable = false;
                break;
            }
        }
    }

    // 2. Vizsgáljuk a típusokat a dimenziók miatt
    for c in &program.constants {
        // Ha összetett adattípusunk van, a Nelder-Mead dimenzió-független 
        // megközelítése már nem elég hatékony. [cite: 645]
        if matches!(c, UniversalScalar::Vec2(_) | UniversalScalar::Vec3(_) | UniversalScalar::Mat2(_) | UniversalScalar::Mat3(_)) { 
            requires_cmaes = true;
            break;
        }
    }

    // 3. Intelligens útválasztás (Routing)
    if is_differentiable {
        // L-BFGS a leggyorsabb és leghatékonyabb a sima (smooth) problémákra.
        // A kvázi-Newton módszereknek kevesebb iteráció is elég a konvergenciához.
        let lbfgs_iters = 15;
        lbfgs::run_lbfgs(ind, dataset, lbfgs_iters);
    } else if requires_cmaes { 
        // Nem differenciálható, de többdimenziós (Linalg) struktúrákat tartalmaz
        let cmaes_iters = (max_iterations / 16).max(2);
        cma_es::run_cma_es(ind, dataset, cmaes_iters);
    } else {
        // Sima skaláris (f32), de nem differenciálható függvényekhez
        // A meglévő, tökéletesen optimalizált Nelder-Mead hívása [cite: 648]
        nelder_mead::run_nelder_mead(ind, dataset, max_iterations);
    }
}