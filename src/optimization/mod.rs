pub mod nelder_mead;
pub mod cma_es;

use crate::engine::individual::Individual;
use crate::metrics::dataset::SimdDataset;
use crate::domain::Domain;
use crate::domain::universal::UniversalScalar;

pub fn optimize_individual_constants<D: Domain<ScalarValue = UniversalScalar>>(
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
    let mut requires_cmaes = false;
    for c in &program.constants {
        // Ha összetett adattípusunk van, a Nelder-Mead dimenzió-független 
        // megközelítése már nem elég hatékony, bevetjük a CMA-ES-t.
        if matches!(c, UniversalScalar::Vec2(_) | UniversalScalar::Vec3(_) | UniversalScalar::Mat2(_) | UniversalScalar::Mat3(_)) {
            requires_cmaes = true;
            break;
        }
    }

    if requires_cmaes {
        let cmaes_iters = (max_iterations / 16).max(2);
        cma_es::run_cma_es(ind, dataset, cmaes_iters);
    } else {
        // A meglévő, tökéletesen optimalizált Nelder-Mead hívása
        nelder_mead::run_nelder_mead(ind, dataset, max_iterations);
    }
}