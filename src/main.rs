use rand::RngExt;
use rsr::ffi::symengine::simplify_symengine;
use rsr::prelude::*;
use rsr::Instruction;
use rsr::eval::basic_domain::BasicOpCode;
use rsr::eval::linalg_domain::LinalgOpCode;
use std::time::Instant;

fn main() {
    //generating the data
    let mut rng = rand::rng();
    let num_samples = 400;

    let mut dx = Vec::with_capacity(num_samples);
    let mut dy = Vec::with_capacity(num_samples);
    for _ in 0..num_samples {
        let e0 = rng.random_range(-5.0..5.0);
        let e1 = rng.random_range(-5.0..5.0);
        let e2 = rng.random_range(-5.0..5.0);
        let v0 = rng.random_range(-5.0..5.0);
        let v1 = rng.random_range(-5.0..5.0);
        let v2 = rng.random_range(-5.0..5.0);
        let b0 = rng.random_range(-5.0..5.0);
        let b1 = rng.random_range(-5.0..5.0);
        let b2 = rng.random_range(-5.0..5.0);

        dx.push(vec![e0, e1, e2, v0, v1, v2, b0, b1, b2]);

        let cx = v1 * b2 - v2 * b1;
        let cy = v2 * b0 - v0 * b2;
        let cz = v0 * b1 - v1 * b0;
        let fx = e0 + cx;
        let fy = e1 + cy;
        let fz: f32 = e2 + cz;

        dy.push((fx * fx + fy * fy + fz * fz).sqrt());
    }

    //set config
    let feature_types = vec![ValueType::Vec3, ValueType::Vec3, ValueType::Vec3];
    let config = Config {
        num_islands: 24,
        island_size: 25,
        max_generations: 3000,
        crossover_rate: 0.10,
        tournament_size: 2,
        migration_interval: 25,
        parsimony_penalty: 0.0,
        target_mse: 1e-7,
        min_improvement: 1e-6,
        opt_prob: 0.2,
        opt_iterations: 100,
        final_opt_iterations: 5000,
        stagnation_threshold: 1000,
        random_injection_rate: 0.10,
        min_random_injection: 2,
        max_tree_size: 32,
        mutation_max_depth: 7,
        mutation_cycles: 5,
        verbose: true,
        allowed_modules: vec![OpModule::Basic],
        custom_ops: vec![
            Instruction::Linalg(LinalgOpCode::AddV3),
            Instruction::Linalg(LinalgOpCode::SubV3),
            Instruction::Linalg(LinalgOpCode::CrossV3),
            Instruction::Linalg(LinalgOpCode::DotV3),
            Instruction::Linalg(LinalgOpCode::NormV3),
            Instruction::Linalg(LinalgOpCode::ScaleV3),
        ],
        excluded_ops: vec![
            Instruction::Basic(BasicOpCode::SinF),
            Instruction::Basic(BasicOpCode::CosF),
            Instruction::Basic(BasicOpCode::ExpF),
            Instruction::Basic(BasicOpCode::LnF),
            Instruction::Basic(BasicOpCode::SqrtF),
            Instruction::Basic(BasicOpCode::SqrF),
        ],
        ..Default::default()
    };

    //start the algorithm
    println!("Starting the algorithm...");
    let dataset = Dataset::new(&dx, &dy, feature_types, false);

    let strategy = StaticStrategy::new(config);
    let allowed_ops = strategy.get_allowed_operators();

    let mut engine = Engine::new(strategy, dataset.get_variable_registry(), allowed_ops);

    let start_time = Instant::now();
    engine.run(&dataset);
    let duration = start_time.elapsed();

    let best_ind = engine.get_global_best();
    let clean_eq = simplify_symengine(&best_ind.to_string());

    println!("\nEquation found in {:.3} seconds!", duration.as_secs_f64());
    println!("Equation: {}", clean_eq);
    println!("MSE:    {:.10}", best_ind.fitness);
}