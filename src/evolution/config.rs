#[derive(Clone, Copy, Debug)]
pub struct EvolutionConfig {
    pub num_islands: usize,
    pub island_size: usize,
    pub max_generations: usize,
    pub crossover_rate: f64,
    pub tournament_size: usize,
    pub migration_interval: usize,
    pub parsimony_penalty: f64,

    pub opt_prob: f64,
    pub opt_iterations: usize,

    pub stagnation_threshold: usize,
    pub target_mse: f64,
    pub min_improvement: f64,

    pub random_injection_rate: f64,
    pub min_random_injection: usize,
    pub verbose: bool
}