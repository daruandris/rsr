use crate::ast::node::Node;
use rand::RngExt;

pub trait Domain: Clone + Copy + Send + Sync + PartialEq + 'static {
    type Operator: Clone + Copy + Send + Sync + std::fmt::Debug + PartialEq;
    type Instruction: Clone + Copy + Send + Sync + std::fmt::Debug;
    type SimdValue: Copy + Send + Sync;
    type ScalarValue: Copy + Send + Sync + std::fmt::Display + PartialEq;
    type TypeId: Clone + Copy + Send + Sync + std::fmt::Debug + PartialEq;

    // --- OPERÁTOR TULAJDONSÁGOK ---
    fn operator_arity(op: &Self::Operator) -> usize;
    fn operator_weight(op: &Self::Operator) -> usize;
    fn format_operator(op: &Self::Operator, args: &[String]) -> String;
    fn random_operator(
        target_type: Self::TypeId,
        allowed_ops: &[Self::Operator],
        rng: &mut impl RngExt
    ) -> Option<Self::Operator>;

    // --- BYTECODE ÉS KIÉRTÉKELÉS ---
    fn compile_operator(op: &Self::Operator) -> Self::Instruction;
    fn load_var_instruction(idx: u8, target_type: Self::TypeId) -> Self::Instruction;
    fn load_const_instruction(idx: u16, target_type: Self::TypeId) -> Self::Instruction;
    
    // Itt delegáljuk a SIMD kiértékelést a specifikus implementációhoz
    fn eval_simd(code: &[Self::Instruction], constants: &[Self::ScalarValue], features: &[Self::SimdValue]) -> Self::SimdValue;
    
    // --- HEURISZTIKÁK ---
    fn simplify(nodes: &[Node<Self>]) -> Vec<Node<Self>>;

    fn random_constant(target_type: Self::TypeId, rng: &mut impl RngExt) -> Option<Self::ScalarValue>;
    fn perturb_constant(val: &mut Self::ScalarValue, rng: &mut impl RngExt);
    
    // A Nelder-Mead (ami belsőleg f32-vel matekozik) konverzióihoz:
    fn scalar_to_f32(val: &Self::ScalarValue) -> f32;
    fn scalar_from_f32(val: f32) -> Self::ScalarValue;

    // --- HIBA SZÁMÍTÁS (A régi mse.rs helyett!) ---
    fn compute_mse(
        code: &[Self::Instruction], 
        constants: &[Self::ScalarValue], 
        dataset: &crate::metrics::dataset::SimdDataset
    ) -> f32;

    // -- TYPE függvények --
    fn return_type(op: &Self::Operator) -> Self::TypeId;
    fn expected_types(op: &Self::Operator) -> Vec<Self::TypeId>;
    fn variable_type() -> Self::TypeId;    
    fn constant_type() -> Self::TypeId;
}

pub mod universal;