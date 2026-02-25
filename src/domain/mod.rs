use crate::ast::node::Node;
use rand::RngExt;

pub trait Domain: Clone + Copy + Send + Sync + PartialEq + 'static {
    type Operator: Clone + Copy + Send + Sync + std::fmt::Debug + PartialEq;
    type Instruction: Clone + Copy + Send + Sync + std::fmt::Debug;
    type SimdValue: Copy + Send + Sync;
    type ScalarValue: Copy + Send + Sync + std::fmt::Display + PartialEq;
    type TypeId: Clone + Copy + Send + Sync + std::fmt::Debug + PartialEq;

    fn operator_arity(op: &Self::Operator) -> usize;
    fn operator_weight(op: &Self::Operator) -> usize;
    fn format_operator(op: &Self::Operator, args: &[String]) -> String;
    fn random_operator(
        target_type: Self::TypeId,
        allowed_ops: &[Self::Operator],
        parent_op: Option<Self::Operator>,
        rng: &mut impl RngExt
        
    ) -> Option<Self::Operator>;

    fn compile_operator(op: &Self::Operator) -> Self::Instruction;
    fn load_var_instruction(idx: u8, target_type: Self::TypeId) -> Self::Instruction;
    fn load_const_instruction(idx: u16, target_type: Self::TypeId) -> Self::Instruction;
    
    fn eval_simd(code: &[Self::Instruction], constants: &[Self::ScalarValue], features: &[Self::SimdValue]) -> Self::SimdValue;
    
    fn simplify(nodes: &[Node<Self>]) -> Vec<Node<Self>>;

    fn random_constant(target_type: Self::TypeId, rng: &mut impl RngExt) -> Option<Self::ScalarValue>;
    fn perturb_constant(val: &mut Self::ScalarValue, rng: &mut impl RngExt);
    
    fn scalar_to_f32(val: &Self::ScalarValue) -> Option<f32>;
    fn scalar_from_f32(val: f32) -> Self::ScalarValue;

    fn compute_mse(
        code: &[Self::Instruction], 
        constants: &[Self::ScalarValue], 
        dataset: &crate::metrics::dataset::SimdDataset
    ) -> f32;

    fn return_type(op: &Self::Operator) -> Self::TypeId;
    fn expected_types(op: &Self::Operator) -> &'static [Self::TypeId];
    fn constant_type() -> Self::TypeId;

    fn type_weight(type_id: &Self::TypeId) -> usize;
    fn compute_mse_with_gradient(
        code: &[Self::Instruction], 
        constants: &[Self::ScalarValue], 
        dataset: &crate::metrics::dataset::SimdDataset
    ) -> (f32, [f32; 32]);
}

pub mod universal;
pub mod dual;