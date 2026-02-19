use crate::ast::node::Node;
use crate::domain::Domain;

#[derive(Clone, Debug)]
pub struct CompiledExpr<D: Domain> {
    pub code: Vec<D::Instruction>,
    pub constants: Vec<D::ScalarValue>,
}

impl<D: Domain> CompiledExpr<D> {
    pub fn from_nodes(nodes: &[Node<D>]) -> Self {
        let mut code = Vec::with_capacity(nodes.len());
        let mut constants = Vec::new();

        for node in nodes {
            match node {
                Node::Variable(idx) => {
                    code.push(D::load_var_instruction(*idx));
                },
                Node::Constant(val) => {
                    let c_idx = constants.len();
                    constants.push(*val);
                    code.push(D::load_const_instruction(c_idx as u16));
                },
                Node::Operator(op) => {
                    code.push(D::compile_operator(op));
                }
            }
        }

        CompiledExpr { code, constants }
    }

    #[inline(always)]
    pub fn eval_simd(&self, features: &[D::SimdValue]) -> D::SimdValue {
        D::eval_simd(&self.code, &self.constants, features)
    }
}