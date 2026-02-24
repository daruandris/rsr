use crate::domain::Domain;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Op {
    // Ezt majd a BasicDomain-nel együtt kivezetjük, 
    // mert az operátorokat a D::Operator fogja kezelni,
    // de amíg él a régi kód, meghagyhatod.
    Add, Sub, Mul, Div, Sin, Cos, Exp, Sqr
}
// ... (A régi Op impl maradhat, amíg nem töröljük a BasicDomain-t)

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Node<D: Domain> {
    Operator(D::Operator),
    Variable(u8, D::TypeId),
    Constant(D::ScalarValue, D::TypeId),
}

impl<D: Domain> Node<D> {
    pub fn arity(&self) -> usize {
        match self {
            Node::Constant(_, _) | Node::Variable(_, _) => 0, // Frissítve [cite: 79, 80]
            Node::Operator(op) => D::operator_arity(op), // Frissítve [cite: 80]
        }
    }

    pub fn weight(&self) -> usize {
        match self {
            Node::Variable(_, _) => 1, 
            Node::Constant(_, type_id) => D::type_weight(type_id), // A súly = paraméterek száma!
            Node::Operator(op) => D::operator_weight(op), 
        }
    }

    pub fn get_type(&self) -> D::TypeId {
        match self {
            Node::Operator(op) => D::return_type(op),
            Node::Variable(_, type_id) => *type_id, // Frissítve
            Node::Constant(_, type_id) => *type_id, // Frissítve
        }
    }
}