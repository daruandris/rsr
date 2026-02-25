#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ValueType {
    Float,
    Int,
    Bool,
    Vec2,
    Vec3,
    Mat2,
    Mat3,
}