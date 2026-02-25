use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Scalar {
    Float(f32),
    Int(i32),
    Bool(bool),
    Vec2([f32; 2]),
    Vec3([f32; 3]),
    Mat2([f32; 4]),
    Mat3([f32; 9]),
}

impl Scalar {
    #[inline]
    pub fn is_zero(&self) -> bool {
        match self {
            Self::Float(f) => f.abs() < 1e-6,
            Self::Vec2(v) => v.iter().all(|&x| x.abs() < 1e-6),
            Self::Vec3(v) => v.iter().all(|&x| x.abs() < 1e-6),
            Self::Mat2(m) => m.iter().all(|&x| x.abs() < 1e-6),
            Self::Mat3(m) => m.iter().all(|&x| x.abs() < 1e-6),
            _ => false,
        }
    }

    #[inline]
    pub fn is_one(&self) -> bool {
        match self {
            Self::Float(f) => (f - 1.0).abs() < 1e-6,
            _ => false,
        }
    }

    #[inline]
    pub fn is_identity(&self) -> bool {
        match self {
            Self::Mat2(m) => {
                (m[0] - 1.0).abs() < 1e-6
                    && m[1].abs() < 1e-6
                    && m[2].abs() < 1e-6
                    && (m[3] - 1.0).abs() < 1e-6
            }
            Self::Mat3(m) => {
                (m[0] - 1.0).abs() < 1e-6
                    && m[1].abs() < 1e-6
                    && m[2].abs() < 1e-6
                    && m[3].abs() < 1e-6
                    && (m[4] - 1.0).abs() < 1e-6
                    && m[5].abs() < 1e-6
                    && m[6].abs() < 1e-6
                    && m[7].abs() < 1e-6
                    && (m[8] - 1.0).abs() < 1e-6
            }
            _ => false,
        }
    }

    #[inline]
    pub fn apply_threshold(&mut self, threshold: f32) {
        match self {
            Self::Float(f) => {
                if f.abs() < threshold {
                    *f = 0.0;
                }
            }
            Self::Vec2(v) => {
                for x in v.iter_mut() {
                    if x.abs() < threshold {
                        *x = 0.0;
                    }
                }
            }
            Self::Vec3(v) => {
                for x in v.iter_mut() {
                    if x.abs() < threshold {
                        *x = 0.0;
                    }
                }
            }
            Self::Mat2(m) => {
                for x in m.iter_mut() {
                    if x.abs() < threshold {
                        *x = 0.0;
                    }
                }
            }
            Self::Mat3(m) => {
                for x in m.iter_mut() {
                    if x.abs() < threshold {
                        *x = 0.0;
                    }
                }
            }
            _ => {}
        }
    }
}

impl fmt::Display for Scalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Scalar::Float(val) => write!(f, "{:.4}", val),
            Scalar::Int(val) => write!(f, "{}", val),
            Scalar::Bool(val) => write!(f, "{}", val),
            Scalar::Vec2(arr) => write!(f, "[{:.2}, {:.2}]", arr[0], arr[1]),
            Scalar::Vec3(arr) => write!(f, "[{:.2}, {:.2}, {:.2}]", arr[0], arr[1], arr[2]),
            Scalar::Mat2(m) => write!(
                f,
                "[{:.2}, {:.2}; {:.2}, {:.2}]",
                m[0], m[2], m[1], m[3]
            ),
            Scalar::Mat3(m) => write!(
                f,
                "[{:.2}, {:.2}, {:.2}; {:.2}, {:.2}, {:.2}; {:.2}, {:.2}, {:.2}]",
                m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]
            ),
        }
    }
}