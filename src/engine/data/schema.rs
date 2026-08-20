use crate::engine::eval::types::ValueType;

#[derive(Clone, Debug)]
pub struct Schema {
    pub feature_types: Vec<ValueType>,
    pub target_col_index: Option<usize>,
    pub normalize: bool,
}

impl Schema {
    pub fn new(feature_types: Vec<ValueType>) -> Self {
        Self {
            feature_types,
            target_col_index: None,
            normalize: true,
        }
    }

    pub fn with_target_index(mut self, index: usize) -> Self {
        self.target_col_index = Some(index);
        self
    }

    pub fn with_normalization(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }
}