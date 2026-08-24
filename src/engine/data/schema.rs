//! Defines the structure and parsing rules for input data.

use crate::engine::eval::types::ValueType;

/// Defines the layout and preprocessing rules for a [`Dataset`](crate::engine::data::dataset::Dataset).
///
/// A `Schema` tells the engine what data types to expect for each feature
/// (e.g., scalars, vectors, matrices), which column contains the target variable,
/// and whether the data should be automatically standardized (mean 0, variance 1).
///
/// # Examples
///
/// ```
/// use rsr::prelude::*;
///
/// // Create a schema for 3 scalar features, auto-normalized.
/// let schema = Schema::new(vec![ValueType::Float, ValueType::Float, ValueType::Float])
///     .with_normalization(true)
///     .with_target_index(3);
/// ```
#[derive(Clone, Debug)]
pub struct Schema {
    /// The types of the input features in order.
    pub feature_types: Vec<ValueType>,
    /// The zero-based index of the target variable column. If `None`, defaults to the last column.
    pub target_col_index: Option<usize>,
    /// Whether to automatically standardize features and targets during dataset creation.
    pub normalize: bool,
}

impl Schema {
    /// Creates a new schema with the specified feature types.
    ///
    /// By default, `normalize` is set to `false`, and `target_col_index` is `None`
    /// (meaning the last column is treated as the target).
    pub fn new(feature_types: Vec<ValueType>) -> Self {
        Self {
            feature_types,
            target_col_index: None,
            normalize: false,
        }
    }

    /// Sets the specific column index that contains the target (Y) variable.
    pub fn with_target_index(mut self, index: usize) -> Self {
        self.target_col_index = Some(index);
        self
    }

    /// Enables or disables automatic Z-score normalization.
    pub fn with_normalization(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }
}
