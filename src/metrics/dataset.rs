use wide::f64x4;


pub struct SimdDataset {
    pub x_batches: Vec<Vec<f64x4>>,
    pub y_batches: Vec<f64x4>,
    pub remainder_x: Vec<Vec<f64>>,
    pub remainder_y: Vec<f64>,
    pub num_features: usize
}

impl SimdDataset {
    pub fn new(data_x: &[Vec<f64>], data_y: &[f64], num_features: usize) -> Self {
        let num_samples = data_x.len();
        let chunk_size = 4;
        let num_chunks = num_samples / chunk_size;

        let mut x_batches = Vec::with_capacity(num_chunks);
        let mut y_batches = Vec::with_capacity(num_chunks);

        for i in 0..num_chunks {
            let start = i* chunk_size;
            let mut batch_features = Vec::with_capacity(num_features);

            for j in 0..num_features {
                let vec4 = f64x4::new([
                    data_x[start][j],
                    data_x[start + 1][j],
                    data_x[start + 2][j],
                    data_x[start + 3][j],
                ]);
                batch_features.push(vec4);
            }
            x_batches.push(batch_features);

            y_batches.push(f64x4::new([
                data_y[start],
                data_y[start + 1],
                data_y[start + 2],
                data_y[start + 3],
            ]));
        }

        let remainder_start = num_chunks * chunk_size;
        let remainder_x = data_x[remainder_start..].to_vec();
        let remainder_y = data_y[remainder_start..].to_vec();

        Self { x_batches, y_batches, remainder_x, remainder_y, num_features }
    }

    pub fn total_samples(&self) -> usize {
        (self.y_batches.len() * 4) + self.remainder_y.len()
    }
}

