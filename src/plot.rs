use crate::engine::data::dataset::Dataset;
use crate::engine::eval::evaluator::eval_simd_mat3;
use crate::api::FitResult;
use plotters::prelude::*;

#[derive(Clone, Copy, Debug)]
pub enum SolidPlotMode {
    /// F11 vs P11 (0-ás indexek)
    UniaxialTension,
    /// Két diagram egymás mellett: F11 vs P11 és F22 vs P22
    BiaxialTension,
    /// F12 vs P12 (1-es indexek)
    Shear,
    /// Egyedi indexek: (X_komponens_F_ből, Y_komponens_P_ből). (0..8)
    Custom(usize, usize),
}

pub struct PlotConfig {
    pub output_path: String,
    pub mode: SolidPlotMode,
    /// A diagramra rajzolt pontok maximális száma (a motor automatikusan ritkít eddig a méretig)
    pub max_points: usize, 
}

impl FitResult {
    /// Mátrixos (Mat3) Solid modell eredményeinek kirajzolása.
    pub fn plot_solid(&self, dataset: &Dataset, config: PlotConfig) -> Result<(), Box<dyn std::error::Error>> {
        let program = self.program.as_ref().ok_or("Nincs lefordított program a FitResult-ban!")?;
        let targets = dataset.target_mat3_batches.as_ref().ok_or("Az adathalmaz nem tartalmaz Mat3 célváltozókat!")?;

        // 1. Adathalmaz ritkítása a gyors és tiszta plotolásért
        let plot_data = dataset.subset(config.max_points);
        let plot_targets = plot_data.target_mat3_batches.as_ref().unwrap();

        // 2. Indexek leképzése a választott mód alapján
        let (idx_x1, idx_y1, idx_x2, idx_y2) = match config.mode {
            SolidPlotMode::UniaxialTension => (0, 0, None, None),
            SolidPlotMode::Shear => (1, 1, None, None),
            SolidPlotMode::BiaxialTension => (0, 0, Some(4), Some(4)),
            SolidPlotMode::Custom(x, y) => (x, y, None, None),
        };

        let mut x1_act = Vec::new(); let mut y1_act = Vec::new(); let mut y1_prd = Vec::new();
        let mut x2_act = Vec::new(); let mut y2_act = Vec::new(); let mut y2_prd = Vec::new();

        // 3. Predikciók legenerálása és kinyerése a SIMD sávokból
        for i in 0..plot_data.num_batches {
            let start_f = i * (plot_data.num_features as usize);
            let features = &plot_data.feature_flat[start_f..start_f + (plot_data.num_features as usize)];
            
            // SIMD kiértékelés (8 sor egyszerre)
            let pred_batch = eval_simd_mat3(program, features);
            let target_batch = plot_targets[i];

            for lane in 0..8 {
                let sample_idx = i * 8 + lane;
                if sample_idx >= plot_data.num_samples { break; }

                let extract = |simd_val: wide::f32x8, l: usize| -> f32 {
                    unsafe { (*(&simd_val as *const _ as *const [f32; 8]))[l] }
                };

                x1_act.push(extract(features[idx_x1], lane));
                y1_act.push(extract(target_batch[idx_y1], lane));
                y1_prd.push(extract(pred_batch[idx_y1], lane));

                if let (Some(x2), Some(y2)) = (idx_x2, idx_y2) {
                    x2_act.push(extract(features[x2], lane));
                    y2_act.push(extract(target_batch[y2], lane));
                    y2_prd.push(extract(pred_batch[y2], lane));
                }
            }
        }

        // 4. Diagram renderelése (Plotters)
        if idx_x2.is_some() {
            // Biaxial mód: 2 diagram egymás mellett (1200x600 px)
            let root = BitMapBackend::new(&config.output_path, (1200, 600)).into_drawing_area();
            root.fill(&WHITE)?;
            let (left, right) = root.split_horizontally(600);
            
            draw_chart(&left, &x1_act, &y1_act, &y1_prd, "11 Komponens")?;
            draw_chart(&right, &x2_act, &y2_act, &y2_prd, "22 Komponens")?;
            root.present()?;
        } else {
            // Single mód: 1 diagram (800x600 px)
            let root = BitMapBackend::new(&config.output_path, (800, 600)).into_drawing_area();
            root.fill(&WHITE)?;
            draw_chart(&root, &x1_act, &y1_act, &y1_prd, "Feszültség - Alakváltozás")?;
            root.present()?;
        }

        Ok(())
    }
}

/// Belső segédfüggvény egyetlen 2D diagram felépítéséhez
fn draw_chart(
    area: &DrawingArea<BitMapBackend, plotters::coord::Shift>,
    x_act: &[f32],
    y_act: &[f32],
    y_pred: &[f32],
    title: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let min_x = x_act.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_x = x_act.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    
    let min_y = y_act.iter().chain(y_pred.iter()).cloned().fold(f32::INFINITY, f32::min);
    let max_y = y_act.iter().chain(y_pred.iter()).cloned().fold(f32::NEG_INFINITY, f32::max);

    let x_margin = (max_x - min_x).max(1e-5) * 0.05;
    let y_margin = (max_y - min_y).max(1e-5) * 0.05;

    let mut chart = ChartBuilder::on(area)
        .caption(title, ("sans-serif", 30).into_font())
        .margin(15)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(
            (min_x - x_margin)..(max_x + x_margin),
            (min_y - y_margin)..(max_y + y_margin),
        )?;

    chart.configure_mesh().draw()?;

    // Zöld pontok: Eredeti kísérleti adatok (szórt grafikon)
    chart.draw_series(
        x_act.iter().zip(y_act.iter()).map(|(&x, &y)| Circle::new((x, y), 3, GREEN.filled()))
    )?
    .label("Kísérleti Adat")
    .legend(|(x, y)| Circle::new((x, y), 3, GREEN.filled()));

    // Előkészítjük a vonalat (X szerint sorbarendezzük a pontokat, hogy folytonos legyen a vonal)
    let mut line_data: Vec<(f32, f32)> = x_act
        .iter()
        .cloned()
        .zip(y_pred.iter().cloned())
        .filter(|(x, y)| x.is_finite() && y.is_finite()) 
        .collect();
    line_data.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // EREDETI PIROS VONAL HELYETT: Piros pontok (Scatter) a predikcióknak
    chart.draw_series(
        x_act.iter().zip(y_pred.iter()).filter(|(x, y)| x.is_finite() && y.is_finite()).map(|(&x, &y)| Circle::new((x, y), 2, RED.filled()))
    )?
    .label("Talált Modell Predikciói")
    .legend(|(x, y)| Circle::new((x, y), 3, RED.filled()));

    chart.configure_series_labels().background_style(&WHITE.mix(0.8)).border_style(&BLACK).draw()?;

    Ok(())
}