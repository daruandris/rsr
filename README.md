# RSR

**RSR** is a high-performance, SIMD-accelerated genetic programming engine written in Rust, specialized for symbolic regression. It is designed to automatically discover mathematical equations and physical invariants directly from dataset observations.

Unlike standard symbolic regression tools, RSR natively supports **Tensor and Linear Algebra** operations(2 and 3 dimensional vectors and matrices), combined with **Forward-Mode Automatic Differentiation** using custom SIMD-optimized dual numbers.

## Key features

*   **Fast execution:** The core expression tree evaluator is built with zero-allocation, fixed-size stack machines.
*   **SIMD acceleration:** Leverages the `wide` crate to evaluate genetic individuals across multiple data points concurrently, maximizing CPU throughput.
*   **Matrix and vector operations:** Capable of generating and evaluating matrix operations (determinant, trace, matrix-vector multiplication, cross products, norm).
*   **Island-based evolution:** Implements a distributed island topology with continuous migration and tournament selection.

## Example: The Lorentz Force
RSR can  discover the laws of physics from numerical data. See `src/main.rs` for an example where the engine discovers the Lorentz Force magnitude formula:

$F = || \mathbf{E} + \mathbf{v} \times \mathbf{B} ||$

Just from 400 data points of 3D vectors (electric field, velocity, magnetic field), the engine pieces together the vector cross product and magnitude in milliseconds. 

You can try it with `cargo run --release`.

## Benchmarks
*(Note: The following times represent the average execution time to find the exact target equation with `MSE < 1e-7` during our release benchmarks).*

| Problem | Target Equation | Features | Execution Time |
| :--- | :--- | :--- | :--- |
| **Square** | `y = 2.5x² - 1.2` | `f32` | **~751 ms** |
| **Electromagnetism** | `norm(E + v × B)` | 3x `Vec3` | **~2.3 s** |
| **3D Transform Error** | `norm(M * v - u)` | `Mat3`, 2x `Vec3` | **~9.8 s** |
| **Inverse & Trace** | `Tr(A⁻¹ * B) + det(A)` | 2x `Mat2` | **~58.2 s** |