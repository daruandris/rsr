Write-Host "Running benchmarks..."

cargo test --release -- --ignored --nocapture --test-threads=1