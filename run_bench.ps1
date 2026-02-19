Write-Host "Running benchmarks..."

cargo test --release -- --nocapture --test-threads=1