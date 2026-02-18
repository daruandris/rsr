Write-Host "Running benchmarks..."

cargo test --release -- --ignored --nocapture --test-threads=1

git add benchmark_output.json
git commit -m "Update Benchmark History"
git push