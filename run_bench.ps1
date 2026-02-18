Write-Host "Benchmarkok futtatása (ez eltarthat pár percig)..."

cargo test --release -- --ignored --nocapture --test-threads=1

git add benchmark_output.json
git commit -m "Update Benchmark History [skip ci]"
git push