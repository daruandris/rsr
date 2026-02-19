use std::process::Command;

fn main() {
    cc::Build::new()
        .cpp(true)
        .file("src/cpp/simplify.cpp")
        .include("vendor/include")
        .flag("/EHsc")
        .compile("simplify_native");

    println!("cargo:rustc-link-search=native=vendor/lib");
    
    println!("cargo:rustc-link-lib=static=symengine");
    println!("cargo:rustc-link-lib=static=gmp");


    let git_hash = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string());

    println!("cargo:rustc-env=GIT_HASH={}", git_hash);
}