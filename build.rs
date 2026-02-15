fn main() {
    cc::Build::new()
        .cpp(true)
        .file("src/simplify.cpp")
        .include("vendor/include")
        .flag("/EHsc")
        .compile("simplify_native");

    println!("cargo:rustc-link-search=native=vendor/lib");
    
    println!("cargo:rustc-link-lib=static=symengine");
    println!("cargo:rustc-link-lib=static=gmp");
}