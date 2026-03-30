fn main() {
    println!("cargo:rerun-if-changed=src/ffi.rs");
    println!("cargo:rerun-if-changed=bonsai.h");
    println!("cargo:rerun-if-changed=Cargo.toml");
}
