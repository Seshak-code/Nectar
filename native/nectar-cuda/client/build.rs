//! On Linux the library is installed as libcuda.so.1, which programs linked with -lcuda name
//! as their dependency, so it carries that name (its SONAME) too.

fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        println!("cargo:rustc-cdylib-link-arg=-Wl,-soname,libcuda.so.1");
    }
}
