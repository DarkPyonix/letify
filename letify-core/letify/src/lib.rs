//! The Rust face of letify.
//!
//! This crate owns the name `letify` on crates.io and the version that is shared with the
//! Python package. It does not hold the letify runtime, which is the Python package.
#![doc = include_str!("../README.md")]

/// The version of this crate, which is also the version of the Python package `letify`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::VERSION;

    #[test]
    fn the_version_is_a_semantic_version() {
        let core = VERSION.split(['-', '+']).next().unwrap();
        let parts: Vec<&str> = core.split('.').collect();
        assert_eq!(parts.len(), 3, "{VERSION} is not major.minor.patch");
        assert!(parts.iter().all(|p| p.parse::<u64>().is_ok()), "{VERSION}");
    }
}
