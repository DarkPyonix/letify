//! The crate `letify` re-exports letify-link whole as `letify::link`, as docs/SPEC.md,
//! Versioning and releases, requires.

#[test]
fn letify_link_is_reachable_as_the_link_module() {
    let identity = letify::link::Identity::generate();
    let again: letify::link::NodeId = identity.node_id();
    assert_eq!(again, identity.node_id());
}

#[test]
fn the_modules_the_sidecar_uses_are_re_exported() {
    let _ = letify::link::udp::UdpConfig::default();
    let _ = letify::link::Config::default();
    let _ = letify::link::sidecar::load_or_create_key;
}

#[test]
fn the_version_constant_is_kept() {
    assert_eq!(letify::VERSION, env!("CARGO_PKG_VERSION"));
}
