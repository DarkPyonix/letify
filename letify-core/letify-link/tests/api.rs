//! Pins SPEC.md letify-link, Library API and embedding item 6: a type in the signature of a
//! public method is nameable from the crate root.

use letify_link::{PathKinds, PathManager};

/// An embedder stores the value in a struct, which needs the type's name.
#[allow(dead_code)]
struct Holder {
    kinds: PathKinds,
}

#[allow(dead_code)]
fn kinds_of(pm: &PathManager) -> Holder {
    Holder { kinds: pm.kinds() }
}

#[test]
fn the_return_type_of_path_manager_kinds_is_nameable_from_the_crate_root() {
    // The check is that this file compiles.
}
