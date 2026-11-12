# letify-link

letify-link is the connection layer of [letify](https://github.com/DarkPyonix/letify). It gives two nodes an encrypted, multiplexed session that survives the loss of its transport.

- **Identity.** Each node holds an ed25519 key, and its public key is its node id.
- **Session.** A Noise_IK_25519_ChaChaPoly_BLAKE2s handshake, then encrypted frames with sequence numbers, acknowledgements, flow controlled streams and resumption on a new path after every path died.
- **Paths.** One session runs over several TCP connections and UDP paths at once. NAT traversal (STUN, simultaneous open and UDP punching) builds the paths.
- **Sidecar.** The binary `letify-link` serves the session to the letify Python package over a framed protocol on its standard streams.

The crate has a Cargo feature `relay`, which adds a node that forwards session ciphertext between two authenticated nodes.

```sh
cargo add letify-link
```

The crate [`letify`](https://crates.io/crates/letify) re-exports this crate as `letify::link`. This crate is released together with the Python package `letify`, and each crate version equals the Python package version.

Source, specification and issues are at <https://github.com/DarkPyonix/letify>. The design is in `docs/SPEC.md`, section letify-link.

## License

MIT.
