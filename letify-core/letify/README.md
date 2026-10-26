# letify

letify is a declarative infrastructure tool for running Python and PyTorch work on remote GPUs. You describe the environment and the function in Python, and letify runs it on a GPU runtime such as Colab, Kaggle, Modal or your own server.

The tool itself is the Python package `letify` on PyPI. Install it with:

```sh
uv add letify
```

Source, documentation and issues are at <https://github.com/DarkPyonix/letify>.

## This crate

This crate is the Rust face of the project. It will re-export the letify components that are meant for use outside letify. The first is the connection library `letify-link`, which provides NAT traversal and multipath links.

Today the crate exports only `VERSION`. Its version follows the Python package: crate `letify` 2.8.6 is released together with PyPI `letify` 2.8.6.

## License

MIT.
