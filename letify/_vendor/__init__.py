"""Third party code carried inside letify, so letify installs nothing else.

cloudpickle 3.1.2, unmodified apart from leaving out cloudpickle_fast.py, which only
re-exports names for backward compatibility. Its BSD 3-clause license is in
cloudpickle/LICENSE. To update it, copy __init__.py and cloudpickle.py from the new release
and replace LICENSE.
"""
