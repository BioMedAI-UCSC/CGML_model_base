#!/usr/bin/env python3
"""CLI shim: delegates to the `preprocess` package.

Full single-file copy (for reference / diff): `preprocess_legacy.py`.
Run from `base_model/` with `module` on PYTHONPATH, same as before:
  python preprocess.py ...
or:
  python -m preprocess ...
"""
from preprocess.__main__ import main

if __name__ == "__main__":
    main()
