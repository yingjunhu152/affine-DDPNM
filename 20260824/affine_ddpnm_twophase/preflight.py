#!/usr/bin/env python3
"""Check the runtime needed by the full benchmark."""

from __future__ import annotations

import importlib
import sys


REQUIRED = ("numpy", "scipy", "dolfinx", "ufl", "basix", "gmsh", "mpi4py", "pyvista")


def main() -> int:
    failures: list[str] = []
    print(f"Python {sys.version.split()[0]}")
    for name in REQUIRED:
        try:
            module = importlib.import_module(name)
        except Exception as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            print(f"MISSING {name}: {exc}")
        else:
            print(f"OK      {name} {getattr(module, '__version__', '')}")
    if failures:
        print("\nPreflight failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("\nPreflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
