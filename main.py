"""Convenience shim so `uv run main.py` invokes the package CLI.

The real entry point is `wwd_i.cli:main` (console script `wwd-i`)."""

from wwd_i.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
