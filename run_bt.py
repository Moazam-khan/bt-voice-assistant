"""PyInstaller entry point for BT.

PyInstaller analyzes a plain script more reliably than `python -m
bt_core.main`, so this thin wrapper is the actual build target — it has
no logic of its own beyond calling into bt_core.main.
"""

from bt_core.main import main

if __name__ == "__main__":
    main()
