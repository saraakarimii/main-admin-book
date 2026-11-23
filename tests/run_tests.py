"""Dependency-free runner (pytest also works: `pytest -q`)."""
import importlib, sys, time, traceback, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
passed = failed = skipped = 0
for mod_name in ("test_palette", "test_integrity", "test_framing_cache", "test_export", "test_notebook_parity", "test_optimize", "test_layout"):
    mod = importlib.import_module(mod_name)
    for name in sorted(n for n in dir(mod) if n.startswith("test_")):
        t0 = time.time()
        try:
            getattr(mod, name)()
            passed += 1; print(f"PASS  {mod_name}::{name}  ({time.time()-t0:.1f}s)")
        except unittest.SkipTest as e:
            skipped += 1; print(f"SKIP  {mod_name}::{name}  {e}")
        except Exception:
            failed += 1; print(f"FAIL  {mod_name}::{name}"); traceback.print_exc()
print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
sys.exit(1 if failed else 0)
