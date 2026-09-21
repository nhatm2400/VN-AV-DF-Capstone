"""Apply preview adjustments inside the existing MuseTalk inference process."""
from pathlib import Path
import os
import runpy
import sys

ROOT = Path(__file__).resolve().parents[3]


if __name__ == '__main__':
    os.chdir(ROOT / 'external/MuseTalk')
    sys.path[:0] = [str(ROOT), str(Path.cwd())]
    # YAPF caches otherwise target the user profile and can stall in restricted runs.
    import platformdirs
    original_cache = platformdirs.user_cache_dir
    platformdirs.user_cache_dir = lambda appname=None, *a, **k: (
        str(ROOT / 'cache/musetalk_yapf') if appname == 'YAPF'
        else original_cache(appname, *a, **k))
    from src.generators.preparation.preview_quality import install_musetalk
    install_musetalk()
    runpy.run_module('scripts.inference', run_name='__main__')
