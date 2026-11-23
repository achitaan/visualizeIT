"""Run prototype modules with explicit package paths; never execute site/.pth hooks."""
import os
from pathlib import Path
import runpy
import sys
import sysconfig


def _resolve_site_packages() -> list[str]:
    win_packages = Path(sys.executable).resolve().parent / 'Lib' / 'site-packages'
    if win_packages.is_dir():
        return [str(win_packages)]

    exe_abs = Path(os.path.abspath(sys.executable))
    venv_root = exe_abs.parent.parent
    if (venv_root / 'pyvenv.cfg').is_file():
        candidates: list[Path] = [venv_root / 'Lib' / 'site-packages']
        if 'venv' in sysconfig.get_scheme_names():
            venv_vars = {'base': str(venv_root), 'platbase': str(venv_root)}
            candidates.append(Path(sysconfig.get_path('purelib', 'venv', vars=venv_vars)))
            candidates.append(Path(sysconfig.get_path('platlib', 'venv', vars=venv_vars)))
        py_ver = f'python{sys.version_info.major}.{sys.version_info.minor}'
        candidates.append(venv_root / 'lib' / py_ver / 'site-packages')
        candidates.append(venv_root / 'lib64' / py_ver / 'site-packages')
        found: list[str] = []
        for path in candidates:
            text = str(path)
            if text not in found and path.is_dir():
                found.append(text)
        if found:
            return found

    fallback: list[str] = []
    paths = sysconfig.get_paths()
    for key in ('purelib', 'platlib'):
        val = paths.get(key)
        if val and val not in fallback and Path(val).is_dir():
            fallback.append(val)
    deb_dist = Path(sys.prefix) / 'lib' / 'python3' / 'dist-packages'
    if deb_dist.is_dir() and str(deb_dist) not in fallback:
        fallback.append(str(deb_dist))
    return fallback or [str(win_packages)]


def main():
    if not sys.flags.no_site:
        raise SystemExit('Use python -B -S tools/prototype_bootstrap.py MODULE [arguments].')
    root = Path(__file__).resolve().parents[1]
    site_packages = _resolve_site_packages()
    sys.path[:0] = [str(root), str(root / '.cache' / 'vision'),
                    str(root / '.cache' / 'plotting'), *site_packages]
    if len(sys.argv) < 2:
        raise SystemExit('A module is required.')
    module = sys.argv.pop(1)
    runpy.run_module(module, run_name='__main__')


if __name__ == '__main__':
    main()
