"""Zip the project (code, assets, notebook, tests) into dist/flappy_bird.zip."""
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ["flappy_rl", "notebooks", "tests", "scripts", "pyproject.toml", "README.md"]
SKIP = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints", ".ruff_cache"}


def main():
    out = ROOT / "dist" / "flappy_bird.zip"
    out.parent.mkdir(exist_ok=True)
    files = sorted(p for name in INCLUDE for p in ([ROOT / name] if (ROOT / name).is_file()
                                                   else (ROOT / name).rglob("*"))
                   if p.is_file() and not SKIP & set(p.parts))
    with ZipFile(out, "w", ZIP_DEFLATED) as bundle:
        for path in files:
            bundle.write(path, path.relative_to(ROOT).as_posix())
    print(f"{out}: {len(files)} files, {out.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
