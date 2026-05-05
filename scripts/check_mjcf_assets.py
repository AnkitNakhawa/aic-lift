#!/usr/bin/env python3
"""
Validate that the MJCF scene is consistent and loadable.

Usage
-----
  # Full check — verify all file refs exist on disk AND MuJoCo can load the scene:
  python3 scripts/check_mjcf_assets.py

  # Structural check only — no mesh files needed (fast, CI-safe for PRs):
  python3 scripts/check_mjcf_assets.py --xml-only

  # Point at a different scene:
  python3 scripts/check_mjcf_assets.py --scene /tmp/aic_mujoco_world/aic_world.xml
"""
import argparse
import hashlib
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MJCF_DIR = REPO_ROOT / "aic_utils" / "aic_mujoco" / "mjcf"
DEFAULT_SCENE = MJCF_DIR / "scene.xml"


def collect_xml_tree(scene: Path) -> list[Path]:
    """Walk <include> chain and return all XML paths in load order."""
    visited: list[Path] = []
    queue = [scene]
    while queue:
        xml = queue.pop(0)
        if xml in visited:
            continue
        visited.append(xml)
        try:
            content = xml.read_text()
        except FileNotFoundError:
            continue
        for m in re.finditer(r'<include\s+file="([^"]+)"', content):
            queue.append(xml.parent / m.group(1))
    return visited


def check_file_refs(scene: Path) -> list[str]:
    """Return error strings for every missing mesh/texture file referenced in the XML tree."""
    errors: list[str] = []
    mjcf_dir = scene.parent
    for xml in collect_xml_tree(scene):
        if not xml.exists():
            errors.append(f"XML not found:     {xml.relative_to(mjcf_dir)}")
            continue
        content = xml.read_text()
        for m in re.finditer(r'\bfile="([^"]+\.(obj|stl|png))"', content):
            ref = mjcf_dir / m.group(1)
            if not ref.exists():
                errors.append(f"Missing asset:     {m.group(1)}")
    return errors


def check_bundle_version(mjcf_dir: Path) -> tuple[str | None, list[str]]:
    """Return (version_string, errors)."""
    vf = mjcf_dir / "ASSET_BUNDLE_VERSION"
    if not vf.exists():
        return None, [f"ASSET_BUNDLE_VERSION not found in {mjcf_dir} — run the mesh export workflow"]
    v = vf.read_text().strip()
    if not v:
        return None, ["ASSET_BUNDLE_VERSION is empty"]
    return v, []


def check_xml_bundle_consistency(scene: Path, declared_version: str | None) -> list[str]:
    """
    Warn when the XML tree references asset hashes that don't belong to any
    known bundle — a sign the XMLs were regenerated without updating ASSET_BUNDLE_VERSION.
    This is a warning-only structural check (no mesh files needed).
    """
    if declared_version is None:
        return []

    # Collect every unique 64-char hex hash prefix from file="<hash>_..." refs.
    seen_prefixes: set[str] = set()
    for xml in collect_xml_tree(scene):
        if not xml.exists():
            continue
        for m in re.finditer(r'\bfile="([0-9a-f]{64})_', xml.read_text()):
            seen_prefixes.add(m.group(1))

    if not seen_prefixes:
        return []  # No hash-prefixed files — stable names in use, nothing to check.

    # Recompute what the bundle version SHOULD be from the XMLs themselves so we
    # can detect stale XML vs. stale ASSET_BUNDLE_VERSION independently.
    xml_derived = hashlib.sha256("|".join(sorted(seen_prefixes)).encode()).hexdigest()[:16]
    if xml_derived == declared_version:
        return []

    return [
        f"ASSET_BUNDLE_VERSION ({declared_version}) does not match the hash prefixes "
        f"found in the XML files ({xml_derived}). "
        "The XMLs and ASSET_BUNDLE_VERSION were probably not exported together. "
        "Re-run the 'Export MuJoCo Mesh Assets' workflow and merge the resulting PR."
    ]


def check_mujoco_load(scene: Path) -> list[str]:
    """Try to load scene.xml with MuJoCo; return errors."""
    try:
        import mujoco  # noqa: PLC0415
    except ImportError:
        return ["mujoco not installed — pip install mujoco"]
    try:
        m = mujoco.MjModel.from_xml_path(str(scene))
        print(f"  MuJoCo: nbody={m.nbody}  ngeom={m.ngeom}  nsensor={m.nsensor}")
        return []
    except Exception as exc:
        return [f"MuJoCo load failed: {exc}"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=str(DEFAULT_SCENE))
    ap.add_argument("--xml-only", action="store_true",
                    help="Skip MuJoCo load; only check file references and bundle version")
    args = ap.parse_args()

    scene = Path(args.scene).resolve()
    mjcf_dir = scene.parent
    errors: list[str] = []

    print(f"Scene:   {scene}")
    print(f"MJCF dir: {mjcf_dir}\n")

    print("[1/4] Checking bundle version...")
    version, errs = check_bundle_version(mjcf_dir)
    if version:
        print(f"  Bundle: {version}")
    errors += errs

    print("[2/4] Checking XML-bundle consistency...")
    errors += check_xml_bundle_consistency(scene, version)

    print("[3/4] Checking file references...")
    errors += check_file_refs(scene)

    if not args.xml_only:
        print("[4/4] Loading scene with MuJoCo...")
        errors += check_mujoco_load(scene)
    else:
        print("[4/4] Skipped (--xml-only)")

    print()
    if errors:
        print(f"FAILED — {len(errors)} error(s):")
        for e in errors:
            print(f"  ✗  {e}")
        sys.exit(1)
    print("OK — all checks passed.")


if __name__ == "__main__":
    main()
