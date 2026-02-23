"""Convert Blender/Mitsuba XML scenes so Sionna can load triangle meshes.

This script focuses on one job only:
- Flatten `<shape type="shapegroup">` + `<shape type="instance">` into
  regular mesh `<shape>` nodes.

Why: Sionna's scene wrapper expects mesh shapes. Keeping instance/shapegroup
nodes can lead to `TypeError: Only triangle meshes are supported`.
"""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import xml.etree.ElementTree as ET

# Backward-compatible defaults
BASE_DIR = Path("/home/js/js/sionna-rt/src/sionna/rt/scenes/kmu_260220")
INPUT_XML = BASE_DIR / "kmu_260220.xml"
OUTPUT_XML = BASE_DIR / "kmu_260220_flat.xml"


def flatten_instances_only(in_path: Path, out_path: Path, base_dir: Path) -> None:
    """Flatten shapegroup/instance nodes and keep everything else unchanged."""
    print(f"🛠️ Start: {in_path}")

    tree = ET.parse(in_path)
    root = tree.getroot()

    # 1) Collect shapegroup templates, then remove shapegroup nodes.
    groups: dict[str, ET.Element] = {}
    removed_groups = 0
    for shape in list(root.findall("shape")):
        if shape.get("type") == "shapegroup":
            gid = shape.get("id")
            inner = shape.find("shape")
            if gid and inner is not None:
                groups[gid] = inner
            root.remove(shape)
            removed_groups += 1

    # 2) Replace each instance by a concrete copy of its referenced shape.
    converted_instances = 0
    for shape in list(root.findall("shape")):
        if shape.get("type") != "instance":
            continue

        ref = shape.find("ref[@name='shape']")
        if ref is None:
            continue

        ref_id = ref.get("id")
        if not ref_id or ref_id not in groups:
            continue

        new_shape = copy.deepcopy(groups[ref_id])

        # Keep instance identity
        if shape.get("id") is not None:
            new_shape.set("id", shape.get("id"))
        if shape.get("name") is not None:
            new_shape.set("name", shape.get("name"))

        # Apply instance transform to copied shape
        inst_tf = shape.find("transform[@name='to_world']")
        if inst_tf is not None:
            old_tf = new_shape.find("transform[@name='to_world']")
            if old_tf is not None:
                new_shape.remove(old_tf)
            new_shape.append(copy.deepcopy(inst_tf))

        # Resolve mesh path to absolute path for robust loading
        for str_node in new_shape.findall(".//string[@name='filename']"):
            value = str_node.get("value", "")
            if "meshes/" in value:
                abs_path = base_dir / "meshes" / os.path.basename(value)
                str_node.set("value", str(abs_path))

        root.remove(shape)
        root.append(new_shape)
        converted_instances += 1

    tree.write(out_path, encoding="utf-8", xml_declaration=True)

    print(f"✅ Removed shapegroups: {removed_groups}")
    print(f"✅ Converted instances: {converted_instances}")
    print(f"🎉 Saved: {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Flatten shapegroup/instance nodes to mesh shapes for Sionna"
    )
    p.add_argument("--input", type=Path, default=INPUT_XML, help="Input XML path")
    p.add_argument("--output", type=Path, default=OUTPUT_XML, help="Output XML path")
    p.add_argument(
        "--base-dir",
        type=Path,
        default=BASE_DIR,
        help="Scene base directory used to resolve meshes/ paths",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.input.exists():
        print(f"❌ File not found: {args.input}")
    else:
        flatten_instances_only(args.input, args.output, args.base_dir)
