"""Convert Blender/Mitsuba XML scenes so Sionna can load triangle meshes.

This script focuses on two practical jobs:
- Flatten `<shape type="shapegroup">` + `<shape type="instance">` into
  regular mesh `<shape>` nodes.
- Sanitize `mat-itu_*` IDs to valid ITU material types so `load_scene()`
  does not fail on invalid names such as `mat-itu_concrete1`.
"""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import re
import xml.etree.ElementTree as ET

# Backward-compatible defaults
BASE_DIR = Path("/home/js/js/sionna-rt/src/sionna/rt/scenes/28g")
INPUT_XML = BASE_DIR / "28g.xml"
OUTPUT_XML = BASE_DIR / "28g_flat.xml"

VALID_ITU_TYPES = {
    "concrete",
    "brick",
    "plasterboard",
    "wood",
    "glass",
    "ceiling_board",
    "chipboard",
    "plywood",
    "marble",
    "floorboard",
    "metal",
    "very_dry_ground",
    "medium_dry_ground",
    "wet_ground",
}

def _infer_itu_type(raw_type: str) -> str:
    """자유 형식 재질 이름에서 유효 ITU 타입을 추론한다."""
    key = raw_type.strip().lower()

    # 한글 주석: mat-, itu_, mat-itu_ 접두가 섞여 들어와도 타입 부분만 추출
    key = re.sub(r"^(mat-itu_|mat-|itu_)", "", key)
    if key in VALID_ITU_TYPES:
        return key

    # 한글 주석: plasterboard_01, plasterboardA, plasterboard2 같은 접미사를 허용
    for valid in VALID_ITU_TYPES:
        if key.startswith(valid):
            return valid

    # 한글 주석: concrete1 같은 숫자 접미사는 제거해서 복구
    key_no_digits = re.sub(r"\d+$", "", key)
    if key_no_digits in VALID_ITU_TYPES:
        return key_no_digits

    # 한글 주석: 도로/아스팔트 계열 이름은 concrete로 안전 매핑
    if any(token in key for token in ["road", "roads", "street", "asphalt", "pavement"]):
        return "concrete"

    # 한글 주석: 자주 나오는 키워드를 대응 ITU 타입으로 매핑
    if any(token in key for token in ["metal", "steel", "iron", "aluminum"]):
        return "metal"
    if any(token in key for token in ["glass", "window"]):
        return "glass"
    if any(token in key for token in ["wood", "tree", "vegetation", "forest"]):
        return "wood"

    # 한글 주석: 매핑 실패 시 concrete로 폴백해 로딩 실패 방지
    return "concrete"

def _sanitize_itu_bsdf_ids(root: ET.Element) -> tuple[int, int]:
    """`mat-itu_*` ID를 정규화하고 중복 bsdf를 정리한다."""
    id_map: dict[str, str] = {}
    canonical: dict[str, ET.Element] = {}
    changed = 0
    removed_duplicates = 0

    for bsdf in list(root.findall("bsdf")):
        old_id = bsdf.get("id", "")

        # 한글 주석: mat-concrete, itu_concrete, mat-itu_concrete 모두 정규화 대상
        if old_id.startswith("mat-itu_"):
            raw_type = old_id[len("mat-itu_"):]
        elif old_id.startswith("mat-"):
            raw_type = old_id[len("mat-"):]
        elif old_id.startswith("itu_"):
            raw_type = old_id[len("itu_"):]
        else:
            continue

        new_id = f"mat-itu_{_infer_itu_type(raw_type)}"
        id_map[old_id] = new_id

        if new_id in canonical and canonical[new_id] is not bsdf:
            root.remove(bsdf)
            removed_duplicates += 1
            changed += 1
            continue

        if old_id != new_id:
            bsdf.set("id", new_id)
            bsdf.set("name", new_id)
            changed += 1

        canonical[new_id] = bsdf

    # 한글 주석: shape의 bsdf ref도 동일한 새 ID로 동기화
    for ref in root.findall(".//ref[@name='bsdf']"):
        rid = ref.get("id")
        if rid in id_map and rid != id_map[rid]:
            ref.set("id", id_map[rid])
            changed += 1

    return changed, removed_duplicates
    
def flatten_instances_only(in_path: Path, out_path: Path, base_dir: Path) -> None:
    """Flatten shapegroup/instance nodes and sanitize ITU material IDs."""
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

    # 3) Normalize mat-itu_* names to valid ITU types
    mat_changed, mat_dedup = _sanitize_itu_bsdf_ids(root)

    tree.write(out_path, encoding="utf-8", xml_declaration=True)

    print(f"✅ Removed shapegroups: {removed_groups}")
    print(f"✅ Converted instances: {converted_instances}")
    print(f"✅ Material IDs normalized: {mat_changed} (duplicates removed: {mat_dedup})")
    print(f"🎉 Saved: {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Flatten shapegroup/instance nodes and sanitize mat-itu_* names"
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
