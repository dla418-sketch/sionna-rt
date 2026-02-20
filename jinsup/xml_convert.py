import argparse
import copy
import os
from pathlib import Path
import xml.etree.ElementTree as ET

# 기본 경로 (기존 스크립트와 호환)
BASE_DIR = Path("/home/js/js/sionna-rt/src/sionna/rt/scenes/kmu_260220")
INPUT_XML = BASE_DIR / "kmu_260220.xml"
OUTPUT_XML = BASE_DIR / "kmu_260220_itu.xml"

# Axis remap preset matrices (row-major 4x4)
# These are applied as shape transforms in world coordinates.
AXIS_REMAP_MATRICES = {
    "none": [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ],
    # (x, y, z) -> (x, z, y)
    "swap_yz": [
        1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ],
    # (x, y, z) -> (x, z, -y)
    # Blender export에서 자주 보이는 회전/축변환과 유사한 보정
    "blender_like": [
        1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, -1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ],
}


def get_material_config(mat_id):
    mat_id_lower = mat_id.lower()

    # 1. 도로/길 (road, path, way, street) -> 콘크리트 (두께 얇게)
    if any(k in mat_id_lower for k in ["road", "path", "way", "street", "step"]):
        return {"type": "concrete", "thickness": 0.15, "desc": "도로/보도"}

    # 2. 식생 (vegetation, tree, grass, leaf) -> 나무
    if any(k in mat_id_lower for k in ["veg", "tree", "plant", "grass", "forest"]):
        return {"type": "wood", "thickness": 0.5, "desc": "식생"}

    # 3. 물 (water, lake, river) -> 콘크리트 (임시)
    if any(k in mat_id_lower for k in ["water", "lake", "river"]):
        return {"type": "concrete", "thickness": 0.1, "desc": "물"}

    # 4. 그 외 -> 콘크리트 기본값
    return {"type": "concrete", "thickness": 0.3, "desc": "건물(기본값)"}


def _find_filename_nodes(shape_node):
    """shape 노드 하위의 filename string 노드를 모두 찾음"""
    return shape_node.findall(".//string[@name='filename']")


def _make_transform_from_matrix(matrix_values):
    tf = ET.Element("transform", {"name": "to_world"})
    ET.SubElement(
        tf,
        "matrix",
        {"value": " ".join(f"{v:.6f}" for v in matrix_values)},
    )
    return tf


def _apply_axis_remap_to_shape(shape_node, remap_mode):
    if remap_mode == "none":
        return False

    remap_matrix = AXIS_REMAP_MATRICES[remap_mode]
    remap_tf = _make_transform_from_matrix(remap_matrix)

    old_tf = shape_node.find("transform[@name='to_world']")
    if old_tf is not None:
        shape_node.remove(old_tf)
        remap_tf.append(old_tf)

    shape_node.append(remap_tf)
    return True


def flatten_and_auto_fix_xml(
    in_path,
    out_path,
    base_dir,
    apply_instance_transform=True,
    axis_remap="none",
):
    print(f"🛠️ XML 변환 시작: {in_path}")
    print(f"   - instance transform 적용: {apply_instance_transform}")
    print(f"   - 축 재매핑 모드: {axis_remap}")

    tree = ET.parse(in_path)
    root = tree.getroot()

    # 1단계: Instance 평탄화
    groups = {}
    for shape in list(root.findall("shape")):
        if shape.get("type") == "shapegroup":
            group_id = shape.get("id")
            inner_shape = shape.find("shape")
            if inner_shape is not None:
                groups[group_id] = inner_shape
            root.remove(shape)

    converted_count = 0
    for shape in list(root.findall("shape")):
        if shape.get("type") == "instance":
            ref_node = shape.find("ref")
            if ref_node is not None:
                ref_id = ref_node.get("id")
                if ref_id in groups:
                    new_shape = copy.deepcopy(groups[ref_id])
                    new_shape.set("id", shape.get("id"))
                    if shape.get("name") is not None:
                        new_shape.set("name", shape.get("name"))

                    if apply_instance_transform:
                        transform = shape.find("transform[@name='to_world']")
                        if transform is not None:
                            old_tf = new_shape.find("transform[@name='to_world']")
                            if old_tf is not None:
                                new_shape.remove(old_tf)
                            new_shape.append(copy.deepcopy(transform))
                    else:
                        old_tf = new_shape.find("transform[@name='to_world']")
                        if old_tf is not None:
                            new_shape.remove(old_tf)

                    for str_node in _find_filename_nodes(new_shape):
                        val = str_node.get("value", "")
                        if "meshes/" in val:
                            abs_path = Path(base_dir) / "meshes" / os.path.basename(val)
                            str_node.set("value", str(abs_path))

                    root.remove(shape)
                    root.append(new_shape)
                    converted_count += 1

    print(f"✅ 구조 평탄화 완료: {converted_count}개 객체 변환")

    # 2단계: 재질 변환
    id_mapping = {}
    mat_count = 0

    for bsdf in root.findall("bsdf"):
        if bsdf.get("type") == "diffuse":
            old_id = bsdf.get("id", "")

            cfg = get_material_config(old_id)
            mat_type = cfg["type"]
            new_id = f"{mat_type}_{old_id.replace('mat-', '')}"

            id_mapping[old_id] = new_id

            bsdf.set("id", new_id)
            if bsdf.get("name"):
                bsdf.set("name", new_id)

            bsdf.set("type", "itu-radio-material")
            for child in list(bsdf):
                bsdf.remove(child)

            ET.SubElement(bsdf, "string", name="type", value=cfg["type"])
            ET.SubElement(bsdf, "float", name="thickness", value=str(cfg["thickness"]))

            mat_count += 1

    print(f"✅ 재질 ID 변경 및 매핑 완료: {mat_count}개")

    # 3단계: ref 업데이트
    ref_update_count = 0
    for shape in root.findall("shape"):
        for ref in shape.findall("ref"):
            if ref.get("name") == "bsdf":
                current_ref_id = ref.get("id")
                if current_ref_id in id_mapping:
                    ref.set("id", id_mapping[current_ref_id])
                    ref_update_count += 1

    print(f"✅ Shape 참조 업데이트 완료: {ref_update_count}개 링크 수정됨")

    # 4단계: 축 재매핑 적용
    remap_count = 0
    if axis_remap != "none":
        for shape in root.findall("shape"):
            if shape.get("type") in {"ply", "obj", "serialized", "mesh"}:
                if _apply_axis_remap_to_shape(shape, axis_remap):
                    remap_count += 1
        print(f"✅ 축 재매핑 transform 적용 완료: {remap_count}개 shape")

    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    print(f"🎉 최종 파일 저장: {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Blender export XML을 Sionna RT 친화 포맷으로 변환"
    )
    parser.add_argument("--input", type=Path, default=INPUT_XML, help="입력 XML 경로")
    parser.add_argument("--output", type=Path, default=OUTPUT_XML, help="출력 XML 경로")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=BASE_DIR,
        help="scene base 디렉토리(메쉬 상대경로 해석용)",
    )
    parser.add_argument(
        "--no-instance-transform",
        action="store_true",
        help="instance의 to_world transform을 적용하지 않음",
    )
    parser.add_argument(
        "--axis-remap",
        choices=tuple(AXIS_REMAP_MATRICES.keys()),
        default="none",
        help="축 재매핑 모드: none | swap_yz | blender_like",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.input.exists():
        print(f"오류: 파일이 없습니다 -> {args.input}")
    else:
        flatten_and_auto_fix_xml(
            in_path=args.input,
            out_path=args.output,
            base_dir=args.base_dir,
            apply_instance_transform=not args.no_instance_transform,
            axis_remap=args.axis_remap,
        )
