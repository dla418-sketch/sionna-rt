import xml.etree.ElementTree as ET
from pathlib import Path
import copy
import os

# 파일 경로 설정
BASE_DIR = Path("/home/js/js/sionna-rt/src/sionna/rt/scenes/kmu_260220")
INPUT_XML = BASE_DIR / "kmu_260220.xml"
OUTPUT_XML = BASE_DIR / "kmu_260220_itu.xml"

# -----------------------------------------------------------
# [핵심] 스마트 재질 감지 로직
# ID를 일일이 적지 않고, 키워드로 판단합니다.
# -----------------------------------------------------------
def get_material_config(mat_id):
    mat_id_lower = mat_id.lower()
    
    # 1. 도로/길 (road, path, way, street) -> 콘크리트 (두께 얇게)
    if any(k in mat_id_lower for k in ["road", "path", "way", "street", "step"]):
        return {"type": "concrete", "thickness": 0.15, "desc": "도로/보도"}
    
    # 2. 식생 (vegetation, tree, grass, leaf) -> 나무
    if any(k in mat_id_lower for k in ["veg", "tree", "plant", "grass"]):
        return {"type": "wood", "thickness": 0.5, "desc": "식생"}
    
    # 3. 물 (water, lake, river) -> (일단 콘크리트로 하되 두껍게? 혹은 커스텀)
    if any(k in mat_id_lower for k in ["water", "lake"]):
        return {"type": "concrete", "thickness": 0.1, "desc": "물"}
        
    # 4. 그 외 (building, wall, default) -> 콘크리트 (두께 두껍게)
    # 건물은 보통 ID에 별다른 특징이 없거나 'material' 등이 붙음
    return {"type": "concrete", "thickness": 0.3, "desc": "건물(기본값)"}


def flatten_and_auto_fix_xml(in_path, out_path):
    print(f"🛠️ XML 변환 시작: {in_path}")
    tree = ET.parse(in_path)
    root = tree.getroot()
    
    # -------------------------------------------------------
    # 1단계: Instance 평탄화 (기존 동일)
    # -------------------------------------------------------
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
                    
                    transform = shape.find("transform")
                    if transform is not None:
                        old_tf = new_shape.find("transform")
                        if old_tf is not None: new_shape.remove(old_tf)
                        new_shape.append(transform)
                    
                    for str_node in new_shape.findall("string"):
                        if str_node.get("name") == "filename":
                            val = str_node.get("value")
                            if "meshes/" in val:
                                abs_path = BASE_DIR / "meshes" / os.path.basename(val)
                                str_node.set("value", str(abs_path))

                    root.remove(shape)
                    root.append(new_shape)
                    converted_count += 1
    
    print(f"✅ 구조 평탄화 완료: {converted_count}개 객체 변환")

    # -------------------------------------------------------
    # [NEW] 2단계: 재질 ID 리네이밍 (Map 생성)
    # -------------------------------------------------------
    id_mapping = {} # {옛날ID : 새ID} 저장소
    mat_count = 0
    
    for bsdf in root.findall("bsdf"):
        if bsdf.get("type") == "diffuse":
            old_id = bsdf.get("id", "")
            
            # 스마트 설정 가져오기
            cfg = get_material_config(old_id)
            mat_type = cfg["type"] # 예: concrete, wood
            
            # [핵심] 새 ID 생성: "재질타입_원래ID"
            # 이렇게 하면 이름에 'concrete'가 들어가면서도, 유일성이 보장됨
            # 예: mat-roads_residential -> concrete_roads_residential
            # 만약 원래 ID 필요없고 무조건 짧게 하고 싶다면 충돌 처리를 따로 해야 함.
            new_id = f"{mat_type}_{old_id.replace('mat-', '')}"
            
            # 매핑 저장
            id_mapping[old_id] = new_id
            
            # 1. BSDF ID 변경
            bsdf.set("id", new_id)
            # (name 속성도 있으면 같이 변경)
            if bsdf.get("name"):
                bsdf.set("name", new_id)

            # 2. 내용물 교체 (itu-radio-material)
            bsdf.set("type", "itu-radio-material")
            for child in list(bsdf):
                bsdf.remove(child)
            
            ET.SubElement(bsdf, "string", name="type", value=cfg["type"])
            ET.SubElement(bsdf, "float", name="thickness", value=str(cfg["thickness"]))
            
            mat_count += 1

    print(f"✅ 재질 ID 변경 및 매핑 완료: {mat_count}개")

    # -------------------------------------------------------
    # [NEW] 3단계: Shape들의 참조(Ref) 업데이트
    # 재질 ID를 바꿨으니, 그걸 쓰는 Shape들도 새 ID를 가리키게 수정
    # -------------------------------------------------------
    ref_update_count = 0
    for shape in root.findall("shape"):
        # shape 안에 있는 <ref name="bsdf" id="..."> 찾기
        for ref in shape.findall("ref"):
            if ref.get("name") == "bsdf":
                current_ref_id = ref.get("id")
                # 매핑된 새 ID가 있다면 교체
                if current_ref_id in id_mapping:
                    ref.set("id", id_mapping[current_ref_id])
                    ref_update_count += 1

    print(f"✅ Shape 참조 업데이트 완료: {ref_update_count}개 링크 수정됨")

    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    print(f"🎉 최종 파일 저장: {out_path}")

if __name__ == "__main__":
    if not INPUT_XML.exists():
        print(f"오류: 파일이 없습니다 -> {INPUT_XML}")
    else:
        flatten_and_auto_fix_xml(INPUT_XML, OUTPUT_XML)