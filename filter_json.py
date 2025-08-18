import json
import re

# Đọc file JSON gốc
with open("file_path.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Hàm chuyển đổi path -> "1/1/16"
def convert_path(path):
    # Tìm Lxx và Vxxx
    l_match = re.search(r"L(\d+)", path)
    v_match = re.search(r"V(\d+)", path)
    # Lấy số frame từ tên file (bỏ .webp, bỏ số 0 ở đầu)
    frame = int(path.split("/")[-1].replace(".webp", ""))
    return f"{int(l_match.group(1))}/{int(v_match.group(1))}/{frame}"

# Chuyển đổi toàn bộ
converted = {k: convert_path(v) for k, v in data.items()}

# Lưu ra file mới
with open("id2index.json", "w", encoding="utf-8") as f:
    json.dump(converted, f, indent=4, ensure_ascii=False)

print("Đã lưu id2index.json")