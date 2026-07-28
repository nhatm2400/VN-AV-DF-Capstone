"""Xác minh bản khóa toàn vẹn của experiment pilot V1 sau khi lưu trữ.

Vì sao cần script này
---------------------
`experiments/pilot_v1_.../manifest_hashes.json` khóa 8 SHA-256: 4 artifact mô hình
và 4 manifest input. Lượt lưu trữ 2026-07-28 đã dời 4 manifest đó sang archive/ và
thay tiền tố đường dẫn bên trong chúng, nên **4 input hash không còn khớp trực tiếp**.

File manifest_hashes.json cố ý KHÔNG được sửa — nó là bản khóa lịch sử, và sửa nó
thành hash hôm nay sẽ xóa mất khả năng phát hiện sửa đổi thật sự.

Thay vào đó, script này đảo ngược đúng phép thế tiền tố trong bộ nhớ rồi đối chiếu.
Khớp nghĩa là: nội dung ngoài đường dẫn chưa hề bị đụng tới, bản khóa vẫn có giá trị.

    D:\\Anaconda\\envs\\vn_av_df\\python.exe archive/pilot_v1/verify_lock.py

Chạy từ thư mục gốc repo. Thoát khác 0 nếu có bất kỳ hash nào lệch.
"""

import csv
import hashlib
import io
import json
import os
import sys

EXPERIMENT = "experiments/pilot_v1_20260720-214741_467f606_b8c61ed7"

# khóa trong manifest_hashes.json  ->  vị trí sau khi lưu trữ
MOVED = {
    "data/03_fake/snvsm/pilot_real_snvsm.csv":
        "archive/pilot_v1/03_fake/snvsm/pilot_real_snvsm.csv",
    "data/03_fake/snvsm/pilot_fake_snvsm.csv":
        "archive/pilot_v1/03_fake/snvsm/pilot_fake_snvsm.csv",
    "data/05_labels/labels_pilot.csv":
        "archive/pilot_v1/05_labels/labels_pilot.csv",
    "data/04_features_pilot/features_index.csv":
        "archive/pilot_v1/04_features_pilot/features_index.csv",
}

# đảo ngược phép thế của lượt lưu trữ 2026-07-28
UNDO = [
    ("archive\\pilot_v1\\03_fake", "data\\03_fake"),
    ("archive/pilot_v1/03_fake", "data/03_fake"),
    ("archive\\pilot_v1\\04_features_pilot", "data\\04_features_pilot"),
    ("archive/pilot_v1/04_features_pilot", "data/04_features_pilot"),
]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def sha256_undone(path):
    """Đảo tiền tố ở mọi cột path rồi hash bản dựng lại trong bộ nhớ."""
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames
        rows = list(reader)
    cols = [c for c in fields if c and "path" in c.lower()]
    undone = 0
    for row in rows:
        for col in cols:
            old = row.get(col) or ""
            new = old
            for a, b in UNDO:
                new = new.replace(a, b)
            if new != old:
                row[col] = new
                undone += 1
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return hashlib.sha256(buf.getvalue().encode("utf-8")).hexdigest().upper(), undone, len(rows)


def main():
    lock_path = os.path.join(EXPERIMENT, "manifest_hashes.json")
    if not os.path.isfile(lock_path):
        sys.exit(f"[LOI] khong thay {lock_path} — chay tu thu muc goc repo")
    with open(lock_path, encoding="utf-8") as fh:
        lock = json.load(fh)

    failed = 0

    print("-- 4 artifact mo hinh (khong bi dich chuyen, doi chieu truc tiep) --")
    for name, want in lock["artifacts"].items():
        path = os.path.join(EXPERIMENT, name)
        if not os.path.isfile(path):
            print(f"  [THIEU] {name}")
            failed += 1
            continue
        ok = sha256_file(path) == want
        failed += not ok
        print(f"  [{'KHOP' if ok else 'LECH'}] {name}")

    print("\n-- 4 manifest input (da luu tru, doi chieu qua phep dao nguoc) --")
    for key, want in lock["inputs"].items():
        path = MOVED.get(key)
        if not path or not os.path.isfile(path):
            print(f"  [THIEU] {key}  -> {path}")
            failed += 1
            continue
        got, undone, rows = sha256_undone(path)
        ok = got == want
        failed += not ok
        print(f"  [{'KHOP' if ok else 'LECH'}] {key}")
        print(f"         {rows} dong | dao nguoc {undone} duong dan | nay o {path}")
        if not ok:
            print(f"         khoa: {want}")
            print(f"         tinh: {got}")

    if failed:
        sys.exit(f"\n[LOI] {failed}/8 hash khong khop — noi dung da bi doi ngoai tien to duong dan")
    print("\n8/8 khop. Ban khoa experiment pilot V1 con nguyen gia tri.")


if __name__ == "__main__":
    main()
