"""
03_sync_score.py — Tính LSE-C / LSE-D (Sync Confidence) — chạy GIỮA trong bộ 02/03/04

Đây là bước DUY NHẤT nối hình với tiếng: so cử động môi với sóng âm theo thời gian.
(Khác bước cắt 04_cut_clips: ở đó VAD chỉ kiểm "có tiếng" và YOLO chỉ kiểm "có mặt"
 một cách RIÊNG RẼ — không biết miệng trên hình có phải nguồn của tiếng đó không.)

Dựa trên joonson/syncnet_python (chuẩn để tính LSE-C/LSE-D trong Wav2Lip).
  LSE-C (confidence) = median(dists) - min(dists) theo sliding window → cao hơn = khớp hơn
  LSE-D (min_dist)   = khoảng cách Euclidean nhỏ nhất  → thấp hơn = khớp hơn

Hai chế độ:
  --calibrate  : chạy thử N clip ngẫu nhiên, in phân bố LSE-C để chọn ngưỡng
  --full       : chạy toàn bộ CSV, thêm cột sync_conf + sync_min_dist + sync_flag_reject

Trình tự bộ script:
  02_score_clips.py  (đo mặt + embedding)
  03_sync_score.py   (đo độ khớp môi-tiếng)        <-- file này
  04_curate.py       (cluster -> gate -> cân bằng -> xuất tập sạch)

SETUP (chạy 1 lần trước khi dùng script này):
  !git clone https://github.com/joonson/syncnet_python /kaggle/working/syncnet_python
  !cd /kaggle/working/syncnet_python && pip install -r requirements.txt -q
  !cd /kaggle/working/syncnet_python && bash download_model.sh
  # Nếu download_model.sh fail (Google Drive quota): tải thủ công và đặt vào
  # /kaggle/working/syncnet_python/data/syncnet_v2.model

LOCAL (TÙY CHỌN — sync khó dựng trên Windows; có thể BỎ, bước 04/05 vẫn chạy):
  cần repo syncnet_python trên máy, truyền --syncnet_dir <đường dẫn local>:
  python 03_sync_score.py --input_csv data/curate/tier1_scored_all.csv \\
      --syncnet_dir <repo_local> --calibrate
  # input mặc định không trỏ sẵn -> luôn truyền --input_csv (và --syncnet_dir).

Ví dụ (KAGGLE):
  # Calibrate trên 20 clip ngẫu nhiên (KHÔNG đặt ngưỡng — chỉ xem phân bố)
  python 03_sync_score.py \\
      --input_csv tier1_scored_tier1.csv \\
      --syncnet_dir /kaggle/working/syncnet_python \\
      --calibrate --n_samples 20

  # Chạy toàn bộ, thêm sync_conf vào CSV (workers=1 cho an toàn VRAM trên 1 GPU)
  python 03_sync_score.py \\
      --input_csv tier1_scored_tier1.csv \\
      --syncnet_dir /kaggle/working/syncnet_python \\
      --full --out tier1_scored_sync.csv --workers 1 --lse_c_threshold <từ calibrate>
"""

import os
import sys
import argparse
import subprocess
import tempfile
import shutil
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd


# ----------------------------- Helpers -----------------------------

def parse_offsets_txt(path):
    """
    Parse file pywork/<ref>/offsets.txt của joonson/syncnet_python.
    Format:
        AV offset:      0
        Min dist:       6.123
        Confidence:     8.456
    Trả về (lse_c, lse_d) hoặc (None, None) nếu không parse được.
    """
    if not os.path.exists(path):
        return None, None
    lse_c, lse_d = None, None
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("Confidence:"):
                    lse_c = float(line.split(":")[-1].strip())
                elif line.startswith("Min dist:"):
                    lse_d = float(line.split(":")[-1].strip())
    except Exception:
        return None, None
    return lse_c, lse_d


def run_syncnet_on_clip(clip_path, ref, syncnet_dir, data_dir):
    """
    Chạy syncnet pipeline + syncnet model trên một clip .mp4.
    Trả về (lse_c, lse_d, error_str).
    error_str = None nếu thành công.
    """
    # Bắt lỗi path sai SỚM (cạm bẫy path Kaggle) thay vì để S3FD fail khó hiểu
    if not isinstance(clip_path, str) or not os.path.isfile(clip_path):
        return None, None, "file không tồn tại (kiểm tra file_path)"

    py = sys.executable
    pipeline_script = os.path.join(syncnet_dir, "run_pipeline.py")
    syncnet_script  = os.path.join(syncnet_dir, "run_syncnet.py")

    # Bước 1: face detection + crop (S3FD)
    r1 = subprocess.run(
        [py, pipeline_script,
         "--videofile", clip_path,
         "--reference", ref,
         "--data_dir", data_dir],
        capture_output=True, text=True, cwd=syncnet_dir
    )
    if r1.returncode != 0:
        return None, None, f"pipeline fail: {r1.stderr[-200:]}"

    # Bước 2: sync model
    r2 = subprocess.run(
        [py, syncnet_script,
         "--videofile", clip_path,
         "--reference", ref,
         "--data_dir", data_dir],
        capture_output=True, text=True, cwd=syncnet_dir
    )
    if r2.returncode != 0:
        return None, None, f"syncnet fail: {r2.stderr[-200:]}"

    offsets_path = os.path.join(data_dir, "pywork", ref, "offsets.txt")
    lse_c, lse_d = parse_offsets_txt(offsets_path)
    if lse_c is None:
        return None, None, "offsets.txt không parse được hoặc không có mặt"
    return lse_c, lse_d, None


def process_one(args):
    """Worker cho ThreadPoolExecutor. args = (clip_id, file_path, syncnet_dir, base_data_dir)"""
    clip_id, file_path, syncnet_dir, base_data_dir = args

    # Mỗi clip dùng thư mục data riêng để tránh xung đột khi chạy song song
    data_dir = os.path.join(base_data_dir, clip_id)
    os.makedirs(data_dir, exist_ok=True)
    ref = "clip"  # tên cố định ổn vì data_dir đã unique

    lse_c, lse_d, err = run_syncnet_on_clip(file_path, ref, syncnet_dir, data_dir)

    # Dọn thư mục tạm ngay để không tràn đĩa
    shutil.rmtree(data_dir, ignore_errors=True)

    return clip_id, lse_c, lse_d, err


# ----------------------------- Calibrate -----------------------------

def calibrate_mode(df, syncnet_dir, n_samples, base_data_dir):
    sample = df.sample(min(n_samples, len(df)), random_state=42)
    print(f"\n===== CALIBRATE: chạy syncnet trên {len(sample)} clip ngẫu nhiên =====")
    print("(Không xóa hay sửa CSV — chỉ in phân bố để chọn ngưỡng)\n")

    results = []
    t0 = time.time()
    for i, row in enumerate(sample.itertuples(index=False)):
        clip_id   = row.clip_id
        file_path = row.file_path
        data_dir  = os.path.join(base_data_dir, f"calib_{clip_id}")
        os.makedirs(data_dir, exist_ok=True)

        lse_c, lse_d, err = run_syncnet_on_clip(file_path, "clip", syncnet_dir, data_dir)
        shutil.rmtree(data_dir, ignore_errors=True)

        status = f"LSE-C={lse_c:.2f} LSE-D={lse_d:.3f}" if lse_c is not None else f"FAIL ({err})"
        print(f"  [{i+1}/{len(sample)}] {clip_id}  {status}")
        results.append({"clip_id": clip_id, "lse_c": lse_c, "lse_d": lse_d, "error": err})

    el = time.time() - t0
    print(f"\nHoàn tất {len(sample)} clip trong {el/60:.1f} phút "
          f"({el/len(sample):.1f}s/clip)")

    rdf = pd.DataFrame(results)
    valid = rdf.dropna(subset=["lse_c"])
    fail_n = rdf["lse_c"].isna().sum()
    print(f"\n-- Kết quả --")
    print(f"  Thành công: {len(valid)} | Fail (không có mặt / lỗi): {fail_n}")

    if len(valid) > 0:
        print("\n-- Phân bố LSE-C (confidence: cao hơn = khớp hơn) --")
        print(valid["lse_c"].describe(percentiles=[.1, .25, .5, .75, .9]).round(2).to_string())
        print("\n-- Phân bố LSE-D (min dist: thấp hơn = khớp hơn) --")
        print(valid["lse_d"].describe(percentiles=[.1, .25, .5, .75, .9]).round(3).to_string())

        p10 = valid["lse_c"].quantile(0.10)
        p25 = valid["lse_c"].quantile(0.25)
        print(f"\n-- Gợi ý ngưỡng --")
        print(f"  Conservative (loại 10% đuôi thấp nhất): LSE-C < {p10:.2f}")
        print(f"  Moderate     (loại 25% đuôi thấp nhất): LSE-C < {p25:.2f}")
        print(f"\n  Nguyên tắc: chỉ loại RÁC RÕ RÀNG (đuôi cực thấp), KHÔNG siết tập real.")
        print(f"  Siết sync mạnh cho real -> đẩy real về 'sync cao' -> model học tắt (leakage).")
        print(f"  NGHE THỬ vài clip có LSE-C thấp nhất để xác nhận trước khi chốt ngưỡng.")

    calib_out = "calibrate_sync_results.csv"
    rdf.to_csv(calib_out, index=False)
    print(f"\nKết quả chi tiết -> {calib_out}")
    print("Tiếp theo: chạy --full --lse_c_threshold <ngưỡng bạn chọn>")


# ----------------------------- Full -----------------------------

def full_mode(df, syncnet_dir, base_data_dir, out_path, workers, lse_c_threshold):
    print(f"\n===== FULL: tính sync_conf cho {len(df)} clip =====")
    if lse_c_threshold is None:
        print(f"  workers={workers} | chưa đặt ngưỡng -> chỉ flag clip FAIL (không có mặt nói)\n")
    else:
        print(f"  workers={workers} | ngưỡng gate LSE-C < {lse_c_threshold} (từ calibrate)\n")

    tasks = [
        (row.clip_id, row.file_path, syncnet_dir, base_data_dir)
        for row in df.itertuples(index=False)
    ]

    results = {}  # clip_id -> (lse_c, lse_d, err)
    t0 = time.time()
    done = 0
    fail = 0

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(process_one, t): t[0] for t in tasks}
        for fut in as_completed(futs):
            clip_id, lse_c, lse_d, err = fut.result()
            results[clip_id] = (lse_c, lse_d, err)
            done += 1
            if err:
                fail += 1
            if done % 200 == 0:
                el = time.time() - t0
                print(f"  {done}/{len(tasks)}  fail={fail}  ({el/60:.1f}m, {done/el:.1f} clip/s)")

    # Ghi kết quả vào CSV
    df = df.copy()
    df["sync_conf"]     = df["clip_id"].map(lambda c: results.get(c, (None,))[0])
    df["sync_min_dist"] = df["clip_id"].map(lambda c: results.get(c, (None, None))[1])

    # Cột flag: True = nên loại. Mặc định (chưa có ngưỡng) CHỈ flag fail hoàn toàn.
    # Có ngưỡng thì thêm "LSE-C quá thấp". Giữ triết lý: chỉ cắt rác rõ ràng.
    if lse_c_threshold is None:
        df["sync_flag_reject"] = df["sync_conf"].isna()
    else:
        df["sync_flag_reject"] = (
            df["sync_conf"].isna() |                # fail = không có mặt nói
            (df["sync_conf"] < lse_c_threshold)     # khớp kém rõ ràng
        )

    el = time.time() - t0
    n_valid   = df["sync_conf"].notna().sum()
    n_reject  = df["sync_flag_reject"].sum()
    print(f"\nXong {len(df)} clip trong {el/60:.1f} phút")
    print(f"  valid: {n_valid} | fail/reject: {n_reject} ({n_reject/len(df)*100:.1f}%)")

    if n_valid > 0:
        print("\n-- Phân bố LSE-C toàn bộ --")
        print(df["sync_conf"].describe(percentiles=[.1,.25,.5,.75,.9]).round(2).to_string())

    df.to_csv(out_path, index=False)
    print(f"\nCSV mới (có cột sync_conf) -> {out_path}")
    print("Tiếp theo: dùng file này làm input cho 04_curate.py "
          "(sẽ tự nhận cột sync_conf vào quality score).")


# ----------------------------- Main -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv",   required=True,  help="tier1_scored_*.csv (output của 03a)")
    ap.add_argument("--syncnet_dir", required=True,  help="đường dẫn đến repo joonson/syncnet_python")
    ap.add_argument("--calibrate",   action="store_true")
    ap.add_argument("--full",        action="store_true")
    ap.add_argument("--n_samples",   type=int, default=20, help="số clip cho calibrate")
    ap.add_argument("--workers",     type=int, default=1,
                    help="số luồng song song. MẶC ĐỊNH 1: mỗi clip nạp S3FD + SyncNet lên "
                         "GPU, chạy 4 luồng trên 1 T4 dễ OOM. Tăng 2 chỉ khi còn dư VRAM.")
    ap.add_argument("--lse_c_threshold", type=float, default=None,
                    help="ngưỡng loại: clip LSE-C < ngưỡng bị đánh dấu sync_flag_reject=True. "
                         "MẶC ĐỊNH None = chỉ flag clip fail. ĐẶT SAU KHI xem calibrate, "
                         "và chỉ nên cắt đuôi cực thấp (tránh leakage cho tập real).")
    ap.add_argument("--out", default="tier1_scored_sync.csv")
    ap.add_argument("--data_dir", default="/tmp/syncnet_tmp",
                    help="thư mục tạm cho syncnet (xóa sau mỗi clip)")
    args = ap.parse_args()

    if not args.calibrate and not args.full:
        print("Chỉ định --calibrate hoặc --full")
        sys.exit(1)

    # Kiểm tra syncnet_dir
    required = ["run_pipeline.py", "run_syncnet.py", "SyncNetInstance.py"]
    missing = [f for f in required if not os.path.exists(os.path.join(args.syncnet_dir, f))]
    if missing:
        print(f"[LỖI] Không tìm thấy {missing} trong {args.syncnet_dir}")
        print("Chạy setup trước:")
        print("  !git clone https://github.com/joonson/syncnet_python <syncnet_dir>")
        print("  !cd <syncnet_dir> && bash download_model.sh")
        sys.exit(1)

    df = pd.read_csv(args.input_csv)
    print(f"Đọc {len(df)} clip từ {args.input_csv}")

    os.makedirs(args.data_dir, exist_ok=True)

    if args.calibrate:
        calibrate_mode(df, args.syncnet_dir, args.n_samples, args.data_dir)
    else:
        full_mode(df, args.syncnet_dir, args.data_dir, args.out, args.workers, args.lse_c_threshold)


if __name__ == "__main__":
    main()
