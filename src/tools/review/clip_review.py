"""
clip_review.py — Công cụ LỌC TAY clip (manual QA) chạy trong trình duyệt.

Mục đích: review clip bằng mắt người, đánh KEEP / REJECT(+lý do) / UNCERTAIN, để so sánh
"lọc tay" với "lọc bằng code" (04_curate -> all_clean.csv). Kết quả ghi sang FOLDER RIÊNG
nên không đụng tới output của pipeline tự động.

Không cần cài gì thêm — chỉ dùng thư viện chuẩn Python + trình duyệt.

CÁCH DÙNG (chạy từ thư mục gốc dự án):
    D:/Anaconda/envs/vn_av_df/python.exe src/tools/review/build_review_manifest.py  # dựng manifest (chạy 1 lần)
    D:/Anaconda/envs/vn_av_df/python.exe src/tools/review/build_roi_preview.py      # dựng ô ROI (chạy 1 lần, ~85 phút)
    D:/Anaconda/envs/vn_av_df/python.exe src/tools/review/clip_review.py            # rồi mở http://127.0.0.1:8000

    # mẻ hiệu chuẩn: N clip chia đều 3 tầng theo một cột đo được
    D:/Anaconda/envs/vn_av_df/python.exe src/tools/review/clip_review.py --sample 60 --stratify motion_median --reviewer nhat

Ô ROI (quan trọng nhất): cạnh video gốc có ô phát CHUỖI ROI THẬT mà stage 04 cắt ra,
ghép với AUDIO GỐC. Video gốc bị tắt tiếng để không vọng đôi. Nhìn miệng + nghe tiếng
cùng lúc là lộ ngay cả ba lỗi: cắt nhầm mặt (miệng đứng im khi tiếng đang nói), lồng
tiếng (môi lệch nhịp), ảnh tĩnh (miệng đóng băng). Dựng trước bằng build_roi_preview.py.

Tùy chọn:
    --csv       manifest cần review     (mặc định manifests/all_clean_review.csv)
    --roi_dir   thư mục preview ROI+tiếng (mặc định data/02_curate/roi_preview)
    --media_root  thư mục chứa clip gốc TRÊN MÁY NÀY — quét đệ quy, tra theo tên file
                  <clip_id>.mp4. BẮT BUỘC khi review trên máy khác máy dựng manifest,
                  vì file_path trong manifest là đường dẫn tuyệt đối của máy đó.
    --out       file ghi quyết định     (mặc định tự sinh theo --csv + rubric + reviewer)
    --compare   file code GIỮ để đối chiếu     (mặc định manifests/all_clean.csv)
    --rejects   file code GATE-REJECT          (mặc định manifests/all_clean_rejects.csv)
    --order     diverse (mặc định) = vòng tròn qua từng video | manifest = thứ tự gốc
    --exclude_channel  bỏ qua kênh, ngăn cách bằng ';' — áp TRƯỚC --sample
    --sample    chỉ review N clip ngẫu nhiên   (0 = toàn bộ)
    --stratify  cột chia 3 tầng cho --sample (vd motion_median); rỗng = ngẫu nhiên thuần
    --seed      seed cho --sample và cho thứ tự trong `diverse` (mặc định 42)
    --reviewer  mã người review, ghi vào CSV   (mặc định biến môi trường USERNAME)
    --col       cột chứa đường dẫn .mp4        (mặc định file_path)
    --port      cổng web                       (mặc định 8000)
    --no-browser   không tự mở trình duyệt

KHÔNG NHẤT THIẾT PHẢI REVIEW HẾT: mục tiêu là đủ số real cho pilot, nên đếm số KEEP
chứ đừng đếm số clip đã xem — tỉ lệ giữ chỉ ước lượng được từ mẫu đã review, sai số
còn rộng. Thứ tự `diverse` bảo đảm dừng lúc nào cũng có tập trải đều nguồn video, NHƯNG
đa dạng của hàng đợi không tự suy ra đa dạng của tập KEEP: phải audit chính tập keep
(tier / speaker / source_video / channel / connected-component) trước khi chốt.

KHÔNG auto-reject theo kênh. `--exclude_channel` là công cụ thủ công để hoãn một kênh
đã biết rõ định dạng; đừng dùng nó thay cho việc xem nội dung clip. Tỉ lệ keep theo kênh
đo trên mẫu nhỏ có khoảng tin cậy rất rộng.

OUTPUT TỰ ĐỘNG VERSION THEO SCOPE: mặc định ghi ra
    data/02_curate/manual/manual_<tên-csv>_<rubric>_<reviewer>.csv
Mỗi manifest + mỗi phiên bản rubric có file riêng, nên KHÔNG bao giờ trộn nhầm quyết định
của hai scope khác nhau (file manual_review.csv cũ thuộc scope tier1_scored_all và chỉ có
3/47 dòng nằm trong all_clean — đừng tái sử dụng).

SO SÁNH 3 TRẠNG THÁI: mỗi clip code có 1 trong 3 trạng thái — KEEP (trong all_clean),
GATE-REJECT (trong all_clean_rejects, = rác), BALANCE-DROP (qua gate nhưng bị cắt do cân
bằng cap/speaker, = clip TỐT bị bỏ vì thừa). Đồng thuận "trục rác" chỉ tính trên clip code
có phán chất lượng (KEEP hoặc GATE-REJECT); balance-drop KHÔNG tính là bất đồng.
Clip đánh UNCERTAIN không tính vào đồng thuận.

GHI CHÚ CHỐNG LEAKAGE: khi lọc tập REAL, chỉ loại "rác rõ ràng". KHÔNG loại theo "khớp môi
tốt/xấu" và tiêu chí phải áp y hệt cho tập fake sau này — nếu không sẽ rò rỉ nhãn.
Real bị loại thì phải loại CẢ CỤM (1 real + 4 fake sinh từ nó), không loại lẻ một fake.
"""
import argparse
import csv
import json
import os
import re
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ----- rubric -----
# Đổi nội dung/ý nghĩa lý do -> TĂNG version, vì quyết định cũ không còn so sánh được.
RUBRIC_VERSION = "v3"
REASONS = [
    ("static",     "Ảnh tĩnh / miệng không động"),
    ("voiceover",  "Người trong hình không nói — tiếng của người ngoài hình"),
    ("dubbed",     "Lồng tiếng — môi không ăn nhịp / khác ngôn ngữ"),
    ("wrong_face", "ROI cắt nhầm người (không phải người đang nói)"),
    ("mouth",      "Miệng bị che / nghiêng quá / ra khỏi ô ROI"),
    ("cut",        "Có chuyển cảnh (chỉ mô tả; thêm static/voiceover nếu đúng)"),
    ("broken",     "Lỗi file / audio hỏng / quá ít tiếng nói"),
]
REASON_KEYS = [k for k, _ in REASONS]
DECISIONS_VALID = ("keep", "reject", "uncertain")

# ----- global state -----
CLIPS = []            # list of dict cần review (toàn bộ HOẶC mẫu sample) — index theo 'i'
ALL_ROWS = []         # toàn bộ manifest (để tra file_path/thứ tự cho mọi clip)
FILEPATH_BY_ID = {}   # clip_id -> file_path (từ toàn bộ manifest)
ORDER_BY_ID = {}      # clip_id -> thứ tự gốc trong manifest (để ghi CSV ổn định)
DECISIONS = {}        # clip_id -> "keep"/"reject"/"uncertain"  (giữ MỌI quyết định)
REASON = {}           # clip_id -> mã lý do (chỉ có nghĩa khi decision == reject)
BAD_INTERVALS = {}    # clip_id -> list[{start_ms,end_ms,reason}]
TS = {}               # clip_id -> timestamp string
REVIEWER = {}         # clip_id -> mã người review
RUBRIC = {}           # clip_id -> rubric version lúc đánh dấu
CODE_KEEP = set()     # clip_id code GIỮ (all_clean.csv)
CODE_GATE = set()     # clip_id code GATE-REJECT (all_clean_rejects.csv) = rác
OUT = ""
PATH_COL = "file_path"
REVIEWER_ID = ""
ROI_DIR = ""
LOCK = threading.Lock()

META_FIELDS = ["tier", "source_video", "duration", "snr", "det_ratio",
               "mean_face_area", "embed_consistency", "face_ratio", "speech_ratio",
               "motion_median", "frac_near_static", "n_faces_med", "ratio_med",
               "sync_conf", "voiced_ms", "temporal_decision", "temporal_reason",
               "unexplained_speech_ratio", "longest_unexplained_speech_ms"]


def normalize_bad_intervals(value):
    """Validate rubric-v3 intervals and return a canonical, sorted list."""
    if isinstance(value, str):
        try:
            value = json.loads(value or "[]")
        except json.JSONDecodeError as error:
            raise ValueError("bad_intervals_json không phải JSON hợp lệ") from error
    if value is None:
        value = []
    if not isinstance(value, list):
        raise ValueError("bad_intervals phải là một danh sách")
    output = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("mỗi interval phải là object")
        reason = str(item.get("reason", ""))
        if reason not in REASON_KEYS:
            raise ValueError(f"lý do interval không hợp lệ: {reason}")
        try:
            start_ms = int(round(float(item["start_ms"])))
            end_ms = int(round(float(item["end_ms"])))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("interval thiếu start_ms/end_ms hợp lệ") from error
        if start_ms < 0 or end_ms <= start_ms:
            raise ValueError("interval phải có 0 <= start_ms < end_ms")
        output.append({"start_ms": start_ms, "end_ms": end_ms, "reason": reason})
    output.sort(key=lambda item: (item["start_ms"], item["end_ms"], item["reason"]))
    return output


def longest_interval_reason(intervals):
    if not intervals:
        return ""
    longest = max(intervals, key=lambda item: (item["end_ms"] - item["start_ms"],
                                               -item["start_ms"]))
    return longest["reason"]


def intervals_are_material(intervals, voiced_ms):
    """Rubric v3: >=800 ms continuous, or >=500 ms total and >=20% voiced."""
    if not intervals:
        return False
    spans = sorted((item["start_ms"], item["end_ms"]) for item in intervals)
    merged = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    lengths = [end - start for start, end in merged]
    total = sum(lengths)
    return max(lengths) >= 800 or (
        total >= 500 and voiced_ms > 0 and total / voiced_ms >= 0.20
    )


def roi_path(clip_id):
    return os.path.join(ROI_DIR, clip_id + ".mp4") if ROI_DIR else ""


def diverse_order(rows, video_col="source_video", seed=42):
    """Vòng tròn qua từng source_video: mỗi vòng lấy 1 clip của mỗi video.

    Mục đích: dừng review lúc nào cũng có tập trải đều nguồn video, thay vì gom hết
    clip của vài video rồi mới sang video khác. (Đa dạng nguồn là tuyến phòng thủ
    chính chống identity leakage — xem PROJECT.md.)

    Trong mỗi video, thứ tự clip được xáo theo `seed` chứ không lấy theo thứ tự
    manifest — manifest sắp theo speaker_id/start_time nên lấy tuần tự sẽ thiên về
    clip đầu buổi quay. Cùng seed -> cùng thứ tự, nên resume được.

    (Thay cho `risk_score` cũ — kiểm trên 60 clip gán nhãn tay: nhóm "rủi ro" bị loại
    60% vs nhóm còn lại 55%, tức không có giá trị dự báo.)
    """
    import random
    by_video = {}
    for r in rows:
        by_video.setdefault(r.get(video_col, ""), []).append(r)
    for v, items in by_video.items():
        random.Random(f"{seed}:{v}").shuffle(items)      # seed riêng mỗi video
    keyed = []
    for v, items in by_video.items():
        for rank, r in enumerate(items):
            keyed.append((rank, v, r))
    keyed.sort(key=lambda t: (t[0], t[1]))
    return [r for _, _, r in keyed]


def index_media(root):
    """Quét đệ quy `root`, trả {tên file: đường dẫn} — để chạy trên MÁY KHÁC.

    `file_path` trong manifest là đường dẫn tuyệt đối trên máy dựng manifest
    (E:\\FPTU\\PRJ\\...), nên reviewer đặt media ở đâu khác là hỏng hết. Tên file
    lại đúng bằng `<clip_id>.mp4` và phải duy nhất trong manifest đang review, nên tra
    theo tên file bền hơn hẳn việc ghép lại tiền tố: reviewer muốn sắp xếp thư
    mục thế nào cũng được, miễn media nằm đâu đó dưới `root`.
    """
    index = {}
    dup = 0
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if not name.lower().endswith(".mp4"):
                continue
            if name in index:
                dup += 1
                continue
            index[name] = os.path.join(dirpath, name)
    return index, dup


def load_clips(csv_path, path_col, media_index=None):
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            fp = (r.get(path_col) or "").strip()
            if media_index is not None:
                fp = media_index.get(r["clip_id"] + ".mp4", fp)
            r["_abspath"] = fp
            rows.append(r)
    return rows


def load_decisions(out_path):
    """Đọc lại quyết định cũ. Chấp nhận file cũ chưa có cột reason/reviewer/rubric."""
    if not os.path.isfile(out_path):
        return
    other_rubric = set()
    with open(out_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            cid = r.get("clip_id")
            dec = r.get("decision")
            if cid and dec in DECISIONS_VALID:
                row_reviewer = r.get("reviewer_id", "") or ""
                if row_reviewer and row_reviewer != REVIEWER_ID:
                    raise SystemExit(
                        f"[LỖI] Output chứa reviewer '{row_reviewer}', "
                        f"không khớp --reviewer '{REVIEWER_ID}': {out_path}"
                    )
                DECISIONS[cid] = dec
                REASON[cid] = r.get("reason", "") or ""
                try:
                    BAD_INTERVALS[cid] = normalize_bad_intervals(
                        r.get("bad_intervals_json", "") or "[]"
                    )
                except ValueError as error:
                    raise SystemExit(f"[LỖI] {cid}: {error}") from error
                TS[cid] = r.get("ts", "")
                REVIEWER[cid] = r.get("reviewer_id", "") or ""
                rv = r.get("rubric_version", "") or ""
                RUBRIC[cid] = rv
                if rv and rv != RUBRIC_VERSION:
                    other_rubric.add(rv)
    if other_rubric:
        print(f"[CẢNH BÁO] File output có quyết định theo rubric {sorted(other_rubric)}, "
              f"khác rubric hiện tại {RUBRIC_VERSION}. Nên review lại các dòng đó.")


def _read_ids(path):
    out = set()
    if path and os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                cid = r.get("clip_id")
                if cid:
                    out.add(cid)
    return out


def load_code_states(keep_path, gate_path):
    CODE_KEEP.update(_read_ids(keep_path))
    CODE_GATE.update(_read_ids(gate_path))


def code_state(cid):
    """Trạng thái của clip theo lọc code: keep / gate (rác) / balance (tốt nhưng thừa)."""
    if cid in CODE_KEEP:
        return "keep"
    if cid in CODE_GATE:
        return "gate"
    return "balance"


def save_decisions():
    """Ghi MỌI quyết định (atomic), theo thứ tự gốc manifest — không phụ thuộc sample."""
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    tmp = OUT + ".tmp"
    items = sorted(DECISIONS.items(), key=lambda kv: ORDER_BY_ID.get(kv[0], 1 << 30))
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["clip_id", "file_path", "decision", "reason", "bad_intervals_json",
                    "reviewer_id", "rubric_version", "ts"])
        for cid, d in items:
            if d:
                w.writerow([cid, FILEPATH_BY_ID.get(cid, ""), d,
                            REASON.get(cid, ""),
                            json.dumps(BAD_INTERVALS.get(cid, []), ensure_ascii=False,
                                       separators=(",", ":")),
                            REVIEWER.get(cid, ""),
                            RUBRIC.get(cid, ""), TS.get(cid, "")])
    os.replace(tmp, OUT)


def now():
    import datetime
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def first_undecided():
    for i, c in enumerate(CLIPS):
        if c["clip_id"] not in DECISIONS:
            return i
    return 0


def compare_summary():
    """
    Ma trận 2x3 giữa lọc tay (keep/reject) và trạng thái code (keep/gate/balance),
    chỉ trên clip ĐÃ đánh dấu tay trong tập đang review (CLIPS).
    Đồng thuận "trục rác" = tay keep&code keep + tay reject&code gate, tính trên các clip
    code có phán chất lượng (keep hoặc gate). Balance-drop KHÔNG tính là bất đồng.
    UNCERTAIN không tính vào ma trận.
    """
    m = {"keep": {"keep": 0, "gate": 0, "balance": 0},
         "reject": {"keep": 0, "gate": 0, "balance": 0}}
    reason_counts = {k: 0 for k in REASON_KEYS}
    n_uncertain = 0
    for c in CLIPS:
        cid = c["clip_id"]
        d = DECISIONS.get(cid)
        if d == "uncertain":
            n_uncertain += 1
        elif d in ("keep", "reject"):
            m[d][code_state(cid)] += 1
            if d == "reject":
                r = REASON.get(cid, "")
                if r in reason_counts:
                    reason_counts[r] += 1
    decided = sum(m[a][b] for a in m for b in m[a])
    qd = m["keep"]["keep"] + m["keep"]["gate"] + m["reject"]["keep"] + m["reject"]["gate"]
    qagree = m["keep"]["keep"] + m["reject"]["gate"]
    return {
        "decided": decided,
        "matrix": m,
        "uncertain": n_uncertain,
        "reason_counts": reason_counts,
        "reason_labels": dict(REASONS),
        "quality_decided": qd,
        "quality_agreement": round(100.0 * qagree / qd, 1) if qd else 0.0,
        "found_missed_garbage": m["reject"]["keep"],   # tay REJECT, code GIỮ -> rác code bỏ sót
        "manual_keep_gate": m["keep"]["gate"],         # tay KEEP, code GATE-reject
        "balance_only": m["keep"]["balance"] + m["reject"]["balance"],
        "has_gate": len(CODE_GATE) > 0,
        "code_keep_total": len(CODE_KEEP),
        "code_gate_total": len(CODE_GATE),
    }


def counts():
    keep = reject = uncertain = 0
    for c in CLIPS:
        d = DECISIONS.get(c["clip_id"])
        if d == "keep":
            keep += 1
        elif d == "reject":
            reject += 1
        elif d == "uncertain":
            uncertain += 1
    return {"keep": keep, "reject": reject, "uncertain": uncertain,
            "decided": keep + reject + uncertain, "total": len(CLIPS)}


# ---------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass  # im lặng

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            self._html()
        elif self.path.startswith("/api/state"):
            self._state()
        elif self.path.startswith("/api/compare"):
            self._json(compare_summary())
        elif self.path.startswith("/video"):
            self._video(roi=False)
        elif self.path.startswith("/roi"):
            self._video(roi=True)
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path.startswith("/api/mark"):
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n) or b"{}")
            i = int(data.get("i", -1))
            dec = data.get("decision")
            if not (0 <= i < len(CLIPS)):
                self._json({"ok": False, "err": "index"}, 400)
                return
            try:
                intervals = normalize_bad_intervals(data.get("bad_intervals", []))
            except ValueError as error:
                self._json({"ok": False, "err": str(error)}, 400)
                return
            if dec == "reject" and not intervals:
                self._json({"ok": False, "err": "rubric v3: reject phải có ít nhất một interval lỗi"}, 400)
                return
            duration_ms = 0
            try:
                duration_ms = int(round(float(CLIPS[i].get("duration", 0)) * 1000))
            except (TypeError, ValueError):
                pass
            if duration_ms and any(item["end_ms"] > duration_ms + 200 for item in intervals):
                self._json({"ok": False, "err": "interval vượt quá thời lượng clip (+200 ms tolerance)"}, 400)
                return
            try:
                voiced_ms = int(round(float(CLIPS[i].get("voiced_ms", 0))))
            except (TypeError, ValueError):
                voiced_ms = 0
            if not voiced_ms:
                voiced_ms = duration_ms
            if dec == "reject" and not intervals_are_material(intervals, voiced_ms):
                self._json({"ok": False, "err": "đoạn lỗi chưa đạt 800 ms liên tục hoặc 500 ms + 20% voiced"}, 400)
                return
            cid = CLIPS[i]["clip_id"]
            with LOCK:
                if dec in DECISIONS_VALID:
                    DECISIONS[cid] = dec
                    BAD_INTERVALS[cid] = intervals if dec == "reject" else []
                    REASON[cid] = longest_interval_reason(intervals) if dec == "reject" else ""
                    TS[cid] = now()
                    REVIEWER[cid] = REVIEWER_ID
                    RUBRIC[cid] = RUBRIC_VERSION
                elif dec == "unset":
                    for d in (DECISIONS, REASON, BAD_INTERVALS, TS, REVIEWER, RUBRIC):
                        d.pop(cid, None)
                else:
                    self._json({"ok": False, "err": "decision"}, 400)
                    return
                save_decisions()
            self._json({"ok": True, "counts": counts(),
                        "reason": REASON.get(cid, ""),
                        "bad_intervals": BAD_INTERVALS.get(cid, [])})
        else:
            self.send_error(404)

    def _state(self):
        clips = []
        for i, c in enumerate(CLIPS):
            cid = c["clip_id"]
            item = {"i": i, "clip_id": cid,
                    "exists": os.path.isfile(c["_abspath"]),
                    "has_roi": bool(ROI_DIR) and os.path.isfile(roi_path(cid)),
                    "dec": DECISIONS.get(cid, ""),
                    "reason": REASON.get(cid, ""),
                    "bad_intervals": BAD_INTERVALS.get(cid, [])}
            for k in META_FIELDS:
                if k in c:
                    item[k] = c[k]
            clips.append(item)
        self._json({"clips": clips, "start": first_undecided(),
                    "counts": counts(), "has_compare": len(CODE_KEEP) > 0,
                    "has_gate": len(CODE_GATE) > 0,
                    "reasons": REASONS, "reviewer": REVIEWER_ID,
                    "rubric": RUBRIC_VERSION})

    def _video(self, roi=False):
        q = re.search(r"[?&]i=(\d+)", self.path)
        if not q:
            self.send_error(400); return
        i = int(q.group(1))
        if not (0 <= i < len(CLIPS)):
            self.send_error(404); return
        path = roi_path(CLIPS[i]["clip_id"]) if roi else CLIPS[i]["_abspath"]
        if not os.path.isfile(path):
            self.send_error(404); return
        fsize = os.path.getsize(path)
        rng = self.headers.get("Range")
        try:
            if rng:
                m = re.match(r"bytes=(\d+)-(\d*)", rng)
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else fsize - 1
                end = min(end, fsize - 1)
                length = end - start + 1
                self.send_response(206)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Range", f"bytes {start}-{end}/{fsize}")
                self.send_header("Content-Length", str(length))
                self.end_headers()
                with open(path, "rb") as f:
                    f.seek(start)
                    remaining = length
                    while remaining > 0:
                        chunk = f.read(min(65536, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            else:
                self.send_response(200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(fsize))
                self.end_headers()
                with open(path, "rb") as f:
                    while True:
                        chunk = f.read(65536)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _html(self):
        body = HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


HTML = r"""<!doctype html><html lang="vi"><head><meta charset="utf-8">
<title>Clip Review — lọc tay</title>
<style>
:root{--bg:#0f1115;--panel:#1a1d24;--mut:#8a93a3;--keep:#2e9e5b;--rej:#d6453d;--unc:#eda100;--acc:#2E5A9C}
*{box-sizing:border-box}body{margin:0;font-family:Segoe UI,Arial,sans-serif;background:var(--bg);color:#e6e9ef}
header{display:flex;align-items:center;gap:16px;padding:10px 16px;background:#12141a;border-bottom:1px solid #262a33;flex-wrap:wrap}
h1{font-size:15px;margin:0;color:#cfd6e4}
.bar{flex:1;height:8px;background:#262a33;border-radius:4px;overflow:hidden;min-width:160px}
.bar > i{display:block;height:100%;background:var(--acc)}
.pill{font-size:12px;color:var(--mut)}
.pill b{color:#e6e9ef}
main{display:flex;gap:16px;padding:16px;align-items:flex-start;flex-wrap:wrap}
#stage{flex:2;min-width:420px}
.vwrap{display:flex;gap:12px;align-items:flex-start}
.vcol{display:flex;flex-direction:column;gap:5px}
.vcol.orig{flex:3;min-width:0}
.vcol.roi{flex:1;min-width:200px}
.vlab{font-size:11px;color:var(--mut);text-transform:uppercase;letter-spacing:.5px}
.vlab b{color:#edc069}
video{width:100%;background:#000;border-radius:10px;outline:none}
#vid{max-height:52vh}
#roivid{image-rendering:pixelated;border:2px solid var(--unc)}
.noroi{font-size:12px;color:#e8b339;padding:10px;border:1px dashed #4a4326;border-radius:8px}
.side{flex:1;min-width:300px;background:var(--panel);border:1px solid #262a33;border-radius:10px;padding:14px}
.meta{font-size:13px;line-height:1.9}
.meta span{color:var(--mut)}.meta b{color:#e6e9ef}
.meta b.alert{color:#f0a07a}
.btns{display:flex;gap:10px;margin:12px 0 6px;flex-wrap:wrap}
button{cursor:pointer;border:0;border-radius:8px;padding:11px 14px;font-size:14px;font-weight:600;color:#fff}
.keep{background:var(--keep)}.unc{background:var(--unc);color:#2b2200}.nav{background:#39404d}.ghost{background:#2a2f3a;color:#cfd6e4}
.rgrid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:8px 0}
.rgrid button{background:#3a2422;border:1px solid var(--rej);color:#f2b3ae;font-size:13px;font-weight:500;text-align:left;padding:9px 11px}
.rgrid button:hover{background:var(--rej);color:#fff}
.rgrid button i{font-style:normal;color:#fff;background:var(--rej);border-radius:4px;padding:0 6px;margin-right:7px;font-weight:700}
.interval-tools{border:1px solid #39404d;border-radius:8px;padding:10px;margin-top:10px}
.interval-list{font-size:12px;color:#cfd6e4;margin:7px 0;line-height:1.7}
.interval-list button{padding:2px 7px;background:#4a2a28;font-size:11px;margin-left:7px}
.reject-final{background:var(--rej);width:100%;margin-top:7px}
.badge{display:inline-block;padding:3px 10px;border-radius:20px;font-size:12px;font-weight:700}
.b-keep{background:rgba(46,158,91,.18);color:#67d699;border:1px solid #2e9e5b}
.b-rej{background:rgba(214,69,61,.18);color:#f08a84;border:1px solid #d6453d}
.b-unc{background:rgba(237,161,0,.18);color:#edc069;border:1px solid #eda100}
.b-none{background:#262a33;color:var(--mut)}
.row{display:flex;gap:8px;align-items:center;margin:8px 0;flex-wrap:wrap}
input[type=number]{width:90px;background:#0f1115;border:1px solid #39404d;color:#fff;border-radius:6px;padding:6px}
.kbd{font-size:11px;color:var(--mut);margin-top:6px;line-height:1.8}
.kbd b{color:#cfd6e4;background:#262a33;padding:1px 6px;border-radius:4px}
#cmp{font-size:13px;line-height:1.7;margin-top:10px;border-top:1px solid #262a33;padding-top:10px}
table.cmp{border-collapse:collapse;margin:8px 0;font-size:12px;width:100%}
table.cmp th,table.cmp td{border:1px solid #2a2f3a;padding:6px 7px;text-align:center}
table.cmp th{color:#cfd6e4;background:#222733;font-weight:600}
table.cmp td.ok{color:#67d699;font-weight:700}
table.cmp td.bad{color:#f0a07a;font-weight:700}
table.cmp td.mut{color:#8a93a3}
.warn{color:#e8b339;font-size:12px;margin-top:8px}
label.cb{font-size:12px;color:var(--mut);display:flex;align-items:center;gap:6px}
.who{font-size:11px;color:var(--mut);border-top:1px solid #262a33;margin-top:10px;padding-top:8px}
</style></head><body>
<header>
  <h1>Clip Review — lọc tay</h1>
  <div class="bar"><i id="prog" style="width:0%"></i></div>
  <div class="pill">#<b id="pos">0</b>/<b id="tot">0</b></div>
  <div class="pill">Keep <b id="ck" style="color:#67d699">0</b></div>
  <div class="pill">Reject <b id="cr" style="color:#f08a84">0</b></div>
  <div class="pill">Chưa chắc <b id="cq" style="color:#edc069">0</b></div>
  <div class="pill">Còn lại <b id="cu">0</b></div>
</header>
<main>
  <div id="stage">
    <div class="vwrap">
      <div class="vcol orig">
        <div class="vlab">video gốc (đã tắt tiếng)</div>
        <video id="vid" controls autoplay loop muted></video>
      </div>
      <div class="vcol roi">
        <div class="vlab"><b>◀ ROI + TIẾNG</b> — thứ model thật sự nhận</div>
        <video id="roivid" controls autoplay loop></video>
        <div class="noroi" id="noroi" style="display:none">Chưa dựng preview ROI cho clip này.</div>
      </div>
    </div>
    <div class="btns">
      <button class="keep" onclick="mark('keep')">✓ Keep (K)</button>
      <button class="unc" onclick="mark('uncertain')">? Chưa chắc (C)</button>
      <button class="nav" onclick="go(-1)">← Trước</button>
      <button class="nav" onclick="go(1)">Sau →</button>
      <button class="ghost" onclick="mark('unset')">Bỏ đánh dấu (U)</button>
    </div>
    <div class="interval-tools">
      <div class="row">
        <button class="ghost" onclick="setBoundary('start')">Đặt đầu đoạn (A)</button>
        <button class="ghost" onclick="setBoundary('end')">Đặt cuối đoạn (B)</button>
        <span id="draft">chưa chọn đoạn</span>
      </div>
      <div class="rgrid" id="rgrid"></div>
      <div class="interval-list" id="intervals">Chưa có interval lỗi.</div>
      <button class="reject-final" onclick="mark('reject')">Reject với các interval đã đánh dấu</button>
    </div>
    <div class="kbd">
      <b>K</b> keep · <b>C</b> chưa chắc · <b>A</b>/<b>B</b> đặt biên ·
      <b>1</b>–<b id="nreason">7</b> thêm interval theo lý do ·
      <b>U</b> bỏ · <b>←</b>/<b>→</b> chuyển · <b>Space</b> phát lại<br>
      Reject BẮT BUỘC có interval đạt quy tắc thời lượng của rubric v3.
    </div>
    <div class="warn" id="missing" style="display:none">⚠ Không tìm thấy file trên đĩa.</div>
  </div>
  <div class="side">
    <div class="row">Trạng thái: <span id="badge" class="badge b-none">chưa đánh dấu</span></div>
    <div class="meta" id="meta"></div>
    <div class="row">
      <label class="cb"><input type="checkbox" id="auto" checked> tự chuyển sau khi đánh dấu</label>
    </div>
    <div class="row">
      Tới clip: <input type="number" id="jump" min="1"> <button class="ghost" onclick="jumpTo()">Đi</button>
      <button class="ghost" onclick="nextUndecided()">Chưa đánh dấu →</button>
    </div>
    <div class="row"><button class="ghost" onclick="doCompare()">So sánh với lọc code</button></div>
    <div id="cmp"></div>
    <div class="who">reviewer: <b id="who">—</b> · rubric: <b id="rub">—</b></div>
  </div>
</main>
<script>
let S={clips:[],i:0,has_compare:false,reasons:[],draftStart:null,draftEnd:null,intervals:[]};
function fmt(v,d=3){if(v===undefined||v===null||v==="")return "—";let n=Number(v);return isNaN(n)?v:n.toFixed(d);}
async function load(){
  let r=await fetch('/api/state');let j=await r.json();
  S.clips=j.clips;S.i=j.start;S.has_compare=j.has_compare;S.has_gate=j.has_gate;S.reasons=j.reasons;
  document.getElementById('tot').textContent=j.counts.total;
  document.getElementById('jump').max=j.counts.total;
  document.getElementById('who').textContent=j.reviewer||'(không đặt)';
  document.getElementById('rub').textContent=j.rubric;
  document.getElementById('rgrid').innerHTML=S.reasons.map((p,k)=>
    `<button onclick="addInterval('${p[0]}')"><i>${k+1}</i>Thêm đoạn: ${p[1]}</button>`).join('');
  document.getElementById('nreason').textContent=S.reasons.length;  // theo rubric, khong hard-code
  setCounts(j.counts);render();
}
function setCounts(c){
  document.getElementById('ck').textContent=c.keep;
  document.getElementById('cr').textContent=c.reject;
  document.getElementById('cq').textContent=c.uncertain;
  document.getElementById('cu').textContent=c.total-c.decided;
  document.getElementById('prog').style.width=(100*c.decided/c.total)+'%';
}
function render(){
  let c=S.clips[S.i];if(!c)return;
  S.intervals=Array.isArray(c.bad_intervals)?JSON.parse(JSON.stringify(c.bad_intervals)):[];
  S.draftStart=null;S.draftEnd=null;renderIntervals();
  document.getElementById('pos').textContent=S.i+1;
  document.getElementById('jump').value=S.i+1;
  let v=document.getElementById('vid');
  v.src='/video?i='+S.i+'&t='+Date.now();v.load();v.play().catch(()=>{});
  let rv=document.getElementById('roivid');
  rv.style.display=c.has_roi?'block':'none';
  document.getElementById('noroi').style.display=c.has_roi?'none':'block';
  if(c.has_roi){rv.src='/roi?i='+S.i+'&t='+Date.now();rv.load();rv.play().catch(()=>{});}
  else{rv.removeAttribute('src');}
  document.getElementById('missing').style.display=c.exists?'none':'block';
  // motion thấp -> tô cảnh báo, nhắc người review nhìn kỹ xem có phải ảnh tĩnh
  let mm=(c.motion_median===undefined||c.motion_median==='')?null:Number(c.motion_median);
  let mcls=(mm!==null&&mm<1.0)?' class="alert"':'';
  let m=`<div><span>clip_id:</span> <b>${c.clip_id}</b></div>`+
    `<div><span>tier:</span> <b>${c.tier||'—'}</b> · <span>video:</span> <b>${c.source_video||'—'}</b></div>`+
    `<div><span>duration:</span> <b>${fmt(c.duration,2)}s</b> · <span>snr:</span> <b>${fmt(c.snr,1)} dB</b></div>`+
    `<div><span>det_ratio:</span> <b>${fmt(c.det_ratio)}</b> · <span>face_area:</span> <b>${fmt(c.mean_face_area,4)}</b></div>`+
    `<div><span>consistency:</span> <b>${fmt(c.embed_consistency)}</b></div>`;
  if(c.motion_median!==undefined)
    m+=`<div><span>motion:</span> <b${mcls}>${fmt(c.motion_median)}</b>`+
       ` · <span>tĩnh:</span> <b${mcls}>${fmt(c.frac_near_static,2)}</b></div>`;
  if(c.n_faces_med!==undefined){
    // ratio cao = mat nhi to xap xi mat nhat -> luat 'mat to nhat' de chon nham nguoi
    let rr=Number(c.ratio_med), rcls=(rr>0.6)?' class="alert"':'';
    m+=`<div><span>số mặt:</span> <b${rcls}>${fmt(c.n_faces_med,0)}</b>`+
       ` · <span>mặt nhì/nhất:</span> <b${rcls}>${fmt(c.ratio_med,2)}</b></div>`;}
  if(c.sync_conf!==undefined)
    m+=`<div><span>sync_conf:</span> <b>${fmt(c.sync_conf,2)}</b></div>`;
  document.getElementById('meta').innerHTML=m;
  let b=document.getElementById('badge');let d=c.dec;
  let lab=Object.fromEntries(S.reasons);
  if(d==='keep'){b.className='badge b-keep';b.textContent='KEEP';}
  else if(d==='reject'){b.className='badge b-rej';b.textContent='REJECT — '+(lab[c.reason]||c.reason||'?');}
  else if(d==='uncertain'){b.className='badge b-unc';b.textContent='CHƯA CHẮC';}
  else{b.className='badge b-none';b.textContent='chưa đánh dấu';}
}
function setBoundary(which){
  let v=document.getElementById('roivid');
  if(!v.src||!isFinite(v.currentTime))v=document.getElementById('vid');
  let ms=Math.max(0,Math.round((v.currentTime||0)*1000));
  if(which==='start')S.draftStart=ms;else S.draftEnd=ms;
  renderIntervals();
}
function addInterval(reason){
  if(S.draftStart===null||S.draftEnd===null||S.draftEnd<=S.draftStart){
    alert('Hãy phát video, đặt đầu đoạn (A) rồi đặt cuối đoạn (B).');return;
  }
  S.intervals.push({start_ms:S.draftStart,end_ms:S.draftEnd,reason});
  S.intervals.sort((a,b)=>a.start_ms-b.start_ms);S.draftStart=null;S.draftEnd=null;
  renderIntervals();
}
function removeInterval(index){S.intervals.splice(index,1);renderIntervals();}
function renderIntervals(){
  let draft=(S.draftStart===null?'?':(S.draftStart/1000).toFixed(2))+'s → '+
            (S.draftEnd===null?'?':(S.draftEnd/1000).toFixed(2))+'s';
  document.getElementById('draft').textContent=draft;
  let labels=Object.fromEntries(S.reasons);
  document.getElementById('intervals').innerHTML=S.intervals.length?S.intervals.map((x,i)=>
    `${(x.start_ms/1000).toFixed(2)}–${(x.end_ms/1000).toFixed(2)}s · ${labels[x.reason]||x.reason}`+
    `<button onclick="removeInterval(${i})">xóa</button>`).join('<br>'):'Chưa có interval lỗi.';
}
async function mark(dec){
  let c=S.clips[S.i];if(!c)return;
  let r=await fetch('/api/mark',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({i:S.i,decision:dec,bad_intervals:dec==='reject'?S.intervals:[]})});
  let j=await r.json();
  if(!j.ok){alert(j.err||'lỗi');return;}
  if(j.counts)setCounts(j.counts);
  c.dec=(dec==='unset')?'':dec;
  c.bad_intervals=j.bad_intervals||[];
  c.reason=j.reason||'';
  if(dec!=='unset'&&document.getElementById('auto').checked){go(1);}else{render();}
}
function go(d){let n=S.i+d;if(n<0||n>=S.clips.length)return;S.i=n;render();}
function jumpTo(){let v=parseInt(document.getElementById('jump').value)-1;if(v>=0&&v<S.clips.length){S.i=v;render();}}
function nextUndecided(){for(let k=S.i+1;k<S.clips.length;k++){if(!S.clips[k].dec){S.i=k;render();return;}}
  for(let k=0;k<S.clips.length;k++){if(!S.clips[k].dec){S.i=k;render();return;}}}
async function doCompare(){
  let el=document.getElementById('cmp');
  if(!S.has_compare){el.innerHTML='<i>Không có file lọc code để so sánh.</i>';return;}
  let j=await(await fetch('/api/compare')).json();
  let m=j.matrix;
  let tbl=`<table class="cmp">`+
    `<tr><th></th><th>code KEEP</th><th>code GATE-REJECT</th><th>code BALANCE-DROP</th></tr>`+
    `<tr><th>Tay KEEP</th><td class="ok">${m.keep.keep}</td><td class="bad">${m.keep.gate}</td><td class="mut">${m.keep.balance}</td></tr>`+
    `<tr><th>Tay REJECT</th><td class="bad">${m.reject.keep}</td><td class="ok">${m.reject.gate}</td><td class="mut">${m.reject.balance}</td></tr>`+
    `</table>`;
  let rc=Object.entries(j.reason_counts).filter(([k,v])=>v>0)
        .sort((a,b)=>b[1]-a[1])
        .map(([k,v])=>`<div><span style="color:var(--mut)">${j.reason_labels[k]}:</span> <b>${v}</b></div>`).join('');
  el.innerHTML=`<b>So sánh tay vs code</b> (trên ${j.decided} clip đã phán keep/reject)`+tbl+
   `Đồng thuận trục RÁC: <b style="color:#67d699">${j.quality_agreement}%</b> `+
   `<span style="color:var(--mut)">(trên ${j.quality_decided} clip code có phán chất lượng)</span><br>`+
   `Rác code BỎ SÓT (tay reject / code keep): <b style="color:#f0a07a">${j.found_missed_garbage}</b><br>`+
   `Tay giữ / code gate-reject: <b style="color:#f0a07a">${j.manual_keep_gate}</b><br>`+
   `Clip tốt code bỏ do CÂN BẰNG (không tính bất đồng): <b style="color:#8a93a3">${j.balance_only}</b><br>`+
   `Chưa chắc (không tính vào ma trận): <b style="color:#edc069">${j.uncertain}</b>`+
   (rc?`<div style="margin-top:8px"><b>Lý do reject</b>${rc}</div>`:'')+
   (j.has_gate?'':`<div class="warn">Chưa có file gate-rejects → cột GATE trống. Chạy lại với --rejects.</div>`);
}
document.addEventListener('keydown',e=>{
  if(e.target.tagName==='INPUT')return;
  if(e.key>='1'&&e.key<='9'){let k=parseInt(e.key)-1;if(k<S.reasons.length)addInterval(S.reasons[k][0]);return;}
  if(e.key==='a'||e.key==='A'){setBoundary('start');return;}
  if(e.key==='b'||e.key==='B'){setBoundary('end');return;}
  if(e.key==='k'||e.key==='K')mark('keep');
  else if(e.key==='c'||e.key==='C')mark('uncertain');
  else if(e.key==='u'||e.key==='U')mark('unset');
  else if(e.key==='ArrowLeft')go(-1);
  else if(e.key==='ArrowRight')go(1);
  else if(e.key===' '){e.preventDefault();
    for(const id of ['vid','roivid']){let v=document.getElementById(id);
      if(v.src){v.currentTime=0;v.play().catch(()=>{});}}}
});
load();
</script></body></html>"""


def stratified_sample(rows, n, col, seed):
    """Chia rows thành 3 tầng đều nhau theo `col` rồi lấy n/3 mỗi tầng.

    Dùng cho mẻ hiệu chuẩn: ép người review gặp cả clip tĩnh lẫn clip động, thay vì
    mẫu ngẫu nhiên thuần (dễ toàn clip bình thường -> không biết rubric có reject được không).
    """
    import random
    rng = random.Random(seed)
    typed = []
    for i, r in enumerate(rows):
        try:
            typed.append((float(r[col]), i))
        except (KeyError, TypeError, ValueError):
            pass
    if len(typed) < 3:
        raise ValueError(f"Cột --stratify '{col}' không có đủ giá trị số")
    typed.sort()
    third = len(typed) // 3
    bands = [typed[:third], typed[third:2 * third], typed[2 * third:]]
    per = max(1, n // 3)
    picked = []
    for band in bands:
        take = rng.sample(band, min(per, len(band)))
        picked.extend(idx for _, idx in take)
    picked.sort()
    labels = ["thấp", "giữa", "cao"]
    print(f"Phân tầng theo '{col}' ({len(typed)} clip có giá trị):")
    for name, band in zip(labels, bands):
        vals = [v for v, _ in band]
        print(f"  tầng {name:5s}: {col} {vals[0]:.3f} – {vals[-1]:.3f}  "
              f"(lấy {min(per, len(band))}/{len(band)})")
    return [rows[i] for i in picked]


def safe_reviewer_name(value):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    if not value:
        raise ValueError("reviewer ID rỗng hoặc không hợp lệ")
    return value


def default_out(csv_path, reviewer):
    stem = os.path.splitext(os.path.basename(csv_path))[0]
    return os.path.join("data", "02_curate", "manual",
                        f"manual_{stem}_{RUBRIC_VERSION}_{safe_reviewer_name(reviewer)}.csv")


def main():
    global OUT, PATH_COL, REVIEWER_ID
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/02_curate/manifests/all_clean_review.csv",
                    help="manifest cần review (bản gộp: + motion + face ambiguity)")
    ap.add_argument("--roi_dir", default="data/02_curate/roi_preview",
                    help="thư mục preview ROI+tiếng (build_roi_preview.py); '' = tắt")
    ap.add_argument("--media_root", default="",
                    help="thư mục chứa clip gốc TRÊN MÁY NÀY — quét đệ quy, tra theo "
                         "tên file <clip_id>.mp4 thay cho file_path trong manifest. "
                         "BẮT BUỘC khi review trên máy khác máy dựng manifest.")
    ap.add_argument("--order", choices=["manifest", "diverse"], default="diverse",
                    help="diverse = vòng tròn qua từng source_video, dừng lúc nào cũng "
                         "có tập đa dạng nguồn")
    ap.add_argument("--exclude_channel", default="",
                    help="bỏ qua kênh (ngăn cách bằng ';'). VD kênh phóng sự lời bình.")
    ap.add_argument("--out", default=None,
                    help="file ghi quyết định (mặc định tự sinh theo --csv + rubric)")
    ap.add_argument("--compare", default="data/02_curate/manifests/all_clean.csv",
                    help="file code GIỮ để đối chiếu (để '' nếu không cần)")
    ap.add_argument("--rejects", default="data/02_curate/manifests/all_clean_rejects.csv",
                    help="file code GATE-REJECT (rác) để đối chiếu 3 trạng thái")
    ap.add_argument("--sample", type=int, default=0,
                    help="chỉ review N clip ngẫu nhiên (0 = toàn bộ)")
    ap.add_argument("--stratify", default="",
                    help="cột chia 3 tầng cho --sample (vd motion_median)")
    ap.add_argument("--seed", type=int, default=42, help="seed cho --sample (để lặp/resume)")
    ap.add_argument("--reviewer", default=os.environ.get("USERNAME", ""),
                    help="mã người review, ghi vào CSV")
    ap.add_argument("--col", default="file_path", help="cột chứa đường dẫn .mp4")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    global ROI_DIR
    PATH_COL = args.col
    REVIEWER_ID = args.reviewer
    ROI_DIR = os.path.abspath(args.roi_dir) if args.roi_dir else ""
    if not REVIEWER_ID:
        print("[LỖI] Bắt buộc truyền --reviewer để không trộn kết quả nhiều người.")
        sys.exit(1)
    OUT = os.path.abspath(args.out or default_out(args.csv, REVIEWER_ID))

    if not os.path.isfile(args.csv):
        print(f"[LỖI] Không thấy manifest: {args.csv}")
        sys.exit(1)
    media_index = None
    if args.media_root:
        if not os.path.isdir(args.media_root):
            print(f"[LỖI] --media_root không phải thư mục: {args.media_root}")
            sys.exit(1)
        media_index, dup = index_media(args.media_root)
        print(f"Media    : quét {args.media_root} -> {len(media_index)} file .mp4"
              + (f" ({dup} tên trùng bị bỏ qua)" if dup else ""))

    global CLIPS, ALL_ROWS
    ALL_ROWS = load_clips(args.csv, args.col, media_index)
    assigned = {r.get("assigned_reviewer", "").strip() for r in ALL_ROWS
                if r.get("assigned_reviewer", "").strip()}
    if assigned and assigned != {REVIEWER_ID}:
        print(f"[LỖI] Assignment dành cho {sorted(assigned)}, "
              f"không phải reviewer '{REVIEWER_ID}'.")
        sys.exit(1)
    for i, r in enumerate(ALL_ROWS):
        FILEPATH_BY_ID[r["clip_id"]] = r.get(PATH_COL, "")
        ORDER_BY_ID[r["clip_id"]] = i

    # Lọc kênh TRƯỚC khi lấy mẫu: làm ngược lại thì --sample N kèm --exclude_channel
    # sẽ ra ÍT hơn N clip và phân tầng bị lệch theo phần bị bỏ.
    pool = ALL_ROWS
    if args.exclude_channel:
        drop = {c.strip() for c in args.exclude_channel.split(";") if c.strip()}
        unknown = drop - {r.get("channel", "") for r in pool}
        if unknown:
            print(f"[CẢNH BÁO] không có kênh {sorted(unknown)} trong manifest")
        before = len(pool)
        pool = [c for c in pool if c.get("channel", "") not in drop]
        print(f"Bỏ kênh  : {sorted(drop)} -> loại {before - len(pool)} clip")

    if args.sample and args.sample < len(pool):
        if args.stratify:
            CLIPS = stratified_sample(pool, args.sample, args.stratify, args.seed)
        else:
            import random
            idx = sorted(random.Random(args.seed).sample(range(len(pool)), args.sample))
            CLIPS = [pool[i] for i in idx]
    else:
        CLIPS = pool

    if args.order == "diverse":
        CLIPS = diverse_order(CLIPS, seed=args.seed)

    load_decisions(OUT)
    load_code_states(args.compare, args.rejects)

    miss = sum(1 for c in CLIPS if not os.path.isfile(c["_abspath"]))
    sampled = len(CLIPS) < len(ALL_ROWS)
    decided_here = sum(1 for c in CLIPS if c["clip_id"] in DECISIONS)
    print("=" * 60)
    print(f"Manifest : {args.csv}  ({len(ALL_ROWS)} clip tổng)")
    if sampled:
        how = f"phân tầng theo {args.stratify}" if args.stratify else "ngẫu nhiên"
        print(f"Sample   : {len(CLIPS)} clip {how} (seed={args.seed}), {miss} thiếu file")
    else:
        print(f"Review   : toàn bộ {len(CLIPS)} clip, {miss} thiếu file")
    print(f"Output   : {OUT}")
    n_roi = sum(1 for c in CLIPS if os.path.isfile(roi_path(c["clip_id"]))) if ROI_DIR else 0
    print(f"ROI      : {n_roi}/{len(CLIPS)} clip có preview"
          + ("" if not ROI_DIR else f"  ({ROI_DIR})"))
    if ROI_DIR and n_roi < len(CLIPS):
        print("           -> chạy src/tools/review/build_roi_preview.py để dựng phần còn thiếu")
    if args.order == "diverse":
        vtot = len({c.get("source_video", "") for c in CLIPS})
        v300 = len({c.get("source_video", "") for c in CLIPS[:300]})
        print(f"Thứ tự   : vòng tròn theo video (seed={args.seed}) — {vtot} video, "
              f"300 clip đầu đã trải {v300}")
    if miss:
        # Thiếu gần hết = gần như chắc chắn sai gốc đường dẫn, không phải thiếu vài file.
        print(f"[CẢNH BÁO] {miss}/{len(CLIPS)} clip không thấy trên đĩa.")
        if not args.media_root:
            print("           Manifest ghi đường dẫn tuyệt đối của máy dựng manifest. "
                  "Nếu đang review trên máy khác, truyền --media_root <thư mục chứa clip>.")
        elif miss > len(CLIPS) // 2:
            print(f"           --media_root='{args.media_root}' có thể trỏ sai chỗ, "
                  "hoặc chưa tải đủ media của assignment này.")
    print(f"Reviewer : {REVIEWER_ID or '(chưa đặt)'} | rubric {RUBRIC_VERSION}")
    print(f"Compare  : code KEEP {len(CODE_KEEP)} | code GATE-REJECT {len(CODE_GATE)}"
          + ("" if CODE_GATE else "  (chưa có --rejects -> cột GATE trống)"))
    print(f"Đã đánh dấu (trong tập này): {decided_here}/{len(CLIPS)} "
          f"| tiếp tục từ clip #{first_undecided()+1}")
    url = f"http://127.0.0.1:{args.port}"
    print(f"Mở: {url}   (Ctrl+C để dừng)")
    print("=" * 60)

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nĐã dừng. Quyết định lưu tại:", OUT)


if __name__ == "__main__":
    main()
