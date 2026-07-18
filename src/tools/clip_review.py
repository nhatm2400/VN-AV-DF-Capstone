"""
clip_review.py — Công cụ LỌC TAY clip (manual QA) chạy trong trình duyệt.

Mục đích: review TOÀN BỘ clip từ đầu bằng mắt người, đánh KEEP/REJECT, để so sánh
"lọc tay" với "lọc bằng code" (04_curate -> all_clean.csv). Kết quả ghi sang FOLDER RIÊNG
nên không đụng tới output của pipeline tự động.

Không cần cài gì thêm — chỉ dùng thư viện chuẩn Python + trình duyệt.

CÁCH DÙNG (chạy từ thư mục gốc dự án):
    python src/tools/clip_review.py
    # rồi mở http://127.0.0.1:8000  (tự mở trình duyệt)

Tùy chọn:
    --csv      manifest đầy đủ cần review   (mặc định data/02_curate/tier1_scored_all.csv)
    --out      file ghi quyết định          (mặc định data/02_curate/manual/manual_review.csv)
    --compare  file code GIỮ để đối chiếu    (mặc định data/02_curate/all_clean.csv)
    --rejects  file code GATE-REJECT          (mặc định data/02_curate/all_clean_rejects.csv)
    --sample   chỉ review N clip NGẪU NHIÊN rải đều (0 = toàn bộ)
    --seed     seed cho --sample (để lặp lại cùng mẫu / resume) (mặc định 42)
    --col      cột chứa đường dẫn .mp4       (mặc định file_path)
    --port     cổng web                      (mặc định 8000)
    --no-browser   không tự mở trình duyệt

SO SÁNH 3 TRẠNG THÁI: mỗi clip code có 1 trong 3 trạng thái — KEEP (trong all_clean),
GATE-REJECT (trong all_clean_rejects, = rác), BALANCE-DROP (qua gate nhưng bị cắt do cân
bằng cap/speaker, = clip TỐT bị bỏ vì thừa). Đồng thuận "trục rác" chỉ tính trên clip code
có phán chất lượng (KEEP hoặc GATE-REJECT); balance-drop KHÔNG tính là bất đồng.

PHÍM TẮT:  K = Keep · X = Reject · U = bỏ đánh dấu · ←/→ = clip trước/sau · Space = phát lại

GHI CHÚ CHỐNG LEAKAGE: khi lọc tập REAL, chỉ loại "rác rõ ràng" (không mặt / che miệng /
nhiều người / sai người). KHÔNG loại theo "khớp môi tốt/xấu" và tiêu chí phải áp y hệt cho
tập fake sau này — nếu không sẽ rò rỉ nhãn.
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

# ----- global state -----
CLIPS = []            # list of dict cần review (toàn bộ HOẶC mẫu sample) — index theo 'i'
ALL_ROWS = []         # toàn bộ manifest (để tra file_path/thứ tự cho mọi clip)
FILEPATH_BY_ID = {}   # clip_id -> file_path (từ toàn bộ manifest)
ORDER_BY_ID = {}      # clip_id -> thứ tự gốc trong manifest (để ghi CSV ổn định)
DECISIONS = {}        # clip_id -> "keep"/"reject"  (giữ MỌI quyết định, kể cả ngoài sample)
TS = {}               # clip_id -> timestamp string
CODE_KEEP = set()     # clip_id code GIỮ (all_clean.csv)
CODE_GATE = set()     # clip_id code GATE-REJECT (all_clean_rejects.csv) = rác
OUT = ""
PATH_COL = "file_path"
LOCK = threading.Lock()

META_FIELDS = ["tier", "source_video", "duration", "snr", "det_ratio",
               "mean_face_area", "embed_consistency", "face_ratio", "speech_ratio"]


def load_clips(csv_path, path_col):
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            fp = (r.get(path_col) or "").strip()
            r["_abspath"] = fp
            rows.append(r)
    return rows


def load_decisions(out_path):
    if not os.path.isfile(out_path):
        return
    with open(out_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            cid = r.get("clip_id")
            dec = r.get("decision")
            if cid and dec in ("keep", "reject"):
                DECISIONS[cid] = dec
                TS[cid] = r.get("ts", "")


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
        w.writerow(["clip_id", "file_path", "decision", "ts"])
        for cid, d in items:
            if d:
                w.writerow([cid, FILEPATH_BY_ID.get(cid, ""), d, TS.get(cid, "")])
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
    """
    m = {"keep": {"keep": 0, "gate": 0, "balance": 0},
         "reject": {"keep": 0, "gate": 0, "balance": 0}}
    for c in CLIPS:
        d = DECISIONS.get(c["clip_id"])
        if d in ("keep", "reject"):
            m[d][code_state(c["clip_id"])] += 1
    decided = sum(m[a][b] for a in m for b in m[a])
    qd = m["keep"]["keep"] + m["keep"]["gate"] + m["reject"]["keep"] + m["reject"]["gate"]
    qagree = m["keep"]["keep"] + m["reject"]["gate"]
    return {
        "decided": decided,
        "matrix": m,
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
    keep = reject = 0
    for c in CLIPS:
        d = DECISIONS.get(c["clip_id"])
        if d == "keep":
            keep += 1
        elif d == "reject":
            reject += 1
    return {"keep": keep, "reject": reject, "decided": keep + reject, "total": len(CLIPS)}


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
            self._video()
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path.startswith("/api/mark"):
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n) or b"{}")
            i = int(data.get("i", -1))
            dec = data.get("decision")
            if 0 <= i < len(CLIPS):
                cid = CLIPS[i]["clip_id"]
                with LOCK:
                    if dec in ("keep", "reject"):
                        DECISIONS[cid] = dec
                        TS[cid] = now()
                    elif dec == "unset":
                        DECISIONS.pop(cid, None)
                        TS.pop(cid, None)
                    save_decisions()
                self._json({"ok": True, "counts": counts()})
            else:
                self._json({"ok": False, "err": "index"}, 400)
        else:
            self.send_error(404)

    def _state(self):
        clips = []
        for i, c in enumerate(CLIPS):
            item = {"i": i, "clip_id": c["clip_id"],
                    "exists": os.path.isfile(c["_abspath"]),
                    "dec": DECISIONS.get(c["clip_id"], "")}
            for k in META_FIELDS:
                if k in c:
                    item[k] = c[k]
            clips.append(item)
        self._json({"clips": clips, "start": first_undecided(),
                    "counts": counts(), "has_compare": len(CODE_KEEP) > 0,
                    "has_gate": len(CODE_GATE) > 0})

    def _video(self):
        q = re.search(r"[?&]i=(\d+)", self.path)
        if not q:
            self.send_error(400); return
        i = int(q.group(1))
        if not (0 <= i < len(CLIPS)):
            self.send_error(404); return
        path = CLIPS[i]["_abspath"]
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
:root{--bg:#0f1115;--panel:#1a1d24;--mut:#8a93a3;--keep:#2e9e5b;--rej:#d6453d;--acc:#2E5A9C}
*{box-sizing:border-box}body{margin:0;font-family:Segoe UI,Arial,sans-serif;background:var(--bg);color:#e6e9ef}
header{display:flex;align-items:center;gap:16px;padding:10px 16px;background:#12141a;border-bottom:1px solid #262a33;flex-wrap:wrap}
h1{font-size:15px;margin:0;color:#cfd6e4}
.bar{flex:1;height:8px;background:#262a33;border-radius:4px;overflow:hidden;min-width:160px}
.bar > i{display:block;height:100%;background:var(--acc)}
.pill{font-size:12px;color:var(--mut)}
.pill b{color:#e6e9ef}
main{display:flex;gap:16px;padding:16px;align-items:flex-start;flex-wrap:wrap}
#stage{flex:2;min-width:420px}
video{width:100%;max-height:62vh;background:#000;border-radius:10px;outline:none}
.side{flex:1;min-width:280px;background:var(--panel);border:1px solid #262a33;border-radius:10px;padding:14px}
.meta{font-size:13px;line-height:1.9}
.meta span{color:var(--mut)}.meta b{color:#e6e9ef}
.btns{display:flex;gap:10px;margin:14px 0}
button{cursor:pointer;border:0;border-radius:8px;padding:12px 14px;font-size:14px;font-weight:600;color:#fff}
.keep{background:var(--keep)}.rej{background:var(--rej)}.nav{background:#39404d}.ghost{background:#2a2f3a;color:#cfd6e4}
.badge{display:inline-block;padding:3px 10px;border-radius:20px;font-size:12px;font-weight:700}
.b-keep{background:rgba(46,158,91,.18);color:#67d699;border:1px solid #2e9e5b}
.b-rej{background:rgba(214,69,61,.18);color:#f08a84;border:1px solid #d6453d}
.b-none{background:#262a33;color:var(--mut)}
.row{display:flex;gap:8px;align-items:center;margin:8px 0;flex-wrap:wrap}
input[type=number]{width:90px;background:#0f1115;border:1px solid #39404d;color:#fff;border-radius:6px;padding:6px}
.kbd{font-size:11px;color:var(--mut);margin-top:6px}
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
</style></head><body>
<header>
  <h1>Clip Review — lọc tay</h1>
  <div class="bar"><i id="prog" style="width:0%"></i></div>
  <div class="pill">#<b id="pos">0</b>/<b id="tot">0</b></div>
  <div class="pill">Keep <b id="ck" style="color:#67d699">0</b></div>
  <div class="pill">Reject <b id="cr" style="color:#f08a84">0</b></div>
  <div class="pill">Còn lại <b id="cu">0</b></div>
</header>
<main>
  <div id="stage">
    <video id="vid" controls autoplay loop></video>
    <div class="btns">
      <button class="rej" onclick="mark('reject')">✗ Reject (X)</button>
      <button class="keep" onclick="mark('keep')">✓ Keep (K)</button>
      <button class="nav" onclick="go(-1)">← Trước</button>
      <button class="nav" onclick="go(1)">Sau →</button>
      <button class="ghost" onclick="mark('unset')">Bỏ đánh dấu (U)</button>
    </div>
    <div class="kbd"><b>K</b> keep · <b>X</b> reject · <b>U</b> bỏ · <b>←</b>/<b>→</b> chuyển · <b>Space</b> phát lại</div>
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
      <button class="ghost" onclick="nextUndecided()">Clip chưa đánh dấu →</button>
    </div>
    <div class="row"><button class="ghost" onclick="doCompare()">So sánh với lọc code</button></div>
    <div id="cmp"></div>
  </div>
</main>
<script>
let S={clips:[],i:0,has_compare:false};
function fmt(v,d=3){if(v===undefined||v==="")return "—";let n=Number(v);return isNaN(n)?v:n.toFixed(d);}
async function load(){
  let r=await fetch('/api/state');let j=await r.json();
  S.clips=j.clips;S.i=j.start;S.has_compare=j.has_compare;S.has_gate=j.has_gate;
  document.getElementById('tot').textContent=j.counts.total;
  document.getElementById('jump').max=j.counts.total;
  setCounts(j.counts);render();
}
function setCounts(c){
  document.getElementById('ck').textContent=c.keep;
  document.getElementById('cr').textContent=c.reject;
  document.getElementById('cu').textContent=c.total-c.decided;
  document.getElementById('prog').style.width=(100*c.decided/c.total)+'%';
}
function render(){
  let c=S.clips[S.i];if(!c)return;
  document.getElementById('pos').textContent=S.i+1;
  document.getElementById('jump').value=S.i+1;
  let v=document.getElementById('vid');
  v.src='/video?i='+S.i+'&t='+Date.now();v.load();v.play().catch(()=>{});
  document.getElementById('missing').style.display=c.exists?'none':'block';
  let m=`<div><span>clip_id:</span> <b>${c.clip_id}</b></div>`+
    `<div><span>tier:</span> <b>${c.tier||'—'}</b> · <span>video:</span> <b>${c.source_video||'—'}</b></div>`+
    `<div><span>duration:</span> <b>${fmt(c.duration,2)}s</b> · <span>snr:</span> <b>${fmt(c.snr,1)} dB</b></div>`+
    `<div><span>det_ratio:</span> <b>${fmt(c.det_ratio)}</b> · <span>face_area:</span> <b>${fmt(c.mean_face_area,4)}</b></div>`+
    `<div><span>consistency:</span> <b>${fmt(c.embed_consistency)}</b></div>`;
  document.getElementById('meta').innerHTML=m;
  let b=document.getElementById('badge');let d=c.dec;
  if(d==='keep'){b.className='badge b-keep';b.textContent='KEEP';}
  else if(d==='reject'){b.className='badge b-rej';b.textContent='REJECT';}
  else{b.className='badge b-none';b.textContent='chưa đánh dấu';}
}
async function mark(dec){
  let c=S.clips[S.i];if(!c)return;
  let r=await fetch('/api/mark',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({i:S.i,decision:dec})});
  let j=await r.json();if(j.counts)setCounts(j.counts);
  c.dec=(dec==='unset')?'':dec;
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
  el.innerHTML=`<b>So sánh tay vs code</b> (trên ${j.decided} clip đã đánh dấu)`+tbl+
   `Đồng thuận trục RÁC: <b style="color:#67d699">${j.quality_agreement}%</b> `+
   `<span style="color:var(--mut)">(trên ${j.quality_decided} clip code có phán chất lượng)</span><br>`+
   `Rác code BỎ SÓT (tay reject / code keep): <b style="color:#f0a07a">${j.found_missed_garbage}</b><br>`+
   `Tay giữ / code gate-reject: <b style="color:#f0a07a">${j.manual_keep_gate}</b><br>`+
   `Clip tốt code bỏ do CÂN BẰNG (không tính bất đồng): <b style="color:#8a93a3">${j.balance_only}</b>`+
   (j.has_gate?'':`<div class="warn">Chưa có file gate-rejects → cột GATE trống. Chạy lại với --rejects.</div>`);
}
document.addEventListener('keydown',e=>{
  if(e.target.tagName==='INPUT')return;
  if(e.key==='k'||e.key==='K')mark('keep');
  else if(e.key==='x'||e.key==='X')mark('reject');
  else if(e.key==='u'||e.key==='U')mark('unset');
  else if(e.key==='ArrowLeft')go(-1);
  else if(e.key==='ArrowRight')go(1);
  else if(e.key===' '){e.preventDefault();let v=document.getElementById('vid');v.currentTime=0;v.play();}
});
load();
</script></body></html>"""


def main():
    global OUT, PATH_COL
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/02_curate/tier1_scored_all.csv",
                    help="manifest đầy đủ cần review")
    ap.add_argument("--out", default="data/02_curate/manual/manual_review.csv",
                    help="file ghi quyết định (folder riêng)")
    ap.add_argument("--compare", default="data/02_curate/all_clean.csv",
                    help="file code GIỮ để đối chiếu (để '' nếu không cần)")
    ap.add_argument("--rejects", default="data/02_curate/all_clean_rejects.csv",
                    help="file code GATE-REJECT (rác) để đối chiếu 3 trạng thái")
    ap.add_argument("--sample", type=int, default=0,
                    help="chỉ review N clip ngẫu nhiên rải đều (0 = toàn bộ)")
    ap.add_argument("--seed", type=int, default=42, help="seed cho --sample (để lặp/resume)")
    ap.add_argument("--col", default="file_path", help="cột chứa đường dẫn .mp4")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    PATH_COL = args.col
    OUT = os.path.abspath(args.out)

    if not os.path.isfile(args.csv):
        print(f"[LỖI] Không thấy manifest: {args.csv}")
        sys.exit(1)

    global CLIPS, ALL_ROWS
    ALL_ROWS = load_clips(args.csv, args.col)
    for i, r in enumerate(ALL_ROWS):
        FILEPATH_BY_ID[r["clip_id"]] = r.get(PATH_COL, "")
        ORDER_BY_ID[r["clip_id"]] = i

    if args.sample and args.sample < len(ALL_ROWS):
        import random
        idx = sorted(random.Random(args.seed).sample(range(len(ALL_ROWS)), args.sample))
        CLIPS = [ALL_ROWS[i] for i in idx]
    else:
        CLIPS = ALL_ROWS

    load_decisions(OUT)
    load_code_states(args.compare, args.rejects)

    miss = sum(1 for c in CLIPS if not os.path.isfile(c["_abspath"]))
    sampled = len(CLIPS) < len(ALL_ROWS)
    decided_here = sum(1 for c in CLIPS if c["clip_id"] in DECISIONS)
    print("=" * 60)
    print(f"Manifest : {args.csv}  ({len(ALL_ROWS)} clip tổng)")
    if sampled:
        print(f"Sample   : {len(CLIPS)} clip ngẫu nhiên (seed={args.seed}), {miss} thiếu file")
    else:
        print(f"Review   : toàn bộ {len(CLIPS)} clip, {miss} thiếu file")
    print(f"Output   : {OUT}")
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
