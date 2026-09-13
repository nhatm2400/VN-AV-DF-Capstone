"""Review-only mouth crops. This is NOT validated AV-HuBERT preprocessing."""
import cv2
import numpy as np

def _crop_mouth(frame, box, size):
    """Crop nửa dưới bbox (vùng miệng), nới ngang 10%, -> grayscale size×size. None nếu suy biến."""
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    mx1 = int(max(0, x1 - 0.10 * bw))
    mx2 = int(min(frame.shape[1], x2 + 0.10 * bw))
    my1 = int(y1 + 0.55 * bh)
    my2 = int(min(frame.shape[0], y2 + 0.10 * bh))
    if mx2 <= mx1 or my2 <= my1:
        return None
    g = cv2.cvtColor(frame[my1:my2, mx1:mx2], cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (size, size))

def detect_and_crop(model, mp4, target_fps, size, detect_every, conf):
    """
    MỘT lần decode: detect (carry-forward + backward-fill) + crop mouth. Trả
    (boxes, mouth[T,size,size]); (None, None) nếu cả clip không có mặt.
    Mỗi output-frame LUÔN có 1 box -> chuỗi ROI không co/lệch với audio. `boxes`
    trả ra để anon ghép cặp tái dùng (crop trên video anon, không detect mặt mờ).
    """
    cap = cv2.VideoCapture(mp4)
    if not cap.isOpened():
        return None, None
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    boxes, rois, pending = [], [], []      # pending: (idx, frame) chưa có box -> backfill sau
    box, fi, oi, det_i = None, 0, 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        while fi >= round(oi * src_fps / target_fps):      # emit output-frame (target_fps thật)
            if det_i % max(1, detect_every) == 0:
                res = model.predict(frame, verbose=False, conf=conf)[0]
                if res.boxes is not None and len(res.boxes) > 0:
                    xyxy = res.boxes.xyxy.cpu().numpy()
                    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
                    box = xyxy[int(areas.argmax())][:4]    # fail -> giữ box cũ (carry-forward)
            det_i += 1
            boxes.append(box)
            if box is None:
                pending.append((oi, frame.copy())); rois.append(None)   # backfill khi có box đầu
            else:
                rois.append(_crop_mouth(frame, box, size))
            oi += 1
        fi += 1
    cap.release()
    first = next((b for b in boxes if b is not None), None)
    if first is None:
        return None, None                                  # cả clip không bắt được mặt
    for k, fr in pending:                                  # backward-fill các None ở đầu
        boxes[k] = first
        rois[k] = _crop_mouth(fr, first, size)
    if not rois or any(r is None for r in rois):
        return boxes, None
    return boxes, np.stack(rois).astype(np.uint8)
