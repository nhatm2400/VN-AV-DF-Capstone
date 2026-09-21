"""Opt-in preview adjustments; models and full-resolution source frames stay intact."""
import importlib
import inspect
import time

import cv2
import numpy as np


DETECTION_WIDTH = 960


def detect_scaled(images, detect, max_width=DETECTION_WIDTH):
    height, width = images.shape[1:3]
    scale = min(1., max_width / max(height, width))
    small_width, small_height = max(1, round(width * scale)), max(1, round(height * scale))
    small = (np.stack([cv2.resize(im, (small_width, small_height), interpolation=cv2.INTER_AREA)
                       for im in images]) if scale < 1 else images)
    boxes = detect(small)
    result = []
    for box in boxes:
        if box is None:
            result.append(None)
        else:
            x1, y1, x2, y2 = box
            result.append((max(0, round(x1 * width / small_width)),
                           max(0, round(y1 * height / small_height)),
                           min(width, round(x2 * width / small_width)),
                           min(height, round(y2 * height / small_height))))
    return result


def install_detector():
    """Upstream keeps inline preprocessing; resize only the detector's input."""
    import torch
    detector = importlib.import_module('face_detection.detection.sfd.sfd_detector')
    for name in ('detect', 'batch_detect'):
        function = getattr(detector, name)
        source = inspect.getsource(function)
        old = 'torch.backends.cudnn.benchmark = True'
        if source.count(old) != 1:
            raise RuntimeError('S3FD changed; review the preview adapter')
        namespace = dict(function.__globals__)
        exec(compile(source.replace(old, 'torch.backends.cudnn.benchmark = False'),
                     '<preview_s3fd>', 'exec'), namespace)
        setattr(detector, name, namespace[name])
    face_alignment = importlib.import_module('face_detection').FaceAlignment
    original = face_alignment.get_detections_for_batch

    def scaled(self, images):
        torch.backends.cudnn.benchmark = False
        return detect_scaled(images, lambda small: original(self, small))

    face_alignment.get_detections_for_batch = scaled
    torch.backends.cudnn.benchmark = False


def lower_face_mask(height, width):
    """Soft ellipse: original eyes/nose bridge and crop edges are preserved."""
    y, x = np.mgrid[:height, :width].astype(np.float32)
    x = (x + .5) / width
    y = (y + .5) / height
    distance = np.sqrt(((x - .5) / .44)**2 + ((y - .76) / .29)**2)
    alpha = np.clip((1 - distance) / .18, 0, 1)
    alpha *= np.clip((y - .50) / .10, 0, 1)
    alpha *= np.clip((1 - y) / .10, 0, 1)
    alpha[-1, :] = 0  # No hard seam where the crop ends below the chin.
    return alpha[..., None]


def blend_lower_face(original, generated):
    if original.shape != generated.shape:
        raise ValueError('Blend requires matching face crops')
    alpha = lower_face_mask(*original.shape[:2])
    return np.rint(original.astype(np.float32) * (1-alpha) +
                   generated.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)


def install_wav2lip(inference):
    install_detector()
    source = inspect.getsource(inference.main)
    old = 'f[y1:y2, x1:x2] = p'
    if source.count(old) != 1:
        raise RuntimeError('Wav2Lip composite changed; review the preview adapter')
    inference.__dict__['blend_lower_face'] = blend_lower_face
    exec(compile(source.replace(old, 'f[y1:y2, x1:x2] = blend_lower_face(f[y1:y2, x1:x2], p)'),
                 '<preview_wav2lip>', 'exec'), inference.__dict__)


def install_musetalk():
    import torch
    from musetalk.utils import preprocessing, blending
    install_detector()
    original_preprocess = preprocessing.get_landmark_and_bbox

    def inline_preprocess(*args, **kwargs):
        # Loading fp32 weights then converting to fp16 leaves a large unused pool.
        # Release that pool in the SAME process; models remain loaded.
        torch.cuda.empty_cache()
        print(f'Inline preprocessing: allocated={torch.cuda.memory_allocated()/2**20:.0f} MiB, '
              f'reserved={torch.cuda.memory_reserved()/2**20:.0f} MiB', flush=True)
        return original_preprocess(*args, **kwargs)

    preprocessing.get_landmark_and_bbox = inline_preprocess
    original_detect = preprocessing.fa.get_detections_for_batch
    original_pose = preprocessing.inference_topdown
    totals = {'s3fd_seconds': 0., 'dwpose_seconds': 0., 'frames': 0}

    def timed_pose(*args, **kwargs):
        start = time.perf_counter()
        result = original_pose(*args, **kwargs)
        totals['dwpose_seconds'] += time.perf_counter() - start
        return result

    def timed_detect(images):
        start = time.perf_counter()
        result = original_detect(images)
        totals['s3fd_seconds'] += time.perf_counter() - start
        totals['frames'] += len(images)
        if totals['frames'] % 50 == 0:
            print(f'Preview detector cumulative: {totals}', flush=True)
        return result

    preprocessing.inference_topdown = timed_pose
    preprocessing.fa.get_detections_for_batch = timed_detect
    original_blend = blending.get_image

    def narrower_blend(image, face, face_box, **kwargs):
        # Retain upstream semantic mask, then limit the edited skin area further.
        result = original_blend(image, face, face_box, **kwargs)
        x1, y1, x2, y2 = map(int, face_box)
        output = image.copy()
        output[y1:y2, x1:x2] = blend_lower_face(
            image[y1:y2, x1:x2], result[y1:y2, x1:x2])
        return output

    blending.get_image = narrower_blend
