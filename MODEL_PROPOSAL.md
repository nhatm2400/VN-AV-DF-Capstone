# Model Proposal for VN-AV-DF

## 1. Context

This repository is currently strongest on dataset construction and curation:

- `src/pipeline/02_curate` produces clean real clips with metadata, `speaker_id`, and quality scores.
- `src/pipeline/03_fake` generates pseudo-fake samples through temporal desync, frame reverse, pitch flatten, and anonymization.
- `PoC` contains an early PAMF-style audio-visual fusion model, but it does not yet use mouth ROI, speaker-disjoint evaluation, or a complete label/split pipeline.

The recommended model should therefore build on the existing data pipeline, not on the current PoC training loop as-is.

## 2. Proposed Model: AVSP-Net

AVSP-Net stands for Audio-Visual Sync + Prosody Network.

It is designed for Vietnamese audio-visual deepfake detection where the core signals are:

- audio-visual timing mismatch,
- mouth-motion inconsistency,
- Vietnamese prosody and F0 contour distortion,
- visual degradation or anonymization artifacts.

## 3. High-Level Architecture

```text
Input video clip
  |
  |-- Audio 16 kHz
  |     -> Vietnamese Wav2Vec2 encoder
  |     -> audio sequence A_t
  |
  |-- Video frames
  |     -> face detection / tracking
  |     -> mouth ROI crop
  |     -> visual temporal encoder
  |     -> visual sequence V_t
  |
  |-- Prosody features
  |     -> F0, energy, voiced/unvoiced, delta-F0
  |     -> prosody encoder
  |     -> prosody sequence P_t
  |
  |-- A_t + V_t
  |     -> cross-attention sync fusion
  |
  |-- fused AV + P_t
        -> temporal transformer / attentive pooling
        -> binary classifier: real / fake
        -> auxiliary offset classifier
```

## 4. Components

### 4.1 Audio Encoder

Use the Vietnamese Wav2Vec2 backbone already used in `PoC/src/feature_extractor.py`.

Recommended setup:

- Start with frozen Wav2Vec2 features.
- Train the fusion and classifier heads first.
- Fine-tune only the last 2-4 transformer layers after the classifier becomes stable.

Output:

```text
A_t: [B, T_audio, D_audio]
```

### 4.2 Visual Encoder

Replace the current full-frame MobileNetV2 approach with mouth ROI modeling.

Required preprocessing:

- detect face,
- track face across frames,
- crop mouth region,
- resize mouth ROI to a fixed size,
- normalize frames consistently.

Reason: full-frame features can leak identity, background, camera, and compression artifacts. The model should focus on mouth movement.

Recommended visual backbones:

- lightweight option: ResNet18 + temporal convolution / BiGRU,
- stronger option: VideoMAE-small or a small TimeSformer-like encoder,
- practical first version: 2D CNN per frame + temporal transformer.

Output:

```text
V_t: [B, T_video, D_visual]
```

### 4.3 Prosody Encoder

Extract Vietnamese prosody features:

- F0 contour,
- delta-F0,
- energy,
- voiced/unvoiced flag,
- local pitch slope,
- optional jitter/creaky-voice proxy.

This branch is important because one fake method in the repo, `03_pitch_flatten.py`, attacks pitch while keeping the video stream unchanged.

Output:

```text
P_t: [B, T_prosody, D_prosody]
```

### 4.4 Cross-Attention Sync Fusion

Use audio as query and visual as key/value:

```text
Q = Linear(A_t)
K = Linear(V_t)
V = Linear(V_t)
F_av = MultiHeadAttention(Q, K, V)
```

This keeps the useful idea from the PoC PAMF model, but applies it to mouth ROI instead of full frames.

### 4.5 Classification Heads

Use two heads:

```text
real_fake_head: binary logit
offset_head: classifies AV offset
```

Offset classes:

```text
[-15f, -7f, -3f, 0f, +3f, +7f, +15f]
```

This matches the pseudo-fake generation logic in `01_temporal_desync.py`.

## 5. Loss Function

Use logits directly:

```text
L = BCEWithLogitsLoss(real_fake)
  + 0.5 * CrossEntropyLoss(offset_class)
  + 0.1 * modality_consistency_loss
```

Do not use `Sigmoid + BCELoss` in the model forward path. `BCEWithLogitsLoss` is numerically more stable.

## 6. Dataset and Split Rules

The split is more important than the model.

Required rules:

- split by `speaker_id`, not by random clip,
- keep all clips from the same `source_video` in the same split,
- keep each fake and its `source_clip` real sample in the same split,
- apply codec, blur, and compression augmentation symmetrically to real and fake,
- report metrics per fake method, not only global accuracy.

Recommended split:

```text
train: 70%
val:   15%
test:  15%
```

Split unit:

```text
speaker_id + source_video group
```

## 7. Required Baselines

Run these baselines before claiming the fusion model works:

1. Audio-only Wav2Vec2 classifier.
2. Visual-only mouth ROI classifier.
3. AV fusion without prosody.
4. AVSP-Net full model.
5. SyncNet score threshold baseline if SyncNet features are available.

The full model is only meaningful if it beats the unimodal baselines on a speaker-disjoint test set.

## 8. Vietnamese-Specific New Direction: VietTone-AVDF

### 8.1 Core Idea

Vietnamese is tonal. A fake video can keep mouth movement and timing plausible while corrupting the tone contour. This is especially relevant for pitch flattening, voice conversion, text-to-speech replacement, and AI-generated speech.

VietTone-AVDF is a tone-aware audio-visual detector:

```text
Detect not only "does the mouth match the sound?"
but also "does the Vietnamese tone contour match the spoken syllable dynamics?"
```

### 8.2 Why This Is Different From Generic AV Deepfake Detection

Generic lip-sync detectors mainly learn timing between mouth opening and phonetic energy.

For Vietnamese, this is incomplete because:

- lexical meaning is tied to tone,
- F0 trajectory carries strong linguistic signal,
- hoi/nga tones often include complex pitch movement and voice quality,
- a fake can preserve timing but damage tone,
- mouth motion alone cannot fully validate Vietnamese speech authenticity.

### 8.3 Training Signal

Generate Vietnamese-specific hard negatives:

1. Tone flattening:
   - flatten F0 over voiced regions,
   - keep video unchanged,
   - label as fake.

2. Tone swapping:
   - transfer F0 contour from another clip of the same speaker or same gender,
   - keep duration and mouth timing close,
   - label as fake.

3. Tone exaggeration:
   - amplify F0 slope or bend,
   - preserve speech timing,
   - label as fake.

4. Local tone corruption:
   - corrupt only 0.5-1.5 seconds,
   - keep the rest real,
   - force the model to detect local prosody anomalies.

These negatives are harder and more Vietnamese-specific than simple global AV offset.

### 8.4 Optional Weak Tone Labels

If transcripts are available or can be generated with a Vietnamese ASR model:

1. segment speech into syllables,
2. infer tone from Vietnamese diacritics,
3. align syllables to time using forced alignment,
4. add an auxiliary tone-contour prediction task.

Auxiliary task:

```text
prosody_branch -> tone class:
  ngang, sac, huyen, hoi, nga, nang
```

This does not need to be perfect. Even weak tone supervision can force the model to attend to Vietnamese-specific acoustic structure.

### 8.5 VietTone Architecture Add-On

```text
Prosody sequence P_t
  -> tone-aware transformer
  -> syllable-level pooling
  -> tone consistency head
  -> fused with AVSP-Net classifier
```

Additional loss:

```text
L_total = L_AVSP
        + 0.3 * CE(tone_class)
        + 0.2 * SmoothL1(reconstructed_F0_contour)
```

Use this only when tone labels or aligned syllable regions are available. Without transcripts, keep only F0 corruption detection.

### 8.6 Evaluation for Vietnamese Data

Report separate metrics for:

- temporal desync fake,
- frame reverse fake,
- pitch flatten fake,
- anonymization fake,
- tone swap fake,
- local tone corruption fake.

Also report:

- per-region performance if tier metadata is reliable,
- performance under noisy audio,
- performance under compression,
- false positive rate on low-SNR real clips.

## 9. Implementation Roadmap

### Phase 1: Fix Data Contract

- Build `data/labels.csv` containing both real and fake rows.
- Add `split` column.
- Ensure fake samples keep `source_clip`, `speaker_id`, and `source_video`.
- Make split generation deterministic.

### Phase 2: Replace PoC Feature Extraction

- Add mouth ROI extraction.
- Save audio, mouth ROI, prosody features per clip.
- Avoid full-frame visual features for the main model.

### Phase 3: Train Baselines

- Audio-only.
- Visual-only.
- AV fusion.
- AV fusion + prosody.

### Phase 4: Add VietTone Negatives

- Implement tone flattening variants beyond full-utterance flattening.
- Add tone swap and local tone corruption.
- Add per-method evaluation.

### Phase 5: Report Correct Metrics

Minimum metrics:

- accuracy,
- precision,
- recall,
- F1,
- ROC-AUC,
- method-wise F1,
- speaker-disjoint test score.

## 10. Expected Advantages

AVSP-Net should handle:

- AV timing mismatch,
- local visual motion reversal,
- pitch/prosody corruption,
- face anonymization or visual degradation.

VietTone-AVDF adds a Vietnamese-specific advantage:

- better detection of fake speech that sounds temporally aligned but has unnatural tone contour,
- better coverage for Vietnamese TTS / voice conversion attacks,
- stronger scientific motivation than a generic lip-sync-only detector.

## 11. Main Risks

- If mouth ROI extraction is unstable, the model may underperform.
- If train/test split leaks speaker or source video, metrics will be inflated.
- If only fake videos are re-encoded, the model may learn codec artifacts.
- If blur/anonymization appears only in fake labels, the model may learn "blur means fake".
- If tone labels from ASR are noisy, tone auxiliary loss should be weighted lightly.

## 12. Recommended Final Claim

Do not claim that the model detects all deepfakes.

A defensible claim is:

```text
The proposed model detects Vietnamese audio-visual inconsistencies by combining
mouth-motion synchrony, audio representation learning, and tone-aware prosody
features. It is evaluated under speaker-disjoint splits and method-wise fake
categories to reduce identity, source, and codec leakage.
```

