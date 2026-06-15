# zeus-wav2lip-worker

RunPod serverless GPU worker — **Wav2Lip + GFPGAN**. Animates ONLY the mouth of a
still face portrait (or video) to match audio, leaving the head perfectly static
(no SadTalker forehead-warp). The right engine for a "talking photo" avatar.

Input `{image_b64 | video_b64, audio_b64, fps?, pads?, enhance?}` → `{video_b64}`.
Weights (wav2lip_gan, s3fd, GFPGAN) cache on the attached network volume at runtime.
