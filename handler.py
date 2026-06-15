"""Zeus Wav2Lip RunPod serverless handler.

Input (event["input"]):
    image_b64    still face portrait (png/jpg), base64   [either this...]
    video_b64    base face clip (mp4), base64            [...or this]
    audio_b64    voiceover (wav), base64   (required)
    fps          static-video fps when image_b64 given (default 25)
    pads         face pad "t b l r" (default "0 12 0 0" — include the chin)
    enhance      "gfpgan" | "" (default "gfpgan")  — restore the mouth crop
Output:
    video_b64    rendered talking-head mp4, base64

Wav2Lip animates ONLY the mouth — the head stays exactly as in the source frame,
so a still portrait talks with zero head motion / no forehead warp.
Weights are cached on the network volume at runtime.
"""
import os, base64, subprocess, tempfile, time, urllib.request

VOL = "/runpod-volume" if os.path.isdir("/runpod-volume") else "/tmp/zeusw2l"
MODELS = f"{VOL}/models"
W2L = "/app/Wav2Lip"
for d in (MODELS, f"{VOL}/tmp", f"{VOL}/hf"):
    os.makedirs(d, exist_ok=True)
os.environ.setdefault("HF_HOME", f"{VOL}/hf")

# Wav2Lip checkpoint + s3fd face detector. HF mirrors (the official GDrive isn't
# automatable). gfpgan/facexlib weights download on first enhance.
DL = {
    f"{MODELS}/wav2lip_gan.pth":
        "https://huggingface.co/camenduru/Wav2Lip/resolve/main/checkpoints/wav2lip_gan.pth",
    f"{W2L}/face_detection/detection/sfd/s3fd.pth":
        "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth",
    f"{MODELS}/GFPGANv1.4.pth":
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
}


def _dl(dst, url):
    if os.path.exists(dst) and os.path.getsize(dst) > 1000:
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "zeus-w2l"})
            with urllib.request.urlopen(req, timeout=600) as r, open(dst, "wb") as f:
                f.write(r.read())
            if os.path.getsize(dst) > 1000:
                return
        except Exception as e:
            last = e
            time.sleep(3)
    raise RuntimeError(f"download failed: {url} ({last})")


def _ensure():
    for dst, url in DL.items():
        _dl(dst, url)


def _write_b64(b64, path):
    with open(path, "wb") as f:
        f.write(base64.b64decode(b64))


def _b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def _enhance(in_mp4, out_mp4):
    """GFPGAN-restore each frame's face so the Wav2Lip mouth crop isn't blurry."""
    import cv2, glob, numpy as np  # noqa
    from gfpgan import GFPGANer
    work = tempfile.mkdtemp(dir=f"{VOL}/tmp")
    frames = f"{work}/f"; os.makedirs(frames, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", in_mp4,
                    f"{frames}/%05d.png"], check=True)
    fps = subprocess.run(["ffprobe", "-v", "0", "-of", "csv=p=0",
                          "-select_streams", "v:0", "-show_entries",
                          "stream=r_frame_rate", in_mp4],
                         capture_output=True, text=True).stdout.strip() or "25"
    rest = GFPGANer(model_path=f"{MODELS}/GFPGANv1.4.pth", upscale=1, arch="clean",
                    channel_multiplier=2, bg_upsampler=None)
    out_frames = f"{work}/o"; os.makedirs(out_frames, exist_ok=True)
    for p in sorted(glob.glob(f"{frames}/*.png")):
        img = cv2.imread(p)
        _, _, restored = rest.enhance(img, has_aligned=False, only_center_face=True,
                                      paste_back=True)
        cv2.imwrite(os.path.join(out_frames, os.path.basename(p)), restored)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-r", fps.split('/')[0] if '/' not in fps else str(eval(fps)),
                    "-i", f"{out_frames}/%05d.png", "-i", in_mp4, "-map", "0:v", "-map", "1:a?",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", out_mp4], check=True)


def handler(event):
    inp = event.get("input") or {}
    if not inp.get("audio_b64"):
        return {"error": "audio_b64 required"}
    if not inp.get("image_b64") and not inp.get("video_b64"):
        return {"error": "image_b64 or video_b64 required"}
    t0 = time.time()
    try:
        _ensure()
    except Exception as e:
        return {"error": f"model download failed: {e}"}

    work = tempfile.mkdtemp(dir=f"{VOL}/tmp")
    audio = f"{work}/a.wav"; _write_b64(inp["audio_b64"], audio)
    fps = str(int(inp.get("fps", 25)))
    res = int(inp.get("res", 1080))  # static-video render width — higher = sharper GFPGAN restore

    if inp.get("image_b64"):
        img = f"{work}/face.png"; _write_b64(inp["image_b64"], img)
        face = f"{work}/face.mp4"
        dur = subprocess.run(["ffprobe", "-v", "0", "-of", "csv=p=0",
                              "-show_entries", "format=duration", audio],
                             capture_output=True, text=True).stdout.strip() or "5"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", img,
                        "-t", dur, "-r", fps,
                        "-vf", f"scale={res}:-2:flags=lanczos,pad={res}:ceil(ih/2)*2",
                        "-pix_fmt", "yuv420p", face], check=True)
    else:
        face = f"{work}/face.mp4"; _write_b64(inp["video_b64"], face)

    raw = f"{work}/raw.mp4"
    pads = str(inp.get("pads", "0 12 0 0")).split()
    cmd = ["python3", f"{W2L}/inference.py",
           "--checkpoint_path", f"{MODELS}/wav2lip_gan.pth",
           "--face", face, "--audio", audio, "--outfile", raw,
           "--pads", *pads, "--nosmooth"]
    r = subprocess.run(cmd, cwd=W2L, capture_output=True, text=True)
    if not os.path.exists(raw):
        return {"error": f"wav2lip failed: {(r.stderr or r.stdout)[-1500:]}"}

    out = raw
    if inp.get("enhance", "gfpgan") == "gfpgan":
        try:
            enh = f"{work}/enh.mp4"; _enhance(raw, enh); out = enh
        except Exception as e:
            out = raw  # fall back to un-enhanced rather than fail the job
            print("enhance skipped:", e)

    return {"video_b64": _b64(out), "seconds": round(time.time() - t0, 1)}


import runpod  # noqa: E402
runpod.serverless.start({"handler": handler})
