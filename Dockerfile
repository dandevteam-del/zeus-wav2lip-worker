# Zeus Wav2Lip worker — RunPod serverless GPU image.
# Wav2Lip animates ONLY the mouth region of a still face + audio → talking head with
# the head perfectly static (no SadTalker forehead-warp). GFPGAN restores the mouth
# crop so it isn't the classic Wav2Lip blur. Weights live on the NETWORK VOLUME.
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    WAV2LIP_DIR=/app/Wav2Lip

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3-dev build-essential \
        git ffmpeg libgl1 libglib2.0-0 ca-certificates wget && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel
RUN pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
        --extra-index-url https://download.pytorch.org/whl/cu121

# Rudrabha/Wav2Lip — the reference implementation.
RUN git clone --depth 1 https://github.com/Rudrabha/Wav2Lip.git /app/Wav2Lip

# numpy first (pinned — same np.float lesson as SadTalker), then the rest.
RUN pip install numpy==1.23.5
RUN pip install \
        librosa==0.10.1 numba==0.57.1 opencv-python-headless==4.9.0.80 \
        scipy==1.10.1 tqdm numba resampy==0.4.2 \
        basicsr==1.4.2 facexlib==0.3.0 gfpgan==1.3.8 \
        av safetensors runpod requests --no-build-isolation

# basicsr imports torchvision functional_tensor (removed in tv>=0.17) — repoint it.
RUN python3 -c "import os,basicsr; f=os.path.join(os.path.dirname(basicsr.__file__),'data','degradations.py'); s=open(f).read().replace('functional_tensor','functional'); open(f,'w').write(s); print('patched',f)"

# Sweep removed numpy aliases (np.float/np.int/np.bool) across Wav2Lip too.
COPY patch_np.py /app/patch_np.py
RUN python3 /app/patch_np.py /app/Wav2Lip

# Re-assert numpy pin last, fail loudly if np.float was dropped.
RUN pip install --force-reinstall --no-deps numpy==1.23.5 && \
    python3 -c "import numpy as np; print('numpy',np.__version__); assert hasattr(np,'float')"

ENV PYTHONPATH="/app/Wav2Lip:${PYTHONPATH}"
COPY handler.py /app/handler.py
CMD ["python3", "-u", "/app/handler.py"]
