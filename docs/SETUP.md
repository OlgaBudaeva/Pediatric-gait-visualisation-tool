## Setup

### Requirements

- **Python 3.12**
- **Windows 11**
- **NVIDIA GPU (optional, recommended):** NVIDIA driver [version], CUDA [version], cuDNN [version]. The app also runs on CPU.

`requirements.txt` lists the exact library versions I used, so the environment can be reproduced.

### 1. Create an environment and install the libraries

```powershell
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Video encoding uses ffmpeg, which is installed automatically through `imageio-ffmpeg`.

**CPU only:** in `requirements.txt`, replace `onnxruntime-gpu` with `onnxruntime`, and in `gait_core.py` change `device="cuda"` to `device="cpu"`.

> Do not install `onnxruntime` and `onnxruntime-gpu` together. If both are present, the GPU provider may not be detected and the model silently runs on the CPU. If that happens, run `pip uninstall onnxruntime onnxruntime-gpu -y` and install only the one you need.

Check that the GPU is visible:

```powershell
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

`CUDAExecutionProvider` should be in the list. If it is not, the usual cause is a CUDA/cuDNN version that does not match your onnxruntime-gpu release.

### 2. Model files

On the first run, rtmlib downloads the detector (YOLOX) and the pose model (RTMW) into its cache (`~/.cache/rtmlib`). The RTMW file is about 200 MB, so on a slow connection the download can stall and leave a corrupted file, which causes a `BadZipFile` error. In that case, delete the partial file from the cache and download the model manually:

1. Download [rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip](https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip).
2. Create the folder `models/rtmw-x-l/` in the project root and extract the zip there.
3. Make sure the file `models/rtmw-x-l/end2end.onnx` exists.

If this file is present, the app uses it instead of downloading. The `models/` folder is not in the repository.

### 3. Run the app

```powershell
python -m streamlit run app.py
```

Open http://localhost:8501. To protect the app with a password, set it before starting:

```powershell
$env:GAIT_APP_PASSWORD = "choose-a-password"
```

Uploaded videos and results are saved in `results/`, which is created automatically and excluded from version control.

### 4. Use it on a phone (optional)

Install [Tailscale](https://tailscale.com) on the computer and on the phone, sign in with the same account, then open `http://<tailscale-ip>:8501` on the phone. The computer must be on and running the app. If the page does not load, check the Windows Firewall rule for port 8501.

### Troubleshooting

| Problem                    | Cause and fix                                                                             |
| -------------------------- | ----------------------------------------------------------------------------------------- |
| `BadZipFile` on first run  | The model download was interrupted. Delete the cached zip and download manually (step 2). |
| "GPU not detected" warning | CPU-only onnxruntime is installed, or CUDA/cuDNN do not match. See step 1.                |
| `streamlit` not recognised | The environment is not active. Use `python -m streamlit run app.py`.                      |
| Video will not open        | Some phones record HEVC. Set the camera to "Most Compatible" or convert to H.264.         |
