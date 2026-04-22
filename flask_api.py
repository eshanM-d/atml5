import base64
import io
import json
import os

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from PIL import Image

# Generator and Critic must be importable — keep wgan_cifar10.py in the same dir.
try:
    from wgan_cifar10 import Generator, LATENT_DIM
except ImportError:
    # Fallback: inline minimal Generator so the API works standalone
    LATENT_DIM   = 100
    CHANNELS_IMG = 3
    FEATURES_G   = 64

    class Generator(nn.Module):
        def __init__(self, latent_dim=LATENT_DIM, features=FEATURES_G):
            super().__init__()
            def _block(i, o, k, s, p):
                return nn.Sequential(
                    nn.ConvTranspose2d(i, o, k, s, p, bias=False),
                    nn.BatchNorm2d(o),
                    nn.ReLU(True),
                )
            self.net = nn.Sequential(
                _block(latent_dim,   features*8, 4, 1, 0),
                _block(features*8,   features*4, 4, 2, 1),
                _block(features*4,   features*2, 4, 2, 1),
                nn.ConvTranspose2d(features*2, CHANNELS_IMG, 4, 2, 1),
                nn.Tanh(),
            )
        def forward(self, z):
            return self.net(z)

# ─── Config ───────────────────────────────────────────────────────────────────

CHECKPOINT_PATH  = os.environ.get("WGAN_CHECKPOINT",  "./checkpoints/wgan_latest.pth")
METRICS_FILE     = os.environ.get("WGAN_METRICS",     "./training_metrics.json")
SAMPLE_DIR       = os.environ.get("WGAN_SAMPLE_DIR",  "./samples")
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Load model once at startup ───────────────────────────────────────────────

generator = Generator(LATENT_DIM).to(DEVICE)
generator.eval()

if os.path.exists(CHECKPOINT_PATH):
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    generator.load_state_dict(ckpt["gen"])
    print(f"[WGAN API] Loaded checkpoint: {CHECKPOINT_PATH}  "
          f"(epoch {ckpt.get('epoch', '?')})")
else:
    print(f"[WGAN API] WARNING — checkpoint not found at {CHECKPOINT_PATH}. "
          "Using random weights (train first!).")

# ─── Helpers ──────────────────────────────────────────────────────────────────

def tensor_to_b64(tensor: torch.Tensor, size: int = 128) -> str:
    """Convert a single (3, H, W) tensor in [-1,1] to a base64 PNG string."""
    img = (tensor.clamp(-1, 1) + 1) / 2          # [0, 1]
    img = TF.to_pil_image(img.cpu())
    img = img.resize((size, size), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def grid_to_b64(tensor: torch.Tensor, nrow: int = 8, size: int = 32) -> str:
    """Tile a batch of images into a grid and return as base64 PNG."""
    import torchvision.utils as vutils
    grid = vutils.make_grid(tensor.clamp(-1, 1), nrow=nrow,
                             normalize=True, value_range=(-1, 1))
    img = TF.to_pil_image(grid.cpu())
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# ─── Flask app ────────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)   # allow React dev-server on a different port


@app.route("/api/health")
def health():
    ckpt_loaded = os.path.exists(CHECKPOINT_PATH)
    return jsonify({
        "status":          "ok",
        "device":          str(DEVICE),
        "checkpoint":      CHECKPOINT_PATH,
        "checkpoint_found": ckpt_loaded,
        "latent_dim":      LATENT_DIM,
    })


@app.route("/api/generate", methods=["POST"])
def generate():
    """
    Body (JSON, all optional):
      n          : int   — number of images (1–64, default 16)
      as_grid    : bool  — return a single grid PNG instead of individual images
      seed       : int   — for reproducibility
      latent_vecs: list  — list of flat float arrays, length LATENT_DIM each
                           (if provided, `n` and `seed` are ignored)
    """
    data = request.get_json(silent=True) or {}
    n         = min(max(int(data.get("n", 16)), 1), 64)
    as_grid   = bool(data.get("as_grid", False))
    seed      = data.get("seed")
    lvecs     = data.get("latent_vecs")

    if seed is not None:
        torch.manual_seed(int(seed))

    with torch.no_grad():
        if lvecs:
            z = torch.tensor(lvecs, dtype=torch.float32, device=DEVICE)
            z = z.view(-1, LATENT_DIM, 1, 1)
            n = z.size(0)
        else:
            z = torch.randn(n, LATENT_DIM, 1, 1, device=DEVICE)

        fake = generator(z)   # (n, 3, 32, 32)

    if as_grid:
        return jsonify({"type": "grid", "image": grid_to_b64(fake)})

    images = [tensor_to_b64(fake[i]) for i in range(n)]
    return jsonify({"type": "individual", "images": images, "count": n})


@app.route("/api/interpolate", methods=["POST"])
def interpolate():
    """
    Linear interpolation between two z vectors.
    Body (JSON):
      steps      : int  — number of frames including start/end (default 10)
      seed_a     : int  — seed for z_A  (optional)
      seed_b     : int  — seed for z_B  (optional)
      z_a        : list — explicit z_A flat array (overrides seed_a)
      z_b        : list — explicit z_B flat array (overrides seed_b)
    Returns:
      { frames: [base64, ...], alphas: [0.0, ..., 1.0] }
    """
    data   = request.get_json(silent=True) or {}
    steps  = min(max(int(data.get("steps", 10)), 2), 30)

    # Build z_A
    if "z_a" in data:
        z_a = torch.tensor(data["z_a"], dtype=torch.float32, device=DEVICE)
    else:
        seed_a = data.get("seed_a", 0)
        torch.manual_seed(int(seed_a))
        z_a = torch.randn(LATENT_DIM, device=DEVICE)

    # Build z_B
    if "z_b" in data:
        z_b = torch.tensor(data["z_b"], dtype=torch.float32, device=DEVICE)
    else:
        seed_b = data.get("seed_b", 1)
        torch.manual_seed(int(seed_b))
        z_b = torch.randn(LATENT_DIM, device=DEVICE)

    alphas = [i / (steps - 1) for i in range(steps)]
    frames = []

    with torch.no_grad():
        for alpha in alphas:
            z = (1 - alpha) * z_a + alpha * z_b
            z = z.view(1, LATENT_DIM, 1, 1)
            img = generator(z)[0]
            frames.append(tensor_to_b64(img, size=128))

    return jsonify({
        "frames": frames,
        "alphas": alphas,
        "steps":  steps,
    })


@app.route("/api/metrics")
def metrics():
    """Return training metrics from training_metrics.json."""
    if not os.path.exists(METRICS_FILE):
        return jsonify({"error": "metrics file not found — train the model first"}), 404
    with open(METRICS_FILE) as f:
        data = json.load(f)
    return jsonify(data)


@app.route("/api/samples")
def list_samples():
    """List generated sample-grid PNG filenames."""
    if not os.path.isdir(SAMPLE_DIR):
        return jsonify({"files": []})
    files = sorted(
        [f for f in os.listdir(SAMPLE_DIR) if f.endswith(".png")],
        reverse=True,
    )
    return jsonify({"files": files, "dir": SAMPLE_DIR})


@app.route("/api/sample/<path:filename>")
def serve_sample(filename):
    """Serve a sample grid image."""
    return send_from_directory(SAMPLE_DIR, filename)


@app.route("/api/random_z")
def random_z():
    """Return a random latent vector (useful for the UI 'randomise' button)."""
    z = torch.randn(LATENT_DIM).tolist()
    return jsonify({"z": z, "dim": LATENT_DIM})




if __name__ == "__main__":
    print(f"[WGAN API] Starting on http://localhost:5000  (device={DEVICE})")
    app.run(host="0.0.0.0", port=5000, debug=False)
