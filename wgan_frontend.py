import base64
import json
import urllib.error
import urllib.request
from io import BytesIO

import streamlit as st

DEFAULT_API = "http://localhost:5000/api"

st.set_page_config(
    page_title="WGAN CIFAR-10 Explorer",
    page_icon="🎛️",
    layout="wide",
)

st.markdown("""
# WGAN CIFAR-10 Explorer
A lightweight Streamlit interface for the WGAN Flask API.
Use the API endpoint below and switch between generation, interpolation, and training dashboard views.
""")

api_base = st.sidebar.text_input("WGAN API base URL", DEFAULT_API)
show_health = st.sidebar.button("Refresh status")


def api_request(path, method="GET", payload=None):
    url = api_base.rstrip("/") + path
    headers = {"Content-Type": "application/json"}
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            raw = resp.read()
            return json.loads(raw.decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return {"error": f"HTTP {exc.code}: {exc.reason}"}
    except urllib.error.URLError as exc:
        return {"error": str(exc.reason)}
    except Exception as exc:
        return {"error": str(exc)}


@st.cache_data(show_spinner=False)
def get_health():
    return api_request("/health")


def decode_image(b64_string):
    png = base64.b64decode(b64_string)
    return BytesIO(png)


health = get_health()
api_online = isinstance(health, dict) and "status" in health and health.get("status") == "ok"

if api_online:
    st.sidebar.success("API online")
    st.sidebar.write(f"**Device:** {health.get('device', 'unknown')}  ")
    st.sidebar.write(f"**Checkpoint:** {health.get('checkpoint', 'none')}  ")
else:
    st.sidebar.error("API offline")
    st.sidebar.write("The Streamlit frontend can still be used, but it will not generate real model output until the Flask API is running.")

if show_health:
    health = get_health()
    api_online = isinstance(health, dict) and "status" in health and health.get("status") == "ok"
    st.experimental_rerun()

pages = st.tabs(["Generate", "Interpolate", "Dashboard", "Samples"])

with pages[0]:
    st.subheader("Generate new images")
    cols = st.columns([3, 1, 1, 1])
    count = cols[0].slider("Number of images", 4, 64, 16, 4)
    seed = cols[1].number_input("Seed", min_value=0, max_value=999999, value=42)
    randomize = cols[2].button("Random seed")
    if randomize:
        import random

        seed = random.randint(0, 999999)
        cols[1].number_input("Seed", min_value=0, max_value=999999, value=seed)
    generate = cols[3].button("Generate")

    if generate and api_online:
        result = api_request("/generate", method="POST", payload={"n": count, "seed": seed})
        if "error" in result:
            st.error(result["error"])
        else:
            st.success(f"Generated {count} images")
            images = [decode_image(x) for x in result.get("images", [])]
            st.image(images, width=120, caption=[f"Seed {seed}#{i+1}" for i in range(len(images))])
    elif generate and not api_online:
        st.warning("API is offline. Start the Flask server to generate real images.")
        placeholder = [BytesIO(base64.b64decode(base64.b64encode(b"\x89PNG\r\n\x1a\n")))]
        st.image(placeholder, width=120)

with pages[1]:
    st.subheader("Latent space interpolation")
    left, right = st.columns(2)
    steps = left.slider("Steps", 4, 20, 10)
    seed_a = left.number_input("Seed A", min_value=0, max_value=999999, value=7)
    seed_b = left.number_input("Seed B", min_value=0, max_value=999999, value=99)
    interpolate = left.button("Interpolate")

    if interpolate and api_online:
        result = api_request(
            "/interpolate",
            method="POST",
            payload={"steps": steps, "seed_a": seed_a, "seed_b": seed_b},
        )
        if "error" in result:
            st.error(result["error"])
        else:
            frames = result.get("frames", [])
            alphas = result.get("alphas", [])
            columns = st.columns(min(10, len(frames)))
            for idx, frame in enumerate(frames):
                columns[idx % len(columns)].image(decode_image(frame), caption=f"α={alphas[idx]:.2f}", use_column_width=True)
    elif interpolate and not api_online:
        st.warning("API is offline. Start the Flask server to interpolate real images.")

with pages[2]:
    st.subheader("Training dashboard")
    if api_online:
        metrics = api_request("/metrics")
        if "error" in metrics:
            st.error(metrics["error"])
        elif isinstance(metrics, list) and metrics:
            import pandas as pd

            df = pd.DataFrame(metrics)
            st.metric("Epochs", int(df["epoch"].iloc[-1]))
            st.metric("Wasserstein dist", float(df["wasserstein_dist"].iloc[-1]))
            st.metric("Generator loss", float(df["gen_loss"].iloc[-1]))
            st.line_chart(df[["epoch", "wasserstein_dist"]].set_index("epoch"))
            st.line_chart(df.set_index("epoch")[["gen_loss", "critic_loss"]])
        else:
            st.info("No metric data available yet. Train the model to populate metrics.")
    else:
        st.warning("API is offline. Metrics require the Flask API.")

with pages[3]:
    st.subheader("Sample galleries")
    if api_online:
        sample_list = api_request("/samples")
        if "error" in sample_list:
            st.error(sample_list["error"])
        else:
            files = sample_list.get("files", [])
            if files:
                selected = st.selectbox("Saved sample grids", files)
                if st.button("Load sample"):
                    image_resp = api_request(f"/sample/{selected}")
                    if isinstance(image_resp, dict) and image_resp.get("error"):
                        st.error(image_resp["error"])
                    else:
                        st.image(f"http://localhost:5000/sample/{selected}", caption=selected)
            else:
                st.info("No saved sample grids found in the sample directory.")
    else:
        st.warning("API is offline. Sample galleries require the Flask API.")
