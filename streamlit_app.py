import streamlit as st
import re
import difflib
import time
import io
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO


CSS_DARK = r"""
:root{
    --bg-1: #00040a;
    --bg-2: #041029;
    --glass: rgba(255,255,255,0.03);
    --muted: #9fb0d9;
    --accent-1: linear-gradient(90deg,#0ea5e9,#7c3aed);
}
*{box-sizing:border-box}
html,body,#root{height:100%}
body{
    margin:0;
    font-family: Inter, system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
    background: linear-gradient(180deg,var(--bg-1) 0%, var(--bg-2) 100%);
    color:#e6f0ff;
}
.app{display:flex;min-height:100vh}
.sidebar{width:300px;padding:18px}
.search{height:44px;background:rgba(255,255,255,0.04);border-radius:14px;margin-bottom:18px}
.status{display:flex;align-items:center;gap:12px;padding:12px;border-radius:10px;background:linear-gradient(180deg, rgba(16,185,129,0.08), rgba(16,185,129,0.03));border:1px solid rgba(16,185,129,0.12)}
.status .checkbox{background:#10b981;color:#042017;padding:6px;border-radius:6px;font-weight:700}
.status-title{font-size:14px}
.controls{margin-top:18px;color:var(--muted)}
.control-row{display:flex;align-items:center;gap:10px}
.control-row input{transform:scale(1.1)}

.main{flex:1;padding:36px}
.glass{background:var(--glass);border:1px solid rgba(255,255,255,0.06);backdrop-filter:blur(8px);border-radius:14px;padding:18px}
.header-card{display:flex;gap:18px;align-items:center;padding:28px;margin-bottom:20px}
.header-card .badge{font-size:34px;background:linear-gradient(90deg,#ff8a65,#ff5252);padding:14px;border-radius:12px}
.header-text h1{margin:0;font-size:28px}
.header-text .lead{margin:6px 0 0;color:#bcd3ff}

.nav-card{display:flex;align-items:center;justify-content:space-between;padding:12px 20px;margin-bottom:18px}
.nav-list{display:flex;gap:18px;list-style:none;padding:0;margin:0}
.nav-list li{padding:10px 16px;border-radius:10px;color:#bcd3ff}
.nav-list li.active{background:rgba(255,255,255,0.03);color:#fff}
.btn-try{background:var(--accent-1);border:0;color:#fff;padding:10px 16px;border-radius:10px;font-weight:600}

.content{display:grid;grid-template-columns:2fr 1fr;gap:18px}
.preview{height:360px;border-radius:12px;display:flex;align-items:center;justify-content:center;background:linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));border:1px solid rgba(255,255,255,0.03)}
.preview-placeholder{color:var(--muted)}
.result-placeholder{color:var(--muted);min-height:200px}

@media(max-width:900px){
    .app{flex-direction:column}
    .sidebar{width:100%;display:flex;gap:12px}
    .content{grid-template-columns:1fr}
}

"""

CSS = CSS_DARK

st.markdown(f"""<style>{CSS}</style>""", unsafe_allow_html=True)

# Sidebar status (styled)
st.sidebar.markdown('<div class="status"><div class="checkbox">✔</div><div><div class="status-title">Model loaded successfully!</div></div></div>', unsafe_allow_html=True)


# --- Recipes database (English only) ---
RECIPES = {
    "apple": {
        "title": "Apple Crumble",
        "content": "Ingredients:\n- 4 apples\n- 100g flour\n- 75g butter\n- 75g brown sugar\n\nSteps:\n1. Slice apples and place in a baking dish.\n2. Mix flour, butter and sugar into crumbs and sprinkle over apples.\n3. Bake at 180°C for 30-35 minutes until golden."
    },
    "banana": {
        "title": "Banana Smoothie",
        "content": "Ingredients:\n- 2 ripe bananas\n- 250ml milk (or plant milk)\n- 1 tbsp honey\n\nSteps:\n1. Blend all ingredients until smooth.\n2. Serve chilled."
    },
    "mango": {
        "title": "Mango Salsa",
        "content": "Ingredients:\n- 1 ripe mango\n- 1/2 red onion\n- Juice of 1 lime\n- Handful cilantro\n\nSteps:\n1. Dice mango and onion.\n2. Mix with lime juice and chopped cilantro.\n3. Serve with chips or grilled fish."
    },
    "orange": {
        "title": "Orange Granita",
        "content": "Ingredients:\n- 500ml fresh orange juice\n- 50g sugar\n\nSteps:\n1. Dissolve sugar into juice.\n2. Freeze in a shallow tray, scraping every 30 minutes until flaky."
    },
    "strawberry": {
        "title": "Strawberry Salad",
        "content": "Ingredients:\n- 250g strawberries\n- Handful of spinach\n- Balsamic vinaigrette\n\nSteps:\n1. Halve strawberries and toss with spinach.\n2. Drizzle with vinaigrette and serve."
    },
    "cucumber": {
        "title": "Cucumber Raita",
        "content": "Ingredients:\n- 1 large cucumber\n- 250g plain yogurt\n- 1/2 tsp roasted cumin powder\n- Salt to taste\n- Fresh cilantro or mint (optional)\n\nSteps:\n1. Peel and grate or finely chop the cucumber.\n2. Mix cucumber with yogurt, cumin powder and salt.\n3. Garnish with chopped cilantro or mint and serve chilled as a side."
    }
}

def extract_fruit_name(label: str) -> str:
    s = label.lower()
    s = s.replace("_", " ")
    s = re.sub(r"\b(fresh|rotten|ripe|unripe|good|bad)\b", "", s)
    s = re.sub(r"[^a-z\s]", "", s)
    s = s.strip()
    parts = s.split()
    if len(parts) == 0:
        return ""
    for p in parts:
        if p in RECIPES:
            return p
    return parts[-1]


def auto_map_fruit(detected_info, conf_thresh=0.3):
    if not detected_info:
        return None
    items = sorted(detected_info, key=lambda x: x.get("conf", 0), reverse=True)
    keys = list(RECIPES.keys())
    for it in items:
        conf = float(it.get("conf", 0))
        if conf < conf_thresh:
            continue
        label = it.get("label", "").lower()
        name = extract_fruit_name(label)
        if name in RECIPES:
            return name
        for k in keys:
            if k in label:
                return k
        match = difflib.get_close_matches(label, keys, n=1, cutoff=0.6)
        if match:
            return match[0]
        for token in label.split():
            match = difflib.get_close_matches(token, keys, n=1, cutoff=0.7)
            if match:
                return match[0]
    return None



# Cache the model so it's loaded once per worker
@st.cache_resource
def load_model(path="best1.pt"):
    model = YOLO(path)
    try:
        model.to("cpu")
    except Exception:
        pass
    return model


model = load_model()


# --- Main header / nav (visual) ---
st.markdown(
        """
        <div class="header-card glass">
            <div class="badge">🍓</div>
            <div class="header-text">
                <h1>Fruit Freshness Detector</h1>
                <p class="lead">Detect whether a fruit is fresh or rotten using YOLO</p>
            </div>
        </div>

        <div class="nav-card glass">
            <ul class="nav-list">
                <li class="active">🍏 Rotten or Not</li>
                <li>📖 Recipe Ideas</li>
                <li>ℹ️ About</li>
            </ul>
            <button class="btn-try">Try it</button>
        </div>
        """,
        unsafe_allow_html=True,
)

# --- Live webcam detection UI (controls remain in a column) ---
# Live webcam header
st.header("🎥 Live Webcam Detection")

# Layout: large preview on left, detection/recipe card on right
cols = st.columns([2, 1])
preview_col = cols[0]
right_col = cols[1]

# Controls moved to sidebar to match prototype layout
start_detection = st.sidebar.button("Start Webcam")
stop_detection = st.sidebar.button("Stop Webcam")
st.sidebar.markdown("---")
st.sidebar.write("**Auto-map recipes:**")
auto = st.sidebar.checkbox("Auto-select best match", value=True)
conf_thresh = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.3, 0.05)
st.sidebar.markdown("---")
st.sidebar.write("**Recipes (manual override):**")
options = sorted(RECIPES.keys())
manual_choice = st.sidebar.selectbox("Select fruit for recipe (override)", options)

FRAME_WINDOW = preview_col.image([], width=640)
det_card = right_col.empty()


# Initialize session state for start/stop
if "running" not in st.session_state:
    st.session_state.running = False

if start_detection:
    st.session_state.running = True
if stop_detection:
    st.session_state.running = False


input_mode = st.sidebar.radio("Input source", ["Browser camera (recommended)", "Local webcam"], index=0)

if st.session_state.running:
    detected_info = []

    if input_mode.startswith("Browser"):
        img_file = preview_col.camera_input("Camera")
        if img_file is not None:
            image = Image.open(img_file).convert("RGB")
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            results = model.predict(frame, conf=0.5, verbose=False)
            pred = results[0]

            if pred.boxes is not None and len(pred.boxes) > 0:
                for box in pred.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = pred.names[cls_id]
                    detected_info.append({"label": label, "conf": float(conf), "cls_id": int(cls_id)})

                    color = (0,255,0) if "fresh" in label.lower() else (0,0,255)
                    cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame_rgb, width=640)

    else:
        # Local webcam: capture a single frame per rerun
        if preview_col.button("Capture frame from local webcam"):
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                st.error("Could not read from local webcam.")
            else:
                frame = cv2.flip(frame, 1)
                results = model.predict(frame, conf=0.5, verbose=False)
                pred = results[0]

                if pred.boxes is not None and len(pred.boxes) > 0:
                    for box in pred.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        label = pred.names[cls_id]
                        detected_info.append({"label": label, "conf": float(conf), "cls_id": int(cls_id)})

                        color = (0,255,0) if "fresh" in label.lower() else (0,0,255)
                        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                        cv2.putText(frame, f"{label} {conf:.2f}", (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(frame_rgb, width=640)

    # show detection + recipe card in right column
    with det_card.container():
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("**Detection Result**")

        # choose fruit: prefer auto-mapped detection, fallback to manual sidebar choice
        chosen_fruit = None
        if detected_info and auto:
            auto_choice = auto_map_fruit(detected_info, conf_thresh=conf_thresh)
            if auto_choice:
                chosen_fruit = auto_choice
                st.success(f"Auto-selected: {chosen_fruit}")

        if not chosen_fruit:
            chosen_fruit = manual_choice

        # show raw detections (if any)
        if detected_info:
            st.write(detected_info)
        else:
            st.info("No fruit detected.")

        # show recipe card for the chosen fruit
        if chosen_fruit in RECIPES:
            r = RECIPES[chosen_fruit]
            st.markdown(f"<div style='margin-top:12px' class='glass'><h4>{r.get('title', chosen_fruit.title())}</h4><pre style='white-space:pre-wrap'>{r.get('content','')}</pre></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

else:
    preview_col.info("Click 'Start Webcam' to begin detection. Use Browser camera for Streamlit Cloud.")
