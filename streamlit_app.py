import os
os.environ["YOLO_CONFIG_DIR"] = "/tmp"

import streamlit as st
import re
import difflib
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Fruit Freshness Detector", layout="wide")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = YOLO("best1.pt")
    model.to("cpu")
    return model

model = load_model()

# ---------------- RECIPES ----------------
RECIPES = {
    "apple": {"title": "Apple Crumble", "content": "Bake apples with flour, butter & sugar at 180°C for 30 mins."},
    "banana": {"title": "Banana Smoothie", "content": "Blend banana + milk + honey."},
    "mango": {"title": "Mango Salsa", "content": "Mix mango + onion + lime."},
    "orange": {"title": "Orange Granita", "content": "Freeze sweetened orange juice."},
    "strawberry": {"title": "Strawberry Salad", "content": "Strawberry + spinach + vinaigrette."},
    "cucumber": {"title": "Cucumber Raita", "content": "Curd + cucumber + cumin."}
}

# ---------------- HELPERS ----------------
def extract_fruit_name(label):
    label = re.sub(r"[^a-z]", " ", label.lower()).strip()
    for word in label.split():
        if word in RECIPES:
            return word
    return None


def auto_map(detections):
    for d in detections:
        fruit = extract_fruit_name(d["label"])
        if fruit:
            return fruit
    return None

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙ Controls")

start = st.sidebar.button("Start Detection")
stop = st.sidebar.button("Stop Detection")

input_mode = st.sidebar.radio(
    "Input Source",
    ["Browser Camera (Cloud)", "Local Webcam (PC only)"]
)

manual_recipe = st.sidebar.selectbox("Manual Recipe", list(RECIPES.keys()))

if "run" not in st.session_state:
    st.session_state.run = False

if start:
    st.session_state.run = True

if stop:
    st.session_state.run = False

# ---------------- HEADER ----------------
st.title("🍓 Fruit Freshness Detector")
st.write("Detect Fresh vs Rotten fruit using YOLO")

col1, col2 = st.columns([2, 1])

frame_window = col1.empty()
result_box = col2.empty()

# ---------------- DETECTION ----------------
if st.session_state.run:

    detections = []

    # -------- BROWSER CAMERA (STREAMLIT CLOUD SAFE) --------
    if input_mode.startswith("Browser"):

        img = col1.camera_input("Take a picture")

        if img:
            image = Image.open(img).convert("RGB")
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            results = model.predict(frame, conf=0.5, verbose=False)[0]

            if results.boxes is not None:
                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = results.names[cls]

                    detections.append({"label": label, "conf": conf})

                    color = (0, 255, 0) if "fresh" in label.lower() else (0, 0, 255)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # -------- LOCAL WEBCAM (ONLY FOR YOUR LAPTOP) --------
    else:
        if col1.button("Capture from webcam"):
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            cap.release()

            if ret:
                frame = cv2.flip(frame, 1)

                results = model.predict(frame, conf=0.5, verbose=False)[0]

                if results.boxes is not None:
                    for box in results.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        label = results.names[cls]

                        detections.append({"label": label, "conf": conf})

                        color = (0, 255, 0) if "fresh" in label.lower() else (0, 0, 255)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # ---------------- RESULT + RECIPE ----------------
    with result_box.container():

        st.subheader("Detection Result")

        if detections:
            st.write(detections)
            fruit = auto_map(detections)
        else:
            st.info("No fruit detected")
            fruit = None

        if not fruit:
            fruit = manual_recipe

        recipe = RECIPES.get(fruit)

        if recipe:
            st.success(f"Recipe for {fruit}")
            st.write(recipe["title"])
            st.write(recipe["content"])

else:
    st.info("Click **Start Detection** to begin.")
