import json
import time
import cv2
import numpy as np
import streamlit as st
mode = st.sidebar.radio("Source mode", ["File/Webcam", "CARLA"], index=0)

# Only import CARLA when actually selected (important for Streamlit Cloud)
if mode == "CARLA":
    import carla

# ------------------ YOLO + device ------------------
YOLO_AVAILABLE = True
DEVICE = "cpu"
try:
    import torch
    DEVICE = 0 if torch.cuda.is_available() else "cpu"
    from ultralytics import YOLO
except Exception:
    YOLO_AVAILABLE = False

st.set_page_config(page_title="Intersection Traffic Monitoring (CARLA Live)", layout="wide")

# ================= UI: SOURCE ========================
st.sidebar.header("Source")
mode = st.sidebar.radio("Source mode", ["File/Webcam", "CARLA"], index=1)
use_camera = st.sidebar.checkbox("Use webcam", value=False, disabled=(mode=="CARLA"))
uploaded = None
if mode == "File/Webcam":
    uploaded = st.sidebar.file_uploader("Upload a video", type=["mp4","avi","mov","mkv"])

# ================= UI: DETECTION =====================
st.sidebar.markdown("---")
st.sidebar.header("Detection")
enable_detection = st.sidebar.checkbox("Enable YOLO detection", value=True)
vehicle_only = st.sidebar.checkbox("Vehicles only (car/bus/truck/motorcycle)", value=True)
show_dets = st.sidebar.checkbox("Show detections (boxes + centroids)", value=True)
confidence = st.sidebar.slider("Confidence", 0.1, 0.9, 0.25, 0.05)
imgsz = st.sidebar.slider("YOLO image size", 640, 1536, 1280, 32)
use_large = st.sidebar.checkbox("Use larger model (yolov8l)", value=False)

# ================= UI: LANE ROIs =====================
st.sidebar.markdown("---")
st.sidebar.header("Lanes (image-normalized polygons)")
st.sidebar.caption("Edit polygons in [0..1]. They’re mapped to pixels at runtime.")
DEFAULT_LANES_JSON = json.dumps({
    "north":  {"poly":[[0.10,0.00],[0.90,0.00],[0.60,0.40],[0.40,0.40]], "group":"NS"},
    "south":  {"poly":[[0.10,1.00],[0.90,1.00],[0.60,0.60],[0.40,0.60]], "group":"NS"},
    "east":   {"poly":[[1.00,0.10],[1.00,0.90],[0.60,0.60],[0.60,0.40]], "group":"EW"},
    "west":   {"poly":[[0.00,0.10],[0.00,0.90],[0.40,0.60],[0.40,0.40]], "group":"EW"}
}, indent=2)
lanes_json = st.sidebar.text_area("Lane polygons JSON", value=DEFAULT_LANES_JSON, height=220)

# ================= UI: SIGNAL PLAN ===================
st.sidebar.markdown("---")
st.sidebar.header("Signal Plan (cycle)")
green_ns = st.sidebar.number_input("NS Green (s)", min_value=5, max_value=120, value=20, step=5)
yellow_ns = st.sidebar.number_input("NS Yellow (s)", min_value=3, max_value=15, value=4, step=1)
green_ew = st.sidebar.number_input("EW Green (s)", min_value=5, max_value=120, value=20, step=5)
yellow_ew = st.sidebar.number_input("EW Yellow (s)", min_value=3, max_value=15, value=4, step=1)

st.title("🚦 Traffic Monitoring — Intersection (Fixed Camera)")

# ------------------ helpers -------------------------
VEH_CLASSES = {2,3,5,7}  # car, motorcycle, bus, truck

def parse_lanes(lanes_json, w, h):
    try:
        cfg = json.loads(lanes_json)
    except Exception:
        cfg = json.loads(DEFAULT_LANES_JSON)
    lanes_px = {}
    for name, item in cfg.items():
        pts = np.array([[int(x*w), int(y*h)] for x,y in item["poly"]], dtype=np.int32)
        group = item.get("group","NS" if name in ("north","south") else "EW")
        lanes_px[name] = {"poly": pts, "group": group}
    return lanes_px

def which_lane(lanes_px, cx, cy):
    p = (cx, cy)
    for name, item in lanes_px.items():
        if cv2.pointPolygonTest(item["poly"], p, False) >= 0:
            return name
    return None

def draw_lane_overlays(frame, lanes_px, state_by_group):
    overlay = frame.copy()
    color_map = {"GREEN": (0,255,0), "YELLOW": (0,255,255), "RED": (0,0,255)}
    for name, item in lanes_px.items():
        group = item["group"]
        state = state_by_group[group]
        color = color_map[state]
        cv2.fillPoly(overlay, [item["poly"]], color)
        M = item["poly"].mean(axis=0).astype(int)
        cv2.putText(overlay, f"{name.upper()} [{state[0]}]", tuple(M),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(overlay, f"{name.upper()} [{state[0]}]", tuple(M),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)
    return cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

# ---------- Non-flicker traffic light FSM ----------
def _phase_sequence():
    return ["NS_G", "NS_Y", "EW_G", "EW_Y"]

def _phase_durations():
    return {"NS_G":green_ns, "NS_Y":yellow_ns, "EW_G":green_ew, "EW_Y":yellow_ew}

def _phase_to_state(phase):
    return {
        "NS_G":{"NS":"GREEN","EW":"RED"},
        "NS_Y":{"NS":"YELLOW","EW":"RED"},
        "EW_G":{"NS":"RED","EW":"GREEN"},
        "EW_Y":{"NS":"RED","EW":"YELLOW"},
    }[phase]

def init_signal_fsm():
    if "sig_phase" not in st.session_state:
        st.session_state.sig_phase = "NS_G"
        st.session_state.sig_started_at = time.monotonic()
        st.session_state.sig_dur_cache = _phase_durations()

def update_signal_fsm():
    durs = _phase_durations()
    if durs != st.session_state.sig_dur_cache:
        st.session_state.sig_started_at = time.monotonic()
        st.session_state.sig_dur_cache = durs
    now = time.monotonic()
    phase = st.session_state.sig_phase
    elapsed = now - st.session_state.sig_started_at
    if elapsed >= durs[phase]:
        seq = _phase_sequence()
        phase = seq[(seq.index(phase)+1)%len(seq)]
        st.session_state.sig_phase = phase
        st.session_state.sig_started_at = now
        elapsed = 0.0
    time_left = durs[phase] - elapsed
    return _phase_to_state(phase), time_left

# ------------------ video sources --------------------
def open_capture(src):
    if hasattr(src, "read") and not isinstance(src, str):
        with open("uploaded_video_tmp", "wb") as f:
            f.write(src.read())
        return cv2.VideoCapture("uploaded_video_tmp")
    return cv2.VideoCapture(src)

def carla_intersection_stream():
    import carla, numpy as np, time
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    bp = world.get_blueprint_library()
    cam_bp = bp.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", "1280")
    cam_bp.set_attribute("image_size_y", "720")
    cam_bp.set_attribute("fov", "90")

    cam_tf = carla.Transform(
        carla.Location(x=0.0, y=0.0, z=30.0),
        carla.Rotation(pitch=-82.0, yaw=0.0, roll=0.0)
    )
    camera = world.spawn_actor(cam_bp, cam_tf)

    holder = {"arr": None}
    def on_img(image):
        arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))[:, :, :3]
        holder["arr"] = arr.copy()
    camera.listen(on_img)

    try:
        while True:
            if holder["arr"] is not None:
                yield holder["arr"]
            else:
                time.sleep(0.01)
    finally:
        camera.stop(); camera.destroy()

# ------------------ app body -------------------------
init_signal_fsm()

frame_slot = st.empty()
c1, c2, c3, c4 = st.columns(4)
m1 = c1.metric("North", 0)
m2 = c2.metric("East",  0)
m3 = c3.metric("South", 0)
m4 = c4.metric("West",  0)
chart = st.line_chart(use_container_width=True)

model = None
if enable_detection and YOLO_AVAILABLE:
    try:
        model = YOLO("yolov8l.pt" if use_large else "yolov8n.pt")
    except Exception as e:
        st.warning(f"YOLO init failed: {e}")

if mode == "CARLA":
    stream = carla_intersection_stream()
    fps = 20.0
else:
    default_demo = 0 if use_camera else "traffic_demo.mp4"
    cap = open_capture(uploaded if uploaded is not None else default_demo)
    if not cap or not cap.isOpened():
        st.error("Could not open source."); st.stop()
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

counts = {"north":0, "east":0, "south":0, "west":0}
prev_lane_by_bucket = {}
history = []
last_plot = time.time()

while True:
    # ---------- frame retrieval ----------
    if mode == "CARLA":
        frame = next(stream, None)
        if frame is None:
            st.stop()
    else:
        ok, frame = cap.read()
        if not ok:
            break

    frame = np.ascontiguousarray(frame)
    h, w = frame.shape[:2]

    lanes_px = parse_lanes(lanes_json, w, h)
    state_by_group, time_left = update_signal_fsm()

    # ---------- YOLO detection ----------
    detections = []
    num_boxes = 0

    if enable_detection and model is not None:
        res = model.predict(
            frame,
            conf=confidence,
            imgsz=imgsz,
            classes=list(VEH_CLASSES) if vehicle_only else None,
            device=DEVICE,
            max_det=300,
            verbose=False
        )
        for r in res:
            if r.boxes is not None:
                boxes = r.boxes.xyxy.cpu().numpy()
                clss  = r.boxes.cls.cpu().numpy().astype(int)
                num_boxes += len(boxes)
                for (x1,y1,x2,y2), cls_id in zip(boxes, clss):
                    cx = int((x1+x2)/2)
                    cy = int((y1+y2)/2)
                    detections.append((cx,cy,(int(x1),int(y1),int(x2),int(y2))))

    # ---------- lane assignment ----------
    current = []
    for (cx,cy,box) in detections:
        lane = which_lane(lanes_px, cx, cy)
        if lane:
            key = (cx//50, cy//50)
            current.append((key, lane, box))

    pm = dict(prev_lane_by_bucket)
    new_prev = {}
    for key, lane, box in current:
        if key not in pm or pm[key] != lane:
            counts[lane] += 1
        new_prev[key] = lane
    prev_lane_by_bucket = new_prev

    # ---------- draw boxes ----------
    if show_dets:
        for _, _, (x1, y1, x2, y2) in current:
            cv2.rectangle(frame, (x1,y1), (x2,y2), (255,255,255), 2)

    # ---------- lane overlays ----------
    frame = draw_lane_overlays(frame, lanes_px, state_by_group)

    # ---------- dashboard header ----------
    hdr = f"NS: {state_by_group['NS']} | EW: {state_by_group['EW']} | Next: {int(time_left)}s | Detections: {num_boxes}"
    cv2.putText(frame, hdr, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3)
    cv2.putText(frame, hdr, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1)

    # ---------- Streamlit output ----------
    frame_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    m1.metric("North", counts["north"])
    m2.metric("East",  counts["east"])
    m3.metric("South", counts["south"])
    m4.metric("West",  counts["west"])

    if time.time() - last_plot > 0.5:
        history.append(counts.copy())
        chart.add_rows(history[-1:])
        last_plot = time.time()

    # ---------- throttle ----------
    time.sleep(max(0, 1.0/(fps+1e-6)))
