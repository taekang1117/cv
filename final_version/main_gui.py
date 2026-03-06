# main_gui.py - Bolt vs Nut classification with Web GUI
import asyncio
import base64
import json
import os
import sys
import threading
import time
import queue
import cv2
import numpy as np
import pandas as pd
import pickle
from aiohttp import web
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set

# =========================
# Configuration
# =========================
WEBCAM_INDEX = 1
FRAME_W, FRAME_H = 960, 540
ROI_X, ROI_Y, ROI_W, ROI_H = 260, 90, 440, 360

BLUR_K      = 5
MORPH_K     = 5
OPEN_ITERS  = 2
CLOSE_ITERS = 2
MIN_AREA    = 800
MAX_AREA    = 40000

# Models
RF_MODEL_FILE  = "bolt_nut_model.pkl"
KNN_MODEL_FILE = "bolt_nut_knn.pkl"
SVM_MODEL_FILE = "bolt_nut_svm.pkl"

BOLT_COLOR  = (0, 200, 255)   # yellow-gold
NUT_COLOR   = (80, 80, 80)    # dark grey

HTTP_HOST = "0.0.0.0"
HTTP_PORT = 8000
VIDEO_FPS = 15
JPEG_QUALITY = 75

# =========================
# Vision Logic
# =========================
def load_model(filepath):
    if not os.path.exists(filepath):
        print(f"ERROR: {filepath} not found!")
        return None, None
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, tuple):
        model, scaler = data
    else:
        model, scaler = data, None
    print(f"Loaded: {filepath}")
    return model, scaler

def clamp_roi(x, y, w, h, W, H):
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return x, y, w, h

def morph_cleanup(mask):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_K, MORPH_K))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=OPEN_ITERS)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=CLOSE_ITERS)
    return mask

def capture_background_gray(cap, roi_rect, n=20):
    rx, ry, rw, rh = roi_rect
    acc = None
    for _ in range(n):
        ret, frame = cap.read()
        if not ret: continue
        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        roi = frame[ry:ry+rh, rx:rx+rw]
        g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
        g = cv2.GaussianBlur(g, (BLUR_K, BLUR_K), 0)
        acc = g if acc is None else acc + g
    return (acc / n).astype(np.uint8)

def get_object_mask(roi_bgr, bg_gray):
    g1 = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    g1 = cv2.GaussianBlur(g1, (BLUR_K, BLUR_K), 0)
    diff = cv2.absdiff(g1, bg_gray)
    _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = morph_cleanup(mask)
    return mask

def get_features_vector(cnt):
    area  = float(cv2.contourArea(cnt))
    perim = float(cv2.arcLength(cnt, True))
    if perim == 0: return None
    circularity = (4.0 * np.pi * area) / (perim * perim)
    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))
    solidity = area / hull_area if hull_area > 0 else 0
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio_invariant = float(max(w, h)) / (min(w, h) + 1e-9)
    return [area, aspect_ratio_invariant, circularity, solidity, perim]

def run_classifier(models, scalers, active_clf, feature_list):
    col_names = ['area', 'aspect_ratio', 'circularity', 'solidity', 'perimeter']
    X = pd.DataFrame(feature_list, columns=col_names)
    
    if active_clf == 3: # Voting Ensemble
        rf_model, knn_model, svm_model = models[:3]
        rf_scaler, knn_scaler, svm_scaler = scalers[:3]
        rf_probs = rf_model.predict_proba(X)
        X_knn = knn_scaler.transform(X)
        knn_probs = knn_model.predict_proba(X_knn)
        X_svm = svm_scaler.transform(X)
        svm_probs = svm_model.predict_proba(X_svm)
        avg_probs = (rf_probs + knn_probs + svm_probs) / 3.0
        preds = np.argmax(avg_probs, axis=1)
        return preds, avg_probs

    model = models[active_clf]
    scaler = scalers[active_clf]
    if scaler is not None:
        X = scaler.transform(X)
    preds = model.predict(X)
    probs = model.predict_proba(X)
    return preds, probs

# =========================
# Web Server State
# =========================
@dataclass
class AppState:
    phase: str = "idle" # idle | initializing | running
    bolts: int = 0
    nuts: int = 0
    decision: str = "NONE"
    active_clf_name: str = "Random Forest"

class SorterApp:
    def __init__(self, log_queue):
        self.log_q = log_queue
        self.lock = threading.Lock()
        self.state = AppState()
        self._latest_jpeg: Optional[bytes] = None
        self._run_enabled = False
        self._stop_event = threading.Event()
        
        # Hardware/Model
        self.cap = None
        self.models = []
        self.scalers = []
        self.bg_gray = None
        self.active_clf = 0
        self.clf_names = ["Random Forest", "KNN (k=5)", "SVM (RBF)", "Voting Ensemble"]

    def load_hardware(self):
        rf_m, rf_s = load_model(RF_MODEL_FILE)
        knn_m, knn_s = load_model(KNN_MODEL_FILE)
        svm_m, svm_s = load_model(SVM_MODEL_FILE)
        self.models  = [rf_m, knn_m, svm_m]
        self.scalers = [rf_s, knn_s, svm_s]
        
        if any(m is None for m in self.models):
            print("ERROR: Models missing. Run train_model.py first.")
            return False

        self.cap = cv2.VideoCapture(WEBCAM_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        if not self.cap.isOpened():
            print(f"ERROR: Could not open webcam {WEBCAM_INDEX}")
            return False
        return True

    def snapshot_state(self):
        with self.lock:
            return {
                "phase": self.state.phase,
                "bolts": self.state.bolts,
                "nuts": self.state.nuts,
                "decision": self.state.decision,
                "active_clf_name": self.state.active_clf_name
            }

    def _update_jpeg(self, frame):
        ok, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if ok:
            with self.lock:
                self._latest_jpeg = buf.tobytes()

    def get_latest_jpeg(self):
        with self.lock:
            return self._latest_jpeg

    def run(self):
        if not self.load_hardware():
            return

        roi_rect = clamp_roi(ROI_X, ROI_Y, ROI_W, ROI_H, FRAME_W, FRAME_H)
        rx, ry, rw, rh = roi_rect
        
        next_video_ts = 0
        video_period = 1.0 / VIDEO_FPS

        while not self._stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            full_bgr = cv2.resize(frame, (FRAME_W, FRAME_H))
            cv2.rectangle(full_bgr, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)
            roi_bgr = full_bgr[ry:ry + rh, rx:rx + rw]
            vis_roi = roi_bgr.copy()

            if self._run_enabled:
                if self.bg_gray is None:
                    self.state.phase = "initializing"
                    self.bg_gray = capture_background_gray(self.cap, roi_rect)
                    self.state.phase = "running"
                else:
                    mask = get_object_mask(roi_bgr, self.bg_gray)
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    feats_list = []
                    bboxes = []
                    for cnt in contours:
                        if MIN_AREA < cv2.contourArea(cnt) < MAX_AREA:
                            vec = get_features_vector(cnt)
                            if vec:
                                feats_list.append(vec)
                                bboxes.append(cv2.boundingRect(cnt))
                    
                    bolts_now, nuts_now = 0, 0
                    if feats_list:
                        preds, probs = run_classifier(self.models, self.scalers, self.active_clf, feats_list)
                        for i, label in enumerate(preds):
                            x, y, w, h = bboxes[i]
                            conf = max(probs[i]) * 100
                            if label == 1:
                                bolts_now += 1
                                color, text = BOLT_COLOR, f"BOLT {conf:.0f}%"
                            else:
                                nuts_now += 1
                                color, text = NUT_COLOR, f"NUT {conf:.0f}%"
                            cv2.rectangle(vis_roi, (x, y), (x + w, y + h), color, 2)
                            cv2.putText(vis_roi, text, (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    with self.lock:
                        self.state.bolts = bolts_now
                        self.state.nuts = nuts_now
                        self.state.decision = f"B:{bolts_now} N:{nuts_now}"
                    
                    full_bgr[ry:ry + rh, rx:rx + rw] = vis_roi

            now = time.time()
            if now >= next_video_ts:
                self._update_jpeg(full_bgr)
                next_video_ts = now + video_period
            
            time.sleep(0.001)

    def cmd_start(self): 
        self._run_enabled = True
        self.state.phase = "running"
    def cmd_stop(self): 
        self._run_enabled = False
        self.state.phase = "idle"
    def cmd_capture_bg(self):
        self.bg_gray = None
    def cmd_switch_clf(self, idx):
        if 0 <= idx < 4:
            self.active_clf = idx
            self.state.active_clf_name = self.clf_names[idx]

# =========================
# Web GUI HTML
# =========================
UI_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Bolt vs Nut Dashboard</title>
  <style>
    body { font-family: 'Inter', system-ui, -apple-system, sans-serif; margin: 0; background:#0b0f14; color:#e6edf3; }
    .wrap { display:grid; grid-template-columns: 1.6fr 1.1fr; gap: 16px; padding: 20px; height: 100vh; box-sizing: border-box; }
    .card { background:#121923; border:1px solid #1f2a37; border-radius: 16px; padding: 20px; box-shadow: 0 4px 20px rgba(0,0,0,0.3); }
    #canvas { width: 100%; height: auto; background:#000; border-radius: 12px; }
    .row { display:flex; gap: 16px; margin-bottom: 16px; }
    .row .card { flex:1; }
    .big { font-size: 42px; font-weight: 800; color: #fff; }
    .label { opacity: .7; font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px; }
    .phase { font-weight: 700; color: #4ade80; }
    .btns { display:flex; gap: 10px; flex-wrap: wrap; margin-top: 15px; }
    button { padding: 12px 18px; border-radius: 10px; border: 1px solid #2b3b4f; background:#1b2533; color:#e0e6ed; cursor:pointer; font-weight: 600; transition: all 0.2s; }
    button:hover { background:#253245; border-color: #3b82f6; }
    button.active { background:#3b82f6; color:#fff; border-color: #3b82f6; }
    #log { height: 200px; overflow:auto; font-family: 'JetBrains Mono', monospace; font-size: 11px; white-space: pre-wrap; background:#0a1017; border-radius: 10px; border:1px solid #1f2a37; padding: 12px; margin-top:10px; color: #8b949e; }
    .muted { opacity:.6; font-size: 12px; }
    .status-capsule { background: #1f2a37; padding: 4px 12px; border-radius: 20px; font-size: 12px; border: 1px solid #2d3b4f; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card" style="display:flex; flex-direction:column; justify-content:center;">
      <canvas id="canvas" width="960" height="540"></canvas>
      <div style="display:flex; justify-content:space-between; margin-top:15px">
        <div class="status-capsule" id="wsStat">Websocket: Connecting...</div>
        <div class="status-capsule">Classifier: <span id="active_clf" style="color:#3b82f6; font-weight:bold">Random Forest</span></div>
      </div>
    </div>

    <div style="display:flex; flex-direction:column; gap:16px overflow:auto;">
      <div class="row">
        <div class="card">
          <div class="label">Bolts Detected</div>
          <div class="big" id="bolts">0</div>
        </div>
        <div class="card">
          <div class="label">Nuts Detected</div>
          <div class="big" id="nuts">0</div>
        </div>
      </div>

      <div class="card">
        <div class="label">System State</div>
        <div class="big phase" id="phase" style="font-size:32px">IDLE</div>
        <div class="muted">Live Decision: <span id="decision">NONE</span></div>
      </div>

      <div class="card">
        <div class="label">Machine Control</div>
        <div class="btns">
          <button id="startBtn">START</button>
          <button id="stopBtn">STOP</button>
          <button id="bgBtn">RESET BG</button>
        </div>
        <div class="label" style="margin-top:20px">Classification Model</div>
        <div class="btns">
          <button class="clf-btn active" data-idx="0">RF</button>
          <button class="clf-btn" data-idx="1">KNN</button>
          <button class="clf-btn" data-idx="2">SVM</button>
          <button class="clf-btn" data-idx="3">ENSEMBLE</button>
        </div>
      </div>

      <div class="card" style="flex:1">
        <div class="label">System Log</div>
        <div id="log"></div>
      </div>
    </div>
  </div>

<script>
(function(){
  const $ = id => document.getElementById(id);
  const logEl = $('log');
  const canvas = $('canvas');
  const ctx = canvas.getContext('2d');
  const img = new Image();

  function addLog(line){
    logEl.textContent += `[${new Date().toLocaleTimeString()}] ${line}\\n`;
    logEl.scrollTop = logEl.scrollHeight;
  }

  const ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onopen = () => { $('wsStat').textContent = 'Websocket: Connected'; addLog('Connected to system'); };
  ws.onclose = () => { $('wsStat').textContent = 'Websocket: Disconnected'; addLog('Disconnected'); };
  
  ws.onmessage = ev => {
    const msg = JSON.parse(ev.data);
    if(msg.type === 'state'){
      const d = msg.data;
      $('phase').textContent = d.phase.toUpperCase();
      $('bolts').textContent = d.bolts;
      $('nuts').textContent = d.nuts;
      $('decision').textContent = d.decision;
      $('active_clf').textContent = d.active_clf_name;
    } else if(msg.type === 'log') addLog(msg.line);
  };

  const videoWs = new WebSocket(`ws://${location.host}/ws/video`);
  videoWs.binaryType = 'arraybuffer';
  videoWs.onmessage = ev => {
    const blob = new Blob([ev.data], {type: 'image/jpeg'});
    const url = URL.createObjectURL(blob);
    img.src = url;
    img.onload = () => { ctx.drawImage(img, 0, 0, canvas.width, canvas.height); URL.revokeObjectURL(url); };
  };

  const send = (type, data) => ws.send(JSON.stringify({type, ...data}));
  $('startBtn').onclick = () => send('cmd', {cmd: 'start'});
  $('stopBtn').onclick = () => send('cmd', {cmd: 'stop'});
  $('bgBtn').onclick = () => send('cmd', {cmd: 'capture_bg'});
  
  document.querySelectorAll('.clf-btn').forEach(btn => {
    btn.onclick = () => {
      document.querySelectorAll('.clf-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      send('cmd', {cmd: 'switch_clf', idx: parseInt(btn.dataset.idx)});
    };
  });
})();
</script>
</body>
</html>"""

# =========================
# Web Handlers
# =========================
async def index(request):
    return web.Response(text=UI_HTML, content_type="text/html")

async def ws_handler(request):
    app = request.app
    sorter = app["sorter"]
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    app["clients"].add(ws)
    
    async def loop():
        while not ws.closed:
            try: await ws.send_str(json.dumps({"type": "state", "data": sorter.snapshot_state()}))
            except: break
            await asyncio.sleep(0.1)
    
    task = asyncio.create_task(loop())
    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                data = json.loads(msg.data)
                if data.get("type") == "cmd":
                    cmd = data.get("cmd")
                    if cmd == "start": sorter.cmd_start()
                    elif cmd == "stop": sorter.cmd_stop()
                    elif cmd == "capture_bg": sorter.cmd_capture_bg()
                    elif cmd == "switch_clf": sorter.cmd_switch_clf(data.get("idx"))
    finally:
        task.cancel()
        app["clients"].discard(ws)
    return ws

async def ws_video_handler(request):
    sorter = request.app["sorter"]
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    while not ws.closed:
        jpg = sorter.get_latest_jpeg()
        if jpg:
            try: await ws.send_bytes(jpg)
            except: break
        await asyncio.sleep(1.0/VIDEO_FPS)
    return ws

def main():
    log_q = queue.Queue()
    sorter = SorterApp(log_q)
    threading.Thread(target=sorter.run, daemon=True).start()
    
    app = web.Application()
    app["sorter"] = sorter
    app["clients"] = set()
    app.router.add_get("/", index)
    app.router.add_get("/ws", ws_handler)
    app.router.add_get("/ws/video", ws_video_handler)
    
    print(f"Server started at http://{HTTP_HOST}:{HTTP_PORT}")
    web.run_app(app, host=HTTP_HOST, port=HTTP_PORT)

if __name__ == "__main__":
    main()
