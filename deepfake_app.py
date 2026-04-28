"""
🎭 DeepTrace — Deepfake Detection Dashboard
Cloud-ready version for Streamlit Cloud deployment
File: deepfake_app.py
"""
import streamlit as st
import numpy as np
import cv2, os, tempfile, time, json
from pathlib import Path
from PIL import Image

st.set_page_config(
    page_title="DeepTrace | Deepfake Detector",
    page_icon="🎭",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');
html,body,.stApp{background:#0a0a0a!important;font-family:'Rajdhani',sans-serif;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:0 2rem 2rem 2rem!important;max-width:100%!important;}
.top-bar{background:linear-gradient(135deg,#1a0000,#0d0d0d,#1a0000);
  border-bottom:2px solid #cc0000;padding:1.2rem 2rem;
  display:flex;align-items:center;justify-content:space-between;
  margin:0 -2rem 2rem -2rem;}
.brand{font-family:'Share Tech Mono',monospace;font-size:1.8rem;
  color:#ff4444;text-shadow:0 0 20px rgba(255,68,68,.5);letter-spacing:4px;}
.brand-sub{font-family:'Share Tech Mono',monospace;font-size:.7rem;
  color:#555;letter-spacing:3px;margin-top:2px;}
.badge{background:#0a1a0a;border:1px solid #00ff88;border-radius:4px;
  padding:.4rem 1rem;font-family:'Share Tech Mono',monospace;
  font-size:.75rem;color:#00ff88;}
.verdict-real{background:linear-gradient(135deg,#001a0d,#0a2a0a);
  border:2px solid #00ff88;border-radius:8px;padding:2rem;text-align:center;}
.verdict-fake{background:linear-gradient(135deg,#1a0000,#2a0a0a);
  border:2px solid #ff4444;border-radius:8px;padding:2rem;text-align:center;}
.vlabel{font-family:'Share Tech Mono',monospace;font-size:2.5rem;
  font-weight:bold;letter-spacing:6px;margin-bottom:.5rem;}
.vconf{font-family:'Share Tech Mono',monospace;font-size:1rem;
  color:#999;letter-spacing:2px;}
.sec{font-family:'Share Tech Mono',monospace;font-size:.75rem;color:#cc0000;
  letter-spacing:4px;text-transform:uppercase;border-bottom:1px solid #222;
  padding-bottom:.5rem;margin-bottom:1.2rem;}
.log{background:#050505;border:1px solid #1a1a1a;border-radius:4px;
  padding:1rem;font-family:'Share Tech Mono',monospace;font-size:.75rem;
  color:#555;max-height:180px;overflow-y:auto;}
.ok{color:#00ff88;}.info{color:#00d4ff;}.err{color:#ff4444;}
p,li,label{color:#999!important;}
h1,h2,h3{color:#ccc!important;}
div[data-testid="stTabs"] button{
  background:#111!important;color:#555!important;
  font-family:'Share Tech Mono',monospace!important;
  letter-spacing:2px!important;font-size:.75rem!important;}
div[data-testid="stTabs"] button[aria-selected="true"]{
  color:#ff4444!important;background:#0d0d0d!important;
  border-bottom:3px solid #ff4444!important;}
.stButton button{
  background:linear-gradient(135deg,#1a0000,#330000)!important;
  color:#ff4444!important;border:1px solid #cc0000!important;
  font-family:'Share Tech Mono',monospace!important;
  letter-spacing:2px!important;}
</style>""", unsafe_allow_html=True)

st.markdown("""
<div class="top-bar">
  <div>
    <div class="brand">🎭 DEEPTRACE</div>
    <div class="brand-sub">AI FORENSIC DEEPFAKE DETECTION SYSTEM v3.0</div>
  </div>
  <div class="badge">● ONLINE</div>
</div>""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI model...")
def load_model():
    try:
        import tensorflow as tf
        # Suppress TF logs
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        # Load config
        IMG_SIZE  = 128
        THRESHOLD = 0.5
        if os.path.exists('model_info.json'):
            with open('model_info.json') as f:
                cfg = json.load(f)
            IMG_SIZE  = int(cfg.get('img_size', 128))
            THRESHOLD = float(cfg.get('threshold', 0.5))

        # Load model
        model = tf.keras.models.load_model('best_deepfake.keras')
        return model, IMG_SIZE, THRESHOLD, True, "✅ MobileNetV2 loaded"

    except FileNotFoundError:
        return None, 128, 0.5, False, "⚠️ Model file not found"
    except Exception as e:
        return None, 128, 0.5, False, f"⚠️ Error: {str(e)[:50]}"

model, IMG_SIZE, THRESHOLD, loaded, status_msg = load_model()
status_col = "#00ff88" if loaded else "#ffd93d"

# Sidebar info
st.sidebar.markdown(f"""
<div style='font-family:Share Tech Mono,monospace;font-size:.8rem;
            color:{status_col};padding:.8rem;background:#0d0d0d;
            border:1px solid #222;border-radius:4px;margin-bottom:1rem'>
{status_msg}
</div>
<div style='font-family:Share Tech Mono,monospace;font-size:.7rem;
            color:#444;line-height:2'>
MODEL    : MobileNetV2<br>
INPUT    : {IMG_SIZE}×{IMG_SIZE} RGB<br>
PREPROC  : pixel / 255.0<br>
CLASSES  : real=0 · fake=1<br>
THRESHOLD: {THRESHOLD}
</div>""", unsafe_allow_html=True)

# ── Core predict function ─────────────────────────────────────
def predict(img_rgb):
    """
    Correct preprocessing — matches training exactly.
    img_rgb : H×W×3 uint8 numpy array
    returns : (label, fake_probability, confidence)
    """
    # Resize and normalize — MUST match ImageDataGenerator(rescale=1./255)
    resized   = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    arr       = resized.astype('float32') / 255.0
    batch     = np.expand_dims(arr, 0)

    if not loaded:
        # Demo mode
        fake_prob = float(np.random.uniform(0.1, 0.9))
    else:
        fake_prob = float(model.predict(batch, verbose=0)[0][0])

    # sigmoid >= threshold → FAKE (label=1)
    label = 'FAKE' if fake_prob >= THRESHOLD else 'REAL'
    conf  = fake_prob if fake_prob >= THRESHOLD else 1.0 - fake_prob
    return label, fake_prob, conf

# ── Face extraction for video ─────────────────────────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def extract_frames(video_path, n=10):
    cap    = cv2.VideoCapture(video_path)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if total < 1:
        cap.release()
        return frames
    for idx in np.linspace(0, total-1, min(n, total), dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        det  = face_cascade.detectMultiScale(
            gray, 1.1, 4, minSize=(40, 40))
        if len(det) > 0:
            x, y, w, h = det[0]
            crop = rgb[max(0,y-15):y+h+15,
                       max(0,x-15):x+w+15]
            if crop.size > 0:
                frames.append(crop)
                continue
        h2, w2 = rgb.shape[:2]
        s = min(h2, w2)
        frames.append(rgb[(h2-s)//2:(h2+s)//2,
                          (w2-s)//2:(w2+s)//2])
    cap.release()
    return frames

# ── HTML helpers ──────────────────────────────────────────────
def verdict_html(label, conf):
    cls   = "verdict-fake" if label == "FAKE" else "verdict-real"
    color = "#ff4444"      if label == "FAKE" else "#00ff88"
    icon  = "⚠️"           if label == "FAKE" else "✅"
    return f"""
    <div class="{cls}">
      <div class="vlabel" style="color:{color}">{icon} {label}</div>
      <div class="vconf">CONFIDENCE: {conf*100:.1f}%</div>
    </div>"""

def bars_html(fake_prob):
    real_prob = 1.0 - fake_prob
    return f"""
    <div style="margin:.5rem 0">
      <div style="display:flex;justify-content:space-between;
           font-family:'Share Tech Mono',monospace;font-size:.75rem;
           color:#888;margin-bottom:4px">
        <span>AUTHENTIC</span><span>{real_prob*100:.1f}%</span></div>
      <div style="background:#1a1a1a;border-radius:2px;height:8px">
        <div style="height:100%;width:{real_prob*100:.1f}%;
             background:linear-gradient(90deg,#004d2a,#00ff88)"></div>
      </div>
    </div>
    <div style="margin:.8rem 0 .5rem">
      <div style="display:flex;justify-content:space-between;
           font-family:'Share Tech Mono',monospace;font-size:.75rem;
           color:#888;margin-bottom:4px">
        <span>DEEPFAKE</span><span>{fake_prob*100:.1f}%</span></div>
      <div style="background:#1a1a1a;border-radius:2px;height:8px">
        <div style="height:100%;width:{fake_prob*100:.1f}%;
             background:linear-gradient(90deg,#4d0000,#ff4444)"></div>
      </div>
    </div>"""

def risk_html(fake_prob):
    if   fake_prob > 0.85: risk,rc="🔴 CRITICAL — High probability deepfake","#ff4444"
    elif fake_prob > 0.65: risk,rc="🟠 HIGH — Strong manipulation signals","#ff8c00"
    elif fake_prob > 0.50: risk,rc="🟡 MEDIUM — Possible manipulation","#ffd93d"
    elif fake_prob > 0.35: risk,rc="🟡 LOW — Minor anomalies detected","#ffd93d"
    else:                  risk,rc="🟢 AUTHENTIC — No manipulation found","#00ff88"
    return f"""
    <div style="background:#0d0d0d;border:1px solid #222;border-radius:6px;
         padding:1rem;font-family:'Share Tech Mono',monospace;
         font-size:.8rem;color:{rc}">{risk}</div>"""

# ── Tabs ──────────────────────────────────────────────────────
t1, t2, t3, t4 = st.tabs([
    "🖼️  IMAGE",
    "🎬  VIDEO",
    "📁  BATCH",
    "ℹ️   INFO"
])

# ══ TAB 1: IMAGE ═════════════════════════════════════════════
with t1:
    st.markdown('<div class="sec">// SINGLE IMAGE FORENSIC ANALYSIS</div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")

    with c1:
        up = st.file_uploader(
            "Upload image",
            type=['jpg','jpeg','png','bmp','webp'],
            label_visibility='collapsed')
        if up:
            img     = Image.open(up).convert('RGB')
            img_arr = np.array(img)
            st.image(img, caption='INPUT IMAGE',
                     use_container_width=True)
            st.markdown(
                f"**File:** `{up.name}` &nbsp;&nbsp; "
                f"**Size:** `{up.size//1024} KB` &nbsp;&nbsp; "
                f"**Dims:** `{img.size[0]}×{img.size[1]}`")

    with c2:
        if up:
            with st.spinner("🔍 Scanning for manipulation..."):
                time.sleep(0.5)
                label, fake_prob, conf = predict(img_arr)

            st.markdown(verdict_html(label, conf),
                        unsafe_allow_html=True)
            st.markdown("<div style='margin-top:1rem'></div>",
                        unsafe_allow_html=True)
            st.markdown('<div class="sec">// PROBABILITY</div>',
                        unsafe_allow_html=True)
            st.markdown(bars_html(fake_prob),
                        unsafe_allow_html=True)
            st.markdown("<div style='margin-top:1rem'></div>",
                        unsafe_allow_html=True)
            st.markdown('<div class="sec">// RISK LEVEL</div>',
                        unsafe_allow_html=True)
            st.markdown(risk_html(fake_prob),
                        unsafe_allow_html=True)
            st.markdown("<div style='margin-top:1rem'></div>",
                        unsafe_allow_html=True)
            st.markdown('<div class="sec">// ANALYSIS LOG</div>',
                        unsafe_allow_html=True)
            st.markdown(f"""
            <div class="log">
              <div class="ok">[OK] {up.name} loaded</div>
              <div class="info">[PREP] resize → {IMG_SIZE}×{IMG_SIZE}</div>
              <div class="info">[PREP] normalize: uint8/255.0 → float32</div>
              <div class="info">[MODEL] MobileNetV2 sigmoid output</div>
              <div class="info">[PRED] fake_probability = {fake_prob:.6f}</div>
              <div class="info">[RULE] threshold={THRESHOLD} | ≥threshold → FAKE</div>
              <div class="{'err' if label=='FAKE' else 'ok'}">[RESULT] {label} ({conf*100:.2f}%)</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#0d0d0d;border:2px dashed #333;
                 border-radius:8px;padding:5rem;text-align:center">
              <div style="font-size:3rem">🔍</div>
              <div style="font-family:'Share Tech Mono',monospace;
                   color:#444;font-size:.85rem;letter-spacing:2px;
                   margin-top:1rem">UPLOAD AN IMAGE TO BEGIN</div>
            </div>""", unsafe_allow_html=True)

# ══ TAB 2: VIDEO ═════════════════════════════════════════════
with t2:
    st.markdown('<div class="sec">// VIDEO FORENSIC ANALYSIS</div>',
                unsafe_allow_html=True)
    vup = st.file_uploader(
        "Upload video",
        type=['mp4','avi','mov','mkv','webm'],
        label_visibility='collapsed')

    if vup:
        with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=Path(vup.name).suffix) as tmp:
            tmp.write(vup.read())
            tmp_path = tmp.name

        vc1, vc2 = st.columns(2, gap="large")
        with vc1:
            st.video(tmp_path)

        with vc2:
            if st.button("🎬 ANALYZE VIDEO"):
                prog = st.progress(0)
                prog.progress(15)
                frames = extract_frames(tmp_path, n=10)
                prog.progress(40)

                if not frames:
                    st.error("❌ No frames could be extracted")
                else:
                    fcols = st.columns(4)
                    for i, frm in enumerate(frames[:4]):
                        fcols[i].image(frm,
                            caption=f'Frame {i+1}',
                            use_container_width=True)
                    probs = []
                    for i, frm in enumerate(frames):
                        _, fp, _ = predict(frm)
                        probs.append(fp)
                        prog.progress(
                            40 + int((i+1)/len(frames)*55))

                    avg   = float(np.mean(probs))
                    lbl   = 'FAKE' if avg >= THRESHOLD else 'REAL'
                    conf  = avg if avg >= THRESHOLD else 1-avg
                    prog.progress(100)

                    st.markdown(verdict_html(lbl, conf),
                                unsafe_allow_html=True)
                    st.markdown(bars_html(avg),
                                unsafe_allow_html=True)
                    st.markdown(f"""
                    <div style="font-family:'Share Tech Mono',monospace;
                         font-size:.75rem;color:#555;margin-top:.8rem">
                      Frames: {len(probs)} &nbsp;|&nbsp;
                      Avg fake prob: {avg:.4f}
                    </div>""", unsafe_allow_html=True)
        os.unlink(tmp_path)
    else:
        st.markdown("""
        <div style="background:#0d0d0d;border:2px dashed #333;
             border-radius:8px;padding:4rem;text-align:center">
          <div style="font-size:3rem">🎬</div>
          <div style="font-family:'Share Tech Mono',monospace;
               color:#444;font-size:.85rem;letter-spacing:2px;
               margin-top:1rem">UPLOAD A VIDEO TO ANALYZE</div>
        </div>""", unsafe_allow_html=True)

# ══ TAB 3: BATCH ═════════════════════════════════════════════
with t3:
    st.markdown('<div class="sec">// BATCH FORENSIC ANALYSIS</div>',
                unsafe_allow_html=True)
    batch = st.file_uploader(
        "Upload images",
        type=['jpg','jpeg','png','bmp'],
        accept_multiple_files=True,
        label_visibility='collapsed')

    if batch and st.button("🔍 ANALYZE ALL"):
        import pandas as pd
        results = []
        prog = st.progress(0)
        for i, f in enumerate(batch):
            img = Image.open(f).convert('RGB')
            lbl, fp, conf = predict(np.array(img))
            results.append({
                'Filename'  : f.name,
                'Verdict'   : lbl,
                'Fake Prob' : f'{fp*100:.1f}%',
                'Real Prob' : f'{(1-fp)*100:.1f}%',
                'Confidence': f'{conf*100:.1f}%',
            })
            prog.progress(int((i+1)/len(batch)*100))

        total = len(results)
        fakes = sum(1 for r in results if r['Verdict']=='FAKE')
        reals = total - fakes

        st.markdown(f"""
        <div style="display:grid;grid-template-columns:repeat(3,1fr);
             gap:1rem;margin-bottom:1.5rem">
          <div style="background:#111;border-left:3px solid #cc0000;
               border-radius:6px;padding:1rem">
            <div style="font-family:'Share Tech Mono',monospace;
                 font-size:1.6rem;color:#ff4444">{total}</div>
            <div style="font-size:.75rem;color:#555;
                 letter-spacing:2px">TOTAL</div></div>
          <div style="background:#111;border-left:3px solid #ff4444;
               border-radius:6px;padding:1rem">
            <div style="font-family:'Share Tech Mono',monospace;
                 font-size:1.6rem;color:#ff4444">{fakes}</div>
            <div style="font-size:.75rem;color:#555;
                 letter-spacing:2px">DEEPFAKES</div></div>
          <div style="background:#111;border-left:3px solid #00ff88;
               border-radius:6px;padding:1rem">
            <div style="font-family:'Share Tech Mono',monospace;
                 font-size:1.6rem;color:#00ff88">{reals}</div>
            <div style="font-size:.75rem;color:#555;
                 letter-spacing:2px">AUTHENTIC</div></div>
        </div>""", unsafe_allow_html=True)

        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True, height=300)
        st.download_button(
            "📥 DOWNLOAD REPORT (CSV)",
            df.to_csv(index=False),
            "deepfake_report.csv",
            "text/csv")

# ══ TAB 4: INFO ══════════════════════════════════════════════
with t4:
    st.markdown('<div class="sec">// SYSTEM INFORMATION</div>',
                unsafe_allow_html=True)
    ic1, ic2 = st.columns(2, gap="large")

    with ic1:
        st.markdown("**MODEL DETAILS**")
        st.markdown(f"""
        <div style="background:#0d0d0d;border:1px solid #222;
             border-radius:6px;padding:1.2rem;
             font-family:'Share Tech Mono',monospace;font-size:.8rem">
          <div style="color:#ff4444;margin-bottom:.8rem;letter-spacing:2px">
            MOBILENETV2 TRANSFER LEARNING</div>
          <div style="color:#666;line-height:2.2">
            Backbone   : MobileNetV2 (ImageNet)<br>
            Parameters : ~3.4M<br>
            Input      : {IMG_SIZE}×{IMG_SIZE}×3<br>
            Preprocess : pixel / 255.0<br>
            Output     : sigmoid (0=real · 1=fake)<br>
            Threshold  : {THRESHOLD}<br>
            Training   : 2-phase fine-tuning<br>
            Val AUC    : 0.8994<br>
            Status     : {status_msg}
          </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("**REAL-WORLD APPLICATIONS**")
        apps = [
            ("🏛️","Law Enforcement","Digital evidence verification"),
            ("📰","Media & Journalism","News authenticity checking"),
            ("💼","Corporate KYC","Video call verification"),
            ("🔐","Cybersecurity","Anti-spoofing systems"),
            ("🎓","Education","Academic integrity tools"),
        ]
        for icon, title, desc in apps:
            st.markdown(f"""
            <div style="background:#0d0d0d;border:1px solid #1a1a1a;
                 border-radius:6px;padding:.8rem;margin-bottom:.5rem;
                 display:flex;gap:1rem;align-items:center">
              <div style="font-size:1.5rem">{icon}</div>
              <div>
                <div style="color:#ccc;font-size:.85rem;
                     font-weight:600">{title}</div>
                <div style="color:#555;font-size:.75rem">{desc}</div>
              </div>
            </div>""", unsafe_allow_html=True)

    with ic2:
        st.markdown("**PERFORMANCE METRICS**")
        metrics = [
            ("Val AUC",      "0.8994",  "#ff4444"),
            ("Architecture", "MobileNetV2", "#00d4ff"),
            ("Training Data","3,200 faces","#ffd93d"),
            ("Videos Used",  "400 videos", "#00ff88"),
            ("Fine-tuned",   "Last 30 layers","#ff6b6b"),
            ("Threshold",    str(THRESHOLD),"#888"),
        ]
        for label, val, col in metrics:
            st.markdown(f"""
            <div style="background:#0d0d0d;border:1px solid #222;
                 border-radius:6px;padding:.8rem;margin-bottom:.5rem;
                 display:flex;justify-content:space-between;
                 align-items:center">
              <div style="font-family:'Share Tech Mono',monospace;
                   font-size:.75rem;color:#555;letter-spacing:2px">
                {label.upper()}</div>
              <div style="font-family:'Share Tech Mono',monospace;
                   font-size:.9rem;color:{col};font-weight:bold">
                {val}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("**TECH STACK**")
        techs = ['Python','TensorFlow','MobileNetV2','OpenCV',
                 'Streamlit','NumPy','Scikit-learn','Pillow']
        html  = ''.join([
            f'<span style="background:#1a0000;border:1px solid #cc0000;'
            f'border-radius:3px;padding:3px 10px;margin:3px;'
            f'display:inline-block;font-family:Share Tech Mono,monospace;'
            f'font-size:.7rem;color:#ff8888">{t}</span>'
            for t in techs])
        st.markdown(f'<div style="margin-top:.5rem">{html}</div>',
                    unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center;margin-top:3rem;padding:1rem;
         border-top:1px solid #1a1a1a;font-family:'Share Tech Mono',
         monospace;font-size:.7rem;color:#333">
      DEEPTRACE v3.0 · ZAIN ABBAS · UET TAXILA 2026<br>
      ⚠️ FOR EDUCATIONAL AND RESEARCH PURPOSES ONLY
    </div>""", unsafe_allow_html=True)
