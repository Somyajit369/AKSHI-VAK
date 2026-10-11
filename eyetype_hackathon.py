"""
EyeType v4 - webcam eye-gaze keyboard for motor-impaired users (TECHNOVA 2026, PS-45)

Triple blink / triple right head-turn -> emergency countdown; triple left -> last-dialled countdown.
Four rapid alternating left-right turns -> local siren + flashing SOS overlay only; no call is placed.
Install:  pip install opencv-contrib-python mediapipe==0.10.14 numpy Pillow pyttsx3 wordfreq pyautogui pyperclip pyserial screen-brightness-control
Run:      python eyetype_hackathon.py                    (normal, webcam gaze; call requests are dry-run)
          python eyetype_hackathon.py --mouse             (UI demo: mouse = gaze, S key = space)
          python eyetype_hackathon.py --live-emergency    (explicitly enable external emergency routing)
Keys:     Esc quit | C recalibrate | M keyboard/board | X cancel emergency + stop siren | K scan current page | L fill-light
No app subscription is built into core EyeType. Tracking, typing, prediction, speech, and siren work OFFLINE after installation.
Low light: screen fill-light, image enhancement, and scan mode. Emoji picker/composer use Pillow + the system emoji font (readable aliases are fallback).
Device/webhook/phone features may need APIs, network, or hardware.

Hands-free controls:
  * Hold gaze on a key ~1 s            -> types it (dwell)
  * Eyes closed 0.7-2.5 s on purpose   -> SPACE (normal blinks ignored)
  * Eyes closed 2.5 s+                 -> SPEAKS the sentence
  * Look at a suggestion box           -> accepts the predicted word
  * MENU board                         -> Quick phrases, personal phrases, smart room, caregiver, SOS
  * MY PHRASES                         -> locally saved editable phrases; double nod up/down to navigate/activate
  * K                                  -> page-aware switch scanning on any board

Features include auto-capitalisation + punctuation, adaptive dwell time, automatic drift correction,
personalizable communication board, switch/head access, smart-room APIs, SOS alarm with optional
phone alert, calibration quality rating, darkness/face-lost analysis, and keystroke-saving metrics.
"""
import cv2, time, json, math, os, csv, sys, threading, subprocess, re, tempfile, shutil
from collections import Counter
import numpy as np
import mediapipe as mp

APP_ROOT = os.path.dirname(os.path.abspath(sys.executable if getattr(sys, "frozen", False) else __file__))
if os.name == "nt":
    _profile = os.environ.get("APPDATA") or os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    APP_DATA_DIR = os.path.join(_profile, "EyeType")
elif sys.platform.startswith("linux"):
    _profile = os.environ.get("XDG_DATA_HOME") or os.path.join(os.path.expanduser("~"), ".local", "share")
    APP_DATA_DIR = os.path.join(_profile, "EyeType")
else:
    APP_DATA_DIR = APP_ROOT
try:
    os.makedirs(APP_DATA_DIR, exist_ok=True)
except OSError:
    APP_DATA_DIR = APP_ROOT


def app_data_path(filename):
    """Return a writable path, migrating a legacy beside-script file once."""
    target = os.path.join(APP_DATA_DIR, filename)
    legacy = os.path.join(APP_ROOT, filename)
    if target != legacy and not os.path.exists(target) and os.path.isfile(legacy):
        try:
            shutil.copy2(legacy, target)
        except OSError:
            pass
    return target


if getattr(sys, "frozen", False) and sys.stdout is None:
    try:
        sys.stdout = open(app_data_path("eyetype.log"), "a", encoding="utf-8", buffering=1)
        sys.stderr = sys.stdout
    except OSError:
        pass

# ---------------- settings ----------------
SCREEN_W_CM = 34.0      # measure your screen width -> used to convert error to degrees
DIST_CM = 55.0          # eye-to-screen distance
CAMERA_WIDTH, CAMERA_HEIGHT = 960, 540  # lower processing load than 720p while retaining gaze features
DWELL_KEY = 0.55        # quick eye typing; increases are still available for users who need them
DWELL_MIN, DWELL_MAX = 0.35, 1.6
LONG_CLOSE = 0.7        # eyes closed this long = space
SPEAK_CLOSE = 2.5       # eyes closed this long = speak
LOCKOUT = 0.18
MODEL_FILE = app_data_path("user_model.json")
METRICS_FILE = app_data_path("metrics.csv")
# Optional phone alert for CALL NURSE / SOS. Example (Telegram bot):
# "https://api.telegram.org/bot<TOKEN>/sendMessage?chat_id=<CHAT_ID>&text={msg}"
ALERT_URL = ""
# Caregiver integration: webhook can be an internet URL or a local-LAN receiver.
CARETAKER_WEBHOOK_URL = ""      # accepts POST JSON: {"event":"message","text":"..."}
CARETAKER_WEBHOOK_TOKEN = ""    # optional Bearer token expected by your webhook
CARETAKER_PHONE = ""            # optional phone for call/SMS; SMS also requires SMS_LIVE=True
CAREGIVER_AUTO_SEND = True      # forward the composed text after a 3 s pause when a webhook URL is set
MOUSE = "--mouse" in sys.argv
APP_BRAND = "Akshi-Vak"

# ---- emergency system ----
# Triggers: 3 deliberate blinks (each 0.25-0.7 s) within 4 s, OR turning the head to the right 3 times, OR the SOS tile.
# A 10 s countdown starts (look at CANCEL or press X to stop). Service tiles open an 8 s confirm/cancel screen.
# Real emergency calls/alerts require the explicit --live-emergency command-line flag.
# Default is demo-safe: show countdowns and local siren, but do not call, SMS, or send emergency webhooks.
EMERGENCY_LIVE = "--live-emergency" in sys.argv
EMERGENCY_COUNTDOWN = 10
EMERGENCY_NUMBERS = {"DEFAULT": "112", "AMBULANCE": "102", "POLICE": "100", "FIRE": "101",   # India shown - change for your country
                     "FAMILY": ""}                                                           # put the family/caregiver number here
HOME_ADDRESS = ""            # added to every alert message so helpers know where to go
BLINK_MIN, BLINK_WINDOW = 0.25, 4.0
YAW_THRESH, TURN_SIGN, TURN_WINDOW = 0.15, 1, 8.0   # if turning RIGHT does not register on your camera, set TURN_SIGN = -1
HEADSHAKE_TURNS, HEADSHAKE_WINDOW = 4, 2.5  # rapid left-right-left-right completed turns trigger siren-only alert
HEADSHAKE_COOLDOWN = 12.0
PITCH_UP_SIGN = -1     # invert if the room-page up/down nods are reversed on your camera
ROLL_LEFT_SIGN = -1    # invert if the room-page left/right head-tilts are reversed
POSE_THRESH = 0.075    # normalized nose-position change required for a nod
ROLL_THRESH_DEG = 13.0 # head roll angle required for a side tilt
ROOM_GESTURE_WINDOW = 5.0

# ---- real device control (optional) ----
HA_URL = ""                  # e.g. "http://192.168.1.20:8123"  (Home Assistant)
HA_TOKEN = ""                # Home Assistant long-lived access token
HA_ENTITIES = {"LIGHT": "light.living_room", "FAN": "switch.fan", "TV": "media_player.tv"}
HA_CLIMATE_ENTITY = "climate.living_room"  # Home Assistant climate entity for AC/thermostat
HA_TEMP_STEP = 1.0
DEVICE_URLS = {}             # or plain URLs, e.g. {"LIGHT": {"on": "http://192.168.1.50/on", "off": "http://192.168.1.50/off"}}
ROOM_CONTROL_URL = ""         # optional generic LAN bridge endpoint; POST JSON device/action/temperature
ROOM_CONTROL_TOKEN = ""       # optional Bearer token for the generic bridge
ROOM_CLIMATE_STATE_URL = ""   # optional GET endpoint returning {"temperature": 24}
ROOM_DEVICE_STATE_URL = ""    # optional GET endpoint returning {"LIGHT":"on","FAN":"off"}
AC_ON_URL = ""                # optional direct GET action URL for AC/cooling on
AC_OFF_URL = ""               # optional direct GET action URL for AC off
AC_SET_TEMP_URL = ""          # optional direct URL template; include {temperature} for a setpoint
LIGHT_ON_URL = ""             # optional direct GET action URL
LIGHT_OFF_URL = ""
FAN_ON_URL = ""
FAN_OFF_URL = ""
DIRECT_AC_TEMP = 24.0          # last requested direct-URL setpoint; prefer a live state URL when available
APP_CONFIG_FILE = app_data_path("eyetype_settings.json")
LAST_DIALED_FILE = app_data_path("last_dialed.json")
LAST_DIALED_FALLBACK = ""    # primary per-install saved call target (UI label: LAST NUMBER)
APP_CONFIG_KEYS = ("ROOM_CONTROL_URL", "ROOM_CONTROL_TOKEN", "ROOM_CLIMATE_STATE_URL", "ROOM_DEVICE_STATE_URL",
                   "AC_ON_URL", "AC_OFF_URL", "AC_SET_TEMP_URL", "LIGHT_ON_URL", "LIGHT_OFF_URL", "FAN_ON_URL", "FAN_OFF_URL",
                   "LAST_DIALED_FALLBACK")


def sync_direct_device_urls():
    for name, on_key, off_key in (("LIGHT", "LIGHT_ON_URL", "LIGHT_OFF_URL"), ("FAN", "FAN_ON_URL", "FAN_OFF_URL")):
        for action, key in (("on", on_key), ("off", off_key)):
            url = globals().get(key, "")
            if url:
                DEVICE_URLS.setdefault(name, {})[action] = url


def load_app_config():
    try:
        with open(APP_CONFIG_FILE, encoding="utf-8") as stream:
            saved = json.load(stream)
        for key in APP_CONFIG_KEYS:
            value = saved.get(key)
            if isinstance(value, str):
                globals()[key] = value
        sync_direct_device_urls()
    except FileNotFoundError:
        pass
    except Exception as ex:
        print(f"Could not load {APP_CONFIG_FILE}: {ex}")


def set_app_config(key, value):
    if key not in APP_CONFIG_KEYS:
        return False, "Unknown setting"
    globals()[key] = value
    binding = {"LIGHT_ON_URL": ("LIGHT", "on"), "LIGHT_OFF_URL": ("LIGHT", "off"),
               "FAN_ON_URL": ("FAN", "on"), "FAN_OFF_URL": ("FAN", "off")}.get(key)
    if binding and not value:
        urls = DEVICE_URLS.get(binding[0], {})
        urls.pop(binding[1], None)
        if not urls:
            DEVICE_URLS.pop(binding[0], None)
    sync_direct_device_urls()
    try:
        temp_path = APP_CONFIG_FILE + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as stream:
            if os.name == "posix":
                os.chmod(temp_path, 0o600)
            json.dump({name: globals().get(name, "") for name in APP_CONFIG_KEYS}, stream, indent=2)
        os.replace(temp_path, APP_CONFIG_FILE)
        if os.name == "posix":
            os.chmod(APP_CONFIG_FILE, 0o600)
        return True, "Setting saved on this computer"
    except Exception as ex:
        return False, f"Could not save settings: {ex}"


load_app_config()

# ---- phone-call handoff / optional GSM SMS ----
GSM_PORT = ""                # e.g. "COM5" or "/dev/ttyUSB0": a SIM800L/SIM900 board or USB GSM dongle with a SIM card
SMS_LIVE = False              # Set True only when you explicitly want this configured modem to send SMS
GSM_BAUD = 9600

# ---- low light ----
DARK_ON, DARK_OFF = 60, 150  # camera brightness (0-255) that switches the fill-light mode on / back off
ENHANCE_BELOW = 90           # frames darker than this are brightened (gamma + CLAHE) before face tracking
SCAN_STEP = 1.0              # seconds each row/key is highlighted in scanning mode
SIREN_SECONDS = 10           # length of the laptop-generated alarm
FILL = {"on": False}         # shared flag: bright 'fill light' screen is on

# ---------------- word prediction ----------------
BASE_WORDS = ("the be to of and a in that have i it for not on with he as you do at this but his by from they we say her she or "
              "an will my one all would there their what so up out if about who get which go me when make can like time no just him "
              "know take people into year your good some could them see other than then now look only come its over think also back "
              "after use two how our work first well way even new want because any these give day most us need water help please "
              "thank hungry pain doctor nurse family love sleep tired cold hot light door window music tv call phone home").split()

SEED_PHRASES = ["i need water", "i am hungry", "please call the nurse", "i am in pain", "thank you very much",
                "i love you", "how are you", "good morning", "i want to sleep", "please turn on the light",
                "can you help me", "i feel cold", "i feel hot", "please open the window", "where is my family",
                "i need the doctor", "please call my family", "i am tired", "i want to listen to music",
                "please turn off the light", "please change the channel", "i want to sit up"]


class Predictor:
    def __init__(self, path=MODEL_FILE):
        self.path = path
        try:
            from wordfreq import top_n_list
            words = [w for w in top_n_list("en", 20000) if w.isalpha() and (len(w) > 1 or w in "ia")]
        except Exception:
            words = BASE_WORDS
        self.words = words
        self.base = {w: 0.5 / (i + 20) for i, w in enumerate(words)}
        self.common = words[:40]
        self.uni, self.bi = Counter(), {}
        self._cache = {}
        if os.path.exists(path):
            d = json.load(open(path))
            self.uni = Counter(d.get("uni", {}))
            self.bi = {k: Counter(v) for k, v in d.get("bi", {}).items()}
        else:
            for p in SEED_PHRASES:
                prev = ""
                for w in p.split():
                    self.learn(prev, w, 0.5)
                    prev = w

    def learn(self, prev, word, weight=1.0):
        word, prev = word.lower().strip(".,?!"), prev.lower().strip(".,?!")
        if not word:
            return
        self.uni[word] += weight
        self.bi.setdefault(prev, Counter())[word] += weight
        self._cache.clear()

    def suggest(self, prefix, prev, k=4):
        key = (prefix, prev)
        if key in self._cache:
            return self._cache[key]
        bi = self.bi.get(prev, {})
        if prefix:
            cands = [w for w in self.words if w.startswith(prefix) and w != prefix]
            cands += [w for w in self.uni if w.startswith(prefix) and w != prefix]
        else:
            cands = list(bi.keys()) + self.common
        sc = {w: self.base.get(w, 0.0) + 5 * self.uni.get(w, 0) + 20 * bi.get(w, 0) for w in set(cands)}
        out = sorted(sc, key=sc.get, reverse=True)[:k]
        self._cache[key] = out
        return out

    def save(self):
        json.dump({"uni": dict(self.uni), "bi": {k: dict(v) for k, v in self.bi.items()}}, open(self.path, "w"))


# ---------------- gaze tracking ----------------
EYES = [(33, 133, 159, 145, 468, [33, 160, 158, 133, 153, 144]),
        (362, 263, 386, 374, 473, [362, 385, 387, 263, 373, 380])]


class Tracker:
    def __init__(self):
        self.fm = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                                  min_detection_confidence=0.6, min_tracking_confidence=0.6)
        self.last_eye = np.zeros(2, dtype=float)

    def process(self, frame):
        h, w = frame.shape[:2]
        res = self.fm.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0].landmark
        P = lambda i: np.array([lm[i].x * w, lm[i].y * h])
        hs, vs, ears, weights = [], [], [], []
        for L, R, U, D, I, E in EYES:
            l, r, u, d, c = P(L), P(R), P(U), P(D), P(I)
            axis = r - l
            n = np.linalg.norm(axis) + 1e-6
            ax = axis / n
            perp = np.array([-ax[1], ax[0]])
            p = [P(i) for i in E]
            ear = (np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])) / (2 * np.linalg.norm(p[0] - p[3]) + 1e-6)
            ears.append(ear)
            horizontal = float((c - (l + r) / 2) @ ax / n)
            vertical = float((c - (u + d) / 2) @ perp / n)
            # Reject obvious iris-landmark jumps; a blink can still use the last
            # valid pupil position for gaze while EAR continues to detect it.
            if np.isfinite(horizontal) and np.isfinite(vertical) and abs(horizontal) <= 0.55 and abs(vertical) <= 0.45:
                hs.append(horizontal); vs.append(vertical)
                weights.append(max(0.05, min(ear, 0.35)))
        iris_quality = len(hs) / 2.0
        if hs:
            if len(hs) == 2 and (abs(hs[0] - hs[1]) > 0.18 or abs(vs[0] - vs[1]) > 0.16):
                # When one eye's iris landmark jumps, trust the more-open eye
                # for this frame instead of averaging two conflicting points.
                best = int(np.argmax(weights))
                self.last_eye = np.array([hs[best], vs[best]])
                iris_quality = 0.5
            else:
                self.last_eye = np.array([np.average(hs, weights=weights), np.average(vs, weights=weights)])
        nose = np.array([lm[1].x, lm[1].y], dtype=float)
        scale = lm[454].x - lm[234].x
        feat = np.array([self.last_eye[0], self.last_eye[1], nose[0], nose[1], scale])
        face_h = abs(lm[152].y - lm[10].y) + 1e-6
        pitch = (lm[1].y - lm[10].y) / face_h
        eye_dx = lm[263].x - lm[33].x
        eye_dy = lm[263].y - lm[33].y
        roll = math.degrees(math.atan2(eye_dy, eye_dx))
        return {"feat": feat, "ear": float(np.mean(ears)), "iris_quality": iris_quality,
                "yaw": (lm[1].x - lm[234].x) / (lm[454].x - lm[234].x + 1e-6),
                "pitch": pitch, "roll": roll}


def poly(x):
    # Low-order surface: the old full 5-D quadratic had 21 coefficients but
    # only 9 distinct calibration targets, so it could overfit badly.
    h, v, face_x, face_y, scale = x
    return np.array([1.0, h, v, face_x, face_y, scale, h * h, h * v, v * v])


class Mapper:
    def fit(self, F, Y, lam=0.12):
        F = np.array(F)
        self.mu = np.median(F, axis=0)
        # Don't amplify tiny head-motion/landmark jitter into a full feature.
        self.sd = np.maximum(F.std(0), np.array([0.04, 0.04, 0.015, 0.015, 0.02]))
        A = np.array([poly((f - self.mu) / self.sd) for f in F])
        reg = lam * np.eye(A.shape[1]); reg[0, 0] = 0.0
        self.W = np.linalg.solve(A.T @ A + reg, A.T @ np.array(Y))

    def predict(self, f):
        return poly((f - self.mu) / self.sd) @ self.W


def run_targets(win, cam, trk, W, H, pts, title, settle=0.8, collect=1.6):
    out = []
    for px, py in pts:
        t0, samples, failed_reads = time.time(), [], 0
        while True:
            ok, frame = cam.read()
            if not ok:
                failed_reads += 1
                if failed_reads >= 30:
                    print("Camera stopped returning frames during calibration.")
                    break
                continue
            failed_reads = 0
            f = cv2.flip(frame, 1)
            r = trk.process(enhance(f, float(f[::8, ::8].mean())))
            t = time.time() - t0
            canvas = np.full((H, W, 3), 235 if FILL["on"] else 0, np.uint8)       # bright screen = light for the face
            rad = int(32 - 22 * min(1, t / (settle + collect)))
            cv2.circle(canvas, (int(px * W), int(py * H)), rad, (0, 0, 255) if FILL["on"] else (0, 255, 255), -1)
            cv2.circle(canvas, (int(px * W), int(py * H)), 4, (0, 0, 0), -1)
            cv2.putText(canvas, title, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (60, 60, 60) if FILL["on"] else (200, 200, 200), 2)
            eye_count = int(round(2 * r.get("iris_quality", 0))) if r else 0
            cv2.putText(canvas, f"IRIS TRACKING: {eye_count}/2 eyes", (30, H - 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (30, 130, 30) if eye_count == 2 else (0, 80, 230), 2, cv2.LINE_AA)
            cv2.imshow(win, canvas)
            if cv2.waitKey(1) == 27:
                sys.exit()
            if t > settle and r and r["ear"] > 0.15 and r.get("iris_quality", 0) > 0:
                samples.append(r)
            if t > settle + collect:
                break
        out.append((np.array([px, py]), samples))
    return out


def calibrate(win, cam, trk, W, H):
    m = 0.1
    pts = [(x, y) for y in (m, 0.5, 1 - m) for x in (m, 0.5, 1 - m)]
    data = run_targets(win, cam, trk, W, H, pts, "AKSHI-VAK | Follow each dot with your eyes")
    F, Y, ears = [], [], []
    for tgt, ss in data:
        if len(ss) < 8:
            print(f"[calibration] target {tuple(tgt)} had only {len(ss)} stable iris samples; repeat with eyes visible and head still")
            return None, None, None
        # One robust center per target avoids treating hundreds of adjacent
        # frames as independent evidence and suppresses isolated tracking jumps.
        F.append(np.median(np.array([s["feat"] for s in ss]), axis=0))
        Y.append(tgt)
        ears.extend(s["ear"] for s in ss)
    if len(F) != len(pts):
        return None, None, None
    mapper = Mapper()
    mapper.fit(F, Y)
    ear_thr = 0.65 * float(np.median(ears))
    vpts = [(0.5, 0.5), (0.25, 0.3), (0.75, 0.3), (0.25, 0.75), (0.75, 0.75)]
    val = run_targets(win, cam, trk, W, H, vpts, "Validation: look at the dots", collect=1.2)
    errs = []
    for tgt, ss in val:
        if len(ss) < 5:
            print(f"[calibration] validation target {tuple(tgt)} was not tracked; check reflection/lighting and recalibrate")
            return None, None, None
        pred = np.median([mapper.predict(s["feat"]) for s in ss], axis=0)
        errs.append(np.linalg.norm((pred - tgt) * [W, H]))
    err_px = float(np.median(errs)) if errs else 999
    err_deg = math.degrees(math.atan(err_px * (SCREEN_W_CM / W) / DIST_CM))
    print(f"[calibration] gaze error {err_px:.0f}px = {err_deg:.2f} deg | eyes-closed threshold {ear_thr:.3f}")
    return mapper, ear_thr, err_deg


def quality(deg):
    if deg < 3: return "GOOD", (0, 200, 0)
    if deg < 5: return "OK", (0, 200, 255)
    return "POOR - press C", (0, 0, 255)


# ---------------- layouts ----------------
ROWS = [[(c, 1) for c in "QWERTYUIOP"], [(c, 1) for c in "ASDFGHJKL"],
        [(c, 1) for c in "ZXCVBNM"] + [("DEL", 1.5), ("MENU", 1.5)],
        [("SPACE", 3), (".", 1), ("?", 1), ("SMS", 1), ("SPEAK", 2), ("CLEAR", 2)]]

BOARD = [["I NEED WATER", "I AM IN PAIN", "CALL NURSE"],
         ["THANK YOU", "I AM HUNGRY", "TOILET PLEASE"],
         ["MY PHRASES", "LIGHT", "FAN"],
         ["SOS", "CALL FAMILY", "KEYBOARD"],
         ["NUMBERS", "SYMBOLS", "EMOJIS"],
         ["CONTROLS", "ROOM", "CURSOR"],
         ["GESTURES", "BIG ABC", "AUTO SHARE"],
         ["SEND TEXT", "CALL CARETAKER", "API SETUP"]]
PHRASES = {"I NEED WATER": "I need water.", "I AM IN PAIN": "I am in pain. Please help.",
           "CALL NURSE": "Please call the nurse.", "THANK YOU": "Thank you.", "I AM HUNGRY": "I am hungry.",
           "TOILET PLEASE": "I need to use the toilet.", "CALL FAMILY": "Please call my family.",
           "SOS": "Emergency! I need help right now!"}
MY_PHRASE_KEYS = ("MY 1", "MY 2", "MY 3")
PERSONAL_PHRASES_FILE = app_data_path("eyetype_phrases.json")
EDITABLE_PHRASES = [label for label in PHRASES if label != "SOS"]
PHRASE_EDIT_ROWS = [EDITABLE_PHRASES[:4], EDITABLE_PHRASES[4:], list(MY_PHRASE_KEYS), ["RESET DEFAULTS", "BACK"]]
ALERTS = {"CALL NURSE", "CALL FAMILY", "SOS", "I AM IN PAIN"}


def load_personal_phrases():
    """Load per-user phrase overrides; built-in safety labels remain fixed."""
    phrases = {**PHRASES, **{key: "" for key in MY_PHRASE_KEYS}}
    try:
        with open(PERSONAL_PHRASES_FILE, encoding="utf-8") as stream:
            saved = json.load(stream)
        if isinstance(saved, dict):
            for key in phrases:
                value = saved.get(key)
                if isinstance(value, str) and len(value) <= 120:
                    phrases[key] = value
    except FileNotFoundError:
        pass
    except Exception as ex:
        print(f"Could not load personal phrases: {ex}")
    return phrases


def save_personal_phrases(phrases):
    """Persist phrase text locally beside the app; it is not sent to a server."""
    try:
        temporary = PERSONAL_PHRASES_FILE + ".tmp"
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(phrases, stream, ensure_ascii=False, indent=2)
        os.replace(temporary, PERSONAL_PHRASES_FILE)
        return True, "Personal phrases saved on this laptop"
    except Exception as ex:
        return False, f"Could not save personal phrases: {ex}"


def build_keyboard(W, H):
    keys, unit, rh, gap, y = [], W * 0.98 / 10, H * 0.155, H * 0.008, H * 0.335
    for row in ROWS:
        x = (W - sum(w for _, w in row) * unit) / 2
        for lab, w in row:
            keys.append((lab, (int(x + gap), int(y), int(x + w * unit - gap), int(y + rh - gap))))
            x += w * unit
        y += rh
    sw = W * 0.98 / 5
    for i in range(4):
        keys.append((f"#{i}", (int(W * 0.01 + i * sw + gap), int(H * 0.18), int(W * 0.01 + (i + 1) * sw - gap), int(H * 0.30))))
    keys.append(("BIG ABC", (int(W * 0.01 + 4 * sw + gap), int(H * 0.18), int(W * 0.99 - gap), int(H * 0.30))))
    return keys


def build_board(W, H):
    keys, tw, rh, gap, y = [], W * 0.98 / 3, H * 0.08, H * 0.008, H * 0.28
    for row in BOARD:
        for i, lab in enumerate(row):
            x = W * 0.01 + i * tw
            keys.append((lab, (int(x + gap), int(y), int(x + tw - gap), int(y + rh - gap))))
        y += rh
    return keys


def build_emergency(W, H):
    keys, gap, tw, rh = [], H * 0.01, W * 0.98 / 3, H * 0.14
    for r, row in enumerate([["AMBULANCE", "POLICE", "FIRE"], ["FAMILY", "CARETAKER", "LAST CALLED"], ["SIREN"]]):
        for i, lab in enumerate(row):
            x = W * 0.01 + (i if len(row) == 3 else 1) * tw if len(row) == 1 else W * 0.01 + i * tw
            y = H * 0.29 + r * rh
            keys.append((lab, (int(x + gap), int(y + gap), int(x + tw - gap), int(y + rh))))
    keys.append(("CANCEL", (int(W * 0.01 + gap), int(H * 0.76), int(W * 0.99 - gap), int(H * 0.93))))
    return keys


NUMBER_PAGE = [["1", "2", "3", "4", "5"], ["6", "7", "8", "9", "0"],
               ["BACKSPACE", "SPACE", "BACK"], ["KEYBOARD", "BACK"]]
LETTER_PAGE_A = [["A", "B", "C", "D", "E"], ["F", "G", "H", "I", "J"],
                 ["K", "L", "M", "SPACE", "BACKSPACE"], ["NEXT", "BACK", "CLEAR", "SPEAK"]]
LETTER_PAGE_B = [["N", "O", "P", "Q", "R"], ["S", "T", "U", "V", "W"],
                 ["X", "Y", "Z", "SPACE", "BACKSPACE"], ["PREVIOUS", "BACK", "CLEAR", "SPEAK"]]
SYMBOL_PAGE = [["@", "#", "$", "%", "^"], ["&", "*", "(", ")", "-"],
               ["_", "+", "=", "!", "?"], ["/", "\\", ":", ";"], ["{", "}", "[", "]"],
               ["KEYBOARD", "BACK"]]
EMOJI_PAGE = [["EMOJI:GRIN", "EMOJI:SMILE", "EMOJI:HEART", "EMOJI:LIKE"], ["EMOJI:HELP", "EMOJI:SAD", "EMOJI:SLEEP", "EMOJI:ILL"], ["EMOJI:MUSIC", "EMOJI:SOS", "EMOJI:HOME", "BACK"]]
EMOJI_MAP = {"GRIN": "😀", "SMILE": "😊", "HEART": "❤️", "LIKE": "👍", "HELP": "🙏",
             "SAD": "😢", "SLEEP": "😴", "ILL": "😷", "MUSIC": "🎵", "SOS": "🚨", "HOME": "🏠"}
CONTROL_PAGE = [["COPY", "PASTE", "BACKSPACE", "ESC FULL"], ["VOL+", "VOL-", "BRIGHT+", "BRIGHT-"], ["CTRL", "ALT", "WIN", "PRED ON"], ["PRED OFF", "CURSOR ON", "CURSOR OFF", "BACK"]]
CURSOR_PAGE = [["CLICK", "DRAG", "SCROLL UP", "SCROLL DOWN"], ["BACK"]]
ROOM_PAGE = [["AC ON", "AC OFF", "TEMP+", "TEMP-"], ["LIGHT", "FAN", "TV", "BACK"]]
API_SETUP_PAGE = [["ROOM BRIDGE", "ROOM TOKEN", "AC ON URL", "AC OFF URL"],
                  ["AC TEMP URL", "LIGHT ON URL", "LIGHT OFF URL", "FAN ON URL"],
                  ["FAN OFF URL", "CLIMATE STATE", "DEVICE STATES", "LAST NUMBER"],
                  ["BACK"]]
CONFIG_FIELD_KEYS = {"ROOM BRIDGE": "ROOM_CONTROL_URL", "ROOM TOKEN": "ROOM_CONTROL_TOKEN",
                     "AC ON URL": "AC_ON_URL", "AC OFF URL": "AC_OFF_URL", "AC TEMP URL": "AC_SET_TEMP_URL",
                     "LIGHT ON URL": "LIGHT_ON_URL", "LIGHT OFF URL": "LIGHT_OFF_URL",
                     "FAN ON URL": "FAN_ON_URL", "FAN OFF URL": "FAN_OFF_URL",
                     "CLIMATE STATE": "ROOM_CLIMATE_STATE_URL", "DEVICE STATES": "ROOM_DEVICE_STATE_URL",
                     "LAST NUMBER": "LAST_DIALED_FALLBACK"}


def build_grid(rows, W, H):
    keys = []
    cols = max(len(row) for row in rows)
    top, bottom = int(H * 0.31), int(H * 0.94)
    rh = (bottom - top) / len(rows)
    cw = W * 0.98 / cols
    for ri, row in enumerate(rows):
        offset = (cols - len(row)) / 2  # center short footer/navigation rows
        for ci, lab in enumerate(row):
            x1 = int(W * 0.01 + (ci + offset) * cw + 5)
            x2 = int(W * 0.01 + (ci + offset + 1) * cw - 5)
            y1 = int(top + ri * rh + 4)
            y2 = int(top + (ri + 1) * rh - 4)
            keys.append((lab, (x1, y1, x2, y2)))
    return keys


def build_phrase_board(W, H):
    labels = [label for label in PHRASES if label != "SOS"] + list(MY_PHRASE_KEYS)
    rows = [labels[i:i + 3] for i in range(0, len(labels), 3)]
    rows.append(["EDIT PHRASES", "BACK"])
    return build_grid(rows, W, H)


def build_phrase_editor(W, H):
    return build_grid(PHRASE_EDIT_ROWS, W, H)


def system_key(key, modifiers=()):
    try:
        import pyautogui
        pyautogui.hotkey(*(list(modifiers) + [key]))
        return True
    except Exception as ex:
        print("System key unavailable (install/configure pyautogui):", ex)
        return False


def set_volume(up):
    key = "volumeup" if up else "volumedown"
    if system_key(key):
        return True
    if sys.platform.startswith("linux"):
        cmd = ["amixer", "-q", "sset", "Master", "5%+" if up else "5%-"]
        try:
            return subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
        except Exception:
            pass
    return False


def direct_url_request(url):
    if not url:
        return False, "No direct action URL is configured"
    try:
        import urllib.request
        headers = {"Authorization": "Bearer " + ROOM_CONTROL_TOKEN} if ROOM_CONTROL_TOKEN else {}
        req = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=5) as response:
            response.read()
        return True, "Device URL accepted the command"
    except Exception as ex:
        return False, f"Device URL request failed: {ex}"


def climate_service(service, data):
    global DIRECT_AC_TEMP
    if not (HA_URL and HA_TOKEN and HA_CLIMATE_ENTITY):
        if service == "set_hvac_mode":
            mode = str(data.get("hvac_mode", "")).lower()
            url = AC_OFF_URL if mode == "off" else (AC_ON_URL if mode in ("cool", "heat", "auto") else "")
            if url:
                return direct_url_request(url)
        if service == "set_temperature":
            if AC_SET_TEMP_URL:
                try:
                    target = float(data["temperature"])
                except (KeyError, TypeError, ValueError):
                    return False, "Temperature setpoint is missing or invalid"
                import urllib.parse
                value = urllib.parse.quote(f"{target:g}", safe=".")
                url = AC_SET_TEMP_URL.replace("{temperature}", value).replace("{temp}", value)
                ok, msg = direct_url_request(url)
                if ok:
                    DIRECT_AC_TEMP = target
                return ok, msg
        if ROOM_CONTROL_URL:
            payload = {"device": "AC", "action": service, **data}
            return room_bridge_request(payload)
        if service == "set_temperature":
            return False, "Configure AC TEMP URL (use {temperature} for the requested setpoint)"
        return False, f"Direct URL mode does not support {service}; configure a room bridge"
    try:
        import urllib.request
        body = dict(data, entity_id=HA_CLIMATE_ENTITY)
        req = urllib.request.Request(
            f"{HA_URL.rstrip('/')}/api/services/climate/{service}",
            data=json.dumps(body).encode(),
            headers={"Authorization": "Bearer " + HA_TOKEN, "Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=5) as response:
            response.read()
        return True, "Home Assistant command sent"
    except Exception as ex:
        return False, f"AC control failed: {ex}"


def get_climate_temperature(default=24.0):
    """Read the current Home Assistant setpoint; use a fallback if unavailable."""
    if not (HA_URL and HA_TOKEN and HA_CLIMATE_ENTITY):
        if not ROOM_CLIMATE_STATE_URL:
            return float(DIRECT_AC_TEMP)
        try:
            import urllib.request
            headers = {"Authorization": "Bearer " + ROOM_CONTROL_TOKEN} if ROOM_CONTROL_TOKEN else {}
            req = urllib.request.Request(ROOM_CLIMATE_STATE_URL, headers=headers)
            with urllib.request.urlopen(req, timeout=5) as response:
                return float(json.loads(response.read().decode()).get("temperature", default))
        except Exception:
            return float(DIRECT_AC_TEMP)
    try:
        import urllib.request
        req = urllib.request.Request(
            f"{HA_URL.rstrip('/')}/api/states/{HA_CLIMATE_ENTITY}",
            headers={"Authorization": "Bearer " + HA_TOKEN})
        with urllib.request.urlopen(req, timeout=5) as response:
            state = json.loads(response.read().decode())
        attrs = state.get("attributes", {})
        return float(attrs.get("temperature", attrs.get("current_temperature", default)))
    except Exception:
        return default


def get_room_device_state(name, default=False):
    """Read an entity's actual on/off state from Home Assistant when configured."""
    entity = HA_ENTITIES.get(name)
    if not (HA_URL and HA_TOKEN and entity):
        if ROOM_DEVICE_STATE_URL:
            try:
                import urllib.request
                headers = {"Authorization": "Bearer " + ROOM_CONTROL_TOKEN} if ROOM_CONTROL_TOKEN else {}
                req = urllib.request.Request(ROOM_DEVICE_STATE_URL, headers=headers)
                with urllib.request.urlopen(req, timeout=5) as response:
                    states = json.loads(response.read().decode())
                return str(states.get(name, "off")).lower() in ("on", "true", "1", "playing")
            except Exception:
                return default
        return default
    try:
        import urllib.request
        req = urllib.request.Request(
            f"{HA_URL.rstrip('/')}/api/states/{entity}",
            headers={"Authorization": "Bearer " + HA_TOKEN})
        with urllib.request.urlopen(req, timeout=5) as response:
            state = json.loads(response.read().decode()).get("state", "")
        return state in ("on", "playing", "home", "open")
    except Exception:
        return default


def room_bridge_request(payload):
    """POST a generic device command to a user-configured local bridge URL."""
    if not ROOM_CONTROL_URL:
        return False, "Configure ROOM_CONTROL_URL or Home Assistant first"
    try:
        import urllib.request
        headers = {"Content-Type": "application/json"}
        if ROOM_CONTROL_TOKEN:
            headers["Authorization"] = "Bearer " + ROOM_CONTROL_TOKEN
        req = urllib.request.Request(ROOM_CONTROL_URL, data=json.dumps(payload).encode(), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=5) as response:
            response.read()
        return True, "Room bridge command sent"
    except Exception as ex:
        return False, f"Room bridge error: {ex}"


def caregiver_webhook(event, text=""):
    """Send a JSON event to the caregiver's configured webhook endpoint."""
    if not CARETAKER_WEBHOOK_URL:
        return False, "Set CARETAKER_WEBHOOK_URL first"
    try:
        import urllib.request
        body = {"event": event, "text": text, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        headers = {"Content-Type": "application/json"}
        if CARETAKER_WEBHOOK_TOKEN:
            headers["Authorization"] = "Bearer " + CARETAKER_WEBHOOK_TOKEN
        req = urllib.request.Request(CARETAKER_WEBHOOK_URL, data=json.dumps(body).encode(), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=8) as response:
            response.read()
        return True, "Caregiver webhook accepted the message"
    except Exception as ex:
        return False, f"Caregiver webhook error: {ex}"


def inside(p, r, margin=0):
    return r[0] - margin <= p[0] <= r[2] + margin and r[1] - margin <= p[1] <= r[3] + margin


def gaze_hit(keys, point, current=None, mode="kb"):
    """Resolve gaze to a key without letting adjacent QWERTY keys trap selection."""
    by_label = {lab: rect for lab, rect in keys}
    if current in by_label:
        rect = by_label[current]
        is_small_letter = mode == "kb" and len(current) == 1 and current.isalpha()
        # Adjacent QWERTY keys nearly touch; the old fixed 30 px sticky margin
        # overlapped neighbors and could hold the previous letter selected.
        margin = 0 if is_small_letter else min(30, max(4, int(min(rect[2] - rect[0], rect[3] - rect[1]) * 0.08)))
        if inside(point, rect, margin):
            return current
    for lab, rect in keys:
        if inside(point, rect):
            return lab
    if mode != "kb":
        return None
    # Gaze calibration has a few pixels of residual error, and the suggestion
    # and SPACE targets can be hard to hit exactly. Let them absorb a small miss.
    tolerant = []
    for lab, rect in keys:
        if not (lab.startswith("#") or lab == "BIG ABC" or lab == "SPACE"):
            continue
        x1, y1, x2, y2 = rect
        margin = max(8, int((y2 - y1) * 0.18))
        if inside(point, rect, margin):
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            distance = (float(point[0]) - cx) ** 2 + (float(point[1]) - cy) ** 2
            tolerant.append((distance, lab))
    if tolerant:
        return min(tolerant)[1]
    letters = [(lab, rect) for lab, rect in keys if len(lab) == 1 and lab.isalpha()]
    if not letters:
        return None
    rows = {}
    for lab, rect in letters:
        rows.setdefault((rect[1], rect[3]), []).append((lab, rect))
    py = float(point[1])
    bounds = min(rows, key=lambda row_bounds: max(row_bounds[0] - py, 0, py - row_bounds[1]))
    row = rows[bounds]
    height = max(1, bounds[1] - bounds[0])
    if max(bounds[0] - py, 0, py - bounds[1]) > max(8, int(height * 0.15)):
        return None
    lab, rect = min(row, key=lambda item: abs(float(point[0]) - (item[1][0] + item[1][2]) / 2))
    width = max(1, rect[2] - rect[0])
    return lab if abs(float(point[0]) - (rect[0] + rect[2]) / 2) <= width * 0.90 else None


# ---------------- helpers ----------------
def speak(text):
    def run():
        try:
            import pyttsx3
            e = pyttsx3.init(); e.say(text); e.runAndWait()
        except Exception as ex:
            print("TTS error:", ex)
    threading.Thread(target=run, daemon=True).start()


SIREN_WAV = os.path.join(tempfile.gettempdir(), "eyetype_siren.wav")
BEEP_WAV = os.path.join(tempfile.gettempdir(), "eyetype_beep.wav")
_player = {"proc": None}


def _write_wav(path, x, sr=22050):
    import wave
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())


def ensure_sounds():
    """Generates the alarm sounds on the laptop itself - no files or internet needed."""
    sr = 22050
    if not os.path.exists(SIREN_WAV):
        t = np.arange(int(SIREN_SECONDS * sr)) / sr
        freq = 1000 + 450 * np.sin(2 * np.pi * t / 1.5)              # siren sweeping up and down
        ph = 2 * np.pi * np.cumsum(freq) / sr
        _write_wav(SIREN_WAV, 0.9 * (0.7 * np.sin(ph) + 0.3 * np.sin(2 * ph)))
    if not os.path.exists(BEEP_WAV):
        t = np.arange(int(0.35 * sr)) / sr
        _write_wav(BEEP_WAV, 0.8 * np.sin(2 * np.pi * 1200 * t) * np.minimum(1, 10 * (0.35 - t)))


def stop_sound():
    try:
        if sys.platform.startswith("win"):
            import winsound
            winsound.PlaySound(None, winsound.SND_PURGE)
        elif _player["proc"]:
            _player["proc"].terminate()
    except Exception:
        pass


def play_wav(path):
    try:
        if sys.platform.startswith("win"):
            import winsound
            winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
            return
        stop_sound()
        for cmd in (["afplay", path], ["aplay", "-q", path], ["paplay", path]):
            try:
                _player["proc"] = subprocess.Popen(cmd)
                return
            except FileNotFoundError:
                continue
        print("\a", end="", flush=True)
    except Exception:
        print("\a", end="", flush=True)


def beep():
    ensure_sounds(); play_wav(BEEP_WAV)


def play_siren():
    ensure_sounds(); play_wav(SIREN_WAV)


def send_alert(msg):
    def run():
        if not ALERT_URL:
            return
        try:
            import urllib.request, urllib.parse
            urllib.request.urlopen(ALERT_URL.replace("{msg}", urllib.parse.quote(msg)), timeout=5)
        except Exception as ex:
            print("Alert failed:", ex)
    threading.Thread(target=run, daemon=True).start()


def control_device(name, on, callback=None):
    def run():
        result = (False, "No supported device URL configured")
        try:
            import urllib.request
            if HA_URL and HA_TOKEN and name in HA_ENTITIES:
                req = urllib.request.Request(
                    f"{HA_URL}/api/services/homeassistant/{'turn_on' if on else 'turn_off'}",
                    data=json.dumps({"entity_id": HA_ENTITIES[name]}).encode(),
                    headers={"Authorization": "Bearer " + HA_TOKEN, "Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(req, timeout=5) as response:
                    response.read()
                result = (True, "Home Assistant accepted the command")
            elif name in DEVICE_URLS and DEVICE_URLS[name].get("on" if on else "off"):
                result = direct_url_request(DEVICE_URLS[name]["on" if on else "off"])
            elif ROOM_CONTROL_URL:
                result = room_bridge_request({"device": name, "action": "on" if on else "off"})
            else:
                result = (False, f"No URL configured for {name} {'on' if on else 'off'}")
        except Exception as ex:
            result = (False, f"Device control failed: {ex}")
        if callback:
            callback(*result)
        elif not result[0]:
            print(result[1])
    threading.Thread(target=run, daemon=True).start()


def _gsm(cmds):
    import serial                                   # pip install pyserial
    with serial.Serial(GSM_PORT, GSM_BAUD, timeout=2) as ser:
        out = ""
        for c, wait in cmds:
            ser.write(c); time.sleep(wait); out += ser.read_all().decode(errors="ignore")
    return out


def gsm_call(number):
    try:
        return "ERROR" not in _gsm([(b"AT\r", 0.5), (f"ATD{number};\r".encode(), 2.0)])
    except Exception as ex:
        print("GSM call failed:", ex); return False


def gsm_sms(number, text):
    try:
        out = _gsm([(b"AT\r", 0.5), (b"AT+CMGF=1\r", 0.5), (f'AT+CMGS="{number}"\r'.encode(), 1.0),
                    (text.encode("ascii", "ignore")[:150] + b"\x1a", 5.0)])
        return "ERROR" not in out and "+CMGS" in out
    except Exception as ex:
        print("GSM sms failed:", ex); return False


def send_sms(number, text):
    """Sends an SMS through the GSM modem (works with no internet). Always logs to sms_outbox.txt."""
    if not number:
        return "no number set"
    def run():
        ok = gsm_sms(number, text) if GSM_PORT and SMS_LIVE else False
        with open(app_data_path("sms_outbox.txt"), "a") as f:
            f.write(f"{'SENT' if ok else 'QUEUED'} | {time.strftime('%Y-%m-%d %H:%M:%S')} | to {number} | {text}\n")
    threading.Thread(target=run, daemon=True).start()
    return "sending via GSM modem" if GSM_PORT and SMS_LIVE else "saved to sms_outbox.txt (SMS_LIVE is off or no GSM modem set)"


def remember_dialed(number):
    try:
        json.dump({"number": number, "time": time.time()}, open(LAST_DIALED_FILE, "w"))
    except Exception:
        pass


def get_last_dialed():
    """Return the per-install saved call target first, then phone/app history if available."""
    if LAST_DIALED_FALLBACK:
        return LAST_DIALED_FALLBACK
    if EMERGENCY_LIVE and not GSM_PORT:
        try:
            out = subprocess.run(["adb", "shell", "content query --uri content://call_log/calls --projection number "
                                  "--where 'type=2' --sort 'date DESC'"], capture_output=True, text=True, timeout=8).stdout
            m = re.search(r"number=([+\d]+)", out)
            if m:
                return m.group(1)
        except Exception:
            pass
    try:
        return json.load(open(LAST_DIALED_FILE)).get("number", "") or LAST_DIALED_FALLBACK
    except Exception:
        return LAST_DIALED_FALLBACK


def open_default_tel_handler(number, who):
    """Ask the operating system to open its registered tel: calling app."""
    cleaned = re.sub(r"[\s().-]", "", str(number or ""))
    if not re.fullmatch(r"\+?\d{2,20}", cleaned):
        return False, f"Invalid or missing call number for {who}"
    try:
        import webbrowser
        opened = webbrowser.open(f"tel:{cleaned}", new=0, autoraise=True)
        if opened:
            remember_dialed(cleaned)
            return True, (f"Sent a tel: request for {who} ({cleaned}); "
                          "check the calling app and confirm the call there if prompted")
        return False, "No default tel: calling app opened; configure one on this laptop"
    except Exception as ex:
        return False, f"Could not open the default tel: app: {ex}"


def dial_number(number, who):
    """Hand off a call to GSM hardware or the operating system's default tel: app."""
    if not number:
        return False, f"No number set for {who}"
    cleaned = re.sub(r"[\s().-]", "", str(number))
    if not re.fullmatch(r"\+?\d{2,20}", cleaned):
        return False, f"Invalid call number for {who}"
    if not EMERGENCY_LIVE:
        return True, f"DRY RUN: would open a call to {who} ({cleaned})"
    if GSM_PORT:
        ok = gsm_call(cleaned)
        if ok: remember_dialed(cleaned)
        return ok, (f"Calling {who} ({cleaned}) via GSM modem" if ok else f"GSM call to {who} failed")
    return open_default_tel_handler(cleaned, who)


def enhance(frame, bright):
    """Brightens dark camera frames (gamma + CLAHE) so the face/iris can still be found."""
    if bright >= ENHANCE_BELOW:
        return frame
    gamma = float(np.clip(1.0 + (ENHANCE_BELOW - bright) / 40.0, 1.2, 2.6))
    lut = (np.linspace(0, 1, 256) ** (1.0 / gamma) * 255).astype(np.uint8)
    lab = cv2.cvtColor(cv2.LUT(frame, lut), cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def set_screen_brightness(level):
    """Optional (pip install screen-brightness-control). Returns the old brightness or None."""
    try:
        import screen_brightness_control as sbc
        old = sbc.get_brightness()[0]
        sbc.set_brightness(level)
        return old
    except Exception:
        return None


def fit(text, maxw, base):
    return min(base, maxw / (max(1, len(text)) * 24))


def rounded_box(img, rect, color, radius=14):
    """Fast filled rounded panel using OpenCV primitives only (no extra UI dependency)."""
    x1, y1, x2, y2 = map(int, rect)
    r = max(2, min(int(radius), (x2 - x1) // 2, (y2 - y1) // 2))
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1, cv2.LINE_AA)
    for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)):
        cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)


def fitted_text(img, text, rect, color, scale, font=None, thickness=None):
    """Draw readable anti-aliased text, scaling down only when it cannot fit its tile."""
    if font is None:
        font = cv2.FONT_HERSHEY_DUPLEX
    x1, y1, x2, y2 = map(int, rect)
    scale = max(0.45, float(scale))
    thick = max(1, int(thickness if thickness is not None else round(scale * 1.5)))
    (tw, th), base = cv2.getTextSize(str(text), font, scale, thick)
    fit_scale = min(1.0, max(1, x2 - x1 - 20) / max(1, tw), max(1, y2 - y1 - 12) / max(1, th + base))
    scale *= fit_scale
    thick = max(1, int(thickness if thickness is not None else round(scale * 1.5)))
    (tw, th), _ = cv2.getTextSize(str(text), font, scale, thick)
    origin = ((x1 + x2 - tw) // 2, (y1 + y2 + th) // 2)
    cv2.putText(img, str(text), origin, font, scale, color, thick, cv2.LINE_AA)


def opencv_safe_message(text):
    """OpenCV Hershey fonts cannot draw emoji; show readable aliases, not question marks."""
    shown = str(text)
    for name, emoji in sorted(EMOJI_MAP.items(), key=lambda item: len(item[1]), reverse=True):
        shown = shown.replace(emoji, f"[{name}]")
    return shown


_EMOJI_FONT_PATH = ""
_EMOJI_FONT_PROBED = False
_EMOJI_BITMAP_CACHE = {}


def find_emoji_font():
    """Find a system color-emoji font once; keep text-name fallback if none is available."""
    global _EMOJI_FONT_PATH, _EMOJI_FONT_PROBED
    if _EMOJI_FONT_PROBED:
        return _EMOJI_FONT_PATH
    _EMOJI_FONT_PROBED = True
    try:
        import PIL  # noqa: F401
    except Exception:
        return ""
    windir = os.environ.get("WINDIR", r"C:\Windows")
    candidates = [
        os.path.join(windir, "Fonts", "seguiemj.ttf"),
        os.path.join(windir, "Fonts", "seguisym.ttf"),
        "/System/Library/Fonts/Apple Color Emoji.ttc",
        "/System/Library/Fonts/AppleColorEmoji.ttc",
        "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
        "/usr/share/fonts/opentype/noto/NotoColorEmoji.ttf",
        "/usr/share/fonts/truetype/noto/NotoEmoji-Regular.ttf",
    ]
    for path in candidates:
        if os.path.isfile(path):
            _EMOJI_FONT_PATH = path
            return path
    if os.name == "posix":
        try:
            result = subprocess.run(["fc-match", "-f", "%{file}", "Noto Color Emoji"],
                                    capture_output=True, text=True, timeout=2)
            path = result.stdout.strip()
            if result.returncode == 0 and os.path.isfile(path):
                _EMOJI_FONT_PATH = path
        except Exception:
            pass
    return _EMOJI_FONT_PATH


def emoji_bitmap(emoji, size, font_path):
    """Return a cached BGRA emoji tile; return None if this font can't draw the glyph."""
    key = (emoji, int(size), font_path)
    if key in _EMOJI_BITMAP_CACHE:
        return _EMOJI_BITMAP_CACHE[key]
    bitmap = None
    try:
        from PIL import Image, ImageDraw, ImageFont
        for font_size in dict.fromkeys((int(size), 109, 96, 72)):
            try:
                font = ImageFont.truetype(font_path, font_size)
            except OSError:
                continue
            bounds = font.getbbox(emoji)
            width, height = max(1, bounds[2] - bounds[0]), max(1, bounds[3] - bounds[1])
            image = Image.new("RGBA", (width + 12, height + 12), (0, 0, 0, 0))
            draw = ImageDraw.Draw(image)
            try:
                draw.text((6 - bounds[0], 6 - bounds[1]), emoji, font=font, embedded_color=True)
            except TypeError:
                draw.text((6 - bounds[0], 6 - bounds[1]), emoji, font=font)
            alpha_bounds = image.getchannel("A").getbbox()
            if not alpha_bounds:
                continue
            image = image.crop(alpha_bounds)
            image.thumbnail((int(size), int(size)), Image.Resampling.LANCZOS)
            rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
            bitmap = rgba[:, :, [2, 1, 0, 3]].copy()
            break
    except Exception:
        bitmap = None
    _EMOJI_BITMAP_CACHE[key] = bitmap
    return bitmap


def draw_composer_message(img, text, rect, color, scale, font_path):
    """Draw normal text with real color emoji glyphs, falling back to [NAME] aliases."""
    x1, y1, x2, y2 = rect
    source = str(text)
    matches = sorted(EMOJI_MAP.items(), key=lambda item: len(item[1]), reverse=True)
    tokens, plain, pos = [], [], 0
    while pos < len(source):
        found = next(((name, emoji) for name, emoji in matches if source.startswith(emoji, pos)), None)
        if not found:
            plain.append(source[pos])
            pos += 1
            continue
        if plain:
            tokens.append(("text", "".join(plain)))
            plain.clear()
        name, emoji = found
        icon = emoji_bitmap(emoji, max(18, int((y2 - y1) * 0.62)), font_path) if font_path else None
        tokens.append(("icon", icon) if icon is not None else ("text", f"[{name}]") )
        pos += len(emoji)
    if plain:
        tokens.append(("text", "".join(plain)))
    if not tokens:
        fitted_text(img, "", rect, color, scale)
        return

    font, thickness = cv2.FONT_HERSHEY_DUPLEX, max(1, int(round(scale * 1.5)))
    base_icon_h = max(16, int((y2 - y1) * 0.62))

    def measure(text_scale, icon_h):
        total, text_h = 0, cv2.getTextSize("Ag", font, text_scale, thickness)[0][1]
        widths = []
        for kind, value in tokens:
            if kind == "text":
                width = cv2.getTextSize(value, font, text_scale, thickness)[0][0]
            else:
                ih, iw = value.shape[:2]
                width = max(1, round(iw * icon_h / max(1, ih)))
            widths.append(width)
            total += width
        return total, max(text_h, icon_h), widths

    available = max(1, x2 - x1 - 24)
    total, height, widths = measure(scale, base_icon_h)
    factor = min(1.0, available / max(1, total), max(1, y2 - y1 - 8) / max(1, height))
    text_scale, icon_h = scale * factor, max(12, int(base_icon_h * factor))
    total, text_h, widths = measure(text_scale, icon_h)
    cursor = (x1 + x2 - total) // 2
    baseline = (y1 + y2 + text_h) // 2
    for (kind, value), width in zip(tokens, widths):
        if kind == "text":
            safe = opencv_safe_message(value)
            cv2.putText(img, safe, (cursor, baseline), font, text_scale, color, thickness, cv2.LINE_AA)
        else:
            ih, iw = value.shape[:2]
            new_w = max(1, width)
            icon = cv2.resize(value, (new_w, icon_h), interpolation=cv2.INTER_AREA) if (iw, ih) != (new_w, icon_h) else value
            iy = baseline - icon_h + max(0, (text_h - icon_h) // 2)
            area = img[iy:iy + icon_h, cursor:cursor + new_w]
            alpha = icon[:, :, 3:4].astype(np.float32) / 255.0
            area[:] = (icon[:, :, :3].astype(np.float32) * alpha + area.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
        cursor += width


# ---------------- application ----------------
class App:
    def __init__(self, W, H, cam, win):
        self.W, self.H, self.cam, self.win = W, H, cam, win
        self.trk = None if MOUSE else Tracker()
        self.emoji_font_path = find_emoji_font()
        self.pred = Predictor()
        self.kb, self.bd, self.em = build_keyboard(W, H), build_board(W, H), build_emergency(W, H)
        self.phrases = load_personal_phrases()
        self.page_cache = {"numbers": build_grid(NUMBER_PAGE, W, H), "symbols": build_grid(SYMBOL_PAGE, W, H),
                           "letters_a": build_grid(LETTER_PAGE_A, W, H), "letters_b": build_grid(LETTER_PAGE_B, W, H),
                           "emojis": build_grid(EMOJI_PAGE, W, H), "controls": build_grid(CONTROL_PAGE, W, H),
                           "cursor": build_grid(CURSOR_PAGE, W, H), "room": build_grid(ROOM_PAGE, W, H),
                           "my_phrases": build_phrase_board(W, H), "phrase_setup": build_phrase_editor(W, H),
                           "api_setup": build_grid(API_SETUP_PAGE, W, H),
                           "confirm": build_grid([["CONFIRM CALL", "CANCEL"]], W, H)}
        self.bg_dark = np.empty((H, W, 3), dtype=np.uint8); self.bg_dark[:] = (38, 27, 20)
        self.bg_light = np.empty((H, W, 3), dtype=np.uint8); self.bg_light[:] = (244, 237, 228)
        self.mode = "kb"
        self.prediction_enabled = True
        self.caregiver_auto_share = CAREGIVER_AUTO_SEND
        self.caregiver_last_shared = ""
        self.caregiver_share_timer = None
        self.config_edit_field = None
        self.phrase_edit_label = None
        self.phrase_focus_i = 0
        self.reset_phrase_deadline = 0.0
        self.cursor_mode = False
        self.dragging = False
        self.modifiers = set()
        self.ha_temperature = 24.0
        self.fullscreen = True
        self.text = ""
        self.gaze = np.array([W / 2, H / 2])
        self.offset = np.zeros(2)
        self.mouse = np.array([W / 2, H / 2])
        self.cur, self.start = None, time.time()
        self.closed_since = None
        self.flash, self.flash_t = "", 0
        self.dwell, self.streak = DWELL_KEY, 0
        self.t_first, self.n_keys, self.n_del, self.saved = None, 0, 0, 0
        self.devices = {"LIGHT": False, "FAN": False, "TV": False}
        self.banner, self.banner_until = "", 0
        self.sos_until, self.next_beep = 0, 0
        self.sos_reason, self.headshake_cooldown_until = "", 0.0
        self.stats = Counter()
        self.reason, self.em_deadline = "manual", 0
        self.pending_service, self.confirm_deadline = None, 0
        self.blink_times, self.turn_times, self.headshake_turns = [], [], []
        self.yaw_base, self.turned = None, 0
        self.pose_base, self.pose_state = None, 0
        self.pose_events, self.roll_state = [], 0
        self.room_pose_lockout = 0.0
        self.mapper, self.ear_thr, self.err_deg = None, 0.2, 0.0
        self.lowlight, self.dark_t, self.bright_t, self.low_since = False, None, None, 0
        self.bright, self.old_brightness = 128.0, None
        self.no_face, self.face_ok, self.scan_auto, self.prev_mode = 0, 0, False, "kb"
        rows = {}
        for lab, rect in self.kb:
            if lab != "MENU":
                rows.setdefault(rect[1], []).append(lab)
        self.scan_rows = [rows[y] for y in sorted(rows)] + [["SOS"]]
        self.label_row = {lab: i for i, row in enumerate(self.scan_rows) for lab in row}
        self.kb_scan = self.kb + [("SOS", (int(W * 0.72), int(H * 0.01), int(W * 0.98), int(H * 0.16)))]
        self.scan_return_mode, self.scan_keys = "kb", self.kb_scan
        self.scan_stage, self.scan_row, self.scan_i, self.scan_t, self.scan_trigger = "row", 0, 0, time.time(), False

    # ----- text logic -----
    def split(self):
        p = self.text.split(" ")
        return p[:-1], p[-1]

    def prev_word(self):
        done, _ = self.split()
        return done[-1].lower().strip(".,?!") if done else ""

    def cap_needed(self):
        _, partial = self.split()
        before = self.text[:len(self.text) - len(partial)]
        return before.strip() == "" or before.rstrip().endswith((".", "?", "!"))

    def sugs(self):
        if not self.prediction_enabled:
            return []
        _, partial = self.split()
        return self.pred.suggest(partial.lower(), self.prev_word())

    def mark_typing(self):
        self.t_first = self.t_first or time.time()
        if not self.config_edit_field and not self.phrase_edit_label and self.caregiver_auto_share and CARETAKER_WEBHOOK_URL:
            if self.caregiver_share_timer:
                self.caregiver_share_timer.cancel()
            self.caregiver_share_timer = threading.Timer(3.0, self.auto_share_caregiver_text)
            self.caregiver_share_timer.daemon = True
            self.caregiver_share_timer.start()

    def auto_share_caregiver_text(self):
        text = self.text.strip()
        if not text or text == self.caregiver_last_shared:
            return
        ok, msg = caregiver_webhook("message", text)
        if ok:
            self.caregiver_last_shared = text
        else:
            print(msg)
            self.banner, self.banner_until = msg, time.time() + 5

    def post_caregiver_text(self, text):
        ok, msg = caregiver_webhook("message", text)
        if ok:
            self.caregiver_last_shared = text
        self.banner = "Caregiver webhook accepted the message" if ok else msg
        self.banner_until = time.time() + 5

    def send_caregiver_text(self):
        text = self.text.strip()
        if not text:
            self.banner = "Type a message first"
            self.banner_until = time.time() + 3
            return
        if self.caregiver_share_timer:
            self.caregiver_share_timer.cancel()
        if CARETAKER_WEBHOOK_URL:
            threading.Thread(target=self.post_caregiver_text, args=(text,), daemon=True).start()
        if CARETAKER_PHONE and GSM_PORT and SMS_LIVE:
            send_sms(CARETAKER_PHONE, text)
        self.banner = "Sending to caregiver webhook..." if CARETAKER_WEBHOOK_URL else (
            "SMS queued to caregiver" if CARETAKER_PHONE and GSM_PORT and SMS_LIVE else
            "Configure caregiver webhook or phone/SMS first")
        self.banner_until = time.time() + 5

    def type_char(self, c):
        _, partial = self.split()
        if not self.config_edit_field and partial == "" and self.cap_needed():
            c = c.upper()
        self.mark_typing(); self.n_keys += 1; self.text += c

    def do_space(self):
        if not self.text or self.text.endswith(" "):
            return
        _, partial = self.split()
        self.pred.learn(self.prev_word(), partial)
        self.mark_typing(); self.n_keys += 1; self.text += " "

    def punct(self, p):
        if not self.text:
            return
        _, partial = self.split()
        if partial:
            self.pred.learn(self.prev_word(), partial)
        self.mark_typing(); self.n_keys += 1
        self.text = self.text.rstrip(" ") + p + " "

    def accept(self, i):
        s = self.sugs()
        if i >= len(s):
            return
        done, partial = self.split()
        w = s[i]
        if partial[:1].isupper() or (partial == "" and self.cap_needed()):
            w = w[0].upper() + w[1:]
        self.saved += max(0, len(w) - len(partial))
        self.pred.learn(self.prev_word(), w)
        self.text = " ".join(done + [w]) + " "
        self.mark_typing(); self.n_keys += 1

    def say(self, sentence, label=""):
        self.banner, self.banner_until = sentence, time.time() + 4
        speak(sentence)
        if label in ALERTS:
            beep(); send_alert(f"EyeType alert: {sentence}")

    # ----- actions -----
    def device_action_result(self, name, target, ok, msg):
        if not ok:
            self.devices[name] = not target
        self.banner = f"{name} {'ON' if target else 'OFF'}: {msg}"
        self.banner_until = time.time() + 5

    def request_device_toggle(self, name, source="tile"):
        self.banner, self.banner_until = f"Checking {name} state...", time.time() + 6
        def run():
            actual = get_room_device_state(name, self.devices.get(name, False))
            target = not actual
            self.devices[name] = target
            self.banner, self.banner_until = f"Sending {name} {'ON' if target else 'OFF'} ({source})...", time.time() + 6
            control_device(name, target, lambda ok, msg: self.device_action_result(name, target, ok, msg))
        threading.Thread(target=run, daemon=True).start()

    def request_climate_action(self, lab):
        self.banner, self.banner_until = f"Sending {lab} command...", time.time() + 6
        def run():
            if lab in ("AC ON", "AC OFF"):
                ok, msg = climate_service("set_hvac_mode", {"hvac_mode": "cool" if lab == "AC ON" else "off"})
                result = f"{lab}: {msg}"
            else:
                current = get_climate_temperature(self.ha_temperature)
                target = max(16, min(30, current + (HA_TEMP_STEP if lab == "TEMP+" else -HA_TEMP_STEP)))
                mode_ok, mode_msg = climate_service("set_hvac_mode", {"hvac_mode": "cool"})
                ok, msg = climate_service("set_temperature", {"temperature": target})
                if ok:
                    self.ha_temperature = target
                    result = (f"Cooling; target {target:.0f}°C" if mode_ok else
                              f"Target {target:.0f}°C sent; AC ON also needed: {mode_msg}")
                else:
                    result = f"AC temperature request failed: {msg}"
            self.banner, self.banner_until = result, time.time() + 6
        threading.Thread(target=run, daemon=True).start()

    def press(self, lab):
        self.flash, self.flash_t = lab, time.time()
        if lab == "CANCEL PHRASE" and self.phrase_edit_label:
            self.phrase_edit_label = None
            self.text = ""
            self.mode, self.cur = "phrase_setup", None
            self.banner, self.banner_until = "Phrase edit cancelled; changes were not saved", time.time() + 4
            return
        if lab == "MENU" and self.phrase_edit_label:
            self.phrase_edit_label = None
            self.text = ""
            self.mode, self.cur = "phrase_setup", None
            self.banner, self.banner_until = "Phrase edit cancelled; changes were not saved", time.time() + 4
            return
        if self.mode == "letters_a" and lab == "NEXT":
            self.mode, self.cur = "letters_b", None
            return
        if self.mode == "letters_b" and lab == "PREVIOUS":
            self.mode, self.cur = "letters_a", None
            return
        if self.mode in ("letters_a", "letters_b") and lab == "BACK":
            self.mode, self.cur = "kb", None
            return
        if lab == "MENU": self.mode = "board"; return
        if lab == "MY PHRASES":
            self.mode, self.cur, self.phrase_focus_i = "my_phrases", None, 0
            return
        if lab == "EDIT PHRASES":
            self.mode, self.cur = "phrase_setup", None
            return
        if self.mode == "phrase_setup" and lab == "RESET DEFAULTS":
            now = time.time()
            if now > self.reset_phrase_deadline:
                self.reset_phrase_deadline = now + 7
                self.banner, self.banner_until = "Select RESET DEFAULTS again within 7 seconds to confirm", now + 7
            else:
                self.phrases = {**PHRASES, **{key: "" for key in MY_PHRASE_KEYS}}
                ok, msg = save_personal_phrases(self.phrases)
                self.reset_phrase_deadline = 0.0
                self.banner, self.banner_until = ("Built-in phrases restored" if ok else msg), now + 4
            return
        if self.mode == "phrase_setup" and lab in self.phrases:
            self.phrase_edit_label = lab
            self.text = self.phrases.get(lab, "")
            self.mode, self.cur = "kb", None
            self.banner = f"Editing {lab}: enter up to 120 characters, then select SAVE PHRASE. CLEAR erases."
            self.banner_until = time.time() + 7
            return
        if lab == "SAVE PHRASE" and self.phrase_edit_label:
            label = self.phrase_edit_label
            self.phrases[label] = self.text.strip()[:120]
            ok, msg = save_personal_phrases(self.phrases)
            self.phrase_edit_label = None
            self.text = ""
            self.mode, self.cur = "phrase_setup", None
            self.banner, self.banner_until = msg, time.time() + 5
            return
        if lab == "CONFIRM CALL" and self.mode == "confirm":
            service = self.pending_service
            self.pending_service = None
            self.dispatch(service or "DEFAULT")
            return
        if lab == "BACK" and self.mode == "phrase_setup":
            self.mode, self.cur = "my_phrases", None
            return
        if lab in ("KEYBOARD", "BACK"):
            self.mode = "kb" if lab == "KEYBOARD" else ("board" if self.mode != "board" else "kb")
            return
        page_modes = {"NUMBERS": "numbers", "SYMBOLS": "symbols", "EMOJIS": "emojis", "BIG ABC": "letters_a",
                      "CONTROLS": "controls", "ROOM": "room", "CURSOR": "cursor", "API SETUP": "api_setup"}
        if lab in page_modes:
            self.mode = page_modes[lab]; return
        if lab in CONFIG_FIELD_KEYS:
            self.config_edit_field = CONFIG_FIELD_KEYS[lab]
            self.text = str(globals().get(self.config_edit_field, ""))
            if self.caregiver_share_timer:
                self.caregiver_share_timer.cancel()
            self.mode, self.cur = "kb", None
            value_type = ("token" if self.config_edit_field == "ROOM_CONTROL_TOKEN" else
                          "phone number" if self.config_edit_field == "LAST_DIALED_FALLBACK" else "URL")
            self.banner = f"Editing {lab}: enter {value_type}, then select SAVE URL. CLEAR replaces it."
            self.banner_until = time.time() + 8
            return
        if lab == "SAVE URL" and self.config_edit_field:
            value = self.text.strip()
            if value and self.config_edit_field == "LAST_DIALED_FALLBACK":
                phone = re.sub(r"[\s().-]", "", value)
                if not re.fullmatch(r"\+?\d{5,20}", phone):
                    self.banner, self.banner_until = "Enter 5-20 digits; optional + at start", time.time() + 5
                    return
                value = phone
            if value and self.config_edit_field not in ("ROOM_CONTROL_TOKEN", "LAST_DIALED_FALLBACK") and not value.lower().startswith(("http://", "https://")):
                self.banner, self.banner_until = "Use a full http:// or https:// endpoint URL", time.time() + 5
                return
            if value and self.config_edit_field == "AC_SET_TEMP_URL" and not any(token in value for token in ("{temperature}", "{temp}")):
                self.banner, self.banner_until = "AC TEMP URL must contain {temperature}", time.time() + 5
                return
            ok, msg = set_app_config(self.config_edit_field, value)
            self.banner, self.banner_until = msg, time.time() + 6
            if ok:
                self.config_edit_field = None
                self.text = ""
                self.mode, self.cur = "api_setup", None
            return
        if lab == "PREDICTION":
            self.prediction_enabled = not self.prediction_enabled
            self.banner = "Word prediction " + ("ON" if self.prediction_enabled else "OFF")
            self.banner_until = time.time() + 3
            return
        if lab == "PRED ON": self.prediction_enabled = True; return
        if lab == "PRED OFF": self.prediction_enabled = False; return
        if lab == "AUTO SHARE":
            self.caregiver_auto_share = not self.caregiver_auto_share
            if not self.caregiver_auto_share and self.caregiver_share_timer:
                self.caregiver_share_timer.cancel()
            self.banner = "Caregiver auto-share " + ("ON" if self.caregiver_auto_share else "OFF")
            if self.caregiver_auto_share and self.text.strip():
                self.mark_typing()
            self.banner_until = time.time() + 4
            return
        if lab == "SEND TEXT":
            self.send_caregiver_text()
            return
        if lab == "CALL CARETAKER":
            self.pending_service = "CARETAKER"
            self.confirm_deadline = time.time() + 8
            self.mode, self.cur = "confirm", None
            self.banner, self.banner_until = "Confirm caregiver call or cancel within 8 seconds", time.time() + 8
            return
        if lab == "CURSOR ON": self.cursor_mode = True; return
        if lab == "CURSOR OFF":
            self.cursor_mode = False
            if self.dragging:
                try:
                    import pyautogui; pyautogui.mouseUp()
                except Exception: pass
                self.dragging = False
            return
        if lab in ("CTRL", "ALT", "WIN"):
            key = {"CTRL": "ctrl", "ALT": "alt", "WIN": "win"}[lab]
            if key in self.modifiers: self.modifiers.remove(key)
            else: self.modifiers.add(key)
            self.banner = "Modifiers: " + (", ".join(sorted(self.modifiers)) or "none")
            self.banner_until = time.time() + 3
            return
        if lab in ("COPY", "PASTE"):
            try:
                import pyperclip
                if lab == "COPY": pyperclip.copy(self.text)
                else:
                    pasted = pyperclip.paste()
                    self.mark_typing()
                    self.text += pasted
                self.banner = "Copied text" if lab == "COPY" else "Pasted clipboard text"
            except Exception:
                self.banner = "Clipboard unavailable: install pyperclip"
            self.banner_until = time.time() + 3
            return
        if lab in ("VOL+", "VOL-"):
            self.banner = "Volume adjusted" if set_volume(lab == "VOL+") else "Volume control unavailable"
            self.banner_until = time.time() + 3
            return
        if lab in ("BRIGHT+", "BRIGHT-"):
            try:
                import screen_brightness_control as sbc
                old = sbc.get_brightness()[0]
                level = max(5, min(100, old + (10 if lab == "BRIGHT+" else -10)))
                sbc.set_brightness(level)
                self.banner = f"Screen brightness {level}%"
            except Exception:
                self.banner = "Brightness control unavailable (install screen-brightness-control)"
            self.banner_until = time.time() + 3
            return
        if lab == "ESC FULL":
            try:
                self.fullscreen = not self.fullscreen
                mode = cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL
                cv2.setWindowProperty(self.win, cv2.WND_PROP_FULLSCREEN, mode)
                self.banner = "Fullscreen ON" if self.fullscreen else "Fullscreen OFF"
            except Exception: self.banner = "Could not change window mode"
            self.banner_until = time.time() + 3
            return
        if lab == "CLICK":
            try:
                import pyautogui; pyautogui.click()
                self.banner = "Mouse click"
            except Exception: self.banner = "Mouse control unavailable (install pyautogui)"
            self.banner_until = time.time() + 2
            return
        if lab == "DRAG":
            try:
                import pyautogui
                self.dragging = not self.dragging
                pyautogui.mouseDown() if self.dragging else pyautogui.mouseUp()
                self.banner = "Drag started" if self.dragging else "Drag released"
            except Exception: self.banner = "Mouse control unavailable (install pyautogui)"
            self.banner_until = time.time() + 2
            return
        if lab in ("SCROLL UP", "SCROLL DOWN"):
            try:
                import pyautogui; pyautogui.scroll(3 if lab == "SCROLL UP" else -3)
            except Exception: self.banner = "Mouse control unavailable (install pyautogui)"
            self.banner_until = time.time() + 2
            return
        if lab in ("AC ON", "AC OFF"):
            self.request_climate_action(lab)
            return
        if lab in ("TEMP+", "TEMP-"):
            self.request_climate_action(lab)
            return
        if lab == "GESTURES":
            self.banner = "Gestures: 4 rapid alternating turns = siren flash only (no call); RIGHT-then-LEFT = numbers; LEFT-then-RIGHT = symbols; triple RIGHT = emergency; triple LEFT = last number."
            self.banner_until = time.time() + 8
            return
        if lab.startswith("EMOJI:"):
            self.text += EMOJI_MAP.get(lab.split(":", 1)[1], "")
            self.mark_typing(); self.n_keys += 1
            self.banner = f"{lab.split(':', 1)[1]} emoji added (Unicode kept in message)"
            self.banner_until = time.time() + 4
            return
        if lab == "CANCEL":
            self.mode, self.cur = ("scan" if self.prev_mode == "scan" else "board"), None
            self.pending_service = None
            self.banner, self.banner_until = "Emergency cancelled", time.time() + 3
            return
        if lab == "SIREN":
            self.trigger_siren("manual")
            return
        if lab in ("AMBULANCE", "POLICE", "FIRE", "FAMILY", "CARETAKER", "LAST CALLED"):
            self.pending_service = lab
            self.confirm_deadline = time.time() + 8
            self.mode, self.cur = "confirm", None
            self.banner, self.banner_until = f"Confirm call: {lab}. Select CONFIRM CALL or CANCEL.", time.time() + 8
            return
        if lab == "SOS":
            self.start_emergency("SOS tile"); return
        if lab in self.devices:
            self.request_device_toggle(lab)
            return
        if lab in self.phrases and self.phrases[lab].strip():
            self.say(self.phrases[lab], lab)
            return
        if lab in self.phrases:
            self.banner, self.banner_until = f"{lab} is empty: open EDIT PHRASES to personalize it", time.time() + 4
            return
        if lab == "SPACE": self.do_space()
        elif lab == "DEL":
            self.mark_typing(); self.text = self.text[:-1]; self.n_del += 1
            self.dwell = min(DWELL_MAX, self.dwell + 0.1); self.streak = 0
            return
        elif lab == "CLEAR":
            self.text = ""
            self.caregiver_last_shared = ""
            if self.caregiver_share_timer:
                self.caregiver_share_timer.cancel()
        elif lab == "SPEAK":
            if self.text.strip(): speak(self.text)
        elif lab == "SMS": self.send_text_sms()
        elif lab in (".", "?"): self.punct(lab)
        elif lab.startswith("#"): self.accept(int(lab[1]))
        elif lab in ("BACKSPACE",):
            self.mark_typing(); self.text = self.text[:-1]
        elif len(lab) == 1 and self.modifiers:
            system_key(lab.lower(), tuple(sorted(self.modifiers)))
            self.modifiers.clear()
        elif len(lab) == 1 and self.mode in ("numbers", "symbols"):
            self.type_char(lab)
        else: self.type_char(lab.lower())
        self.streak += 1
        if self.streak >= 6:                      # adaptive dwell: speed up while the user is accurate
            self.dwell = max(DWELL_MIN, self.dwell - 0.05); self.streak = 0

    def need(self, lab):
        if self.mode == "emergency": return 0.8 if lab == "CANCEL" else 1.5
        if lab == "SOS": return 2.0
        if lab.startswith("#"): return max(0.30, 0.55 * self.dwell)
        if self.mode == "board": return 1.2 * self.dwell
        return self.dwell

    def active(self):
        if self.mode in self.page_cache:
            return self.page_cache[self.mode]
        if self.mode == "kb" and self.config_edit_field:
            return [("SAVE URL" if lab == "SMS" else lab, rect) for lab, rect in self.kb]
        if self.mode == "kb" and self.phrase_edit_label:
            return [("SAVE PHRASE" if lab == "SMS" else ("CANCEL PHRASE" if lab == "MENU" else lab), rect)
                    for lab, rect in self.kb]
        return {"kb": self.kb, "board": self.bd, "emergency": self.em, "scan": self.scan_keys}[self.mode]

    # ----- emergency -----
    def start_emergency(self, reason):
        if self.mode == "emergency":
            return
        self.prev_mode = self.mode
        self.reason, self.mode, self.cur = reason, "emergency", None
        self.em_deadline = time.time() + EMERGENCY_COUNTDOWN
        self.stats["emergencies"] += 1
        speak("Emergency mode. Look at cancel to stop.")
        beep()

    def trigger_siren(self, reason="head-shake"):
        """Start local siren audio and flashing SOS overlay; this function never dials."""
        now = time.time()
        self.sos_reason = reason
        self.sos_until = now + SIREN_SECONDS
        self.banner = "HEAD SHAKE DETECTED - SIREN ACTIVE (NO CALL)" if reason == "head-shake" else "Siren activated"
        self.banner_until = now + 5
        play_siren()
        speak("Emergency siren activated. No call has been placed.")

    def dispatch(self, service):
        now = time.time()
        last = get_last_dialed()
        if service == "SIREN":
            targets = []
        elif service == "LAST CALLED":
            targets = [("last dialed number", last)]
        elif service == "CARETAKER":
            targets = [("caregiver", CARETAKER_PHONE)]
        elif service == "FAMILY":
            targets = [("family", EMERGENCY_NUMBERS.get("FAMILY", ""))]
        else:
            targets = [("emergency services" if service == "DEFAULT" else service.lower(),
                        EMERGENCY_NUMBERS.get(service, EMERGENCY_NUMBERS["DEFAULT"]))]
        who = targets[0][0] if targets else "help (siren only)"
        msg = f"EMERGENCY: {who} needed. Trigger: {self.reason}. {HOME_ADDRESS}".strip()
        if service == "CARETAKER" and EMERGENCY_LIVE:
            threading.Thread(target=caregiver_webhook, args=("call_request", msg), daemon=True).start()
        results, reached = [], False
        for name, num in targets:
            ok, text = dial_number(num, name)
            results.append(text); reached = reached or ok
        if targets and not reached and service not in ("LAST CALLED", "CARETAKER"):
            ok, text = dial_number(last, "last dialed number")
            results.append("fallback: " + text)
        if EMERGENCY_LIVE:
            for n in {EMERGENCY_NUMBERS.get("FAMILY", ""), CARETAKER_PHONE, last} - {""}:      # caregiver/family/last number may receive SMS
                send_sms(n, msg)
            send_alert(msg)                                              # online alert (optional)
        play_siren()                                                     # loud alarm generated by the laptop
        self.sos_reason = "emergency"
        result = " | ".join(results) or "Siren on"
        with open(app_data_path("emergency_log.txt"), "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg} | {result}\n")
        if EMERGENCY_LIVE:
            speak((f"Emergency. Call request sent for {who}. Check the calling app." if reached else
                   f"Emergency. Could not start a call to {who}.") if targets else "Emergency. I need help.")
        else:
            result = ("DRY RUN — NO CALL OR ALERT SENT | " + result) if targets else "SIREN ONLY — NO CALL OR ALERT SENT"
            speak("Demo dry run. No call or alert was sent. Siren is active.")
        self.banner, self.banner_until = result, now + 8
        self.sos_until = now + SIREN_SECONDS
        self.mode, self.cur = ("scan" if self.prev_mode == "scan" else "board"), None

    def check_head_turn(self, yaw, now):
        if self.yaw_base is None:
            self.yaw_base = yaw
        dev = TURN_SIGN * (yaw - self.yaw_base)
        direction = 1 if dev > YAW_THRESH else (-1 if dev < -YAW_THRESH else 0)
        if self.turned == 0 and direction:
            self.turned = direction
        elif self.turned and abs(dev) < YAW_THRESH * 0.4:
            completed = self.turned
            self.turned = 0
            self.turn_times = [(sgn, t) for sgn, t in self.turn_times if now - t < TURN_WINDOW]
            self.turn_times.append((completed, now))
            self.headshake_turns = [(sgn, t) for sgn, t in self.headshake_turns if now - t < HEADSHAKE_WINDOW]
            self.headshake_turns.append((completed, now))
            recent = [sgn for sgn, _ in self.turn_times]
            shake = self.headshake_turns[-HEADSHAKE_TURNS:]
            alternating = all(shake[i][0] != shake[i - 1][0] for i in range(1, len(shake)))
            shake_fast = len(shake) == HEADSHAKE_TURNS and shake[-1][1] - shake[0][1] <= HEADSHAKE_WINDOW
            if now >= self.headshake_cooldown_until and alternating and shake_fast:
                self.turn_times = []
                self.headshake_turns = []
                self.headshake_cooldown_until = now + HEADSHAKE_COOLDOWN
                self.trigger_siren("head-shake")
            elif len(recent) >= 3 and recent[-3:] == [1, 1, 1]:
                self.turn_times = []
                self.headshake_turns = []
                self.start_emergency("triple head turn right")
            elif len(recent) >= 3 and recent[-3:] == [-1, -1, -1]:
                self.turn_times = []
                self.headshake_turns = []
                self.start_emergency("triple head turn left (last number)")
            elif self.mode != "emergency" and len(recent) >= 2 and recent[-2:] == [1, -1]:
                self.turn_times = []
                self.mode, self.cur = "numbers", None
                self.banner, self.banner_until = "Number keypad", now + 3
            elif self.mode != "emergency" and len(recent) >= 2 and recent[-2:] == [-1, 1]:
                self.turn_times = []
                self.mode, self.cur = "symbols", None
                self.banner, self.banner_until = "Symbols keypad", now + 3
        if abs(dev) < 0.06:
            self.yaw_base += 0.02 * (yaw - self.yaw_base)

    def check_room_pose(self, pitch, roll, now):
        """Use guarded double-nods for room controls or My Phrases head navigation."""
        if self.mode not in ("room", "my_phrases") or now < self.room_pose_lockout:
            self.pose_base, self.pose_state, self.roll_state = None, 0, 0
            if self.mode not in ("room", "my_phrases"):
                self.pose_events.clear()
            return
        if self.pose_base is None:
            self.pose_base = pitch
        dev = PITCH_UP_SIGN * (pitch - self.pose_base)
        direction = 1 if dev > POSE_THRESH else (-1 if dev < -POSE_THRESH else 0)
        if self.pose_state == 0 and direction:
            self.pose_state = direction
        elif self.pose_state in (-1, 1) and abs(dev) < POSE_THRESH * 0.35:
            event = "nod_up" if self.pose_state == 1 else "nod_down"
            self.pose_state = 0
            self.pose_events = [(name, t) for name, t in self.pose_events if now - t < ROOM_GESTURE_WINDOW]
            self.pose_events.append((event, now))
            same = [name for name, _ in self.pose_events]
            if len(same) >= 2 and same[-2:] == [event, event]:
                self.pose_events.clear()
                if self.mode == "room" and event == "nod_up":
                    self.press("TEMP+")  # wakes cooling mode and raises target by one degree
                elif self.mode == "room":
                    self.press("TEMP-")  # lowers target by one degree
                else:
                    items = [lab for lab, _ in self.page_cache["my_phrases"]]
                    if event == "nod_up":
                        self.phrase_focus_i = (self.phrase_focus_i + 1) % len(items)
                        self.banner = f"Head focus: {items[self.phrase_focus_i]} (double nod down to activate)"
                        self.banner_until = now + 3
                    else:
                        self.press(items[self.phrase_focus_i % len(items)])
                self.room_pose_lockout = now + 2.0
        if abs(dev) < POSE_THRESH * 0.25:
            self.pose_base += 0.03 * (pitch - self.pose_base)

        if self.mode != "room":
            return

        left_tilt = ROLL_LEFT_SIGN * roll < -ROLL_THRESH_DEG
        right_tilt = ROLL_LEFT_SIGN * roll > ROLL_THRESH_DEG
        roll_dir = -1 if left_tilt else (1 if right_tilt else 0)
        if self.roll_state == 0 and roll_dir:
            self.roll_state = roll_dir
        elif self.roll_state and roll_dir == 0:
            completed, self.roll_state = self.roll_state, 0
            event = "tilt_left" if completed < 0 else "tilt_right"
            self.pose_events = [(name, t) for name, t in self.pose_events if now - t < ROOM_GESTURE_WINDOW]
            self.pose_events.append((event, now))
            same = [name for name, _ in self.pose_events]
            if len(same) >= 2 and same[-2:] == [event, event]:
                self.pose_events.clear()
                if event == "tilt_left":
                    self.toggle_room_device("LIGHT")
                else:
                    self.toggle_room_device("FAN")
                self.room_pose_lockout = now + 2.0

    def toggle_room_device(self, name):
        self.request_device_toggle(name, "head gesture")

    # ----- low light, scanning keyboard, SMS -----
    def update_lighting(self, now, bright):
        if not self.lowlight:
            if bright < DARK_ON:
                self.dark_t = self.dark_t or now
                if now - self.dark_t > 1.0: self.set_lowlight(True)
            else:
                self.dark_t = None
        elif now - self.low_since > 30:                      # stay in fill-light mode at least 30 s (avoids flicker)
            if bright > DARK_OFF:
                self.bright_t = self.bright_t or now
                if now - self.bright_t > 3.0: self.set_lowlight(False)
            else:
                self.bright_t = None

    def set_lowlight(self, on):
        if on == self.lowlight:
            return
        self.lowlight, FILL["on"] = on, on
        self.dark_t = self.bright_t = None
        if on:
            self.low_since = time.time()
            self.old_brightness = set_screen_brightness(100)
            self.banner, self.banner_until = "Low light: screen fill-light ON", time.time() + 5
            speak("Low light detected. The screen is now lighting your face.")
        else:
            if self.old_brightness is not None: set_screen_brightness(self.old_brightness)
            self.banner, self.banner_until = "Light is fine again", time.time() + 3

    def enter_scan(self, auto):
        if self.mode == "scan":
            return
        self.scan_return_mode = self.mode
        if self.mode == "kb":
            rows = {}
            for lab, rect in self.kb:
                if lab != "MENU":
                    rows.setdefault(rect[1], []).append(lab)
            self.scan_rows = [rows[y] for y in sorted(rows)] + [["SOS"]]
            self.scan_keys = self.kb_scan
        else:
            self.scan_keys = list(self.active())
            rows = {}
            for lab, rect in self.scan_keys:
                rows.setdefault(rect[1], []).append(lab)
            self.scan_rows = [rows[y] for y in sorted(rows)]
        self.label_row = {lab: i for i, row in enumerate(self.scan_rows) for lab in row}
        self.mode, self.cur, self.scan_auto = "scan", None, auto
        self.scan_stage, self.scan_row, self.scan_i, self.scan_t = "row", 0, 0, time.time() + 1.0
        self.no_face = 0
        speak("Switch scanning on. Close your eyes for one second, or press any key, to select.")

    def leave_scan(self):
        self.mode, self.scan_auto, self.cur = self.scan_return_mode, False, None

    def scan_tick(self, now):
        if now - self.scan_t >= SCAN_STEP:
            self.scan_t = now
            self.scan_i += 1
            n = len(self.scan_rows) if self.scan_stage == "row" else len(self.scan_rows[self.scan_row])
            if self.scan_i >= n:
                self.scan_i = 0
                self.scan_stage = "row"                      # a full pass with no choice -> back to the rows
        if self.scan_trigger:
            self.scan_trigger = False
            if self.scan_stage == "row":
                self.scan_row, self.scan_stage, self.scan_i = self.scan_i, "item", 0
            else:
                lab = self.scan_rows[self.scan_row][self.scan_i]
                self.scan_stage, self.scan_i = "row", 0
                self.press(lab)
            self.scan_t = now

    def send_text_sms(self):
        msg = self.text.strip()
        if not msg:
            return
        to = EMERGENCY_NUMBERS.get("FAMILY") or get_last_dialed()
        status = send_sms(to, msg)
        self.banner, self.banner_until = f"SMS to {to or '(no number)'}: {status}", time.time() + 5
        self.text = ""

    # ----- calibration -----
    def calibrate_now(self):
        if MOUSE:
            return True
        res = calibrate(self.win, self.cam, self.trk, self.W, self.H)
        if res[0] is None:
            self.banner = "Calibration failed: check eye visibility/glare, then press C to retry"
            self.banner_until = time.time() + 8
            return False
        self.mapper, self.ear_thr, self.err_deg = res
        self.offset[:] = 0
        if self.mode == "scan" and self.scan_return_mode == "kb":
            self.leave_scan()
        self.banner = f"Calibration complete: {self.err_deg:.1f}° median error"
        self.banner_until = time.time() + 5
        return True

    # ----- one frame -----
    def step(self):
        W, H, now = self.W, self.H, time.time()
        if MOUSE:
            frame, r = np.zeros((720, 1280, 3), np.uint8), None
            self.gaze = 0.6 * self.gaze + 0.4 * self.mouse
        else:
            ok, frame = self.cam.read()
            if not ok:
                return
            frame = cv2.flip(frame, 1)
            self.stats["frames"] += 1
            bright = float(frame[::8, ::8].mean())
            self.bright = bright
            if bright < DARK_ON: self.stats["dark_frames"] += 1
            self.update_lighting(now, bright)
            frame = enhance(frame, bright)                    # brighten dark frames before tracking
            r = self.trk.process(frame)
            if r is None:
                self.stats["face_lost_frames"] += 1
                self.closed_since = None
                self.no_face, self.face_ok = self.no_face + 1, 0
                if self.lowlight and self.mode == "kb" and self.no_face > 90:
                    self.enter_scan(True)                     # camera cannot see the user -> scanning keyboard
            else:
                self.no_face, self.face_ok = 0, self.face_ok + 1
                if self.scan_auto and self.mode == "scan" and self.face_ok > 120:
                    self.leave_scan()                         # camera sees the user again -> gaze keyboard
                if r["ear"] < self.ear_thr:
                    self.closed_since = self.closed_since or now      # freeze gaze while closed
                else:
                    if self.closed_since is not None:
                        d = now - self.closed_since
                        if self.mode == "kb" and d >= SPEAK_CLOSE:
                            self.stats["speak_closes"] += 1; self.press("SPEAK")
                        elif self.mode == "kb" and d >= LONG_CLOSE:
                            self.stats["space_closes"] += 1; self.press("SPACE")
                        elif self.mode == "scan" and d >= LONG_CLOSE:
                            self.scan_trigger = True          # long eye-close = select in scanning mode
                        elif d < LONG_CLOSE:
                            self.stats["blinks_ignored"] += 1
                            if d >= BLINK_MIN:                      # deliberate blink: 3 in a row = emergency
                                self.blink_times = [t for t in self.blink_times if now - t < BLINK_WINDOW] + [now]
                                if len(self.blink_times) >= 3:
                                    self.blink_times = []
                                    if self.mode == "emergency":
                                        self.trigger_siren("triple-blink")
                                    else:
                                        self.start_emergency("triple blink")
                        self.closed_since = None
                    if self.mapper is not None and r.get("iris_quality", 0) > 0:
                        raw = self.mapper.predict(r["feat"]) * [W, H] + self.offset
                        # Faster than the previous 25% update, while keeping
                        # small landmark jitter from bouncing the dwell target.
                        self.gaze = 0.60 * self.gaze + 0.40 * np.clip(raw, 0, [W - 1, H - 1])
                    self.check_head_turn(r["yaw"], now)
                    self.check_room_pose(r["pitch"], r["roll"], now)
                    if self.cursor_mode:
                        try:
                            import pyautogui
                            pyautogui.moveTo(int(self.gaze[0]), int(self.gaze[1]), duration=0)
                        except Exception:
                            pass

        if self.mode == "confirm" and now >= self.confirm_deadline:
            self.pending_service = None
            self.mode, self.cur = "board", None
            self.banner, self.banner_until = "Call confirmation timed out; no call placed", now + 5
        if self.mode == "emergency" and now >= self.em_deadline:
            self.dispatch("LAST CALLED" if "last number" in self.reason else "DEFAULT")

        if self.mode == "scan":
            self.scan_tick(now)
            self.draw(frame, r, now, 0.0)
            return

        # Keep ordinary keys steady, but do not let the sticky margin overlap
        # adjacent letters; the resolver also catches tiny gaps between letters.
        keys = self.active()
        hit = gaze_hit(keys, self.gaze, self.cur, self.mode)
        if hit != self.cur:
            self.cur, self.start = hit, now
        prog = 0.0
        if self.cur and self.closed_since is None:
            prog = max(0.0, now - self.start) / self.need(self.cur)
        if prog >= 1:
            lab, rect = self.cur, dict(keys)[self.cur]
            mode_before = self.mode
            self.press(lab)
            if (not MOUSE and lab not in ("DEL", "MENU", "KEYBOARD") and rect[2] - rect[0] < 0.18 * W):
                center = np.array([(rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2])   # automatic drift correction
                lim = min(W, H) * 0.08
                self.offset = np.clip(self.offset + 0.15 * (center - self.gaze), -lim, lim)
            if self.mode != mode_before: self.cur = None
            self.start, prog = now + LOCKOUT, 0.0
        self.draw(frame, r, now, prog)

    # ----- drawing -----
    def draw(self, frame, r, now, prog):
        W, H = self.W, self.H
        canvas = (self.bg_light if self.lowlight else self.bg_dark).copy()
        dark = not self.lowlight
        panel = (51, 39, 29) if dark else (235, 224, 210)
        tile = (73, 55, 42) if dark else (255, 252, 249)
        tile_alt = (93, 70, 52) if dark else (242, 233, 222)
        ink = (250, 248, 244) if dark else (59, 43, 28)
        muted = (197, 179, 160) if dark else (120, 103, 83)
        accent = (172, 211, 87) if dark else (111, 145, 28)
        danger = (62, 65, 190)
        keys = self.active()
        s_list = self.sugs() if self.mode == "kb" and not self.config_edit_field and not self.phrase_edit_label and self.prediction_enabled else []
        emoji_icons = {}
        if self.mode == "emojis" and self.emoji_font_path:
            for lab, rect in keys:
                if lab.startswith("EMOJI:"):
                    x1, y1, x2, y2 = rect
                    size = max(20, min(int((y2 - y1) * 0.56), int((x2 - x1) * 0.28)))
                    icon = emoji_bitmap(EMOJI_MAP.get(lab.split(":", 1)[1], ""), size, self.emoji_font_path)
                    if icon is not None:
                        emoji_icons[lab] = icon

        # A quiet floating tray visually separates the virtual keyboard from the background.
        tray_y = int(H * (0.16 if self.mode in ("kb", "scan") else 0.25))
        rounded_box(canvas, (int(W * 0.012) + 4, tray_y + 5, int(W * 0.988) + 4, int(H * 0.965) + 5),
                    (10, 14, 23) if dark else (210, 197, 181), max(14, int(H * 0.025)))
        rounded_box(canvas, (int(W * 0.012), tray_y, int(W * 0.988), int(H * 0.965)), panel,
                    max(14, int(H * 0.025)))

        for lab, rect in keys:
            x1, y1, x2, y2 = rect
            col = tile if len(lab) == 1 or lab.startswith("#") else tile_alt
            if lab in self.devices and self.devices[lab]: col = accent
            if lab == "SOS" or lab in ALERTS or (self.mode == "emergency" and lab not in ("CANCEL", "SIREN")):
                col = danger
            elif lab == "SIREN": col = (15, 143, 239)
            elif lab == "CANCEL": col = accent if self.mode == "emergency" else tile_alt
            if lab == self.flash and now - self.flash_t < 0.25: col = accent
            elif lab == self.cur: col = (227, 155, 79) if dark else (247, 220, 184)
            elif self.mode == "my_phrases" and lab == self.page_cache["my_phrases"][self.phrase_focus_i % len(self.page_cache["my_phrases"])][0]:
                col = (227, 155, 79) if dark else (247, 220, 184)
            if self.mode == "scan":
                ri = self.label_row.get(lab, -1)
                if self.scan_stage == "row" and ri == self.scan_i: col = (79, 155, 227)
                elif self.scan_stage == "item" and ri == self.scan_row:
                    col = (60, 180, 245) if lab == self.scan_rows[self.scan_row][self.scan_i] else tile_alt
            radius = max(8, int(H * 0.018))
            rounded_box(canvas, (x1 + 2, y1 + 4, x2 + 2, y2 + 4), (12, 16, 24) if dark else (189, 204, 216), radius)
            rounded_box(canvas, rect, col, radius)
            if lab == self.cur and prog > 0:
                bar_y = y2 - max(5, int(H * 0.009))
                cv2.rectangle(canvas, (x1 + 5, bar_y),
                              (min(x2 - 5, x1 + 5 + int((x2 - x1 - 10) * min(prog, 1))), y2 - 3), accent, -1)
            label = lab
            if lab == "AUTO SHARE":
                label = f"AUTO SHARE {'ON' if self.caregiver_auto_share else 'OFF'}"
            if lab.startswith("#"):
                i = int(lab[1]); label = f"{i + 1}  {s_list[i]}" if i < len(s_list) else ""
            if lab.startswith("EMOJI:"):
                label = "" if lab in emoji_icons else lab.split(":", 1)[1].title()
            if lab in self.devices: label = f"{lab}  {'ON' if self.devices[lab] else 'OFF'}"
            if lab in MY_PHRASE_KEYS:
                label = lab if not self.phrases.get(lab, "").strip() else opencv_safe_message(self.phrases[lab].strip())
            is_letter = len(lab) == 1 and lab.isalpha() and self.mode == "kb"
            scale = (1.65 if is_letter else (1.15 if len(label) <= 7 else 0.95)) * (H / 720)
            if label:
                fitted_text(canvas, label, rect, ink, scale)

        for lab, rect in keys:
            icon = emoji_icons.get(lab)
            if icon is None:
                continue
            x1, y1, x2, y2 = rect
            ih, iw = icon.shape[:2]
            x = (x1 + x2 - iw) // 2
            y = y1 + max(1, (y2 - y1 - ih - 18) // 2)
            alpha = icon[:, :, 3:4].astype(np.float32) / 255.0
            area = canvas[y:y + ih, x:x + iw]
            area[:] = (icon[:, :, :3].astype(np.float32) * alpha + area.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
            name = lab.split(":", 1)[1].title()
            (tw, _), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_DUPLEX, 0.34 * H / 540, 1)
            cv2.putText(canvas, name, ((x1 + x2 - tw) // 2, y2 - 6), cv2.FONT_HERSHEY_DUPLEX,
                        0.34 * H / 540, ink, 1, cv2.LINE_AA)

        # Composer and camera preview share a clear top bar with strong text contrast.
        composer = (int(W * 0.012), int(H * 0.018), int(W * 0.70), int(H * 0.158))
        rounded_box(canvas, (composer[0] + 3, composer[1] + 4, composer[2] + 3, composer[3] + 4),
                    (10, 14, 23) if dark else (210, 197, 181), max(12, int(H * 0.02)))
        rounded_box(canvas, composer, panel, max(12, int(H * 0.02)))
        composer_hint = (self.banner if now < self.banner_until and self.banner else
                         (f"{APP_BRAND} · Look at keys to type · SPACE adds a gap"
                          if self.mode == "kb" else f"{APP_BRAND} · NO SUBSCRIPTION | OFFLINE CORE"))
        fitted_text(canvas, composer_hint, (composer[0] + 12, composer[1] + 4, composer[2] - 12, composer[1] + int(H * 0.052)),
                    accent if composer_hint != "YOUR MESSAGE" else muted, 0.62 * H / 720)
        if self.config_edit_field and len(self.text) > 40:
            visible_text = self.text[:20] + "…" + self.text[-19:]
        else:
            visible_text = self.text[-38:]
        while visible_text and (visible_text[0] in "\ufe0f\u200d" or 0x1F3FB <= ord(visible_text[0]) <= 0x1F3FF):
            visible_text = visible_text[1:]
        shown_text = (visible_text if self.text else "Select BIG ABC for large letter keys") + (" |" if self.text else "")
        draw_composer_message(canvas, shown_text,
                              (composer[0] + 12, composer[1] + int(H * 0.052), composer[2] - 12, composer[3] - 8),
                              ink, 1.25 * H / 720, self.emoji_font_path)

        px, py, pw, ph = int(W * 0.72), int(H * 0.018), int(W * 0.268), int(H * 0.14)
        rounded_box(canvas, (px - 3, py - 3, px + pw + 3, py + ph + 3), panel, max(10, int(H * 0.015)))
        if self.mode != "scan":
            canvas[py:py + ph, px:px + pw] = cv2.resize(frame, (pw, ph))
        cv2.rectangle(canvas, (px, py), (px + pw, py + ph), accent if r is not None else danger, 2, cv2.LINE_AA)
        eye_count = int(round(2 * r.get("iris_quality", 1))) if isinstance(r, dict) else 0
        warn = ("MOUSE DEMO" if MOUSE else ("FACE NOT FOUND" if r is None else
                ("IRIS NOT TRACKED" if eye_count == 0 else ("ONE IRIS TRACKED" if eye_count == 1 else
                 ("LOW LIGHT" if self.lowlight else "CAMERA")))))
        fitted_text(canvas, warn, (px + 4, py + ph - int(H * 0.045), px + pw - 4, py + ph - 2),
                    (255, 255, 255), 0.65 * H / 720)

        title = {"board": "QUICK ACTIONS", "my_phrases": "MY PHRASES", "phrase_setup": "EDIT PHRASES", "numbers": "NUMBER KEYPAD",
                 "letters_a": "BIG LETTERS A-M", "letters_b": "BIG LETTERS N-Z", "symbols": "SYMBOLS",
                 "emojis": "EMOJI SHORTCUTS", "controls": "KEYBOARD CONTROLS", "cursor": "CURSOR CONTROL",
                 "room": "SMART ROOM", "api_setup": "DEVICE API SETUP", "emergency": "EMERGENCY OPTIONS", "confirm": "CALL CONFIRMATION",
                 "scan": f"ACCESSIBILITY SCAN · {self.scan_return_mode.replace('_', ' ').upper()}", "kb": "EYE-TYPING KEYBOARD"}.get(self.mode, "EYETYPE")
        if self.mode != "kb" and (self.mode != "scan" or self.scan_return_mode != "kb"):
            fitted_text(canvas, title, (int(W * 0.025), int(H * 0.18), int(W * 0.975), int(H * 0.225)), accent, 0.88 * H / 720)

        if self.mode == "emergency":
            countdown_text = (f"Call starts in {max(0, self.em_deadline - now):.0f}s unless cancelled"
                              if EMERGENCY_LIVE else
                              f"DRY RUN — no call will be placed — {max(0, self.em_deadline - now):.0f}s")
            fitted_text(canvas, countdown_text,
                        (int(W * 0.025), int(H * 0.245), int(W * 0.975), int(H * 0.285)), danger, 0.75 * H / 720)
        elif self.mode == "confirm":
            confirm_text = (f"Confirm {self.pending_service} call? {max(0, self.confirm_deadline - now):.0f}s"
                            if EMERGENCY_LIVE else
                            f"DRY RUN — no call will be placed — {max(0, self.confirm_deadline - now):.0f}s")
            fitted_text(canvas, confirm_text,
                        (int(W * 0.025), int(H * 0.245), int(W * 0.975), int(H * 0.285)), danger, 0.75 * H / 720)
        elif self.mode == "room":
            fitted_text(canvas, "Nod twice: adjust AC  |  Tilt left/right twice: toggle light/fan",
                        (int(W * 0.025), int(H * 0.245), int(W * 0.975), int(H * 0.285)), muted, 0.7 * H / 720)
        elif self.mode == "my_phrases":
            fitted_text(canvas, "Double nod up: move focus  |  Double nod down: activate  |  gaze or switch-scan also works",
                        (int(W * 0.025), int(H * 0.245), int(W * 0.975), int(H * 0.285)), muted, 0.68 * H / 720)
        elif self.mode == "phrase_setup":
            fitted_text(canvas, "Select a phrase to edit; type up to 120 characters, then choose SAVE PHRASE. Changes stay local.",
                        (int(W * 0.025), int(H * 0.245), int(W * 0.975), int(H * 0.285)), muted, 0.68 * H / 720)
        elif self.mode in ("letters_a", "letters_b"):
            fitted_text(canvas, "Gaze at letters to type. SPACE adds a word gap; BACKSPACE corrects; NEXT/PREVIOUS changes page.",
                        (int(W * 0.025), int(H * 0.245), int(W * 0.975), int(H * 0.285)), muted, 0.68 * H / 720)
        elif self.mode == "scan":
            fitted_text(canvas, "Switch-scan this page: close eyes for 1 second or press any key to select",
                        (int(W * 0.025), int(H * 0.245), int(W * 0.975), int(H * 0.285)), muted, 0.68 * H / 720)
        elif self.mode == "api_setup":
            fitted_text(canvas, "Choose a field, type its value, then select SAVE URL. AC TEMP needs {temperature}.",
                        (int(W * 0.025), int(H * 0.235), int(W * 0.975), int(H * 0.27)), muted, 0.72 * H / 720)
            fitted_text(canvas, "LAST NUMBER is saved per laptop; calls request its OS tel: app (or configured GSM). Devices need an API.",
                        (int(W * 0.025), int(H * 0.27), int(W * 0.975), int(H * 0.305)), muted, 0.72 * H / 720)

        # status line
        wpm = 0.0
        if self.t_first and time.time() - self.t_first > 5:
            wpm = (len(self.text) / 5) / ((time.time() - self.t_first) / 60)
        kss = 100 * self.saved / max(1, self.saved + self.n_keys)
        q, qcol = quality(self.err_deg) if not MOUSE else ("MOUSE MODE", (200, 200, 200))
        eye_status = "mouse" if MOUSE else f"iris {eye_count}/2"
        info = (f"Gaze error {self.err_deg:.1f} deg [{q}] | {eye_status} | dwell {self.dwell:.2f}s | WPM {wpm:.1f} | "
                f"saved {kss:.0f}% | drift ({self.offset[0]:.0f},{self.offset[1]:.0f})px"
                f"{' | FILL LIGHT' if self.lowlight else ''}")
        if self.mode == "scan":
            info = f"SCAN: close eyes 0.7 s to select | step {SCAN_STEP:.1f}s"
        fitted_text(canvas, info, (int(W * 0.015), int(H * 0.963), int(W * 0.985), int(H * 0.995)),
                    muted if dark else qcol, 0.57 * H / 720)

        if self.mode != "scan":
            cv2.circle(canvas, (int(self.gaze[0]), int(self.gaze[1])), max(8, int(H * 0.012)), accent, 2, cv2.LINE_AA)
            cv2.circle(canvas, (int(self.gaze[0]), int(self.gaze[1])), max(2, int(H * 0.003)), ink, -1, cv2.LINE_AA)

        # SOS alarm overlay
        if now < self.sos_until:
            if int(now * 4) % 2 == 0:
                rounded_box(canvas, (int(W * 0.035), int(H * 0.39), int(W * 0.965), int(H * 0.59)), danger,
                            max(12, int(H * 0.025)))
            message = "HEAD SHAKE - SIREN ACTIVE - NO CALL" if self.sos_reason == "head-shake" else "EMERGENCY - HELP REQUESTED"
            fitted_text(canvas, message, (int(W * 0.05), int(H * 0.405), int(W * 0.95), int(H * 0.575)),
                        (255, 255, 255), 1.7 * H / 720)
        cv2.imshow(self.win, canvas)

    # ----- main loop + metrics -----
    def run(self):
        ensure_sounds()
        if not MOUSE:
            for _ in range(15): self.cam.read()               # let camera auto-exposure settle
            ok, f = self.cam.read()
            if ok and float(f[::8, ::8].mean()) < DARK_ON:
                self.set_lowlight(True)                       # dark room: calibrate on a bright screen
        if not self.calibrate_now():
            print("Calibration incomplete; switch-scan is available. Check eye visibility/glare and press C to retry.")
            self.enter_scan(False)
            self.banner = "Calibration incomplete — switch-scan works; check glare/lighting and press C to retry"
            self.banner_until = time.time() + 10
        while True:
            self.step()
            k = cv2.waitKey(1) & 0xFF
            if k == 27: break
            if k in (ord("m"), ord("M")) and self.mode in ("kb", "board"): self.mode = "board" if self.mode == "kb" else "kb"; self.cur = None
            if k in (ord("c"), ord("C")): self.calibrate_now()
            if k in (ord("x"), ord("X")):
                stop_sound(); self.sos_until = 0; self.sos_reason = ""
                if self.mode in ("emergency", "confirm"): self.press("CANCEL")
            if k in (ord("k"), ord("K")):
                self.leave_scan() if self.mode == "scan" else self.enter_scan(False)
            if k in (ord("l"), ord("L")): self.set_lowlight(not self.lowlight)
            if self.mode == "scan" and k not in (255, 27) + tuple(ord(c) for c in "cCkKlLxXmM"):
                self.scan_trigger = True                      # any other key = switch press (works in total darkness)
            if MOUSE and k in (ord("s"), ord("S")): self.press("SPACE")
        self.finish()

    def finish(self):
        stop_sound()
        if self.caregiver_share_timer:
            self.caregiver_share_timer.cancel()
        if self.old_brightness is not None:
            set_screen_brightness(self.old_brightness)
        self.pred.save()
        if self.t_first and self.n_keys:
            mins = (time.time() - self.t_first) / 60
            wpm = (len(self.text) / 5) / mins if mins > 0 else 0
            frames = max(1, self.stats["frames"])
            row = [time.strftime("%H:%M:%S"), len(self.text), round(mins, 2), round(wpm, 1),
                   round(self.n_del / max(1, self.n_keys), 3),
                   round(100 * self.saved / max(1, self.saved + self.n_keys), 1),
                   round(self.err_deg, 2), self.stats["space_closes"], self.stats["blinks_ignored"],
                   round(100 * self.stats["face_lost_frames"] / frames, 1),
                   round(100 * self.stats["dark_frames"] / frames, 1), round(self.dwell, 2)]
            new = not os.path.exists(METRICS_FILE)
            with open(METRICS_FILE, "a", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["time", "chars", "minutes", "wpm", "backspace_rate", "keystroke_saving_pct",
                                "gaze_error_deg", "eye_close_spaces", "blinks_ignored", "face_lost_pct",
                                "dark_pct", "final_dwell_s"])
                w.writerow(row)
            print(f"Session: {wpm:.1f} WPM | backspace rate {row[4]:.1%} | keystrokes saved {row[5]}% | gaze error {row[6]} deg")
        self.cam.release(); cv2.destroyAllWindows()


def main():
    try:
        import tkinter as tk
        r = tk.Tk(); W, H = r.winfo_screenwidth(), r.winfo_screenheight(); r.destroy()
    except Exception:
        W, H = 1280, 720
    cam = None
    if not MOUSE:
        cam = cv2.VideoCapture(0)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH); cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)       # not every webcam backend honors this
        if not cam.isOpened():
            print("Camera unavailable: connect a webcam or run with --mouse for the UI demo.")
            return
    win = f"{APP_BRAND} | EyeType"
    cv2.namedWindow(win, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    app = App(W, H, cam, win)
    if MOUSE:
        def on_mouse(e, x, y, f, p): app.mouse = np.array([x, y], float)
        cv2.setMouseCallback(win, on_mouse)
        app.cam = type("NoCam", (), {"release": lambda s: None})()
    app.run()


if __name__ == "__main__":
    main()
