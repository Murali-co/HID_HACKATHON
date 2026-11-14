"use strict";

const get = id => document.getElementById(id);
const video = get("camera");
const id = get("subject");
const displayNameInput = get("display-name");
const operatorIdInput = get("operator-id");
const tokenInput = get("token");
const cameraSelect = get("camera-select");
const refreshBtn = get("refresh-cameras");
const exportCount = get("export-count");
const participantDisplay = get("participant-display");
const diagnosticsDrawer = get("diagnostics-drawer");
const diagnosticsContent = get("diagnostics-content");

const canvas = document.createElement("canvas");
canvas.width = 640;
canvas.height = 480;
const ctx = canvas.getContext("2d", {alpha: false});
const deadlineMs = 2000;

// Load persisted session rows from sessionStorage to survive page reloads
let rows = [];
try {
  const saved = sessionStorage.getItem("liveguard_session_rows");
  if (saved) rows = JSON.parse(saved);
} catch (e) {
  rows = [];
}

let busy = false;
let ready = false;
let retry = null;
let stream = null;
let selectedDeviceId = "";
let activeTrackLabel = "";

const messages = {
  verified: "Live face, matching identity and valid access window.",
  presentation_attack: "The capture showed signs of a presentation attack.",
  outside_schedule: "This campus ID has no active access window.",
  identity_mismatch: "The face did not match the claimed campus ID.",
  identity_not_enrolled: "The claimed campus ID needs operator enrollment.",
  liveness_uncertain: "Center your face and hold still in even light.",
  identity_uncertain: "Face the camera directly and try again.",
  center_face_in_guide: "Move closer and center your face inside the guide.",
  one_face_required: "Make sure only one person is inside the face guide.",
  improve_light_or_focus: "Improve the lighting and hold still.",
  enrollment_face_changed: "Keep still during enrollment — face changed between frames.",
  enrollment_presentation_attack: "Enrollment rejected: presentation attack detected.",
  enrollment_liveness_uncertain: "Enrollment uncertain: ensure live presence and good lighting.",
  template_corrupt: "Enrolled template is corrupted or unreadable. Please re-enroll.",
  invalid_frame_size: "Unsupported frame resolution. Check camera resolution.",
  invalid_frame: "Could not read camera frame. Check camera connection.",
  invalid_embedding: "Could not generate face embedding. Please retry.",
  invalid_display_name: "Participant display name cannot be blank.",
  invalid_evidence: "Biometric evidence failed consistency validation.",
  verification_failed: "The retake could not verify access. Contact the operator.",
  retry_expired_or_invalid: "Start a new verification.",
  policy_unavailable: "The access policy is unavailable or expired. Contact the operator.",
  decision_timeout: "Verification timed out. Start a new scan.",
  camera_pipeline_busy: "The kiosk is busy. Try again.",
  models_unavailable: "The kiosk needs its local face models.",
  pipeline_unavailable: "The verification service is unavailable.",
  audit_unavailable: "The local audit store is unavailable.",
  operator_required: "Enter a valid operator ID and credential.",
  consent_required: "Confirm the participant's consent first.",
  identity_already_enrolled: "This campus ID already has an enrollment.",
  capture_failed: "Check the camera and try again.",
};

function updateExportCounter() {
  if (exportCount) {
    exportCount.textContent = `${rows.length} recorded attempt${rows.length === 1 ? "" : "s"}`;
  }
}

function show(outcome, message, elapsed, score, display) {
  get("result").dataset.outcome = outcome;
  get("outcome").textContent = {
    ALLOW: "ALLOW", DENY: "DENY", REQUEST_VERIFICATION: "REQUEST VERIFICATION",
    ENROLLED: "ENROLLED", busy: "Verifying…", idle: "Ready to verify"
  }[outcome] || outcome;
  get("message").textContent = message;
  get("duration").textContent = elapsed == null ? "—" : `${Math.round(elapsed)} ms`;
  get("score").textContent = score == null ? "—" : score.toFixed(3);

  if (participantDisplay) {
    if (display) {
      participantDisplay.textContent = `Participant: ${display}`;
      participantDisplay.hidden = false;
    } else {
      participantDisplay.hidden = true;
    }
  }
}

function controls() {
  const hasStream = Boolean(stream && stream.active);
  get("scan").disabled = busy || !ready || !hasStream;
  get("enroll").disabled = busy || !ready || !hasStream || !get("consent").checked;
  get("delete").disabled = busy;
  get("start").disabled = busy || !ready || hasStream;
  get("stop").disabled = busy || !hasStream;
  if (cameraSelect) cameraSelect.disabled = busy;
  if (refreshBtn) refreshBtn.disabled = busy;
  get("retry").hidden = !retry || id.value !== retry.subject;
  get("retry").disabled = busy;
}

function timeoutSignal(ms) {
  if (typeof AbortSignal !== "undefined" && typeof AbortSignal.timeout === "function") {
    return AbortSignal.timeout(ms);
  }
  const controller = new AbortController();
  setTimeout(() => controller.abort(new DOMException("TimeoutError", "TimeoutError")), ms);
  return controller.signal;
}

function stopCurrentStream() {
  if (stream) {
    try {
      stream.getTracks().forEach(track => {
        track.stop();
      });
    } catch (e) {
      console.warn("Error stopping media tracks:", e);
    }
    stream = null;
  }
  if (video.srcObject) {
    video.srcObject = null;
  }
  activeTrackLabel = "";
}

function handleCameraDisconnect() {
  stopCurrentStream();
  get("camera-caption").textContent = "Camera disconnected";
  show("REQUEST_VERIFICATION", "Camera disconnected. Reconnect your phone/webcam and click Refresh.");
  controls();
}

function handleCameraError(err) {
  let msg = "Camera unavailable. Check browser permissions and connection.";
  if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
    msg = "Camera permission denied. Click the lock/camera icon in your address bar to allow access.";
  } else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
    msg = "No camera found. Connect your Android phone or USB webcam and click Refresh.";
  } else if (err.name === "NotReadableError" || err.name === "TrackStartError") {
    msg = "Camera is in use by another application. Close other camera apps and try again.";
  } else if (err.name === "OverconstrainedError") {
    msg = "Selected camera settings are not supported. Try selecting another camera.";
  } else if (!window.isSecureContext) {
    msg = "Camera requires a secure context. Access via http://localhost:8000 or http://127.0.0.1:8000.";
  }
  get("camera-caption").textContent = "Camera error";
  show("REQUEST_VERIFICATION", msg);
}

async function updateCameraList(promptIfEmpty = false) {
  if (!navigator.mediaDevices?.enumerateDevices) {
    get("camera-caption").textContent = "MediaDevices API unsupported";
    return;
  }
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter(d => d.kind === "videoinput");
    if (!cameraSelect) return;

    cameraSelect.innerHTML = "";
    if (videoDevices.length === 0) {
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = "No cameras detected";
      cameraSelect.appendChild(opt);
      if (promptIfEmpty) {
        show("REQUEST_VERIFICATION", "No cameras detected. Connect your Android phone or USB webcam, then click Refresh.");
      }
    } else {
      let matched = false;
      videoDevices.forEach((device, index) => {
        const opt = document.createElement("option");
        opt.value = device.deviceId;
        const fallbackName = `Camera ${index + 1}${device.deviceId ? " (" + device.deviceId.slice(0, 8) + "…)" : ""}`;
        opt.textContent = device.label || fallbackName;
        if (device.deviceId && device.deviceId === selectedDeviceId) {
          opt.selected = true;
          matched = true;
        }
        cameraSelect.appendChild(opt);
      });
      if (!matched && videoDevices.length > 0) {
        selectedDeviceId = videoDevices[0].deviceId;
        cameraSelect.value = selectedDeviceId;
      }
    }
  } catch (err) {
    console.warn("Failed to enumerate video devices:", err);
  }
}

async function startCamera(deviceIdToUse) {
  if (busy) return;
  stopCurrentStream();
  busy = true;
  controls();
  get("camera-caption").textContent = "Connecting camera…";

  const targetDeviceId = deviceIdToUse || selectedDeviceId || (cameraSelect ? cameraSelect.value : "");
  const constraints = {
    audio: false,
    video: {
      width: {ideal: 640},
      height: {ideal: 480},
      frameRate: {ideal: 30}
    }
  };
  if (targetDeviceId) {
    constraints.video.deviceId = {exact: targetDeviceId};
  }

  try {
    // Request exact camera without silent fallback to prevent wrong-camera traps
    stream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = stream;
    await video.play();

    const activeTrack = stream.getVideoTracks()[0];
    if (activeTrack) {
      const settings = activeTrack.getSettings ? activeTrack.getSettings() : {};
      if (settings.deviceId) {
        selectedDeviceId = settings.deviceId;
      }
      activeTrackLabel = activeTrack.label || "Camera";
      activeTrack.addEventListener("ended", handleCameraDisconnect);
    }

    await updateCameraList();
    if (selectedDeviceId && cameraSelect) {
      cameraSelect.value = selectedDeviceId;
    }

    get("camera-caption").textContent = `Active: ${activeTrackLabel}`;
    show("idle", "Camera ready. Enter Campus ID, then click Verify access.");
  } catch (err) {
    stopCurrentStream();
    handleCameraError(err);
  } finally {
    busy = false;
    controls();
  }
}

function stopCamera() {
  stopCurrentStream();
  get("camera-caption").textContent = "Camera off";
  show("idle", "Camera stopped. Select camera and click Start camera.");
  controls();
}

// Deterministic frame waiting with clean cancellation
function nextFrame(signal) {
  return new Promise((resolve, reject) => {
    if (signal.aborted) return reject(signal.reason);
    let handleId = null;
    let isVFC = false;
    let done = false;

    const onAbort = () => {
      cleanup();
      reject(signal.reason);
    };

    const cleanup = () => {
      if (done) return;
      done = true;
      signal.removeEventListener("abort", onAbort);
      if (handleId !== null) {
        if (isVFC && "cancelVideoFrameCallback" in video) {
          video.cancelVideoFrameCallback(handleId);
        } else {
          cancelAnimationFrame(handleId);
        }
        handleId = null;
      }
    };

    signal.addEventListener("abort", onAbort, {once: true});

    if ("requestVideoFrameCallback" in video) {
      isVFC = true;
      handleId = video.requestVideoFrameCallback((now, metadata) => {
        cleanup();
        resolve(metadata);
      });
    } else {
      isVFC = false;
      handleId = requestAnimationFrame(() => {
        cleanup();
        resolve();
      });
    }
  });
}

async function capture(signal) {
  if (!stream || !video.videoWidth || video.readyState < 2) throw Error("capture_failed");
  const width = video.videoWidth;
  const height = video.videoHeight;
  // Center 4:3 crop matching object-fit: cover on 4:3 preview
  const cropWidth = Math.min(width, height * 4 / 3);
  const cropHeight = Math.min(height, width * 3 / 4);
  const frames = [];

  for (let i = 0; i < 3; i++) {
    await nextFrame(signal);
    ctx.drawImage(video, (width - cropWidth) / 2, (height - cropHeight) / 2,
      cropWidth, cropHeight, 0, 0, 640, 480);
    frames.push(canvas.toDataURL("image/jpeg", 0.80).split(",")[1]);
    if (i < 2) await new Promise(resolve => setTimeout(resolve, 115));
  }
  return frames;
}

async function post(path, body, signal, isOperator = false) {
  const headers = {"Content-Type": "application/json"};
  if (isOperator) {
    const opToken = tokenInput ? tokenInput.value : "";
    const opId = operatorIdInput ? operatorIdInput.value.trim() : "";
    if (opToken) headers["X-Operator-Token"] = opToken;
    if (opId) headers["X-Operator-Id"] = opId;
  }
  const response = await fetch(path, {method: "POST", headers, body: JSON.stringify(body), signal});
  const result = await response.json();
  if (!response.ok) throw Error(result.detail);
  return result;
}

function renderDiagnostics(data) {
  if (!diagnosticsContent) return;
  const diag = data.diagnostics || {};
  const frames = diag.frames || [];
  let html = `<p><strong>Camera:</strong> ${escapeHtml(activeTrackLabel || "Default")} | <strong>Server latency:</strong> ${data.server_ms ?? "—"} ms</p>`;

  if (frames.length > 0) {
    html += `<table class="diag-table">
      <thead>
        <tr><th>Frame</th><th>Light (mean)</th><th>Focus (var)</th><th>PAD V2 (2.7×)</th><th>PAD V1SE (4.0×)</th><th>Identity</th></tr>
      </thead>
      <tbody>`;
    for (const f of frames) {
      html += `<tr>
        <td>#${f.frame}</td>
        <td>${f.light ?? "—"}</td>
        <td>${f.focus ?? "—"}</td>
        <td>${f.pad_v2 != null ? f.pad_v2.toFixed(4) : "—"}</td>
        <td>${f.pad_v1se != null ? f.pad_v1se.toFixed(4) : "—"}</td>
        <td>${f.identity_score != null ? f.identity_score.toFixed(4) : "—"}</td>
      </tr>`;
    }
    html += `</tbody></table>`;
  } else {
    html += `<p class="muted">Summary: PAD ${data.pad ?? "—"} (Score: ${data.live_score ?? "—"}), Identity: ${data.identity_score ?? "—"}</p>`;
  }

  diagnosticsContent.innerHTML = html;
}

function escapeHtml(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function escapeCsvField(val) {
  if (val == null) return "";
  let s = String(val);
  // Neutralize formula injection
  if (/^[=+\-@]/.test(s)) {
    s = "'" + s;
  }
  if (s.includes('"') || s.includes(',') || s.includes('\n') || s.includes('\r')) {
    return '"' + s.replace(/"/g, '""') + '"';
  }
  return s;
}

get("start").addEventListener("click", () => startCamera());
get("stop").addEventListener("click", () => stopCamera());

if (refreshBtn) {
  refreshBtn.addEventListener("click", async () => {
    await updateCameraList(true);
    if (cameraSelect && cameraSelect.options.length > 0 && cameraSelect.value) {
      show("idle", `Detected ${cameraSelect.options.length} video source(s). Click Start camera.`);
    }
  });
}

if (cameraSelect) {
  cameraSelect.addEventListener("change", async (e) => {
    const newId = e.target.value;
    if (!newId) return;
    selectedDeviceId = newId;
    if (stream && stream.active) {
      await startCamera(newId);
    }
  });
}

if (navigator.mediaDevices?.addEventListener) {
  navigator.mediaDevices.addEventListener("devicechange", async () => {
    await updateCameraList();
    if (stream && !stream.getVideoTracks().some(t => t.readyState === "live")) {
      handleCameraDisconnect();
    }
  });
}

async function verify(isRetry) {
  if (busy || !id.reportValidity()) return;
  const subject = id.value;
  const rawScenario = get("scenario").value;
  const scenario = rawScenario ? rawScenario : "unlabeled";
  const retryCode = isRetry ? retry?.code : null;
  if (isRetry && !retryCode) return;
  retry = null;
  busy = true;
  controls();
  const started = performance.now();
  const signal = timeoutSignal(deadlineMs);
  let data;
  show("busy", "Keep your face centered in the guide.");

  try {
    const frames = await capture(signal);
    data = await post("/api/scan", {subject, frames, retry: retryCode}, signal);
  } catch (error) {
    data = {
      outcome: "REQUEST_VERIFICATION",
      pad: "UNCERTAIN",
      reason: signal.aborted ? "decision_timeout" : error.message,
      subject,
    };
  }

  const elapsed = performance.now() - started;
  if (elapsed >= deadlineMs) {
    data = {outcome: "REQUEST_VERIFICATION", pad: "UNCERTAIN", reason: "decision_timeout", subject};
  }

  if (data.retry) retry = {code: data.retry, subject};
  show(data.outcome, messages[data.reason] || messages.capture_failed, elapsed, data.live_score, data.display_name);
  renderDiagnostics(data);

  // Record attempt in session log
  rows.push([
    scenario,
    isRetry ? "retake" : "initial",
    data.outcome,
    data.pad,
    Math.round(elapsed * 10) / 10,
    data.server_ms ?? "",
    data.reason ?? "",
    data.event_id ?? "",
  ]);

  try {
    sessionStorage.setItem("liveguard_session_rows", JSON.stringify(rows));
  } catch (e) {}

  updateExportCounter();
  busy = false;
  controls();
}

get("scan-form").addEventListener("submit", event => {
  event.preventDefault();
  verify(false);
});
get("retry").addEventListener("click", () => verify(true));
id.addEventListener("input", controls);
get("consent").addEventListener("change", controls);

async function enrollment(remove) {
  if (busy || !id.reportValidity()) return;
  busy = true;
  const subject = id.value;
  const displayName = displayNameInput ? displayNameInput.value.trim() : null;
  retry = null;
  controls();
  let clearToken = true;

  try {
    const signal = timeoutSignal(6000);
    if (remove) {
      const result = await post("/api/delete-enrollment", {subject}, signal, true);
      show("idle", result.deleted ? "Enrollment removed." : "No enrollment for that campus ID.");
      if (participantDisplay) participantDisplay.hidden = true;
    } else {
      if (!get("consent").checked) throw Error("consent_required");
      const frames = await capture(signal);
      const payload = {subject, frames, consent: true};
      if (displayName) payload.display_name = displayName;
      const res = await post("/api/enroll", payload, signal, true);
      get("consent").checked = false;
      show("ENROLLED", "Enrollment saved locally. Use Verify access to test it.", null, null, res.display_name);
    }
  } catch (error) {
    show("REQUEST_VERIFICATION", messages[error.message] || messages.capture_failed);
    // Retain operator token if failure was a face/lighting/consent glitch
    if (error.message !== "operator_required") {
      clearToken = false;
    }
  } finally {
    if (clearToken && tokenInput) {
      tokenInput.value = "";
    }
    busy = false;
    controls();
  }
}

get("enroll").addEventListener("click", () => enrollment(false));
get("delete").addEventListener("click", () => enrollment(true));

get("export").addEventListener("click", () => {
  if (rows.length === 0) {
    show("REQUEST_VERIFICATION", "No scan attempts recorded in this session to export.");
    return;
  }
  const headers = ["scenario", "attempt", "outcome", "pad", "browser_ms", "server_ms", "reason", "event_id"];
  const csvLines = [headers.join(",")];

  for (const row of rows) {
    csvLines.push(row.map(escapeCsvField).join(","));
  }

  // Prepend UTF-8 BOM so spreadsheet applications render Unicode & Kannada names properly
  const csvContent = "\uFEFF" + csvLines.join("\r\n") + "\r\n";
  const blob = new Blob([csvContent], {type: "text/csv;charset=utf-8"});
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.download = `liveguard-session-${new Date().toISOString().slice(0, 10)}.csv`;
  link.href = url;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  setTimeout(() => URL.revokeObjectURL(url), 2000);
});

function applyGuideGeometry(guide) {
  const guideEl = document.querySelector(".guide");
  if (!guideEl || !guide) return;
  if (typeof guide.left === "number") guideEl.style.left = `${guide.left * 100}%`;
  if (typeof guide.top === "number") guideEl.style.top = `${guide.top * 100}%`;
  if (typeof guide.width === "number") guideEl.style.width = `${guide.width * 100}%`;
  if (typeof guide.height === "number") guideEl.style.height = `${guide.height * 100}%`;
}

fetch("/api/status").then(response => response.json()).then(data => {
  ready = data.models_ready;
  if (!ready) show("REQUEST_VERIFICATION", messages.models_unavailable);
  if (data.guide) applyGuideGeometry(data.guide);
  controls();
}).catch(() => show("REQUEST_VERIFICATION", messages.pipeline_unavailable));

window.addEventListener("pagehide", () => stopCurrentStream());

// Initial setup
updateCameraList();
updateExportCounter();
controls();
