const API_BASE_URL = window.location.protocol.startsWith("http") && window.location.port === "8000"
  ? window.location.origin
  : "http://localhost:8000";

const SUBMISSION_KEY = "bioplexmdt_submission";
const $ = (id) => document.getElementById(id);
let latestGigatimeUrls = [];

function normalizeUrl(url) {
  if (!url) return "";
  if (/^https?:\/\//i.test(url)) return url;
  if (url.startsWith("/")) return `${API_BASE_URL}${url}`;
  return `${API_BASE_URL}/${url}`;
}

function imageMarkerLabel(url) {
  const filename = decodeURIComponent(String(url || "").split("/").pop() || "");
  if (!filename) return "Simulated immunomap";
  if (/overlay/i.test(filename)) return "All markers overlay";
  if (/Map_00_DAPI/i.test(filename)) return "DAPI";

  const markerMatch = filename.match(/^Map_\d+_(.+?)_DAPI_merge\./i);
  if (markerMatch) return `${markerMatch[1].replace(/_/g, " ")} + DAPI`;

  const knownMarkers = ["PD-L1", "CD68", "CD8", "CD4", "CK", "Ki67", "DAPI"];
  const found = knownMarkers.filter((marker) => filename.toLowerCase().includes(marker.toLowerCase()));
  return found.length ? found.join(" + ") : filename.replace(/\.[^.]+$/, "").replace(/_/g, " ");
}

function sortImmunomapUrls(urls) {
  const order = [
    "Map_00_DAPI",
    "CD68",
    "CD8",
    "PD-L1",
    "CK",
    "Ki67",
    "CD4",
    "Overlay"
  ];
  return [...urls].sort((a, b) => {
    const an = decodeURIComponent(String(a).split("/").pop() || "");
    const bn = decodeURIComponent(String(b).split("/").pop() || "");
    const ai = order.findIndex((key) => an.toLowerCase().includes(key.toLowerCase()));
    const bi = order.findIndex((key) => bn.toLowerCase().includes(key.toLowerCase()));
    return (ai === -1 ? 99 : ai) - (bi === -1 ? 99 : bi) || an.localeCompare(bn);
  });
}

function drawNeuroMap() {
  const canvas = $("neuro-map");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  let t = 0;

  function frame() {
    t += 0.014;
    const grd = ctx.createLinearGradient(0, 0, w, h);
    grd.addColorStop(0, "#101827");
    grd.addColorStop(0.55, "#183245");
    grd.addColorStop(1, "#0e3d42");
    ctx.fillStyle = grd;
    ctx.fillRect(0, 0, w, h);

    ctx.save();
    ctx.translate(w / 2, h / 2);
    for (let ring = 0; ring < 9; ring += 1) {
      ctx.beginPath();
      const rx = 36 + ring * 14;
      const ry = 24 + ring * 10;
      for (let i = 0; i <= 160; i += 1) {
        const a = (Math.PI * 2 * i) / 160;
        const wave = Math.sin(a * 4 + t * 3 + ring) * 4;
        const x = Math.cos(a) * (rx + wave);
        const y = Math.sin(a) * (ry + wave * 0.5);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.strokeStyle = `rgba(${80 + ring * 14}, ${166 + ring * 4}, 255, ${0.18 + ring * 0.035})`;
      ctx.lineWidth = ring === 4 ? 2.2 : 1.2;
      ctx.stroke();
    }

    for (let i = 0; i < 42; i += 1) {
      const a = i * 0.72 + t;
      const r = 40 + (i % 9) * 12;
      const x = Math.cos(a) * r;
      const y = Math.sin(a * 1.2) * r * 0.62;
      ctx.beginPath();
      ctx.arc(x, y, i % 7 === 0 ? 3 : 1.8, 0, Math.PI * 2);
      ctx.fillStyle = i % 7 === 0 ? "rgba(255,197,96,0.88)" : "rgba(142,219,217,0.66)";
      ctx.fill();
    }

    ctx.beginPath();
    ctx.arc(46 + Math.sin(t * 2) * 8, -18 + Math.cos(t * 1.5) * 5, 24, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(212,90,72,0.74)";
    ctx.fill();
    ctx.strokeStyle = "rgba(255,255,255,0.78)";
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.restore();

    requestAnimationFrame(frame);
  }

  frame();
}

function renderGigatimeImages(urls = latestGigatimeUrls) {
  const gallery = $("gigatime-gallery");
  const empty = $("gigatime-empty");
  const status = $("immuno-status");
  if (!gallery || !empty) return;

  const cleanUrls = sortImmunomapUrls(Array.from(new Set((urls || []).filter(Boolean)))).slice(0, 8);
  latestGigatimeUrls = cleanUrls;
  gallery.innerHTML = "";

  if (!cleanUrls.length) {
    gallery.classList.remove("has-images");
    empty.style.display = "grid";
    if (status) status.textContent = "No simulated immunomap available.";
    return;
  }

  cleanUrls.forEach((url) => {
    const href = normalizeUrl(url);
    const link = document.createElement("a");
    link.href = href;
    link.target = "_blank";
    link.rel = "noopener";
    link.className = "gigatime-image-card";

    const img = document.createElement("img");
    img.src = href;
    img.alt = imageMarkerLabel(url);
    img.loading = "lazy";

    const caption = document.createElement("span");
    caption.textContent = imageMarkerLabel(url);

    link.appendChild(img);
    link.appendChild(caption);
    gallery.appendChild(link);
  });

  empty.style.display = "none";
  gallery.classList.add("has-images");
  if (status) status.textContent = `${cleanUrls.length} high-resolution simulated immunomap image(s) displayed.`;
}

function splitAnswerSections(answer) {
  const text = String(answer || "").trim();
  const result = { mdt: "", patient: "", professional: "" };
  if (!text) return result;

  const headers = [
    { key: "mdt", re: /(MDT\s*recommendations?|multidisciplinary\s*recommendations?|treatment\s*recommendations?)/i },
    { key: "patient", re: /(patient-friendly\s*explanation|patient\s*explanation|for\s*the\s*patient|family\s*explanation)/i },
    { key: "professional", re: /(professional\s*answer|clinical\s*answer|question\s*answer|specialist\s*answer)/i }
  ];
  const lines = text.split(/\n/);
  const hits = [];

  lines.forEach((line, index) => {
    const normalized = line.replace(/^[#*\-\s\d.、()[\]【】]+/, "").trim();
    headers.forEach((item) => {
      if (item.re.test(normalized)) hits.push({ key: item.key, index });
    });
  });

  if (hits.length) {
    const ordered = hits.sort((a, b) => a.index - b.index);
    ordered.forEach((hit, index) => {
      const next = ordered[index + 1]?.index ?? lines.length;
      if (!result[hit.key]) result[hit.key] = lines.slice(hit.index, next).join("\n").trim();
    });
  }

  if (!result.professional) result.professional = text;
  if (!result.mdt) result.mdt = result.professional;
  return result;
}

function setResultFileLink(targetId, url, label) {
  const target = $(targetId);
  if (!target) return;
  let link = document.querySelector(`[data-result-link="${targetId}"]`);
  if (!url) {
    if (link) link.remove();
    return;
  }
  if (!link) {
    link = document.createElement("a");
    link.dataset.resultLink = targetId;
    link.className = "result-file-link";
    link.target = "_blank";
    link.rel = "noopener";
    target.insertAdjacentElement("afterend", link);
  }
  link.href = normalizeUrl(url);
  link.textContent = label;
}

function renderAnswerModules(payload) {
  const answer = typeof payload === "string" ? payload : (payload?.answer || "");
  const sections = typeof payload === "object" && payload
    ? {
        mdt: payload.mdt_output || "",
        patient: payload.patient_friendly_output || "",
        professional: payload.professional_answer_output || ""
      }
    : splitAnswerSections(answer);

  if (!sections.mdt && !sections.patient && !sections.professional) {
    Object.assign(sections, splitAnswerSections(answer));
  }

  if ($("mdt-output")) $("mdt-output").textContent = sections.mdt || "No MDT recommendations yet.";
  if ($("patient-output")) $("patient-output").textContent = sections.patient || "No patient-friendly explanation yet.";
  if ($("professional-output")) $("professional-output").textContent = sections.professional || "No professional answer yet.";
  if ($("answer-output")) $("answer-output").textContent = answer || "No results yet.";

  setResultFileLink("mdt-output", payload?.mdt_file_url, "View MDT recommendation file");
  setResultFileLink("patient-output", payload?.patient_friendly_file_url, "View patient-friendly explanation file");
  setResultFileLink("professional-output", payload?.professional_answer_file_url, "View professional answer file");
}

function setBusy(isBusy) {
  ["ask-btn", "parse-btn"].forEach((id) => {
    const btn = $(id);
    if (btn) btn.disabled = isBusy;
  });
}

async function checkHealth() {
  const healthStatus = $("health-status");
  if (!healthStatus) return;
  healthStatus.textContent = "Checking...";
  try {
    const resp = await fetch(`${API_BASE_URL}/api/health`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    healthStatus.innerHTML = data.status === "ok"
      ? '<span class="ok">API connected</span>'
      : '<span class="warn">API responded</span>';
  } catch (_) {
    healthStatus.innerHTML = '<span class="bad">Not connected</span>';
  }
}

async function parseQuestion() {
  const questionInput = $("question-input");
  const modelSelect = $("model-select");
  const qaStatus = $("qa-status");
  const parsedOutput = $("parsed-output");
  if (!questionInput || !modelSelect || !qaStatus || !parsedOutput) return;

  const question = questionInput.value.trim();
  if (!question) {
    qaStatus.innerHTML = '<span class="warn">Please enter case information first.</span>';
    return;
  }

  setBusy(true);
  qaStatus.textContent = "Parsing question...";
  try {
    const resp = await fetch(`${API_BASE_URL}/api/parse-question`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, model: modelSelect.value })
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    parsedOutput.textContent = JSON.stringify(data.struct || data, null, 2);
    qaStatus.innerHTML = '<span class="ok">Question parsed.</span>';
  } catch (error) {
    qaStatus.innerHTML = `<span class="bad">Parse failed: </span>${error.message}`;
  } finally {
    setBusy(false);
  }
}

function composeQuestion(submission) {
  const extras = [];
  if (submission.case_id) extras.push(`Case ID: ${submission.case_id}`);
  if (submission.tiff_path) extras.push(`TIFF path: ${submission.tiff_path}`);
  if (submission.dicom_dir) extras.push(`DICOM path: ${submission.dicom_dir}`);
  if (submission.scrna_h5ad_path) extras.push(`h5ad path: ${submission.scrna_h5ad_path}`);
  if (submission.tenx_dir) extras.push(`10x path: ${submission.tenx_dir}`);
  return extras.length ? `${submission.question}\n\n[Additional Data]\n${extras.join("\n")}` : submission.question;
}

function collectSubmission() {
  return {
    question: $("question-input")?.value.trim() || "",
    model: $("model-select")?.value || "",
    case_id: $("case-id")?.value.trim() || "",
    tiff_path: $("tiff-path")?.value.trim() || "",
    dicom_dir: $("dicom-dir")?.value.trim() || "",
    scrna_h5ad_path: $("h5ad-path")?.value.trim() || "",
    tenx_dir: $("tenx-dir")?.value.trim() || "",
    created_at: new Date().toISOString()
  };
}

function saveSubmissionAndGo() {
  const qaStatus = $("qa-status");
  const submission = collectSubmission();
  if (!submission.question) {
    if (qaStatus) qaStatus.innerHTML = '<span class="warn">Please enter case information first.</span>';
    return;
  }

  const hasFileSelection = ($("mri-files")?.files.length || 0)
    || ($("tiff-files")?.files.length || 0)
    || ($("dicom-files")?.files.length || 0)
    || ($("h5ad-file")?.files.length || 0);
  if (hasFileSelection && qaStatus) {
    qaStatus.innerHTML = '<span class="warn">Opening the results page. Cross-page submission uses local paths; please confirm TIFF, DICOM, h5ad, or 10x paths are filled.</span>';
  }

  sessionStorage.setItem(SUBMISSION_KEY, JSON.stringify(submission));
  window.location.href = "result.html";
}

function handlePossibleImagePayload(payload) {
  const urls = payload?.gigatime_output_urls || payload?.gigatime_preview_urls || payload?.immunomap_urls;
  if (!urls || !urls.length) return;
  renderGigatimeImages(urls);
}

function exportResultFile() {
  const mdt = $("mdt-output")?.textContent.trim() || "";
  const patient = $("patient-output")?.textContent.trim() || "";
  const professional = $("professional-output")?.textContent.trim() || "";
  const createdAt = new Date().toLocaleString("en-US", { hour12: false });
  const content = [
    "BioPlexMDT Analysis Results",
    `Export time: ${createdAt}`,
    "",
    "## MDT Recommendations",
    mdt,
    "",
    "## Patient-Friendly Explanation",
    patient,
    "",
    "## Professional Answer",
    professional,
    ""
  ].join("\n");
  const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
  link.href = url;
  link.download = `BioPlexMDT_result_${stamp}.txt`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

async function runAnalysisFromSubmission(submission) {
  const qaStatus = $("qa-status");
  const parsedOutput = $("parsed-output");
  const traceOutput = $("trace-output");
  if (!qaStatus || !parsedOutput || !traceOutput) return;

  if (!submission?.question) {
    qaStatus.innerHTML = '<span class="warn">No submitted data found. Please return to the submission page and submit again.</span>';
    return;
  }

  setBusy(true);
  parsedOutput.textContent = "{}";
  traceOutput.textContent = "Submitting...";
  renderAnswerModules("Waiting for backend response...");
  renderGigatimeImages([]);
  qaStatus.textContent = "Calling agent...";

  try {
    const resp = await fetch(`${API_BASE_URL}/api/ask-stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: composeQuestion(submission),
        model: submission.model,
        tiff_path: submission.tiff_path,
        dicom_dir: submission.dicom_dir,
        scrna_h5ad_path: submission.scrna_h5ad_path,
        tenx_dir: submission.tenx_dir
      })
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    if (!resp.body) throw new Error("This browser does not support stream reading");

    const reader = resp.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";
    traceOutput.textContent = "";

    function handleEvent(name, dataText) {
      let payload = dataText;
      try { payload = JSON.parse(dataText); } catch (_) {}

      if (name === "log") {
        traceOutput.textContent += `${payload}\n`;
        traceOutput.scrollTop = traceOutput.scrollHeight;
      }
      if (name === "meta") {
        traceOutput.textContent += `[meta] ${JSON.stringify(payload, null, 2)}\n`;
        handlePossibleImagePayload(payload);
      }
      if (name === "final") {
        renderAnswerModules(payload);
        if (payload.trace) traceOutput.textContent = payload.trace;
        handlePossibleImagePayload(payload);
        qaStatus.innerHTML = '<span class="ok">Agent analysis completed.</span>';
        if ($("health-status")) $("health-status").innerHTML = '<span class="ok">Completed</span>';
      }
      if (name === "error") {
        qaStatus.innerHTML = `<span class="bad">Backend error: </span>${payload.detail || payload}`;
        if ($("health-status")) $("health-status").innerHTML = '<span class="bad">Run failed</span>';
      }
    }

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let index;
      while ((index = buffer.indexOf("\n\n")) !== -1) {
        const frame = buffer.slice(0, index);
        buffer = buffer.slice(index + 2);
        let eventName = "message";
        const dataLines = [];
        frame.split("\n").forEach((line) => {
          if (line.startsWith("event:")) eventName = line.slice(6).trim();
          if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
        });
        if (dataLines.length) handleEvent(eventName, dataLines.join("\n"));
      }
    }
  } catch (error) {
    qaStatus.innerHTML = `<span class="bad">Call failed: </span>${error.message}`;
    traceOutput.textContent = `Please confirm the backend service is running at ${API_BASE_URL}`;
    if ($("health-status")) $("health-status").innerHTML = '<span class="bad">Run failed</span>';
  } finally {
    setBusy(false);
  }
}

async function buildIndex(kind) {
  const status = $("index-status");
  if (!status) return;
  const endpoint = kind === "facts" ? "/api/build-facts" : "/api/build-index";
  status.textContent = "Submitting index task...";
  try {
    const resp = await fetch(`${API_BASE_URL}${endpoint}`, { method: "POST" });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    status.innerHTML = `<span class="ok">Index task completed.</span> ${data.detail || ""}`;
  } catch (error) {
    status.innerHTML = `<span class="bad">Index task failed: </span>${error.message}`;
  }
}

function clearSubmissionPage() {
  ["question-input", "case-id", "tiff-path", "dicom-dir", "h5ad-path", "tenx-dir"].forEach((id) => {
    if ($(id)) $(id).value = "";
  });
  if ($("parsed-output")) $("parsed-output").textContent = "{}";
  if ($("qa-status")) $("qa-status").textContent = "Cleared.";
}

function bindSubmitPage() {
  $("parse-btn")?.addEventListener("click", parseQuestion);
  $("ask-btn")?.addEventListener("click", saveSubmissionAndGo);
  $("clear-btn")?.addEventListener("click", clearSubmissionPage);
  $("build-index-btn")?.addEventListener("click", () => buildIndex("text"));
  $("build-facts-btn")?.addEventListener("click", () => buildIndex("facts"));
}

function bindResultPage() {
  $("refresh-gigatime-btn")?.addEventListener("click", () => renderGigatimeImages());
  $("export-result-btn")?.addEventListener("click", exportResultFile);
  renderGigatimeImages([]);
  const raw = sessionStorage.getItem(SUBMISSION_KEY);
  let submission = null;
  try { submission = raw ? JSON.parse(raw) : null; } catch (_) {}
  runAnalysisFromSubmission(submission);
}

function init() {
  $("health-btn")?.addEventListener("click", checkHealth);
  drawNeuroMap();
  const page = document.body.dataset.page;
  if (page === "submit") bindSubmitPage();
  if (page === "result") bindResultPage();
}

init();
