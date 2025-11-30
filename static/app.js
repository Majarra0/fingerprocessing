const state = {
  jobId: null,
  pollHandle: null,
};

const $ = (sel) => document.querySelector(sel);

const apiUrl = (path) => new URL(path, window.location.origin).toString();

function setMessage(text, tone = "info") {
  const el = $("#uploadMessage");
  if (!el) return;
  el.textContent = text;
  el.style.color = tone === "error" ? "#ff6b6b" : tone === "success" ? "#52e3a5" : "#9bb0d4";
}

function toggleLoading(isLoading) {
  const btn = $("#submitBtn");
  if (!btn) return;
  btn.disabled = isLoading;
  btn.textContent = isLoading ? "Sending…" : "Start processing";
}

function badgeClass(status) {
  if (!status) return "badge";
  if (status === "completed" || status === "success") return "badge ok";
  if (status === "failed" || status === "error") return "badge danger";
  if (status === "processing") return "badge warn";
  return "badge";
}

async function uploadHandler(event) {
  event.preventDefault();
  const filesInput = $("#fileInput");
  const outputFolder = $("#outputFolder").value || "processed_images";
  const filenamePrefix = $("#filenamePrefix").value || "enhanced";

  if (!filesInput.files.length) {
    setMessage("Please choose at least one file.", "error");
    return;
  }

  const formData = new FormData();
  Array.from(filesInput.files).forEach((file) => formData.append("files", file));
  formData.append("output_folder", outputFolder);
  formData.append("filename_prefix", filenamePrefix);

  toggleLoading(true);
  setMessage("Uploading files and queuing job…");

  try {
    const res = await fetch(apiUrl("/upload/"), { method: "POST", body: formData });
    if (!res.ok) {
      const error = await res.json().catch(() => ({}));
      throw new Error(error.detail || "Upload failed");
    }
    const data = await res.json();
    state.jobId = data.job_id;
    setMessage(`Job ${data.job_id} queued. Tracking status…`, "success");
    startPolling(data.job_id);
  } catch (err) {
    console.error(err);
    setMessage(err.message, "error");
  } finally {
    toggleLoading(false);
  }
}

async function fetchStatus(jobId) {
  if (!jobId) return null;
  const res = await fetch(apiUrl(`/status/${jobId}`));
  if (!res.ok) throw new Error("Unable to fetch status");
  return res.json();
}

async function fetchDownloads(jobId) {
  if (!jobId) return null;
  const res = await fetch(apiUrl(`/download/${jobId}`));
  if (!res.ok) throw new Error("No downloadable files yet");
  return res.json();
}

function renderStatus(data) {
  const card = $("#statusCard");
  if (!data || !card) return;

  const statusLabel = data.status || "unknown";
  const badge = `<span class="${badgeClass(statusLabel)}">${statusLabel}</span>`;
  const created = data.created_at ? new Date(data.created_at).toLocaleString() : "–";
  const stateInfo = data.task_states?.[0];
  const workerState = stateInfo?.state || "–";
  const workerStep = stateInfo?.info?.step || stateInfo?.info?.detail || "";
  const workerDisplay = workerStep ? `${workerState} • ${workerStep}` : workerState;

  card.innerHTML = `
    <div class="status-row">
      <strong>Job</strong>
      <code>${data.job_id || "n/a"}</code>
      ${badge}
    </div>
    <div class="status-grid">
      <div class="stat"><span>Created</span>${created}</div>
      <div class="stat"><span>File count</span>${data.file_count ?? "–"}</div>
      <div class="stat"><span>Completed</span>${data.completed_tasks ?? 0}</div>
      <div class="stat"><span>Pending</span>${data.pending_tasks ?? 0}</div>
      <div class="stat"><span>Failed</span>${data.failed_tasks ?? 0}</div>
      <div class="stat"><span>Output folder</span>${data.output_folder || "–"}</div>
      <div class="stat"><span>Worker</span>${workerDisplay}</div>
    </div>
    <div class="status-row">
      <div class="${data.errors?.length ? "badge danger" : "badge ok"}">
        Results: ${data.results?.length || 0} | Errors: ${data.errors?.length || 0}
      </div>
    </div>
  `;
}

function renderResults(downloadData, statusData) {
  const container = $("#resultsCard");
  if (!container) return;
  if (!downloadData) {
    container.innerHTML = `<p class="hint">Results will appear once processing finishes.</p>`;
    return;
  }

  const items = downloadData.available_files || [];
  if (!items.length) {
    container.innerHTML = `<p class="hint">No files found in ${downloadData.output_folder}. If the job just finished, try refreshing.</p>`;
    return;
  }

  const outputFolder = downloadData.output_folder;
  const rows = items
    .map((file, i) => {
      const url = new URL(downloadData.download_urls?.[i] || `/file/${outputFolder}/${file}`, window.location.origin).toString();
      const isImage = /\.(png|jpg|jpeg|bmp|tif|tiff)$/i.test(file);
      return `
        <div class="result-item">
          <div>
            <div><strong>${file}</strong></div>
            <div class="hint">${outputFolder}</div>
          </div>
          <div class="result-item__links">
            <a class="button ghost" href="${url}" download>Download</a>
            ${isImage ? `<a class="button subtle" href="${url}" target="_blank" rel="noopener">Preview</a>` : ""}
          </div>
        </div>
      `;
    })
    .join("");

  container.innerHTML = `
    <div class="status-row">
      <strong>Files in ${outputFolder}</strong>
      <span class="${badgeClass(statusData?.status)}">${statusData?.status || "unknown"}</span>
    </div>
    ${rows}
  `;
}

function startPolling(jobId) {
  if (!jobId) return;
  clearInterval(state.pollHandle);
  state.jobId = jobId;
  getStatusAndRender(jobId);
  state.pollHandle = setInterval(() => getStatusAndRender(jobId, true), 2500);
}

async function getStatusAndRender(jobId, silent = false) {
  try {
    const statusData = await fetchStatus(jobId);
    renderStatus(statusData);
    if (!silent) {
      setMessage(`Job ${jobId}: ${statusData.status}`, statusData.status === "failed" ? "error" : "info");
    }
    if (statusData.status === "completed" || statusData.status === "failed") {
      clearInterval(state.pollHandle);
      state.pollHandle = null;
      try {
        const downloadData = await fetchDownloads(jobId);
        renderResults(downloadData, statusData);
      } catch (err) {
        renderResults(null, statusData);
      }
    }
  } catch (err) {
    if (!silent) setMessage(err.message || "Failed to get status", "error");
  }
}

async function handleLookup() {
  const input = $("#lookupJobId");
  const jobId = input.value.trim();
  if (!jobId) return;
  setMessage(`Looking up job ${jobId}…`);
  startPolling(jobId);
}

function stopPolling() {
  if (state.pollHandle) {
    clearInterval(state.pollHandle);
    state.pollHandle = null;
    setMessage("Polling stopped. Use lookup to resume.", "info");
  }
}

async function refreshOutputs() {
  const container = $("#outputsList");
  if (!container) return;
  container.innerHTML = `<p class="hint">Loading folders…</p>`;
  try {
    const res = await fetch(apiUrl("/list-outputs/"));
    if (!res.ok) throw new Error("Failed to list outputs");
    const data = await res.json();
    const folders = data.output_folders || [];
    if (!folders.length) {
      container.innerHTML = `<p class="hint">No output folders detected yet.</p>`;
      return;
    }
    container.innerHTML = folders
      .map(
        (folder) => `
        <div class="output-card">
          <h4>${folder.folder_name}</h4>
          <div class="hint">${folder.file_count} file(s)</div>
          <div class="hint">${folder.files.slice(0, 3).join(", ") || "Empty folder"}</div>
        </div>
      `
      )
      .join("");
  } catch (err) {
    container.innerHTML = `<p class="hint" style="color:#ff6b6b">${err.message}</p>`;
  }
}

function wireDropzone() {
  const dropzone = $("#dropzone");
  const input = $("#fileInput");
  if (!dropzone || !input) return;

  const preventDefaults = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  ["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
    dropzone.addEventListener(eventName, preventDefaults, false);
  });

  ["dragenter", "dragover"].forEach((eventName) => {
    dropzone.addEventListener(eventName, () => dropzone.classList.add("active"), false);
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropzone.addEventListener(eventName, () => dropzone.classList.remove("active"), false);
  });

  dropzone.addEventListener("drop", (e) => {
    const dt = e.dataTransfer;
    if (!dt?.files?.length) return;
    const fileList = new DataTransfer();
    Array.from(dt.files).forEach((file) => fileList.items.add(file));
    input.files = fileList.files;
    setMessage(`${input.files.length} file(s) ready to upload.`);
  });
}

function init() {
  $("#uploadForm")?.addEventListener("submit", uploadHandler);
  $("#lookupBtn")?.addEventListener("click", handleLookup);
  $("#stopPolling")?.addEventListener("click", stopPolling);
  $("#refreshOutputs")?.addEventListener("click", refreshOutputs);
  $("#refreshOutputsSecondary")?.addEventListener("click", refreshOutputs);
  wireDropzone();
  refreshOutputs();
}

document.addEventListener("DOMContentLoaded", init);
