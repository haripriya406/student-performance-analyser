document.addEventListener("DOMContentLoaded", () => {
  const errorBox = document.getElementById("dataset-error");
  const uploadStatus = document.getElementById("upload-status");
  const escapeText = (value) => String(value).replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  })[character]);
  const loadInfo = async () => {
    errorBox.classList.add("is-hidden");
    try {
      const response = await fetch("/api/dataset-info");
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Dataset information could not be loaded.");
      document.getElementById("dataset-content").classList.remove("is-hidden");
      document.getElementById("dataset-rows").textContent = data.rows.toLocaleString();
      document.getElementById("dataset-columns").textContent = data.columns;
      document.getElementById("dataset-duplicates").textContent = data.duplicate_records.toLocaleString();
      document.getElementById("dataset-filename").textContent = data.filename;
      document.getElementById("dataset-distribution").innerHTML = Object.entries(data.class_distribution).map(([label, value]) => `<div class="distribution-row"><span>${label}</span><strong>${value.toLocaleString()} records</strong></div>`).join("");
      document.getElementById("dataset-columns-body").innerHTML = data.column_names.map((column) => `<tr><td><code>${escapeText(column)}</code></td><td>${escapeText(data.column_descriptions[column] || "")}</td><td>${data.missing_values[column]}</td></tr>`).join("");
      const uploadForm = document.getElementById("upload-form");
      uploadForm.title = data.upload_behavior;
    } catch (error) {
      errorBox.textContent = error.message;
      errorBox.classList.remove("is-hidden");
    }
  };
  document.getElementById("upload-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    if (!form.reportValidity()) return;
    const button = document.getElementById("upload-submit");
    button.disabled = true;
    uploadStatus.className = "notice notice-info";
    uploadStatus.textContent = "Validating the uploaded CSV…";
    try {
      const response = await fetch("/api/upload-dataset", { method: "POST", body: new FormData(form) });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Upload could not be completed.");
      uploadStatus.className = "notice notice-success";
      uploadStatus.textContent = `${data.message} ${data.rows.toLocaleString()} records validated.`;
      form.reset();
      await loadInfo();
    } catch (error) {
      uploadStatus.className = "notice notice-error";
      uploadStatus.textContent = error.message;
    } finally {
      button.disabled = false;
    }
  });
  loadInfo();
});
