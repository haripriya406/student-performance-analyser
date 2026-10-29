document.addEventListener("DOMContentLoaded", async () => {
  const form = document.getElementById("prediction-form");
  const status = document.getElementById("prediction-status");
  const result = document.getElementById("prediction-result");
  const submit = document.getElementById("predict-submit");
  const showMessage = (message, kind) => {
    status.className = `notice notice-${kind}`;
    status.textContent = message;
    status.classList.remove("is-hidden");
  };
  try {
    const response = await fetch("/api/dataset-info");
    const info = await response.json();
    if (!response.ok) throw new Error(info.error || "Dataset categories could not be loaded.");
    for (const select of form.querySelectorAll("[data-category]")) {
      const field = select.dataset.category;
      select.replaceChildren(new Option("Select a category", ""));
      for (const value of info.categories[field] || []) select.add(new Option(value, value));
    }
  } catch (error) {
    showMessage(error.message, "error");
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    form.classList.add("was-validated");
    if (!form.reportValidity()) return;
    const payload = Object.fromEntries(new FormData(form).entries());
    for (const field of ["age", "study_hours_per_day", "attendance_percentage", "previous_score", "assignments_completed_out_of_10", "sleep_hours_per_day", "extracurricular_hours_per_week"]) {
      payload[field] = Number(payload[field]);
    }
    submit.disabled = true;
    submit.textContent = "Getting estimate…";
    result.classList.add("is-hidden");
    showMessage("The model is processing this estimate…", "info");
    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Prediction could not be completed.");
      const riskClass = data.prediction === "At Risk" ? "at-risk" : "not-at-risk";
      let probabilityMarkup = "";
      if (data.class_probabilities) {
        probabilityMarkup = `<h3>Model output scores</h3><div class="probability-list">${Object.entries(data.class_probabilities).map(([label, probability]) => `<div class="probability-row"><span>${label}</span><span class="probability-track"><span class="probability-fill" style="width:${Math.max(0, Math.min(100, probability * 100))}%"></span></span><strong>${(probability * 100).toFixed(1)}%</strong></div>`).join("")}</div><p class="data-note">${data.probability_notice}</p>`;
      }
      result.innerHTML = `<h2>Prediction result</h2><p class="result-category ${riskClass}">${data.prediction}</p><p class="data-note">${data.estimate_notice}</p>${probabilityMarkup}`;
      result.classList.remove("is-hidden");
      showMessage("Estimate completed.", "success");
    } catch (error) {
      showMessage(error.message, "error");
    } finally {
      submit.disabled = false;
      submit.innerHTML = 'Predict academic risk <span aria-hidden="true">→</span>';
    }
  });
});
