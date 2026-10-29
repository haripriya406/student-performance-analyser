document.addEventListener("DOMContentLoaded", async () => {
  const errorBox = document.getElementById("evaluation-error");
  try {
    const response = await fetch("/api/model-metrics");
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Model metrics could not be loaded.");
    document.getElementById("evaluation-content").classList.remove("is-hidden");
    document.getElementById("training-count").textContent = data.training_records.toLocaleString();
    document.getElementById("testing-count").textContent = data.testing_records.toLocaleString();
    document.getElementById("accuracy-value").textContent = `${(data.metrics.accuracy * 100).toFixed(1)}%`;
    document.getElementById("precision-value").textContent = (data.metrics.precision_macro * 100).toFixed(1) + "%";
    document.getElementById("recall-value").textContent = (data.metrics.recall_macro * 100).toFixed(1) + "%";
    document.getElementById("f1-value").textContent = (data.metrics.f1_macro * 100).toFixed(1) + "%";

    const labels = data.confusion_matrix.labels;
    const values = data.confusion_matrix.values;
    const matrix = document.getElementById("confusion-matrix");
    matrix.innerHTML = `<span class="matrix-axis">Predicted class →</span>${values.map((row, rowIndex) => row.map((value, columnIndex) => `<div class="matrix-cell"><strong>${value}</strong><small>Actual ${labels[rowIndex]} · Predicted ${labels[columnIndex]}</small></div>`).join("")).join("")}`;
    document.getElementById("class-distribution").innerHTML = Object.entries(data.class_distribution).map(([label, value]) => `<div class="distribution-row"><span>${label}</span><strong>${value.toLocaleString()}</strong></div>`).join("");

    const reportBody = document.getElementById("classification-report");
    reportBody.innerHTML = labels.map((label) => {
      const item = data.classification_report[label];
      return `<tr><td>${label}</td><td>${(item.precision * 100).toFixed(1)}%</td><td>${(item.recall * 100).toFixed(1)}%</td><td>${(item["f1-score"] * 100).toFixed(1)}%</td><td>${item.support}</td></tr>`;
    }).join("");
    document.getElementById("evaluation-note").textContent = data.evaluation_note;
  } catch (error) {
    errorBox.textContent = error.message;
    errorBox.classList.remove("is-hidden");
  }
});
