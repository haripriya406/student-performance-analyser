document.addEventListener("DOMContentLoaded", async () => {
  const errorBox = document.getElementById("dashboard-error");
  const sampleBody = document.getElementById("dashboard-sample");
  const escapeText = (value) => String(value).replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  })[character]);
  const showError = (message) => {
    errorBox.textContent = message;
    errorBox.classList.remove("is-hidden");
    sampleBody.innerHTML = `<tr><td colspan="4" class="empty-state">${escapeText(message)}</td></tr>`;
    document.querySelectorAll(".chart-empty").forEach((item) => item.classList.remove("is-hidden"));
  };
  try {
    const response = await fetch("/api/dashboard");
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Dashboard data could not be loaded.");
    document.querySelector('[data-stat="total"]').textContent = data.total_students.toLocaleString();
    document.querySelector('[data-stat="score"]').textContent = `${data.average_final_score.toFixed(1)} / 100`;
    document.querySelector('[data-stat="attendance"]').textContent = `${data.average_attendance.toFixed(1)}%`;
    document.querySelector('[data-stat="risk"]').textContent = data.risk_counts["At Risk"].toLocaleString();
    document.querySelector('[data-stat="not-risk"]').textContent = data.risk_counts["Not At Risk"].toLocaleString();

    sampleBody.innerHTML = data.sample_students.map((student) => {
      const riskClass = student.academic_risk === "At Risk" ? "at-risk" : "not-at-risk";
      return `<tr><td>${escapeText(student.student_id)}</td><td>${Number(student.attendance_percentage).toFixed(1)}%</td><td>${Number(student.final_score).toFixed(1)}</td><td><span class="risk-badge ${riskClass}">${escapeText(student.academic_risk)}</span></td></tr>`;
    }).join("") || '<tr><td colspan="4" class="empty-state">No student records found.</td></tr>';

    if (window.Chart) {
      const common = { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: "bottom", labels: { usePointStyle: true, boxWidth: 7, color: "#718096", font: { size: 10 } } } } };
      new Chart(document.getElementById("risk-chart"), {
        type: "doughnut",
        data: { labels: data.risk_distribution.labels, datasets: [{ data: data.risk_distribution.values, backgroundColor: ["#e77777", "#54b894"], borderWidth: 0, hoverOffset: 5 }] },
        options: { ...common, cutout: "70%" }
      });
      new Chart(document.getElementById("score-chart"), {
        type: "bar",
        data: { labels: data.score_distribution.labels, datasets: [{ label: "Students", data: data.score_distribution.values, backgroundColor: "#6d91ed", borderRadius: 4, maxBarThickness: 26 }] },
        options: { ...common, plugins: { ...common.plugins, legend: { display: false } }, scales: { x: { grid: { display: false }, ticks: { color: "#8995a8", font: { size: 9 } } }, y: { beginAtZero: true, grid: { color: "#eef1f6" }, ticks: { color: "#8995a8", precision: 0 } } } }
      });
      new Chart(document.getElementById("attendance-chart"), {
        type: "scatter",
        data: { datasets: [{ label: "Student records", data: data.attendance_performance.attendance.map((value, index) => ({ x: value, y: data.attendance_performance.scores[index] })), backgroundColor: "rgb(65 116 230 / 42%)", pointRadius: 3 }] },
        options: { ...common, plugins: { ...common.plugins, legend: { display: false } }, scales: { x: { title: { display: true, text: "Attendance (%)", color: "#778398" }, min: 0, max: 100, grid: { color: "#eef1f6" } }, y: { title: { display: true, text: "Final score", color: "#778398" }, min: 0, max: 100, grid: { color: "#eef1f6" } } } }
      });
    } else {
      document.querySelectorAll(".chart-empty").forEach((item) => { item.textContent = "Charts could not load. Check your internet connection and refresh."; item.classList.remove("is-hidden"); });
    }
  } catch (error) {
    showError(error.message);
  }
});
