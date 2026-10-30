document.addEventListener("DOMContentLoaded", () => {
  const dataElement = document.getElementById("dashboard-chart-data");
  if (!dataElement) return;

  const data = JSON.parse(dataElement.textContent);
  const notice = (canvasId, message) => {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const paragraph = document.createElement("p");
    paragraph.className = "empty-state";
    paragraph.textContent = message;
    canvas.replaceWith(paragraph);
  };
  if (!window.Chart) {
    ["risk-chart", "study-chart", "attendance-chart"].forEach((id) => {
      notice(id, "Charts could not load. Check your internet connection and refresh.");
    });
    return;
  }

  const colors = ["#e77777", "#54b894"];
  const commonOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: "bottom",
        labels: { usePointStyle: true, boxWidth: 8, color: "#718096" }
      }
    }
  };
  const riskCanvas = document.getElementById("risk-chart");
  if (riskCanvas) {
    new Chart(riskCanvas, {
      type: "doughnut",
      data: {
        labels: data.risk.labels,
        datasets: [{
          data: data.risk.values,
          backgroundColor: colors,
          borderWidth: 0,
          hoverOffset: 5
        }]
      },
      options: { ...commonOptions, cutout: "68%" }
    });
  }

  const makeBarChart = (canvasId, chartData, label, color, maximum) => {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    new Chart(canvas, {
      type: "bar",
      data: {
        labels: chartData.labels,
        datasets: [{
          label,
          data: chartData.values,
          backgroundColor: color,
          borderRadius: 6,
          maxBarThickness: 54
        }]
      },
      options: {
        ...commonOptions,
        plugins: { ...commonOptions.plugins, legend: { display: false } },
        scales: {
          x: { grid: { display: false }, ticks: { color: "#8995a8" } },
          y: {
            beginAtZero: true,
            ...(maximum ? { max: maximum } : {}),
            grid: { color: "#eef1f6" },
            ticks: { color: "#8995a8" }
          }
        }
      }
    });
  };
  makeBarChart("study-chart", data.study, "Average study hours per day", "#4a83e8");
  makeBarChart("attendance-chart", data.attendance, "Average attendance (%)", "#22a68a", 100);
});
