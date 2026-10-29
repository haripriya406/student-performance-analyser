document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("student-filters");
  const body = document.getElementById("students-body");
  const errorBox = document.getElementById("students-error");
  const count = document.getElementById("students-count");
  const indicator = document.getElementById("page-indicator");
  let page = 1;
  let totalPages = 0;
  let requestNumber = 0;
  const safe = (value) => String(value).replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  })[character]);
  const load = async () => {
    const currentRequest = ++requestNumber;
    const params = new URLSearchParams(new FormData(form));
    params.set("page", String(page));
    params.set("per_page", "20");
    errorBox.classList.add("is-hidden");
    body.innerHTML = '<tr><td colspan="6" class="empty-state">Loading student records…</td></tr>';
    try {
      const response = await fetch(`/api/students?${params}`);
      const data = await response.json();
      if (currentRequest !== requestNumber) return;
      if (!response.ok) throw new Error(data.error || "Student records could not be loaded.");
      totalPages = data.total_pages;
      body.innerHTML = data.records.length ? data.records.map((student) => {
        const riskClass = student.academic_risk === "At Risk" ? "at-risk" : "not-at-risk";
        return `<tr><td>${safe(student.student_id)}</td><td>${student.age}</td><td>${Number(student.attendance_percentage).toFixed(1)}%</td><td>${Number(student.previous_score).toFixed(1)}</td><td>${Number(student.final_score).toFixed(1)}</td><td><span class="risk-badge ${riskClass}">${safe(student.academic_risk)}</span></td></tr>`;
      }).join("") : '<tr><td colspan="6" class="empty-state">No records match these filters.</td></tr>';
      count.textContent = `${data.total.toLocaleString()} matching records`;
      indicator.textContent = `Page ${data.total_pages ? data.page : 0} of ${data.total_pages}`;
      document.getElementById("previous-page").disabled = page <= 1;
      document.getElementById("next-page").disabled = page >= totalPages;
    } catch (error) {
      if (currentRequest !== requestNumber) return;
      errorBox.textContent = error.message;
      errorBox.classList.remove("is-hidden");
      body.innerHTML = '<tr><td colspan="6" class="empty-state">Records are unavailable.</td></tr>';
      count.textContent = "";
      indicator.textContent = "";
    }
  };
  let timer;
  form.addEventListener("input", () => { page = 1; clearTimeout(timer); timer = setTimeout(load, 250); });
  form.addEventListener("change", () => { page = 1; load(); });
  form.addEventListener("reset", () => { page = 1; setTimeout(load, 0); });
  document.getElementById("previous-page").addEventListener("click", () => { if (page > 1) { page -= 1; load(); } });
  document.getElementById("next-page").addEventListener("click", () => { if (page < totalPages) { page += 1; load(); } });
  load();
});
