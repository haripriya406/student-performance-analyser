document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("upload-form");
  const status = document.getElementById("upload-status");
  const submit = document.getElementById("upload-submit");
  if (!form || !status || !submit) return;

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!form.reportValidity()) return;

    submit.disabled = true;
    status.className = "notice notice-info";
    status.textContent = "Validating the uploaded CSV...";
    try {
      const response = await fetch(form.action, {
        method: "POST",
        body: new FormData(form)
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "The dataset could not be uploaded.");
      status.className = "notice notice-success";
      status.textContent = `${data.message} ${data.rows.toLocaleString()} records validated. Refreshing the preview...`;
      window.setTimeout(() => window.location.reload(), 700);
    } catch (error) {
      status.className = "notice notice-error";
      status.textContent = error.message;
      submit.disabled = false;
    }
  });
});
