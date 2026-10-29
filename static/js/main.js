document.addEventListener("DOMContentLoaded", () => {
  const menuButton = document.querySelector(".menu-toggle");
  const sidebar = document.querySelector(".sidebar");
  if (!menuButton || !sidebar) return;

  menuButton.addEventListener("click", () => {
    const open = sidebar.classList.toggle("is-open");
    menuButton.setAttribute("aria-expanded", String(open));
    menuButton.setAttribute("aria-label", open ? "Close navigation" : "Open navigation");
  });
  document.addEventListener("click", (event) => {
    if (!sidebar.classList.contains("is-open")) return;
    if (sidebar.contains(event.target) || menuButton.contains(event.target)) return;
    sidebar.classList.remove("is-open");
    menuButton.setAttribute("aria-expanded", "false");
    menuButton.setAttribute("aria-label", "Open navigation");
  });
});
