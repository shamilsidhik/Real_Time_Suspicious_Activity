(() => {
  const sidebar = document.getElementById("appSidebar");
  const backdrop = document.getElementById("sidebarBackdrop");
  const openButton = document.getElementById("sidebarOpen");
  const closeButton = document.getElementById("sidebarClose");

  const openSidebar = () => {
    sidebar?.classList.add("open");
    backdrop?.classList.add("open");
    document.body.style.overflow = "hidden";
  };

  const closeSidebar = () => {
    sidebar?.classList.remove("open");
    backdrop?.classList.remove("open");
    document.body.style.overflow = "";
  };

  openButton?.addEventListener("click", openSidebar);
  closeButton?.addEventListener("click", closeSidebar);
  backdrop?.addEventListener("click", closeSidebar);

  const publicMenuButton = document.getElementById("publicMenuButton");
  const publicNavLinks = document.getElementById("publicNavLinks");

  publicMenuButton?.addEventListener("click", () => {
    publicNavLinks?.classList.toggle("open");
  });
})();
