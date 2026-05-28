const menuButton = document.querySelector(".menu-button");
const navLinks = document.querySelector(".nav-links");
const copyButton = document.querySelector(".copy-button");
const toast = document.querySelector(".toast");

menuButton?.addEventListener("click", () => {
  const isOpen = navLinks.classList.toggle("is-open");
  menuButton.setAttribute("aria-expanded", String(isOpen));
});

navLinks?.addEventListener("click", (event) => {
  if (event.target instanceof HTMLAnchorElement) {
    navLinks.classList.remove("is-open");
    menuButton?.setAttribute("aria-expanded", "false");
  }
});

copyButton?.addEventListener("click", async () => {
  const citation = document.querySelector(".bibtex code")?.textContent?.trim();
  if (!citation) return;

  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(citation);
    } else {
      copyWithFallback(citation);
    }
    showToast("BibTeX copied");
  } catch {
    try {
      copyWithFallback(citation);
      showToast("BibTeX copied");
    } catch {
      showToast("BibTeX ready");
    }
  }
});

function showToast(message) {
  toast.textContent = message;
  toast.classList.add("is-visible");
  window.setTimeout(() => toast.classList.remove("is-visible"), 1800);
}

function copyWithFallback(text) {
  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "");
  textarea.style.position = "fixed";
  textarea.style.top = "-999px";
  document.body.append(textarea);
  textarea.select();
  const copied = document.execCommand("copy");
  textarea.remove();
  if (!copied) {
    throw new Error("Copy command failed");
  }
}
