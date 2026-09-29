// Start the two API trees closed in the sidebar.
//
// Shibuya expands the sidebar in its own script, from globaltoc_expand_depth,
// and does it at load time -- so the classes in the HTML are overwritten and
// this has to run after it. Both scripts are deferred and this one is listed
// later, so it does. A section containing the current page is left open.

const COLLAPSED_SECTIONS = ["api/index.html", "python/index.html"];

document.querySelectorAll(".globaltoc li.toctree-l1._expand").forEach((li) => {
  const link = li.querySelector(":scope > a");
  const href = link && link.getAttribute("href");
  if (!href || li.classList.contains("current")) return;
  if (!COLLAPSED_SECTIONS.some((target) => href.endsWith(target))) return;

  li.classList.replace("_expand", "_collapse");
  const button = li.querySelector(":scope > button");
  if (button) button.setAttribute("aria-label", "Expand " + link.textContent);
});
