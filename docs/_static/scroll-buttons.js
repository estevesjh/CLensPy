/*
 * Jump-to-top / jump-to-bottom buttons, fixed bottom-right. The API
 * reference and the longer Theory pages run to many screens; a plain
 * scrollbar drag is the only alternative without this.
 */
(function () {
  function init() {
    var wrap = document.createElement("div");
    wrap.className = "clenspy-scroll-buttons";

    var up = document.createElement("button");
    up.type = "button";
    up.className = "clenspy-scroll-btn clenspy-scroll-up";
    up.setAttribute("aria-label", "Scroll to top");
    up.innerHTML = "&#9650;";

    var down = document.createElement("button");
    down.type = "button";
    down.className = "clenspy-scroll-btn clenspy-scroll-down";
    down.setAttribute("aria-label", "Scroll to bottom");
    down.innerHTML = "&#9660;";

    wrap.appendChild(up);
    wrap.appendChild(down);
    document.body.appendChild(wrap);

    up.addEventListener("click", function () {
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
    down.addEventListener("click", function () {
      window.scrollTo({
        top: document.documentElement.scrollHeight,
        behavior: "smooth",
      });
    });

    function updateVisibility() {
      var scrollTop = window.scrollY || document.documentElement.scrollTop;
      var atBottom =
        window.innerHeight + scrollTop >=
        document.documentElement.scrollHeight - 2;
      up.classList.toggle("clenspy-scroll-hidden", scrollTop < 200);
      down.classList.toggle("clenspy-scroll-hidden", atBottom);
    }

    window.addEventListener("scroll", updateVisibility, { passive: true });
    window.addEventListener("resize", updateVisibility);
    updateVisibility();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
