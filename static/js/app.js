"use strict";

const root = document.documentElement;
const form = document.getElementById("scan-form");
const fileInput = document.getElementById("images");
const dropzone = document.getElementById("dropzone");
const previews = document.getElementById("previews");
const submitBtn = document.getElementById("submit");
const formError = document.getElementById("form-error");
const resultEl = document.getElementById("result");
const hint = document.getElementById("drop-hint");
const deviceEl = document.getElementById("device");

const CATEGORY_NAMES = {
  raw_produce: "Fruit or vegetable",
  packaged_solid: "Packaged food",
  packaged_liquid: "Packaged drink",
  packaged_exempt: "Oil, ghee or honey",
};
const VERDICT_WORDS = {
  SAFE: "Good to eat",
  CAUTION: "Eat with care",
  UNSAFE: "Better not eat this",
  REJECTED: "We can't check this photo",
  MISMATCH: "That doesn't match",
};
// Icons drawn in a 48-unit box, placed in the middle of the seal.
const SEAL_ICONS = {
  REJECTED: "M10 10l28 28M24 4a20 20 0 1 0 0 40 20 20 0 0 0 0-40Z",
  MISMATCH: "M14 6 6 14l8 8M6 14h26M34 42l8-8-8-8m8 8H16",
  EXEMPT: "M24 6c-7 10-12 15-12 22a12 12 0 0 0 24 0c0-7-5-12-12-22Z",
};
const STAGE_NAMES = [
  [/^Stage 1/, "Is it food?"],
  [/^Stage 2/, "Is it the kind of food you picked?"],
  [/^Stage 3/, "Is it the product you named?"],
  [/^Pipeline A/, "Freshness check"],
  [/^Pipeline B/, "Label reading and FSSAI limits"],
];
const STAGE_STATUS = { PASS: "Passed", FAIL: "Stopped here", WARN: "Needs a look", SKIP: "Skipped" };
const STEPS = {
  raw: ["Making sure it is food", "Confirming it is a fruit or vegetable", "Checking it is what you named",
        "Looking for signs of spoilage", "Writing your review"],
  packaged: ["Making sure it is food", "Reading the label", "Matching the product",
             "Checking FSSAI limits", "Writing your review"],
};

let files = [];

const el = (tag, attrs = {}, ...children) => {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else node.setAttribute(k, v);
  }
  for (const c of children) if (c != null && c !== false) node.append(c);
  return node;
};
const svgEl = (tag, attrs = {}) => {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  return node;
};
const pretty = (s) => (s || "").replaceAll("_", " ");
const capital = (s) => s.charAt(0).toUpperCase() + s.slice(1);
const category = () => form.querySelector("input[name=category]:checked").value;
const isRaw = () => category() === "raw_produce";
const maxFiles = () => (isRaw() ? 1 : 3);
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const reducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// ----------------------------------------------------------------- theme --

document.getElementById("theme-toggle").addEventListener("click", () => {
  const current = root.dataset.theme
    || (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  const next = current === "dark" ? "light" : "dark";
  root.dataset.theme = next;
  try { localStorage.setItem("fg-theme", next); } catch (e) { /* private mode: theme just won't persist */ }
});

// ---------------------------------------------------------------- inputs --

function onCategoryChange() {
  const raw = isRaw();
  hint.textContent = raw
    ? "One clear photo of the item, filling the frame."
    : "Up to 3 photos: the nutrition table first, then the front of the pack.";
  form.querySelectorAll(".packaged-only").forEach((n) => { n.hidden = raw; });
  if (files.length > maxFiles()) { files = files.slice(0, maxFiles()); renderPreviews(); }
}
form.querySelectorAll("input[name=category]").forEach((r) => r.addEventListener("change", onCategoryChange));

function addFiles(list) {
  for (const f of list) {
    if (!f.type.startsWith("image/")) { formError.textContent = `${f.name} is not an image.`; continue; }
    if (files.length >= maxFiles()) files.shift();
    files.push(f);
    formError.textContent = "";
  }
  renderPreviews();
}
function renderPreviews() {
  previews.replaceChildren(...files.map((f, i) => {
    const img = el("img", { alt: f.name });
    img.src = URL.createObjectURL(f);
    const btn = el("button", { type: "button", "aria-label": `Remove ${f.name}`, text: "×" });
    btn.onclick = () => { files.splice(i, 1); renderPreviews(); };
    return el("figure", {}, img, btn);
  }));
}
fileInput.addEventListener("change", () => { addFiles(fileInput.files); fileInput.value = ""; });
["dragenter", "dragover"].forEach((e) => dropzone.addEventListener(e, (ev) => { ev.preventDefault(); dropzone.classList.add("drag"); }));
["dragleave", "drop"].forEach((e) => dropzone.addEventListener(e, (ev) => { ev.preventDefault(); dropzone.classList.remove("drag"); }));
dropzone.addEventListener("drop", (ev) => addFiles(ev.dataTransfer.files));

// ------------------------------------------------------------ processing --

function showProcessing() {
  const frag = document.getElementById("tpl-processing").content.cloneNode(true);
  const img = frag.querySelector(".scan-img");
  img.src = URL.createObjectURL(files[0]);
  const list = frag.querySelector(".steps");
  const steps = (isRaw() ? STEPS.raw : STEPS.packaged).map((s) => el("li", { text: s }));
  list.append(...steps);
  resultEl.replaceChildren(frag);
  if (window.innerWidth < 920) resultEl.scrollIntoView({ behavior: "smooth", block: "start" });

  // OCR makes packaged checks slower, so their steps advance more slowly.
  // The last step stays active until the server answers.
  const pace = isRaw() ? 900 : 2200;
  let i = 0;
  steps[0].classList.add("active");
  const timer = setInterval(() => {
    if (i >= steps.length - 1) return;
    steps[i].classList.replace("active", "done");
    steps[++i].classList.add("active");
  }, pace);
  return {
    async finish() {
      clearInterval(timer);
      steps.forEach((s) => { s.classList.remove("active"); s.classList.add("done"); });
      if (!reducedMotion()) await wait(450);
    },
  };
}

// ---------------------------------------------------------------- submit --

form.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  formError.textContent = "";
  if (!files.length) { formError.textContent = "Add a photo first."; return; }

  const data = new FormData(form);
  data.delete("images");
  files.forEach((f) => data.append("images", f));
  if (isRaw()) { data.delete("barcode"); data.delete("expiry_date"); }

  submitBtn.disabled = true;
  submitBtn.textContent = "Checking…";
  const progress = showProcessing();
  const started = Date.now();
  try {
    const res = await fetch("/api/analyze", { method: "POST", body: data });
    const json = await res.json().catch(() => ({ error: `The server answered with error ${res.status}.` }));
    if (!res.ok) throw new Error(json.error || `The server answered with error ${res.status}.`);
    if (Date.now() - started < 1200 && !reducedMotion()) await wait(1200 - (Date.now() - started));
    await progress.finish();
    renderResult(json);
  } catch (err) {
    await progress.finish();
    resultEl.replaceChildren(errorState(err.message));
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Check this food";
  }
});

function errorState(message) {
  return el("div", { class: "empty" },
    el("h2", { text: "The check did not finish" }),
    el("p", { text: message }),
    el("p", { text: "Make sure the FoodGuard server is still running, then try again." }));
}

// ---------------------------------------------------------------- render --

function setSeal(frag, r) {
  const v = r.verdict.status;
  const review = r.review;
  const svg = frag.querySelector(".seal svg");
  const score = frag.querySelector(".seal-score");
  const sub = frag.querySelector(".seal-sub");
  let icon = null;

  if (review && review.score != null) {
    score.textContent = review.score;
    sub.textContent = "out of 100";
  } else if (review && review.rating === "Exempt") {
    icon = SEAL_ICONS.EXEMPT; sub.textContent = "not scored";
  } else {
    icon = SEAL_ICONS[v] || SEAL_ICONS.REJECTED;
    sub.textContent = v === "MISMATCH" ? "mismatch" : "not checked";
  }
  if (icon) {
    score.remove();
    const g = svgEl("g", { transform: "translate(61 44) scale(.8)" });
    g.append(svgEl("path", { d: icon, class: "seal-icon" }));
    svg.append(g);
  }
  if (!reducedMotion()) frag.querySelector(".seal").classList.add("stamp-in");
}

function stars(n) {
  const wrap = el("span", { class: "stars", role: "img", "aria-label": `${n} out of 5 stars` });
  const fill = el("i", { text: "★★★★★" });
  fill.style.width = `${(n / 5) * 100}%`;
  wrap.append(fill);
  return wrap;
}

function hexagon(h) {
  const svg = svgEl("svg", { viewBox: "0 0 112 98", role: "img", "aria-label": `${h.label} warning` });
  svg.append(svgEl("polygon", { points: "28,3 84,3 110,49 84,95 28,95 2,49" }));
  const words = h.label.split(" ");
  const lines = words.length > 2 ? [words.slice(0, 2).join(" "), words.slice(2).join(" ")]
              : words.length === 2 ? words : [h.label];
  lines.forEach((line, i) => {
    const t = svgEl("text", { x: 56, y: (lines.length === 1 ? 54 : 46) + i * 18 });
    t.textContent = line;
    svg.append(t);
  });
  return el("div", { class: "hex" }, svg, el("small", { text: `${h.value} g, limit ${h.limit} g` }));
}

function section(title, lede, ...body) {
  return el("section", { class: "reveal" }, el("h3", { text: title }), lede ? el("p", { class: "lede", text: lede }) : null, ...body);
}

function aboutSection(r) {
  const info = r.review.info;
  const facts = el("dl", { class: "facts" },
    ...info.facts.map((f) => el("div", {}, el("dt", { text: f.label }), el("dd", { text: f.value }))));
  const notes = el("div", { class: "about-notes" });
  const note = (title, body) => body && notes.append(el("div", {}, el("h4", { text: title }), body));
  if (info.known_for && info.known_for.length) note("Known for", el("ul", {}, ...info.known_for.map((k) => el("li", { text: k }))));
  if (info.storage) note("How to store it", el("p", { text: info.storage }));
  if (info.shelf_life) note("How long it keeps", el("p", { text: info.shelf_life }));
  if (info.spoilage_signs) note("Signs it has gone off", el("p", { text: info.spoilage_signs }));
  if (info.ingredients && info.ingredients.length) {
    const more = info.ingredient_count > info.ingredients.length ? ` and ${info.ingredient_count - info.ingredients.length} more` : "";
    note("Ingredients", el("p", { text: info.ingredients.join(", ") + more }));
  }
  return section(info.title, info.subtitle, info.facts.length ? facts : null,
    notes.childElementCount ? notes : null, info.source ? el("p", { class: "source", text: `Source: ${info.source.replace(/\.+$/, "")}.` }) : null);
}

function bar(name, valueText, fraction, limitAt, over, note) {
  const track = el("div", { class: "bar" });
  const fill = el("i", { class: over ? "over" : "" });
  fill.style.width = `${Math.min(1, fraction) * 100}%`;
  track.append(fill);
  if (limitAt != null) {
    const mark = el("span", { class: "limit", title: "FSSAI limit" });
    mark.style.left = `${limitAt * 100}%`;
    track.append(mark);
  }
  return el("div", { class: "bar-row" },
    el("div", { class: "bar-head" }, el("span", { text: name }), el("b", { text: valueText })),
    track, note ? el("p", { class: "bar-note", text: note }) : null);
}

function limitsSection(r) {
  const c = r.packaged.compliance;
  if (c.exempt) return null;
  const unit = (c.basis || "100g").slice(3) || "g";
  const rows = c.checks.map((ch) => {
    if (ch.value == null) return bar(ch.name, "not readable", 0, 1 / 1.5, false, ch.note);
    const scale = Math.max(ch.limit * 1.5, ch.value);
    return bar(ch.name, `${ch.less_than ? "<" : ""}${ch.value} g of ${ch.limit} g allowed`,
      ch.value / scale, ch.limit / scale, ch.status === "EXCEEDED", ch.note);
  });
  return section("Against the FSSAI limits", `Per 100 ${unit}. The dark line marks the limit.`, ...rows);
}

function dailySection(info) {
  if (!info.daily_share || !info.daily_share.length) return null;
  return section("Share of your day", info.daily_note,
    ...info.daily_share.map((d) => bar(d.name, `${d.percent}% of ${d.daily_max}`, d.percent / 100, null, d.percent >= 100)));
}

function scoreSection(r) {
  const rev = r.review;
  const items = rev.breakdown.map((b) => {
    const pts = b.points;
    const label = r.produce ? `${pts}` : pts === 0 ? "0" : pts > 0 ? `+${pts}` : `${pts}`;
    return el("li", {}, el("span", { text: b.factor }), el("b", { class: pts < 0 ? "neg" : r.produce ? "pos" : "", text: label }));
  });
  const lede = r.produce
    ? "For fruit and vegetables the score is how sure the spoilage model is that the item is fresh."
    : rev.score == null ? "Single-ingredient foods are not scored against the HFSS limits."
    : "Packaged food starts at 100 and loses points as sugar, saturated fat and sodium approach their FSSAI limits.";
  const readings = r.verdict.details.length
    ? el("details", { class: "inline" }, el("summary", { text: "Readings behind this result" }),
         el("ul", {}, ...r.verdict.details.map((d) => el("li", { text: d }))))
    : null;
  return section("How the score was worked out", lede, el("ul", { class: "breakdown" }, ...items), readings);
}

function checksSection(r) {
  const items = r.stages.map((s) => {
    const name = (STAGE_NAMES.find(([re]) => re.test(s.name)) || [null, s.name])[1];
    return el("li", { class: s.status },
      el("div", {}, el("b", { text: name }), el("span", {}, el("em", { text: STAGE_STATUS[s.status] || s.status }), ` ${s.detail}`)));
  });
  return section("How we checked", "Each photo goes through these checks in order and stops at the first one it fails.",
    el("ol", { class: "checks" }, ...items));
}

function proofSection(r) {
  if (!r.evidence.length) return null;
  const figs = r.evidence.map((ev) => {
    const img = el("img", { alt: ev.title, loading: "lazy" });
    img.src = `data:image/png;base64,${ev.png}`;
    return el("figure", {}, img, el("figcaption", { text: ev.title }));
  });
  const ocr = r.packaged && r.packaged.ocr_text.length
    ? el("details", { class: "inline" }, el("summary", { text: `Text read from the label (${r.packaged.ocr_text.length} lines)` }),
         el("pre", { text: r.packaged.ocr_text.join("\n") }))
    : null;
  return section("What the model looked at", r.produce
    ? "Warm colours show the parts of the photo that mattered most to the freshness decision."
    : "Boxes show every piece of text found; the nutrition rows are outlined in red.",
    el("div", { class: "proof" }, ...figs), ocr);
}

function renderResult(r) {
  const frag = document.getElementById("tpl-result").content.cloneNode(true);
  const v = r.verdict;
  const rev = r.review;

  frag.querySelector(".verdict").classList.add(`v-${v.status}`);
  setSeal(frag, r);
  frag.querySelector(".verdict-word").textContent = VERDICT_WORDS[v.status] || capital(v.headline.toLowerCase());
  frag.querySelector(".verdict-reason").textContent = v.reason;
  const rating = frag.querySelector(".verdict-rating");
  if (rev && rev.score != null) rating.append(stars(rev.stars), `Rated ${rev.rating}, ${rev.score} out of 100`);
  else if (rev) rating.textContent = "Exempt from the HFSS limits, so no health score.";
  else rating.remove();

  frag.querySelector(".hexagons").append(...(r.hexagons || []).map(hexagon));

  if (r.suggested_category && r.suggested_category !== r.category) {
    const btn = el("button", { type: "button", text: `Check again as ${CATEGORY_NAMES[r.suggested_category] || pretty(r.suggested_category)}` });
    btn.onclick = () => {
      const radio = form.querySelector(`input[name=category][value=${r.suggested_category}]`);
      if (radio) { radio.checked = true; onCategoryChange(); form.requestSubmit(); }
    };
    frag.querySelector(".suggest").append(btn);
  }
  frag.querySelector(".notes").append(...r.warnings.map((w) => el("p", { text: w })));

  const sections = [];
  if (rev) sections.push(aboutSection(r));
  if (r.packaged) sections.push(limitsSection(r), dailySection(rev.info));
  if (rev) sections.push(scoreSection(r));
  sections.push(checksSection(r), proofSection(r));
  frag.querySelector(".sections").append(...sections.filter(Boolean));

  const slim = JSON.parse(JSON.stringify(r));
  slim.evidence = slim.evidence.map((e) => ({ title: e.title, png: `<${e.png.length} base64 characters>` }));
  frag.querySelector(".raw pre").textContent = JSON.stringify(slim, null, 2);

  resultEl.replaceChildren(frag);
  if (resultEl.getBoundingClientRect().top < 0 || window.innerWidth < 920) {
    resultEl.scrollIntoView({ behavior: reducedMotion() ? "auto" : "smooth", block: "start" });
  }
}

// ---------------------------------------------------------------- health --

async function checkHealth() {
  try {
    const h = await (await fetch("/api/health")).json();
    const where = h.device === "cuda" ? "GPU" : "CPU";
    if (!h.spoilage_model_trained) {
      deviceEl.textContent = "Spoilage model not trained";
      return;
    }
    const ready = ["clip", "ocr", "spoilage"].every((k) => h.loaded.includes(k));
    deviceEl.textContent = ready ? `Ready on ${where}` : `Loading models on ${where}`;
    deviceEl.classList.toggle("ready", ready);
    if (!ready) setTimeout(checkHealth, 3000);
  } catch (e) {
    deviceEl.textContent = "Server offline";
  }
}

checkHealth();
onCategoryChange();
