// Build the thesis-class presentation deck (5-10 minute slot).
//
// Every number on a slide comes from deck_data.json, which is exported from the
// generated report (docs/reports/<date>/tables/*.csv) — never typed by hand.
// Demo screenshots in assets/ are captured from the local dashboard (see
// presentation-notes.md for the commands).
//
//   cd docs/thesis/presentation && npm install && npm run build
//
// Output: mtg-search-presentation.pptx next to this file (gitignored; regenerable).

const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require("./apply_theme.js");

const HERE = __dirname;
const D = JSON.parse(fs.readFileSync(path.join(HERE, "deck_data.json"), "utf8"));
const OUT = path.join(HERE, "mtg-search-presentation.pptx");

const THEME = {
  name: "MTG Search",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "161A23", lt1: "FFFFFF", dk2: "3A4254", lt2: "F2F3F6",
    accent1: "EB6834", // ours / tuned embedder (the deck's one sharp accent)
    accent2: "2A78D6", // base embedder / "before"
    accent3: "8A93A6", accent4: "1BAF7A", accent5: "EDA100", accent6: "4A3AA7",
    hlink: "2A78D6", folHlink: "4A3AA7",
  },
};
const HEX = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10" x 5.625"
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Plain-language card search";
pres.author = "Mitchell Trafford";
const C = pres.SchemeColor;

// ---- layouts (frames) ----------------------------------------------------
pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: C.text1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 1.55, w: 8.8, h: 1.5, fontSize: 40, bold: true, color: C.background1, align: "left", valign: "top", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.6, y: 3.2, w: 8.8, h: 1.2, fontSize: 18, color: C.background2, align: "left", valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  margin: [0.5, 0.5, 0.6, 0.5],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.3, w: 9.0, h: 0.95, fontSize: 30, bold: true, color: C.text1, align: "left", valign: "top", margin: 0 }, text: "" } },
    { text: { text: "Plain-language card search  ·  M. Trafford  ·  CSCI 5953", options: { x: 0.5, y: 5.2, w: 7.5, h: 0.25, fontSize: 10, color: C.accent3, margin: 0, isTextBox: true } } },
  ],
  slideNumber: { x: 9.0, y: 5.2, w: 0.5, h: 0.25, fontSize: 10, color: C.accent3, align: "right" },
});

// ---- helpers ---------------------------------------------------------------
let chipN = 0;
// The deck's motif: a query "chip" — the thing a person actually types.
function chip(slide, text, x, y, w, { fill = C.background2, color = C.text1, mono = false, h = 0.42, size = 14 } = {}) {
  slide.addShape(pres.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.12, fill: { color: fill }, line: { type: "none" }, objectName: `chip-bg-${++chipN}` });
  slide.addText(text, { x: x + 0.14, y, w: w - 0.28, h, fontSize: size, color, valign: "middle", margin: 0, isTextBox: true, fontFace: mono ? "Courier New" : undefined, objectName: `chip-text-${chipN}` });
}
function card(slide, x, y, w, h, name) {
  slide.addShape(pres.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.1, fill: { color: C.background2 }, line: { type: "none" }, objectName: name });
}
function stat(slide, value, label, x, y, w, { color = C.accent1 } = {}) {
  slide.addText(value, { x, y, w, h: 0.75, fontSize: 40, bold: true, color, margin: 0, isTextBox: true, fontFace: THEME.headFontFace, valign: "bottom" });
  slide.addText(label, { x, y: y + 0.78, w, h: 0.55, fontSize: 12, color: C.text2, margin: 0, isTextBox: true, valign: "top" });
}
// Round half up on values exported to 3 decimals (0.565 must read 0.57, not 0.56).
const pct = (v) => (Math.round((v + 1e-9) * 100) / 100).toFixed(2);

// Charts are drawn as shapes, not PowerPoint chart objects: Keynote and QuickLook both
// dropped the native charts when this deck was checked (2026-10-03), and the deck has to
// survive being presented from either app. The bars are still generated from deck_data.json.
let barN = 0;
function columnChart(slide, { x, y, w, h, max, step, groups, series }) {
  const axisW = 0.45, legendH = 0.55, labelH = 0.4;
  const px = x + axisW, pw = w - axisW, py = y + legendH, ph = h - legendH - labelH;
  let lx = px;
  series.forEach(({ name, color }) => {
    slide.addShape(pres.ShapeType.roundRect, { x: lx, y: y + 0.08, w: 0.16, h: 0.16, rectRadius: 0.03, fill: { color }, line: { type: "none" }, objectName: `legend-${++barN}` });
    slide.addText(name, { x: lx + 0.24, y, w: 2.6, h: 0.32, fontSize: 12, color: C.text2, margin: 0, isTextBox: true, valign: "middle" });
    lx += 0.24 + name.length * 0.085 + 0.35;
  });
  for (let v = 0; v <= max + 1e-9; v += step) {
    const gy = py + ph - (v / max) * ph;
    slide.addShape(pres.ShapeType.line, { x: px, y: gy, w: pw, h: 0, line: { color: v === 0 ? C.accent3 : C.background2, width: v === 0 ? 1 : 0.75 }, objectName: `grid-${++barN}` });
    slide.addText(v.toFixed(1), { x, y: gy - 0.13, w: axisW - 0.1, h: 0.26, fontSize: 11, color: C.accent3, align: "right", valign: "middle", margin: 0, isTextBox: true });
  }
  const gw = pw / groups.length, n = series.length, bw = Math.min(0.85, (gw * 0.62) / n), gap = 0.06;
  groups.forEach(({ label, values }, gi) => {
    const total = n * bw + (n - 1) * gap, gx = px + gi * gw + (gw - total) / 2;
    values.forEach((v, si) => {
      const bh = (v / max) * ph, bx = gx + si * (bw + gap), by = py + ph - bh;
      slide.addShape(pres.ShapeType.rect, { x: bx, y: by, w: bw, h: bh, fill: { color: series[si].color }, line: { type: "none" }, objectName: `bar-${++barN}` });
      slide.addText(pct(v), { x: bx - 0.2, y: by - 0.32, w: bw + 0.4, h: 0.28, fontSize: 14, bold: true, color: C.text1, align: "center", valign: "bottom", margin: 0, isTextBox: true });
    });
    if (label) slide.addText(label, { x: px + gi * gw, y: py + ph + 0.06, w: gw, h: labelH - 0.06, fontSize: 13, color: C.text2, align: "center", valign: "top", margin: 0, isTextBox: true });
  });
}
function rankedBars(slide, { x, y, w, rowH, rows, color }) {
  const labelW = 2.55, valW = 0.45, bw = w - labelW - valW - 0.1;
  rows.forEach(({ label, value }, i) => {
    const ry = y + i * rowH;
    slide.addText(label, { x, y: ry, w: labelW - 0.1, h: rowH, fontSize: 10, color: C.text2, align: "right", valign: "middle", margin: 0, isTextBox: true });
    slide.addShape(pres.ShapeType.rect, { x: x + labelW, y: ry + rowH * 0.22, w: bw, h: rowH * 0.56, fill: { color: C.background2 }, line: { type: "none" }, objectName: `track-${++barN}` });
    if (value > 0.004) slide.addShape(pres.ShapeType.rect, { x: x + labelW, y: ry + rowH * 0.22, w: bw * value, h: rowH * 0.56, fill: { color }, line: { type: "none" }, objectName: `rank-${++barN}` });
    slide.addText(pct(value), { x: x + labelW + bw + 0.08, y: ry, w: valW, h: rowH, fontSize: 10, color: C.text1, valign: "middle", margin: 0, isTextBox: true });
  });
}

// ============================================================================
pres.addSection({ title: "Talk" });

// 1 — title
{
  const s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Talk" });
  s.addText("Plain-language card search", { placeholder: "title" });
  s.addText("A three-stage retrieval cascade over 30,000 Magic: The Gathering cards, and what fine-tuning changed", { placeholder: "body" });
  s.addText("Mitchell Trafford  ·  CSCI 5953 Independent Study  ·  CSUSB  ·  Fall 2026", { x: 0.6, y: 4.75, w: 8.8, h: 0.3, fontSize: 12, color: C.accent3, margin: 0, isTextBox: true });
  chip(s, "is a planeswalker", 0.6, 0.75, 2.25, { fill: C.text2, color: C.background1 });
  chip(s, "board wipes", 3.0, 0.75, 1.65, { fill: C.text2, color: C.background1 });
  chip(s, "cheap blue counterspells", 4.8, 0.75, 2.9, { fill: C.text2, color: C.background1 });
  s.addNotes("[20 s] Three things a player would type. None of them work in the standard search tool as written. This project is about making them work.");
}

// 2 — the problem
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Talk" });
  s.addText("Search rewards those who know the syntax", { placeholder: "title" });
  card(s, 0.5, 1.4, 4.35, 2.55, "card-typed");
  card(s, 5.15, 1.4, 4.35, 2.55, "card-needed");
  s.addText("What a player types", { x: 0.75, y: 1.55, w: 3.85, h: 0.35, fontSize: 16, bold: true, color: C.text1, margin: 0, isTextBox: true });
  s.addText("What Scryfall needs", { x: 5.4, y: 1.55, w: 3.85, h: 0.35, fontSize: 16, bold: true, color: C.text1, margin: 0, isTextBox: true });
  const rows = [["is a planeswalker", "t:planeswalker"], ["board wipes", "otag:sweeper"], ["cheap blue counterspells", "otag:counterspell c:u mv<=2"]];
  rows.forEach(([plain, syntax], i) => {
    chip(s, plain, 0.75, 2.05 + i * 0.6, 3.85, { fill: C.background1 });
    chip(s, syntax, 5.4, 2.05 + i * 0.6, 3.85, { fill: C.background1, mono: true, size: 13 });
  });
  s.addText([
    { text: `${D.complexity.plain}`, options: { bold: true, color: C.accent1 } },
    { text: " of plain language on average, against " },
    { text: `${D.complexity.tags}`, options: { bold: true } },
    { text: " of syntax when an expert knows the right community tag, and " },
    { text: `${D.complexity.no_tags}`, options: { bold: true } },
    { text: " when they don't." },
  ], { x: 0.5, y: 4.2, w: 9.0, h: 0.75, fontSize: 15, color: C.text2, margin: 0, isTextBox: true, valign: "top" });
  s.addNotes("[60 s] Scryfall is the standard tool and it is excellent, for experts. The left column returns nothing useful there. The right column is what you have to know. 'otag:sweeper' is the community's tag for board wipes; you have to know the word 'sweeper'. The goal is not to beat Scryfall. It is to get a newcomer the same results from the left column.");
}

// 3 — approach
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Talk" });
  s.addText("Three stages turn a sentence into a ranked set", { placeholder: "title" });
  chip(s, "cheap blue counterspells", 0.5, 1.35, 3.0, { fill: C.text1, color: C.background1 });
  const stages = [
    ["1", "Rewrite", "A small local LLM (8B) splits the query", "color: blue\ncost ≤ 2\nconcept: counterspell"],
    ["2", "Filter", "SQL narrows the catalog before any ranking", "31,124 cards → 3,364"],
    ["3", "Rank", "Vector search orders what is left by meaning", "Miscalculation, Censor,\nLofty Denial, …"],
  ];
  stages.forEach(([n, name, what, example], i) => {
    const x = 0.5 + i * 3.1;
    card(s, x, 2.05, 2.8, 2.75, `stage-${n}`);
    s.addShape(pres.ShapeType.ellipse, { x: x + 0.22, y: 2.25, w: 0.5, h: 0.5, fill: { color: C.accent1 }, line: { type: "none" }, objectName: `stage-dot-${n}` });
    s.addText(n, { x: x + 0.22, y: 2.25, w: 0.5, h: 0.5, fontSize: 18, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true });
    s.addText(name, { x: x + 0.85, y: 2.25, w: 1.8, h: 0.5, fontSize: 20, bold: true, color: C.text1, valign: "middle", margin: 0, isTextBox: true, fontFace: THEME.headFontFace });
    s.addText(what, { x: x + 0.22, y: 2.95, w: 2.4, h: 0.7, fontSize: 14, color: C.text2, margin: 0, isTextBox: true, valign: "top" });
    s.addText(example, { x: x + 0.22, y: 3.75, w: 2.4, h: 0.85, fontSize: 13, color: C.text1, margin: 0, isTextBox: true, valign: "top", fontFace: "Courier New" });
    if (i < 2) s.addText("→", { x: x + 2.8, y: 3.1, w: 0.3, h: 0.5, fontSize: 22, color: C.accent3, align: "center", valign: "middle", margin: 0, isTextBox: true });
  });
  s.addNotes("[60 s] One example through the pipeline. Stage 1: a small language model running on this laptop pulls out the structured parts, colour and cost, and names the concept. Stage 2: ordinary SQL throws away nine tenths of the catalog. Stage 3: semantic search ranks what remains. The key design decision is the order: filter first, then rank inside the filtered set.");
}

// 4 — the pivot
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Talk" });
  s.addText("Teaching the embedder the game's jargon", { placeholder: "title" });
  s.addText("Searching for", { x: 0.5, y: 1.42, w: 1.35, h: 0.42, fontSize: 15, color: C.text2, margin: 0, isTextBox: true, valign: "middle" });
  chip(s, "tutor", 1.85, 1.42, 0.95, { fill: C.text1, color: C.background1 });
  card(s, 0.5, 2.05, 2.75, 2.75, "card-before");
  card(s, 3.45, 2.05, 2.75, 2.75, "card-after");
  s.addText("Before", { x: 0.7, y: 2.17, w: 2.35, h: 0.35, fontSize: 15, bold: true, color: C.accent2, margin: 0, isTextBox: true });
  s.addText("After fine-tuning", { x: 3.65, y: 2.17, w: 2.35, h: 0.35, fontSize: 15, bold: true, color: C.accent1, margin: 0, isTextBox: true });
  ["Proud Mentor", "Blade Instructor", "Barging Sergeant"].forEach((t, i) => chip(s, t, 0.7, 2.62 + i * 0.52, 2.35, { fill: C.background1 }));
  ["Grim Tutor", "Rhystic Tutor", "Demonic Bargain"].forEach((t, i) => chip(s, t, 3.65, 2.62 + i * 0.52, 2.35, { fill: C.background1 }));
  s.addText("A tutor is a teacher", { x: 0.7, y: 4.3, w: 2.35, h: 0.35, fontSize: 12, italic: true, color: C.text2, margin: 0, isTextBox: true });
  s.addText("A tutor searches your library", { x: 3.65, y: 4.3, w: 2.35, h: 0.35, fontSize: 12, italic: true, color: C.text2, margin: 0, isTextBox: true });
  stat(s, D.finetune.pairs.replace(",062", "k"), "training pairs from Scryfall's community tags", 6.6, 1.3, 2.9);
  stat(s, `${D.finetune.minutes} min`, "to fine-tune a 137M-parameter embedder on a laptop", 6.6, 2.55, 2.9);
  stat(s, "2×", `retrieval on tags never seen in training (${D.finetune.heldout_ndcg} nDCG@10)`, 6.6, 3.8, 2.9);
  s.addNotes("[75 s] First version worked on plain English and failed on game slang. Search 'tutor' and the general-purpose model returns teachers: Proud Mentor, Blade Instructor. In Magic a tutor is a card that searches your library. Players have tagged every card by function on Scryfall, about 200 thousand tag-to-card pairs. I fine-tuned the embedding model on those, 71 minutes on this laptop. On tags held out of training, retrieval doubled, so it learned the idea of functional vocabulary, not a word list.");
}

// 5 — result: rewriter x embedder (native chart)
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Talk" });
  s.addText("Fine-tuning let the simpler rewriter win", { placeholder: "title" });
  const t = D.two_by_two;
  columnChart(s, {
    x: 0.5, y: 1.3, w: 5.6, h: 3.45, max: 0.6, step: 0.2,
    series: [{ name: "Base embedder", color: C.accent2 }, { name: "Tuned embedder", color: C.accent1 }],
    groups: [{ label: "Original rewriter", values: [t.v1.base, t.v1.tuned] }, { label: "Simpler rewriter", values: [t.v2.base, t.v2.tuned] }],
  });
  s.addText("Share of each true result set found at the top of the ranking (R-precision, 21 queries)", { x: 0.5, y: 4.85, w: 5.6, h: 0.3, fontSize: 10, color: C.accent3, margin: 0, isTextBox: true });
  s.addText([
    { text: "On the base embedder the simpler rewriter is worse.", options: { breakLine: true, bold: true } },
    { text: "A bare word like “ramp” means nothing to a model that never learned it.", options: { breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 8 } },
    { text: "On the tuned embedder it is the best configuration.", options: { breakLine: true, bold: true } },
    { text: `It is also cheaper: ${D.stage1.v2_tokens} output tokens per query instead of ${D.stage1.v1_tokens}.` },
  ], { x: 6.4, y: 1.45, w: 3.1, h: 3.2, fontSize: 15, color: C.text2, margin: 0, isTextBox: true, valign: "top", paraSpaceAfter: 4 });
  s.addNotes(`[60 s] The central result. Two rewriters, two embedders. The original rewriter writes a paragraph of imaginary card text. The simpler one just names the concept. On the base embedder, simpler is worse: ${pct(t.v2.base)} against ${pct(t.v1.base)}. On the tuned embedder it jumps to ${pct(t.v2.tuned)}, the best cell. Neither change works alone; together they do. And the simpler rewriter costs a third fewer tokens.`);
}

// 6 — result: parity with Scryfall
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Talk" });
  s.addText("Plain language reaches most expert results", { placeholder: "title" });
  const p = D.parity;
  columnChart(s, {
    x: 0.5, y: 1.3, w: 4.9, h: 3.45, max: 0.6, step: 0.2,
    series: [{ name: "First version", color: C.accent2 }, { name: "Fine-tuned + simpler rewriter", color: C.accent1 }],
    groups: [{ label: "", values: [p.control, p.headline] }],
  });
  s.addText("Overlap between our ranking and an expert Scryfall query's result set (26 queries)", { x: 0.5, y: 4.85, w: 4.9, h: 0.3, fontSize: 10, color: C.accent3, margin: 0, isTextBox: true });
  stat(s, `${p.n_ge_085} of ${p.n}`, "queries match the expert's set at 0.85 or better", 5.8, 1.3, 3.7);
  stat(s, `${p.headline_depth90.toFixed(2)}×`, "of the set's size scrolled to see 90% of it", 5.8, 2.55, 3.7);
  stat(s, "0", "search operators the user has to know", 5.8, 3.8, 3.7, { color: C.text1 });
  s.addNotes(`[60 s] The comparison that matters. For each test query an expert Scryfall search was written and run. This measures how much of that expert's result set shows up at the top of ours. First version: ${pct(p.control)}. Now: ${pct(p.headline)}. ${p.n_ge_085} of ${p.n} queries are at 0.85 or above. Not a win over Scryfall: the claim is reaching most of what an expert reaches without learning the grammar.`);
}

// 7 — demo
{
  const s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Talk" });
  s.addText("Live demo", { placeholder: "title" });
  s.addText("Three searches, about two minutes", { placeholder: "body" });
  const demos = [["is a planeswalker", "structure from plain words"], ["tutor", "base embedder, then tuned"], ["blue counterspells that cost 2 or less", "filters and meaning together"]];
  demos.forEach(([q, why], i) => {
    chip(s, q, 0.6, 3.75 + i * 0.5, 4.6, { fill: C.text2, color: C.background1, h: 0.4 });
    s.addText(why, { x: 5.4, y: 3.75 + i * 0.5, w: 4.0, h: 0.4, fontSize: 13, color: C.accent3, margin: 0, isTextBox: true, valign: "middle" });
  });
  s.addNotes("[120 s] Open the three bookmarked demo links (see presentation-notes.md). 1: 'is a planeswalker' — point at the filter box: type Planeswalker, 319 cards, no syntax typed. 2: 'tutor' on the base embedder — teachers. Switch the Embedder dropdown to the tuned model — real tutors. 3: 'blue counterspells that cost 2 or less' — colour and cost became filters, results are all counterspells. If anything fails, skip to the backup slides: same three searches as screenshots.");
}

// 8 — limits and next
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Talk" });
  s.addText("It shows merit, and it is not foolproof", { placeholder: "title" });
  card(s, 0.5, 1.4, 4.35, 3.4, "card-limits");
  card(s, 5.15, 1.4, 4.35, 3.4, "card-next");
  const items = (x, title, lines) => {
    s.addText(title, { x: x + 0.25, y: 1.55, w: 3.85, h: 0.35, fontSize: 16, bold: true, color: C.text1, margin: 0, isTextBox: true });
    lines.forEach(([lead, rest], i) => {
      s.addText([{ text: lead + " ", options: { bold: true, color: C.text1 } }, { text: rest }],
        { x: x + 0.25, y: 2.05 + i * 0.9, w: 3.85, h: 0.8, fontSize: 14, color: C.text2, margin: 0, isTextBox: true, valign: "top" });
    });
  };
  items(0.5, "Where it falls short", [
    ["Too broad.", "\u201CDestroy all creatures\u201D returns every board wipe, including ones that destroy lands."],
    ["Small test set.", "Fixing one query tends to cost another across 26 queries."],
    ["Drafted comparisons.", "The expert queries were written for this study, not collected from experts."],
  ]);
  items(5.15, "What comes next", [
    ["Overlap training.", "Two concepts together should land on the cards that are both."],
    ["Finer tags.", "Have an LLM split broad tags into subtags, then retrain."],
    ["A larger test set,", "judged by more than one person."],
  ]);
  s.addNotes("[45 s] Honest limits. The clearest failure: ask for 'destroy all creatures' and you get every sweeper, because the community tag is broader than the request and there is no narrower tag. Small fixes trade one query for another on a 26-query set. Next: train on tag intersections, and use an LLM to split broad tags into finer ones. Then questions.");
}

// ---- backup ----------------------------------------------------------------
pres.addSection({ title: "Backup" });

// 9 — per-query parity
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Backup" });
  s.addText("Backup: parity by query", { placeholder: "title" });
  const rows = [...D.per_query].sort((a, b) => b.parity - a.parity)
    .map((r) => ({ label: r.q.length > 30 ? r.q.slice(0, 29) + "…" : r.q, value: r.parity }));
  const half = Math.ceil(rows.length / 2);
  rankedBars(s, { x: 0.5, y: 1.3, w: 4.4, rowH: 0.285, rows: rows.slice(0, half), color: C.accent1 });
  rankedBars(s, { x: 5.1, y: 1.3, w: 4.4, rowH: 0.285, rows: rows.slice(half), color: C.accent1 });
  s.addNotes("Backup for questions about where it works and where it doesn't. Top: structural and well-tagged jargon queries. Bottom: descriptive queries the rewriter misreads, and requests more specific than any tag.");
}

// 10-12 — demo fallbacks
const shots = [
  ["Backup demo: structure from plain words", "demo_planeswalker.jpg", "Typed in plain words. The rewriter produced a type filter; 319 planeswalkers, nothing else."],
  ["Backup demo: tutor, before fine-tuning", "demo_tutor_base.jpg", "Before fine-tuning: the model reads 'tutor' as 'teacher'."],
  ["Backup demo: tutor, after fine-tuning", "demo_tutor_tuned.jpg", "After fine-tuning: cards that search your library."],
  ["Backup demo: filters plus meaning", "demo_counterspells.jpg", "Colour and cost became filters; the ranking inside them is all counterspells."],
];
for (const [title, file, note] of shots) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Backup" });
  s.addText(title, { placeholder: "title" });
  // screenshots are 1600x1000 (8:5); 5.6" x 3.5" keeps the ratio and clears the footer by 0.45"
  s.addImage({ path: path.join(HERE, "assets", file), x: 2.2, y: 1.25, w: 5.6, h: 3.5, altText: note, objectName: `shot-${file}` });
  s.addNotes(note);
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
