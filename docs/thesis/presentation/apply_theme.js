// Write the deck's colour scheme into ppt/theme/theme1.xml.
// pptxgenjs can reference scheme colours (text1, accent1, ...) but cannot set
// them, so until this runs they resolve to Office's stock palette.
const fs = require("fs");

const SLOTS = ["dk1", "lt1", "dk2", "lt2", "accent1", "accent2", "accent3", "accent4", "accent5", "accent6", "hlink", "folHlink"];
const xmlEscape = (s) => String(s).replace(/[<>&"]/g, (c) => ({ "<": "&lt;", ">": "&gt;", "&": "&amp;", '"': "&quot;" })[c]);

async function applyTheme(deckPath, theme) {
  // jszip ships as a dependency of pptxgenjs; resolve it from there.
  const JSZip = require(require.resolve("jszip", { paths: [require.resolve("pptxgenjs")] }));
  const zip = await JSZip.loadAsync(fs.readFileSync(deckPath));
  const part = "ppt/theme/theme1.xml";
  const xml = await zip.file(part).async("string");
  for (const k of SLOTS) {
    if (!/^[0-9A-Fa-f]{6}$/.test(theme.colors[k] || "")) throw new Error(`theme.colors.${k} must be a 6-digit hex colour`);
  }
  const scheme =
    `<a:clrScheme name="${xmlEscape(theme.name)}">` +
    SLOTS.map((k) => `<a:${k}><a:srgbClr val="${theme.colors[k].toUpperCase()}"/></a:${k}>`).join("") +
    "</a:clrScheme>";
  const out = xml.replace(/<a:clrScheme\b[\s\S]*?<\/a:clrScheme>/, () => scheme);
  if (!out.includes(scheme)) throw new Error(`${part} has no <a:clrScheme> to replace`);
  zip.file(part, out);
  fs.writeFileSync(deckPath, await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
}

module.exports = { applyTheme };
