#!/usr/bin/env node
"use strict";

/**
 * Render static SVG previews for the Epiplexity proof/declaration UMAP embedding.
 *
 * Inputs:
 *   - artifacts/visuals/epiplexity_proofs.json
 *
 * Outputs:
 *   - artifacts/visuals/epiplexity_2d_preview.svg
 *   - artifacts/visuals/epiplexity_3d_preview.svg
 *   - artifacts/visuals/epiplexity_3d_preview_animated.svg
 */

const fs = require("fs");
const path = require("path");

function fail(msg) {
  console.error(`E: ${msg}`);
  process.exit(1);
}

function hash32(s) {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

function colorForFamily(family) {
  // Custom colors for Epiplexity families
  const colors = {
    "Core": { fill: "hsl(210 70% 58%)", stroke: "hsl(210 70% 32%)" },
    "MDL": { fill: "hsl(280 70% 58%)", stroke: "hsl(280 70% 32%)" },
    "Information": { fill: "hsl(120 70% 58%)", stroke: "hsl(120 70% 32%)" },
    "Bounds": { fill: "hsl(45 70% 58%)", stroke: "hsl(45 70% 32%)" },
    "Conditional": { fill: "hsl(180 70% 58%)", stroke: "hsl(180 70% 32%)" },
    "Programs": { fill: "hsl(340 70% 58%)", stroke: "hsl(340 70% 32%)" },
    "Emergence": { fill: "hsl(30 90% 55%)", stroke: "hsl(30 90% 35%)" },
    "Prelude": { fill: "hsl(0 0% 60%)", stroke: "hsl(0 0% 40%)" },
    "Crypto/Axioms": { fill: "hsl(0 70% 58%)", stroke: "hsl(0 70% 32%)" },
    "Crypto/CSPRNG": { fill: "hsl(60 70% 50%)", stroke: "hsl(60 70% 30%)" },
    "Crypto/Factorization": { fill: "hsl(300 70% 58%)", stroke: "hsl(300 70% 32%)" },
    "Crypto/HeavySet": { fill: "hsl(160 70% 50%)", stroke: "hsl(160 70% 30%)" },
    "Crypto/PRF": { fill: "hsl(240 70% 58%)", stroke: "hsl(240 70% 32%)" },
  };
  if (colors[family]) return colors[family];
  const h = hash32(family) % 360;
  return { fill: `hsl(${h} 70% 58%)`, stroke: `hsl(${h} 70% 32%)` };
}

function esc(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function svgDoc({ width, height, background, body }) {
  return (
    `<?xml version="1.0" encoding="utf-8"?>\n` +
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" ` +
    `viewBox="0 0 ${width} ${height}" role="img" aria-label="UMAP preview">\n` +
    `<rect x="0" y="0" width="${width}" height="${height}" fill="${background}"/>\n` +
    `${body}\n` +
    `</svg>\n`
  );
}

function renderLegend({ families, counts, x, y, lineH }) {
  const entries = [...families].sort(
    (a, b) => (counts.get(b) ?? 0) - (counts.get(a) ?? 0)
  );
  let out = "";
  let cy = y;
  for (const fam of entries) {
    const n = counts.get(fam) ?? 0;
    const c = colorForFamily(fam);
    out += `<rect x="${x}" y="${cy}" width="10" height="10" fill="${c.fill}" stroke="${c.stroke}" stroke-width="1"/>\n`;
    out += `<text x="${x + 16}" y="${cy + 9}" fill="#e6eef7" font-size="12" font-family="ui-sans-serif,system-ui,Segoe UI,Roboto,Helvetica,Arial">${esc(
      fam
    )} <tspan fill="#b8c7d9">(${n})</tspan></text>\n`;
    cy += lineH;
  }
  return out;
}

function map2dPoints({ items, getPos, margin, plotW, plotH }) {
  function toCanvas(p) {
    const x = margin + p.x * plotW;
    const y = margin + (1 - p.y) * plotH;
    return { x, y };
  }
  return items.map((it) => toCanvas(getPos(it)));
}

function render2d({ data, outPath }) {
  const items = data.items ?? [];
  const edges = data.edges ?? [];

  const width = 1500;
  const height = 900;
  const background = "#0b0f14";

  const margin = 50;
  const legendW = 310;
  const plotW = width - margin * 2 - legendW;
  const plotH = height - margin * 2;
  const legendX = margin + plotW + 30;

  const counts = new Map();
  for (const it of items) {
    const fam = it.family ?? "Other";
    counts.set(fam, (counts.get(fam) ?? 0) + 1);
  }
  const families = [...counts.keys()];

  const pts = map2dPoints({
    items,
    getPos: (it) => it.pos,
    margin,
    plotW,
    plotH,
  });

  let body = "";
  const title = "UMAP 2D — Epiplexity Proof/Declaration Map";
  const subtitle = "Points: declarations • Colors: module family • Edges: intra-family similarity";
  body += `<text x="${margin}" y="${margin - 18}" fill="#ffffff" font-size="20" font-family="ui-sans-serif,system-ui,Segoe UI,Roboto,Helvetica,Arial">${esc(title)}</text>\n`;
  body += `<text x="${margin}" y="${margin - 2}" fill="#b8c7d9" font-size="12" font-family="ui-sans-serif,system-ui,Segoe UI,Roboto,Helvetica,Arial">${esc(subtitle)}</text>\n`;

  body += `<rect x="${margin}" y="${margin}" width="${plotW}" height="${plotH}" fill="#0f1721" stroke="#1c2a3a" stroke-width="1"/>\n`;

  // Edges
  for (const [i, j] of edges) {
    const a = pts[i];
    const b = pts[j];
    if (!a || !b) continue;
    body += `<line x1="${a.x.toFixed(2)}" y1="${a.y.toFixed(2)}" x2="${b.x.toFixed(2)}" y2="${b.y.toFixed(2)}" stroke="#3b4b5d" stroke-opacity="0.25" stroke-width="1"/>\n`;
  }

  // Nodes
  for (let idx = 0; idx < items.length; idx++) {
    const it = items[idx];
    const p = pts[idx];
    if (!p) continue;
    const fam = it.family ?? "Other";
    const c = colorForFamily(fam);
    body += `<circle cx="${p.x.toFixed(2)}" cy="${p.y.toFixed(2)}" r="5" fill="${c.fill}" stroke="#0b0f14" stroke-width="1">\n`;
    body += `  <title>${esc(it.name ?? it.id ?? "item")} (${it.kind})</title>\n`;
    body += `</circle>\n`;
  }

  // Legend
  body += `<text x="${legendX}" y="${margin}" fill="#ffffff" font-size="16" font-family="ui-sans-serif,system-ui,Segoe UI,Roboto,Helvetica,Arial">Legend</text>\n`;
  body += renderLegend({ families, counts, x: legendX, y: margin + 14, lineH: 18 });

  const svg = svgDoc({ width, height, background, body });
  fs.writeFileSync(outPath, svg, "utf8");
}

function rotateX([x, y, z], a) {
  const ca = Math.cos(a);
  const sa = Math.sin(a);
  return [x, y * ca - z * sa, y * sa + z * ca];
}

function rotateY([x, y, z], a) {
  const ca = Math.cos(a);
  const sa = Math.sin(a);
  return [x * ca + z * sa, y, -x * sa + z * ca];
}

function render3dAnimated({ data, outPath }) {
  const items = data.items ?? [];

  const width = 1500;
  const height = 900;
  const background = "#0b0f14";

  const margin = 50;
  const plotW = width - margin * 2;
  const plotH = height - margin * 2;

  const pitch = 0.48;
  const cameraDist = 3.0;

  const frames = 72;
  const durSec = 14;

  const p3s = items.map((it) => it.pos3).filter(Boolean);
  if (!p3s.length) {
    fail(`cannot render animated 3D preview: no pos3 coordinates present`);
  }

  let minx = 1e9, maxx = -1e9, miny = 1e9, maxy = -1e9, minz = 1e9, maxz = -1e9;
  for (const p of p3s) {
    minx = Math.min(minx, p.x); maxx = Math.max(maxx, p.x);
    miny = Math.min(miny, p.y); maxy = Math.max(maxy, p.y);
    minz = Math.min(minz, p.z); maxz = Math.max(maxz, p.z);
  }
  const cx = (minx + maxx) / 2;
  const cy = (miny + maxy) / 2;
  const cz = (minz + maxz) / 2;
  const scale = 2 / Math.max(1e-6, maxx - minx, maxy - miny, maxz - minz);

  const xyz = items.map((it) => {
    const p = it.pos3;
    if (!p) return null;
    return { x: (p.x - cx) * scale, y: (p.y - cy) * scale, z: (p.z - cz) * scale };
  });

  function projectAtYaw(yaw) {
    return xyz.map((p) => {
      if (!p) return null;
      let v = [p.x, p.y, p.z];
      v = rotateY(v, yaw);
      v = rotateX(v, pitch);
      const [rx, ry, rz] = v;
      const s = cameraDist / (cameraDist - rz);
      return { x: rx * s, y: ry * s, z: rz };
    });
  }

  const projFrames = [];
  for (let t = 0; t <= frames; t++) {
    const yaw = (2 * Math.PI * t) / frames;
    projFrames.push(projectAtYaw(yaw));
  }

  let gx0 = 1e9, gx1 = -1e9, gy0 = 1e9, gy1 = -1e9;
  for (const f of projFrames) {
    for (const p of f) {
      if (!p) continue;
      gx0 = Math.min(gx0, p.x); gx1 = Math.max(gx1, p.x);
      gy0 = Math.min(gy0, p.y); gy1 = Math.max(gy1, p.y);
    }
  }
  const invW = 1 / Math.max(1e-6, gx1 - gx0);
  const invH = 1 / Math.max(1e-6, gy1 - gy0);

  const normFrames = projFrames.map((f) =>
    f.map((p) => {
      if (!p) return null;
      return { x: (p.x - gx0) * invW, y: (p.y - gy0) * invH };
    })
  );

  const counts = new Map();
  for (const it of items) {
    const fam = it.family ?? "Other";
    counts.set(fam, (counts.get(fam) ?? 0) + 1);
  }
  const families = [...counts.keys()];

  let body = "";
  const title = "UMAP 3D — Epiplexity Proof Map (animated)";
  const subtitle = "Rotation preview of 3D embedding";
  body += `<text x="${margin}" y="${margin - 18}" fill="#ffffff" font-size="20" font-family="ui-sans-serif,system-ui,Segoe UI,Roboto,Helvetica,Arial">${esc(title)}</text>\n`;
  body += `<text x="${margin}" y="${margin - 2}" fill="#b8c7d9" font-size="12" font-family="ui-sans-serif,system-ui,Segoe UI,Roboto,Helvetica,Arial">${esc(subtitle)}</text>\n`;
  body += `<rect x="${margin}" y="${margin}" width="${plotW}" height="${plotH}" fill="#0f1721" stroke="#1c2a3a" stroke-width="1"/>\n`;

  const legendX = margin + 18;
  const legendY = margin + 18;
  body += renderLegend({ families, counts, x: legendX, y: legendY, lineH: 18 });

  function toCanvas2(p) {
    const x = margin + p.x * plotW;
    const y = margin + (1 - p.y) * plotH;
    return { x, y };
  }

  for (let idx = 0; idx < items.length; idx++) {
    const it = items[idx];
    const fam = it.family ?? "Other";
    const c = colorForFamily(fam);
    const valuesX = [];
    const valuesY = [];
    for (const f of normFrames) {
      const p = f[idx];
      if (!p) continue;
      const q = toCanvas2(p);
      valuesX.push(q.x.toFixed(2));
      valuesY.push(q.y.toFixed(2));
    }
    if (!valuesX.length) continue;
    body += `<circle r="5" fill="${c.fill}" stroke="#0b0f14" stroke-width="1">\n`;
    body += `  <title>${esc(it.name ?? it.id ?? "item")}</title>\n`;
    body += `  <animate attributeName="cx" dur="${durSec}s" repeatCount="indefinite" values="${valuesX.join(";")}"/>\n`;
    body += `  <animate attributeName="cy" dur="${durSec}s" repeatCount="indefinite" values="${valuesY.join(";")}"/>\n`;
    body += `</circle>\n`;
  }

  const svg = svgDoc({ width, height, background, body });
  fs.writeFileSync(outPath, svg, "utf8");
}

function render3dStatic({ data, outPath }) {
  const items = data.items ?? [];

  const width = 1500;
  const height = 900;
  const background = "#0b0f14";

  const margin = 50;
  const legendW = 310;
  const plotW = width - margin * 2 - legendW;
  const plotH = height - margin * 2;
  const legendX = margin + plotW + 30;

  const counts = new Map();
  for (const it of items) {
    const fam = it.family ?? "Other";
    counts.set(fam, (counts.get(fam) ?? 0) + 1);
  }
  const families = [...counts.keys()];

  const pitch = 0.48;
  const yaw = 0.72;
  const cameraDist = 3.0;

  const p3s = items.map((it) => it.pos3).filter(Boolean);
  if (!p3s.length) {
    fail(`cannot render static 3D preview: no pos3 coordinates present`);
  }

  let minx = 1e9, maxx = -1e9, miny = 1e9, maxy = -1e9, minz = 1e9, maxz = -1e9;
  for (const p of p3s) {
    minx = Math.min(minx, p.x); maxx = Math.max(maxx, p.x);
    miny = Math.min(miny, p.y); maxy = Math.max(maxy, p.y);
    minz = Math.min(minz, p.z); maxz = Math.max(maxz, p.z);
  }
  const cx = (minx + maxx) / 2;
  const cy = (miny + maxy) / 2;
  const cz = (minz + maxz) / 2;
  const scale = 2 / Math.max(1e-6, maxx - minx, maxy - miny, maxz - minz);

  const proj = items.map((it) => {
    const p = it.pos3;
    if (!p) return null;
    let v = [(p.x - cx) * scale, (p.y - cy) * scale, (p.z - cz) * scale];
    v = rotateY(v, yaw);
    v = rotateX(v, pitch);
    const [rx, ry, rz] = v;
    const s = cameraDist / (cameraDist - rz);
    return { x: rx * s, y: ry * s };
  });

  let gx0 = 1e9, gx1 = -1e9, gy0 = 1e9, gy1 = -1e9;
  for (const p of proj) {
    if (!p) continue;
    gx0 = Math.min(gx0, p.x); gx1 = Math.max(gx1, p.x);
    gy0 = Math.min(gy0, p.y); gy1 = Math.max(gy1, p.y);
  }
  const invW = 1 / Math.max(1e-6, gx1 - gx0);
  const invH = 1 / Math.max(1e-6, gy1 - gy0);

  const pts = proj.map((p) => {
    if (!p) return null;
    return { x: (p.x - gx0) * invW, y: (p.y - gy0) * invH };
  });

  function toCanvas2(p) {
    const x = margin + p.x * plotW;
    const y = margin + (1 - p.y) * plotH;
    return { x, y };
  }

  let body = "";
  const title = "UMAP 3D — Epiplexity Proof Map (static)";
  const subtitle = "Static projection of 3D embedding";
  body += `<text x="${margin}" y="${margin - 18}" fill="#ffffff" font-size="20" font-family="ui-sans-serif,system-ui,Segoe UI,Roboto,Helvetica,Arial">${esc(title)}</text>\n`;
  body += `<text x="${margin}" y="${margin - 2}" fill="#b8c7d9" font-size="12" font-family="ui-sans-serif,system-ui,Segoe UI,Roboto,Helvetica,Arial">${esc(subtitle)}</text>\n`;
  body += `<rect x="${margin}" y="${margin}" width="${plotW}" height="${plotH}" fill="#0f1721" stroke="#1c2a3a" stroke-width="1"/>\n`;

  for (let idx = 0; idx < items.length; idx++) {
    const it = items[idx];
    const p = pts[idx];
    if (!p) continue;
    const q = toCanvas2(p);
    const fam = it.family ?? "Other";
    const c = colorForFamily(fam);
    body += `<circle cx="${q.x.toFixed(2)}" cy="${q.y.toFixed(2)}" r="5" fill="${c.fill}" stroke="#0b0f14" stroke-width="1">\n`;
    body += `  <title>${esc(it.name ?? it.id ?? "item")}</title>\n`;
    body += `</circle>\n`;
  }

  body += `<text x="${legendX}" y="${margin}" fill="#ffffff" font-size="16" font-family="ui-sans-serif,system-ui,Segoe UI,Roboto,Helvetica,Arial">Legend</text>\n`;
  body += renderLegend({ families, counts, x: legendX, y: margin + 14, lineH: 18 });

  const svg = svgDoc({ width, height, background, body });
  fs.writeFileSync(outPath, svg, "utf8");
}

function main() {
  const root = path.resolve(__dirname, "..");
  const inFile = path.join(root, "artifacts", "visuals", "epiplexity_proofs.json");
  if (!fs.existsSync(inFile)) {
    fail(`missing input: ${inFile}`);
  }
  const data = JSON.parse(fs.readFileSync(inFile, "utf8"));

  const out2d = path.join(root, "artifacts", "visuals", "epiplexity_2d_preview.svg");
  const out3d = path.join(root, "artifacts", "visuals", "epiplexity_3d_preview.svg");
  const out3dAnimated = path.join(root, "artifacts", "visuals", "epiplexity_3d_preview_animated.svg");

  render2d({ data, outPath: out2d });
  render3dStatic({ data, outPath: out3d });
  render3dAnimated({ data, outPath: out3dAnimated });

  console.log(`[render_umap_previews] wrote:\n- ${out2d}\n- ${out3d}\n- ${out3dAnimated}`);
}

main();
