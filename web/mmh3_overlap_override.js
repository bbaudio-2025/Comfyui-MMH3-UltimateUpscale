// Front-end inline editor for the "MMH3 Overlap Fade Override" node.
//
// The server node exposes only a hidden JSON string (`override_data`) and a
// `tile_count` widget. This extension replaces the raw JSON editing with a
// visual, numbered tile map plus one row of native number widgets per tile and
// aggregates them back into `override_data`, so the server receives the same
// { index: { overlap_width, overlap_height, fade_width, fade_height } } map.
import { app } from "../../scripts/app.js";

const TARGET = "MMH3OverlapFadeOverride";
const DATA_WIDGET = "override_data";
const COUNT_WIDGET = "tile_count";
const FIELDS = ["overlap_width", "overlap_height", "fade_width", "fade_height"];
const FIELDS_LABEL = ["overlap W", "overlap H", "fade W", "fade H"];
// Selectable overlap/fade values (pixels). "default" is the unset state (falls
// back to the main node's global value); "0" is a real "no overlap/fade on this
// dimension" value; the rest are 32-multiples so stepping stays on integers.
const OPTIONS = ["default", "0", "32", "64", "96", "128", "160", "192", "224",
                 "256", "288", "320", "384", "448", "512", "640", "768", "896",
                 "1024", "1280", "1536", "1792", "2048"];
const MAX_TILES = 12;
const MAP_H = 46;
const TAG = "[MMH3-UltimateUpscale]";

function parseData(widget) {
    try {
        const v = JSON.parse(widget.value || "[]");
        return Array.isArray(v) ? v : [];
    } catch {
        return [];
    }
}

app.registerExtension({
    name: "MMH3UltimateUpscale.OverlapFadeOverrideGrid",
    nodeCreated(node) {
        const cls = node.comfyClass || node.type;
        if (cls !== TARGET) return;

        const dataWidget = node.widgets && node.widgets.find((w) => w.name === DATA_WIDGET);
        const countWidget = node.widgets && node.widgets.find((w) => w.name === COUNT_WIDGET);
        if (!dataWidget || !countWidget) {
            console.error(TAG, "missing widgets on", cls);
            return;
        }
        dataWidget.hidden = true; // raw JSON is edited via the grid below

        // Decorative map: anchor tile 0 centered, tiles alternating +1.. on
        // right, -1.. on left (the main node's "alternating" plan). A tile with
        // a complete override is green; partial is amber; empty is grey.
        const MAP_BOX = 44;
        const MAP_GAP = 8;
        const mapWidget = node.addWidget("custom", "mmh3_tile_map", null, null, {
            widget: {
                computeSize(w) {
                    return [w, MAP_H];
                },
                draw(ctx, node, width, y) {
                    const entries = parseData(dataWidget);
                    const flags = flagsFor(entries);
                    const n = Math.max(countWidget.value || 1, 1);
                    const cx = width / 2;
                    const top = y + 4;
                    const h = MAP_BOX - 10;
                    ctx.save();
                    ctx.textAlign = "center";
                    ctx.textBaseline = "middle";
                    ctx.font = "11px sans-serif";
                    const box = (idx, color) => {
                        ctx.fillStyle = color;
                        ctx.fillRect(cx + idx * (MAP_BOX + MAP_GAP) - MAP_BOX / 2, top, MAP_BOX, h);
                        ctx.fillStyle = "#fff";
                        ctx.fillText(String(idx), cx + idx * (MAP_BOX + MAP_GAP), top + h / 2);
                    };
                    box(0, "#3a6ea5");
                    for (let idx = 1; idx <= n; idx++) {
                        const side = idx % 2 === 1 ? 1 : -1; // odd -> right, even -> left
                        const k = Math.floor((idx + 1) / 2) * side;
                        const fl = flags[idx];
                        const color = fl && fl.complete ? "#3f9142" : fl && fl.any ? "#b3872f" : "#555";
                        box(k, color);
                    }
                    ctx.restore();
                },
            },
        });
        mapWidget.computeSize = (w) => [w, MAP_H];

        const byTile = {}; // tile index -> { field -> widget }

        function flagsFor(entries) {
            const flags = {};
            for (const e of entries) {
                if (!e || typeof e.index !== "number" || e.index < 1) continue;
                const done = {};
                for (const f of FIELDS) done[f] = typeof e[f] === "number" && e[f] > 0;
                const any = Object.values(done).some(Boolean);
                flags[e.index] = { any, complete: any && FIELDS.every((f) => done[f]) };
            }
            return flags;
        }

        function ensureTiles(n) {
            for (let idx = 1; idx <= n; idx++) {
                if (byTile[idx]) continue;
                const row = {};
                for (const f of FIELDS) {
                    const w = node.addWidget("combo", `tile ${idx} ${FIELDS_LABEL[FIELDS.indexOf(f)]}`, "default",
                        refresh, { values: OPTIONS });
                    w.computeSize = (x) => [x, 20];
                    w.options = Object.assign(w.options || {}, { values: OPTIONS });
                    w.values = OPTIONS;
                    w.tileIndex = idx;
                    w.tileField = f;
                    row[f] = w;
                }
                byTile[idx] = row;
            }
        }

        function labelToValue(label) {
            if (label === "default") return null;
            const n = Number(label);
            return Number.isFinite(n) ? n : null;
        }
        function valueToLabel(v) {
            if (v === null || v === undefined) return "default";
            const n = Math.round(Number(v) / 32) * 32;
            return (n === 0 || OPTIONS.includes(String(n))) ? String(n) : "default";
        }

        function clampInt(v) {
            const x = Math.round(Number(v));
            if (Number.isNaN(x)) return 1;
            return Math.min(MAX_TILES, Math.max(1, x));
        }

        function refresh() {
            const n = clampInt(countWidget.value);
            countWidget.value = n;
            ensureTiles(n);

            // Hide tiles above count (kept values persist).
            for (const idx of Object.keys(byTile)) {
                const visible = Number(idx) <= n;
                for (const f of FIELDS) byTile[idx][f].hidden = !visible;
            }

            // Aggregate visible tiles into the override map. A field set to
            // "default" is emitted as null so the backend falls back to the main
            // node's global value; an explicit "0" is kept as a real value (no
            // overlap/fade on that dimension).
            const payload = [];
            for (let idx = 1; idx <= n; idx++) {
                const row = byTile[idx];
                const entry = { index: idx };
                for (const f of FIELDS) {
                    const v = labelToValue(row[f].value);
                    entry[f] = v === null ? null : v;
                }
                if (FIELDS.some((f) => entry[f] !== null)) payload.push(entry);
            }
            dataWidget.value = JSON.stringify(payload);

            if (node.graph) {
                node.onResize?.(node.size);
                app.graph.setDirtyCanvas(true, true);
            }
        }

        function restore() {
            const entries = parseData(dataWidget);
            let maxIdx = 0;
            for (const e of entries) {
                if (e && typeof e.index === "number" && e.index >= 1) {
                    maxIdx = Math.max(maxIdx, e.index | 0);
                    ensureTiles(e.index | 0);
                    const row = byTile[e.index | 0];
                    for (const f of FIELDS) row[f].value = valueToLabel(e[f]);
                }
            }
            if (countWidget.value < maxIdx) countWidget.value = maxIdx;
            if (countWidget.value < 1) countWidget.value = 1;
            refresh();
        }

        const origCallback = countWidget.callback;
        countWidget.callback = function () {
            const res = origCallback ? origCallback.apply(this, arguments) : undefined;
            refresh();
            return res;
        };

        restore();

        const origConfigure = node.onConfigure;
        node.onConfigure = function () {
            const res = origConfigure ? origConfigure.apply(this, arguments) : undefined;
            restore();
            return res;
        };
    },
});
