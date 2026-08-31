// Front-end helper for MMH3SpatialInpaintParams / MMH3FunControlnetParams:
// auto-show/hide the percentage vs step input groups depending on the
// start_end_set combo. Hidden widgets keep their stored values and serialize
// as usual - the server side resolves the range from whichever mode is
// selected, this is purely cosmetic.
import { app } from "../../scripts/app.js";

const TARGET_NODES = new Set(["MMH3SpatialInpaintParams", "MMH3FunControlnetParams"]);
const MODE_WIDGET = "start_end_set";
const PERCENT_WIDGETS = ["start_percent", "end_percent"];
const STEP_WIDGETS = ["start_step", "end_step"];

function findWidget(node, name) {
    return node.widgets ? node.widgets.find((w) => w.name === name) : null;
}

function setWidgetVisible(node, name, visible) {
    const w = findWidget(node, name);
    if (!w) return;
    if ("hidden" in w) {
        // modern ComfyUI frontend: proper hidden flag, excluded from layout
        w.hidden = !visible;
        return;
    }
    // fallback for older frontends: park the widget as "converted"
    if (visible) {
        if (w.type === "converted-widget" && w.origType) {
            w.type = w.origType;
            if (w.origComputeOptions) w.computeOptions = w.origComputeOptions;
        }
    } else if (w.type !== "converted-widget") {
        w.origType = w.type;
        w.origComputeOptions = w.computeOptions;
        w.type = "converted-widget";
        w.computeOptions = () => {};
    }
}

function applyMode(node) {
    const mode = findWidget(node, MODE_WIDGET);
    if (!mode) return false;
    const stepMode = mode.value === "step";
    for (const name of PERCENT_WIDGETS) setWidgetVisible(node, name, !stepMode);
    for (const name of STEP_WIDGETS) setWidgetVisible(node, name, stepMode);
    if (node.onResize) node.onResize(node.size);
    app.graph.setDirtyCanvas(true, true);
    return true;
}

app.registerExtension({
    name: "MMH3UltimateUpscale.StartEndSet",
    nodeCreated(node) {
        const cls = node.comfyClass || node.type;
        if (!TARGET_NODES.has(cls)) return;
        console.info("[MMH3-UltimateUpscale] start_end_set UI hook installed on", cls);

        const mode = findWidget(node, MODE_WIDGET);
        if (mode) {
            const origCallback = mode.callback;
            mode.callback = function () {
                const res = origCallback ? origCallback.apply(this, arguments) : undefined;
                applyMode(node);
                return res;
            };
        }
        const origConfigure = node.onConfigure;
        node.onConfigure = function () {
            const res = origConfigure ? origConfigure.apply(this, arguments) : undefined;
            setTimeout(() => applyMode(node), 0);
            return res;
        };
        setTimeout(() => applyMode(node), 0);
    },
});
