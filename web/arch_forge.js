import { app } from "../../scripts/app.js";

const GROUPS = [
  ["Models", 
	[
	  "checkpoint_name",
	  "lora_name_1", "lora_strength_1_model", "lora_strength_1_clip",
      "lora_name_2", "lora_strength_2_model", "lora_strength_2_clip",
      "lora_name_3", "lora_strength_3_model", "lora_strength_3_clip"
	]
  ],
  ["Prompts", ["positive_prompt", "negative_prompt"]],
  [
    "Canny",
    [
      "canny_controlnet", "canny_strength", "canny_start", "canny_end",
      "canny_enable_threshold", "canny_threshold_low", "canny_threshold_high"
    ]
  ],
  [
    "Depth",
    [
      "depth_controlnet", "depth_strength", "depth_start", "depth_end",
      "depth_resolution",
      "depth_midas_a", "depth_midas_bg_threshold"
    ]
  ],
  [
    "Persp wrap",
    [
      "enable_nuke",
      "nuke_to1_x", "nuke_to1_y",
      "nuke_to2_x", "nuke_to2_y",
      "nuke_to3_x", "nuke_to3_y",
      "nuke_to4_x", "nuke_to4_y",
      "nuke_filter",
      "corner_pin_scope"
    ]
  ],
  [
    "Sampling",
    [
      "seed", "steps", "cfg", "sampler_name", "scheduler",
      "start_at_step", "end_at_step", "control_after_generate"
    ]
  ],
  [
    "Ambient Occlusion",
    [
      "enable_ssao",
      "ssao_resolution", "ssao_strength", "ssao_radius", "ssao_blur",
      "ssao_specular_threshold", "ssao_enable_specular", "ssao_tile_size",
      "ssao_blend_factor", "ssao_blend_mode"
    ]
  ],
  [
    "Sharpen",
    [
      "enable_lucy", "lucy_iterations", "lucy_kernel_size",
      "enable_final_sharpen",
      "sharpen_radius", "sharpen_sigma", "sharpen_alpha"
    ]
  ],
  [
    "Persp Straighen",
    ["enable_perspective_correction", "perspective_strength", "perspective_auto_crop"]
  ],
  ["Upscale", ["upscale_model"]]
];

app.registerExtension({
  name: "ArchForge.UI",
  nodeCreated(node) {
    if (node.comfyClass !== "ArchForgeNode" || node._archForgeUI) return;
    node._archForgeUI = true;

    const snapshots = new WeakMap();
    const open = new Set();

    const el = document.createElement("div");
    el.style.cssText =
      "display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px;padding:10px;box-sizing:border-box;width:100%;background:#0f172a;border-radius:8px;";

    const panel = node.addDOMWidget("arch_forge_ui", "af", el, {
      serialize: false,
      getMinHeight: () => 220,
      getMaxHeight: () => 220,
    });

    function hide(w) {
      if (!snapshots.has(w)) {
        const saved = {};
        for (const key of ["type", "hidden", "computeSize", "draw", "mouse"]) {
          saved[key] = {
            own: Object.prototype.hasOwnProperty.call(w, key),
            value: w[key],
          };
        }
        snapshots.set(w, saved);
      }
      w.hidden = true;
      w.type = "hidden";
      w.computeSize = () => [0, -4];
      w.draw = () => {};
      w.mouse = () => false;
      if (w.element) w.element.style.display = "none";
    }

    function show(w) {
      const saved = snapshots.get(w);
      if (!saved) return;
      for (const [key, entry] of Object.entries(saved)) {
        if (entry.own) w[key] = entry.value;
        else delete w[key];
      }
      if (w.element) w.element.style.removeProperty("display");
    }

    const buttons = GROUPS.map(([label, names], i) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.textContent = label;
      btn.style.cssText =
        "padding:8px 5px;background:#1e293b;color:#e2e8f0;border:1px solid #334155;border-radius:6px;cursor:pointer;font:12px system-ui;";
      btn.addEventListener("pointerdown", (event) => event.stopPropagation());
      btn.addEventListener("click", (event) => {
        event.stopPropagation();
        open.has(i) ? open.delete(i) : open.add(i);
        sync();
      });
      el.appendChild(btn);
      return btn;
    });

    const attribution = document.createElement("a");
    attribution.textContent = "by jeeltcraft";
    attribution.href = "https://www.jeeltcraft.com/";
    attribution.target = "_blank";
    attribution.rel = "noopener noreferrer";
    attribution.style.cssText =
      "grid-column:1/-1;text-align:center;color:#94a3b8;font:10px system-ui;text-decoration:none;";
    el.appendChild(attribution);

    // FORCE INITIAL SIZE - prevents huge load height (worked before)
    setTimeout(() => {
      node.setSize([Math.max(node.size[0], 420), 280]);
    }, 10);

    function sync() {
      for (const w of node.widgets || []) {
        if (w === panel) continue;
        const i = GROUPS.findIndex(([, names]) => names.includes(w.name));
        if (i < 0) continue;
        open.has(i) ? show(w) : hide(w);
      }

      buttons.forEach((btn, i) => {
        btn.setAttribute("aria-expanded", String(open.has(i)));
        btn.style.borderColor = open.has(i) ? "#22c55e" : "#334155";
      });

      requestAnimationFrame(() => {
        if (!node.graph) return;
        const size = node.computeSize();
        node.setSize([Math.max(node.size[0], 420), size[1]]);
        node.setDirtyCanvas?.(true, true);
      });
    }

    const configure = node.onConfigure;
    node.onConfigure = function (...args) {
      const result = configure?.apply(this, args);
      sync();
      return result;
    };

    sync();
  },
});