import { app } from "../../scripts/app.js";

// Groups shown as buttons. "Persp wrap" was removed for both workflows.
const GROUPS = [
  ["Models",
    [
      "checkpoint_name", "vae_name",
      "moss_unet_name", "moss_clip_name", "moss_vae_name",
      "lora_name_1", "lora_strength_1_model", "lora_strength_1_clip",
      "lora_name_2", "lora_strength_2_model", "lora_strength_2_clip",
      "lora_name_3", "lora_strength_3_model", "lora_strength_3_clip",
      "lora_name_4", "lora_strength_4_model", "lora_strength_4_clip"
    ]
  ],
  ["Prompts", ["positive_prompt", "negative_prompt"]],
  [
    "Canny",
    [
      "canny_controlnet", "canny_strength", "canny_start", "canny_end",
      "canny_enable_threshold", "canny_threshold_low", "canny_threshold_high",
      "keep_artifacts", "moss_canny_lora",
      "moss_canny_low", "moss_canny_high", "moss_canny_resolution"
    ]
  ],
  [
    "Depth",
    [
      "depth_controlnet", "moss_depth_lora",
      "depth_strength", "depth_start", "depth_end",
      "depth_resolution",
      "depth_midas_a", "depth_midas_bg_threshold"
    ]
  ],
  [
    "Sampling",
    [
      "seed", "steps", "cfg", "sampler_name", "scheduler",
      "start_at_step", "end_at_step", "control_after_generate",
      "moss_leftover_noise_handoff"
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

// Widgets that only apply to one of the two workflows.
const SDXL_ONLY = new Set([
  "checkpoint_name", "vae_name",
  "canny_controlnet", "canny_start", "canny_end",
  "canny_enable_threshold", "canny_threshold_low", "canny_threshold_high",
  "depth_controlnet", "depth_start", "depth_end",
]);
const MOSS_ONLY = new Set([
  "moss_unet_name", "moss_clip_name", "moss_vae_name",
  "moss_depth_lora", "moss_canny_lora", "keep_artifacts",
  "moss_canny_low", "moss_canny_high", "moss_canny_resolution",
  "moss_leftover_noise_handoff",
]);

const LEGACY_KEYS = ["type", "computeSize", "draw", "mouse"];

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
      getMinHeight: () => 250,
      getMaxHeight: () => 250,
    });

    // Widgets are always looked up lazily: they may not exist yet when nodeCreated fires.
    const widgetByName = (name) => (node.widgets || []).find((w) => w.name === name);

    // DOM widgets (the multiline prompt boxes) are managed by the frontend, so they are
    // only toggled through hidden flags and their element. Plain widgets also get the
    // classic type/computeSize override so they take no space while collapsed.
    function setVisible(w, visible) {
      if (!w) return;
      const isDom = !!w.element;
      if (!isDom && !snapshots.has(w)) {
        const saved = {};
        for (const key of LEGACY_KEYS) {
          saved[key] = { own: Object.prototype.hasOwnProperty.call(w, key), value: w[key] };
        }
        snapshots.set(w, saved);
      }
      if (!isDom) {
        if (visible) {
          const saved = snapshots.get(w);
          for (const [key, entry] of Object.entries(saved)) {
            if (entry.own) w[key] = entry.value;
            else delete w[key];
          }
        } else {
          w.type = "hidden";
          w.computeSize = () => [0, -4];
          w.draw = () => {};
          w.mouse = () => false;
        }
      }
      try { w.hidden = !visible; } catch (_) { /* read-only on some frontends */ }
      if (w.options) w.options.hidden = !visible;
      if (isDom) {
        w.element.hidden = !visible;
        if (visible) w.element.style.removeProperty("display");
        else w.element.style.display = "none";
      }
    }

    // Workflow switch: the real "workflow" widget stays hidden (it still serialises)
    // and is driven by a select pinned at the top of the panel.
    const select = document.createElement("select");
    select.style.cssText =
      "grid-column:1/-1;padding:6px 8px;background:#1e293b;color:#e2e8f0;border:1px solid #22c55e;border-radius:6px;font:12px system-ui;";
    for (const value of ["SDXL", "MOSS"]) {
      const opt = document.createElement("option");
      opt.value = value;
      opt.textContent = `Workflow: ${value}`;
      select.appendChild(opt);
    }
    select.addEventListener("pointerdown", (event) => event.stopPropagation());
    select.addEventListener("change", (event) => {
      event.stopPropagation();
      const wf = widgetByName("workflow");
      if (wf) {
        wf.value = select.value;
        wf.callback?.(select.value);
      }
      sync();
    });
    el.appendChild(select);

    const mode = () => (widgetByName("workflow")?.value === "MOSS" ? "MOSS" : "SDXL");
    const allowed = (name) =>
      mode() === "MOSS" ? !SDXL_ONLY.has(name) : !MOSS_ONLY.has(name);

    const buttons = GROUPS.map(([label], i) => {
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
      node.setSize([Math.max(node.size[0], 420), 300]);
    }, 10);

    let logged = false;
    function sync() {
      const wf = widgetByName("workflow");
      if (wf) setVisible(wf, false);
      select.value = mode();

      for (const w of node.widgets || []) {
        if (w === panel || w === wf) continue;
        const i = GROUPS.findIndex(([, names]) => names.includes(w.name));
        if (i < 0) continue;
        setVisible(w, open.has(i) && allowed(w.name));
      }

      if (!logged && (node.widgets || []).length > 1) {
        logged = true;
        console.debug("[ArchForge] widgets:", (node.widgets || []).map((w) => `${w.name}:${w.type}${w.element ? "(dom)" : ""}`).join(", "));
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

    // Keep the layout in sync if the workflow value changes through any other path.
    const widgetChanged = node.onWidgetChanged;
    node.onWidgetChanged = function (name, ...args) {
      const result = widgetChanged?.call(this, name, ...args);
      if (name === "workflow") sync();
      return result;
    };

    sync();
    // Widgets can be created after nodeCreated in some frontends; re-sync once they exist.
    setTimeout(sync, 50);
    setTimeout(sync, 500);
  },
});
