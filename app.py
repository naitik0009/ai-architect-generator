import json
import os
import math
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Wedge
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from post_process import post_process, has_closed_exterior_loop

# --- Configuration (overridable via env vars, e.g. for Colab/cloud) ---
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-Coder-14B-Instruct")
# Local adapter dir by default; set ADAPTER_PATH=naitik12kumar/qwen-kalkulio-lora-14b-v4
# to pull straight from the HF Hub (no local files needed).
ADAPTER_PATH = os.environ.get("ADAPTER_PATH", "./qwen-kalkulio-lora-14b-v4/final")
# Set LOAD_IN_4BIT=1 to fit the 14B on a 16 GB GPU (e.g. Colab free T4).
LOAD_IN_4BIT = os.environ.get("LOAD_IN_4BIT", "0") == "1"
SYSTEM_PROMPT = "You are an expert architectural AI. Generate a valid JSON floor plan for a single-family house."

# --- Global Model State ---
model = None
tokenizer = None

def load_model():
    global model, tokenizer
    if model is None:
        print(f"Loading {BASE_MODEL}  (4-bit={LOAD_IN_4BIT})  + adapter {ADAPTER_PATH} ...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        load_kwargs = dict(device_map="auto", attn_implementation="sdpa")
        if LOAD_IN_4BIT:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            load_kwargs["dtype"] = torch.bfloat16
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **load_kwargs)
        model = PeftModel.from_pretrained(base, ADAPTER_PATH)
        model.eval()
        print("Model loaded successfully.")

# --- Visualization ---
ROOM_COLORS = {
    "LivingRoom": "#ffe4b5",
    "Bedroom":    "#e6e6fa",
    "Kitchen":    "#ffdab9",
    "Bath":       "#e0ffff",
    "Entry":      "#f5f5dc",
    "Dining":     "#fffacd",
    "DiningRoom": "#fffacd",
    "Hall":       "#f0f8ff",
    "Hallway":    "#f0f8ff",
    "Vestibule":  "#f5f5dc",
    "Closet":     "#f5fffa",
    "Storage":    "#f5fffa",
    "Garage":     "#dcdcdc",
    "Office":     "#fff0f5",
    "Sauna":      "#ffe4e1",
    "Laundry":    "#e0f7fa",
    "Pantry":     "#fff8dc",
    "Utility":    "#f5fffa",
    "Room":       "#f0f0f0",
    "Pokoj":      "#f0f0f0",
    "Undefined":  "#f0f0f0",
}

ROOM_LABEL = {
    "LivingRoom": "Obývák",
    "Bedroom":    "Ložnice",
    "Kitchen":    "Kuchyně",
    "Bath":       "Koupelna",
    "Entry":      "Vstup",
    "Dining":     "Jídelna",
    "DiningRoom": "Jídelna",
    "Hall":       "Hala",
    "Hallway":    "Chodba",
    "Vestibule":  "Vestibul",
    "Closet":     "Šatna",
    "Storage":    "Sklad",
    "Garage":     "Garáž",
    "Office":     "Pracovna",
    "Sauna":      "Sauna",
    "Laundry":    "Prádelna",
    "Pantry":     "Spíž",
    "Utility":    "Tech.",
    "Room":       "Pokoj",
    "Pokoj":      "Pokoj",
    "Undefined":  "Místnost",
}


def _opening_endpoints(wall, pozice, sirka):
    """Return the (x, y) start/end of an opening along its host wall."""
    x1, y1 = wall["od"]
    x2, y2 = wall["do"]
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return None
    ux, uy = dx / length, dy / length
    sx = x1 + ux * pozice
    sy = y1 + uy * pozice
    ex = x1 + ux * (pozice + sirka)
    ey = y1 + uy * (pozice + sirka)
    return (sx, sy, ex, ey, ux, uy, length)


def render_floor_plan(plan):
    """Render a Kalkulio floor plan as a clean matplotlib figure.

    Drawing order (back to front):
      1. Rooms (filled polygons + thin outline)
      2. Walls (thick dark segments) — drawn over rooms so any room polygon
         that extends past the walls is visually clipped at the wall edge
      3. Openings (windows = light fill on wall, doors = arc + frame)
      4. Room labels (on top so they stay readable)
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    rooms = plan.get("prostory", [])
    walls = plan.get("steny", [])
    openings = plan.get("otvory", [])
    walls_by_id = {w["id"]: w for w in walls if "id" in w}

    # 1. Rooms
    for r in rooms:
        poly = r.get("polygon")
        if not poly:
            continue
        rtype = r.get("typ", "Undefined")
        color = ROOM_COLORS.get(rtype, "#f0f0f0")
        ax.add_patch(Polygon(
            poly, closed=True,
            facecolor=color, edgecolor="#cccccc",
            linewidth=0.5, alpha=0.85, zorder=1,
        ))

    # 2. Walls (on top of rooms; thick exterior, thinner partitions)
    for w in walls:
        x1, y1 = w["od"]
        x2, y2 = w["do"]
        thickness = w.get("tloustka") or 0.15
        is_exterior = w.get("typ") in ("obvodova", "nosna")
        lw = max(3.5, thickness * 14) if is_exterior else max(2.0, thickness * 12)
        ax.plot(
            [x1, x2], [y1, y2],
            color="#2c3e50", linewidth=lw,
            solid_capstyle="round", zorder=3,
        )

    # 3. Openings
    for o in openings:
        wall = walls_by_id.get(o.get("stena"))
        if not wall:
            continue
        pozice = o.get("pozice")
        sirka = o.get("sirka")
        if pozice is None or sirka is None:
            continue
        info = _opening_endpoints(wall, pozice, sirka)
        if info is None:
            continue
        sx, sy, ex, ey, ux, uy, _ = info
        otype = o.get("typ", "")
        if otype == "okno":
            ax.plot([sx, ex], [sy, ey], color="white",
                    linewidth=4, solid_capstyle="butt", zorder=4)
            ax.plot([sx, ex], [sy, ey], color="#5fa8d3",
                    linewidth=2, solid_capstyle="butt", zorder=5)
        elif otype == "dvere":
            ax.plot([sx, ex], [sy, ey], color="white",
                    linewidth=4, solid_capstyle="butt", zorder=4)
            nx, ny = -uy, ux
            cx, cy = sx, sy
            angle_along = math.degrees(math.atan2(uy, ux))
            ax.add_patch(Wedge(
                center=(cx, cy), r=sirka,
                theta1=angle_along, theta2=angle_along + 90,
                facecolor="none", edgecolor="#e67e22",
                linewidth=1.2, zorder=6,
            ))
            ax.plot(
                [cx, cx + ux * sirka],
                [cy, cy + uy * sirka],
                color="#e67e22", linewidth=1.2, zorder=6,
            )

    # 4. Room labels on top
    for r in rooms:
        poly = r.get("polygon")
        if not poly:
            continue
        rtype = r.get("typ", "Undefined")
        label = r.get("nazev") or ROOM_LABEL.get(rtype) or "Pokoj"
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        area = r.get("plocha_m2", 0)
        ax.text(
            cx, cy, f"{label}\n{area:.1f} m²",
            ha="center", va="center", fontsize=10, color="#2c3e50",
            bbox=dict(facecolor="white", alpha=0.85,
                      edgecolor="#cccccc", boxstyle="round,pad=0.3"),
            zorder=10,
        )

    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.margins(0.05)
    ax.axis("off")
    plt.tight_layout()
    return fig

# --- Generation Logic ---

def _score_plan(plan: dict, target_area: float) -> float:
    """Score a cleaned plan. Higher is better. Used to pick the best attempt.

    Rewards watertight geometry and area accuracy; penalizes the things that
    make a plan look bad to a human: tiny slivers, one room hogging the house,
    and over-fragmentation.
    """
    if not plan:
        return -1e9
    score = 0.0
    rooms = plan.get("prostory") or []
    interior = [r for r in rooms if not r.get("venkovni", False)]

    if has_closed_exterior_loop(plan):
        score += 100

    # Reward a sensible room count (diminishing past ~8)
    score += min(len(interior), 8) * 2

    interior_area = sum(r.get("plocha_m2", 0) for r in interior)
    if target_area > 0 and interior_area > 0:
        score -= abs(interior_area - target_area) / target_area * 30

    # Penalty: tiny rooms (< 4 m²) — slivers / unrealistic spaces
    tiny = sum(1 for r in interior if r.get("plocha_m2", 0) < 4.0)
    score -= tiny * 8

    # Penalty: one room hogging the house, by fraction (small plans)...
    if interior and interior_area > 0:
        largest_frac = max(r.get("plocha_m2", 0) for r in interior) / interior_area
        if largest_frac > 0.45:
            score -= (largest_frac - 0.45) * 100

    # ...and by absolute size (large plans) — e.g. a 62 m² living room is
    # unrealistic for a single-family house regardless of the total.
    for r in interior:
        a = r.get("plocha_m2", 0)
        if a > 45:
            score -= (a - 45) * 1.5

    # Penalty: over-fragmentation (too many rooms for the area)
    if target_area > 0:
        expected_max_rooms = target_area / 10.0 + 3
        if len(interior) > expected_max_rooms:
            score -= (len(interior) - expected_max_rooms) * 4

    return score


def _is_good_enough(plan: dict, score: float) -> bool:
    """Good enough to stop early (keeps generation fast): watertight and not
    heavily fragmented. We allow up to one small room so we don't burn extra
    attempts (each ~minutes on 14B) chasing perfection."""
    if not plan or not has_closed_exterior_loop(plan):
        return False
    interior = [r for r in plan.get("prostory", []) if not r.get("venkovni", False)]
    tiny = sum(1 for r in interior if r.get("plocha_m2", 0) < 4.0)
    return tiny <= 1


def _fallback_plan(area_m2: float) -> dict:
    """Procedural fallback: a simple 2-room rectangular house sized to `area_m2`.

    Used only if every attempt fails to produce valid JSON. Guarantees the UI
    always shows SOMETHING and the JSON output is valid Kalkulio format.
    """
    side = max(4.0, area_m2 ** 0.5)
    w, h = side, area_m2 / side
    return {
        "steny": [
            {"id": "W1", "od": [0, 0],   "do": [w, 0],   "tloustka": 0.3, "typ": "obvodova", "od_constraint": None, "do_constraint": None},
            {"id": "W2", "od": [w, 0],   "do": [w, h],   "tloustka": 0.3, "typ": "obvodova", "od_constraint": None, "do_constraint": None},
            {"id": "W3", "od": [w, h],   "do": [0, h],   "tloustka": 0.3, "typ": "obvodova", "od_constraint": None, "do_constraint": None},
            {"id": "W4", "od": [0, h],   "do": [0, 0],   "tloustka": 0.3, "typ": "obvodova", "od_constraint": None, "do_constraint": None},
            {"id": "W5", "od": [w / 2, 0], "do": [w / 2, h], "tloustka": 0.15, "typ": "pricka", "od_constraint": None, "do_constraint": None},
        ],
        "otvory": [
            {"id": "D1", "stena": "W5", "pozice": h / 2 - 0.45, "sirka": 0.9, "typ": "dvere", "smer_otvirani": "left_in", "pocet_kridel": 1, "typ_dveri": "jednostranne"},
            {"id": "O1", "stena": "W1", "pozice": w / 4,       "sirka": 1.5, "typ": "okno",  "smer_otvirani": None,     "pocet_kridel": 1, "typ_dveri": None},
            {"id": "O2", "stena": "W3", "pozice": w * 3 / 4,   "sirka": 1.5, "typ": "okno",  "smer_otvirani": None,     "pocet_kridel": 1, "typ_dveri": None},
        ],
        "prostory": [
            {"id": "P1", "typ": "LivingRoom", "podtyp": None,
             "polygon": [[0, 0], [w / 2, 0], [w / 2, h], [0, h], [0, 0]],
             "plocha_m2": round(w / 2 * h, 2), "nazev": "Obývák", "venkovni": False},
            {"id": "P2", "typ": "Bedroom", "podtyp": None,
             "polygon": [[w / 2, 0], [w, 0], [w, h], [w / 2, h], [w / 2, 0]],
             "plocha_m2": round(w / 2 * h, 2), "nazev": "Ložnice", "venkovni": False},
        ],
    }


def generate_plan(area_m2, max_attempts=2, progress=gr.Progress()):
    load_model()

    best_plan = None
    best_score = float("-inf")
    best_attempt = 0
    invalid_count = 0

    temperatures = [0.3, 0.6]

    for attempt in range(1, max_attempts + 1):
        progress((attempt - 1) / max_attempts, desc=f"Attempt {attempt}/{max_attempts}...")
        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Generate a floor plan for a house with an approximate area of {area_m2}m2."},
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            temp = temperatures[min(attempt - 1, len(temperatures) - 1)]
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=3072,   # plans <=180 m² fit easily; caps runaways
                    do_sample=True,
                    temperature=temp,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                )

            raw = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

            try:
                plan = json.loads(raw)
            except json.JSONDecodeError:
                invalid_count += 1
                continue

            try:
                cleaned = post_process(plan, target_area=area_m2)
            except Exception as e:
                print(f"  [attempt {attempt}] post_process crashed: {e}")
                continue

            score = _score_plan(cleaned, area_m2)
            if score > best_score:
                best_plan = cleaned
                best_score = score
                best_attempt = attempt

            # Stop as soon as we have a watertight, non-fragmented plan. With
            # only 2 attempts and ~minutes each on 14B, we don't chase perfection.
            if _is_good_enough(cleaned, score):
                break

        except Exception as e:
            print(f"  [attempt {attempt}] unexpected error: {e}")
            invalid_count += 1
            continue

    # Build status message based on outcome
    if best_plan is None:
        # Total failure — fall back to procedural plan so the UI never breaks
        best_plan = _fallback_plan(area_m2)
        status_msg = (f"❌ Model failed all {max_attempts} attempts ({invalid_count} invalid JSON). "
                      f"Showing a procedural fallback house. Try a different area.")
    elif has_closed_exterior_loop(best_plan):
        status_msg = (f"✅ Watertight floor plan selected (best of {best_attempt} "
                      f"attempt{'s' if best_attempt > 1 else ''}, score {best_score:.0f}).")
    else:
        status_msg = (f"⚠️ No fully watertight plan found in {max_attempts} attempts. "
                      f"Showing best-scoring attempt (score={best_score:.0f}). "
                      f"The house may have minor gaps.")

    try:
        fig = render_floor_plan(best_plan)
    except Exception as e:
        print(f"render crash: {e}")
        fig = None
        status_msg += f"  (render error: {e})"

    return json.dumps(best_plan, indent=2, ensure_ascii=False), fig, status_msg

# --- Modification Logic (Bonus) ---
def modify_plan(plan_json_str, instruction):
    # Guard: empty input box (Gradio passes None/"") — don't crash on json.loads(None)
    if not plan_json_str or not str(plan_json_str).strip():
        return plan_json_str, None, "❌ Paste a floor plan JSON first — generate one in the 'Generate New Plan' tab, then copy it here."
    if not instruction or not str(instruction).strip():
        return plan_json_str, None, "❌ Enter a modification instruction (e.g. 'add a bedroom')."

    load_model()

    try:
        # Validate input JSON
        json.loads(plan_json_str)
    except (json.JSONDecodeError, TypeError):
        return plan_json_str, None, "❌ Invalid input JSON — paste a complete floor plan from the first tab."

    messages = [
        {"role": "system", "content": "You are an expert architectural AI. You modify JSON floor plans based on user instructions. Output ONLY the modified JSON."},
        {"role": "user", "content": f"Here is the current floor plan:\n```json\n{plan_json_str}\n```\n\nInstruction: {instruction}\n\nOutput the complete modified JSON."}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=8192,
            do_sample=True,
            temperature=0.3, # Lower temp for modification to preserve structure
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    raw = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    # Strip markdown code blocks if the model added them
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0].strip()
        
    try:
        modified_plan = json.loads(raw)
        # We post-process the modified plan to ensure it's still geometrically valid
        # We don't force a target area here, we let the modification dictate it
        cleaned = post_process(modified_plan)
        fig = render_floor_plan(cleaned)
        return json.dumps(cleaned, indent=2, ensure_ascii=False), fig, "✅ Modification successful."
    except json.JSONDecodeError:
        return raw, None, "❌ Failed to parse modified JSON."

# --- Gradio UI ---
with gr.Blocks(title="Kalkulio AI Architect", theme=gr.themes.Default(primary_hue="blue")) as demo:
    gr.Markdown(
        """
        # 🏗️ Kalkulio AI Architect
        Generate and modify single-family house floor plans using a fine-tuned Qwen2.5-Coder-14B model.
        """
    )
    
    with gr.Tabs():
        # TAB 1: GENERATION
        with gr.TabItem("✨ Generate New Plan"):
            with gr.Row():
                with gr.Column(scale=1):
                    area_slider = gr.Slider(minimum=60, maximum=130, value=90, step=5, label="Target Area (m²)")
                    generate_btn = gr.Button("Generate Floor Plan", variant="primary")
                    gen_status = gr.Textbox(label="Status", interactive=False)
                    
                    gr.Markdown("### How it works\n1. The AI generates a raw floor plan.\n2. A deterministic post-processor cleans up the geometry (snaps walls, drops overlaps, scales to exact area).\n3. A best-of-N sampler ensures the final house is 'watertight'.")
                
                with gr.Column(scale=2):
                    gen_plot = gr.Plot(label="Floor Plan Visualization")
                    gen_json = gr.Code(label="Output JSON (Kalkulio Format)", language="json", interactive=False)
            
            generate_btn.click(
                fn=generate_plan,
                inputs=[area_slider],
                outputs=[gen_json, gen_plot, gen_status]
            )

        # TAB 2: MODIFICATION (BONUS)
        with gr.TabItem("🛠️ Modify Existing Plan (Bonus)"):
            with gr.Row():
                with gr.Column(scale=1):
                    mod_input_json = gr.Code(label="Input JSON", language="json", interactive=True)
                    mod_instruction = gr.Textbox(label="Instruction", placeholder="e.g., Add a second bathroom, or make the living room larger.", lines=2)
                    modify_btn = gr.Button("Apply Modification", variant="primary")
                    mod_status = gr.Textbox(label="Status", interactive=False)
                    
                with gr.Column(scale=2):
                    mod_plot = gr.Plot(label="Modified Floor Plan Visualization")
                    mod_output_json = gr.Code(label="Modified JSON", language="json", interactive=False)
                    
            modify_btn.click(
                fn=modify_plan,
                inputs=[mod_input_json, mod_instruction],
                outputs=[mod_output_json, mod_plot, mod_status]
            )

if __name__ == "__main__":
    # Queue with a longer per-request timeout (model + 4 attempts ~60s)
    demo.queue(default_concurrency_limit=2, max_size=20)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
    )
