import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

# ---------- helpers ----------

def draw_panel(ax, x, y, w, h, title):
    """Colored panel with a title bar."""
    panel = Rectangle((x, y), w, h, facecolor="#e9f3ff",
                      edgecolor="#5b8bd1", linewidth=1.5)
    ax.add_patch(panel)

    title_h = 0.5
    title_box = Rectangle((x, y + h - title_h), w, title_h,
                          facecolor="#5b8bd1", edgecolor="#5b8bd1")
    ax.add_patch(title_box)

    ax.text(x + w/2, y + h - title_h/2, title,
            ha="center", va="center", color="white",
            fontsize=10, weight="bold")

def draw_rounded_box(ax, x, y, w, h, text):
    """Rounded white box for each step."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.15,rounding_size=0.15",
        linewidth=1.4, edgecolor="#333333", facecolor="white"
    )
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text,
            ha="center", va="center",
            fontsize=8, wrap=True)

def draw_arrow(ax, x1, y1, x2, y2):
    """Arrow from (x1,y1) to (x2,y2)."""
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="->",
        mutation_scale=14,
        linewidth=1.3,
        color="#444444"
    )
    ax.add_patch(arr)

# ---------- figure ----------

fig, ax = plt.subplots(figsize=(11, 6))
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis("off")

panel_w = 4.8
panel_h = 6.5

# Panels: Data, LSTM, Hybrid/Eval
draw_panel(ax, 0.4, 1.2, panel_w, panel_h, "Data Processing")
draw_panel(ax, 5.4, 1.2, panel_w, panel_h, "LSTM Autoencoder")
draw_panel(ax, 10.4, 1.2, panel_w, panel_h, "Hybrid Classification & Evaluation")

box_w = 3.7
box_h = 1.0

# ---------- Data panel boxes ----------
y_top = 6.8
x_data = 0.95

draw_rounded_box(ax, x_data, y_top, box_w, box_h,
                 "Raw Sensor Data\n(CSV, shoe IMU)")
draw_rounded_box(ax, x_data, y_top - 1.4, box_w, box_h,
                 "Signal Normalisation")
draw_rounded_box(ax, x_data, y_top - 2.8, box_w, box_h,
                 "Windowing\n(100×3 samples)")
draw_rounded_box(ax, x_data, y_top - 4.2, box_w, box_h,
                 "Labeled Windows\n(0 = normal, 1 = abnormal)")

cx = x_data + box_w/2
draw_arrow(ax, cx, y_top + box_h*0.1,
           cx, y_top - 1.4 + box_h*0.9)
draw_arrow(ax, cx, y_top - 1.4 + box_h*0.1,
           cx, y_top - 2.8 + box_h*0.9)
draw_arrow(ax, cx, y_top - 2.8 + box_h*0.1,
           cx, y_top - 4.2 + box_h*0.9)

# ---------- LSTM panel boxes ----------
x_lstm = 5.95
y_lstm_top = 6.8

draw_rounded_box(ax, x_lstm, y_lstm_top, box_w, box_h,
                 "Train LSTM Autoencoder\n(on normal gait windows)")
draw_rounded_box(ax, x_lstm, y_lstm_top - 1.6, box_w, box_h,
                 "Reconstruction Error\n(MSE per window)")
draw_rounded_box(ax, x_lstm, y_lstm_top - 3.2, box_w, box_h,
                 "Threshold Selection\n(95th percentile of normal error)")
draw_rounded_box(ax, x_lstm, y_lstm_top - 4.8, box_w, box_h,
                 "LSTM-based Classification\n(normal vs abnormal)")

cx_lstm = x_lstm + box_w/2
draw_arrow(ax, cx_lstm, y_lstm_top + box_h*0.1,
           cx_lstm, y_lstm_top - 1.6 + box_h*0.9)
draw_arrow(ax, cx_lstm, y_lstm_top - 1.6 + box_h*0.1,
           cx_lstm, y_lstm_top - 3.2 + box_h*0.9)
draw_arrow(ax, cx_lstm, y_lstm_top - 3.2 + box_h*0.1,
           cx_lstm, y_lstm_top - 4.8 + box_h*0.9)

# ---------- Hybrid / evaluation panel boxes ----------
x_hyb = 10.95
y_hyb_top = 6.8

draw_rounded_box(ax, x_hyb, y_hyb_top, box_w, box_h,
                 "LSTM Encoder Latent Features\n(64-D vector per window)")
draw_rounded_box(ax, x_hyb, y_hyb_top - 1.6, box_w, box_h,
                 "XGBoost Classifier\n(5-fold cross-validation)")
draw_rounded_box(ax, x_hyb, y_hyb_top - 3.2, box_w, box_h,
                 "Model Comparison\nLSTM AE + threshold\nvs KNN vs XGBoost")

cx_hyb = x_hyb + box_w/2
draw_arrow(ax, cx_hyb, y_hyb_top + box_h*0.1,
           cx_hyb, y_hyb_top - 1.6 + box_h*0.9)
draw_arrow(ax, cx_hyb, y_hyb_top - 1.6 + box_h*0.1,
           cx_hyb, y_hyb_top - 3.2 + box_h*0.9)

# ---------- cross-panel arrows (now with ax!) ----------

# Labeled windows -> Train LSTM AE
draw_arrow(
    ax,
    x_data + box_w,             y_top - 4.2 + box_h/2,
    x_lstm,                     y_lstm_top + box_h/2
)

# Train LSTM AE -> Latent features
draw_arrow(
    ax,
    x_lstm + box_w/2,           y_lstm_top,
    x_hyb + box_w/2,            y_hyb_top + box_h
)

# LSTM-based Classification -> Model Comparison
draw_arrow(
    ax,
    x_lstm + box_w,             y_lstm_top - 4.8 + box_h/2,
    x_hyb,                      y_hyb_top - 3.2 + box_h/2
)

plt.tight_layout()
plt.savefig("methodology_flowchart_pretty.png", dpi=300, bbox_inches="tight")
print("✅ Saved diagram to methodology_flowchart_pretty.png")
plt.show()
