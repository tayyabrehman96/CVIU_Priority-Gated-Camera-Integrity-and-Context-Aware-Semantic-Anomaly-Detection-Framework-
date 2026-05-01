from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


def add_lifeline(ax, x, label, y_top=0.90, y_bottom=0.08, box_w=None, font_size=8.0):
    ax.plot([x, x], [y_bottom, y_top], linestyle="--", linewidth=1.0, color="#475569")
    if box_w is None:
        lines = label.split("\n")
        max_line_len = max(len(line) for line in lines) if lines else 8
        box_w = min(0.16, max(0.09, 0.06 + 0.0048 * max_line_len))
    head = FancyBboxPatch(
        (x - box_w / 2, y_top + 0.02),
        box_w,
        0.055,
        boxstyle="round,pad=0.01,rounding_size=0.015",
        linewidth=1.2,
        edgecolor="#0f172a",
        facecolor="#e2e8f0",
    )
    ax.add_patch(head)
    ax.text(x, y_top + 0.047, label, ha="center", va="center", fontsize=font_size, fontweight="bold")


def add_activation(ax, x, y, h, w=0.018, color="#bfdbfe"):
    rect = Rectangle((x - w / 2, y - h), w, h, linewidth=0.8, edgecolor="#1d4ed8", facecolor=color)
    ax.add_patch(rect)


def message(ax, x1, x2, y, label, color="#111827"):
    ax.annotate(
        "",
        xy=(x2, y),
        xytext=(x1, y),
        arrowprops=dict(arrowstyle="->", lw=1.1, color=color),
    )
    ax.text((x1 + x2) / 2, y + 0.012, label, ha="center", va="bottom", fontsize=7.8)


def return_msg(ax, x1, x2, y, label):
    ax.annotate(
        "",
        xy=(x2, y),
        xytext=(x1, y),
        arrowprops=dict(arrowstyle="->", lw=1.0, linestyle="--", color="#334155"),
    )
    ax.text((x1 + x2) / 2, y + 0.01, label, ha="center", va="bottom", fontsize=7.5, color="#334155")


def add_fragment(ax, x, y, w, h, title):
    rect = Rectangle((x, y - h), w, h, linewidth=1.0, edgecolor="#475569", facecolor="none")
    ax.add_patch(rect)
    ax.text(x + 0.01, y - 0.01, title, ha="left", va="top", fontsize=8.0, fontweight="bold", color="#1f2937")


def main():
    fig, ax = plt.subplots(figsize=(19.5, 8.5), dpi=320)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor("white")

    # Lifeline positions
    x = {
        "source": 0.07,
        "ingest": 0.20,
        "ci": 0.35,
        "context": 0.50,
        "semantic": 0.64,
        "decision": 0.81,
        "sink": 0.965,
    }

    add_lifeline(ax, x["source"], "Data Source\n(Public / Live)", box_w=0.15, font_size=8.2)
    add_lifeline(ax, x["ingest"], "ONVIF\nIngestion", box_w=0.12, font_size=8.2)
    add_lifeline(ax, x["ci"], "CI Guard", box_w=0.105, font_size=8.2)
    add_lifeline(ax, x["context"], "Context\nScorer", box_w=0.12, font_size=8.2)
    add_lifeline(ax, x["semantic"], "Semantic\nDetector", box_w=0.13, font_size=8.2)
    add_lifeline(ax, x["decision"], "Decision +\nLogger", box_w=0.13, font_size=8.2)
    add_lifeline(ax, x["sink"], "Alert /\nReport Sink", box_w=0.13, font_size=8.2)

    # Activations
    add_activation(ax, x["ingest"], 0.84, 0.70, color="#dbeafe")
    add_activation(ax, x["ci"], 0.80, 0.56, color="#dbeafe")
    add_activation(ax, x["context"], 0.58, 0.24, color="#dbeafe")
    add_activation(ax, x["semantic"], 0.54, 0.27, color="#dbeafe")
    add_activation(ax, x["decision"], 0.80, 0.67, color="#dbeafe")
    add_activation(ax, x["sink"], 0.46, 0.33, color="#e2e8f0")

    # Messages
    y = 0.86
    message(ax, x["source"], x["ingest"], y, "1) Frame stream $I_t$ from selected mode")
    y -= 0.06
    message(ax, x["ingest"], x["ingest"], y, "2) Timestamp sync and buffering")
    y -= 0.05
    message(ax, x["ingest"], x["ci"], y, "3) Decode and normalize")
    y -= 0.06
    message(ax, x["ci"], x["ci"], y, "4) CI checks: block/cover/blur/move")

    # alt fragment
    add_fragment(ax, 0.27, 0.65, 0.68, 0.49, "alt  Camera Integrity Outcome")
    ax.plot([0.27, 0.95], [0.54, 0.54], color="#64748b", lw=1.0)
    ax.text(0.28, 0.56, "[CI fault]", fontsize=9.0, color="#0f172a", fontweight="bold")
    ax.text(0.28, 0.37, "[CI normal]", fontsize=9.0, color="#0f172a", fontweight="bold")

    # CI fault path
    y = 0.50
    message(ax, x["ci"], x["decision"], y, "5a) Emit CI label")
    y -= 0.05
    message(ax, x["decision"], x["decision"], y, "6a) Compose rationale")
    y -= 0.05
    message(ax, x["decision"], x["decision"], y, "7a) Log event metadata")
    y -= 0.05
    message(ax, x["decision"], x["sink"], y, "8a) Dispatch guard alert")
    y -= 0.05
    return_msg(ax, x["sink"], x["source"], y, "9a) Next frame request")

    # CI normal path
    y = 0.32
    message(ax, x["ci"], x["context"], y, "5b) Compute context score $c_t$")
    y -= 0.05
    message(ax, x["context"], x["context"], y, "6b) Update running scene statistics")
    y -= 0.05
    message(ax, x["context"], x["semantic"], y, "7b) Run semantic inference")
    y -= 0.05
    message(ax, x["semantic"], x["semantic"], y, "8b) YOLO + fire/smoke + motion cues")
    y -= 0.05
    message(ax, x["semantic"], x["decision"], y, "9b) Send scores + evidence")
    y -= 0.05
    message(ax, x["decision"], x["decision"], y, "10b) Fuse scores and assign class")
    y -= 0.05
    message(ax, x["decision"], x["decision"], y, "11b) Compose rationale + metadata")
    y -= 0.05
    message(ax, x["decision"], x["sink"], y, "12b) Publish event packet")
    y -= 0.05
    return_msg(ax, x["sink"], x["source"], y, "13b) Continue stream")

    # Optional benchmarking branch for public datasets
    add_fragment(ax, 0.02, 0.26, 0.32, 0.18, "opt  Public dataset mode")
    message(ax, x["source"], x["sink"], 0.18, "A) Store predictions for PR/F1/AUC")
    return_msg(ax, x["sink"], x["source"], 0.13, "B) Batch evaluation summary")

    fig.subplots_adjust(left=0.03, right=0.985, top=0.96, bottom=0.05)
    fig.savefig("Images/proposed_model_activity_diagram.png", bbox_inches="tight", pad_inches=0.08)


if __name__ == "__main__":
    main()
