import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(11, 7.6))
ax.set_xlim(0, 100); ax.set_ylim(0, 72); ax.axis("off")

ax.text(50, 69, "The IFC-XAI Integrative Framework", ha="center", fontsize=19, fontweight="bold", color="#1B365D")
ax.text(50, 65.6, "Embedding Explainable AI into Internal Financial & Accounting Control", ha="center", fontsize=11, style="italic", color="#5B6B7C")

layers = [
    ("Layer 1  -  Data & Intelligent Modeling", "#E8EEF5", "#2A7F8E", "#1B365D",
     ["Financial &\nLedger Data", "ERP / Audit\nLogs", "ML / DL\nModels", "Risk &\nFraud Engine"]),
    ("Layer 2  -  Explainability Engine (XAI)", "#E8EEF5", "#4A6070", "#1B365D",
     ["SHAP\n(global/local)", "LIME\n(local)", "Counter-\nfactuals", "Rule / Tree\nSurrogates", "Feature\nAttribution"]),
    ("Layer 3  -  Governance & Internal-Control Mapping (COSO-aligned)", "#E8EEF5", "#1F3B5C", "#1B365D",
     ["Control\nEnvironment", "Risk\nAssessment", "Control\nActivities", "Information &\nCommunication", "Monitoring &\nAudit Trail"]),
    ("Layer 4  -  Institutional Sustainability Outcomes", "#E8F3EA", "#3F7D5A", "#2F6B47",
     ["Transparency\n& Trust", "Accountability\n& Compliance", "Operational\nEfficiency", "Long-term\nResilience"]),
]
top = 61.0; h = 13.2; gap = 2.6; x0, x1 = 9, 97
for i, (title, bg, boxc, tc, items) in enumerate(layers):
    y = top - i * (h + gap) - h
    ax.add_patch(FancyBboxPatch((x0, y), x1 - x0, h, boxstyle="round,pad=0.0,rounding_size=0.8", fc=bg, ec="#3D4B5C" if i < 3 else "#3F7D5A", lw=1.3))
    ax.text(x0 + 1.5, y + h - 2.1, title, fontsize=11.5, fontweight="bold", color=tc, va="center")
    n = len(items); bw = 14.6 if n == 5 else 18.2; g = ((x1 - x0) - 3 - n * bw) / (n - 1) if n > 1 else 0
    for k, t in enumerate(items):
        bx = x0 + 1.5 + k * (bw + g)
        ax.add_patch(FancyBboxPatch((bx, y + 1.3), bw, h - 5.4, boxstyle="round,pad=0.0,rounding_size=0.6", fc=boxc, ec="none"))
        ax.text(bx + bw / 2, y + 1.3 + (h - 5.4) / 2, t, ha="center", va="center", fontsize=10, fontweight="bold", color="white", linespacing=1.15)
    if i < 3:
        ax.add_patch(FancyArrowPatch((x1 - 3, y - 0.2), (x1 - 3, y - gap + 0.2), arrowstyle="-|>", mutation_scale=16, color="#C97A1C", lw=2))

ytop = top - 0.5; ybot = top - 3 * (h + gap) - h + 3
ax.add_patch(FancyArrowPatch((5.5, ybot), (5.5, ytop), arrowstyle="-|>", mutation_scale=16, color="#6B7A8C", lw=2))
ax.text(3.2, (ytop + ybot) / 2, "Feedback / continuous monitoring loop", rotation=90, ha="center", va="center", fontsize=10, color="#6B7A8C")
plt.savefig("fig4_ifc_xai_framework.png", dpi=150, bbox_inches="tight", pad_inches=0.2, facecolor="white")
