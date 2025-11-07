import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

results_dir = "results"

for filename in os.listdir(results_dir):
    if not filename.endswith(".json"):
        continue

    with open(os.path.join(results_dir, filename), "r") as f:
        results = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(filename, fontsize=14, fontweight="bold")

    recons = results.get("recons", {})
    if recons:
        axes[0].bar(recons.keys(), recons.values(), color='skyblue')
        axes[0].set_title("Reconstruction")
        axes[0].tick_params(axis='x', rotation=45)
    else:
        axes[0].text(0.5, 0.5, "No data", ha="center", va="center")

    trans = results.get("trans", {})
    if trans:
        cos_metrics = {k: v for k, v in trans.items() if k.endswith("_cos")}
        se_metrics = {k.replace("_cos", "_cos_se"): trans.get(k.replace("_cos", "_cos_se")) for k in cos_metrics}
        labels, means, errors = [], [], []
        for k, v in cos_metrics.items():
            labels.append(k.replace("_", "→"))
            means.append(v)
            errors.append(se_metrics.get(k.replace("_cos", "_cos_se"), 0))
        axes[1].bar(labels, means, yerr=errors, capsize=5, color="lightgreen")
        axes[1].set_title("Translation Cosine ± SE")
        axes[1].tick_params(axis='x', rotation=45)
    else:
        axes[1].text(0.5, 0.5, "No data", ha="center", va="center")

    heatmap = results.get("heatmap", {})
    if heatmap:
        top1_keys = [k for k in heatmap.keys() if "top_1_acc" in k]
        top16_keys = [k for k in heatmap.keys() if "top_16_acc" in k]
        models = sorted({k.split("_top")[0] for k in top1_keys + top16_keys})
        data = {m: [
            heatmap.get(f"{m}_top_1_acc (avg. 4 batches)", 0),
            heatmap.get(f"{m}_top_16_acc (avg. 4 batches)", 0)
        ] for m in models}
        df = pd.DataFrame(data, index=["top_1_acc", "top_16_acc"])
        sns.heatmap(df, annot=True, cmap="Blues", fmt=".3f", ax=axes[2], cbar=False)
        axes[2].set_title("Heatmap")
    else:
        axes[2].text(0.5, 0.5, "No data", ha="center", va="center")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(results_dir, os.path.splitext(filename)[0] + "_summary.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"saved {out}")
