import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


def load_data(json_path: str) -> pd.DataFrame:
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {json_path}")

    with open(path, "r") as f:
        data = json.load(f)
    
    df = pd.DataFrame.from_dict(data, orient="index")
    df = df[(df["unsup_pre_translation"] <= 0.6) & (df["sup_pre_translation"] >= 0.85)]
    df["mode_information_gain"] = df["sup_pre_translation"] - df["unsup_pre_translation"] # e.g. audio-level similarity - text-level sim (the bigger the more translation should make text-level bigger)
    # WARNING: looking at the cos metrics in the paper it actually represents the similarity between F(x_i) and x_i. therefore the below quantity must tend to 0
    # for example in this case: df["unsup_translated"] - df["unsup_pre_translation"] you would be subtracting to similar quantities.
    # this i argue could happen if the two models share some training data. 
    # maybe check if clap and gte do. 
    # but then, which metric shuold we use ? should we only keep unsup_translated ? 

    df["unsup_translation_gain"] = df["unsup_translated"] - df["unsup_pre_translation"]

    print(df.shape)
    return df

def plot(df: pd.DataFrame, save_path: str = None):
    # Compute Pearson correlation
    pearson_r = df["mode_information_gain"].corr(df["unsup_translation_gain"])

    # Create plot
    plt.figure(figsize=(7, 5))
    sns.scatterplot(
        data=df,
        x="mode_information_gain",
        y="unsup_translation_gain",
        s=80,
        color="blue",
    )

    # Add regression line
    sns.regplot(
        data=df,
        x="mode_information_gain",
        y="unsup_translation_gain",
        scatter=False,
        color="red",
        ci=None,
        line_kws={"linestyle": "--"},
    )

    # Titles and labels
    plt.title(f"Unsupervised Translation Gain vs Mode Information Gain\nPearson r = {pearson_r:.3f}")
    plt.xlabel("mode_information_gain (supervised - unsupervised pre-translation)")
    plt.ylabel("unsup_translation_gain (unsup translated - unsup pre-translation)")
    plt.grid(True, linestyle="--", alpha=0.4)

    # Save or show
    if save_path:
        print(f"Saving plot to: {save_path}")
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    else:
        plt.show()


def main(json_path: str, save_path: str = "image.png"):
    df = load_data(json_path)
    plot(df, save_path)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python plot.py <path_to_json> [<save_path>]")
        sys.exit(1)
    json_path = sys.argv[1]
    save_path = sys.argv[2] if len(sys.argv) > 2 else "image.png"
    main(json_path, save_path)
