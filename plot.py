import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["PTHREAD_POOL_SIZE"] = "1"


def load_data(json_path: str) -> pd.DataFrame:
    df = pd.read_json(path_or_buf=json_path, lines=True)
    df = df.loc[(df["sup_pre_translation"] < 1) & (df["unsup_pre_translation"] < 1)].copy()
    return df


def plot_raw_distributions(df: pd.DataFrame):
    plt.figure(figsize=(7, 5))

    sns.histplot(
        data=df,
        x="unsup_pre_translation",
        stat="density",
        alpha=0.3,
        kde=True
    )

    sns.histplot(
        data=df,
        x="sup_pre_translation",
        stat="density",
        alpha=0.3,
        kde=True
    )

    plt.title("Distributions of Pre-Translation Scores")
    plt.savefig("distributions.png", bbox_inches="tight", dpi=300)
    plt.close()


def wrangle(df: pd.DataFrame):
    df = df.copy()

    df["modality_gap"] = abs(df["unsup_pre_translation"] - df["sup_pre_translation"])
    x_thresh = df["unsup_pre_translation"].quantile(0.5)
    y_thresh = df["sup_pre_translation"].quantile(0.5)

    similar_audio_distant_text = df[
        (df["unsup_pre_translation"] > x_thresh) &
        (df["sup_pre_translation"] < y_thresh)
    ].copy()

    similar_text_dist_audio = df[
        (df["unsup_pre_translation"] < x_thresh) &
        (df["sup_pre_translation"] > y_thresh)
    ].copy()

    # Gains: ASIS (similar audio, distant text)
    similar_audio_distant_text["sup_latents_gain"] = (
        similar_audio_distant_text["sup_latents"] -
        similar_audio_distant_text["sup_pre_translation"]
    )
    similar_audio_distant_text["unsup_latents_gain"] = (
        similar_audio_distant_text["unsup_latents"] -
        similar_audio_distant_text["unsup_pre_translation"]
    )
#    similar_audio_distant_text["sup_trans_gain"] = (
 #       similar_audio_distant_text["sup_translated"] -
  #      similar_audio_distant_text["sup_pre_translation"]
   # )
    #similar_audio_distant_text["unsup_trans_gain"] = (
     #   similar_audio_distant_text["unsup_translated"] -
      #  similar_audio_distant_text["unsup_pre_translation"]
    #)
    similar_audio_distant_text["Source"] = "ASIS"

    # Gains: ISAS (similar text, distant audio)
    similar_text_dist_audio["sup_latents_gain"] = (
        similar_text_dist_audio["sup_latents"] -
        similar_text_dist_audio["sup_pre_translation"]
    )
    similar_text_dist_audio["unsup_latents_gain"] = (
        similar_text_dist_audio["unsup_latents"] -
        similar_text_dist_audio["unsup_pre_translation"]
    )
    #similar_text_dist_audio["sup_trans_gain"] = (
     #   similar_text_dist_audio["sup_translated"] -
      #  similar_text_dist_audio["sup_pre_translation"]
    # )
    #similar_text_dist_audio["unsup_trans_gain"] = (
     #   similar_text_dist_audio["unsup_translated"] -
      #  similar_text_dist_audio["unsup_pre_translation"]
    #)
    similar_text_dist_audio["Source"] = "ISAS"

    print(similar_audio_distant_text.shape, similar_text_dist_audio.shape)

    df = pd.concat([similar_audio_distant_text, similar_text_dist_audio], ignore_index=True)
    return df


def plot(df: pd.DataFrame, save_path: str):
    metrics = ["sup_latents_gain", "unsup_latents_gain"]

    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    axs = axs.flatten()

    for ax, metric in zip(axs, metrics):
        sns.kdeplot(
            data=df,
            x=metric,
            hue="Source",
            ax=ax
        )
        ax.set_title(metric)

    fig.suptitle("Gain Distributions between Embedding Spaces Before/After Translation")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        print(f"Saving plot to: {save_path}")
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
    else:
        plt.show()


def main(json_path: str, fn: str):
    df = load_data(json_path)
    df = wrangle(df)
    save_path = f"{fn}.png"
    plot(df, save_path)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python plot.py <path_to_json> [<save_path_without_ext>]")
        sys.exit(1)

    json_path = sys.argv[1]
    save_path = json_path.split(".")[-2]
    main(json_path, save_path)

