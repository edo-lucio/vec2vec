import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_data(json_path: str):
    asis_rows = []
    isas_rows = []

    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)

            u = obj["unsup_pre_translation"]
            s = obj["sup_pre_translation"]

            if u >= 0.85 and s <= 0.75:
                obj["sup_gain"] = obj["sup_translated"] - obj["sup_pre_translation"]
                asis_rows.append(obj)
                continue

            if u <= 0.75 and s >= 0.85:
                obj["sup_gain"] = obj["sup_translated"] - obj["sup_pre_translation"]
                isas_rows.append(obj)

    df_asis = pd.DataFrame(asis_rows)
    df_isas = pd.DataFrame(isas_rows)

    return df_asis, df_isas


def plot(df_asis: pd.DataFrame, df_isas: pd.DataFrame, save_path: str = None):
    plt.figure(figsize=(8, 5))
    
    print(df_asis.shape, df_isas.shape)
    
    df_combined = pd.concat([
        df_asis.assign(Source="ASIS"),
        df_isas.assign(Source="ISAS")
    ])
    
    sns.histplot(
        data=df_combined,
        x="sup_gain",
        hue="Source",        
        stat="density",      
        common_norm=False,   
        element="step",     
        alpha=0.3,           
        kde=True             
    )
    
    plt.title("Normalized Distribution of sup_gain (Fair Comparison)")
    plt.xlabel("Sup Gain")
    plt.ylabel("Density")
    
    if save_path:
        print(f"Saving plot to: {save_path}")
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    else:
        plt.show()

def main(json_path: str, save_path: str = "image.png"):
    asis, isas = load_data(json_path)
    plot(asis, isas, save_path)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python plot.py <path_to_json> [<save_path>]")
        sys.exit(1)
    json_path = sys.argv[1]
    save_path = sys.argv[2] if len(sys.argv) > 2 else "image.png"
    main(json_path, save_path)
