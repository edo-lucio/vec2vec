from datasets import load_dataset

dset = load_dataset(
    "laion/in‑the‑wild‑sound‑events",
    streaming=True
)

# Example: iterate over first few entries
for i, sample in enumerate(dset):
    if i >= 5:
        break
    print(sample)
