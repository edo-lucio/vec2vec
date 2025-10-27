import os
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from types import SimpleNamespace
import toml
from sys import argv
import argparse

from utils.collate import MultiencoderTokenizedDataset, TokenizedCollator
from utils.utils import get_num_proc, load_n_translator
from utils.model_utils import load_encoder
from utils.streaming_utils import load_streaming_embeddings

class EmbeddingVisualizer:
    def __init__(self, cfg, translator, encoders, dataset, device=None):
        self.cfg = cfg
        self.translator = translator
        self.encoders = encoders
        self.dataset = dataset
        self.device = device if device else next(translator.parameters()).device

    def compute_embeddings(self):
        num_workers = get_num_proc()
        dataloader = DataLoader(
            MultiencoderTokenizedDataset(
                dataset=self.dataset,
                encoders=self.encoders,
                n_embs_per_batch=2,
                batch_size=self.cfg.val_bs,
                max_length=self.cfg.max_seq_length,
                seed=self.cfg.sampling_seed
            ),
            batch_size=self.cfg.val_bs,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=TokenizedCollator(),
            drop_last=False
        )

        embeddings_dict = {name: [] for name in self.encoders.keys()}
        latent_list = []

        self.translator.eval()
        with torch.no_grad():
            for batch in dataloader:
                print(batch.keys())  # debug

                batch_embeddings = {}

                for name, encoder in self.encoders.items():
                    # Handle standard HF-style encoders (e.g., e5)
                    if "e5" in name.lower():
                        encoder_batch = {
                            "input_ids": batch["e5_input_ids"].to(self.device),
                            "attention_mask": batch["e5_attention_mask"].to(self.device)
                        }
                        if "e5_token_type_ids" in batch:
                            encoder_batch["token_type_ids"] = batch["e5_token_type_ids"].to(self.device)

                        emb = encoder(**encoder_batch).detach().cpu()

                    elif "clap" in name.lower():
                        # Grab raw text from dataset
                        texts = batch.get("clap_text", None)
                        if texts is None:
                            # fallback: maybe the dataset itself has a text field
                            texts = [item["text"] for item in self.dataset]  # careful: match batch size!
                        
                        # If the batch already contains a list of texts
                        if isinstance(texts, torch.Tensor):
                            # Convert tensor of token IDs back to strings if needed
                            # (requires your tokenizer)
                            texts = [self.encoders[name].tokenizer.decode(ids) for ids in texts]
                        
                        emb = encoder.encode(
                            texts,
                            convert_to_tensor=True
                        ).detach().cpu()

                        embeddings_dict[name].append(emb)
                        batch_embeddings[name] = emb.to(self.device)

                # Encode through the translator
                latent_emb = self.translator.encode(batch_embeddings).detach().cpu()
                latent_list.append(latent_emb)

        # Concatenate embeddings
        for k in embeddings_dict.keys():
            embeddings_dict[k] = torch.cat(embeddings_dict[k], dim=0)
        latent_embeddings = torch.cat(latent_list, dim=0)

        return embeddings_dict, latent_embeddings

    def plot_tsne(self, title_suffix="", output_dir="results/plots"):
        embeddings_dict, latent_embeddings = self.compute_embeddings()

        combined = torch.cat(list(embeddings_dict.values()) + [latent_embeddings], dim=0)
        tsne = TSNE(n_components=2, init='pca', random_state=42)
        tsne_results = tsne.fit_transform(combined.numpy())

        sizes = [len(v) for v in embeddings_dict.values()] + [len(latent_embeddings)]
        indices = torch.cumsum(torch.tensor([0] + sizes), dim=0)

        encoder_names = list(embeddings_dict.keys())
        colors = plt.cm.get_cmap('tab10', len(encoder_names))
        plt.figure(figsize=(10, 8))

        # Original embeddings
        for i, name in enumerate(encoder_names):
            start, end = indices[i], indices[i+1]
            plt.scatter(tsne_results[start:end, 0], tsne_results[start:end, 1],
                        label=f"{name} (original)", color=colors(i), alpha=0.6, marker='o')

        # Latent embeddings
        latent_start, latent_end = indices[-2], indices[-1]
        n_per_encoder = latent_embeddings.shape[0] // len(encoder_names)
        for i, name in enumerate(encoder_names):
            start_idx = i * n_per_encoder
            end_idx = (i + 1) * n_per_encoder if i < len(encoder_names) - 1 else latent_embeddings.shape[0]
            plt.scatter(tsne_results[latent_start + start_idx:latent_start + end_idx, 0],
                        tsne_results[latent_start + start_idx:latent_start + end_idx, 1],
                        label=f"{name} (latent)", color=colors(i), alpha=0.8, marker='x')

        plt.legend()
        plt.title(f"t-SNE of Original and Translator Embeddings {title_suffix}")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.grid(True)

        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"tsne_embeddings{title_suffix}.png")
        plt.savefig(out_path)
        print(f"Saved t-SNE plot to {out_path}")
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot encoder and translator embeddings with t-SNE")
    parser.add_argument("--config", type=str, required=True, help="Path to config directory containing config.toml")
    parser.add_argument("--dataset", type=str, default="train", help="Dataset split to use: train/val/test")
    parser.add_argument("--output", type=str, default="results/plots", help="Output directory for plots")
    parser.add_argument("--sup_emb", type=str,)
    parser.add_argument("--unsup_emb", type=str,)

    args = parser.parse_args()
    args.config = f"finetuning_unsupervised/{args.config}"

    cfg = toml.load(os.path.join(args.config, "config.toml"))
    cfg = SimpleNamespace(**cfg)

    cfg.sup_emb = args.sup_emb if args.sup_emb else cfg.sup_emb
    cfg.unsup_emb = args.unsup_emb if args.unsup_emb else cfg.unsup_emb

    # Load dataset
    dset = load_streaming_embeddings(cfg.dataset)
    if args.dataset == "train":
        dataset = dset
    elif args.dataset == "val":
        dataset = dset.train_test_split(test_size=cfg.val_size, seed=cfg.val_dataset_seed)["test"]
    else:
        dataset = dset

    # Load encoders
    sup_enc = load_encoder(cfg.sup_emb)
    unsup_enc = load_encoder(cfg.unsup_emb)
    encoders = {cfg.sup_emb: sup_enc, cfg.unsup_emb: unsup_enc}

    encoder_dims = {cfg.sup_emb: sup_enc.get_sentence_embedding_dimension(),
                    cfg.unsup_emb: unsup_enc.get_sentence_embedding_dimension()}
    translator = load_n_translator(cfg, encoder_dims)
    translator.load_state_dict(torch.load(os.path.join(args.config, "model.pt"), map_location="cpu"))
    translator.eval()

    # Run visualizer
    visualizer = EmbeddingVisualizer(cfg, translator, encoders, dataset)
    visualizer.plot_tsne(title_suffix=f"_{args.dataset}", output_dir=args.output)


if __name__ == "__main__":
    main()
