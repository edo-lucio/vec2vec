import torch
import torch.nn as nn
import numpy as np
from typing import List, Union, Optional, Dict, Any
from transformers import ClapModel, ClapProcessor
import warnings


class AudioSentenceTransformer:
    def __init__(
        self,
        model_name_or_path: str = "laion/clap-htsat-unfused",
        device: Optional[str] = None,
        cache_folder: Optional[str] = None,
        use_safetensors: bool = True,
        project_dim: int = 768,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load pretrained CLAP
        self.model = ClapModel.from_pretrained(
            model_name_or_path,
            cache_dir=cache_folder,
            use_safetensors=use_safetensors,
        ).to(self.device)

        # Freeze CLAP weights (optional — you can unfreeze later)
        for p in self.model.parameters():
            p.requires_grad = False

        in_dim = self.model.config.projection_dim  # typically 512
        self.adapter = nn.Sequential(
            nn.Linear(in_dim, project_dim),
            nn.LayerNorm(project_dim),
            nn.GELU(),
        ).to(self.device)

        # Update config + processor
        self.model.config.projection_dim = project_dim
        self.processor = ClapProcessor.from_pretrained(
            model_name_or_path, cache_dir=cache_folder
        )

        self.model_name = model_name_or_path
        self.max_seq_length = None  # for compatibility

    # ------------------------------------------------------------------
    # Encoding (Text)
    # ------------------------------------------------------------------
    def encode(
        self,
        *args,
        texts: Union[str, List[str]] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        batch_size: int = 32,
        show_progress_bar: bool = False,
        output_value: str = "sentence_embedding",
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        normalize_embeddings: bool = False,
        **kwargs,
    ) -> Union[np.ndarray, torch.Tensor]:
        self.model.eval()
        all_embeddings = []

        if input_ids is not None and attention_mask is not None:
            for i in range(0, input_ids.size(0), batch_size):
                batch = {
                    "input_ids": input_ids[i:i + batch_size].to(self.device),
                    "attention_mask": attention_mask[i:i + batch_size].to(self.device),
                }
                if token_type_ids is not None:
                    batch["token_type_ids"] = token_type_ids[i:i + batch_size].to(self.device)

                with torch.no_grad():
                    emb = self.model.get_text_features(**batch)
                    emb = self.adapter(emb)
                all_embeddings.append(emb.cpu())

        elif texts is not None:
            if isinstance(texts, str):
                texts = [texts]

            iterator = range(0, len(texts), batch_size)
            if show_progress_bar:
                try:
                    from tqdm import tqdm
                    iterator = tqdm(iterator, desc="Encoding texts")
                except ImportError:
                    warnings.warn("tqdm not installed, progress bar disabled")

            for i in iterator:
                batch_texts = texts[i:i + batch_size]
                inputs = self.processor(
                    text=batch_texts, return_tensors="pt", **kwargs
                ).to(self.device)
                with torch.no_grad():
                    emb = self.model.get_text_features(**inputs)
                    emb = self.adapter(emb)
                all_embeddings.append(emb.cpu())
        else:
            raise ValueError("Must provide either `texts` or tokenized tensors.")

        embeddings = torch.cat(all_embeddings, dim=0)

        if normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)

        if convert_to_numpy:
            return embeddings.numpy()
        elif convert_to_tensor:
            return embeddings
        return embeddings.numpy()

    # ------------------------------------------------------------------
    # Audio Processing
    # ------------------------------------------------------------------
    def _prepare_audio_inputs(
        self,
        audios: Union[np.ndarray, List[np.ndarray], Dict[str, Any], List[Dict[str, Any]]],
        target_sr: int,
    ) -> List[np.ndarray]:
        if isinstance(audios, np.ndarray):
            if audios.ndim == 1:
                return [audios]
            else:
                return list(audios)

        if isinstance(audios, dict) and "array" in audios:
            return [audios["array"]]

        if isinstance(audios, list):
            result = []
            for audio in audios:
                if isinstance(audio, dict) and "array" in audio:
                    result.append(audio["array"])
                elif isinstance(audio, np.ndarray):
                    result.append(audio)
                else:
                    raise ValueError(f"Unsupported audio format: {type(audio)}")
            return result

        raise ValueError(f"Unsupported audio input type: {type(audios)}")

    def encode_audio(
        self,
        audios: Union[np.ndarray, List[np.ndarray], Dict[str, Any], List[Dict[str, Any]]],
        batch_size: int = 8,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = False,
        show_progress_bar: bool = False,
        **kwargs,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Encode audio clips with adapter."""
        self.model.eval()
        all_embeddings = []

        audios = self._prepare_audio_inputs(audios, target_sr=48000)

        iterator = range(0, len(audios), batch_size)
        if show_progress_bar:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc="Encoding audios")
            except ImportError:
                pass

        for i in iterator:
            batch_audios = audios[i:i + batch_size]
            inputs = self.processor(audios=batch_audios, return_tensors="pt").to(self.device)
            with torch.no_grad():
                emb = self.model.get_audio_features(**inputs)
                emb = self.adapter(emb)
            all_embeddings.append(emb.cpu())

        embeddings = torch.cat(all_embeddings, dim=0)

        if normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)

        if convert_to_numpy:
            return embeddings.numpy()
        return embeddings

    # ------------------------------------------------------------------
    # Similarity
    # ------------------------------------------------------------------
    def similarity(
        self,
        embeddings1: Union[np.ndarray, torch.Tensor],
        embeddings2: Union[np.ndarray, torch.Tensor],
    ) -> torch.Tensor:
        """Compute cosine similarity between embeddings."""
        if isinstance(embeddings1, np.ndarray):
            embeddings1 = torch.from_numpy(embeddings1)
        if isinstance(embeddings2, np.ndarray):
            embeddings2 = torch.from_numpy(embeddings2)

        embeddings1 = torch.nn.functional.normalize(embeddings1, p=2, dim=1)
        embeddings2 = torch.nn.functional.normalize(embeddings2, p=2, dim=1)
        return torch.mm(embeddings1, embeddings2.T)

    # ------------------------------------------------------------------
    # Forward (SentenceTransformer compatibility)
    # ------------------------------------------------------------------
    def forward(
        self,
        features: Dict[str, torch.Tensor],
        output_value: str = "sentence_embedding",
        normalize_embeddings: bool = False,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        self.model.eval()
        features = {k: v.to(self.device) for k, v in features.items() if isinstance(v, torch.Tensor)}

        with torch.no_grad():
            if output_value == "sentence_embedding":
                if "input_ids" in features:
                    emb = self.model.get_text_features(**features)
                else:
                    emb = self.model.get_audio_features(**features)
                emb = self.adapter(emb)
            else:
                if "input_ids" in features:
                    outputs = self.model.text_model(**features)
                else:
                    outputs = self.model.audio_model(**features)
                emb = outputs.last_hidden_state

        if normalize_embeddings:
            emb = torch.nn.functional.normalize(emb, p=2, dim=-1)

        return {"sentence_embedding": emb}

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------
    def encode_text(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = False,
        show_progress_bar: bool = False,
        **kwargs,
    ) -> Union[np.ndarray, torch.Tensor]:
        self.model.eval()
        if isinstance(texts, str):
            texts = [texts]
        all_embeddings = []

        iterator = range(0, len(texts), batch_size)
        if show_progress_bar:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc="Encoding texts")
            except ImportError:
                pass

        for i in iterator:
            batch = texts[i:i + batch_size]
            inputs = self.processor(
                text=batch, padding=True, truncation=True, return_tensors="pt", **kwargs
            ).to(self.device)
            with torch.no_grad():
                emb = self.model.get_text_features(**inputs)
                emb = self.adapter(emb)
            all_embeddings.append(emb.cpu())

        embeddings = torch.cat(all_embeddings, dim=0)
        if normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)

        if convert_to_numpy:
            return embeddings.numpy()
        return embeddings

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def save(self, path: str, safe_serialization: bool = True):
        """Save model and adapter."""
        self.model.save_pretrained(path, safe_serialization=safe_serialization)
        torch.save(self.adapter.state_dict(), f"{path}/adapter.pt")
        self.processor.save_pretrained(path)

    def load_adapter(self, path: str):
        """Reload a saved adapter."""
        state = torch.load(f"{path}/adapter.pt", map_location=self.device)
        self.adapter.load_state_dict(state)

    def to(self, device: Union[str, torch.device]):
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        self.adapter = self.adapter.to(self.device)
        return self

    def eval(self):
        self.model.eval()
        self.adapter.eval()
        return self

    def train(self):
        warnings.warn("Adapter is trainable; CLAP base is frozen by default.")
        self.adapter.train()
        return self

    def get_sentence_embedding_dimension(self) -> int:
        return self.model.config.projection_dim

    @property
    def embedding_dimension(self) -> int:
        return self.get_sentence_embedding_dimension()

    @property
    def tokenizer(self):
        return getattr(self.processor, "tokenizer", None)
