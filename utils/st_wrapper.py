import torch
import numpy as np
from typing import List, Union, Optional, Dict, Any
from transformers import ClapModel, ClapProcessor
import warnings


class AudioSentenceTransformer:
    """
    A SentenceTransformer-like wrapper for CLAP audio models.
    Provides the same interface as sentence-transformers but for audio embeddings.
    """
    
    def __init__(
        self,
        model_name_or_path: str = "laion/clap-htsat-unfused",
        device: Optional[str] = None,
        cache_folder: Optional[str] = None,
        use_safetensors: bool = True,
    ):
        """
        Initialize the AudioSentenceTransformer.
        
        Args:
            model_name_or_path: HuggingFace model identifier or path
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
            cache_folder: Path to cache folder for models
            use_safetensors: Use safetensors format (required for torch < 2.6)
        """
        # Determine device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load model and processor with safetensors
        try:
            self.model = ClapModel.from_pretrained(
                model_name_or_path,
                cache_dir=cache_folder,
                use_safetensors=use_safetensors
            ).to(self.device)
        except Exception as e:
            if "torch.load" in str(e) or "weights_only" in str(e):
                print("Attempting to load with safetensors format...")
                self.model = ClapModel.from_pretrained(
                    model_name_or_path,
                    cache_dir=cache_folder,
                    use_safetensors=True,
                    ignore_mismatched_sizes=False
                ).to(self.device)
            else:
                raise e
        
        self.processor = ClapProcessor.from_pretrained(
            model_name_or_path,
            cache_dir=cache_folder
        )
        
        self.model_name = model_name_or_path
        self.max_seq_length = None  # For compatibility

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        output_value: str = 'sentence_embedding',
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        normalize_embeddings: bool = False,
        **kwargs
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Encode text inputs into embeddings using the CLAP text encoder.
        
        Args:
            texts: A string or list of text strings to encode.
            batch_size: Number of samples to encode per batch.
            show_progress_bar: Whether to display a progress bar during encoding.
            output_value: Type of embeddings to return ('sentence_embedding' or 'token_embeddings').
            convert_to_numpy: Whether to return embeddings as numpy arrays.
            convert_to_tensor: Whether to return embeddings as torch tensors.
            normalize_embeddings: Whether to normalize embeddings to unit length.
            **kwargs: Additional arguments passed to the processor.
            
        Returns:
            Embeddings as a numpy array or PyTorch tensor.
        """
        self.model.eval()

        # Ensure list input
        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []

        # Setup progress bar if requested
        iterator = range(0, len(texts), batch_size)
        if show_progress_bar:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc="Encoding texts", disable=not show_progress_bar)
            except ImportError:
                warnings.warn("tqdm not installed, progress bar disabled")

        # Process text in batches
        for i in iterator:
            batch = texts[i:i + batch_size]

            # Tokenize text batch
            inputs = self.processor(
                text=batch,
                return_tensors="pt",
                **kwargs
            ).to(self.device)

            # Forward pass
            with torch.no_grad():
                if output_value == 'sentence_embedding':
                    embeddings = self.model.get_text_features(**inputs)
                else:
                    # Return token embeddings (hidden states)
                    outputs = self.model.text_model(**inputs)
                    embeddings = outputs.last_hidden_state

            all_embeddings.append(embeddings.cpu())

        # Concatenate all batches
        embeddings = torch.cat(all_embeddings, dim=0)

        # Normalize if requested
        if normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)

        # Convert to desired format
        if convert_to_numpy:
            return embeddings.numpy()
        elif convert_to_tensor:
            return embeddings
        else:
            return embeddings.numpy()

    def _prepare_audio_inputs(
        self,
        audios: Union[np.ndarray, List[np.ndarray], Dict[str, Any], List[Dict[str, Any]]],
        target_sr: int
    ) -> List[np.ndarray]:
        """
        Prepare audio inputs into a consistent format.
        
        Args:
            audios: Raw audio inputs
            target_sr: Target sampling rate
            
        Returns:
            List of audio arrays
        """
        # Handle single audio sample
        if isinstance(audios, np.ndarray):
            if audios.ndim == 1:
                return [audios]
            else:
                return list(audios)
        
        # Handle dict format (like from datasets)
        if isinstance(audios, dict) and 'array' in audios:
            return [audios['array']]
        
        # Handle list of samples
        if isinstance(audios, list):
            result = []
            for audio in audios:
                if isinstance(audio, dict) and 'array' in audio:
                    result.append(audio['array'])
                elif isinstance(audio, np.ndarray):
                    result.append(audio)
                else:
                    raise ValueError(f"Unsupported audio format: {type(audio)}")
            return result
        
        raise ValueError(f"Unsupported audio input type: {type(audios)}")
    
    def similarity(
        self,
        embeddings1: Union[np.ndarray, torch.Tensor],
        embeddings2: Union[np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute cosine similarity between two sets of embeddings.
        
        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings
            
        Returns:
            Similarity matrix
        """
        if isinstance(embeddings1, np.ndarray):
            embeddings1 = torch.from_numpy(embeddings1)
        if isinstance(embeddings2, np.ndarray):
            embeddings2 = torch.from_numpy(embeddings2)
        
        # Normalize embeddings
        embeddings1 = torch.nn.functional.normalize(embeddings1, p=2, dim=1)
        embeddings2 = torch.nn.functional.normalize(embeddings2, p=2, dim=1)
        
        # Compute cosine similarity
        return torch.mm(embeddings1, embeddings2.transpose(0, 1))
 
    def forward(
        self,
        features: Dict[str, torch.Tensor],
        output_value: str = "sentence_embedding",
        normalize_embeddings: bool = False,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass compatible with SentenceTransformer-style pipelines.
        Expects pre-tokenized features (e.g., from a collator or dataset).

        Args:
            features: A dict of tensors containing model inputs
            output_value: Either 'sentence_embedding' or 'token_embeddings'
            normalize_embeddings: Whether to L2-normalize embeddings
            **kwargs: Extra args (ignored)
        
        Returns:
            dict: { "sentence_embedding": torch.Tensor }
        """
        self.model.eval()

        # Move inputs to the correct device
        features = {k: v.to(self.device) for k, v in features.items() if isinstance(v, torch.Tensor)}

        with torch.no_grad():
            if output_value == "sentence_embedding":
                # CLAP unified API for both text/audio encoders
                if "input_ids" in features:
                    # Text branch
                    embeddings = self.model.get_text_features(**features)
                else:
                    # Audio branch
                    embeddings = self.model.get_audio_features(**features)
            else:
                # Token-level embeddings
                if "input_ids" in features:
                    outputs = self.model.text_model(**features)
                else:
                    outputs = self.model.audio_model(**features)
                embeddings = outputs.last_hidden_state

        if normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)

        return {"sentence_embedding": embeddings}

    def encode_text(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = False,
        show_progress_bar: bool = False,
        **kwargs
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Encode text descriptions (CLAP supports audio-text matching).
        
        Args:
            texts: Text strings or list of text strings
            batch_size: Batch size for encoding
            convert_to_numpy: Convert output to numpy array
            normalize_embeddings: Normalize embeddings to unit length
            show_progress_bar: Whether to show progress bar
            **kwargs: Additional arguments
            
        Returns:
            Text embeddings
        """
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
                text=batch,
                return_tensors="pt",
                **kwargs
            ).to(self.device)
            
            with torch.no_grad():
                embeddings = self.model.get_text_features(**inputs)
            
            all_embeddings.append(embeddings.cpu())
        
        embeddings = torch.cat(all_embeddings, dim=0)
        
        if normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        
        if convert_to_numpy:
            return embeddings.numpy()
        return embeddings
    
    def __call__(self, *args, **kwargs):
        """Allow direct calling like model(audio)"""
        return self.encode(*args, **kwargs)
    
    def save(self, path: str, safe_serialization: bool = True):
        """
        Save model to path using safetensors format.
        
        Args:
            path: Directory path to save to
            safe_serialization: Use safetensors format (recommended)
        """
        self.model.save_pretrained(path, safe_serialization=safe_serialization)
        self.processor.save_pretrained(path)
    
    def to(self, device: Union[str, torch.device]):
        """Move model to device"""
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        return self
    
    def eval(self):
        """Set model to evaluation mode"""
        self.model.eval()
        return self
    
    def train(self):
        """Set model to training mode"""
        warnings.warn("AudioSentenceTransformer training not implemented")
        self.model.train()
        return self
    
    def get_sentence_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings"""
        return self.model.config.projection_dim
    
    @property
    def embedding_dimension(self) -> int:
        """Get the dimension of the embeddings"""
        return self.get_sentence_embedding_dimension()
    
    @property
    def tokenizer(self):
        """Expose tokenizer for compatibility with SentenceTransformer-like encoders."""
        return getattr(self.processor, "tokenizer", None)