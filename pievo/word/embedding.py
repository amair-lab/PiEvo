import os
import torch
import torch.nn.functional as F
from typing import List


class EmbeddingService:
    def __init__(self):
        self.embedding_cache = {}
        self.local_tokenizer = None
        self.local_model = None

    @staticmethod
    def _last_token_pool(last_hidden_states, attention_mask):
        """
        Performs last token pooling on the hidden states.
        This is the strategy recommended for the Qwen embedding model.
        """
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def _initialize_local_model(self):
        """
        Lazily initializes and loads the local Hugging Face model and tokenizer
        the first time they are needed.
        """
        if self.local_tokenizer is None or self.local_model is None:
            from transformers import AutoTokenizer, AutoModel

            # Define the model name
            model_name = 'Qwen/Qwen3-Embedding-0.6B'

            # Standard CPU/GPU loading
            self.local_tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
            self.local_model = AutoModel.from_pretrained(model_name)

            os.environ["http_proxy"] = ""
            os.environ["https_proxy"] = ""

    def embed_with_local_model(self, sentence: str) -> List[float]:
        """
        Generates an embedding using the local Qwen-Embedding model.
        """
        if sentence in self.embedding_cache:
            return self.embedding_cache[sentence]
        
        self._initialize_local_model()

        max_length = 8192  # Max sequence length for the model

        # Tokenize the input text. Note: it's processed as a batch of one.
        batch_dict = self.local_tokenizer(
            [sentence],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        # Move tensors to the same device as the model
        batch_dict.to(self.local_model.device)

        # Run model inference
        with torch.no_grad():
            outputs = self.local_model(**batch_dict)

        # Pool the output hidden states to get the final embedding
        embedding = self._last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

        # Normalize the embedding to a unit vector, as is common practice
        normalized_embedding = F.normalize(embedding, p=2, dim=1)

        # Save the model for embedding cache.
        self.embedding_cache[sentence] = normalized_embedding[0].cpu().tolist()
        return normalized_embedding[0].cpu().tolist()

    def embed_with_llm(self, sentence: str) -> List[float]:
        """
        NOTE: We do not use this LLM embedding due to high cost, we only use the local LLM embedding.
        """
        from openai import OpenAI

        # first match in the cache for avoiding duplicated costing.
        if sentence in self.embedding_cache:
            return self.embedding_cache[sentence]
        
        embedding = (
            OpenAI(
                base_url=os.environ["EMBEDDING_MODEL_URL"],
                api_key=os.environ["EMBEDDING_MODEL_API_KEY"],
            )
            .embeddings.create(
                model=os.environ["EMBEDDING_MODEL_NAME"],
                input=sentence,
                dimensions=int(os.environ["EMBEDDING_MODEL_DIMENSIONS"]),
                encoding_format="float",
            )
            .data[0]
            .embedding
        )
        self.embedding_cache[sentence] = embedding
        return embedding