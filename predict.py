from typing import List, Optional
from cog import BasePredictor, Input
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import torch

CACHE_DIR = 'weights'


MODEL_NAME = 'tiiuae/falcon-7b'

class Predictor(BasePredictor):
    def setup(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=MODEL_NAME,
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device=0,
        )
        

    def predict(self,prompt: str = Input(description=f"Text prompt to send to the model.")) -> List[str]:
        sequences = self.pipeline(
            prompt,
            max_length=200,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        outputs = sequences[0]['generated_text']
        return outputs