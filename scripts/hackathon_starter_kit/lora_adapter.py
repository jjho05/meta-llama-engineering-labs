"""
Adaptador LoRA para Inferencia con Modelos de Lenguaje PEFT.
"""
class LoRAInferenceEngine:
    def __init__(self, base_model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", adapter_path="./lora_checkpoint"):
        self.base_model_id = base_model_id
        self.adapter_path = adapter_path
        self._tokenizer = None
        self._model = None

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel

            self._tokenizer = AutoTokenizer.from_pretrained(self.base_model_id)
            base = AutoModelForCausalLM.from_pretrained(
                self.base_model_id,
                dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto"
            )
            try:
                self._model = PeftModel.from_pretrained(base, self.adapter_path)
            except Exception:
                self._model = base

    def generate(self, prompt: str, max_new_tokens=80) -> str:
        import torch
        self._load()
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            outputs = self._model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
