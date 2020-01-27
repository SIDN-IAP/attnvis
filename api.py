import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
)


class AttentionGetter:
    def __init__(self, model_name: str):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_name, output_attentions=True).to(
            self.device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bos_token_id = self.tokenizer.bos_token_id

    def analyze_text(self, text: str):
        # Process Input
        toked = self.tokenizer.encode(text)
        # Convert to PyTorch Tensor
        start_token = torch.full(
            (1, 1), self.bos_token_id, device=self.device, dtype=torch.long,
        )
        context = torch.tensor(toked, device=self.device, dtype=torch.long).unsqueeze(0)
        context = torch.cat([start_token, context], dim=1)
        # Run Forward Pass
        model_output = self.model(context)
        # Grab the attention from the output
        # Format as Layer x Head x From x To
        attn = torch.cat([l for l in model_output[2]], dim=0)
        return {
            "tokens": self.tokenizer.convert_ids_to_tokens(context[0][1:]),
            "attention": attn.cpu().tolist(),
        }


if __name__ == "__main__":
    model = AttentionGetter("gpt2")
    payload = model.analyze_text("This is a test.")
    print(payload)
    print("checking successful!")
