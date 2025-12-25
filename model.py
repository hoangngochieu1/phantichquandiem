import torch
import torch.nn as nn

class JointACDSPCModel(nn.Module):
    def __init__(self, num_aspects, hf_model_name=None, hf_available=True):
        super().__init__()
        self.num_aspects = num_aspects
        self.hf_available = hf_available and (hf_model_name is not None)

        if self.hf_available:
            from transformers import AutoModel
            self.encoder = AutoModel.from_pretrained(hf_model_name)
            self.pooler_dim = self.encoder.config.hidden_size
        else:
            raise RuntimeError("Fallback encoder not supported in inference mode")

        self.acd_head = nn.Sequential(
            nn.Linear(self.pooler_dim, self.pooler_dim // 2),
            nn.ReLU(),
            nn.Linear(self.pooler_dim // 2, num_aspects)
        )

        self.spc_head = nn.Sequential(
            nn.Linear(self.pooler_dim, self.pooler_dim // 2),
            nn.ReLU(),
            nn.Linear(self.pooler_dim // 2, num_aspects * 3)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state[:, 0, :]

        acd_logits = self.acd_head(pooled)
        spc_logits = self.spc_head(pooled)
        spc_logits = spc_logits.view(-1, self.num_aspects, 3)

        return acd_logits, spc_logits
