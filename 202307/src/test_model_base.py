from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


model_id = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(model_id)

tokenizer = AutoTokenizer.from_pretrained(model_id)

input = """令和三年デジタル庁令第三号
  名称: デジタル庁の所管する法令に係る情報通信技術を活用した行政の推進等に関する法律施行規則
  内容: 
  第一条"""
inputs_id = tokenizer(input, return_tensors="pt")

with torch.no_grad():
    out = model.generate(inputs_id['input_ids'], max_length=128, do_sample=False, temperature=0)

print("\nllama-2-7b-hf")
print(tokenizer.decode(out[0]))
"""
PeftModelForCausalLM(
  (base_model): LoraModel(
    (model): LlamaForCausalLM(
      (model): LlamaModel(
        (embed_tokens): Embedding(32000, 4096, padding_idx=0)
        (layers): ModuleList(
          (0-31): 32 x LlamaDecoderLayer(
            (self_attn): LlamaAttention(
              (q_proj): Linear(
                in_features=4096, out_features=4096, bias=False
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.01, inplace=False)
                )
"""