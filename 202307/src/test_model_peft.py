from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch


model_id = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(model_id)

peft_model_id = "/groups/gcf51099/uchiyama.fumiya/llmx-2F/models/40668545/meta-llama/Llama-2-7b-hf"
peft_model = PeftModel.from_pretrained(model, peft_model_id)


tokenizer = AutoTokenizer.from_pretrained(model_id)

input = """令和三年デジタル庁令第三号
  名称: デジタル庁の所管する法令に係る情報通信技術を活用した行政の推進等に関する法律施行規則
  内容: 
  第一条"""
inputs_id = tokenizer(input, return_tensors="pt")

with torch.no_grad():
    peft_out = peft_model.generate(input_ids=inputs_id['input_ids'], max_length=128, do_sample=False, temperature=0)


print("\nllama-2-7b-hf-peft")
print(tokenizer.decode(peft_out[0]))




peft_model.base_model.model.model.layers[0].self_attn.q_proj.lora_A.default.weight
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