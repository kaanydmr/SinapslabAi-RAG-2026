import os
import glob
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# --- AYARLAR ---
# --- AYARLAR ---
max_seq_length = 2048
model_name = "unsloth/Qwen2.5-7B-bnb-4bit"

# DİKKAT: Yolları resmi Docker yapısına (/workspace) göre güncelledik
output_dir = "/workspace/output_model"
data_dir = "/workspace/training_data"

# --- MODEL YÜKLEME ---
print(f"Model yükleniyor: {model_name}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# --- VERİ SETİ (GÜNCELLENEN KISIM) ---
alpaca_prompt = """### Instruction:
{}

### Input:
{}

### Response:
{}""" + tokenizer.eos_token

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        texts.append(alpaca_prompt.format(instruction, input, output))
    return { "text" : texts, }

# Klasördeki tüm .json dosyalarını bul
json_files = glob.glob(os.path.join(data_dir, "*.json"))

if not json_files:
    raise ValueError(f"HATA: '{data_dir}' klasöründe hiç .json dosyası bulunamadı!")

print(f"Bulunan dosya sayısı: {len(json_files)}. Birleştiriliyor...")

# Hepsini tek seferde yükle
dataset = load_dataset("json", data_files=json_files, split="train")
print(f"Toplam eğitim verisi satır sayısı: {len(dataset)}")

dataset = dataset.map(formatting_prompts_func, batched = True,)

# --- EĞİTİM ---
print("Eğitim başlıyor...")
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 1,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        output_dir = "checkpoints",
    ),
)

trainer.train()

# --- KAYDETME ---
print("Model GGUF formatına çevriliyor...")
model.save_pretrained_gguf(output_dir, tokenizer, quantization_method = "q4_k_m")
print(f"BİTTİ! Dosyalar '{output_dir}' klasörüne kaydedildi.")