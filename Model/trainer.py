from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset
import os

def load_all_chunks(folder_path):
    files = sorted([f for f in os.listdir(folder_path) if f.startswith("chunk_") and f.endswith(".txt")])
    texts = []
    for file in files:
        with open(os.path.join(folder_path, file), "r", encoding="utf-8") as f:
            text = f.read().strip()
            if text:
                texts.append({"text": text})
    return Dataset.from_list(texts)

dataset = load_all_chunks("Data/all_llc_chunks")

model_name = "gpt2"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize, batched=True)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./llc_gpt2",
    per_device_train_batch_size=1,
    num_train_epochs=1,
    save_steps=10,
    save_total_limit=2,
    logging_steps=5,
)

trainer = Trainer(
    model = AutoModelForCausalLM.from_pretrained("gpt2"),
    tokenizer = AutoTokenizer.from_pretrained("gpt2"),
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

trainer.train()

trainer.save_model("./llc_gpt2_clean")
tokenizer.save_pretrained("./llc_gpt2_clean")
