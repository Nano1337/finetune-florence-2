import os
import wandb

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AdamW, AutoModelForCausalLM, AutoProcessor,
                          get_scheduler)

from data import FilteringVQADataset

from peft import LoraConfig, get_peft_model

TARGET_MODULES = [
    "q_proj", "o_proj", "k_proj", "v_proj", 
    "linear", "Conv2d", "lm_head", "fc2"
]

# pr 6 for base, pr 10 for large
REVISION = "refs/pr/10"

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=TARGET_MODULES,
    task_type="CAUSAL_LM",
    lora_dropout=0.05,
    bias="none",
    inference_mode=False,
    use_rslora=True,
    init_lora_weights="gaussian",
    revision=REVISION
)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize wandb
wandb.init(project="florence2-finetuning")

# Load the model and processor
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Florence-2-large-ft", trust_remote_code=True, revision=REVISION
).to(device)
processor = AutoProcessor.from_pretrained(
    "microsoft/Florence-2-large-ft", trust_remote_code=True, revision=REVISION
)

peft_model = get_peft_model(model, config)
peft_model.print_trainable_parameters()

def collate_fn(batch):
    questions, answers, images = zip(*batch)
    inputs = processor(
        text=list(questions), images=list(images), return_tensors="pt", padding=True, max_length=1024, truncation=True
    ).to(device)
    return inputs, answers


# Create datasets
train_dataset = FilteringVQADataset("train")
val_dataset = FilteringVQADataset("val")

# Create DataLoader
batch_size = 8
num_workers = 0

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=collate_fn,
    num_workers=num_workers,
    shuffle=True,
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers
)


def train_model(train_loader, val_loader, model, processor, epochs=10, lr=1e-6):
    optimizer = AdamW(model.parameters(), lr=lr)
    num_training_steps = epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        i = -1
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
            i += 1
            inputs, answers = batch

            input_ids = inputs["input_ids"]
            pixel_values = inputs["pixel_values"]
            labels = processor.tokenizer(
                text=answers,
                return_tensors="pt",
                padding=True,
                return_token_type_ids=False,
            ).input_ids.to(device)

            outputs = model(
                input_ids=input_ids, pixel_values=pixel_values, labels=labels
            )
            loss = outputs.loss

            if i % 200 == 0:
                print(loss)

                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3,
                )
                generated_texts = processor.batch_decode(
                    generated_ids, skip_special_tokens=False
                )

                for generated_text, answer in zip(generated_texts, answers):
                    parsed_answer = processor.post_process_generation(
                        generated_text,
                        task="<FilteringVQA>",
                        image_size=(
                            inputs["pixel_values"].shape[-2],
                            inputs["pixel_values"].shape[-1],
                        ),
                    )
                    print("GT:", answer)
                    print("Pred:", parsed_answer["<FilteringVQA>"])

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Average Training Loss: {avg_train_loss}")
        wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss})

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(
                val_loader, desc=f"Validation Epoch {epoch + 1}/{epochs}"
            ):
                inputs, answers = batch

                input_ids = inputs["input_ids"]
                pixel_values = inputs["pixel_values"]
                labels = processor.tokenizer(
                    text=answers,
                    return_tensors="pt",
                    padding=True,
                    return_token_type_ids=False,
                ).input_ids.to(device)

                outputs = model(
                    input_ids=input_ids, pixel_values=pixel_values, labels=labels
                )
                loss = outputs.loss

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Average Validation Loss: {avg_val_loss}")
        wandb.log({"epoch": epoch + 1, "val_loss": avg_val_loss})

        # Save model checkpoint
        output_dir = f"./model_checkpoints/epoch_{epoch+1}"
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)



# run training
model = torch.compile(model)
train_model(train_loader, val_loader, peft_model, processor, epochs=3)
