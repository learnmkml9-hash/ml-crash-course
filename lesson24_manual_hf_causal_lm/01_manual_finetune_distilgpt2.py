# Lesson 24: Manual Hugging Face Causal-LM Fine-tuning without Trainer
#
# Goal:
#   Fine-tune a pretrained GPT-style model using a manual PyTorch training loop.
#
# Compared with Lesson 23:
#   Lesson 23:
#       Trainer handled DataLoader, optimizer, training loop, evaluation,
#       logging, checkpointing.
#
#   Lesson 24:
#       We write those pieces ourselves.
#
# We still use:
#   AutoTokenizer
#   AutoModelForCausalLM
#   DataCollatorForLanguageModeling
#
# But we do NOT use:
#   Trainer
#   TrainingArguments

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import math
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)


# ------------------------------------------------------------
# 1. Device selection
# ------------------------------------------------------------

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# ------------------------------------------------------------
# 2. Tiny corpus
# ------------------------------------------------------------

def build_tiny_story_corpus():
    base_text = """
    The robot entered the laboratory and looked at the glowing console.
    The scientist asked the robot to inspect the strange signal.
    The robot noticed that the signal repeated every seven seconds.
    The scientist wrote the pattern in a notebook and smiled.

    The drone flew over the forest and mapped the river.
    The engineer asked the drone to return before sunset.
    The drone found a hidden bridge near the old trees.
    The engineer studied the map and planned a safer route.

    The rover crossed the dusty valley and climbed the red hill.
    The mission team asked the rover to search for ice.
    The rover detected a faint trace beneath the rocks.
    The mission team celebrated the discovery.

    The assistant read the question carefully and wrote a clear answer.
    The student tested the code and found a small bug.
    The assistant explained the error and suggested a simple fix.
    The student ran the program again and the result was correct.
    """

    text = "\n".join(
        line.strip()
        for line in base_text.strip().splitlines()
        if line.strip()
    )

    text = (text + "\n") * 200

    return text


# ------------------------------------------------------------
# 3. Dataset creation
# ------------------------------------------------------------

def create_text_dataset(text, train_fraction=0.9):
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    rng = np.random.default_rng(0)
    indices = rng.permutation(len(lines))

    num_train = int(train_fraction * len(lines))

    train_lines = [lines[i] for i in indices[:num_train]]
    val_lines = [lines[i] for i in indices[num_train:]]

    train_dataset = Dataset.from_dict({"text": train_lines})
    val_dataset = Dataset.from_dict({"text": val_lines})

    return train_dataset, val_dataset


# ------------------------------------------------------------
# 4. Tokenize and chunk
# ------------------------------------------------------------

def tokenize_and_group_texts(train_dataset, val_dataset, tokenizer, block_size):
    def tokenize_function(examples):
        return tokenizer(examples["text"])

    tokenized_train = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )

    tokenized_val = val_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )

    def group_texts(examples):
        concatenated = {}

        for key in examples.keys():
            concatenated[key] = sum(examples[key], [])

        total_length = len(concatenated["input_ids"])
        total_length = (total_length // block_size) * block_size

        result = {}

        for key, tokens in concatenated.items():
            result[key] = [
                tokens[i : i + block_size]
                for i in range(0, total_length, block_size)
            ]

        return result

    lm_train = tokenized_train.map(group_texts, batched=True)
    lm_val = tokenized_val.map(group_texts, batched=True)

    return lm_train, lm_val


# ------------------------------------------------------------
# 5. Manual evaluation
# ------------------------------------------------------------

@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()

    total_loss = 0.0
    total_examples = 0

    for batch in data_loader:
        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        outputs = model(**batch)
        loss = outputs.loss

        batch_size = batch["input_ids"].shape[0]
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    average_loss = total_loss / total_examples
    perplexity = math.exp(average_loss)

    return average_loss, perplexity


# ------------------------------------------------------------
# 6. Manual training loop
# ------------------------------------------------------------

def train_model(
    model,
    train_loader,
    val_loader,
    device,
    learning_rate,
    weight_decay,
    num_epochs,
    checkpoint_path,
):
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_val_loss = float("inf")

    global_step = 0

    for epoch in range(num_epochs):
        model.train()

        total_train_loss = 0.0
        total_train_examples = 0

        for batch_index, batch in enumerate(train_loader):
            global_step += 1

            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            batch_size = batch["input_ids"].shape[0]
            total_train_loss += loss.item() * batch_size
            total_train_examples += batch_size

            if global_step % 20 == 0:
                print(
                    f"Epoch {epoch + 1:2d} | "
                    f"Step {global_step:4d} | "
                    f"Batch loss = {loss.item():.4f}"
                )

        train_loss = total_train_loss / total_train_examples
        train_ppl = math.exp(train_loss)

        val_loss, val_ppl = evaluate(
            model=model,
            data_loader=val_loader,
            device=device,
        )

        print(
            f"\nEpoch {epoch + 1:2d}/{num_epochs} summary | "
            f"Train loss = {train_loss:.4f} | "
            f"Train ppl = {train_ppl:.2f} | "
            f"Val loss = {val_loss:.4f} | "
            f"Val ppl = {val_ppl:.2f}\n"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "val_ppl": val_ppl,
                },
                checkpoint_path,
            )

            print("Saved new best checkpoint.")

    return best_val_loss


# ------------------------------------------------------------
# 7. CPU-safe generation
# ------------------------------------------------------------

def generate_text(model, tokenizer, prompt, max_new_tokens):
    """
    CPU-safe greedy generation.

    This avoids MPS generation crashes on some macOS setups.
    """

    original_device = next(model.parameters()).device
    was_training = model.training

    model.to("cpu")
    model.eval()

    old_use_cache = getattr(model.config, "use_cache", None)
    model.config.use_cache = False

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=False,
        )

    text = tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True,
    )

    if old_use_cache is not None:
        model.config.use_cache = old_use_cache

    model.to(original_device)

    if was_training:
        model.train()

    return text


# ------------------------------------------------------------
# 8. Count parameters
# ------------------------------------------------------------

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return total_params, trainable_params


# ------------------------------------------------------------
# 9. Main
# ------------------------------------------------------------

def main():
    device = get_device()

    print("Using device:", device)
    print("PyTorch version:", torch.__version__)

    model_checkpoint = "distilgpt2"
    # model_checkpoint = "sshleifer/tiny-gpt2"

    print("\nModel checkpoint:")
    print(model_checkpoint)

    text = build_tiny_story_corpus()

    train_dataset, val_dataset = create_text_dataset(
        text=text,
        train_fraction=0.9,
    )

    print("\nRaw dataset sizes:")
    print("train lines:", len(train_dataset))
    print("validation lines:", len(val_dataset))

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Tokenizer vocab size:", tokenizer.vocab_size)
    print("EOS token:", tokenizer.eos_token)
    print("PAD token:", tokenizer.pad_token)

    block_size = 64

    print("\nTokenizing and grouping text...")

    lm_train, lm_val = tokenize_and_group_texts(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
        block_size=block_size,
    )

    print("\nLanguage modeling dataset sizes:")
    print("train chunks:", len(lm_train))
    print("validation chunks:", len(lm_val))

    print("\nFirst chunk decoded:")
    print(tokenizer.decode(lm_train[0]["input_ids"]))

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    train_loader = DataLoader(
        lm_train,
        batch_size=4,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=0,
    )

    val_loader = DataLoader(
        lm_val,
        batch_size=4,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=0,
    )

    print("\nInspect one batch:")

    first_batch = next(iter(train_loader))

    for key, value in first_batch.items():
        print(key, value.shape, value.dtype)

    print("\nLoading pretrained causal LM...")

    model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
    model.config.pad_token_id = tokenizer.pad_token_id

    model.to(device)

    total_params, trainable_params = count_parameters(model)

    print("\nModel:")
    print(model.__class__.__name__)
    print("Total parameters:", total_params)
    print("Trainable parameters:", trainable_params)

    print("\nGeneration before fine-tuning:")
    print(
        generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt="The robot",
            max_new_tokens=60,
        )
    )

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(
        checkpoint_dir,
        "manual_hf_causal_lm_best.pt",
    )

    learning_rate = 5e-5
    weight_decay = 0.01
    num_epochs = 3

    print("\nEvaluating before fine-tuning...")

    val_loss_before, val_ppl_before = evaluate(
        model=model,
        data_loader=val_loader,
        device=device,
    )

    print("Validation loss before:", val_loss_before)
    print("Validation perplexity before:", val_ppl_before)

    print("\nManual fine-tuning...\n")

    best_val_loss = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_epochs=num_epochs,
        checkpoint_path=checkpoint_path,
    )

    print("\nBest validation loss:", best_val_loss)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    print("\nLoaded best checkpoint:")
    print("Epoch:", checkpoint["epoch"] + 1)
    print("Global step:", checkpoint["global_step"])
    print("Validation loss:", checkpoint["val_loss"])
    print("Validation perplexity:", checkpoint["val_ppl"])

    val_loss_after, val_ppl_after = evaluate(
        model=model,
        data_loader=val_loader,
        device=device,
    )

    print("\nValidation loss after:", val_loss_after)
    print("Validation perplexity after:", val_ppl_after)

    print("\nGeneration after fine-tuning:")

    for prompt in ["The robot", "The scientist", "The drone", "The assistant"]:
        print("\nPrompt:", prompt)
        print(
            generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=80,
            )
        )


if __name__ == "__main__":
    main()