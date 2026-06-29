import argparse
import copy

import torch
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_causal_lm import (
    LOCAL_TINY_DATASET_NAME,
    create_causal_lm_dataloaders,
    load_text_datasets,
    tokenize_and_group_texts,
)
from utils_cluster import (
    configure_runtime,
    count_parameters,
    ensure_output_dir,
    get_device,
    safe_perplexity,
    set_seed,
    write_summary,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Manual Hugging Face causal-LM fine-tuning."
    )
    parser.add_argument("--model_checkpoint", default="distilgpt2")
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--output_dir", default="runs/lesson25_causal_lm")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--cpu_safe_generation", action="store_true")
    parser.add_argument("--dataset_name", default=LOCAL_TINY_DATASET_NAME)
    parser.add_argument("--dataset_config", default=None)
    parser.add_argument("--text_column", default="text")
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--validation_split", default="validation")
    parser.add_argument("--max_train_examples", type=int, default=None)
    parser.add_argument("--max_val_examples", type=int, default=None)
    parser.add_argument("--streaming", action="store_true")
    return parser.parse_args()


def move_batch_to_device(batch, device):
    return {
        key: value.to(device)
        for key, value in batch.items()
    }


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()

    total_loss = 0.0
    total_examples = 0

    for batch in data_loader:
        batch = move_batch_to_device(batch, device)

        outputs = model(**batch)
        loss = outputs.loss

        batch_size = batch["input_ids"].shape[0]
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    average_loss = total_loss / total_examples
    perplexity = safe_perplexity(average_loss)

    return average_loss, perplexity


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    learning_rate,
    weight_decay,
    num_epochs,
):
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_val_loss = float("inf")
    best_checkpoint = None
    global_step = 0

    for epoch in range(num_epochs):
        model.train()

        total_train_loss = 0.0
        total_train_examples = 0

        for batch in train_loader:
            global_step += 1

            batch = move_batch_to_device(batch, device)

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
        train_ppl = safe_perplexity(train_loss)

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
            best_checkpoint = {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "val_loss": val_loss,
                "val_ppl": val_ppl,
            }
            print("Saved new best checkpoint in memory.")

    return best_checkpoint


def generate_text(model, tokenizer, prompt, max_new_tokens, cpu_safe_generation):
    original_device = next(model.parameters()).device
    was_training = model.training

    if cpu_safe_generation:
        model.to("cpu")
        generation_device = torch.device("cpu")
    else:
        generation_device = original_device

    model.eval()

    old_use_cache = getattr(model.config, "use_cache", None)

    if cpu_safe_generation:
        model.config.use_cache = False

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )
    inputs = {
        key: value.to(generation_device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=not cpu_safe_generation,
        )

    text = tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True,
    )

    if old_use_cache is not None:
        model.config.use_cache = old_use_cache

    if cpu_safe_generation:
        model.to(original_device)

    if was_training:
        model.train()

    return text


def load_tokenizer_and_model(model_checkpoint, device):
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Tokenizer vocab size:", tokenizer.vocab_size)
    print("EOS token:", tokenizer.eos_token)
    print("PAD token:", tokenizer.pad_token)

    print("\nLoading pretrained causal LM...")
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)

    return tokenizer, model


def build_dataloaders(tokenizer, args):
    train_dataset, val_dataset, dataset_metadata = load_text_datasets(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        text_column=args.text_column,
        train_split=args.train_split,
        validation_split=args.validation_split,
        max_train_examples=args.max_train_examples,
        max_val_examples=args.max_val_examples,
        streaming=args.streaming,
        seed=args.seed,
    )

    print("\nRaw dataset sizes:")
    print("train examples:", len(train_dataset))
    print("validation examples:", len(val_dataset))

    print("\nTokenizing and grouping text...")
    lm_train, lm_val = tokenize_and_group_texts(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
        block_size=args.block_size,
        text_column=dataset_metadata["text_column"],
    )

    print("\nLanguage modeling dataset sizes:")
    print("train chunks:", len(lm_train))
    print("validation chunks:", len(lm_val))

    print("\nFirst chunk decoded:")
    print(tokenizer.decode(lm_train[0]["input_ids"]))

    train_loader, val_loader = create_causal_lm_dataloaders(
        lm_train=lm_train,
        lm_val=lm_val,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
    )

    dataset_metadata["train_chunks"] = len(lm_train)
    dataset_metadata["val_chunks"] = len(lm_val)

    return train_loader, val_loader, dataset_metadata


def run_dry_run(model, train_loader, device):
    print("\nDry run: inspecting one batch and running one forward pass.")
    first_batch = next(iter(train_loader))

    for key, value in first_batch.items():
        print(key, value.shape, value.dtype)

    model.eval()
    batch = move_batch_to_device(first_batch, device)

    with torch.no_grad():
        outputs = model(**batch)

    print("Dry-run loss:", outputs.loss.item())


def main():
    configure_runtime()
    args = parse_args()
    set_seed(args.seed)

    output_dir = ensure_output_dir(args.output_dir)
    device = get_device()

    print("Using device:", device)
    print("PyTorch version:", torch.__version__)
    print("\nModel checkpoint:")
    print(args.model_checkpoint)
    print("\nDataset:")
    print(args.dataset_name)
    print("\nOutput directory:")
    print(output_dir)

    tokenizer, model = load_tokenizer_and_model(
        model_checkpoint=args.model_checkpoint,
        device=device,
    )

    train_loader, val_loader, dataset_metadata = build_dataloaders(
        tokenizer=tokenizer,
        args=args,
    )

    total_params, trainable_params = count_parameters(model)

    print("\nModel:")
    print(model.__class__.__name__)
    print("Total parameters:", total_params)
    print("Trainable parameters:", trainable_params)

    if args.dry_run:
        run_dry_run(
            model=model,
            train_loader=train_loader,
            device=device,
        )
        return

    prompts = ["The robot", "The scientist", "The drone", "The assistant"]

    print("\nGeneration before fine-tuning:")
    before_samples = {}
    before_samples["The robot"] = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt="The robot",
        max_new_tokens=60,
        cpu_safe_generation=args.cpu_safe_generation,
    )
    print(before_samples["The robot"])

    print("\nEvaluating before fine-tuning...")
    val_loss_before, val_ppl_before = evaluate(
        model=model,
        data_loader=val_loader,
        device=device,
    )

    print("Validation loss before:", val_loss_before)
    print("Validation perplexity before:", val_ppl_before)

    print("\nManual fine-tuning...\n")

    best_checkpoint = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
    )

    if best_checkpoint is None:
        raise RuntimeError("Training finished without producing a checkpoint.")

    model.load_state_dict(best_checkpoint["model_state_dict"])

    print("\nLoaded best checkpoint from memory:")
    print("Epoch:", best_checkpoint["epoch"] + 1)
    print("Global step:", best_checkpoint["global_step"])
    print("Validation loss:", best_checkpoint["val_loss"])
    print("Validation perplexity:", best_checkpoint["val_ppl"])

    val_loss_after, val_ppl_after = evaluate(
        model=model,
        data_loader=val_loader,
        device=device,
    )

    print("\nValidation loss after:", val_loss_after)
    print("Validation perplexity after:", val_ppl_after)

    print("\nGeneration after fine-tuning:")
    after_samples = {}

    for prompt in prompts:
        print("\nPrompt:", prompt)
        sample = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=80,
            cpu_safe_generation=args.cpu_safe_generation,
        )
        after_samples[prompt] = sample
        print(sample)

    summary = {
        "model_checkpoint": args.model_checkpoint,
        "model_name": model.__class__.__name__,
        "hyperparameters": {
            "block_size": args.block_size,
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "cpu_safe_generation": args.cpu_safe_generation,
        },
        "parameter_counts": {
            "total": total_params,
            "trainable": trainable_params,
        },
        "dataset": dataset_metadata,
        "validation": {
            "loss_before": val_loss_before,
            "perplexity_before": val_ppl_before,
            "loss_after": val_loss_after,
            "perplexity_after": val_ppl_after,
        },
        "best_checkpoint": {
            "epoch": best_checkpoint["epoch"] + 1,
            "global_step": best_checkpoint["global_step"],
            "val_loss": best_checkpoint["val_loss"],
            "val_ppl": best_checkpoint["val_ppl"],
        },
        "generated_samples": {
            "before": before_samples,
            "after": after_samples,
        },
    }

    summary_path = write_summary(output_dir, summary)
    print("\nWrote summary:", summary_path)


if __name__ == "__main__":
    main()
