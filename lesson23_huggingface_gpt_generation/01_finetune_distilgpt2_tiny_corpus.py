# Lesson 23: Fine-tuning a Pretrained GPT-style Transformer with Hugging Face
#
# Goal:
#   Fine-tune a pretrained causal language model on a tiny custom corpus.
#
# Model:
#   DistilGPT2 by default.
#
# Workflow:
#   raw text
#       -> tokenizer
#       -> token chunks
#       -> AutoModelForCausalLM
#       -> next-token prediction loss
#       -> generate text
#
# This is the Hugging Face counterpart of Lessons 19 and 21.

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import math
import numpy as np
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)


# ------------------------------------------------------------
# 1. Device information
# ------------------------------------------------------------

def print_device_info():
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if hasattr(torch.backends, "mps"):
        print("MPS available:", torch.backends.mps.is_available())

    if torch.cuda.is_available():
        print("Using CUDA if Trainer selects it.")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("Using Apple MPS if supported by Trainer.")
    else:
        print("Using CPU.")


# ------------------------------------------------------------
# 2. Tiny custom corpus
# ------------------------------------------------------------

def build_tiny_story_corpus():
    """
    Build a tiny repeated text corpus.

    The corpus is intentionally small so that fine-tuning is fast.
    It has repeated structure so that we can see generation change.
    """

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

    text = "\n".join(line.strip() for line in base_text.strip().splitlines() if line.strip())

    # Repeat the corpus so the tiny fine-tuning run has enough examples.
    text = (text + "\n") * 200

    return text


# ------------------------------------------------------------
# 3. Dataset creation
# ------------------------------------------------------------

def create_text_dataset(text, train_fraction=0.9):
    """
    Create a Hugging Face Dataset from local text.

    We split the text into sentence-like lines.
    """

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
# 4. Tokenization and chunking
# ------------------------------------------------------------

def tokenize_and_group_texts(train_dataset, val_dataset, tokenizer, block_size):
    """
    Tokenize text and group tokens into fixed-length chunks.

    For causal language modeling, we want examples like:

        input_ids: [t0, t1, ..., t_{block_size-1}]

    The labels are created by the data collator/model so the model learns
    next-token prediction.
    """

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
        """
        Concatenate tokenized text and split into chunks of block_size.
        """

        concatenated = {}

        for key in examples.keys():
            concatenated[key] = sum(examples[key], [])

        total_length = len(concatenated["input_ids"])

        # Drop the small remainder so every chunk has equal length.
        total_length = (total_length // block_size) * block_size

        result = {}

        for key, tokens in concatenated.items():
            result[key] = [
                tokens[i : i + block_size]
                for i in range(0, total_length, block_size)
            ]

        return result

    lm_train = tokenized_train.map(
        group_texts,
        batched=True,
    )

    lm_val = tokenized_val.map(
        group_texts,
        batched=True,
    )

    return lm_train, lm_val


# ------------------------------------------------------------
# 5. Generation helper
# ------------------------------------------------------------

def generate_text(model, tokenizer, prompt, max_new_tokens, temperature, top_k):
    """
    Generate text from a prompt.

    Safer version:
      - runs generation on CPU
      - disables KV cache
      - starts with greedy decoding to avoid sampling-related crashes
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
            do_sample=False,   # start safely with greedy decoding
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=False,
        )

    generated_text = tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True,
    )

    if old_use_cache is not None:
        model.config.use_cache = old_use_cache

    model.to(original_device)

    if was_training:
        model.train()

    return generated_text


# ------------------------------------------------------------
# 6. Main script
# ------------------------------------------------------------

def main():
    print_device_info()

    # --------------------------------------------------------
    # Model choice
    # --------------------------------------------------------
    # Recommended default:
    #   distilgpt2
    #
    # Faster/smaller fallback:
    #   sshleifer/tiny-gpt2
    #
    # If downloading is slow, first try sshleifer/tiny-gpt2.
    # --------------------------------------------------------

    model_checkpoint = "distilgpt2"
    # model_checkpoint = "sshleifer/tiny-gpt2"

    print("\nModel checkpoint:")
    print(model_checkpoint)

    # --------------------------------------------------------
    # Build local dataset
    # --------------------------------------------------------

    text = build_tiny_story_corpus()

    print("\nCorpus:")
    print("Number of characters:", len(text))
    print("First 500 characters:")
    print(text[:500])

    train_dataset, val_dataset = create_text_dataset(
        text=text,
        train_fraction=0.9,
    )

    print("\nRaw dataset sizes:")
    print("train lines:", len(train_dataset))
    print("validation lines:", len(val_dataset))

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # GPT-2 style tokenizers often do not have a pad token.
    # For causal LM fine-tuning, a common simple fix is to use EOS as PAD.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Tokenizer vocab size:", tokenizer.vocab_size)
    print("EOS token:", tokenizer.eos_token)
    print("PAD token:", tokenizer.pad_token)

    # --------------------------------------------------------
    # Tokenize and group into chunks
    # --------------------------------------------------------

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

    print("\nFirst tokenized chunk:")
    print(lm_train[0]["input_ids"])
    print("\nDecoded first chunk:")
    print(tokenizer.decode(lm_train[0]["input_ids"]))

    # --------------------------------------------------------
    # Data collator
    # --------------------------------------------------------
    # mlm=False means causal language modeling, not masked LM.
    # The collator creates labels from input_ids.
    # --------------------------------------------------------

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("\nLoading pretrained causal LM...")

    model = AutoModelForCausalLM.from_pretrained(model_checkpoint)

    # Ensure model knows the pad token id.
    model.config.pad_token_id = tokenizer.pad_token_id

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("\nModel:")
    print(model.__class__.__name__)
    print("Total parameters:", total_params)
    print("Trainable parameters:", trainable_params)

    # --------------------------------------------------------
    # Generation before fine-tuning
    # --------------------------------------------------------

    # prompt = "The robot"

    # print("\nGeneration before fine-tuning:")

    # before_text = generate_text(
    #     model=model,
    #     tokenizer=tokenizer,
    #     prompt=prompt,
    #     max_new_tokens=40,
    #     temperature=0.8,
    #     top_k=50,
    # )

    # print(before_text)
    print("\nSkipping generation before fine-tuning for stability.")

    # --------------------------------------------------------
    # Training setup
    # --------------------------------------------------------

    output_dir = "hf_outputs/distilgpt2_tiny_corpus"

    training_args = TrainingArguments(
        output_dir=output_dir,

        # Evaluation and checkpointing
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # Training hyperparameters
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,

        # Logging
        logging_steps=20,
        report_to="none",

        # Reproducibility
        seed=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_train,
        eval_dataset=lm_val,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    # --------------------------------------------------------
    # Evaluate before fine-tuning
    # --------------------------------------------------------

    print("\nEvaluating before fine-tuning...")

    before_metrics = trainer.evaluate()

    print("\nMetrics before fine-tuning:")
    print(before_metrics)

    if "eval_loss" in before_metrics:
        print("Perplexity before:", math.exp(before_metrics["eval_loss"]))

    # --------------------------------------------------------
    # Fine-tuning
    # --------------------------------------------------------

    print("\nFine-tuning GPT-style causal LM...")

    trainer.train()

    # --------------------------------------------------------
    # Evaluate after fine-tuning
    # --------------------------------------------------------

    print("\nEvaluating after fine-tuning...")

    after_metrics = trainer.evaluate()

    print("\nMetrics after fine-tuning:")
    print(after_metrics)

    if "eval_loss" in after_metrics:
        print("Perplexity after:", math.exp(after_metrics["eval_loss"]))

    # --------------------------------------------------------
    # Save model and tokenizer
    # --------------------------------------------------------

    save_dir = "saved_models/distilgpt2_tiny_corpus"

    os.makedirs(save_dir, exist_ok=True)

    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)

    print("\nSaved fine-tuned model to:")
    print(save_dir)

    # --------------------------------------------------------
    # Generate after fine-tuning
    # --------------------------------------------------------

    print("\nGeneration after fine-tuning:")

    prompts = [
        "The robot",
        "The scientist",
        "The drone",
        "The assistant",
    ]

    for prompt in prompts:
        generated = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=120,
            temperature=0.8,
            top_k=50,
        )

        print("\nPrompt:")
        print(prompt)
        print("Generated:")
        print(generated)

    print("\nLower-temperature generation:")

    generated_low_temp = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt="The robot",
        max_new_tokens=120,
        temperature=0.4,
        top_k=20,
    )

    print(generated_low_temp)


if __name__ == "__main__":
    main()