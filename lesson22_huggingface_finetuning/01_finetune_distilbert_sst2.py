# Lesson 22: Fine-tuning a Pretrained Transformer with Hugging Face
#
# Goal:
#   Fine-tune DistilBERT on SST-2 sentiment classification.
#
# Workflow:
#   1. Load GLUE/SST-2 dataset.
#   2. Load pretrained tokenizer.
#   3. Tokenize text examples.
#   4. Load pretrained Transformer for sequence classification.
#   5. Fine-tune using Hugging Face Trainer.
#   6. Evaluate on validation set.
#   7. Run predictions on custom sentences.

import os
import numpy as np
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
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
        print("Using CUDA GPU")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("Using Apple MPS if supported by Trainer")
    else:
        print("Using CPU")


# ------------------------------------------------------------
# 2. Metrics
# ------------------------------------------------------------

def compute_metrics(eval_pred):
    """
    Hugging Face Trainer passes:
        eval_pred.predictions: logits
        eval_pred.label_ids: integer labels

    We compute accuracy manually using NumPy.
    """

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = np.mean(predictions == labels)

    return {"accuracy": accuracy}


# ------------------------------------------------------------
# 3. Main script
# ------------------------------------------------------------

def main():
    print_device_info()

    # --------------------------------------------------------
    # Model and dataset choices
    # --------------------------------------------------------

    model_checkpoint = "distilbert-base-uncased"

    print("\nModel checkpoint:")
    print(model_checkpoint)

    # GLUE/SST-2:
    #   sentence: text
    #   label: 0 negative, 1 positive
    print("\nLoading dataset...")

    raw_datasets = load_dataset("nyu-mll/glue", "sst2") 

    print(raw_datasets)

    print("\nDataset columns:")
    print(raw_datasets["train"].column_names)

    print("\nFirst training example:")
    print(raw_datasets["train"][0])

    print("\nLabel names:")
    print(raw_datasets["train"].features["label"].names)

    # --------------------------------------------------------
    # For fast crash-course training, use small subsets.
    # You can set these to larger values later.
    # --------------------------------------------------------

    train_subset_size = 4000
    validation_subset_size = 872

    small_train_dataset = (
        raw_datasets["train"]
        .shuffle(seed=0)
        .select(range(min(train_subset_size, len(raw_datasets["train"]))))
    )

    small_validation_dataset = (
        raw_datasets["validation"]
        .shuffle(seed=0)
        .select(range(min(validation_subset_size, len(raw_datasets["validation"]))))
    )

    

    print("\nSubset sizes:")
    print("train:", len(small_train_dataset))
    print("validation:", len(small_validation_dataset))

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    def tokenize_function(examples):
        return tokenizer(
            examples["sentence"],
            truncation=True,
        )

    print("\nTokenizing datasets...")

    tokenized_train = small_train_dataset.map(
        tokenize_function,
        batched=True,
    )

    tokenized_validation = small_validation_dataset.map(
        tokenize_function,
        batched=True,
    )

    print("\nTokenized example:")
    example = tokenized_train[0]
    print("sentence:", example["sentence"])
    print("input_ids:", example["input_ids"])
    print("attention_mask:", example["attention_mask"])
    print("label:", example["label"])

    decoded = tokenizer.decode(example["input_ids"])
    print("\nDecoded input_ids:")
    print(decoded)

    # Dynamic padding:
    #   pads each batch to the longest sequence in that batch.
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("\nLoading pretrained model...")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint,
        num_labels=2,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    print("\nModel:")
    print(model.__class__.__name__)
    print("Total parameters:", total_params)
    print("Trainable parameters:", trainable_params)

    # --------------------------------------------------------
    # TrainingArguments
    # --------------------------------------------------------

    output_dir = "hf_outputs/distilbert_sst2"

    training_args = TrainingArguments(
        output_dir=output_dir,

        # Evaluation and checkpointing
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,

        # Training hyperparameters
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=2,
        weight_decay=0.01,

        # Logging
        logging_steps=50,
        report_to="none",

        # Reproducibility
        seed=0,
    )

    # --------------------------------------------------------
    # Trainer
    # --------------------------------------------------------


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_validation,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # --------------------------------------------------------
    # Baseline evaluation before fine-tuning
    # --------------------------------------------------------

    print("\nEvaluating before fine-tuning...")

    before_metrics = trainer.evaluate()

    print("\nMetrics before fine-tuning:")
    print(before_metrics)

    # --------------------------------------------------------
    # Fine-tuning
    # --------------------------------------------------------

    print("\nFine-tuning model...")

    trainer.train()

    # --------------------------------------------------------
    # Evaluation after fine-tuning
    # --------------------------------------------------------

    print("\nEvaluating after fine-tuning...")

    after_metrics = trainer.evaluate()

    print("\nMetrics after fine-tuning:")
    print(after_metrics)

    # --------------------------------------------------------
    # Save model and tokenizer
    # --------------------------------------------------------

    save_dir = "saved_models/distilbert_sst2"

    os.makedirs(save_dir, exist_ok=True)

    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)

    print("\nSaved fine-tuned model to:")
    print(save_dir)

    # --------------------------------------------------------
    # Custom predictions
    # --------------------------------------------------------

    print("\nCustom predictions:")

    test_sentences = [
        "This movie was surprisingly thoughtful and beautifully acted.",
        "The plot was boring and the characters were completely unconvincing.",
        "I would not recommend this film to anyone.",
        "A charming and emotionally satisfying story.",
    ]

    model.eval()

    inputs = tokenizer(
        test_sentences,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    # Move inputs to same device as model.
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(probabilities, dim=-1)

    label_names = raw_datasets["train"].features["label"].names

    for sentence, pred, probs in zip(
        test_sentences,
        predictions.cpu().numpy(),
        probabilities.cpu().numpy(),
    ):
        print("\nSentence:")
        print(sentence)
        print("Predicted label:", label_names[pred])
        print("Probabilities:", probs)


if __name__ == "__main__":
    main()