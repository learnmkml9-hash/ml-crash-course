import numpy as np
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling


LOCAL_TINY_DATASET_NAME = "tiny_stories_local"


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

    return (text + "\n") * 200


def create_text_dataset(text, train_fraction=0.9, seed=0):
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(lines))

    num_train = int(train_fraction * len(lines))

    train_lines = [lines[i] for i in indices[:num_train]]
    val_lines = [lines[i] for i in indices[num_train:]]

    train_dataset = Dataset.from_dict({"text": train_lines})
    val_dataset = Dataset.from_dict({"text": val_lines})

    return train_dataset, val_dataset


def _dataset_length(dataset):
    try:
        return len(dataset)
    except TypeError:
        return None


def _select_examples(dataset, max_examples, seed):
    if max_examples is None:
        return dataset

    max_examples = min(max_examples, len(dataset))

    return (
        dataset
        .shuffle(seed=seed)
        .select(range(max_examples))
    )


def _materialize_streaming_dataset(dataset, text_column, max_examples):
    if max_examples is None:
        raise ValueError(
            "--streaming requires --max_train_examples and --max_val_examples "
            "so the iterable dataset can be materialized for token grouping."
        )

    rows = []

    for example in dataset.take(max_examples):
        rows.append(example[text_column])

    return Dataset.from_dict({text_column: rows})


def load_text_datasets(
    dataset_name,
    dataset_config,
    text_column,
    train_split,
    validation_split,
    max_train_examples,
    max_val_examples,
    streaming,
    seed,
):
    if dataset_name == LOCAL_TINY_DATASET_NAME:
        text = build_tiny_story_corpus()
        train_dataset, val_dataset = create_text_dataset(
            text=text,
            train_fraction=0.9,
            seed=seed,
        )

        train_dataset = _select_examples(
            dataset=train_dataset,
            max_examples=max_train_examples,
            seed=seed,
        )
        val_dataset = _select_examples(
            dataset=val_dataset,
            max_examples=max_val_examples,
            seed=seed,
        )

        metadata = {
            "dataset_name": dataset_name,
            "dataset_config": dataset_config,
            "text_column": "text",
            "train_split": "local_train",
            "validation_split": "local_validation",
            "streaming": False,
            "max_train_examples": max_train_examples,
            "max_val_examples": max_val_examples,
            "raw_train_examples": len(train_dataset),
            "raw_val_examples": len(val_dataset),
        }

        return train_dataset, val_dataset, metadata

    load_kwargs = {
        "path": dataset_name,
        "split": train_split,
        "streaming": streaming,
    }

    if dataset_config:
        load_kwargs["name"] = dataset_config

    train_dataset = load_dataset(**load_kwargs)

    load_kwargs["split"] = validation_split
    val_dataset = load_dataset(**load_kwargs)

    if streaming:
        train_dataset = _materialize_streaming_dataset(
            dataset=train_dataset,
            text_column=text_column,
            max_examples=max_train_examples,
        )
        val_dataset = _materialize_streaming_dataset(
            dataset=val_dataset,
            text_column=text_column,
            max_examples=max_val_examples,
        )
    else:
        train_dataset = _select_examples(
            dataset=train_dataset,
            max_examples=max_train_examples,
            seed=seed,
        )
        val_dataset = _select_examples(
            dataset=val_dataset,
            max_examples=max_val_examples,
            seed=seed,
        )

    metadata = {
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "text_column": text_column,
        "train_split": train_split,
        "validation_split": validation_split,
        "streaming": streaming,
        "max_train_examples": max_train_examples,
        "max_val_examples": max_val_examples,
        "raw_train_examples": _dataset_length(train_dataset),
        "raw_val_examples": _dataset_length(val_dataset),
    }

    return train_dataset, val_dataset, metadata


def tokenize_and_group_texts(
    train_dataset,
    val_dataset,
    tokenizer,
    block_size,
    text_column="text",
):
    def tokenize_function(examples):
        return tokenizer(examples[text_column])

    tokenized_train = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names,
    )

    tokenized_val = val_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=val_dataset.column_names,
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


def create_causal_lm_dataloaders(lm_train, lm_val, tokenizer, batch_size):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    train_loader = DataLoader(
        lm_train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=0,
    )

    val_loader = DataLoader(
        lm_val,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=0,
    )

    return train_loader, val_loader
