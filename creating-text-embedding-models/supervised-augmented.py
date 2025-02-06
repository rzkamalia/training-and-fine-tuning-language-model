import numpy as np
import pandas as pd

from datasets import Dataset, load_dataset
from sentence_transformers import (
    cross_encoder,
    datasets,
    evaluation, 
    InputExample,
    losses, 
    SentenceTransformer,  
    trainer,
    training_args
)
from tqdm import tqdm


dataset = load_dataset(
    "glue",
    "mnli",
    split="train"
).select(range(10_000))

mapping = {2: 0, 1: 0, 0: 1}

gold_examples = [
    InputExample(
        texts=[
            row["premise"],
            row["hypothesis"]
        ],
        label=mapping[row["label"]]
    )
    for row in tqdm(dataset)
]
gold_dataloader = datasets.NoDuplicatesDataLoader(gold_examples, batch_size=32)

gold = pd.DataFrame(
    {
        "sentence1": dataset["premise"],
        "sentence2": dataset["hypothesis"],
        "label": [mapping[label] for label in dataset["label"]]
    }
)

# STEP 1: fine-tune a cross-encoder (BERT) using gold dataset.
cross_encoder = cross_encoder.CrossEncoder("bert-base-uncased", num_labels=2)
cross_encoder.fit(
    train_dataloader=gold_dataloader,
    epochs=1,
    show_progress_bar=True,
    warmup_steps=100,
    use_amp=False
)

# STEP 2: create new sentence pairs called silver dataset.
silver = load_dataset(
    "glue",
    "mnli",
    split="train"
).select(range(10_000, 50_000))
pairs = list(zip(silver["premise"], silver["hypothesis"]))

# STEP 3: label silver dataset with fine-tuned cross-encoder
output = cross_encoder.predict(
    pairs,
    apply_softmax=True,
    show_progress_bar=True
)

silver = pd.DataFrame(
    {
        "sentence1": silver["premise"],
        "sentence2": silver["hypothesis"],
        "label": np.argmax(output, axis=1)
    }
)

# STEP 4: train bi-encoder on the gold + silver dataset.
data = pd.concat([gold, silver], ignore_index=True, axis=0)
data = data.drop_duplicates(subset=["sentence1", "sentence2"], keep="first")
train_dataset = Dataset.from_pandas(data, preserve_index=False)

val_stsb = load_dataset(
    "glue",
    "stsb",
    split="validation"
)
evaluator = evaluation.EmbeddingSimilarityEvaluator(
    sentences1=val_stsb["sentence1"],
    sentences2=val_stsb["sentence2"],
    scores=val_stsb["label"],
    main_similarity="cosine"
)

args = training_args.SentenceTransformerTrainingArguments(
    output_dir="augmented_embedding_model",
    num_train_epochs=1,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=100,
    fp16=True,
    eval_steps=100,
    logging_steps=100
)

embedding_model = SentenceTransformer("bert-base-uncased") 

train_loss = losses.CosineSimilarityLoss(model=embedding_model)

trainer = trainer.SentenceTransformerTrainer(
    model=embedding_model,
    args=args,
    train_dataset=train_dataset,
    loss=train_loss,
    evaluator=evaluator
)
trainer.train()

print(evaluator(embedding_model))
