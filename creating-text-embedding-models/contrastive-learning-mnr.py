import random

from datasets import Dataset, load_dataset
from sentence_transformers import (
    evaluation, 
    losses, 
    SentenceTransformer,  
    trainer,
    training_args
)
from tqdm import tqdm


mnli = load_dataset(
    "glue",
    "mnli",
    split="train"
).select(range(50_000))
mnli = mnli.remove_columns("idx")
mnli = mnli.filter(lambda x: True if x["label"] == 0 else False)

train_dataset = {
    "anchor": [],
    "positive": [],
    "negative": []
}
soft_negatives = mnli["hypothesis"]
random.shuffle(soft_negatives)
for row, soft_negatives in tqdm(zip(mnli, soft_negatives)):
    train_dataset["anchor"].append(row["premise"])
    train_dataset["positive"].append(row["hypothesis"])
    train_dataset["negative"].append(soft_negatives)
train_dataset = Dataset.from_dict(train_dataset)

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
    output_dir="mnrloss_embedding_model",
    num_train_epochs=1,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=100,
    fp16=True,
    eval_steps=100,
    logging_steps=100
)

embedding_model = SentenceTransformer("bert-base-uncased") 

train_loss = losses.MultipleNegativesRankingLoss(model=embedding_model)

trainer = trainer.SentenceTransformerTrainer(
    model=embedding_model,
    args=args,
    train_dataset=train_dataset,
    loss=train_loss,
    evaluator=evaluator
)
trainer.train()

print(evaluator(embedding_model))
