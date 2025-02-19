from datasets import Dataset, load_dataset
from sentence_transformers import (
    evaluation, 
    losses, 
    SentenceTransformer,  
    trainer,
    training_args
)


train_dataset = load_dataset(
    "glue",
    "mnli",
    split="train"
).select(range(50_000))
train_dataset = train_dataset.remove_columns("idx")

mapping = {2: 0, 1: 0, 0: 1}
train_dataset = Dataset.from_dict(
    {
        "sentence1": train_dataset["premise"],
        "sentence2": train_dataset["hypothesis"],
        "label": [float(mapping[label]) for label in train_dataset["label"]]
    }
)

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
    output_dir="cosineloss_embedding_model",
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