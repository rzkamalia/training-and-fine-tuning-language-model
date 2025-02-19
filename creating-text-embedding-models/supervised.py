from datasets import load_dataset
from sentence_transformers import (
    evaluation, 
    losses, 
    SentenceTransformer, 
    training_args, 
    trainer
)


train_dataset = load_dataset(
    "glue",
    "mnli",
    split="train"
).select(range(50_000))
train_dataset = train_dataset.remove_columns("idx")

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

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

train_loss = losses.MultipleNegativesRankingLoss(model=embedding_model)

args = training_args.SentenceTransformerTrainingArguments(
    output_dir="finetuned_embedding_model",
    num_train_epochs=1,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=100,
    fp16=True,
    eval_steps=100,
    logging_steps=100
)

trainer = trainer.SentenceTransformerTrainer(
    model=embedding_model,
    args=args,
    train_dataset=train_dataset,
    loss=train_loss,
    evaluator=evaluator
)
trainer.train()

print(evaluator(embedding_model))