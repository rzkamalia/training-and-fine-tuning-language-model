import nltk

from datasets import Dataset, load_dataset
from sentence_transformers import (
    datasets,
    evaluation, 
    losses, 
    models,
    SentenceTransformer,  
    trainer,
    training_args
)
from tqdm import tqdm


nltk.download("punkt_tab")

mnli = load_dataset(
    "glue",
    "mnli",
    split="train"
).select(range(25_000))
flat_sentences = mnli["premise"] + mnli["hypothesis"]

damaged_data = datasets.DenoisingAutoEncoderDataset(list(set(flat_sentences)))

train_dataset = {
    "damaged_sentence": [],
    "original_sentence": []
}
for data in tqdm(damaged_data):
    train_dataset["damaged_sentence"].append(data.texts[0])
    train_dataset["original_sentence"].append(data.texts[1])
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
    output_dir="tsdae_embedding_model",
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=100,
    fp16=True,
    eval_steps=100,
    logging_steps=100
)

word_embedding_model = models.Transformer("bert-base-uncased")
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), "cls")
embedding_model = SentenceTransformer(modules=[word_embedding_model, pooling_model]) 

train_loss = losses.DenoisingAutoEncoderLoss(embedding_model, tie_encoder_decoder=True)
train_loss.decoder = train_loss.decoder.to("cuda")

trainer = trainer.SentenceTransformerTrainer(
    model=embedding_model,
    args=args,
    train_dataset=train_dataset,
    loss=train_loss,
    evaluator=evaluator
)
trainer.train()

print(evaluator(embedding_model))
