# Creating Text Embedding Models

Unstructured textual data by itself is often quite hard to process. They are not values we can directly process, visualize, and create actionable results from. We first have to convert this textual data to something that we can easily process: numeric representations. This process is often referred to as embedding the input to output usable vectors, namely *embeddings*. This process of embedding the input is typically performed by an LLM, which we refer to as an *embedding model*. **The main purpose of such a model is to be as accurate as possible in representing the textual data as an embedding**.

The definition of accurate in representing means that we expect vectors of documents that are similar to one another to be similar, whereas the embeddings of documents that each discuss something entirely different should be dissimilar. 

There are many ways in which we can train, fine-tune, and guide embedding models, but one of the strongest and most widely used techniques is called *contrastive learning*.

## Contrastive Learning
One major technique for both training and fine-tuning text embedding models is called contrastive learning. Contrastive learning is a technique that aims to train an embedding model such that similar documents are closer in vector space while dissimilar documents are further apart. The underlying idea of contrastive learning is that the best way to learn and model similarity/dissimilarity between documents is by feeding a model examples of similar and dissimilar pairs.

Although there are many forms of contrastive learning, one framework that has popularized the technique within the natural language processing community is `sentence-transformers`. Before `sentence-transformers`, sentence embeddings often used an architectural structure called cross-encoders with BERT.

|![cross-encoder](/creating-text-embedding-models/assets/cross-encoder.jpeg)| 
|:-:| 
|Figure 1. The cross-encoder architecture. Both sentences are concatenated, separated with <SEP> token, and fed to the model simultaneously. (source: [1])|

A cross-encoder processes two sentences together through a Transformer network to predict their similarity. It achieves this by adding a classification head to the model, which outputs a similarity score. However, this approach becomes computationally expensive when dealing with large collections. For example, comparing every pair in a set of 10,000 sentences would require n⋅(n−1)/2 = 49,995,000 inferences, resulting in significant computational overhead. Additionally, cross-encoders do not generate standalone embeddings for the input sentences, as illustrated in Figure 1. Instead, they directly output a similarity score for each pair of sentences.

Unlike a cross-encoder, in `sentence-transformers` the classification head is dropped, and instead mean pooling is used on the final output layer to generate an embedding. This pooling layer averages the word embeddings and gives back a fixed dimensional output vector. This ensures a fixed-size embedding. 

|![bi-encoder](/creating-text-embedding-models/assets/bi-encoder.jpeg)| 
|:-:| 
|Figure 2. An original architecture of `sentence-transformers` model, which leverages a Siamese network, also called a bi-encoder. (source: [1])|

The training for `sentence-transformers` uses a Siamese architecture. In this architecture, as visualized in Figure 2, we have two identical BERT models that share the same weights and neural architecture. These models are fed the sentences from which embeddings are generated through the pooling of token embeddings. Then, models are optimized through the similarity of the sentence embeddings. Since the weights are identical for both BERT models, we can use a single model and feed it the sentences one after the other.

The optimization process of these pairs of sentences is done through loss function which can have a major impact on the model's performance. During training, the embeddings for each sentence are concatenated together with the difference between the embeddings. Then, this resulting embedding is optimized through a softmax classifier. 

The resulting architecture is also referred to as a bi-encoder or SBERT for sentence-BERT. Although a bi-encoder is quite fast and create accurate sentence representations, cross-encoder generally achieve better performance than a bi-encoder but do not generate embeddings. 

Both, cross-encoder and bi-encoder, leverages contrastive learning. To perform contrastive learning, we need two things:
1. We need data that constitutes similar/dissimilar pairs.
2. We will need to define how the model defines and optimize similarity.

## Create an Embedding Model using Contrastive Learning

### Generate contrastive learning
When pretraining our embedding model, we will often see data being used from natural language (NLI) datasets. NLI refers to the task of investigating whether, for given premise, it entails the hypothesis (entailment), contradicts it (contradiction), or neither (neutral). The example of NL dataset shown in Figure 3. 

|![nli-example](/creating-text-embedding-models/assets/nli-example.jpeg)| 
|:-:| 
|Figure 3. The example of the NLI dataset. (source: [1])|

The data that we are going to be using throughout creating and fine-tuning embedding models is derived from the General Language Understanding Evaluation benchmark (GLUE). The GLUE benchmark is a collection of resources for training, evaluating, and analyzing natural language understanding systems. One of these collections is Multi-Genre Natural Language Inference (MNLI) corpus, which is a collection pairs annotated with entailment (contradiction, neutral, entailment). 

Here is an example of data from MNLI that we will use. 
```
{
    'premise': 'Conceptually cream skimming has two basic dimensions - product and geography.',
    'hypothesis': 'Product and geography are what make cream skimming work. ',
    'label': 1
}
```

The MNLI dataset from GLUE contains three labels: entailment (0), neutral (1), and contradiction (2). The example above shows label 1, which indicates a neutral relationship between the premise and the hypothesis.

We will use only 50,000 sentence pairs from the MNLI dataset. The code for training the MNLI dataset using the softmax loss function can be found in [contrastive-learning-softmax.py](contrastive-learning-softmax.py). Below are the training process and evaluation results.

|![contrastive-softmax](/creating-text-embedding-models/assets/results/contrastive-softmax.png)| 
|:-:| 
|Figure 4. Training proses and evaluation results of contrastive learning using the softmax loss function.|

### Using cosine similarity loss function
Instead of having strictly positive and negative pairs of sentences, we assume pairs of sentences that are similar and dissimilar to certain degree. Typically, this value lies between 0 and 1 to indicate dissimilarity and similarity, respectively. Cosine similarity loss function straightforward - it calculates the cosine similarity between the two embeddings of the two texts and compares that to the labeled similarity score. The model will learn to recognize the degree of similarity between sentences. 

To apply this loss function to our NLI dataset, we need to convert the labels for entailment (0), neutral (1), and contradiction (2) into values between 0 and 1. Entailment represents a high similarity between sentences, so it is assigned a similarity score of 1. In contrast, both neutral and contradiction represent dissimilarity, and are therefore given a similarity score of 0. 

Here is an example of data that has been converted.
```
{
    'sentence1': 'Conceptually cream skimming has two basic dimensions - product and geography.', 
    'sentence2': 'Product and geography are what make cream skimming work. ', 
    'label': 0.0
}
```

The code for training the MNLI dataset using the cosine similarity loss function can be found in [contrastive-learning-cosine.py](contrastive-learning-cosine.py). Below are the training process and evaluation results.

|![contrastive-cosine](/creating-text-embedding-models/assets/results/contrastive-cosine.png)| 
|:-:| 
|Figure 5. Training process and evaluation results of contrastive learning using the cosine similarity loss function.|

### Using multiple negatives ranking loss function
Multiple Negatives Ranking (MNR) loss function is a loss that uses either positives pairs of sentence or triplets that contain a pair of positives sentences and an additional unrelated sentence. This unrelated sentence is called a negative and represents the dissimilarity between the positives sentences.

For example, we might have pairs of question-answer, image-image caption, paper title-paper abstract. The great thing about these pairs is that we can be confident they are hard positive pairs. In MNR loss function, negative pairs are constructed by mixing a positive pair with another positive pair. In the example of paper title and abstract we would generate a negative pair by combining the title of a paper with a completely different abstract. These negatives are called *in-batch negatives* and can also be used to generate the triplets. 

Once the positive and negative pairs are generated, we calculate their embeddings and compute the cosine similarity. These similarity scores are then used to determine whether the pairs are negative or positive. In other words, this is treated as a classification task, where we can optimize the model using cross-entropy loss.

To make these triplets we start with an anchor sentence (i.e., labeled as the "premise") which is used to compare other sentences. Then using the MNLI dataset we only select sentence pairs that are positive (i.e., labeled as "entailment"). To add negative sentences, we randomly sample sentences as the "hypothesis". 

Here is the data used for training with the MNR loss function.
```
{
    'anchor': 'you know during the season and i guess at at your level uh you lose them to the next level if if they decide to recall the the parent team the Braves decide to call to recall a guy from triple A then a double A guy goes up to replace him and a single A guy goes up to replace him',
    'positive': 'You lose the things to the following level if the people recall.',
    'negative': 'She pulled back at the last moment.'
}
```

The code for training the MNLI dataset using the MNR loss function can be found in [contrastive-learning-mnr.py](contrastive-learning-mnr.py). Below are the training process and evaluation results.

|![contrastive-mnr](/creating-text-embedding-models/assets/results/contrastive-mnr.png)| 
|:-:| 
|Figure 6. Training process and evaluation results of contrastive learning using the MNR loss function.|

There is a downside to how we used this loss function. Since negatives are sampled from other question-answer pairs, these in-batch or "easy" negatives that used cloud potentially be completely unrelated to the question. As a result, the embedding model's task of then finding the right answer to a question becomes quite easy. Instead, we would like to have negatives that are very related to the question but not the right answer. These negatives are called *hard negatives*. Since this would make the task more difficult for the embedding model as it has to learn more nuanced representations, the embedding model's performance generally improve quite a bit. A good example of hard negative is the following.

|![hard-negative](/creating-text-embedding-models/assets/hard-negative.jpeg)| 
|:-:| 
|Figure 7. An example of easy, semi-hard, and hard negatives. (source: [1])|

+ Easy negatives: through randomly sampling documents as we did Before.
+ Semi-hard negatives: using a pretrained embedding model, we can apply cosine similarity on all sentence embeddings to find those that are highly related. Generally, this does not lead to hard negatives since method merely finds similar sentences, not question-answer pairs. 
+ Hard negatives: these often need to be either manually labeled or you can use a generative model to either judge or generate sentence pairs.

## Fine-Tuning an Embedding Model
There are several ways to fine-tune your model, depending on data availability and the specific domain. In this book, we will explore two such methods and demonstrate the advantages of using pretrained embedding models.

### Supervised
We will use the same data as we used to train model using MNR loss. The code for fine-tune using the supervised method can be found in [supervised.py](supervised.py). Below are the training process and evaluation results.

|![supervised](/creating-text-embedding-models/assets/results/supervised.png)| 
|:-:| 
|Figure 8. Training process and evaluation results using supervised method.|

Training from scratch or fine-tuning embedding models often requires substantial data, with many models being trained on over a billion sentence pairs. However, extracting such a large number of pairs for specific use cases is typically impractical, as only a few thousand labeled data points are often available. Fortunately, a method called *Augmented SBERT* enables training embedding models even with limited labeled data.

|![augmented-sbert](/creating-text-embedding-models/assets/augmented-sbert.jpeg)| 
|:-:| 
|Figure 9. Augmented SBERT flow. (source: [1])|

Augmented SBET involves the following steps:
1. Fine-tune a cross-encoder (BERT) using a small annotated dataset (gold dataset).
2. Create new sentence pairs.
3. Label new sentence pairs with fine-tuned cross-encoder (silver dataset).
4. Train a bi-encoder (SBERT) on the extended dataset (gold + silver dataset).

The model was fine-tuned using a supervised method with 50,000 sentence pairs from the MNLI dataset. For augmentation, only 10,000 pairs were used to simulate limited annotated data. 
The code for augmented SBERT can be found in [supervised-augmented.py](supervised-augmented.py). We explain step-by-step in our code. Below are the training process and evaluation results.

|![supervised-augmented-20](/creating-text-embedding-models/assets/results/supervised-augmented-20.png)| 
|:-:| 
|Figure 10. Training process and evaluation results using Augmented SBERT.|

The previous augmentation used 10,000 pairs; we also experimented with 25,000 sentence pairs. Below are the training process and evaluation results with 25,000 sentence pairs.

|![supervised-augmented-50](/creating-text-embedding-models/assets/results/supervised-augmented-50.png)| 
|:-:| 
|Figure 11. Training process and evaluation results using Augmented SBERT with 25,000 sentence pairs.|

### Unsupervised
Not all real-world datasets come with a convenient set of labels. Instead, we explore techniques to train the model without predetermined labels, such as unsupervised learning. One approach is the Transformer-based Sequential Denoising Auto-Encoder (TSDAE).

TSDAE assumes that we have no labeled data at all and does not require us to artificially create labels. The underlying idea of TSDAE is that we add noise to the input sentence by removing a certain percentage of words from it. This "damaged" sentence is put through an encoder, with a pooling layer on top of it, to map it to a sentence embedding. From this sentence embedding, a decoder tries to reconstruct the original sentence from the "damaged" sentence but without the artificial noise. The main concept here is that the more accurate the sentence embedding is, the more accurate the reconstructed sentence will be.

This method is very similar to masked language modeling, where we try to reconstruct and learn certain masked words. Here instead of reconstructing masked words we try to reconstruct the entire sentence.

After training, we can use the encoder to generate embeddings from text since the decoder is only used for judging whether the embeddings can accurately reconstruct the original sentence. 

|![tsdae](/creating-text-embedding-models/assets/tsdae.jpeg)| 
|:-:| 
|Figure 12. Transformer-based Sequential Denoising Auto-Encoder flow. (source: [1])|

For the experiment, we (still) use the dataset from MNLI. Here is an example of data after we demaged the dataset.
```
{
    'damaged_sentence': 'I immediately tail, fast could opposite',
    'original_sentence': 'I immediately turned tail and ran, heading fast as I could in the opposite direction.'
}
```

Next, we run the training as before but with the [CLS] token as the pooling strategy instead of the mean pooling of the token embeddings. In the TSDAE paper, this was shown in to be more effective since mean pooling loses the position information, which is not the case when using then [CLS] token:

```
word_embedding_model = models.Transformer("bert-base-uncased")
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), "cls")
embedding_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
```

Using our sentence pairs, we will need a loss function that attempts to reconstruct the original sentence using the noise sentence, namely `DenoisingAutoEncoderLoss`. By doing so, it will learn how to accurately represent the data. It is similar to masking but without knowing where the actual masks are. Moreover, we tie the parameters of both models. Instead of having separate weights for the encoder's embedding layer and the decoder's output layer, they share the same weights. This means that any updates to the weights in one layer will be reflected in the other layer as well: 

```
train_loss = losses.DenoisingAutoEncoderLoss(embedding_model, tie_encoder_decoder=True)
```

Full code for this experiment can be seen in [unsupervised.py](unsupervised.py). Below are the training process and evaluation results.

|![unsupervised](/creating-text-embedding-models/assets/results/unsupervised.png)| 
|:-:| 
|Figure 13. Training process and evaluation results using unsupervised method.|

### Adaptive pretraining
When limited or no labeled data is available, unsupervised learning is often used to create text embedding models. However, unsupervised methods generally perform worse than supervised ones and struggle to capture domain-specific concepts. This challenge is addressed through *domain adaptation*, which updates existing embedding models to better align with a specific textual domain that differs from the source domain. The target domain often includes unique words and topics not present in the source domain. A common approach for domain adaptation is *adaptive pretraining*, where a domain-specific corpus is first pretrained using unsupervised techniques like TSDAE. The pretrained model is then fine-tuned with a training dataset, ideally from the target domain, though out-domain data can also work effectively since the initial unsupervised training is specific to the target domain.

|![adaptive](/creating-text-embedding-models/assets/adaptive.jpeg)| 
|:-:| 
|Figure 14. Domain adaptation can be perfomed with adaptive pretraining. (source[1])|

# Reference
[1] Alammar, J., & Grootendorst, M. (2024). *Hands-On Large Language Models*. O'Reilly Media, Inc.

# Note
All experiments are conducted using several rental servers in vast.ai with the following specifications:
+ CPU: Minimum 32 cores
+ RAM: Minimum 32GB
+ VRAM (GPU): Minimum 15GB (using 8GB is soooooooooo slow, lol)
+Storage Disk: Minimum 48GB
+ CUDA Version: Minimum 12.1