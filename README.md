<h1 align='center'> French Text Summarization </h1>

# Introduction

This Module is a Summarizer using [Hugging face's Transformers library] (https://huggingface.co/transformers/), which provides general-purpose architectures for NLP, in particual the BERT architecture.

It can be used with two NLP pre-trained french BERT based models:
* [CamemBERT](https://camembert-model.fr/)
* [FlauBERT](https://arxiv.org/abs/1912.05372)

There are 3 type of extractive summarization techniques that can be used in this module:
### Mean Summarization:
This technique is the simplest one. It computes the mean embedding of the text and return the top closer sentences to compose the summary.
### Clustering Summarization:
The clustering summarization model acts as the mean summarization one, only here the summarizer performs a clustering algorithm (K-means) on the data embedding first.
`nb_clusers` centroids embedding are computed, one for each cluster. Then `nb_top` closer sentences are selected for each cluster to compose the summary. 
With this method, a cluster labels in the 2D space after TSNE dimension reduction can be visualized.
### Graph Summary:
This third summarization method makes a similarity graph between the different lines of the text data. Then, it calculates a score for each sentence using the pagerank algorithm. This score is used to produce the final summary.
To use this method, call the `graph_summary` method on the summarizer. Once again, you can chose how may sentence you want for the summary with the `nb_sentences` parameter.
