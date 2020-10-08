<h1 align='center'> French Text Summarization <img alt="APM" src="https://img.shields.io/apm/l/npm"> </h1> 

# Introduction

This Module is a Summarizer using [Hugging face's Transformers library](https://huggingface.co/transformers/), which provides general-purpose architectures for NLP, in particual the BERT architecture.

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

# Usage
### Requirements
The code is designed to run on Python 3 and works with `pytorch 1.6`. Some major dependencies is needed to be installed : 
<pre><code>pip install transformers
python3 -m spacy download fr_core_news_md
</code></pre>

### Running 
<pre><code>python3 main.py --text_path=path/to/text --model='flaubert' --method='clustering' --nb_sentences=5 </code></pre>

You can choose `flaubert` or `camembert` for the model and `clustering` , `mean` or `graph` for the summarization methods.

### Demos
The original text can be seen in the first part of this [article](https://fr.wikipedia.org/wiki/Terre).
The following text is the summary produced by graph method : 
<pre><code>
La Terre est la troisième planète par ordre d'éloignement au Soleil et la cinquième plus grande aussi bien par la masse que le diamètre du Système solaire.
L'axe de rotation de la Terre possède une inclinaison de 23°, ce qui cause l'apparition des saisons.
Une combinaison de facteurs tels que la distance de la Terre au Soleil (environ 150 millions de kilomètres, aussi appelée unité astronomique), son atmosphère, sa couche d'ozone, son champ magnétique et son évolution géologique ont permis à la vie d'évoluer et de se développer.
Elle est la planète la plus dense du Système solaire ainsi que la plus grande et massive des quatre planètes telluriques.
La structure interne de la Terre est géologiquement active, le noyau interne solide et le noyau externe liquide (composés tous deux essentiellement de fer) permettant notamment de générer le champ magnétique terrestre par effet dynamo et la convection du manteau terrestre (composé de roches silicatées) étant la cause de la tectonique des plaques
</code></pre>
