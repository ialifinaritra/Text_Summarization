from Summarizer import Summarizer
from absl import app
from absl import flags
import numpy as np


FLAGS = flags.FLAGS

flags.DEFINE_string(
    "text_path",
    "",
    "path to the text to summarize"
)

flags.DEFINE_string(
    "model",
    "flaubert",
    "Use either 'camembert' or 'flaubert' for model"
)

flags.DEFINE_string(
    "method",
    "mean",
    "mean / Clustering or Graph summarization"
)

flags.DEFINE_integer(
    "nb_sentences",
    5,
    "number of sentences in the summary"
)

flags.DEFINE_bool(
    "visualization",
    False,
    "visualize text or not"
)


def summarize(model, method, text, nb_sentences,viz=False):
    summarizer = Summarizer()
    summarizer.init_model(model, log=True)

    summarizer.fit(text)
    
    if method == 'mean':
        summary = summarizer.mean_similarity_summary(nb_sentences=nb_sentences)

    elif method == 'clustering':
        summary, cluster_results = summarizer.clustering_summary(nb_clusters=nb_sentences, nb_top=2, return_clusters=True)
        labels, cluster_indices = cluster_results
        if viz:
            summarizer.text_visualization(cluster_labels=labels, plot_lib='plotly')

    elif method == 'graph':
        summary = summarizer.graph_summary(nb_sentences=nb_sentences)

    return summary


def load_preprocess_text(path):
    with open(path, 'r') as f:
        text = f.read()
    
    text = text.split('.')

    return np.array(text)


def main(_):
    model = FLAGS.model 
    method = FLAGS.method 
    path_text = FLAGS.text_path
    nb_sentences = FLAGS.nb_sentences

    text = load_preprocess_text(path_text)
    summary = summarize(model, method, text, nb_sentences,viz)
    for sentence in summary:
        print(sentence)


if __name__ == '__main__':
    app.run(main)