# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import torch
from transformers import FlaubertModel, FlaubertTokenizer
from scipy.spatial.distance import cosine
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import plotly.express as px
import networkx as nx
import fr_core_news_md

class Summarizer():

  def __init__(self, stop_words=None):
    self.nlp = fr_core_news_md.load()
    self.stop_words = stop_words

###### Load Model Methods ######

  def init_model(self, model='flaubert', device=None, log=False):
    # Choosing device for language model
    if device is None:
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.device = device

    try:
      # Flaubert model
      if model=='flaubert':
        model_name = 'flaubert/flaubert_large_cased'
        flaubert = FlaubertModel.from_pretrained(model_name)
        tokenizer = FlaubertTokenizer.from_pretrained(model_name,
                                                      do_lowercase=False)
        self.model = flaubert
        self.tokenizer = tokenizer
        self.model_name = model_name
      # Camembert model
      elif model=='camembert':
        model_name = 'camembert'
        self.model = torch.hub.load('pytorch/fairseq', 'camembert')
        self.model_name = model_name
    except:
        print(f'Error while loading the {model} model.')
        return

    # Model Inference
    self.model.to(self.device)
    self.model.eval()

    # Log Info
    if log:
      self.init_log(self.model_name, self.device)

  def init_log(self, model_name, device):
    print(f'Summarizer: \'{model_name}\' successfully loaded on {device}.')

  def to(self, device):
    """
    Moves and/or casts the NLP model parameters and buffers.

    Parameters
    ----------
      device: string | device name.
    """

    self.device = device
    self.model.to(device)

###### Sentence Selection Methods ######

  def reference_selection(self, reference_embeddings, embeddings, threshold):
    """
    Filter embeddings based on similarity with a reference embedding.
    Return selected embeddings with similarity score higher than thresold.
    """

    selected_indices = []
    for reference_embedding in reference_embeddings:
      similarities = np.array(self.get_similarities(reference_embedding,
                                                    embeddings))
      
      filtered_indices = np.where(similarities > threshold)[0]
      selected_indices.extend(filtered_indices.tolist())

    return sorted(list(set(selected_indices)))

###### Keyword Selection Methods #####

  def word_embedding(self, word):
    """
    Return model embedding of the given word.
    """
    # Camembert
    if self.model_name=='camembert':
      token = self.model.encode(word).to(self.device)

      with torch.no_grad():
        encoded_layers = self.model.extract_features(token,
                                                     return_all_hiddens=False)
        embedded_word = encoded_layers[0][0].cpu().numpy()

      return embedded_word
    # Flaubert
    else:
      token_ids = torch.tensor([self.tokenizer.encode(word,
                                                      add_special_tokens=False)])
      token_ids = token_ids.to(self.device)
    
      with torch.no_grad():
        last_layers = self.model(token_ids)

      token_embedding = torch.stack(last_layers, dim=0)[0]
      word_embedding = torch.mean(token_embedding,dim=1)
      embedded_word = word_embedding.cpu().numpy()
      
      return embedded_word

  def remove_stop_words(self, sentence):
    """
    Remove stop words form a text sentence.
    """
    split = [word for word in sentence.split(' ') if len(word) > 2]
    sentence = ' '.join(split)
    sentence = self.nlp(sentence)
    tokens = [token.text for token in sentence]
    clean_sentence = tokens
    if self.stop_words is not None:
      clean_sentence = [word for word in tokens if not word in self.stop_words]
    clean_sentence[:] = [item for item in clean_sentence if item != ' ']

    return clean_sentence

  def content_words_embedding(self, text):
    """
    Return the word granularity text embedding of the given text.
    """

    text_content_words = [self.remove_stop_words(sentence) for sentence in text]

    content_words_embedding = []
    for words in text_content_words:
      content_words_embedding.append([self.word_embedding(word) \
                                      for word in words])
    
    return content_words_embedding

  def keyword_similarity(self, content_words_embedding, keyword_embedding):
    
    keyword_similarities = []
    for words_embedding in content_words_embedding:
      if len(words_embedding) != 0:
        sim = [1 - cosine(keyword_embedding, w) for w in words_embedding]
      else:
        sim = [0.]
      keyword_similarities.append(sim)
    
    return keyword_similarities

  def keyword_selection(self, content_words_embedding, keywords_embeddings,
                        method='max', threshold=0.6):
    """
    Return selected text indices based on max/mean similarity with keywords.
    """
    kw_similarities = [self.keyword_similarity(content_words_embedding,
                                               kw) for kw in keywords_embeddings]
    
    top_indices = []
    for kw_similarity in kw_similarities:

      top_sim = []
      if method=='max':
        max_sim_sentence = [max(sentence) for sentence in kw_similarity]
        max_sim_sentence = np.array(max_sim_sentence)
        top_sim = np.where(max_sim_sentence >= threshold)[0]
      else:
        mean_sim_sentence = [np.mean(sentence, axis=0) for sentence in kw_similarity]
        mean_sim_sentence = np.array(mean_sim_sentence)
        top_sim = np.where(mean_sim_sentence >= threshold)[0].tolist()

      top_indices.extend(top_sim)

    return list(set(top_indices))

###### "FIT" methods ######

  def fit(self, text,
          reference_sentences=None,
          reference_threshold=0.6,
          keywords=None,
          keywords_method='max',
          keywords_threshold=0.6,
          log=True):
    # Embed all the text
    try:
      if not isinstance(text, pd.core.series.Series):
        text = pd.Series(text)
    except:
      print('Data input error: text should be a numpy ndarray or a pandas '
            'series of str sentences')
      return
    
    self.text = text.to_numpy()
    if self.model_name == 'camembert':
      self.text_embeddings = self.camembert_text_embedding(self.text)
    else:
      self.text_embeddings = self.flaubert_text_embedding(self.text)

    # Reference Sentence Selection
    if reference_sentences is not None:
      if self.model_name == 'camembert':
        reference_embeddings = self.camembert_text_embedding(reference_sentences)
      else:
        reference_embeddings = self.flaubert_text_embedding(reference_sentences)

      selected_indices = self.reference_selection(reference_embeddings,
                                                  self.text_embeddings,
                                                  reference_threshold)
      self.text = self.text[selected_indices]
      self.text_embeddings = self.text_embeddings[selected_indices]

    # Keyword Sentence Selection
    if keywords is not None:
      keywords_embeddings = [self.word_embedding(keyword) for keyword in keywords]

      content_words_embedding = self.content_words_embedding(self.text)
      selected_indices = self.keyword_selection(content_words_embedding,
                                                keywords_embeddings,
                                                method=keywords_method,
                                                threshold=keywords_threshold)
      self.text = self.text[selected_indices]
      self.text_embeddings = self.text_embeddings[selected_indices]

    # Log Info
    if log:
      print(f'Summarizer fit: computed {self.text_embeddings.shape[0]} '
            f'embeddings of dim {self.text_embeddings.shape[1]}.')

  def camembert_embedding(self, sentence):
    """Return sentence CLS embedding computed with the camemBERT model"""
    # Get sentence tokens
    tokens = self.model.encode(sentence).to(self.device)

    # Pass to model
    with torch.no_grad():
      encoded_layers = self.model.extract_features(tokens,
                                                   return_all_hiddens=True)

    token_embeddings = torch.stack(encoded_layers, dim=0)
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    sum_vec = torch.sum(token_embeddings[-4:], dim=0)
    sentence_embedding = torch.mean(sum_vec,dim=0)
    return sentence_embedding.cpu().numpy()

  def camembert_text_embedding(self, text):
    """
    Return the camembert model embeddings associated with the text input.

    Parameters
    ----------
      text: numpy array of string | All text sentences      
    """
    embedded_sentences = list(map(self.camembert_embedding, text))

    return np.vstack(embedded_sentences)

  def flaubert_text_embedding(self, text):
    """
    Return the flaubert model embeddings associated with the text input.

    Parameters
    ----------
      text: numpy array of string | All text sentences      
    """
    input_ids = [self.tokenizer.encode(sentence) for sentence in text]
    padded = np.array([i + [0]*(300-len(i)) for i in input_ids])

    attention_mask = np.where(padded != 0, 1, 0)
    input_ids_tensor = torch.tensor(padded).to(self.device)
    masks_tensor = torch.tensor(attention_mask).to(self.device)

    with torch.no_grad():
      encoded_layers = self.model(input_ids_tensor, masks_tensor)

    token_embeddings = torch.stack(encoded_layers, dim=0)[0]
    sentence_embedding = torch.mean(token_embeddings,dim=1)
    embedded_sentences = sentence_embedding.cpu().numpy()

    return embedded_sentences

###### "Summary" methods ######

  def get_similarities(self, reference_embedding, embeddings):
    """
    Return the similarity scores between reference and embeddings.
    """
    similarities = []
    for i in range(len(embeddings)):
        sim = 1 - cosine(reference_embedding, embeddings[i])
        similarities.append(sim)
    
    return similarities

  def top_similarities(self, reference_embedding, embeddings, nb_top):
    """
    Return the nb_top embeddings indices closer to the reference_embedding.

    Parameter
    ---------
      reference_embedding: np.array | reference embedding for distance.
      embeddings: np.ndarray | embeddings to sort according to ref distance.
      nb_top: int | number of top indices to return.
    """

    # Compute similarity according to distance to reference.
    similarities = self.get_similarities(reference_embedding, embeddings)

    # Return nb_top indices
    top_indices = np.array(similarities).argsort()[::-1][:nb_top]
    return top_indices

  def mean_similarity_summary(self, nb_sentences=10, return_indices=False):
    """
    Perform summarization over the text_embeddings with mean similarity method.
    The mean embedding is used as reference for similarity.

    Return the summary of length nb_sentences.
    (optional) Return nb_sentences indices ordered by distance to the mean.

    Parameters
    ----------
      nb_sentences: int | length of the summary.
      return_indices: bool | return sentences indices if set to True.
    """
    # Compute mean sentence embedding
    mean_sentence_embedding = np.mean(self.text_embeddings, axis=0)

    top_indices = self.top_similarities(mean_sentence_embedding,
                                        self.text_embeddings,
                                        nb_sentences)
    
    summary = self.text[sorted(top_indices)]

    if return_indices:
      return summary, top_indices

    return summary

  def text_visualization(self, cluster_labels=None, plot_lib='plotly'):
    """
    Plot embedded text visualization in 2 dimension with tsne.
    Use either plotly or matplotlib library.

    Parameters
    ----------
      cluster_labels: sequence of labels
      plot_lib: string | Library name. Either 'plotly' or 'pyplot'.
    """
    # TSNE dimension reduction
    mapped_embeddings = TSNE(n_components=2,
                             metric='cosine',
                             init='pca').fit_transform(self.text_embeddings)
    # Plot with plotly
    if plot_lib == 'plotly':
      try:
        df_visu = pd.DataFrame({'x': mapped_embeddings[:, 0],
                                'y': mapped_embeddings[:, 1],
                                'text': self.text})
        color = None
        if cluster_labels is not None:
          df_visu['cluster'] = cluster_labels
          color = 'cluster'

        fig = px.scatter(df_visu, x='x', y='y', hover_data=['text'],
                         color=color)
        fig.update(layout_coloraxis_showscale=False)
        fig.show()
        return
      except:
        print('Error with plotly scatter method, using pyplot instead.')
        plot_lib = 'pyplot'

    # Plot with matplotlib
    if plot_lib == 'pyplot':
      plt.figure(figsize=(50,50))
      x = mapped_embeddings[:, 0]
      y = mapped_embeddings[:, 1]
      plt.scatter(x, y, c=cluster_labels)
      for i, txt in enumerate(self.text):
          plt.annotate(txt[:10], (x[i], y[i]))
      return

    # Case for wrong plot lib parameter
    print('Wrong plot_lib parameter. Use \'plotly\' or \'pyplot\'.')

  def cluster_embeddings(self, nb_clusters, nb_top):
    """
    Perform Kmeans clustering over the text embeddings.

    Return:
      dict{cluster_id, embeddings_indices}
      &
      dict{cluster_id, nb_top_embeddings_indices}
    
    Parameters
    ----------
      nb_clusters: int | number of clusters.
      nb_top: int | number of embbedings closer to cluster centroid to select. 
    """
    # Perform Kmeans
    Kmean_cluster = KMeans(n_clusters=nb_clusters).fit(self.text_embeddings)
    centroid = Kmean_cluster.cluster_centers_
    embeddings_cluster_labels = Kmean_cluster.labels_

    top_clustered_indices = {}

    # Go through each cluster
    for cluster_label in range(centroid.shape[0]):
      # Get all sentence indices
      indices = np.where(embeddings_cluster_labels == cluster_label)[0]

      # Get top sentences indices
      cluster_embeddings = self.text_embeddings[indices]
      top = self.top_similarities(centroid[cluster_label],
                                        cluster_embeddings, nb_top)
      top_clustered_indices[cluster_label] = indices[top].tolist()

    return embeddings_cluster_labels, top_clustered_indices

  def clustering_summary(self, nb_clusters, nb_top, return_clusters=False):
    """
    Perform clustering summarization method.

    Return the summary.
    (optional) Return clustering results.

    Parameters
    ----------
      nb_clusters: int | number of clusters.
      nb_top: int | number of embbedings closer to cluster centroid to select.
      return_clusters: Bool | return clustering results if True.
    """
    # Kmeans clustering
    cluster_results = self.cluster_embeddings(nb_clusters, nb_top)

    # Get top sentence indices per cluster
    _, top_clustered_indices = cluster_results

    # Construct the summary
    summary_indices = []
    for indices in top_clustered_indices.values():
      summary_indices += indices
    summary = self.text[sorted(summary_indices)]

    if return_clusters:
      return summary, cluster_results

    return summary

  def graph_summary(self, nb_sentences=10, return_indices=False):
    """
    Perform graph summarization method.
    Return the summary.
    (optional) Return nb_sentences indices ordered by pagerank score.
    """

    nb_embeddings = self.text_embeddings.shape[0]

    # Building similarity graph
    sim_matrix = np.zeros([nb_embeddings, nb_embeddings])
    for i in range(nb_embeddings):
      for j in range(nb_embeddings):
        if i != j:
          sim_matrix[i][j] = 1 - cosine(self.text_embeddings[i],
                                        self.text_embeddings[j])
    nx_graph = nx.from_numpy_array(sim_matrix)

    # Pagerank algorithm
    scores = nx.pagerank(nx_graph)

    summary_indices = np.argsort([scores[i] for i in range(nb_embeddings)])[::-1]

    summary = self.text[sorted(summary_indices[:nb_sentences])]

    if return_indices:
      return summary, summary_indices
    else:
      return summary