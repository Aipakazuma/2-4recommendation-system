# -*- coding:utf-8 -*-

from gensim import corpora, models, similarities

def create_similarity_matrix(movie_features, movies_data):
  """
  類似度計算用の行列オブジェクトを作る。
  :param movie_features:
  :param movies_data:
  :return:
  """
  dicitionary = corpora.Dictionary(movie_features)
  bow_actors = [dicitionary.doc2bow(actors) for actors in movies_data]
  tfidf = models.TfidfModel(bow_actors)
  similarity_matrix = similarities.MatrixSimilarity(tfidf[bow_actors])
  return dicitionary, similarity_matrix

def calc_similarity(dicitionary, similarity_matrix, user_features):
  """

  :param dicitionary:
  :param similarity_matrix:
  :param user_features:
  :return:
  """
  return [
    (movie_index, similarity) for movie_index, similarity
    in enumerate(similarity_matrix[dicitionary.doc2bow(user_features)])
  ]

