# -*- coding:utf-8 -*-

import numpy
from scipy.spatial.distance import cosine

def calc_item_score(target_user_index, user_rating_matrix):

  """
  指定したアイテムの評価値を計算する
  :param target_user_index:  int: ユーザーのindex
  :param user_rating_matrix: numpy.ndarray: ユーザーアイテムの評価値行列
  :return:  float: 指定したアイテムの評価値
  """

  target_user_ratings = user_rating_matrix[target_user_index]
  item_similarity = numpy.zeros(len(target_user_ratings))
  for compare_user_index in range(len(user_rating_matrix)):
    compare_user_ratings = user_rating_matrix[compare_user_index]
    if compare_user_index == target_user_index:
      # 同一ユーザーのときは類似度計算しない
      continue

    # ユーザーの類似度をコサイン類似度から求める
    user_similarity = 1.0 - cosine(target_user_ratings, compare_user_ratings)

    # 求めたコサイン類似度をそのユーザーの評価値に乗じて足し合わせる
    item_similarity += user_similarity * compare_user_ratings

  return item_similarity
