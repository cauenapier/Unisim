def skew(v, return_dv=False):
  """
  Returns the skew-symmetric matrix of a vector
  Ref: https://github.com/dreamdragon/Solve3Plus1/blob/master/skew3.m

  Also known as the cross-product matrix [v]_x such that
  the cross product of (v x w) is equivalent to the
  matrix multiplication of the cross product matrix of
  v ([v]_x) and w

  In other words: v x w = [v]_x * w
  """
  sk = np.float32([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])

  if return_dv:
      dV = np.float32([[0, 0, 0],
                       [0, 0, -1],
                       [0, 1, 0],
                       [0, 0, 1],
                       [0, 0, 0],
                       [-1, 0, 0],
                       [0, -1, 0],
                       [1, 0, 0],
                       [0, 0, 0]])
      return sk, dV
  else:
      return sk
