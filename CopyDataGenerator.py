import numpy as np

def get_episodic_copy_data(total_sequences = 16 * 100, batch_size = 16,
                           blanks_len = 100, sequence_len = 10):
  '''
  Returns a 4-d tensor containing the data to be used for the copy problem.

  Args:
  total_sequences -- the total number of sequences across all batches
      (default 1600)
  batch_size -- the batch size (default 16)
  blanks_len -- the length of the sequence between taking the input and
      producing the output including 1 character for delimiter (default 100)
  sequence_len -- the length of the sequence required to be memorized and then
      returned after the delimiter (default 10)

  Returns:
  one_hot_x -- a 4-d tensor of shape:
      (total_sequences // batch_size) x batch_size x
      (sequence_len + blank_len + sequence_len) x 10
      This tensor represents the input sequence in the copy problem.


  one_hot_y -- a 4-d tensor of shape:
      (total_sequences // batch_size) x batch_size x
      (sequence_len + blank_len + sequence_len) x 10
      This tensor represents the output sequence in the copy problem.
  '''

  seq = np.random.randint(1, high=9, size=(total_sequences, sequence_len))

  zeros1 = np.zeros((total_sequences, blanks_len - 1))
  zeros2 = np.zeros((total_sequences, blanks_len))
  delimiter = 9 * np.ones((total_sequences, 1))
  zeros3 = np.zeros((total_sequences, sequence_len))

  x = np.concatenate((seq, zeros1, delimiter, zeros3), axis=1).astype(np.int32)
  y = np.concatenate((zeros3, zeros2, seq), axis=1).astype(np.int32)

  x = x.reshape(total_sequences // batch_size, batch_size, 1, -1)
  x = np.swapaxes(x, 2, 3)
  x = np.swapaxes(x, 1, 2)
  x = x[..., 0]

  z = np.zeros(x.shape)

  one_hot_x = np.zeros((x.shape[0], x.shape[1], x.shape[2], 10))
  for c in range(10):
    z = z * 0
    z[np.where(x == c)] = 1
    one_hot_x[..., c] += z

  y = y.reshape(total_sequences // batch_size, batch_size, 1, -1)
  y = np.swapaxes(y, 2, 3)
  y = np.swapaxes(y, 1, 2)
  y = y[..., 0]

  z = np.zeros(y.shape)

  one_hot_y = np.zeros((y.shape[0], y.shape[1], y.shape[2], 10))

  for c in range(9):
    z = z * 0
    z[np.where(y == c)] = 1
    one_hot_y[..., c] += z

  one_hot_x = np.swapaxes(one_hot_x, 1, 2)
  one_hot_y = np.swapaxes(one_hot_y, 1, 2)

  return one_hot_x, one_hot_y
