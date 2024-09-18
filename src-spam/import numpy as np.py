# import numpy as np
# words = np.array([[1,2,0], [1,2,3], [0,2,5]])
# labels = [0, 1, 1]
# print(np.sum(words[labels == 1, :], axis = 1))
import numpy as np

words = np.array([[0,0,1],[1,2,9], [1,2,3], [0,2,5]])
labels = np.array([0, 0, 1, 1])

print(np.sum(words[labels == 1, :], axis = 1))