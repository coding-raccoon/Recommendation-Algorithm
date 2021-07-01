import numpy as np
import json

if __name__ == "__main__":
    # f = open('test.json', 'r')
    # for line in f:
    #     d = json.loads(line)
    #     print(d['hello'])
    # d = {'hello': [[2, 3], [4, 5]], "world": [[3, 4], [5, 6]]}
    # with open('test.json', 'w') as outfile:
    #     json.dump(d, outfile)
    matrix = np.zeros((4, 3))
    P = np.array([1, 2, 3, 4])
    Q = np.array([4, 5, 6])
    D = np.dot(np.expand_dims(P, axis=1), np.expand_dims(Q, axis=0))
    print(D)
    np.random.shuffle(D)
    print(D)

    # print(np.dot(np.expand_dims(P), Q.T))