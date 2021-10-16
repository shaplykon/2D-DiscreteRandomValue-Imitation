import numpy as np


class RandomVariableGenerator:
    def __init__(self, X, Y, theoretical_p):
        self.X = X
        self.Y = Y
        self.p = theoretical_p
        self.q = list()
        self.l = list()
        self.l.append(0)
        for i in range(len(self.Y)):
            self.q.append(self.p.sum(axis=1)[i])
            self.l.append(sum(self.q))

    def generate_value(self):
        x = np.random.uniform()
        k = r = 1
        for i in range(len(self.l) - 1):
            if self.l[i] < x <= self.l[i + 1]:
                for j in range(len(self.X)):
                    if sum(self.p[k - 1][0:j + 1]) + self.l[k - 1] < x:
                        r += 1
                    else:
                        break
                break
            else:
                k += 1
        return k - 1, r - 1
