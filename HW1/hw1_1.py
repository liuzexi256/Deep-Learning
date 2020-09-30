import math
import random
import numpy as np
import matplotlib.pyplot as plt

random.seed(0)

def makeMatrix(I, J, fill=0.0):

    m = []
    for i in range(I):
        m.append([fill] * J)
    return m


def randomizeMatrix(matrix, a, b):

    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            matrix[i][j] = random.uniform(a, b)
    


def sigmoid(x):

    return 1.0 / (1.0 + math.exp(-x))


def dsigmoid(y):

    return y * (1 - y)


class NN:
    def __init__(self, ni, nh, no):

        self.ni = ni + 1
        self.nh = nh
        self.no = no

        self.ai = [1.0] * self.ni
        self.ah = [1.0] * self.nh
        self.ao = [1.0] * self.no

        self.wi = makeMatrix(self.ni, self.nh)
        self.wo = makeMatrix(self.nh, self.no)

        randomizeMatrix(self.wi, -0.2, 0.2)
        randomizeMatrix(self.wo, -2.0, 2.0)

        self.ci = makeMatrix(self.ni, self.nh)
        self.co = makeMatrix(self.nh, self.no)

    def runNN(self, inputs):

        if len(inputs) != self.ni - 1:
            print('incorrect number of inputs')

        for i in range(self.ni - 1):
            self.ai[i] = inputs[i]

        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                sum += ( self.ai[i] * self.wi[i][j] )
            self.ah[j] = sigmoid(sum)

        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum += ( self.ah[j] * self.wo[j][k] )
            self.ao[k] = sigmoid(sum)

        return self.ao


    def backPropagate(self, targets, N):

        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k] - self.ao[k]
            output_deltas[k] = error * dsigmoid(self.ao[k])

        for j in range(self.nh):
            for k in range(self.no):

                change = output_deltas[k] * self.ah[j]

                self.wo[j][k] += N * change
                self.co[j][k] = change


        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error += output_deltas[k] * self.wo[j][k]
            hidden_deltas[j] = error * dsigmoid(self.ah[j])


        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j] * self.ai[i]

                self.wi[i][j] += N * change
                self.ci[i][j] = change


        error = 0.0
        for k in range(len(targets)):
            error = 0.5 * (targets[k] - self.ao[k]) ** 2
        return error


    def test(self, test):

        if self.runNN(test)[0] > 0.5:
            return 1
        else:
            return 0

    def train(self, patterns, max_iterations, N):
        er = []
        for i in range(max_iterations):
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.runNN(inputs)
                error = self.backPropagate(targets, N)
                
            if i % 50 == 0:
                er.append(error)
                print('Combined error', error)
        x_values = [i*50 for i in range(max_iterations//50)]
        plt.plot(x_values, er)
        plt.xlabel('epoch')
        plt.ylabel('error')
        plt.title('learning curve')



def main():
    pat = [
        [[1, 0], [1]],
        [[0, 1], [1]],
        [[-1, 0], [1]],
        [[0, -1], [1]],
        [[0.5, 0.5], [0]],
        [[0.5, -0.5], [0]],
        [[-0.5, 0.5], [0]],
        [[-0.5, -0.5], [0]],
        #[[0, 0], [0]],
        
    ]
    myNN = NN(2, 8, 1)
    myNN.train(pat, 20000, 0.5)

    test = [[[0,0] for i in range(200)] for j in range(200)]
    res = [[0 for i in range(200)] for j in range(200)]
    for i in range(200):
        for j in range(200):
            test[i][j] = [j/100 - 1, -i/100 + 1]
            res[i][j] = myNN.test(test[i][j])
    plt.matshow(res)
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()
