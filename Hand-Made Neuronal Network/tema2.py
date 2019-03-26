import pickle, gzip
from numpy import random, dot


def activation(z):
    if z > 0:
        return 1
    return 0


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
f.close()

# init and train
# 9 bios

# perceptron = []
# for i in range(10):
#    perceptron.append(random.rand(28 * 28))

# b = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]

b = pickle.load(open("myBios.txt", "rb"))
learnRate = 0.05
nrIterations = 1
z = 0
allClasified = False
perceptron = pickle.load(open("myPerceptron.txt", "rb"))
ok = 0
while nrIterations > 0 and not allClasified:  # if all the perceptrons clasified everything corect
    for i in range(10):
        allClasified = True
        for index in range(50000):
            x = train_set[0][index]
            t = train_set[1][index]
            z = dot(perceptron[i], x) + b[i]
            output = activation(z)  # calculate if the perceptron recognises the digit
            if t == i:
                ok = 1
            else:
                ok = 0
            perceptron[i] = perceptron[i] + (ok - output) * x * learnRate  # adjust the wieghts of the perceptron
            b[i] = b[i] + (ok - output) * learnRate  # adjust the bios of the perceptron
            if t != output:  # test if we have the expected output
                allClasified = False
            z = 0
            if index % 1000 == 0:
                print("Nr iteration: " + str(nrIterations) + " index: " + str(index))
    nrIterations = nrIterations - 1

# validation
totalMatch = 0.0
goodMatch = 0.0

# same as for traing-ing, but without the learning part
for index in range(10000):
    x = valid_set[0][index]
    t = valid_set[1][index]
    max = 0
    myPerceptron = 0
    for i in range(10):
        z = dot(perceptron[i], x) + b[i]
        output = activation(z)
        if output == 1:
            if z > max:
                max = z
                myPerceptron = i

    if t == myPerceptron:
        goodMatch = goodMatch + 1

    totalMatch = totalMatch + 1  # number of cases until now
    print("Acuracy: " + str(100 * goodMatch / totalMatch) + "%")
    badMatch = 0
pickle.dump(perceptron, open("myPerceptron.txt", "wb"))
pickle.dump(b, open("myBios.txt", "wb"))
