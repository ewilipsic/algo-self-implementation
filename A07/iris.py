import random

def knn(data,q,k):
    def give_dist(c1,c2):
        sum = 0
        for i in range(len(c1)):
            sum += (c1[i] - c2[i])**2
        return sum**0.5

    dists = sorted([(give_dist(q,x[1]),x[0]) for x in data],key = lambda s : s[0])

    d = {}
    for i in range(k):
        if dists[i][1] in d: d[dists[i][1]] += 1
        else: d[dists[i][1]] = 1

    max = 0
    maxkey = ''
    for x,y in d.items():
        if(y>max):
            max = y
            maxkey = x
    return maxkey

iris = []

with open("iris.csv","r") as f:
    f.readline()
    for line in f:
        line = line.strip().split(',')
        iris.append((line[-1],[float(x) for x in line[:-1]]))
random.shuffle(iris)

AB_trainingData = iris[:74]
AB_validationData = iris[74:105]
AB_testData = iris[105:]

AB_maxk = 0
AB_maxcur = 0

for AB_k in range(1,len(AB_trainingData) + 1,2):
    AB_cor = 0
    for AB_y,AB_x in AB_validationData:
        if knn(AB_trainingData,AB_x,AB_k) == AB_y:
            AB_cor += 1
    if AB_cor > AB_maxcur:
        AB_maxcur = AB_cor
        AB_maxk = AB_k

print("best k = ",AB_maxk)

correct = 0
for x in AB_testData:
    pred = knn(AB_trainingData,x[1],AB_maxk)
    if(pred == x[0]):correct += 1

print(correct/len(AB_testData))

