
AB_trainingData = [('comedy',(100,0)),('action',(0,100)),('action',(15,90)),('comedy',(85,20))]

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

AB_validationData = [('action',(10,95)),('comedy',(85,15))]

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

AB_testData = [(6,70),(93,23),(50,50)]
AB_preds = []
for x in AB_testData:
    AB_preds.append(knn(AB_trainingData,x,AB_maxk))

print(AB_testData)
print(AB_preds)


