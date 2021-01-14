import numpy
import matplotlib.pyplot as plt
import math

data = numpy.random.binomial(1, 0.25, (100000,1000))
epsilons = [0.5,0.25,0.1,0.01,0.001]

#seif a
for i in range(5):
    Xm = []
    sum = 0
    for j in range(1000):
        sum += data[i][j]
        Xm.append(sum/(j+1))
    plt.plot(Xm, label="sequence "+str(i+1))
plt.axis([0,1000,0,1])
plt.title("mean estimation as a function of the number of tosses")
plt.xlabel("number of tosses")
plt.ylabel("probability")
plt.legend()
plt.show()

#funcs for seif b
def chebyshev_upperbound(m,epsilon):
    return 1/(4*m*(epsilon**2))

def hoeffding_upperbound(m,epsilon):
    return 2*math.exp(-2*m*(epsilon**2))

#func for seif c:
def percentage(epsilon):
    percent_lst = []
    estimation_lst = [0]*100000
    p = 0.25
    for m in range(1000):
        counter = 0
        for i in range(100000):
            estimation_lst[i] += data[i][m]
            estimation = estimation_lst[i]/(m+1)
            if abs(estimation - p) >= epsilon:
                counter += 1
        percent_lst.append(counter / 100000)
    return percent_lst


for e in epsilons:
    chebyshev_arr = []
    hoeffding_arr = []
    #for seif b
    for m in range(1,1001):
        chebyshev_arr.append(min(chebyshev_upperbound(m,e),1))
        hoeffding_arr.append(min(hoeffding_upperbound(m,e),1))

    #for seif c
    sequence_satisfication = percentage(e)

    plt.plot(chebyshev_arr, label = "chebyshev")
    plt.plot(hoeffding_arr, label = "hoefding")
    plt.plot(sequence_satisfication, label = "percentage")
    plt.title("epsilon = "+str(e))
    plt.xlabel("m")
    # plt.ylabel("upper bounds size")
    plt.ylabel("upper bounds and percentage of satisfying sequences")
    plt.legend()
    plt.show()

