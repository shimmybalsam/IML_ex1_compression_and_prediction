from numpy import linalg, dot
import pandas
import matplotlib.pyplot as plt

data = pandas.read_csv("kc_house_data.csv")
noise = set()
for i in range(len(data['price'])):
    if data['price'][i] < 0 or data['sqft_lot15'][i] < 0 or data['bedrooms'][i] >=10:
        noise.add(i)
data.drop(noise,axis=0,inplace=True)
data.dropna(axis=0,how='any',inplace=True)
data.drop(['id','date','sqft_living'],axis=1,inplace=True)
data = pandas.get_dummies(data, columns= ["zipcode"], drop_first=True)
noise_control = []
for i in range(len(data['price'])):
    noise_control.append(1)
data.insert(loc=0, column='noise', value=noise_control)
training_error = []
test_error = []
for x in range(1, 101):
    training_loss = 0
    test_loss = 0
    for learning_loop in range(10):
        training_set = data.sample(frac=(x/100))
        test_set = data.drop(training_set.index,axis=0,inplace=False)
        y_training = training_set['price']
        training_set.drop(['price'],axis=1,inplace=True)
        y_test = test_set['price']
        test_set.drop(['price'],axis=1,inplace=True)
        w = dot(linalg.pinv(training_set),y_training)
        y_hat = dot(training_set,w)
        training_loss += ((((y_hat - y_training)**2).mean())**0.5)
        y_target = dot(test_set,w)
        test_loss += ((((y_target - y_test)**2).mean())**0.5)

    training_error.append(training_loss/10)
    test_error.append(test_loss/10)

plt.plot(training_error, label="training error")
plt.plot(test_error, label="test error")
plt.xlabel("x%")
plt.ylabel("error")
plt.legend()
plt.title("error as a function of the data percentage taken for training")
plt.show()
