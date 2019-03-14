import matplotlib.pyplot as plt
import pandas
from pandas.plotting import scatter_matrix
from sklearn import linear_model
from sklearn import model_selection
from sklearn import svm

# Load dataset
names = ['QIN', 'TIN', 'pHIN', 'CondIN', 'CODIN', 'SSIN', 'BOD5IN']
dataset = pandas.read_csv('aritma.csv', names=names)

# Split-out validation dataset
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values
test_rate = 0.25
xTrain, xTest, yTrain, yTest = model_selection.train_test_split(x, y, test_size=test_rate, random_state=0)

classifiers = [('SVM', svm.SVR()), ('SGDRegressor', linear_model.SGDRegressor()),
               ('PassiveAggressiveRegressor', linear_model.PassiveAggressiveRegressor())]

for name, model in classifiers:
    clf = model
    clf.fit(xTrain, yTrain)
    yPredicted = clf.predict(xTest)
    plt.figure()
    print("yPredicted", yPredicted)
    plt.title(name + ' Test' + ' Score: ' + str(clf.score(xTrain, yTrain)))
    plt.plot(yPredicted, color='red', label='predicted', marker='.')
    plt.plot(yTest, color='blue', label='Actual', marker='*')
    plt.legend()
    plt.savefig(name + '_Test' + '_plot.svg', dpi=300)
    file = open('modeller.txt', 'a')
    file.write("\n\nModel Adı: {}, \n {}".format(name, model))
    file.close()

# Training values
for name, model in classifiers:
    clf = model
    clf.fit(xTrain, yTrain)
    yPredicted = clf.predict(xTrain)
    plt.figure()
    plt.title(name + ' Train' + ' Score: ' + str(clf.score(xTrain, yTrain)))
    plt.plot(yPredicted, color='red', label='predicted')
    plt.plot(yTrain, color='blue', label='Actual')
    plt.legend()
    plt.savefig(name + '_Train' + '_plot.svg', dpi=300)

# box and whisker
x_values = dataset.loc[:, ['QIN', 'TIN', 'pHIN', 'CondIN', 'CODIN', 'SSIN']]
x_values.plot(kind='box', subplots=True, layout=(2, 3), sharex=False, sharey=True)
plt.savefig('box-whisker.svg', dpi=300)

# histograms
dataset.hist(sharex=True, sharey=True)
plt.savefig('histograms.svg', dpi=300)

# scatter plot matrix
scatter_matrix(dataset, range_padding=0.75)
plt.savefig('scatter-plots.svg', dpi=300)

tfile = open('tanimlayici_istatistik.txt', 'a')
statistics = dataset.describe()
tfile.write(statistics.to_string())
tfile.close()
