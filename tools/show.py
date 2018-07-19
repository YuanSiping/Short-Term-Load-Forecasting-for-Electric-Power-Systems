from pandas import read_csv
from matplotlib import pyplot
# load dataset
dataset = read_csv('load.csv', header=0, index_col=0)
values = dataset.values
# plot each column
i = 1
pyplot.figure()
for group in range(0, 7):
	pyplot.subplot(7, 1, i)
	pyplot.plot(values[:, group])
	pyplot.title(dataset.columns[group], y=0.5, loc='right')
	i += 1
pyplot.show()