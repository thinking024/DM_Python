import numpy as np
from scipy import stats
import readFile as f


print('sepal_length:')
print('mean =', np.mean(f.sepal_length))
print('median =', np.median(f.sepal_length))
print('mode =', stats.mode(f.sepal_length)[0][0])
print('\n=====================\n')

print('sepal_width:')
print('mean =', np.mean(f.sepal_width))
print('median =', np.median(f.sepal_width))
print('mode =', stats.mode(f.sepal_width)[0][0])
print('\n=====================\n')

print('petal_length')
print('mean =', np.mean(f.petal_length))
print('median =', np.median(f.petal_length))
print('mode =', stats.mode(f.petal_length)[0][0])
print('\n=====================\n')

print('petal_width')
print('mean =', np.mean(f.petal_width))
print('median =', np.median(f.petal_width))
print('mode =', stats.mode(f.petal_width)[0][0])
print('\n=====================\n')
