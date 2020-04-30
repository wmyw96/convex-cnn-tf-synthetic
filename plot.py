import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Times New Roman"
plt.style.use('seaborn-darkgrid')
palette = plt.get_cmap('Set1')
plt.figure(figsize=(8, 6))

same_result = np.load('matching_result_same.npy')
diff_result = np.load('matching_result_diff.npy')

result = [same_result, diff_result]
indicator = ['same', 'diff']
marker_shape = ['o', 'v', 's']

for i in range(len(indicator)):
    for j in range(len(marker_shape)):
        plt.plot(np.arange(len(result[i][:, j])) + 1, 
                 result[i][:, j], marker=marker_shape[j],
                 color=palette(i), 
                 label='{}-layer{}'.format(indicator[i],j + 1))
plt.xlabel('number of epoches')
plt.ylabel('mean matched l2 distance')
plt.legend(frameon=True)
#plt.show()
plt.savefig('result.pdf')
plt.close()
plt.clf()