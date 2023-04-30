import numpy as np
import os

N = [3, 4, 5, 7, 10, 15, 20, 25, 30]

avg_acc = np.zeros(9)
avg_recall = np.zeros(9)
avg_precision = np.zeros(9)

for idx, n in enumerate(N):

    files = [filename for filename in os.listdir('res') if filename.startswith(f'{n}_')]
    FOLDER = 'res/'

    for file in files:
        container = np.load(FOLDER + file)
        pred = container['pred']
        true = container['true']

        # accuracy
        from sklearn.metrics import accuracy_score

        avg_acc[idx] = avg_acc[idx] + accuracy_score(true, pred)

        #####
        # type 1 and 2 errors
        #####

        # precision
        from sklearn.metrics import precision_score

        avg_precision[idx] = avg_precision[idx] + precision_score(true, pred)

        # recall
        from sklearn.metrics import recall_score

        avg_recall[idx] = avg_recall[idx] + recall_score(true, pred)

    avg_acc[idx] /= len(files)
    avg_precision[idx] /= len(files)
    avg_recall[idx] /= len(files)

print(avg_acc)
print(avg_precision)
print(avg_recall)

import matplotlib.pyplot as plt
# plt.style.use('seaborn-whitegrid')

fig, ax = plt.subplots()
x = range(len(N))
ax.set_ylim([97, 100.5])
# ax.grid('on')
ax.set_ylabel('Average accuracy (%)')
ax.set_xlabel('Number of nodes')
ax.set_title('Average accuracy')
ax.bar(x, avg_acc * 100, 0.8, color='g', align='center')
ax.set_xticks(x)
_ = ax.set_xticklabels(N)
plt.show()

# A high recall score means that the model has a low rate of false negatives!
fig, ax = plt.subplots()
x = range(len(N))
# ax.set_ylim([0.9, 1.005])
# ax.grid('on')
ax.set_ylabel('Average precision')
ax.set_xlabel('Number of nodes')
ax.set_title('Average precision')
ax.bar(x, avg_precision, 0.8, color='g', align='center')
ax.set_xticks(x)
_ = ax.set_xticklabels(N)
plt.show()

# A high precision score means that the model has a low rate of false positives!
fig, ax = plt.subplots()
x = range(len(N))
# ax.set_ylim([0.9, 1.005])
# ax.grid('on')
ax.set_ylabel('Average recall')
ax.set_xlabel('Number of nodes')
ax.set_title('Average recall')
ax.bar(x, avg_recall, 0.8, color='g', align='center')
ax.set_xticks(x)
_ = ax.set_xticklabels(N)
plt.show()
