import numpy as np
import matplotlib.pyplot as plt

# creating the dataset
data = {'plaintext': 0.012353, 'Paillier': 2.8, 'BFV': 6963}
courses = list(data.keys())
values = list(data.values())

fig = plt.figure()

# creating the bar plot
plt.bar(courses, values, color='maroon', width=0.4)

plt.ylabel("seconds")
plt.title("Average elapsed time for 5 x 10 instances. \n Experiments repeated 5 times.")
plt.yscale('log')
plt.show()
