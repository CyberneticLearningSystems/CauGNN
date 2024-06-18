import matplotlib as plt

import matplotlib.pyplot as plt
import numpy as np
file = './TENet_master/TE/form41_aggregated_quarterly_LargeUSC_reduced_TE.txt'
A = np.loadtxt(file)
A = np.array(A,dtype=np.float32)

# Display matrix
plt.matshow(A)

plt.show()
