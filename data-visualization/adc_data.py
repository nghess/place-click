import numpy as np
import matplotlib.pyplot as plt
import time

# Import .dat and zip raw and smoothed into DF
data = np.fromfile('data-visualization/sample-data/amplifierdata.bin', dtype=float)

#data = np.fromfile('data-visualization/sample-data/digitaldata.bin', dtype=float)

print(data.shape)
window_end = 20000


# plt.plot(data[1:window_end:8])
# plt.plot(data[2:window_end:8])
# plt.plot(data[3:window_end:8])
plt.plot(data[8:-1:8])
# plt.plot(data[5:window_end:8])
# plt.plot(data[6:window_end:8])
# plt.plot(data[7:window_end:8])


plt.show()



