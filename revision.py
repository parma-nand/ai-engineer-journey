
import matplotlib.pyplot as plt
import numpy as np

# x = np.linspace(0, 10, 100)

# plt.subplot(1, 3, 1)          # (rows, cols, which plot)
# plt.plot(x, np.sin(x))
# plt.title("Sin")

# plt.subplot(1, 3, 2)          # move to 2nd plot
# plt.plot(x, np.cos(x))
# plt.title("Cos")

# plt.subplot(1, 3, 3)          # move to 3rd plot
# plt.plot(x, np.tan(x))
# plt.title("Tan")

# plt.show()
# x = np.linspace(0, 10, 100)

axes = plt.subplots(1, 3, figsize=(12, 4))  # all 3 at once

axes[0].plot(x, np.sin(x))
axes[0].set_title("Sin")

axes[1].plot(x, np.cos(x))
axes[1].set_title("Cos")

axes[2].plot(x, np.tan(x))
axes[2].set_title("Tan")

plt.show()
