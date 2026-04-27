import numpy as np
import matplotlib.pyplot as plt

normal = np.random.normal(0, 1, 1000)
uniform = np.random.uniform(0, 1, 1000)
exponential = np.random.exponential(1, 1000)
binomial = np.random.binomial(10, 0.5, 1000)

plt.figure(figsize=(12, 4))

plt.subplot(1, 4, 1)
plt.hist(normal, bins=30, color='blue')
plt.title("Normal")

plt.subplot(1, 4, 2)
plt.hist(uniform, bins=30, color='green')
plt.title("Uniform")

plt.subplot(1, 4, 3)
plt.hist(exponential, bins=30, color='red')
plt.title("Exponential")

plt.subplot(1, 4, 4)
plt.hist(binomial, bins=30, color='orange')
plt.title("Binomial")

plt.tight_layout()
plt.show()

print("Normal   - Mean:", round(np.mean(normal), 2), "Std:", round(np.std(normal), 2))
print("Uniform  - Mean:", round(np.mean(uniform), 2), "Std:", round(np.std(uniform), 2))
print("Exponential - Mean:", round(np.mean(exponential), 2))
print("Binomial - Mean:", round(np.mean(binomial), 2))
