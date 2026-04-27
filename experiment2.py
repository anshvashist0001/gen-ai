import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

true_mu = 5
true_sigma = 2
data = np.random.normal(true_mu, true_sigma, 500)

estimated_mu = np.mean(data)
estimated_sigma = np.std(data)

print("True mu:", true_mu, "| Estimated mu:", round(estimated_mu, 4))
print("True sigma:", true_sigma, "| Estimated sigma:", round(estimated_sigma, 4))
print("Error mu:", round(abs(true_mu - estimated_mu), 4))
print("Error sigma:", round(abs(true_sigma - estimated_sigma), 4))

x = np.linspace(data.min(), data.max(), 100)

plt.hist(data, bins=30, density=True, alpha=0.6, label="Data")
plt.plot(x, norm.pdf(x, estimated_mu, estimated_sigma), 'r', label="MLE Fit")
plt.plot(x, norm.pdf(x, true_mu, true_sigma), 'g--', label="True")
plt.legend()
plt.title("MLE - Normal Distribution")
plt.show()
