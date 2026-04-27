import numpy as np
from scipy.stats import norm, expon
from scipy.optimize import minimize
import matplotlib.pyplot as plt

true_mu = 4.0
true_sigma = 1.5
true_lambda = 0.5

data_normal = np.random.normal(true_mu, true_sigma, 1000)
data_exp = np.random.exponential(1 / true_lambda, 1000)

est_mu = np.mean(data_normal)
est_sigma = np.std(data_normal)
est_lambda = 1 / np.mean(data_exp)

print("=== Normal Distribution ===")
print(f"True     mu={true_mu}, sigma={true_sigma}")
print(f"Estimated mu={est_mu:.4f}, sigma={est_sigma:.4f}")
print(f"Error    mu={abs(est_mu - true_mu):.4f}, sigma={abs(est_sigma - true_sigma):.4f}")

print("\n=== Exponential Distribution ===")
print(f"True     lambda={true_lambda}")
print(f"Estimated lambda={est_lambda:.4f}")
print(f"Error    lambda={abs(est_lambda - true_lambda):.4f}")

sizes = [10, 50, 100, 500, 1000, 5000]
mu_errors = []
sigma_errors = []

for n in sizes:
    samples = np.random.normal(true_mu, true_sigma, n)
    mu_errors.append(abs(np.mean(samples) - true_mu))
    sigma_errors.append(abs(np.std(samples) - true_sigma))

print("\n=== Error vs Sample Size ===")
for i, n in enumerate(sizes):
    print(f"n={n:5d}  mu_error={mu_errors[i]:.4f}  sigma_error={sigma_errors[i]:.4f}")

plt.figure(figsize=(14, 4))

plt.subplot(1, 3, 1)
x = np.linspace(data_normal.min(), data_normal.max(), 200)
plt.hist(data_normal, bins=40, density=True, alpha=0.6, label="Data")
plt.plot(x, norm.pdf(x, true_mu, true_sigma), 'r--', label="True")
plt.plot(x, norm.pdf(x, est_mu, est_sigma), 'g', label="MLE")
plt.title("Normal MLE")
plt.legend()

plt.subplot(1, 3, 2)
x2 = np.linspace(0, data_exp.max(), 200)
plt.hist(data_exp, bins=40, density=True, alpha=0.6, label="Data")
plt.plot(x2, expon.pdf(x2, scale=1/true_lambda), 'r--', label="True")
plt.plot(x2, expon.pdf(x2, scale=1/est_lambda), 'g', label="MLE")
plt.title("Exponential MLE")
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(sizes, mu_errors, 'o-', label="mu error")
plt.plot(sizes, sigma_errors, 's-', label="sigma error")
plt.xscale('log')
plt.xlabel("Sample Size")
plt.ylabel("Error")
plt.title("Error vs Sample Size")
plt.legend()

plt.tight_layout()
plt.show()
