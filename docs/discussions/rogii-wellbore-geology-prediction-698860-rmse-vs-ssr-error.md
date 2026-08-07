# rmse_vs_ssr_error

- archived_at: 2026-06-11T13:50:34Z
- source: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/698860

Topic #698860: rmse_vs_ssr_error
  Author: jose mata
  Posted: 2026-05-11 20:08:44.625000
  Votes: 3  Comments: 1

I don't know if it will be of any use, however I'm sharing it just in case it might be helpful to someone.
`"""

General Equation for RMSE:

RMSE = sqrt( (1/n) * Σ(y_i - ŷ_i)² )

Definition of the Sum of Squared Residuals (SSR):

SSR = Σ(y_i - ŷ_i)²

Simplified Form (for n = 14151):

RMSE = 8.41e-3 * sqrt(SSR)

Interpretation:

Since k = 1/sqrt(14151) ≈ 8.41e-3, the RMSE is defined as

the product of the statistical scale factor and the linear magnitude of the total error (L2 norm).

"""

import numpy as np

import matplotlib.pyplot as plt

ssr = np.linspace(0, 3700000, 1000)

k = 8.41e-3 

y = k * np.sqrt(ssr)

plt.figure(figsize=(10, 6))

plt.plot(ssr, y, color='black', linewidth=2, label='RMSE')

plt.fill_between(ssr, 8, 16, color='r', alpha=0.3, label='Critical region')

plt.fill_between(ssr, 2, 8, color='g', alpha=0.3, label='Optimal region')

plt.fill_between(ssr, 0.9, 2, color='b', alpha=0.3, label='Excellent region')

plt.fill_between(ssr, 0, 0.9, color='y', alpha=0.3, label='Overfitting')

plt.title('RMSE vs SSR (n = 14151)')

plt.xlabel('SSR (ft²)')

plt.ylabel('RMSE (ft)')

plt.xlim(0, 3700000)

plt.ylim(0, 18)

plt.legend(loc='upper left')

plt.grid(True, linestyle='--', alpha=0.6)

plt.show()`

Comments:
├─ jose mata (2026-05-11 20:09:23.287000) [+0]
