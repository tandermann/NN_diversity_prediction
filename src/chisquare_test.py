import numpy as np
import scipy.stats as stats

good_counts = np.array([47,41,41])
medium_counts = np.array([9,8,10])
bad_counts = np.array([3,10,8])

data = np.array([good_counts,medium_counts,bad_counts])
data.sum(axis=1)
data.sum(axis=0)/data.sum()
data.sum(axis=1)/data.sum()

observed_counts = data
h0_ratios = np.array([1/3,1/3,1/3]) # null hypothesis, they are all the same between classes
expected_counts = (sum(observed_counts)*h0_ratios).astype(int)
chi_squared_stat = (((observed_counts-expected_counts)**2)/expected_counts).sum()

crit = stats.chi2.ppf(q = 0.95, # Find the critical value for 95% confidence*
                      df = len(good_counts)-1)   # Df = number of variable categories - 1

print("Critical value")
print(crit)

p_value = 1 - stats.chi2.cdf(x=chi_squared_stat,  # Find the p-value
                             df=len(good_counts)-1)
print("P value")
print(p_value**2)

#array = np.random.choice(['low','medium','high'],p=[0.2,0.3,0.5],size=300)
#np.savetxt('/Users/tobiasandermann/Desktop/sim_values.txt',array,fmt='%s')


import calendar
print(calendar.calendar(9999999999999999))
