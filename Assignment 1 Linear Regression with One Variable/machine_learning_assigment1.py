import numpy as np
import matplotlib.pyplot as plt

# Load and divade the columns
data = np.loadtxt("Assignment1data.txt", delimiter=',')
population = data[:, 0]
profit = data[:, 1]

print(population)
print(profit)


# Visualize Dataset
plt.scatter(population, profit, color='red', label='Data points')
plt.xlabel('Population')
plt.ylabel('Profit')
plt.title('City Population vs. Profit')
plt.show()

## Calculate Linear Regression 
#  means
population_mean = np.mean(population)
profit_mean = np.mean(profit)

# slope 
numerator = np.sum((population - population_mean) * (profit - profit_mean))  # top part of the slope formula
denominator = np.sum((population - population_mean) ** 2)
slope = numerator / denominator

#intercept
intercept = profit_mean - slope * population_mean

# The line equation 
line_y = slope * population + intercept

#Prediction
predicted_profit_35k = slope * 3.5 + intercept  
predicted_profit_70k = slope * 7.0 + intercept  

print(f"Predicted profit for a population of 35,000: ${predicted_profit_35k * 10_000:.2f}")
print(f"Predicted profit for a population of 70,000: ${predicted_profit_70k * 10_000:.2f}")

# Vizualization
plt.scatter(population, profit, color='red', label='Data points')
plt.xlabel('Population')
plt.ylabel('Profit')
plt.title('City Population vs. Profit')
plt.plot(population, line_y, color='red', label='Regression Line')
plt.legend()
 
plt.show()
