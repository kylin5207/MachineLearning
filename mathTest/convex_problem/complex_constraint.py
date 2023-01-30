import cvxpy as cp
import numpy as np
import pandas as pd
"""
cvxpy包用于计算凸优化问题
"""

data = data = pd.read_excel("测试数据集.xlsx", header=0)
n = data.shape[0]
print(data.head())

a = data["an"].values
b = data["bn"].values
power_lower = data["lower Pn"].values
power_upper = data["Upper Pn"].values

# Construct the problem.
x = cp.Variable(n)

objective = cp.Minimize(cp.sum(0.5 * a @ x**2 + b @ x))
constraints = [power_lower <= x, x <= power_upper, cp.sum(x) == 0]
prob = cp.Problem(objective, constraints)
result = prob.solve()

print("===x value======")
print(x.value)
print("====result====")
print(result)

value_df = pd.DataFrame(x.value, columns=["power"])
value_df.to_csv("truth_value.csv", index=False)
