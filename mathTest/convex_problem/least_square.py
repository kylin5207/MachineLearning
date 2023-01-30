import cvxpy as cp
import numpy as np

"""
cvxpy包用于计算凸优化问题
例如解决带有约束条件的最小二乘问题
"""

# Problem data.
m = 30
n = 20
np.random.seed(1)
A = np.random.randn(m, n)
b = np.random.randn(m)

# Construct the problem.
# 定义变量的个数
x = cp.Variable(n)
# 最小化问题
objective = cp.Minimize(cp.sum_squares(A @ x - b))
# 约束条件
constraints = [0 <= x, x <= 1]
# 定义问题
prob = cp.Problem(objective, constraints)

# The optimal objective value is returned by `prob.solve()`.
result = prob.solve()

# The optimal value for x is stored in `x.value`.
print("====x的取值=====")
print(x.value)

# 约束的最佳拉格朗日乘数存储在constraint.dual_value中
print("====最佳拉格朗日乘数===")
print(constraints[0].dual_value)