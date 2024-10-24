## 4
#### a) Is this matrix symmetric?
- $A_3$ is symmetric. $A_1$ and $A_2$ are not symmetric.

#### b) Is this matrix diagonally dominant?
- Neither $A_1$ nor $A_2$ nor $A_3$ is diagonally dominant. Refer to the is_diagonally_dominant function for methodology.

#### c) Based on your answers to the preceding questions, what would be a good numerical method to solve a system of equations where this was the system matrix?
- Gaussian elimination with back-substitution would be the best numerical method to use to solve matrices $A_1$ and $A_2$ since they are both non-symmetric and non-diagonally-dominant, so methods such as the Conjugate Gradient algorithm or the Gauss-Seidel algorithm will not be suitable. However, for matrix $A_3$, Conjugate Gradient method would be the best method to use because the matrix is symmetric.

## 5
#### For each of the matrices, determine if it is well-conditioned, ill-conditioned, or singular, and determine its condition number. Explain how you arrived at your answer.
- Matrix $A_4$ is ill-conditioned because all of the values in the diagonal of the matrix are eigenvalues, and when the largest eigenvalue is divided by the smallest eigenvalue, the condition number can be calculated: $\frac{1000}{5} = 200$. The higher the condition number, the worse conditioned the number is.
- Matrix $A_5$ is ill-conditioned because when this matrix is plugged into the numpy function for finding a condition number, the condition number comes out to 27,478.573.
- Matrix $A_6$ is singular because the columns are multiples of each other, so the bottom row will come out to all zeros, which means that there is no eigenvalue of the last row, so a condition number cannot be found.
- Matrix $A_7$ is well-conditioned because like $A_4$ the eigenvalues are on the diagonal so when we divide the largest eigenvalue (1.5) by the smallest eigenvalue (1), we find that the condition number is 1.5.

## 6
#### a) What problem will you encounter if you perform Gaussian elimination on matrix $A_s$?
- If you perform Gaussian elimination on the matrix, the last row will come out to all zeroes because the last two rows are multiples of each other.
#### b) Is $A_s$ well-conditioned, ill-conditioned, or singular, and why?
- $A_s$ is singular because the last row comes out to all zeroes, meaning that the number of rows in the matrix does not match the number of equations.
#### c) What would be an alternative method to solving linear systems that have $A_s$ as their system matrix?
- There is no alternative method for solving linear systems that have $A_s$ as their system matrix because $A_s$ has no inverse.