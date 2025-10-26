# house-value-prediction-assignment
Tried all methods of Linear Regression and Ensemble Learning from ML Course Part - 1

Detailed report and Model Summary is available in Exploratory Data Analysis Notebook / PDF (`exploratory_data_analysis.ipynb`) 

Data was noticeable skewed for some columns, so Box-Cox Transformation was applied.  
### Why BoxCox is Superior to Log Transformation

BoxCox finds the **optimal transformation parameter (λ)** for each feature individually, rather than blindly applying log to everything.

The BoxCox Family of Transformations:
$$
y(\lambda) = 
\begin{cases}
\frac{y^\lambda - 1}{\lambda} & \text{if } \lambda \neq 0 \\
\log(y) & \text{if } \lambda = 0
\end{cases}
$$

where λ is chosen to maximize normality (minimize skewness).
