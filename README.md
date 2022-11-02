# OGA_HDIC_Trim

The first step is to sequentially select input variables via orthogonal greedy algorithm (OGA). The second step is to determine the number of OGA iterations using high-dimensional information criterion (HDIC). The third step is to remove irrelevant variables remaining in the second step using HDIC.

# Usage

Ohit(X,y,Kn = None,c1 = 5,c2 = 2,c3 = 2,HDIC_Type='HDAIC',init = None,conf = 0.01,intercept = True)

# Arguments

- X input numpy/dataframe/list of n rows and p columns.
- y Response of length n.
- Kn The numbert of OGA iterations. Kn must be a positive integer between 1 and p. Default is Kn=max(1, min(floor(c1*sqrt(n/log(p))), p)), where c1 is a tuning parameter.
- c1 The tuning parameter for the number of OGA iterations. Default is c1=5.
-c2	The tuning parameter for HDIC_Type="HDAIC". Default is c2=2.
-c3	The tuning parameter for HDIC_Type="HDHQ". Default is c3=2.01.
- HDIC_Type	High-dimensional information criterion. The value must be "HDAIC", "HDBIC" or "HDHQ". The formula is n*log(rmse)+k_use*omega_n*log(p) where rmse is the residual mean squared error and k_use is the number of variables used to fit the model. For HDIC_Type="HDAIC", it is HDIC with omega_n=c2. For HDIC_Type="HDBIC", it is HDIC with omega_n=log(n). For HDIC_Type="HDHQ", it is HDIC with omega_n=c3*log(log(n)). Default is HDIC_Type="HDAIC".
- inint The user specifed set which need to be added in OGA and final set. Defult is None.
- conf The confidence level used in predictions.
- intercept If True, the final model will include intercept. Defult is None.

# References
Ing, C.-K. and Lai, T. L. (2011). A stepwise regression method and consistent model selection for high-dimensional sparse linear models. Statistica Sinica, 21, 1473–1513.
