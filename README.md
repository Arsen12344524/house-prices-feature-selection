# House Price Prediction: Feature Selection Analysis

This project explores various **Feature Selection** techniques to optimize a regression model for predicting house prices. The goal is to identify which methods best capture the relationship between house characteristics and their final sale price.

##  Project Objective

The main focus is to compare three distinct approaches to feature selection:

**Filter Method**: Using statistical tests (`SelectKBest` with `f_regression`) to rank features.


**Wrapper Method**: Using Recursive Feature Elimination (`RFE`) to find the best feature combinations.


**Embedded Method**: Leveraging `Random Forest` feature importance to select variables during the model training process.



## Tech Stack

**Language**: Python 


**Libraries**: Pandas, NumPy, Scikit-Learn 


**Dataset**: House Prices (Advanced Regression Techniques) 



##  Methodology & Workflow

1. **Data Preprocessing**: Handled missing values and scaled 14 numerical features using `StandardScaler`.


2. **Baseline Model**: Trained a Linear Regression model on all 14 features to establish a performance benchmark.


3. **Feature Selection ( )**:
* Selected the top 7 features using each of the three methods mentioned above.


* Retrained the Linear Regression model on these subsets to compare MSE and  metrics.



##  Key Insights

**Filter vs. Wrapper**: Unlike filter methods that evaluate features individually, **RFE** can find better combinations even if individual features have low correlation with the target.



**L1 vs. L2 Regularization**: This project clarifies why **L1 (Lasso)** acts as a feature selection tool by shrinking coefficients to zero, whereas L2 (Ridge) only minimizes them.
