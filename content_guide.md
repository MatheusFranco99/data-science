# Data science steps and content guide

## Data profiling
- granularity, distribution, dimensionality, sparsity

## Feature engineering
- Data Cleaning
    - outliers detection and removal (boxplots, manual removal with scatter plot, z-score, IQR, isolation forest, Local Outlier Factor (LOF))
- Data Preparation
    - missing values imputation
    - dummification
    - normalization  (min-max scaling, z-score transformation, log transformation, box-cox transformation, quantile normalization, robust scaling)
- Data Balancing
    - class weights, cost-sensitive learning, undersampling, near miss undersampling, oversampling, SMOTE (Syntheic Minority Over-sampling technique), ensemble methods (baging, boosting,...)
- Feature extraction
    - features interactions (e.g. feature area, price -> price per unit area). Common text-relted methods as: bag of words (BoW), TF-IDF, BM25, stemming, tokenization, etc.
- Feature transformation
    - common with signals with methods as: Wavelet transform, Fourier transform, binning / discretization (dividing countinuous numerical features into bins (intervals)), enconding (categorical features into numerical (e.g. one-hot encoding, label encoding, target encoding)),
- Feature Selection
    - dimensionality reduction (PCA, LDA, t-SNE), redundant, irrelevant, noise feature removal, randomized methods, correlation-based (pearson correlation coefficient, spearman rank correlation), filter methods (chi-square test, ANOVA, information gain)

## Model Selection
- Linear models:
    - linear regression, logistic regression, ridge regression, lasso regression
- Tree-based models:
    - decision trees, random forest, gradient boosting trees, XGBosst, LightGBM, CatBoost
- Neural Networks:
    - multi-layer perceptron, convolutional neural networks, recurrent neural networks, autoencoders, generative adversarial networks
- Supposert Vector Machine:
    - Linear SVM, Non-Linear SVM
- Clusteering models:
    - K-Means, Hierarchical, DBSCAN, Gaussian Mixture models
- Ensemble Models:
    - Bagging, Boosting, Stacking
- Bayesian Models:
    - Naive Bayes, Bayesian Networks, Gaussian Processes
- Evolutionary models:
    - genetic algorithms, particle swarm optimization
- k-Nearest Neighbors

## Model Tuning

## Model Evaluation
- Metrics: Accuracy, Precision, Recall, F1 score, ROC AUC score, Cohen's Kappa, Matthews correlation coefficient (MCC), Balanced accuracy (BA), G-mean, Classification error, sensitivity, specificity, Log loss