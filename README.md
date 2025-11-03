# house-value-prediction-assignment
Tried all methods of Linear Regression and Ensemble Learning from ML Course Part - 1

Summarized report at `hotel_value_chicken_biryani.pdf`

Detailed report and Model Summary is available in Exploratory Data Analysis Notebook / PDF (`exploratory_data_analysis.ipynb`) 

Data was noticeable skewed for some columns, so Box-Cox Transformation was applied.  

<img width="777" height="179" alt="Screenshot 2025-10-26 at 6 52 32 PM" src="https://github.com/user-attachments/assets/740a65bf-1ecc-4025-988d-3dba4670e34d" />

Log RMSLE: 0.0674  
Kaggle Score: 17900~18000  

================================================================================
🏆 DATA DRIVEN APPROACH - HOTEL VALUE PREDICTION
================================================================================

📊 Loading data...
Train set size: (1200, 81)
Test set size: (260, 80)
START data processing 2025-11-03 18:35:29.970590

🔍 Removing outliers...

🎯 Transforming target variable...
Combined features shape: (1458, 79)

🔄 Converting column types...

🔧 Handling missing values with domain knowledge...
✅ Missing values handled

📊 Correcting skewness with BoxCox transformation...
  Found 25 features with skewness > 0.5
  ✓ Applied transformations to 25/25 skewed features

🔧 Feature Engineering...
  ✓ Created 5 binary features
  ✓ Total features shape: (1458, 89)

🔤 One-hot encoding categorical features...
  ✓ Shape after encoding: (1458, 340)

✂️  Data split: X=(1198, 340), y=(1198,), X_sub=(260, 340)

🎯 Removing outliers from training data...
  ✓ Removed 4 outliers

🔍 Removing overfit features...
  ✓ Removed 2 overfit features

✅ Final shapes: X=(1194, 338), y=(1194,), X_sub=(260, 338)

🤖 Setting up models...

📊 TEST score on CV (10-fold):
Ridge score:        0.1080 (0.0071) 2025-11-03 18:35:32.655931
Lasso score:        0.1074 (0.0074) 2025-11-03 18:35:38.458851
ElasticNet score:   0.1072 (0.0075) 2025-11-03 18:36:03.620368
Bayesian Ridge:     0.1082 (0.0070) 2025-11-03 18:36:04.123965
GradientBoosting:   0.1108 (0.0119) 2025-11-03 18:36:40.935494
XGBoost score:      0.1138 (0.0094) 2025-11-03 18:37:14.034557

🚀 Fitting models on full training data...
2025-11-03 18:37:14.034589 StackingRegressor
2025-11-03 18:38:09.321620 elasticnet
2025-11-03 18:38:11.732026 lasso
2025-11-03 18:38:12.233450 ridge
2025-11-03 18:38:12.510558 bayesian
2025-11-03 18:38:12.561596 GradientBoosting
2025-11-03 18:38:16.587748 xgboost
✅ All models trained!

📈 RMSLE score on train data:
  0.0685

🎪 Generating predictions...

Base prediction stats:
  Min: $52,778
  Max: $629,494
  Mean: $176,161
  Median: $149,475

✅ V1 (Original): 0.45%/99% quantiles, 0.88/1.1 multipliers
✅ V2 (No Clipping): Raw predictions
✅ V3 (Conservative): 1%/99% quantiles, 0.95/1.05 multipliers
✅ V4 (Aggressive): 0.5%/99.5% quantiles, 0.85/1.15 multipliers
✅ V5 (Lower Only): 1% quantile, 0.90 multiplier on low end
✅ V6 (Upper Only): 99% quantile, 1.08 multiplier on high end
✅ V7 (Median+Std): Clipped to median ± 2.5 std dev
✅ V8 (IQR): Capped at Q1-1.5*IQR and Q3+1.5*IQR

================================================================================
✅ Generated 8 submission variations!
================================================================================

Submission files created:
  1. submission_v1_original.csv (= submission_grandmaster.csv)
  2. submission_v2_no_clipping.csv
  3. submission_v3_conservative.csv
  4. submission_v4_aggressive.csv
  5. submission_v5_lower_only.csv
  6. submission_v6_upper_only.csv
  7. submission_v7_median_std.csv
  8. submission_v8_iqr.csv

📊 Sample from V1 (Original):
     Id  HotelValue
0   893    145873.0
1  1106    332263.0
2   414    103135.0
3   523    156950.0
4  1037    322158.0
5   615     74631.0
6   219    240348.0
7  1161    141124.0
8   650     74004.0
9   888    142880.0

🏆 Training complete using DATA DRIVEN approach!
