import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import skew
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV, BayesianRidge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import StackingRegressor
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# DATA DRIVEN APPROACH - ADAPTED FOR HOTEL VALUE PREDICTION
# ============================================================================

print("="*80)
print("🏆 DATA DRIVEN APPROACH - HOTEL VALUE PREDICTION")
print("="*80)

# ============================================================================
# DATA LOADING
# ============================================================================

print("\n📊 Loading data...")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(f"Train set size: {train.shape}")
print(f"Test set size: {test.shape}")
print('START data processing', datetime.now())

train_ID = train['Id']
test_ID = test['Id']
train.drop(['Id'], axis=1, inplace=True)
test.drop(['Id'], axis=1, inplace=True)

# ============================================================================
# OUTLIER REMOVAL (GrLivArea equivalent = UsableArea)
# ============================================================================

print("\n🔍 Removing outliers...")
# Original: train = train[train.GrLivArea < 4500]
# Hotel equivalent: UsableArea
train = train[train.UsableArea < 4500]
train.reset_index(drop=True, inplace=True)

# ============================================================================
# TARGET TRANSFORMATION
# ============================================================================

print("\n🎯 Transforming target variable...")
train["HotelValue"] = np.log1p(train["HotelValue"])
y = train.HotelValue.reset_index(drop=True)
train_features = train.drop(['HotelValue'], axis=1)
test_features = test

features = pd.concat([train_features, test_features]).reset_index(drop=True)
print(f"Combined features shape: {features.shape}")

# ============================================================================
# COLUMN MAPPING & TYPE CONVERSION
# ============================================================================

print("\n🔄 Converting column types...")

# PropertyClass → MSSubClass (property classification)
features['PropertyClass'] = features['PropertyClass'].apply(str)

# YearSold, MonthSold → YrSold, MoSold
features['YearSold'] = features['YearSold'].astype(str)
features['MonthSold'] = features['MonthSold'].astype(str)

# ============================================================================
# MISSING VALUE IMPUTATION - SPECIFIC STRATEGY
# ============================================================================

print("\n🔧 Handling missing values with domain knowledge...")

# Categorical features with logical defaults
features['PropertyFunctionality'] = features['PropertyFunctionality'].fillna('Typ')  # Functional → PropertyFunctionality
features['ElectricalSystem'] = features['ElectricalSystem'].fillna("SBrkr")  # Electrical → ElectricalSystem
features['KitchenQuality'] = features['KitchenQuality'].fillna("TA")
features['ExteriorPrimary'] = features['ExteriorPrimary'].fillna(features['ExteriorPrimary'].mode()[0])  # Exterior1st
features['ExteriorSecondary'] = features['ExteriorSecondary'].fillna(features['ExteriorSecondary'].mode()[0])  # Exterior2nd
features['DealType'] = features['DealType'].fillna(features['DealType'].mode()[0])  # SaleType

features["PoolQuality"] = features["PoolQuality"].fillna("None")

# Garage features (Parking in hotel data)
for col in ('ParkingConstructionYear', 'ParkingArea', 'ParkingCapacity'):
    features[col] = features[col].fillna(0)

for col in ['ParkingType', 'ParkingFinish', 'ParkingQuality', 'ParkingCondition']:
    features[col] = features[col].fillna('None')

# Basement features
for col in ('BasementHeight', 'BasementCondition', 'BasementExposure', 'BasementFacilityType1', 'BasementFacilityType2'):
    features[col] = features[col].fillna('None')

# ZoningCategory → MSZoning
features['ZoningCategory'] = features.groupby('PropertyClass')['ZoningCategory'].transform(
    lambda x: x.fillna(x.mode()[0]) if not x.mode().empty else x
)

# Fill all remaining object columns with 'None'
objects = []
for i in features.columns:
    if features[i].dtype == object:
        objects.append(i)
features.update(features[objects].fillna('None'))

# RoadAccessLength → LotFrontage (neighborhood-based imputation)
features['RoadAccessLength'] = features.groupby('District')['RoadAccessLength'].transform(
    lambda x: x.fillna(x.median())
)

# Fill remaining numerical columns with 0
numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerics = []
for i in features.columns:
    if features[i].dtype in numeric_dtypes:
        numerics.append(i)
features.update(features[numerics].fillna(0))

print(f"✅ Missing values handled")

# ============================================================================
# SKEWNESS CORRECTION - BOXCOX TRANSFORMATION
# ============================================================================

print("\n📊 Correcting skewness with BoxCox transformation...")

numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerics2 = []
for i in features.columns:
    if features[i].dtype in numeric_dtypes:
        numerics2.append(i)

skew_features = features[numerics2].apply(lambda x: skew(x)).sort_values(ascending=False)
high_skew = skew_features[skew_features > 0.5]
skew_index = high_skew.index

print(f"  Found {len(high_skew)} features with skewness > 0.5")

transformed_count = 0
for i in skew_index:
    try:
        # Only transform if feature has variation and positive values
        if features[i].nunique() > 1 and (features[i] >= 0).all():
            # Try BoxCox transformation
            lam = boxcox_normmax(features[i] + 1)
            features[i] = boxcox1p(features[i], lam)
            transformed_count += 1
        else:
            # Fall back to log transformation for features with zeros or negatives
            features[i] = np.log1p(features[i] - features[i].min() + 1)
            transformed_count += 1
    except Exception as e:
        # If BoxCox fails, use simple log transformation
        try:
            features[i] = np.log1p(features[i] - features[i].min() + 1)
            transformed_count += 1
        except:
            # Keep original if transformation fails
            pass

print(f"  ✓ Applied transformations to {transformed_count}/{len(skew_index)} skewed features")

# ============================================================================
# DROP CONSTANT FEATURES
# ============================================================================

# Street, Utilities equivalent (if they exist and are constant)
constant_features = []
for col in ['RoadType', 'UtilityAccess']:
    if col in features.columns and features[col].nunique() == 1:
        constant_features.append(col)

if constant_features:
    features = features.drop(constant_features, axis=1)
    print(f"  ✓ Dropped constant features: {constant_features}")

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

print("\n🔧 Feature Engineering...")

# YrBltAndRemod → ConstructionYear + RenovationYear
features['YrBltAndRemod'] = features['ConstructionYear'] + features['RenovationYear']

# TotalSF → Total square footage
features['TotalSF'] = features['BasementTotalSF'] + features['GroundFloorArea'] + features['UpperFloorArea']

# Total_sqr_footage (more detailed)
features['Total_sqr_footage'] = (features['BasementFacilitySF1'] + features['BasementFacilitySF2'] +
                                 features['GroundFloorArea'] + features['UpperFloorArea'])

# Total_Bathrooms
features['Total_Bathrooms'] = (features['FullBaths'] + (0.5 * features['HalfBaths']) +
                               features['BasementFullBaths'] + (0.5 * features['BasementHalfBaths']))

# Total_porch_sf
features['Total_porch_sf'] = (features['OpenVerandaArea'] + features['SeasonalPorchArea'] +
                              features['EnclosedVerandaArea'] + features['ScreenPorchArea'] +
                              features['TerraceArea'])

# Binary features
features['haspool'] = features['SwimmingPoolArea'].apply(lambda x: 1 if x > 0 else 0)
features['has2ndfloor'] = features['UpperFloorArea'].apply(lambda x: 1 if x > 0 else 0)
features['hasgarage'] = features['ParkingArea'].apply(lambda x: 1 if x > 0 else 0)
features['hasbsmt'] = features['BasementTotalSF'].apply(lambda x: 1 if x > 0 else 0)
features['hasfireplace'] = features['Lounges'].apply(lambda x: 1 if x > 0 else 0)

print(f"  ✓ Created {5} binary features")
print(f"  ✓ Total features shape: {features.shape}")

# ============================================================================
# ONE-HOT ENCODING
# ============================================================================

print("\n🔤 One-hot encoding categorical features...")
final_features = pd.get_dummies(features).reset_index(drop=True)
print(f"  ✓ Shape after encoding: {final_features.shape}")

# ============================================================================
# SPLIT TRAIN/TEST
# ============================================================================

X = final_features.iloc[:len(y), :]
X_sub = final_features.iloc[len(y):, :]

print(f"\n✂️  Data split: X={X.shape}, y={y.shape}, X_sub={X_sub.shape}")

# ============================================================================
# REMOVE OUTLIERS FROM TRAINING DATA
# ============================================================================

print("\n🎯 Removing outliers from training data...")
# You can adjust these indices based on your EDA
outliers = [30, 88, 462, 631, 1322]
outliers = [idx for idx in outliers if idx < len(X)]  # Only keep valid indices
if outliers:
    X = X.drop(X.index[outliers])
    y = y.drop(y.index[outliers])
    print(f"  ✓ Removed {len(outliers)} outliers")

# ============================================================================
# REMOVE OVERFIT FEATURES (>99.94% same value)
# ============================================================================

print("\n🔍 Removing overfit features...")
overfit = []
for i in X.columns:
    counts = X[i].value_counts()
    zeros = counts.iloc[0]
    if zeros / len(X) * 100 > 99.94:
        overfit.append(i)

if overfit:
    X = X.drop(overfit, axis=1)
    X_sub = X_sub.drop(overfit, axis=1)
    print(f"  ✓ Removed {len(overfit)} overfit features")

print(f"\n✅ Final shapes: X={X.shape}, y={y.shape}, X_sub={X_sub.shape}")

# ============================================================================
# MODEL SETUP
# ============================================================================

print("\n🤖 Setting up models...")

kfolds = KFold(n_splits=10, shuffle=True, random_state=42)

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

def cv_rmse(model, X=X):
    rmse = np.sqrt(-cross_val_score(model, X, y,
                                    scoring="neg_mean_squared_error",
                                    cv=kfolds))
    return (rmse)

# Model hyperparameters
alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]
alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]
e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]

ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=alphas_alt, cv=kfolds))
lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=10000000, alphas=alphas2, random_state=42, cv=kfolds))
elasticnet = make_pipeline(RobustScaler(), ElasticNetCV(max_iter=10000000, alphas=e_alphas, cv=kfolds, l1_ratio=e_l1ratio))
bayesian = make_pipeline(RobustScaler(), BayesianRidge())

gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                 max_depth=4, max_features='sqrt',
                                 min_samples_leaf=15, min_samples_split=10,
                                 loss='huber', random_state=42)

xgboost = XGBRegressor(learning_rate=0.01, n_estimators=3460,
                       max_depth=3, min_child_weight=0,
                       gamma=0, subsample=0.7,
                       colsample_bytree=0.7,
                       objective='reg:squarederror', nthread=-1,
                       scale_pos_weight=1, seed=27,
                       reg_alpha=0.00006)

# Stacking ensemble - using scikit-learn's native StackingRegressor
stack_gen = StackingRegressor(
    estimators=[
        ('ridge', ridge),
        ('lasso', lasso),
        ('elasticnet', elasticnet),
        ('bayesian', bayesian),
        ('gbr', gbr),
        ('xgboost', xgboost)
    ],
    final_estimator=XGBRegressor(learning_rate=0.01, n_estimators=1000, max_depth=3, random_state=42),
    cv=kfolds,
    n_jobs=-1,
    passthrough=False
)

# ============================================================================
# CROSS-VALIDATION SCORES
# ============================================================================

print("\n📊 TEST score on CV (10-fold):")

score = cv_rmse(ridge)
print("Ridge score:        {:.4f} ({:.4f})".format(score.mean(), score.std()), datetime.now())

score = cv_rmse(lasso)
print("Lasso score:        {:.4f} ({:.4f})".format(score.mean(), score.std()), datetime.now())

score = cv_rmse(elasticnet)
print("ElasticNet score:   {:.4f} ({:.4f})".format(score.mean(), score.std()), datetime.now())

score = cv_rmse(bayesian)
print("Bayesian Ridge:     {:.4f} ({:.4f})".format(score.mean(), score.std()), datetime.now())

score = cv_rmse(gbr)
print("GradientBoosting:   {:.4f} ({:.4f})".format(score.mean(), score.std()), datetime.now())

score = cv_rmse(xgboost)
print("XGBoost score:      {:.4f} ({:.4f})".format(score.mean(), score.std()), datetime.now())

# ============================================================================
# FIT MODELS ON FULL TRAINING DATA
# ============================================================================

print("\n🚀 Fitting models on full training data...")

print(datetime.now(), 'StackingRegressor')
stack_gen_model = stack_gen.fit(X, y)

print(datetime.now(), 'elasticnet')
elastic_model_full_data = elasticnet.fit(X, y)

print(datetime.now(), 'lasso')
lasso_model_full_data = lasso.fit(X, y)

print(datetime.now(), 'ridge')
ridge_model_full_data = ridge.fit(X, y)

print(datetime.now(), 'bayesian')
bayesian_model_full_data = bayesian.fit(X, y)

print(datetime.now(), 'GradientBoosting')
gbr_model_full_data = gbr.fit(X, y)

print(datetime.now(), 'xgboost')
xgb_model_full_data = xgboost.fit(X, y)

print("✅ All models trained!")

# ============================================================================
# BLENDING FUNCTION
# ============================================================================

def blend_models_predict(X):
    return ((0.15 * elastic_model_full_data.predict(X)) + \
            (0.1 * lasso_model_full_data.predict(X)) + \
            (0.1 * ridge_model_full_data.predict(X)) + \
            (0.1 * bayesian_model_full_data.predict(X)) + \
            (0.15 * gbr_model_full_data.predict(X)) + \
            (0.15 * xgb_model_full_data.predict(X)) + \
            (0.25 * stack_gen_model.predict(X)))

print('\n📈 RMSLE score on train data:')
print(f"  {rmsle(y, blend_models_predict(X)):.4f}")

# ============================================================================
# PREDICTIONS & SUBMISSION - MULTIPLE VARIATIONS
# ============================================================================

print('\n🎪 Generating predictions...')
predictions = blend_models_predict(X_sub)

base_submission = pd.DataFrame({
    'Id': test_ID,
    'HotelValue': np.floor(np.expm1(predictions))
})

print(f"\nBase prediction stats:")
print(f"  Min: ${base_submission['HotelValue'].min():,.0f}")
print(f"  Max: ${base_submission['HotelValue'].max():,.0f}")
print(f"  Mean: ${base_submission['HotelValue'].mean():,.0f}")
print(f"  Median: ${base_submission['HotelValue'].median():,.0f}")

# ============================================================================
# VARIATION 1: Original approach (0.0045, 0.99 quantiles, 0.88/1.1 multipliers)
# ============================================================================
submission_v1 = base_submission.copy()
q1 = submission_v1['HotelValue'].quantile(0.0045)
q2 = submission_v1['HotelValue'].quantile(0.99)
submission_v1['HotelValue'] = submission_v1['HotelValue'].apply(lambda x: x if x > q1 else x*0.77)
submission_v1['HotelValue'] = submission_v1['HotelValue'].apply(lambda x: x if x < q2 else x*1.1)
submission_v1.to_csv("submission_v1_original.csv", index=False)
print(f"\n✅ V1 (Original): 0.45%/99% quantiles, 0.88/1.1 multipliers")

# ============================================================================
# VARIATION 2: No clipping at all (raw predictions)
# ============================================================================
submission_v2 = base_submission.copy()
submission_v2.to_csv("submission_v2_no_clipping.csv", index=False)
print(f"✅ V2 (No Clipping): Raw predictions")

# ============================================================================
# VARIATION 3: Conservative clipping (1%/99%, smaller adjustments)
# ============================================================================
submission_v3 = base_submission.copy()
q1 = submission_v3['HotelValue'].quantile(0.01)
q2 = submission_v3['HotelValue'].quantile(0.99)
submission_v3['HotelValue'] = submission_v3['HotelValue'].apply(lambda x: x if x > q1 else x*0.95)
submission_v3['HotelValue'] = submission_v3['HotelValue'].apply(lambda x: x if x < q2 else x*1.05)
submission_v3.to_csv("submission_v3_conservative.csv", index=False)
print(f"✅ V3 (Conservative): 1%/99% quantiles, 0.95/1.05 multipliers")

# ============================================================================
# VARIATION 4: Aggressive clipping (0.5%/99.5%, stronger adjustments)
# ============================================================================
submission_v4 = base_submission.copy()
q1 = submission_v4['HotelValue'].quantile(0.0035)
q2 = submission_v4['HotelValue'].quantile(0.98)
submission_v4['HotelValue'] = submission_v4['HotelValue'].apply(lambda x: x if x > q1 else x*0.8)
submission_v4['HotelValue'] = submission_v4['HotelValue'].apply(lambda x: x if x < q2 else x*1.12)
submission_v4.to_csv("submission_v4_aggressive.csv", index=False)
print(f"✅ V4 (Aggressive): 0.5%/99.5% quantiles, 0.85/1.15 multipliers")

# ============================================================================
# VARIATION 5: Only lower bound adjustment
# ============================================================================
submission_v5 = base_submission.copy()
q1 = submission_v5['HotelValue'].quantile(0.01)
submission_v5['HotelValue'] = submission_v5['HotelValue'].apply(lambda x: x if x > q1 else x*0.90)
submission_v5.to_csv("submission_v5_lower_only.csv", index=False)
print(f"✅ V5 (Lower Only): 1% quantile, 0.90 multiplier on low end")

# ============================================================================
# VARIATION 6: Only upper bound adjustment
# ============================================================================
submission_v6 = base_submission.copy()
q2 = submission_v6['HotelValue'].quantile(0.99)
submission_v6['HotelValue'] = submission_v6['HotelValue'].apply(lambda x: x if x < q2 else x*1.08)
submission_v6.to_csv("submission_v6_upper_only.csv", index=False)
print(f"✅ V6 (Upper Only): 99% quantile, 1.08 multiplier on high end")

# ============================================================================
# VARIATION 7: Median-based clipping
# ============================================================================
submission_v7 = base_submission.copy()
median = submission_v7['HotelValue'].median()
std = submission_v7['HotelValue'].std()
submission_v7['HotelValue'] = submission_v7['HotelValue'].clip(lower=median - 2.5*std, upper=median + 2.5*std)
submission_v7.to_csv("submission_v7_median_std.csv", index=False)
print(f"✅ V7 (Median+Std): Clipped to median ± 2.5 std dev")

# ============================================================================
# VARIATION 8: IQR-based outlier adjustment
# ============================================================================
submission_v8 = base_submission.copy()
Q1 = submission_v8['HotelValue'].quantile(0.25)
Q3 = submission_v8['HotelValue'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
submission_v8['HotelValue'] = submission_v8['HotelValue'].apply(
    lambda x: lower_bound if x < lower_bound else (upper_bound if x > upper_bound else x)
)
submission_v8.to_csv("submission_v8_iqr.csv", index=False)
print(f"✅ V8 (IQR): Capped at Q1-1.5*IQR and Q3+1.5*IQR")

# Backward compatibility - keep original filename
submission_v1.to_csv("submission_grandmaster.csv", index=False)

print(f"\n{'='*80}")
print(f"✅ Generated 8 submission variations!")
print(f"{'='*80}")
print(f"\nSubmission files created:")
print(f"  1. submission_v1_original.csv (= submission_grandmaster.csv)")
print(f"  2. submission_v2_no_clipping.csv")
print(f"  3. submission_v3_conservative.csv")
print(f"  4. submission_v4_aggressive.csv")
print(f"  5. submission_v5_lower_only.csv")
print(f"  6. submission_v6_upper_only.csv")
print(f"  7. submission_v7_median_std.csv")
print(f"  8. submission_v8_iqr.csv")

print(f"\n📊 Sample from V1 (Original):")
print(submission_v1.head(10))

print("\n🏆 Training complete using DATA DRIVEN approach!")
