# DAY 3: Data Cleaning & Feature Engineering
# ============================================

print("Day 3: Data Cleaning & Feature Engineering!")
print("=" * 60)

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

print("Libraries imported successfully!")

# Load the cleaned data from yesterday
print("\nLoading cleaned data from Day 2...")
try:
    df = pd.read_csv('telco_churn_cleaned.csv')
    print(f"Data loaded! Shape: {df.shape}")
except:
    print("Couldn't find cleaned data, loading fresh...")
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    df = pd.read_csv(url)
    
    # Basic cleaning (from Day 2)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    zero_tenure_mask = df['tenure'] == 0
    nan_mask = df['TotalCharges'].isna()
    df.loc[zero_tenure_mask & nan_mask, 'TotalCharges'] = 0
    median_charges = df['TotalCharges'].median()
    df.loc[df['TotalCharges'].isna(), 'TotalCharges'] = median_charges

print(f"\n📊 Initial data shape: {df.shape}")
print(f"📋 Columns: {list(df.columns)}")

print("STEP 1: DATA QUALITY AUDIT")
print("=" * 60)

def audit_data_quality(df):
    """Perform comprehensive data quality check"""
    
    print("\nData Quality Report:")
    print("-" * 40)
    
    # 1. Missing values
    print("\nMissing Values Analysis:")
    missing_data = df.isnull().sum()
    missing_percentage = (missing_data / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Missing Count': missing_data,
        'Percentage': missing_percentage
    })
    
    missing_df = missing_df[missing_df['Missing Count'] > 0]
    
    if len(missing_df) > 0:
        print("Found missing values:")
        print(missing_df)
    else:
        print("No missing values found!")

    #2.duplicates
    duplicates=df.duplicated().sum()
    print(f'The Total Number of Duplicates:{duplicates}')
    if duplicates >0:
        print(f'Removing the {duplicates}')
        df=df.drop_duplicates()
    else:
        print('No Duplicates Found')


    #3.Datatype Checks
    print('Data type Analysis')
    dtype_df=pd.DataFrame(df.dtypes,columns=['Data Type'])
    dtype_df['Count']=1
    dtype_summary=dtype_df.groupby('Data Type').count()
    print(dtype_summary)

    # 4. Unique values per column
    print("\nUnique Values per Column:")
    unique_counts = {}
    for col in df.columns:
        unique_counts[col] = df[col].nunique()
    
    unique_df = pd.DataFrame.from_dict(unique_counts, orient='index', 
                                       columns=['Unique Values'])
    unique_df = unique_df.sort_values('Unique Values', ascending=False)
    print(unique_df.head(10))

    #zero/blank Values in Important Columns
    print("\n5️⃣ Zero/Blank Values in Key Columns:")
    key_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
    for col in key_columns:
        if col in df.columns:
            zeros = (df[col] == 0).sum()
            if pd.api.types.is_numeric_dtype(df[col]):
                blanks = df[col].isna().sum()
                print(f"   {col}: {zeros} zeros, {blanks} blanks")
    
    return df

# Run data audit
df= audit_data_quality(df)
print(f"\n📊 Data shape after audit: {df.shape}")
print("STEP 2: CATEGORICAL VARIABLE ANALYSIS")
categorical_cols=df.select_dtypes(include=['object']).columns.tolist()
if 'customerID' in categorical_cols:
    categorical_cols.remove('customerID')
if 'churn' in categorical_cols:
    categorical_cols.remove('Churn')

print(f'categorical col:{len(categorical_cols)} categorical columns')


cat_summary = []
for col in categorical_cols:
    unique_vals = df[col].unique()
    num_unique = len(unique_vals)
    cat_summary.append({
        'Column': col,
        'Unique Values': num_unique,
        'Values': str(unique_vals[:5])[1:-1] + ('...' if num_unique > 5 else '')
    })
cat_df=pd.DataFrame(cat_summary)
print(cat_df.to_string(index=False))


# Visualize distribution of categorical variables
fig,axes=plt.subplots(4,3,figsize=(15,12))
axes=axes.flatten()

for idx,col in enumerate(categorical_cols[:12]):
    ax=axes[idx]

    value_counts=df[col].value_counts().head(10)

    bar=ax.bar(range(len(value_counts)),value_counts.values,
               color=plt.cm.Set3(range(len(value_counts))))
    
    ax.set_title(f'{col}',fontsize=10,fontweight='bold')
    ax.set_label('')
    ax.set_ylabel(range(len(value_counts)))
    ax.set_xticklabels(value_counts.index,rotation=45,ha='right',fontsize=7)

    for i,v in enumerate(value_counts.values):
        ax.text(i,v,str(v),ha='center',va='bottom',fontsize=7)


for idx in range(len(categorical_cols[:12]),len(axes)):
    axes[idx].set_visibile(False)

plt.suptitle('Distribution of Categorical Variables', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('categorical_distributions.png', dpi=100, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("STEP 3: FEATURE ENGINEERING")
print("=" * 60)

print("\n🔧 Creating new features from existing data...")

# Create a copy for feature engineering
df_features = df.copy()

# 1. Customer Value Features
print("\n1️⃣ Customer Value Features:")

# Monthly charge per service (estimate)
df_features['AvgChargePerMonth'] = df_features['TotalCharges'] / (df_features['tenure'] + 1)

# Charge ratio
df_features['MonthlyToTotalRatio'] = df_features['MonthlyCharges'] / (df_features['TotalCharges'] + 1)

# Customer lifetime value estimate
df_features['EstimatedLifetimeValue'] = df_features['MonthlyCharges'] * 12 * 3  # 3 years projection

print(f"   Created: AvgChargePerMonth, MonthlyToTotalRatio, EstimatedLifetimeValue")

# 2. Service Count Features
print("\n2️⃣ Service Count Features:")

# List of service columns
service_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 
                'OnlineBackup', 'DeviceProtection', 'TechSupport',
                'StreamingTV', 'StreamingMovies']

# Convert services to binary (Yes=1, No/No service=0)
for col in service_cols:
    if col in df_features.columns:
        df_features[f'{col}_binary'] = df_features[col].apply(
            lambda x: 1 if x in ['Yes', 'DSL', 'Fiber optic'] else 0
        )

# Total number of services
binary_cols = [col for col in df_features.columns if '_binary' in col]
df_features['TotalServices'] = df_features[binary_cols].sum(axis=1)

# Premium services count (streaming, security, etc)
premium_services = ['OnlineSecurity_binary', 'OnlineBackup_binary', 
                    'DeviceProtection_binary', 'TechSupport_binary',
                    'StreamingTV_binary', 'StreamingMovies_binary']
df_features['PremiumServices'] = df_features[premium_services].sum(axis=1)

print(f"   Created: TotalServices ({df_features['TotalServices'].mean():.1f} avg)")
print(f"   Created: PremiumServices ({df_features['PremiumServices'].mean():.1f} avg)")

# 3. Behavioral Features
print("\n3️⃣ Behavioral Features:")

# Tenure groups
def categorize_tenure(months):
    if months == 0:
        return 'New'
    elif months <= 12:
        return 'Short-term'
    elif months <= 36:
        return 'Mid-term'
    else:
        return 'Long-term'

df_features['TenureGroup'] = df_features['tenure'].apply(categorize_tenure)

# Monthly charge groups
df_features['ChargeGroup'] = pd.qcut(df_features['MonthlyCharges'], 
                                     q=4, 
                                     labels=['Low', 'Medium-Low', 'Medium-High', 'High'])

# Contract risk score (higher for month-to-month)
contract_risk = {'Month-to-month': 3, 'One year': 2, 'Two year': 1}
df_features['ContractRisk'] = df_features['Contract'].map(contract_risk)

# Payment risk score
payment_risk = {'Electronic check': 3, 'Mailed check': 2, 
                'Bank transfer (automatic)': 1, 'Credit card (automatic)': 1}
df_features['PaymentRisk'] = df_features['PaymentMethod'].map(payment_risk)

print(f"   Created: TenureGroup, ChargeGroup, ContractRisk, PaymentRisk")

# 4. Interaction Features
print("\n4️⃣ Interaction Features:")

# Service charge efficiency
df_features['ChargePerService']=df_features['MonthlyCharges']/(df_features['TotalServices']+1)

# Tenure to charge ratio
df_features['TenureChargeRatio']=df_features['tenure']/(df_features['MonthlyCharges']+1)

# Senior citizen with dependents
df_features['SeniorwithDependents']=((df_features['SeniorCitizen']==1) &
                                     (df_features['Dependents']=='Yes')).astype(int)

print(f"   Created: ChargePerService, TenureChargeRatio, SeniorWithDependents")

# 5. Risk Score (Composite Feature)
print("\n5️⃣ Composite Risk Score:")

df_features['RiskScore']=(
    df_features['ContractRisk']*0.3+
    df_features['PaymentRisk']*0.2+
    (df_features['MonthlyCharges']>df_features['MonthlyCharges'].median()).astype(int)*0.2+
    (df_features['tenure']<12).astype(int)*0.3
)

print(f"   Created: RiskScore (0-1 scale)")
print(f"   RiskScore stats: Mean={df_features['RiskScore'].mean():.2f}, "
      f"Std={df_features['RiskScore'].std():.2f}")


print(f'Total number New Features Created: {(len(df_features.columns)-len(df.columns))}')
print(f'The Shape of new Feature are:{df_features.shape}')
print("\n" + "=" * 60)
print("STEP 4: VISUALIZING NEW FEATURES")
print("=" * 60)

# Select some of the new features to visualize
new_features = ['TotalServices', 'PremiumServices', 'ContractRisk', 
                'PaymentRisk', 'RiskScore', 'AvgChargePerMonth']

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, feature in enumerate(new_features[:6]):
    ax = axes[idx]
    
    if feature in df_features.columns:
        # Create subplot based on feature type
        if df_features[feature].nunique() <= 10:  # Categorical-like
            # Plot distribution
            value_counts = df_features[feature].value_counts().sort_index()
            bars = ax.bar(range(len(value_counts)), value_counts.values,
                         color=plt.cm.Set2(range(len(value_counts))))
            ax.set_xticks(range(len(value_counts)))
            ax.set_xticklabels([str(x) for x in value_counts.index], rotation=45)
            ax.set_ylabel('Count')
            
        else:  # Continuous
            # Plot histogram
            ax.hist(df_features[feature], bins=30, alpha=0.7, 
                   color='skyblue', edgecolor='black')
            ax.axvline(df_features[feature].mean(), color='red', 
                      linestyle='--', linewidth=2, label='Mean')
            ax.axvline(df_features[feature].median(), color='green', 
                      linestyle='--', linewidth=2, label='Median')
            ax.legend()
            ax.set_ylabel('Frequency')
        
        ax.set_title(feature, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel(feature)

plt.suptitle('Distribution of New Engineered Features', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('new_features_distribution.png', dpi=100, bbox_inches='tight')
plt.show()


df_features['Churn_numeric']=df_features['Churn'].map({'Yes':1,'No':0})

new_feature_corr={}
for feature in new_features:
    if feature in df_features.columns:
        corr = df_features[[feature, 'Churn_numeric']].corr().iloc[0, 1]
        new_feature_corr[feature] = corr

# Sort by absolute correlation
sorted_corr = sorted(new_feature_corr.items(), key=lambda x: abs(x[1]), reverse=True)

print("\n📈 Top correlations with churn:")
for feature, corr in sorted_corr:
    direction = "positive" if corr > 0 else "negative"
    print(f"   {feature:20s}: {corr:7.3f} ({direction} correlation)")

    print("\n" + "=" * 60)
print("STEP 5: ENCODING CATEGORICAL VARIABLES")
print("=" * 60)

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
print("\n🔧 Preparing categorical variables for machine learning...")

df_encoded=df_features.copy()

label_encoder=LabelEncoder()

binary_categorical=[]
for col in categorical_cols:
    if df_encoded[col].nunique()==2:
        binary_categorical.append(col)
print(f'found: {(len(binary_categorical))} binary_categorical')

for col in binary_categorical:
    orginal_vals=df_encoded[col].unique()
    df_encoded['f{col}-encoded']=label_encoder.fit_transform(df_encoded[col])
    print(f'---{col}:{orginal_vals}->[0,1]')

print('\nOne_Hot Encoding(for Multi-Class variable)')
multi_cetegorical=[col for col in categorical_cols if col not in binary_categorical]


print(f'Found {(len(multi_cetegorical))} Multi Class Variables')
for col in categorical_cols[:5]:
    unique_vals=df_encoded[col].unique()
    print(f'--- {col}:{len(unique_vals)}unique vals')
    if len(unique_vals)<=5:
        print(f'values{unique_vals}')

if len(multi_cetegorical)>5:
    print(f'...and {len(multi_cetegorical)-5}more')

print("\n3️⃣ Manual Encoding (for ordinal variables):")

# Contract: Two year < One year < Month-to-month (reverse for risk)
internet_mapping={'No':0,'DSL':1,'Fiber optic':2}
df_encoded['InternetService_encoded']=df_encoded['InternetService'].map(internet_mapping)

# Contract: Two year < One year < Month-to-month (reverse for risk)
contract_mapping={'Two year':0,'One year':1,'Month-to-Month':2}
df_encoded['Contract_encoded']=df_encoded['Contract'].map(contract_mapping)
print(f" Created encoded versions for InternetService and Contract")


# 4. Target variable encoding
print("\n4️⃣ Target Variable Encoding:")

df_encoded['Churn_encoded']=label_encoder.fit_transform(df_encoded['Churn'])
print(f' Churn:{df_encoded['Churn'].unique()} ->{df_encoded['Churn_encoded'].unique()}')

print(f"\n✅ Encoding complete!")
print(f"📊 Encoded data shape: {df_encoded.shape}")
print(f"📋 Total columns after encoding: {len(df_encoded.columns)}")
print("\n" + "=" * 60)


print("\n" + "=" * 60)
print("STEP 6: FEATURE SELECTION ANALYSIS")
print("=" * 60)

# Calculate feature importance using simple methods
print("\n🔍 Analyzing feature importance...")

# 1. Correlation with target
print("\n1️⃣ Correlation with Churn:")

# Select numeric columns (excluding ID and target)
numeric_cols = df_encoded.select_dtypes(include=[np.number]).columns.tolist()
# Remove non-feature columns
exclude_cols = ['customerID', 'Churn_numeric', 'Churn_encoded']
numeric_features = [col for col in numeric_cols if col not in exclude_cols]

# Calculate correlation
correlations = []
for col in numeric_features:
    if col in df_encoded.columns:
        corr = df_encoded[[col, 'Churn_encoded']].corr().iloc[0, 1]
        correlations.append((col, corr))

# Sort by absolute correlation
correlations.sort(key=lambda x: abs(x[1]), reverse=True)

print(f"\n📈 Top 10 features by correlation with churn:")
for i, (col, corr) in enumerate(correlations[:10], 1):
    direction = "+" if corr > 0 else "-"
    print(f"   {i:2d}. {col:30s}: {corr:7.3f} ({direction})")

# 2. Visualize top correlations
print("\n📊 Visualizing top correlated features...")

top_features = [col for col, _ in correlations[:6]]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, feature in enumerate(top_features[:6]):
    ax = axes[idx]
    
    # Create boxplot by churn status
    data_to_plot = [
        df_encoded[df_encoded['Churn_encoded'] == 0][feature],
        df_encoded[df_encoded['Churn_encoded'] == 1][feature]
    ]
    
    box = ax.boxplot(data_to_plot, patch_artist=True, labels=['No Churn', 'Churn'])
    
    # Color boxes
    colors = ['#2ecc71', '#e74c3c']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_title(f'{feature}\nCorr: {correlations[idx][1]:.3f}', 
                 fontsize=11, fontweight='bold')
    ax.set_ylabel('Value')
    ax.grid(True, alpha=0.3)

plt.suptitle('Top Features by Correlation with Churn', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('top_features_correlation.png', dpi=100, bbox_inches='tight')
plt.show()


#3.Feature Variance analysis
feature_variance={}
for col in numeric_features:
    if col in df_encoded.columns:
        variance=df_encoded[col].var()
        feature_variance[col]=variance


sorted_variance=sorted(feature_variance.items(),key=lambda x: x[1],reverse=True)

print(f'\n top 5 Variance Features')
for i,(corr,var) in enumerate(sorted_variance[:5],1):
    print(f'{i:2d}.{col:30s}:{var:10.2f}')

print(f'\n Top 5 Lower Varience features')

for i,(corr,var)in enumerate(sorted_variance[-5:],1):
    print(f'{i:2d}.{col:30s}:{var:10.2f}')

print("\n" + "=" * 60)
print("STEP 7: PREPARING FOR MACHINE LEARNING")
print("=" * 60)

print("\n🔧 Creating final feature set and target variable...")

# Define our feature set
# We'll select a combination of original and engineered features
selected_features = [
    # Original features
    'tenure',
    'MonthlyCharges',
    'TotalCharges',
    'SeniorCitizen',
    
    # Engineered features
    'TotalServices',
    'PremiumServices',
    'ContractRisk',
    'PaymentRisk',
    'RiskScore',
    
    # Encoded features
    'gender_encoded',
    'Partner_encoded',
    'Dependents_encoded',
    'PhoneService_encoded',
    'PaperlessBilling_encoded',
    'InternetService_encoded',
    'Contract_encoded'
]

# Check which features we have
available_features = [col for col in selected_features if col in df_encoded.columns]
print(f"📋 Selected {len(available_features)} features for modeling:")

# Display features by category
print("\n📊 Feature Categories:")
print("   Demographic: SeniorCitizen, gender_encoded, Partner_encoded, Dependents_encoded")
print("   Usage: tenure, MonthlyCharges, TotalCharges")
print("   Services: TotalServices, PremiumServices, PhoneService_encoded")
print("   Contract: Contract_encoded, PaperlessBilling_encoded")
print("   Risk: ContractRisk, PaymentRisk, RiskScore")
print("   Internet: InternetService_encoded")

# Create X (features) and y (target)
X = df_encoded[available_features]
y = df_encoded['Churn_encoded']

print(f"\n✅ Feature matrix shape: {X.shape}")
print(f"✅ Target vector shape: {y.shape}")

# Check class balance
print(f"\n🎯 Target distribution:")
class_counts = y.value_counts()
class_percentages = y.value_counts(normalize=True) * 100
for cls in sorted(class_counts.index):
    print(f"   Class {cls}: {class_counts[cls]:,} samples ({class_percentages[cls]:.1f}%)")

# Save the prepared data
print("\n💾 Saving prepared data...")
prepared_data = {
    'X': X,
    'y': y,
    'feature_names': available_features,
    'df_encoded': df_encoded
}

import joblib
joblib.dump(prepared_data, 'prepared_data.pkl')
print("✅ Prepared data saved as 'prepared_data.pkl'")

# Also save as CSV for inspection
df_encoded[available_features + ['Churn_encoded']].to_csv('ml_ready_data.csv', index=False)
print("✅ ML-ready data saved as 'ml_ready_data.csv'")


