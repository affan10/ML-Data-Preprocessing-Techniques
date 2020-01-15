import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

dataframe = pd.read_csv('biomech_accel_labels.csv')

# Dropping first column
dataframe = dataframe.drop(dataframe.columns[0], axis=1)

# Imputing NaN values with the mean of column
fill_NaN   = SimpleImputer(missing_values=np.nan, strategy='mean')
imputed_DF = pd.DataFrame(fill_NaN.fit_transform(dataframe), columns = dataframe.columns)
export_csv = imputed_DF.to_csv('biomech_accel_labels_imputed.csv', index=False)

dataframe = pd.read_csv('biomech_accel_labels_imputed.csv')
array     = dataframe.values

# Separate the dataframe into input and output components
X = array[:,0:11]
Y = array[:,12]

# Applying normalization to all columns separately with appropriate feature_range
# NOTE: Did not apply normalization to the 'movement' column as its values were already between 0 to 1
scalerPositive                   = MinMaxScaler(feature_range=(0, 1))
scalerNegative                   = MinMaxScaler(feature_range=(-1, 1))
scalerFullNegative               = MinMaxScaler(feature_range=(-1, 1))
imputed_DF['rotation']           = scalerNegative.fit_transform(imputed_DF[['rotation']])
imputed_DF['aperture']           = scalerPositive.fit_transform(imputed_DF[['aperture']])
imputed_DF['supination']         = scalerPositive.fit_transform(imputed_DF[['supination']])
imputed_DF['smoothness']         = scalerFullNegative.fit_transform(imputed_DF[['smoothness']])
imputed_DF['thumb extension']    = scalerNegative.fit_transform(imputed_DF[['thumb extension']])
imputed_DF['thumb abduction']    = scalerPositive.fit_transform(imputed_DF[['thumb abduction']])
imputed_DF['wrist extension']    = scalerPositive.fit_transform(imputed_DF[['wrist extension']])
imputed_DF['elbow extension']    = scalerPositive.fit_transform(imputed_DF[['elbow extension']])
imputed_DF['finger extension']   = scalerPositive.fit_transform(imputed_DF[['finger extension']])
imputed_DF['external rotation']  = scalerNegative.fit_transform(imputed_DF[['external rotation']])
imputed_DF['shoulder elevation'] = scalerPositive.fit_transform(imputed_DF[['shoulder elevation']])

export_csv = imputed_DF.to_csv('biomech_accel_labels_imputed_rescaled.csv', index=False)

array = imputed_DF.values
X     = array[:,0:11]
Y     = array[:,12]

# Applying PCA to get Explained Variance Ratio (EVR)
# Variance Ratio is the ratio between the variance of the principal component and the total variance
# The higher the EVR, the more important that component is for better prediction
pca = PCA(n_components=3)
fit = pca.fit(X)
print('Explained Variance: %s' % fit.explained_variance_ratio_)
print(fit.components_)

# Applying Recursive Feature Elimination (RFE) with the model as Logistic Regression
# The choice is model can be any standard model like SVC, Random Forest etc.
model = LogisticRegression()

# The parameter 3 indicates that we want the 3 best features
rfe = RFE(model, 3)
fit = rfe.fit(X, Y)
print("Num Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)