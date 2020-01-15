# Data Preprocessing Techniques
This repository contains a tabular dataset on which I applied some standard preprocessing techniques. The result was a final, cleansed dataset that can be fed to different predictive algorithms.

###### Dataset: 
(n_samples, n_features) == (45, 13)  

###### Libraries used:  
--> Scikit-Learn  
--> Pandas  
--> Numpy  

###### Problems with the original dataset:  
--> Unwanted column of data  
--> Missing/NaN values  
--> Irregular range of values for all features  
--> High dimensional data  
--> Possibly unwanted features  

###### Solution in chronological order:  
--> Removed column through standard Pandas .drop() function  
--> Filled Missing NaN values through Sklearn's SimpleImputer()  
--> Normalized data through Sklearn's MinMaxScaler()  
	-Used feature_range between 0 to 1 for columns with positive values only  
	-Used feature_range between -1 to 1 for columns with both positive and negative values  
--> For Feature Engineering:  
	- Applied Principle Component Analysis (PCA) to get 3 best n_components  
	- Applied unwanted feature elimination technique Recursive Feature Elimination (RFE)  
	- Could not apply SelectKBest Features as it cannot be applied to data with negative values  

Most of the code is taken from Machine Learning Mastery with Python by Jason Brownlee and some from towardsdatascience.com.
