## Description
# This python script include functions for calculating the feature value of each feature by month, by proportion %


# import packages 
# import packages
import os
import pandas as pd
import seaborn as sns
import numpy as np
import datetime
from dateutil.parser import parser


## Summary statistics table 
def get_summary_statistics(data_type, feature_list, df):
	
	ds = df[feature_list].describe(include = 'all')
	ds = ds.T.reset_index()
	ds = ds.rename(columns = {'index' : 'feature name'})
	ds.insert(loc=0, column='type', value=data_type)
	
	# shape of the data 
	tot_row, tot_col = df.shape 
	
	# missing values 
	ds['missing values'] = tot_row - ds['count']
	ds['missing values'] = ds['missing values'].astype('int64')
	
	return ds 
	
## Get feature value count for categorical variable
def get_feature_value_count_categorical(feature_list, df, target_feature):
	
	# initialise
	cat_feature_df = pd.DataFrame(columns = ['feature value', 'monthyear', target_feature, 'count', 'feature name'])
	
	if feature_list:
		
		# for each feature, get value count
		for fc in feature_list:
			
			# count based on feature value, monthyear
			count_df = df.groupby([fc, 'monthyear', target_feature])[fc].count()
			null_df = df.groupby(['monthyear', target_feature])[fc].apply(lambda _df: _df.isna().sum())
			
			# set as data frame 
			count_df = count_df.to_frame(name='count').reset_index()
			null_df = null_df.to_frame(name='count').reset_index()
			
			# rename feature column 
			count_df = count_df.rename(columns={fc: 'feature value'})
			null_df['feature value'] = 'NA'
			
			# include constant feature name 
			count_df['feature name'] = fc
			null_df['feature name'] = fc 
			
			# join the feature count with the null count 
			full_df = pd.concat([count_df, null_df], ignore_index=True)
			
			# append to master table 
			cat_feature_df = pd.concat([cat_feature_df, full_df], ignore_index=True)
		
		# add data type 
		cat_feature_df['type'] = 'Categorical'
		
		return cat_feature_df[['type', 'feature name', 'feature value', target_feature, 'monthyear', 'count']]
		
		
## Get feature value count for continuous variable 
def get_feature_value_count_continuous(feature_list, df, target_feature):

	# initialise
	con_feature_df = pd.DataFrame(columns = ['feature value', 'monthyear', target_feature, 'count', 'feature name'])
	
	if feature_list:
		
		# for each feature, get value count
		for fc in feature_list:
		
			# count total unique values 
			tot_unique = df[fc].nunique()
			
			# if tot unique > 10, divide 10 equal sized group of data 
			if tot_unique > 10:
				nbins = 10
			else:
				nbins = tot_unique 
				
			# replace non-numeric value with NaN
			df[fc] = df[df.applymap(isnumber)][fc]
			
			# create x equal size of groupings of the data 
			if nbins < 10:
				df['feature value'] = df[fc].astype(str)
				df['feature value'] = df['feature value'].replace('nan', 'NA')
			else:
				df['feature value'] = pd.cut(df[fc], bins=nbins, precision=0)
			
			# count based on feature value, monthyear
			count_df = df.groupby(['feature value', 'monthyear', target_feature])[fc].count()
			null_df = df.groupby(['monthyear', target_feature])[fc].apply(lambda _df: _df.isna().sum())
			
			# set as data frame 
			count_df = count_df.to_frame(name='count').reset_index()
			null_df = null_df.to_frame(name='count').reset_index()
			
			null_df['feature value'] = 'NA'
			
			# include constant feature name 
			count_df['feature name'] = fc
			null_df['feature name'] = fc 
			
			# join the feature count with the null count 
			full_df = pd.concat([count_df, null_df], ignore_index=True)
			
			# append to master table 
			con_feature_df = pd.concat([con_feature_df, full_df], ignore_index=True)
		
		# add data type 
		con_feature_df['type'] = 'Continuous'
		
		return con_feature_df[['type', 'feature name', 'feature value', target_feature, 'monthyear', 'count']]
		
## Get median value from each continuous feature
def get_median_value_continuous_feature(feature_list, df):

	# initialise
	con_feature_median_df = pd.DataFrame(columns = ['feature value', 'monthyear', 'median'])
	
	if feature_list:
		
		# for each feature, get value count
		for fc in feature_list:
		
			# replace non-numeric value with NaN
			df[fc] = df[df.applymap(isnumber)][fc]
			
			# calculate the median by month and year
			median_df = df.groupby(['monthyear'])[fc].median()
			
			# set as data frame 
			median_df = median_df.to_frame(name='median').reset_index()
			
			# include constant feature name 
			median_df['feature name'] = fc 
			
			# append to the master table 
			con_feature_median_df = pd.concat([con_feature_median_df, median_df], ignore_index=True)
			
	# replace null as 0
	con_feature_median_df['median'] = con_feature_median_df['median'].fillna(0)
	
	return con_feature_median_df[['feature name', 'monthyear', 'median']]
	
	
## Count by feature value and second/third feature 
def count_by_feature_value(df, second_feat, third_feat):
	
	df_feature_count = []
	
	# count by 1 feature 
	if len(second_feat) == 0 and len(third_feat) == 0:
		df_feature_count = df.groupby(['type', 'feature name', 'feature value'])['count'].sum().reset_index()
		
	# count with second feature 
	elif len(second_feat) > 0 and len(third_feat) == 0:
		df_feature_count = df.groupby(['type', 'feature name', 'feature value', second_feat])['count'].sum().reset_index()
		
	# count with third feature 
	elif len(second_feat) == 0 and len(third_feat) > 0:
		df_feature_count = df.groupby(['type', 'feature name', 'feature value', third_feat])['count'].sum().reset_index()
	
	# count with second and third feat 
	else:
		df_feature_count = df.groupby(['type', 'feature name', 'feature value', second_feat, third_feat])['count'].sum().reset_index()
	
	return df_feature_count
	

## Count target feature by month and year
def count_target_by_month(df, target_feature):
	
	target_month_count = df.groupby([target_feature, 'monthyear'])[target_feature].count()
	target_month_count = target_month_count.to_frame(name='count').reset_index()
	
	# rename 
	target_month_count = target_month_count.rename(columns={target_feature: 'feature value'})
	target_month_count['feature name'] = target_feature
	target_month_count['type'] = 'Categorical'
	
	return target_month_count
	
	
## Count by feature value and target feature, with proportion by target feature 
def count_by_feature_value_target_feature(df, target_feature):

	df_feature_target_count = []
	df_feature_target_count = df.groupby(['type', 'feature name', 'feature value', target_feature])['count'].sum().reset_index()
	
	# get total count by feature name and value, exclude NA
	total_count_tf = df_feature_target_count.groupby(['type', 'feature name', 'feature value']).apply(lambda dft: pd.Series({'total': dft['count'].sum()})).reset_index()
	
	# join the total with feature count 
	df_feature_target_count = df_feature_target_count.merge(total_count_tf, how='left', left_on=['type', 'feature name', 'feature value'], right_on=['type', 'feature name', 'feature value'], indicator=True)
	
	# calculate proportion 
	df_feature_target_count['total'] = df_feature_target_count['total'].replace(0, np.nan)
	df_feature_target_count['proportion (%)'] = df_feature_target_count['count'].div(df_feature_target_count['total'])*100
	
	return df_feature_target_count
	
## Count missing values by feature name and month year
def count_by_month_missing_values(df, missing_value):
	
	total_count_month = []
	
	# get total count by feature name and month
	total_count_month = df.groupby(['type', 'feature name', 'monthyear'])['count'].sum().reset_index()
	total_count_month.rename(columns={'count':'total by monthyear'}, inplace=True)
	
	# get total count of missing values by feature name and month
	df_month_nulls_count = df[df['feature value'] == missing_value].reset_index(drop=True)
	df_month_nulls_count.rename(columns={'count': 'total missing value'}, inplace=True)
	
	# join the total with the missing value count 
	df_month_nulls_count = df_month_nulls_count.merge(total_count_month, how='outer', left_on=['type', 'feature name', 'monthyear'], right_on=['type', 'feature name', 'monthyear'], indicator=True)
	
	# calculate the proportion
	#df_month_nulls_count['missing value (%)'] = df_month_nulls_count['total missing value'].div(df_month_nulls_count['total by monthyear']).replace(np.inf, 0)*100
	df_month_nulls_count['missing value (%)'] = df_month_nulls_count['total missing value'].div(df_month_nulls_count['total by monthyear'].where(df_month_nulls_count['total by monthyear'] != 0, np.nan)) * 100
	
	# replace nan to 0
	df_month_nulls_count['feature value'] = df_month_nulls_count['feature value'].fillna(missing_value)
	df_month_nulls_count['missing value (%)'] = df_month_nulls_count['missing value (%)'].fillna(0)
	
	return df_month_nulls_count
	

## Get total outliers 
def find_outliers_IQR(df):

	q1 = df.quantile(0.25)
	q3 = df.quantile(0.75)
	IQR = q3-q1
	outliers = df[((df<(q1-1.5*IQR)) | (df>(q3+1.5*IQR)))]
	
	return len(outliers)
	
## Show number if numeric
def isnumber(x):
	try:
		float(x)
		return True
	except:
		return False 
	