## Description
# This python script include functions for plotting time series, bar charts and histogram


# import packages
import os
import pandas as pd
import seaborn as sns
import numpy as np
import datetime
from dateutil.parser import parser
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


## Inputs
FIGSIZE_RANGE = (20,8)
MAX_DISTINCT_VAL = 10 #max distinct feature values


## Plot missing values by month with missing value % on secondary axis 
def plot_null_by_month(feature_list, df, rotate_x_axis, pdfsave):
	
	for fc in feature_list:
		
		# selected feature name
		sel_df = df[df['feature name'] == fc]
		
		# data type
		data_type = sel_df['type'].iloc[0]
		
		# title 
		title = fc + ': Total missing values by month'
		
		# plot the barchart and line graph
		plot1 = plot_bar_chart_line_plot('', 'monthyear', 'total missing value', 'missing value (%)', 100, title, rotate_x_axis, sel_df, sel_df)
		
		# save figure to pdf 
		pdfsave.savefig(plot1, bbox_inches='tight')
		

## plot target feature missing values by month with missing values % on secondary axis
def plot_target_feature_null_by_month(target_df, target_feature, rotate_x_axis, pdfsave):
	
	# plot the barchart and line graph for assessment decision
	title = target_feature + ': Total missing values by month'
	plot1 = plot_bar_chart_line_plot('', 'monthyear', 'total missing value', 'missing value (%)', 100, title, rotate_x_axis, target_df, target_df)
	
	# save figure to pdf 
	pdfsave.savefig(plot1, bbox_inches='tight')
	

## plot histogram of count by featur value and target proportion %
def plot_count_by_feature_value(feature_list, df, df_proportion, proportion_title, rotate_x_axis, pdfsave):
	
	for fc in feature_list:
		
		# selected feature name
		sel_df = df[df['feature name'] == fc].reset_index(drop=True)
		sel_df2 = df_proportion[df_proportion['feature name'] == fc].reset_index(drop=True)
		
		# data type
		data_type = sel_df['type'].iloc[0]
		
		#merge data by feature name and value 
		# for continuous data, merge by index
		if data_type == 'Continuous':
			sel_merge = pd.merge(sel_df, sel_df2[[proportion_title]], left_index=True, right_index=True, how='outer')
		else:
			sel_merge = pd.merge(sel_df, sel_df2[['feature name', 'feature value', proportion_title]], how='outer', on=['feature name', 'feature value'])
			
		sel_merge[proportion_title] = sel_merge[proportion_title].fillna(0)
		
		# sort na to last
		sel_merge['feature value'] = sel_merge['feature value'].replace('NA', np.nan)
		sel_merge = sel_merge.sort_values('feature value', na_position='last')
		sel_merge['feature value'] = sel_merge['feature value'].fillna('NA')
		
		# if total distinct values > max distinct values, only plot the highest target proportion 
		if sel_merge['feature value'].nunique() >= MAX_DISTINCT_VAL:
			sel_merge = sel_merge.nlargest(n=MAX_DISTINCT_VAL, columns=[proportion_title])
			title = fc + ': Barchart of count by feature value with Top ' + str(MAX_DISTINCT_VAL) + ' highest ' + proportion_title
		else:
			title = fc + ': Barchart of count by feature value with ' + proportion_title
			
		# plot 
		plot1 = plot_bar_chart_line_plot(data_type, 'feature value', 'count', proportion_title, 0, title, rotate_x_axis, sel_merge, sel_merge)
		
		# save figure to pdf
		pdfsave.savefig(plot1, bbox_inches='tight')
		
		
## Plot time series count of feature values by month and year
def plot_time_series_feature_values(feature_list, df, pdfsave):
	
	for fc in feature_list:
		
		# feature name 
		sel_df = df[df['feature name'] == fc]
		
		# if total distinct values > max distinct value, only plot the top highest count 
		if sel_df['feature value'].nunique() >= MAX_DISTINCT_VAL:
			_list = sel_df['feature value'].value_counts()[:MAX_DISTINCT_VAL].index.tolist()
			sel_df = sel_df[sel_df['feature value'].isin(_list)]
			title = fc + ': Top ' + str(MAX_DISTINCT_VAL) + ' highest count of features by month and year'
		else:
			title = fc + ': Count features  by month and year'
			
		# plot the line graph
		plot1 = plot_time_series('monthyear', 'count', 'feature value', title, sel_df)
		
		# save figure to pdf 
		pdfsave.savefig(plot1, bbox_inches='tight')
		

## Plot time series of median by month and year
def plot_time_series_median(feature_list, df, pdfsave):
	
	for fc in feature_list:
	
		# feature name 
		sel_df = df[df['feature name'] == fc]
		
		# title
		title = fc + ': Median value by month and year'
		
		# plot the line graph 
		plot1 = plot_time_series('monthyear', 'median', '', title, sel_df)
		
		# save figure to pdf 
		pdfsave.savefig(plot1, bbox_inches='tight')
		
		
## Plot bar charts 
def plot_bar_chart(x_axis, y_axis, title, rotate_x_axis, df):
	
	plt.figure(figsize = FIGSIZE_RANGE)
	ax = sns.barplot(x = x_axis, y = y_axis, data=df)
	
	# rotate x axis 
	if rotate_x_axis == 'Y':
		ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
		
	# set title
	ax.set_title(title)
	
	plt.show
	
	
## Plot bar chart and line plot 
def plot_bar_chart_line_plot(data_type, x_axis, y_axis, y_axis2, yrange, title, rotate_x_axis, df, df2):
	
	plt.figure(figsize = FIGSIZE_RANGE)
	ax = sns.barplot(x=x_axis, y=y_axis, color="#FFA500", data=df)
	
	# rotate x axis
	if rotate_x_axis == 'Y':
		ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
	
	# plot secondary line plot 
	ax2 = ax.twinx()
	sns.pointplot(x=x_axis, y=y_axis2, data=df2, markers='o', errwidth=3, ax=ax2)
	
	# set secondary y axis limit
	if yrange != 0:
		ax2.set_ylim([0, yrange])
		
	# set title 
	ax.set_title(title)
	
	# set x axis 
	if data_type == 'Continuous':
		ax.set(xlabel = 'range value')
	else:
		ax.set(xlabel = x_axis)
	
	plt.show 
	
## Plot time series 
def plot_time_series(x_axis, y_axis, legend_feature, title, df):
	
	plt.figure(figsize = FIGSIZE_RANGE)
	
	if legend_feature != '':
		ax = sns.pointplot(data=df, x=x_axis, y=y_axis, hue=legend_feature)
	else:
		ax = sns.pointplot(data=df, x=x_axis, y=y_axis)
		
	# set title and label
	ax.set_title(title)
	ax.set_ylabel(y_axis)
	
	plt.show 
