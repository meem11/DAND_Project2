#!/usr/bin/env python
# coding: utf-8

# # Project: Why Do People Miss their Appointments?!
# # By: Abdullah Alghamdi
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# 
# This dataset collects information from 100k medical appointments in Brazil and is focused on the question of whether or not patients show up for their appointment. A number of characteristics about the patient are included in each row.
# 
# <ul>
# <li>'PatientId' is a unique number of each patient</li>
# <li>'AppointmentID' is a unique number of each appointment</li>
# <li>'Gender' M for male, and F for female</li>    
# <li>‘ScheduledDay’ tells us on what day the patient set up their appointment.</li>
# <li>‘AppointmentDay’ tells us on what day is the appointment.</li>
# <li>'Age' is age of each patient</li>
# <li>‘Neighborhood’ indicates the location of the hospital</li>
# <li>‘Scholarship’ indicates whether or not the patient is enrolled in Brasilian welfare program Bolsa Família.</li>
# <li>'Hipertension' indicated true of false of having high blood pressure</li>    
# <li>'Diabetes' indicated true of false of having diabetes</li>    
# <li>'Alcoholism' indicated true of false of having alcoholism</li>    
# <li>'Handcap' indicated true of false of being disabled</li>    
# <li>'SMS_received' indicated true of false of receiving a reminder SMS</li>    
# <li>'No-show' with "yes" for not showing up and "No" if patient showed up</li>    
# </ul>

# In[87]:


# import packages 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set_style('white')

#the below line is to disable false positive error messages
pd.options.mode.chained_assignment = None  


# <a id='wrangling'></a>
# ## Data Wrangling
# 
# ### General Properties

# In[88]:


df = pd.read_csv('data.csv')
df.head()


# ## Data Cleaning

# In[89]:


# Start by looking for any missing values
df.info()


# ###### As can be seen above, no missing values in the provided data as all columns have 110,527 rows.

# ### Columns Renaming:
# The columns were renamed to follow conventional naming of lower case and underscore between the words

# In[90]:


#columns titles will be renamed
df.columns = df.columns.str.lower()
df.columns = ['patient_id', 'appointment_id', 'gender', 'scheduled_day',
       'appointment_day', 'age', 'neighbourhood', 'scholarship', 'hypertension',
       'diabetes', 'alcoholism', 'handicap', 'sms_received', 'no_show']


# ### Checking for duplicates:
# No duplicates were found as shown below.

# In[91]:


df.duplicated().sum()
# No Duplicated Data


# ### Data Types correcting:
# Changed data type of two date columns to be datetime data type instead of object.
# 

# In[92]:


df_clean = df
df_clean['scheduled_day'] = pd.to_datetime(df['scheduled_day'], infer_datetime_format = True)
df_clean['appointment_day'] = pd.to_datetime(df['appointment_day'], infer_datetime_format = True)
df_clean['scheduled_day'] = pd.DatetimeIndex(df_clean.scheduled_day).normalize()

df_clean.head()


# ### Data Validatation:
# Age column has one invalid datapoint which show negative value. This row was excluded from the dataset.

# In[93]:


#Age has invalid values "< 0", which will be dropped
df_clean = df_clean[df_clean['age'] >= 0]
df_clean.shape


# ### Fixing No Show column:
# No show column consist of two values, "Yes" indicates absence and "No" which means attendance. This can be confusing. Also using numbers (1,0) will be better in mathematical operations. 

# In[94]:


#start by renaming no_show column to showed
df_clean.rename(columns={'no_show':'showed'}, inplace=True)
df_clean.head()


# In[95]:


df_clean.showed[df_clean['showed']=='Yes'] = 0
df_clean.showed[df_clean['showed']=='No'] = 1

df_clean['showed'].value_counts()


# <a id='eda'></a>
# ## Exploratory Data Analysis
# 
# We start with plotting histograms of all of the columns to explore distribution of data. below are some comments on some attributes.
# ### Age:

# In[96]:


plt.hist(df_clean.age[df_clean['showed'] == 1], alpha=.5, label='showed')
plt.hist(df_clean.age[df_clean['showed'] == 0], alpha=1, label='not showed')
plt.legend(loc='upper right')
plt.show()


# most of the dataset population are on the younger side of the spectrum. This might be because of frequent checkups and vaccinations of children aged 10 and below.

# ### Gender:

# In[97]:


plt.hist(df_clean.gender[df_clean['showed'] == 1], alpha=.5, label='showed')
plt.hist(df_clean.gender[df_clean['showed'] == 0], alpha=1, label='not showed')
plt.legend(loc='upper right')
plt.show()


# Females are more common in this dataset, which could be caused by frequent checkups for females during pregnancy. However, the relationship between showing up to appointments and gender will be explored in the next steps.

# ### Other attributes:
# 
# distribution of Several health issues can be seen below. Hypertension is the most common one of them.

# In[98]:


df_clean.hist(figsize = (15,15));


# In[99]:


#checking number of people showing up to appointments
df_clean['showed'].value_counts()


# about 82% of patient showed up for their appointment

# ### Research Question 1: Do Male Miss Appointments more than Women

# ##### Start by counting showing up by gender. Then data was normalized by dividing count of attendance show-up of each gender by the total number of appointment of each gender
# 

# In[100]:


counts_gender = df_clean.groupby(['gender','showed'])['appointment_id'].count()
totals_gender = df_clean.groupby(['gender'])['appointment_id'].count()
prop_gender = counts_gender / totals_gender
prop_gender


# In[101]:


ind = [0,1]  # the x locations for the groups
width = 0.35       # the width of the bars
# plot bars
female_bars = plt.bar(ind, prop_gender['F'], width, color='r', alpha=.7, label='Females')
male_bars = plt.bar([.35,1.35] , prop_gender['M'], width, color='g', alpha=.7, label='Males')

# title and labels
plt.ylabel('Proportion')
plt.xlabel('Appointment Show up')
plt.title('Proportion by Gender and Appointment Show-up ')
locations = ([.35/2,2.35/2]) # xtick locations
labels = ['No', 'Yes']  # xtick labels
plt.xticks(locations, labels)

# legend
plt.legend();


# ###### The graph above shows that males and females had the same number of missing appointments. Thus no relationship between the two variable is found

# ### Research Question 2: Does health status of the patient affect his probability of showing up

# ###### Start by creating a new column in the data frame to define sick persons. people who are diabetic, having high blood pressure, or handicapped were classified as having health issues.

# In[102]:


df_clean['sick'] = 0


# In[103]:


df_clean.sick[df_clean['handicap'] == 1] = 1
df_clean.sick[df_clean['hypertension'] == 1] = 1
df_clean.sick[df_clean['diabetes'] == 1] = 1


# ###### Below, proportion of sick people who showed up and missed the appointment to the sick population were calculated

# In[104]:


counts_sick = df_clean.groupby(['sick','showed'])['appointment_id'].count()
totals_sick = df_clean.groupby(['sick'])['appointment_id'].count()
prop_sick = counts_sick / totals_sick
prop_sick


# In[105]:


ind = [0,1]  # the x locations for the groups
width = 0.35       # the width of the bars
# plot bars
healthy_bars = plt.bar(ind, prop_sick[0], width, color='g', alpha=.7, label='Healthy')
sick_bars = plt.bar([.35,1.35] , prop_sick[1], width, color='r', alpha=.7, label='Sick')

# title and labels
plt.ylabel('Proportion')
plt.xlabel('Appointment Show up')
plt.title('Proportion by Health and Appointment Show-up ')
locations = ([.35/2,2.35/2]) # xtick locations
labels = ['No', 'Yes']  # xtick labels
plt.xticks(locations, labels)

# legend
plt.legend();


# ###### Healthy individuals tend to miss more appointments than sick ones. This could be related to the urgency and the need of car that is more urgent in sick people (diabetic, high blood pressure, and handicapped). The relationship is not significant and might be due to the limitation of the dataset.

# ### Research Question 3: Does Reminding by SMS work?

# In[106]:



plt.hist(df_clean.sms_received[df_clean['showed'] == 1], alpha=.5, label='Showed')
plt.hist(df_clean.sms_received[df_clean['showed'] == 0], alpha=1, label='No Show')
# title and labels
plt.ylabel('Count of Appointments')
plt.xlabel('Received Reminder SMS')
plt.title('Count of people recevied SMS by Appointment Show-up ')
locations = ([.05,.95]) # xtick locations
labels = ['Yes', 'No']  # xtick labels
plt.xticks(locations, labels)

# legend
plt.legend(loc='upper right')
plt.show();


# ###### From the above chart, it can be seen that from the population who received SMS message, about 40% missed the appointment. However, most of the people who did not receive this message, showed up for the appointment. This might be due to the way the system send these SMS messages only as reminder for people with previous missing appointments.

# ### Research Question 4: Does the time period between scheduling and the appointment affect the show up status?

# ###### We start by creating a column that calculates difference between the appointment day and the scheduled day

# In[107]:


df_clean['days_to_appointment'] = df_clean['appointment_day'] - df_clean['scheduled_day']
df_clean['days_to_appointment'] = df_clean['days_to_appointment'].dt.days
df_clean['days_to_appointment_bins'] = 0


# ###### Then we create several bins to show the trend of the relationship between the gap between the two dates and the no show status.

# In[108]:


df_clean.days_to_appointment_bins[df_clean['days_to_appointment'] <= 7] = '1 Week'
df_clean.days_to_appointment_bins[(df_clean['days_to_appointment'] <= 14) & (df_clean['days_to_appointment'] > 7)] = '2 Weeks'
df_clean.days_to_appointment_bins[(df_clean['days_to_appointment'] <= 21) & (df_clean['days_to_appointment'] > 14)] = '3 Weeks'
df_clean.days_to_appointment_bins[(df_clean['days_to_appointment'] <= 28) & (df_clean['days_to_appointment'] > 21)] = '4 Weeks'
df_clean.days_to_appointment_bins[(df_clean['days_to_appointment'] <= 60) & (df_clean['days_to_appointment'] > 28)] = '8 Weeks'
df_clean.days_to_appointment_bins[df_clean['days_to_appointment'] > 60] = '>12 Weeks'


# ###### The below code is to create a graph to show the relationship

# In[109]:



counts_days_diff = df_clean.groupby(['days_to_appointment_bins','showed'])['appointment_id'].count()
totals_days_diff = df_clean.groupby(['days_to_appointment_bins'])['appointment_id'].count()
prop_days_diff = counts_days_diff/totals_days_diff

# plot data
fig, ax = plt.subplots(figsize=(15,7))
# use unstack()
prop_days_diff.unstack().plot(ax=ax, sort_columns=True, kind = 'bar', stacked = True)
h,l = ax.get_legend_handles_labels()
ax.legend(h[:2],["No Show", "Show"], loc='upper right', bbox_to_anchor=(1, 1.05),
          ncol=3)
plt.ylabel('Proportion')
plt.xlabel('Time between appointment and scheduled dates')
plt.title('Proportion of show up status vs time between scheduling and appointment dates',loc='left');


# ###### From the chart above, it can be seen some correlation between time between scheduling and appointment dates with the no show status. The closer the appointment to the scheduling date, the more likely the patient will make it to the appointment.

# <a id='conclusions'></a>
# ## Conclusions
# 
# 
# ### Several findings were observed from the brazilian medical appointments data set, such as:
# 
# 
# 1. It was found that the gender of the patient has no effect on the likelihood of missing appointments.
# 1. A relationship between the health status of patients and the likelihood of showing up for the appointment was found. Patients with health issues are more likely to no miss their appointments. This could be due to the urgency of the care in these situation in contrast to general appointment visits.
# 1. SMS reminder effectivity was investigated,patients who received an SMS reminder were compared the ones who did not receive one. almost half of the people who received a message did not show up to the appointment, were that number was way lower at around 15% at the population who received the message. This might be due to the way the message mechanism worked, for example only send an SMS if the patient had already missed one appointment before. This information is not available, thus only speculations can be made.
# 1. As time period of between the the appointment day and the day the appointment was scheduled, the more likely the appointment will be missed. This could be explained by the tendency of people to forget or get involved with other unforeseen activities which prevent them from meeting the appointment time.
# 
# 
# ### Limitations
# 1. Dataset might be biased to focus on no show side due to the scope and objective of collecting this data.
# 1. More explanation of information can be beneficial on drawing more accurate conclusions.

# In[110]:


#Creating html version of the notebook
from subprocess import call
call(['python', '-m', 'nbconvert', 'Investigate_a_Dataset.ipynb'])

