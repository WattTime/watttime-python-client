#!/usr/bin/env python
# coding: utf-8

# In[58]:


import numpy as np
import pandas as pd
import datetime
import random
import pytz
from datetime import datetime, timedelta
import random


# In[2]:


# utilities and functions
# get rid of this one when you get real data
def generate_normal_distribution_series(mean, variance, observations):
    std_dev = np.sqrt(variance)
    random_draws = np.random.normal(loc=mean, scale=std_dev,size=observations)
    series = pd.Series(random_draws)
    return series

def generate_random_plug_time(date):
    """
    Generate a random datetime ona the given date, uniformly distributed between 5pm and 9 pm.

    Parameters:
    date (datetime.date): The date on which to generate the random time.

    Returns:
    - datetime: A datetime object for the given date with a random time between 5 PM and 9 PM
    """
    #  Define the start and end times for the interval (5 PM to 9PM)
    start_time = datetime.combine(date, datetime.strptime("17:00:00", "%H:%M:%S").time())
    end_time = datetime.combine(date, datetime.strptime("21:00:00", "%H:%M:%S").time())
    
    # Calculate the total number of seconds between start and end times 
    total_seconds = int((end_time - start_time).total_seconds())
    
    # Generate a random number of seconds within the interval
    total_seconds = random.randint(0, total_seconds)

    # Add the random seconds to the start time to get the random datetime
    random_datetime = start_time + timedelta(seconds=total_seconds)

    random_datetime_utc = pytz.utc.localize(random_datetime)

    return random_datetime_utc

def generate_random_unplug_time(random_plug_time,mean,stddev):
    """ 
    Adds a number of sconds drawn from a normal distribution to the given datetime.

    Parameters:
    -datetime_obj
    -mean
    -stddev

    REturns 
    -pd.Timestamp: the new datetime after adding the random seconds
    """
    random_seconds = np.random.normal(loc=mean,scale=stddev)

    #convert to timedelta
    random_timedelta = timedelta(seconds=random_seconds)
    new_datetime = random_plug_time + random_timedelta

    if not isinstance(new_datetime,pd.Timestamp):
        new_datetime = pd.Timestamp(new_datetime)
    return new_datetime


# # Mock data because the server stopped responding

# In[3]:


start = '2023-04-01 00:00:00+00:00'
end = '2023-05-01 00:00:00+00:00'
freq = '5min'

# create some mock data because the server stopped responding
date_range = pd.date_range(start= start, end = end, freq=freq, tz='UTC')
values = generate_normal_distribution_series(1400,61,len(date_range))
df = pd.DataFrame({'point_time':date_range, 'values':values.values})


# # Single user problem

# In[52]:


# characterize a single user
rate= 11 #7.4, 11, or 22 kWh
total_capacity= 118 #ranges from 21kW to 123 kW
mean_length_charge = 36000 
std_length_charge = 7200


# ### Single user/single day problem

# For each user-day, the following dataframe summarizes their relationship with the grid: 
# 1. plug_in_time : random number from a uniform distribution between 5 and 9 pm, when the user plugs in their EV
# 2. unplug_time : random number from a normal distribution with inital mean given by user (example here is 10 hours, stddev is 3 hrs)
# 3. inital_charge : what percentage of battery was full at plug in time
# 4. total seconds to 95% : how long would it take to charge the battery to the desired level
# 5. length of interval : plug_in_time - unplug_time
# 6. final battery level : how charged was the battery in the end
# 7. uncharged: boolean is the battery more than 80% full at unplug time
# 8. MOER: sum of MOER in the grid for that day (number to minimize)

# ### Dataframe implementation: 1 user, all days for which we have data (in the mock dataframe DF)

# In[53]:


distinct_dates = df['point_time'].dt.date.unique()
distinct_dates_utc = [pd.Timestamp(date).replace(tzinfo=pytz.UTC) for date in distinct_dates]
user_df = pd.DataFrame(distinct_dates_utc, columns=['distinct_dates']).sort_values(by='distinct_dates').copy()


# In[54]:


user_df['user_type'] =  'r'+str(rate)+'_tc'+str(total_charge)+'_avglc'+str(mean_length_charge)+'_sdlc'+str(std_length_charge)
user_df['plug_in_time'] = user_df['distinct_dates'].apply(generate_random_plug_time)
user_df['unplug_time'] = user_df['plug_in_time'].apply(lambda x: generate_random_unplug_time(x,mean_length_charge,std_length_charge))
user_df['initial_charge'] = user_df.apply(lambda _: random.uniform(0,0.3), axis=1)
user_df['total_seconds_to_95'] = user_df['initial_charge'].apply(lambda x: total_capacity*(0.95-x)/(rate/3600))

user_df['full_charge_time']= user_df['plug_in_time'] + pd.to_timedelta(user_df['total_seconds_to_95'],unit='s')
user_df['length_plugged_in'] = (user_df.unplug_time - user_df.plug_in_time) / pd.Timedelta(seconds=1)

user_df['session_charge'] = user_df[['total_seconds_to_95','length_plugged_in']].min(axis=1)*(rate/3600)
user_df['final_perc_charged'] = user_df.session_charge.apply(lambda x: x/total_capacity)
user_df['final_perc_charged'] = user_df.final_perc_charged + user_df.initial_charge

user_df['final_charge_time'] = user_df[['full_charge_time', 'unplug_time']].min(axis=1)
user_df['uncharged'] = np.where(user_df['final_perc_charged'] <0.80, True, False)


# In[55]:


moer_temp = user_df[['user_type','distinct_dates','plug_in_time','final_charge_time']].copy()
moer_temp['key'] = 1
df['key'] = 1


# In[56]:


moer_temp = pd.merge(moer_temp,df, on='key').drop(columns='key')
moer_temp = moer_temp[(moer_temp['point_time']>moer_temp['plug_in_time'])&(moer_temp['point_time']<moer_temp['final_charge_time'])]


# In[57]:


moer_temp = moer_temp.groupby(['distinct_dates','user_type']).agg(
    sum_moer=('values','sum'),
    count_moer_intervals=('values','count'),
    avg_moer=('values','mean')
).reset_index()


# In[34]:


user_df= pd.merge(user_df,moer_temp, on=['user_type','distinct_dates'])


# In[35]:


user_df.to_csv('test.csv')


# In[43]:


i = user_df.loc[3]


# In[50]:


f"For user {i['user_type']}, on date {i['distinct_dates']}, the EV was plugged in at {i['plug_in_time']}, and unplugged at {i['unplug_time']}.It had an initial charge of {i['initial_charge']:.1%}. At max rate, full charge time would be {i['full_charge_time']}. This leaves it with a final charge of {i['final_perc_charged']}. The total MOER was {i['sum_moer']:,}"


# # Simulate 1K users to get a distribution

# In[76]:


all_users = pd.DataFrame()
for i in range(1000):
    rate = random.choice([11,7.4,22])
    total_capacity =  round(random.uniform(21,123))
    mean_length_charge = round(random.uniform(20000,30000))
    std_length_charge = round(random.uniform(6800,8000))
    
    print(f"working on user with {total_capacity} total_capacity, {rate} rate of charge, and ({mean_length_charge/3600},{std_length_charge/3600}) charging behavior.") 

    user_df = pd.DataFrame(distinct_dates_utc, columns=['distinct_dates']).sort_values(by='distinct_dates').copy() 

    user_df['user_type'] =  'r'+str(rate)+'_tc'+str(total_charge)+'_avglc'+str(mean_length_charge)+'_sdlc'+str(std_length_charge)
    user_df['plug_in_time'] = user_df['distinct_dates'].apply(generate_random_plug_time)
    user_df['unplug_time'] = user_df['plug_in_time'].apply(lambda x: generate_random_unplug_time(x,mean_length_charge,std_length_charge))
    user_df['initial_charge'] = user_df.apply(lambda _: random.uniform(0,0.3), axis=1)
    user_df['total_seconds_to_95'] = user_df['initial_charge'].apply(lambda x: total_capacity*(0.95-x)/(rate/3600))
    
    user_df['full_charge_time']= user_df['plug_in_time'] + pd.to_timedelta(user_df['total_seconds_to_95'],unit='s')
    user_df['length_plugged_in'] = (user_df.unplug_time - user_df.plug_in_time) / pd.Timedelta(seconds=1)
    
    user_df['session_charge'] = user_df[['total_seconds_to_95','length_plugged_in']].min(axis=1)*(rate/3600)
    user_df['final_perc_charged'] = user_df.session_charge.apply(lambda x: x/total_capacity)
    user_df['final_perc_charged'] = user_df.final_perc_charged + user_df.initial_charge
    
    user_df['final_charge_time'] = user_df[['full_charge_time', 'unplug_time']].min(axis=1)
    user_df['uncharged'] = np.where(user_df['final_perc_charged'] <0.80, True, False)

    moer_temp = user_df[['user_type','distinct_dates','plug_in_time','final_charge_time']].copy()
    moer_temp['key'] = 1
    df['key'] = 1

    moer_temp = pd.merge(moer_temp,df, on='key').drop(columns='key')
    moer_temp = moer_temp[(moer_temp['point_time']>moer_temp['plug_in_time'])&(moer_temp['point_time']<moer_temp['final_charge_time'])]

    moer_temp = moer_temp.groupby(['distinct_dates','user_type']).agg(
        sum_moer=('values','sum'),
        count_moer_intervals=('values','count'),
        avg_moer=('values','mean')
    ).reset_index()

    user_df= pd.merge(user_df,moer_temp, on=['user_type','distinct_dates'])
    all_users = pd.concat([all_users, user_df], axis=0)
    print(all_users.shape)


# In[79]:


all_users.to_csv('dummy_evaluation_data.csv')


# In[78]:


all_users.shape


# In[ ]:




