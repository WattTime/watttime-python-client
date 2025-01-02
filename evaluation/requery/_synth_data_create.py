import evaluation.eval_framework as evu
import data.s3 as s3u

s3 = s3u.s3_utils()

'''
- 2024 dates only
- 1000 users
- sanity check on 9 current regions + 9 randomly selected other regions
- set of requery increments to test: none, 5,15,60,180
- charging windows of lengths 3,6,12 hours
- Charge needed at least 45 minutes (25% of smallest window)
'''

dates_2023 = evu.generate_random_dates(2023)
dates_2024 = evu.generate_random_dates(2024)
distinct_date_list = dates_2023 + dates_2024

dates_2024_only = list(filter(lambda x: x.year == 2024, distinct_date_list))

req_kwargs = {
    "power_output_max_rates": [42.5], # BMW average
    "max_percent_capacity": 0.95,  # highest level of charge achieved by battery
    "power_output_efficiency": 0.75,  # power loss. 1 = no power loss.
    "average_battery_starting_capacity": 0.5,  # average starting percent charged
    "start_hour": "00:00:00",  # earliest session can start
    "end_hour": "23:59:00",  # latest session can start
    "user_charge_tolerance":1, # must complete
    "session_lengths":[x*60*60 for x  in [3,6,12]] # convert hours to seconds
}

df_req = evu.execute_synth_data_process(
    distinct_date_list=dates_2024_only, number_of_users=1000, **req_kwargs
)

df_req = df_req.query('usage_time_required_minutes > 45')

print(f"total sample frame will be: {df_req.shape[0]*18}")

s3.store_csvdataframe(
    df_req, f"requery_data/20241203_1k_synth_users_96_days.csv"
)