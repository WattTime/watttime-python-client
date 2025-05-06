
from watttime_optimizer.evaluator.evaluator import RecalculationOptChargeEvaluator
from watttime_optimizer.evaluator.evaluator import OptChargeEvaluator
from watttime_optimizer.evaluator.evaluator import ImpactEvaluator
import numpy as np
import tqdm


# 4 seconds per row, mostly API call
def analysis_loop(region, input_dict,username,password):
    oce = OptChargeEvaluator(username=username,password=password)
    results = {}
    for key,value in tqdm.tqdm(input_dict.items()):
        value.update({'region':region,'tz_convert':True, "verbose":False})
        df = oce.get_schedule_and_cost_api(**value)
        m, b = np.polyfit(np.arange(len(df.pred_moer.values)),df.pred_moer.values, 1)
        stddev = df.pred_moer.std()
        r = ImpactEvaluator(username,password,df).get_all_emissions_values(region=region)
        r.update({'m':m,'b':b,'stddev':stddev})
        results.update({key:r})
    return results

# 4 seconds per row, mostly API call
def analysis_loop_requery(region, input_dict, interval,username,password):
    roce = RecalculationOptChargeEvaluator(username,password)
    results = {}
    for key,value in tqdm.tqdm(input_dict.items()):
        value.update(
            {'region':region,
            'tz_convert':True, 
            "optimization_method": "auto", 
            "verbose":False,
            "interval":interval,
            "charge_per_segment":None}
            )
        df = roce.fit_recalculator(**value)
        m, b = np.polyfit(np.arange(len(df.pred_moer.values)),df.pred_moer.values, 1)
        stddev = df.pred_moer.std()
        r = ImpactEvaluator(username,password,df).get_all_emissions_values(region=region)
        r.update({'m':m,'b':b,'stddev':stddev})
        results.update({key:r})
    return results

# 4 seconds per row, mostly API call
def analysis_loop_requery_contiguous(region, input_dict, interval,username,password):
    roce = RecalculationOptChargeEvaluator(username,password)
    results = {}
    for key,value in tqdm.tqdm(input_dict.items()):
        try:
            value.update(
                {'region':region,
                'tz_convert':True,
                "optimization_method": "auto", 
                "verbose":False,
                "interval":interval,
                "contiguous":True
                }
                )
            df = roce.fit_recalculator(**value).get_combined_schedule()
            m, b = np.polyfit(np.arange(len(df.pred_moer.values)),df.pred_moer.values, 1)
            stddev = df.pred_moer.std()
            r = ImpactEvaluator(username,password,df).get_all_emissions_values(region=region)
            r.update({'m':m,'b':b,'stddev':stddev})
            results.update({key:r})
        except:
            print('error')
            pass
    return results