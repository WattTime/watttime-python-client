from dataclasses import dataclass
import pandas as pd
from typing import List

@dataclass
class EvalMetrics:
    baseline_col: str 
    ideal_col: str
    forecast_col: str
    actuals_col: str

    def percent_diff(self, y,x):
        return ((y-x)/x)*100

    def forecast_less_baseline(self, results_frame: pd.DataFrame, percent = False): # correct
        if percent == True:
            return self.percent_diff(results_frame[self.forecast_col],results_frame[self.baseline_col])
        else:
            return results_frame[self.forecast_col] - results_frame[self.baseline_col]
   
    def forecast_less_ideal(self, results_frame: pd.DataFrame, percent = False): # correct
        if percent == True:
           return self.percent_diff(results_frame[self.forecast_col],results_frame[self.ideal_col])
        else:
            return   results_frame[self.forecast_col] - results_frame[self.ideal_col]
    
    def forecast_less_actuals(self, results_frame: pd.DataFrame, percent = False): # correct
        if percent == True:
           return self.percent_diff(results_frame[self.actuals_col],results_frame[self.forecast_col])
        else:
            return  results_frame[self.forecast_col] - results_frame[self.actuals_col]
    
    def acutals_less_baseline(self, results_frame: pd.DataFrame, percent = False): # correct
        if percent == True:
           return self.percent_diff(results_frame[self.actuals_col],results_frame[self.baseline_col])
        else:
            return results_frame[self.actuals_col] - results_frame[self.baseline_col]
        
    def calculate_results(self, results_frame: pd.DataFrame, keep_cols: List[str], percent = False):
            columns = ["emissions_avoided","expected_avoidance","nearness_to_ideal","forecast_error"]
            df = pd.DataFrame(
                    list(
                        zip(
                            self.acutals_less_baseline(results_frame, percent=percent).to_list(),
                            self.forecast_less_baseline(results_frame, percent=percent).to_list(),
                            self.forecast_less_ideal(results_frame, percent=percent).to_list(),
                            self.forecast_less_actuals(results_frame, percent=percent).to_list()
                            )
                        ),
                    columns = columns  
                )
            if percent == True:
                columns = [c + "_percent" for c in columns]
            
            df.columns = columns

            return pd.concat([results_frame[keep_cols],df])