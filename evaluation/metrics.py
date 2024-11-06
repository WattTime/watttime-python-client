from dataclasses import dataclass
import pandas as pd
import numpy as np

@dataclass
class EvalMetrics:
    baseline_col: str 
    ideal_col: str
    forecast_col: str
    actuals_col: str

    def percent_diff(self, y,x):
        return (y-x)/x

    def baseline_difference(self, results_frame: pd.DataFrame, percent = False):
        if percent == True:
            return self.percent_diff(results_frame[self.forecast_col],results_frame[self.baseline_col])
        else:
            return results_frame[self.forecast_col] - results_frame[self.baseline_col]
   
    def ideal_difference(self, results_frame: pd.DataFrame, percent = False):
        if percent == True:
           return self.percent_diff(results_frame[self.forecast_col],results_frame[self.ideal_col])
        else:
            return results_frame[self.forecast_col] - results_frame[self.ideal_col]
    
    def actuals_difference(self, results_frame: pd.DataFrame, percent = False):
        if percent == True:
           return self.percent_diff(results_frame[self.forecast_col],results_frame[self.actuals_col])
        else:
            return results_frame[self.forecast_col] - results_frame[self.actuals_col]
    
    def emissions_avoided(self, results_frame: pd.DataFrame, percent = False):
        if percent == True:
           return self.percent_diff(results_frame[self.baseline_col],results_frame[self.actuals_col])
        else:
            return results_frame[self.baseline_col] - results_frame[self.actuals_col]
        
    def calculate_results(self, results_frame: pd.DataFrame, percent = False):
            columns = ["emissions_avoided","baseline_difference","ideal_difference","actuals_difference"]
            df = pd.DataFrame(
                    list(
                        zip(
                            self.emissions_avoided(results_frame, percent=percent),
                            self.baseline_difference(results_frame, percent=percent),
                            self.ideal_difference(results_frame, percent=percent),
                            self.actuals_difference(results_frame, percent=percent).to_list()
                            )
                            ),
                    columns = columns  
                )
            if percent == True:
                columns = [c + "_percent" for c in columns]
            
            df.columns = columns
            return df