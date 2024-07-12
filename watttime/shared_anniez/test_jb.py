from alg import optCharger, moer

m = moer.Moer(
    mu = [10,10,10,13,3,1,1,], # 1 MWh = 10 MT of CO2e. .5 MWH = 5 MT of co2e
    isDiagonal = False
)

print(len(m))

# greedy
print("No extra costs ... ")
model = optCharger.OptCharger(.5)
model.fit(totalCharge=1.5, totalTime=7, moer=m, asap=True); model.summary()

# simple
print("No extra costs ... ")
model = optCharger.OptCharger(.5)
model.fit(totalCharge=1.5, totalTime=7, moer=m, asap=False); model.summary()


moer_data = pd.DataFrame(
    {'point_time': [0, 1, 2, 3, 4, 5, 6],
     'value': [10,10,10,13,3,1,1,],
    })

def sum_moer_actuals(moer_data, MWh_fraction,plug_in_time, number_conseq_intervals):
    index_lower_limit = moer_data[moer_data.point_time >= plug_in_time].index[0]
    index_upper_limit = index_lower_limit + int(number_conseq_intervals)
    return sum(
        moer_data[index_lower_limit: index_upper_limit]["value"] * MWh_fraction
        )

sum_moer_actuals(
    moer_data = moer_data,
    MWh_fraction = 0.5,
    plug_in_time = 0,
    number_conseq_intervals = 3
)