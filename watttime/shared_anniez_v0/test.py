from alg import optCharger, moer

m = moer.Moer(
    mu = [10,10,10,1,13,3,2,3], 
    isDiagonal = True,
)
print(len(m))

print("No extra costs ... ")
model = optCharger.OptCharger(
    fixedChargeRate = 1,
)
model.fit(totalCharge=3, totalTime=8, moer=m, asap=True); model.summary()
model.fit(totalCharge=3, totalTime=8, moer=m, asap=False); model.summary()
model.fit(totalCharge=3, totalTime=8, moer=m, asap=False, emission_multiplier_fn=lambda x,y: [1.0,2.0,1.0][x]); model.summary()
model.fit(totalCharge=3, totalTime=8, moer=m, asap=False, constraints = {0:(1,None), 1:(2,None)}); model.summary()
model.fit(totalCharge=3, totalTime=8, moer=m, asap=False, totalIntervals=1); model.summary()
model.fit(totalCharge=3, totalTime=8, moer=m, asap=False, totalIntervals=1, emission_multiplier_fn=lambda x,y: [1.0,0.1,1.0][x]); model.summary()
model.fit(totalCharge=3, totalTime=8, moer=m, asap=False, totalIntervals=1, constraints = {0:(1,None)}); model.summary()
model.fit(totalCharge=3, totalTime=8, moer=m, asap=False, totalIntervals=2, constraints = {0:(1,None)}); model.summary()

# print("Introducing emission overheads...")
# model = optCharger.OptCharger(
#     minChargeRate = 1,
#     maxChargeRate = 5,
#     startEmissionOverhead = 10.,
#     keepEmissionOverhead = 1.
# )
# model.fit(totalCharge=10, totalTime=5, moer=m, asap=True); model.summary()
# model.fit(totalCharge=10, totalTime=5, moer=m, asap=False); model.summary()
# model.fit(totalCharge=10, totalTime=5, moer=m, asap=False, constraints = {0:(1,None), 1:(3,None)}); model.summary()