from alg import optCharger, moer

m = moer.Moer(
    mu = [10,10,10,13,3,1,1,], 
    isDiagonal = True,
    sig2 = 1.0,
)
print(len(m))

print("No extra costs ... ")
model = optCharger.OptCharger(
    minChargeRate = 1,
    maxChargeRate = 5,
)
model.fit(totalCharge=10, totalTime=5, moer=m, asap=True); model.summary()
model.fit(totalCharge=10, totalTime=5, moer=m, asap=False); model.summary()
model.fit(totalCharge=10, totalTime=5, moer=m, asap=False, constraints = {0:(1,None), 1:(3,None)}); model.summary()

print("Introducing emission overheads...")
model = optCharger.OptCharger(
    minChargeRate = 1,
    maxChargeRate = 5,
    startEmissionOverhead = 10.,
    keepEmissionOverhead = 1.
)
model.fit(totalCharge=10, totalTime=5, moer=m, asap=True); model.summary()
model.fit(totalCharge=10, totalTime=5, moer=m, asap=False); model.summary()
model.fit(totalCharge=10, totalTime=5, moer=m, asap=False, constraints = {0:(1,None), 1:(3,None)}); model.summary()