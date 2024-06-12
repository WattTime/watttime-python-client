from alg import optCharger, moer

m = moer.Moer(
    mu = [10,10,10,13,3], 
    isDiagonal = True,
    sig2 = 1.0,
    # ac1 = 0.9, 
    ra = 0.
)
print(len(m))
# print(m.isDiagonal())
# print(m.getMarginalCost(2,3))
# print(m.getTotalCost([0,0,0,1,1]))
# print(m.getTotalUtil([0,2,0,1,1]))

model = optCharger.OptCharger(
    totalCharge = 10, 
    moer = m, 
    minChargeRate = 1,
    maxChargeRate = 4,
    startChargeCost = 0,
    keepChargeCost = 0,
    constraints = {2:(5,None)}
)
model.fit()
model.summary()