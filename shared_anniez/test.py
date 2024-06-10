from alg import optCharger, moer

m = moer.Moer(
    mu = [10,10,10,13,3], 
    sig2 = 1.0,
    ac1 = 0.9, 
    ra = 0.
)
# print(m.length())
# print(m.isDiagonal())
# print(m.getMarginalCost(2,3))
# print(m.getTotalCost([0,0,0,1,1]))
# print(m.getTotalUtil([0,2,0,1,1]))

model = optCharger.OptCharger(
    totalCharge = 5, 
    moer = m, 
    minChargeRate = 1,
    maxChargeRate = 2,
    startChargeCost = 15,
    keepChargeCost = 1,
)
model.fit()
model.summary()