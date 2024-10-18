from alg import moer, optCharger

m = moer.Moer(
    mu = [10,10,10,1,13,3,2,3], 
)
print("Length of schedule:", len(m))

model = optCharger.OptCharger()
print("greedy algo")
model.fit(totalCharge=3, totalTime=8, moer=m, optimization_method='baseline'); model.summary()
print("simple sorting algo")
model.fit(totalCharge=3, totalTime=8, moer=m, ); model.summary()
print("sophisticated algo that produces same answer as simple")
model.fit(totalCharge=3, totalTime=8, moer=m, optimization_method='sophisticated'); model.summary()
print("incorrect pairing of simple sorting algo + variable charge rate")
model.fit(totalCharge=3, totalTime=8, moer=m, emission_multiplier_fn=lambda x,y: [1.0,2.0,1.0][x], optimization_method='simple'); model.summary()
print("sophisticated algo + variable charge rate")
model.fit(totalCharge=3, totalTime=8, moer=m, emission_multiplier_fn=lambda x,y: [1.0,2.0,1.0][x]); model.summary()
print("sophisticated algo + constraints")
model.fit(totalCharge=3, totalTime=8, moer=m, constraints = {2:(2,None)}); model.summary()

m = moer.Moer(
    mu = [2,1,10,10,10,1,13,3], 
)

# Contiguous
print("One contiguous interval")
model.fit(totalCharge=3, totalTime=8, moer=m, charge_per_interval=[(0,3)]); model.summary()
print("Two contiguous intervals")
model.fit(totalCharge=3, totalTime=8, moer=m, charge_per_interval=[(1,2),(0,3)]); model.summary()
print ("Two contiguous intervals + variable power rate")
model.fit(totalCharge=3, totalTime=8, moer=m, charge_per_interval=[(1,2),(1,2)], emission_multiplier_fn=lambda x,y: [1.0,0.1,1.0][x]); model.summary()
print ("Two contiguous intervals + variable power rate")
model.fit(totalCharge=4, totalTime=8, moer=m, charge_per_interval=[(1,3),(0,3)]); model.summary()
print ("Two contiguous intervals + variable power rate + constraints")
model.fit(totalCharge=4, totalTime=8, moer=m, charge_per_interval=[(1,3),(0,3)], constraints = {2:(None,1)}); model.summary()
print ("Two contiguous intervals + variable power rate + constraints")
model.fit(totalCharge=4, totalTime=8, moer=m, charge_per_interval=[(1,3),(0,3)], constraints = {2:(None,1),5:(3,None)}); model.summary()