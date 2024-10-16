from alg import moer_new, optCharger_new

m = moer_new.Moer(
    mu = [10,10,10,1,13,3,2,3], 
)
print(len(m))

print("No extra costs ... ")
model = optCharger_new.OptCharger()
model.fit(totalCharge=3, totalTime=8, moer=m, asap=True); model.summary()
model.fit(totalCharge=3, totalTime=8, moer=m, asap=False, ); model.summary()
model.fit(totalCharge=3, totalTime=8, moer=m, asap=False, optimization_method='sophisticated'); model.summary()
model.fit(totalCharge=3, totalTime=8, moer=m, asap=False, emission_multiplier_fn=lambda x,y: [1.0,2.0,1.0][x]); model.summary()
model.fit(totalCharge=3, totalTime=8, moer=m, asap=False, emission_multiplier_fn=lambda x,y: [1.0,2.0,1.0][x], optimization_method='simple'); model.summary()

model.fit(totalCharge=3, totalTime=8, moer=m, asap=False, constraints = {0:(1,None), 1:(2,None)}); model.summary()
model.fit(totalCharge=3, totalTime=8, moer=m, asap=False, charge_per_interval=[(1,2),(0,3)]); model.summary()
model.fit(totalCharge=3, totalTime=8, moer=m, asap=False, charge_per_interval=[(1,2),(0,1)], emission_multiplier_fn=lambda x,y: [1.0,0.1,1.0][x]); model.summary()
model.fit(totalCharge=3, totalTime=8, moer=m, asap=False, charge_per_interval=[(1,2),(0,3)], constraints = {0:(1,None)}); model.summary()
model.fit(totalCharge=3, totalTime=8, moer=m, asap=False, charge_per_interval=[(1,2),(0,3)], constraints = {0:(1,None),6:(None,2)}); model.summary()