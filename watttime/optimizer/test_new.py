from alg import moer, optCharger
model = optCharger.OptCharger()

m = moer.Moer(
    mu = [10,10,10,1,13,3,2,3], 
)
print("Length of schedule:", len(m))

print("greedy algo")
model.fit(total_charge=3, total_time=8, moer=m, optimization_method='baseline'); model.summary()
print("simple sorting algo")
model.fit(total_charge=3, total_time=8, moer=m, ); model.summary()
print("sophisticated algo that produces same answer as simple")
model.fit(total_charge=3, total_time=8, moer=m, optimization_method='sophisticated'); model.summary()
print("incorrect pairing of simple sorting algo + variable charge rate")
model.fit(total_charge=3, total_time=8, moer=m, emission_multiplier_fn=lambda x,y: [1.0,2.0,1.0][x], optimization_method='simple'); model.summary()
print("sophisticated algo + variable charge rate")
model.fit(total_charge=3, total_time=8, moer=m, emission_multiplier_fn=lambda x,y: [1.0,2.0,1.0][x]); model.summary()
print("sophisticated algo + constraints")
model.fit(total_charge=3, total_time=8, moer=m, constraints = {2:(2,None)}); model.summary()

m = moer.Moer(
    mu = [2,1,10,10,10,1,13,3], 
)

# Fixed Contiguous
print("One contiguous interval")
model.fit(total_charge=3, total_time=8, moer=m, charge_per_interval=[3]); model.summary()
print("Two contiguous intervals")
model.fit(total_charge=3, total_time=8, moer=m, charge_per_interval=[2,1]); model.summary()
print ("Two contiguous intervals, one of which given as intervals + variable power rate")
model.fit(total_charge=3, total_time=8, moer=m, charge_per_interval=[(2,2),1], emission_multiplier_fn=lambda x,y: [1.0,0.1,1.0][x]); model.summary()
print ("Two contiguous intervals, one of which given as intervals + variable power rate")
model.fit(total_charge=3, total_time=8, moer=m, charge_per_interval=[2,(1,1)], emission_multiplier_fn=lambda x,y: [1.0,0.1,1.0][x]); model.summary()
print ("Two contiguous intervals, one of which given as intervals + variable power rate")
model.fit(total_charge=3, total_time=8, moer=m, charge_per_interval=[(2,2),(1,1)], emission_multiplier_fn=lambda x,y: [1.0,0.1,1.0][x]); model.summary()
print ("Two contiguous intervals + variable power rate + constraints")
model.fit(total_charge=4, total_time=8, moer=m, charge_per_interval=[3,1], constraints = {2:(None,1),5:(3,None)}); model.summary()

# Variable Contiguous
print("One contiguous interval")
model.fit(total_charge=3, total_time=8, moer=m, charge_per_interval=[(0,3)]); model.summary()
print("Two contiguous intervals")
model.fit(total_charge=3, total_time=8, moer=m, charge_per_interval=[(1,2),(0,3)]); model.summary()
print ("Two contiguous intervals + variable power rate")
model.fit(total_charge=3, total_time=8, moer=m, charge_per_interval=[(1,2),(1,2)], emission_multiplier_fn=lambda x,y: [1.0,0.1,1.0][x]); model.summary()
print ("Two contiguous intervals + variable power rate")
model.fit(total_charge=4, total_time=8, moer=m, charge_per_interval=[(1,3),(0,3)]); model.summary()
print ("Two contiguous intervals + variable power rate + constraints")
model.fit(total_charge=4, total_time=8, moer=m, charge_per_interval=[(1,3),(0,3)], constraints = {2:(None,1)}); model.summary()
print ("Two contiguous intervals + variable power rate + constraints")
model.fit(total_charge=4, total_time=8, moer=m, charge_per_interval=[(1,3),(0,3)], constraints = {2:(None,1),5:(3,None)}); model.summary()

m = moer.Moer(
    mu = [10,1,1,1,10,1,1,1], 
)
print ("Three contiguous intervals of fixed lengths")
model.fit(total_charge=6, total_time=8, moer=m, charge_per_interval=[2]*3); model.summary()
print ("Three contiguous intervals of variable lengths")
model.fit(total_charge=6, total_time=8, moer=m, charge_per_interval=[(2,6)]*3); model.summary()
print ("Three contiguous intervals of variable lengths, but doesnt need to charge all intervals")
model.fit(total_charge=6, total_time=8, moer=m, charge_per_interval=[(2,6)]*3, use_all_intervals=False); model.summary()

