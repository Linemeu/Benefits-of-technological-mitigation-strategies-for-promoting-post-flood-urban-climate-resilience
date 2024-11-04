
#-------------CAP
data = df.iloc[96:384]
data['Rain_l1'] = data['Rain'].shift(-1)
formula = """d_CAPI~UAV+Rain+LSD+REE
             REE~UAV+Rain+LSD
             LSD~UAV+Rain
             # DW~LSD+Rain
             """
mod = semopy.Model(formula)
res = mod.fit(data)
from semopy import Optimizer

opt = Optimizer(mod)
opt.optimize('DWLS')
print(res)
CAP_ins = mod.inspect()
print(CAP_ins)
CAP_stats = semopy.calc_stats(mod)
print(CAP_stats.T)


#-------------CP
data = df.iloc[96:384]
data['Rain_l1'] = data['Rain'].shift(-1)
formula = """d_CPI~UAV+Rain+LSD+REE
             REE~UAV+Rain+LSD
             LSD~UAV+Rain
             """
mod = semopy.Model(formula)
res = mod.fit(data)
from semopy import Optimizer

opt = Optimizer(mod)
opt.optimize('DWLS')
# opt.optimize(objective='MLW')
# opt = Optimizer(model)
# objective_function_value = opt.optimize()
print(res)
CP_ins = mod.inspect()
# g = semopy.semplot(ins, "pd.png")
print(CP_ins)
CP_stats = semopy.calc_stats(mod)
print(CP_stats.T)

#-------------TP
data = df.iloc[96:384]
data['Rain_l1'] = data['Rain'].shift(-1)
formula = """d_TPI~Rain+AMap+REE+UAV
             REE~Rain+AMap+DW
             AMap~UAV+Rain
             # DW~+Rain
             
             # DW~Rain+AMap+UAV
             """
mod = semopy.Model(formula)
res = mod.fit(data)
from semopy import Optimizer

opt = Optimizer(mod)
opt.optimize('DWLS')

print(res)
TP_ins = mod.inspect()
print(TP_ins)
TP_stats = semopy.calc_stats(mod)
print(TP_stats.T)





#-------------WEP
data = df.iloc[96:384]
data['Rain_l1'] = data['Rain'].shift(-1)
formula = """d_WPI~UAV+Rain+LSD+REE
             REE~UAV+Rain+LSD
             LSD~UAV+Rain
             """
mod = semopy.Model(formula)
res = mod.fit(data)
from semopy import Optimizer

opt = Optimizer(mod)
opt.optimize('DWLS')

print(res)
WP_ins = mod.inspect()
print(WP_ins)
WP_stats = semopy.calc_stats(mod)
print(WP_stats.T)


#-------------BP
data = df.iloc[96:384]
data['Rain_l1'] = data['Rain'].shift(-1)
formula = """d_BPI~UAV+Rain+DW+REE
             REE~UAV+Rain+DW
             DW~UAV+Rain
             """
mod = semopy.Model(formula)
res = mod.fit(data)
from semopy import Optimizer

opt = Optimizer(mod)
opt.optimize('DWLS')

print(res)
BP_ins = mod.inspect()
print(BP_ins)
BP_stats = semopy.calc_stats(mod)
print(BP_stats.T)
