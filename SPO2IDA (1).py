#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import math


# In[3]:


# [Base Shear Vb [kN],Roof Displacement 𝛥 [m]]
Yielding= [378.2, 0.008]
Hardening= [378.2, 0.023]
Softening= [112.5, 0.041]
Residual= [112.5, 0.05]
Degradation= [0.00, 0.16]
all_=[Yielding, Hardening, Softening, Residual, Degradation]
R=[]
Mu=[]
for i in all_:
    R.append(i[0]/Yielding[0])
    Mu.append(i[1]/Yielding[1])
story= 2
modal= [(32.8, 0.38), (37.91, 0.87)]

Total_mass= 0
for item in modal:
    Total_mass += float(item[0])
    
SDOF_mass = 0.0
for item in modal:
    product = float(item[0]) * float(item[1])
    SDOF_mass += product
    
a = 0.0
for item in modal:
    product = float(item[0]) * float(item[1]) * float(item[1])  
    a += product
    Transformation_factor = SDOF_mass/a
    
SDOF_Yield_displacement= Yielding[1] / Transformation_factor
SDOF_Yield_force= Yielding[0] / Transformation_factor
T= 2 *math.pi*((SDOF_mass*SDOF_Yield_displacement/SDOF_Yield_force)**0.5)


# In[8]:


R= np.array(R)
mu= np.array(Mu)

# Define coefficients {16%, 50%, 84%}
# Hardening branch
a_alpha_1 = np.array([
    [0.146, 0.8628, 1.024],
    [0.5926, 0.9235, 0.6034],
    [0.07312, 0.9195, 0.2466],
    [0.2965, 0.9632, 0.06141],
    [0.02688, 0.4745, 0.2511],
    [1.063, 0.0654, 0.0001],
    [0.3127, 0.04461, 0.07086]
])

b_alpha_1 = np.array([
    [0.5335, 0.7624, 0.9018],
    [0.4161, 0.5041, 0.1928],
    [0.4495, 0.1785, 0.4758],
    [0.2215, 1.022, 0.6903],
    [0.3699, 0.3253, 0.3254],
    [1.003, 0.4064, 0.939],
    [0.1462, 0.4479, 0.3948]
])

c_alpha_1 = np.array([
    [0.03444, 0.1643, 0.6555],
    [0.3194, 0.1701, 0.1072],
    [0.01667, 0.1147, 0.1232],
    [0.1087, 0.1694, 0.05664],
    [0.0158, 0.09403, 0.07067],
    [0.646, 0.02054, 0.00132],
    [0.07181, 0.01584, 0.02287]
])

a_beta_1 = np.array([
    [0.2008, -0.1334, 0.7182],
    [0.179, 0.3312, 0.132],
    [0.1425, 0.7985, 0.1233],
    [0.1533, 0.0001, 0.09805],
    [3.623E+12, 0.1543, 0.1429],
    [0.09451, 0.9252, 0.6547],
    [0.1964, 0.2809, 0.0001]
])

b_beta_1 = np.array([
    [1.093, 0.7771, 0.04151],
    [0.7169, 0.7647, 0.6058],
    [0.4876, 0.04284, 0.4904],
    [0.5709, 0.5721, 0.5448],
    [97.61, 0.4788, 0.3652],
    [0.4424, 0.8165, 0.8431],
    [0.3345, 0.3003, 0.7115]
])

c_beta_1 = np.array([
    [0.5405, 0.04907, 0.09018],
    [0.08836, 0.000986, 0.04845],
    [0.04956, 0.09365, 0.04392],
    [0.07256, 0.0001, 0.01778],
    [17.94, 0.105, 0.09815],
    [0.06262, 0.51, 0.7126],
    [0.09522, 0.1216, 0.0001803]
])

# Softening branch
a_alpha_2 = np.array([0.03945, 0.01833, 0.009508])
b_alpha_2 = np.array([-0.03069, -0.01481, -0.007821])
a_beta_2 = np.array([1.049, 0.8237, 0.4175])
b_beta_2 = np.array([0.2494, 0.04082, 0.03164])
a_gamma_2 = np.array([-0.7326, -0.7208, -0.0375])
b_gamma_2 = np.array([1.116, 1.279, 1.079])

# Residual plateau branch
a_alpha_3 = np.array([-5.075, -2.099, -0.382])
b_alpha_3 = np.array([7.112, 3.182, 0.6334])
c_alpha_3 = np.array([-1.572, -0.6989, -0.051])
d_alpha_3 = np.array([0.1049, 0.0481, 0.002])

a_beta_3 = np.array([16.16, 8.417, -0.027])
b_beta_3 = np.array([-26.5, -14.51, -1.80])
c_beta_3 = np.array([10.92, 6.75, 2.036])
d_beta_3 = np.array([1.055, 0.9061, 1.067])

# Strength degradation branch
a_alpha_4 = np.array([-1.564, -0.5954, -0.06693])
b_alpha_4 = np.array([2.193, 0.817, 0.1418])
c_alpha_4 = np.array([-0.352, -0.09191, 0.0124])
d_alpha_4 = np.array([0.0149, 0.001819, -0.002012])
a_beta_4 = np.array([1.756, 0.7315, -0.408])
b_beta_4 = np.array([-8.719, -3.703, -1.333])
c_beta_4 = np.array([8.285, 4.391, 2.521])
d_beta_4 = np.array([1.198, 1.116, 1.058])

# Compute the parameters for each fractile (16%, 50%, 84%)
alpha_1 = np.zeros(3)
beta_1 = np.zeros(3)
alpha_2 = np.zeros(3)
beta_2 = np.zeros(3)
gamma_2 = np.zeros(3)
alpha_3 = np.zeros(3)
beta_3 = np.zeros(3)
alpha_4 = np.zeros(3)
beta_4 = np.zeros(3)

for i in range(3):
    # Hardening branch
    alpha_1[i] = np.sum(a_alpha_1[:, i] * np.exp(-((T - b_alpha_1[:, i]) / c_alpha_1[:, i]) ** 2))
    beta_1[i] = np.sum(a_beta_1[:, i] * np.exp(-((T - b_beta_1[:, i]) / c_beta_1[:, i]) ** 2))

    # Softening branch
    alpha_2[i] = a_alpha_2[i] * T + b_alpha_2[i]
    beta_2[i] = a_beta_2[i] * T + b_beta_2[i]
    gamma_2[i] = a_gamma_2[i] * T + b_gamma_2[i]

    # Residual branch
    alpha_3[i] = a_alpha_3[i] * T ** 3 + b_alpha_3[i] * T ** 2 + c_alpha_3[i] * T + d_alpha_3[i]
    beta_3[i] = a_beta_3[i] * T ** 3 + b_beta_3[i] * T ** 2 + c_beta_3[i] * T + d_beta_3[i]

    # Strength degradation branch
    alpha_4[i] = a_alpha_4[i] * T ** 3 + b_alpha_4[i] * T ** 2 + c_alpha_4[i] * T + d_alpha_4[i]
    beta_4[i] = a_beta_4[i] * T ** 3 + b_beta_4[i] * T ** 2 + c_beta_4[i] * T + d_beta_4[i]

# Fit the branches and adjust discontinuities
mu_1 = np.linspace(1, mu[0], 10)
mu_2 = np.linspace(mu[0], mu[1], 10)
mu_3 = np.linspace(mu[1], mu[2], 10)
mu_4 = np.linspace(mu[2], mu[3], 10)

Rdyn_1 = np.zeros((3, 10))
Rdyn_2 = np.zeros((3, 10))
Rdyn_3 = np.zeros((3, 10))
Rdyn_4 = np.zeros((3, 10))

for j in range(3):
    for i in range(10):
        Rdyn_1[j, i] = alpha_1[j] * mu_1[i] ** beta_1[j]

for j in range(3):
    for i in range(10):
        Rdyn_2[j, i] = alpha_2[j] * mu_2[i] ** 2 + beta_2[j] * mu_2[i] + gamma_2[j]

for j in range(3):
    for i in range(10):
        Rdyn_3[j, i] = alpha_3[j] * mu_3[i] + beta_3[j]

for j in range(3):
    for i in range(10):
        Rdyn_4[j, i] = alpha_4[j] * mu_4[i] + beta_4[j]

Rdyn = np.concatenate((Rdyn_1, Rdyn_2, Rdyn_3, Rdyn_4), axis=1).T
mudyn = np.concatenate((mu_1, mu_2, mu_3, mu_4))

# Hardening Initiation
Rdiff0 = 1 - Rdyn[0, :3]
for i in range(3):
    if Rdiff0[i] < 0:
        Rdyn[:10, i] -= abs(Rdiff0[i])
    else:
        Rdyn[:10, i] += abs(Rdiff0[i])

# Connection Hardening-Softening
Rdiff1 = Rdyn[9, :3] - Rdyn[10, :3]
for i in range(3):
    if Rdiff1[i] < 0:
        Rdyn[10:20, i] -= abs(Rdiff1[i])
    else:
        Rdyn[10:20, i] += abs(Rdiff1[i])

# Connection Softening-Plateau
Rdiff2 = Rdyn[19, :3] - Rdyn[20, :3]
for i in range(3):
    if Rdiff2[i] < 0:
        Rdyn[20:30, i] -= abs(Rdiff2[i])
    else:
        Rdyn[20:30, i] += abs(Rdiff2[i])

# Connection Plateau-Degradation
Rdiff3 = Rdyn[29, :3] - Rdyn[30, :3]
for i in range(3):
    if Rdiff3[i] < 0:
        Rdyn[30:40, i] -= abs(Rdiff3[i])
    else:
        Rdyn[30:40, i] += abs(Rdiff3[i])

# Add a flatline point
Rdyn = np.vstack((Rdyn, Rdyn[-1, :]))
mudyn = np.append(mudyn, mudyn[-1] + 5)

# Plot the diagram
plt.figure()
plt.plot(mudyn, Rdyn[:, 0], '-r')
plt.plot(mudyn, Rdyn[:, 1], '-b')
plt.plot(mudyn, Rdyn[:, 2], '-m')
plt.plot([0, 1, *mu], [0, 1, *R], '-k')
plt.legend(['16%', '50%', '84%', 'SPO'], loc='upper right')  # Change 'southeast' to 'upper right'
plt.xlabel('Ductility \u03BC')
plt.ylabel('Strength Ratio R')
plt.grid(True)
plt.show()


# In[ ]:




