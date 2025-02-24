from scipy import special
import numpy as np
import matplotlib.pyplot as plt

G = 4
M = (np.pi)**2

plt.style.use('dark_background')
ecc = 0.3
semiM = 1
phi_M = 0
phi_E = 0
phi = 0
dt = 0.0001
totalTime = 10
samples = int(totalTime / dt)
omega = 0
theta = 0
omegaList = [None]*samples
thetaList = [None]*samples

# calculating true anomaly
t = np.linspace(0,totalTime,samples + 1)
phi_M = 2*(np.pi)*t
def theotherbit(terms,besselTerms):
    sum = 0
    for n in range(1,terms):
        sum += (2/n) * special.jv(besselTerms,n*ecc) * np.sin(n*phi_M)
    return sum
phi_E = phi_M + theotherbit(500,500)
phi = 2 * (np.arctan(np.sqrt((1 + ecc)/(1 - ecc)) * (np.tan((phi_E)/2))))

# single RK4 timestep
def RK4(timeIndex, x, v):
    dx1 = dt * (v)
    dv1 = dt * -(3/2)*(G*M)*(1/((semiM*(1 - ecc**2)/(1 + ecc*(np.cos(phi[timeIndex]))))**3))*(np.sin(2*(x - phi[timeIndex])))
    dx2 = dt * (v + dv1/2)
    dv2 = dt * -(3/2)*(G*M)*(1/((semiM*(1 - ecc**2)/(1 + ecc*(np.cos((phi[timeIndex]+phi[timeIndex + 1])/2))))**3))*(np.sin(2*((x + dx1/2) - (phi[timeIndex]+phi[timeIndex + 1])/2)))
    dx3 = dt * (v + dv2/2)
    dv3 = dt * -(3/2)*(G*M)*(1/((semiM*(1 - ecc**2)/(1 + ecc*(np.cos((phi[timeIndex]+phi[timeIndex + 1])/2))))**3))*(np.sin(2*((x + dx2/2) - (phi[timeIndex]+phi[timeIndex + 1])/2)))
    dx4 = dt * (v + dv3)
    dv4 = dt * -(3/2)*(G*M)*(1/((semiM*(1 - ecc**2)/(1 + ecc*(np.cos(phi[timeIndex + 1]))))**3))*(np.sin(2*((x + dx3) - phi[timeIndex + 1])))
    x += (dx1 + 2*dx2 + 2*dx3 + dx4)/6
    v += (dv1 + 2*dv2 + 2*dv3 + dv4)/6
    return (x,v)

# assigning theta and omega
for i in range(0,samples):
    omegaList[i] = omega
    thetaList[i] = theta
    theta, omega = RK4(i, theta, omega)

# plot
fig, ax = plt.subplots()

# fudge to match len(t) with len(omegaList)
t = np.delete(t, -1)
ax.plot(t, omegaList, linewidth=2.0, color = "white")

plt.show()