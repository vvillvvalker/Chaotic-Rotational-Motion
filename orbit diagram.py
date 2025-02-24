from scipy import special
import numpy as np
import matplotlib.pyplot as plt

G = 4
M = (np.pi)**2

plt.style.use('dark_background')
semiM = 1
phi_M = 0
phi_E = 0
phi = 0
dt = 0.001
totalTime = 100
samples = int(totalTime / dt)
eccSamples = 10000
# eccentricity is strictly less than 1
ecc = np.linspace(0,0.9999,eccSamples)
omega = 0
theta = 0
omegaList = [None]*samples
thetaList = [None]*samples
omegaBiList = np.array([None]*samples)

def theotherbit(terms,besselTerms):
        sum = 0
        for n in range(1,terms):
            sum += (2/n) * special.jv(besselTerms,n*ecc[k]) * np.sin(n*phi_M)
        return sum

# Here you'll see a lot of omega = 0 and theta = 0. One of them is important and I do not know which.

for k in range(0,len(ecc)):
    omega = 0
    theta = 0
    # calculating true anomaly
    t = np.linspace(0,totalTime,samples + 1)
    phi_M = 2*(np.pi)*t
    # 50 term summation and 50 terms in Bessel approximation are probably overkill
    phi_E = phi_M + theotherbit(50,50)
    phi = 2 * (np.arctan(np.sqrt((1 + ecc[k])/(1 - ecc[k])) * (np.tan((phi_E)/2))))

    omega = 0
    theta = 0

    # single RK4 timestep
    # provides results similar to Euler (just more accurate) as a sanity check
    def RK4(timeIndex, x, v):
        dx1 = dt * (v)
        dv1 = dt * -(3/2)*(G*M)*(1/((semiM*(1 - (ecc[k])**2)/(1 + (ecc[k])*(np.cos(phi[timeIndex]))))**3))*(np.sin(2*(x - phi[timeIndex])))
        dx2 = dt * (v + dv1/2)
        dv2 = dt * -(3/2)*(G*M)*(1/((semiM*(1 - (ecc[k])**2)/(1 + (ecc[k])*(np.cos((phi[timeIndex]+phi[timeIndex + 1])/2))))**3))*(np.sin(2*((x + dx1/2) - (phi[timeIndex]+phi[timeIndex + 1])/2)))
        dx3 = dt * (v + dv2/2)
        dv3 = dt * -(3/2)*(G*M)*(1/((semiM*(1 - (ecc[k])**2)/(1 + (ecc[k])*(np.cos((phi[timeIndex]+phi[timeIndex + 1])/2))))**3))*(np.sin(2*((x + dx2/2) - (phi[timeIndex]+phi[timeIndex + 1])/2)))
        dx4 = dt * (v + dv3)
        dv4 = dt * -(3/2)*(G*M)*(1/((semiM*(1 - (ecc[k])**2)/(1 + (ecc[k])*(np.cos(phi[timeIndex + 1]))))**3))*(np.sin(2*((x + dx3) - phi[timeIndex + 1])))
        x += (dx1 + 2*dx2 + 2*dx3 + dx4)/6
        v += (dv1 + 2*dv2 + 2*dv3 + dv4)/6
        return (x,v)

    # assigning theta and omega
    for i in range(0,samples):
        if i > int(50 / dt):
            omegaList[i] = omega
            thetaList[i] = theta
        theta, omega = RK4(i, theta, omega)

    t = np.delete(t, -1)
    
    omega = 0
    theta = 0

    # 2x2 matrix where each column is one full simulation
    omegaBiList[k] = list(omegaList)

    # progress bar for the impatient
    print(k, " / ", len(ecc))

# plot
fig, ax = plt.subplots()

# selects omega for periodic t (t = 0,1,2...)
reducedOmegaList = [None]*eccSamples
for k in range(0,int(np.floor(totalTime / 1))):
    if int(k * (1 / dt)) > 50:
        for i in range(0,eccSamples):
            reducedOmegaList[i] = omegaBiList[i][int(k * (1 / dt))]
    plt.scatter(ecc, reducedOmegaList, s = 0.01, color="white")

ax.set(xlim=(0, 1), ylim=(-80, 80))

plt.show()