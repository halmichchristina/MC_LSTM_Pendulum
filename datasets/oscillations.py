import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def create_Oscillation(lam,
                       friction,
                       pendulum_length = 1,
                       initial_amplitude = 0.3):
    # Gravitational constant g in m/s^2
    g = 9.81
    #Mass in kg
    m = 1
    #Length of Pendulum in m
    r = pendulum_length
    #Initial amplitude
    a = initial_amplitude

    if(friction == False):
        lam = 0

    #Lists for dataset creation
    #G,M,L,Lam,A,T,K,P = [],[],[],[],[],[],[],[]
    #Implement formulas from https://nrich.maths.org/content/id/6478/Paul-not%20so%20simple%20pendulum%202.pdf
    k = np.sqrt((m*g)/r)
    b = lam
    h = np.sqrt((k**2)/m - (b**2)/(4*(m**2)))
    t = np.arange(0,25,0.025)
    y = a*np.exp((-lam/(2*m))*t)*(np.cos(h*t)+(lam/(2*m*h))*np.sin(h*t))
    #dy/dt for kinetic energy calculation
    w = (- a * ((lam**2) + (4*(h**2)*(m**2)))*np.exp(-(lam*t)/(2*m))*np.sin(h*t))/(4*h*m**2)
    #Energy calculation
    E_pot = m*g*r*(1-np.cos(y))
    E_kin = 0.5*m*(w**2)*(r**2)
    #Deflection
    s = r*y

    #Show Energy graph
    # plt.title('Amplitude of pendulum over time')
    # plt.ylabel('Amplitude ')
    # plt.xlabel('Time in s')
    # plt.autoscale(axis='t', tight=True)
    # plt.plot(t,s, label = 'Amplitude')
    # #plt.plot(t,w, label = 'W')
    # plt.legend()
    # plt.savefig("./Figures/Amplitude.jpg", bbox_inches='tight')
    # plt.clf()


    #print('Angle max, min', y.max(), y.min())

    #Append new sample to dataset
    #G.append(g)
    #M.append(m)
    #L.append(r)
    #Lam.append(lam)
    #A.append(a)
    #T.append(t)
    #K.append(E_kin)
    #P.append(E_pot)

    #Save dataset
    data = {'Time': t,  #T
            'Kinetic Energy': E_kin,  #K
            'Potential Energy': E_pot, #P
            'Angle':y,
            'Deflection':s, 
            'Damping Factor': lam, #Lam
            'Acceleration': g,#G
            'Length of String': r,#L
            'Mass':m,#M
            'Initial Amplitude':a}#A
    df = pd.DataFrame(data, columns=['Time','Kinetic Energy','Potential Energy','Angle','Deflection','Damping Factor','Acceleration','Length of String','Mass','Initial Amplitude'])
    return df
