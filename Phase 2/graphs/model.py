import numpy as np
import scipy as sp

def modelCov(pars, days):
    S = [67500000]
    I00 = pars['lol']*150000
    E = [pars['lol1']*I00]
    IS = [pars['lol3']*(1-pars['lol1'])*(1-pars['alpha2'])*I00]
    ISD = [(1-pars['lol3'])*(1-pars['lol1'])*(1-pars['alpha2'])*I00]
    IA = [pars['lol4']*(1-pars['lol1'])*(pars['alpha2'])*I00]
    IAD = [(1-pars['lol4'])*(1-pars['lol1'])*(pars['alpha2'])*I00]
    D = [41358]
    R = [0]
    N = [S[0]+E[0]+IS[0]+ISD[0]+IA[0]+IAD[0]+R[0]]
    IDD = [677]
    I = [IAD[0]+ISD[0]]

    for t in range(days-1):
        N.append(S[t] + E[t] + IA[t] + IAD[t] + IS[t] + ISD[t] + R[t])

        S.append(S[t] - (pars['beta1']*IA[t] + pars['beta2']*IS[t] + pars['beta3']*IAD[t] + pars['beta4']*ISD[t])*S[t]/N[t])
        E.append(E[t] + (pars['beta1']*IA[t] + pars['beta2']*IS[t] + pars['beta3']*IAD[t] + pars['beta4']*ISD[t])*S[t]/N[t] - pars['alpha1']*pars['alpha2']*E[t] - pars['alpha1']*(1-pars['alpha2'])*E[t])

        IA.append(IA[t] + pars['alpha1']*pars['alpha2']*E[t] - pars['gamma1']*IA[t] - pars['rho1']*IA[t])
        IAD.append(IAD[t] + pars['rho1']*IA[t]  - pars['gamma1']*IAD[t])

        IS.append(IS[t] + pars['alpha1']*(1-pars['alpha2'])*E[t] - pars['gamma2']*IS[t] - pars['epsilon']*IS[t] - pars['rho2']*IS[t])
        ISD.append(ISD[t] + pars['rho2']*IS[t]  - pars['gamma2']*ISD[t] - pars['epsilon']*ISD[t])

        D.append(D[t] + pars['epsilon']*(IS[t]+ISD[t]))
        R.append(R[t] + pars['gamma1']*(IA[t]+IAD[t]) + pars['gamma2']*(IS[t]+ISD[t]))

        I.append(IAD[t] + ISD[t])
        IDD.append(pars['rho1']*IA[t] + pars['rho2']*IS[t])

    G = []
    G.append(np.average(I[3:9]))
    G.append(np.average(I[12:18]))

    return N, S, E, IA, IAD, IS, ISD, D, R, I, IDD, G
