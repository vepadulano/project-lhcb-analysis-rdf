#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
start_time = time.time()

import datetime
now = datetime.datetime.now()
print(f"Starting execution at: {now.hour}:{now.minute}")

# In[2]:


import uproot
import numpy as np
import matplotlib.pylab as plt
import matplotlib.colors as colors
from scipy.optimize import curve_fit # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html


# In[3]:


events_down = uproot.open('B2HHH_MagnetDown.root')
events_up  = uproot.open('B2HHH_MagnetUp.root')
#events_up = uproot.open('500k_B2HHH_MagnetUp.root')


# In[4]:


# Check what's in the tree. 
# Note that the simulation tree is called 'PhaseSpaceTree' and does not have the ProbPi/K variables filled.
print('Input data variables:')
print(events_up['DecayTree'].keys())

# These are the arrays to hold the data
pT = []
pX = []
pY = []
pZ = []
m_pipipi = []
m_KKK = []
m_Kpipi = []
mKKlow = []
mKKhigh = []
mKpi = []
mpipi = []
mpipilow = []
mpipihigh = []
probK = []
probPi = []

mpi = 140.
mK = 494.
mD = 1865.
mJp = 3097.
mp2S = 3686.
mB = 5280.
probK_all = 0.5
probK_high = 0.8
probK_sum = 2.1
probK_max = 0.7
probpi_all = 0.6
probpi_sum2 = 1.3
probpi_sum = 2.4
mass_cut = 50.
mass_cut_D = 20.


# In[ ]:


# A counter for bookkeeping
event_counter = 0

# If set to a value greater than 0, limits the number of events analysed
MAX_EVENTS = -1

# Process only the events_up file
for data in events_up['DecayTree'].iterate(['H1_PX', 'H1_PY', 'H1_PZ','H1_Charge','H1_ProbPi', 'H1_ProbK', 'H1_isMuon', 'H2_PX', 'H2_PY', 'H2_PZ','H2_Charge','H2_ProbPi', 'H2_ProbK', 'H2_isMuon', 'H3_PX', 'H3_PY', 'H3_PZ','H3_Charge','H3_ProbPi', 'H3_ProbK', 'H3_isMuon']):
    # As Python can handle calculations with arrays, we can calculate derived quantities here
    pT_H1 = np.sqrt(data['H1_PX']**2+data['H1_PY']**2)
    pT_H2 = np.sqrt(data['H2_PX']**2+data['H2_PY']**2)
    pT_H3 = np.sqrt(data['H3_PX']**2+data['H3_PY']**2)
    e_pi1 = np.sqrt(data['H1_PX']**2+data['H1_PY']**2+data['H1_PZ']**2+mpi**2)
    e_pi2 = np.sqrt(data['H2_PX']**2+data['H2_PY']**2+data['H2_PZ']**2+mpi**2)
    e_pi3 = np.sqrt(data['H3_PX']**2+data['H3_PY']**2+data['H3_PZ']**2+mpi**2)
    e_K1 = np.sqrt(data['H1_PX']**2+data['H1_PY']**2+data['H1_PZ']**2+mK**2)
    e_K2 = np.sqrt(data['H2_PX']**2+data['H2_PY']**2+data['H2_PZ']**2+mK**2)
    e_K3 = np.sqrt(data['H3_PX']**2+data['H3_PY']**2+data['H3_PZ']**2+mK**2)
    p12sq = (data['H1_PX']+data['H2_PX'])**2 + (data['H1_PY']+data['H2_PY'])**2 + (data['H1_PZ']+data['H2_PZ'])**2
    p23sq = (data['H2_PX']+data['H3_PX'])**2 + (data['H2_PY']+data['H3_PY'])**2 + (data['H2_PZ']+data['H3_PZ'])**2
    p13sq = (data['H1_PX']+data['H3_PX'])**2 + (data['H1_PY']+data['H3_PY'])**2 + (data['H1_PZ']+data['H3_PZ'])**2

    m12KKsq = (e_K1+e_K2)**2 - p12sq
    m23KKsq = (e_K3+e_K2)**2 - p23sq
    m13KKsq = (e_K1+e_K3)**2 - p13sq
    m12pipisq = (e_pi1+e_pi2)**2 - p12sq
    m23pipisq = (e_pi3+e_pi2)**2 - p23sq
    m13pipisq = (e_pi1+e_pi3)**2 - p13sq
    m12Kpisq = (e_K1+e_pi2)**2 - p12sq
    m23Kpisq = (e_K3+e_pi2)**2 - p23sq
    m13Kpisq = (e_K1+e_pi3)**2 - p13sq
    m12piKsq = (e_pi1+e_K2)**2 - p12sq
    m23piKsq = (e_pi3+e_K2)**2 - p23sq
    m13piKsq = (e_pi1+e_K3)**2 - p13sq

    psq = (data['H1_PX']+data['H2_PX']+data['H3_PX'])**2 + (data['H1_PY']+data['H2_PY']+data['H3_PY'])**2 + (data['H1_PZ']+data['H2_PZ']+data['H3_PZ'])**2

    mKKKinv = np.sqrt((e_K1+e_K2+e_K3)**2 - psq)
    mpipipiinv = np.sqrt((e_pi1+e_pi2+e_pi3)**2 - psq)
    mKpipiinv = np.sqrt((e_K1+e_pi2+e_pi3)**2 - psq)
    mpiKpiinv = np.sqrt((e_pi1+e_K2+e_pi3)**2 - psq)
    mpipiKinv = np.sqrt((e_pi1+e_pi2+e_K3)**2 - psq)

    # This loop will go over individual events
    for i in range(0,len(data['H1_PZ'])):
        event_counter += 1
        if 0 < MAX_EVENTS and MAX_EVENTS < event_counter: break
        if 0 == (event_counter % 100000): print('Read', event_counter, 'events')
        # Decide here which events to analyse
        if (data['H1_PZ'][i] < 0) or (data['H2_PZ'][i] < 0) or (data['H3_PZ'][i] < 0): continue
        if data['H1_isMuon'][i] or data['H2_isMuon'][i] or data['H3_isMuon'][i]: continue
        # Fill arrays of events to be plotted and analysed further below
        # Adding values for all three hadrons to the same variable here
        pT.append(pT_H1[i])
        pT.append(pT_H2[i])
        pT.append(pT_H3[i])
        pX.append(data['H1_PX'][i])
        pX.append(data['H2_PX'][i])
        pX.append(data['H3_PX'][i])
        pY.append(data['H1_PY'][i])
        pY.append(data['H2_PY'][i])
        pY.append(data['H3_PY'][i])
        pZ.append(data['H1_PZ'][i])
        pZ.append(data['H2_PZ'][i])
        pZ.append(data['H3_PZ'][i])
        probK.append(data['H1_ProbK'][i])
        probPi.append(data['H1_ProbPi'][i])
        probK.append(data['H2_ProbK'][i])
        probPi.append(data['H2_ProbPi'][i])
        probK.append(data['H3_ProbK'][i])
        probPi.append(data['H3_ProbPi'][i])

        if ((data['H1_ProbK'][i] > probK_all) 
            and (data['H2_ProbK'][i] > probK_all) 
            and (data['H3_ProbK'][i] > probK_all)
            and (data['H1_ProbK'][i] + data['H2_ProbK'][i] + data['H3_ProbK'][i] > probK_sum)):
            if data['H1_Charge'][i] == data['H2_Charge'][i]:
                if m13KKsq[i] > m23KKsq[i]:
                    if np.abs(mKKKinv[i]-mB) < mass_cut: 
                        mKKlow.append(m23KKsq[i])
                        mKKhigh.append(m13KKsq[i])
                    if np.abs(m23KKsq[i]-mD) > mass_cut_D: 
                        m_KKK.append(mKKKinv[i])
                else:
                    if np.abs(mKKKinv[i]-mB) < mass_cut: 
                        mKKlow.append(m13KKsq[i])
                        mKKhigh.append(m23KKsq[i])
                    if np.abs(m13KKsq[i]-mD) > mass_cut_D: 
                        m_KKK.append(mKKKinv[i])
            elif data['H1_Charge'][i] == data['H3_Charge'][i]:
                if m12KKsq[i] > m23KKsq[i]:
                    if np.abs(mKKKinv[i]-mB) < mass_cut: 
                        mKKlow.append(m23KKsq[i])
                        mKKhigh.append(m12KKsq[i])
                    if np.abs(m23KKsq[i]-mD) > mass_cut_D: 
                        m_KKK.append(mKKKinv[i])
                else:
                    if np.abs(mKKKinv[i]-mB) < mass_cut: 
                        mKKlow.append(m12KKsq[i])
                        mKKhigh.append(m23KKsq[i])
                    if np.abs(m12KKsq[i]-mD) > mass_cut_D: 
                        m_KKK.append(mKKKinv[i])
            elif data['H2_Charge'][i] == data['H3_Charge'][i]:
                if m12KKsq[i] > m13KKsq[i]:
                    if np.abs(mKKKinv[i]-mB) < mass_cut: 
                        mKKlow.append(m13KKsq[i])
                        mKKhigh.append(m12KKsq[i])
                    if np.abs(m13KKsq[i]-mD) > mass_cut_D: 
                        m_KKK.append(mKKKinv[i])
                else:
                    if np.abs(mKKKinv[i]-mB) < mass_cut: 
                        mKKlow.append(m12KKsq[i])
                        mKKhigh.append(m13KKsq[i])
                    if np.abs(m12KKsq[i]-mD) > mass_cut_D: 
                        m_KKK.append(mKKKinv[i])


        elif ((data['H1_ProbPi'][i] > probpi_all) 
              and (data['H2_ProbPi'][i] > probpi_all) 
              and (data['H3_ProbPi'][i] > probpi_all)
              and (data['H1_ProbK'][i] < probK_max)
              and (data['H2_ProbK'][i] < probK_max)
              and (data['H3_ProbK'][i] < probK_max)
              and (data['H1_ProbPi'][i] + data['H2_ProbPi'][i] + data['H3_ProbPi'][i] > probpi_sum)):
            if data['H1_Charge'][i] == data['H2_Charge'][i]:
                if np.abs(mpipipiinv[i]-mB) < mass_cut:
                    if m13pipisq[i] > m23pipisq[i]:
                        mpipilow.append(m23pipisq[i])
                        mpipihigh.append(m13pipisq[i])
                    else:
                        mpipilow.append(m13pipisq[i])
                        mpipihigh.append(m23pipisq[i])
                if (np.abs(m13pipisq[i]-mD) > mass_cut_D) and (np.abs(m23pipisq[i]-mD) > mass_cut_D): 
                    m_pipipi.append(mpipipiinv[i])
            elif data['H1_Charge'][i] == data['H3_Charge'][i]:
                if np.abs(mpipipiinv[i]-mB) < mass_cut:
                    if m12pipisq[i] > m23pipisq[i]:
                        mpipilow.append(m23pipisq[i])
                        mpipihigh.append(m12pipisq[i])
                    else:
                        mpipilow.append(m12pipisq[i])
                        mpipihigh.append(m23pipisq[i])
                if (np.abs(m12pipisq[i]-mD) > mass_cut_D) and (np.abs(m23pipisq[i]-mD) > mass_cut_D): 
                    m_pipipi.append(mpipipiinv[i])
            elif data['H2_Charge'][i] == data['H3_Charge'][i]:
                if np.abs(mpipipiinv[i]-mB) < mass_cut:
                    if m12pipisq[i] > m13pipisq[i]:
                        mpipilow.append(m13pipisq[i])
                        mpipihigh.append(m12pipisq[i])
                    else:
                        mpipilow.append(m12pipisq[i])
                        mpipihigh.append(m13pipisq[i])
                if (np.abs(m13pipisq[i]-mD) > mass_cut_D) and (np.abs(m12pipisq[i]-mD) > mass_cut_D): 
                    m_pipipi.append(mpipipiinv[i])

        elif ((data['H1_ProbK'][i] > data['H2_ProbK'][i]) and (data['H1_ProbK'][i] > data['H3_ProbK'][i])
              and (data['H2_Charge'][i] != data['H3_Charge'][i])
              and (data['H1_ProbK'][i] > probK_high)
              and (data['H2_ProbPi'][i] > probpi_all) 
              and (data['H3_ProbPi'][i] > probpi_all)
              and (data['H2_ProbPi'][i] + data['H3_ProbPi'][i] > probpi_sum2)):
            if (data['H1_Charge'][i] != data['H2_Charge'][i]):
                if np.abs(mKpipiinv[i]-mB) < mass_cut: 
                    mKpi.append(m12Kpisq[i])
                    mpipi.append(m23pipisq[i])
                if np.abs(m12Kpisq[i]-mD) > mass_cut_D: 
                    m_Kpipi.append(mKpipiinv[i])
            else:
                if np.abs(mKpipiinv[i]-mB) < mass_cut: 
                    mKpi.append(m13Kpisq[i])
                    mpipi.append(m23pipisq[i])
                if np.abs(m13Kpisq[i]-mD) > mass_cut_D: 
                    m_Kpipi.append(mKpipiinv[i])

        elif ((data['H2_ProbK'][i] > data['H1_ProbK'][i]) and (data['H2_ProbK'][i] > data['H3_ProbK'][i])
              and (data['H1_Charge'][i] != data['H3_Charge'][i])
              and (data['H2_ProbK'][i] > probK_high)
              and (data['H1_ProbPi'][i] > probpi_all) 
              and (data['H3_ProbPi'][i] > probpi_all)
              and (data['H1_ProbPi'][i] + data['H3_ProbPi'][i] > probpi_sum2)):
            if (data['H2_Charge'][i] != data['H1_Charge'][i]):
                if np.abs(mpiKpiinv[i]-mB) < mass_cut: 
                    mKpi.append(m12piKsq[i])
                    mpipi.append(m13pipisq[i])
                if np.abs(m12piKsq[i]-mD) > mass_cut_D: 
                    m_Kpipi.append(mpiKpiinv[i])
            else:
                if np.abs(mpiKpiinv[i]-mB) < mass_cut: 
                    mKpi.append(m23Kpisq[i])
                    mpipi.append(m13pipisq[i])
                if np.abs(m23Kpisq[i]-mD) > mass_cut_D: 
                    m_Kpipi.append(mpiKpiinv[i])

        elif ((data['H3_ProbK'][i] > data['H2_ProbK'][i]) and (data['H3_ProbK'][i] > data['H1_ProbK'][i])
              and (data['H1_Charge'][i] != data['H2_Charge'][i])
              and (data['H3_ProbK'][i] > probK_high)
              and (data['H2_ProbPi'][i] > probpi_all) 
              and (data['H1_ProbPi'][i] > probpi_all)
              and (data['H2_ProbPi'][i] + data['H1_ProbPi'][i] > probpi_sum2)):
            if (data['H3_Charge'][i] != data['H1_Charge'][i]):
                if np.abs(mpipiKinv[i]-mB) < mass_cut: 
                    mKpi.append(m13piKsq[i])
                    mpipi.append(m12pipisq[i])
                if np.abs(m13piKsq[i]-mD) > mass_cut_D: 
                    m_Kpipi.append(mpipiKinv[i])
            else:
                if np.abs(mpipiKinv[i]-mB) < mass_cut: 
                    mKpi.append(m23piKsq[i])
                    mpipi.append(m12pipisq[i])
                if np.abs(m23piKsq[i]-mD) > mass_cut_D: 
                    m_Kpipi.append(mpipiKinv[i])

print('Read {:d} events'.format(event_counter))


# In[ ]:


# Plot some histograms -- NOTE: this can take several minutes

# This line produces two plots side-by-side. 
# It's still useful just for one plot as you can set the size and get the fig,ax objects
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
fig.subplots_adjust(wspace=0.3) # increase horizontal space between plots

# This plots two 1D-histograms.
# The color is changed automatically, the styles are set by hand
# keep hold of the pT histogram data for fitting later
print('Plotting 1D histogram')
values_pT,bins_pT,patches_pT = ax[0].hist(pT, bins = 200, range = [0, 100000],histtype='step',label='$p_{T}$')
ax[0].hist(pZ, bins = 200, range = [0, 100000],histtype='stepfilled',alpha=0.3,label='$p_{z}$')
ax[0].set_xlabel('Momentum in MeV')
ax[0].set_ylabel('Entries per 100 MeV')
ax[0].legend()
plt.savefig('pTpZ.pdf')

# This plots a 2D-histogram with values converted to GeV and with a logarithmic colour scale
print('Plotting 2D histogram')
#h2d = ax[1].hist2d(np.true_divide(pX,1000), np.divide(pY,1000), bins = [100,100], range = [[-10,10],[-10,10]],norm=colors.LogNorm())
h2d = ax[1].hist2d(probK, probPi, bins = [50,50], range = [[0,1],[0,1]],norm=colors.LogNorm())
ax[1].set_xlabel('probK')
ax[1].set_ylabel('probPi')
fig.colorbar(h2d[3],ax=ax[1]) # let's add the colour scale
plt.savefig('probKpi.pdf')

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
fig.subplots_adjust(wspace=0.3) # increase horizontal space between plots
values_mKKK,bins_mKKK,patches_mKKK = ax[0].hist(m_KKK, bins = 300, range = [4500, 6000],histtype='step',label='$m_{KKK}$')
values_mpipipi,bins_mpipipi,patches_mpipipi = ax[1].hist(m_pipipi, bins = 150, range = [4500, 6000],histtype='step',label='$m_{\pi\pi\pi}$')
values_mKpipi,bins_mKpipi,patches_mKpipi = ax[2].hist(m_Kpipi, bins = 300, range = [4500, 6000],histtype='step',label='$m_{K\pi\pi}$')
ax[0].set_xlabel('Mass in MeV/$c^2$')
ax[0].set_ylabel('Entries per 5 MeV/$c^2$')
ax[0].legend()
plt.savefig('mKKK.pdf')
ax[1].set_xlabel('Mass in MeV/$c^2$')
ax[1].set_ylabel('Entries per 5 MeV/$c^2$')
ax[1].legend()
plt.savefig('mpipipi.pdf')
ax[2].set_xlabel('Mass in MeV/$c^2$')
ax[2].set_ylabel('Entries per 5 MeV/$c^2$')
ax[2].legend()
plt.savefig('mKpipi.pdf')

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
fig.subplots_adjust(wspace=0.3) # increase horizontal space between plots
ax[0].hist(np.sqrt(mKKlow), bins = 500, range = [200, 5200],histtype='step',label='$m_{KK,low}$')
ax[0].hist(np.sqrt(mKKhigh), bins = 500, range = [200, 5200],histtype='step',label='$m_{KK,high}$')

ax[1].hist(np.sqrt(mpipilow), bins = 500, range = [200, 5200],histtype='step',label='$m_{\pi\pi,low}$')
ax[1].hist(np.sqrt(mpipihigh), bins = 500, range = [200, 5200],histtype='step',label='$m_{\pi\pi,high}$')

ax[2].hist(np.sqrt(mKpi), bins = 500, range = [200, 5200],histtype='step',label='$m_{K\pi}$')
ax[2].hist(np.sqrt(mpipi), bins = 500, range = [200, 5200],histtype='step',label='$m_{\pi\pi}$')

ax[0].set_xlabel('Mass in MeV/$c^2$')
ax[0].set_ylabel('Entries per 10 MeV/$c^2$')
ax[0].legend()
plt.savefig('mKK.pdf')
ax[1].set_xlabel('Mass in MeV/$c^2$')
ax[1].set_ylabel('Entries per 10 MeV/$c^2$')
ax[1].legend()
plt.savefig('mpipi.pdf')
ax[2].set_xlabel('Mass in MeV/$c^2$')
ax[2].set_ylabel('Entries per 10 MeV/$c^2$')
ax[2].legend()
plt.savefig('mKpi_pipi.pdf')



# In[ ]:


snb = values_mpipipi[78]
b=(values_mpipipi[69]+values_mpipipi[87])/2
s = snb-b
significance = s/np.sqrt(snb)

print('Kmax',probK_max,'piAll',probpi_all,'piSum',probpi_sum,'S',s,'B',b,'S/B',s/b,'sig',significance)


# In[ ]:


fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
fig.subplots_adjust(wspace=0.3) # increase horizontal space between plots
values_mKKK,bins_mKKK,patches_mKKK = ax[0].hist(m_KKK, bins = 300, range = [5000, 5600],histtype='step',label='$m_{KKK}$')
values_mpipipi,bins_mpipipi,patches_mpipipi = ax[1].hist(m_pipipi, bins = 300, range = [5000, 5600],histtype='step',label='$m_{\pi\pi\pi}$')
values_mKpipi,bins_mKpipi,patches_mKpipi = ax[2].hist(m_Kpipi, bins = 300, range = [5000, 5600],histtype='step',label='$m_{K\pi\pi}$')
ax[0].set_xlabel('Mass in MeV/$c^2$')
ax[0].set_ylabel('Entries per 2 MeV/$c^2$')
ax[0].legend()
plt.savefig('mKKK_narrow.pdf')
ax[1].set_xlabel('Mass in MeV/$c^2$')
ax[1].set_ylabel('Entries per 2 MeV/$c^2$')
ax[1].legend()
plt.savefig('mpipipi_narrow.pdf')
ax[2].set_xlabel('Mass in MeV/$c^2$')
ax[2].set_ylabel('Entries per 2 MeV/$c^2$')
ax[2].legend()
plt.savefig('mKpipi_narrow.pdf')


# In[ ]:


# decaying exponential function
def exponential(x, norm, decay):
    return np.array( norm * np.exp(-(x-5000)/decay) )

# constant function
def constant(x, norm):
    return np.array( norm )

# Gaussian function
def gauss(x, norm, mu, sigma):
    return np.array( norm / np.sqrt(2*np.pi) / sigma * np.exp( -(x-mu)**2 / 2 / sigma**2 ) )


# fit function combining two individual functions
def fit_function(x, norm4, sigma4, normG, muG, sigmaG, normE, decay):
    return np.array( gauss(x, norm4, 5100, sigma4) + gauss(x, normG, muG, sigmaG) + exponential(x, normE, decay) )

def fit_data(bins, values, minX, maxX, p0):
    # determine bin centres
    bin_centres = [(a+b)/2 for a,b in zip(bins[0:-1],bins[1:]) ] # uses simultaneous loop over two arrays

    # reduce range to fit only part of curve
    bin_centres_red = [] 
    values_red = []
    for c,v in zip(bin_centres,values):
        if c < minX or c > maxX: continue
        bin_centres_red.append(c)
        values_red.append(v)

    # execute the fit with starting values 5000 and 10^-4
    coeff_fit,cov_fit = curve_fit(fit_function,bin_centres_red,values_red,p0) # fit
    fit_vals = [fit_function(x,coeff_fit[0],coeff_fit[1],coeff_fit[2],coeff_fit[3],coeff_fit[4],coeff_fit[5],coeff_fit[6]) for x in bin_centres_red]
    chi2parts = np.array( ( np.divide( np.array(values_red) - np.array(fit_vals), np.sqrt( values_red ), 
                                      out = np.array(values_red), where = np.array(values_red) != 0 ) )**2 )
    chi2 = np.sum( chi2parts )
    return coeff_fit,cov_fit, bin_centres, bin_centres_red, chi2, len(chi2parts)

def print_results(coeff,cov,chi2,ndf):
    perr = np.sqrt(np.diag(cov)) # extract errors from covarianve matrix
    # output fit results
    print('Fit results with chi2/ndf', chi2,'/',ndf)
    parcount = 0
    for p,e in zip(coeff,perr):
        parcount += 1
        print('Par {:d}: {:f} +/- {:f}'.format(parcount,p,e))

def plot_results(a,bin_centres,bin_centres_red,values,coeff_fit,fname):
    # plot the data, this time as dots with error bars (sqrt(N) errors)
    a.errorbar(bin_centres,values,yerr=np.sqrt(values),linestyle='',marker='.',
               markerfacecolor='k',markeredgecolor='k',ecolor='k',label='Data')

    # plot the fit: create x values, then calculate the corresponding y values and plot
    x_fit = np.linspace(bin_centres_red[0],bin_centres_red[-1],100)
    y_fit = fit_function(x_fit,coeff_fit[0],coeff_fit[1],coeff_fit[2],coeff_fit[3],coeff_fit[4],coeff_fit[5],coeff_fit[6])
    a.plot(x_fit,y_fit,label='Fit',color='r',zorder=10) # zorder makes sure the fit line is on top

    # plot decoration
    a.legend()
    a.set_xlabel('$m_{inv}$ in MeV')
    a.set_ylabel('Entries per 2 MeV')
    plt.savefig(fname)

coeff_KKK_fit,cov_KKK_fit, bin_centres_KKK, bin_centres_red_KKK, chi2_KKK, ndf_KKK = fit_data( bins_mKKK, values_mKKK, 5125, 5500, [2000,30,30000,5285,20,40,500] )
coeff_pipipi_fit,cov_pipipi_fit, bin_centres_pipipi, bin_centres_red_pipipi, chi2_pipipi, ndf_pipipi = fit_data( bins_mpipipi, values_mpipipi, 5100, 5500, [8000,30,10000,5285,20,1000,500] )
coeff_Kpipi_fit,cov_Kpipi_fit, bin_centres_Kpipi, bin_centres_red_Kpipi, chi2_Kpipi, ndf_Kpipi = fit_data( bins_mKpipi, values_mKpipi, 5100, 5500, [15000,30,60000,5285,20,400,500] )

print_results(coeff_KKK_fit,cov_KKK_fit, chi2_KKK, ndf_KKK)
print_results(coeff_pipipi_fit,cov_pipipi_fit, chi2_pipipi, ndf_pipipi)
print_results(coeff_Kpipi_fit,cov_Kpipi_fit, chi2_Kpipi, ndf_Kpipi)

# plot results
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 6))

plot_results(ax[0],bin_centres_KKK,bin_centres_red_KKK,values_mKKK,coeff_KKK_fit,'mfit_KKK.pdf')
plot_results(ax[1],bin_centres_pipipi,bin_centres_red_pipipi,values_mpipipi,coeff_pipipi_fit,'mfit_pipipi.pdf')
plot_results(ax[2],bin_centres_Kpipi,bin_centres_red_Kpipi,values_mKpipi,coeff_Kpipi_fit,'mfit_Kpipi.pdf')


# In[ ]:


end_time = time.time()
elapsed = end_time - start_time
minutes = int(elapsed // 60)
seconds = int(elapsed % 60)
print(f"Total execution time: {minutes} min {seconds} sec")


# In[ ]:




