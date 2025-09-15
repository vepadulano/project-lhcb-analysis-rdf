#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
start_time = time.time()

import datetime
now = datetime.datetime.now()
print(f"Starting execution at: {now.hour}:{now.minute}")
print("Done")


# In[2]:


import uproot
import numpy as np
import matplotlib.pylab as plt
import matplotlib.colors as colors
from scipy.optimize import curve_fit # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
print("Done")


# In[3]:


events_down = uproot.open('B2HHH_MagnetDown.root')
events_up  = uproot.open('B2HHH_MagnetUp.root')
#events_up = uproot.open('500k_B2HHH_MagnetUp.root')
print("Done")


# In[4]:


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
print("Done")


# In[ ]:


import uproot
import numpy as np
import time

start_time = time.time()

events_down = uproot.open('B2HHH_MagnetDown.root')
events_up = uproot.open('B2HHH_MagnetUp.root')
print("Done")
print('Input data variables:')
print(events_up['DecayTree'].keys())

# Arrays to hold the data
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

# Constants
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

# Counters for event statistics
total_events = 0
events_after_basic_cuts = 0
kkk_candidates = 0
kkk_selected = 0
kkk_mass_window = 0
pipipi_candidates = 0
pipipi_selected = 0
pipipi_mass_window = 0
kpi_candidates = 0
kpi_selected = 0
kpi_mass_window = 0

# A counter for bookkeeping
event_counter = 0

# If set to a value greater than 0, limits the number of events analysed
MAX_EVENTS = -1

# Process both files using nested loop structure
for tree in [events_down['DecayTree'], events_up['DecayTree']]:
    for data in tree.iterate(['H1_PX', 'H1_PY', 'H1_PZ','H1_Charge','H1_ProbPi', 'H1_ProbK', 'H1_isMuon', 
                             'H2_PX', 'H2_PY', 'H2_PZ','H2_Charge','H2_ProbPi', 'H2_ProbK', 'H2_isMuon', 
                             'H3_PX', 'H3_PY', 'H3_PZ','H3_Charge','H3_ProbPi', 'H3_ProbK', 'H3_isMuon']):

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

        # Count total events in this batch
        total_events += len(data['H1_PZ'])

        # This loop will go over individual events
        for i in range(0,len(data['H1_PZ'])):
            event_counter += 1
            if 0 < MAX_EVENTS and MAX_EVENTS < event_counter: break
            if 0 == (event_counter % 100000): print('Read', event_counter, 'events')

            # Decide here which events to analyse
            if (data['H1_PZ'][i] < 0) or (data['H2_PZ'][i] < 0) or (data['H3_PZ'][i] < 0): continue
            if data['H1_isMuon'][i] or data['H2_isMuon'][i] or data['H3_isMuon'][i]: continue

            # Count events after basic cuts
            events_after_basic_cuts += 1

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

            # --- KKK Selection ---
            if ((data['H1_ProbK'][i] > probK_all) 
                and (data['H2_ProbK'][i] > probK_all) 
                and (data['H3_ProbK'][i] > probK_all)
                and (data['H1_ProbK'][i] + data['H2_ProbK'][i] + data['H3_ProbK'][i] > probK_sum)):

                kkk_candidates += 1

                # Check if in mass window
                in_mass_window = np.abs(mKKKinv[i]-mB) < mass_cut
                if in_mass_window:
                    kkk_mass_window += 1

                if data['H1_Charge'][i] == data['H2_Charge'][i]:
                    if m13KKsq[i] > m23KKsq[i]:
                        if in_mass_window: 
                            mKKlow.append(m23KKsq[i])
                            mKKhigh.append(m13KKsq[i])
                        if np.abs(np.sqrt(m23KKsq[i])-mD) > mass_cut_D: 
                            m_KKK.append(mKKKinv[i])
                            kkk_selected += 1
                    else:
                        if in_mass_window: 
                            mKKlow.append(m13KKsq[i])
                            mKKhigh.append(m23KKsq[i])
                        if np.abs(np.sqrt(m13KKsq[i])-mD) > mass_cut_D: 
                            m_KKK.append(mKKKinv[i])
                            kkk_selected += 1
                elif data['H1_Charge'][i] == data['H3_Charge'][i]:
                    if m12KKsq[i] > m23KKsq[i]:
                        if in_mass_window: 
                            mKKlow.append(m23KKsq[i])
                            mKKhigh.append(m12KKsq[i])
                        if np.abs(np.sqrt(m23KKsq[i])-mD) > mass_cut_D: 
                            m_KKK.append(mKKKinv[i])
                            kkk_selected += 1
                    else:
                        if in_mass_window: 
                            mKKlow.append(m12KKsq[i])
                            mKKhigh.append(m23KKsq[i])
                        if np.abs(np.sqrt(m12KKsq[i])-mD) > mass_cut_D: 
                            m_KKK.append(mKKKinv[i])
                            kkk_selected += 1
                elif data['H2_Charge'][i] == data['H3_Charge'][i]:
                    if m12KKsq[i] > m13KKsq[i]:
                        if in_mass_window: 
                            mKKlow.append(m13KKsq[i])
                            mKKhigh.append(m12KKsq[i])
                        if np.abs(np.sqrt(m13KKsq[i])-mD) > mass_cut_D: 
                            m_KKK.append(mKKKinv[i])
                            kkk_selected += 1
                    else:
                        if in_mass_window: 
                            mKKlow.append(m12KKsq[i])
                            mKKhigh.append(m13KKsq[i])
                        if np.abs(np.sqrt(m12KKsq[i])-mD) > mass_cut_D: 
                            m_KKK.append(mKKKinv[i])
                            kkk_selected += 1

            # --- πππ Selection ---
            elif ((data['H1_ProbPi'][i] > probpi_all) 
                  and (data['H2_ProbPi'][i] > probpi_all) 
                  and (data['H3_ProbPi'][i] > probpi_all)
                  and (data['H1_ProbK'][i] < probK_max)
                  and (data['H2_ProbK'][i] < probK_max)
                  and (data['H3_ProbK'][i] < probK_max)
                  and (data['H1_ProbPi'][i] + data['H2_ProbPi'][i] + data['H3_ProbPi'][i] > probpi_sum)):

                pipipi_candidates += 1

                # Check if in mass window
                in_mass_window = np.abs(mpipipiinv[i]-mB) < mass_cut
                if in_mass_window:
                    pipipi_mass_window += 1

                if data['H1_Charge'][i] == data['H2_Charge'][i]:
                    if in_mass_window:
                        if m13pipisq[i] > m23pipisq[i]:
                            mpipilow.append(m23pipisq[i])
                            mpipihigh.append(m13pipisq[i])
                        else:
                            mpipilow.append(m13pipisq[i])
                            mpipihigh.append(m23pipisq[i])
                    if (np.abs(np.sqrt(m13pipisq[i])-mD) > mass_cut_D) and (np.abs(np.sqrt(m23pipisq[i])-mD) > mass_cut_D): 
                        m_pipipi.append(mpipipiinv[i])
                        pipipi_selected += 1
                elif data['H1_Charge'][i] == data['H3_Charge'][i]:
                    if in_mass_window:
                        if m12pipisq[i] > m23pipisq[i]:
                            mpipilow.append(m23pipisq[i])
                            mpipihigh.append(m12pipisq[i])
                        else:
                            mpipilow.append(m12pipisq[i])
                            mpipihigh.append(m23pipisq[i])
                    if (np.abs(np.sqrt(m12pipisq[i])-mD) > mass_cut_D) and (np.abs(np.sqrt(m23pipisq[i])-mD) > mass_cut_D): 
                        m_pipipi.append(mpipipiinv[i])
                        pipipi_selected += 1
                elif data['H2_Charge'][i] == data['H3_Charge'][i]:
                    if in_mass_window:
                        if m12pipisq[i] > m13pipisq[i]:
                            mpipilow.append(m13pipisq[i])
                            mpipihigh.append(m12pipisq[i])
                        else:
                            mpipilow.append(m12pipisq[i])
                            mpipihigh.append(m13pipisq[i])
                    if (np.abs(np.sqrt(m13pipisq[i])-mD) > mass_cut_D) and (np.abs(np.sqrt(m12pipisq[i])-mD) > mass_cut_D): 
                        m_pipipi.append(mpipipiinv[i])
                        pipipi_selected += 1

            # --- Kππ Selection ---
            elif ((data['H1_ProbK'][i] > data['H2_ProbK'][i]) and (data['H1_ProbK'][i] > data['H3_ProbK'][i])
                  and (data['H2_Charge'][i] != data['H3_Charge'][i])
                  and (data['H1_ProbK'][i] > probK_high)
                  and (data['H2_ProbPi'][i] > probpi_all) 
                  and (data['H3_ProbPi'][i] > probpi_all)
                  and (data['H2_ProbPi'][i] + data['H3_ProbPi'][i] > probpi_sum2)):

                kpi_candidates += 1

                # Check if in mass window
                in_mass_window = np.abs(mKpipiinv[i]-mB) < mass_cut
                if in_mass_window:
                    kpi_mass_window += 1

                if (data['H1_Charge'][i] != data['H2_Charge'][i]):
                    if in_mass_window: 
                        mKpi.append(m12Kpisq[i])
                        mpipi.append(m23pipisq[i])
                    if np.abs(np.sqrt(m12Kpisq[i])-mD) > mass_cut_D: 
                        m_Kpipi.append(mKpipiinv[i])
                        kpi_selected += 1
                else:
                    if in_mass_window: 
                        mKpi.append(m13Kpisq[i])
                        mpipi.append(m23pipisq[i])
                    if np.abs(np.sqrt(m13Kpisq[i])-mD) > mass_cut_D: 
                        m_Kpipi.append(mKpipiinv[i])
                        kpi_selected += 1

            elif ((data['H2_ProbK'][i] > data['H1_ProbK'][i]) and (data['H2_ProbK'][i] > data['H3_ProbK'][i])
                  and (data['H1_Charge'][i] != data['H3_Charge'][i])
                  and (data['H2_ProbK'][i] > probK_high)
                  and (data['H1_ProbPi'][i] > probpi_all) 
                  and (data['H3_ProbPi'][i] > probpi_all)
                  and (data['H1_ProbPi'][i] + data['H3_ProbPi'][i] > probpi_sum2)):

                kpi_candidates += 1

                # Check if in mass window
                in_mass_window = np.abs(mpiKpiinv[i]-mB) < mass_cut
                if in_mass_window:
                    kpi_mass_window += 1

                if (data['H2_Charge'][i] != data['H1_Charge'][i]):
                    if in_mass_window: 
                        mKpi.append(m12piKsq[i])
                        mpipi.append(m13pipisq[i])
                    if np.abs(np.sqrt(m12piKsq[i])-mD) > mass_cut_D: 
                        m_Kpipi.append(mpiKpiinv[i])
                        kpi_selected += 1
                else:
                    if in_mass_window: 
                        mKpi.append(m23Kpisq[i])
                        mpipi.append(m13pipisq[i])
                    if np.abs(np.sqrt(m23Kpisq[i])-mD) > mass_cut_D: 
                        m_Kpipi.append(mpiKpiinv[i])
                        kpi_selected += 1

            elif ((data['H3_ProbK'][i] > data['H2_ProbK'][i]) and (data['H3_ProbK'][i] > data['H1_ProbK'][i])
                  and (data['H1_Charge'][i] != data['H2_Charge'][i])
                  and (data['H3_ProbK'][i] > probK_high)
                  and (data['H2_ProbPi'][i] > probpi_all) 
                  and (data['H1_ProbPi'][i] > probpi_all)
                  and (data['H2_ProbPi'][i] + data['H1_ProbPi'][i] > probpi_sum2)):

                kpi_candidates += 1

                # Check if in mass window
                in_mass_window = np.abs(mpipiKinv[i]-mB) < mass_cut
                if in_mass_window:
                    kpi_mass_window += 1

                if (data['H3_Charge'][i] != data['H1_Charge'][i]):
                    if in_mass_window: 
                        mKpi.append(m13piKsq[i])
                        mpipi.append(m12pipisq[i])
                    if np.abs(np.sqrt(m13piKsq[i])-mD) > mass_cut_D: 
                        m_Kpipi.append(mpipiKinv[i])
                        kpi_selected += 1
                else:
                    if in_mass_window: 
                        mKpi.append(m23piKsq[i])
                        mpipi.append(m12pipisq[i])
                    if np.abs(np.sqrt(m23piKsq[i])-mD) > mass_cut_D: 
                        m_Kpipi.append(mpipiKinv[i])
                        kpi_selected += 1

print('Read {:d} events'.format(event_counter))

# --- Final Results ---
print(f"\nEvent selection completed. Processed {total_events} events in total.")
print(f"Events after basic cuts: {events_after_basic_cuts}")
print(f"KKK candidates: {kkk_candidates}, selected: {kkk_selected}")
print(f"πππ candidates: {pipipi_candidates}, selected: {pipipi_selected}")
print(f"Kππ candidates: {kpi_candidates}, selected: {kpi_selected}")

# --- Calculate selection efficiencies ---
print("\n=== DETAILED SELECTION REPORT ===")
if kkk_candidates > 0:
    print(f"KKK selection efficiency: {kkk_selected/kkk_candidates*100:.2f}% ({kkk_selected}/{kkk_candidates})")
if pipipi_candidates > 0:
    print(f"πππ selection efficiency: {pipipi_selected/pipipi_candidates*100:.2f}% ({pipipi_selected}/{pipipi_candidates})")
if kpi_candidates > 0:
    print(f"Kππ selection efficiency: {kpi_selected/kpi_candidates*100:.2f}% ({kpi_selected}/{kpi_candidates})")

# Calculate and print elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time}")
print("Done")


# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

def print_histogram_details_numpy(values, bins, name, max_entries=20):
    """Print detailed information about numpy histogram contents"""
    print(f"\n{'='*60}")
    print(f"HISTOGRAM DETAILS: {name}")
    print(f"{'='*60}")

    # Calculate histogram statistics
    total_entries = np.sum(values)
    bin_width = bins[1] - bins[0]
    x_min, x_max = bins[0], bins[-1]

    # Calculate bin centers
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Calculate mean and RMS from histogram data
    if total_entries > 0:
        mean = np.average(bin_centers, weights=values)
        variance = np.average((bin_centers - mean)**2, weights=values)
        rms = np.sqrt(variance)
    else:
        mean = 0
        rms = 0

    print(f"Total entries: {total_entries:.0f}")
    print(f"Number of bins: {len(values)}")
    print(f"X-axis range: [{x_min:.2f}, {x_max:.2f}]")
    print(f"Bin width: {bin_width:.2f}")
    print(f"Mean: {mean:.2f}")
    print(f"RMS: {rms:.2f}")

    # Find non-empty bins
    non_empty_indices = np.where(values > 0)[0]
    non_empty_bins = []

    for i in non_empty_indices:
        bin_center = bin_centers[i]
        bin_low = bins[i]
        bin_high = bins[i+1]
        content = values[i]
        non_empty_bins.append((i+1, bin_center, bin_low, bin_high, content))

    print(f"Non-empty bins: {len(non_empty_bins)}")
    print(f"\nShowing first {min(max_entries, len(non_empty_bins))} non-empty bins:")
    print(f"{'Bin#':<6} {'Center':<12} {'Low Edge':<12} {'High Edge':<12} {'Content':<10}")
    print("-" * 62)

    for i, (bin_num, center, low, high, content) in enumerate(non_empty_bins[:max_entries]):
        print(f"{bin_num:<6} {center:<12.2f} {low:<12.2f} {high:<12.2f} {content:<10.1f}")

    if len(non_empty_bins) > max_entries:
        print(f"... and {len(non_empty_bins) - max_entries} more non-empty bins")

def print_2d_histogram_details_numpy(hist_data, x_edges, y_edges, name, max_entries=20):
    """Print detailed information about 2D numpy histogram contents"""
    print(f"\n{'='*60}")
    print(f"2D HISTOGRAM DETAILS: {name}")
    print(f"{'='*60}")

    # Calculate histogram statistics
    total_entries = np.sum(hist_data)
    x_bin_width = x_edges[1] - x_edges[0]
    y_bin_width = y_edges[1] - y_edges[0]
    x_min, x_max = x_edges[0], x_edges[-1]
    y_min, y_max = y_edges[0], y_edges[-1]

    print(f"Total entries: {total_entries:.0f}")
    print(f"X bins: {len(x_edges)-1}, Y bins: {len(y_edges)-1}")
    print(f"X-axis range: [{x_min:.3f}, {x_max:.3f}]")
    print(f"Y-axis range: [{y_min:.3f}, {y_max:.3f}]")
    print(f"X bin width: {x_bin_width:.3f}")
    print(f"Y bin width: {y_bin_width:.3f}")

    # Find non-empty bins
    non_empty_bins = []
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    for i in range(len(x_centers)):
        for j in range(len(y_centers)):
            content = hist_data[i, j]
            if content > 0:
                x_center = x_centers[i]
                y_center = y_centers[j]
                non_empty_bins.append((i+1, j+1, x_center, y_center, content))

    print(f"Non-empty bins: {len(non_empty_bins)}")
    print(f"\nShowing first {min(max_entries, len(non_empty_bins))} non-empty bins:")
    print(f"{'X Bin':<6} {'Y Bin':<6} {'X Center':<10} {'Y Center':<10} {'Content':<10}")
    print("-" * 52)

    for i, (x_bin, y_bin, x_center, y_center, content) in enumerate(non_empty_bins[:max_entries]):
        print(f"{x_bin:<6} {y_bin:<6} {x_center:<10.3f} {y_center:<10.3f} {content:<10.1f}")

    if len(non_empty_bins) > max_entries:
        print(f"... and {len(non_empty_bins) - max_entries} more non-empty bins")

# Plot some histograms -- NOTE: this can take several minutes
# This line produces two plots side-by-side. 
# It's still useful just for one plot as you can set the size and get the fig,ax objects
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
fig.subplots_adjust(wspace=0.3) # increase horizontal space between plots

# This plots two 1D-histograms.
# The color is changed automatically, the styles are set by hand
# keep hold of the pT histogram data for fitting later
print('Plotting 1D histogram')
values_pT, bins_pT, patches_pT = ax[0].hist(pT, bins=200, range=[0, 100000], histtype='step', label='$p_{T}$')
values_pZ, bins_pZ, patches_pZ = ax[0].hist(pZ, bins=200, range=[0, 100000], histtype='stepfilled', alpha=0.3, label='$p_{z}$')

# Print detailed information for pT and pZ histograms
print_histogram_details_numpy(values_pT, bins_pT, "pT Distribution (Combined H1+H2+H3)", max_entries=30)
print_histogram_details_numpy(values_pZ, bins_pZ, "pZ Distribution (Combined H1+H2+H3)", max_entries=30)

ax[0].set_xlabel('Momentum in MeV')
ax[0].set_ylabel('Entries per 100 MeV')
ax[0].legend()
plt.savefig('pTpZ.pdf')

# This plots a 2D-histogram with values converted to GeV and with a logarithmic colour scale
print('Plotting 2D histogram')
h2d = ax[1].hist2d(probK, probPi, bins=[50, 50], range=[[0, 1], [0, 1]], norm=colors.LogNorm())

# Extract 2D histogram data for detailed analysis
hist_2d_data, x_edges_2d, y_edges_2d, _ = h2d
print_2d_histogram_details_numpy(hist_2d_data, x_edges_2d, y_edges_2d, "probK vs probPi (Combined H1+H2+H3)", max_entries=50)

ax[1].set_xlabel('probK')
ax[1].set_ylabel('probPi')
fig.colorbar(h2d[3], ax=ax[1]) # let's add the colour scale
plt.savefig('probKpi.pdf')

# Triple mass distributions
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
fig.subplots_adjust(wspace=0.3) # increase horizontal space between plots

values_mKKK, bins_mKKK, patches_mKKK = ax[0].hist(m_KKK, bins=300, range=[4500, 6000], histtype='step', label='$m_{KKK}$')
values_mpipipi, bins_mpipipi, patches_mpipipi = ax[1].hist(m_pipipi, bins=150, range=[4500, 6000], histtype='step', label='$m_{\pi\pi\pi}$')
values_mKpipi, bins_mKpipi, patches_mKpipi = ax[2].hist(m_Kpipi, bins=300, range=[4500, 6000], histtype='step', label='$m_{K\pi\pi}$')

# Print detailed information for triple mass histograms
print_histogram_details_numpy(values_mKKK, bins_mKKK, "KKK Mass Distribution", max_entries=25)
print_histogram_details_numpy(values_mpipipi, bins_mpipipi, "πππ Mass Distribution", max_entries=25)
print_histogram_details_numpy(values_mKpipi, bins_mKpipi, "Kππ Mass Distribution", max_entries=25)

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

# Pair mass distributions
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
fig.subplots_adjust(wspace=0.3) # increase horizontal space between plots
#####################################################################################################3
values_mKKlow, bins_mKKlow, patches_mKKlow = ax[0].hist(np.sqrt(mKKlow), bins=500, range=[200, 5200], histtype='step', label='$m_{KK,low}$')
values_mKKhigh, bins_mKKhigh, patches_mKKhigh = ax[0].hist(np.sqrt(mKKhigh), bins=500, range=[200, 5200], histtype='step', label='$m_{KK,high}$')

values_mpipilow, bins_mpipilow, patches_mpipilow = ax[1].hist(np.sqrt(mpipilow), bins=500, range=[200, 5200], histtype='step', label='$m_{\pi\pi,low}$')
values_mpipihigh, bins_mpipihigh, patches_mpipihigh = ax[1].hist(np.sqrt(mpipihigh), bins=500, range=[200, 5200], histtype='step', label='$m_{\pi\pi,high}$')

values_mKpi, bins_mKpi, patches_mKpi = ax[2].hist(np.sqrt(mKpi), bins=500, range=[200, 5200], histtype='step', label='$m_{K\pi}$')
values_mpipi, bins_mpipi, patches_mpipi = ax[2].hist(np.sqrt(mpipi), bins=500, range=[200, 5200], histtype='step', label='$m_{\pi\pi}$')
##########################################################################################################
# Print detailed information for pair mass histograms
print_histogram_details_numpy(values_mKKlow, bins_mKKlow, "KK Low Mass Distribution", max_entries=25)
print_histogram_details_numpy(values_mKKhigh, bins_mKKhigh, "KK High Mass Distribution", max_entries=25)
print_histogram_details_numpy(values_mpipilow, bins_mpipilow, "ππ Low Mass Distribution", max_entries=25)
print_histogram_details_numpy(values_mpipihigh, bins_mpipihigh, "ππ High Mass Distribution", max_entries=25)
print_histogram_details_numpy(values_mKpi, bins_mKpi, "Kπ Mass Distribution", max_entries=25)
print_histogram_details_numpy(values_mpipi, bins_mpipi, "ππ Mass Distribution (in Kππ)", max_entries=25)

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

print("All histogram analysis completed with detailed information!")
print("Done")


# In[ ]:


snb = values_mpipipi[78]
b=(values_mpipipi[69]+values_mpipipi[87])/2
s = snb-b
significance = s/np.sqrt(snb)

print('Kmax',probK_max,'piAll',probpi_all,'piSum',probpi_sum,'S',s,'B',b,'S/B',s/b,'sig',significance)
print("Done")


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
print("Done")


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

print("Done")


# In[ ]:


end_time = time.time()
elapsed = end_time - start_time
minutes = int(elapsed // 60)
seconds = int(elapsed % 60)
print(f"Total execution time: {minutes} min {seconds} sec")
print("Done")


# In[ ]:




