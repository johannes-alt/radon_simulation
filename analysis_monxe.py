import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import binom as binomcoeff
from scipy.optimize import curve_fit
from scipy.integrate import quad


def filter_400(data_dict, return_poped_wf = False):
    # remove all waveforms with the flag 0x400 and with an pulse_height_adc = 0
    # if return_poped_wf == True a dict with all removed waveforms will be returned
    
    timesp_list = list(data_dict.keys())
    poped_waveforms = {}
    for i in timesp_list:
        pop_it = False
        if data_dict[i]['waveform'][0] == np.str('0x400') or data_dict[i]['pulse_height_adc'] == 0:
            poped_waveforms[i] = data_dict[i]
            pop_it = True
        elif data_dict[i]['waveform'][0] != np.str('0x0'):
            print('waveform with timestamp ',i, ' has not flag 0x0')
            pop_it = True
        if pop_it:
            data_dict.pop(i)
            
    if return_poped_wf:
        return poped_waveforms
    
def calc_baseline(data_dict):
    # calculating the baseline and adding it in the data dict
    
    timesp_list = list(data_dict.keys())
    
    for i in timesp_list:
        baseline_samples = list(map(int,data_dict[i]['waveform'][1:600]))
        baseline = np.mean(baseline_samples)
        data_dict[i]['baseline'] = baseline

def calc_height(data_dict):
    # claculating the height of the peaks via highest sample to baseline
    
    timesp_list = list(data_dict.keys())
    
    for i in timesp_list:
        max_sample = max(list(map(int,data_dict[i]['waveform'][1:-2])))
        height = max_sample - data_dict[i]['baseline']
        data_dict[i]['height'] = height
        
        
def create_array(data_dict, mode = 'fit_max'):
    # create an array with timestamp and peak heigth (with calc_height) similar to the one obtained if waveforms are not saved
    
    timestamp_data_dtype = np.dtype([
        ("timestamp_ps", np.uint64), # timestamp in ps
        ("pulse_height_adc", np.int16) # max adc channel is ~16000, np.int16 ranges from -32768 to 32767
    ])
    
    data_array = []
    timesp_list = list(data_dict.keys())
    if mode == 'asymptotic':
        for i in timesp_list:
            data_array.append((data_dict[i]['timestamp_ps'], data_dict[i]['fit_3'][0][2]))
    elif mode == 'fit_max':
        for i in timesp_list:
            data_array.append((data_dict[i]['timestamp_ps'], data_dict[i]['fit_max']))
    elif mode == 'wf_max':
        for i in timesp_list:
            data_array.append((data_dict[i]['timestamp_ps'], data_dict[i]['height']))
    return np.array(data_array, timestamp_data_dtype)

def expo(x, h, t, c):
    return h * np.exp(-x/t) + c

def fit_expo(data_dict):
    # fitting an exponential function on the waveforms
    
    timesp_list = list(data_dict.keys())
    
    for i in timesp_list:
        waveform = np.array(list(map(int,data_dict[i]['waveform'][1:-2])))
        max_index = np.where(waveform == np.amax(waveform))[0][0]
        #print(max_index)
        waveform = waveform[max_index:]
        waveform = waveform[1000:10000]
        x = np.linspace(0,len(waveform)-1,len(waveform))
        popt, pcov = curve_fit(expo, x, waveform)
        
        data_dict[i]['fit_parameter'] = (popt, pcov)
        
        #plt.plot(x, waveform)
        #plt.plot(expo(x, popt[0], popt[1], popt[2]))
        #plt.show()
        
def fit_complete(x, x_bsl, baseline, A_rise, lamba_rise, A_dec, lamba_dec):
    # function that is fitted on to the data
    
    x_bsl = int(x_bsl)
    x_ls = x[x_bsl:]
    x_rd = baseline*np.ones(len(x_ls)) + A_rise * (1-np.exp(-lamba_rise*(x_ls-x_bsl))) - A_dec * (1-np.exp(-lamba_dec*(x_ls-x_bsl)))
    const = np.concatenate((baseline * np.ones(x_bsl),np.zeros(len(x_rd))))
    rise_fall = np.concatenate((np.zeros(x_bsl),x_rd))
    return const + rise_fall

def fit(data_dict):
    # fitting
    
    for i in data_dict:
        waveform = np.array(list(map(int,data_dict[i]['waveform'][1:-2])))
        x = np.linspace(0, len(waveform)-1, len(waveform))
        popt, pcov = curve_fit(fit_complete, x, waveform,
                              p0=[645, 4800, data_dict[i]['fit_parameter'][0][0]-500, 1/200,
                                  data_dict[i]['fit_parameter'][0][0]+500,
                                  1/data_dict[i]['fit_parameter'][0][1]])
        data_dict[i]['fit_3'] = (popt, pcov)
        data_dict[i]['fit_max'] = fit_complete(x, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5]) - popt[1]