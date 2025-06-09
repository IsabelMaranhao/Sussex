# -*- coding: utf-8 -*-
"""
Class Trial - contains information about each trial (of length a few seconds) for analysing mouse 2 photon time series
Class Neuron - contains information about each neuron identified in experiments and label applied
@author: Paul Kinghorn
"""




class Trial():

    def __init__(self,num_trial_in_experiment, len_trial, cues,dff,licks,trial_type,mouse_id, date, ROI,trial_outcome):
        super(Trial, self).__init__()
        self.num_trial = num_trial_in_experiment
        self.len_trial = len_trial
        self.cues = cues
        self.dff = dff
        self.licks = licks
        self.trial_type = trial_type
        self.mouse_id = mouse_id
        self.date = date
        self.ROI = ROI
        self.trial_outcome = trial_outcome # hit miss etc
        self.neuron_num= int(str(mouse_id) + str(date)+str(ROI))

class Neuron():
    
    def __init__(self, mouse_id, date, ROI, categorised_time_series,mean_time_series,std_time_series,trial_count,manual_label=1e6,label=1e6,paul_manual_label=1e6):
        super(Neuron, self).__init__()
        self.neuron_num= int(str(mouse_id) + str(date)+str(ROI))
        self.mouse_id= mouse_id
        self.date= date
        self.ROI= ROI
        self.categorised_time_series= categorised_time_series #4 arrays of the underlying time_series for "Hit", "Miss", "CR", "FA" 
        self.mean_time_series= mean_time_series #4 mean time_series arrays for "Hit", "Miss", "CR", "FA" 
        self.std_time_series= std_time_series #4 std time_series arrays for "Hit", "Miss", "CR", "FA" 
        self.trial_count= trial_count # array of 4 numbers - number of trials used to create means for "Hit", "Miss", "CR", "FA" 
        self.manual_label=manual_label # defaults to 1e6 if not provided
        self.paul_manual_label=paul_manual_label # defaults to 1e6 if not provided
        self.label=label # defaults to 1e6 if not provided

 