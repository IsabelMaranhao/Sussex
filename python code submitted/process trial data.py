# -*- coding: utf-8 -*-
"""
Code to bring in 2 photon time series data and categorise
"""
import numpy as np
import os
import scipy.io as sio
from dataClasses import Trial,Neuron


'''
Script to bring in time series data
Data is retrieved from a series of .MAT files - they need to cotnain dictionary in correct format and have corect naming convention
Hardcoded aspects:
    - file naming convention
    - cues and go / nogo signals must coincide in time series data.  post is at variable "cue_location".  
'''

def import_trial_data(source_dir):   
    cue_location=2 # eg if=2 then cue is at 3rd time series (0,1,2,....])
    experiments=[] #list of of all the experiments - made up of many trials (it is all the data in one file)
    trials=[] #list of all the trials - each list item is a trial object    
    for file in os.listdir(source_dir) :
        print("processing file "+file)
        # bring in each file
        mat=sio.loadmat(str(source_dir)+"/" + str(file))
               
        ### split into trials and create an object for each trial       
        # whenever cue=1 is a new trial
        trial_starts=np.where(mat['cues']==1)[1]-cue_location
        trial_starts=np.append(trial_starts,mat['cues'].shape[1])
        mouse_id=file[0:4]
        date=file[5:13]
        ROI_index= int(file.find("_timeseries",0,-1))
        if ROI_index==16:
            ROI=file[ROI_index-2:ROI_index]
        elif ROI_index==15:
            ROI=file[ROI_index-1:ROI_index]
        elif ROI_index==17:
            ROI=file[ROI_index-3:ROI_index]
        else:
            print(file + " has wrong naming convention")
        experiments.append([int(mouse_id),int(date),int(ROI)])    

        ###
        top_and_tail=1 # number of trials to remove at begninng and end of experiment. set this to zero if not to happen
        for i in range(top_and_tail,len(trial_starts)-top_and_tail-1):  
 
            cues=mat['cues'][0,trial_starts[i]:trial_starts[i+1]]            
            dff=mat['dff'][0,trial_starts[i]:trial_starts[i+1]]                  
            licks=mat['licks'][0,trial_starts[i]:trial_starts[i+1]]
            punish=mat['punish'][0,trial_starts[i]:trial_starts[i+1]]
            reward=mat['reward'][0,trial_starts[i]:trial_starts[i+1]]
            ##### categorise Go / NOGO           
            if mat['go'][0,trial_starts[i]+cue_location]==1:
                trial_type="go"
            elif mat['nogo'][0,trial_starts[i]+cue_location]==1:
                trial_type="nogo"
            else:
                print(i, "PROBLEM in trial object creation - neither go nor nogo" )
            ###### cateorise H/M/FA/CR
            if np.any(punish==1):
                trial_outcome="FA"
            elif np.any(reward==1):
                trial_outcome="Hit"
            elif trial_type=="go":  # neither reward nor punish label given. So if is GO then is a Miss, if nogo then is Correct Rejection
                trial_outcome="Miss"
            elif trial_type=="nogo":
                trial_outcome="CR"
            else:
                trial_outcome="Unknown"
            my_trial=Trial(i,len(cues),cues,dff,licks,trial_type,mouse_id, date, ROI,trial_outcome) #create the trial object            
            trials.append(my_trial) # add the trial object to a list of all trials
    return experiments, trials           

def setup_trial_arrays(trials):
    ###### getting mean dff beahviour relative to start of trial
    # create criteria for filtering
    print("Total Number of trials: " , len(trials))
    print("Trial lengths: " , len(trials[0].dff))
    my_dff=np.empty(shape=(len(trials[0].dff),len(trials)),dtype=float)
    my_dff_MA=np.empty(shape=(len(trials[0].dff),len(trials)),dtype=float) # moving average of last N timesteps
    my_licks=np.empty(shape=(len(trials[0].licks),len(trials)),dtype=float)  #### new for licks data
    my_mouse_id=np.empty(shape=(len(trials)),dtype=int)
    my_date=np.empty(shape=(len(trials)),dtype=int)
    my_ROI=np.empty(shape=(len(trials)),dtype=int)
    my_go_nogo=np.empty(shape=(len(trials)),dtype=bool)
    my_num_trial=np.empty(shape=(len(trials)),dtype=int)
    my_trial_outcome=np.empty(shape=(len(trials)),dtype=np.dtype('U8'))

    for i in range(len(trials)):   
        my_dff[:,i]=trials[i].dff
        ############# create MA of dff
        N=5 ### cant change this due to next line being hardcoded
        my_dff_MA[4:my_dff.shape[0],i]=1/N*(my_dff[0:my_dff.shape[0]-N+1,i]+my_dff[1:my_dff.shape[0]-N+2,i]+my_dff[2:my_dff.shape[0]-N+3,i]+my_dff[3:my_dff.shape[0]-N+4,i]+my_dff[4:my_dff.shape[0]-N+5,i])
        my_dff_MA[2:my_dff.shape[0]-2,i]=my_dff_MA[4:my_dff.shape[0],i]
        #just make first 2 and last flat.  this will empahsise them but better than leaving unsmoothed.
        my_dff_MA[0:2,i]=my_dff_MA[2,i]
        my_dff_MA[my_dff.shape[0]-2:my_dff.shape[0],i]=my_dff_MA[my_dff.shape[0]-3,i]
        ##############
        my_licks[:,i]=trials[i].licks
        my_mouse_id[i]=trials[i].mouse_id
        my_date[i]=trials[i].date
        my_ROI[i]=trials[i].ROI
        my_num_trial[i]=trials[i].num_trial
        if trials[i].trial_type=="go":        
            my_go_nogo[i]=True
        elif trials[i].trial_type=="nogo":        
           my_go_nogo[i]=False 
        else:
           print("PROBLEM in reading in trial objects - Neither go nor nogo" ) 
        my_trial_outcome[i]=trials[i].trial_outcome

    return my_dff,my_dff_MA,my_mouse_id,my_date,my_ROI,my_go_nogo,my_num_trial,my_trial_outcome, my_licks


def save_all_neurons(experiments, my_dff, my_licks):
    '''
    code to take all the trials for a neuron and save the mean and std of dff data for each of the 4 category types
    '''
    all_means=[]
    all_stes=[]
    all_neurons=[];
    for i in range(len(experiments)): 

        print(experiments[i])
        mask=(my_mouse_id==experiments[i][0]) 
        mask=mask & (my_date==experiments[i][1])
        mask=mask & (my_ROI==experiments[i][2]) 
        #mask=mask & (my_go_nogo==True)
        #mask=mask & (my_num_trial==0) # individual spike monitor - so only 1 trial.
        filtered_dff=my_dff[:,mask]
        num_trials=filtered_dff.shape[1]
        print(i, "Total filtered number of trials ", num_trials)
        
        #time_series=0
        means=[]
        categorised_time_series=[]
        stds=[]
        stes=[]
        trial_count=np.zeros(6)
        j=0
        # create the mean and std dff for each category (all dff data is relative to stimulus)
        for category in ["Hit", "Miss", "CR", "FA"]:

            mask=(my_mouse_id==experiments[i][0]) 
            mask=mask & (my_date==experiments[i][1])
            mask=mask & (my_ROI==experiments[i][2]) 
            #mask=mask & (my_go_nogo==True)
            #mask=mask & (my_num_trial==0) # individual spike monitor - so only 1 trial.
            mask= mask & (my_trial_outcome==category) # choices are Hit Miss FA CR 
            filtered_dff=my_dff[:,mask]
                
            num_trials=filtered_dff.shape[1]
            print("Filtered number of trials ", category, num_trials)
            mean_dff=np.mean(filtered_dff,axis=1)
            std_dff=np.std(filtered_dff,axis=1)
            temp_num_trials=np.maximum(num_trials,1)
            ste_dff=np.std(filtered_dff,axis=1)/np.sqrt(temp_num_trials)

            ###################################
            # rebase all the means so that first timestep starts on zero
            mean_dff=mean_dff-mean_dff[0]
            # if there are no instances in that category then just give mean of zero
            if num_trials==0:
                mean_dff[:]=0
                std_dff[:]=0
                ste_dff[:]=0            
            means.append(mean_dff) # mean_dff is a time series showing mean for that category for that experiment
            categorised_time_series.append(filtered_dff) # filtered_dff is all the underlying time_series for the category
            stds.append(std_dff)
            stes.append(ste_dff)
            trial_count[j]=num_trials
            j+=1
            #time_series+=1
        
        # create the mean and std dff for Hit and FP category relative to LICK
        # also the trial count for each (will be the same as the "nromal Hit and FP")
        for category in ["Hit", "FA"]:
            mask=(my_mouse_id==experiments[i][0]) 
            mask=mask & (my_date==experiments[i][1])
            mask=mask & (my_ROI==experiments[i][2]) 
            mask= mask & (my_trial_outcome==category) # choices are Hit Miss FA CR 
            filtered_dff=my_dff[:,mask]
            filtered_licks=my_licks[:,mask]
                
            num_trials=filtered_dff.shape[1]
            lick_starts=np.argmax(filtered_licks==1,axis=0)

            # create, instead of filtered_dff and mean_dff, filtered_dff_rel_licks and mean_dff_rel_licks
            # for each trial remove timseries of dff before first lick - so time series of dff_rel_licks starts at time of first lick
            # so these will noe be shorter than full time series, but all adjusted to start relative to lick time
            # we will only save the data after licking starts and then take mean and std
            # need to mkae them all the same length, so work out shortest time series for all of neuron's trials and just keep that length of data as an array
            filtered_dff_rel_licks=[]            
            for temp_trial_count in range(filtered_dff.shape[1]):
                trial_filtered_dff_rel_licks=np.zeros(filtered_dff.shape[0])
                # when did mouse first lick
                lick_start_idx=lick_starts[temp_trial_count]
                # we want to rebase everything relative to first lick minus 5 timesteps
                new_start_idx=lick_start_idx-4 # index 4 is 5th index
                if new_start_idx>=0:                    
                    # move the array of licks left so that timeseries starst at lick time
                    trial_filtered_dff_rel_licks[0:filtered_dff.shape[0]-new_start_idx]=filtered_dff[new_start_idx: ,temp_trial_count]
                    ## need to mske the last few timesteps on the new series = final timestep on original series
                    trial_filtered_dff_rel_licks[filtered_dff.shape[0]-new_start_idx :]=filtered_dff[-1,temp_trial_count]
                else: 
                    # move the array of licks right so that timeseries starst at lick time
                    trial_filtered_dff_rel_licks[0-new_start_idx :]=filtered_dff[0:filtered_dff.shape[0]+new_start_idx ,temp_trial_count]
                    ## need to mske the first few timesteps on the new series = first timestep on original series
                    trial_filtered_dff_rel_licks[0:-new_start_idx]=filtered_dff[0 ,temp_trial_count]
                
                filtered_dff_rel_licks.append(trial_filtered_dff_rel_licks)
            filtered_dff_rel_licks=np.asarray(filtered_dff_rel_licks).transpose()
            
            print("Filtered number of trials ", category, "relative to Lick", num_trials)
            ### if there are no instances of hit or FA the need to just force mean and std to zero for each timeslot
            if filtered_dff_rel_licks.shape[0]==0:
                mean_dff=np.zeros(filtered_dff.shape[0])
                std_dff=np.zeros(filtered_dff.shape[0])
                ste_dff=np.zeros(filtered_dff.shape[0])
            else:    
                mean_dff=np.mean(filtered_dff_rel_licks,axis=1)
                std_dff=np.std(filtered_dff_rel_licks,axis=1)
                ste_dff=np.std(filtered_dff_rel_licks,axis=1)/np.sqrt(num_trials)
            ###################################
            # rebase all the means so that first timestep starts on zero
            mean_dff=mean_dff-mean_dff[0]
            # if there are no instances in that category then just give mean of zero
            if num_trials==0:
                mean_dff[:]=0
                std_dff[:]=0
                ste_dff[:]=0            
            means.append(mean_dff) # mean_dff is a time series showing mean for that category for that experiment
            categorised_time_series.append(filtered_dff_rel_licks) # filtered_dff is all the underlying time_series for the category
            stds.append(std_dff)
            stes.append(ste_dff)
            trial_count[j]=num_trials
            j+=1
        
        all_means.append(means) # means is list of lenght 4 containing mean time series for each category
        
        all_stes.append(stes)
        my_neuron=Neuron(mouse_id=experiments[i][0],date=experiments[i][1],ROI=experiments[i][2],categorised_time_series=categorised_time_series,mean_time_series=means,std_time_series=stds, trial_count=trial_count) #create the neuron object            
        all_neurons.append(my_neuron) # add the trial object to a list of all trials
    all_means=np.array(all_means) # eg all_means is 289*4*41
    all_stes=np.array(all_stes)
    all_neurons=np.array(all_neurons)
    return all_neurons
 

data_sets = ["S1","S1naive","PPC"]
for brain_area in data_sets:
    source_dir='i:/Data/'+ brain_area# choose which brain area
    experiments, trials=import_trial_data(source_dir)
    print("Saving trial data to disk")
    np.save("processed_data/"+brain_area+"_all_trials.npy",trials)

    my_dff,my_dff_MA,my_mouse_id,my_date,my_ROI,my_go_nogo,my_num_trial,my_trial_outcome,my_licks=setup_trial_arrays(trials)
    
    all_neurons=save_all_neurons(experiments, my_dff, my_licks) # all_maens and all_stes take the shape (num_experiments, 4, len(trial))- each of the 4 are hit, miss, cr, fa in that order
    
    print("Saving neuron data to disk")
    np.save("processed_data/"+brain_area+"_all_neurons.npy",all_neurons)
        


