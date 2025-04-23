# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 18:17:49 2019

@author: User

THIS IS AN EXTENDED VERSION OF SVM gonogo etc

IN THE ORIGINAL PROGRAM, IF NEURON FAILS THE 25/75 RULE THEN IT IS EXCLUDED
IE ACCURACY AND RANK ARE NOT CACLUATED FOR THAT NEURON

IN THIS VERSION, TRIAL ARE RANDOMLY DELETED UNTIL THE 25/75 RULE IS SATISFIED
(NOTE THAT IT STILL HAS TO PASS A TEST FOR MINIMUM NUMBER OF TRIALS)

"""
import numpy as np
from dataClasses import Trial,Neuron
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn import svm
from sklearn.model_selection import cross_validate, GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix,classification_report
from scipy import stats
import warnings
import random
import math


############ SELECT VALUES HERE
brain_area="S1"  ### choices are "S1" "S1naive" or "PPC"
type_of_analysis= "GO" ### choices are "GO" or "LICK"
#########################


def cutback(X,y,ratio_size):
## used to randomly delete some data points and labels in order to make the ratio a certain size
## Y LABELS MUST BE IN THE FORM 0 AND 1
    if (y == 0).sum()/y.shape[0]<ratio_size: 
    # need to delete some label 1
        num_to_delete=(y == 1).sum()-(y == 0).sum()*((1/ratio_size)-1)
        num_to_delete=int(math.ceil(num_to_delete))
        print("cutting back",(y == 0).sum(),(y == 1).sum()," to ",(y == 0).sum(),(y == 1).sum()-num_to_delete)
        for i in range(num_to_delete):
            rate=np.sum(y)
            cumrate=np.cumsum(y)
            my_rand=np.random.rand()*rate
            event = np.where(cumrate>my_rand)[0][0]
            y=np.delete(y,event,axis=0)
            X=np.delete(X,event,axis=0)
    elif (y == 1).sum()/y.shape[0]<ratio_size:
    # need to delete some label 0
        y=np.where(y==1,2,y)
        y=np.where(y==0,1,y)
        y=np.where(y==2,0,y)
        num_to_delete=(y == 1).sum()-(y == 0).sum()*((1/ratio_size)-1)
        num_to_delete=int(math.ceil(num_to_delete))
        print("cutting back",(y == 0).sum(),(y == 1).sum()," to ",(y == 0).sum(),(y == 1).sum()-num_to_delete)
        for i in range(num_to_delete):
            rate=np.sum(y)
            cumrate=np.cumsum(y)
            my_rand=np.random.rand()*rate
            event = np.where(cumrate>my_rand)[0][0]
            y=np.delete(y,event,axis=0)
            X=np.delete(X,event,axis=0)
        np.where(y==0,2,y)
        np.where(y==1,0,y)
        np.where(y==2,1,y)
    else:
        print("Something wrong - no reason to have called the cutback function")
    return X,y    




'''
# bring in all the trials, mouse id etc, dff series, type of trial
# create the two time series - one relative to lick, one to stimulus
'''

all_trials=np.load("processed_data/"+brain_area + "_all_trials.npy",allow_pickle=True)
trial_list=np.empty((all_trials.shape[0],3))
trial_dff=np.empty((all_trials.shape[0],41)) 
trial_licks=np.empty((all_trials.shape[0],41))

for trial in range(all_trials.shape[0]):
    
    trial_list[trial,0]=all_trials[trial].neuron_num

    
    if type_of_analysis=="GO":
        # for labelling go nogo
        if (all_trials[trial].trial_type)=="go":
            trial_list[trial,1]=1
        elif (all_trials[trial].trial_type)=="nogo":
            trial_list[trial,1]=0
        else:
            print("gone wrong on neuron ",all_trials[trial].neuron_num )
    elif type_of_analysis=="LICK":    
        #for labelling lick nolick
        if (all_trials[trial].trial_outcome)=="FA":
            trial_list[trial,1]=1
        elif (all_trials[trial].trial_outcome)=="Hit":
            trial_list[trial,1]=1
        elif (all_trials[trial].trial_outcome)=="Miss":
            trial_list[trial,1]=0
        elif (all_trials[trial].trial_outcome)=="CR":
            trial_list[trial,1]=0
        else:
            print("gone wrong on neuron ",all_trials[trial].neuron_num )
    else:
        print("Type of analysis can only be 'GO' or 'LICK'")


    trial_list[trial,2]=all_trials[trial].mouse_id

    if (all_trials[trial].trial_outcome)=="FA" or (all_trials[trial].trial_outcome)=="Hit":
        trial_dff[trial,0:41]=all_trials[trial].dff 
        #rebase  dff 
        trial_licks[trial,:]=all_trials[trial].licks
        lick_start=np.argmax(trial_licks[trial,:]==1,axis=0)
        trial_dff_rel_licks=np.zeros(trial_licks.shape[1])
        new_start=lick_start-4
        if new_start>=0:                    
            # move the array of licks left so that timeseries starst at lick time
            trial_dff_rel_licks[0:trial_licks.shape[1]-new_start]=all_trials[trial].dff[new_start: ]
            ## need to mske the last few timesteps on the new series = final timestep on original series
            trial_dff_rel_licks[trial_licks.shape[1]-new_start :]=all_trials[trial].dff[-1]
        else: 
            # move the array of licks right so that timeseries starst at lick time
            trial_dff_rel_licks[0-new_start :]=all_trials[trial].dff[0:all_trials[trial].dff.shape[0]+new_start]
            ## need to mske the first few timesteps on the new series = first timestep on original series
            trial_dff_rel_licks[0:-new_start]=all_trials[trial].dff[ 0 ]
            #trial_dff[trial,41:82]=trial_dff_rel_licks
    else:
        trial_dff[trial,0:41]=all_trials[trial].dff 
        #trial_dff[trial,41:82]=all_trials[trial].dff



print("Array of neuron_num and trial type",trial_list.shape)
print("Array of dff",trial_dff.shape)
neuron_list=np.unique(trial_list[:,0]).reshape(-1,1)
print("Num of neurons before filter=",neuron_list.shape[0])




'''
# we now should have, for each trial, neuronid, list of X= timeseries data, y= label
## now for each neuron we train an SVM:
'''
accuracies=[]
dummy_distn=[]
mouse_ids=[]
Cs=[]
ranks=[]
experiments=[]
count=0



for neuron in neuron_list:
    print("\n",count, str(neuron_list[count,0]), "XXXXXXXXXXXXXXXXXXXX")
    # filter down to just that neuron
    neuron_data_list= trial_list[trial_list[:,0]==neuron].copy()
    neuron_data_dff= trial_dff[trial_list[:,0]==neuron].copy()
    X=neuron_data_dff.copy()
    y=neuron_data_list[:,1]
    ratio_size=.25

    if (y == 0).sum()/y.shape[0]>=ratio_size and (y == 1).sum()/y.shape[0]>=ratio_size:
        _=0
        # print("No cut down needed")
    else:
        X,y=cutback(X,y,ratio_size)

    ####  don;t create SVM if less than a certain % of the trials are in one of the categories OR if less than certain number of total trials
    #### eg if set to .15 then the split must be at least 15%/85%. 14/86 or 86/14 would fail
    if (y == 0).sum()/y.shape[0]>=ratio_size and (y == 1).sum()/y.shape[0]>=ratio_size and y.shape[0]>100:
        kfold_size=min(5,(y == 0).sum(),(y == 1).sum())
        print("Included. Number in each category: No go (or lick)",(y == 0).sum(),"Yes go (or lick)", (y == 1).sum())#, ". kfold size", kfold_size)
        # do pca on the time series
        whitening=False
        pca = PCA(whiten=whitening)#, n_components=25)  
        pca.fit(X)
        X_transformed = pca.transform(X)
        
        ### normalize
        scaler = StandardScaler()    
        scaler.fit(X_transformed)
        X_transformed=scaler.transform(X_transformed)

        
        ## print out explained variance
        print("n components: ",pca.n_components_)
        print("Variance sum of:",np.sum(pca.explained_variance_ratio_))
    
        my_class_weight="balanced"
        
        '''
        instead of RFECV, just do a CV test on data with PCA set to no reduction
        '''
        highest_score=0
        optimalC=0
        for C in [.001, 0.01,.1, 1]:
                        svc = svm.SVC(cache_size=1000, class_weight=my_class_weight,kernel="linear", C=C)
                        cv_results = cross_validate(svc, X_transformed, y, cv=StratifiedKFold(kfold_size),scoring='accuracy')
                        #cv_results = cross_validate(svc, X_transformed, y, cv=kfold_size,scoring='accuracy')
                        mean_CV=np.mean(cv_results['test_score'])
                        print(C, "check the score - CV test score", mean_CV)
                        if mean_CV>highest_score:
                            highest_score=mean_CV
                            optimalC=C
        real_highest_score=highest_score.copy()
        print("Best ", optimalC, highest_score)
        accuracies.append(highest_score)
        Cs.append(optimalC)
        mouse_ids.append(str(neuron_list[count,0])[:4])
        experiments.append(str(neuron_list[count,0]))
        count+=1

        '''
        then do 100 shuffles of the label anc see how well can create classifier.
        this gives a distibution which we can compare to the best classifer using actual data
        '''
        y_new=y.copy()       
        dummy_accuracies=[]
        for my_dummy in range(100):
            random.seed()
            random.shuffle(y_new)
            highest_score=0
            optimalC=0
            for C in [.001, 0.01,.1, 1]:
                        svc = svm.SVC(cache_size=1000, class_weight=my_class_weight,kernel="linear", C=C)
                        cv_results = cross_validate(svc, X_transformed, y_new, cv=StratifiedKFold(kfold_size),scoring='accuracy')
                        #cv_results = cross_validate(svc, X_transformed, y, cv=kfold_size,scoring='accuracy')
                        mean_CV=np.mean(cv_results['test_score'])
                        if mean_CV>highest_score:
                            highest_score=mean_CV
                            optimalC=C
            dummy_accuracies.append(highest_score)
            
        my_rank=stats.percentileofscore(dummy_accuracies, real_highest_score)
        print("Rank ", my_rank)
        
        # plt histogram of shuffled label perofrmancs versus red dot of real label
        plt.figure()
        bins=np.arange(.3   ,1,.025)
        plt.hist(dummy_accuracies,bins=bins,label="shuffled labels")
        plt.plot(real_highest_score,1,"r*",markersize=20,label="with genuine label")
        plt.title("Distribution of classifier performance")
        plt.xlabel("Classifier performance")
        plt.ylabel("Count across 100 shuffles")
        rank_text="Rank="+str(int(my_rank))
        plt.text(0.3,2,rank_text)
        plt.legend()
        plt.show()
        
        
        dummy_distn.append(dummy_accuracies)
        ranks.append(my_rank)
    else:
        accuracies.append(-100)
        ranks.append(-100)
        print("Excluded. Number in each category: No go (or lick)",(y == 0).sum(),"Yes go (or lick)", (y == 1).sum())
        count+=1

experiments=np.array(experiments)
accuracies=np.array(accuracies)
dummy_distn=np.array(dummy_distn)
Cs=np.array(Cs)
ranks=np.array(ranks)




####
print("Saving data to disk")
print("Size being saved:", experiments.shape)
np.save("SVM_outputs/"+brain_area+"_"+type_of_analysis+"_experiments.npy",experiments)
np.save("SVM_outputs/"+brain_area+"_"+type_of_analysis+"_accuracies.npy",accuracies)
np.save("SVM_outputs/"+brain_area+"_"+type_of_analysis+"_ranks.npy",ranks)
np.save("SVM_outputs/"+brain_area+"_"+type_of_analysis+"_dummy_distn.npy",dummy_distn)
np.save("SVM_outputs/"+brain_area+"_"+type_of_analysis+"_Cs.npy",Cs)
np.save("SVM_outputs/"+brain_area+"_"+type_of_analysis+"_mouse_ids.npy",mouse_ids)

