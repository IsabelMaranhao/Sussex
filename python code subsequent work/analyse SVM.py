# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 08:55:56 2019

@author: User

analyse output of SVM gonogo etc
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture
import scipy.stats as stats
from scipy.cluster.vq import kmeans2
import numpy.ma as ma

brain_area="PPC" ### choices are "S1" "S1naive" or "PPC"
directory="SVM_outputs/" # where the SVM output files can be found
#directory="SVM_outputs/orig from 2020/" # where the SVM output files can be found
#directory="SVM_outputs/2022 - with 100 minimum size/" # where the SVM output files can be found

def onebyone(directory,brain_area_in,type_of_analysis):
    brain_area=directory+brain_area_in
    ## bring in data
    
    if type_of_analysis == "LICK":
        ##lick nolick - all neurons
        print("*******************************")
        print("Loading data from disk")
        accuracies= np.load(brain_area+"_LICK_accuracies.npy")
        mouse_ids=np.load(brain_area+"_LICK_mouse_ids.npy")
        ranks=np.load(brain_area+"_LICK_ranks.npy")
        expts=np.load(brain_area+"_LICK_experiments.npy")
        text="LICK/NOLICK"
    elif type_of_analysis == "GO":  
        # go nogo - all
        print("*******************************")
        print("Loading data from disk")
        accuracies= np.load(brain_area+"_GO_accuracies.npy")
        mouse_ids=np.load(brain_area+"_GO_mouse_ids.npy")
        ranks=np.load(brain_area+"_GO_ranks.npy")
        expts=np.load(brain_area+"_GO_experiments.npy")
        text="GO/NOGO"
    elif type_of_analysis == "REWARD":  
        # go nogo - all
        print("*******************************")
        print("Loading data from disk")
        accuracies= np.load(brain_area+"_REWARD_accuraciesPREVIOUS.npy")
        mouse_ids=np.load(brain_area+"_REWARD_mouse_idsPREVIOUS.npy")
        ranks=np.load(brain_area+"_REWARD_ranksPREVIOUS.npy")
        expts=np.load(brain_area+"_REWARD_experimentsPREVIOUS.npy")
        text="REWARD"
            
    #remove any accuracies =-100 (these are neurons which didnt have enought data to run SVM)
    accuracies=accuracies[[accuracies!=-100]] 
    ranks=ranks[[ranks!=-100]]
    
    
    # for S1, manually remove mouse id 406, session 20180409 due to bad data
    # need to remove from sorted_accuracies and amend position of red and black vertical lines
    if brain_area_in=="S1":
        mouse_and_session= np.char.ljust(expts,11)
        mask=(mouse_and_session=="40620180409")
        expts=np.ma.masked_array(expts)
        expts[mask]=np.ma.masked
        expts=expts.compressed()
        
        mouse_ids=np.ma.masked_array(mouse_ids)
        mouse_ids[mask]=np.ma.masked
        mouse_ids=mouse_ids.compressed()
        
        ranks=np.ma.masked_array(ranks)
        ranks[mask]=np.ma.masked
        ranks=ranks.compressed()
        
        accuracies=np.ma.masked_array(accuracies)
        accuracies[mask]=np.ma.masked
        accuracies=accuracies.compressed()

    ## how many neurons meet the criteria
    meet_criteria=(ranks >=95)
    print("number of neurons passing criteria", np.count_nonzero(meet_criteria), " out of", ranks.shape[0])

 


    #####################
    # plot accuracies score 
    # draw lines when move on to a new mouse
    plt.figure()
    
    # bodge to create expereiments wit all the same lenght numbers.  careful with this
    tidied_expts=np.empty_like(expts)
    for count,e in enumerate(expts):
        if len(e)==14:
            new_e=e[:11]+'0'+e[11:]
        else:
            new_e=e
        tidied_expts[count]=new_e
    order=tidied_expts.argsort()
    sorted_expts=tidied_expts[order]
    sorted_ranks=ranks[order]
    sorted_accuracies=accuracies[order]
    sorted_mouse_ids=mouse_ids[order]
    
    ##create a single csv output file
    ## manually change name
    csv_output=sorted_expts.reshape(-1,1).copy()
    csv_output=np.append(csv_output,sorted_accuracies.reshape(-1,1),axis=1)
    csv_output=np.append(csv_output,sorted_ranks.reshape(-1,1),axis=1)
    for count,e in enumerate(sorted_expts):
        if len(e)==15 and str(sorted_expts[count])[:4] in ["1140","1218","1260","1268","1298"]:
            new_e=e[:12]+'0'+e[12:]
        else:
            new_e=e
        csv_output[count,0]=new_e
    
    np.savetxt(directory+"PPCpreviousTrial.csv",csv_output,fmt="%s",delimiter=",",header='Expt,Accuracy,Rank')       
    ###############
    

    lines=[]
    red_lines=[]
    old_mouse_id=0
    old_session_id=0
    for i in range(len(sorted_expts)):
        ## we get the mouse id and session id from sorted_expts. but problem is the mouse ids are varying legnths
        ## coud do auto but jsut set manually for now
        if str(sorted_expts[i])[:3] in ["301","406","410","531","588","756","588","756"]:
            cut_off1=11
            cut_off2=3
        elif str(sorted_expts[i])[:4] in ["1140","1218","1260","1268","1298"]:
            cut_off1=12
            cut_off2=4
        else:
            print("PROBLEM WITH MOUSEID")
        new_session_id=str(sorted_expts[i])[:cut_off1] 
        if old_session_id!=new_session_id:
            lines.append(i)        
        old_session_id=new_session_id
        new_mouse_id=str(sorted_expts[i])[:cut_off2]  
        if old_mouse_id!=new_mouse_id:
            red_lines.append(i)        
        old_mouse_id=new_mouse_id
    for i in range(len(lines)):
        plt.plot([lines[i],lines[i]],[0.4,1],"k--")
    for i in range(len(red_lines)):
        plt.plot([red_lines[i],red_lines[i]],[0.4,1],"r-")
    x=np.arange(sorted_ranks.shape[0])
    col = np.where(sorted_ranks<95,'k','r')
    plt.scatter(x,sorted_accuracies,c=col)
    plt.ylim([0.4,1])
    plt.xlim(left=0)
    plt.xlabel("Neuron")
    plt.ylabel("Discriminator accuracy")
    plt.title(text )
    plt.show()
    

    
    


def GMM_2d(file2_lick, file2_go, file_lick,file_go,file1_lick, file1_go,text):
    accuracies_lick=np.load(file_lick)
    accuracies_go=np.load(file_go)
    ranks_lick=np.load(file1_lick)
    ranks_go=np.load(file1_go)
    expts_lick=np.load(file2_lick)
    expts_go=np.load(file2_go)
        
    #remove any accuracies =-100 (these are neurons which didnt have enought data to run SVM)   
    accuracies=np.concatenate((accuracies_lick.reshape(-1,1),accuracies_go.reshape(-1,1)),axis=1)
    ranks=np.concatenate((ranks_lick.reshape(-1,1),ranks_go.reshape(-1,1)),axis=1)

    
       
    accuracies=accuracies[accuracies[:,0]!=-100]
    accuracies=accuracies[accuracies[:,1]!=-100]
    
    
    ranks=ranks[ranks[:,0]!=-100]
    ranks=ranks[ranks[:,1]!=-100]
    
    
    
    col = np.where(((ranks[:,0]>95) & (ranks[:,1]>95)),'black',np.where(((ranks[:,0]>95) & (ranks[:,1]<=95)),'blue',np.where(((ranks[:,0]<=95) & (ranks[:,1]>95)),'purple','grey')))

    
    print("*******************************")
    print("Number of neurons in 2d chart",accuracies.shape [0])


    plt.figure()
    plt.scatter(accuracies[:,1],accuracies[:,0],c=col)
    plt.plot([.45,1],[.45,1],"k--")
    plt.xlabel("go accuracies")
    plt.ylabel("lick accuracies")
    plt.title(text)
    plt.xlim([0.45,1])
    plt.ylim([0.45,1])


# onebyone(directory,brain_area,"LICK")
# onebyone(directory,brain_area,"GO")
# GMM_2d(directory+brain_area+"_LICK_experiments.npy",directory+brain_area+"_GO_experiments.npy",directory+brain_area+"_LICK_accuracies.npy",directory+brain_area+"_GO_accuracies.npy",directory+brain_area+"_LICK_ranks.npy",directory+brain_area+"_GO_ranks.npy",brain_area+" all neurons")

onebyone(directory,brain_area,"REWARD")