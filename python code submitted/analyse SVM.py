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

brain_area="S1" ### choices are "S1" "S1naive" or "PPC"
directory="SVM_outputs/" # where the SVM output files can be found

def onebyone(brain_area,type_of_analysis):
    ## bring in data
    
    if type_of_analysis == "LICK":
        ##lick nolick - all neurons
        print("*******************************")
        print("Loading data from disk")
        accuracies= np.load(brain_area+"_LICK_accuracies.npy")
        mouse_ids=np.load(brain_area+"_LICK_mouse_ids.npy")
        ranks=np.load(brain_area+"_LICK_ranks.npy")
        text="LICK/NOLICK"
    elif type_of_analysis == "GO":  
        # go nogo - all
        print("*******************************")
        print("Loading data from disk")
        accuracies= np.load(brain_area+"_GO_accuracies.npy")
        mouse_ids=np.load(brain_area+"_GO_mouse_ids.npy")
        ranks=np.load(brain_area+"_GO_ranks.npy")
        text="GO/NOGO"
            
    #remove any accuracies =-100 (these are neurons which didnt have enought data to run SVM)
    accuracies=accuracies[[accuracies!=-100]] 
    ranks=ranks[[ranks!=-100]]

    ## how many neurons meet the criteria
    meet_criteria=(ranks >=95)
    print("number of neurons passing criteria", np.count_nonzero(meet_criteria), " out of", ranks.shape[0])

    
    #####################
    # plot accuracies score 
    # draw lines when move on to a new mouse
    plt.figure()
    
    lines=[]
    old_mouse_id=0
    for i in range(len(mouse_ids)):
        new_mouse_id=str(mouse_ids[i])[:5]
        #print(new_mouse_id)
        if old_mouse_id!=new_mouse_id:
            lines.append(i)
        old_mouse_id=new_mouse_id
    for i in range(len(lines)):
        plt.plot([lines[i],lines[i]],[0.4,1],"k--")
    x=np.arange(ranks.shape[0])
    col = np.where(ranks<95,'k','r')
    plt.scatter(x,accuracies,c=col)
    plt.ylim([0.4,1])
    plt.xlim(left=0)
    plt.title(text)
    plt.show()
    



def GMM_2d(file_lick,file_go,file1_lick, file1_go,text):
    accuracies_lick=np.load(file_lick)
    accuracies_go=np.load(file_go)
    ranks_lick=np.load(file1_lick)
    ranks_go=np.load(file1_go)
        
    #remove any accuracies =-100 (these are neurons which didnt have enought data to run SVM)   
    accuracies=np.concatenate((accuracies_lick.reshape(-1,1),accuracies_go.reshape(-1,1)),axis=1)
    accuracies=accuracies[accuracies[:,0]!=-100]
    accuracies=accuracies[accuracies[:,1]!=-100]
    
    ranks=np.concatenate((ranks_lick.reshape(-1,1),ranks_go.reshape(-1,1)),axis=1)
    ranks=ranks[ranks[:,0]!=-100]
    ranks=ranks[ranks[:,1]!=-100]
    col = np.where(((ranks[:,0]>=95) & (ranks[:,1]>=95)),'black',np.where(((ranks[:,0]>=95) & (ranks[:,1]<95)),'blue',np.where(((ranks[:,0]<95) & (ranks[:,1]>=95)),'purple','grey')))

    
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


onebyone(directory+brain_area,"LICK")
onebyone(directory+brain_area,"GO")
GMM_2d(directory+brain_area+"_LICK_accuracies.npy",directory+brain_area+"_GO_accuracies.npy",directory+brain_area+"_LICK_ranks.npy",directory+brain_area+"_GO_ranks.npy",brain_area+" all neurons")
