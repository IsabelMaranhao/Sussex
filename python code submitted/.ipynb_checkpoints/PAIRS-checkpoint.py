# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 15:03:29 2020

@author: User


PAIRS
"""
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats


# bring in neuron data - this uses previously created files.  you need to chooes which file to use (one for each brain area)
all_neurons=np.load("processed_data/PPC_all_neurons.npy",allow_pickle=True)


#choose which random method to use as follows:
#0: normalize the PCA features and then randomly select according to normal distribution
#1: non-parametric. select a random  
random_method=1

# number of PCA features to use
num_features=8

# number fo nearest neighbours to use in PAIRS analysis
n_neighbours=3# this is the samllest value for which random median>=pi/8
#############################################
    
    
    
data=np.empty((len(all_neurons), 41*6))
for i in range(all_neurons.shape[0]):
    all_means_single_neuron=np.concatenate((all_neurons[i].mean_time_series[0],all_neurons[i].mean_time_series[1],all_neurons[i].mean_time_series[2],all_neurons[i].mean_time_series[3],all_neurons[i].mean_time_series[4],all_neurons[i].mean_time_series[5]),axis=0)
    data[i,:]=all_means_single_neuron
#carry out PCA
pca = PCA(n_components=num_features,whiten=False)  
pca.fit(data)


## print out explained variance
print("n components ",pca.n_components_)
print("Variance sum of % ",np.sum(pca.explained_variance_ratio_))
data_transformed = pca.transform(data) 
print("shape of data after PCA", data_transformed.shape)

##### if not using the whole set, then filter by mouse id and experiment date.
#my_mouse_id=1268
#my_date=20190814
#i=0
#my_mask=np.zeros((all_neurons.shape),dtype=bool)
#for neuron in all_neurons:
#    if neuron.mouse_id==my_mouse_id and neuron.date==my_date:
#        my_mask[i]=True
#    else:
#        my_mask[i]=False 
#    i=i+1
#data_transformed=data_transformed[my_mask]      
#print("shape of data after reducing to required experiment", data_transformed.shape)
################



num_neurons=data_transformed.shape[0]
num_features=data_transformed.shape[1]

if random_method==0:
    #normalize the features
    scaler = StandardScaler()    
    scaler.fit(data_transformed)
    data_transformed=scaler.transform(data_transformed)





'''
produce N*N matrix showing the angles between each of the neurons, where N is numbe rof neurons
'''
numerator=np.dot(data_transformed,data_transformed.T)
size=np.sqrt(np.diag(numerator)).reshape(-1,1)
denominator=np.dot(size,size.T)
data_angles_cosine=numerator/denominator # this is a symmetrix matrix showing the angles between each neuron
# there seems to be a rounding error. if>1 make cosine=1
data_angles_cosine=np.where(data_angles_cosine>1,1,data_angles_cosine)
# what comes up as eg +5degrees might be -5 degrees,but doesnt matter - no need to correct for that
data_angles=np.arccos(data_angles_cosine)

'''
for each neuron find the difference angle to the 3 nearest angles of the other 288 neurons
and take the average of these three
'''
# we now have a histogram of what the average angle is between a neuron and its nearest neighbours
print("n_neighbours",n_neighbours)
mean_angles=np.empty(data_angles.shape[0])
# each i (row) of data_angles is for one neuron, the 289 angles to all the other neurons.  call the is nueron_distances
# for a chosen i, distances is then 289*4 = the 4 nearest neighbour angles for each of the neurons
#  mean_angle is the mean of the first row of distances ie the mean distance from that neuron to its nearest neighbours
for i in range(data_angles.shape[0]):
    neuron_distances=data_angles[i,:]
    distances=np.sort(neuron_distances)[1:n_neighbours+1]
    mean_angle=np.mean(distances)
    mean_angles[i]=mean_angle
actual_mean_angles=mean_angles.copy()
actual_data_transformed=data_transformed.copy()   



'''
for compaison with the real data:
create random distributions of nearest neighbours
for 10k different runs, generate 289 neurons according to gaussian distn with mean and stdev of the actual data
average over thos 10k runs will be histogram of the nearest neighbour distribution if the distribution was random
i.e this is how clustering would likely be if there was no clustering
'''

# work out mean and stdevs of the features
# so the the gaussina distibution estimate of each of the PCs in the real data
means=np.mean(data_transformed,axis=0)
stdevs=np.std(data_transformed,axis=0)  

## generate 10k data sets of 289 random vectors
# each row is a simulation of 289 neurons, each of which is descirbed by the number of PCs used in the real data.
all_mean_angles=[]
num_runs=10000
for run in range(num_runs):
    if run%100==0:
        print("run ", run, " of ",num_runs )
    data_transformed=np.empty((num_neurons,num_features))
    #generate random data
    if random_method==0:
        ### assume each PCA is a normal distribution and generate according to that 
        # for this appoach to be valid needs the PCA to be normalized
        for i in range(num_features):        
            data_transformed[:,i]=means[i]+ stdevs[i]*np.random.randn(num_neurons)
    elif random_method==1:
        for i in range(num_features): 
            data_transformed[:,i]=np.random.choice(actual_data_transformed[:,i], data_transformed.shape[0])    

    '''
    produce 289*289 matrix showing the angles between each of the neurons
    '''
    numerator=np.dot(data_transformed,data_transformed.T)
    size=np.sqrt(np.diag(numerator)).reshape(-1,1)
    denominator=np.dot(size,size.T)
    data_angles_cosine=numerator/denominator # this is a symmetrix matrix showing the angles between each neuron
    # there seems to be a rounding error. if>1 make cosine=1
    data_angles_cosine=np.where(data_angles_cosine>1,1,data_angles_cosine)
    # convert cosineinto angle
    # what comes up as eg +5degrees might be -5 degrees,but doesnt matter - no need to correct for that
    data_angles=np.arccos(data_angles_cosine)
    #temp=data_angles*180/np.pi # produce 0 to 180 degrees
     
    '''
    for each neuron find the difference angle to the 3 nearest angles of the other 288 neurons
    and take the average of these three
    '''
    # we now have a histogram of what the average angle is between a neuron and its nearest neighbours
    mean_angles=np.empty(data_angles.shape[0])
    # each i (row) of data_angles is for one neuron, the 289 angles to all the other neurons.  call the is nueron_distances
    # for a chosen i, distances is then 289*4 = the 4 nearest neighbour angles for each of the neurons
    #  mean_angle is the mean of the first row of distances ie the mean distance from that neuron to its nearest neighbours
    for i in range(data_angles.shape[0]):
        neuron_distances=data_angles[i,:]
        distances=np.sort(neuron_distances)[1:n_neighbours+1]
        mean_angle=np.mean(distances)
        mean_angles[i]=mean_angle
    all_mean_angles.append(mean_angles)
dummy_mean_angles_shaped=np.asarray(all_mean_angles)
dummy_mean_angles=dummy_mean_angles_shaped.ravel()


#for i in range(num_features):    
#    plt.figure()
#    bins=np.arange(-10,10,0.5)
#    plt.hist(actual_data_transformed[:,i],bins=bins)
#    plt.hist(data_transformed[:,i],bins=bins) # distn of just one of the dummy_runs
#    plt.title("Distribution of PCA component "+str(i))
#    plt.show()



plt.figure()
step=0.01
bins=np.pi*np.arange(0,0.5+step,step)
actual_hist=plt.hist(actual_mean_angles,bins=bins,label="genuine data") # actual data
step1=0.01
bins=np.pi*np.arange(0,0.5+step1,step1)
dummy_hist=np.histogram(dummy_mean_angles,bins=bins)# actual data
plt.plot(dummy_hist[1][:-1]+step1/2, dummy_hist[0]/num_runs*step/step1,"r", label="random")
plt.title("Distribution of mean angle to nearest neighbours\n S1")
plt.xlabel("Nearest neighbour mean angle(rad)")
plt.ylabel("Number of pairs")    
plt.legend()
plt.plot()
plt.show()  


'''
now compare median of actual data with median distribution of all the dummy runs
'''
print("median real data", np.median(actual_mean_angles)) 
print("median random", np.median(dummy_mean_angles))
all_dummy_medians=np.median(dummy_mean_angles_shaped,axis=1)
plt.figure()
medians_hist=plt.hist(all_dummy_medians, label="dummy neuron sets")
plt.plot(np.median(actual_mean_angles),5,"r*",markersize=20,label="genuine neuron set")
plt.title("Distribution of median of mean angle to nearest neighbours\n S1")
plt.xlabel("Median of nearest neighbour mean angle(rad)")
plt.ylabel("Count")
plt.legend()
plt.show()

''' work out where the real data sits in the list of 10,000 dummy runs 
(so p<.05 requires it to be less than 500)
'''
print("pseudo pvalue=", stats.percentileofscore(all_dummy_medians,np.median(actual_mean_angles)))
