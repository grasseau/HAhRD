import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
#Adding full screen support based on answer on Stack Overflow
#https://stackoverflow.com/questions/32428193/
    #saving-matplotlib-graphs-to-image-as-full-screen?
    #noredirect=1&lq=1
# manager = plt.get_current_fig_manager()
# manager.resize(*manager.window.maxsize())

def load_data(filename):
    '''
    DESCRIPTION:
        This function will load the compressed label and prediction
        array in the dictionary and can be retreived by accessing
        the dictionary with the name
        1.labels
        2.predictions
    USAGE:
        INPUT:
            filename  : the file name of the the compressed data
        OUTPUT:
            data      : the dictionary containing the labels and
                        predictions
    '''
    data=np.load(filename)
    return data

def variance_aggregate_function(bin_values):
    '''
    DESCRIPTION:
        This function is an aggregation function of the values present
        in the bins to calculate the standard deviation of the labels
        in the specific bins
    '''
    #Returning the nan values if the array is empty
    if bin_values.shape[0]==0:
        return np.nan

    #Calculating the aggregate value with degree of freedom=0
    std=np.std(bin_values,ddof=0)

    return std  #scalar values

def get_rms(bin_values):
    '''
    DESCRIPTION:
        This function will be used as an aggregation function to calculate
        the bin statistics to get the RMS value of that bin
    '''
    #Returning the nan values if the array is empty
    if bin_values.shape[0]==0:
        return np.nan

    #Calculating the RMS value
    rms=np.power(np.mean(np.power(bin_values,2)),0.5)

    return rms  #scalar values


def plot_histogram(predictions,labels):
    '''
    DESCRIPTION:
        This function will plot the histogram of prediction and
        labels by randomly sampling few examples out of whole
        prediction predictions
    USAGE:
        INPUT:
            predictions : the prediction of the model from the saved weights
            labels      : the target labels that was assigned for that input
        OUTPUT:
    '''
    #Creating a permuation to suffle the array
    perm=np.random.permutation(predictions.shape[0])
    #Shuffling the array based on above permutation
    predictions=predictions[perm,:]
    labels=labels[perm,:]
    #Specifying the number of bins
    bins=50
    #Plotting the Histogram in Energy
    fig=plt.figure()
    fig.suptitle('Relative Error Histogram and Profile Histogram: Energy')
    #plotting the Relative Error Histogram
    ax1=fig.add_subplot(121)
    ax1.hist(((predictions[:,0]-labels[:,0])/labels[:,0]),
                    ec='k',alpha=0.7,bins=bins)
    ax1.set_title('Relative Energy Error Histogram: [(prediction-labels)/labels] ')
    ax1.set_xlabel('Relative Energy Error')
    ax1.set_ylabel('Counts (out of total {} test samples)'.format(labels.shape[0]))
    #Plotting the Profile-Histogram for the Energy Prediction
    #Specifying the data (Energy) to bin over
    ax2=fig.add_subplot(122)
    x=labels[:,0]
    #Specifying the values to calculate the statistics on, in that bin
    values=((predictions[:,0]-labels[:,0])/labels[:,0])
    #Calculating the bin statistics
    mean,bin_edges,_=stats.binned_statistic(x,values,
                                                    statistic='mean',
                                                    bins=bins)
    rms,_,_=stats.binned_statistic(x,values,
                                        statistic=get_rms,
                                        bins=bins)
    #Now calculating the midpoint of the bins to plot the profile
    bin_length=bin_edges[1]-bin_edges[0]
    midpoints=bin_edges[1:]-bin_length
    #now plotting the bin_statistics at the midpoints
    # print mean
    # print std
    ax2.errorbar(midpoints,mean,yerr=rms,fmt='b.-',
                mfc='r',mec='r',ecolor='y')
    ax2.set_title('Profile Histogram')
    ax2.set_xlabel('Energy of Examples')
    ax2.set_ylabel('Mean Value of Relative Error (in bins)')
    plt.show()
    #plt.savefig('energy_hist.png',bbox_inches='tight')
    plt.close()
    #Error histogram in Eta prediction
    plt.hist(predictions[:,1]-labels[:,1],ec='k',alpha=0.7,bins=bins)
    plt.title('Eta Error Histogram: [prediction-labels] ')
    plt.xlabel('Eta Error')
    plt.ylabel('Counts (out of total {} test samples)'.format(labels.shape[0]))
    plt.show()
    plt.close()
    #Error histogram in Phi prediction
    plt.hist(predictions[:,2]-labels[:,2],ec='k',alpha=0.7,bins=bins)
    plt.title('Phi Error Histogram: [prediction-labels] ')
    plt.xlabel('Phi Error')
    plt.ylabel('Counts (out of total {} test samples)'.format(labels.shape[0]))
    plt.show()
    plt.close()

    #Taking the random shuffled sample
    no_samples=100
    x=range(1,no_samples+1)
    predictions=predictions[0:no_samples,:]
    labels=labels[0:no_samples,:]

    #Plotting these sample : Energy
    fig=plt.figure()
    fig.suptitle('Energy Prediction:Label')
    #Plotting the overalyed Bar Graph
    ax1=fig.add_subplot(211)
    ax1.bar(x,predictions[:,0],alpha=1,color='#ff9933',label='prediction')
    ax1.bar(x,labels[:,0],alpha=0.5,color='b',label='label')
    ax1.set_xlabel('{} random sample prediction and labels'.format(no_samples))
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.set_title('Bar Graph: Prediction and Label overlayed')
    #Plotting the corresponding difference
    ax2=fig.add_subplot(212)
    ax2.bar(x,((predictions[:,0]-labels[:,0])/labels[:,0]),
                        alpha=0.7,color='r',label='Error')
    ax2.set_xlabel('Same random Sample as above')
    ax2.set_ylabel('Percentage Error')
    ax2.legend()
    ax2.set_title('(Prediction-Label)/Label')
    plt.show()
    plt.close()


    #Plotting the samples : Eta
    fig=plt.figure()
    fig.suptitle('Eta Prediction:Label')
    #Plotting the overalyed Bar Graph
    ax1=fig.add_subplot(211)
    ax1.bar(x,predictions[:,1],alpha=1,color='#ff9933',label='prediction')
    ax1.bar(x,labels[:,1],alpha=0.5,color='b',label='label')
    ax1.set_xlabel('{} random sample prediction and labels'.format(no_samples))
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.set_title('Bar Graph: Prediction and Label overlayed')
    #Plotting the corresponding difference
    ax2=fig.add_subplot(212)
    ax2.bar(x,predictions[:,1]-labels[:,1],alpha=0.7,color='r',label='Error')
    ax2.set_xlabel('Same random Sample as above')
    ax2.set_ylabel('Error')
    ax2.legend()
    ax2.set_title('Prediction-Label')
    plt.show()
    plt.close()

    #Plotting the samples :Phi
    fig=plt.figure()
    fig.suptitle('Phi Prediction:Label')
    #Plotting the overalyed Bar Graph
    ax1=fig.add_subplot(211)
    ax1.bar(x,predictions[:,2],alpha=1,color='#ff9933',label='prediction')
    ax1.bar(x,labels[:,2],alpha=0.5,color='b',label='label')
    ax1.set_xlabel('{} random sample prediction and labels'.format(no_samples))
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.set_title('Bar Graph: Prediction and Label overlayed')
    #Plotting the corresponding difference
    ax2=fig.add_subplot(212)
    ax2.bar(x,predictions[:,2]-labels[:,2],alpha=0.7,color='r',label='Error')
    ax2.set_xlabel('Same random Sample as above')
    ax2.set_ylabel('Error')
    ax2.legend()
    ax2.set_title('Prediction-Label')
    plt.show()
    plt.close()



if __name__=='__main__':
    #Loading the prediction and label data
    filename='results_mode_valid.npz'
    data=load_data(filename)
    predictions=data['predictions']
    labels=data['labels']

    #Now plotting the predictions and labels
    plot_histogram(predictions,labels)
