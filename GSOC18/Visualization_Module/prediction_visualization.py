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
    #Printing the accuracy statistics
    print 'Printing the average absolute error'
    print np.mean(np.abs(predictions[:,1:4]-labels[:,1:4]),axis=0)

    #Creating a permuation to suffle the array
    perm=np.random.permutation(predictions.shape[0])
    #Shuffling the array based on above permutation
    predictions=predictions[perm,:]
    labels=labels[perm,:]
    #Specifying the number of bins
    bins=25

    regression_length=4
    regression_names=['Energy','pos-x','pos-y','pos-z']
    for pred_i in range(regression_length):
        #Plotting the Histogram in Energy
        fig=plt.figure()
        fig.suptitle('Relative Error Histogram and Profile Histogram: {}'.format(
                                                            regression_names[pred_i]))
        #plotting the Relative Error Histogram
        ax1=fig.add_subplot(131)
        values=None
        if pred_i==0:
            values=((predictions[:,pred_i]-labels[:,pred_i])/labels[:,pred_i])
            ax1.set_title('Relative Error Histogram: [(prediction-labels)/labels] ')
            ax1.set_xlabel('Relative Error')
        else:
            values=(predictions[:,pred_i]-labels[:,pred_i])
            ax1.set_title('Error Histogram: [prediction-labels] ')
            ax1.set_xlabel('Error')

        ax1.hist(values,ec='k',alpha=0.7,bins=bins)
        ax1.set_ylabel('Counts ( out of total {} samples )'.format(labels.shape[0]))
        #Plotting the Profile-Histogram for the Energy Prediction
        #Specifying the data (Energy) to bin over
        ax2=fig.add_subplot(132)
        x=labels[:,pred_i]
        #Specifying the values to calculate the statistics on, in that bin
        values=None
        if pred_i==0:
            values=np.abs((predictions[:,pred_i]-labels[:,pred_i])/labels[:,pred_i])
            ax2.set_ylabel('Mean Value of Absolute Relative Error (in bins)')
        else:
            values=np.abs(predictions[:,pred_i]-labels[:,pred_i])
            ax2.set_ylabel('Mean Value of Absolute Error (in bins)')


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
        ax2.errorbar(midpoints,mean,yerr=rms,fmt='b.-',
                    mfc='r',mec='r',ecolor='y')
        ax2.set_title('Profile Histogram')
        ax2.set_xlabel('Value of {} bins'.format(regression_names[pred_i]))


        #Plotting the RMS values of the relative Error in energy bins
        ax3=fig.add_subplot(133)
        #print midpoints,rms
        #ax3.bar(midpoints,rms,bin_length,ec='k',color='y')
        ax3.plot(midpoints,rms,'b.-',mfc='r',mec='r')
        if pred_i==0:
            ax3.set_title('RMS of Absolute Relative Error in {} Bins'.format(
                                                regression_names[pred_i]))
            ax3.set_ylabel('RMS of Absolute Relative Error in the bin')
        else:
            ax3.set_title('RMS of Absolute Error in {} Bins'.format(
                                                regression_names[pred_i]))
            ax3.set_ylabel('RMS of Absolute Error in the bin')

        ax3.set_xlabel('{} of bins'.format(regression_names[pred_i]))

        #plt.savefig('energy_hist.png',bbox_inches='tight')
        plt.show()
        plt.close()

        #Taking the random shuffled sample
        no_samples=100
        x=range(1,no_samples+1)
        predictions_slice=predictions[0:no_samples,:]
        labels_slice=labels[0:no_samples,:]

        #Plotting these sample : Energy
        fig=plt.figure()
        fig.suptitle('{} Prediction and Label'.format(regression_names[pred_i]))
        #Plotting the overalyed Bar Graph
        ax1=fig.add_subplot(211)
        ax1.bar(x,predictions_slice[:,pred_i],alpha=1,color='#ff9933',label='prediction')
        ax1.bar(x,labels_slice[:,pred_i],alpha=0.5,color='b',label='label')
        ax1.set_xlabel('{} random sample prediction and labels'.format(no_samples))
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.set_title('Bar Graph: Prediction and Label overlayed')
        #Plotting the corresponding difference
        ax2=fig.add_subplot(212)
        if pred_i==0:
            values=((predictions_slice[:,pred_i]-labels_slice[:,pred_i])/labels_slice[:,pred_i])
            ax2.set_title('(Prediction-Label)/Label')
            ax2.set_ylabel('Relative Error')
        else:
            values=(predictions_slice[:,pred_i]-labels_slice[:,pred_i])
            ax2.set_title('(Prediction-Label)')
            ax2.set_ylabel('Error')

        ax2.bar(x,values,alpha=0.7,color='r',label='Error')
        ax2.set_xlabel('Same random Sample as above')
        ax2.legend()

        plt.show()
        plt.close()


    ############### posx/y profile over Energy Histogram #########
    #Specifying the data (Energy) to bin over
    x=labels[:,0]
    for i in range(1,3):
        fig=plt.figure()
        fig.suptitle('Varaition of Error in Barycenter position with Eenrgy')

        ax1=fig.add_subplot(121)
        values=np.abs(predictions[:,i]-labels[:,i])
        if i==1:
            ax1.set_ylabel('Mean Value of Absolute Error in posx (in bins)')
            ax1.set_title('Profile Histogram of posx')
        else:
            ax1.set_ylabel('Mean Value of Absolute Error in posy (in bins)')
            ax1.set_title('Profile Histogram of posy')

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
        ax1.errorbar(midpoints,mean,yerr=rms,fmt='b.-',
                    mfc='r',mec='r',ecolor='y')
        ax1.set_xlabel('Value of energy bins')


        #Plotting the RMS values of the relative Error in energy bins
        ax2=fig.add_subplot(122)
        ax2.plot(midpoints,rms,'b.-',mfc='r',mec='r')
        if i==1:
            ax2.set_title('RMS of Absolute Error in posx')
            ax2.set_ylabel('RMS of Absolute Error')
        else:
            ax2.set_title('RMS of Absolute Error in posy')
            ax2.set_ylabel('RMS of Absolute Error')

        ax2.set_xlabel('Energy of bins')

        #plt.savefig('energy_hist.png',bbox_inches='tight')
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
