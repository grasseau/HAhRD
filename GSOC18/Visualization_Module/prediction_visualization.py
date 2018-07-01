import numpy as np
import matplotlib.pyplot as plt
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

    #Plotting the error histogram in Energy
    plt.hist(predictions[:,0]-labels[:,0],ec='k',alpha=0.7,bins=100)
    plt.title('Energy Error Histogram: [prediction-labels] ')
    plt.xlabel('Energy Error')
    plt.ylabel('Counts (out of total {} test samples)'.format(labels.shape[0]))
    plt.show()
    #plt.savefig('energy_hist.png',bbox_inches='tight')
    plt.close()
    #Error histogram in Eta prediction
    plt.hist(predictions[:,1]-labels[:,1],ec='k',alpha=0.7,bins=100)
    plt.title('Eta Error Histogram: [prediction-labels] ')
    plt.xlabel('Eta Error')
    plt.ylabel('Counts (out of total {} test samples)'.format(labels.shape[0]))
    plt.show()
    plt.close()
    #Error histogram in Phi prediction
    plt.hist(predictions[:,2]-labels[:,2],ec='k',alpha=0.7,bins=100)
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
    ax2.bar(x,predictions[:,0]-labels[:,0],alpha=0.7,color='r',label='Error')
    ax2.set_xlabel('Same random Sample as above')
    ax2.set_ylabel('Error')
    ax2.legend()
    ax2.set_title('Prediction-Label')
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
    filename='results.npz'
    data=load_data(filename)
    predictions=data['predictions']
    labels=data['labels']

    #Now plotting the predictions and labels
    plot_histogram(predictions,labels)
