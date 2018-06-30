import numpy as np
import matplotlib.pyplot as plt

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

    #Taking the random shuffled sample
    no_samples=100
    predictions=predictions[0:no_samples,:]
    labels=labels[0:no_samples,:]

    #Plotting these sample
    x=range(1,no_samples+1)
    plt.bar(x,predictions[:,0],color='#ff9933')
    plt.bar(x,labels[:,0],alpha=0.5,color='g')
    plt.show()

    plt.clf()
    plt.bar(x,predictions[:,1],color='#ff9933')
    plt.bar(x,labels[:,1],alpha=0.5,color='g')
    plt.show()

    plt.clf()
    plt.bar(x,predictions[:,2],color='#ff9933')
    plt.bar(x,labels[:,2],alpha=0.5,color='g')
    plt.show()


if __name__=='__main__':
    #Loading the prediction and label data
    filename='results.npz'
    data=load_data(filename)
    predictions=data['predictions']
    labels=data['labels']

    #Now plotting the predictions and labels
    plot_histogram(predictions,labels)
