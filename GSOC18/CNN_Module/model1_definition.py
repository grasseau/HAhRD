import tensorflow as tf

def calculate_model_accuracy(Z,Y):
    '''
    DESCRIPTION:
        This function will be used to calculate the model accuracy.
        Both regression accuracy and classification accuracy
        will be displayed separately in the tensorboard.
        This formulation of accuracy will be unique for each
        model description.
    USAGE:
        INPUT:
            Z   :the activation from the final layer of the model
            Y   :the target label of the model to aim at (Arjun)
        OUPUT:
            regression_accuracy     : 100- (% error)(WRONG, Think better metric)
            classification_accuracy : (correct/total)*100

            These output will be optionally caught by the caller
            if we want to display them in the terminal also and
            then run in the ops along with losses.
    '''
    with tf.name_scope('accuracy_calculation'):
        #Regression accuracy calcuation as 100-%error
        #Energy_accuracy/error
        energy_abs_diff=tf.losses.absolute_difference(Z[:,0],
                                            Y[:,0],reduction=None)
        energy_error=(tf.reduce_mean(tf.divide(
                                energy_abs_diff,Y[:,0]+1e-10)))*100
        tf.summary.scalar('%_energy_error',energy_error)

        #Eta Accuracy/Error
        eta_abs_diff=tf.losses.absolute_difference(Z[:,1],
                                                    Y[:,1],reduction=None)
        eta_error=(tf.reduce_mean(tf.divide(
                                    eta_abs_diff,Y[:,1]+1e-10)))*100
        tf.summary.scalar('%_eta_error',eta_error)

        #Phi Accuracy/Error
        phi_abs_diff=tf.losses.absolute_difference(Z[:,2],
                                                    Y[:,2],reduction=None)
        phi_error=(tf.reduce_mean(tf.divide(
                                    phi_abs_diff,Y[:,2]+1e-10)))*100
        tf.summary.scalar('%_phi_error',phi_error)


        #Calculation of Classification error
        regression_len=3
        prediction=tf.argmax(Z[:,regression_len:],axis=1)
        label=tf.argmax(Y[:,regression_len:],axis=1)

        correct=tf.equal(prediction,label)
        classification_accuracy=tf.reduce_mean(tf.cast(correct,'float'))*100
        tf.summary.scalar('%_classification_accuracy',classification_accuracy)

    return None,classification_accuracy





def calculate_total_loss(Z,Y,scope=None):
    '''
    DESCRIPTION:
        This function combines all the losses i.e the model loss
        + L2Regularization loss

        Also, it will filter the L2 loss based on the namescope
        from all_losses collection to give specific loss for
        each GPU.
        (Though the L2Loss will be same for both GPU, but both
        GPU need to calculate their complete loss for computing
        gradient, so its better to do it from their local copy
        of variables)
    USAGE:
        INPUT:
            Z       :the final layer's unnormalized output of model
            Y       :the actual target label to train model on
            scope   :the namescope of the current tower to compute
                        the tower specific loss from the local copy
                        of the parameter present on each tower.
        OUTPUT:
            total_cost: the total cost of the model, which will be
                        used to calculate the gradient
    '''
    with tf.name_scope('loss_calculation'):
        #A list to combine all losses
        all_loss_list=[]

        #Calculating the L2_loss(scalar)
        reg_loss_list=tf.get_collection('all_losses',scope=scope)
        l2_reg_loss=0.0
        if not len(reg_loss_list)==0:
            l2_reg_loss=tf.add_n(reg_loss_list,name='l2_reg_loss')
            tf.summary.scalar('l2_reg_loss',l2_reg_loss)
        all_loss_list.append(l2_reg_loss)

        #Calculating the model loss
        #Calculating the regression loss(scalar)
        regression_len=3
        regression_output=Z[:,0:regression_len]#0,1,2
        regression_label=Y[:,0:regression_len]#TRUTH
        regression_loss=tf.losses.mean_squared_error(regression_output,
                                    regression_label)
        tf.summary.scalar('regression_loss',regression_loss)
        all_loss_list.append(regression_loss)

        #Calculating the x-entropy loss(scalar)
        class_output=Z[:,regression_len:]#3,4,....
        class_label=Y[:,regression_len:]#TRUTH
        categorical_loss=tf.losses.softmax_cross_entropy(class_label,
                                                        class_output)
        tf.summary.scalar('categorical_loss',categorical_loss)
        all_loss_list.append(categorical_loss)

        #calculating the total loss
        total_loss=tf.add_n(all_loss_list,name='merge_loss')
        tf.summary.scalar('total_loss',total_loss)

    return total_loss


def model1(X,is_training):
    '''
    DESCRIPTION:
        Defining the first CNN model for the HGCAL. This function will
        take the input X and apply the CNN as defined in this model
        and give the unrectified output of the last layer as output
        for the computing the cost/error for training.

    USAGE:
        INPUT:
            X           : the input "image" to the model
            is_training : the training flag to know whether we are in
                            training mode or not which is required
                            for batch norm and dropout layers to modify
                            thier behaviour accordingly.
        OUTPUT:
            Z           : the output (un rectified output) of this
                            model to furthur compute the loss and hence
                            gradient.
    '''
    #Model Hyperparameter Section
    #(A separate handling mechanism will be developed for this later)
    bn_decision=False           #Batch-Normalization (ON/OFF)
    lambd=0.0                   #control L2-Regularization of weights
    dropout_rate=0.0            #The dropout rates of activation in any layer


    #Reshaping the input to appropriate format
    X_img=tf.expand_dims(X,axis=-1,name='channel_dim')

    #Passing it through the first layer
