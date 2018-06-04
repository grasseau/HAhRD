import tensorflow as tf

def _binary_parse_function(serialized_example_protocol):
    '''
    DESCRIPTION:
        This is a parser function to convert the binary data read
        from the tfrecords file which were originally save in
        "example" protocol, and reconvert them to tensors to be
        ready to use by model.

    USAGE:
        Dont use this function directly.
    '''
    features=\
            {'image': tf.FixedLenFeature((),tf.string),
             'label': tf.FixedLenFeature((),tf.int64)}
    parsed_feature=tf.parse_single_example(
                        serialized_example_protocol,features)

    #Perform required transformation to bring the flattened
    #data to appropriate shape
    #testing for cifar
    height=32
    width=32
    depth=3

    image=tf.decode_raw(parsed_feature['image'],tf.uint8)
    image.set_shape([depth*height*width])

    #Reshaping
    image=tf.cast(
            tf.transpose(
                tf.reshape(image,[depth,height,width]),[1,2,0]),
            tf.float32)
    label=tf.cast(parsed_feature['label'],tf.int32)

    #returning the example as a tuple. So each "element" is a tuple
    return image,label

def parse_tfrecords_file(train_filename_list,test_filename_list,mini_batch_size):
    '''
    DESCRIPTION:
        This function will read the dataset stored in tfrecords
        file and return an re-initializable iterator and
        corresponding
            1.train_iter_init_op and,
            2.test_iter_init_op
    USAGE:
        INPUT:
            mini_batch_size      : the size of minibatch to extract each time
            train_filename_list  : list of tfrecords name for training
            test_filename_list   : list of tfrecords name for testing
        OUPUT:
            next_mini_batch      : an handle of iterator.get_next()
            train_iter_init_op   : an op which need to be run to make iterator
                                    point to start of training dataset
            test_iter_init_op    : an op which when run in a session will point
                                    point to start of the testing dataset
    '''
    #Reading the tfRecords File
    train_dataset=tf.data.TFRecordDataset(train_filename_list)
    test_dataset =tf.data.TFRecordDataset(test_filename_list)
    #print '\ndirect binary'
    #print (train_dataset.output_types,train_dataset.output_shapes)

    #Applying appropriate decoding from binary
    train_dataset=train_dataset.map(_binary_parse_function)
    test_dataset=test_dataset.map(_binary_parse_function)
    #print '\nparsed data'
    #print (train_dataset.output_types,train_dataset.output_shapes)

    #Now creating appropraite batches from the tupled element
    train_dataset=train_dataset.batch(mini_batch_size)
    test_dataset=test_dataset.batch(mini_batch_size)
    #print '\nbatched data'
    #print (train_dataset.output_types,train_dataset.output_shapes)

    #now creating the re_initializable iterator
    iterator=tf.data.Iterator.from_structure(
                            train_dataset.output_types,
                            train_dataset.output_shapes)
    next_element=iterator.get_next()

    #Also getting the iterator initialization op
    train_iter_init_op=iterator.make_initializer(train_dataset)
    test_iter_init_op=iterator.make_initializer(test_dataset)

    #Returning the required elements
    return next_element,train_iter_init_op,test_iter_init_op
