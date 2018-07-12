import tensorflow as tf
import multiprocessing
ncpu=multiprocessing.cpu_count()

def _binary_parse_function_cifar(serialized_example_protocol):
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

def _binary_parse_function_image(serialized_example_protocol):
    '''
    DESCRIPTION:
        This function will be used to parse the binary data stored
        in tfrecords in the way the images are saved.
        Also the byte order in which data is saved is same as it
        is retreived by numpy tensorflow i.e
        Saving: depth->columns->rows
        Retreival: depth->columns->rows
        See the numpy docs of tobytes and tf docs of tf.reshape
    USAGE:
        INPUT:
            serialized_example_protocol : the binary serialized data

    DONT USE it directly. This function handle will be passed to the
    mapper function.
    '''
    #Parsing the binary data based on the "feature" we saved
    features={
        'image':tf.FixedLenFeature((),tf.string),
        'event':tf.FixedLenFeature((),tf.int64)
    }
    parsed_feature=tf.parse_single_example(
                    serialized_example_protocol,features)

    #Now decoding the raw binary features
    #Decoding the image feature
    height=514
    width=513
    depth=40

    image=tf.decode_raw(parsed_feature['image'],tf.float32)#BEWARE of dtype
    image.set_shape([depth*height*width])
    #Now reshape in usual way since reshape automatically read in c-order
    image=tf.reshape(image,[height,width,depth])

    #Decoding the event feature for check of sequential access
    event=tf.cast(parsed_feature['event'],tf.int32)

    return image,event

def _binary_parse_function_label(serialized_example_protocol):
    '''
    DESCRIPTION:
        This function will parse the label form the serialized
        example protocol.

    '''
    features={
        'label' :   tf.FixedLenFeature((),tf.string),
        'event' :   tf.FixedLenFeature((),tf.int64)
    }
    parsed_feature=tf.parse_single_example(serialized_example_protocol,
                                            features)

    #Now decoding the raw features to specified form
    target_len=5
    label=tf.decode_raw(parsed_feature['label'],tf.float32)
    label.set_shape([target_len])
    label=tf.reshape(label,[target_len,])

    #Decoding the event id for seq access check
    event=tf.cast(parsed_feature['event'],tf.int32)

    return label,event

def _binary_parse_function_example(serialized_example_protocol):
    '''
    DESCRIPTION:
        This function will deserialize, decompress and then transform
        the image and label in the appropriate shape based on the (new) merged
        structure of the dataset.
    '''
    #Parsing the exampe from the binary format
    features={
        'image':    tf.FixedLenFeature((),tf.string),
        'label':    tf.FixedLenFeature((),tf.string)
    }
    parsed_feature=tf.parse_single_example(serialized_example_protocol,
                                            features)

    #Now setting the appropriate tranformation (decoding and reshape)
    height=514
    width=513
    depth=40
    #Decoding the image from biary
    image=tf.decode_raw(parsed_feature['image'],tf.float32)#BEWARE of dtype
    image.set_shape([depth*height*width])
    #Now reshape in usual way since reshape automatically read in c-order
    image=tf.reshape(image,[height,width,depth])

    #Now decoding the label
    target_len=6
    label=tf.decode_raw(parsed_feature['label'],tf.float32)
    label.set_shape([target_len])
    #Reshaping appropriately
    label=tf.reshape(label,[target_len,])

    #Returing the example tuple finally
    return image,label

################# TRAIN DATASET PIPELINE #####################
def parse_tfrecords_file_v1(train_image_filename_list,train_label_filename_list,
                        test_image_filename_list,test_label_filename_list,
                        mini_batch_size,buffer_size=5000):
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
            iterator             : an handle of iterator
            train_iter_init_op   : an op which need to be run to make iterator
                                    point to start of training dataset
            test_iter_init_op    : an op which when run in a session will point
                                    point to start of the testing dataset
    '''
    #Reading the tfRecords File
    comp_type='ZLIB'
    #Reading the training dataset
    train_dataset_image=tf.data.TFRecordDataset(train_image_filename_list,
                                compression_type=comp_type,num_parallel_reads=ncpu/2)
    train_dataset_label=tf.data.TFRecordDataset(train_label_filename_list,
                                compression_type=comp_type,num_parallel_reads=ncpu/2)
    #Applying the apropriate transformation to map from binary
    train_dataset_image=train_dataset_image.map(_binary_parse_function_image,
                                                num_parallel_calls=ncpu/2)
    train_dataset_label=train_dataset_label.map(_binary_parse_function_label,
                                                num_parallel_calls=ncpu/2)
    #Stitching the image and label together
    train_dataset=tf.data.Dataset.zip((train_dataset_image,
                                        train_dataset_label))

    #Reading the test dataset
    test_dataset_image=tf.data.TFRecordDataset(test_image_filename_list,
                                compression_type=comp_type,num_parallel_reads=ncpu/2)
    test_dataset_label=tf.data.TFRecordDataset(test_label_filename_list,
                                compression_type=comp_type,num_parallel_reads=ncpu/2)
    #print '\ndirect binary'
    #print (train_dataset.output_types,train_dataset.output_shapes)
    #Applying the appropriate transformation to map from binary
    test_dataset_image=test_dataset_image.map(_binary_parse_function_image,
                                        num_parallel_calls=ncpu/2)
    test_dataset_label=test_dataset_label.map(_binary_parse_function_label,
                                        num_parallel_calls=ncpu/2)
    #Stitching the image and label together
    test_dataset=tf.data.Dataset.zip((test_dataset_image,
                                        test_dataset_label))
    #print '\nparsed data'
    #print (train_dataset.output_types,train_dataset.output_shapes)

    #Shuffling the data before creating the minibatch
    train_dataset=train_dataset.shuffle(buffer_size=buffer_size)
    test_dataset=test_dataset.shuffle(buffer_size=buffer_size)

    #Now creating appropraite batches from the tupled element
    train_dataset=train_dataset.batch(mini_batch_size)
    test_dataset=test_dataset.batch(mini_batch_size)
    #print '\nbatched data'
    #print (train_dataset.output_types,train_dataset.output_shapes)

    #Adding the prefetching so that the above steps are pipelined with below
    train_dataset=train_dataset.prefetch(4)
    test_dataset=test_dataset.prefetch(4)

    #now creating the re_initializable iterator
    iterator=tf.data.Iterator.from_structure(
                            train_dataset.output_types,
                            train_dataset.output_shapes)
    next_element=iterator.get_next()

    #Also getting the iterator initialization op
    train_iter_init_op=iterator.make_initializer(train_dataset)
    test_iter_init_op=iterator.make_initializer(test_dataset)

    #Returning the required elements (dont return next element return iterator)
    return iterator,train_iter_init_op,test_iter_init_op

def parse_tfrecords_file(train_filename_list,test_filename_list,
                        mini_batch_size,shuffle_buffer_size):
    '''
    DESCRIPTION:
        This function will create the piepline based on the merged example
        format where a single file contains both labels and images.
    USAGE:
        INPUT:
            train_filename_list : the list of filename to create training
                                    dataset
            test_filename_list  : the list of the filename to create test
                                    dataset
            mini_batch_size     : the batch size of the dataset
            shuffle_buffer_size : the buffere size to shuffle data from
                                    (here shuffling will be donr on the filename)
        OUTPUT:

    '''
    comp_type='ZLIB'
    #Creating the training dataset
    train_dataset=tf.data.TFRecordDataset(train_filename_list,
                                        compression_type=comp_type,
                                        num_parallel_reads=20)
    #Shuffling the file list
    train_dataset=train_dataset.shuffle(buffer_size=shuffle_buffer_size)
    #Mapping the parser function on file on each element to decode them
    train_dataset=train_dataset.map(_binary_parse_function_example,
                                    num_parallel_calls=ncpu)
    #Batching the dataset
    train_dataset=train_dataset.batch(mini_batch_size)
    #Prefetching the batches
    train_dataset=train_dataset.prefetch(4)


    #Now making the re-initializable iterator
    iterator=tf.data.Iterator.from_structure(
                                        train_dataset.output_types,
                                        train_dataset.output_shapes)
    train_iter_init_op=iterator.make_initializer(train_dataset)

    return iterator,train_iter_init_op,None



################ INFERENCE DATASET PIPELINE #################
def parse_tfrecords_file_inference(test_image_filename_list,
                                    test_label_filename_list,
                                    mini_batch_size):
    '''
    DESCRIPTION:
        This function will make the one-shot iterator for runnning
        the imference on the test/validation set ,without any
        shuffling etc.
    USAGE:
        INPUTS:
            test_image_filename_list : the list of tfrecords containing
                                        the images for inference
            test_label_filename_list : the list of tfrecords with
                                        the labels of the corresponding images
            mini_batch_size          : since this time no gradient ops
                                        will be made on graph, we could have
                                        bigger batch size to parallely make
                                        inference
        OUTPUT:
            one_shot_iterator       : the one-shot iterator for the dataset
    '''
    #Reading the tfrecords
    comp_type='ZLIB'
    test_dataset_image=tf.data.TFRecordDataset(test_image_filename_list,
                                    compression_type=comp_type,
                                    num_parallel_reads=ncpu-2)
    test_dataset_label=tf.data.TFRecordDataset(test_label_filename_list,
                                    compression_type=comp_type,
                                    num_parallel_reads=ncpu-2)
    #Decoding the binary file to numberical format
    test_dataset_image=test_dataset_image.map(_binary_parse_function_image,
                                                num_parallel_calls=ncpu-2)
    test_dataset_label=test_dataset_label.map(_binary_parse_function_label,
                                                num_parallel_calls=ncpu-2)

    #Now Zipping them together to make on combined example dataset
    test_dataset=tf.data.Dataset.zip((test_dataset_image,
                                    test_dataset_label))

    #Making the batches for parallel inference
    test_dataset=test_dataset.batch(mini_batch_size)

    #Finally making the one-shot iterator
    one_shot_iterator=test_dataset.make_one_shot_iterator()

    return one_shot_iterator
