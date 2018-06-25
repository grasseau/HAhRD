import tensorflow as tf

def _binary_parse_function(serialized_example_protocol):
    '''
    DESCRIPTION:
        This function is similar to the one used in io_pipeline
        file for the training.
        This will be used the map the data set which is a byte
        to the requires numerical format according the the
        example protocol we have saved them.
    USAGE:
        DONT use this function directly.
    '''
    features={
        'image':tf.FixedLenFeature((),tf.string),
    }
    parsed_feature=tf.parse_single_example(
                serialized_example_protocol,features)

    ###### THIS IS NEEDED TO BE CHANGe EACH TIME
    ####### NEW DATA SET IS CREATED
    height=514
    width=513
    depth=40

    image=tf.decode_raw(parsed_feature['image'],tf.float32)
    image.set_shape([depth*height*width])

    #Giving out the value in HWC format
    # image=tf.transpose(#i.e dont transpose its correct
    #         #we just reshape it intuitively. automatically the
    #         #row major format will be filled by reshape function
    #         tf.reshape(image,[height,width,depth]),[0,1,2]
    #     )
    image=tf.reshape(image,[height,width,depth])

    return image


def get_tf_records_iterator(filename_list):
    '''
    DESCRIPTION:
        This function will be used for getting the iterator for
        the event "image" stored in the form of tf records so that
        we dont have to separately store the numpy array for
        visualization.

        This format will be similar to the ones to be used during the
        training process.
    USAGE:
        INPUT:
            filename_list   : the list of name of the event "image"
                                stored in the tfrecords format.
        OUTPUT:
            next_element    : the one-shot-iterated next element
                                handle
    '''
    print '>>> Reading the images from tf records'
    #Creating the dataset from the tfrecords file
    dataset=tf.data.TFRecordDataset(filename_list,
                                        compression_type='ZLIB')

    #Decoding the binary data from the tfrecords
    dataset=dataset.map(_binary_parse_function)

    #Since we are not training we will make it one-shot
    iterator=dataset.make_one_shot_iterator()
    #since we are not in multigpu setting this next_element is enough
    next_element=iterator.get_next()

    return next_element
