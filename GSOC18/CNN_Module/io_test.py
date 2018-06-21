import tensorflow as tf
from io_pipeline import *
import sys

if __name__=='__main__':
    base_path='/home/abhinav/Desktop/HAhRD/GSOC18/GeometryUtilities-master/interpolation/image_data/'
    #Giving the train filename list
    train_image_filename_list=[base_path+'image0batchsize20zside0.tfrecords']
    train_label_filename_list=[base_path+'label0batchsize20.tfrecords']

    #Giving the test filename list
    test_image_filename_list=[base_path+'image0batchsize20zside1.tfrecords']
    test_label_filename_list=[base_path+'label0batchsize20.tfrecords']

    mini_batch_size=1
    buffer_size=10*mini_batch_size

    #Getting the iterator and the training and test set
    iterator,train_iter_init_op,test_iter_init_op=parse_tfrecords_file(
                                    train_image_filename_list,
                                    train_label_filename_list,
                                    test_image_filename_list,
                                    test_label_filename_list,
                                    mini_batch_size,
                                    buffer_size
                                    )

    with tf.Session() as sess:
        sess.run(train_iter_init_op)
        while True:
            examples=sess.run(iterator.get_next())
            # print '\n Printing examples:'
            # print examples[0][0].shape
            # print 'eventid:',examples[0][1]
            # print examples[1][0]
            # print 'eventid:',examples[1][1]

            print examples[0][1]==examples[1][1]
            #sys.exit(1)
