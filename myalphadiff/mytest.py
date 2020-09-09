#fr[om keras.preprocessing.image import load_img, img_to_array, array_to_img, ImageDataGenerator
#from keras.models import Sequential
#from keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense, Dropout
#from keras import backend as K
#from keras.layers import Input, Lambda
#from keras.models import Model
#from keras.optimizers import RMSprop
#import keras.backend as K
#from keras.utils import np_utils
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
#config.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=config))

from tensorflow.contrib import keras
from tensorflow.contrib.keras import preprocessing
from tensorflow.contrib.keras import models
from tensorflow.contrib.keras import layers
from tensorflow.contrib.keras import optimizers
from tensorflow.contrib.keras import backend as K
from tensorflow.contrib.keras import utils
import numpy as np
import matplotlib.pyplot as plt
#import tensorflow as tf
#import generator
import os
import pyflann
import math
from scipy import spatial
import json
import sys


batch_size = 64
epochs = 100
input_shape = (100, 100, 1)
input_shape_degree = (2, )
input_shape_callseq = (4200, )
train_samples = 400000
val_samples = 80000
test_samples = 72075

#name = 'siamese_fc_' + str(400000) + '-' + str(80000) + '_epoch-' + str(50) + '_batch-' + str(128) + '_Conv-20-5-50-5-80-5-80-5-Fc-500-1024'
name_avg = 'named-avgpool-siamese_fc_' + str(800000) + '-' + str(200000) + '_epoch-' + str(65) + '_batch-' + str(16) + '_Conv-20-5-50-5-80-5-80-5-Fc-500-1024'
name_offline = 'offline-named-avgpool-siamese_fc_' + str(800000) + '-' + str(200000) + '_epoch-' + str(62) + '_batch-' + str(16) + '_Conv-20-5-50-5-80-5-80-5-Fc-500-1024'
name_fileunit = 'fileunit-named-avgpool-siamese_fc_' + str(800000) + '-' + str(200000) + '_epoch-' + str(65) + '_batch-' + str(16) + '_Conv-20-5-50-5-80-5-80-5-Fc-500-1024'
name_file_online = 'online-fileunit-named-avgpool-siamese_fc_' + str(800000) + '-' + str(200000) + '_epoch-' + str(200) + '_batch-' + str(16) + '_Conv-20-5-50-5-80-5-80-5-Fc-500-1024'
name_file_online_fromzero = 'online-fileunit-fromzero-named-avgpool-siamese_fc_' + str(800000) + '-' + str(200000) + '_epoch-' + str(100) + '_batch-' + str(16) + '_Conv-20-5-50-5-80-5-80-5-Fc-500-1024'
name_file_online_fromzero_maxpool = 'online-fileunit-fromzero-named-maxpool-siamese_fc_' + str(800000) + '-' + str(200000) + '_epoch-' + str(100) + '_batch-' + str(16) + '_Conv-20-5-50-5-80-5-80-5-Fc-500-1024'
name_bigger = 'bigger-named-avgpool-siamese_fc_' + str(800000) + '-' + str(200000) + '_epoch-' + str(50) + '_batch-' + str(16) + '_Conv-20-3-50-3-50-5-80-5-80-5-80-5-100-5-Fc-500-1024'
name_more_kernels = 'more-kernels-named-avgpool-siamese_fc_' + str(800000) + '-' + str(200000) + '_epoch-' + str(30) + '_batch-' + str(16) + '_Conv-32-3-64-5-96-5-96-5-Fc-500-1024' 
name_more_kernels_fileunit_online = 'more-kernels-fileunit-online-named-avgpool-siamese_fc_' + str(800000) + '-' + str(200000) + '_epoch-' + str('21+10+50+10+20+50') + '_batch-' + str(16) + '_Conv-32-3-64-5-96-5-96-5-Fc-500-1024'
name_more_kernels_fileunit_online_crossval = 'more-kernels-fileunit-online-crossval-named-avgpool-siamese_fc_' + str(800000) + '-' + str(200000) + '_epoch-' + str('21+90+60') + '_batch-' + str(16) + '_Conv-32-3-64-5-96-5-96-5-Fc-500-1024'
name_more_kernels_fileunit_func_online = 'more-kernels-fileunit-func_online-named-avgpool-siamese_fc_' + str(800000) + '-' + str(200000) + '_epoch-' + str('21+140+5') + '_batch-' + str(16) + '_Conv-32-3-64-5-96-5-96-5-Fc-500-1024'
name_more_kernels_deeper = 'more-kernels-fileunit-func_online-named-avgpool-siamese_fc_' + str(800000) + '-' + str(200000) + '_epoch-' + str('9+1') + '_batch-' + str(16) + '_Conv-32-3-32-3-64-3-64-3-96-3-96-3-96-3-96-3-Fc-500-1024'
name_more_kernels_deeper_newdata = 'newdata-more-kernels-fileunit-func_online-named-avgpool-siamese_fc_' + str(800000) + '-' + str(200000) + '_epoch-' + str('15+6') + '_batch-' + str(16) + '_Conv-32-3-32-3-64-3-64-3-96-3-96-3-96-3-96-3-Fc-500-1024'
name_more_kernels_deeper_newdata_BN = 'BN-newdata-more-kernels-fileunit-func_online-named-avgpool-siamese_fc_' + str(800000) + '-' + str(200000) + '_epoch-' + str('15') + '_batch-' + str(16) + '_Conv-32-3-32-3-64-3-64-3-96-3-96-3-96-3-96-3-Fc-500-1024'
name_more_kernels_deeper_wholedata_BN = 'wholedata-BN-newdata-more-kernels-fileunit-func_online-named-avgpool-siamese_fc_' + str(800000) + '-' + str(200000) + '_epoch-' + str('25') + '_batch-' + str(16) + '_Conv-32-3-32-3-64-3-64-3-96-3-96-3-96-3-96-3-Fc-500-1024'
name_more_kernels_deeper_wholedata_BN = 'wholedata-BN-newdata-more-kernels-fileunit-func_online-named-avgpool-siamese_fc_800000-200000_epoch-50_batch-16_Conv-32-3-32-3-64-3-64-3-96-3-96-3-96-3-96-3-Fc-500-1024epoch-07-0.01611'
#name_more_kernels_deeper_wholedata_BN = 'wholedata-BN-newdata-more-kernels-fileunit-func_online-named-avgpool-siamese_fc_800000-200000_epoch-69_batch-16_Conv-32-3-32-3-64-3-64-3-96-3-96-3-96-3-96-3-Fc-500-1024epoch-00-0.01720'
name_more_kernels_deeper_wholedata_BN = 'wholedata-BN-newdata-more-kernels-fileunit-func_online-named-avgpool-siamese_fc_800000-200000_epoch-70_batch-16_Conv-32-3-32-3-64-3-64-3-96-3-96-3-96-3-96-3-Fc-500-1024epoch-05-0.01682'
name_more_kernels_deeper_wholedata_BN_triple = 'triple-wholedata-BN-newdata-more-kernels-fileunit-func_online-named-avgpool-siamese_fc_' + str(800000) + '-' + str(200000) + '_epoch-' + str('10+2') + '_batch-' + str(16) + '_Conv-32-3-32-3-64-3-64-3-96-3-96-3-96-3-96-3-Fc-500-1024'
name_more_kernels_deeper_wholedata_BN_padding = 'paddiing0-wholedata-BN-newdata-more-kernels-fileunit-func_online-named-avgpool-siamese_fc_' + str(800000) + '-' + str(200000) + '_epoch-' + str('15+1') + '_batch-' + str(16) + '_Conv-32-3-32-3-64-3-64-3-96-3-96-3-96-3-96-3-Fc-500-1024'
name_more_kernels_deeper_wholedata_BN_padding_start = 'fromstart-paddiing0-wholedata-BN-newdata-more-kernels-fileunit-func_online-named-avgpool-siamese_fc_' + str(800000) + '-' + str(200000) + '_epoch-' + str('15') + '_batch-' + str(16) + '_Conv-32-3-32-3-64-3-64-3-96-3-96-3-96-3-96-3-Fc-500-1024'
name_more_kernels_deeper_wholedata_BN_padding_start = 'fromstart-paddiing0-wholedata-BN-newdata-more-kernels-fileunit-func_online-named-avgpool-siamese_fc_800000-200000_epoch-25_batch-16_Conv-32-3-32-3-64-3-64-3-96-3-96-3-96-3-96-3-Fc-500-1024epoch-03-0.02'
name_more_kernels_deeper_wholedata_BN_padding_start_savebest = 'fromstart-paddiing0-wholedata-BN-newdata-more-kernels-fileunit-func_online-named-avgpool-siamese_fc_800000-200000_epoch-40_batch-16_Conv-32-3-32-3-64-3-64-3-96-3-96-3-96-3-96-3-Fc-500-1024epoch-01-0.01949'
name_more_kernels_deeper_wholedata_BN_padding_start_savebest = 'fromstart-paddiing0-wholedata-BN-newdata-more-kernels-fileunit-func_online-named-avgpool-siamese_fc_800000-200000_epoch-53_batch-16_Conv-32-3-32-3-64-3-64-3-96-3-96-3-96-3-96-3-Fc-500-1024epoch-08-0.01688'
#name_more_kernels_deeper_wholedata_BN_padding_start_savebest  = 'fromstart-paddiing0-wholedata-BN-newdata-more-kernels-fileunit-func_online-named-avgpool-siamese_fc_800000-200000_epoch-70_batch-16_Conv-32-3-32-3-64-3-64-3-96-3-96-3-96-3-96-3-Fc-500-1024epoch-16-0.01818'
name_more_kernels_deeper_wholedata_BN_triple_savebest = 'triple-fromstart-paddiing0-wholedata-BN-newdata-more-kernels-fileunit-func_online-named-avgpool-siamese_fc_800000-200000_epoch-50_batch-200_Conv-32-3-32-3-64-3-64-3-96-3-96-3-96-3-96-3-Fc-500-1024epoch-35-0.02844'
name_more_kernels_deeper_wholedata_BN_triple_coreutils = 'coreutils-triple-fromstart-paddiing0-wholedata-BN-newdata-more-kernels-fileunit-func_online-named-avgpool-siamese_fc_15757-15757_epoch-50_batch-200_Conv-32-3-32-3-64-3-64-3-96-3-96-3-96-3-96-3-Fc-500-1024epoch-42-0.04618'
name_more_kernels_deeper_wholedata_BN_triple_openssl = 'openssl-triple-fromstart-paddiing0-wholedata-BN-newdata-more-kernels-fileunit-func_online-named-avgpool-siamese_fc_15757-15757_epoch-50_batch-200_Conv-32-3-32-3-64-3-64-3-96-3-96-3-96-3-96-3-Fc-500-1024epoch-02-0.09033'
name_more_kernels_deeper_wholedata_BN_online_openssl = 'openssl-online-fromstart-paddiing0-wholedata-BN-newdata-more-kernels-fileunit-func_online-named-avgpool-siamese_fc_15757-15757_epoch-50_batch-16_Conv-32-3-32-3-64-3-64-3-96-3-96-3-96-3-96-3-Fc-500-1024epoch-41-0.01298'
name_evalutation_triple_embedding_2014_prefix = 'triple-fromstart-paddiing0-wholedata-BN-newdata-more-kernels-fileunit-func_online-named-avgpool-siamese_fc_800000-200000_epoch-50_batch-200_Conv-32-3-32-3-64-3-64-3-96-3-96-3-96-3-96-3-Fc-500-1024epoch-'
name_prefix = 'triple-fromstart-paddiing0-wholedata-BN-newdata-more-kernels-fileunit-func_online-named-avgpool-siamese_fc_800000-200000_epoch-30_batch-200_Conv-32-3-32-3-64-3-64-3-96-3-96-3-96-3-96-3-Fc-500-64epoch-'
name_prefix = 'triple-fromstart-paddiing0-wholedata-BN-newdata-more-kernels-fileunit-func_online-named-avgpool-siamese_fc_800000-200000_epoch-20_batch-200_Conv-32-3-32-3-64-3-64-3-96-3-96-3-96-3-96-3-Fc-500-256epoch-'
name_prefix = 'maxpool-triple-fromstart-paddiing0-wholedata-BN-newdata-more-kernels-fileunit-func_online-named-avgpool-siamese_fc_800000-200000_epoch-20_batch-200_Conv-32-3-32-3-64-3-64-3-96-3-96-3-96-3-96-3-Fc-500-1024epoch-'
#name_more_kernels_deeper_wholedata_BN_padding_start_savebest  = 'fromstart-paddiing0-wholedata-BN-newdata-more-kernels-fileunit-func_online-named-avgpool-siamese_fc_800000-200000_epoch-70_batch-16_Conv-32-3-32-3-64-3-64-3-96-3-96-3-96-3-96-3-Fc-500-1024epoch-35-0.01580'
#name_more_kernels_deeper_wholedata_BN_padding_start_savebest = 'fromstart-paddiing0-wholedata-BN-newdata-more-kernels-fileunit-func_online-named-avgpool-siamese_fc_800000-200000_epoch-100_batch-16_Conv-32-3-32-3-64-3-64-3-96-3-96-3-96-3-96-3-Fc-500-1024epoch-03-0.01570'
name_more_kernels_deeper_wholedata_BN_padding_start_savebest_reonline = 'fromstart-paddiing0-reonline-wholedata-BN-newdata-more-kernels-fileunit-func_online-named-avgpool-siamese_fc_800000-200000_epoch-30_batch-16_Conv-32-3-32-3-64-3-64-3-96-3-96-3-96-3-96-3-Fc-500-1024epoch-01-0.01324'
name_more_kernels_deeper_wholedata_BN_context_online_savebest = 'context-online-fromstart-paddiing0-wholedata-BN-newdata-more-kernels-fileunit-func_online-named-avgpool-siamese_fc_800000-200000_epoch-50_batch-20_Conv-32-3-32-3-64-3-64-3-96-3-96-3-96-3-96-3-Fc-500-1024epoch-01-0.02335'
name_more_kernels_deeper_wholedata_BN_context_triple_savebest = 'context-triple-fromstart-paddiing0-wholedata-BN-newdata-more-kernels-fileunit-func_online-named-avgpool-siamese_fc_800000-200000_epoch-50_batch-200_Conv-32-3-32-3-64-3-64-3-96-3-96-3-96-3-96-3-Fc-500-1024epoch-07-0.31367'
name_more_kernels_deeper_error50 = 'error50-paddiing0-BN-newdata-more-kernels-fileunit-func_online-named-avgpool-siamese_fc_800000-200000_epoch-40_batch-16_Conv-32-3-32-3-64-3-64-3-96-3-96-3-96-3-96-3-Fc-500-1024epoch-16-0.01669'
name_prefix = 'vgg_hard_neg_pairs_epoch-30-batch-100-embedding-128-maxpoolepoch-'
name_prefix = 'epoch-200-batch-100-dense2-512-embedding128epoch-'
#print(name)


def create_base_network(input_shape):
    #with tf.device('/cpu:0'):
    seq = models.Sequential()
    # CONV => RELU => POOL
    #with tf.device('/gpu:0'):
    seq.add(layers.Conv2D(32, kernel_size=3, padding="same", input_shape=input_shape))
    seq.add(layers.BatchNormalization(axis=3))
    seq.add(layers.Activation("relu"))
    seq.add(layers.Conv2D(32, kernel_size=3, padding="same", input_shape=input_shape))
    seq.add(layers.BatchNormalization(axis=3))
    seq.add(layers.Activation("relu"))
    seq.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    #seq.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
    # CONV => RELU => POOL
    #with tf.device('/gpu:1'):
    seq.add(layers.Conv2D(64, kernel_size=3, padding="same"))
    seq.add(layers.BatchNormalization(axis=3))
    seq.add(layers.Activation("relu"))
    seq.add(layers.Conv2D(64, kernel_size=3, padding="same"))
    seq.add(layers.BatchNormalization(axis=3))
    seq.add(layers.Activation("relu"))
    seq.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    #seq.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
    # CONV => RELU => POOL
    seq.add(layers.Conv2D(96, kernel_size=3, padding='same'))
    seq.add(layers.BatchNormalization(axis=3))
    seq.add(layers.Activation('relu'))
    seq.add(layers.Conv2D(96, kernel_size=3, padding='same'))
    seq.add(layers.BatchNormalization(axis=3))
    seq.add(layers.Activation('relu'))
    seq.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
    #seq.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
    # CONV => RELU => POOL
    seq.add(layers.Conv2D(96, kernel_size=3, padding='same'))
    seq.add(layers.BatchNormalization(axis=3))
    seq.add(layers.Activation('relu'))
    seq.add(layers.Conv2D(96, kernel_size=3, padding='same'))
    seq.add(layers.BatchNormalization(axis=3))
    seq.add(layers.Activation('relu'))
    seq.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
    #seq.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
    # Flatten => RELU
    #seq.add(Flatten())
    #with tf.device('/gpu:2'):
    seq.add(layers.Dense(512, activation="relu"))
    seq.add(layers.Flatten())
    seq.add(layers.Dense(128))
    #seq.add(Dropout(0.1))
    #seq.add(Dense(500))
    #seq.add(Dense(500)) 

    return seq

def create_network(input_shape_img, input_shape_degree, input_shape_callseq, weight):
    input_img = layers.Input(shape=input_shape_img)
    base_network = create_base_network(input_shape_img)
    embedding_img = base_network(input_img)
    #print embedding_img, embedding_img.shape

    base_model = models.Model(inputs=[input_img], outputs=embedding_img)
    base_model.load_weights(weight)

    model = models.Model(inputs=input_img, outputs=embedding_img)
    model.summary()
    return model


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0],1)


def cosine_distance(vecs, normalize=False):
    x, y = vecs
    if normalize:
        x = K.l2_normalize(x, axis=0)
        y = K.l2_normalize(x, axis=0)
    return K.prod(K.stack([x, y], axis=1), axis=1)

def cosine_distance_output_shape(shapes):
    return shapes[0]

def compute_accuracy(preds, labels):
    return labels[preds.ravel() < 0.5].mean()

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    # print("y_true", y_true, "y_pred", y_pred[0])
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

weight_file = "epoch-200-batch-100-dense2-512-embedding128epoch-199-0.09604.h5"
def test(bechmark_path,func_name,top_k=5):

    #======================================
    # load model
    #======================================
    base_network = create_network(input_shape, input_shape_degree, input_shape_callseq, weight_file)
    image_left = layers.Input(shape=input_shape)
    degree_left = layers.Input(shape=input_shape_degree)
    callseq_left = layers.Input(shape=input_shape_callseq)
    vector_left = base_network(image_left)
    model = models.Model(inputs=image_left, outputs=vector_left)
    model.summary()
    rms = optimizers.RMSprop()
    model.compile(loss=contrastive_loss, optimizer=rms)

    #======================================
    # statis variables
    #======================================
    sum_rate = 0.0
    sum_nn_rate = 0.0
    sum_rate5 = 0.0
    statis_count = 0
    statis = []
    f_statis = open('testset-num>=100_maxpool_batch100_vgghardneg_epoch30_recall.log', 'w')
    
    #======================================
    # iterate each binary pair of beachmark
    #======================================
    for diffdir in os.listdir(bechmark_path):
        pre_dir = os.path.join(bechmark_path, diffdir, 'A')
        post_dir = os.path.join(bechmark_path, diffdir, 'B')

        #============================
        # 'signature' file constains inter-func feature & inter-mod feature data,
        # which is necessary for using feature 2 and 3.
        # Here, parsing signature file's content
        #============================
        if not os.path.isfile(os.path.join(pre_dir, 'signature')) or \
            not os.path.isfile(os.path.join(post_dir, 'signature')):
            print("nnnnnnnnnnnnnnnnnnno")
            print(os.path.join(pre_dir, 'signature'))
            return
            continue
        context_a = {}
        with open(os.path.join(pre_dir, 'signature'), 'r') as f:
            count = 0
            for line in f:
                if count == 0 or '::' in line[:-2]:
                    count = count + 1
                    continue
                elems = line[:-2].split(':')
                context_a[elems[0]] = ([int(d) for d in elems[1].split('##')[0].split(' ') if d != ''], [int(c) for c in elems[1].split('##')[1][1:-1].split(', ') if c != ''])
        context_b = {}    
        with open(os.path.join(post_dir, 'signature'), 'r') as f:
            count = 0
            for line in f:
                if count == 0 or '::' in line[:-2]:
                    count = count + 1
                    continue
                elems = line[:-2].split(':')
                context_b[elems[0]] = ([int(d) for d in elems[1].split('##')[0].split(' ') if d != ''], [int(c) for c in elems[1].split('##')[1][1:-1].split(', ') if c != ''])

        #===================================
        #  Parsing the directory of each binary pair, which is stored in a folder.
        #  In the folder, each function's raw byte is stored in a file
        # ==================================
        file_paths = []
        for root, _, files in os.walk(pre_dir):
            for fname in files:
                if fname == 'pairs' or fname == 'signature':
                    continue
                if fname is None:
                    continue
                file_paths.append(os.path.join(root, fname))

        file_paths_newer = []
        for root, _, files in os.walk(post_dir):
            for fname in files:
                if fname == 'pairs' or fname == 'signature':
                    continue
                if fname is None:
                    continue
                file_paths_newer.append(os.path.join(root, fname))
        
        total_of_pre = len(file_paths)
        total_of_post = len(file_paths_newer)
        #if total_of_pre < 2 or total_of_post < 2:
        #    continue
        #print "lenth of primary", len(file_paths), len(file_paths_newer)

        #======================================
        # convert each feature into a (embedding) vector
        #   intra-func: rawByte --> embedding (CNN)
        #   inter-func: (IN, OUT) --> 2-d V
        #   inter-module: set of common IMPORTs --> n-d V 
        #======================================
        result = []
        count = 0

        index_A = []
        i = 0
        for fpath in file_paths:
            fname = fpath.split('/')[-1]
            ##size = int(fname.split('##')[3].split('*')[0])
            size = int(fname.split('_')[-1].split('x')[0])
            funcname = fpath.split('/')[-1].split("_0x")[0]
            ##funcname = fpath.split('/')[-1].split("##")[2]
            if size <= 2:
                #print '[size <= 2]', fpath.split('/')[-1].split("_0x")[0]
                continue
            if funcname not in context_a.keys():
                #print '[not in context]', funcname
                continue
            #===================
            # intra-func: rawByte --> embedding
            #===================
            image_x = preprocessing.image.load_img(fpath, grayscale='grayscale', target_size=(100, 100))
            image_x = preprocessing.image.img_to_array(image_x, data_format='channels_last')
            image_x = image_x.reshape(100 * 100)
            image_x[size*size:] = 0
            image_x = image_x / 255
            image_x = image_x.reshape(1, 100, 100, 1)
            embedding = model.predict(image_x)

            #=========================
            # inter-func feature
            #=========================
            degree_input = np.array(context_a[funcname][0])
            degree_input = degree_input.reshape((1, 2))
            ctx_input = np.zeros(4200)
            for c in context_a[funcname][1]:
                ctx_input[c] = 1
            ctx_input = ctx_input.reshape((1, 4200))
                       
            index_A.append(funcname)

            if i == 0:
                testset = embedding
            else:
                testset = np.concatenate((testset, embedding), axis=0)
            i = i + 1

        index_B = []
        i = 0
        for fpath in file_paths_newer:
            fname = fpath.split('/')[-1]
            ##size = int(fname.split('##')[3].split('*')[1].split('.jpeg')[0])
            funcname = fpath.split('/')[-1].split("_0x")[0]
            ##funcname = fpath.split('/')[-1].split("##")[2]
            size = int(fname.split('_')[-1].split('x')[0])
            if size <= 2:
                continue
            if funcname not in context_b.keys():
                continue
            #=========================
            # intra-func feature
            #=========================
            image_x = preprocessing.image.load_img(fpath, grayscale='grayscale', target_size=(100, 100))
            image_x = preprocessing.image.img_to_array(image_x, data_format='channels_last')
            image_x = image_x.reshape(100 * 100)
            image_x[size*size:] = 0
            image_x = image_x / 255
            image_x = image_x.reshape(1, 100, 100, 1)
            embedding = model.predict(image_x)
            
            #===========================
            # inter-func feature
            #============================
            degree_input = np.array(context_b[funcname][0])
            degree_input = degree_input.reshape((1, 2))

            #============================
            # inter-mod feature
            #============================
            ctx_input = np.zeros(4200)
            for c in context_b[funcname][1]:
                ctx_input[c] = 1
            ctx_input = ctx_input.reshape((1, 4200))

            index_B.append(funcname)
            if i == 0:
                dataset = embedding
            else:
                dataset = np.concatenate((dataset, embedding), axis=0)
            i = i + 1

        #=======================
        # total counter used for percentage and recall computation
        #=======================
        total_of_pre = len(testset)
        total_of_post = len(dataset) 
        total = 0
        for func_a in index_A:
            if func_a in index_B:
                total = total + 1
        if total <= 0:
            continue

        #====================================
        # For each function in A, computing the distance between it and each function of B
        # the distance = intra-func distance + inter-func distance + inter-mod distance
        #=====================================
        s_result = []
        for n_i in range(0, len(index_A)):
            n_result = []
            for n_j in range(0, len(index_B)):                
                dist_degree = spatial.distance.euclidean(np.array(context_a[index_A[n_i]][0]), np.array(context_b[index_B[n_j]][0]))
                
                ctx_A = np.zeros(4200)
                for c in context_a[index_A[n_i]][1]:
                    ctx_A[c] = 1
                ctx_B = np.zeros(4200)
                for c in context_b[index_B[n_j]][1]:
                    ctx_B[c] = 1            
                dist_ctx = spatial.distance.euclidean(ctx_A, ctx_B)
                        
                dist_img = spatial.distance.euclidean(testset[n_i], dataset[n_j]) 

                value = dist_img + dist_ctx  + (1 - math.pow((3.0/4.0), dist_degree)) #dist_img #+ dist_ctx + (1 - math.pow((3.0/4.0), dist_degree))
                n_result.append((index_B[n_j], value))

            n_result.sort(key = lambda x: x[1])
            s_result.append(n_result)

        #=====================================
        # statis the result of last step
        #=====================================
        count_1 = 0
        count_k = 0

        for n_i in range(0, len(s_result)):
            # for item in s_result[n_i]:
            #     result_file.write(index_A[n_i]+" "+str(item)+"\n")
            if index_A[n_i] == func_name:
                import pickle
                dumped_file = os.path.abspath("./_result.pkl")
                with open(dumped_file, "wb") as f:
                    pickle.dump(s_result[n_i], f, -1)
                return dumped_file

            if index_A[n_i] == s_result[n_i][0][0]:
                #print index_A[n_i], '@1', s_result[n_i][0]
                count_1 = count_1 + 1
                count_k = count_k + 1
            else:
                #print index_A[n_i], '@k:'
                for n_j in range(0, min(top_k, len(s_result[n_i]))):
                    if index_A[n_i] == s_result[n_i][n_j][0]:
                        count_k = count_k + 1
                    #print '     ', s_result[n_i][n_j]
        result_file.close()

        statis_count = statis_count + 1
        total_num = min(len(index_A), len(index_B))
        rate = float(count_1)/float(total)
        rate_k = float(count_k)/float(total)
        sum_rate = sum_rate + rate
        sum_rate5 = sum_rate5 + rate_k

        statis_elem = {}
        statis_elem['bindiff'] = diffdir
        statis_elem['total_of_pre'] = len(index_A)
        statis_elem['total_of_post'] = len(index_B)
        statis_elem['total'] = total
        statis_elem['@1'] = count_1
        statis_elem['@K'] = count_k
        statis_elem['rate'] = rate 
        statis.append(statis_elem)

        #f_statis.write('bindiff:' + diffdir + ';\ttotal_of_pre:' + str(len(index_A)) + ';\ttotal_of_post:' + str(len(index_B)) + ';\t@1:' + str(count_1) + ';\t@k:' + str(count_k) + ';\trate:' + str(rate) + '\n')
        print 'bindiff:', diffdir, 'total:', total, 'total_of_pre:', len(index_A), 'total_of_post:', len(index_B), '@1:', count_1, '@k:', count_k, 'rate:', rate
        #break

    f_statis.close()

    weight_one_statis = {}
    weight_one_statis['id'] = weight_file.split('-')[-2]
    weight_one_statis['avg_of_@1'] = sum_rate/statis_count
    weight_one_statis['avg_of_@5'] = sum_rate5/statis_count  
    print weight_one_statis

if __name__ == "__main__":
    bechmark_path = sys.argv[1]
    # func_name = sys.argv[2]
    # top_k = sys.argv[3]

    # test(weights_file, bechmark_path)    
