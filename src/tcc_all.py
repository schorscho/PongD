import sys
import os
import logging.config
from time import time
from datetime import datetime
from math import log

import pandas as pd

import numpy as np

from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, precision_score, recall_score

import tensorflow as tf

import keras
import keras.models as km
import keras.layers as kl
from keras import backend
from keras.models import load_model
from keras.callbacks import Callback
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

from att_layer import AttentionWeightedAverage


DEFAULT_LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': { 
        'standard': { 
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
    },
    'handlers': { 
        'default': { 
            'level': 'DEBUG',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
    },
    'loggers': { 
        'root': { 
            'handlers': ['default'],
            'level': 'INFO'
        },
        'TCC': { 
            'handlers': ['default'],
            'level': 'INFO',
            'propagate': False
        },
    } 
}


logging.config.dictConfig(DEFAULT_LOGGING)


logger = logging.getLogger('TCC')


class Tcc:
    PROJECT_ROOT_DIR = os.environ['TCC_ROOT_DIR']
    INPUT_DIR = os.path.join(PROJECT_ROOT_DIR, "input")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT_DIR, "output")

    TRAIN_FILE = "train.csv"
    TRAIN_PREP_FILE = "tcc_train_prep"
    VAL_PREP_FILE = "tcc_val_prep"
    TEST_PREP_FILE = "tcc_test_prep"
    COMP_FILE = "test.csv"
    COMP_PREP_FILE = "tcc_comp_prep"

    SUB_FILE = "sample_submission"
    
    EMB_MAT_FILE = "tcc_emb_mat"

    PAD = '___PAD___'
    OOV = '___OOV___'
    EMPTY = '___VERY_EMPTY___'

    CMT_ID_COL = 'id'
    CMT_TXT_COL = 'comment_text'
    CMT_INDEX_COL = 'tc_id'
    TOXIC = "toxic"
    SEVERE_TOXIC = "severe_toxic"
    OBSCENE = "obscene"
    THREAT = "threat"
    INSULT = "insult"
    IDENTITY_HATE = "identity_hate"
    CLASSES = [TOXIC, SEVERE_TOXIC, OBSCENE, THREAT, INSULT, IDENTITY_HATE]
    EM_INDEX_COL = 'word_id'
    
    TRAIN_SIZE = 0.95
    VAL_SIZE = 0.05
    TEST_SIZE = 0.0
    DP = 'DP01R00'
    
    VOCAB_FILE = 'glove.6B.200d.txt'
    VOCAB_SIZE = 40000
    MAX_LEN = 200
    EMBED_DIMS = 200
    WP = 'WP02R00'

    COMPREHENSIVE = 'comprehensive'
    INDIVIDUAL = 'individual'
    MV = 'MV04R00'
    
    BATCH_SIZE = 64
    OV = 'OV03R00'

    START_EP = 0
    END_EP = 4
    LOAD_MODEL_TOXIC = 'TR010_MV04R00_OV03R00_WP02R00_DP01R00_SE00_EP04-0.0972_toxic_20180227-080003.hdf5'
    LOAD_MODEL_SEVERE_TOXIC = 'TR010_MV04R00_OV03R00_WP02R00_DP01R00_SE00_EP04-0.0263_severe_toxic_20180227-083816.hdf5'
    LOAD_MODEL_OBSCENE = 'TR010_MV04R00_OV03R00_WP02R00_DP01R00_SE00_EP04-0.0494_obscene_20180227-091628.hdf5'
    LOAD_MODEL_THREAT = 'TR010_MV04R00_OV03R00_WP02R00_DP01R00_SE00_EP04-0.0092_threat_20180227-095436.hdf5'
    LOAD_MODEL_INSULT = 'TR010_MV04R00_OV03R00_WP02R00_DP01R00_SE00_EP04-0.0737_insult_20180227-103251.hdf5'
    LOAD_MODEL_IDENTITY_HATE = 'TR010_MV04R00_OV03R00_WP02R00_DP01R00_SE00_EP04-0.0309_identity_hate_20180227-111109.hdf5'
    LOAD_MODEL = 'TR010_MV04R00_OV03R00_WP02R00_DP01R00_SE00_EP02-0.0412_comprehensive_20180226-172746.hdf5'
    #LOAD_MODEL_TOXIC = None
    #LOAD_MODEL_SEVERE_TOXIC = None
    #LOAD_MODEL_OBSCENE = None
    #LOAD_MODEL_THREAT = None
    #LOAD_MODEL_INSULT = None
    #LOAD_MODEL_IDENTITY_HATE = None
    #LOAD_MODEL = None
    SAVE_MODEL = 'TR010'
    

def build_keras_model(comprehensive, emb_dims, vocab_size, max_len, emb_matrix):
    ip = kl.Input(shape=(max_len,))
    x = kl.Embedding(input_dim=vocab_size, output_dim=emb_dims, 
                     weights=[emb_matrix], trainable=True, name='X_emb')(ip)
    # x = kl.Activation('tanh')(x)
    x = kl.SpatialDropout1D(0.4)(x)
    x1 = kl.Bidirectional(kl.CuDNNGRU(200, return_sequences=True))(x)
    x2 = kl.Bidirectional(kl.CuDNNGRU(200, return_sequences=True))(x1)
    x = kl.concatenate([x1, x2, x])
    x = AttentionWeightedAverage(name='attlayer', return_attention=False)(x)
    x = kl.Dropout(0.4)(x)
    op = kl.Dense(6 if comprehensive else 1, activation="sigmoid")(x)

    model = km.Model(inputs=[ip], outputs=op)
    
    return model


def build_keras_model_mv03r00(emb_dims, vocab_size, max_len, emb_matrix):
    ip = kl.Input(shape=(max_len,))
    x = kl.Embedding(vocab_size, emb_dims, weights=[emb_matrix], trainable=True, name='X_emb')(ip)
    x = kl.SpatialDropout1D(0.5)(x)
    x = kl.Bidirectional(kl.CuDNNGRU(200, return_sequences=True))(x)
    x = kl.Bidirectional(kl.CuDNNGRU(200, return_sequences=True))(x)
    x = kl.GlobalMaxPool1D()(x)
    x = kl.Dense(100, activation="relu")(x)
    x = kl.Dropout(0.5)(x)
    op = kl.Dense(6, activation="sigmoid")(x)

    model = km.Model(inputs=[ip], outputs=op)

    return model    


def lr_schedule(ep):
    lr = [0] * 10

    lr[0] = 0.001
    lr[1] = 0.001
    lr[2] = 0.0005
    lr[3] = 0.0001
    lr[4] = 0.0001
    lr[5] = 0.0001
    lr[6] = 0.00005
    lr[7] = 0.00001
    lr[8] = 0.00001
    lr[9] = 0.00001
    
    logger.info('New learning rate: %01.10f', lr[ep])
    
    return lr[ep]


def compile_keras_model(model):
    model.compile(optimizer='adam', loss=backend.binary_crossentropy, metrics=['accuracy', tf_auc_roc])

    return model


def time_it(start, end):
    h, r = divmod(end - start, 3600)
    m, s = divmod(r, 60)
    
    return "{:0>2}:{:0>2}:{:06.3f}".format(int(h), int(m), s)


def load_data(file_name):
    if file_name == Tcc.TRAIN_FILE or file_name == Tcc.COMP_FILE or file_name == Tcc.SUB_FILE:
        index_col = None
        path = os.path.join(Tcc.INPUT_DIR, file_name)
    else:
        index_col = Tcc.CMT_INDEX_COL
        path = os.path.join(Tcc.OUTPUT_DIR, file_name + '_' + Tcc.DP + '_' + Tcc.WP + '.csv')
   
    data = pd.read_csv(filepath_or_buffer=path, sep=',', header=0, index_col=index_col)
    
    return data


def save_data(data, file_name):
    file_name += '_' + Tcc.DP + '_' + Tcc.WP + '.csv'

    data.to_csv(path_or_buf=os.path.join(Tcc.OUTPUT_DIR, file_name))


def split_data(x, train_size, val_size, test_size):
    if val_size + test_size > 0:
        x_tr, x_v = split_train_test(x, train_size, val_size+test_size)
    
        if val_size == 0:
            x_te = x_v
            x_v = None
        elif test_size == 0:
            x_te = None
        else:
            x_v, x_te = split_train_test(x_v, val_size/(val_size+test_size), test_size/(val_size+test_size))
    else:
        x_tr = x
        x_v = None
        x_te = None
        
    return x_tr, x_v, x_te


def split_train_test(x, train_size, test_size):
    split = ShuffleSplit(n_splits=1, train_size=train_size, test_size=test_size, random_state=None)
    res = split.split(x)

    x_tr = None
    x_te = None
 
    for train_i, test_i in res:
        if isinstance(x, pd.DataFrame):
            x_tr = x.iloc[train_i]
            x_te = x.iloc[test_i]
        else:
            x_tr = x[train_i]
            x_te = x[test_i]
    
    return x_tr, x_te


def load_all_data(train_set, val_set, test_set, comp_set, initialization):
    train_data = None
    val_data = None
    test_data = None
    comp_data = None
    
    if train_set:
        if initialization:
            logger.info("Loading initial training data ...")
        
            train_data = load_data(file_name=Tcc.TRAIN_FILE)

            logger.info("Loading initial training data done.")
        else:
            logger.info("Loading prepared training data ...")
        
            train_data = load_data(file_name=Tcc.TRAIN_PREP_FILE)
        
            logger.info("Loading prepared training data done.")

    if val_set:
        logger.info("Loading prepared validation data ...")

        val_data = load_data(file_name=Tcc.VAL_PREP_FILE)

        logger.info("Loading prepared validation data done.")

    if test_set:
        logger.info("Loading prepared test data ...")

        test_data = load_data(Tcc.TEST_PREP_FILE)

        logger.info("Loading prepared test data done.")

    if comp_set:
        if initialization:
            logger.info("Loading initial competition data ...")
        
            comp_data = load_data(file_name=Tcc.COMP_FILE)

            logger.info("Loading initial competition data done.")
        else:
            logger.info("Loading prepared competition data ...")
        
            comp_data = load_data(file_name=Tcc.COMP_PREP_FILE)
        
            logger.info("Loading prepared competition data done.")

    return train_data, val_data, test_data, comp_data


def save_all_prepared_data(train_data, val_data, test_data, comp_data):
    if train_data is not None:
        logger.info("Saving prepared training data ...")

        save_data(train_data, Tcc.TRAIN_PREP_FILE)

        logger.info("Saving prepared training data done.")

    if val_data is not None:
        logger.info("Saving prepared validation data ...")
    
        save_data(val_data, Tcc.VAL_PREP_FILE)

        logger.info("Saving prepared validation data done.")

    if test_data is not None:
        logger.info("Saving prepared test data ...")
    
        save_data(test_data, Tcc.TEST_PREP_FILE)

        logger.info("Saving prepared test data done.")
    
    if comp_data is not None:
        logger.info("Saving prepared competition data ...")
    
        save_data(comp_data, Tcc.COMP_PREP_FILE)

        logger.info("Saving prepared competition data done.")


def prepare_data(data):
    data[Tcc.CMT_INDEX_COL] = [i for i in range(len(data))]
    
    data.set_index(Tcc.CMT_INDEX_COL, inplace=True)


def execute_data_initialization(train_data, comp_data):
    logger.info("Starting initial data preparation ...")

    val_data = None
    test_data = None
    
    if train_data is not None:
        logger.info("Preparing training data ...")

        prepare_data(train_data)

        logger.info("Preparing training data done.")

        logger.info("Splitting prepared training data ...")

        train_data_as_test = None

        if Tcc.TEST_SIZE == 0:
            train_data_as_test = train_data
        
        train_data, val_data, test_data = split_data(x=train_data,
                                                     train_size=Tcc.TRAIN_SIZE, 
                                                     val_size=Tcc.VAL_SIZE,
                                                     test_size=Tcc.TEST_SIZE)

        if Tcc.TEST_SIZE == 0:
            test_data = train_data_as_test

        logger.info("Splitting prepared training data done.")

    if comp_data is not None:
        logger.info("Preparing competition data ...")

        prepare_data(comp_data)

        logger.info("Preparing competition data done.")

    return train_data, val_data, test_data, comp_data
        

def build_tokenizer(items, vocab_size):
    filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'

    tokenizer = Tokenizer(num_words=vocab_size, filters=filters, oov_token=Tcc.OOV)    
    
    tokenizer.fit_on_texts(items.values)    
    
    return tokenizer


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def build_emb_matrix(tokenizer, vocab_size):
    emb_index = dict(get_coefs(*o.strip().split())
                     for o in open(os.path.join(Tcc.INPUT_DIR, Tcc.VOCAB_FILE)))
    
    all_embs = np.stack(emb_index.values())
    emb_mean = all_embs.mean()
    emb_std = all_embs.std()
    
    word_index = tokenizer.word_index

    if vocab_size is None:
        vocab_size = len(word_index)

    vocab_size = min(vocab_size, len(word_index))
    
    emb_matrix = np.random.normal(emb_mean, emb_std, (vocab_size, Tcc.EMBED_DIMS))
    
    for word, i in word_index.items():
        if i >= vocab_size:
            continue
        
        emb_vector = emb_index.get(word)
        
        if emb_vector is not None: 
            emb_matrix[i] = emb_vector

    emb_matrix = pd.DataFrame(emb_matrix)
    
    emb_matrix[Tcc.EM_INDEX_COL] = [i for i in range(len(emb_matrix))]
    
    columns = [str(i) for i in range(Tcc.EMBED_DIMS)]
    
    columns.append(Tcc.EM_INDEX_COL)
    
    emb_matrix.columns = columns
    
    emb_matrix.set_index([Tcc.EM_INDEX_COL], inplace=True)
    
    emb_matrix.sort_index(inplace=True)

    return emb_matrix


def load_emb_matrix():
    file_name = Tcc.EMB_MAT_FILE + '_' + Tcc.DP + '_' + Tcc.WP + '.csv'

    emb_matrix = pd.read_csv(
        filepath_or_buffer=os.path.join(Tcc.OUTPUT_DIR, file_name),
        header=0, index_col=Tcc.EM_INDEX_COL)
    
    return emb_matrix


def save_emb_matrix(emb_matrix):
    file_name = Tcc.EMB_MAT_FILE + '_' + Tcc.DP + '_' + Tcc.WP + '.csv'

    emb_matrix.to_csv(path_or_buf=os.path.join(Tcc.OUTPUT_DIR, file_name))   


def execute_word2i(embedding, train_data, val_data, test_data, comp_data):
    emb_matrix = None

    then = time()

    logger.info("Executing word2i ...")

    logger.info("Building tokenizer ...")
    
    if Tcc.TEST_SIZE > 0:
        data = pd.concat([train_data[Tcc.CMT_TXT_COL], val_data[Tcc.CMT_TXT_COL], 
                          test_data[Tcc.CMT_TXT_COL], comp_data[Tcc.CMT_TXT_COL]])
    else:
        data = pd.concat([train_data[Tcc.CMT_TXT_COL], val_data[Tcc.CMT_TXT_COL], comp_data[Tcc.CMT_TXT_COL]])

    tokenizer = build_tokenizer(items=data, vocab_size=Tcc.VOCAB_SIZE)

    logger.info("Building tokenizer done.")

    if embedding:
        logger.info("Building embedding matrix ...")

        emb_matrix = build_emb_matrix(tokenizer=tokenizer, vocab_size=Tcc.VOCAB_SIZE)

        logger.info("Building embedding matrix done.")

        logger.info("Saving embedding matrix ...")

        save_emb_matrix(emb_matrix)

        logger.info("Saving embedding matrix done.")

    logger.info("Executing word2i done in %s.", time_it(then, time()))     

    return tokenizer, emb_matrix


def build_indexed_data(data, tokenizer, max_words):
    list_sentences = data[Tcc.CMT_TXT_COL].values                    
    
    list_tokenized = tokenizer.texts_to_sequences(list_sentences)
    
    indexed_data = sequence.pad_sequences(list_tokenized, maxlen=max_words, padding='post', truncating='post')

    seq_cols = [str(i) for i in range(Tcc.MAX_LEN)]
    
    indexed_data = pd.DataFrame(data=indexed_data, columns=seq_cols)

    data.reset_index(inplace=True)

    indexed_data = pd.concat([data, indexed_data], axis=1)
    
    indexed_data.set_index(Tcc.CMT_INDEX_COL, inplace=True)
    
    return indexed_data


def execute_indexation(train_data, val_data, test_data, comp_data, tokenizer):
    logger.info("Starting indexation ...")

    then = time()

    train_prop = {'data': train_data, 'context': 'training data'}
    val_prop = {'data': val_data, 'context': 'validation data'}
    test_prop = {'data': test_data, 'context': 'test data'}
    comp_prop = {'data': comp_data, 'context': 'competition data'}

    for prop in [train_prop, val_prop, test_prop, comp_prop]:
        data = prop['data']
        
        if data is not None:
            logger.info("Indexation for %s ...", prop['context'])

            indexed_data = build_indexed_data(data=data, tokenizer=tokenizer, max_words=Tcc.MAX_LEN)
            
            prop['data'] = indexed_data

            logger.info("Indexation for %s done.", prop['context'])

    logger.info("Done with indexation in %s.", time_it(then, time()))
    
    return train_prop['data'], val_prop['data'], test_prop['data'], comp_prop['data']


def get_data_packages(data, submission):
    seq_cols = [str(i) for i in range(Tcc.MAX_LEN)]

    x_seq = data[seq_cols].as_matrix()
    
    if not submission:
        y = data[Tcc.CLASSES].as_matrix()
    else:
        y = None
    
    cmt_id = data[Tcc.CMT_ID_COL].values
        
    return x_seq, y, cmt_id

def load_keras_models(tox_classes):
    models = {}

    if Tcc.LOAD_MODEL is not None and Tcc.COMPREHENSIVE in tox_classes:
        logger.info("Loading model for %s ...", Tcc.COMPREHENSIVE)

        models[Tcc.COMPREHENSIVE] = load_keras_model(Tcc.LOAD_MODEL)

        logger.info("Loading model for %s done.", Tcc.COMPREHENSIVE)

    if Tcc.LOAD_MODEL_TOXIC is not None and Tcc.TOXIC in tox_classes:
        logger.info("Loading model for %s ...", Tcc.TOXIC)

        models[Tcc.TOXIC] = load_keras_model(Tcc.LOAD_MODEL_TOXIC)

        logger.info("Loading model for %s done.", Tcc.TOXIC)

    if Tcc.LOAD_MODEL_SEVERE_TOXIC is not None and Tcc.SEVERE_TOXIC in tox_classes:
        logger.info("Loading model for %s ...", Tcc.SEVERE_TOXIC)

        models[Tcc.SEVERE_TOXIC] = load_keras_model(Tcc.LOAD_MODEL_SEVERE_TOXIC)

        logger.info("Loading model for %s done.", Tcc.SEVERE_TOXIC)

    if Tcc.LOAD_MODEL_OBSCENE is not None and Tcc.OBSCENE in tox_classes:
        logger.info("Loading model for %s ...", Tcc.OBSCENE)

        models[Tcc.OBSCENE] = load_keras_model(Tcc.LOAD_MODEL_OBSCENE)

        logger.info("Loading model for %s done.", Tcc.OBSCENE)

    if Tcc.LOAD_MODEL_THREAT is not None and Tcc.THREAT in tox_classes:
        logger.info("Loading model for %s ...", Tcc.THREAT)

        models[Tcc.THREAT] = load_keras_model(Tcc.LOAD_MODEL_THREAT)

        logger.info("Loading model for %s done.", Tcc.THREAT)

    if Tcc.LOAD_MODEL_INSULT is not None and Tcc.INSULT in tox_classes:
        logger.info("Loading model for %s ...", Tcc.INSULT)

        models[Tcc.INSULT] = load_keras_model(Tcc.LOAD_MODEL_INSULT)

        logger.info("Loading model for %s done.", Tcc.INSULT)

    if Tcc.LOAD_MODEL_IDENTITY_HATE is not None and Tcc.IDENTITY_HATE in tox_classes:
        logger.info("Loading model for %s ...", Tcc.IDENTITY_HATE)

        models[Tcc.IDENTITY_HATE] = load_keras_model(Tcc.LOAD_MODEL_IDENTITY_HATE)

        logger.info("Loading model for %s done.", Tcc.IDENTITY_HATE)

    return models


def load_keras_model(model_file):
    model = load_model(os.path.join(Tcc.OUTPUT_DIR, model_file), 
                       custom_objects={'tf_auc_roc': tf_auc_roc,
                                       'AttentionWeightedAverage': AttentionWeightedAverage})

    return model
    

def save_keras_model(model, model_file):
    model.save(os.path.join(Tcc.OUTPUT_DIR, model_file))


class Evaluation(Callback):
    def __init__(self, val_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_v, self.y_v = val_data

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_v, verbose=0)
            roc_auc = roc_auc_score(self.y_v, y_pred)
            bce = log_loss(self.y_v, y_pred)

            print(" - val_skl_bce ({:.6f}), val_skl_roc_auc ({:.6f})".format(bce, roc_auc))


def tf_auc_roc(y_true, y_pred):
    # any tensorflow metric
    value, update_op = tf.contrib.metrics.streaming_auc(y_pred, y_true)

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value
    

def my_bce(y_true, y_pred):
    res = 0
    for i in range(y_true.shape[0]):
        r = 0
        for j in range(y_true.shape[1]):
            r -= y_true[i, j] * log(y_pred[i, j])
        
        res += r
        
    return res
    

def execute_training(tox_classes, start_epoch, end_epoch, build_on_models, save_models_as, train_data, val_data, emb_matrix):
    x_t, y_t, _ = get_data_packages(train_data, False)
    x_v, y_v, _ = get_data_packages(val_data, False)

    y_ta = y_t
    y_va = y_v

    models = {}
        
    for tox_class in tox_classes:
        single = (tox_class == Tcc.COMPREHENSIVE)
        build_on_model = None

        if not single:
            ind = [i for i, v in enumerate(Tcc.CLASSES) if v == tox_class]
            ind = ind[0]
            y_ta = y_t[:, ind]
            y_va = y_v[:, ind]

        if tox_class in build_on_models.keys():
            build_on_model = build_on_models[tox_class]

        model = build_and_train_model(tox_class, start_epoch, end_epoch, build_on_model, save_models_as,
                                      x_t, y_ta, x_v, y_va, emb_matrix)

        models[tox_class] = model

    return models


def build_and_train_model(tox_class, start_epoch, end_epoch, build_on_model, save_model_as,
                          x_t, y_t, x_v, y_v, emb_matrix):
    logger.info('Building/compiling model for %s ...', tox_class)

    if build_on_model is None:
        model = build_keras_model(comprehensive=(tox_class == Tcc.COMPREHENSIVE),
                                  emb_dims=Tcc.EMBED_DIMS,
                                  vocab_size=Tcc.VOCAB_SIZE,
                                  max_len=Tcc.MAX_LEN,
                                  emb_matrix=emb_matrix)
    else:
        model = build_on_model

    model = compile_keras_model(model)

    callbacks = [keras.callbacks.LearningRateScheduler(lr_schedule),
                 Evaluation(val_data=(x_v, y_v), interval=1)]

    if save_model_as is not None:
        file = save_model_as + '_{0}_{1}_{2}_{3}'.format(Tcc.MV, Tcc.OV, Tcc.WP, Tcc.DP)
        file += '_' + tox_class
        file += '_SE{0:02d}'.format(start_epoch)
        file += '_EP{epoch:02d}-{val_loss:.4f}'
        file += '_' + datetime.utcnow().strftime("%Y%m%d-%H%M%S") + '.hdf5'

        mc_callback = keras.callbacks.ModelCheckpoint(filepath=os.path.join(Tcc.OUTPUT_DIR, file),
                                                      monitor='val_loss', verbose=0, save_best_only=False,
                                                      save_weights_only=False, mode='min', period=1)

        callbacks.append(mc_callback)

    logger.info('Building/compiling model for %s done.', tox_class)

    logger.info('Fitting model for %s ...', tox_class)

    history = model.fit(
        x=[x_t], y=y_t,
        batch_size=Tcc.BATCH_SIZE,
        epochs=end_epoch,
        verbose=1,
        callbacks=callbacks,
        shuffle=True,
        initial_epoch=start_epoch,
        steps_per_epoch=None,
        validation_data=[[x_v], y_v])

    logger.info('Fitting model for %s done.', tox_class)

    return model


def execute_test(tox_classes, models, test_data):
    x_t, y_t, _ = get_data_packages(test_data, False)
    y_pred = []
    y_true = []
    result = None

    comprehensive = (tox_classes[0] == Tcc.COMPREHENSIVE)
    individual = (len(tox_classes) == len(Tcc.CLASSES))

    if not comprehensive:
        if individual:
            tox_classes = Tcc.CLASSES

        for tox_class in tox_classes:
            logger.info("Evaluating model for %s ...", tox_class)

            class_i = [i for i, v in enumerate(Tcc.CLASSES) if v == tox_class]
            class_i = class_i[0]
            y_ta = y_t[:, class_i]

            result = evaluate_model(x_t, y_ta, models[tox_class])

            log_specific_results(tox_class, y_ta, result, False)

            y_pred.append(result[3][:, 0])
            y_true.append(y_ta)

            logger.info("Evaluating model for %s done.", tox_class)

        y_true = np.stack(y_true, axis=1)
        y_pred = np.stack(y_pred, axis=1)

        result = None
    elif comprehensive:
        logger.info("Evaluating model for %s ...", Tcc.COMPREHENSIVE)

        result = evaluate_model(x_t, y_t, models[Tcc.COMPREHENSIVE])

        y_pred = result[3]
        y_true = y_t

        for class_i in range(len(Tcc.CLASSES)):
            result[3] = y_pred[:, class_i]
            y_ta = y_t[:, class_i]

            log_specific_results(Tcc.CLASSES[class_i], y_ta, result, True)

        logger.info("Evaluating model for %s done.", Tcc.COMPREHENSIVE)

    log_comprehensive_results(y_true, y_pred, result)


def evaluate_model(x_t, y_t, model):
    keras_bce, keras_acc, tf_roc_auc = model.evaluate(x=[x_t], y=y_t, batch_size=None, verbose=1, sample_weight=None, steps=None)

    y_pred = model.predict(x_t, verbose=1)

    return [keras_bce, keras_acc, tf_roc_auc, y_pred]


def log_specific_results(tox_class, y_true, result, comprehensive):
    keras_bce = result[0]
    keras_acc = result[1]
    tf_roc_auc = result[2]
    y_pred = result[3]

    skl_roc_auc = roc_auc_score(y_true, y_pred)
    skl_bce = log_loss(y_true, y_pred)

    y_pred = y_pred - 0.5
    y_pred = y_pred > 0

    skl_acc = accuracy_score(y_true, y_pred)
    skl_prec = precision_score(y_true, y_pred)
    skl_rec = recall_score(y_true, y_pred)

    no_class_samples = len(y_true[y_true > 0])

    if comprehensive:
        logger.info("Spec. results: "
                    "skl_bce (%.6f), "
                    "skl_roc_auc (%.6f), "
                    "skl_acc (%.6f), "
                    "skl_prec (%.6f), skl_rec (%.6f) - "
                    "%s (%d / %d = %.2f%%)",
                    skl_bce,
                    skl_roc_auc,
                    skl_acc,
                    skl_prec, skl_rec,
                    tox_class, no_class_samples, len(y_true), no_class_samples * 100 / len(y_true))
    else:
        logger.info("Spec. results: "
                    "keras_bce (%.6f), skl_bce (%.6f), "
                    "tf_roc_auc (%.6f), skl_roc_auc (%.6f), "
                    "keras_acc (%.6f), skl_acc (%.6f), "
                    "skl_prec (%.6f), skl_rec (%.6f) - "
                    "%s (%d / %d = %.2f%%)",
                    keras_bce, skl_bce,
                    tf_roc_auc, skl_roc_auc,
                    keras_acc, skl_acc,
                    skl_prec, skl_rec,
                    tox_class, no_class_samples, len(y_true), no_class_samples * 100 / len(y_true))

    # correct = (y_true == y_pred)

    # print("Classification Rate: ", len(correct[correct]) / len(y_true))


def log_comprehensive_results(y_true, y_pred, result):
    skl_roc_auc = roc_auc_score(y_true, y_pred)
    skl_bce = log_loss(y_true, y_pred)

    y_pred = y_pred - 0.5
    y_pred = y_pred > 0

    skl_acc = accuracy_score(y_true, y_pred)

    if result is not None:
        keras_bce = result[0]
        keras_acc = result[1]
        tf_roc_auc = result[2]

        logger.info("Comprehensive results: "
                    "keras_bce (%.6f), skl_bce (%.6f), "
                    "tf_roc_auc (%.6f), skl_roc_auc (%.6f), "
                    "keras_acc (%.6f), skl_acc (%.6f)",
                    keras_bce, skl_bce,
                    tf_roc_auc, skl_roc_auc,
                    keras_acc, skl_acc)
    else:
        logger.info("Comprehensive results: "
                    "skl_bce (%.6f), skl_roc_auc (%.6f), "
                    "skl_acc (%.6f)",
                    skl_bce, skl_roc_auc,
                    skl_acc)


def execute_submission(model, comp_data):
    x_s, _, cmt_id = get_data_packages(comp_data, True)
    
    y_pred = model.predict(x=[x_s], batch_size=None, verbose=1, steps=None)

    y_pred = pd.DataFrame(data=y_pred, columns=Tcc.CLASSES)

    cmt_id = pd.DataFrame(data=cmt_id, columns=[Tcc.CMT_ID_COL])

    sub_data = pd.concat(objs=[cmt_id, y_pred], axis=1)
    
    sub_data.set_index(Tcc.CMT_ID_COL, inplace=True)

    sub_data.to_csv(path_or_buf=os.path.join(Tcc.OUTPUT_DIR, Tcc.SUB_FILE + '.gz'), compression='gzip')   

    
def main():
    overall = time()

    logger.info("Main script started ...")     
    
    initialization = False
    embedding = False
    indexation = False
    training = False
    validation = False
    test = False
    submission = False
    
    train_set = False
    val_set = False
    test_set = False
    comp_set = False

    tox_classes = []
    
    tokenizer = None
    emb_matrix = None
    models = None
    
    for arg in sys.argv[1:]:
        if arg == 'initialization':
            initialization = True
        elif arg == 'embedding':
            embedding = True
        elif arg == 'indexation':
            indexation = True
        elif arg == 'training':
            training = True
        elif arg == 'validation':
            validation = True
        elif arg == 'test':
            test = True
        elif arg == 'submission':
            submission = True
        elif arg == 'train_set':
            train_set = True
        elif arg == 'val_set':
            val_set = True
        elif arg == 'test_set':
            test_set = True
        elif arg == 'comp_set':
            comp_set = True
        elif arg == Tcc.INDIVIDUAL:
            tox_classes = Tcc.CLASSES
        elif arg in Tcc.CLASSES:
            tox_classes.append(arg)

    if not initialization and not embedding and not indexation \
            and not training and not validation and not test and not submission:
        initialization = True
        embedding = True
        indexation = True
        training = True
    
    if not train_set and not test_set and not val_set and not comp_set:
        train_set = True
        val_set = True
        test_set = True
        comp_set = True

    if len(tox_classes) == 0:
        tox_classes = [Tcc.COMPREHENSIVE]
        
    then = time()

    logger.info("Data preparation started ...")     

    train_data, val_data, test_data, comp_data = load_all_data(
        train_set=(train_set and initialization) or (embedding) or (train_set and indexation) or (training),
        val_set=((embedding) or (val_set and indexation) or (training) or (validation)) and not initialization,
        test_set=((embedding) or (train_set and indexation) or (test)) and not initialization,
        comp_set=(embedding) or (comp_set and initialization) or (comp_set and indexation) or (submission),
        initialization=initialization)

    if initialization:
        train_data, val_data, test_data, comp_data = execute_data_initialization(
            train_data=train_data, comp_data=comp_data)

    if embedding or indexation:
        tokenizer, emb_matrix = execute_word2i(embedding, train_data, val_data, test_data, comp_data)

    if indexation:
        train_data, val_data, test_data, comp_data = execute_indexation(
            train_data=train_data if train_set else None, 
            val_data=val_data if val_set else None, 
            test_data=test_data if test_set else None, 
            comp_data=comp_data if comp_set else None, 
            tokenizer=tokenizer)

    if initialization or indexation:
        save_all_prepared_data(
            train_data=train_data if train_set else None, 
            val_data=val_data if val_set else None, 
            test_data=test_data if test_set else None,
            comp_data=comp_data if comp_set else None)

    logger.info("Data preparation done in %s.", time_it(then, time()))  

    if training or validation or test or submission:
        models = load_keras_models(tox_classes)

    if training:
        logger.info("Executing training ...")    

        then = time()
        
        if emb_matrix is None:
            logger.info("Loading embedding matrix ...")    

            emb_matrix = load_emb_matrix()
            
            logger.info("Embedding matrix loaded.")    

        logger.info("Begin training ...")    

        models = execute_training(tox_classes, start_epoch=Tcc.START_EP, end_epoch=Tcc.END_EP,
                                  build_on_models=models, save_models_as=Tcc.SAVE_MODEL,
                                  train_data=train_data, val_data=val_data, emb_matrix=emb_matrix)

        logger.info("Training completed.")    
        
        logger.info("Done executing training in %s.", time_it(then, time()))    

    if validation:
        logger.info("Executing validation ...")    

        then = time()

        execute_test(tox_classes, models, val_data)
    
        logger.info("Done executing validation in %s.", time_it(then, time()))    

    if test:
        logger.info("Executing test ...")    

        then = time()

        execute_test(tox_classes, models, test_data)
    
        logger.info("Done executing test in %s.", time_it(then, time()))    
        
    if submission:
        logger.info("Executing submission ...")    

        then = time()

        #execute_submission(models, comp_data)

        logger.info("Done executing submission in %s.", time_it(then, time()))    

        logger.info("Main script finished in %s.", time_it(overall, time()))    
        

if __name__ == "__main__":
    main()
