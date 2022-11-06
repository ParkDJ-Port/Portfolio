import logging
import os
import numpy as np
import keras.backend as K
from keras.layers import Dense, Activation, Embedding, Input
from keras.models import Model
from keras.constraints import MaxNorm

from my_layers import Attention, Average, WeightedSum, WeightedAspectEmb, MaxMargin

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def create_model(args, maxlen, vocab):
    def ortho_reg(weight_matrix):
        ### orthogonal regularization for aspect embedding matrix ###
        w_n = K.l2_normalize(weight_matrix, axis=-1)
        reg = K.sum(K.square(K.dot(w_n, K.transpose(w_n)) - K.eye(w_n.get_shape().as_list()[0])))
        return args.ortho_reg * reg

    vocab_size = len(vocab)

    if args.emb_name:
        from w2vEmbReader import W2VEmbReader as EmbReader
        emb_reader = EmbReader(os.path.join("..", "preprocessed_data", args.domain), args.emb_name)
        aspect_matrix = emb_reader.get_aspect_matrix(args.seed_word, args.aspect_size)
        args.aspect_size = emb_reader.aspect_size
        args.emb_dim = emb_reader.emb_dim

    ##### Inputs #####
    sentence_input = Input(shape=(maxlen,), dtype='int32', name='sentence_input')
    neg_input = Input(shape=(args.neg_size, maxlen), dtype='int32', name='neg_input')

    ##### Construct word embedding layer #####
    word_emb = Embedding(vocab_size, args.emb_dim,
                         mask_zero=True, name='word_emb',
                         embeddings_constraint=MaxNorm(10))

    ##### Compute sentence representation #####
    e_w = word_emb(sentence_input)
    y_s = Average()(e_w)
    att_weights = Attention(name='att_weights',
                            W_constraint=MaxNorm(10),
                            b_constraint=MaxNorm(10))([e_w, y_s])
    z_s = WeightedSum()([e_w, att_weights]) # context(sentence) vector

    ##### Compute representations of negative instances #####
    e_neg = word_emb(neg_input)
    z_n = Average()(e_neg) # negative vector

    ##### Reconstruction #####
    p_t = Dense(args.aspect_size)(z_s)
    p_t = Activation('softmax', name='p_t')(p_t)
    r_s = WeightedAspectEmb(args.aspect_size, args.emb_dim, name='aspect_emb',
                            W_constraint=MaxNorm(10),
                            W_regularizer=ortho_reg)(p_t) # reconstruction vector

    ##### Loss #####
    loss = MaxMargin(name='max_margin')([z_s, z_n, r_s])


    print(sentence_input.shape)
    ##### Indicative #####
    sent_dic = {}
    for i, sent in enumerate(sentence_input):
        sent_dic[i] = sent.strip()

    tw_dic = {}
    vocab = {i:{} for i in range(5)}

    for i, tw in enumerate(p_t):
        tw_dic[i] = np.argmax(tw)

    for ind, asp in tw_dic.items():
        sent = sent_dic[ind]
        for word in sent.split():
            if word in vocab[asp]:
                vocab[asp][word] += 1
            else:
                vocab[asp][word] = 1

    sorted_vocab = {key : dict(sorted(val.items(), key = lambda x : x[1], reverse=True)) for key, val in vocab.items()}
    
    indicative_dic = {}
    for i in range(5):
        A_j = sum(sorted_vocab.get(i).values())
        A_j_w = np.fromiter(sorted_vocab.get(i).values(), dtype=float)
        indicative_dic[i] = A_j_w / A_j


    ##### Distinctive #####

    distinctive_dic = {}
    for i in range(5):
        A_k_w_list = []
        for w, f in sorted_vocab.get(i).items():
            A_k_w_sub = []
            A_j_w = f
            for j in range(5):
                try:
                    A_k_w = sorted_vocab.get(j)[w]
                except:
                    A_k_w = 0
                A_k_w_sub.append(A_k_w)
            A_k_w_list.append(A_j_w / max(A_k_w_sub))
        distinctive_dic[i] = np.array(A_k_w_list)

    
    ##### gmean Indicative & Distinctive #####

    gmean_dic = {}
    for i in range(5):
        gmean_dic[i] = gmean([indicative_dic[i], distinctive_dic[i]], axis=0)
    


    model = Model(inputs=[sentence_input, neg_input], outputs=[loss])

    ### Word embedding and aspect embedding initialization ######
    if args.emb_name:
        from w2vEmbReader import W2VEmbReader as EmbReader
        logger.info('Initializing word embedding matrix')
        embs = model.get_layer('word_emb').embeddings
        K.set_value(embs, emb_reader.get_emb_matrix_given_vocab(vocab, K.get_value(embs)))
        logger.info('Initializing aspect embedding matrix as centroid of kmean clusters')
        K.set_value(model.get_layer('aspect_emb').W, aspect_matrix)

    return model
