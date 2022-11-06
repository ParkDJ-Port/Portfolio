import numpy as np
import os
from keras.models import load_model
from tqdm import tqdm
from sklearn.metrics import classification_report
import keras.backend as K
from keras.preprocessing import sequence

import utils as U
import reader as dataset
from my_layers import Attention, Average, WeightedSum, WeightedAspectEmb, MaxMargin

######### Get hyper-params in order to rebuild the model architecture ###########
# The hyper parameters should be exactly the same as those used for training

parser = U.add_common_args()
parser.add_argument("-as", "--aspect-size", dest="aspect_size", type=int, metavar='<int>', default=5,
                    help="The number of aspects specified by users (default=5)")
parser.add_argument("--seed-word", dest="seed_word", type=str,
                    help="given seed word", default="none")
args = parser.parse_args()

if os.path.exists(args.seed_word):
    if args.seed_word.split('/')[-1] == 'seed_words.txt':
        tag = 'arya'
    elif args.seed_word.split('/')[-1] == 'seed_words_init.txt':
        tag = 'init'
    else: 
        tag='none'

out_dir = args.out_dir_path + '/' + args.domain
# out_dir = '../pre_trained_model/' + args.domain
U.print_args(args)

# assert args.domain in {'restaurant', 'beer'}

###### Get test data #############
vocab, train_x, test_x, overall_maxlen = dataset.get_data(args.domain, vocab_size=args.vocab_size, maxlen=args.maxlen)
# test_x = train_x
test_x = sequence.pad_sequences(test_x, maxlen=overall_maxlen)
test_length = test_x.shape[0]
splits = []
for i in range(1, test_length // args.batch_size):
    splits.append(args.batch_size * i)
if test_length % args.batch_size:
    splits += [(test_length // args.batch_size) * args.batch_size]
test_x = np.split(test_x, splits)

############# Build model architecture, same as the model used for training #########

## Load the save model parameters
model = load_model(out_dir + f'/model_param_{args.aspect_size}_{tag}',
                   custom_objects={"Attention": Attention, "Average": Average, "WeightedSum": WeightedSum,
                                   "MaxMargin": MaxMargin, "WeightedAspectEmb": WeightedAspectEmb,
                                   "max_margin_loss": U.max_margin_loss},
                   compile=True)


################ Evaluation ####################################

def evaluation(true, predict, domain):
    true_label = []
    predict_label = []
    
    with open(true, 'r') as f:
        for t in f.readlines():
            true_label.append(t.strip())
            
    for p in predict:
        predict_label.append(p.strip())

    print(classification_report(true_label, predict_label, ['location', 'drinks', 'food', 'ambience', 'service'], digits=3))



def prediction(test_labels, aspect_probs, cluster_map, domain):
    label_ids = np.argsort(aspect_probs, axis=1)[:, -1]
    predict_labels = [cluster_map[label_id] for label_id in label_ids]
    evaluation(test_labels, predict_labels, domain)


## Create a dictionary that map word index to word 
vocab_inv = {}
for w, ind in vocab.items():
    vocab_inv[ind] = w

test_fn = K.function([model.get_layer('sentence_input').input, K.learning_phase()],
                     [model.get_layer('att_weights').output, model.get_layer('p_t').output])
att_weights, aspect_probs = [], []
for batch in tqdm(test_x):
    cur_att_weights, cur_aspect_probs = test_fn([batch, 0])
    att_weights.append(cur_att_weights)
    aspect_probs.append(cur_aspect_probs)

att_weights = np.concatenate(att_weights)
aspect_probs = np.concatenate(aspect_probs)

######### Topic weight ###################################

def save_topic_weights():
    topic_weight_out = open(out_dir + f'/topic_weights_{args.aspect_size}_{tag}', 'wt', encoding='utf-8')
    labels_out = open(out_dir + f'/labels_{tag}.txt', 'wt', encoding='utf-8')
    print('Saving topic weights on test sentences...')

    for probs in aspect_probs:
        labels_out.write(str(np.argmax(probs)) + "\n")
        weights_for_sentence = ""
        for p in probs:
            weights_for_sentence += str(p) + "\t"
        weights_for_sentence.strip()
        topic_weight_out.write(weights_for_sentence + "\n")

    for probs in aspect_probs:
        weights_for_sentence = ""
        for p in probs:
            weights_for_sentence += str(p) + "\t"
        weights_for_sentence.strip()
        topic_weight_out.write(weights_for_sentence + "\n")
        
save_topic_weights()
print(aspect_probs)

## Save attention weights on test sentences into a file

def save_attention_weigts():
    att_out = open(out_dir + '/att_weights', 'wt', encoding='utf-8')
    print('Saving attention weights on test sentences...')
    test_x = np.concatenate(test_x)
    for c in range(len(test_x)):
        att_out.write('----------------------------------------\n')
        att_out.write(str(c) + '\n')

        word_inds = [i for i in test_x[c] if i != 0]
        line_len = len(word_inds)
        weights = att_weights[c]
        weights = weights[(overall_maxlen - line_len):]

        words = [vocab_inv[i] for i in word_inds]
        att_out.write(' '.join(words) + '\n')
        for j in range(len(words)):
            att_out.write(words[j] + ' ' + str(round(weights[j], 3)) + '\n')
            
# save_attention_weigts()

######################################################
# Uncomment the below part for F scores
######################################################

## cluster_map need to be specified manually according to the top words in each inferred aspect (save in aspect.log)

# map for the pr e-trained restaurant model (under pre_trained_model/restaurant)
cluster_map = {0: 'location', 1: 'drinks', 2: 'food', 3: 'ambience',
           4: 'service'}

print('--- Results on %s domain ---' % (args.domain))
test_labels = '../preprocessed_data/%s/test_label.txt' % (args.domain)
prediction(test_labels, aspect_probs, cluster_map, domain=args.domain)
