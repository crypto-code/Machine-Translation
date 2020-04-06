# Making the necessary imports
import tensorflow as tf
import model
import json
import numpy as np
import glob
import argparse
import os
from keras.preprocessing.text import tokenizer_from_json
# This will force execute tensorflow in eager execution mode 
tf.enable_eager_execution()

parser = argparse.ArgumentParser(description='Translation Bot Trainer')
parser.add_argument('--name',type=str,required=True,help='Name of the Dataset')
parser.add_argument('--epochs',type=int,default=50,help='Number of Training Epochs')
parser.add_argument('--resume', action="store_true", help='Resume Training')
parser.add_argument('--min_len',type=int,default=1,help='Minimum number of words in a scentence')
parser.add_argument('--max_len',type=int,default=15,help='Maximum number of words in a scentence')
    
args = parser.parse_args()

# load the tokenziers
# The JSON files contain the encoding corresponding to each word in the input and output lines (done in preprocess.py)
with open('./processed_data_'+args.name+'/inp_lang.json', 'r') as f:
    json_data = json.load(f)
    inp_lang = tokenizer_from_json(json_data)
    f.close()

print('Input Language Loaded...')    

with open('./processed_data_'+args.name+'/targ_lang.json', 'r') as f:
    json_data = json.load(f)
    targ_lang = tokenizer_from_json(json_data)
    f.close()

print('Target Language Loaded...')    

# load the dataset
# The .npz file contains two arrays containing vectorized form of the input and output lines (arr_0: input & arr_1: output)
npzfile = np.load('./processed_data_'+args.name+'/data.npz')    


# define hyperparameters
BUFFER_SIZE = len(npzfile['arr_0'])
BATCH_SIZE = 64
steps_per_epoch = len(npzfile['arr_0'])//BATCH_SIZE
embedding_dim = 128
units = 256
vocab_inp_size = len(inp_lang.word_index)+1
vocab_tar_size = len(targ_lang.word_index)+1
max_sentence_length = 15

# create tensorflow dataset pipeline for faster processing
dataset = tf.data.Dataset.from_tensor_slices((npzfile['arr_0'], npzfile['arr_1'])).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
print('Loaded dataset into memory...')


# create encoder from Chatbot class
encoder = model.create_encoder(vocab_inp_size, embedding_dim, units, max_sentence_length)
encoder.summary()

# create decoder from Chatbot class
decoder = model.create_decoder(vocab_tar_size, embedding_dim, units, units, max_sentence_length)
decoder.summary()


# there are lots of parameters, so more training would yield better results
optimizer = tf.keras.optimizers.Adam(1e-2)

# the training step function that performs the optimization
@tf.function
def train_step(inp, targ):
    loss = 0
    # Firstly <start> is passed to the decoder to begin the process of teacher forcing
    # Check out this article https://machinelearningmastery.com/teacher-forcing-for-recurrent-neural-networks/
    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp)  # pass the input to the encoder, get encoder_output and state
        dec_hidden = enc_hidden   # set the decoder hidden state same as encoder final state
        dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)

        # Teacher forcing - feeding the target as the next input
        for t in range(1, targ.shape[1]):
            # passing enc_output to the decoder
            predictions, dec_hidden = decoder([enc_output, dec_hidden, dec_input])

            loss += model.loss_func(targ[:, t], predictions)

            # using teacher forcing
            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss / int(targ.shape[1]))
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return batch_loss

def best_weight(files):
    files.sort(key=lambda x:float(x[len(files)-9:-3]))
    return files[0]

def get_epoch(files):
    files.sort(key=lambda x:int(x.split("_")[2][6:]), reverse=True)
    eno = files[0].split("_")[2][6:]
    return int(eno)

## Here you have the training step
def training(EPOCHS, name, resume = False):
    show_output = 10
    start=0
    if resume:
        enc_models = glob.glob(f'models_'+name+'/encoder_*.h5', recursive=True)
        dec_models = glob.glob(f'models_'+name+'/decoder_*.h5', recursive=True)
        encoder.load_weights(best_weight(enc_models))
        decoder.load_weights(best_weight(dec_models))
        start = get_epoch(enc_models)
        print("Encoder Model: ",best_weight(enc_models))
        print("Decoder Model: ",best_weight(dec_models))
        print("Weights Loaded")
        print("Resuming from Epoch", start+1)

    if not os.path.exists('models_'+name):
        os.mkdir('models_'+name)
        
    for epoch in range(start, EPOCHS):
        print('=' * 80)
        print('EPOCH: ', epoch+1)
        total_loss = 0

        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step(inp, targ)
            total_loss += batch_loss

            if batch % show_output == 0:
                print(str(batch/show_output) + '\t\t Loss: ' + str(batch_loss))

        print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / steps_per_epoch))

        r_loss = str(total_loss / steps_per_epoch)
        r_loss = r_loss[10:16]
        # after training save the weights
        encoder.save_weights('models_'+name+'/encoder_epoch-{}_loss-{}.h5'.format(str(epoch+1), str(r_loss)))
        decoder.save_weights('models_'+name+'/decoder_epoch-{}_loss-{}.h5'.format(str(epoch+1), str(r_loss)))

    
# when performing training for first time, First_time = True, else First_time = False
training(args.epochs, args.name, resume = args.resume) 
