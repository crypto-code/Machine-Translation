# Making the necessary imports
import tensorflow as tf
import model
from preprocess import clean_text
import numpy as np
import json
from keras.preprocessing.text import tokenizer_from_json
import glob
import argparse
# This will force execute tensorflow in eager execution mode 
tf.enable_eager_execution()

parser = argparse.ArgumentParser(description='Interactive Translation Bot')
parser.add_argument('--name',type=str,required=True,help='Name of the Dataset')
parser.add_argument('--min_len',type=int,default=1,help='Minimum number of words in a scentence')
parser.add_argument('--max_len',type=int,default=15,help='Maximum number of words in a scentence')
    
args = parser.parse_args()


# load the tokenziers
with open('processed_data_'+args.name+'/inp_lang.json', 'r') as f:
    json_data_i = json.load(f)
    inp_lang = tokenizer_from_json(json_data_i)
    f.close()
    
with open('processed_data_'+args.name+'/targ_lang.json', 'r') as f:
    json_data_t = json.load(f)
    targ_lang = tokenizer_from_json(json_data_t)
    f.close()

# define hyperparameters
embedding_dim = 128
units = 256
vocab_inp_size = len(inp_lang.word_index)+1
vocab_tar_size = len(targ_lang.word_index)+1
max_sentence_length = 15

def best_weight(files):
    files.sort(key=lambda x:float(x[len(files)-9:-3]))
    return files[0]

# create encoder from Chatbot class
encoder = model.create_encoder(vocab_inp_size, embedding_dim, units, max_sentence_length)
print('Encoder model initialized...')

# create decoder from Chatbot class
decoder = model.create_decoder(vocab_tar_size, embedding_dim, units, units, max_sentence_length)
print('Decoder model initialized...')

enc_models = glob.glob(f'models_'+args.name+'/encoder_*.h5', recursive=True)
dec_models = glob.glob(f'models_'+args.name+'/decoder_*.h5', recursive=True)
encoder.load_weights(best_weight(enc_models))
decoder.load_weights(best_weight(dec_models))
print("Encoder Model: ",best_weight(enc_models))
print("Decoder Model: ",best_weight(dec_models))

def evaluate(sentence, samp_type = 1):
    sentence = clean_text(sentence)
    inputs = []
    # split the sentence and replace unknown words by <unk> token.
    for i in sentence.split(' '):
        try:
            inputs.append(inp_lang.word_index[i])
        except KeyError:
            print("Error")
            inputs.append(inp_lang.word_index['<unk>'])
    
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen = max_sentence_length, padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''
    enc_output, enc_hidden = encoder(inputs)
    
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)
    
    for t in range(max_sentence_length):
        predictions, dec_hidden = decoder([enc_output, dec_hidden, dec_input])
        if samp_type == 1:
            # Atin -- argamax implies using greedy algorithim
            predicted_id = tf.argmax(predictions[0]).numpy()
        elif samp_type == 2:
            # Atin - This is for random sampling with probability propotional to size
            predicted_id = np.random.choice(vocab_tar_size, p = predictions[0].numpy())
        elif samp_type == 3:
            # Atin - Takes top 3 samples
            _ , indices = tf.math.top_k(predictions[0], k = 3)
            predicted_id = np.random.choice(indices)

        if predicted_id!= 0:
            if targ_lang.index_word[predicted_id] == '<end>':
                return result, sentence
            else:
                result += targ_lang.index_word[predicted_id] + ' '
        
        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence

print('\n')
print('=' * 50)
print('+' * 15 + ' TRANSLATION BOT ' + '+' * 15)
print('=' * 50)
print('\nChoose A Sampling Method')
print('Input 1 for => Greedy Sampling')
print('Input 2 for => Probability Proportional Sampling')
print('Input 3 for => Top-3 Sampling')
print('=' * 50)
samp_type = int(input('Input your preferred sampling choice: '))
print('=' * 50)
print()
if samp_type not in [1,2,3]:
    raise NotImplementedError

while True:
    inputs = input('Input :> ')
    if inputs == 'quit' or inputs == 'Quit':
        break
    result, sentence = evaluate(inputs, samp_type)
    print('Output :> ' + result)

