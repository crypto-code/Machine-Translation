# The necessary imports
import tensorflow as tf
import re
import numpy as np
import json
import argparse
import os

def clean_text(text):
    """
    A function that cleans the text by removing the common abbreviations and unwanted characters or puntuations
    It also ends up adding a <start> tag at the beginning of the text and
    and <end> tag at the last of the text
    """
    text = text.lower().strip()   # lowercase and remove trailing whitespaces
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"there's", "there is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r'[" "]+', " ", text)   # remove extra spaces in between
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    text = '<start> ' + text + ' <end>'
    return text


def preprocess(dataset_file_path, len_bound, num_examples = None, reverse = False):
    """
    It reads the required files, creates input output pairs.
    """
    min_sentence_length = len_bound[0]
    max_sentence_length = len_bound[1]
    
    lines = open(str(dataset_file_path), encoding='utf-8', errors = 'ignore').read().strip().split('\n')

    if num_examples is not None:
        lines = lines[:num_examples] #  This takes only some lines
        
    input_lang = []
    output_lang = []
    seen = set()
    for line in lines:
        _line = line.split('\t') # seperate the input line and output line
        if (len(_line[0].split(" "))>min_sentence_length and len(_line[0].split(" "))<max_sentence_length
            and len(_line[1].split(" "))>min_sentence_length and len(_line[1].split(" "))<max_sentence_length):
            inp = clean_text(_line[0])
            if inp in seen:
                continue
            seen.add(inp)
            input_lang.append(inp)
            output_lang.append(clean_text(_line[1]))
            

    assert len(input_lang) == len(output_lang) #  make both equal
    print("Read %s sentence pairs" % len(input_lang))

    if reverse:
        return (input_lang, output_lang)
    else:
        return (output_lang, input_lang)


def tokenize(lang, oov=True):
    """
    Tokenize sentences into words, and correspondingly create an index based representation for vocabulary
    """
    if oov:
        lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token = '<unk>')
    else:
        lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
    return tensor, lang_tokenizer


def load_dataset(dataset_file_path, len_bound, num_examples = None, reverse=False):
    # creating cleaned input, output pairs
    targ_lang, inp_lang = preprocess(dataset_file_path, len_bound, num_examples, reverse=reverse)

    input_tensor, inp_lang_tokenizer = tokenize(inp_lang, oov = True)   # in the input language, we allow OOV words
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang, oov = False)   # in the output language, we do not allow OOV words

    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dataset Preprocessor')
    parser.add_argument('--input',type=str,required=True,help='File contining training dataset')
    parser.add_argument('--name',type=str,required=True,help='Name of the Dataset')
    parser.add_argument('--reverse', action="store_true", help='Translates to English')
    parser.add_argument('--num_examples',type=int,default=None,help='Number of Examples to take from training set')
    parser.add_argument('--min_len',type=int,default=1,help='Minimum number of words in a scentence')
    parser.add_argument('--max_len',type=int,default=15,help='Maximum number of words in a scentence')
    
    args = parser.parse_args()
    dataset_file_path = args.input # the path to the folder 
    len_bounds = [args.min_len, args.max_len]   # minimum and maximum permissible length of a sentence to be considered.
    
    input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(dataset_file_path, len_bounds, num_examples = args.num_examples, reverse=args.reverse)
    
    inp_lang_json = inp_lang.to_json()
    targ_lang_json = targ_lang.to_json()
    if not os.path.exists('processed_data_'+args.name):
        os.mkdir('processed_data_'+args.name)
    
    with open('processed_data_'+args.name+'/inp_lang.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(inp_lang_json, ensure_ascii=False))
        f.close()
    print('Input Language Tokenizer saved...')
        
    with open('processed_data_'+args.name+'/targ_lang.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(targ_lang_json, ensure_ascii=False))
        f.close()
    print('Target Language Tokenizer saved...')
        
    np.savez('processed_data_'+args.name+'/data.npz', input_tensor, target_tensor)
    print('Final Dataset saved...')
    
    
