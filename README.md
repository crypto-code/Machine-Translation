# Machine-Translation
Train a Seq2Seq Model with Attention to Translate from One Language to Another

## Seq2Seq Model with Attention

Sequence to Sequence(Seq2Seq) models uses an encoder-decoder architecture like the one shown below.

<p align="center">
<img src="https://github.com/crypto-code/Machine-Translation/blob/master/assets/seq2seq.png" align="middle" />   </p>

A simple Seq2Seq model is normally composed of an encoder-decoder architecture, where the encoder processes the input sequence and encodes/compresses/summarizes the information into a context vector (also called as the “thought vector”) of a fixed length. This representation is expected to be a good summary of the entire input sequence. The decoder is then initialized with this context vector, using which it starts generating the transformed output.

### So what is wrong with a Seq2Seq Model ?

A critical and apparent disadvantage of this fixed-length context vector design is the incapability of the system to remember longer sequences. Often is has forgotten the earlier parts of the sequence once it has processed the entire the sequence. The attention mechanism was born to resolve this problem.

### Bahdanau Attention

<p align="center">
<img src="https://github.com/crypto-code/Machine-Translation/blob/master/assets/attention.gif" height="500" align="middle" />   </p>

For implementing the Attention Mechanism:

**Step 0:** Prepare hidden states.

**Step 1:** Obtain a score for every encoder hidden state.

**Step 2:** Run all the scores through a softmax layer.

**Step 3:** Multiply each encoder hidden state by its softmaxed score.

**Step 4:** Sum up the alignment vectors.

**Step 5:** Feed the context vector into the decoder.

In a Banhdanau Attention Mechanism, the input to the next decoder step is the concatenation between the generated word from the previous decoder time step and context vector from the current time step.

## Usage:

- For Training, the language dataset is available [here](http://www.manythings.org/anki/). It consists of pairs from English to different languages. Once downloaded, you can use preprocess.py to process the dataset and prepare the tokenized dataset.
```
python preprocess.py --help

usage: preprocess.py [-h] --input INPUT --name NAME [--reverse]
                     [--num_examples NUM_EXAMPLES] [--min_len MIN_LEN]
                     [--max_len MAX_LEN]

Dataset Preprocessor

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT         File contining training dataset eg. dataset/fra.txt
  --name NAME           Name of the Dataset
  --reverse             Translates to English
  --num_examples NUM_EXAMPLES
                        Number of Examples to take from training set
  --min_len MIN_LEN     Minimum number of words in a scentence
  --max_len MAX_LEN     Maximum number of words in a scentence
```
**Note: Translation is done usually from English. --reverse can be used to make the Translation to English.**

- Once the dataset is prepared, you can start the training using train.py
```
python train.py --help

usage: train.py [-h] --name NAME [--epochs EPOCHS] [--resume]
                [--min_len MIN_LEN] [--max_len MAX_LEN]

Translation Bot Trainer

optional arguments:
  -h, --help         show this help message and exit
  --name NAME        Name of the Dataset
  --epochs EPOCHS    Number of Training Epochs
  --resume           Resume Training
  --min_len MIN_LEN  Minimum number of words in a scentence
  --max_len MAX_LEN  Maximum number of words in a scentence
```
**Note: Training by default starts from scratch. --resume can be used to continue Training from the last epoch loading the best weight**

- After the training is completed, you can run test.py to start an interactive session with the Translation Bot
```
ML¬ python test.py --help

usage: test.py [-h] --name NAME [--min_len MIN_LEN] [--max_len MAX_LEN]

Interactive Translation Bot

optional arguments:
  -h, --help         show this help message and exit
  --name NAME        Name of the Dataset
  --min_len MIN_LEN  Minimum number of words in a scentence
  --max_len MAX_LEN  Maximum number of words in a scentence
```
**Note: The Translation Bot can operate in three modes:**

**1.Greedy Sampling** 

**2.Probability Proportional Sampling** 

**3.Top-3 Sampling**

## Examples:

# G00D LUCK

For doubts email me at:
atinsaki@gmail.com
