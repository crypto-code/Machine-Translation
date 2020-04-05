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
