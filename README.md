# Machine-Translation
Train a Seq2Seq Model with Attention to Translate from One Language to Another

## Seq2Seq Model with Attention

Sequence to Sequence(Seq2Seq) models uses an encoder-decoder architecture like the one shown below.

<p align="center">
<img src="https://github.com/crypto-code/Machine-Translation/blob/master/assets/seq2seq.png" height="400" align="middle" />   </p>

In a simple Seq2Sq model, we feed in each word from left to right one at a time. By the end, the Encoder encodes the information about the whole sentence into a numerical format. This encoded sequence along with the hidden state from each step is passed into the decoder. The decoder outputs the corresponding translated words having information about the words already produced and somecontext of the words preceding it.  

## Usage:

- Firstly, download the dataset you require from [here](http://www.manythings.org/anki/)
