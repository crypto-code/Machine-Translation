# Machine-Translation
Train a Seq2Seq Model with Attention to Translate from One Language to Another

## Seq2Seq Model with Attention

Sequence to Sequence(Seq2Seq) models uses an encoder-decoder architecture like the one shown below.

<p align="center">
<img src="https://github.com/crypto-code/Machine-Translation/blob/master/assets/seq2seq.png" height="400" align="middle" />   </p>

A simple Seq2Seq model is normally composed of an encoder-decoder architecture, where the encoder processes the input sequence and encodes/compresses/summarizes the information into a context vector (also called as the “thought vector”) of a fixed length. This representation is expected to be a good summary of the entire input sequence. The decoder is then initialized with this context vector, using which it starts generating the transformed output.

**So what is wrong with a Seq2Seq Model ?**

A critical and apparent disadvantage of this fixed-length context vector design is the incapability of the system to remember longer sequences. Often is has forgotten the earlier parts of the sequence once it has processed the entire the sequence. The attention mechanism was born to resolve this problem.

### B

## Usage:

- Firstly, download the dataset you require from [here](http://www.manythings.org/anki/)
