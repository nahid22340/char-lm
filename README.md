
# Character Language Model (`char-lm`)

Bigram and Trigram language models are character-level language models. Bigram model consists of 1 context window (e.g., bigram model has knowledge about only one previous letter and predicts the next letter). On the other hand, trigram model has 2 context windows (e.g., trigram model has knowledge about previous two letters and predicts the next letter).

I will upload Count Matrix and Neural Net approach of both models in distinct files. After uploading all of them, I will explicitly explain each model here in the README section. I will use no advanced built-in models or high-level frameworks—pure PyTorch is the only requirement.

## Project Structure

```text
character-language-model/
│
├── 01_bigram_counting.py     # 2D Frequency counting matrix for bigrams
├── 02_bigram_nn.py           # Bigram model trained via Cross-Entropy Loss & Gradient Descent
├── 03_trigram_counting.py    # 3D Frequency counting matrix for trigrams 
├── 04_trigram_nn.py          # Trigram neural network with dual-character context
├── README.md                 # Project documentation and performance benchmarks
└── names.txt                 # Dataset of names used for training
```