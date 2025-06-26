# Sentiment Analysis

## Test 1: `main1.py`

### Constants
- Using `MultiHeadedAttention` with only 1 attention head for attention layer.
- Using `PositionalEncodingSinusoidal` for positional encoding.
- For the final decision of sentiment, all of the embedding is joined tail-to-head. This means that if the context length is `d_context` and each token is embedded into `d_model` dimensions, the final layer after the encoder layers has dimension `(d_model * d_context, 2)`.

### Variables
- Padding mask is done differently. 

### Results

## Test 2: `main2.py`


## Test 3: `main3.py`


## Test 4: `main4.py`