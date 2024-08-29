# SocialGNN Encoding Results


## Figures Overview

The figures are stored in the `result_figures/behavioral_encoding` directory and represent the final states (the last 
hidden layer) of different models tested. 

Ridge regression: Ridge regression is performed using RidgeCV, which automatically selects the best regularization strength 
(alpha) from 10^(-5), 10^-4， 10^-3, 10^-2, 10^-1, 10. The model is trained on the training set and evaluated on both a cross-validation set and the test set.

### 1. Linear-SocialGNN Final State

![Linear-SocialGNN Final State](./result_figures/behavioral_encoding/linear-SocialGNN-final_state.png)

### 2. Linear-LSTM Relation Final State

![Linear-LSTM Relation Final State](./result_figures/behavioral_encoding/linear-LSTM_Relation-final_state.png)

### 3. Linear-LSTM Final State

![Linear-LSTM Final State](./result_figures/behavioral_encoding/linear-LSTM-final_state.png)

### 4. Ridge-SocialGNN Final State

![Ridge-SocialGNN Final State](./result_figures/behavioral_encoding/ridge-SocialGNN-final_state.png)

### 5. Ridge-LSTM Relation Final State

![Ridge-LSTM Relation Final State](./result_figures/behavioral_encoding/ridge-LSTM_Relation-final_state.png)

### 6. Ridge-LSTM Final State

![Ridge-LSTM Final State](./result_figures/behavioral_encoding/ridge-LSTM-final_state.png)

