# SocialGNN Encoding Results


## Figures Overview

The figures are stored in the `result_figures/behavioral_encoding` directory and represent the final states (the last 
hidden layer) of different models tested. 

Ridge regression is performed using RidgeCV, which automatically selects the best regularization strength 
(alpha) from seven values sampled from a logspace of 10e-2, 10e5. Performance was measured as the Pearson correlation between the predicted behavioral or neural response and the true
response
### 1. SocialGNN Linear & Ridge Regression
<div style="display: flex; justify-content: space-between;">
    <img src="./result_figures/behavioral_encoding/linear-SocialGNN-final_state.png" alt="Linear-SocialGNN Final State" style="width: 49%;">
    <img src="./result_figures/behavioral_encoding/ridge-SocialGNN-final_state.png" alt="Linear-LSTM Relation Final State" style="width: 49%;">
</div>

### 2. LSTM Relation Linear & Ridge Regression

<div style="display: flex; justify-content: space-between;">
    <img src="./result_figures/behavioral_encoding/linear-LSTM_Relation-final_state.png" alt="Linear-SocialGNN Final State" style="width: 49%;">
    <img src="./result_figures/behavioral_encoding/ridge-LSTM_Relation-final_state.png" alt="Linear-LSTM Relation Final State" style="width: 49%;">
</div>

### 3. LSTM Linear & Ridge Regression

<div style="display: flex; justify-content: space-between;">
    <img src="./result_figures/behavioral_encoding/linear-LSTM-final_state.png" alt="Linear-SocialGNN Final State" style="width: 49%;">
    <img src="./result_figures/behavioral_encoding/ridge-LSTM-final_state.png" alt="Linear-LSTM Relation Final State" style="width: 49%;">
</div>

