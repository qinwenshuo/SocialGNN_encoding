# SocialGNN Encoding Results


## Figures Overview

The figures are stored in the `result_figures/behavioral_encoding` directory and represent the different layers of different models tested. 

The following reproduce the methods in Garcia & Kathy et al:

Train (200) and test (50) data are predefined. Ridge regression is performed using RidgeCV, which automatically selects the best regularization strength 
(alpha) from seven values sampled from a logspace of 10e-2, 10e5. Performance was measured as the Pearson correlation between the predicted behavioral or neural response and the true
respons. 

Garcia, Kathy, et al. "Modeling dynamic social vision highlights gaps between deep learning and humans." (2024).


### Ridge regression behavioral encoding 

model tested: 
* LSTM, final_state
* SocialGNN, final_state
* LSTM_Relation, final_state

![ridge regression behavioral encoding](result_figures/behavioral_encoding/ridge_beh_encoding.png)

model tested: 
* LSTM, final_state
* SocialGNN, RNN_output
* LSTM_Relation, final_state

![ridge regression behavioral encoding](result_figures/behavioral_encoding/RNN_ridge_beh_encoding.png)

### Linear regression behavioral encoding results

model tested: 
* LSTM, final_state
* SocialGNN, final_state
* LSTM_Relation, final_state

![linear regression behavioral encoding](result_figures/behavioral_encoding/linear_beh_encoding.png)

model tested: 
* LSTM, final_state
* SocialGNN, RNN_output
* LSTM_Relation, final_state

![linear regression behavioral encoding](result_figures/behavioral_encoding/RNN_linear_beh_encoding.png)
