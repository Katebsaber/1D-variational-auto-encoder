# Variational Auto Encoder time series forecasting system - KERAS

This model is the key component of a forecasting system based on Variational inference written in Keras. 

As the name suggests, this model is categorized as an Auto Encoder so the input and output data should be in the same shape.  This model is variational so it uses KL Divergence as loss function to compare the real and hypothetical probability distributions.