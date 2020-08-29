# EVA5 - Session 6 - Batch Normalization and Regularization
## Problem Statement
### Run the best code from previous session and report findings:

•	with L1 + BN

•	with L2 + BN

•	with L1 and L2 with BN

•	with GBN

•	with L1 and L2 with GBN

# Validation Accuracy:
https://github.com/srivatsanmurugan96/Extensive-Vision-Program-5.0/blob/master/S6/val_acc.jpg


# Validation Loss:
https://github.com/srivatsanmurugan96/Extensive-Vision-Program-5.0/blob/master/S6/val_loss.jpg


L1+ Batch Normalization: The loss curve is displayed in Blue color. The curve shows that negative log loss increases at first and decreases suddenly and it become stable after 10 epochs. This model is not efficient.
L2 + Batch Normalization: The loss curve is displayed in Orange color. The curve shows that negative log loss is low from the start and there is slight increase in loss from 6th epoch to 9th epoch and then it become stable.
L1 + L2 + Batch Normalization: The loss curve is displayed in Green color. The curve shows that model doesn’t have some huge fluctuations like L1+BN but still this model shows some fluctuations for every 2 epochs till 10th epoch and then it become stable.
Ghost Batch Normalization: The loss curve is displayed in Red color. The curve shows that loss is very low ,steady and consistent till the end. This model is a good one compared to all other models.  
L1 + L2 + Ghost Batch Normalization: The loss curve is displayed in Purple color. The curve shows some minor fluctuation at start and then there is huge spike over 7th epoch and then there is huge drop, and then it become stable after 10th epoch. This model is almost similar to L1 + BN.
When comparing both accuracy and loss, we can able to infer that Ghost Batch Normalization performs great in both accuracy and negative log loss where it is very much consistent and the L1 + L2 + Ghost Batch Normalization performs poor in both aspects where we can see some huge fluctuations and it is not consistent.
### Inference: From the graph, we can see that L1 + L2 + BatchNormalization and GhostBatchNormalization gives the best accuracy score, but the model with L1 + L2 + BatchNormalization is not consistent. We can see the loss and accuracy are fluctuating. The Model with GhostBatchNormalization gives good accuracy with consistency. So this is a good model comparatively.

# 25 Mis-Classified Images:
https://github.com/srivatsanmurugan96/Extensive-Vision-Program-5.0/blob/master/S6/incorrect_images.jpg

