# AxonSegmentation 

Axon segmentation with microscopic images and based on deep convolutional neural networks.
The U-Net (from Ronneberger et al.) is used as a classification model.
Then an MRF is applied for the post-processing.
The resulting axon segmentation mask feeds a myelin detector (Zaimi et al.)

<img src="https://github.com/vherman3/AxonSegmentation/blob/master/schema.png" width="600px" align="middle" />
