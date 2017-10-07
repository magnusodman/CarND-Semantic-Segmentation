Semantic segmentation
=====================


Summary
-------

This project detects free road in images/videos using a FCN-8 network. 


Network structure
-----------------

The network is based on the FCN-8 network with vgg16 as the encoder part. Layer 3, 4 and 7 are transformed using a 1x1 convolution and connected to the decoder part of the network. 

```python
    c1_1x1_layer7 = conv_1x1(vgg_layer7_out, num_classes)
    c1_1x1_layer4 = conv_1x1(vgg_layer4_out, num_classes)
    c1_1x1_layer3 = conv_1x1(vgg_layer3_out, num_classes)

```

The decoder consists of transposed convolutional layers that upsamples the layers to create an output image. The skip layers retain the spatial information that is lost in the convolutions.  

```python
    c1_1x1_layer7_upsampled = upsample(c1_1x1_layer7, num_classes, 4, 2)
    skip1 = tf.add(c1_1x1_layer7_upsampled, c1_1x1_layer4)

    skip1_upsampled = upsample(skip1, num_classes, 4, 2)
    skip2 = tf.add(skip1_upsampled, c1_1x1_layer3)
    # Create output layer that matches image size
    output = upsample(skip2, num_classes, 16, 8)
```

Network training
----------------

The training is done using the AdamOptimizer and a softmax cost function.

```python
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    optimizer = tf.train.AdamOptimizer(learning_rate)

    train_op = optimizer.minimize(cross_entropy_loss)
```

The network is trained for 25 epochs and the learning rate was set to 0.001. 

Results
-------
images/um_000038.png
![alt text](file://images/um_000038.png "Result from inference")

