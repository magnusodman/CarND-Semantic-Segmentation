import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    graph = tf.get_default_graph()
    input_layer = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return input_layer, keep_prob, layer3_out, layer4_out, layer7_out

tests.test_load_vgg(load_vgg, tf)

#INITIALIZER = tf.contrib.layers.xavier_initializer()
INITIALIZER = tf.truncated_normal_initializer(stddev=0.01)
REGULARIZER = tf.contrib.layers.l2_regularizer(1e-3)


def conv_1x1(layer, num_classes):
    return tf.layers.conv2d(layer, num_classes, kernel_size=1, padding='same',
                              kernel_regularizer=REGULARIZER, kernel_initializer=INITIALIZER)

def upsample(layer, num_classes, size, stride):
    return tf.layers.conv2d_transpose(layer, num_classes, size, strides=stride, padding='same',
                               kernel_regularizer=REGULARIZER, kernel_initializer=INITIALIZER)

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output

    The encoder for FCN-8 is the VGG16 model pretrained on ImageNet for classification.
    The fully-connected layers are replaced by 1-by-1 convolutions
    """

    c1_1x1_layer7 = conv_1x1(vgg_layer7_out, num_classes)

    c1_1x1_layer4 = conv_1x1(vgg_layer4_out, num_classes)

    c1_1x1_layer3 = conv_1x1(vgg_layer3_out, num_classes)


    #Match layer sizes (upsample) and create skip layers

    c1_1x1_layer7_upsampled = upsample(c1_1x1_layer7, num_classes, 4, 2)
    skip1 = tf.add(c1_1x1_layer7_upsampled, c1_1x1_layer4)


    skip1_upsampled = upsample(skip1, num_classes, 4, 2)

    skip2 = tf.add(skip1_upsampled, c1_1x1_layer3)

    # Create output layer that matches image size
    output = upsample(skip2, num_classes, 16, 8)

    return output

tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """

    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))

    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    optimizer = tf.train.AdamOptimizer(learning_rate)

    train_op = optimizer.minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss

tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """

    print()
    for epoch in range(epochs):
        sum_loss = 0
        for images, labels in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss],
                     feed_dict={input_image: images, correct_label: labels, keep_prob: 0.5, learning_rate: 0.001})
            sum_loss = sum_loss + loss

        print('\rEpoch {0} cross entropy: {1}'.format(epoch, sum_loss/batch_size), end="")

tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg mod
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network


        input_layer, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, data_dir + "/vgg")
        output_layer = layers(layer3_out, layer4_out, layer7_out, num_classes)

        correct_label = tf.placeholder(tf.float32, shape=[None, None, None, num_classes])
        learning_rate = tf.placeholder(tf.float32)

        logits, train_op, cross_entropy_loss = optimize(output_layer, correct_label, learning_rate, num_classes)

        # TODO: Train NN using the train_nn function

        batch_size = 5
        epochs = 25
        sess.run(tf.global_variables_initializer())

        print("Training started")

        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_layer, correct_label, keep_prob, learning_rate)

        print("Training done!")

        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_layer)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
