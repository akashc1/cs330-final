import numpy as np
import sys
import tensorflow as tf
from functools import partial
import csv
import pickle
import random
import pandas as pd

from data import DataGenerator
from util import cross_entropy_loss, accuracy




def conv_block(inp, cweight, bweight, bn, activation=tf.nn.relu, residual=False):
    """ Perform, conv, batch norm, nonlinearity, and max pool """
    stride, no_stride = [1,2,2,1], [1,1,1,1]

    conv_output = tf.nn.conv2d(input=inp, filters=cweight, strides=no_stride, padding='SAME') + bweight
    normed = bn(conv_output)
    normed = activation(normed)
    return normed

class ConvLayers(tf.keras.layers.Layer):
    def __init__(self, channels, dim_hidden, dim_output, img_size):
        super(ConvLayers, self).__init__()
        self.channels = channels
        self.dim_hidden = dim_hidden
        self.dim_output = dim_output
        self.img_size = img_size

        weights = {}

        dtype = tf.float32
        weight_initializer =  tf.keras.initializers.GlorotUniform()
        k = 3

        weights['conv1'] = tf.Variable(weight_initializer(shape=[k, k, self.channels, self.dim_hidden]), name='conv1', dtype=dtype)
        weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden]), name='b1')
        self.bn1 = tf.keras.layers.BatchNormalization(name='bn1')
        weights['conv2'] = tf.Variable(weight_initializer(shape=[k, k, self.dim_hidden, self.dim_hidden]), name='conv2', dtype=dtype)
        weights['b2'] = tf.Variable(tf.zeros([self.dim_hidden]), name='b2')
        self.bn2 = tf.keras.layers.BatchNormalization(name='bn2')
        weights['conv3'] = tf.Variable(weight_initializer(shape=[k, k, self.dim_hidden, self.dim_hidden]), name='conv3', dtype=dtype)
        weights['b3'] = tf.Variable(tf.zeros([self.dim_hidden]), name='b3')
        self.bn3 = tf.keras.layers.BatchNormalization(name='bn3')
        weights['conv4'] = tf.Variable(weight_initializer([k, k, self.dim_hidden, self.dim_hidden]), name='conv4', dtype=dtype)
        weights['b4'] = tf.Variable(tf.zeros([self.dim_hidden]), name='b4')
        self.bn4 = tf.keras.layers.BatchNormalization(name='bn4')
        weights['w5'] = tf.Variable(weight_initializer(shape=[self.dim_hidden, self.dim_output]), name='w5', dtype=dtype)
        weights['b5'] = tf.Variable(tf.zeros([self.dim_output]), name='b5')
        self.conv_weights = weights

    def call(self, inp, weights):
        channels = self.channels
        inp = tf.reshape(inp, [-1, self.img_size, self.img_size, channels])
        hidden1 = conv_block(inp, weights['conv1'], weights['b1'], self.bn1)
        hidden2 = conv_block(hidden1, weights['conv2'], weights['b2'], self.bn2)
        hidden3 = conv_block(hidden2, weights['conv3'], weights['b3'], self.bn3)
        hidden4 = conv_block(hidden3, weights['conv4'], weights['b4'], self.bn4)
        hidden4 = tf.reduce_mean(input_tensor=hidden4, axis=[1, 2])
        
        return tf.matmul(hidden4, weights['w5']) + weights['b5']

class MAML(tf.keras.Model):
    def __init__(self, dim_input=1, dim_output=1,
                num_inner_updates=1,
                inner_update_lr=0.4, num_filters=32, k_shot=5, learn_inner_update_lr=False):
        super(MAML, self).__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.inner_update_lr = inner_update_lr
        self.loss_func = cross_entropy_loss
        self.dim_hidden = num_filters
        self.channels = 1
        self.img_size = int(np.sqrt(self.dim_input/self.channels))

        seed = 42

        # outputs_ts[i] and losses_ts_post[i] are the output and loss after i+1 inner gradient updates
        losses_tr_pre, outputs_tr, losses_ts_post, outputs_ts = [], [], [], []
        accuracies_tr_pre, accuracies_ts = [], []

        # for each loop in the inner training loop
        outputs_ts = [[]]*num_inner_updates
        losses_ts_post = [[]]*num_inner_updates
        accuracies_ts = [[]]*num_inner_updates

        # Define the weights - these should NOT be directly modified by the
        # inner training loop
        tf.random.set_seed(seed)
        self.conv_layers = ConvLayers(self.channels, self.dim_hidden, self.dim_output, self.img_size)

        self.learn_inner_update_lr = learn_inner_update_lr

        if self.learn_inner_update_lr:
            self.inner_update_lr_dict = {}

            for key in self.conv_layers.conv_weights.keys():
                self.inner_update_lr_dict[key] = [
                            tf.Variable(self.inner_update_lr,
                                name=f'inner_update_lr_{key}_{j}') for j in range(num_inner_updates)
                                                ]

    def call(self, inp, meta_batch_size=25, num_inner_updates=1):
        def task_inner_loop(inp, reuse=True,
                        meta_batch_size=25, num_inner_updates=1):
            """
                Perform gradient descent for one task in the meta-batch (i.e. inner-loop).
                Args:
                inp: a tuple (input_tr, input_ts, label_tr, label_ts), where input_tr and label_tr are the inputs and
                    labels used for calculating inner loop gradients and input_ts and label_ts are the inputs and
                    labels used for evaluating the model after inner updates.
                    Should be shapes:
                    input_tr: [N*K, 784]
                    input_ts: [N*K, 784]
                    label_tr: [N*K, N]
                    label_ts: [N*K, N]
                Returns:
                task_output: a list of outputs, losses and accuracies at each inner update
            """
            # the inner and outer loop data
            input_tr, input_ts, label_tr, label_ts = inp

            # weights corresponds to the initial weights in MAML (i.e. the meta-parameters)
            weights = self.conv_layers.conv_weights

            # the predicted outputs, loss values, and accuracy for the pre-update model (with the initial weights)
            # evaluated on the inner loop training data
            task_output_tr_pre, task_loss_tr_pre, task_accuracy_tr_pre = None, None, None

            # lists to keep track of outputs, losses, and accuracies of test data for each inner_update
            # where task_outputs_ts[i], task_losses_ts[i], task_accuracies_ts[i] are the output, loss, and accuracy
            # after i+1 inner gradient updates
            task_outputs_ts, task_losses_ts, task_accuracies_ts = [], [], []

            with tf.GradientTape(persistent=True) as g:

                # perform initial forward pass with meta-parameters 
                initial_out_tr = self.conv_layers(input_tr, weights)
                task_output_tr_pre = initial_out_tr
                task_loss_tr_pre = self.loss_func(initial_out_tr, label_tr)
                grads = g.gradient(task_loss_tr_pre, list(weights.values()))

                # created copy of weights with first update
                if self.learn_inner_update_lr:
                    updated_weights = {var_name: var - self.inner_update_lr_dict[var_name][0] * grad
                                    for (var_name, var), grad in zip(weights.items(), grads)}
                else:
                    updated_weights = {var_name: var - self.inner_update_lr * grad
                                    for (var_name, var), grad in zip(weights.items(), grads)}


                out_ts = self.conv_layers(input_ts, updated_weights)
                loss_ts = self.loss_func(out_ts, label_ts)
                task_outputs_ts.append(out_ts)
                task_losses_ts.append(loss_ts)

                # iteratively update parameters (not meta-parameters)
                for i in range(num_inner_updates - 1):
                    
                    out_tr = self.conv_layers(input_tr, updated_weights)
                    loss = self.loss_func(out_tr, label_tr)

                    grads = g.gradient(loss, list(updated_weights.values()))

                    if self.learn_inner_update_lr:
                        updated_weights = {var_name: var - self.inner_update_lr_dict[var_name][i + 1] * grad
                                    for (var_name, var), grad in zip(weights.items(), grads)}
                    else:
                        updated_weights = {var_name: var - self.inner_update_lr * grad
                                    for (var_name, var), grad in zip(weights.items(), grads)}

                    out_ts = self.conv_layers(input_ts, updated_weights)
                    loss_ts = self.loss_func(out_ts, label_ts)
                    task_outputs_ts.append(out_ts)
                    task_losses_ts.append(loss_ts)

            # Compute accuracies from output predictions
            task_accuracy_tr_pre = accuracy(tf.argmax(input=label_tr, axis=1),
                                            tf.argmax(input=tf.nn.softmax(task_output_tr_pre), axis=1))

            for j in range(num_inner_updates):
                task_accuracies_ts.append(accuracy(tf.argmax(input=label_ts, axis=1),
                                          tf.argmax(input=tf.nn.softmax(task_outputs_ts[j]), axis=1)))

            task_output = [task_output_tr_pre, task_outputs_ts, task_loss_tr_pre, task_losses_ts, task_accuracy_tr_pre, task_accuracies_ts]

            return task_output

        input_tr, input_ts, label_tr, label_ts = inp
        # to initialize the batch norm vars, might want to combine this, and not run idx 0 twice.
        unused = task_inner_loop((input_tr[0], input_ts[0], label_tr[0], label_ts[0]),
                            False,
                            meta_batch_size,
                            num_inner_updates)
        out_dtype = [tf.float32, [tf.float32]*num_inner_updates, tf.float32, [tf.float32]*num_inner_updates]
        out_dtype.extend([tf.float32, [tf.float32]*num_inner_updates])
        task_inner_loop_partial = partial(task_inner_loop,
                                        meta_batch_size=meta_batch_size,
                                        num_inner_updates=num_inner_updates)
        result = tf.map_fn(task_inner_loop_partial,
                        elems=(input_tr, input_ts, label_tr, label_ts),
                        dtype=out_dtype,
                        parallel_iterations=meta_batch_size)

        return result


def outer_train_step(inp, model, optim, meta_batch_size=25, num_inner_updates=1):
    with tf.GradientTape(persistent=False) as outer_tape:
        result = model(inp, meta_batch_size=meta_batch_size, num_inner_updates=num_inner_updates)

        outputs_tr, outputs_ts, losses_tr_pre, losses_ts, accuracies_tr_pre, accuracies_ts = result

        total_losses_ts = [tf.reduce_mean(loss_ts) for loss_ts in losses_ts]

    # perform meta-update
    gradients = outer_tape.gradient(total_losses_ts[-1], model.trainable_variables)
    optim.apply_gradients(zip(gradients, model.trainable_variables))

    total_loss_tr_pre = tf.reduce_mean(losses_tr_pre)
    total_accuracy_tr_pre = tf.reduce_mean(accuracies_tr_pre)
    total_accuracies_ts = [tf.reduce_mean(accuracy_ts) for accuracy_ts in accuracies_ts]

    return outputs_tr, outputs_ts, total_loss_tr_pre, total_losses_ts, total_accuracy_tr_pre, total_accuracies_ts


def outer_eval_step(inp, model, meta_batch_size=25, num_inner_updates=1):
  result = model(inp, meta_batch_size=meta_batch_size, num_inner_updates=num_inner_updates)

  outputs_tr, outputs_ts, losses_tr_pre, losses_ts, accuracies_tr_pre, accuracies_ts = result

  total_loss_tr_pre = tf.reduce_mean(losses_tr_pre)
  total_losses_ts = [tf.reduce_mean(loss_ts) for loss_ts in losses_ts]

  total_accuracy_tr_pre = tf.reduce_mean(accuracies_tr_pre)
  total_accuracies_ts = [tf.reduce_mean(accuracy_ts) for accuracy_ts in accuracies_ts]

  return outputs_tr, outputs_ts, total_loss_tr_pre, total_losses_ts, total_accuracy_tr_pre, total_accuracies_ts  


def meta_train_fn(model, exp_string, data_generator,
               n_way=5, meta_train_iterations=15000, meta_batch_size=25,
               log=True, logdir='../logs/', k_shot=1, num_inner_updates=1, meta_lr=0.001,
               output_filename=None):
    
    SUMMARY_INTERVAL = 10
    SAVE_INTERVAL = 100
    PRINT_INTERVAL = 10  
    TEST_PRINT_INTERVAL = PRINT_INTERVAL*5

    pre_accuracies, post_accuracies = [], []

    num_classes = data_generator.num_classes

    optimizer = tf.keras.optimizers.Adam(learning_rate=meta_lr)

    # pandas DF to track running stats
    output_data = pd.DataFrame(columns=[
                                        'iter', 'pre_inner_tr_acc',
                                        'post_inner_ts_acc',
                                        'pre_tr_loss', 'ts_loss',
                                        'val_pre_tr_acc', 'val_post_ts_acc',
                                        'val_pre_tr_loss', 'val_ts_loss'
                                        ])

    full_log_filename = logdir + output_filename + '.csv'

    for itr in range(meta_train_iterations):

        # sample batch & partition into support/query sets, reshape
        all_images, all_labels = data_generator.sample_batch("meta_train",
                                                            batch_size=meta_batch_size,
                                                            shuffle=True, swap=False)
        
        new_shape_im = (meta_batch_size, n_way * k_shot, all_images.shape[-1])
        new_shape_lbl = (meta_batch_size, n_way * k_shot, n_way)
        input_tr = tf.reshape(all_images[:, :, :k_shot, :], new_shape_im)
        input_ts = tf.reshape(all_images[:, :, k_shot:, :], new_shape_im)
        label_tr = tf.reshape(all_labels[:, :, :k_shot, :], new_shape_lbl)
        label_ts = tf.reshape(all_labels[:, :, k_shot:, :], new_shape_lbl)

        inp = (input_tr, input_ts, label_tr, label_ts)
        
        result = outer_train_step(inp, model, optimizer, meta_batch_size=meta_batch_size, num_inner_updates=num_inner_updates)

        if itr % SUMMARY_INTERVAL == 0:
            pre_accuracies.append(result[-2])
            post_accuracies.append(result[-1][-1])

        if (itr!=0) and itr % PRINT_INTERVAL == 0:
            print_str = 'Iteration %d: pre-inner-loop train accuracy: %.5f, post-inner-loop test accuracy: %.5f' % (itr, np.mean(pre_accuracies), np.mean(post_accuracies))
            print(print_str)
            print(f"Total training loss pre: {result[2]}\tTotal test loss: {result[3][-1].numpy()}")

            output_data = output_data.append({
                'iter': itr,
                'pre_inner_tr_acc': np.mean(pre_accuracies),
                'post_inner_ts_acc': np.mean(post_accuracies),
                'pre_tr_loss': result[2],
                'ts_loss': result[3][-1].numpy()
            }, ignore_index=True)

            pre_accuracies, post_accuracies = [], []

        if (itr!=0) and itr % TEST_PRINT_INTERVAL == 0:

            # sample batch & partition into support/query sets, reshape
            all_images, all_labels = data_generator.sample_batch("meta_val",
                                                                batch_size=meta_batch_size,
                                                                shuffle=True, swap=False)
            
            new_shape_im = (meta_batch_size, n_way * k_shot, all_images.shape[-1])
            new_shape_lbl = (meta_batch_size, n_way * k_shot, n_way)
            input_tr = tf.reshape(all_images[:, :, :k_shot, :], new_shape_im)
            input_ts = tf.reshape(all_images[:, :, k_shot:, :], new_shape_im)
            label_tr = tf.reshape(all_labels[:, :, :k_shot, :], new_shape_lbl)
            label_ts = tf.reshape(all_labels[:, :, k_shot:, :], new_shape_lbl)

            inp = (input_tr, input_ts, label_tr, label_ts)
            result = outer_eval_step(inp, model, meta_batch_size=meta_batch_size, num_inner_updates=num_inner_updates)

            if (output_data['iter'] == itr).any():
            
                row = output_data.loc[output_data['iter'] == itr].index[0]
                output_data.loc[row, 'val_pre_tr_acc'] = result[-2].numpy()
                output_data.loc[output_data['iter'] == itr, 'val_post_ts_acc'] = result[-1][-1].numpy()
                output_data.loc[output_data['iter'] == itr, 'val_pre_tr_loss'] = result[2].numpy()
                output_data.loc[output_data['iter'] == itr, 'val_ts_loss'] = result[3][-1].numpy()
            else:
                output_data = output_data.append({
                    'iter': itr,
                    'val_pre_tr_acc': result[-2].numpy(),
                    'val_post_ts_acc': result[-1][-1].numpy(),
                    'val_pre_tr_loss': result[2].numpy(),
                    'val_ts_loss': result[3][-1].numpy().numpy()
                }, ignore_index=True)

            print('Meta-validation pre-inner-loop train accuracy: %.5f, meta-validation post-inner-loop test accuracy: %.5f' % (result[-2], result[-1][-1]))

        if itr != 0 and itr % SAVE_INTERVAL == 0 and output_filename is not None:
            output_data.to_csv(full_log_filename)

    model_file = logdir + '/models/' + exp_string +  '/model' + str(itr)
    print("Saving to ", model_file)
    model.save_weights(model_file)

    return output_data


def meta_test_fn(model, data_generator, n_way=5, meta_batch_size=25, k_shot=1,
              num_inner_updates=1):

    NUM_META_TEST_POINTS = 600
  
    num_classes = data_generator.num_classes

    np.random.seed(1)
    random.seed(1)

    meta_test_accuracies = []

    for _ in range(NUM_META_TEST_POINTS):
        
        # sample batch & partition into support/query sets, reshape
        all_images, all_labels = data_generator.sample_batch("meta_test",
                                                            batch_size=meta_batch_size,
                                                            shuffle=True, swap=False)
        
        new_shape_im = (meta_batch_size, n_way * k_shot, all_images.shape[-1])
        new_shape_lbl = (meta_batch_size, n_way * k_shot, n_way)
        input_tr = tf.reshape(all_images[:, :, :k_shot, :], new_shape_im)
        input_ts = tf.reshape(all_images[:, :, k_shot:, :], new_shape_im)
        label_tr = tf.reshape(all_labels[:, :, :k_shot, :], new_shape_lbl)
        label_ts = tf.reshape(all_labels[:, :, k_shot:, :], new_shape_lbl)

        inp = (input_tr, input_ts, label_tr, label_ts)
        result = outer_eval_step(inp, model, meta_batch_size=meta_batch_size, num_inner_updates=num_inner_updates)

        meta_test_accuracies.append(result[-1][-1])

    meta_test_accuracies = np.array(meta_test_accuracies)
    means = np.mean(meta_test_accuracies)
    stds = np.std(meta_test_accuracies)
    ci95 = 1.96*stds/np.sqrt(NUM_META_TEST_POINTS)

    print('Mean meta-test accuracy/loss, stddev, and confidence intervals')
    print((means, stds, ci95))

def run_maml(n_way=5, k_shot=1, meta_batch_size=25, meta_lr=0.001,
             inner_update_lr=0.4, num_filters=32, num_inner_updates=1,
             learn_inner_update_lr=False,
             resume=False, resume_itr=0, log=True, logdir='/tmp/data',
             data_path='../omniglot_resized',meta_train=True,
             meta_train_iterations=15000, meta_train_k_shot=-1,
             meta_train_inner_update_lr=-1, output_file=None):


    # call data_generator and get data with k_shot*2 samples per class
    data_generator = DataGenerator(n_way, k_shot*2, n_way, k_shot*2, config={'data_folder': data_path})

    # set up MAML model
    dim_output = data_generator.dim_output
    dim_input = data_generator.dim_input
    model = MAML(dim_input,
                dim_output,
                num_inner_updates=num_inner_updates,
                inner_update_lr=inner_update_lr,
                k_shot=k_shot,
                num_filters=num_filters,
                learn_inner_update_lr=learn_inner_update_lr)

    if meta_train_k_shot == -1:
        meta_train_k_shot = k_shot
    if meta_train_inner_update_lr == -1:
        meta_train_inner_update_lr = inner_update_lr

    exp_string = 'cls_'+str(n_way)+'.mbs_'+str(meta_batch_size) + '.k_shot_' + str(meta_train_k_shot) + '.inner_numstep_' + str(num_inner_updates) + '.inner_updatelr_' + str(meta_train_inner_update_lr) + '.learn_inner_update_lr_' + str(learn_inner_update_lr)


    if meta_train:
        filename = output_file if output_file is not None else exp_string
        output = meta_train_fn(model, exp_string, data_generator,
                    n_way, meta_train_iterations, meta_batch_size, log, logdir,
                    k_shot, num_inner_updates, meta_lr, output_filename=filename)
        
        output.to_csv(filename)

    else:
        meta_batch_size = 1
        print(logdir + '/models/' + exp_string)
        os.listdir(logdir + '/models/' + exp_string)
        model_file = tf.train.latest_checkpoint(logdir + '/models/' + exp_string)
        print("Restoring model weights from ", model_file)
        model.load_weights(model_file)

        meta_test_fn(model, data_generator, n_way, meta_batch_size, k_shot, num_inner_updates)


if __name__ == "__main__":
    run_maml(n_way=5, k_shot=1, inner_update_lr=0.04, num_inner_updates=1,
         meta_train_iterations=8000, logdir="../logs/")
  