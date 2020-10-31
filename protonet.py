import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from PIL import Image
import os, glob, sys, pickle, random
import pandas as pd
from data import DataGenerator
from util import cross_entropy_loss, accuracy

class ProtoNet(tf.keras.Model):

    def __init__(self, num_filters, latent_dim):
        super(ProtoNet, self).__init__()
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        num_filter_list = self.num_filters + [latent_dim]
        self.convs = []
        for i, num_filter in enumerate(num_filter_list):
            block_parts = [
                layers.Conv2D(
                filters=num_filter,
                kernel_size=3,
                padding='SAME',
                activation='linear'),
            ]

            block_parts += [layers.BatchNormalization()]
            block_parts += [layers.Activation('relu')]
            block_parts += [layers.MaxPool2D()]
            block = tf.keras.Sequential(block_parts, name='conv_block_%d' % i)
            self.__setattr__("conv%d" % i, block)
            self.convs.append(block)

        self.flatten = tf.keras.layers.Flatten()

    def call(self, inp):
        out = inp
        for conv in self.convs:
            out = conv(out)
        out = self.flatten(out)
        return out

def ProtoLoss(x_latent, q_latent, labels_onehot, num_classes, num_support, num_queries):
    """
        calculates the prototype network loss using the latent representation of x
        and the latent representation of the query set
        Args:
        x_latent: latent representation of supports with shape [N*S, D], where D is the latent dimension
        q_latent: latent representation of queries with shape [N*Q, D], where D is the latent dimension
        labels_onehot: one-hot encodings of the labels of the queries with shape [N, Q, N]
        num_classes: number of classes (N) for classification
        num_support: number of examples (S) in the support set
        num_queries: number of examples (Q) in the query set
        Returns:
        ce_loss: the cross entropy loss between the predicted labels and true labels
        acc: the accuracy of classification on the queries
    """
    
    latent_dim = x_latent.shape[-1]

    # prototypes are just centroids of each class's examples in latent space
    x_class_split = tf.reshape(x_latent, (num_classes, num_support, latent_dim))
    prototypes = tf.reduce_mean(x_class_split, axis=1) # (num_classes, latent_dim)

    # need to repeat prototypes for easy distance calculation
    query_split = tf.reshape(q_latent, (num_classes, num_queries, 1, latent_dim))
    expanded = tf.expand_dims(prototypes, axis=0) # (1, num_classes, latent_dim)
    expanded = tf.expand_dims(expanded, axis=0) # (1, 1, num_classes, latent_dim)
    expanded = tf.repeat(expanded, repeats=(num_classes), axis=0) # (num_classes, 1, num_classes, latent_dim)
    expanded = tf.repeat(expanded, repeats=(num_queries), axis=1) # (num_classes, num_queries, num_classes, latent_dim)
    
    # calculate distances (L2 norm), add small value for degenerate case
    dists = tf.norm(query_split - expanded, axis=3) + np.random.normal(1e-5, scale=1e-6)

    # use negative distance as logits for CE loss
    ce_loss = cross_entropy_loss(-1*dists, labels_onehot)

    # predictions use argmin if distance, argmax if logits/normalized distribution
    preds = tf.argmin(dists, axis=2)
    gt = tf.argmax(labels_onehot, axis=2)
    acc = accuracy(gt, preds)

    return ce_loss, acc

def proto_net_train_step(model, optim, x, q, labels_ph):
    num_classes, num_support, im_height, im_width, channels = x.shape
    num_queries = q.shape[1]
    x = tf.reshape(x, [-1, im_height, im_width, channels])
    q = tf.reshape(q, [-1, im_height, im_width, channels])

    with tf.GradientTape() as tape:
        x_latent = model(x)
        q_latent = model(q)
        ce_loss, acc = ProtoLoss(x_latent, q_latent, labels_ph, num_classes, num_support, num_queries)

    gradients = tape.gradient(ce_loss, model.trainable_variables)
    optim.apply_gradients(zip(gradients, model.trainable_variables))
    return ce_loss, acc

def proto_net_eval(model, x, q, labels_ph):
    num_classes, num_support, im_height, im_width, channels = x.shape
    num_queries = q.shape[1]
    x = tf.reshape(x, [-1, im_height, im_width, channels])
    q = tf.reshape(q, [-1, im_height, im_width, channels])

    x_latent = model(x)
    q_latent = model(q)
    ce_loss, acc = ProtoLoss(x_latent, q_latent, labels_ph, num_classes, num_support, num_queries)

    return ce_loss, acc 

def run_protonet(data_path='../omniglot_resized', n_way=20, k_shot=1, n_query=5,
                 n_meta_test_way=20, k_meta_test_shot=5, n_meta_test_query=5,
                 logdir="../logs/"):
    n_epochs = 20
    n_episodes = 100

    im_width, im_height, channels = 28, 28, 1
    num_filters = 32
    latent_dim = 16
    num_conv_layers = 3
    n_meta_test_episodes = 1000

    output_data = pd.DataFrame(columns=[
                                        'iter', 'tr_acc',
                                        'val_acc',
                                        'tr_loss', 'val_loss',
                                        ])

    model = ProtoNet([num_filters]*num_conv_layers, latent_dim)
    optimizer = tf.keras.optimizers.Adam()

    exp_string = ('proto_tr_cls_'+str(n_way)+'.tr_k_shot_' + str(k_shot) +
                    'ts_cls_' + str(n_meta_test_way) + '.ts_k_shot_' + str(k_meta_test_shot))
    
    full_log_file = logdir + exp_string + '.csv'

    # call DataGenerator with k_shot+n_query samples per class
    data_generator = DataGenerator(n_way, k_shot+n_query, n_meta_test_way, k_meta_test_shot+n_meta_test_query,
                                    config={'data_folder': data_path})

    for ep in range(n_epochs):
        for epi in range(n_episodes):
        
            # sample batch, partition into support/query, reshape
            images, labels = data_generator.sample_batch("meta_train", batch_size=1, shuffle=False)
            support = tf.reshape(images[0, :, :k_shot, :],
                                shape=(n_way, k_shot, im_height, im_width, 1))
            query = tf.reshape(images[0, :, k_shot:, :],
                                shape=(n_way, n_query, im_height, im_width, 1))
            labels = tf.reshape(labels[0, :, k_shot:, :], shape=(n_way, n_query, n_way))

            ls, ac = proto_net_train_step(model, optimizer, x=support, q=query, labels_ph=labels)
            if (epi+1) % 50 == 0:

                # sample batch, partition into support/query, reshape
                images, labels = data_generator.sample_batch("meta_val", batch_size=1, shuffle=False)
                support = tf.reshape(images[0, :, :k_shot, :],
                                shape=(n_way, k_shot, im_height, im_width, 1))
                query = tf.reshape(images[0, :, k_shot:, :],
                                    shape=(n_way, n_query, im_height, im_width, 1))
                labels = tf.reshape(labels[0, :, k_shot:, :], shape=(n_way, n_query, n_way))
                val_ls, val_ac = proto_net_eval(model, x=support, q=query, labels_ph=labels)
                print(f'[epoch {ep + 1}/{n_epochs}, episode {epi + 1}/{n_episodes}] => meta-training loss: {ls:.5f}, meta-training acc: {ac:.5f}, meta-val loss: {val_ls:.5f}, meta-val acc: {val_ac:.5f}')
                
                output_data = output_data.append({'iter': ep*(n_episodes) + epi + 1,
                                            'tr_acc': ac.numpy(),
                                            'val_acc': val_ac.numpy(),
                                            'tr_loss': ls.numpy(),
                                            'val_loss': val_ls.numpy(),
                                            }, ignore_index=True)
            
        output_data.to_csv(full_log_file)
    
    print("Testing...")

    meta_test_accuracies = []
    for epi in range(n_meta_test_episodes):

        # sample batch, partition into support/query, reshape
        images, labels = data_generator.sample_batch("meta_test", batch_size=1, shuffle=False)
        support = tf.reshape(images[0, :, :k_meta_test_shot, :],
                            shape=(n_meta_test_way, k_meta_test_shot, im_height, im_width, 1))
        query = tf.reshape(images[0, :, k_meta_test_shot:, :],
                            shape=(n_meta_test_way, n_meta_test_query, im_height, im_width, 1))
        labels = tf.reshape(labels[0, :, k_meta_test_shot:, :],
                            shape=(n_meta_test_way, n_meta_test_query, n_meta_test_way))
        val_ls, val_ac = proto_net_eval(model, x=support, q=query, labels_ph=labels)

        ls, ac = proto_net_eval(model, x=support, q=query, labels_ph=labels)
        meta_test_accuracies.append(ac)
        
        if (epi+1) % 50 == 0:
            print('[meta-test episode {}/{}] => loss: {:.5f}, acc: {:.5f}'.format(epi+1, n_meta_test_episodes, ls, ac))
    
    avg_acc = np.mean(meta_test_accuracies)
    stds = np.std(meta_test_accuracies)
    print('Average Meta-Test Accuracy: {:.5f}, Meta-Test Accuracy Std: {:.5f}'.format(avg_acc, stds))

if __name__ == "__main__":
    run_protonet()
