import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from PIL import Image
import os, glob, sys, pickle, random
import pandas as pd
import numpy as np
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

class Sampler(tf.keras.Model):
    def __init__(self, hidden_dim, num_classes):
        super(Sampler, self).__init__()
        self.layer1 = tf.keras.layers.Dense(hidden_dim)
        self.layer2 = tf.keras.layers.Dense(hidden_dim)
        self.layer3 = tf.keras.layers.Dense(num_classes)
        #self.num_unlabeled = num_unlabeled

    def call(self, input):
        x = self.layer1(input)
        x = tf.keras.activations.relu(x)
        x = self.layer2(x)
        x = tf.keras.activations.relu(x)
        x = self.layer3(x)
        update_weights = tf.keras.activations.relu(x) # will be (n_way, k_shot, latent_dim)
        return update_weights
        #return self.pick_samples(desired_latent, data)

    
def pick_samples(target, data):
        # target is (n_way, k_shot, latent_dim)
        # data is (num_unlabeled*k, latent_dim), arbitrary k

        n_way, k_shot, latent_dim = target.shape
        shaped_data = tf.expand_dims(data, axis=0) # (1, num_unlabeled*k, latent_dim)

        distances = tf.norm(target - shaped_data, axis=2)
        min_inds = tf.argmin(distances, axis=1, output_type=tf.dtypes.int64)

        sampled_unlabeled_points = tf.reshape([data[ind, :] for ind in min_inds], [n_way, latent_dim])
        
        return sampled_unlabeled_points

def get_prototypes(labels, latent, num_classes, num_samples):
    class_split = tf.reshape(latent, (num_classes, num_samples, -1))
    prototypes = tf.reduce_mean(class_split, axis=1) # (num_classes, latent_dim)
    return prototypes


def ProtoLoss(prototypes, x_latent, q_latent, labels_onehot, num_classes, num_unlabeled, num_support, num_queries):
    """
        calculates the prototype network loss using the latent representation of x
        and the latent representation of the query set
        Args:
        desired_latent: num_classes unlabeled examples to refine 
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
    #x_class_split = tf.reshape(x_latent, (num_classes, num_support, latent_dim))
    #prototypes = tf.reduce_mean(x_class_split, axis=1) # (num_classes, latent_dim)

    #closest_centroids = []
    #for u in range(num_unlabeled):
      #  dists = []
      #  for p in range(num_classes):
      #      dists.append(tf.norm(desired_latent[u] - prototypes[p]))
     #   closest_centroids.append(tf.argmax(dists))
    #for u in range(num_unlabeled):
        #new_class = x_class_split[closest_centroids[u],:,:]
        #unlabeled_sample = tf.expand_dims(desired_latent[u], axis=0)
        #new_class = tf.concat([new_class, unlabeled_sample], axis = 1)
        #prototypes = prototypes[closest_centroids[u]].assign(tf.reduce_mean(new_class, axis = 1))


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

def update_protos(u_latent, u_weights, labeled_prototypes, num_support):
    num_classes = labeled_prototypes.shape[0]
    num_unlabeled = u_latent.shape[0]
    latent_dim = u_latent.shape[1]

    u_class_totals = tf.zeros([num_classes, latent_dim])
    for i in range(num_unlabeled):
        u_latent_i = tf.expand_dims(u_latent[i], axis=0)
        u_latent_i = tf.tile(u_latent_i, multiples = [num_classes, 1]) #(num_classes, latent_dim)
        u_weights_i = tf.expand_dims(u_weights[i], axis=-1) # (1, num_classes)
        u_weights_i = tf.tile(u_weights_i, multiples=[1, latent_dim])
        u_class_totals += tf.multiply(u_latent_i, u_weights_i)

    # new_prototypes = prototypes + (1/(num_support + total_unlabeled_weight))*(update - prototypes)
    # u_weights.shape = (num_unlabeled, num_classes)
    u_weight_totals = tf.expand_dims(tf.reduce_sum(u_weights, axis=0), axis=-1)
    aggregate_protos = u_class_totals + labeled_prototypes*num_support
    aggregate_weights = num_support + u_weight_totals
    new_prototypes = tf.divide(aggregate_protos, aggregate_weights)
    return new_prototypes #(num_classes, latent_dim)


def proto_net_train_step(embedder, weighter, optim, x, q, u, labels_ph, eval=False, baseline=False):
    num_classes, num_support, im_height, im_width, channels = x.shape
    num_queries = q.shape[1]
    x = tf.reshape(x, [-1, im_height, im_width, channels])
    q = tf.reshape(q, [-1, im_height, im_width, channels])
    u = tf.reshape(u, [-1, im_height, im_width, channels])

    q_labels = labels_ph[:, num_support:num_support+num_queries, :]
    
    num_unlabeled = u.shape[0]

    x_labels = labels_ph[:, :num_support, :]
    q_labels = labels_ph[:, num_support:, :]

    x = tf.reshape(x, [-1, im_height, im_width, channels])
    q = tf.reshape(q, [-1, im_height, im_width, channels])
    u = tf.reshape(u, [-1, im_height, im_width, channels]) # (num_unlabeled*n_way, h, w, c)

    with tf.GradientTape() as embedder_tape, tf.GradientTape() as weighter_tape:
        x_latent = embedder(x)
        q_latent = embedder(q)
        u_latent = embedder(u) #(num_unlabeled, latent_dim)

        latent_dim = x_latent.shape[-1]

        labeled_prototypes = get_prototypes(x_labels, x_latent, num_classes, num_support) # (num_classes, latent_dim)
        if not baseline:
            #Create weights for unlabeled samples
            flattened_protos = tf.reshape(labeled_prototypes, (-1,)) #(num_classes * latent_dim,)
            flattened_protos = tf.expand_dims(flattened_protos, axis=0)
            flattened_protos = tf.tile(flattened_protos, multiples=[num_unlabeled, 1]) # (num_unlabeled, num_classes*latent_dim)
       
            weight_input = tf.concat([u_latent, flattened_protos], axis=1) #(num_unlabeled, latent_dim*num_classes+1)
            u_weights = weighter(weight_input) #(num_unlabeled, num_classes)

            new_prototypes = update_protos(u_latent, u_weights, labeled_prototypes, num_support=num_support)
        else:
            new_prototypes = labeled_prototypes

        ce_loss, acc = ProtoLoss(new_prototypes, x_latent, q_latent, q_labels, num_classes, num_unlabeled, num_support, num_queries)

    if not eval:
        if not baseline:
            weighter_grads = weighter_tape.gradient(ce_loss, weighter.trainable_variables)
            embedder_grads = embedder_tape.gradient(ce_loss, embedder.trainable_variables)
            optim.apply_gradients(zip(embedder_grads, embedder.trainable_variables))
            optim.apply_gradients(zip(weighter_grads, weighter.trainable_variables))

        else:
            embedder_grads = embedder_tape.gradient(ce_loss, embedder.trainable_variables)
            optim.apply_gradients(zip(embedder_grads, embedder.trainable_variables))
    return ce_loss, acc

#def proto_net_eval(embedder, weighter, optim, x, q, u, labels_ph):


def proto_net_eval(model, x, q, labels_ph):
    #print(f"Initial label shape: {labels_ph.shape}")
    num_classes, num_support, im_height, im_width, channels = x.shape
    num_queries = q.shape[1]
    x = tf.reshape(x, [-1, im_height, im_width, channels])
    q = tf.reshape(q, [-1, im_height, im_width, channels])

    x_latent = model(x)
    q_latent = model(q)
    #u_latent = model(u)
    x_labels = labels_ph[:, :num_support, :]
    q_labels = labels_ph[:, num_support:, :]
    #print(f"sliced Query label shape: {labels_ph.shape}")

    prototypes = get_prototypes(x_labels, x_latent, num_classes, num_support)
    # ProtoLoss(new_prototypes, x_latent, q_latent, q_labels, num_classes, num_unlabeled, num_support, num_queries)
    ce_loss, acc = ProtoLoss(prototypes, x_latent, q_latent, q_labels, num_classes, 0, num_support, num_queries)

    return ce_loss, acc 

def run_protonet(data_path='../omniglot_resized', baseline=False, n_way=20, k_shot=1, n_query=5, n_unlabeled = 5,
                 n_meta_test_way=20, k_meta_test_shot=5, n_meta_test_query=5, n_meta_test_unlabeled = 5,
                 logdir="../logs/"):
    n_epochs = 20
    n_episodes = 100

    im_width, im_height, channels = 28, 28, 1
    num_filters = 32
    latent_dim = 16
    hidden_dim = 32
    num_conv_layers = 3
    n_meta_test_episodes = 1000

    if baseline:
        print("Running baseline model (no unlabeled refinement of prototypes)")

    output_data = pd.DataFrame(columns=[
                                        'iter', 'tr_acc',
                                        'val_acc',
                                        'tr_loss', 'val_loss',
                                        ])

    model = ProtoNet([num_filters]*num_conv_layers, latent_dim)

    weighter = Sampler(hidden_dim, n_way)
    
    optimizer = tf.keras.optimizers.Adam()

    exp_string = ('proto_tr_cls_'+str(n_way)+'.tr_k_shot_' + str(k_shot) +
                    'ts_cls_' + str(n_meta_test_way) + '.ts_k_shot_' + str(k_meta_test_shot))
    
    full_log_file = logdir + exp_string + '.csv'

    # call DataGenerator with k_shot+n_query samples per class
    # Increase number of samples for each task (u unlabeled samples)
    data_generator = DataGenerator(n_way, k_shot+n_query+n_unlabeled, n_meta_test_way, k_meta_test_shot+n_meta_test_query+n_meta_test_unlabeled,
                                    config={'data_folder': data_path})


    for ep in range(n_epochs):
        for epi in range(n_episodes):
            #print("epi ", epi)
            # sample batch, partition into support/query/unlabeled, reshape
            images, labels = data_generator.sample_batch("meta_train", batch_size=1, shuffle=False)
            support = tf.reshape(images[0, :, :k_shot, :],
                                shape=(n_way, k_shot, im_height, im_width, 1))
            query = tf.reshape(images[0, :, k_shot:k_shot + n_query, :],
                                shape=(n_way, n_query, im_height, im_width, 1))
            unlabeled = tf.reshape(images[0, :, n_query + k_shot:, :],
                                shape=(n_way, n_unlabeled, im_height, im_width, 1))

            #print(f"Shape of labels: {labels.shape}\tDesired shape: {(n_way, n_query, n_way)}")
            labels = tf.reshape(labels[0, :, :k_shot + n_query, :], shape=(n_way, k_shot+n_query, n_way)) # (5, 10, 5)
            
            ls, ac = proto_net_train_step(model, weighter, optimizer, x=support, q=query, u=unlabeled, labels_ph=labels, baseline=baseline)
            if (epi+1) % 50 == 0:
                #print("hello")
                # sample batch, partition into support/query, reshape NOT unlabeled
                images, labels = data_generator.sample_batch("meta_val", batch_size=1, shuffle=False)
                support = tf.reshape(images[0, :, :k_shot, :],
                                shape=(n_way, k_shot, im_height, im_width, 1))
                query = tf.reshape(images[0, :, k_shot:k_shot + n_query, :],
                                    shape=(n_way, n_query, im_height, im_width, 1))
                unlabeled = tf.reshape(images[0, :, n_query + k_shot:, :],
                                shape=(n_way, n_unlabeled, im_height, im_width, 1))
                # unlabeled = tf.reshape(images[0, :, n_query + k_shot:, :],
                                #shape=(n_way, n_unlabeled, im_height, im_width, 1))
                labels = tf.reshape(labels[0, :, :k_shot + n_query, :], shape=(n_way, k_shot+n_query, n_way))
                
                val_ls, val_ac = proto_net_train_step(model, weighter, optimizer, x=support, q=query, u=unlabeled, labels_ph=labels, eval=True)
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
        query = tf.reshape(images[0, :, k_meta_test_shot:k_meta_test_shot+n_meta_test_query, :],
                            shape=(n_meta_test_way, n_meta_test_query, im_height, im_width, 1))
        labels = tf.reshape(labels[0, :, :k_meta_test_shot+n_meta_test_query, :],
                            shape=(n_meta_test_way, k_meta_test_shot+n_meta_test_query, n_meta_test_way))
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
