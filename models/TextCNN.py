# -*- coding:UTF-8 -*-
# !/usr/bin/env python3

import tensorflow as tf
import torch
from scripts.data_utils import MyDataset


class TextCNN(object):
    def __init__(self, dataset, flags=None):
        # TODO input flags
        self.dataset = dataset
        # parameter
        # batch_size = 2
        # sequence_length = 40
        # embedding_dim = 100
        # kernel_sizes = [3, 4, 5]
        # kernel_num = 2  # conv1 output channel
        # class_num = 2
        self.build_graph()
        pass

    def build_graph(self):
        sequence_length = 40
        embedding_dim = 100
        kernel_sizes = [3, 4, 5]
        kernel_num = 2  # conv1 output channel
        class_num = 4
        # input
        self.input_x = tf.placeholder(tf.int32, shape=[None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, shape=[None, class_num], name="input_y")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        # Embedding layer
        vocab = self.dataset.vocab.vectors
        self.embedding = tf.Variable(initial_value=vocab, trainable=True)  # [vocab_size, embedding_dim]
        # self.embedding = tf.Variable(tf.random_uniform([len(vocab), embedding_dim], -1.0, 1.0))
        self.embedding_h = tf.nn.embedding_lookup(self.embedding,
                                                  self.input_x)  # [batch_size, vocab_size, embedding_dim]
        self.embedding_h_expand = tf.expand_dims(self.embedding_h, -1)  # [batch_size, vocab_size, embedding_dim, 1]
        # Convolution layer, max pooling layer
        self.pooled = []
        for i, kernel_size in enumerate(kernel_sizes):
            with tf.name_scope(f"conv-maxpool-{kernel_size}"):
                kernel_w = tf.get_variable(f"kernel-w-{kernel_size}", shape=[kernel_size, embedding_dim, 1, kernel_num])
                kernel_b = tf.get_variable(f"kernel-b-{kernel_size}", shape=[kernel_num])
                conv = tf.nn.conv2d(self.embedding_h_expand,
                                    kernel_w,
                                    strides=[1, 1, 1, 1],
                                    padding='VALID',
                                    name="conv")
                conv_h = tf.nn.relu(tf.nn.bias_add(conv, kernel_b))
                max_pool_h = tf.nn.max_pool(conv_h,
                                            ksize=[1, sequence_length - kernel_size + 1, 1, 1],
                                            strides=[1, 1, 1, 1],
                                            padding='VALID',
                                            data_format='NHWC',  # number, in_height, in_width, in_channels
                                            )
                self.pooled.append(max_pool_h)
        self.pooled_h = tf.concat(self.pooled, 3)
        fc_in = kernel_num * len(kernel_sizes)
        self.pooled_h_flat = tf.reshape(self.pooled_h, [-1, fc_in])
        # drop out
        self.dropout_h = tf.nn.dropout(self.pooled_h_flat, keep_prob=self.keep_prob)
        # Fully connected layer
        self.fc_w = tf.get_variable(name="fc_w", shape=[fc_in, class_num],
                                    initializer=tf.contrib.layers.xavier_initializer())
        self.fc_b = tf.Variable(tf.constant(0.1, name="fc_b", shape=[class_num]))
        self.y_ = tf.nn.xw_plus_b(self.dropout_h, self.fc_w, self.fc_b)

        # loss
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.y_, labels=self.input_y)
        # self.cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.y_, labels=self.input_y)  # multilabel
        self.loss = tf.reduce_mean(self.cross_entropy)
        # TODO accuracy
        self.arg_y_ = tf.argmax(self.y_, 1)
        self.arg_y = tf.argmax(self.input_y, 1)
        self.correct_predictions = tf.equal(self.arg_y, self.arg_y_)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, tf.float32), name="accuracy")

    def train(self):
        learning_rate = 0.001
        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        # self.optimizer = tf.train.AdamOptimizer(learning_rate)
        # self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
        # self.train_op = self.optimizer.apply_gradients(self.grads_and_vars, global_step=self.global_step)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(500):
                for j, batch in enumerate(self.dataset.train_iter):
                    text, label = batch.text, batch.label
                    input_x = text.data.transpose(1, 0)
                    temp1 = label.data.transpose(1, 0)
                    # TODO transfer tensor from [batch, 1] to [batch, class_num]
                    temp2 = torch.ones(temp1.shape).long() - temp1
                    input_y = torch.cat((temp1, temp2), 1)
                    _, loss, acc, y, y_, arg_y, arg_y_, cross_entropy = sess.run(
                        [self.train_op, self.loss, self.accuracy, self.input_y, self.y_, self.arg_y, self.arg_y_,
                         self.cross_entropy],
                        feed_dict={self.input_x: input_x, self.input_y: input_y,
                                   self.keep_prob: 1.0})
                    print(f"Epoch: {i} | batch: {j} | loss: {loss} | accuracy: {acc}")


def main():
    print("Run main()")
    dataset = MyDataset(is_train=True)
    cnn = TextCNN(dataset)

    train_op = tf.train.AdamOptimizer(0.001).minimize(cnn.loss)

    global_step = tf.Variable(0, name="global_step", trainable=False)
    # optimizer = tf.train.AdamOptimizer(1e-2)
    # grads_and_vars = optimizer.compute_gradients(cnn.loss)
    # train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(500):
            for i, batch in enumerate(dataset.train_iter):
                text, label = batch.text, batch.label
                x_batch = text.data.transpose(1, 0)
                temp1 = label.data.transpose(1, 0)
                temp2 = torch.ones(temp1.shape).long() - temp1
                y_batch = torch.cat((temp1, temp2), 1)
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.keep_prob: 0.5
                }
                _, step, loss, accuracy, y, y_ = sess.run(
                    [train_op, global_step, cnn.loss, cnn.accuracy, cnn.input_y, cnn.y_],
                    feed_dict)
                print("{}: step {}, loss {:g}, acc {:g}".format(i, step, loss, accuracy))
    print("Run main()")


def main2():
    print("Run main2()")
    dataset = MyDataset()
    cnn = TextCNN(dataset)
    cnn.train()
    print("Run main2()")


if __name__ == '__main__':
    # main()
    main2()
