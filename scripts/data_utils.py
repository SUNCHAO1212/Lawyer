# -*- coding:UTF-8 -*-
# !/usr/bin/env python3

import pickle
import time
import os
import tensorflow as tf
from torchtext import data
from scripts.text_cnn import TextCNN
import torch


class MyDataset(object):
    def __init__(self, flags=None, is_train=True):

        # parameters, need flags
        self.flags = flags
        dataset_path = "../corpus/training_data.csv"
        save_path = "../savepath"
        vocab_file = "latest_vocab.pkl"
        batch_size = 64

        # Field
        self.TEXT = data.Field(tokenize=self.tokenizer, fix_length=40, sequential=True)
        self.LABEL = data.Field(unk_token=None, pad_token=None, sequential=True)
        # print(vars(self.TEXT))
        # Dataset
        self.dataset = data.TabularDataset(
            path=dataset_path, format='csv',
            fields={
                'text': ('text', self.TEXT),
                'label': ('label', self.LABEL)
            }
        )
        # Vocab
        if is_train:
            # modelpath = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            self.TEXT.build_vocab(self.dataset, vectors="glove.6B.100d")
            self.LABEL.build_vocab(['0', '1'])  # TODO multilabel
            self.vocab = self.TEXT.vocab
            with open(os.path.join(save_path, vocab_file), 'wb') as fo:
                pickle.dump(self.vocab, fo)
                print(f"[info] Vocab saved to path: {os.path.join(save_path, vocab_file)}")
        else:
            if os.path.exists(os.path.join(save_path, vocab_file)):
                with open(os.path.join(save_path, vocab_file), 'rb') as f:
                    self.vocab = pickle.load(f)
                    print(f"[info] Vocab loaded from path: {os.path.join(save_path, vocab_file)}")
            else:
                print(f"[Error] Path not exist: {save_path}")
        # iterator
        self.train_iter = data.BucketIterator(dataset=self.dataset,
                                              batch_size=batch_size,
                                              device=-1,
                                              repeat=False,
                                              shuffle=True)

    @staticmethod
    def tokenizer(string):
        return list(string.split())


def main():
    dataset = MyDataset(is_train=True)
    cnn = TextCNN(sequence_length=40, num_classes=2, vocab_size=len(dataset.vocab.itos),
                  embedding_size=100, filter_sizes=[3, 4, 5], num_filters=2)
    # cnn = TextCNN()

    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(1e-3)
    grads_and_vars = optimizer.compute_gradients(cnn.loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

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
                    cnn.dropout_keep_prob: 0.5
                }
                _, step, loss, accuracy, y, y_ = sess.run(
                    [train_op, global_step, cnn.loss, cnn.accuracy, cnn.input_y, cnn.scores],
                    feed_dict)
                # print(y)
                # print(y_)
                print("{}: step {}, loss {:g}, acc {:g}".format(i, step, loss, accuracy))

    pass


if __name__ == '__main__':
    main()

