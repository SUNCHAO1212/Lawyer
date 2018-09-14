# -*- coding:UTF-8 -*-
# !/usr/bin/env python3


import pickle
import os
import tensorflow as tf
from torchtext import data
from models.TextCNN import TextCNN


class MyDataset(object):
    def __init__(self, flags=None, is_train=True):

        # parameters, need flags
        self.flags = flags
        # dataset_path = "../corpus/training_data.tsv"
        dataset_path = "../corpus/train.csv"
        save_path = "../savepath"
        vocab_file = "latest_vocab.pkl"
        batch_size = 64

        # Field
        self.TEXT = data.Field(tokenize=self.tokenizer, fix_length=40, sequential=True)
        self.LABEL = data.Field(tokenize=self.tokenizer, unk_token=None, sequential=True)
        # print(vars(self.TEXT))
        # Dataset
        self.dataset = data.TabularDataset(
            path=dataset_path, format='csv',
            fields={
                'text': ('text', self.TEXT),
                'label': ('label', self.LABEL)
            }
        )
        # todo how to get label set
        labels = []
        for label in self.dataset.label:
            labels = list(set(label) | set(labels))
        labels = [labels]
        # Vocab
        if is_train:
            # modelpath = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            self.TEXT.build_vocab(self.dataset, vectors="glove.6B.100d")
            self.LABEL.build_vocab(labels)  # TODO multilabel
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

    def embed_label(self, batch_label):
        y_batch = []  # [batch_size, num_class]
        for i, row in enumerate(batch_label.data.transpose(1, 0)):
            temp = [0] * (len(self.LABEL.vocab.itos) - 1)
            for j, col in enumerate(row):
                if col != 0:
                    temp[col-1] = 1
            y_batch.append(temp)

        return y_batch

    @staticmethod
    def tokenizer(string):
        return list(string.split())


def main():
    # dataset = MyDataset(is_train=True)
    #
    # for i, batch in enumerate(dataset.train_iter):
    #     # text, label = batch.text, batch.label
    #
    #     # x_batch = text.data.transpose(1, 0)
    #     # temp1 = label.data.transpose(1, 0)
    #     # temp2 = torch.ones(temp1.shape).long() - temp1
    #     # y_batch = torch.cat((temp1, temp2), 1)
    #
    #     text, label = dataset.get_text_and_label(batch.text, batch.label)
    #     pass

    dataset = MyDataset(is_train=True)
    cnn = TextCNN(dataset)

    train_op = tf.train.AdamOptimizer(0.001).minimize(cnn.loss)

    global_step = tf.Variable(0, name="global_step", trainable=False)
    # optimizer = tf.train.AdamOptimizer(1e-2)
    # grads_and_vars = optimizer.compute_gradients(cnn.loss)
    # train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(50000):
            for i, batch in enumerate(dataset.train_iter):
                text = batch.text.data.transpose(1, 0)
                label = dataset.embed_label(batch.label)

                feed_dict = {
                    cnn.input_x: text,
                    cnn.input_y: label,
                    cnn.keep_prob: 0.5
                }
                _, step, loss, accuracy, y, y_ = sess.run(
                    [train_op, global_step, cnn.loss, cnn.accuracy, cnn.input_y, cnn.y_],
                    feed_dict)
                print("{}: step {}, loss {:g}, acc {:g}".format(i, step, loss, accuracy))
    pass


if __name__ == '__main__':
    main()
    # print(__doc__)
