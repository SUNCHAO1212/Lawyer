# -*- coding:UTF-8 -*-
# !/usr/bin/env python3


import tensorflow as tf

from models.TextCNN import TextCNN
from data_loader.MyDataset import MyDataset

# All paths
tf.flags.DEFINE_string("data_path", "corpus/training_data.csv", "data_path")
tf.flags.DEFINE_string("save_path", "savepath/", "save_path")
# Training parameters
tf.flags.DEFINE_float("learning_rate", 1e-3, "learning_rate")
tf.flags.DEFINE_integer("batch_size", 64, "batch_size")
tf.flags.DEFINE_integer("num_epochs", 500, "num_epoch")
# Model hyperparameters
tf.flags.DEFINE_integer("sequence_length", 40, "sequence_length")
tf.flags.DEFINE_list("kernel_sizes", [3, 4, 5], "kernel_sizes/filter_sizes")
tf.flags.DEFINE_integer("kernel_num", 2, "number of each kernel_size")
# TODO Logging


FLAGS = tf.flags.FLAGS
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print(f"{attr.upper()} = {value.value}")


def train():
    dataset = MyDataset(flags=FLAGS)
    text_vocab = dataset.TEXT.vocab
    cnn = TextCNN(dataset, flags=FLAGS)  # TODO input vocab, class_num

    # train
    train_op = tf.train.AdamOptimizer(0.001).minimize(cnn.loss)
    global_step = tf.Variable(0, name="global_step", trainable=False)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(500):
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


def main(argv=None):
    train()


if __name__ == '__main__':
    tf.app.run()
