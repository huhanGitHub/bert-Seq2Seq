import tensorflow as tf
import pickle
from Seq2seq.model import Model
from Seq2seq.utils import batch_iter, buildDicFromVocab, loadData
import numpy as np
vocal_path = '../data/config/vocab.txt'
output_path='../data/test_output.txt'
article_max_len = 64
summary_max_len = 64
input_path = '../data/code_test.txt'
model_path = '/sda/qiuyuanchen/saved_model/'

with open("args.pickle", "rb") as f:
    args = pickle.load(f)

print("Loading dictionary...")
#word_dict, reversed_dict, article_max_len, summary_max_len = build_dict("valid", args.toy)
word_dict, reversed_dict = buildDicFromVocab(vocal_path)
print("Loading validation dataset...")
#valid_x = build_dataset("valid", word_dict, article_max_len, summary_max_len, args.toy)
valid_x = loadData(word_dict, input_path)

valid_x_len = [len([y for y in x if y != 0]) for x in valid_x]

with tf.Session() as sess:
    print("Loading saved model...")
    model = Model(reversed_dict, article_max_len, summary_max_len, args, forward_only=True)
    saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state(model_path)
    saver.restore(sess, ckpt.model_checkpoint_path)

    batches = batch_iter(valid_x, [0] * len(valid_x), args.batch_size, 1)

    print("Writing summaries to" + output_path)
    for batch_x, _ in batches:
        batch_x_len = [len([y for y in x if y != 0]) for x in batch_x]
        for i in batch_x:
            while len(i) < 64:
                print(len(i))
                i.append(0)
        #batch_x = batch_x.replace('\n', '')
        batch_x = np.array([np.array(i) for i in batch_x])

        valid_feed_dict = {
            model.batch_size: len(batch_x),
            model.X: batch_x,
            model.X_len: batch_x_len,
        }

        prediction = sess.run(model.prediction, feed_dict=valid_feed_dict)
        prediction_output = [[reversed_dict[y] for y in x] for x in prediction[:, 0, :]]

        with open(output_path, "w") as f:
            for line in prediction_output:
                summary = list()
                for word in line:
                    if word == "[SEP]":
                        break
                    if word not in summary:
                        summary.append(word)
                print(" ".join(summary), file=f)

    print('Summaries are saved to "result.txt"...')
