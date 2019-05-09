import linecache
import json
import random
from tokenization import convert_tokens_to_ids
from utils import buildDicFromVocab
def handleVectors(src_path, dst_path):
    try:
        lines = linecache.getlines(src_path)
        out = open(dst_path, 'w+')
        for line in lines:
            line = json.loads(line)
            line = line['features'][0]['layers'][0]['values']
            print(len(line))
            line = str(line)
            line = line .replace('[', '')
            line = line.replace(']', '')
            line = line.replace(' ', '')
            out.writelines(str(line) + '\n')
    except IOError:
        print("path not found")
    finally:
        out.close()

def buildVocabDic(vocab_path, out_vocab_dic_path):
    reversed_dict, word_dict = buildDicFromVocab(vocab_path)
    try:
        f = open(out_vocab_dic_path, 'w+', encoding='UTF-8')
        f.write(str(reversed_dict))
        f.close()
    except IOError:
        print("path not found")


def convertVocab2Embeddings(train_data_path, embedding_vector_path, vocab_path, new_vocab_embedding_path):
    try:
        train_data = open(train_data_path, 'r', encoding='UTF-8')
        embedding_vector = open(embedding_vector_path, 'r', encoding='UTF-8')
        vocab= open(vocab_path, 'r', encoding='UTF-8')
        new_vocab_embedding = open(new_vocab_embedding_path, 'w+', encoding='UTF-8')
        list_train_data = train_data.readlines()
        list_embedding_vector = embedding_vector.readlines()
        dic_vocab, _ = buildDicFromVocab(vocab_path)
        list_vocab = vocab.readlines()
        list_vocab = [random.uniform(-1, 1) for i in list_vocab]

        #new_vocab = tf.random_uniform([len(list_vocab)], -1.0, 1.0)
        for (line_train_data, line_embedding_vector) in zip(list_train_data, list_embedding_vector):
            line_train_data = line_train_data.replace("\n", '')
            line_train_data = line_train_data.split(' ')
            line_embedding_vector = line_embedding_vector[1:len(line_embedding_vector)-2]
            line_embedding_vector = line_embedding_vector.replace("\n", '')
            line_embedding_vector = line_embedding_vector.split(',')
            ouput = convert_tokens_to_ids(dic_vocab, line_train_data)

            for (o, embedding) in zip(ouput, line_embedding_vector):
                embedding = embedding.strip()
                embedding = float(embedding)
                if o < len(list_vocab)-1:
                    list_vocab[o] = embedding
        new_vocab_embedding.write(str(list_vocab))
    except IOError:
        print("path not found")

    finally:
        train_data.close()
        embedding_vector.close()
        new_vocab_embedding.close()

handleVectors('../data/config/vocab-output.json', '../data/config/vocab-output-formal.txt')
#buildVocabDic('../uncased_L-12_H-768_A-12/vocab.txt', '../data/vocab_dict.txt')
#convertVocab2Embeddings('../data/code-100.txt', '../data/code-100-formal.txt', '../uncased_L-12_H-768_A-12/vocab.txt', '../data/vocab_embedding.txt')