#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import re
list = []
list.append('[PAD]')
list.append('[CLS]')
list.append('[SEP]')
list.append('[MASK]')
list.append('[UNK]')
vocab = open('../Seq2seq/vocab.txt', 'w+')
def gci(filepath, suffix):
#遍历filepath下所有文件，包括子目录
  files = os.listdir(filepath)
  for fi in files:
    fi_d = os.path.join(filepath, fi)
    if os.path.isdir(fi_d):
      gci(fi_d, suffix)
    else:
      #print(fi_d)
      if os.path.splitext(fi_d)[1] == suffix:
          print(fi_d)
          with open(fi_d) as f:
              lines = f.readlines()
              for line in lines:
                  print(line)
                  line = re.sub(' +',' ',line)
                  line = line.replace('\n', '')
                  print(line)
                  line = line.split(' ')
                  for item in line:
                    if item not in list:
                        list.append(item)

#递归遍历/root目录下所有文件
gci('../data/vocab/', '.txt')
for line in list:
    if line != '':
        vocab.write(str(line)+'\n')
vocab.close()