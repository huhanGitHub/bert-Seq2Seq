#!/usr/bin/python
# -*- coding: utf-8 -*-

list = []
with open('../data/vocab/code.txt', 'r') as code, open('../data/vocab/code_train.txt', 'w+') as code_train, open('../data/vocab/comment.txt', 'r') as comment, open('../data/vocab/comment_train.txt', 'w+') as comment_train:
    code = code.readlines()
    comment = comment.readlines()
    for i in range(len(code)):
        ii = code[i].split(' ')
        if len(ii) < 64 and len(ii) >1 and len(comment[i].split(' ')) < 64:
             code_train.write(str(code[i]))
             comment_train.write(str(comment[i]))