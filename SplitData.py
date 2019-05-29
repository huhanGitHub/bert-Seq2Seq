data_path = 'data/input.txt'
pretrain_data_out = 'data/test/pretrain_data.txt'
train_data_out = 'data/test/train_data.txt'
ratio = 0.1
pretrain_data_path = open(pretrain_data_out, 'w+')
train_data_path = open(train_data_out, 'w+')

with open(data_path) as f:
    list = f.readlines()
    print(len(list))
    train_data_len = round(ratio * len(list))
    pretrain_data = list[:len(list)-train_data_len]
    train_data = list[len(list)-train_data_len:]
    for line in pretrain_data:
        if line != '':
            pretrain_data_path.write(str(line))
    pretrain_data_path.close()

    for line in train_data:
        if line != '':
            train_data_path.write(str(line) )
    train_data_path.close()


