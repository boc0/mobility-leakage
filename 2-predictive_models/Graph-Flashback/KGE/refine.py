import os
from .constant import DATA_NAME, SCHEME


def refine_triplets():
    new_train_triplets = './dataset/{}/{}_scheme{}/new_final_train_triplets.txt'.format(DATA_NAME, DATA_NAME, SCHEME)
    new_test_triplets = './dataset/{}/{}_scheme{}/new_final_test_triplets.txt'.format(DATA_NAME, DATA_NAME, SCHEME)
    os.makedirs(os.path.dirname(new_train_triplets), exist_ok=True)
    f_new_train = open(new_train_triplets, 'w+')
    f_new_test = open(new_test_triplets, 'w+')

    train_friend_triplets = set()
    train_spatial_triplets = set()
    ratio = 0.8

    # 原封不动地把interact和temporal三元组写入新文件中
    with open('./dataset/{}/{}_scheme{}/final_test_triplets.txt'.format(DATA_NAME, DATA_NAME, SCHEME), 'r') as f_test:
        for line in f_test.readlines():
            tokens = tuple(line.strip('\n').split('\t'))
            if len(tokens) != 3:
                continue
            h, t, r = tokens  # str
            f_new_test.write(h + '\t')
            f_new_test.write(t + '\t')
            f_new_test.write(r + '\n')

    train_path = './dataset/{}/{}_scheme{}/final_train_triplets.txt'.format(DATA_NAME, DATA_NAME, SCHEME)
    with open(train_path, 'r') as f_train:
        for line in f_train.readlines():
            tokens = tuple(line.strip('\n').split('\t'))
            if len(tokens) != 3:
                continue
            h, t, r = tokens  # str
            relation = int(tokens[2])
            if relation == 0:  # interact
                f_new_train.write(h + '\t')
                f_new_train.write(t + '\t')
                f_new_train.write(r + '\n')
            elif relation == 1:  # temporal
                f_new_train.write(h + '\t')
                f_new_train.write(t + '\t')
                f_new_train.write(r + '\n')
            elif relation == 2:  # spatial (symmetric)
                if (t, h, r) not in train_spatial_triplets:
                    train_spatial_triplets.add((h, t, r))
            else:  # friend (symmetric)
                if (t, h, r) not in train_friend_triplets:
                    train_friend_triplets.add((h, t, r))
    print(len(train_spatial_triplets))  # size spatial unique (undirected)
    print(len(train_friend_triplets))  # size friend unique (undirected)

    train_spatial_triplets_len = int(len(train_spatial_triplets) * ratio)
    train_friend_triplets_len = int(len(train_friend_triplets) * ratio)

    count_spatial = 0
    count_friend = 0

    for elem_spatial in train_spatial_triplets:
        h, t, r = elem_spatial
        if count_spatial < train_spatial_triplets_len:
            f_new_train.write(h + '\t')
            f_new_train.write(t + '\t')
            f_new_train.write(r + '\n')

            f_new_train.write(t + '\t')
            f_new_train.write(h + '\t')
            f_new_train.write(r + '\n')
            count_spatial += 1
        else:
            f_new_test.write(h + '\t')
            f_new_test.write(t + '\t')
            f_new_test.write(r + '\n')

            f_new_test.write(t + '\t')
            f_new_test.write(h + '\t')
            f_new_test.write(r + '\n')
    print(count_spatial)
    for elem_friend in train_friend_triplets:
        h, t, r = elem_friend
        if count_friend < train_friend_triplets_len:
            f_new_train.write(h + '\t')
            f_new_train.write(t + '\t')
            f_new_train.write(r + '\n')

            f_new_train.write(t + '\t')
            f_new_train.write(h + '\t')
            f_new_train.write(r + '\n')
            count_friend += 1
        else:
            f_new_test.write(h + '\t')
            f_new_test.write(t + '\t')
            f_new_test.write(r + '\n')

            f_new_test.write(t + '\t')
            f_new_test.write(h + '\t')
            f_new_test.write(r + '\n')
    print(count_friend)
    f_new_train.close()
    f_new_test.close()


if __name__ == '__main__':
    refine_triplets()
