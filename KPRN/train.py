import argparse
import os.path
import pickle
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim

import consts as consts
from data.dataloader import TrainInteractionData, my_collate, sort_batch
from format import format_path
from model import KPRN
from path_extraction import find_paths_user_to_songs

random.seed(1)

argparse = argparse.ArgumentParser(description='Model Parameters')

argparse.add_argument('--subnetwork_type', type=str, default='standard',
                      choices=['dense', 'standard', 'sparse', 'full'])
argparse.add_argument('--type_embedding_dim', type=int, default=32)
argparse.add_argument('--relation_embedding_dim', type=int, default=32)
argparse.add_argument('--entity_embedding_dim', type=int, default=64)
argparse.add_argument('--lstm_hidden_dim', type=int, default=256)

argparse.add_argument('--epochs', type=int, default=200)
argparse.add_argument('--batch_size', type=int, default=256)
argparse.add_argument('--lr', type=float, default=0.002)
argparse.add_argument('--l2_reg', type=float, default=0.0001)
argparse.add_argument('--optimizer', type=str, default='Adam')
argparse.add_argument('--gamma', type=float, default=1, help='weighted pooling')

argparse.add_argument('--user_limit', type=int, default=100, help='max number of users to find paths for')
argparse.add_argument('--samples', type=int, default=-1,
                      help='number of paths to sample for each interaction(-1 means all paths)')

argparse.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
argparse.add_argument('--mode', type=str, default='train', choices=['train', 'eval'])

argparse.add_argument('--kg_path_file', type=str, default='train_inters_standard_100.txt')
argparse.add_argument('--model_path', type=str, default='kprn.pth')
argparse.add_argument('--find_path', type=bool, default=False)

args = argparse.parse_args()


def makedir(dir):
    try:
        os.mkdir(dir)
    except FileExistsError:
        print("The directory has already existed!")


def sample_paths(paths, samples):
    index_list = list(range(len(paths)))
    random.shuffle(index_list)
    indices = index_list[:samples]
    return [paths[i] for i in indices]


# the file in song_ix_mapping with name x_x_to_ix.dict
def load_string_to_ix_dicts():
    prefix = 'data/' + consts.IX_MAPPING_DIR + '/' + args.subnetwork_type + '_'
    with open(prefix + consts.TYPE_TO_IX, 'rb') as file:
        type_to_ix = pickle.load(file)
    with open(prefix + consts.RELATION_TO_IX, 'rb') as file:
        relation_to_ix = pickle.load(file)
    with open(prefix + consts.ENTITY_TO_IX, 'rb') as file:
        entity_to_ix = pickle.load(file)

    return type_to_ix, relation_to_ix, entity_to_ix


def load_song_ix_data():
    prefix = 'data/' + consts.IX_DATA_DIR + '/' + args.subnetwork_type + '_ix_'
    with open(prefix + consts.SONG_PERSON_DICT, 'rb') as file:
        song_person = pickle.load(file)
    with open(prefix + consts.PERSON_SONG_DICT, 'rb') as file:
        person_song = pickle.load(file)
    with open(prefix + consts.SONG_USER_DICT, 'rb') as file:
        song_user = pickle.load(file)
    with open(prefix + consts.USER_SONG_DICT, 'rb') as file:
        user_song = pickle.load(file)

    return song_person, person_song, song_user, user_song


def load_data(song_person, person_song, all_user_song, all_song_user, train_user_song, train_song_user, neg_samples,
              entity_to_ix, type_to_ix, relation_to_ix, kg_path_file, len_3_branch, len_5_branch, limit=10,
              version='train', samples=-1):
    prefix = 'data/' + consts.PATH_DATA_DIR + '/'
    makedir(prefix)

    write_file = open(prefix + kg_path_file, 'w')

    pos_path_not_found = 0
    pos_interactions_nums = 0
    neg_interactions_nums = 0
    avg_pos_path_nums, avg_neg_path_nums = 0, 0

    for user, pos_songs in tqdm(list(train_user_song.items())[:limit]):
        pos_interactions_nums += len(pos_songs)
        # song_to_paths key: song(the song may not be interacted with, may be by other user or person), value: a list of path from user to song
        # neg_songs_with_path key: negative song value: a list of path from user to negative song
        song_to_paths, neg_songs_with_paths = None, None
        # the pointer in the neg song list, to check if there is enough negative samples(parameter: neg_sample)
        cur_idx = 0

        for pos_song in pos_songs:
            # use in eval stages to store all paths(neg and pos), needs to record 100 neg path + 1 pos path
            interactions = []
            if song_to_paths is None:
                if version == 'train':
                    song_to_paths = find_paths_user_to_songs(user, song_person, person_song, train_song_user,
                                                             train_user_song, 3, len_3_branch)
                    song_to_paths_len5 = find_paths_user_to_songs(user, song_person, person_song, train_song_user,
                                                                  train_user_song, 5, len_5_branch)
                # eval
                else:
                    song_to_paths = find_paths_user_to_songs(user, song_person, person_song, all_song_user,
                                                             all_user_song, 3, len_3_branch)
                    song_to_paths_len5 = find_paths_user_to_songs(user, song_person, person_song, all_song_user,
                                                                  all_user_song, 5, len_5_branch)

                # song_to_path is defaultlist, so it will not error
                for song in song_to_paths_len5.keys():
                    song_to_paths[song].extend(song_to_paths_len5[song])

                # for a user, each pos_song, sampling neg_samples
                all_pos_songs = set(all_user_song[user])
                # pos_songs
                songs_with_paths = set(song_to_paths.keys())
                # get neg use set difference
                neg_songs_with_paths = list(songs_with_paths.difference(all_pos_songs))

                top_neg_songs = neg_songs_with_paths
                random.shuffle(top_neg_songs)

            pos_paths = song_to_paths[pos_song]
            if len(pos_paths) > 0:
                # samples==-1 means use all path
                if samples != -1:
                    pos_paths = sample_paths(pos_paths, samples)
                # where 1 means positive label
                interaction = (format_path(pos_paths, entity_to_ix, type_to_ix, relation_to_ix), 1)

                if version == 'train':
                    write_file.write(repr(interaction) + '\n')
                elif version == 'eval':
                    interactions.append(interaction)

                avg_pos_path_nums += len(pos_paths)

            else:
                pos_path_not_found += 1
                # if not pos paths, either not neg paths(continue)
                continue

            # for eval stage, it is a flag whether 101 paths are sampled
            found_all_samples = True
            for i in range(neg_samples):
                if cur_idx >= len(top_neg_songs):
                    print("not enought neg paths, only found:", str(i))
                    found_all_samples = False
                    break

                neg_song = top_neg_songs[cur_idx]
                neg_paths = song_to_paths[neg_song]

                if samples != -1:
                    neg_paths = sample_paths(neg_paths, samples)
                interaction = (format_path(neg_paths, entity_to_ix, type_to_ix, relation_to_ix), 0)
                if version == 'train':
                    write_file.write(repr(interaction) + '\n')
                elif version == 'eval':
                    interactions.append(interaction)

                avg_neg_path_nums += len(neg_paths)
                neg_interactions_nums += 1
                cur_idx += 1

            if found_all_samples and version == 'eval':
                write_file.write(repr(interactions) + '\n')

    avg_neg_path_nums = avg_neg_path_nums / neg_interactions_nums
    avg_pos_path_nums = avg_pos_path_nums / (pos_interactions_nums - pos_path_not_found)

    print(f"number of pos paths attempted to find: {pos_interactions_nums}")
    print(f"number of pos paths not found: {pos_path_not_found}")
    print(f"avg num paths per positive interaction: {avg_pos_path_nums}")
    print(f"avg num paths per negative interaction: {avg_neg_path_nums}")

    write_file.close()


# need train
if args.mode == 'train' or (args.mode == 'eval' and os.path.exists(args.model_path) is False):

    # load data
    type_to_ix, relation_to_ix, entity_to_ix = load_string_to_ix_dicts()
    song_person, person_song, song_user, user_song = load_song_ix_data()

    print('Finding paths...')

    if args.find_path:
        # training data
        prefix = 'data/' + consts.IX_DATA_DIR + '/' + args.subnetwork_type + '_train_ix_'
        with open(prefix + consts.USER_SONG_DICT, 'rb') as file:
            train_user_song = pickle.load(file)
        with open(prefix + consts.SONG_USER_DICT, 'rb') as file:
            train_song_user = pickle.load(file)

        load_data(song_person, person_song, user_song, song_user, train_user_song, train_song_user,
                  consts.NEG_SAMPLES_TRAIN, entity_to_ix, type_to_ix, relation_to_ix, args.kg_path_file,
                  consts.LEN_3_BRANCH, consts.LEN_5_BRANCH_TRAIN, limit=args.user_limit, version='train',
                  samples=args.samples)

    interaction_data = TrainInteractionData(args.kg_path_file)
    dataloader = DataLoader(dataset=interaction_data, collate_fn=my_collate, batch_size=args.batch_size, shuffle=True)

    # create model and put it on device(cuda/cpu)
    kprn = KPRN(args.type_embedding_dim, args.relation_embedding_dim, args.entity_embedding_dim, args.lstm_hidden_dim,
                len(type_to_ix), len(relation_to_ix), len(entity_to_ix), args.gamma).to(args.device)
    optimizer = optim.Adam(kprn.parameters(), lr=args.lr, weight_decay=args.l2_reg)
    criterion = nn.NLLLoss()

    for epoch in range(args.epochs):
        total_loss = 0
        for interaction_batch, targets in tqdm(dataloader):
            # to store the data in batch to be sorted
            paths, lengths, inter_ids = [], [], []
            for inter_id, inter_paths in enumerate(interaction_batch):
                for path, length in inter_paths:
                    paths.append(path)
                    lengths.append(length)
                inter_ids.extend([inter_id for i in range(len(inter_paths))])

            paths = torch.LongTensor(paths)
            lengths = torch.LongTensor(lengths)
            inter_ids = torch.LongTensor(inter_ids)

            # why need to sort?
            # in model, it uses "pack_padded_sequence" and its parameter enforce_sorted is True, mean the sequence length should be from large to small
            s_path_batch, s_inter_ids, s_lengths = sort_batch(paths, inter_ids, lengths)

            interactions_path_plausibility = kprn(s_path_batch.to(args.device), s_lengths)

            first = True
            # i is the index of row of data in batch, which is a set of paths from user to song
            # the dimension 0 of output is the num of total paths in one batch, this step is to find the paths that from the same row data
            for i in range(len(interaction_batch)):
                masks = s_inter_ids == i

                # shape: (2, )
                pool_score = kprn.weighted_pooling(interactions_path_plausibility[masks])

                # pool_scores: (batch_size, 2)
                if first:
                    # the shape of NLLLoss input should be (N, C) where C is the number of classes
                    # chang it to 2 dimension
                    pool_scores = pool_score.unsqueeze(0)
                    first = False
                else:
                    pool_scores = torch.cat([pool_scores, pool_score.unsqueeze(0)], dim=0)

            # (batch_size, 2)
            predict = torch.log_softmax(pool_scores, dim=1)

            optimizer.zero_grad()
            loss = criterion(predict.to(args.device), targets.to(args.device))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"epoch: {epoch}, loss: {total_loss / len(dataloader)}")

# eval and load model directly
else:
    kprn = torch.load(args.model_path)
