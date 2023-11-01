import argparse
import os
import pickle
import random
import sys
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('./consts.py'))))
from KPRN import consts

parser = argparse.ArgumentParser(description='the parameter of file')

parser.add_argument('--song_data', type=str, default='../song_dataset/songs.csv', help='the data to build KG')
parser.add_argument('--interaction_data', type=str, default='../song_dataset/train.csv',
                    help='the data of what items user interact with')
parser.add_argument('--subnetwork_type', type=str, default='dense', choices=['dense', 'standard', 'sparse', 'full'],
                    help='the type of subnetwork')

args = parser.parse_args()


def make_person_list(row):
    person_set = set()

    col_names = ['artist_name', 'lyricist', 'composer']

    for col_name in col_names:
        # if more than 1 person, use '|' to spilt
        if not isinstance(row[col_name], float):
            for person in row[col_name].split('|'):
                person_set.add(person.strip())

    return list(person_set)


# use dict(adjacent list) to represent map
def prepare_song_data():
    songs = pd.read_csv(args.song_data, encoding='utf-8')
    interactions = pd.read_csv(args.interaction_data, encoding='utf-8')

    # selct 4 columns from songs df
    # there is null value and -1 in data(float type), need to cleaned

    # includes 3 entity types: user, song, artist(artist name, lyricist, composer)

    # person reference to artist, not user
    # song_person.dict (key: song_id, value: person_list)
    songs4cols = songs[['song_id', 'artist_name', 'composer', 'lyricist']]

    person_list = songs4cols.apply(lambda x: make_person_list(x), axis=1)
    song_person = pd.DataFrame({
        'song_id': songs4cols['song_id'],
        'person_list': person_list
    })

    song_person_dict = song_person.set_index('song_id')['person_list'].to_dict()

    with open(consts.DATA_DIR + '/' + consts.SONG_PERSON_DICT, 'wb') as file:
        pickle.dump(song_person_dict, file, protocol=pickle.HIGHEST_PROTOCOL)
    # end

    # person_song.dict (key: person, value: song_id list related to the person)
    person_song_dict = {}
    for idx, row in song_person.iterrows():
        for person in row['person_list']:
            if person not in person_song_dict.keys():
                person_song_dict[person] = []
            person_song_dict[person].append(row['song_id'])

    with open(consts.DATA_DIR + '/' + consts.PERSON_SONG_DICT, 'wb') as file:
        pickle.dump(person_song_dict, file, protocol=pickle.HIGHEST_PROTOCOL)
    # end

    interactions2cols = interactions[['msno', 'song_id']]

    # user_song.dict (key: user, value: song list the user interact with)
    user_song_dict = defaultdict(list)
    for idx, row in interactions2cols.iterrows():
        user_song_dict[row['msno']].append(row['song_id'])

    with open(consts.DATA_DIR + '/' + consts.USER_SONG_DICT, 'wb') as file:
        pickle.dump(user_song_dict, file, protocol=pickle.HIGHEST_PROTOCOL)
    # end

    # song_user.dict (key: song, value: user list who interact with this song)
    song_user_dict = defaultdict(list)
    for idx, row in interactions2cols.iterrows():
        song_user_dict[row['song_id']].append(row['msno'])

    with open(consts.DATA_DIR + '/' + consts.SONG_USER_DICT, 'wb') as file:
        pickle.dump(song_user_dict, file, protocol=pickle.HIGHEST_PROTOCOL)
    # end


# each type of nodes remain (factor) of total number and their edge
def find_subnetwork(factor=0.1):
    with open(consts.DATA_DIR + '/' + consts.PERSON_SONG_DICT, 'rb') as file:
        person_song = pickle.load(file)

    with open(consts.DATA_DIR + '/' + consts.SONG_PERSON_DICT, 'rb') as file:
        song_person = pickle.load(file)

    with open(consts.DATA_DIR + '/' + consts.USER_SONG_DICT, 'rb') as file:
        user_song = pickle.load(file)

    with open(consts.DATA_DIR + '/' + consts.SONG_USER_DICT, 'rb') as file:
        song_user = pickle.load(file)

    person_song = defaultdict(list, person_song)
    song_person = defaultdict(list, song_person)
    user_song = defaultdict(list, user_song)
    song_user = defaultdict(list, song_user)

    # get node
    # song degree(key: song, value: person list and user list  related to the song)
    song_degree_dict = {}
    for song, user_list in song_user.items():
        song_degree_dict[song] = user_list
    for song, person_list in song_person.items():
        if song not in song_degree_dict.keys():
            song_degree_dict[song] = person_list
        else:
            # must x = x + y
            # it can not be += or extend, because these 2 operations will change the origin data
            song_degree_dict[song] = song_degree_dict[song] + person_list

    song_degree = [(k, len(v)) for (k, v) in song_degree_dict.items()]
    # sort by decreasing
    song_degree.sort(key=lambda x: -x[1])

    # person degree(key: person, value: item list relate to the person)
    person_degree = [(k, len(v)) for (k, v) in person_song.items()]
    person_degree.sort(key=lambda x: -x[1])

    # user degree(key: user, value: item list relate to the user)
    user_degree = [(k, len(v)) for (k, v) in user_song.items()]
    user_degree.sort(key=lambda x: -x[1])

    # each type of node choose top 10% by degree
    if args.subnetwork_type == 'dense':
        song_nodes_tuples = song_degree[:int(len(song_degree) * factor)]
        song_nodes = [song_nodes_tuple[0] for song_nodes_tuple in song_nodes_tuples]

        user_nodes_tuples = user_degree[:int(len(user_degree) * factor)]
        user_nodes = [user_nodes_tuple[0] for user_nodes_tuple in user_nodes_tuples]

        person_nodes_tuples = person_degree[:int(len(person_degree) * factor)]
        person_nodes = [person_nodes_tuple[0] for person_nodes_tuple in person_nodes_tuples]
    # each type of node random choose 10%
    elif args.subnetwork_type == 'standard':
        song_nodes_tuples = random.sample(song_degree, int(len(song_degree) * factor))
        song_nodes = [song_nodes_tuple[0] for song_nodes_tuple in song_nodes_tuples]

        user_nodes_tuples = random.sample(user_degree, int(len(user_degree) * factor))
        user_nodes = [user_nodes_tuple[0] for user_nodes_tuple in user_nodes_tuples]

        person_nodes_tuples = random.sample(person_degree, int(len(person_degree) * factor))
        person_nodes = [person_nodes_tuple[0] for person_nodes_tuple in person_nodes_tuples]

    # each type of node choose last 10% by degree
    elif args.subnetwork_type == 'sparse':
        song_nodes_tuples = song_degree[-int(len(song_degree) * factor):]
        song_nodes = [song_nodes_tuple[0] for song_nodes_tuple in song_nodes_tuples]

        user_nodes_tuples = user_degree[-int(len(user_degree) * factor):]
        user_nodes = [user_nodes_tuple[0] for user_nodes_tuple in user_nodes_tuples]

        person_nodes_tuples = person_degree[-int(len(person_degree) * factor):]
        person_nodes = [person_nodes_tuple[0] for person_nodes_tuple in person_nodes_tuples]
    # too big and not implement
    elif args.subnetwork_type == 'full':
        return

    nodes = song_nodes + user_nodes + person_nodes
    print(
        f'The {args.subnetwork_type} subnetwork has {len(nodes)} nodes: {len(song_nodes)} songs, {len(user_nodes)} users, {len(person_nodes)} persons.')
    # end

    # get edges(the edges connect with the sampled nodes, the origin edges are in the ".dict" file)
    # (h, r, t) indicates a relationship from head entity "h" to tail entity "t", which means it is a direacted graph

    # song->user
    edges_type1 = []
    # song->person
    edges_type2 = []
    # user->song
    edges_type3 = []
    # person->song
    edges_type4 = []
    nodes_set = set(nodes)

    # u type can be: user, song and person
    # typeX_set can be empty
    for u in tqdm(nodes_set):
        type1_set = set(song_user[u]).intersection(nodes_set)
        for v in type1_set:
            edges_type1.append((u, v))

        type2_set = set(song_person[u]).intersection(nodes_set)
        for v in type2_set:
            edges_type2.append((u, v))

        type3_set = set(user_song[u]).intersection(nodes_set)
        for v in type3_set:
            edges_type3.append((u, v))

        type4_set = set(person_song[u]).intersection(nodes_set)
        for v in type4_set:
            edges_type4.append((u, v))

    edges = edges_type1 + edges_type2 + edges_type3 + edges_type4
    print(f'The {args.subnetwork_type} subnetwork has {len(edges)} edges')

    # get dense adjacent map
    # dense_person_song.dict
    person_song_dict = defaultdict(list)
    for edge in edges_type4:
        person = edge[0]
        song = edge[1]
        person_song_dict[person].append(song)

    filename = consts.DATA_DIR + '/' + args.subnetwork_type + '_' + consts.PERSON_SONG_DICT
    with open(filename, 'wb') as file:
        pickle.dump(person_song_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

    # dense_song_person.dict
    song_person_dict = defaultdict(list)
    for edge in edges_type2:
        song = edge[0]
        person = edge[1]
        song_person_dict[song].append(person)

    filename = consts.DATA_DIR + '/' + args.subnetwork_type + '_' + consts.SONG_PERSON_DICT
    with open(filename, 'wb') as file:
        pickle.dump(song_person_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

    # dense_user_song
    user_song_dict = defaultdict(list)
    for edge in edges_type3:
        user = edge[0]
        song = edge[1]
        user_song_dict[user].append(song)

    filename = consts.DATA_DIR + '/' + args.subnetwork_type + '_' + consts.USER_SONG_DICT
    with open(filename, 'wb') as file:
        pickle.dump(user_song_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

    # dense_song_user
    song_user_dict = defaultdict(list)
    for edge in edges_type1:
        song = edge[0]
        user = edge[1]
        song_user_dict[song].append(user)

    filename = consts.DATA_DIR + '/' + args.subnetwork_type + '_' + consts.SONG_USER_DICT
    with open(filename, 'wb') as file:
        pickle.dump(song_user_dict, file, protocol=pickle.HIGHEST_PROTOCOL)


def convert_to_ix(entity_to_ix, origin_dict, start_type, end_type):
    new_dict = {}

    for k, v in origin_dict.items():
        k_ix = entity_to_ix[(k, start_type)]
        v_ixs = []
        for v_entity in v:
            v_ixs.append(entity_to_ix[(v_entity, end_type)])
        new_dict[k_ix] = v_ixs

    return new_dict


def ix_mapping():
    # still unknow what it use to do, maybe in extra the path?
    pad_token = consts.PAD_TOKEN
    type_to_ix = {'person': consts.PERSON_TYPE, 'user': consts.USER_TYPE, 'song': consts.SONG_TYPE,
                  pad_token: consts.PAD_TYPE}
    relation_to_ix = {'song_person': consts.SONG_PERSON_REL, 'person_song': consts.PERSON_SONG_REL,
                      'user_song': consts.USER_SONG_REL, 'song_user': consts.SONG_USER_REL,
                      '#UNK_RELATION': consts.UNK_REL,
                      '#END_RELATION': consts.END_REL, pad_token: consts.PAD_REL}

    prefix = consts.DATA_DIR + '/' + args.subnetwork_type + '_'
    with open(prefix + consts.SONG_USER_DICT, 'rb') as file:
        song_user = pickle.load(file)
    with open(prefix + consts.SONG_PERSON_DICT, 'rb') as file:
        song_person = pickle.load(file)
    with open(prefix + consts.USER_SONG_DICT, 'rb') as file:
        user_song = pickle.load(file)
    with open(prefix + consts.PERSON_SONG_DICT, 'rb') as file:
        person_song = pickle.load(file)

    # get ids
    songs = set(song_user.keys()).union(song_person.keys())
    users = set(user_song.keys())
    persons = set(person_song.keys())

    # print(song_user)

    # mapping id to idx
    entity_to_ix = {(song, consts.SONG_TYPE): ix for ix, song in enumerate(songs)}
    entity_to_ix.update(
        {(user, consts.USER_TYPE): ix + len(songs) for ix, user in enumerate(users)}
    )
    entity_to_ix.update(
        {(person, consts.PERSON_TYPE): ix + len(users) + len(persons) for ix, person in enumerate(persons)}
    )
    entity_to_ix[consts.PAD_TOKEN] = len(entity_to_ix)

    # mapping idx to id
    ix_to_relation = {v: k for k, v in relation_to_ix.items()}
    ix_to_type = {v: k for k, v in type_to_ix.items()}
    ix_to_entity = {v: k for k, v in entity_to_ix.items()}

    prefix = consts.IX_MAPPING_DIR + '/' + args.subnetwork_type + '_'
    with open(prefix + consts.TYPE_TO_IX, 'wb') as file:
        pickle.dump(type_to_ix, file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(prefix + consts.RELATION_TO_IX, 'wb') as file:
        pickle.dump(relation_to_ix, file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(prefix + consts.ENTITY_TO_IX, 'wb') as file:
        pickle.dump(entity_to_ix, file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(prefix + consts.IX_TO_TYPE, 'wb') as file:
        pickle.dump(ix_to_type, file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(prefix + consts.IX_TO_RELATION, 'wb') as file:
        pickle.dump(ix_to_relation, file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(prefix + consts.IX_TO_ENTITY, 'wb') as file:
        pickle.dump(ix_to_entity, file, protocol=pickle.HIGHEST_PROTOCOL)

    song_user_ix = convert_to_ix(entity_to_ix, song_user, consts.SONG_TYPE, consts.USER_TYPE)
    user_song_ix = convert_to_ix(entity_to_ix, user_song, consts.USER_TYPE, consts.SONG_TYPE)
    song_person_ix = convert_to_ix(entity_to_ix, song_person, consts.SONG_TYPE, consts.PERSON_TYPE)
    person_song_ix = convert_to_ix(entity_to_ix, person_song, consts.PERSON_TYPE, consts.SONG_TYPE)

    prefix = consts.IX_DATA_DIR + '/' + args.subnetwork_type + '_ix_'
    with open(prefix + consts.SONG_USER_DICT, 'wb') as file:
        pickle.dump(song_user_ix, file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(prefix + consts.USER_SONG_DICT, 'wb') as file:
        pickle.dump(user_song_ix, file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(prefix + consts.SONG_PERSON_DICT, 'wb') as file:
        pickle.dump(song_person_ix, file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(prefix + consts.PERSON_SONG_DICT, 'wb') as file:
        pickle.dump(person_song_ix, file, protocol=pickle.HIGHEST_PROTOCOL)


def makedir(dir):
    try:
        os.mkdir(dir)
    except FileExistsError:
        print("The directory has already existed!")


makedir(consts.DATA_DIR)

print("Building KG...")
# the number of file in song_data dir < 4, it should be reproduced network
if len(os.listdir(consts.DATA_DIR)) < 4:
    prepare_song_data()

print("Building Subnetwork...")
# the number of file in song_data dir < 4, it should be reproduced subnetwork
if len(os.listdir(consts.DATA_DIR)) < 8:
    find_subnetwork()

makedir(consts.IX_MAPPING_DIR)
makedir(consts.IX_DATA_DIR)
print("Mapping id from zero...")
ix_mapping()