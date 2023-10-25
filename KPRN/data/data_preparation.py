import argparse
import os
import pickle
import random
from collections import defaultdict
from tqdm import tqdm

import pandas as pd

parser = argparse.ArgumentParser(description='the parameter of file')

parser.add_argument('--data_dir', type=str, default='song_data', help='the directory of prepared data')
parser.add_argument('--song_person_dict', type=str, default='song_person.dict', help='song_person file name')
parser.add_argument('--person_song_dict', type=str, default='person_song.dict', help='person_song file name')
parser.add_argument('--user_song_dict', type=str, default='user_song.dict', help='user_song file name')
parser.add_argument('--song_user_dict', type=str, default='song_user.dict', help='song_user file name')

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

    with open(args.data_dir + '/' + args.song_person_dict, 'wb') as file:
        pickle.dump(song_person_dict, file, protocol=pickle.HIGHEST_PROTOCOL)
    # end

    # person_song.dict (key: person, value: song_id list related to the person)
    person_song_dict = {}
    for idx, row in song_person.iterrows():
        for person in row['person_list']:
            if person not in person_song_dict.keys():
                person_song_dict[person] = []
            person_song_dict[person].append(row['song_id'])

    with open(args.data_dir + '/' + args.person_song_dict, 'wb') as file:
        pickle.dump(person_song_dict, file, protocol=pickle.HIGHEST_PROTOCOL)
    # end

    interactions2cols = interactions[['msno', 'song_id']]

    # user_song.dict (key: user, value: song list the user interact with)
    user_song_dict = defaultdict(list)
    for idx, row in interactions2cols.iterrows():
        user_song_dict[row['msno']].append(row['song_id'])

    with open(args.data_dir + '/' + args.user_song_dict, 'wb') as file:
        pickle.dump(user_song_dict, file, protocol=pickle.HIGHEST_PROTOCOL)
    # end

    # song_user.dict (key: song, value: user list who interact with this song)
    song_user_dict = defaultdict(list)
    for idx, row in interactions2cols.iterrows():
        song_user_dict[row['song_id']].append(row['msno'])

    with open(args.data_dir + '/' + args.song_user_dict, 'wb') as file:
        pickle.dump(song_user_dict, file, protocol=pickle.HIGHEST_PROTOCOL)
    # end


# each type of nodes remain (factor) of total number and their edge
def find_subnetwork(factor=0.1):
    with open(args.data_dir + '/' + args.person_song_dict, 'rb') as file:
        person_song = pickle.load(file)

    with open(args.data_dir + '/' + args.song_person_dict, 'rb') as file:
        song_person = pickle.load(file)

    with open(args.data_dir + '/' + args.user_song_dict, 'rb') as file:
        user_song = pickle.load(file)

    with open(args.data_dir + '/' + args.song_user_dict, 'rb') as file:
        song_user = pickle.load(file)

    # get node
    # song degree(key: song, value: person list and user list  related to the song)
    song_degree_dict = {}
    for song, user_list in song_user.items():
        song_degree_dict[song] = user_list
    for song, person_list in song_person.items():
        if song not in song_degree_dict.keys():
            song_degree_dict[song] = person_list
        else:
            song_degree_dict[song].extend(person_list)

    song_degree = [(k, len(v)) for (k, v) in song_degree_dict.items()]
    # sort by decreasing
    song_degree.sort(key=lambda x: x[1], reverse=True)

    # person degree(key: person, value: item list relate to the person)
    person_degree = [(k, len(v)) for (k, v) in person_song.items()]
    person_degree.sort(key=lambda x: x[1], reverse=True)

    # user degree(key: user, value: item list relate to the user)
    user_degree = [(k, len(v)) for (k, v) in user_song.items()]
    user_degree.sort(key=lambda x: x[1], reverse=True)

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
    print(f'The {args.subnetwork_type} subnetwork has {len(nodes)} nodes: {len(song_nodes)} songs, {len(user_nodes)} users, {len(person_nodes)} persons.')
    # end

    # get edges


def makedir(dir):
    try:
        os.mkdir(dir)
    except FileExistsError:
        print("The directory has already existed!")


makedir(args.data_dir)
print("Building KG...")
# the number of file in song_data dir < 4, it should be reproduced
if len(os.listdir(args.data_dir)) < 4:
    prepare_song_data()
print("Building Subnetwork...")
find_subnetwork()
