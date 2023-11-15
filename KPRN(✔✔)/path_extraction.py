import copy
import random
from collections import defaultdict

import consts as consts


class PathState:
    def __init__(self, path, length, entities):
        # array of (entity, type of entity, relation to next)
        self.path = path
        self.length = length
        # set to keep track of the entities in the path to avoid cycles
        self.entities = entities


def get_random_index(nums, max_length):
    index_list = list(range(max_length))
    random.shuffle(index_list)

    return index_list[:nums]


def find_paths_user_to_songs(start_user, song_person, person_song, song_user, user_song, max_length, sample_nums):
    song_to_paths = defaultdict(list)
    # use List to imitate Stack
    stack = []
    # in initial, there is no path so the relation to the next is <End>
    # in end, the relation to next of last node will be <End>
    start = PathState([[start_user, consts.USER_TYPE, consts.END_REL]], 0, {start_user})
    stack.append(start)

    # use stack to implement dfs
    while len(stack) > 0:
        front = stack.pop()
        # the last node in the path
        entity, entity_type = front.path[-1][0], front.path[-1][1]

        # the last node of path is song and the path length satisfies the required length, add it to the
        if entity_type == consts.SONG_TYPE and front.length == max_length:
            song_to_paths[entity].append(front.path)

        if front.length == max_length:
            continue

        # there is a default rule:
        # the next entity of user must be music, which means user had interacted with this song
        # the next entity of song can be person or user, which means it made(or other relationships) by person or interacted by user
        # the next entity of person must be music, which means person made(or other relationships) this music

        # the type of dealing node is user
        # if type == consts.USER_TYPE and entity in user_song.keys() (user can not in user_song? I think it must in user_song, if can't, modifing it)
        if entity_type == consts.USER_TYPE:
            song_list = user_song[entity]
            index_list = get_random_index(sample_nums, len(song_list))
            for index in index_list:
                song = song_list[index]
                # avoid cycle
                if song not in front.entities:
                    new_path = copy.deepcopy(front.path)
                    # change relation to next to user_song relationship
                    new_path[-1][2] = consts.USER_SONG_REL
                    new_path.append([song, consts.SONG_TYPE, consts.END_REL])
                    new_state = PathState(new_path, front.length + 1, {song} | front.entities)
                    stack.append(new_state)

        # the type of dealing node is song(song can from user and person, can in user/person but not in person/user)
        elif entity_type == consts.SONG_TYPE:
            if entity in song_user:
                user_list = song_user[entity]
                index_list = get_random_index(sample_nums, len(user_list))
                for index in index_list:
                    user = user_list[index]
                    if user not in front.entities:
                        new_path = copy.deepcopy(front.path)
                        new_path[-1][2] = consts.SONG_USER_REL
                        new_path.append([user, consts.USER_TYPE, consts.END_REL])
                        new_state = PathState(new_path, front.length + 1, {user} | front.entities)
                        stack.append(new_state)

            if entity in song_person:
                person_list = song_person[entity]
                index_list = get_random_index(sample_nums, len(person_list))
                for index in index_list:
                    person = person_list[index]
                    if person not in front.entities:
                        new_path = copy.deepcopy(front.path)
                        new_path[-1][2] = consts.SONG_PERSON_REL
                        new_path.append([person, consts.PERSON_TYPE, consts.END_REL])
                        new_state = PathState(new_path, front.length + 1, {person} | front.entities)
                        stack.append(new_state)

        # the type of dealing node is song
        # the situation is same as when entity_type=consts.USER_TYPE
        # elif entity_type == consts.PERSON_TYPE and entity in person_song:
        elif entity_type == consts.PERSON_TYPE:
            song_list = person_song[entity]
            index_list = get_random_index(sample_nums, len(song_list))
            for index in index_list:
                song = song_list[index]
                if song not in front.entities:
                    new_path = copy.deepcopy(front.path)
                    new_path[-1][2] = consts.PERSON_SONG_REL
                    new_path.append([song, consts.SONG_TYPE, consts.END_REL])
                    new_state = PathState(new_path, front.length+1, {song}| front.entities)
                    stack.append(new_state)

    return song_to_paths
