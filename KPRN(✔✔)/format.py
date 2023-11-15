import consts as consts


def format_path(pos_paths, entity_to_ix, type_to_ix, relation_to_ix):
    # why it should
    new_paths = []
    for path in pos_paths:
        path_len =len(path)
        pad_path(path, entity_to_ix, type_to_ix, relation_to_ix, consts.MAX_PATH_LEN, consts.PAD_TOKEN)
        new_paths.append((path, path_len))

    return new_paths


def pad_path(path, entity_to_ix, type_to_ix, relation_to_ix, max_len, padding_token):
    entity_padding = entity_to_ix[padding_token]
    type_padding = type_to_ix[padding_token]
    relation_padding = relation_to_ix[padding_token]

    while len(path) < max_len:
        path.append([entity_padding, type_padding, relation_padding])

    return path