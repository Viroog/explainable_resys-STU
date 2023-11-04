DATA_DIR = 'song_data'
IX_MAPPING_DIR = 'song_ix_mapping'
IX_DATA_DIR = 'song_ix_data'
PATH_DATA_DIR = 'path_data'

SONG_PERSON_DICT = 'song_person.dict'
PERSON_SONG_DICT = 'person_song.dict'
USER_SONG_DICT = 'user_song.dict'
SONG_USER_DICT = 'song_user.dict'

TYPE_TO_IX = 'type_to_ix.dict'
RELATION_TO_IX = 'relation_to_ix.dict'
ENTITY_TO_IX = 'entity_to_ix.dict'
IX_TO_TYPE = 'ix_to_type.dict'
IX_TO_RELATION = 'ix_to_relation.dict'
IX_TO_ENTITY = 'ix_to_entity.dict'

PAD_TOKEN = '#PAD_TOKEN'
SONG_TYPE = 0
USER_TYPE = 1
PERSON_TYPE = 2
PAD_TYPE = 3

SONG_PERSON_REL = 0
PERSON_SONG_REL = 1
USER_SONG_REL = 2
SONG_USER_REL = 3
UNK_REL = 4
END_REL = 5
PAD_REL = 6

MAX_PATH_LEN = 6
# when training, for each positive interaction has 4 negative interactions
NEG_SAMPLES_TRAIN = 4
# when training, for each positive interaction has 100 negative interactions
# just same as the SASRec
NEG_SAMPLES_EVAL = 100

# length 3 means 4 nodes and length 5 means 5 nodes

# branching factor in each layer of dfs
# it means from a node in the graph, randomly sample k edges from that node to use as edges
LEN_3_BRANCH = 50
LEN_5_BRANCH_TRAIN = 6
LEN_5_BRANCH_EVAL = 10
