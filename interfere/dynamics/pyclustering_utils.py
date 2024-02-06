from pyclustering.nnet import conn_type

# Maps string arguments to pyclustering arguments
CONN_TYPE_MAP = {
    "all_to_all": conn_type.ALL_TO_ALL,
    "grid_four": conn_type.GRID_FOUR,
    "grid_eight": conn_type.GRID_EIGHT,
    "list_bdir": conn_type.LIST_BIDIR,
    "dynamic": conn_type.DYNAMIC
}