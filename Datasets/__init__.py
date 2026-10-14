from Datasets.UORED import UORED
from Datasets.BJTU_RAO import BJTU_RAO

# Compatibility: expose dataset under both names
# BJTU-RAO accessible as attribute `BJTU_RAO` (train_utils will accept hyphenated name)
BJTU_RAO = BJTU_RAO