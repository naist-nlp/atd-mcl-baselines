import pickle

from logzero import logger


TARGET_KEYS = ['name', 'prefecture']


class Database(object):
    gid2entry_ids: dict[str, set[str]] # entry group ID -> set of entry IDs
    entry_id2gid: dict[str, str]
    key2value2gids: dict[str, dict[str, set[str]]]

    def __init__(self) -> None:
        self.gid2entry_ids = {}
        self.entry_id2gid = {}
        self.key2value2gids = {key: {} for key in TARGET_KEYS}


    def add_entry_group(self, gid: str, entry_ids: set[str]) -> None:
        self.gid2entry_ids[gid] = entry_ids

        for kv in gid.split('|'):
            array = kv.split('=') # key and value of a OSM tag
            if len(array) == 1 or len(array) > 2:
                logger.warning(f'Skip entry with invalid format: {gid}')
                continue

            key, value = array
            if value == 'None':
                continue

            if key in TARGET_KEYS:
                if not value in self.key2value2gids[key]:
                    self.key2value2gids[key][value] = set()
                self.key2value2gids[key][value].add(gid)
                
                
    def update_entry_id2gid(self) -> None:
        self.entry_id2gid = {}
        # To be implemented


    def get_gids_with_exact_name(self, entity_name: str) -> set[str]:
        if not entity_name:
            return set()

        if entity_name in self.key2value2gids['name']:
            gids = self.key2value2gids['name'][entity_name]
            return gids
        else:
            return set()


    def are_entry_ids_in_gids(
            self,
            entry_ids: list[str],
            gids: list[str],
            top_k: int = 1,
    ) -> bool:

        if gids is None:
            return False

        for gid in gids[:top_k]:
            if not gid in self.gid2entry_ids:
                logger.warning('Predicted entry group ID was not in the database.')
                continue

            entry_ids_in_group = self.gid2entry_ids[gid]
            for entry_id in entry_ids:
                if entry_id in entry_ids_in_group:
                    return True

        return False


"""
[Format of entry database text file (tentative)]
Num_entries	GID_Wikipedia	GID_OSM	OSM_entries
---
[Example rows]
6556	name=東海道新幹線|prefecture=静岡県|railway=rail	way/759909766,way/797675701,...
955	name=JR函館本線|prefecture=北海道|railway=rail	way/798332497,way/24371890

"""
def load_entry_database(
        input_path: str,
) -> Database:
    db = Database()

    with open(input_path) as f:
        logger.info(f'Read: {input_path}')
        for line in f:
            line    = line.rstrip('\n')
            array   = line.split('\t')
            gid_osm = array[1]                   # entry group ID for OSM-based systems
            entry_ids = set(array[2].split(',')) # set of OSM entry IDs
            db.add_entry_group(gid_osm, entry_ids)

    logger.info(f'Finish: database construction')
    return db


def serialize_entry_database(db: Database, output_path: str) -> None:
    with open(output_path, 'wb') as fw:
        pickle.dump(db, fw)
        logger.info(f'Write: {output_path}')
    return db


def deserialize_entry_database(input_path: str) -> Database:
    with open(input_path, 'rb') as f:
        db = pickle.load(f)
        logger.info(f'Finish loading: {input_path}')
    return db
