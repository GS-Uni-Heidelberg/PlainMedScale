"""
Helper functions for search
"""
import json
import unicodedata
from typing import Dict, List
from sqlitevec import SQLiteVec


def load_dictionary_tables():
    icd10_table = {}
    for lang in ['en', 'de']:
        icd10_table[lang] = SQLiteVec(
            table=f"icd10_{lang}",
            db_file="icd10.db",
            lang=lang,
        )

    mesh_table = {}
    for lang in ['en', 'de']:
        mesh_table[lang] = SQLiteVec(
            table=f"mesh_{lang}",
            db_file="mesh.db",
            lang=lang,
        )

    do_table = {}
    for lang in ['en']:
        do_table[lang] = SQLiteVec(
            table="disease_ontology",
            db_file="disease_ontology.db",
            lang=lang,
        )

    pschyrembel_table = {}
    for lang in ['en', 'de']:
        pschyrembel_table[lang] = SQLiteVec(
            table=f"pschyrembel_{lang}",
            db_file="pschyrembel.db",
            lang=lang,
        )

    return icd10_table, mesh_table, do_table, pschyrembel_table


def load_data_tables():
    msd_table = {}
    for lang in ['en', 'de']:
        msd_table[lang] = SQLiteVec(
            table=f"msd_doc_{lang}",
            db_file="msd.db",
            lang=lang,
        )

    gesund_table = {}
    for lang in ['en', 'de']:
        gesund_table[lang] = SQLiteVec(
            table=f"gesund_bund_doc_{lang}",
            db_file="gesund_bund.db",
            lang=lang,
        )

    nhs_table = {}
    for lang in ['en']:
        nhs_table[lang] = SQLiteVec(
            table="nhs_doc",
            db_file="nhs.db",
            lang=lang,
        )

    apoum_table = {}
    for lang in ['de']:
        apoum_table[lang] = SQLiteVec(
            table="apotheken_umschau_doc",
            db_file="apotheken_umschau.db",
            lang=lang,
        )
    return msd_table, gesund_table, nhs_table, apoum_table


def _clean(query: str) -> str:
    # remove double-quotes and parenthese
    query = query.replace('"', '')
    query = query.replace('(', '').replace(')', '')
    query = unicodedata.normalize('NFKD', query)
    return query.strip()


def _format_query(query: str) -> str:
    query = _clean(query)
    if ',' in query: # set of words in no particular order
        return f"({' OR '.join(['"'+q.strip()+'"' for q in query.split(',')])})"
    else: # phrase in specific order
        return f'"{query}"'


def lookup_title(table: SQLiteVec, query: str, keys: List[str], limit: int = 5) -> Dict[str, List]:
    """Case-insensitive lookup"""
    found = table.fulltext_search_with_highlight(
        query = f'title_lower: {_format_query(query.lower())}', limit = limit,
    )
    hits = {}
    if len(found) > 0:
        for key in keys:
            if key in found[0]:
                hits[key] = list(set([f[key] for f in found if isinstance(f[key], str) and len(f[key]) > 0]))
    return hits


def search_by_similarity(
    data_table: SQLiteVec, dict_table: SQLiteVec, emb: Dict, instance_id: str, key: str, limit: int=5
) -> Dict[str, float]:
    hits = []
    for lang, table in dict_table.items():
    
        # lookup the dict_table by similarity
        if lang in data_table:
            dict_found = table.similarity_search_with_score_by_vector(
                embedding=emb[lang], k=limit)
        
            # validate, if the similar instances actually contains the original query
            hit = None
            for f in dict_found:
                # Case-insensitive lookup
                dict_found_rev = data_table[lang].fulltext_search_with_highlight(
                    query = f'title_lower: {_format_query(f["title"].lower())}', limit = limit,
                )
                instance_ids_rev = [f['instance_id'] for f in dict_found_rev]
                if instance_id in instance_ids_rev:
                    hit = f
                    break
    
            if hit:
                hits.append(hit)
    return _get_distance(hits, key, distance={})


def check_by_similarity(dict_table: SQLiteVec, emb, ids, key, limit = 25):
    distance = {}
    for lang, table in dict_table.items():
        if lang in emb:
            kwargs = {'embedding': emb[lang], 'k': limit, key: ids}
            found = table.similarity_search_with_score_by_vector(**kwargs)
            distance = _get_distance(found, key, distance=distance)
    return distance


def _get_distance(found: List[Dict], key: str, distance: Dict = {}) -> Dict[str,float]:
    # get instances ordered by distance
    for f in found:
        if f[key] in distance:
            if f['distance'] < distance[f[key]]:
                distance[f[key]] = f['distance']
        else:
            distance[f[key]] = f['distance']
    return distance


def _add_to_dict(hits: Dict, external_ids: Dict, instance_id: str) -> Dict:
    key_map = {'doid': 'DOID', 'icd10': 'ICD-10CM', 'mesh': 'MeSH', 'descriptor_ui': 'MeSH', 'pschyrembel_id': 'Pschyrembel'}
    for key, values in hits.items():
        for value in values:
            if key in ['mesh', 'descriptor_ui'] and ';' in value:
                for desc in value.split(';'):
                    assert desc.startswith('D')
                    external_ids[instance_id][key_map[key]].add(desc.strip())
            elif key in ['instance_id']:
                if '-' in value:
                    external_ids[instance_id]['MSD'].add(value)
                else:
                    external_ids[instance_id]['GesundBund'].add(value)
            else:
                external_ids[instance_id][key_map[key]].add(value)
    return external_ids
