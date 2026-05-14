import json
import os
from collections import defaultdict


# locations for metadata from MissionMAST:
missions = ['jwst', 'roman', 'hst']
table_data_dir = os.path.join(os.path.dirname(__file__), 'data')
unique_column_path = os.path.join(table_data_dir, 'unique_columns_per_mission.json')
column_descriptions_path = os.path.join(table_data_dir, 'column_descriptions.json')

# manual definitions for `list_products` queries:
list_products = [
    {
        "name": "product_key",
        "data_type": "string",
        "description": "Unique product key used to identify the product within MAST"
    },
    {
        "name": "access",
        "data_type": "string",
        "description": "Proprietary status of the data product"
    },
    {
        "name": "dataset",
        "data_type": "string",
        "description": "Data set identifier"
    },
    {
        "name": "instrument_name",
        "data_type": "string",
        "description": "Instrument used to produce the data product"
    },
    {
        "name": "filters",
        "data_type": "string",
        "description": "Filter(s) used for this data product"
    },
    {
        "name": "filename",
        "data_type": "string",
        "description": "Unique filename in MAST"
    },
    {
        "name": "uri",
        "data_type": "string",
        "description": "URI for product within MAST"
    },
    {
        "name": "authz_primary_identifier",
        "data_type": "string",
        "description": "Undefined"
    },
    {
        "name": "authz_secondary_identifier",
        "data_type": "string",
        "description": "Undefined"
    },
    {
        "name": "file_suffix",
        "data_type": "string",
        "description": ('The file suffix for observations is usually the last characters '
                        'before the file extension. For other products, the file suffix '
                        'may be the extension.')
    },
    {
        "name": "category",
        "data_type": "string",
        "description": ('Data product category. May be a level (like "1b", "2b", ...)'
                        'or a description of the file (like "REFERENCE" or "ASSOCIATION")')
    },
    {
        "name": "size",
        "data_type": "integer",
        "description": "Size of file in bytes"
    },
    {
        "name": "type",
        "data_type": "string",
        "description": 'Data product type, like "science", "reference", "assocation", "exposure"...'
    },
]


def update_mast_column_lists(update_column_descriptions=True, update_unique_columns=True):
    """
    Return a dictionary of columns in observation query
    results from astroqery.mast.missions.MastMissions
    that are unique to each mission.

    To update the cached columns, run:

        from mast_table.table import validate
        validate.update_mast_column_lists()
    """
    from astroquery.mast import MastMissions

    mast = MastMissions()

    columns_available = dict()
    unique_columns = dict()
    column_descriptions = defaultdict(list)

    for mission in missions:
        mast.mission = mission
        column_list = mast.get_column_list()

        for row in column_list.iterrows():
            column_descriptions[mission].append(
                {k: str(v).strip() for k, v in zip(column_list.colnames, row)}
            )

        column_names = column_list['name'].tolist()
        columns_available[mission] = set(sorted(column_names))

    columns_available['list_products'] = set(sorted(col['name'] for col in list_products))
    column_descriptions['list_products'] = sorted(list_products, key=lambda x: x['name'])

    if update_column_descriptions:
        with open(column_descriptions_path, 'w') as json_file:
            json.dump(
                column_descriptions,
                json_file,
                indent=4,
                sort_keys=True
            )

    compare_to_missions = missions + ['list_products']

    for mission in compare_to_missions:
        other_missions = list(compare_to_missions)
        other_missions.remove(mission)

        other_mission_columns = set()
        for other_mission in other_missions:
            other_mission_columns |= columns_available[other_mission]

        unique_columns[mission] = list(
            columns_available[mission].difference(
                other_mission_columns
            )
        )

    if update_unique_columns:
        with open(unique_column_path, 'w') as json_file:
            json.dump(
                unique_columns,
                json_file,
                indent=4,
                sort_keys=True
            )

    return unique_columns, column_descriptions


def detect_mission_or_products(table):
    """
    Detect which mission was queried by the results from an astroquery.mast.missions
    query by looking for unique columns in the `astropy.table.Table`.
    """
    unique_columns = json.load(open(unique_column_path, 'r'))
    columns = table.colnames

    for mission in missions + ['list_products']:
        if any(name in unique_columns[mission] for name in columns):
            return mission


def get_column_descriptions(mission):
    column_descriptions = json.load(open(column_descriptions_path, 'r'))
    return column_descriptions[mission]
