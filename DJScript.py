import json
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan

es = Elasticsearch(
    hosts=['http://es.backpackbang.com:9200'],
    timeout=30, max_retries=2, retry_on_timeout=True
)

cursor = scan(es,
              query={"_source": ["dimensions"], "query": {"match_all": {}}},
              index="products",
              doc_type="amazon"
              )

with open('data/djdataF.txt', 'w') as f:
    for i, doc in enumerate(cursor):
        res = dict()
        res['asin'] = doc['_id']
        res['dimensions'] = doc['_source'].get('dimensions', '')
        json.dump(res, f)
        f.write("\n")
        if i % 1000 == 0 and i:
            print('done:', i)
