from elasticsearch import Elasticsearch
import elasticsearch.helpers

es = Elasticsearch(['http://es.backpackbang.com:9200/'])


def get(cat, asin):
    try:
        res = es.search(index="products", doc_type="amazon", body={"query": {"match": {"_id": asin}}})

        res = res['hits']['hits']
        if len(res) == 0 or cat not in res[0]['_source']:
            return ""
        else:
            return res[0]['_source'][cat]

    except Exception as e:
        print(e)
        return ""


if __name__ == "__main__":
    # print(get("title", "B071GRBWG6"))
    # print(get("title", "B00N2AY7WM"))
    # print(get("title", "B07CY1X7YL"))
    # print(get("title", "B06XFGTGC4"))
    # print(get("title", "B07FQMBWQG"))
    # print(get("title", "B078P2XN9M"))
    # print(get("title", "B01LXTK4T6"))
    # print(get("title", "B01D402Z28"))
    # print(get("title", "B0711T4RDF"))
    # print(get("title", "B0785MMPJ1"))
    print(get("dimensions", "B06XCM9LJ4"))
