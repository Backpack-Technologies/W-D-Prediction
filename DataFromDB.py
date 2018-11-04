from elasticsearch import Elasticsearch
import elasticsearch.helpers
from collections import OrderedDict
from operator import itemgetter
import json
import random

TOP = 400000

DUMPED_DATA_FILE = "data/search-query.json"
NUMBER_OF_DATAS = str(TOP)
P2V_DATA_FILE = "data/p2v-data" + NUMBER_OF_DATAS
P2V_QUE_FILE = "data/p2v-question" + NUMBER_OF_DATAS

datas = []


def get_data(file_load):
    count = dict()
    if file_load:
        with open(DUMPED_DATA_FILE, 'a') as outfile:

            try:
                es = Elasticsearch(['http://es.backpackbang.com:9200/'])
                results = elasticsearch.helpers.scan(es,
                                                     index="search-results-v1",
                                                     doc_type="amazon",
                                                     request_timeout=30000,
                                                     scroll=u'1m',
                                                     size="10000",
                                                     query={"query": {"match_all": {}}})

                cnt = 0
                outfile.write('[')
                mf = False
                for item in results:
                    cnt += 1
                    if mf:
                        outfile.write(',\n')
                    mf = True
                    flag = False
                    outfile.write('[\n')
                    for data in item['_source']['results']:
                        if flag:
                            outfile.write(',\n')
                        flag = True
                        json.dump(data, outfile)
                    outfile.write('\n]')
                    if cnt % 10000 == 0:
                        print(cnt)

                outfile.write(']')
            except Exception as e:
                print(e)

    with open(DUMPED_DATA_FILE, 'r') as infile:
        res = []
        for line in infile:
            res.append(json.loads(line))

        print(len(res))

        for doc in res:
            tmpData = []
            for result in doc:
                asin = result['asin']
                tmpData.append(asin)

                if asin not in count:
                    count[asin] = 0
                count[asin] += 1

            datas.append(tmpData)

        count_sorted = OrderedDict(sorted(count.items(), key = itemgetter(1), reverse = True))
        count_final = dict()

        for ind, asin in enumerate(count_sorted):
            if ind >= TOP:
                break
            count_final[asin] = count_sorted[asin]

        del count_sorted
        del count

        tmpData = datas.copy()
        datas.clear()

        for data in tmpData:
            tmp = []
            for asin in data:
                if asin in count_final:
                    tmp.append(asin)
            datas.append(tmp)
        del tmpData

        # print(datas)
        # print(count_final)
        print(len(count_final))

        with open(P2V_DATA_FILE, "w") as fp:
            with open(P2V_QUE_FILE, "w") as qp:
                for data in datas:
                    if len(data) > 3:
                        fp.write(" ".join(data))
                        fp.write('\n')
                        if random.randint(0, 1) == 1:
                            qp.write(" ".join(data[0:4]))
                            qp.write("\n")


if __name__ == '__main__':
    get_data(False)
