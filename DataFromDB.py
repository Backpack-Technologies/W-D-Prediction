from elasticsearch import Elasticsearch
import elasticsearch.helpers
from collections import OrderedDict
from operator import itemgetter
import json
import random

TOP = 2000000
NUMBER_OF_DATAS = str(TOP)
P2V_DATA_FILE = "data/p2v-data" + NUMBER_OF_DATAS
P2V_QUE_FILE = "data/p2v-question" + NUMBER_OF_DATAS

DUMPED_DATA_FILE = "data/search-query.json"
datas = []


def init_file_name():
    global NUMBER_OF_DATAS
    global P2V_DATA_FILE
    global P2V_QUE_FILE

    NUMBER_OF_DATAS = str(TOP)
    P2V_DATA_FILE = "data/p2v-data" + NUMBER_OF_DATAS
    P2V_QUE_FILE = "data/p2v-question" + NUMBER_OF_DATAS


def get_data(file_load):
    init_file_name()

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
        loop = 0
        for line in infile:
            res.append(json.loads(line))
            loop += 1
            if loop % 1000 == 0:
                print("Got search query:", loop)
        print("All search Found. len:", len(res))

        for doc in res:
            tmpData = []
            for result in doc:
                asin = result['asin']
                tmpData.append(asin)

                if asin not in count:
                    count[asin] = 0
                count[asin] += 1

            datas.append(tmpData)

            if len(datas) % 1000 == 0:
                print("Data collected:", len(datas))
        print("All data collected. len:", len(datas))

        count_sorted = OrderedDict(sorted(count.items(), key=itemgetter(1), reverse=True))
        count_final = dict()

        for ind, asin in enumerate(count_sorted):
            if ind >= TOP:
                break
            count_final[asin] = count_sorted[asin]

        del count_sorted
        del count

        tmpData = datas.copy()
        datas.clear()

        loop = 0
        for data in tmpData:
            tmp = []
            for asin in data:
                if asin in count_final:
                    tmp.append(asin)
            datas.append(tmp)

            loop += 1
            if loop % 1000 == 0:
                print("Final data collected", loop, "in", len(tmpData))
        print("Final data collection over. len:", len(datas))

        del tmpData

        with open(P2V_QUE_FILE, "w") as qp:
            qp.write(": a-b-c")
            qp.write("\n")

        loop = 0
        with open(P2V_DATA_FILE, "w") as fp:
            with open(P2V_QUE_FILE, "a") as qp:
                for data in datas:
                    if len(data) > 3:
                        fp.write(" ".join(data))
                        fp.write('\n')
                        if random.randint(0, 1) == 1:
                            qp.write(" ".join(data[0:4]))
                            qp.write("\n")

                    loop += 1
                    if loop % 1000 == 0:
                        print("File written", loop, "in", len(datas))


if __name__ == '__main__':
    get_data(False)
