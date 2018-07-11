//
// Created by LooperXX on 2018/6/22.
//

#include <bits/stdc++.h>
#include <Windows.h>
#define psc(a, b) pair<string, classV>(a,b)

using std::string;
using std::ifstream;
using std::cout;
using std::endl;
using std::vector;
using std::map;
using std::pair;

class read {
public:
    enum classV {
        Irissetosa = 0,
        Irisversicolor = 1,
        Irisvirginica = 2
    };
    map<string, classV> classVmap;
    string name;
public:
    void setName(string filename) {
        name = filename;
    }

    void initClassVMap() {
        classVmap.insert(psc("Irissetosa", Irissetosa));
        classVmap.insert(psc("Irisversicolor", Irisversicolor));
        classVmap.insert(psc("Irisvirginica", Irisvirginica));
    }

    void split(std::string &s, std::string &delim, vector<string> *ret) {
        size_t last = 0;
        size_t index = s.find_first_of(delim, last);
        while (index != std::string::npos) {
            ret->push_back(s.substr(last, index - last));
            last = index + 1;
            index = s.find_first_of(delim, last);
        }
        if (index - last > 0) {
            ret->push_back(s.substr(last, index - last));
        }
    }

    void readData(float *dataList, int *classList, const int featureCount, const int classIndex,
                  const int sequenceName_index, string *seqName) {
        ifstream infile;
        infile.open(name.data());   //将文件流对象与文件连接起来
        assert(infile.is_open());   //若失败,则输出错误消息,并终止程序运行
        string s;
        string delim = ",";
        vector<string> data;
        while (getline(infile, s)) {
            split(s, delim, &data);
        }
        int dataListCount = 0;
        int classListCount = 0;
        int seqNameCount = 0;
        for (int i = 0; i < data.size();) {
            for (int j = 0; j <= featureCount; j++) {
                if (j == classIndex) {
                    classList[classListCount++] = classVmap[data[i++].data()];
                } else if (j == sequenceName_index) {
                    seqName[seqNameCount++] = data[i++];
                } else {
                    dataList[dataListCount++] = (float) atof(data[i++].data());
                }

            }
        }
        infile.close();
    }
};







