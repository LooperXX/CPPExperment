//
// Created by LooperXX on 2018/7/3.
//
#include <bits/stdc++.h>
#include "BPnet.h"
#include "read.cpp"
using namespace std;
#define psc(a, b) pair<string, classV>(a,b)
#define pis(a, b) pair<int, string>(a,b)
int main()
{
    string name = "C:\\Users\\LooperXX\\CLionProjects\\MyNativeBayes\\data_iris.txt";
    int classCount = 3;//类别数 = 输出层节点数
    int sum = 1 + classCount * 2;//测试结果输出
    int classIndex = 4;//label在train数据中的index
    int sequenceName_index = -1; // -1:no 以index作为sequenceName
    int featureCount = 4;
    unsigned int sampleCount = 150;
    float testRatio = 0.10f;
    int testRound = 10;
    int testCount = (int) (sampleCount * testRatio);
    int trainCount = sampleCount - testCount;
    int achi = 0;
    double threshold = 0.1;
    map<int, map<int, string>> IndexMap; // < 测试次数,<index,sequenceName>>
    read t;
    t.setName(name);
    t.initClassVMap();
    float dataList[sampleCount * featureCount] = {};
    string seqName[sampleCount] = {};
    int classList[sampleCount] = {};
    if(sequenceName_index != -1){
        t.readData(dataList, classList, featureCount + 1, classIndex, sequenceName_index, seqName);
    }else{
        t.readData(dataList, classList, featureCount, classIndex, sequenceName_index, seqName);
    }


    for(int round = 0; round < testRound; round++)
    {
        // 随机生成不重复的testCount个 数
        srand((unsigned) time(nullptr));
        for (int i = 0; i < testCount; i++) {
            int temp = rand() % sampleCount;
            bool flag = true;
            for(int j = 0; j <= round; j++){
                if (IndexMap[j].count(temp) == 0)
                    continue;
                flag = false;
                i--;
                break;
            }
            if (flag) {
                IndexMap[round].insert(pis(temp, std::to_string(temp)));
                if (sequenceName_index != -1) {
                    IndexMap[round][temp] = seqName[temp];
                }
                cout << temp << "|";
            }
        }
        cout << endl;

        // 生成测试数据
        sample sampleTest[testCount];
        vector<int> testIndexList;
        int testIndex = 0;
        for (auto iter = IndexMap[round].begin(); iter != IndexMap[round].end(); iter++) {
            int index = iter->first;
            for (int i = 0; i < featureCount; i++) {
                sampleTest[testIndex].in.push_back(dataList[index * featureCount + i]);
            }
            testIndexList.push_back(index);
            sampleTest[testIndex].out.push_back(classList[index]);
//            int temp = classList[index];
//            for(int i = 0; i < classCount; i++){
//                if(i == temp){
//                    sampleTest[testIndex].out.push_back(1);
//                }else{
//                    sampleTest[testIndex].out.push_back(0);
//                }
//            }
            testIndex++;
        }
        if (testIndex != testCount) {
            cout << "Something wrong with testList dude:<" << endl;
        }
        vector<sample> testGroup(sampleTest, sampleTest + testCount);

        // 生成训练数据
        sample sampleTrain[trainCount];
        int trainIndex = 0;
        for (int index = 0; index < sampleCount; index++) {
            if (IndexMap[round].count(index) == 0) {
                for (int i = 0; i < featureCount; i++) {
                    sampleTrain[trainIndex].in.push_back(dataList[index * featureCount + i]);
                }
                int temp = classList[index];
                for(int i = 0; i < classCount; i++){
                    if(i == temp){
                        sampleTrain[trainIndex].out.push_back(1);
                    }else{
                        sampleTrain[trainIndex].out.push_back(0);
                    }
                }
                trainIndex++;
            }
        }
        if (trainIndex != trainCount) {
            cout << "Something wrong with trainList dude:<" << endl;
        }
        vector<sample> sampleGroup(sampleTrain, sampleTrain + trainCount);

        //训练
        BpNet testNet;
        testNet.training(sampleGroup, threshold);

        // 预测测试数据，并输出结果
        testNet.predict(testGroup);
        for (int i = 0; i < testCount; i++)
        {
            cout << "SequenceIndex: " << IndexMap[round][testIndexList[i]] << "\t";
            int maxIndex = 0;
            double max_temp = 0;
            for (int j = 0; j < classCount; j++){
                if(max_temp < testGroup[i].out[j]){
                    max_temp = testGroup[i].out[j];
                    maxIndex = j;
                }
            }
            auto pre = read::classV(maxIndex);
            string preS;
            auto res = read::classV(sampleTest[i].out[0]);
            string resS;
            for (auto iter = t.classVmap.begin(); iter != t.classVmap.end(); iter++) {
                string key = iter->first;
                auto value = iter->second;
                if (value == pre) {
                    preS = key;
                }
                if (value == res) {
                    resS = key;
                }
            }
            cout << "predict: " << preS << ";\ttrue : " << resS << ";" << endl;
            if(maxIndex == sampleTest[i].out[0])
                achi++;
        }
    }
    float acc = ((float) achi / (float) (testCount * testRound)) * 100;
    printf("Accuracy: %.2f%%\n", acc);
    system("pause");
    return 0;
}