//
// Created by LooperXX on 2018/6/19.
//


#include <bits/stdc++.h>
#include <Windows.h>
#include "BayesClassifier.h"
#include "read.cpp"
#define pis(a, b) pair<int, string>(a,b)
using std::string;
using std::ifstream;
using std::cout;
using std::endl;
using std::vector;
using std::map;
using std::pair;

// TODO: sequenceName
///@details: 本程序的数据要求 除类别数据外 均为数值型数据 即要求自行处理离散型数据为数值型数据
///@details: 本程序中所指特征值 不包括sequenceName 与 classData 即不作为离散型数据  不必计入disCount与featureCount
///@details: 此外，对于sequenceName，如果不存在，则将sequenceName_index 设为-1 测试结果中显示其在数据集中的index 如果存在 则给出其column_index 测试结果中将显示其sequenceName

int main() {
    string name = "C:\\Users\\LooperXX\\CLionProjects\\MyNativeBayes\\data_balance-scale.txt";
    // 初始化训练参数: 类别数目 类别坐标 离散特征数目 连续特征数目 样本数目 数据类型 测试比率
    unsigned int sampleCount = 625;
    int classCount = 3;
    int classIndex = 0;
    int disCount = 4;
    int conCount = 0;
    int sequenceName_index = -1; // -1:no 以index作为sequenceName
    int type = 0;// 0: 离散 1: 连续 2: 混合
    int disIndex[disCount] = {};// 若为混合型
    int conIndex[conCount] = {};
    float testRatio = 0.10f;
    int testRound = 10;
    int featureCount = disCount + conCount;
    int sum = 1 + classCount * 2;
    int testCount = (int) (sampleCount * testRatio);
    int trainCount = sampleCount - testCount;
    int achi = 0;
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


    for(int round = 0; round < testRound; round++) {
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
        float dataList_test[testCount * featureCount] = {};
        int classList_test[testCount] = {};
        int testIndex = 0;
        int testIndexList[testCount] = {};
        for (auto iter = IndexMap[round].begin(); iter != IndexMap[round].end(); iter++) {
            int index = iter->first;
            for (int i = 0; i < featureCount; i++) {
                dataList_test[testIndex * featureCount + i] = dataList[index * featureCount + i];
            }
            testIndexList[testIndex] = index;
            classList_test[testIndex++] = classList[index];
        }
        if (testIndex != testCount) {
            cout << "Something wrong with testList dude:<" << endl;
        }

        // 生成训练数据
        float dataList_train[trainCount * featureCount] = {};
        int classList_train[trainCount] = {};
        int trainIndex = 0;
        for (int index = 0; index < sampleCount; index++) {
            if (IndexMap[round].count(index) == 0) {
                for (int i = 0; i < featureCount; i++) {
                    dataList_train[trainIndex * featureCount + i] = dataList[index * featureCount + i];
                }
                classList_train[trainIndex++] = classList[index];
            }
        }
        if (trainIndex != trainCount) {
            cout << "Something wrong with trainList dude:<" << endl;
        }
        // 定义样本矩阵
        // 定义样本的类别向量
        float predictValue1[sum] = {};
        float predictValue2[sum] = {};
        BayesClassifier classifier1;
        BayesClassifier classifier2;
        BayesClassifier classifier3;
        BayesClassifier classifier4;
        // train_matrix
        Matrix<int> classVector(trainCount, 1, classList_train);
        Matrix<float> trainMatrix(trainCount, featureCount, dataList_train);
        // test_matrix
        Matrix<int> result(testCount, 1, classList_test);
        Matrix<float> testMatrix(testCount, featureCount, dataList_test);
        // 训练与预测
        if (type == 0) {
            BayesProblem problem1(trainMatrix, classVector, BAYES_FEATURE_DISCRETE);
            classifier1.TrainModel(problem1);
            Matrix<float> newSample(1, featureCount);
            for (int i = 0; i < testCount; i++) {
                for (int j = 0; j < featureCount; j++) {
                    newSample[0][j] = testMatrix[i][j];
                }
                classifier1.Predict(newSample, predictValue1);
                cout << "SequenceIndex: " << IndexMap[round][testIndexList[i]] << " ;";
                auto pre = read::classV((int) predictValue1[sum - 1]);
                string preS;
                auto res = read::classV(result[i][0]);
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
                cout << "predict: " << preS << ";true : " << resS << ";" << endl;
//                printf("predict: %d ;true : %d\n", (int) predictValue1[sum - 1],
//                       result[i][0]);
                if ((int) predictValue1[sum - 1] == result[i][0]) achi++;
            }
        } else if (type == 1) {
            BayesProblem problem2(trainMatrix, classVector, BAYES_FEATURE_CONTINUS);
            classifier2.TrainModel(problem2);
            Matrix<float> newSample(1, featureCount);
            for (int i = 0; i < testCount; i++) {
                for (int j = 0; j < featureCount; j++) {
                    newSample[0][j] = testMatrix[i][j];
                }
                classifier2.Predict(newSample, predictValue2);
                //                for(int k = 0; k < sum; k++){
                //                    cout << predictValue2[k] << " | ";
                //                }
                //                如结果有误，可打印查看
                cout << "SequenceIndex: " << IndexMap[round][testIndexList[i]] << " ;";
                auto pre = read::classV((int) predictValue2[sum - 1]);
                string preS;
                auto res = read::classV(result[i][0]);
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
                cout << "predict: " << preS << ";true : " << resS << ";" << endl;
                //printf("predict: %d ;true : %d\n", (int) predictValue2[sum - 1], result[i][0]);
                if ((int) predictValue2[sum - 1] == result[i][0]) achi++;
            }
        } else if (type == 2) {
            // train
            float data_train_con[trainCount * conCount] = {};
            float data_train_dis[trainCount * disCount] = {};
            for (int i = 0; i < trainCount; i++) {
                for (int j = 0; j < conCount; j++) {
                    data_train_con[i * conCount + j] = dataList_train[i * featureCount + conIndex[j]];
                }
                for (int j = 0; j < disCount; j++) {
                    data_train_dis[i * disCount + j] = dataList_train[i * featureCount + disIndex[j]];
                }
            }
            Matrix<float> trainMatrix_con(trainCount, conCount, data_train_con);
            Matrix<float> trainMatrix_dis(trainCount, disCount, data_train_dis);
            BayesProblem problem3(trainMatrix_dis, classVector, BAYES_FEATURE_DISCRETE);
            classifier3.TrainModel(problem3);
            BayesProblem problem4(trainMatrix_con, classVector, BAYES_FEATURE_CONTINUS);
            classifier4.TrainModel(problem4);
            // test
            Matrix<float> newDisSample(1, disCount);
            Matrix<float> newConSample(1, conCount);
            for (int i = 0; i < testCount; i++) {
                for (int j = 0; j < conCount; j++) {
                    newConSample[0][j] = testMatrix[i][conIndex[j]];
                }
                for (int j = 0; j < disCount; j++) {
                    newDisSample[0][j] = testMatrix[i][disIndex[j]];
                }
                classifier3.Predict(newDisSample, predictValue1);
                classifier4.Predict(newConSample, predictValue2);
                if (predictValue1[sum - 1] == predictValue2[sum - 1]) {
                    cout << "SequenceIndex: " << IndexMap[round][testIndexList[i]] << " ;";
                    printf("predict: %d ;true : %d\n", (int) predictValue1[sum - 1],
                           result[i][0]);
                    if ((int) predictValue1[sum - 1] == result[i][0]) achi++;
                } else {
                    float temp;
                    float max = 0.0f;
                    int index = 1;
                    for (unsigned int j = 1; j < sum; j = j + 2) {
                        temp = predictValue1[j] * predictValue2[j];
                        if (temp > max) {
                            max = temp;
                            index = j;
                        }
                    }
                    cout << "SequenceIndex: " << IndexMap[round][testIndexList[i]] << " ;";

                    auto pre = read::classV(index - 1);
                    string preS;
                    auto res = read::classV(result[i][0]);
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
                    cout << "predict: " << preS << ";true : " << resS << ";" << endl;

                    //printf("predict: %d ;true : %d\n", index - 1, result[i][0]);
                    if (index - 1 == result[i][0]) achi++;
                }
            }
        } else {
            cout << "Something wrong dude :<\n";
        }
    }
    float acc = ((float) achi / (float) (testCount * testRound)) * 100;
    printf("Accuracy: %.2f%%\n", acc);
    system("pause");
    return 0;
}