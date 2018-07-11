//
// Created by LooperXX on 2018/6/19.
//

#include "BayesClassifier.h"
#include <cmath>
#include <map>
#include <vector>

using std::map;
using std::vector;

/// @brief 贝叶斯分类器虚基类
class SBayesClassifier {
public:
    /// @brief 析构函数
    virtual ~SBayesClassifier() {};

    /// @brief 训练模型
    /// @param[in] problem 贝叶斯问题
    /// @return 成功返回true, 失败返回false, 参数错误的情况下会返回false
    virtual bool TrainModel(IN const BayesProblem &problem) = 0;

    /// @brief 使用训练好的模型进行预测
    ///
    /// 请保证需要预测的样本的特征长度和训练样本的特征长度相同
    /// @param[in] sample 需要预测的样本
    /// @param[out] sampleClassValue 存储预测结果, 不能为0
    /// @return 成功预测返回true, 失败返回false, 参数错误或模型未训练的情况下会返回false
    virtual bool Predict(IN const Matrix<float> &sample, OUT float *sampleClassValue) = 0;
};

/// @brief 特征类别计数类 离散数据的计数类
class FeatureClassCount {

private:
    map<float, map<int, unsigned int>> featureClassMap; ///< 特征映射, <特征值, <类别值, 类别计数>>

public:
    /// @brief 将指定特征的指定的类别计数加1
    /// @param[in] featureValue 特征值
    /// @param[in] classValue 类别值
    void CountInc(IN float featureValue, IN int classValue) {
        featureClassMap[featureValue][classValue]++;
    }

    /// @brief 获取指定特征的指定类别的计数
    /// @param[in] featureValue 特征值
    /// @param[in] classValue 类别值
    /// @return 类别的计数
    unsigned int GetCount(IN float featureValue, IN int classValue) {
        return featureClassMap[featureValue][classValue];
    }

    /// @brief 获取指定特征的总计数
    /// @param[in] featureValue 特征值
    /// @return 特征值的所有类别的总计数
    unsigned int GetTotalCount(IN float featureValue) {
        auto classMap = featureClassMap[featureValue];
        unsigned int totalCount = 0;
        for (auto iter = classMap.begin(); iter != classMap.end(); iter++) {
            totalCount += iter->second;
        }

        return totalCount;
    }

    /// @brief 清除数据
    void Clear() {
        featureClassMap.clear();
    }
};

/// @brief 贝叶斯分类器(离散)实现类
class BayesClassifierDiscrete : public SBayesClassifier {

private:
    vector<FeatureClassCount> featureClassCountList; ///< 特征类别计数组 vector中的每个 都是一个FeatureClassCount的对象 对象有着map<特征值, <类别值, 类别计数>>
    map<int, unsigned int> sampleClassCount; ///< 训练样本类别计数
    unsigned int featureCount; ///< 样本特征数量
    unsigned int sampleCount; ///< 训练样本总数

public:
    ///@brief 构造函数
    BayesClassifierDiscrete() {
        sampleCount = 0;
        featureCount = 0;
    }

    /// @brief 析构函数
    ~BayesClassifierDiscrete() {
        sampleCount = 0;
        featureCount = 0;
        sampleClassCount.clear();
        featureClassCountList.clear();
    }

    /// @brief 训练模型
    /// @param[in] problem 贝叶斯问题
    /// @return 成功返回true, 失败返回false, 参数错误的情况下会返回false
    virtual bool TrainModel(IN const BayesProblem &problem) {
        // 进行参数检查
        if (problem.XMatrix.ColumnLen < 1)
            return false;
        if (problem.XMatrix.RowLen < 1)
            return false;
        if (problem.YVector.ColumnLen != 1)
            return false;
        if (problem.XMatrix.RowLen != problem.YVector.RowLen)
            return false;

        sampleClassCount.clear();
        featureClassCountList.clear();
        sampleCount = problem.XMatrix.RowLen;///< 训练样本总数
        featureCount = problem.XMatrix.ColumnLen;///< 样本特征数量
        for (unsigned int i = 0; i < featureCount; i++) {
            featureClassCountList.push_back(FeatureClassCount());//生成每个特征的特征类别计数类对象
        }

        for (unsigned int row = 0; row < sampleCount; row++) {
            int classValue = problem.YVector[row][0];
            sampleClassCount[classValue]++;
            for (unsigned int col = 0; col < featureCount; col++) {
                float featureValue = problem.XMatrix[row][col];
                featureClassCountList[col].CountInc(featureValue, classValue);
            }
        }

        return true;
    }

    /// @brief 使用训练好的模型进行预测
    /// 请保证需要预测的样本的特征长度和训练样本的特征长度相同
    /// @param[in] sample 需要预测的样本
    /// @param[out] sampleClassValue 存储预测结果, 不能为0
    /// @return 成功预测返回true, 失败返回false, 参数错误或模型未训练的情况下会返回false
    virtual bool Predict(IN const Matrix<float> &sample, OUT float *sampleClassValue) {
        // 检查参数
        if (1 != sample.RowLen)
            return false;
        if (featureCount != sample.ColumnLen)
            return false;
        if (nullptr == sampleClassValue)
            return false;
        if (sampleCount == 0)
            return false;

        float maxProb = 0;
        float bestClassValue;//存储结果
        int count = 0;

        for (auto iter = sampleClassCount.begin(); iter != sampleClassCount.end(); iter++) {
            int classValue = iter->first;
            float prob = this->GetProbSampleInClass(sample, classValue);
            sampleClassValue[count++] = classValue;
            sampleClassValue[count++] = prob;
            if (prob > maxProb) {
                maxProb = prob;
                bestClassValue = (float) classValue;
            }
        }

        sampleClassValue[count++] = bestClassValue;

        return true;
    }

private:
    /// @brief 获取指定样本属于指定类别的概率值, P(class | sample)
    /// @param[in] sample 样本
    /// @param[in] classValue 类别值
    /// @return 概率值
    float GetProbSampleInClass(IN const Matrix<float> &sample, IN int classValue) {
        // 贝叶斯公式:
        // P(y|x) = P(x|y) * P(y) / P(x)
        // 对于每个样本来说P(x)值相同, 所以我们只需考察分母的值, 也就是P(x|y) * P(y)
        // 因为各个特征独立所以
        // P(x|y) * P(y) = P(a1|y) * P(a2|y) * ... * P(an|y) * P(y)
        unsigned int classCount = sampleClassCount[classValue];
        float prob = 1.0f;
        for (unsigned int col = 0; col < featureCount; col++) {
            float featureValue = sample[0][col];
            unsigned int featureClassCount = featureClassCountList[col].GetCount(featureValue, classValue);
            float basicProb = (float) featureClassCount / (float) classCount;
            unsigned int featureTotalCount = featureClassCountList[col].GetTotalCount(featureValue);
            // w = 0.5 + totalCount/(1 + totalCount) * (basicProb - 0.5)  实质为 (b+0.5c/a) / (c+c/a) 其中 a: 本特征的这一特征值出现的总次数 b: 本特征的这一特征值的该类别的出现次数 c: 该类别的样本总数 原本的P= b/c
            // 使用权重概率可以解决以下问题:
            // 特征值在指定分类出现次数为0导致概率为0的情况
            float weightProb = 0.5f + (float) featureTotalCount / (1.0f + (float) featureTotalCount) * (basicProb - 0.5f);
            prob *= weightProb;
        }
        prob *= (float) classCount / (float) sampleCount;
        return prob;
    }
};

/// @brief 连续特征类别数据结构
struct FeatureClassData {
    map<int, vector<float>> DataMap;///< 类别数据映射, <类别值, 数据列表>
};

/// @brief 高斯分布结构
struct Gauss {
    float Mean; ///< 均值
    float Div; ///< 标准差
};

/// @brief 特征类别高斯分布结构
struct FeatureClassGauss {
    map<int, Gauss> GaussMap; ///< 类别高斯分布映射, <类别值, 高斯分布>
};

/// @brief 贝叶斯分类器连续(非离散)实现类
class BayesClassifierContinues : public SBayesClassifier {
private:
    vector<FeatureClassGauss> featureClassGaussList; ///< 特征类别高斯分布列表
    map<int, unsigned int> sampleClassCount; ///< 训练样本类别计数
    unsigned int featureCount; ///< 样本特征数量
    unsigned int sampleCount; ///< 训练样本总数

public:
    /// @brief 构造函数
    BayesClassifierContinues() {
        sampleCount = 0;
        featureCount = 0;
    }

    /// @brief 析构函数
    ~BayesClassifierContinues() {
        sampleCount = 0;
        featureCount = 0;
        sampleClassCount.clear();
        featureClassGaussList.clear();
    }

    /// @brief 训练模型
    /// @param[in] problem 贝叶斯问题
    /// @return 成功返回true, 失败返回false, 参数错误的情况下会返回false
    virtual bool TrainModel(IN const BayesProblem &problem) {
        // 进行参数检查
        if (problem.XMatrix.ColumnLen < 1)
            return false;
        if (problem.XMatrix.RowLen < 1)
            return false;
        if (problem.YVector.ColumnLen != 1)
            return false;
        if (problem.XMatrix.RowLen != problem.YVector.RowLen)
            return false;

        vector<FeatureClassData> featureClassDataList;

        sampleClassCount.clear();
        featureClassGaussList.clear();
        sampleCount = problem.XMatrix.RowLen;
        featureCount = problem.XMatrix.ColumnLen;
        for (unsigned int i = 0; i < featureCount; i++) {
            featureClassDataList.push_back(FeatureClassData());
            featureClassGaussList.push_back(FeatureClassGauss());
        }

        // 将每列特征值按类别归类
        for (unsigned int row = 0; row < sampleCount; row++) {
            int classValue = problem.YVector[row][0];
            sampleClassCount[classValue]++;
            for (unsigned int col = 0; col < featureCount; col++) {
                float featureValue = problem.XMatrix[row][col];
                FeatureClassData &featureClassData = featureClassDataList[col];
                featureClassData.DataMap[classValue].push_back(featureValue);
            }
        }

        // 计算数据的高斯分布
        for (unsigned int i = 0; i < featureCount; i++) {
            for (auto iter = sampleClassCount.begin(); iter != sampleClassCount.end(); iter++) {
                int classValue = iter->first;
                FeatureClassData &featureClassData = featureClassDataList[i];
                Gauss gauss = this->CalculateGauss(featureClassData.DataMap[classValue]);
                if (gauss.Div == 0.0f) { // 方差为0, 表示数据有问题, 无法使用高斯分布
                    sampleCount = 0;
                    featureCount = 0;
                    sampleClassCount.clear();
                    featureClassGaussList.clear();
                    return false;
                }
                featureClassGaussList[i].GaussMap[classValue] = gauss;
            }
        }
        return true;
    }

    /// @brief 使用训练好的模型进行预测
    /// 请保证需要预测的样本的特征长度和训练样本的特征长度相同
    /// @param[in] sample 需要预测的样本
    /// @param[out] sampleClassValue 存储预测结果, 不能为0
    /// @return 成功预测返回true, 失败返回false, 参数错误或模型未训练的情况下会返回false
    virtual bool Predict(IN const Matrix<float> &sample, OUT float *sampleClassValue) {
        // 检查参数
        if (1 != sample.RowLen)
            return false;
        if (featureCount != sample.ColumnLen)
            return false;
        if (nullptr == sampleClassValue)
            return false;
        if (sampleCount == 0)
            return false;

        float maxProb = 0.0f;
        float bestClassValue;
        int count = 0;
        for (auto iter = sampleClassCount.begin(); iter != sampleClassCount.end(); iter++) {
            int classValue = iter->first;
            float prob = this->GetProbSampleInClass(sample, classValue);
            sampleClassValue[count++] = classValue;
            sampleClassValue[count++] = prob;
            if (prob > maxProb) {
                maxProb = prob;
                bestClassValue = (float) classValue;
            }
        }
        sampleClassValue[count++] = bestClassValue;
        return true;
    }

private:
    /// @brief 获取指定样本属于指定类别的概率值, Pr(class | sample)
    /// @param[in] sample 样本
    /// @param[in] classValue 类别值
    /// @return 概率值
    float GetProbSampleInClass(IN const Matrix<float> &sample, IN int classValue) {
        // 贝叶斯公式:
        // P(y|x) = P(x|y) * P(y) / P(x)
        // 对于每个样本来说P(x)值相同, 所以我们只需考察分母的值, 也就是P(x|y) * P(y)
        // 因为各个特征独立所以
        // P(x|y) * P(y) = P(a1|y) * P(a2|y) * ... * P(an|y) * P(y)
        unsigned int classCount = sampleClassCount[classValue];
        float prob = 1.0f;
        for (unsigned int col = 0; col < featureCount; col++) {
            float featureValue = sample[0][col];
            const Gauss &gauss = featureClassGaussList[col].GaussMap[classValue];
            float temp1 = 1.0f / (float) (sqrt(2.0f * 3.14159f) * gauss.Div);
            float temp2 = (float) featureValue - gauss.Mean;
            float temp3 = exp(-1.0f * temp2 * temp2 / (2.0f * gauss.Div * gauss.Div));
            float gaussProb = temp1 * temp3;
            prob *= gaussProb;
        }
        prob *= (float) classCount / (float) sampleCount;
        return prob;
    }

    /// @brief 计算数据的高斯分布
    /// @param[in] dataList 数据列表
    /// @return 高斯分布结构
    Gauss CalculateGauss(IN const vector<float> &dataList) {
        Gauss gauss;
        gauss.Mean = 0.0f;
        gauss.Div = 0.0f;
        int len = dataList.size();
        if (len < 1)
            return gauss;
        float sum = 0.0f;
        for (unsigned int i = 0; i < len; i++) {
            sum += (float) dataList[i];
        }
        gauss.Mean = sum / len;
        float div = 0.0f;
        for (unsigned int i = 0; i < len; i++) {
            float temp = (float) dataList[i] - gauss.Mean;
            div += (temp * temp);
        }
        div = div / len;
        gauss.Div = sqrt(div);
        return gauss;
    }
};

BayesClassifier::BayesClassifier() {
    sBayesClassifier = nullptr;
}

BayesClassifier::~BayesClassifier() {
    if (nullptr != sBayesClassifier) {
        delete sBayesClassifier;
        sBayesClassifier = nullptr;
    }
}

bool BayesClassifier::TrainModel(IN const BayesProblem &problem) {
    if (nullptr != sBayesClassifier) {
        delete sBayesClassifier;
        sBayesClassifier = nullptr;
    }

    if (problem.FeatureDistribution == BAYES_FEATURE_DISCRETE)
        sBayesClassifier = new BayesClassifierDiscrete();
    else if (problem.FeatureDistribution == BAYES_FEATURE_CONTINUS)
        sBayesClassifier = new BayesClassifierContinues();
    else
        return false;


    return sBayesClassifier->TrainModel(problem);
}

bool BayesClassifier::Predict(IN const Matrix<float> &sample, OUT float *sampleClassValue) {
    return sBayesClassifier->Predict(sample, sampleClassValue);
}


