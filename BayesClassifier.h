//
// Created by LooperXX on 2018/6/19.
//
/// @file BayesClassifier.h
/// @brief 贝叶斯分类器
/// @details
/// 贝叶斯分类器为: 生成模型, 有监督分类, 多元分类
/// 本文件实现的贝叶斯分类器的样本特征值数据可为连续型数据或离散型数据, 但不可混合使用, 即样本数据不可同时出现离散型数据和连续型数据
/// 连续型数据如: 人的身高
/// 离散型数据如: 文本分类中的单词数量
/// 贝叶斯分类器主要依据贝叶斯公式: P(Y|X) = P(Y, X)/P(X) = P(X, Y)/P(X) = P(X|Y) * P(Y)/P(X)

#ifndef MYNATIVEBAYES_BAYESCLASSIFIER_H
#define MYNATIVEBAYES_BAYESCLASSIFIER_H

#ifndef _BAYESCLASSIFIER_H_
#define _BAYESCLASSIFIER_H_

#include "Matrix.h"

//typedef Matrix<float> BayesMatrix; ///< 贝叶斯矩阵

/// @brief 贝叶斯特征值分布
enum BayesFeatureDistribution {
    BAYES_FEATURE_DISCRETE = 1, ///< 离散
    BAYES_FEATURE_CONTINUS = 2, ///< 连续
};

/// @brief 贝叶斯原始问题结构
/// 特征数据为非离散(连续)表示值服从高斯分布
struct BayesProblem {
    /// @brief 构造函数
    /// @param[in] sampleMatrix 样本矩阵, 每一行为一个样本, 每行中的值为样本的特征值
    /// @param[in] classVector 类别向量(列向量), 行数为样本矩阵的行数, 列数为1
    /// @param[in] dataType 特征值分布
    BayesProblem(
            IN const Matrix<float> &sampleMatrix,
            IN const Matrix<int> &classVector,
            IN BayesFeatureDistribution distribution)
            : XMatrix(sampleMatrix), YVector(classVector), FeatureDistribution(distribution) {
    }

    const Matrix<float> &XMatrix; ///< 样本矩阵
    const Matrix<int> &YVector; ///< 类别向量(列向量)
    const BayesFeatureDistribution FeatureDistribution; ///< 贝叶斯特征值分布
};

class SBayesClassifier;

/// @brief 贝叶斯分类器接口类
class BayesClassifier {
public:
    /// @brief 构造函数
    BayesClassifier();

    /// @brief 析构函数
    ~BayesClassifier();

    /// @brief 训练模型
    /// @param[in] problem 贝叶斯问题
    /// @return 成功返回true, 失败返回false, 参数错误的情况下会返回false
    bool TrainModel(IN const BayesProblem &problem);

    /// @brief 使用训练好的模型进行预测
    /// 请保证需要预测的样本的特征长度和训练样本的特征长度相同
    /// @param[in] sample 需要预测的样本
    /// @param[out] sampleClassValue 存储预测结果, 不能为0
    /// @return 成功预测返回true, 失败返回false, 参数错误或模型未训练的情况下会返回false
    bool Predict(IN const Matrix<float> &sample, OUT float *sampleClassValue);

private:
    SBayesClassifier *sBayesClassifier;///< 贝叶斯分类器实现对象

    // 禁止拷贝构造函数和赋值操作符
    BayesClassifier(const BayesClassifier &);

    BayesClassifier &operator=(const BayesClassifier &);
};

#endif
#endif //MYNATIVEBAYES_BAYESCLASSIFIER_H
