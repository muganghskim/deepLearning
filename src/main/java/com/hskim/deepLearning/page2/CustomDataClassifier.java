package com.hskim.deepLearning.page2;

import lombok.extern.slf4j.Slf4j;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.springframework.stereotype.Component;

import java.io.IOException;

@Slf4j
@Component
public class CustomDataClassifier {
    public void initModel() throws IOException {
        // 데이터셋 반복자 생성
        DataSetIterator customIter = new CustomDataSetIterator("src/main/resources/data.csv", 10, 3, 10);

        // 신경망 구성
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(1234)
                .updater(new Adam(0.001))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(new DenseLayer.Builder().nIn(10).nOut(20)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .nIn(20).nOut(3).build())
                .build();

        // 모델 초기화
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(10)); // 성능 추적을 위해 모델 학습 중 출력

        log.info("Training model...");

        // 모델 학습
        for (int i = 0; i < 1000; i++) {
            model.fit(customIter);
        }

        // 모델 평가 준비
        customIter.reset(); // 데이터셋 반복자를 초기화하여 평가를 준비합니다.
        Evaluation eval = new Evaluation(3); // 3개의 클래스에 대한 평가 객체 생성
        while (customIter.hasNext()) {
            DataSet dataSet = customIter.next();
            INDArray features = dataSet.getFeatures();
            INDArray targetLabels = dataSet.getLabels();
            INDArray predictions = model.output(features);
            eval.eval(targetLabels, predictions);
        }

        log.info(eval.stats());

        /*

            ========================Evaluation Metrics========================
             # of classes:    3
             Accuracy:        0.2500
             Precision:       0.2500	(2 classes excluded from average)
             Recall:          0.3333
             F1 Score:        0.4000	(2 classes excluded from average)
            Precision, recall & F1: macro-averaged (equally weighted avg. of 3 classes)

            Warning: 2 classes were never predicted by the model and were excluded from average precision
            Classes excluded from average precision: [0, 1]

            =========================Confusion Matrix=========================
             0 1 2
            -------
             0 0 2 | 0 = 0
             0 0 1 | 1 = 1
             0 0 1 | 2 = 2

            Confusion matrix format: Actual (rowClass) predicted as (columnClass) N times

            이 평가 결과는 여러 가지 원인으로 인해 발생할 수 있지만, 가장 주목해야 할 몇 가지 핵심 포인트는 다음과 같습니다:

            데이터셋의 크기: 예, 데이터 개수가 매우 적을 경우 모델이 충분히 학습되지 않아 이러한 결과가 나올 수 있습니다. 더 많은 데이터를 사용하여 모델을 학습시키면 성능이 개선될 가능성이 높습니다.

            모델의 구조나 파라미터: 모델의 구조나 학습 파라미터가 데이터셋에 적절하지 않을 수 있습니다. 예를 들어, 너무 많거나 너무 적은 수의 뉴런, 적절하지 않은 학습률 등이 문제일 수 있습니다.

            클래스 불균형: 평가 결과에 따르면 2개의 클래스(0과 1)가 모델에 의해 전혀 예측되지 않았습니다. 이는 클래스 불균형으로 인해 모델이 주로 한 클래스(여기서는 2)만을 예측하도록 학습되었을 수 있음을 나타냅니다. 데이터셋에 각 클래스의 데이터가 충분히 균형있게 포함되어 있는지 확인해야 합니다.

            과소 적합(underfitting): 제공된 정보에 따르면 모델이 데이터의 패턴을 충분히 학습하지 못한 것으로 보입니다. 이는 모델이 너무 단순하거나, 학습이 충분히 이루어지지 않았거나, 데이터가 충분하지 않음을 의미할 수 있습니다.

            해결책으로는 다음과 같은 방법이 있습니다:

            데이터 확장: 가능하다면 더 많은 데이터를 수집하거나, 데이터 증강 기법을 사용하여 기존 데이터를 확장하세요.
            모델 구조 조정: 레이어 수를 늘리거나 줄이고, 뉴런의 수를 조정하는 등 모델 구조를 실험적으로 조정해보세요.
            학습 파라미터 조정: 학습률, 배치 크기, 에폭 수 등의 학습 파라미터를 조정해보세요.
            클래스 가중치 조정: 클래스 불균형이 문제라면, 손실 함수에 클래스 가중치를 적용하여 덜 대표된 클래스의 중요성을 증가시킬 수 있습니다.
            교차 검증 사용: 모델의 일반화 능력을 평가하기 위해 교차 검증을 사용해보세요.
            실험을 통해 이러한 조정이 모델의 성능에 어떤 영향을 미치는지 관찰하고, 가장 좋은 결과를 얻을 수 있는 조합을 찾아야 합니다.

        */
    }
}
