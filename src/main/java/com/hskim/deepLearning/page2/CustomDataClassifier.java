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
        /*

            배치사이즈는 10인데 10번을 배치돌리는것과 1000번 epoch 하는것의 차이가 무엇인가??

            배치 사이즈와 에포크의 정의
            배치 사이즈: 한 번에 모델에 공급되는 데이터 샘플의 수입니다. 배치 사이즈가 크면 한 번의 학습 단계에서 더 많은 데이터를 처리하지만, 메모리 사용량이 증가하고, 모델 업데이트가 덜 자주 발생합니다.
            에포크: 전체 데이터셋이 모델을 통해 한 번 전달되는 과정입니다. 모든 데이터 샘플이 모델 학습에 정확히 한 번씩 사용된 경우, 그것을 하나의 에포크로 간주합니다.
            배치 사이즈와 에포크의 차이점
            학습 동안의 가중치 업데이트 빈도: 배치 사이즈가 작으면 가중치 업데이트가 더 자주 발생합니다. 에포크 수가 많으면 동일한 데이터에 대해 여러 번 학습을 반복하여, 모델이 데이터의 패턴을 더 잘 학습할 수 있게 합니다.
            학습 시간과 효율성: 큰 배치 사이즈는 보통 더 빠른 계산을 가능하게 하지만, 너무 큰 배치 사이즈는 학습의 효율성을 저하시킬 수 있습니다. 반면, 많은 에포크 수는 모델이 데이터의 패턴을 더 잘 학습하게 하지만, 과적합(overfitting)을 일으킬 위험이 있습니다.
            1000번 에포크의 의미
            1000번의 에포크를 실행한다는 것은 전체 데이터셋을 1000번 반복하여 모델을 학습시킨다는 의미입니다. 이는 모델이 데이터의 패턴을 더 깊이 학습할 기회를 제공하지만, 과적합의 위험이 동반됩니다.
            1000번 에포크 동안 모델은 데이터의 미묘한 패턴을 포착하려고 시도하며, 학습 과정에서 모델 성능이 어떻게 변화하는지 관찰하는 것이 중요합니다.
            학습 과정에서의 조정
            조기 종료(Early Stopping): 학습 과정에서 검증 세트(validation set)에 대한 성능을 모니터링하고, 성능이 더 이상 개선되지 않을 때 학습을 조기에 종료하는 기법을 사용할 수 있습니다.
            하이퍼파라미터 튜닝: 최적의 모델 성능을 달성하기 위해 배치 사이즈, 에포크 수, 학습률 등의 하이퍼파라미터를 조정해야 할 수 있습니다.
            결론적으로, 1000번의 에포크가 의미 있는 학습 과정이 될 수 있으나, 과적합을 방지하고 최적의 모델 성능을 달성하기 위해 적절한 학습 전략과 하이퍼파라미터 조정이 필요합니다.
            데이터와 모델에 따라 최적의 설정이 달라질 수 있으므로, 다양한 설정을 실험해보고 결과를 비교하는 것이 좋습니다.

        */

        // 모델 평가 준비
        customIter.reset(); // 데이터셋 반복자를 초기화하여 평가를 준비합니다.
        Evaluation eval = new Evaluation(3); // 3개의 클래스에 대한 평가 객체 생성

        /*

            customIter.reset(); 부분은 DataSetIterator의 reset() 메서드를 호출하여 데이터셋 반복자를 초기 상태로 되돌립니다.
            즉, 모델 학습이 끝난 후에 이 메서드를 호출함으로써 데이터셋 반복자가 다시 처음부터 데이터를 제공할 수 있게 되어 평가 과정에서 모든 데이터를 사용할 수 있습니다.
            이는 특히, 데이터셋을 여러 에포크(epoch)에 걸쳐 모델에 공급할 때 중요합니다. 학습 과정에서 데이터셋을 모두 소진했다면, 평가를 시작하기 전에 반복자를 리셋해야 평가 과정에서 데이터를 순차적으로 다시 읽을 수 있습니다.

            테스트 데이터셋을 별도로 준비하지 않고, 학습에 사용된 동일한 데이터셋(customIter)으로 모델을 평가하는 것은 가능합니다.
            그러나 일반적으로 모델의 성능을 객관적으로 평가하기 위해 학습 데이터셋과는 별개의 테스트 데이터셋을 사용하는 것이 좋습니다.
            학습 데이터셋만을 사용하여 모델을 평가하면 과적합(overfitting)된 모델의 성능을 과대평가할 위험이 있습니다.

            따라서, 이상적인 접근 방법은 다음과 같습니다:

            데이터셋을 학습용과 테스트용으로 분리합니다.
            학습용 데이터셋(trainIter)으로 모델을 학습시킵니다.
            학습이 완료된 후, 테스트용 데이터셋(testIter)으로 모델을 평가합니다.
            테스트 데이터셋을 사용하지 않고 현재 방식으로 모델을 평가하는 것은 모델의 초기 성능 확인에는 유용할 수 있지만, 모델의 일반화 능력을 정확히 평가하기 위해서는 별도의 테스트 데이터셋을 준비하는 것이 중요합니다.
            만약 실제 운영 환경에서 모델을 적용하기 전이라면, 검증 데이터셋(validation set)을 추가로 분리하여 모델의 하이퍼파라미터를 조정하는 과정에서도 사용할 수 있습니다.

        */

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
