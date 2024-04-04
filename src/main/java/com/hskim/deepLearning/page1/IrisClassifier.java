package com.hskim.deepLearning.page1;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.springframework.stereotype.Component;

@Slf4j
@Component
public class IrisClassifier {

    public void initModel() {

        // IrisDataSetIterator는 Iris 데이터셋을 로드하는 데 사용됩니다. 여기서 첫 번째 150은 데이터셋의 총 샘플 수를, 두 번째 150은 한 번에 로드할 샘플 수(배치 크기)를 의미합니다.
        /*

            질문 1 총데이터는 301개이고 배치수를 150으로 한다면?

            만약 총 데이터 샘플이 301개 있고, 배치 크기를 150으로 설정한다면, 이는 데이터 로딩 과정에서 데이터셋이 배치로 나뉘어 처리됨을 의미
            첫 번째 배치: 첫 번째로 로드하는 배치에는 150개의 샘플이 포함됩니다.
            두 번째 배치: 두 번째 배치에서는 다음 150개의 샘플을 로드하려고 시도합니다. 하지만, 실제로는 301 - 150 = 151개의 샘플만 남아 있으므로, 두 번째 배치에는 151개의 샘플이 포함됩니다.
            배치 처리 완료: 이로 인해 전체 데이터셋이 두 배치로 나누어 처리되며, 첫 번째 배치에는 150개, 두 번째 배치에는 151개의 샘플이 포함됩니다.
            데이터셋 크기보다 배치 크기를 작게 설정하는 것이 일반적이며, 이를 통해 메모리 사용량을 관리하고,
            학습 과정에서 미니배치 경사 하강법과 같은 최적화 알고리즘을 효율적으로 사용할 수 있습니다.

        */
        DataSetIterator irisIter = new IrisDataSetIterator(150, 150);

        // 신경망 모델의 구조를 정의합니다. NeuralNetConfiguration.Builder를 사용하여 모델의 각종 설정을 초기화합니다.
        // .seed(123)는 모델 학습시 난수 생성의 일관성을 위해 시드 값을 설정합니다.
        // .weightInit(WeightInit.XAVIER)는 가중치 초기화 방식으로 Xavier 초기화 방법을 사용합니다. 이는 레이어 간의 정보가 적절히 흐르도록 도와줍니다.
        /*

            Xavier 초기화는 신경망의 각 레이어의 가중치를 초기화할 때 입력 노드와 출력 노드의 수를 기반으로 한 값의 범위 내에서 무작위로 선택하여 초기화하는 방법입니다.

            이 초기화 방법은 신경망을 학습할 때 가중치의 적절한 시작 값을 제공함으로써, 학습 초기 단계에서의 그라디언트(gradient) 소실이나 폭발 문제를 완화하는 데 도움을 줍니다.

            Xavier 초기화의 원리
            Xavier 초기화는 신경망의 전방향 전파(forward propagation)와 역방향 전파(backward propagation) 중에 그라디언트가 적절하게 유지되도록 설계되었습니다.
            구체적으로는, 각 레이어의 입력과 출력 사이에서 분산이 유지되도록 하여 그라디언트가 너무 커지거나 작아지는 것을 방지합니다.

            이 초기화 방법에서 가중치는 다음과 같은 분포에서 무작위로 선택됩니다:

            균등 분포(U(-a, a)) 또는 정규 분포 N(0, σ^2)에서 추출되며,
            여기서 a 또는 σ는 각각 균등 분포의 범위 또는 정규 분포의 표준편차를 나타내며,
            이 값들은 레이어의 입력 노드 수(fan-in)와 출력 노드 수(fan-out)에 따라 결정됩니다.
            Xavier 초기화는 다음과 같은 공식을 사용하여 계산될 수 있습니다:

            균등 분포의 경우: a = sqrt(6 / (fan_in + fan_out))
            정규 분포의 경우: σ = sqrt(2 / (fan_in + fan_out))

            Xavier 초기화의 효과
            신경망의 초기 가중치가 너무 작으면 신호가 너무 약해져 그라디언트 소실 문제가 발생할 수 있습니다.
            반대로 초기 가중치가 너무 크면 그라디언트가 폭발할 수 있습니다.
            Xavier 초기화는 이러한 문제를 완화하고 신경망의 학습 속도를 개선하기 위해 가중치의 초기값을 적절한 범위 내에서 설정합니다.
            Xavier 초기화는 주로 활성화 함수로 Sigmoid나 Tanh가 사용되는 전통적인 심층 신경망에 적합합니다. ReLU 활성화 함수를 사용할 경우,
            He 초기화(또는 Kaiming 초기화)가 더 적합할 수 있습니다. He 초기화는 Xavier 초기화의 변형으로, ReLU의 특성을 고려하여 가중치를 초기화하는 방법입니다.

        */
        // .updater(new Sgd(0.1))는 모델의 가중치를 업데이트하는 방법으로 확률적 경사 하강법(SGD)을 사용하며, 학습률(learning rate)은 0.1입니다.
        /*

            확률적 경사 하강법 (SGD)
            경사 하강법(Gradient Descent) 은 비용 함수(Cost Function)의 기울기(Gradient)를 계산하여,
            이 기울기의 반대 방향으로 조금씩 이동시키며 최소값을 찾아가는 최적화 방법입니다. 이 과정에서 모델의 가중치가 점차적으로 업데이트됩니다.
            확률적(Stochastic) 이라는 용어는 전체 데이터셋 대신 무작위로 선택된 하나의 데이터 또는 소수의 데이터 샘플(미니배치)을 사용하여 경사를 계산한다는 의미입니다.
            이 방법은 전체 데이터셋을 사용하는 것에 비해 더 빠르게 수렴할 수 있으며, 로컬 미니멈(Local Minimum)에 빠질 위험을 줄여줍니다.

            학습률 (Learning Rate)
            학습률은 매 학습 단계에서 가중치를 얼마나 조정할 것인지를 결정합니다. 즉, 비용 함수의 기울기에 학습률을 곱한 값만큼 가중치를 업데이트합니다.
            학습률이 너무 높으면, 모델이 최소값을 지나치거나 발산할 위험이 있습니다. 반면, 학습률이 너무 낮으면 학습 속도가 매우 느려지고, 최적값에 도달하기 전에 학습이 멈출 수 있습니다.
            적절한 학습률을 설정하는 것은 모델의 성능과 학습 속도에 중요한 영향을 미칩니다. 일반적으로 실험을 통해 최적의 학습률 값을 찾아야 합니다.

            SGD의 활용
            SGD는 다양한 변형(예: Momentum, Nesterov Accelerated Gradient 등)과 함께 딥러닝 모델의 학습에 널리 사용됩니다.
            이러한 변형들은 일반적인 SGD의 단점을 개선하고, 학습 과정의 안정성과 수렴 속도를 높이는 데 도움을 줍니다.

        */
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .weightInit(WeightInit.XAVIER)
                .updater(new Sgd(0.1))
                .list()
                .layer(0, new DenseLayer.Builder().nIn(4).nOut(10)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .nIn(10).nOut(3).build())
                .build();
        /*

            첫 번째 레이어: DenseLayer
            DenseLayer(완전 연결 레이어): 이 레이어의 모든 뉴런(노드)은 이전 레이어의 모든 뉴런과 연결되어 있습니다. DenseLayer는 입력된 데이터의 패턴을 학습하는 데 사용됩니다.
            입력과 출력: 이 예제에서 첫 번째 DenseLayer는 4개의 입력을 받아서 10개의 출력을 생성합니다.
            4개의 입력은 Iris 데이터셋의 특성(꽃받침 길이, 꽃받침 너비, 꽃잎 길이, 꽃잎 너비)을 나타내며, 10개의 출력은 다음 레이어로 전달됩니다.
            활성화 함수 ReLU: 활성화 함수로는 ReLU(Rectified Linear Unit)가 사용됩니다. ReLU 함수는 입력이 0보다 클 경우 그 입력을 그대로 출력하고,
            0 이하일 경우 0을 출력합니다. 이는 비선형성을 도입하여 신경망이 복잡한 문제를 더 잘 해결할 수 있게 해줍니다.

                ReLU 함수의 정의
                ReLU 함수는 입력값 x에 대해 다음과 같이 정의됩니다:

                f(x)=max(0,x)

                이 수학적 표현은 입력 x가 0보다 클 경우 x를 그대로 출력하고, 0 이하일 경우 0을 출력한다는 의미입니다.

                ReLU의 특징
                비선형성: ReLU는 간단한 비선형 함수입니다. 비선형 활성화 함수는 신경망이 선형 모델로 표현할 수 없는 복잡한 패턴과 관계를 학습할 수 있게 해줍니다.
                계산 효율성: ReLU의 계산은 매우 간단합니다. 최대값 연산만 필요하기 때문에, 다른 활성화 함수(예: 시그모이드, tanh)에 비해 계산 비용이 낮습니다.
                희소성: ReLU를 사용하면 네트워크의 출력이 0이 될 가능성이 높아져, 결과적으로 뉴런의 활성화가 희소해집니다. 희소성은 네트워크의 효율성과 일반화 능력을 향상시키는 데 도움이 될 수 있습니다.

                ReLU의 장점
                그라디언트 소실 문제 완화: 심층 신경망에서는 그라디언트가 레이어를 역전파하면서 점점 작아지는(소실하는) 현상이 발생할 수 있습니다. ReLU는 양수 입력에 대해 그라디언트가 상수(1)이기 때문에, 이 문제를 일정 부분 완화할 수 있습니다.
                학습 속도 향상: ReLU의 간단한 계산 덕분에 신경망의 학습 속도가 다른 활성화 함수를 사용할 때보다 빨라질 수 있습니다.

                ReLU의 단점
                죽은 ReLU 문제(Dead ReLU Problem): 특정 뉴런이 항상 0만을 출력하게 되는 현상을 말합니다. 이는 뉴런의 가중치가 특정 값으로 초기화되거나,
                학습 도중 큰 그라디언트 값으로 인해 가중치가 크게 업데이트되어 발생할 수 있습니다. 이 문제를 해결하기 위해 Leaky ReLU나 Parametric ReLU와 같은 ReLU의 변형들이 제안되었습니다.

            두 번째 레이어: OutputLayer
            OutputLayer: 신경망의 마지막 레이어로, 실제 예측을 출력합니다. 분류 문제에서는 클래스별 확률을 출력하기 위해 사용됩니다.
            입력과 출력: 이 예제에서 OutputLayer는 10개의 입력을 받고, Iris 데이터셋의 3개 클래스에 대한 확률을 출력합니다.
            10개의 입력은 이전 DenseLayer의 출력이며, 3개의 출력은 각각 Iris 꽃의 종류(셋오사, 버시컬러, 버지니카)에 대한 확률을 나타냅니다.
            활성화 함수 Softmax: Softmax 활성화 함수는 다중 클래스 분류 문제에서 출력 레이어에서 주로 사용됩니다. 각 클래스에 대한 확률을 출력하며,
            모든 클래스의 확률 합은 1이 됩니다. 이를 통해 가장 높은 확률을 가진 클래스를 최종 예측 결과로 선택할 수 있습니다.
            손실 함수 Negative Log Likelihood: 분류 문제에서 모델의 예측과 실제 레이블 사이의 차이를 측정하는 데 사용됩니다. 이 손실 함수는 모델이 정답 클래스에 높은 확률을 할당하도록 학습을 유도합니다.
            위의 설명에서 DenseLayer는 입력 데이터에서 특성을 학습하는 역할을 하며, OutputLayer는 학습된 특성을 바탕으로 각 클래스에 대한 확률을 출력하여 최종 분류를 수행합니다.
            ReLU와 Softmax 활성화 함수는 각각의 레이어에서 적절한 비선형성과 확률적 출력을 제공합니다. Negative Log Likelihood 손실 함수는 이러한 예측이 실제 레이블과 얼마나 잘 일치하는지를 평가하여 모델 학습을 지도합니다.

            레이어는 순차적인가? 하나의 레이어의 출력은 항상 다음 레이어의 입력인가?
            일반적인 신경망 구조에서 레이어는 순차적으로 배열되며, 한 레이어의 출력은 다음 레이어의 입력으로 사용됩니다. 이러한 구조는 "피드포워드 신경망(Feedforward Neural Networks)"에서 가장 흔히 볼 수 있으며, 가장 기본적이고 널리 사용되는 신경망 구조 중 하나입니다.

            순차적 레이어 구조의 특징:
            순차적 처리: 데이터는 신경망의 첫 번째 레이어로 입력되어, 각 레이어를 순차적으로 통과하며 처리됩니다. 마지막 레이어에서는 최종 예측 결과가 출력됩니다.
            정보의 흐름: 각 레이어는 이전 레이어로부터 받은 입력에 대해 연산(가중치 곱 및 활성화 함수 적용)을 수행하고, 그 결과를 다음 레이어로 전달합니다. 이 과정에서 정보가 신경망을 통해 흐르게 됩니다.
            출력과 입력의 관계: 정의에 따라, 한 레이어의 출력 크기(output size)는 다음 레이어의 입력 크기(input size)와 일치해야 합니다. 이는 각 레이어가 서로 호환되어 정보가 손실 없이 전달될 수 있게 합니다.
            예외적인 구조:
            병렬 구조와 스킵 연결: 일부 신경망 아키텍처는 순차적이지 않은 구조를 가질 수 있습니다. 예를 들어, "인셉션 네트워크(Inception Networks)"는 여러 크기의 필터를 병렬로 적용하고,
            "잔차 네트워크(Residual Networks, ResNet)"는 입력을 일부 레이어를 건너뛰고(스킵하고) 직접 출력에 더하는 스킵 연결(skip connections)을 사용합니다.
            순환 신경망(RNN): RNN과 같은 순환 신경망은 과거의 정보를 순차적으로 처리하는 구조이지만, 각 타임 스텝에서의 출력이 다음 타임 스텝의 입력으로 사용되는 특별한 형태의 연결 구조를 가집니다.
            이는 시계열 데이터나 자연어 처리와 같이 시퀀스 데이터를 처리하는 데 적합합니다.
            따라서, 대부분의 신경망에서 레이어 간의 정보 흐름은 순차적이며, 각 레이어의 출력은 다음 레이어의 입력으로 직접 연결됩니다.
            하지만, 특정 아키텍처나 목적에 따라 이러한 구조가 변형되어 더 복잡한 정보 처리 방식을 구현하기도 합니다.

        */

        // MultiLayerNetwork 객체를 생성하고, 앞서 정의한 구성으로 초기화합니다.
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        // fit 메소드를 사용하여 주어진 데이터셋으로 모델을 학습시킵니다. 여기서는 총 1000번의 에폭(epoch) 동안 학습합니다.
        for (int i = 0; i < 1000; i++) {
            model.fit(irisIter);
        }

        /*

            테스트 데이터셋 준비: 모델 평가를 위해 별도의 테스트 데이터셋을 준비해야 합니다. 이 예시에서는 간단히 동일한 IrisDataSetIterator를 사용했지만, 일반적으로는 학습 데이터와 분리된 테스트 데이터셋을 사용합니다.

            Evaluation 객체 생성: new Evaluation(클래스 수)를 호출하여 Evaluation 객체를 생성합니다. Iris 데이터셋의 경우 클래스 수는 3입니다.

            평가 실행: 테스트 데이터셋의 각 배치에 대해, model.output(features, false)를 호출하여 모델의 예측을 수행합니다. 그런 다음, eval.eval(labels, predicted)를 호출하여 실제 레이블과 예측된 레이블을 비교하고, 평가 지표를 계산합니다.

            평가 결과 출력: eval.stats()를 호출하여 모델의 정확도, 정밀도, 재현율, F1 점수 등 다양한 평가 지표를 포함하는 평가 결과를 출력합니다.

        */
        DataSetIterator testIter = new IrisDataSetIterator(150, 150);

        // Evaluation 클래스의 인스턴스 생성
        Evaluation eval = new Evaluation(3); // 분류 문제의 클래스 수 (Iris 데이터셋의 경우 3)

        // 모델 평가 수행
        while(testIter.hasNext()) {
            org.nd4j.linalg.dataset.DataSet t = testIter.next();
            org.nd4j.linalg.api.ndarray.INDArray features = t.getFeatures();
            org.nd4j.linalg.api.ndarray.INDArray labels = t.getLabels();
            org.nd4j.linalg.api.ndarray.INDArray predicted = model.output(features, false);
            eval.eval(labels, predicted);
        }

        // 평가 결과 출력
        log.info("평가 결과 출력 : {}", eval.stats());

        /*

            평가 결과 해석
            클래스의 수: 모델은 3개의 클래스(셋오사, 버시컬러, 버지니카)로 Iris 꽃을 분류합니다.
            정확도(Accuracy): 약 97.33%. 이는 모델이 전체 데이터 중 약 97.33%를 올바르게 분류했다는 것을 의미합니다.
            정밀도(Precision): 약 97.53%. 정밀도는 모델이 어떤 클래스로 분류한 항목들 중 실제로 그 클래스에 속하는 항목의 비율입니다. 이 경우, 모델이 특정 클래스라고 예측한 경우, 그 예측이 얼마나 정확한지를 나타냅니다.
            재현율(Recall): 약 97.33%. 재현율은 실제 클래스에 속하는 항목들 중 모델이 얼마나 많은 항목을 올바르게 감지했는지를 나타냅니다.
            F1 점수(F1 Score): 약 97.33%. F1 점수는 정밀도와 재현율의 조화 평균으로, 두 측정치의 균형을 나타냅니다.

            혼동 행렬(Confusion Matrix)
              0  1  2
            ----------
             50  0  0 | 0 = 0
              0 46  4 | 1 = 1
              0  0 50 | 2 = 2
            혼동 행렬은 모델의 성능을 보다 상세하게 이해할 수 있게 해줍니다. 각 행은 실제 클래스를, 각 열은 모델이 예측한 클래스를 나타냅니다.
            첫 번째 클래스(0)의 경우, 50개 모두를 정확히 분류했습니다(50개 True Positive).
            두 번째 클래스(1)의 경우, 46개를 정확히 분류하고, 4개를 세 번째 클래스(2)로 잘못 분류했습니다(4개 False Positive).
            세 번째 클래스(2)는 모두 정확하게 분류되었습니다(50개 True Positive).

        */
    }

}
