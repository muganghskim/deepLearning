package com.hskim.deepLearning.page3;

import lombok.extern.slf4j.Slf4j;
import org.datavec.api.split.FileSplit;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.ResizeImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
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
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.springframework.stereotype.Component;

import java.io.File;
import java.io.IOException;

@Slf4j
@Component
public class CatClassifier {
    public void initModel() throws IOException {
        int height = 50;  // 이미지의 높이
        int width = 50;   // 이미지의 너비
        // 모든 이미지를 신경망이 처리하기 쉽게 50*50으로 변환하여 처리함

        int channels = 1; // 이미지의 채널 수 (RGB)
        /*

            channels 변수는 이미지 데이터의 채널 수를 나타냅니다. 이미지의 채널은 색상 정보의 차원을 의미하며, 일반적으로 다음과 같이 구성됩니다:

            흑백 이미지: 채널 수가 1입니다. 이미지의 각 픽셀은 단일 명암 값(그레이스케일)으로 표현됩니다.
            컬러 이미지 (RGB): 채널 수가 3입니다. 이미지의 각 픽셀은 빨강(Red), 초록(Green), 파랑(Blue)의 세 가지 색상 값으로 구성됩니다.
            각 색상 채널은 해당 색상의 강도를 나타내며, 이 세 가지 값을 조합하여 다양한 색상을 표현합니다.
            컬러 이미지 (RGBA): 채널 수가 4인 경우도 있습니다. 여기에는 RGB에 알파(Alpha) 채널이 추가되어, 투명도 정보를 포함합니다.
            channels = 3으로 설정된 경우, 이는 이미지가 RGB 컬러 이미지임을 나타냅니다. 즉, 각 이미지가 빨강, 초록, 파랑의 세 가지 색상 채널을 갖고 있으며,
            이를 통해 다채로운 색상의 이미지를 처리할 수 있음을 의미합니다. DeepLearning4J의 ImageRecordReader를 사용할 때 이 값은 로딩할 이미지 데이터의 형식을 정확히 반영해야 합니다.

        */

        // ImageRecordReader 초기화
        ImageRecordReader imageRecordReader = new ImageRecordReader(height, width, channels);
        ImageRecordReader testImageRecordReader = new ImageRecordReader(height, width, channels);

        // 이미지 변환 설정 (옵션)
        ImageTransform transform = new ResizeImageTransform(width, height);

        // ImageRecordReader 구성
        imageRecordReader.initialize(new FileSplit(new File("src/main/resources/image/train/")), transform);
        testImageRecordReader.initialize(new FileSplit(new File("src/main/resources/image/test/")), transform);

        int batchSize = 64; // 한 번에 처리할 이미지 수
        boolean train = true; // 훈련 데이터인지 여부

        DataSetIterator dataSetIterator = new RecordReaderDataSetIterator(imageRecordReader, batchSize, 1, 2);
        dataSetIterator.setPreProcessor(new ImagePreProcessingScaler(0, 1)); // 이미지 데이터 정규화

        DataSetIterator testDataSetIterator = new RecordReaderDataSetIterator(testImageRecordReader, batchSize, 1, 2);
        testDataSetIterator.setPreProcessor(new ImagePreProcessingScaler(0, 1)); // 테스트 이미지 데이터 정규화

        // 신경망 구성
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .updater(new Adam(0.001))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(height * width * channels)
                        .nOut(1000)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                        .nIn(1000)
                        .nOut(2)
                        .activation(Activation.SOFTMAX)
                        .build())
                .build();

        // 신경망 초기화
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(10)); // 모델의 학습 과정에서 점수를 로그로 출력

        // 모델 훈련
        log.info("Train model...");
        for (int i = 0; i < 20; i++) {
            model.fit(dataSetIterator);
        }

        // 모델 평가
        log.info("Evaluate model...");
        Evaluation eval = new Evaluation(2); // 2개 클래스 (고양이, 비고양이)
        while (testDataSetIterator.hasNext()) {
            DataSet testDataSet = testDataSetIterator.next();
            INDArray output = model.output(testDataSet.getFeatures());
            eval.eval(testDataSet.getLabels(), output);
        }

        log.info(eval.stats());
    }
}
