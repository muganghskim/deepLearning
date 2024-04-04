package com.hskim.deepLearning.page2;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.NoSuchElementException;

/*

    기타 유형의 데이터셋 처리
    1. 다변량 시계열 데이터
    다변량 시계열 데이터는 여러 시간에 걸친 여러 변수의 관측값을 포함합니다. 예를 들어, 날씨 데이터에서 온도, 습도, 바람의 속도 등이 모두 다변량 시계열 데이터의 예입니다.
    이 경우, 각 변수를 별도의 채널로 처리하거나, 모든 변수를 하나의 큰 벡터로 결합하여 DataSetIterator에 제공할 수 있습니다.
    2. 그래프 데이터
    그래프 데이터는 노드와 엣지로 구성되며, 소셜 네트워크 분석이나 분자 구조 모델링과 같은 분야에서 사용됩니다.
    그래프 데이터를 처리하기 위해서는 노드 임베딩, 엣지 변환 등을 통해 그래프 구조를 수치 벡터로 변환하는 과정이 필요합니다.
    3. 오디오 데이터
    오디오 데이터는 일반적으로 웨이브폼(waveform)이나 스펙트로그램(spectrogram) 형태로 처리됩니다. 오디오 파일을 직접 파싱하거나,
    라이브러리를 사용하여 특정 형식으로 변환한 후, 이를 DataSetIterator에 적합한 형태로 제공할 수 있습니다.
    커스텀 DataSetIterator 구현시 고려사항
    데이터 로딩: RecordReader나 다른 데이터 로딩 메커니즘을 사용하여 원시 데이터를 로드합니다.
    데이터 전처리: 필요한 데이터 전처리 작업(정규화, 표준화, 벡터화 등)을 수행합니다.
    배치 생성: 데이터를 배치 크기에 맞게 분할하고, 각 배치를 DataSet 객체로 변환하여 제공합니다.
    배치 순서: 데이터셋이 시퀀스 데이터인 경우, 배치를 생성할 때 시퀀스의 순서를 유지하는 것이 중요할 수 있습니다.
    커스텀 DataSetIterator를 만드는 과정은 데이터의 특성과 모델의 요구 사항에 따라 매우 다양할 수 있으며, 여기서 중요한 것은 원시 데이터를 모델이 이해할 수 있는 수치형 데이터로 적절히 변환하고,
    학습에 적합한 형태로 제공하는 것입니다. DeepLearning4J는 이러한 과정을 지원하기 위한 다양한 도구와 클래스를 제공합니다.

*/
public class CustomDataSetIterator implements DataSetIterator {

    // 예를 들어, data.csv 파일의 내용은 다음과 같을 수 있습니다:
    /*
        0.5, 0.2, 0.1, 0.4, 0.6, 0.9, 0.2, 0.8, 0.7, 0.3, 0
        0.6, 0.7, 0.8, 0.2, 0.1, 0.4, 0.3, 0.5, 0.9, 0.2, 1
        0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 2
        0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8, 0.0, 0
    */
    // 이 파일에는 10개의 특성(feature)과 1개의 레이블(label)이 포함되어 있습니다. 첫 번째부터 열 번째 열까지는 특성을 나타내고, 마지막 열은 해당 데이터 포인트의 레이블을 나타냅니다. 레이블은 0, 1, 또는 2 중 하나입니다.

    private String filePath; // 데이터 파일 경로
    private int numFeatures; // 데이터 포인트 당 특성의 수
    private int numLabels; // 예측해야 할 레이블의 수
    private int batchSize; // 배치 크기
    private List<DataSet> dataSets = new ArrayList<>();
    private int cursor = 0;

    public CustomDataSetIterator(String filePath, int numFeatures, int numLabels, int batchSize) throws IOException {
        this.filePath = filePath;
        this.numFeatures = numFeatures;
        this.numLabels = numLabels;
        this.batchSize = batchSize;
        loadData();
    }

    private void loadData() throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(filePath));
        List<INDArray> featureBatchList = new ArrayList<>();
        List<INDArray> labelBatchList = new ArrayList<>();

        String line;
        while ((line = reader.readLine()) != null) {
            String[] tokens = line.split(",");
            double[] features = new double[numFeatures];
            for (int i = 0; i < numFeatures; i++) {
                features[i] = Double.parseDouble(tokens[i]);
            }
            int labelIndex = Integer.parseInt(tokens[tokens.length - 1]);

            INDArray featureVector = Nd4j.create(features);
            /*

                첫 번째 코드
                여기서 INDArray labelVector = Nd4j.zeros(1, numLabels);는 2차원 배열을 생성합니다. 이 배열은 1xnumLabels의 형태를 가지며, 여기서 numLabels는 클래스 또는 레이블의 총 수입니다.
                labelVector.putScalar(new int[]{0, labelIndex}, 1); 호출은 labelIndex 위치의 원소를 1로 설정합니다. 이는 2차원 배열에서 사용될 때, [0, labelIndex] 위치를 지정하는 것입니다.

            */
//            INDArray labelVector = Nd4j.zeros(1, numLabels); // 3개 클래스에 대한 one-hot 벡터 생성
//            labelVector.putScalar(new int[]{0, labelIndex}, 1);
            INDArray labelVector = Nd4j.zeros(numLabels); // 수정된 부분
            labelVector.putScalar(labelIndex, 1); // 수정된 부분
            /*

                두 번째 코드
                INDArray labelVector = Nd4j.zeros(numLabels);는 1차원 배열을 생성합니다. 배열의 크기는 numLabels입니다. 이는 레이블 또는 클래스의 총 수에 해당합니다.
                labelVector.putScalar(labelIndex, 1);는 이 1차원 배열에서 labelIndex 위치의 원소를 1로 설정합니다. 여기서는 인덱스가 1차원 배열에 직접 적용됩니다.
                차이점
                첫 번째 방법은 2차원 배열(실제로는 1xN 형태의 행렬)을 생성하고, 레이블 인덱스를 설정할 때 2차원 좌표를 사용합니다. 이 방식은 일부 상황에서 필요할 수 있지만, 대부분의 경우 레이블 인코딩에는 1차원 배열이면 충분합니다.
                두 번째 방법은 더 간단하고 직관적입니다. 1차원 배열을 사용하여 각 레이블에 대한 one-hot 인코딩을 직접 생성합니다. 대부분의 머신 러닝 프레임워크와 작업할 때 이 방식이 선호됩니다.
                따라서, 두 번째 방법은 one-hot 인코딩을 위해 더 일반적으로 사용되는 접근 방식입니다. 이 방식은 코드의 가독성과 사용의 단순성 면에서 장점을 가지고 있습니다.

            */


            featureBatchList.add(featureVector);
            labelBatchList.add(labelVector);

            // 현재 배치 리스트가 batchSize에 도달하면 DataSet을 생성하고 초기화합니다.
            if (featureBatchList.size() == batchSize) {
                INDArray featureMatrix = Nd4j.vstack(featureBatchList.toArray(new INDArray[0]));
                INDArray labelMatrix = Nd4j.vstack(labelBatchList.toArray(new INDArray[0]));
                dataSets.add(new DataSet(featureMatrix, labelMatrix));

                // 다음 배치를 위해 리스트를 초기화합니다.
                featureBatchList.clear();
                labelBatchList.clear();
            }
        }
        reader.close();

        // 파일 끝에 도달했을 때, 남은 데이터가 있다면 마지막 DataSet을 추가합니다.
        if (!featureBatchList.isEmpty()) {
            INDArray featureMatrix = Nd4j.vstack(featureBatchList.toArray(new INDArray[0]));
            INDArray labelMatrix = Nd4j.vstack(labelBatchList.toArray(new INDArray[0]));
            dataSets.add(new DataSet(featureMatrix, labelMatrix));
        }
    }

    @Override
    public DataSet next(int num) {
        if (!hasNext()) throw new NoSuchElementException();
        DataSet nextDataSet = dataSets.get(cursor++);
        return nextDataSet;
    }

    // 입력 데이터의 특성 수(컬럼 수)를 반환합니다. 구현이 완료되지 않았으므로 0을 반환합니다.
    @Override
    public int inputColumns() {
        return 0;
    }

    // 가능한 결과(레이블 또는 클래스)의 총 수를 반환합니다. 구현이 완료되지 않았으므로 0을 반환합니다.
    @Override
    public int totalOutcomes() {
        return 0;
    }

    // 이 DataSetIterator가 reset() 메소드를 지원하는지 여부를 나타냅니다. 이 경우 false를 반환하여 지원하지 않음을 나타냅니다.
    @Override
    public boolean resetSupported() {
        return false;
    }

    // 비동기적으로 데이터를 로드하는 기능을 지원하는지 여부를 나타냅니다. 이 경우 false를 반환합니다.
    @Override
    public boolean asyncSupported() {
        return false;
    }

//    @Override
//    public boolean hasNext() {
//        // 다음 배치 데이터가 있는지 확인하는 로직을 구현합니다.
//        return true; // 예제 코드에서는 항상 true를 반환합니다.
//    }

    @Override
    public boolean hasNext() {
        return cursor < dataSets.size();
    }

    // 다음 데이터 배치를 반환합니다.
    @Override
    public DataSet next() {
        return next(batchSize); // 이렇게 수정하면, next(int num) 메서드와 일관된 동작을 합니다.
    }

//    @Override
//    public void reset() {
//        // 데이터셋을 처음으로 되돌리는 로직을 구현합니다.
//    }

    @Override
    public void reset() {
        cursor = 0;
    }

    // 현재 배치 크기를 반환합니다. 구현이 완료되지 않았으므로 0을 반환합니다.
    @Override
    public int batch() {
        return 0;
    }

    // 데이터셋에 대한 사전 처리기를 설정합니다. 이 예제에서는 사전 처리 로직이 구현되지 않았습니다.
    @Override
    public void setPreProcessor(DataSetPreProcessor dataSetPreProcessor) {

    }

    // 설정된 데이터셋 사전 처리기를 반환합니다. 구현되지 않았으므로 null을 반환합니다.
    @Override
    public DataSetPreProcessor getPreProcessor() {
        return null;
    }

    // 레이블의 리스트를 반환합니다. 이 예제에서는 구현되지 않았으므로 null을 반환합니다.
    @Override
    public List<String> getLabels() {
        return null;
    }

}
