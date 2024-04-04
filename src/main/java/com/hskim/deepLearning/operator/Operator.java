package com.hskim.deepLearning.operator;

import com.hskim.deepLearning.page1.IrisClassifier;
import com.hskim.deepLearning.page2.CustomDataClassifier;
import jakarta.annotation.PostConstruct;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.io.IOException;

@Slf4j
@Component
public class Operator {
    private final IrisClassifier irisClassifier;

    private final CustomDataClassifier customDataClassifier;

    public Operator(IrisClassifier irisClassifier, CustomDataClassifier customDataClassifier){
        this.irisClassifier = irisClassifier;
        this.customDataClassifier = customDataClassifier;
    }

    // 작동기 : 작동할 컴포넌트를 골라 실험해 보세요.
    @PostConstruct
    public void operate() throws IOException {
//        irisClassifier.initModel(); // 아이리스 분류기
        customDataClassifier.initModel(); // 커스텀 데이터셋 사용 분류기
    }
}
