package com.hskim.deepLearning.operator;

import com.hskim.deepLearning.page1.IrisClassifier;
import jakarta.annotation.PostConstruct;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

@Slf4j
@Component
public class Operator {
    private final IrisClassifier irisClassifier;

    public Operator(IrisClassifier irisClassifier){
        this.irisClassifier = irisClassifier;
    }

    // 작동기 : 작동할 컴포넌트를 골라 실험해 보세요.
    @PostConstruct
    public void operate(){
        irisClassifier.initModel(); // 아이리스 분류기
    }
}
