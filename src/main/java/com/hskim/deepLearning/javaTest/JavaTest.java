package com.hskim.deepLearning.javaTest;

import jakarta.annotation.PostConstruct;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.io.*;
import java.time.LocalDate;
import java.time.temporal.ChronoUnit;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;


@Slf4j
@Component
public class JavaTest {
    // 1번 문제 Java 8에서 추가된 스트림 API를 사용하여 주어진 리스트에서 짝수만을 필터링하고, 각 값에 3을 곱한 후, 결과 값을 리스트로 반환하는 코드를 작성하세요.
    public List<Integer> streamApi(){
        List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);

        List<Integer> result = numbers.stream() // 스트림 생성
                .filter(n -> n % 2 == 0) // 짝수 필터링
                .map(n -> n * 3) // 각 값에 3을 곱함
                .collect(Collectors.toList()); // 결과를 리스트로 수집

       return result;
    }

    // 2번 문제 1000개의 스레드를 생성하고 각 스레드에서 공유 카운터를 1씩 증가시키는 프로그램을 작성하세요. 모든 스레드가 실행 완료한 후, 최종 카운터 값이 올바르게 나오도록 동기화를 구현하세요.
    private int counter = 0;
    private synchronized void incrementCounter() {
        counter++;
    }
    @PostConstruct
    public void init() throws InterruptedException {
        Thread[] threads = new Thread[1000];

        for (int i = 0; i < threads.length; i++) {
            threads[i] = new Thread(new Runnable() {
                @Override
                public void run() {
                    incrementCounter();
                }
            });
            threads[i].start();
        }

        // 모든 스레드가 종료될 때까지 기다립니다.
        for (Thread thread : threads) {
            thread.join();
        }

        log.info("최종 카운터 값: {}", counter);
    }
    // 3번 문제 100만 개의 정수를 저장해야 하는 상황에서 ArrayList와 LinkedList 중 어느 것을 사용할지 결정하고 그 이유를 설명하세요.
    // 답 : ArrayList는 무작위 접근에 유리하고, LinkedList는 데이터의 삽입 및 삭제가 빈번할 때 유리합니다.
    /*

        100만 개의 정수를 저장해야 하는 상황에서는 ArrayList를 사용하는 것이 더 적합합니다. 그 이유는 다음과 같습니다:

        접근 시간: ArrayList는 인덱스를 통해 직접 접근이 가능한 배열 기반의 데이터 구조입니다. 이는 특정 위치의 데이터에 빠르게 접근할 수 있음을 의미하며,
        O(1)의 시간 복잡도를 가집니다. 반면, LinkedList는 각 요소가 이전 및 다음 요소의 참조를 가지고 있는 연결 리스트 기반의 데이터 구조로,
        특정 인덱스의 요소에 접근하기 위해서는 처음부터 해당 위치까지 순차적으로 탐색해야 하므로 O(n)의 시간 복잡도를 가집니다.

        메모리 사용량: LinkedList는 각 요소가 데이터와 더불어 다음 및 이전 요소의 참조(포인터)를 가지고 있어야 하기 때문에, 같은 수의 데이터를 저장할 때 ArrayList에 비해 더 많은 메모리를 사용합니다.

        데이터 추가 및 삭제: LinkedList는 특정 노드의 추가 및 삭제가 빈번하게 발생할 경우 ArrayList보다 효율적일 수 있습니다.
        이는 LinkedList가 요소의 추가 및 삭제 시에 배열의 재배치 없이 포인터만 조정하면 되기 때문입니다. 그러나 문제 상황에서는 데이터의 저장만 언급되어 있고, 추가 또는 삭제에 대한 언급이 없습니다.

        결론적으로, 100만 개의 정수를 단순히 저장하고 빠르게 접근해야 하는 경우, ArrayList가 더 우수한 성능을 보일 것입니다.
        ArrayList는 랜덤 액세스가 필요할 때 훨씬 더 효율적이며, 메모리 사용량도 LinkedList보다 효율적입니다.

    */
    // 4번 문제 람다 표현식을 사용한 Runnable 인스턴스를 생성하고 사용해보세요.
    @PostConstruct
    public void secondInit(){
        // Runnable 인터페이스의 익명 클래스 인스턴스 생성
        Runnable task = () -> {
            System.out.println("람다 표현식을 사용한 Runnable의 run 메소드 실행");
        };

        // 스레드 생성 및 시작
        Thread thread = new Thread(task);
        thread.start();
    }
    // 5번 문제 싱글톤 디자인 패턴을 Java로 구현하는 방법을 설명하고 코드 예제를 제시하세요. 따로 공부해야할듯...
//    public class Singleton {
//        // 1. 클래스 내부에 유일한 인스턴스를 생성합니다. (private & static)
//        private static Singleton instance = new Singleton();
//
//        // 2. 생성자를 private으로 선언하여 외부에서 인스턴스 생성을 방지합니다.
//        private Singleton() {}
//
//        // 3. 유일한 인스턴스에 접근하기 위한 public 메소드를 제공합니다.
//        public static Singleton getInstance() {
//            return instance;
//        }
//
//        // 이 클래스의 다른 메소드들...
//    }
    /*

        위의 코드는 가장 기본적인 싱글톤 패턴의 구현 방법을 보여줍니다. 이 방법은 멀티스레딩 환경에서도 안전하게 사용할 수 있습니다.
        왜냐하면 인스턴스가 클래스 로딩 시점에 생성되기 때문에, 별도의 동기화 처리 없이도 스레드 안전(thread-safe)을 보장합니다.

        그러나 이 방식은 클래스가 로딩될 때 인스턴스가 생성되므로, 사용되지 않을 경우에도 메모리를 차지하게 됩니다.
        이러한 문제를 해결하기 위한 다른 싱글톤 구현 방법들도 존재합니다(예: Lazy Initialization, Double-checked Locking, Initialization-on-demand holder idiom 등).

    */

    // 6번 문제 Java 메모리 모델과 관련하여, 스택(Stack)과 힙(Heap)의 차이를 설명하고, 가비지 컬렉션(Garbage Collection)의 작동 원리에 대해 간략하게 설명하세요.
    // 스택은 메서드 호출과 지역 변수에 사용되며, 힙은 객체와 인스턴스를 저장하는데 사용됩니다. 가비지 컬렉션은 힙 메모리에서 더 이상 참조되지 않는 객체를 찾아내어 메모리를 회수하는 과정입니다.

    // 7번 문제 커스텀 예외 클래스를 생성하고, try-catch 블록을 사용하여 예외를 처리하는 예제 코드를 작성하세요. 좀더 보완해보자...
    public void ex7() throws CustomException{
        try{
            commethod();
        }catch (Exception e){
            throw new CustomException();
        };
    }
    public class CustomException extends Exception{

    }
    public Integer commethod(){
        return 1;
    }

    // 8번 문제 Java에서 파일을 읽고 쓰는 두 가지 방법(FileReader/FileWriter, FileInputStream/FileOutputStream)을 비교하고 각각의 사용 예제를 작성하세요.
    public class FileReadWriteExample {
        public static void main(String[] args) {
            String inputFile = "input.txt";
            String outputFile = "output.txt";

            try (FileReader fr = new FileReader(inputFile); FileWriter fw = new FileWriter(outputFile)) {
                int c;
                while ((c = fr.read()) != -1) {
                    fw.write(c);
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    public class FileStreamExample {
        public static void main(String[] args) {
            String inputFile = "input.bin";
            String outputFile = "output.bin";

            try (FileInputStream fis = new FileInputStream(inputFile); FileOutputStream fos = new FileOutputStream(outputFile)) {
                int byteData;
                while ((byteData = fis.read()) != -1) {
                    fos.write(byteData);
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    /*

        사용 데이터 타입:
        FileReader/FileWriter: 문자 데이터를 위한 스트림. 텍스트 파일을 읽거나 쓸 때 사용합니다.
        FileInputStream/FileOutputStream: 바이트 데이터를 위한 스트림. 바이너리 파일을 읽거나 쓸 때 사용합니다.
        성능:
        문자 데이터의 경우, FileReader/FileWriter를 사용하는 것이 더 효율적입니다.
        바이너리 데이터의 경우, FileInputStream/FileOutputStream을 사용하는 것이 더 적합합니다.
        용도:
        FileReader/FileWriter: 텍스트 파일 처리
        FileInputStream/FileOutputStream: 바이너리 파일 처리 (예: 이미지, 오디오, 비디오 파일)

    */

    // 9번 문제 Java 8에서 추가된 날짜와 시간 API를 사용하여 현재 시간부터 2주 후까지의 날짜를 계산하는 코드를 작성하세요.
    public class DateCalculationExample {
        public static void main(String[] args) {
            // 현재 날짜 가져오기
            LocalDate today = LocalDate.now();
            System.out.println("현재 날짜: " + today);

            // 2주 후의 날짜 계산
            LocalDate twoWeeksLater = today.plus(2, ChronoUnit.WEEKS);
            System.out.println("2주 후 날짜: " + twoWeeksLater);
        }
    }

    // 10번 문제 Java에서 함수형 인터페이스(Functional Interface)의 개념을 설명하고, Stream API를 사용하여 리스트의 문자열 요소들 중 "Java"를 포함하고 있는 요소들만 필터링하고 대문자로 변환하는 코드를 작성하세요.

    /*

        java에서 함수형 인터페이스(Functional Interface)는 오직 하나의 추상 메소드를 가진 인터페이스입니다.
        이러한 인터페이스는 람다 표현식이나 메소드 참조를 통해 간단히 구현될 수 있으며, Java 8부터는 @FunctionalInterface 어노테이션을 사용하여 이를 명시할 수 있습니다.
        함수형 인터페이스는 Java의 함수형 프로그래밍을 가능하게 하는 핵심 요소 중 하나입니다.

    */
    @FunctionalInterface
    public interface Comparator {
        int compare(int a, int b);
    }
    public class FunctionalInterfaceExample {
        public static void main(String[] args) {
            // 람다 표현식을 사용하여 Comparator 인터페이스의 구현체를 제공
            Comparator comparator = (a, b) -> {
                if (a > b) return 1;
                else if (a == b) return 0;
                else return -1;
            };

            // comparator를 사용하여 두 정수 비교
            int result1 = comparator.compare(5, 3);
            int result2 = comparator.compare(2, 2);
            int result3 = comparator.compare(1, 4);

            // 결과 출력
            System.out.println(result1); // 1
            System.out.println(result2); // 0
            System.out.println(result3); // -1
        }
    }

    public class StreamExample {
        public static void main(String[] args) {
            List<String> list = Arrays.asList("Hello, Java", "Java 8", "Functional Interface", "Stream API", "Java Stream Example");

            List<String> filteredList = list.stream() // 스트림 생성
                    .filter(s -> s.contains("Java")) // "Java"를 포함하는 요소 필터링
                    .map(String::toUpperCase) // 모든 요소를 대문자로 변환
                    .collect(Collectors.toList()); // 결과를 리스트로 수집

            System.out.println(filteredList);
        }
    }
}
