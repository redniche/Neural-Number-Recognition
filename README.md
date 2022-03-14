<!-- @format -->

## 소개

정사각형 데이터패턴으로 이루어진 숫자를 인식하는 다층 퍼셉트론 기반 신경망입니다. 숫자가 아니어도 패턴을 유추해낼 수 있습니다.
학습과 질의를 위해 MNIST 데이터셋을 사용하는 것을 추천합니다.
**정식 릴리즈되지 않은 코드를 빌드, 사용시 어떠한 법적 책임도 지지 않음을 명시합니다.**

## 기술스택

<img src="https://img.shields.io/badge/C%23-007ACC?style=flat&logo=CSharp&logoColor=white">
<img src="https://img.shields.io/badge/.Net Framework-007ACC?style=flat&logo=DotNet&logoColor=white">

## Build Environment

- Framework: .Net Framework 4.7.1
- IDE: Visual Studio 2019
- Language: C#

---

## 데이터셋

- [MNIST in CSV 링크](https://pjreddie.com/projects/mnist-in-csv/)의 train set, test set 같은 형식의 csv 파일
- 파일명이 정답인 이미지 파일(.png, .jpg, .jpeg 형식)
  - 예시) 1.png, 2.jpg

## 주요 사용법

![주요 사용법1](/image/신경망초기화.gif)

### `Help`

도움말을 보여줍니다.

### `Create`

신경망을 생성합니다.  
본 예제에서는 은닉층이 200노드고 출력이 10개(숫자 수)인 이미지(가로세로 28px)를 인식할 수 있는 신경망 객체를 생성합니다.

### `ShowStatus`

신경망의 상태를 확인합니다. 대략적인 신경망 객체의 정보를 볼 수 있습니다.

### `CsvTrain`

![주요 사용법1](/image/Csv학습.gif)

csv파일로 신경망을 학습시킬 수 있습니다.  
예시 학습용 csv파일: [링크](https://pjreddie.com/media/files/mnist_train.csv)

### `CsvQuery`

![주요 사용법1](/image/Csv질의.gif)

학습된 신경망 객체를 csv파일로 테스팅할 수 있습니다.  
csv파일로 신경망에 질의하는 명령입니다.  
대용량 테스팅을 할 때 활용할 수 있으며 쿼리의 세부 결과는 로그로 남습니다.  
예시 테스팅용 csv파일: [링크](https://pjreddie.com/media/files/mnist_test.csv)

### `ImageQuery`

![주요 사용법1](/image/이미지쿼리.gif)

학습된 신경망 객체를 이미지 파일로 테스팅할 수 있습니다.  
png, jpg, jpeg 파일을 지원합니다. 여러 파일을 한 번에 테스팅 할 수 있습니다.
쿼리의 결과가 콘솔창에 그대로 나타납니다.

### `Save`

![주요 사용법1](/image/신경망저장.gif)

신경망 객체를 저장할 수 있습니다.

### `Open`

![주요 사용법1](/image/신경망불러오기.gif)

저장된 신경망 객체를 로드합니다.
예제 테스트 데이터에 대해 대략 97%의 정확도를 보여주는 신경망 객체 파일을 제공합니다.  
[신경망 객체 파일.neu](/trained_file/my_train_data.neu)

## 과거 코드

제 블로그입니다.
[블로그 링크](https://blog.naver.com/redniche/221401615403)
