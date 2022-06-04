# Meal Detection and Recording Application

## 프로젝트 소개

객체 탐지를 이용한 자동 식단 기록 애플리케이션

**Youtube**: https://www.youtube.com/watch?v=97yLDvm1MZ8 

![image-20211205183522853](https://user-images.githubusercontent.com/70505378/144741326-b6a0e257-dfaf-4ecb-a17f-da7f1932c9b9.png)

<br>

## 개발 내용

### Deep Learning

**1. 데이터 다운로드**

![image](https://user-images.githubusercontent.com/70505378/171992077-6f64ce09-fc4f-44d3-8f2e-ee1c69af5443.png) 

출처: https://aihub.or.kr/aidata/30747 

<br>

상위 400 종의 한국인 외식 데이터셋 다운로드

전체 용량이 약 2.4TB 가량 되기 때문에 예산으로 원본 데이터셋 + 변환된 데이터셋을 저장할 수 있는 약 5TB의 외장 하드를 구매하고 데이터셋을 다운로드



![image](https://user-images.githubusercontent.com/70505378/171992084-b523a54e-76bb-44b6-b609-f13652eb09ee.png)  



<br>

**2. TFRecord 변환**

빠른 모델 학습을 위해 이미지-어노테이션 데이터 쌍을 TFRecrod 포맷의 데이터셋으로 변환. 

![image](https://user-images.githubusercontent.com/70505378/171992087-38096719-228e-4a54-bf95-0a5850e2df16.png)  



<br>

xml 파일(어노테이션 파일)을 파싱하는 코드. 각 어노테이션 파일에서 TFRecord 변환에서 필요한 정보들을 추출하여 딕셔너리 형태로 저장 후 반환. 



![image](https://user-images.githubusercontent.com/70505378/171992091-9651120e-5dab-4dfe-a1c6-74b7d70cb57f.png)  

![image](https://user-images.githubusercontent.com/70505378/171992095-ad7de530-cacc-472e-bba3-49a0623d851e.png)  



<br>

앞에서 파싱한 정보들을 가지고 tf.train.Example 객체 생성. TFRecord 데이터셋 변환을 위해서는 객체 정보에 해당하는 feature들을 Example 객체로 변환해야 한다. 



![image](https://user-images.githubusercontent.com/70505378/171992099-f4fd01b0-6d67-41f4-a823-37b1efd65656.png)  



<br>

앞에서 생성한 Example 객체를 tfrecord 파일에 write한다. 

이 과정을 모든 어노테이션-이미지 쌍에 대해 반복하면 전체 데이터에 대한 TFRecord 데이터셋을 생성할 수 있다. 

TFRecord 데이터셋은 객체 탐지 모델이 필요로 하는 어노테이션-이미지 데이터 쌍을 한 번에 저장하여 데이터를 읽는 속도를 빠르게 하고, 추가로 데이터셋의 용량도 약 70% 정도로 축소시킬 수 있다. 



<br>

**3. 모델 선택**

객체 탐지 모델에는 여러 모델이 있다. 

크게 영역 제안(Region proposal)과 분류(Classification)을 한 번에 수행하는 1-stage detector와 단계적으로 수행하는 2-stage detector가 있다. 

당연히 1-stage detector는 빠른 속도를 보이지만 작은 객체를 탐지해내는 성능이 2-stage detector보다는 떨어진다. 

하지만 어느 모델이든 완벽한 탐지를 해낼 수는 없고, 어플리케이션이라는 특성을 이용해 탐지해내지 못 한 객체에 대해서는 사용자 커스터마이징에 맡기면 된다. 

따라서 빠른 추론 속도를 보장하는 1-stage detector 중 대표적인 모델인 YOLO 모델을 사용한다. 

![image](https://user-images.githubusercontent.com/70505378/171992102-1f95bd76-ed72-494e-9a98-3b3d6af3a8a6.png)  



<br>

Github에는 Tensorflow, Pytorch 등으로 구현된 YOLOv2~YOLOv5 의 모델들이 많이 있고, 이 중 성능을 보장하면서도 사용하기에 편리한 YOLOv3를 선택한다. 

또한 추가적으로 이 프로젝트는 온-디바이스 머신러닝을 사용하는 프로젝트로, 뒤에서 모델을 tflite 포맷의 파일로 변환할 것이기 때문에 Tensorflow 프레임워크로 구현된 모델을 사용하는 것이 편리하다. 

여러 조건을 고려하여 다음 리포지토리의 YOLO 모델을 사용한다. 

![image](https://user-images.githubusercontent.com/70505378/171992105-74b84e81-b7b3-40df-b7b0-160b9f1fbc4f.png)  

출처: https://github.com/wowo0709/yolov3-tf2 



<br>

위 리포지토리는 기존에 있던 zzh8829 유저가 만든 리포지토리를 fork한 개인 리포지토리이다. 이 리포지토리에서 필요한 코드들만을 참빛 프로젝트용 리포지토리에 복사한다. 



위 이미지를 보면 여러 사물에 대한 탐지 및 분류를 수행하는 것을 볼 수 있는데, 이 학습된 모델을 앞에서 변환한 한국인 외식 TFRecord 데이터셋으로 학습하여 프로젝트 목적에 맞는 모델로 변환한다. 



<br>

**4. 모델 학습**

사용할 모델을 결정하고 데이터셋 준비까지 끝마쳤으면 모델을 학습시킨다. 

 ![image](https://user-images.githubusercontent.com/70505378/171992112-032674cc-3e87-43a2-b848-adc0068ff8bd.png)  



모델에 여러 옵션들을 줌으로써 커스텀하게 모델을 학습시킬 수 있다. 



모델을 학습시키기 위해 사용한 옵션은 아래와 같다. 

![image](https://user-images.githubusercontent.com/70505378/171992117-f4c2129d-50bb-4510-aeb6-ce2fc5ab517d.png)  



<br>

모델의 학습에 필요한 epoch와 batch size, learning rate 등을 고려하여 모델을 학습시킨다. 

이렇게 학습된 모델은 .tf 파일 포맷으로 저장된다. 

![image](https://user-images.githubusercontent.com/70505378/171992121-df68cb3a-a217-41d1-9edf-103edf7381e0.png)  



<br>

**5. TFLite 변환**

.tf 포맷의 파일로 저장된 모델 파일을 안드로이드 프로젝트에 포함시키기 위해서는 .tflite 포맷의 파일로 변환해야 한다. 

TFLite 파일로 변환할 때에도 마찬가지로 여러 옵션을 줄 수 있는데, 여기서는 양자화에 대한 것만 다루겠다. 

모델의 가중치는 float32 의 precision을 가지도록 저장된다. 가중치의 precision을 낮춤으로써 모델 파일의 용량을 크게 줄이면서도 성능은 크게 뒤떨어지지 않도록 할 수 있다. 

이러한 양자화 기법을 ‘학습 후 양자화’라고 하며, 다른 여러 양자화 기법들도 존재하지만 그 시간과 복잡도에 비해 효과가 크지 않아 학습 후 양자화 기법을 많이 사용한다. 

![image](https://user-images.githubusercontent.com/70505378/171992126-9f79d929-32e7-48ef-a898-12df88684935.png)  


<br>

학습 후 양자화 기법에는 dynamic range 양자화, float 16 양자화, Integer 양자화 등이 있다. 

이 중 가장 용량 축소 효과가 큰 dynamic range 양자화를 사용한다. 

dynamic range 양자화를 사용하면 모델의 용량을 기존 모델 용량의 약 1/4 수준으로 줄일 수 있다. 

![image](https://user-images.githubusercontent.com/70505378/171992129-175c6663-76c1-40e7-92a3-b02d117b1936.png)  





<br>

**6. tflite 파일을 안드로이드 프로젝트에 포함**

이렇게 변환된 tflite 모델 파일을 안드로이드 프로젝트의 assets 폴더에 포함시키면 안드로이드 프로젝트에서 모델을 호출하여 추론을 수행할 수 있다. 







<br>

### Android

**0. TFLiteSupportLibrary 의존성 추가**

안드로이드에서 TFLite 파일을 통한 모델 추론을 수행하고 싶다면 tflitesupportlibrary 외부 라이브러리에 대한 의존성을 추가하여 편리하게 수행할 수 있다. 

이 라이브러리는 TFLite에서 사용할 수 있는 연산들에 대한 접근을 제공해주는데, 아직 tensorflow의 모든 연산에 대한 접근성을 제공해주고 있지는 않다. 

따라서 tflitesupportlibrary에서 제공하고 있지 않은 tensorflow 연산에 대해서는 따로 외부 라이브러리 의존성 추가를 통해 해결해야 한다. 

![image](https://user-images.githubusercontent.com/70505378/171992201-6c322ba5-a74f-49d1-8a84-ec47a33d0139.png)  

![image](https://user-images.githubusercontent.com/70505378/171992205-bf3a3caf-6a12-494b-acb0-9f68eef77387.png)  



위와 같이 의존성을 추가하면 모델 추론 과정에서 사용하는 tensorflow 연산들을 모두 사용할 수 있다. 



<br>

**1. 권한 명세 및 카메라/갤러리 접근**

어플리케이션에서 이미지를 가져오기 위해서는 사용자 카메라/갤러리로의 접근이 필요하고, 접근을 위해서는 사용자에게 권한을 명세받아야 한다. 

![image](https://user-images.githubusercontent.com/70505378/171992206-3b94a0de-2322-4a75-9338-00b4bf98c482.png)  



권한을 요청하는 BaseActivity를 클래스로 생성해놓으면 MainActivity에서 BaseActivity를 상속받음으로써 필요한 권한들을 요청할 수 있다. 

![image](https://user-images.githubusercontent.com/70505378/171992207-09c48ea7-b5d7-4912-bf1a-d2e9a5cb3944.png)  

![image](https://user-images.githubusercontent.com/70505378/171992209-79ec0455-e4d5-4ff7-8013-a7a2c2c012d5.png)  



추상 메서드로 구현된 permissionGranted와 permissionDenied는 BaseActivity 클래스를 상속받는 MainActivity에서 구현한다. 


<br>

MainActivity에서는 권한 요청에 대한 추상 메서드들을 구현하고, 카메라/갤러리 요청 코드 및 접근 코드를 작성한다. 

(카메라 호출 메서드)

![image](https://user-images.githubusercontent.com/70505378/171992211-7a73e956-8888-406d-9f5c-cdcccca38eb8.png)  



(갤러리 호출 메서드)

![image](https://user-images.githubusercontent.com/70505378/171992213-85113af2-cdc3-4368-9503-763dad7b5850.png)  



<br>

**2. 선택한 이미지를 모델에 전달**

사용자가 카메라/갤러리를 통해 이미지를 촬영/선택하면 객체 탐지 모델을 호출한다. 

MainActivity에는 모델 호출을 위한 callFoodDetector 메서드가 있다. 

![image](https://user-images.githubusercontent.com/70505378/171992216-43fe3415-96c6-4d75-890c-f4336fe9bc0f.png)  



위 메서드를 호출하면 모델이 생성되고 입력 이미지를 전달받게 된다. 


<br>

**3. 입력 이미지 전처리**

모델이 이미지를 전달받으면 먼저 이미지 전처리를 수행한다. 

전처리에는 resizing, interpolation, normalizaing, converting pixel range 등의 과정이 포함되며, 코드로는 다음과 같다. 

![image](https://user-images.githubusercontent.com/70505378/171992220-d12e7982-e6c0-4ff4-a540-63b9b31ccc03.png)  



![image](https://user-images.githubusercontent.com/70505378/171992221-19d4a97f-2aa3-4e6d-bcab-9975eb8e40fe.png)  



<br>

**4. 모델 추론**

입력 이미지 전처리가 완료되면 모델에 이미지를 전달하여 추론을 수행한다. 

텐서플로 라이트를 이용해 모델 추론을 수행할 때는 모델 추론 값을 반환 받을 텐서버퍼를 먼저 할당해주어야 한다. 



<br>

YOLO 모델은 입력 이미지에 대해 탐지박스 좌표([xmin, ymin, xmax, ymax], shape=(1,100,4)), 클래스에 속할 확률(shape=(1,100)), 속하는 클래스(shape=(1,100)), 총 탐지된 객체 수(shape=(1,))를 추론 결과로 반환한다. 

따라서 이 추론값들을 반환받기 위한 텐서버퍼를 먼저 할당한다. 

![image](https://user-images.githubusercontent.com/70505378/171992225-ece7d48e-4b1e-416f-9e69-ab937c145252.png)  



<br>

그리고 나서 입력 이미지와 텐서 버퍼를 함께 모델에 전달하여 추론을 수행한다. 

기본적으로 tflite 모델 호출 시에는 run이라는 메서드를 사용하는데, run 메서드는 모델의 반환 값이 하나일 때만 사용할 수 있다. 

YOLO 모델의 경우 반환 값이 4개이기 때문에 여러 개의 반환 값을 받을 수 있는 runForMultipleInputsOutputs() 메서드를 사용한다. 

![image](https://user-images.githubusercontent.com/70505378/171992229-44ed7412-20d9-4093-802a-fca4b6ef953a.png)  



위 메서드를 호출하고 나면 앞에서 할당한 텐서버퍼에 모델 추론 값이 담기게 된다. 



<br>

**5. 모델 추론 값 후처리**

모델의 반환 값을 받고 나면 이에 대한 후처리가 필요하다. 

후처리 과정에서는 추론된 클래스의 인덱스에 맞게 클래스명을 매핑해주고, 탐지박스 리스트와 음식이름+확률 리스트만을 추출하여 callFoodDetector를 호출한 MainActivity에게 반환한다. 

![image](https://user-images.githubusercontent.com/70505378/171992232-0d01d449-0bea-4098-96e0-e8fc222df24e.png)  


<br>

후처리된 모델 추론 값을 전달받은 MainActivity에서는 이 추론 값을 사용자가 확인할 수 있도록 출력해주는 과정을 시작한다. 

이 과정은 PhotoFragment라는 프래그먼트에서 수행된다. 

![image](https://user-images.githubusercontent.com/70505378/171992235-7f271b91-1d63-45d3-8a13-5d119e75d626.png)  


setPhotoFragment메서드에서 전달받은 모델 추론 값을 전달해준다. 



<br>

**6. 모델 추론 값 출력**

PhotoFragment에서는 전달 받은 모델 추론 값을 사용자가 확인할 수 있도록 출력한다. 

출력을 위해서 bitmap과 recycler view를 사용하며, 이에 대한 코드는 너무 많으니 생략한다. 



<br>

먼저 이미지 뷰에 사용자가 선택한 이미지를 출력하고, 전달받은 탐지박스 좌표 리스트를 이용하여 탐지된 음식들에 박스를 친다. 추가로 박스 위에 음식 이름과 확률을 출력하여 사용자가 구별할 수 있도록 한다. 

이를 위해 안드로이드의 Canvas 객체를 사용하고 코드는 아래와 같다. 

![image](https://user-images.githubusercontent.com/70505378/171992236-027a554a-68ef-4884-8cb6-2fee40c9b6a9.png)  



![image](https://user-images.githubusercontent.com/70505378/171992239-6d4ea8e9-d0d6-41f0-84ab-414ad89ea7fa.png)  



<br>

다음으로 모델이 탐지한 음식이름+확률 리스트를 recycler view에 출력한다. 

여기서 사용자는 모델의 탐지 결과를 커스터마이징 할 수 있다. 

![image](https://user-images.githubusercontent.com/70505378/171992241-c60fe3d3-3abd-47c8-9929-f0044c92f01a.png)  



![image](https://user-images.githubusercontent.com/70505378/171992245-885f1e11-66a9-44d7-8a11-f6689042897d.png)  



코드의 대략적인 흐름은 위와 같다. 

또한 출력을 위한 레이아웃 파일을 생성한다. 

![image](https://user-images.githubusercontent.com/70505378/171992264-91a6bd05-940a-44a6-be86-b43814702f34.png)





사용자가 추론된 아이템들을 확인하고 추가/변경/삭제가 가능하다. 

결과 화면은 다음과 같다.

![image](https://user-images.githubusercontent.com/70505378/171992270-adb19b47-b699-4728-891f-fea8c0626c79.png)


<br>

**7. 사용자 식단 저장**

사용자가 위 화면에서 [저장]을 선택하면 식단 기록을 저장해야 한다. 

이를 위해서 Android의 경량화 데이터베이스인 Sqlite를 사용한다. 

![image](https://user-images.githubusercontent.com/70505378/171992274-f36b06f8-bb5c-4090-bbce-2ae52e520d3f.png)  



위 코드는 식단 리스트를 저장하기 위한 클래스이고, DB 테이블을 생성하는 코드이다. 

코드의 양이 많아 보일 수는 없지만 아이템 추가/수정/삭제 등을 위한 메서드가 추가로 작성되어 있다. 



<br>

**8. 사용자 식단 리스트 출력**

이제 DB에 저장된 식단 리스트를 불러와 ListFragment라는 프래그먼트에 recycler view로 출력한다. 

![image](https://user-images.githubusercontent.com/70505378/171992278-3e826575-1728-461a-baa8-034091c47426.png)  



<br>

아이템 출력을 위해서는 마찬가지로 레이아웃이 필요하고, 다음과 같이 생성한다. 

  

![image](https://user-images.githubusercontent.com/70505378/171992290-1e933f69-fb49-4b8b-bbc4-8d45aafd2d56.png)

또한 위를 보면 [즐겨찾기] 탭이 있는데, 사용자가 즐겨찾기 등록한 식단들만을 따로 관리할 수 있도록 하기 위해 이에 해당하는 리스너를 달아준다. 

![image](https://user-images.githubusercontent.com/70505378/171992292-67e4b739-69a3-4489-b617-6540ead9d72c.png)  


<br>

결과 화면은 다음과 같다.

![image](https://user-images.githubusercontent.com/70505378/171992293-745f4b68-9e83-44f2-beef-127137e304b3.png)  












<br>







