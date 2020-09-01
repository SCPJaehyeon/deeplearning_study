# 딥러닝에 필요한 케라스 함수 호출
from keras.models import load_model
from keras.utils import np_utils

# 필요 라이브러리 호출
import numpy as np

# csv 파일을 읽어 ','기준으로 나눠 Dataset에 불러오기
Dataset = np.loadtxt("test.csv", delimiter=",")

# 환자 정보는 0-16번(17개)까지이므로 해당 부분까지 X에 담기
X = Dataset[:, 0:17]
# 수술 후 결과 정보인 예측값 변수 초기화
Y = []

# 모델 불러오기
model = load_model('Predict_Model.h5')

# X 값에 대한 predict_classes 함수 실행결과 Y에 저장
Y = model.predict_classes(X)

print('Predict : ', Y)