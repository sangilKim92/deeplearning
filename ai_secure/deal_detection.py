import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv('./datasets/payment_fraud.csv')

df = pd.get_dummies(df, columns=['paymentMethod'])
# paymentMethod 속성은 범주형으로 값이 연속적인 데이터가 아니다. 이를 신경망 계산하기 위해선 원핫 인코딩을 사용해야 하는데 원핫 인코딩으로 바꾸어주는 함수이다.

X_train, X_test, Y_train, Y_test = train_test_split(df.drop('label',axis=1),df['label'], test_size=0.2,random_state=17)

clf = LogisticRegression().fit(X_train,Y_train)

y_pred = clf.predict(X_test)

print(accuracy_score(y_pred,Y_test))

print(confusion_matrix(Y_test,y_pred))

#confusion_matrix를 이용해서 정확도뿐만 아니라 정밀도 재현율도 계산해 낸다.
#대부분의 머신러닝 모델이 그렇지만 로지스틱 회귀도 각각의 속성이 독립적이어야 한다. 모든 속성은 수치값이어야 한다.
#만약 범주형을 사용하고 싶다면 의사결정 트리와 같은 모델을 사용하면 된다.
#의사결정 트리는 범주형, 실수형에 복합적으로 사용할 수 있지만, 정확도와 강인성이 떨어진다. 작은 변화에 영향을 크게 받을 수 있다.
#결과에 대한 설명하기에 좋은 모델이다.

#랜덤포레스트는 의사결정 트리를 여러 개 앙상블 한 것으로 트리각각을 학습시킨후 전체를 투표형태로 결과를 예측하는 것이다.

#어떤 모델을 사용하면 좋을지 정할때는 연산 복잡도, 산술 복잡도(선형데이터인지 비선형 데이터인지), 설명 가능성을 고려하면 좋다.
# 연산 복잡도 같은경우 신경망, SVM은 오래 걸릴 수 있고
#산술 복잡도는 의사결정 포레스트, 커널 svm, 신경망등은 비선형이기에 사용하면 좋다.
