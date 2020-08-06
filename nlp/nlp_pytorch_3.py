#딥러닝의 역사
#초기에는 규칙기반으로 시소러스같은 방식을 이용함
#이런 방식이 통계기반 기법으로 옮겨졌는데, CBOW, Bow, RNN, LSTM같은 기술을 뜻한다.
#한 층에 모든 통계에 기반한 결과를 가져오는 방식이다.

#2015년 seq2seq 방식이 소개되며 어텐션까지 도입되고 나서야 음성 인식분야에도 딥러닝으로 바뀌었다.


#단어 학습을 어렵게 하는 요소들 단어의 모호성(다의어), 다양한 표현, 불연속적인 데이터의 벡터화

import torch
import torch.nn as nn
import random
import numpy as np

x = torch.Tensor([[1,2],[3,4]])
#numpy.array() 와 비슷한 용도, 그래프와 경사도가 추가됨
#torch.Tensor == torch.FloatTensor
#x= torch.from_numpy(np.array([[1,2],[3,4]]))
#print(x)

#x= np.array([[1,2],[3,4]])
#print(x)

#autograd => 값을 앞으로 피드포워드하며 계산만해도, backward() 호출 한번에 역전파 알고리즘을 수행한다.
x = torch.Tensor(2,2)
y= torch.Tensor(2,2)
y.requires_grad_(True)
# 동적 그래프 자동으로 그래프가 생성된다. 최대 장점 그래프의 기울기 변화를 파악하기 좋다.
z = (x +y ) + torch.Tensor(2,2)


def liner(x,W,b):
    y = torch.mm(x,W) + b

    return y
x = torch.Tensor(16,10)
W = torch.Tensor(10,5)
b = torch.Tensor(5)
#print(x)
#print(W)
#print(b)
y = liner(x,W,b)
#print(y)

class MYLinear(nn.Module): ##nn.Module을 상속받은 클래스는 내부에 nn.module을 상속한 클래스 객체를 소유할 수 있다.
    def __init__(self, input_size, output_size):
        super().__init__()

        self.W = torch.Tensor(input_size,output_size)
        self.b = torch.Tensor(output_size)

    def forward(self, x):
        y = torch.mm(x,self.W)+ self.b #torch.mm은 피드 포워드함수이다.

        return y #이렇게 오버라이딩해서 피드포워딩하면 자동으로 오차역전파법을 해준다.

X = torch.Tensor(16,10)
#print("X: ",X)
linear = MYLinear(10,5)
y=linear.forward(X)
#print(y)
# nn.Parameters() 함수는 모듈 내에 선언된 학습이 필요한 파라미터들을 반환하는 이터레이터이다.
print("학습이 필요하다고 설정안한 경우: ",[p.size() for p in linear.parameters()]) #생각하기에는 W와 b 두 개가 학습이 필요할 것같은데 나오지 않는다.




#학습이 필요하다고 판단한 것은 따로 지정해 주어야 한다.
class MYLinear2(nn.Module): ##nn.Module을 상속받은 클래스는 내부에 nn.module을 상속한 클래스 객체를 소유할 수 있다.
    def __init__(self, input_size, output_size):
        super().__init__()

        self.W = nn.Parameter(torch.Tensor(input_size,output_size), requires_grad=True)
        self.b = nn.Parameter(torch.Tensor(output_size), requires_grad=True)
        #print("W: ",self.W)
        #print("b: ",self.b)

        #위의 식을 간단하게 구현하려면 nn.Linear()를 사용하면 된다.
        #nn.module을 상속받았기에 class내부에 nn 변수를 지정할 수 있다.
        #self.linear = nn.Linear(input_size,output_size)

    def forward(self, x):
        y = torch.mm(x,self.W)+ self.b #torch.mm은 피드 포워드함수이다.

        return y #이렇게 오버라이딩해서 피드포워딩하면 자동으로 오차역전파법을 해준다.

x2 = torch.Tensor(16,10)
linear2 =MYLinear2(10,5)
linear2.forward(x2)
print("학습 변수를 설정:",[x.size() for x in linear2.parameters()])


class MYLinear3(nn.Module): ##nn.Module을 상속받은 클래스는 내부에 nn.module을 상속한 클래스 객체를 소유할 수 있다.
    def __init__(self, input_size, output_size):
        super(MYLinear3,self).__init__()

        #nn.module을 상속받았기에 class내부에 nn 변수를 지정할 수 있다.
        self.linear = nn.Linear(input_size,output_size)

    def forward(self, x):
        y =self.linear(x)

        return y #이렇게 오버라이딩해서 피드포워딩하면 자동으로 오차역전파법을 해준다.
#이제 오차역전파법을 진행해보자.
target= 100
#정답 레이블 값
print("\n")
linear3 = MYLinear3(10,5)
y=linear3(x2) # == linear3.forward(x2) 왜 같은지 모르겠음
print(y)
loss = (target - y.sum())**2
#print(loss)
#print(loss.backward()) #기울기 계산
#print(linear3.eval())
#print(linear3.train())

class MyModel(nn.Module):
    def __init__(self,input_size,output_size):
        super(MyModel, self).__init__()

        self.linear = nn.Linear(input_size,output_size)

    def forward(self,X):
        y = self.linear(X)

        return y

def ground_truth(x):
    return 3*x[:,0]+x[:,1]-2*x[:,2]

def train(model, x, y,optim):
    optim.zero_grad() # 가중치 초기화

    y_hat = model(x) #feed-forward

    loss = ((y-y_hat)**2).sum()/x.size(0) # 손실함수 계산 직접 구현

    loss.backward()  #backward는 자동으로 해줌

    return loss.data

batch_size = 1
n_epochs = 1000
n_iter = 10000

model = MyModel(3,1)
optim = torch.optim.SGD(model.parameters(), lr= 1e-4, momentum=0.1)

print(model)

for epoch in range(n_epochs):
    avg_loss = 0

    for i in range(n_iter):
        x = torch.rand(batch_size, 3)
        y = ground_truth(x.data)

        loss = train(model, x, y,optim)

        avg_loss += loss
        avg_loss = avg_loss / n_iter

    x_valid = torch.Tensor([[.3,.2,.1]])
    y_valid = ground_truth(x_valid.data)

    model.eval()
    y_hat = model(x_valid)
    model.train()

    print(avg_loss,y_valid.data[0],y_hat.data[0,0])

    if(avg_loss <0.001):
        break

#nn.Module 클래스를 상속받아 모델 아키텍처 클래스 선언
#해당 클래스 객체 생성
#SGD나 Adam 등의 옵티마이저를 생성하고, 생성한 모델의 파라미터를 최적화 대상으로 등록
#데이터로 미니배치를 구성하여 피드포워드 연산 그래프 생성
# 손실 함수를 통해 최종 결괏값과 손실값 계산
# 손실에 대해서 backward() 호출 -> 텐서들의 기울기가 채워짐
# 3번의 옵티마이저에서 step() 을 호출하여 경사하강법 1 스텝 수행
# 4번으로 돌아가 수렴 조건이 만족할때까지 반복