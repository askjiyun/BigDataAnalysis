# FFT와 시각화
import matplotlib.pyplot as plt
import numpy as np
import scipy

T = 0.001 # Sample 간격 (Sample Interval)
f = 300 # 첫번째 주파수 Frequency 1
f2 = 140 # 두번째 주파수 Frequency 2
x = np.arange(0,100,T)
#y = np.arange(0,200,2*T)

# 신호 생성
y = np.sin(2*np.pi*f*x) + 0.3*np.sin(2*np.pi*f2*x)
# 주파수 축 정의
x1 = np.linspace(-1/(2*T),1/(2*T),1024)
# FFT 계산
y1 = scipy.fft.fft(y,1024)
# 중앙 주파수 0hz가 중심에 오도록 정렬
y1 = scipy.fft.fftshift(y1)

plt.plot(x1,abs(y1))
plt.xlabel("Frequency")
plt.ylabel("Value")
plt.title("Example 1")
plt.grid()
plt.show()