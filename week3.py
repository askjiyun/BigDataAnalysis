import numpy as np
list1 = [[0, 2, 5, 7],[np.nan,2,3,4]]
arr1 = np.array(list1)
print(type(list1))
print(type(arr1))
print(arr1.dtype)
print(arr1.shape)
print(arr1.ndim)
arr1 = arr1.T
print(arr1.shape)
print(arr1)

# zero 1/2 dim array
arr = np.zeros(5)
print(type(arr))
print(arr.shape)
print(arr.dtype)

# empty array
arr = np.empty((2,2))
print(arr)
arr = np.empty((2,2), dtype=int)
print(arr)

# eye array - 주대각선 행렬
arr = np.eye(3, dtype=int)
print(arr)
arr = np.eye(3, k=1)
print(arr)
arr1_2 = 2*np.eye(4)
print(arr1_2)

# 배열을 1로 초기화
arr1_1 = 2*np.ones((4,1), dtype=int)
print(arr1_1)

#2~5까지 4개의 요소 생성, 시작점부터 끝점까지 일정한 간격으로 배열 생성
#endpoint = False - 마지막 값은 포함하지 않음
arr2 = np.linspace(2,5,4, endpoint=False)
print(arr2)
# 1 ~ 20까지 1의 간격으로 배열 생성
arr2 = np.arange(1,21,1)
print(arr2)
# 배열의 형태 변환
# -1 : numpy가 자동으로 열의 개수 계산
# 10x2 크기의 배열로 변경
arr2 = arr2.reshape(10,-1)
print(arr2)
# 다시 1차원으로 변경
arr2 = arr2.reshape(-1)
print(arr2)

# 3차원 배열 생성
# reshape(dim1, dim2, dim3)
# 2개의 블록, 각 블록에 3개의 행, 각 행에 4개의 요소
arr = np.arange(24).reshape(2,3,4)
print(arr)

arr = np.array([[[0,  1,  2,  3],
                         [4,  5,  6,  7],
                         [8,  9, 10, 11]],
                        [[12, 13, 14, 15],
                         [16, 17, 18, 19],
                         [20, 21, 22, 23]]])
print(arr.ndim) #3 - 배열의 차원 수
print(arr.shape) #(2,3,4) - 배열 모양
print(arr.size) # 24 - 배열 전체 요소 개수 반환
print(arr.dtype) #int64 - 데이터 타입
print(arr.itemsize) #8 - 각 요소가 차지하는 바이트 수
print(arr.strides) #(96,32,8) - 각 차원에서의 바이트 이동크기

# hello - 64bit 부동소수점으로 정의
# world - 10글자의 문자열로 지정
n_1_d = np.dtype([('hello', 'f8'), ('world', 'S10')])
# 크기가 3인 배열 생성, dtype은 위에 설정한 데이터 타입 따름
# 각각은 0.0과 빈문자열로 초기화
arr3 = np.zeros(3,dtype=n_1_d)
print(arr3)
print(arr3.dtype)
# hello 필드에 대해 모든 요소를 1000으로 설정
arr3['hello'] = 1000
print(arr3)

# 32비트 부동소수점 형식으로 5.0저장
a = np.float32(5.0)
print(a.dtype)
# 정수형 데이터 지정 & 리스트를 정수형 배열로 변환
b = np.int_([1, 2, 3])
print(b.dtype)
# 0-4까지 정수 배열 생성
c = np.arange(5, dtype=np.uint16)
print(c.dtype)
# 구조화된 배열 생성
# name : 최대 10글자 유니코드 문자열
# 두번째 필드 : 정수형 i4 빈문자열로 정의
# weight : 32비트 부동소수점
arr = np.array([('jin', 25, 67), ('suho', 18, 77)],
               dtype=[('name', 'U10'), ('', 'i4'), ('weight', 'f4')])
print(arr)
# 배열의 첫번째 값 수정
arr[0] = ("joon", "18", "72")
print(arr)
# 배열의 두번째 값 수정
arr[1] = ("soohee", "21", "45")
print(arr.dtype)
# 두번째 필드를 가리키며 각 요소의 값을 변경
arr["f1"] = [100,23] #[('joon', 100, 72.) ('soohee',  23, 45.)]
print(arr)

# 구조화된 배열과 다양한 정보 출력
arr_d = arr.dtype # arr 데이터타입 지정
print(arr_d.names)  #dtype 내에 정의된 필드이름 반환
print(arr_d.fields) # 각 필드에 대한 정보를 담은 딕셔너리 반환

# 필드 offset 출력
# offset : 메모리 내에서 각 필드가 시작되는 위치를 의미
# itemsize : 배열의 각 요소가 차지하는 바이트 크기
def print_offsets(d):
    print('offsets:', [d.fields[name][1] for name in d.names])
    print('itemsize:', d.itemsize)

print_offsets(np.dtype('u1,u1,i4,u1,i8,u2'))
d = np.dtype('u1,u1,i4,u1,i8,u2')
print(d)
print(d.itemsize) #각 요소의 크기
print(d.fields)    # 필드이름과 필드의 데이터타입 및 오프셋이 딕셔너리 형태로 저장된것
print(d.names)  # 필드의 이름들
print(d.fields['f0'])   # 필드 f0의 데이터타입과 오프셋
print(d.fields['f0'][1])    #필드 f0의 오프셋
# 메모리 정렬 강제
print_offsets(np.dtype('u1,u1,i4,u1,i8,u2', align=True))
#
print("-----------Structured Array & DataType----------------")
# 구조화된 배열 수정
a = np.array([(1, 2, 3), (4, 5, 6)], dtype='i8, f4, f8')
a[1] = (7, 8, 9)
print(a)

# 구조화된 배열에서 다양한 데이터타입 사용
# 64비트 , 32비트 부동소수점, 부울 값, 1글자 문자열
a = np.zeros(2, dtype='i8, f4, ?, S1')
print(a)

# 배열 전체에 값을 할당
a[:] = 7 # 배열 a의 모든 요소에 7 할당
print(a)
a[:] = np.arange(2) # 배열 a의 모든 요소에 [0,1]로 할당
print(a)

# 모든 값이 0으로 채워진 구조화된 배열
a = np.zeros(3, dtype=[('a', 'i8'), ('b', 'f4'), ('c', 'S3')])
# 숫자 필드는 1, 나머지 필드는 기본값으로 초기화된 배열
b = np.ones(3, dtype=[('x', 'f4'), ('y', 'S3'), ('z', 'O')])
print(a)
print(b)

# 배열 간 데이터 복사
b[:] = a #배열b에 배열 a의 모든 값을 복사
print(a)
print(b)

# 다차원 배열 생성
arr1 = np.arange(10) # 0-9까지 연속된 숫자 1차원 배열
arr2 = np.arange(9).reshape(3,3) # 0-8까지 3x3 배열
arr3 = np.reshape(np.arange(24), (2,3,4)) # 0-23까지 2x3x4 3차원의 배열


print("--------Slicing--------")
print(arr1[1]) # 배열에서 두번째 요소 (인덱스 1에 해당하는 값 반환)
# 첫번째 요소부터 특정 범위까지 선택
print(arr1[:6])
print(arr1[0:5])
# step을 지정하여 일정 간격으로 요소 선택
print(arr1[::2]) # 2씩 증가
print(arr1[1::2])   # 두번재 요소부터 2씩 증가
#start : end : step 형식
print(arr1[1:7:2])

# 인덱스 뒤에서부터 추출
print(arr1[-3:9]) # 배열 뒤에서 세번째 요소
print(arr1[:-3]) # 시작부터 끝에서 세번째 직전 요소까지
print(arr1[-3:2:-1]) # 배열 뒤에서 세번째 요소에서 시작해 인덱스 2까지 역순으로
print(arr1[5:2]) #시작인덱스 값이 끝 인덱스 값보다 크면 빈 배열 반환

print(arr1[5:]) # 인덱스 5부터 끝까지

# 2차원 배열 슬라이싱
print(arr2[1:]) # 2차원 배열에서 인덱스 1부터 끝가지의 행 슬라이싱
print(arr2)
print(arr2[:2, :2]) #배열의 첫번째와 두번째 행 선택, 각 행에서 첫번재와 두번째 열 선택
print(arr2[:, ::-1]) # 모든 행에서 각 행의 열을 역순으로 선택

print(arr2[:, :]) # 배열 전체 선택
print(arr2[1, :]) # 두번째 행 (인덱스 0 선택)
print(arr2[1, 2]) # 두번째 행의 세번째 열

# 3차원 배열 슬라이싱
print(arr3[:2, 1:, :2])

arr1= np.array([1, 2, 3, 4, 5])
arr2 = np.array([11, 12, 13])
#print(arr1 + arr2)
# 배열 차원 확장 - 1차원 배열을 2차원 배열로 확장
# 세로로 변환 - 각 요소가 행
arr1_nx = arr1[:, np.newaxis]
arr2_nx = arr2[:, np.newaxis]
print(arr1_nx)
print(arr2_nx)
# 배열 덧셈 진행
# arr1_nx : 5x1 크기의 배열
# arr2_nx : 3x1 크기의 배열
# broadcasting을 통해 두 배열의 크기를 맞춤
print(arr1_nx + arr2) # arr2의 요소가 arr1_nx 각 행에 더해짐
print(arr2_nx + arr1) # arr1의 요소가 arr2_nx 각 행에 더해짐
