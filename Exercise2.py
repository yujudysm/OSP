import cv2

# 1. 얼굴 검출을 위한 학습된 모델(XML) 불러오기
# OpenCV에서 기본 제공하는 정면 얼굴 검출기입니다.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 2. 웹캠 연결 (0은 기본 내장 카메라)
cap = cv2.VideoCapture(0)

print("얼굴 검출을 시작합니다. 종료하려면 'q'를 누르세요.")

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        break

    # 3. 속도와 정확도를 위해 그레이스케일(흑백)로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 4. 얼굴 검출 수행
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 5. 검출된 얼굴 위치에 사각형 그리기
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # 화면에 결과 출력
    cv2.imshow('Face Detection', frame)

    # 'q' 키를 누르면 루프 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 메모리 해제 및 창 닫기
cap.release()
cv2.destroyAllWindows()