import cv2

# 마우스 이벤트 콜백 함수
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, mode, counter

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.rectangle(frame, (ix, iy), (x, y), (0, 255, 0), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(frame, (ix, iy), (x, y), (0, 255, 0), 2)
        # 선택한 영역 잘라내기
        cropped = frame[iy:y, ix:x]
        # 이미지 파일로 저장
        cv2.imwrite(f'cropped_image_{counter}.jpg', cropped)
        counter += 1

# 웹캠 열기
cap = cv2.VideoCapture(0)

# 마우스 이벤트 콜백 함수 등록
cv2.namedWindow('frame')
cv2.setMouseCallback('frame', draw_rectangle)

# 영상 처리 및 출력
drawing = False
ix, iy = -1, -1
counter = 1

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
