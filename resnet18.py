import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F

# 사전 학습된 ResNet-18 모델 불러오기
resnet18 = models.resnet18(pretrained=True)
resnet18.eval()  # 평가 모드로 설정 (드롭아웃 등을 비활성화)
# 분류기 부분을 제거하여 특징 추출기 부분만을 사용하는 새로운 모델 정의
feature_extractor = torch.nn.Sequential(*list(resnet18.children())[:-1])  # 마지막 FC 레이어 제외

# 이미지 전처리를 위한 변환 정의
preprocess = transforms.Compose([
    transforms.Resize(256),  # 이미지 크기 조정
    transforms.CenterCrop(224),  # 이미지 가운데 부분을 잘라냄
    transforms.ToTensor(),  # 이미지를 텐서로 변환
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 이미지를 정규화
])

# 이미지 불러오기 및 전처리
img1_path = 'cropped_image_4.jpg'  # 이미지 파일 경로
img2_path = 'cropped_image_5.jpg'
img1 = Image.open(img1_path)  # 이미지 열기
img2 = Image.open(img2_path)
input_tensor1 = preprocess(img1)  # 전처리된 이미지를 텐서로 변환
input_tensor2 = preprocess(img2)

# 배치 차원 추가 (모델은 배치를 기대하므로)
input_batch1 = input_tensor1.unsqueeze(0)  
input_batch2 = input_tensor2.unsqueeze(0)  

# 모델에 입력 전달하여 예측값 얻기
with torch.no_grad():  # 그라디언트 계산 비활성화
    output1 = feature_extractor(input_batch1)
    output2 = feature_extractor(input_batch2)

# 예측 결과 해석 (이 경우, ImageNet 클래스에 대한 확률)
vec1 = output1.view(-1)
vec2 = output2.view(-1)

# Compute the cosine similarity
cos_sim = F.cosine_similarity(vec1, vec2, dim=0)

print("Cosine similarity:", cos_sim.item())


# def read_classes_from_file(file_path):
#     classes = {}
#     with open(file_path, 'r') as file:
#         for line in file:
#             index, class_name = line.strip().split(': ')
#             classes[int(index)] = class_name
#     return classes

# def get_class_name(class_number, classes):
#     return classes.get(class_number, 'Unknown class')

# # 클래스 정보 파일 경로
# file_path = 'imagenet1000_classes.txt'

# # 텍스트 파일에서 클래스 읽어오기
# classes = read_classes_from_file(file_path)

# # 예측 결과 출력
# print("Predicted label:", predicted_label1)
# print(get_class_name(predicted_label1, classes))
# print("Predicted label:", predicted_label2)
# print(get_class_name(predicted_label2, classes))
