'''
Sample Code for Keypoint matching and Image Registeration
'''

from main import SuperGlueInfer


obj = SuperGlueInfer()
img1_path = "/sample/58130349_0.jpg"
img2_path = "/sample/59249805_1.jpg"
output = obj.predict_kps(img1_path, img2_path, 'output')
arr = output['mkconf']
average_conf = sum(arr)/ len(arr)
print(average_conf)