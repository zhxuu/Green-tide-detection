from osgeo import gdal
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# 1. 导入原图像和分类图像
try:
    img_ds = gdal.Open('1016img.tif')
    label_ds = gdal.Open('1016img-classified.tif')
except Exception as e:
    print("无法打开图像文件:", e)
    exit()

# 2. 获取图像尺寸
img_width = img_ds.RasterXSize
img_height = img_ds.RasterYSize

# 3. 将图像调整为二维数组
img_data = img_ds.ReadAsArray()
img_data = np.transpose(img_data, (1, 2, 0))
img_data = img_data.reshape((img_width * img_height, img_data.shape[2]))
#总之，这段代码的主要目的是将原始图像数据重塑为一个二维数组，
# 其中每行包含一个像素的所有颜色通道信息，以便进行后续的图像处理或机器学习任务。
# 这种形式通常更适合处理图像数据，因为它可以轻松地与各种图像处理和机器学习库一起使用。
label_data = label_ds.ReadAsArray().flatten()
#总之，这段代码的目的是将从分类标签数据集中读取的标签数据从多维数组转换为一维数组，以便后续在机器学习模型中使用。
# 这种展平操作通常是为了与许多机器学习算法兼容，因为它们通常期望一维标签数组作为输入。
# 4. 创建支持向量机模型
clf = svm.SVC(kernel='linear')

# 5. 训练模型
clf.fit(img_data, label_data)

# 6. 加载待预测的图像
try:
    predict_ds = gdal.Open('1016img.tif')
except Exception as e:
    print("无法打开待预测的图像文件:", e)
    exit()

predict_data = predict_ds.ReadAsArray()
predict_data = np.transpose(predict_data, (1, 2, 0))
predict_data = predict_data.reshape((img_width * img_height, predict_data.shape[2]))

# 7. 预测图像类别
predict_label = clf.predict(predict_data)

# 将预测结果从一维数组转换为二维数组
predict_label = predict_label.reshape((img_height, img_width))

# 8. 将预测结果保存为图像
#try:
#    output_ds = gdal.GetDriverByName('GTiff').Create('predict_result.tif', img_width, img_height, 1, gdal.GDT_Byte)
#    output_ds.GetRasterBand(1).WriteArray(predict_label)
#    output_ds = None  # 释放数据集
#     print("预测结果已保存为 'predict_result.tif'")
#except Exception as e:
#    print("保存预测结果时出错:", e)

# 9. 计算模型精度
train_predict = clf.predict(img_data)
accuracy = accuracy_score(label_data, train_predict)
print('模型精度为:', accuracy)


# 创建颜色映射
cmap = ListedColormap(['#FF0000', '#0000FF'])  # 使用红色和蓝色表示两个类别

# 10. 可视化预测结果
plt.figure(figsize=(8, 6))
plt.imshow(predict_label, cmap=cmap)
plt.colorbar(ticks=[0, 1])
plt.title("SVM Predicted Result")
plt.show()

