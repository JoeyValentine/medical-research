import sys
import scipy
import numpy as np
import pyqtgraph.opengl as gl
import matplotlib.pyplot as plt
import scipy.ndimage.morphology
import regularize_contour as reg

from skimage import measure
from pyqtgraph.Qt import QtGui


def setdiff_mask(mask1, mask2):
    nx = mask1.shape[0]
    ny = mask1.shape[1]
    mask1_ = mask1.reshape((nx*ny, 1))
    mask2_ = mask2.reshape((nx*ny, 1))
    ind1 = np.where(mask1_ == 1)
    ind2 = np.where(mask2_ == 1)

    index1 = ind1[0]
    index2 = ind2[0]

    s1 = set(index1)
    s2 = set(index2)
    s3 = s1 - s2

    mask3 = np.zeros((nx, ny))

    for j in s3:
        mask3[int(j/ny), j % ny] = 1

    return mask3


def shapebasedSliceInterp(mask_diastole_epi, mask_diastole_endo, numOfInsertedPicture):

    mask_diastole_myo = []

    _, _, slno = mask_diastole_epi.shape

    for i in range(0, slno-1):
        mask1 = mask_diastole_endo[:, :, i+1]
        mask2 = mask_diastole_epi[:, :, i+1]
        mask1_1 = mask_diastole_endo[:, :, i]
        mask2_1 = mask_diastole_epi[:, :, i]

        Di_1_mask1 = scipy.ndimage.morphology.distance_transform_edt(mask1_1) - scipy.ndimage.morphology.distance_transform_edt(1 - mask1_1)
        Di_mask1 = scipy.ndimage.morphology.distance_transform_edt(mask1) - scipy.ndimage.morphology.distance_transform_edt(1 - mask1)
        Di_1_mask2 = scipy.ndimage.morphology.distance_transform_edt(mask2_1) - scipy.ndimage.morphology.distance_transform_edt(1 - mask2_1)
        Di_mask2 = scipy.ndimage.morphology.distance_transform_edt(mask2) - scipy.ndimage.morphology.distance_transform_edt(1 - mask2)

        mask_diastole_myo.append(setdiff_mask(mask2_1, mask1_1))
        for j in range(1, numOfInsertedPicture + 1):
            weight_Di = j / (numOfInsertedPicture + 1)
            weight_Di_1 = 1 - weight_Di
            image_1 = weight_Di_1 * Di_1_mask1 + weight_Di * Di_mask1
            image_2 = weight_Di_1 * Di_1_mask2 + weight_Di * Di_mask2
            binary_1 = image_1 > 0
            binary_2 = image_2 > 0
            binary_1 = binary_1.astype(float)
            binary_2 = binary_2.astype(float)
            binary = setdiff_mask(binary_2, binary_1)
            mask_diastole_myo.append(binary)

    mask_diastole_myo.append(setdiff_mask(mask2, mask1))
    mask_diastole_myo = np.dstack(mask_diastole_myo)
    return mask_diastole_myo

# result = np.load('contour_results.npy')
result = np.load('contour_results_2.npy')

print(result.shape)

cine_images = result[0]  # cine는 4차원영상

print(cine_images.shape)  # 272: 행 갯수, 232:열 , 30:시간측(갯수) , 9: 슬라이스갯수

dicom_info = result[1]

print(dicom_info.PixelSpacing)  # 픽셀실제크기 mm
print(dicom_info.SpacingBetweenSlices)  # mm 10/1.28 배 z축 슬라이스 늘림

mask_diastole_endo = result[2]  # 확장기mask : binary
mask_diastole_epi = result[3]
frameno_diastole = result[4]  # diastole일때  idx
slicelocation = result[5]  # 위치 : 크기차이 == SpacingBetweenSlices

print(slicelocation[0:4])
print(mask_diastole_endo.shape)
print(frameno_diastole)

plt.figure()
plt.imshow(cine_images[:, :, frameno_diastole, 7], cmap='gray')
plt.contour(mask_diastole_epi[:, :, 7], [0.5], colors='r')
plt.contour(mask_diastole_endo[:, :, 7], [0.5], colors='r')

start_idx = 2
mask_diastole = []
row, col, sliceNum = mask_diastole_endo.shape
numOfInsertedPicture = round(dicom_info.SpacingBetweenSlices / dicom_info.PixelSpacing[0])

mask_diastole_epi = mask_diastole_epi.astype(float)
mask_diastole_endo = mask_diastole_endo.astype(float)

for slno in range(3, 8):
    mask_diastole_epi[:, :, slno] = reg.regularize_contour(mask_diastole_epi[:, :, slno])
    mask_diastole_endo[:, :, slno] = reg.regularize_contour(mask_diastole_endo[:, :, slno])

plt.figure()
plt.imshow(cine_images[:, :, frameno_diastole, 7], cmap='gray')
plt.contour(mask_diastole_epi[:, :, 7], [0.5], colors='r')
plt.contour(mask_diastole_endo[:, :, 7], [0.5], colors='r')

mask_diastole = shapebasedSliceInterp(mask_diastole_epi[:, :, 3:8], mask_diastole_endo[:, :, 3:8], numOfInsertedPicture)

# Create an PyQT4 application object.
app = QtGui.QApplication(sys.argv)

# Create a window object.
window = gl.GLViewWidget()
window.resize(500, 500)
window.setCameraPosition(distance=100)
window.setWindowTitle('pyqtgraph : GLIsosurface')
window.show()

# uniform_filter() is equivalent to smooth3() in matlab.
mask_diastole = scipy.ndimage.uniform_filter(mask_diastole, [4, 4, 4], mode='nearest')

# Using marching cubes algorithm to get a polygonal mesh of an isosurface
verts, faces = measure.marching_cubes(mask_diastole, 0.15)  # 값을 바꾸니까 오류가 사라지는군?!
meshdata = gl.MeshData(vertexes=verts, faces=faces)
mesh = gl.GLMeshItem(meshdata=meshdata, smooth=True, color=(1.0, 0.0, 0.0, 0.2), shader='balloon', glOptions='additive')

# Translation
[avgX, avgY, avgZ] = map(np.mean, zip(*verts))
mesh.translate(-avgX, -avgY, -avgZ)
window.addItem(mesh)

plt.show()

sys.exit()
