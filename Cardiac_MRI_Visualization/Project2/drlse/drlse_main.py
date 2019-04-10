import scipy
import numpy as np
import drlse_edge4 as drlse
import matplotlib.pyplot as plt
import regularize_contour as reg
import scipy.ndimage.morphology as snm
import skimage.morphology as sm


result = np.load('contour_results_2.npy')

cine_images = result[0]  # cine는 4차원영상
dicom_info = result[1]
mask_diastole_endo = result[2]  # 확장기mask : binary
mask_diastole_epi = result[3]
frameno_diastole = result[4]  # diastole일때  idx
slicelocation = result[5]  # 위치 : 크기차이 == SpacingBetweenSlices

slno = 7

# mask_diastole_epi before regularization
mask = mask_diastole_epi[:, :, slno]
img = cine_images[:, :, frameno_diastole, slno]

plt.figure()
plt.imshow(img, cmap='gray')  # result mask
plt.contour(mask, [0.5], colors='r')
plt.title('mask_diastole_epi before regularization')

# mask_diastole_epi after regularization
mask_smooth_epi = reg.regularize_contour(mask)
plt.figure()
plt.imshow(img, cmap='gray')  # result mask
plt.contour(mask_smooth_epi, [0.5], colors='r')
plt.title('mask_diastole_epi after regularization')

# mask_diastole_endo before regularization
mask_endo = mask_diastole_endo[:, :, slno]
plt.figure()
plt.imshow(img, cmap='gray')  # result mask
plt.contour(mask_endo, [0.5], colors='r')
plt.title('mask_diastole_endo before regularization')


# mask_diastole_endo after regularization
mask_smooth = reg.regularize_contour(mask_endo)
plt.figure()
plt.imshow(img, cmap='gray')  # result mask
plt.contour(mask_smooth, [0.5], colors='r')
plt.title('mask_diastole_endo after regularization')

# mask_endo_dilate = scipy.ndimage.morphology.binary_dilation(mask_endo)  # 어느정도 blurr?

mask_endo_dilate = sm.dilation(mask_endo, sm.disk(5))

img = img.astype(float)
img = 255.0 * img/np.amax(img)  # 영상 intensity를 [0 255] 범위로 맞추어 계산해야 level set 알고리즘이 올바로 동작함.

#
# parameter setting
#

timestep = 1  # time step
mu = 0.2/timestep  # coefficient of the distance regularization term R(phi)
niter = 150
lambd = 3  # coefficient of the weighted length term L(phi)
alfa = -2  # coefficient of the weighted area term A(phi)
epsilon = 1.5  # papramater that specifies the width of the DiracDelta function
c0 = 2  # height of binary step function in level set function
sigma = 3  # scale parameter in Gaussian kernel

img_smooth = scipy.ndimage.filters.gaussian_filter(img, sigma)  # smooth image by Gaussiin convolution
temp = np.dstack(np.gradient(img_smooth))
ix, iy = temp[:, :, 0], temp[:, :, 1]
f = ix**2 + iy**2
g = 1./(1+f)  # edge indicator function

plt.figure()
plt.imshow(img_smooth, cmap='gray')  # result mask


# initialize LSF as binary step function

# chan paper와 다른 점 : 초기화할 때 bwdist 사용함
# drlse에서는 초기화 더 간단하게 phi0
nx, ny = mask_endo_dilate.shape
initialLSF = np.zeros(mask_endo_dilate.shape)

idx = np.where(mask_endo_dilate.reshape(nx * ny, 1) == 0)[0]
for i in idx:
    initialLSF[int(i / ny), i % ny] = c0

idx = np.where(mask_endo_dilate.reshape(nx * ny, 1) == 1)[0]
for i in idx:
    initialLSF[int(i / ny), i % ny] = -c0

phi = initialLSF
plt.figure()
plt.imshow(phi)

plt.figure()
plt.imshow(img, cmap='gray')
plt.contour(phi, 0, colors='r')
plt.title('Initial zero level contour')

plt.figure()
plt.imshow(img_smooth, cmap='gray')
plt.title('smooth image')

plt.figure()
plt.imshow(g, cmap='gray')
plt.title('edge indicator function')

phi = drlse.drlse_edge_detect(phi, g, lambd, mu, alfa, epsilon, timestep, niter)

# refine the zero level contour by further level set evolution with alfa=0

alfa = 0
iter_refine = 10
phi = drlse.drlse_edge_detect(phi, g, lambd, mu, alfa, epsilon, timestep, iter_refine)

finalLSF = phi

plt.figure()
plt.imshow(img, cmap='gray')
plt.contour(phi, 0, colors='r')
#  plt.title('Final zero level contour, ' + niter+iter_refine + ' iterations')

plt.show()
