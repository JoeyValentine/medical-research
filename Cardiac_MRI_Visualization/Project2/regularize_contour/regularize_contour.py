import scipy.ndimage.morphology
import numpy as np


def divergence(field):
    px = field[:, :, 0]
    py = field[:, :, 1]
    row, col = px.shape
    fx = (np.append(px[1:, :], px[-1, :]).reshape(row, col) - np.append(px[0, :], px[:-1, :]).reshape(row, col)) / 2
    fx[0, :] = +px[1, :] / 2 + px[0, :]
    fx[1, :] = +px[2, :] / 2 - px[0, :]
    fx[-1, :] = -px[-1, :] - px[-2, :] / 2
    fx[-2, :] = px[-1, :] - px[-3, :] / 2
    fy = (np.append(py[:, 1:], py[:, -1].reshape(row, 1), axis=1).reshape(row, col) - np.append(py[:, 0].reshape(row, 1), py[:, :-1], axis=1).reshape(row, col)) / 2
    fy[:, 0] = +py[:, 1] / 2 + py[:, 0]
    fy[:, 1] = +py[:, 2] / 2 - py[:, 0]
    fy[:, -1] = -py[:, -1] - py[:, -2] / 2
    fy[:, -2] = py[:, -1] - py[:, -3] / 2
    return fx+fy


def regularize_contour(mask):

    # phi0 : initial contour( singed distance function 멀면 양수고 표면은 0 내부는 음수 == level set function )
    phi0 = scipy.ndimage.morphology.distance_transform_edt(1-mask) - scipy.ndimage.morphology.distance_transform_edt(mask) + mask.astype(float) - 0.5
    tmax = 20
    tau = 0.1

    niter = round(tmax / tau)

    phi = phi0

    eps = 2 ** (-52)

    '''
    a피/at = 그 지점이 직선이면 시간에 따라서 안움직임
    첨점에서는 not zero 값을 가짐 : 시간에 따라서 움직인다.
    뾰족한 부분은 => 변화율이 작은 부분으로 간다.
    직선인 부분은 => 그대로 둔다.
    이러한 방식으로 부드럽게 만든다.
    '''

    for j in range(1, niter+1):
        g0 = np.dstack(np.gradient(phi))
        d = np.maximum(eps, np.sqrt(np.sum(g0 ** 2, axis=2)))  # norm of del phi
        g = g0 / np.tile(d[:, :, np.newaxis], (1, 1, 2))  # normalized 된 g0
        K = d * divergence(g)
        phi = phi + tau * K

        '''
        phi(다음의 phi) - phi(이전의 phi) = tau*k 변화율,변화율은 항상 양수
        등고선은 일정한데 level 올라가는 방향으로 변함
        따라서 원래 이미지보다 작아진다. (단점)
        '''

    nx, ny = phi.shape
    tempmask = phi.reshape(nx*ny, 1)
    idx = np.where(tempmask < 0)[0]

    ret = np.zeros(mask.shape)
    for i in idx:
        ret[int(i/ny), i % ny] = 1

    return ret
