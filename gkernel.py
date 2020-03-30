import numpy as np

def gaussian_kernel(size, Sigma):
    '''
    generate a Gaussian kernel.

    size    - kernel size
    Sigma   - Covariance Matrix

    output  - Gaussian kernel
    '''
    center = size / 2.0 - 0.5

    X, Y = np.meshgrid(np.arange(center-size+1,size-center), np.arange(center-size+1, size-center))
    pos=np.stack([X, Y],axis=2)
    posT=np.reshape(pos, [size, size, 1, 2])
    pos=np.transpose(posT, [0, 1, 3, 2])

    Sigma=np.linalg.inv(Sigma)
    Sigma = np.stack([Sigma] * size*size, 0)
    Sigma= np.reshape(Sigma, [size, size, 2, 2])

    inner=np.matmul(posT, np.matmul(Sigma, pos))
    k = np.exp(-0.5*inner)
    k = k / np.sum(k)

    return k

def anisotropic_Gaussian(ksize,theta,l1,l2):
    '''
    generate an anisotropic Gaussian kernel.

    Param
    ksize - kernel size
    theta - [0,  pi], rotation angle range
    l1    - [0.1,10], eigenvalue 1
    l2    - [0.1,l1], eigenvalue 2

    output
    k     - Gaussian kernel
    '''

    c=np.cos(theta)
    s=np.sin(theta)

    V = np.asarray([[c, -s],[s, c]])
    V_inv = np.asarray([[c, s],[-s, c]])

    D = np.asarray([[l1, 0],[0, l2]])
    Sigma = np.matmul(V, np.matmul(D, V_inv))
    k = gaussian_kernel(ksize, Sigma)

    return np.squeeze(k)

def generate_kernel(k1, ksize=15):
    '''
    generate random anisotropic Gaussian kernel
    '''
    theta = np.pi * np.random.random(1)[0]
    l1 = 1.0 + (k1 - 1.0) * np.random.random(1)[0]
    l2 = 1.0 + (l1 - 1.0) * np.random.random(1)[0]

    kernel = anisotropic_Gaussian(ksize, theta, l1, l2)
    return kernel


