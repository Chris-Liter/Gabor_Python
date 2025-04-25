import numpy as np
import cv2
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import time
from scipy.signal import convolve2d
from numba import njit

# Gabor kernel en CPU
def generate_gabor_kernel(ksize, sigma, theta, lambd, psi, gamma):
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    half = ksize // 2
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    for y in range(-half, half+1):
        for x in range(-half, half+1):
            x_theta = x * cos_theta + y * sin_theta
            y_theta = -x * sin_theta + y * cos_theta
            gauss = np.exp(-(x_theta**2 + gamma**2 * y_theta**2) / (2 * sigma**2))
            wave = np.cos(2 * np.pi * x_theta / lambd + psi)
            kernel[y + half, x + half] = gauss * wave

    return kernel

#Metodo CPU
# def applyGaborCPU(img, kernel):
#     channels = []
#     for c in range(3):
#         convolved = convolve2d(img[:, :, c], kernel, mode='same', boundary='symm')
#         channels.append(np.clip(convolved, 0, 255).astype(np.uint8))
#     return cv2.merge(channels)

@njit
def applyGaborCPU_numba(img, kernel, ksize):
    half = ksize // 2
    height, width, _ = img.shape
    output = np.zeros_like(img)

    for y in range(half, height - half):
        for x in range(half, width - half):
            for c in range(3):
                suma = 0.0
                for i in range(-half, half + 1):
                    for j in range(-half, half + 1):
                        pixel = img[y + i, x + j, c]
                        weight = kernel[i + half, j + half]
                        suma += weight * pixel
                output[y, x, c] = min(max(int(suma), 0), 255)

    return output
    

# CUDA kernel
kernel_code = """
__global__ void applyGaborCUDA(uchar3* input, uchar3* output, float* kernel, 
                               int ksize, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int half = ksize / 2;

    if (x >= half && y >= half && x < width - half && y < height - half) {
        float sumB = 0.0f, sumG = 0.0f, sumR = 0.0f;
        for (int ky = -half; ky <= half; ky++) {
            for (int kx = -half; kx <= half; kx++) {
                int imgX = x + kx;
                int imgY = y + ky;
                int idx = imgY * width + imgX;
                int kidx = (ky + half) * ksize + (kx + half);
                uchar3 pixel = input[idx];
                float weight = kernel[kidx];
                sumR += weight * pixel.x;  // Red
                sumG += weight * pixel.y;  // Green
                sumB += weight * pixel.z;  // Blue

            }
        }
        int outIdx = y * width + x;
        output[outIdx].x = min(max(int(sumR), 0), 255);
        output[outIdx].y = min(max(int(sumG), 0), 255);
        output[outIdx].z = min(max(int(sumB), 0), 255);

    }
}
"""

mod = SourceModule(kernel_code)
apply_gabor = mod.get_function("applyGaborCUDA")

# Cargar imagen y preparar datos
img = cv2.imread("./imagenArbol.jpg")
if img is None:
    raise Exception("No se pudo cargar la imagen")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
height, width = img.shape[:2]
img_flat = img.reshape(-1, 3).astype(np.uint8)

# Parámetros
mask = 0


for i in range(3):
    
    if mask == 13:
        mask = 21
        print("Mascara 21x21")
    
    if mask == 9:
        mask = 13
        print("Mascara 13x13")
    
    if mask == 0:
        mask = 9
        print("Mascara 9x9")
    
    
    
    kernel = generate_gabor_kernel(mask, 5.0, np.pi/4, 10.0, 2.0, 2.0)

    ####CPU
    
    start = time.time()
    
    cpu = applyGaborCPU_numba(img, kernel, mask)
    
    end = time.time()
    lapso = (end - start) * 1000
    print("Tiempo CPU:", lapso, "ms")

    output_img = cv2.cvtColor(cpu, cv2.COLOR_RGB2BGR)
    cv2.imwrite("gabor_cpu_py_mask_" + str(mask) + ".jpg", output_img)

    ####GPU 
    
    kernel = generate_gabor_kernel(mask, 5.0, np.pi/4, 10.0, 2.0, 2.0).flatten()

    # Memoria GPU
    d_input = cuda.mem_alloc(img_flat.nbytes)
    d_output = cuda.mem_alloc(img_flat.nbytes)
    d_kernel = cuda.mem_alloc(kernel.nbytes)

    cuda.memcpy_htod(d_input, img_flat)
    cuda.memcpy_htod(d_kernel, kernel)

    # Ejecutar kernel
    block = (16, 16, 1)
    grid = ((width + 15) // 16, (height + 15) // 16)

    start = time.time()
    apply_gabor(d_input, d_output, d_kernel, 
                np.int32(mask), np.int32(width), np.int32(height), 
                block=block, grid=grid)
    cuda.Context.synchronize()
    end = time.time()
    print("Tiempo GPU:", (end - start) * 1000, "ms")

    # Obtener imagen de salida
    output = np.empty_like(img_flat)
    cuda.memcpy_dtoh(output, d_output)
    output_img = output.reshape((height, width, 3))
    output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("gabor_cuda_py_mask_" + str(mask) + ".jpg", output_img)
