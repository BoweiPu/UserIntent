import cv2
import numpy as np
def mask_and_fill_with_external_values(image_path, polygons):
    """
    使用多边形区域外部值填充多边形区域。
    
    :param image_path: 输入图片路径。
    :param polygons: 多边形坐标列表，例如 [[[x1, y1], [x2, y2], ...], ...]
    :param output_path: 输出图片路径。
    """
    # 1. 读取图片
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("无法加载图片，请检查路径是否正确。")
    
    # 2. 创建掩膜
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for polygon_coords in polygons:
        polygon_coords = np.array([polygon_coords], dtype=np.int32)
        cv2.fillPoly(mask, polygon_coords, 255)
    
    # 3. 使用掩膜外部值进行填充
    result = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    

    return result

def global_blur(image_path, blur_kernel=(89, 89)):
    """
    对图片进行全局模糊处理。
    
    :param image_path: 输入图片路径。
    :param blur_kernel: 模糊核大小，例如 (25, 25)。
    :param output_path: 输出图片路径。
    """
    # 1. 读取图片
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("无法加载图片，请检查路径是否正确。")
    
    # 2. 全局模糊处理
    blurred_image = cv2.GaussianBlur(image, blur_kernel, 0)

    return blurred_image