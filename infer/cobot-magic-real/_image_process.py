#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Responsibilities:
- Crop RGB/Depth to a configured polygon/box
- Convert BGR to RGB for images (to align with downstream expectations)
- Optional denoise for depth (median blur)
- Robust JPEG bytes -> BGR decoding (strip trailing nulls)

Usage:
    from image_processing import ImageProcessor
    ip = ImageProcessor(crop_polygon=[(290,185),(290,350),(540,350),(540,185)], depth_denoise_ksize=5)
    rgb = ip.process_rgb(bgr_image)
    depth = ip.process_depth(depth_image)
"""

from typing import List, Sequence, Tuple, Optional
import numpy as np
import cv2
import traceback
try:
    from termcolor import cprint
except Exception:  # fallback if termcolor isn't installed
    def cprint(msg, color=None):
        print(msg)


Point = Tuple[float, float]


class ImageProcessor:
    def __init__(
        self,
        crop_polygon: Optional[Sequence[Point]] = None,
        depth_denoise_ksize: int = 5,
        bgr_to_rgb: bool = True,
    ) -> None:
        self.crop_polygon: Optional[np.ndarray] = (
            np.array(crop_polygon, dtype=np.float32) if crop_polygon is not None else None
        )
        self.depth_denoise_ksize = depth_denoise_ksize
        self.bgr_to_rgb = bgr_to_rgb
        self._last_crop_offset: Tuple[int, int] = (0, 0)

    @staticmethod
    def _crop_safely(image: np.ndarray, region_points: Sequence[Point]) -> Tuple[np.ndarray, Tuple[int, int]]:
        try:
            x_min = max(0, int(min(p[0] for p in region_points)))
            y_min = max(0, int(min(p[1] for p in region_points)))
            x_max = int(max(p[0] for p in region_points))
            y_max = int(max(p[1] for p in region_points))

            h, w = image.shape[:2]
            x_max = min(w, x_max)
            y_max = min(h, y_max)
            return image[y_min:y_max, x_min:x_max], (x_min, y_min)
        except Exception as e:
            cprint(f"裁剪图像出错: {e}", "red")
            traceback.print_exc()
            return image, (0, 0)

    def crop_to_region(self, image: np.ndarray) -> np.ndarray:
        if self.crop_polygon is None:
            self._last_crop_offset = (0, 0)
            return image
        cropped, offset = self._crop_safely(image, self.crop_polygon)
        self._last_crop_offset = offset
        return cropped

    def get_last_crop_offset(self) -> Tuple[int, int]:
        return self._last_crop_offset

    @staticmethod
    def decode_jpeg_bytes_to_bgr(jpeg_bytes: bytes) -> Optional[np.ndarray]:
        try:
            if isinstance(jpeg_bytes, (bytes, bytearray, np.void)):
                try:
                    jpeg_bytes = bytes(jpeg_bytes).rstrip(b"\x00")
                except Exception:
                    jpeg_bytes = bytes(jpeg_bytes)
            arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return img  # BGR
        except Exception as e:
            cprint(f"JPEG解码失败: {e}", "red")
            traceback.print_exc()
            return None

    def process_rgb(self, image_bgr: np.ndarray) -> np.ndarray:
        """Crop and convert BGR->RGB (if configured). Returns float32 image.
        Assumes input is BGR from OpenCV decoding or camera API.
        """
        img = image_bgr
        if self.crop_polygon is not None:
            img = self.crop_to_region(img)
        if img.ndim == 3 and img.shape[2] == 3 and self.bgr_to_rgb:
            img = img[..., ::-1].copy()
        cv2.imwrite( '0_crop_test_rgb' + '.jpg',img)

        return img.astype(np.float32)

    def process_depth(self, depth: np.ndarray) -> np.ndarray:
        d = depth
        if d.ndim == 3 and d.shape[2] == 1:
            d = d[:, :, 0]
        elif d.ndim > 2:
            d = d[:, :, 0]
        if self.crop_polygon is not None:
            d = self.crop_to_region(d)
        # denoise
        try:
            if self.depth_denoise_ksize and self.depth_denoise_ksize >= 3:
                d = cv2.medianBlur(d.astype(np.float32), int(self.depth_denoise_ksize))
        except Exception as e:
            cprint(f"深度图去噪处理出错: {e}", "red")
        cv2.imwrite( '0_crop_test_depth' + '.jpg',d)
        return d.astype(np.float32)
