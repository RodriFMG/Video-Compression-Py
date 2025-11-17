import cv2
import numpy as np
from Core.GOP import GOP


def load_frames(video_path: str) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {video_path}")

    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames


def VideoCompression(frames_video: list[np.ndarray], block_size: int, alpha: float) -> list[GOP]:
    GOPs: list[GOP] = []

    n = len(frames_video)
    size_gop: int = GOP().get_size_group()

    for idx in range(0, n, size_gop):
        # se puede paralelizar xd
        gop_i: GOP = GOP().encode(frames_video[idx: idx + size_gop], block_size, alpha)
        GOPs.append(gop_i)

    return GOPs


def VideoDescompression(GOPs: list[GOP]) -> list[np.ndarray]:
    frames_desc: list[np.ndarray] = []

    for gop_i in GOPs:
        frames_desc.extend(gop_i.decode())

    return frames_desc


if __name__ == "__main__":
    video_path = "test.mp4"
    frames = load_frames(video_path)

    print(f"Total de frames cargados: {len(frames)}")
    print("Dimensiones del primer frame:", frames[0].shape if frames else "No hay frames")
