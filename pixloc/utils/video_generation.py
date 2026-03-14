import logging
import os
import re

import cv2

logger = logging.getLogger(__name__)


def create_video_from_images(image_folder, output_video_path, image_size=(512, 288), fps=25):
    """Create a video from images in a folder.

    Args:
        image_folder: Directory containing image files.
        output_video_path: Output video file path.
        image_size: Output video resolution (width, height).
        fps: Output video frame rate.
    """
    images = [os.path.join(image_folder, img) for img in os.listdir(image_folder)
              if img.endswith(".jpg") or img.endswith(".png")]
    def sort_key(img_path):
        base_name = os.path.basename(img_path)
        match = re.search(r'(\d+)\.png', base_name)
        if match:
            return int(match.group(1))
        return 0

    images.sort(key=sort_key)

    if not images:
        logger.warning("No matching image files found in '%s'.", image_folder)
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, image_size)

    for img_path in images:
        img = cv2.imread(img_path)
        if img is None:
            logger.warning("Failed to read image: %s", img_path)
            continue

        img_resized = cv2.resize(img, image_size)

        video_writer.write(img_resized)

    video_writer.release()
    logger.info("Video saved to: %s", output_video_path)

