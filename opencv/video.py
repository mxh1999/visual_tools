import cv2

def generate_video(video_matrials, outname, size=None, fps=4):
    if size is None:
        img = cv2.imread(video_matrials[0])
        height, width, _ = img.shape
    else:
        height, width = size
    fps=fps
    video = cv2.VideoWriter(outname, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width,height))
    for file in video_matrials:
        img = cv2.imread(file)
        video.write(img)
    video.release()