from utils import video_crop_to_images

video_path = "D:/GMS/Documents/Dong/Co3O4/15_DP3.avi"
rec = (30,4,512,512)
save_path = "D:/GMS/Documents/Dong/data/15_DP3/"

video_crop_to_images(video_path, rec, save_path)
