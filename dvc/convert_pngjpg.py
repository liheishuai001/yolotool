def png_jpg():
    import os
    import cv2
    path = "..\\convertpng\\"
    for filename in os.listdir(path):
        if os.path.splitext(filename)[1] == '.png':
            print(filename)
            img = cv2.imread(path + filename)
            print(filename.replace(".png", ".jpg"))
            newfilename = filename.replace(".png", ".jpg")
            cv2.imwrite(path + newfilename, img)


if __name__ == '__main__':
    png_jpg()