import glob, os
count = 5462
dir = "image_generator_train/rename_y"
for pathAndFilename in glob.iglob(os.path.join(dir, '*')):
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))
    os.rename(pathAndFilename, os.path.join(dir, '%04d' %(count) + ext))
    count += 1