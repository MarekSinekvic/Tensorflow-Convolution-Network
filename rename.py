import os

print("Target path: ")
targetFolder = input()#"Inputs/class_3/"
if (targetFolder[len(targetFolder)-1] != '/' and targetFolder[len(targetFolder)-1] != '\\'): targetFolder += '/'
os.mkdir(targetFolder+"new")
files = os.listdir(targetFolder)
ind = 0
for file in files:
    if file == "new": continue
    os.rename(targetFolder+file,targetFolder+"new/img"+str(ind)+".jpg")
    print(file)
    ind += 1