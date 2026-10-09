from MonocularCues import get_monocular_cues;
import os;
import shutil;
from PIL import Image;

a = ["000", "001", "002", "003", "004", "005", "006", "007", "008", "009", "010", "011"]

DATASET_PATH = "/home/leusteanstefan/Documents/Practica/datasets/objaverse_pilot"
os.chdir(DATASET_PATH)

print("Working Directory: ", os.getcwd())

os.chdir("./objects")
OBJ_PATH = os.getcwd()

print("Moving to Objects: ", os.getcwd())

objects = os.listdir(os.getcwd())
print("Objects: ", len(objects))

for obj in objects:
    CURRENT_OBJ = OBJ_PATH + "/" + obj

    os.chdir(CURRENT_OBJ)

    print("Modifying: ", obj)

    print("Creating MoGE directory:")

    rem = True
    if os.path.exists(CURRENT_OBJ + "/moge"):
        
        if os.path.exists(CURRENT_OBJ + "/moge/depth"):
            count = len([n for n in os.listdir(CURRENT_OBJ + "/moge/depth")])

            if count != 12:
                rem = True    
            else:
                rem = False 
        else:
            rem = True

    if rem == False:
        print("Object already done!")
        os.chdir(OBJ_PATH)
        continue
    else:
        print("Doing Object")
        if os.path.exists(CURRENT_OBJ + "/moge"):
            shutil.rmtree(CURRENT_OBJ + "/moge")

    os.makedirs(CURRENT_OBJ + "/moge", exist_ok=True)
    os.makedirs(CURRENT_OBJ + "/moge/depth", exist_ok=True)
    os.makedirs(CURRENT_OBJ + "/moge/normal", exist_ok=True)

    for rgb_image in a:
        rgb_path = "rgb/" + rgb_image + ".png"

        depth_array, normal_array = get_monocular_cues(rgb_path)

        img = Image.fromarray(depth_array)
        img.save(CURRENT_OBJ + "/moge/depth/" + rgb_image + "_depth.png")

        img = Image.fromarray(normal_array, "RGB")
        img.save(CURRENT_OBJ + "/moge/normal/" + rgb_image + "_normal.png")

    os.chdir(OBJ_PATH)