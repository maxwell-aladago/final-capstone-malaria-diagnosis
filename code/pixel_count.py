from dataset import Data

""" 
This utility class was created to simply count the number of piixels belonging to each 
of the classes - normal cells, gametocytes and trophozoites

@author Maxwell Aladago

"""
data = Data()

# do not split data into training and validation. Load all as training
_, masks, _, _ = data.load_data(directory="../")


print("Total Number of Pixels: {} ".format(masks.shape[0] * masks.shape[1]* masks.shape[2]))
print("Total Negative of Pixels: {} ".format(len(masks[masks == 0])))
print("Total Trophpzoites of Pixels: {} ".format(len(masks[masks == 1])))
print("Total Gametocytes of Pixels: {} ".format(len(masks[masks == 2])))

del masks

_, masks, _, _ = data.load_data(directory="../", mode="test")
print("Test")
print("Total Number of Pixels: {} ".format(masks.shape[0] * masks.shape[1] * masks.shape[2]))
print("Total Negative of Pixels: {} ".format(len(masks[masks == 0])))
print("Total Trophpzoites of Pixels: {} ".format(len(masks[masks == 1])))
print("Total Gametocytes of Pixels: {} ".format(len(masks[masks == 2])))
