import os
import numpy as np

def getFiles(brick, flatConcrete, steelTile):

    brick_paths = []
    flatConcrete_paths = []
    steelTile_paths = []

    for root, dirs, files in os.walk(brick):
        for file in files:
            brick_paths.append(os.path.join(root, file))

    for root, dirs, files in os.walk(flatConcrete):
        for file in files:
            flatConcrete_paths.append(os.path.join(root, file))

    for root, dirs, files in os.walk(steelTile):
        for file in files:
            steelTile_paths.append(os.path.join(root, file))
        
    brick_paths.sort()
    flatConcrete_paths.sort()
    steelTile_paths.sort()

    brick_img, brick_label = [brick_paths[i] for i in range(0,len(brick_paths),2)], [brick_paths[i] for i in range(1,len(brick_paths),2)]
    flatConcrete_img, flatConcrete_label = [flatConcrete_paths[i] for i in range(0,len(flatConcrete_paths),2)], [flatConcrete_paths[i] for i in range(1,len(flatConcrete_paths),2)]
    steelTile_img, steelTile_label = [steelTile_paths[i] for i in range(0,len(steelTile_paths),2)], [steelTile_paths[i] for i in range(1,len(steelTile_paths),2)]

    images = np.concatenate((brick_img, flatConcrete_img, steelTile_img))
    masks = np.concatenate((brick_label,flatConcrete_label,steelTile_label  ))

    return images, masks