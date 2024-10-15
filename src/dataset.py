import numpy as np
from torch.utils.data import Dataset

# Create dataloader for multi-cam BEV
class Cam2BEVDataset(Dataset):
    def __init__(self, input_dirs, label_dir):
        self.input_dirs = input_dirs
        self.label_dir = label_dir
        self.images = sorted(os.listdir(input_dirs[0]))
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        inputs = []
        # Load one-hot encoded image 
        # Input: front, left, rear, right
        for i in self.input_dirs:
            one_hot_image_path = os.path.join(i, self.images[idx])
            image = np.load(one_hot_image_path).astype(np.float32)
            inputs.append(torch.from_numpy(image))
        
        # Label image -> load image -> one-hot encoded image
        label_path = os.path.join(self.label_dir, self.images[idx])
        label = np.load(label_path).astype(np.float32)
        label = torch.from_numpy(label)
        return (inputs, label)


INPUT_PALETTE = [
    [np.array([128,  64, 128])],                                                       # road  
    [np.array([244,  35, 232]), np.array([250, 170, 160])],                            # sidewalk
    [np.array([255,   0,   0])],                                                       # person   
    [np.array([32, 47, 70]), np.array([  0,   0, 142]), np.array([  0,   0, 110])],    # car
    [np.array([ 0,  0, 70])],                                                          # truck
    [np.array([  0,  60, 100]), np.array([ 0,  0, 90])],                               # bus
    [np.array([220,  20,  60]), np.array([  0,   0, 230]), np.array([119,  11,  32])], # two-wheelers
    [np.array([0, 0, 0]), np.array([111,  74,   0]), np.array([81,  0, 81]),           # static obstacles
    np.array([230, 150, 140]), np.array([70, 70, 70]), np.array([102, 102, 156]),      
    np.array([190, 153, 153]), np.array([180, 165, 180]), np.array([150, 100, 100]),   
    np.array([150, 120,  90]), np.array([153, 153, 153]), np.array([153, 153, 153]),
    np.array([250, 170,  30]), np.array([220, 220,   0]), np.array([  0,  80, 100])],
    [np.array([152, 251, 152]),np.array([107, 142,  35])],                             # vegetation
    [np.array([ 70, 130, 180])]                                                        # Sky
]


OUTPUT_PALETTE = [
    [np.array([128,  64, 128])],
    [np.array([244,  35, 232]), np.array([250, 170, 160])],
    [np.array([255,   0,   0])],
    [np.array([  0,   0, 142]), np.array([  0,   0, 110])],
    [np.array([ 0,  0, 70])],
    [np.array([  0,  60, 100]), np.array([ 0,  0, 90])],
    [np.array([220,  20,  60]), np.array([  0,   0, 230]), np.array([119,  11,  32])],
    [np.array([0, 0, 0]), np.array([111,  74,   0]), np.array([81,  0, 81]),
     np.array([230, 150, 140]), np.array([70, 70, 70]), np.array([102, 102, 156]),
     np.array([190, 153, 153]), np.array([180, 165, 180]), np.array([150, 100, 100]),
     np.array([150, 120,  90]), np.array([153, 153, 153]), np.array([153, 153, 153]),
     np.array([250, 170,  30]), np.array([220, 220,   0]), np.array([  0,  80, 100]), np.array([ 70, 130, 180])],
    [np.array([107, 142,  35]), np.array([152, 251, 152])], 
    [np.array([150, 150, 150])]              # OCCLUSION CLASS
]