# -*- coding: utf-8 -*-
from block_tools.blocks_class import BlocksImage
from image_tools.ImageTools import ImageTools
from helpers.utils import md5Binary, sha256Binary
from transforms.DqKT import DqKT
from transforms.DAT import DAT

from PIL import Image
import numpy as np
from copy import deepcopy
import hashlib
from scipy import misc
import cv2

#PyTorch
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable

from pathlib import Path


class Avila2019Fragil():
    """
    Método de marca de agua digital frágil para imágenes RGB
    """
    def __init__(self, key='password'):
        self.key = key
        # Hash of key
        self.binary_hash_key = md5Binary(self.key)

    def insert(self, cover_image):
        # Image to array
        cover_array = misc.fromimage(cover_image)

        for color in range(3):        
            # Color component
            color_cover_array = cover_array[:, :, color]
            color_cover_array.setflags(write=1)
            # Dividing in 32x32 blocks
            blocks32x32 = BlocksImage(color_cover_array, 32, 32)
            # Touring each block 32x32
            for num_block in range(blocks32x32.max_num_blocks()):
                # Copying block 32x32
                blocksCopy32x32 = deepcopy(blocks32x32.get_block(num_block))
                # Dividing in 16x16 blocks
                blocksCopy16x16 = BlocksImage(blocksCopy32x32, 16, 16)
                # Get first block
                first_block = blocksCopy16x16.get_block(0)
                # Pariar
                for i in range(16):
                    for y in range(16):
                        if (first_block[i, y] % 2) == 1:
                            first_block[i, y] -= 1
                # Hash of blocksCopy32x32 pareado
                blocksHash = sha256Binary(blocksCopy32x32.tolist())
                # Insert data
                for i in range(16):
                    for y in range(16):
                        first_block[i, y] += int(blocksHash[16*i + y])
                # Update block
                blocks32x32.set_block(blocksCopy32x32, num_block)
        watermarked_image = Image.fromarray(cover_array)
        return watermarked_image

    def extract(self, watermarked_image):
        # To array
        watermarked_array = np.asarray(watermarked_image)
        
        # Validate each color component
        for color in range(3):
            # color component
            color_watermarked_array = watermarked_array[:, :, color]
            color_watermarked_array.setflags(write=1)
            color_watermarked_array_noise = color_watermarked_array.copy()
            # Dividing in 32x32 blocks
            blocks32x32 = BlocksImage(color_watermarked_array_noise, 32, 32)
            # Touring each block 32x32
            modifiedBlocks = []
            for num_block in range(blocks32x32.max_num_blocks()):
                # Copying block 32x32
                blockCopy32x32 = deepcopy(blocks32x32.get_block(num_block))
                # Dividing in 16x16 blocks
                blocksCopy16x16 = BlocksImage(blockCopy32x32, 16, 16)
                # Get first block
                first_block = blocksCopy16x16.get_block(0)
                # Watermark
                w = ''
                # Pariar
                for i in range(16):
                    for y in range(16):
                        if (first_block[i, y] % 2) == 1:
                            first_block[i, y] -= 1
                            w += '1'
                        else:
                            w += '0'
                # Hash of blocksCopy32x32 pareado
                blocksHash = sha256Binary(blockCopy32x32.tolist())
                if w != blocksHash:
                    modifiedBlocks.append(num_block)
        modifiedBlocks = list(set(modifiedBlocks))
        print(modifiedBlocks)
        for item in modifiedBlocks:
            coord = blocks32x32.get_coord(item)
            cv2.rectangle(
                watermarked_array, (coord[1], coord[0]),
                (coord[3], coord[2]), (0, 255, 0), 1)
        return Image.fromarray(watermarked_array)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3 * 8 * 8, 5000)
        self.fc2 = nn.Linear(5000, 9)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class Clasification():
    
    def __init__(self):
        self.model = Net()
        checkpoint = torch.load(Path('data/fnn600_with_jpeg_20.pt'), map_location='cpu')
        self.model.load_state_dict(checkpoint)
        self.model.eval()

        self.data_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
    
    def predict(self, image_array):
        image = Image.fromarray(image_array)
        image_tensor = self.data_transforms(image)

        # PyTorch pretrained models expect the Tensor dims to be
        # (num input imgs, num color channels, height, width).
        # Currently however, we have (num color channels, height, width);
        # let's fix this by inserting a new axis.
        image_tensor = image_tensor.unsqueeze(0)

        output = self.model(Variable(image_tensor))

        return np.argmax(output.detach().numpy())
    

class Avila2019Robust():
    """
    Método de marca de agua digital robusto para imágenes RGB
    """
    def __init__(self, key):
        self.key = key
        # Hash of key
        self.binary_hash_key = md5Binary(self.key)
        # Instancia
        dat = DAT()
        # Cargando watermark
        watermark = Image.open("static/Watermarking.png").convert("1")
        # obteniendo array de la watermark
        watermark_array = np.asarray(watermark)
        # Datos de la watermark como lista
        watermark_as_list = watermark_array.reshape((1, watermark_array.size))[0]
        # Tomando solo los valores correspondientes a los datos
        watermark_list = []
        for p in myqr.get_pos():
            if watermark_as_list[p]:
                watermark_list.append(0)
            else:
                watermark_list.append(255)                
        # Convertir a matriz cuadrada
        for p in range(8):
            watermark_list.append(0)
        true_watermark_array = np.reshape(np.asarray(watermark_list), (38, 38))
        # Obtener imagen correspondiente
        true_watermark_image = misc.toimage(true_watermark_array)
        # Utilizando Arnold Transforms
        # Para imagen 38x38 el periodo es 40
        for i in range(30):
            true_watermark_image = dat.dat2(true_watermark_image)
        true_watermark_array_scambred = np.asarray(true_watermark_image)
        # Calculando e imprimeindo datos iniciales
        self.len_of_watermark = true_watermark_array.size
        print('Cantidad de bit a insertar: ', self.len_of_watermark)
        # Datos de la watermark como lista
        self.list_bit_of_watermark = true_watermark_array_scambred.reshape((1, self.len_of_watermark))[0]

        # Asignando a v para tener una misma v en todos los experimentos
        self.v = [
            131840, 84237, 46815, 54681, 97958, 131552, 38851, 147140, 89791,
            23580, 23180, 142232, 26755, 127416, 124163, 61299, 80119, 134993,
            69773, 84627, 114263, 39221, 54895, 45552, 138270, 78155, 85025,
            75583, 111312, 67369, 91647, 45888, 52095, 129152, 40287, 37771,
            50710, 111804, 109780, 112304, 35212, 99184, 109522, 149238, 48917,
            80127, 146975, 99235, 53891, 74614, 114598, 76051, 125557, 85602,
            85019, 55935, 65454, 128219, 91659, 66131, 77709, 31129, 60087,
            89703, 94543, 124759, 46807, 53622, 56068, 38912, 47970, 54605,
            43117, 54167, 34462, 80924, 117682, 38655, 33169, 97674, 44050,
            80532, 140564, 60246, 46024, 64387, 129963, 60141, 110025, 140439,
            66055, 101039, 135019, 151197, 110769, 121379, 140515, 113448,
            116596, 135153, 77302, 78597, 63208, 104279, 49864, 64976, 78729,
            77931, 66439, 148252, 67321, 78870, 120408, 103107, 74566, 128812,
            54177, 127200, 56347, 113937, 33476, 38939, 75849, 95222, 126624,
            149601, 35876, 71409, 87916, 87971, 150907, 123871, 89968, 95675,
            38404, 68842, 103021, 27629, 43942, 109683, 113800, 139765, 142147,
            105373, 66716, 52648, 85166, 53791, 67237, 115103, 113797, 22378,
            48973, 56820, 29522, 113314, 145755, 143664, 130912, 52268, 108111,
            44063, 27736, 109348, 43586, 80009, 29105, 48373, 128959, 76536,
            31763, 88780, 50852, 94939, 63054, 26735, 62698, 95476, 26142, 77481,
            24401, 92343, 51954, 48964, 62615, 52656, 90014, 104809, 50089,
            64153, 29027, 80972, 29790, 139606, 66237, 39383, 38222, 92673,
            54740, 32663, 128309, 56471, 66809, 90230, 113113, 125344, 127476,
            90798, 137430, 91670, 133727, 98295, 81205, 140824, 55433, 87552,
            68857, 23576, 138013, 70192, 78075, 65630, 108868, 59263, 99182,
            60697, 105891, 109555, 29724, 95979, 113395, 38507, 142114, 113830,
            106608, 57315, 124054, 144493, 51734, 27414, 137276, 128380, 31103,
            104134, 45317, 106026, 82181, 47353, 149841, 75070, 80487, 56848,
            25116, 70973, 49252, 77359, 81774, 108558, 63089, 145446, 61663,
            140104, 48231, 103973, 141160, 30553, 121740, 121012, 67702, 115102,
            133506, 45113, 138749, 25128, 65024, 133076, 29703, 61703, 62952,
            46831, 147092, 92971, 102629, 149842, 80855, 115874, 106054, 114306,
            135545, 80095, 61658, 83522, 148005, 88115, 29947, 108151, 103991,
            146633, 97532, 30037, 105936, 103056, 135150, 116877, 71765, 53645,
            111704, 114675, 118807, 142438, 67680, 111729, 30195, 65680, 84254,
            136976, 33976, 58149, 127402, 36576, 70853, 150348, 36473, 123351,
            70131, 33368, 148505, 35864, 63488, 42671, 49639, 108871, 149030,
            150433, 125724, 64474, 42541, 96700, 83328, 117815, 58908, 124617,
            141390, 36141, 80257, 69860, 30653, 61708, 151211, 53831, 116303,
            82241, 62086, 93442, 114307, 26003, 24693, 151164, 53223, 54591,
            143438, 35602, 38142, 41450, 50731, 127238, 61666, 131377, 30151,
            64594, 65882, 113361, 28032, 83105, 120922, 111816, 33157, 36301,
            133846, 85113, 114679, 148318, 25778, 145006, 89343, 82587, 74456,
            25224, 137983, 117454, 57206, 38931, 124195, 137884, 116301, 59654,
            87028, 147453, 64786, 22161, 117826, 145115, 129910, 130464, 87755,
            134747, 85130, 93910, 83000, 42649, 65526, 68546, 33045, 142246,
            126323, 24111, 62382, 125737, 59239, 93708, 93950, 24688, 150403,
            48939, 68025, 31712, 119553, 28554, 137481, 38786, 61408, 35543,
            82050, 86064, 134197, 47766, 98319, 101330, 48855, 87549, 23150,
            45589, 147461, 36776, 75597, 135713, 128637, 143698, 111759, 100537,
            135354, 79858, 124721, 136956, 49067, 50541, 121791, 119266, 64179,
            97562, 141679, 113731, 61859, 139883, 81491, 64252, 124247, 147220,
            132384, 108116, 106311, 138943, 115552, 80018, 84764, 36282, 120116,
            48252, 126736, 89411, 48891, 113326, 79842, 102622, 40958, 111289,
            129129, 61671, 23039, 64661, 80473, 88171, 141995, 53196, 82576,
            68311, 126618, 83145, 63105, 97313, 61813, 141531, 48936, 75695,
            44091, 111679, 60143, 145777, 91615, 30897, 115978, 74712, 127860,
            32210, 102215, 105420, 35533, 128484, 95867, 132409, 56934, 146296,
            149667, 91924, 82909, 43038, 56760, 96324, 141310, 32956, 88480,
            138910, 135951, 88412, 124542, 141224, 88002, 23611, 76329, 87504,
            40959, 72353, 80407, 63822, 24924, 89281, 60540, 149755, 37785,
            97143, 111296, 50611, 35587, 43262, 145375, 85479, 60988, 60870,
            31991, 94189, 110126, 30303, 88739, 62343, 117067, 56690, 136167,
            133307, 119645, 150756, 109182, 122406, 73464, 46551, 64575, 35957,
            68477, 92136, 66918, 60265, 36899, 104278, 24038, 137029, 91676,
            115515, 151121, 68363, 81278, 114798, 52143, 31806, 44192, 26917,
            133232, 134335, 36173, 67356, 122845, 80506, 24273, 85915, 62515,
            55255, 45268, 79569, 123104, 95525, 127405, 147922, 149614, 28654,
            112294, 78024, 68143, 104114, 24520, 75068, 52738, 115118, 120729,
            113763, 117657, 126807, 91000, 143889, 56386, 70542, 41109, 145774,
            29367, 100863, 93038, 53913, 25785, 28277, 104755, 133114, 70045,
            77628, 64025, 103103, 113722, 33791, 29204, 87135, 117273, 145705,
            148080, 80106, 106664, 78740, 90135, 76477, 118920, 88170, 110175,
            150822, 64275, 137521, 90538, 137902, 91007, 140582, 146603, 105403,
            111864, 41960, 42413, 35342, 43442, 63041, 61498, 53348, 106208, 119521, 27384, 104600, 105515, 68413, 125047, 85177, 25623, 126988, 68414, 76413, 90148, 117521, 65175, 50122, 31947, 83774, 77085, 100994, 85909, 29645, 116801, 34656, 143435, 71108, 74646, 113330, 41546, 26303, 85080, 40922, 89203, 144439, 145409, 120178, 45979, 66760, 150296, 51942, 127485, 55475, 62333, 52287, 112659, 54051, 86442, 35173, 51272, 31005, 31517, 92094, 93385, 64637, 32743, 91416, 25333, 34871, 135450, 120625, 55204, 146663, 58465, 75596, 133656, 135283, 149309, 91098, 120822, 139144, 135337, 113380, 56711, 102312, 25967, 121648, 129927, 72355, 70466, 86481, 132784, 60456, 55302, 25726, 101068, 150016, 126418, 119783, 70582, 85232, 54889, 130452, 86740, 96357, 76859, 76525, 133385, 96191, 93378, 114550, 41467, 41017, 120399, 146212, 78180, 100112, 122524, 104213, 145089, 146217, 97423, 75448, 112046, 48961, 133309, 126554, 135986, 137270, 129184, 63126, 109502, 85976, 95913, 34709, 131712, 57951, 63556, 31773, 57794, 68086, 87909, 136162, 128025, 145861, 107869, 33922, 47582, 102503, 138235, 137927, 144705, 121980, 113971, 105632, 137415, 44712, 40999, 112956, 43147, 75201, 122093, 144689, 127261, 144303, 134195, 63789, 98121, 140367, 23635, 31028, 90549, 83380, 32843, 47270, 34425, 70293, 64043, 56662, 67958, 131550, 24895, 141391, 89568, 25975, 122450, 51792, 72579, 23289, 80054, 25245, 76860, 127223, 67257, 114358, 131427, 28097, 131668, 55834, 81505, 100220, 49643, 148230, 68449, 42976, 42786, 55944, 25515, 23437, 110984, 116683, 146194, 141341, 77622, 130981, 40646, 104143, 34813, 66250, 135876, 142902, 130198, 128470, 74020, 120073, 47976, 150869, 68415, 64356, 126677, 125956, 34206, 62544, 121239, 94930, 36529, 139338, 83077, 97139, 29851, 68820, 125130, 106731, 106696, 140530, 53413, 61358, 69390, 57159, 36376, 144109, 130373, 83067, 28534, 34076, 146164, 66826, 135522, 24887, 118905, 105621, 74552, 106909, 49911, 148226, 134506, 49914, 114154, 83365, 31885, 98861, 131820, 142787, 44443, 75942, 50317, 91604, 104386, 45969, 31585, 44026, 145453, 50912, 37204, 126580, 107131, 104362, 131268, 73173, 117152, 99126, 88591, 38484, 67691, 59889, 22870, 140134, 48214, 139246, 58531, 62902, 109332, 103901, 37402, 59833, 37125, 128322, 100462, 110402, 120364, 96884, 118078, 151395, 87625, 129669, 60552, 138938, 128032, 136842, 35551, 149424, 151299, 49684, 106055, 53094, 93697, 120025, 124700, 66356, 125568, 68904, 138769, 36781, 144560, 99344, 114251, 104801, 136574, 139127, 147512, 38164, 80404, 99660, 67192, 149110, 23533, 43220, 146349, 86496, 67698, 113416, 114368, 54312, 82177, 48766, 96311, 46498, 114270, 35767, 43077, 32825, 148061, 78410, 35170, 93763, 43794, 117721, 113894, 65946, 143324, 139951, 91720, 69861, 27246, 35136, 73749, 95764, 92236, 71835, 126816, 50479, 141207, 90870, 127246, 137490, 63561, 96847, 105037, 40729, 138395, 136263, 22436, 104699, 102881, 84202, 111865, 70573, 34091, 54822, 147605, 116172, 26414, 115906, 70108, 45620, 75414, 127044, 76492, 27332, 135495, 98380, 107248, 80160, 75121, 117877, 24402, 120982, 130205, 108739, 43449, 91795, 78135, 56157, 97094, 27220, 52131, 104281, 101351, 139142, 67895, 116225, 120083, 137926, 84596, 59846, 22284, 26872, 53785, 96247, 95938, 77915, 144623, 67473, 22304, 71380, 142502, 42703, 113074, 102865, 68465, 77079, 80627, 60442, 89830, 146693, 142840, 142890, 132908, 57962, 94670, 64249, 63444, 117502, 93338, 29916, 34859, 67314, 76363, 109317, 114649, 107684, 82873, 109071, 105201, 138826, 117210, 119997, 96768, 100480, 104950, 49745, 100814, 110114, 81910, 127704, 46090, 104526, 80514, 52381, 64857, 125295, 54879, 133478, 95883, 75160, 110181, 99242, 85064, 76682, 70121, 129079, 92699, 128475, 24751, 76424, 64288, 50641, 103799, 82079, 69281, 42279, 31336, 149883, 135926, 109953, 94314, 148624, 102530, 77581, 102106, 96209, 52618, 66027, 76894, 42242, 40433, 33046, 98761, 100657, 61120, 128123, 26456, 46516, 45442, 70858, 102884, 62562, 144968, 100643, 33274, 101496, 30277, 130406, 145034, 43469, 144660, 44051, 52923, 33894, 87286, 111694, 71646, 37277, 132079, 54376, 74687, 74774, 127943, 128454, 139563, 52610, 103952, 37759, 59447, 146561, 107168, 112840, 71011, 124559, 87505, 94673, 32415, 72499, 150717, 121628, 132451, 71805, 67478, 136241, 80499, 66290, 93796, 96895, 41813, 41695, 65939, 132444, 134746, 60705, 40495, 122459, 54727, 42748, 118757, 150816, 28672, 137102, 75402, 69373, 98059, 128319, 51039, 65008, 89738, 51504, 131325, 110613, 104233, 110967, 49337, 127543, 121010, 65470, 79436, 65703, 104678, 148651, 31305, 60673, 61117, 24498, 135779, 120744, 79147, 105105, 149652, 133693, 125354, 83802, 57657, 131209, 130017, 73534, 101328, 36908, 93689, 40502, 93325, 43441, 93509, 108425, 37618, 122020, 83117, 84996, 94725, 41006, 146626, 22590, 80526, 99280, 135804, 95371, 63037, 86058, 144900, 122633, 129458, 83138, 102212, 142599, 81642, 97862, 55609, 91895, 142422, 99520, 116787, 111403, 90857, 55020, 147443, 134066, 107575, 117464, 44598, 62522, 37439, 49657, 25357, 122631, 38216, 40201, 96045, 103467, 144260, 77567, 22831, 64767, 70279, 87219, 142053, 40480, 109943, 89138, 113521, 72341, 22291, 110574, 94802, 138830, 50677, 109801, 148175, 64880, 136745, 41568, 53970, 110912, 150905, 40023, 53014, 108431, 149522, 134731, 119652, 141804, 102280, 74624, 40173, 94179, 123368, 85527, 68424, 26916, 100227, 108514, 98140, 115748, 87324, 137952, 35148, 129908, 64208, 42284, 62685, 48408, 63017, 141266, 116382, 122610, 41174, 94819, 59713, 23960, 34282, 74675, 123119, 76093, 87077, 97031, 149434, 79426, 28424, 135937, 63367, 105197, 51069, 46680, 58409, 147964, 118233, 146247, 95802, 125897, 136626, 62256, 125215, 57355, 149007, 79570, 102663, 108303, 73740, 51776, 78901, 42833, 124175, 24482, 38100, 81263]

    def insert(self, cover_image):
        # Image to array
        cover_array = misc.fromimage(cover_image)
        # Instancia
        itools = ImageTools()
        clasification = Clasification()
        dqkt = DqKT()
        # Convirtiendo a modelo de color YCbCr
        cover_ycbcr_array = itools.rgb2ycbcr(cover_image)
        # Componente Y
        cover_array = cover_ycbcr_array[:, :, 0]
        # Instance a la clase Bloque
        bt_of_cover = BlocksImage(cover_array)
        bt_of_rgb_cover = BlocksImage(misc.fromimage(cover_image))

        # print("Seleccionando bloques...")
        # # Utilizar Bloques segun key
        # dic = {'semilla': 0.00325687, 'p': 0.22415897}
        # valores = []
        # cantidad = bt_of_cover.max_blocks()
        # for i in range(cantidad):
        #     valores.append(i)
        # v = pwlcm.mypwlcm_limit(dic, valores, len_of_watermark)

        # # Simulando pwlcm para AG
        # import random
        # v = []
        # cantidad = bt_of_cover.max_num_blocks()            
        # x, y = cover_image.size
        # izq_min = x//64
        # der_max = (x//8 - x//64)
        # arr_min = y//64
        # aba_max = (y//8 - y//64)
        # while len(v) < len_of_watermark:
        #     val = random.randrange(cantidad)
        #     columna = val//(x//8)
        #     fila = val - columna*(x//8)
        #     if val not in v:
        #         if columna > izq_min and columna < der_max and fila > arr_min and fila < aba_max:
        #             v.append(val)

        print("Insertando...")
        # Marcar los self.len_of_watermark bloques
        for i in range(self.len_of_watermark):
            block = bt_of_cover.get_block(self.v[i])
            # Predict
            p = clasification.predict(
                bt_of_rgb_cover.get_block(self.v[i]))
            if p == 1:
                c = 17
                delta = 90
            elif p == 4:
                c = 19
                delta = 60
            elif p == 5:
                c = 20
                delta = 130
            elif p == 7:
                c = 28
                delta = 94
            elif p == 8:
                c = 34
                delta = 130
            else:
                c = 19
                delta = 130

            dqkt_block = dqkt.dqkt2(block)

            negative = False
            if dqkt_block[get_indice(c)[0], get_indice(c)[1]] < 0:
                negative = True

            if self.list_bit_of_watermark[i % self.len_of_watermark] == 0:
                # Bit a insertar 0
                dqkt_block[get_indice(c)[0], get_indice(c)[1]] = 2*delta*round(abs(dqkt_block[get_indice(c)[0], get_indice(c)[1]])/(2.0*delta)) - delta/2.0
            else:
                # Bit a insertar 1
                dqkt_block[get_indice(c)[0], get_indice(c)[1]] = 2*delta*round(abs(dqkt_block[get_indice(c)[0], get_indice(c)[1]])/(2.0*delta)) + delta/2.0

            if negative:
                dqkt_block[get_indice(c)[0], get_indice(c)[1]] *= -1
            idqkt_block = dqkt.idqkt2(dqkt_block)
            inv = idqkt_block
            for x in range(8):
                for y in range(8):
                    if (inv[x, y] - int(inv[x, y])) < 0.5:
                        inv[x, y] = int(inv[x, y])
                    else:
                        inv[x, y] = int(inv[x, y]) + 1
                    if inv[x, y] > 255:
                        inv[x, y] = 255
                    if inv[x, y] < 0:
                        inv[x, y] = 0
            bt_of_cover.set_block(idqkt_block, self.v[i])
        
        cover_marked_ycbcr_array = cover_ycbcr_array

        image_rgb_array = itools.ycbcr2rgb(cover_marked_ycbcr_array)
        watermarked_image = Image.fromarray(image_rgb_array)
        return watermarked_image

    def extract(self, watermarked_image):
        import cv2
        # To array
        watermarked_array = np.asarray(watermarked_image)
        # Blue component
        blue_watermarked_array = watermarked_array[:, :, 2]
        blue_watermarked_array.setflags(write=1)
        blue_watermarked_array_noise = blue_watermarked_array.copy()
        # Dividing in 32x32 blocks
        blocks32x32 = BlocksImage(blue_watermarked_array_noise, 32, 32)
        # Touring each block 32x32
        modifiedBlocks = []
        for num_block in range(blocks32x32.max_num_blocks()):
            # Copying block 32x32
            blockCopy32x32 = deepcopy(blocks32x32.get_block(num_block))
            # Dividing in 16x16 blocks
            blocksCopy16x16 = BlocksImage(blockCopy32x32, 16, 16)
            # Get first block
            first_block = blocksCopy16x16.get_block(0)
            # Watermark
            w = ''
            # Pariar
            for i in range(16):
                for y in range(16):
                    if (first_block[i, y] % 2) == 1:
                        first_block[i, y] -= 1
                        w += '1'
                    else:
                        w += '0'
            # Hash of blocksCopy32x32 pareado
            blocksHash = sha256Binary(blockCopy32x32.tolist())
            if w != blocksHash:
                modifiedBlocks.append(num_block)
        print(modifiedBlocks)
        for item in modifiedBlocks:
            coord = blocks32x32.get_coord(item)
            cv2.rectangle(
                watermarked_array, (coord[1], coord[0]),
                (coord[3], coord[2]), (0, 255, 0), 1)
        return Image.fromarray(watermarked_array)
