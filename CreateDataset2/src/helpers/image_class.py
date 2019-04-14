

from helpers.blocks_class import BlocksImage


class Block_RGBImage():
    def __init__(self, image_array, sblock_rows=8, sblock_cols=8):
        self.image = image_array
        self.size_block_rows = sblock_rows
        self.size_block_cols = sblock_cols

    def get_image(self):
        return self.image

    def image_size(self):
        return self.image.shape

    def max_num_blocks_image(self):
        image_dims = self.image_size()
        blocks_in_rows = image_dims[0] // self.size_block_rows
        blocks_in_cols = image_dims[1] // self.size_block_cols
        return blocks_in_rows * blocks_in_cols * image_dims[2]

    def get_coord_block_image(self, num_block):
        L = []
        red_instantce = BlocksImage(
            self.image[:, :, 0], self.size_block_rows, self.size_block_cols)
        green_instantce = BlocksImage(
            self.image[:, :, 1], self.size_block_rows, self.size_block_cols)
        blue_instantce = BlocksImage(
            self.image[:, :, 2], self.size_block_rows, self.size_block_cols)
        red_max = red_instantce.max_num_blocks()
        green_max = 2 * red_max
        blue_max = 3 * red_max
        if num_block < self.max_num_blocks_image():
            if num_block < red_max:
                L = red_instantce.get_coord(num_block)
                L.append(0)
            elif red_max <= num_block and num_block < green_max:
                L = green_instantce.get_coord(num_block % red_max)
                L.append(1)
            elif green_max <= num_block and num_block < blue_max:
                L = blue_instantce.get_coord(num_block % green_max)
                L.append(2)
            return L
        raise Exception("There is no such block")

    def get_block_image(self, num_block):
        try:
            pos = self.get_coord_block_image(num_block)
            return self.image[pos[0]:pos[1], pos[2]:pos[3], pos[4]]
        except Exception:
            return None

    def set_block_image(self, block, num_block):
        pos = self.get_coord_block_image(num_block)
        self.image[pos[0]:pos[1], pos[2]:pos[3], pos[4]] = block
