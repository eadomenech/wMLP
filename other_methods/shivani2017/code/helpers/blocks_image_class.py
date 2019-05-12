

class BlocksImage3D():
    def __init__(self, image_block, sblock_rows=8, sblock_cols=8):
        self.image_block_3d = image_block
        self.size_block_rows = sblock_rows
        self.size_block_cols = sblock_cols
        self.blocks_in_rows = (
            self.image_block_3d.shape[0] // self.size_block_rows
        )
        self.blocks_in_cols = (
            self.image_block_3d.shape[1] // self.size_block_cols
        )

    def get_image(self):
        return self.image_block_3d

    def image_size(self):
        return self.image_block_3d.shape

    def max_num_blocks_image_3d(self):
        image_dims = self.image_size()
        blocks_in_rows = image_dims[0] // self.size_block_rows
        blocks_in_cols = image_dims[1] // self.size_block_cols
        return blocks_in_rows * blocks_in_cols

    def get_coord_block_image_3d(self, num_block):
        if num_block < self.max_num_blocks_image_3d():
            L = []
            row_block = int(num_block / self.blocks_in_cols)
            col_block = num_block % self.blocks_in_cols
            L.append(row_block * self.size_block_rows)
            L.append((row_block + 1) * self.size_block_rows)
            L.append(col_block * self.size_block_cols)
            L.append((col_block + 1) * self.size_block_cols)
            return L
        raise Exception("There is no such block")

    def get_block_image_3d(self, num_block):
        try:
            pos = self.get_coord_block_image_3d(num_block)
            return self.image_block_3d[pos[0]:pos[1], pos[2]:pos[3]]
        except Exception:
            return None

    def set_block_image_3d(self, block, num_block):
        pos = self.get_coord_block_image_3d(num_block)
        self.image_block_3d[pos[0]:pos[1], pos[2]:pos[3]] = block
