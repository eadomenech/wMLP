

class BlocksImage():
    def __init__(self, matrix, sblock_rows=8, sblock_cols=8):
        self.matrix = matrix
        self.size_block_rows = sblock_rows
        self.size_block_cols = sblock_cols
        self.blocks_in_cols = len(self.matrix) // self.size_block_rows
        self.blocks_in_rows = len(self.matrix[1]) // self.size_block_cols

    def get(self):
        return self.matrix

    def max_num_blocks(self):
        return self.blocks_in_rows * self.blocks_in_cols

    def image_size(self):
        return self.matrix.shape

    def get_coord(self, num_block):
        # Se cuentan los bloques a partir de 0
        if num_block < self.max_num_blocks():
            L = []
            x1 = num_block // self.blocks_in_rows
            y1 = num_block % self.blocks_in_rows
            L.append(x1 * self.size_block_rows)
            L.append(y1 * self.size_block_cols)
            L.append((x1 + 1) * self.size_block_rows)
            L.append((y1 + 1) * self.size_block_cols)
            return L
        raise Exception("There is no such block")

    def get_block(self, num_block):
        """
        Retorna la matrix correspondiente al bloque en cuestion.
        Se cuentan los bloques a partir de 0
        """
        try:
            pos = self.get_coord(num_block)
            return self.matrix[pos[0]:pos[2], pos[1]:pos[3]]
        except Exception:
            return None

    def set_block(self, block, num_block):
        # Se cuentan los bloques a partir de 0
        pos = self.get_coord(num_block)
        self.matrix[pos[0]:pos[2], pos[1]:pos[3]] = block
    
    # Poner todos los pixel del bloque de un mismo color
    def set_color(self, num_block):
        try:
            lista = self.get_coord(num_block)

            blanco = 0
            negro = 0

            for x in range(lista[0], lista[2]):
                for y in range(lista[1], lista[3]):
                    if self.matrix[x, y] == 0:
                        negro += 1
                    else:
                        blanco += 1

            if negro > blanco:
                self.matrix[lista[0]:lista[2], lista[1]:lista[3]] = 0
            else:
                self.matrix[lista[0]:lista[2], lista[1]:lista[3]] = 255
            return None
        except Exception as e:
            print(e)
            return None