import numpy as np
import multiprocessing

from AIPlayer import AutoplayS, Dispatcher
from BoardMover import BoardMoverWithScore


class AItest:
    def __init__(self, path):
        self.mover = BoardMoverWithScore()
        self.board_encoded, self.board, self.ai_player, self.score, self.history = self.new_game()
        self.ai_dispatcher = Dispatcher(self.board, self.board_encoded)
        self.died = False
        self.path = path

    def ai_step(self):
        # AI 计算步骤
        empty_slots = np.sum(self.ai_player.board == 0)
        big_nums = (self.ai_player.board > 128).sum()
        if self.is_mess():
            big_nums2 = (self.ai_player.board > 512).sum()
            depth = 6
            self.ai_player.start_search(depth)
            while self.ai_player.node < 320000 * big_nums2 ** 2 and depth < 9:
                depth += 1
                self.ai_player.start_search(depth)
        elif empty_slots > 9 or big_nums < 1:
            self.ai_player.start_search(1)
        elif empty_slots > 4 and big_nums < 2:
            self.ai_player.start_search(2)
        elif (empty_slots > 3 > big_nums) or (big_nums < 2):
            self.ai_player.start_search(3)
        else:
            depth = 4 if big_nums < 4 else 5
            self.ai_player.start_search(depth)
            while self.ai_player.node < 20000 * depth and depth < 8:
                depth += 1
                self.ai_player.start_search(depth)
        return {1:'Left', 2:'Right', 3:'Up', 4:'Down'}.get(self.ai_player.best_operation,'')

    def new_game(self):
        board_encoded = np.uint64(self.mover.gen_new_num(self.mover.gen_new_num(np.uint64(0))[0])[0])
        board = self.mover.decode_board(board_encoded)
        ai_player = AutoplayS(board)
        score = 0
        history = [(board_encoded, score, 0)]
        return board_encoded, board, ai_player, score, history

    def one_step(self):
        self.ai_dispatcher.reset(self.board, self.board_encoded)
        best_move = self.ai_dispatcher.dispatcher()
        if best_move == 'AI':
            self.ai_player.board = self.board
            ai_move = self.ai_step()
            if ai_move:
                self.do_move(ai_move.capitalize())
            else:
                self.died = True
        else:
            self.do_move(best_move.capitalize())

    def do_move(self, direction):
        # print(direction)
        direction = {'Left':1, 'Right':2, 'Up':3, 'Down':4}[direction.capitalize()]
        board_encoded_new, new_score = self.mover.move_board(self.board_encoded, direction)
        board_encoded_new = np.uint64(board_encoded_new)
        if board_encoded_new != self.board_encoded:
            self.board_encoded = board_encoded_new
            self.board = self.mover.decode_board(self.board_encoded)
            self.score += new_score
            self.gen_new_num()
        else:
            self.died = True

    def gen_new_num(self):
        self.board_encoded = np.uint64(self.mover.gen_new_num(self.board_encoded)[0])
        self.board = self.mover.decode_board(self.board_encoded)
        self.history.append((self.board_encoded, self.score, self.ai_dispatcher.last_operator))

    def play(self):
        while not self.died:
            self.one_step()
            if np.sum((self.board == 32768)) == 2:
                positions = np.where(self.board == 32768)
                first_position = (positions[0][0], positions[1][0])
                self.board[first_position] = 2
                self.board_encoded = np.uint64(self.mover.encode_board(self.board))
                self.score += 65536
                self.history.append((self.board_encoded, self.score, self.ai_dispatcher.last_operator))

            if len(self.history) % 931==128:
                if self.score > 1400000:
                    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
                elif self.score > 900000:
                    print('**************************************************')
                print(self.score)
                print(self.board)
                print(self.path)
                if self.score > 1400000:
                    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
                elif self.score > 900000:
                    print('**************************************************')

        print('--------------------------------------------------')
        print(self.score)
        print(self.board)
        print(self.path, 'died')
        print('--------------------------------------------------')
        record = np.array(self.history, dtype='uint64,uint32,uint8')
        record.tofile(self.path)

    def is_mess(self):
        """检查是否乱阵"""
        board = self.board
        large_tiles = (board > 64).sum()
        board_flatten = board.flatten()
        if large_tiles < 3:
            return False
        elif large_tiles > 3:
            top4_pos = np.argpartition(board_flatten, -4)[-4:]
            if len(np.unique(board_flatten[top4_pos])) < 4:
                return False
            top4_pos = tuple(sorted(top4_pos))
            return top4_pos not in (
                (10, 11, 14, 15), (11, 13, 14, 15), (3, 7, 11, 15), (8, 12, 13, 14), (4, 8, 12, 13), (8, 9, 12, 13),
                (0, 1, 2, 4), (1, 2, 3, 7), (0, 1, 4, 5), (0, 1, 4, 8), (2, 3, 7, 11), (2, 3, 6, 7), (0, 1, 2, 3),
                (0, 4, 8, 12), (3, 7, 11, 15), (12, 13, 14, 15))
        else:
            top3_pos = np.argpartition(board_flatten, -3)[-3:]
            if len(np.unique(board_flatten[top3_pos])) < 3:
                return False
            top3_pos = tuple(sorted(top3_pos))
            return top3_pos not in (
                (11, 14, 15), (13, 14, 15), (7, 11, 15), (12, 13, 14), (4, 8, 12), (8, 12, 13), (0, 1, 2), (1, 2, 3),
                (0, 1, 4), (0, 4, 8), (3, 7, 11), (2, 3, 7))


def run_test(index):
    ai_test = AItest(f"{index}")
    ai_test.play()


def main():
    multiprocessing.freeze_support()
    with multiprocessing.Pool(processes=10) as pool:
        pool.map(run_test, range(0, 300))


if __name__ == "__main__":
    main()
# run_test(0)
