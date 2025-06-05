import multiprocessing
import os

import numpy as np

from AIPlayer import AIPlayer, Dispatcher
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
        empty_slots = self.ai_dispatcher.counts[0]
        big_nums = np.sum(self.ai_dispatcher.counts[8:])
        if self.is_mess():
            big_nums2 = np.sum(self.ai_dispatcher.counts[9:])
            depth = 5 if np.max(self.ai_dispatcher.counts[8:]) == 1 else 6
            if self.ai_player.check_corner(self.board_encoded):
                depth = 8
            self.ai_player.start_search(depth)
            while self.ai_player.node < 200000 * big_nums2 ** 2 and depth < 9:
                depth += 1
                self.ai_player.start_search(depth)
        elif empty_slots > 9 or big_nums < 1:
            self.ai_player.start_search(1)
        elif empty_slots > 4 and big_nums < 2:
            self.ai_player.start_search(2)
        elif (empty_slots > 3 > big_nums) or (big_nums < 2):
            self.ai_player.start_search(4)
        else:
            depth = 4 if big_nums < 4 else 5
            self.ai_player.start_search(depth)
            while self.ai_player.node < 24000 * depth * big_nums ** 1.25 and depth < 8:
                depth += 1
                self.ai_player.start_search(depth)
        return {1:'Left', 2:'Right', 3:'Up', 4:'Down'}.get(self.ai_player.best_operation,'')

    def new_game(self):
        board_encoded = np.uint64(self.mover.gen_new_num(self.mover.gen_new_num(np.uint64(0))[0])[0])
        board = self.mover.decode_board(board_encoded)
        ai_player = AIPlayer(board)
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

                if positions[0][0] == positions[0][1] and abs(positions[1][0] - positions[1][1]) == 1:
                    self.board[first_position] = 0
                    self.board_encoded = np.uint64(self.mover.encode_board(self.board))
                    self.score += 65536
                    self.do_move('Right')
                    self.died = False

                elif positions[1][0] == positions[1][1] and abs(positions[0][0] - positions[0][1]) == 1:
                    self.board[first_position] = 0
                    self.board_encoded = np.uint64(self.mover.encode_board(self.board))
                    self.score += 65536
                    self.do_move('Down')
                    self.died = False

            if len(self.history) % 931==128:
                print(self.score)
                print(self.board)
                print(os.path.basename(self.path))
                if self.score > 900000:
                    print('**************************************************')
                else:
                    print('                                                  ')

        print(self.score)
        print(self.board)
        print(os.path.basename(self.path), 'died')
        print('--------------------------------------------------')
        record = np.array(self.history, dtype='uint64,uint32,uint8')
        record.tofile(self.path)

    def is_mess(self):
        """检查是否乱阵"""
        board = self.board
        large_tiles = np.sum(self.ai_dispatcher.counts[7:])
        board_flatten = board.flatten()
        if large_tiles < 3:
            return False
        elif large_tiles > 3:
            top4_pos = np.argpartition(board_flatten, -4)[-4:]
            if len(np.unique(board_flatten[top4_pos])) < 4:
                return False
            top4_pos = tuple(sorted(top4_pos))
            return top4_pos not in ((0, 1, 2, 3), (0, 4, 8, 12), (12, 13, 14, 15), (3, 7, 11, 15),
                                    (0, 1, 2, 4), (4, 8, 12, 13), (11, 13, 14, 15), (2, 3, 7, 11),
                                    (0, 1, 4, 8), (8, 12, 13, 14), (7, 11, 14, 15), (1, 2, 3, 7),
                                    (0, 1, 4, 5), (8, 9, 12, 13), (10, 11, 14, 15), (2, 3, 6, 7),
                                    (2, 3, 14, 15), (0, 1, 12, 13), (8, 11, 12, 15), (0, 3, 4, 7))
        else:
            # [[0, 1, 2, 3],
            #  [4, 5, 6, 7],
            #  [8, 9,10,11],
            #  [12,13,14,15]]
            top3_pos = np.argpartition(board_flatten, -3)[-3:]
            if len(np.unique(board_flatten[top3_pos])) < 3:
                return False
            top3_pos = tuple(sorted(top3_pos))
            return top3_pos not in (
                (0, 1, 2), (1, 2, 3), (3, 7, 11), (7, 11, 15), (13, 14, 15), (12, 13, 14), (4, 8, 12), (0, 4, 8),
                (0, 1, 3), (0, 2, 3), (3, 7, 15), (3, 11, 15), (12, 14, 15), (12, 13, 15), (0, 8, 12), (0, 4, 12),
                (0, 1, 12), (0, 3, 4), (0, 3, 7), (2, 3, 15), (3, 14, 15), (11, 12, 15), (8, 12 ,15), (0, 12, 13),
                (0, 1, 4), (2, 3, 7), (11, 14, 15), (8, 12, 13))


def run_test(index):
    ai_test = AItest(fr"C:\Users\Administrator\Desktop\record\{index}")
    ai_test.play()


def main():
    multiprocessing.freeze_support()
    with multiprocessing.Pool(processes=30) as pool:
        pool.map(run_test, range(0, 1800))


if __name__ == "__main__":
    main()
    #run_test(0)
