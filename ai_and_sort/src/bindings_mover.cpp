#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/vector.h>

#include "CommonMover.h"
#include "BoardMover.h"
#include "VBoardMover.h"

namespace nb = nanobind;
using namespace nb::literals;

// 定义导出宏
#define EXPORT_MOVER_INTERFACE(submodule, MoverType) \
    submodule.def("move_board", &MoverType::move_board, "board"_a, "direction"_a); \
    submodule.def("move_all_dir", &MoverType::move_all_dir, "board"_a); \
    submodule.def("s_move_board", &MoverType::s_move_board, "board"_a, "direction"_a); \
    submodule.def("s_move_board_all", &MoverType::s_move_board_all, "board"_a); \
    submodule.def("move_left", &MoverType::move_left, "board"_a); \
    submodule.def("move_right", &MoverType::move_right, "board"_a); \
    submodule.def("move_up", &MoverType::move_up, "board"_a); \
    submodule.def("move_down", &MoverType::move_down, "board"_a); \
    submodule.def("s_move_left", &MoverType::s_move_left, "board"_a); \
    submodule.def("s_move_right", &MoverType::s_move_right, "board"_a); \
    submodule.def("s_move_up", &MoverType::s_move_up, "board"_a); \
    submodule.def("s_move_down", &MoverType::s_move_down, "board"_a);

NB_MODULE(mover_core, m) {
    m.doc() = "Independent 2048 Bitboard Mover Module";

    // --- 必须注册 MoveResult，否则 Stubgen 无法解析返回类型 ---
    nb::class_<MoveResult>(m, "MoveResult")
        .def_rw("board", &MoveResult::board)
        .def_rw("score", &MoveResult::score)
        .def_rw("is_valid", &MoveResult::is_valid);

    // 公共工具函数
    m.def("reverse", &reverse_board, "board"_a);
    m.def("canonical_diagonal", &canonical_diagonal, "board"_a);

    m.def("gen_new_num", &gen_new_num, "t"_a, "p"_a = 0.1f);
    m.def("s_gen_new_num", &s_gen_new_num, "t"_a, "p"_a = 0.1f);

    // 标准子模块
    auto m_std = m.def_submodule("std", "Standard 2048 logic");
    EXPORT_MOVER_INTERFACE(m_std, BoardMover);

    // 隔离墙子模块
    auto m_v = m.def_submodule("v", "32768 Wall logic");
    EXPORT_MOVER_INTERFACE(m_v, VBoardMover);
}
