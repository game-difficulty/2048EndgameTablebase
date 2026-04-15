#include <nanobind/nanobind.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>


#include "AIPlayer.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(ai_core, m) {
    m.def("find_best_egtb_move", &find_best_egtb_move, 
          "board"_a, "table_type"_a);

    m.def("resolve_32768_doubles", &resolve_32768_doubles, nb::arg("board"));
          
    nb::class_<AIPlayer>(m, "AIPlayer")
        .def(nb::init<uint64_t>())
        .def("reset_board", &AIPlayer::reset_board, "new_board"_a)
        .def("clear_cache", &AIPlayer::clear_cache)
        .def("update_spawn_rate", &AIPlayer::update_spawn_rate, "new_rate"_a)
        .def("evaluate", &AIPlayer::evaluate, "board"_a)
        .def("start_search", &AIPlayer::start_search, 
             "depth"_a = 3, 
             nb::call_guard<nb::gil_scoped_release>())
        
        .def_rw("max_d", &AIPlayer::max_d)
        .def_rw("best_operation", &AIPlayer::best_operation)
        .def_rw("board", &AIPlayer::board)
        .def_rw("max_threads", &AIPlayer::max_threads)
        .def_rw("do_check", &AIPlayer::do_check)
        .def_rw("spawn_rate4", &AIPlayer::spawn_rate4)
        .def_rw("prune", &AIPlayer::prune)

        .def_ro("masked_count", &AIPlayer::masked_count)
        .def_ro("fixed_mask", &AIPlayer::fixed_mask)
        .def_ro("dead_score", &AIPlayer::dead_score)

        .def_prop_ro("scores", [](const AIPlayer &self) {
            return std::make_tuple(self.top_scores[0], self.top_scores[1], 
                                   self.top_scores[2], self.top_scores[3]);
        })
        .def_prop_ro("node", [](const AIPlayer &self) {
            return self.node.load();
        })
        .def_prop_rw("stop_search", 
            [](const AIPlayer &self) {
                return self.stop_search.load(std::memory_order_relaxed);
            }, 
            [](AIPlayer &self, bool value) {
                self.stop_search.store(value, std::memory_order_relaxed);
        });

    // =================================================================
    // 4. EvilGen 类导出
    // =================================================================
    nb::class_<EvilGen>(m, "EvilGen")
        .def(nb::init<uint64_t>(), "initial_board"_a)
        .def("reset_board", &EvilGen::reset_board, "new_board"_a)
        .def("start_search", &EvilGen::start_search, 
             "depth"_a = 4, 
             nb::call_guard<nb::gil_scoped_release>())
        .def("dispatcher", &EvilGen::dispatcher, 
             "board"_a, 
             nb::call_guard<nb::gil_scoped_release>())
        .def("gen_new_num", &EvilGen::gen_new_num, 
             "depth"_a = 4, 
             nb::call_guard<nb::gil_scoped_release>())
        
        .def_rw("max_d", &EvilGen::max_d)
        .def_rw("hardest_pos", &EvilGen::hardest_pos)
        .def_rw("hardest_num", &EvilGen::hardest_num)
        .def_rw("board", &EvilGen::board)
        
        .def_prop_ro("node", [](const EvilGen &self) {
            return self.node;
        });
}
