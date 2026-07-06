#include "AIPlayer - wasm.h"

#include <emscripten/bind.h>

using namespace emscripten;

val wrap_evil_gen_new_num(EvilGen& self, int depth) {
    auto result = self.gen_new_num(depth);
    val output = val::array();
    output.set(0, std::get<0>(result));
    output.set(1, std::get<1>(result));
    output.set(2, std::get<2>(result));
    return output;
}
EMSCRIPTEN_BINDINGS(evil_core_module) {
    class_<EvilGen>("EvilGen")
        .constructor<uint64_t>()
        .function("reset_board", &EvilGen::reset_board)
        .function("evaluate", &EvilGen::evaluate)
        .function("dispatcher", &EvilGen::dispatcher)
        .function("start_search", &EvilGen::start_search)
        .function("gen_new_num", &wrap_evil_gen_new_num)
        .property("max_d", &EvilGen::max_d)
        .property("hardest_pos", &EvilGen::hardest_pos)
        .property("hardest_num", &EvilGen::hardest_num)
        .property("board", &EvilGen::board)
        .property("node", &EvilGen::node)
        .property("dead_score", &EvilGen::dead_score);
}
