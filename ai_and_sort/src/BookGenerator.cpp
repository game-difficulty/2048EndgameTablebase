#include "BookGenerator.h"
#include "BookGeneratorUtils.h"
#include "BoardMover.h"
#include "VBoardMover.h"
#include "Calculator.h"

#include <omp.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <cstring>
#include <numeric>

#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <iomanip>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace BookGenerator {

    // ========================================================================
    // 内部辅助函数：处理动态正则化
    // ========================================================================
    inline uint64_t apply_canonical(uint64_t board, int symm_mode) {
        switch (static_cast<SymmMode>(symm_mode)) {
            case SymmMode::Full:       return Calculator::canonical_full(board);
            case SymmMode::Diagonal:   return Calculator::canonical_diagonal(board);
            case SymmMode::Horizontal: return Calculator::canonical_horizontal(board);
            case SymmMode::Min33:      return Calculator::canonical_min33(board);
            case SymmMode::Min24:      return Calculator::canonical_min24(board);
            case SymmMode::Min34:      return Calculator::canonical_min34(board);
            case SymmMode::Identity:   
            default:                   return Calculator::canonical_identity(board);
        }
    }


    // ========================================================================
    // 核心函数
    // ========================================================================
    template <typename Mover>
    GenBoardsResult gen_boards(
        const uint64_t* arr0, size_t arr0_size,
        int target,
        const PatternConfig& config,
        uint64_t* hashmap1, size_t hashmap1_size,
        uint64_t* hashmap2, size_t hashmap2_size,
        uint64_t* arr1, uint64_t* arr2,
        size_t capacity, // 预分配数组的容量，用于分配线程起始点
        int num_threads,
        bool do_check,
        bool isfree
    ) {
        uint64_t target_stacked = (uint64_t)target * 0x1111111111111111ULL;
        uint64_t hashmask1 = hashmap1_size - 1;
        uint64_t hashmask2 = hashmap2_size - 1;

        // 为每个线程分配固定的写入起始位置
        std::vector<size_t> starts(num_threads);
        std::vector<size_t> c1(num_threads);
        std::vector<size_t> c2(num_threads);
        for (int i = 0; i < num_threads; ++i) {
            starts[i] = (capacity / num_threads) * i;
        }

        // chunking 逻辑，防止数据倾斜导致某线程耗时过长
        size_t chunk_size = std::min<size_t>(1000000ULL, arr0_size / (num_threads * 5) + 1) * num_threads;
        size_t chunks_count = (arr0_size + chunk_size - 1) / chunk_size;

        #pragma omp parallel num_threads(num_threads)
        {
            int s = omp_get_thread_num();
            size_t c1t = starts[s];
            size_t c2t = starts[s];

            for (size_t chunk = 0; chunk < chunks_count; ++chunk) {
                size_t chunk_start = chunk * chunk_size;
                size_t chunk_end = std::min(chunk_start + chunk_size, arr0_size);

                size_t thread_start = chunk_start + s * (chunk_size / num_threads);
                size_t thread_end = thread_start + (chunk_size / num_threads);
                size_t start = std::max(thread_start, chunk_start);
                size_t end = std::min(thread_end, chunk_end);

                // 确保有效区间存在
                if (start >= end) continue;

                for (size_t b = start; b < end; ++b) {
                    uint64_t t = arr0[b];
                    
                    if (do_check && is_success(t, config.success_mask, target_stacked)) {
                        continue;
                    }

                    for (int i = 0; i < 16; ++i) {
                        // 检查第 i 位置是否为空
                        if (((t >> (4 * i)) & 0xF) == 0) {
                            
                            // -----------------------------------------
                            // 填入数字 2 (值为 1)
                            // -----------------------------------------
                            uint64_t t1 = t | (1ULL << (4 * i));
                            auto moves1 = Mover::move_all_dir(t1);
                            
                            // 展开 4 个方向的结果
                            uint64_t nb1[4] = {std::get<0>(moves1), std::get<1>(moves1), std::get<2>(moves1), std::get<3>(moves1)};
                            for (int d = 0; d < 4; ++d) {
                                uint64_t newt = nb1[d];
                                if (newt != t1 && is_pattern(newt, config.masks)) {
                                    newt = apply_canonical(newt, config.symm_mode);
                                    size_t hashed = BookGeneratorUtils::hash_board(newt) & hashmask1;
                                    
                                    // 无锁哈希预筛 (允许数据竞争)
                                    if (hashmap1[hashed] != newt) {
                                        hashmap1[hashed] = newt;
                                        arr1[c1t++] = newt;
                                    }
                                }
                            }

                            // -----------------------------------------
                            // 填入数字 4 (值为 2)
                            // -----------------------------------------
                            uint64_t t2 = t | (2ULL << (4 * i));
                            auto moves2 = Mover::move_all_dir(t2);
                            
                            uint64_t nb2[4] = {std::get<0>(moves2), std::get<1>(moves2), std::get<2>(moves2), std::get<3>(moves2)};
                            for (int d = 0; d < 4; ++d) {
                                uint64_t newt = nb2[d];
                                if (newt != t2 && is_pattern(newt, config.masks)) {
                                    newt = apply_canonical(newt, config.symm_mode);
                                    size_t hashed = BookGeneratorUtils::hash_board(newt) & hashmask2;
                                    
                                    if (hashmap2[hashed] != newt) {
                                        hashmap2[hashed] = newt;
                                        arr2[c2t++] = newt;
                                    }
                                }
                            }
                        }
                    } // end for i
                } // end for b
            } // end for chunk
            
            c1[s] = c1t;
            c2[s] = c2t;
        }

        std::vector<size_t> counts1_out(num_threads);
        std::vector<size_t> counts2_out(num_threads);
        for (int i = 0; i < num_threads; ++i) {
            counts1_out[i] = c1[i] - starts[i];
            counts2_out[i] = c2[i] - starts[i];
        }

        GenBoardsResult res;
        res.total_arr1 = BookGeneratorUtils::merge_inplace(arr1, c1, starts);
        res.total_arr2 = BookGeneratorUtils::merge_inplace(arr2, c2, starts);
        res.counts1 = std::move(counts1_out); // 导出
        res.counts2 = std::move(counts2_out); // 导出

        return res;
    }


    // ========================================================================
    // 核心函数：轻量级并行生成引擎 (无哈希预筛)
    // ========================================================================
    template <typename Mover>
    std::tuple<std::vector<uint64_t>, std::vector<uint64_t>> 
    gen_boards_simple(
        const uint64_t* arr0, size_t arr0_size,
        int target,
        const PatternConfig& config,
        bool do_check,
        bool isfree
    ) {
        uint64_t target_stacked = (uint64_t)target * 0x1111111111111111ULL;
        
        // 严格对照 Python 逻辑预计算最大长度
        size_t length = isfree ? std::max<size_t>(arr0_size * 8, 19999999ULL) 
                               : std::max<size_t>(arr0_size * 6, 9999999ULL);

        std::vector<uint64_t> arr_out1;
        std::vector<uint64_t> arr_out2;

        // 严格限制仅使用 2 个线程：一个负责 p=1 (填2), 另一个负责 p=2 (填4)
        #pragma omp parallel for num_threads(2)
        for (int p = 1; p <= 2; ++p) {
            
            // 使用 reserve 预分配物理内存，避免零初始化开销 (等价于 np.empty)
            // 使用 push_back 自带索引递增功能 (等价于 ct++)
            std::vector<uint64_t> arr;
            arr.reserve(length);

            for (size_t b = 0; b < arr0_size; ++b) {
                uint64_t t = arr0[b];
                
                // 如果当前棋盘状态已经符合成功条件，跳过
                if (do_check && is_success(t, config.success_mask, target_stacked)) {
                    continue; 
                }
                
                for (int i = 0; i < 16; ++i) {
                    // 检查第 i 位置是否为空
                    if (((t >> (4 * i)) & 0xF) == 0) {
                        
                        // 用 p (1或2) 填充当前空位
                        uint64_t t1 = t | ((uint64_t)p << (4 * i)); 
                        
                        // 尝试所有四个方向上的移动
                        auto moves = Mover::move_all_dir(t1);
                        uint64_t new_boards[4] = {std::get<0>(moves), std::get<1>(moves), std::get<2>(moves), std::get<3>(moves)};
                        
                        for (int d = 0; d < 4; ++d) {
                            uint64_t newt = new_boards[d];
                            if (newt != t1 && is_pattern(newt, config.masks)) {
                                arr.push_back(apply_canonical(newt, config.symm_mode));
                            }
                        }
                    }
                }
            }

            // 将局部结果通过 std::move 零拷贝转移给外部输出变量 (等价于 arrs[p-1] = arr[:ct])
            if (p == 1) {
                arr_out1 = std::move(arr);
            } else {
                arr_out2 = std::move(arr);
            }
        }
        
        // 返回包含可能的新棋盘状态的两个 vector
        return {std::move(arr_out1), std::move(arr_out2)};
    }

    // ========================================================================
    // 显式模板实例化
    // ========================================================================
    template GenBoardsResult gen_boards<BoardMover>(const uint64_t*, size_t, int, const PatternConfig&, uint64_t*, size_t, uint64_t*, size_t, uint64_t*, uint64_t*, size_t, int, bool, bool);
    template GenBoardsResult gen_boards<VBoardMover>(const uint64_t*, size_t, int, const PatternConfig&, uint64_t*, size_t, uint64_t*, size_t, uint64_t*, uint64_t*, size_t, int, bool, bool);

    template std::tuple<std::vector<uint64_t>, std::vector<uint64_t>> gen_boards_simple<BoardMover>(const uint64_t*, size_t, int, const PatternConfig&, bool, bool);
    template std::tuple<std::vector<uint64_t>, std::vector<uint64_t>> gen_boards_simple<VBoardMover>(const uint64_t*, size_t, int, const PatternConfig&, bool, bool);

    // ========================================================================
    // 宏观分段调度 (对应 gen_boards_big)
    // ========================================================================
    std::tuple<std::vector<std::vector<uint64_t>>, std::vector<std::vector<uint64_t>>, std::vector<uint64_t>, std::vector<uint64_t>>
    gen_boards_big(
        const std::vector<uint64_t>& arr0, int target, const PatternConfig& config,
        std::vector<uint64_t>& hashmap1, std::vector<uint64_t>& hashmap2, int num_threads,
        std::vector<std::vector<double>>& lf_list, double& lf_multiplier, bool do_check, bool isfree, bool is_variant
    ) {
        size_t segs_count = lf_list.size();
        std::vector<std::vector<uint64_t>> arr1s, arr2s;
        std::vector<size_t> act_len1(segs_count * num_threads), act_len2(segs_count * num_threads);
        std::vector<size_t> seg_limits = allocate_seg(lf_list, arr0.size());

        for (size_t s_idx = 0; s_idx < segs_count; ++s_idx) {
            size_t s_start = seg_limits[s_idx], s_end = seg_limits[s_idx+1], s_len = s_end - s_start;
            
            double lf = predict_next_length_factor_quadratic(lf_list[s_idx]);
            lf *= (s_len > 1e8) ? 1.25 : 1.5;
            lf *= lf_multiplier;

            size_t cap = std::max<size_t>(isfree ? 99999999 : 69999999, static_cast<size_t>(s_len * lf));
            auto a1_ptr = std::make_unique<uint64_t[]>(cap);
            auto a2_ptr = std::make_unique<uint64_t[]>(cap);

            GenBoardsResult res = is_variant ? 
                gen_boards<VBoardMover>(arr0.data()+s_start, s_len, target, config, hashmap1.data(), hashmap1.size(), hashmap2.data(), hashmap2.size(), a1_ptr.get(), a2_ptr.get(), cap, num_threads, do_check, isfree) :
                gen_boards<BoardMover>(arr0.data()+s_start, s_len, target, config, hashmap1.data(), hashmap1.size(), hashmap2.data(), hashmap2.size(), a1_ptr.get(), a2_ptr.get(), cap, num_threads, do_check, isfree);

            validate_length_and_balance(s_len, res.total_arr2, res.total_arr1, res.counts1, res.counts2, lf, true);
            
            std::copy(res.counts1.begin(), res.counts1.end(), act_len1.begin() + s_idx * num_threads);
            std::copy(res.counts2.begin(), res.counts2.end(), act_len2.begin() + s_idx * num_threads);
            
            lf_list[s_idx].erase(lf_list[s_idx].begin());
            lf_list[s_idx].push_back(static_cast<double>(res.total_arr2) / (1.0 + s_len));

            BookGeneratorUtils::sort_array(a1_ptr.get(), res.total_arr1, num_threads);
            BookGeneratorUtils::sort_array(a2_ptr.get(), res.total_arr2, num_threads);
            arr1s.emplace_back(a1_ptr.get(), a1_ptr.get() + BookGeneratorUtils::parallel_unique(a1_ptr.get(), res.total_arr1, num_threads));
            arr2s.emplace_back(a2_ptr.get(), a2_ptr.get() + BookGeneratorUtils::parallel_unique(a2_ptr.get(), res.total_arr2, num_threads));
        }

        auto [new_list, new_mult] = update_parameters_big(act_len2, act_len1, num_threads, lf_list);
        lf_list = new_list; lf_multiplier = new_mult;

        return {std::move(arr1s), std::move(arr2s), std::move(hashmap1), std::move(hashmap2)};
    }


    // ========================================================================
    // 主生成管线 (对应 generate_process)
    // ========================================================================
    std::tuple<bool, std::vector<uint64_t>, std::vector<uint64_t>> 
    generate_process(const std::vector<uint64_t>& arr_init, const PatternConfig& config, int target, int steps, int tile_sum, const std::string& pathname, bool isfree, bool is_variant) {
        bool started = false;
        std::vector<uint64_t> d0, d1, h1, h2;
        int n_threads = omp_get_max_threads();
        auto p = initialize_parameters_internal(n_threads, pathname, isfree);

        int t_val = 1 << target;
        int d_check_step = (t_val / 2) - (tile_sum % t_val) / 2;

        for (int i = 1; i < steps - 1; ++i) {
            auto [run, rd0, rd1] = handle_restart(i, pathname, arr_init, started);
            if (!rd0.empty()) d0 = std::move(rd0);
            if (!rd1.empty()) d1 = std::move(rd1);
            if (!run) continue;

            started = true;
            bool do_check = (i > d_check_step);
            double t_start = get_current_time();

            if (d0.size() < p.segment_size) {
                std::vector<uint64_t> d1t, d2;
                if (d0.size() < 10000 || arr_init[0] == 0xffff00000000ffffULL) {
                    std::tie(d1t, d2) = is_variant ? gen_boards_simple<VBoardMover>(d0.data(), d0.size(), target, config, do_check, isfree) : gen_boards_simple<BoardMover>(d0.data(), d0.size(), target, config, do_check, isfree);
                } else {
                    if (h1.empty()) { update_hashmap_length(h1, d0.size()); update_hashmap_length(h2, d0.size()); }
                    size_t cap = std::max<size_t>(isfree ? 99999999 : 69999999, d0.size() * p.length_factor * p.length_factor_multiplier);
                    auto a1 = std::make_unique<uint64_t[]>(cap), a2 = std::make_unique<uint64_t[]>(cap);
                    GenBoardsResult res = is_variant ? 
                        gen_boards<VBoardMover>(d0.data(), d0.size(), target, config, h1.data(), h1.size(), h2.data(), h2.size(), a1.get(), a2.get(), cap, n_threads, do_check, isfree) :
                        gen_boards<BoardMover>(d0.data(), d0.size(), target, config, h1.data(), h1.size(), h2.data(), h2.size(), a1.get(), a2.get(), cap, n_threads, do_check, isfree);

                    validate_length_and_balance(d0.size(), res.total_arr2, res.total_arr1, res.counts1, res.counts2, p.length_factor, false);
                    BookGeneratorUtils::sort_array(a1.get(), res.total_arr1, n_threads);
                    BookGeneratorUtils::sort_array(a2.get(), res.total_arr2, n_threads);
                    d1t.assign(a1.get(), a1.get() + BookGeneratorUtils::parallel_unique(a1.get(), res.total_arr1, n_threads));
                    d2.assign(a2.get(), a2.get() + BookGeneratorUtils::parallel_unique(a2.get(), res.total_arr2, n_threads));
                    
                    auto [nf, nl] = update_parameters(d0.size(), d2.size(), p.length_factors, p.lf_path);
                    p.length_factors = nf; p.length_factors_list = nl;
                    p.length_factor_multiplier = static_cast<double>(*std::max_element(res.counts2.begin(), res.counts2.end())) / (std::accumulate(res.counts2.begin(), res.counts2.end(), 0ULL) / n_threads);
                }
                std::vector<uint64_t> pivots;
                for(int pt=1; pt<n_threads; ++pt) pivots.push_back(d0.empty() ? pt*(1ULL<<50)/n_threads : d0[pt*d0.size()/n_threads]);
                std::vector<std::vector<uint64_t>> d1_in = {std::move(d1), std::move(d1t)};
                d1 = BookGeneratorUtils::concatenate(BookGeneratorUtils::merge_deduplicate_all(d1_in, pivots, n_threads));
                d0 = std::move(d1); d1 = std::move(d2);
                if (!h1.empty()) { update_hashmap_length(h1, d1.size()); update_hashmap_length(h2, d1.size()); }
                log_performance(i, t_start, get_current_time(), get_current_time(), get_current_time(), d1.size());
            } else {
                if (h1.empty()) { size_t h_cap = BookGeneratorUtils::largest_power_of_2(20971520ULL * static_cast<size_t>(get_system_memory_gb()*0.75)); h1.assign(h_cap, 0); h2.assign(h_cap, 0); }
                auto [d1s, d2s, nh1, nh2] = gen_boards_big(d0, target, config, h1, h2, n_threads, p.length_factors_list, p.length_factor_multiplier, do_check, isfree, is_variant);
                h1 = std::move(nh1); h2 = std::move(nh2);
                std::vector<uint64_t> pivots;
                for(int pt=1; pt<n_threads; ++pt) pivots.push_back(d0[pt*d0.size()/n_threads]);
                d0.clear(); d1s.push_back(std::move(d1));
                d1 = BookGeneratorUtils::concatenate(BookGeneratorUtils::merge_deduplicate_all(d2s, pivots, n_threads));
                d0 = BookGeneratorUtils::concatenate(BookGeneratorUtils::merge_deduplicate_all(d1s, pivots, n_threads));
                p.length_factors = harmonic_mean_by_column(p.length_factors_list);
                save_length_factors_to_disk(p.lf_path, p.length_factors_list);
                log_performance(i, t_start, get_current_time(), get_current_time(), get_current_time(), d0.size());
            }
            std::ofstream out(pathname + std::to_string(i), std::ios::binary);
            out.write(reinterpret_cast<const char*>(d0.data()), d0.size()*sizeof(uint64_t));
            std::swap(h1, h2);
        }
        return {started, std::move(d0), std::move(d1)};
    }

    double predict_next_length_factor_quadratic(const std::vector<double>& length_factors) {
        if (length_factors.empty()) return 3.0;
        
        bool all_close = true;
        double last = length_factors.back();
        for (double val : length_factors) {
            if (std::abs(val - last) > 0.1) {
                all_close = false;
                break;
            }
        }
        if (all_close) return last;

        double next_val = solve_quadratic_and_predict(length_factors);
        double sum = std::accumulate(length_factors.begin(), length_factors.end(), 0.0);
        double mean = sum / static_cast<double>(length_factors.size());

        double result = std::max(next_val, mean);
        result = std::min(result, last * 2.5);
        
        return std::isnan(result) ? 3.0 : result;
    }

    // 对应 split_length_factor_list
    std::vector<std::vector<double>> split_length_factor_list(const std::vector<std::vector<double>>& length_factor_list) {
        if (length_factor_list.empty()) return {};

        std::vector<std::vector<double>> new_list;
        
        // 处理第一个元素: i * 1.5
        std::vector<double> first = length_factor_list[0];
        for (double& val : first) val *= 1.5;
        new_list.push_back(first);

        // 复制两遍
        for (const auto& sublist : length_factor_list) {
            new_list.push_back(sublist);
            new_list.push_back(sublist);
        }
        
        // pop_back
        if (!new_list.empty()) new_list.pop_back();
        
        return new_list;
    }

    // 对应 reverse_split_length_factor_list
    std::vector<std::vector<double>> reverse_split_length_factor_list(const std::vector<std::vector<double>>& length_factor_list) {
        std::vector<std::vector<double>> new_list;
        for (const auto& sublist : length_factor_list) {
            std::vector<double> sliced;
            for (size_t i = 0; i < sublist.size(); i += 2) {
                sliced.push_back(sublist[i]);
            }
            new_list.push_back(sliced);
        }
        return new_list;
    }

    // 对应 harmonic_mean_by_column
    std::vector<double> harmonic_mean_by_column(const std::vector<std::vector<double>>& matrix) {
        if (matrix.empty() || matrix[0].empty()) return {};

        size_t num_cols = matrix[0].size();
        std::vector<double> harmonic_means;
        harmonic_means.reserve(num_cols);

        for (size_t col = 0; col < num_cols; ++col) {
            std::vector<double> non_zero_values;
            for (const auto& row : matrix) {
                if (col < row.size() && row[col] != 0.0) {
                    // 修正：使用 push_back 替代 append
                    non_zero_values.push_back(row[col]);
                }
            }

            if (non_zero_values.empty()) {
                harmonic_means.push_back(1.5);
                continue;
            }

            double reciprocals_sum = 0.0;
            for (double val : non_zero_values) {
                reciprocals_sum += 1.0 / val;
            }
            harmonic_means.push_back(static_cast<double>(non_zero_values.size()) / reciprocals_sum);
        }
        return harmonic_means;
    }

    // 对应 allocate_seg
    std::vector<size_t> allocate_seg(const std::vector<std::vector<double>>& length_factors_list, size_t arr_length) {
        if (length_factors_list.size() <= 1) {
            return {0, arr_length};
        }

        size_t num_segs = length_factors_list.size();
        std::vector<double> weights;
        weights.reserve(num_segs);
        double total_weight = 0;

        for (const auto& list : length_factors_list) {
            double f = list.empty() ? 1.5 : list.back();
            double w = 1.0 / (f + 0.2);
            weights.push_back(w);
            total_weight += w;
        }

        std::vector<size_t> result_segs;
        result_segs.reserve(num_segs + 1);
        result_segs.push_back(0);
        
        double cumulative_weight = 0;
        for (size_t i = 0; i < num_segs - 1; ++i) {
            cumulative_weight += weights[i];
            size_t pos = static_cast<size_t>((cumulative_weight / total_weight) * static_cast<double>(arr_length));
            result_segs.push_back(pos);
        }
        
        result_segs.push_back(arr_length);
        return result_segs;
    }

    // ========================================================================
    // 内部辅助函数：获取系统物理内存 (等价于 psutil.virtual_memory().total)
    // ========================================================================
    double get_system_memory_gb() {
    #ifdef _WIN32
        MEMORYSTATUSEX status;
        status.dwLength = sizeof(status);
        GlobalMemoryStatusEx(&status);
        return static_cast<double>(status.ullTotalPhys) / (1024.0 * 1024.0 * 1024.0);
    #else
        long pages = sysconf(_SC_PHYS_PAGES);
        long page_size = sysconf(_SC_PAGE_SIZE);
        return static_cast<double>(pages * page_size) / (1024.0 * 1024.0 * 1024.0);
    #endif
    }

    // ========================================================================
    // 哈希表动态扩容 (update_hashmap_length)
    // ========================================================================
    void update_hashmap_length(std::vector<uint64_t>& hashmap, size_t current_arr_size) {
        size_t length = std::max<size_t>(BookGeneratorUtils::largest_power_of_2(current_arr_size), 1048576ULL);
        
        if (hashmap.size() < length) {
            size_t old_size = hashmap.size();
            if (old_size > 0) {
                // 扩容并镜像复制前半部分 (模拟 np.empty + 切片赋值)
                std::vector<uint64_t> new_hashmap(old_size * 2);
                std::memcpy(new_hashmap.data(), hashmap.data(), old_size * sizeof(uint64_t));
                std::memcpy(new_hashmap.data() + old_size, hashmap.data(), old_size * sizeof(uint64_t));
                hashmap = std::move(new_hashmap);
            } else {
                // 初始化必须为 0，防止野指针误判
                hashmap.assign(length, 0); 
            }
        }
    }

    // ========================================================================
    // 断点续传逻辑 (handle_restart)
    // ========================================================================
    std::tuple<bool, std::vector<uint64_t>, std::vector<uint64_t>> 
    handle_restart(int step_i, const std::string& pathname, const std::vector<uint64_t>& arr_init, bool started) {
        auto path_i = pathname + std::to_string(step_i);
        auto path_i_plus_1 = pathname + std::to_string(step_i + 1);
        auto path_i_minus_1 = pathname + std::to_string(step_i - 1);

        // 检查是否存在当前步或下一步的缓存文件
        if ((fs::exists(path_i_plus_1) && fs::exists(path_i)) ||
            (fs::exists(path_i_plus_1 + ".book") && fs::exists(path_i)) ||
            (fs::exists(path_i_plus_1 + ".z") && fs::exists(path_i)) ||
            fs::exists(path_i + ".book") ||
            fs::exists(path_i + ".z") ||
            fs::exists(path_i + ".book.7z") ||
            fs::exists(path_i + ".7z")) {
            
            std::cout << "[DEBUG] skipping step " << step_i << std::endl;
            return {false, {}, {}};
        }

        if (step_i == 1) {
            // 第一步：持久化初始局面 arr_init
            std::ofstream out(path_i_minus_1, std::ios::binary);
            if (out) {
                out.write(reinterpret_cast<const char*>(arr_init.data()), arr_init.size() * sizeof(uint64_t));
            }
            return {true, arr_init, std::vector<uint64_t>()};
        } 
        else if (!started) {
            // 断点重启：从磁盘读取上一层和本层的 .book 原始二进制文件
            auto read_binary = [](const std::string& p) {
                std::vector<uint64_t> data;
                std::ifstream file(p, std::ios::binary | std::ios::ate);
                if (file) {
                    size_t size = file.tellg();
                    file.seekg(0, std::ios::beg);
                    data.resize(size / sizeof(uint64_t));
                    file.read(reinterpret_cast<char*>(data.data()), size);
                }
                return data;
            };

            std::vector<uint64_t> d0 = read_binary(path_i_minus_1);
            std::vector<uint64_t> d1 = read_binary(path_i);
            return {true, d0, d1};
        }

        // 如果已经在循环中，不需要重新加载
        return {true, {}, {}};
    }

    namespace {

    // ========================================================================
    // 性能日志打印 (log_performance)
    // ========================================================================
    void log_performance(int step_i, double t0, double t1, double t2, double t3, size_t d1_len) {
        if (t3 > t0) {
            double mbps = (d1_len / (t3 - t0)) / 1000000.0;
            double gen_ratio = (t1 - t0) / (t3 - t0);
            double sort_ratio = (t2 - t1) / (t3 - t0);
            double dedup_ratio = (t3 - t2) / (t3 - t0);
            
            std::cout << "[DEBUG] step " << step_i << " generated: " << std::fixed << std::setprecision(2) << mbps << " mbps\n";
            std::cout << "[DEBUG] generate/sort/deduplicate: " 
                    << gen_ratio << "/" << sort_ratio << "/" << dedup_ratio << "\n\n";
        }
    }

    // ========================================================================
    // 长度与负载均衡校验 (validate_length_and_balance)
    // ========================================================================
    void validate_length_and_balance(size_t len_d0, size_t len_d2, size_t len_d1t, 
                                    const std::vector<size_t>& counts1, const std::vector<size_t>& counts2, 
                                    double length_factor, bool isbig) {
        if (len_d0 < 1999999 || len_d2 < 1999999) return;

        size_t max_c1 = *std::max_element(counts1.begin(), counts1.end());
        size_t max_c2 = *std::max_element(counts2.begin(), counts2.end());
        size_t sum_c1 = std::accumulate(counts1.begin(), counts1.end(), 0ULL);
        size_t sum_c2 = std::accumulate(counts2.begin(), counts2.end(), 0ULL);
        
        size_t length_needed = std::max(max_c1, max_c2) * counts1.size();
        double length_factor_actual = static_cast<double>(length_needed) / static_cast<double>(len_d0);
        size_t allocated_length = std::max<size_t>(69999999ULL, static_cast<size_t>(len_d0 * length_factor));
        
        bool is_valid_length = length_needed <= allocated_length;

        if (!is_valid_length) {
            std::cerr << "[CRITICAL] length multiplier " << length_factor 
                    << ", need " << length_factor_actual << "\n";
            throw std::out_of_range("The length multiplier is not big enough. Please restart.");
        }

        if (!isbig) {
            std::cout << "[DEBUG] length " << len_d1t << ", " << len_d2 
                    << ", Using " << length_factor << ", Need " << length_factor_actual << "\n";
        } else if (len_d0 > 0) {
            std::cout << "[DEBUG] length " << len_d1t << ", " << len_d2 
                    << ", Using " << length_factor << ", Need " << length_factor_actual << "\n";
        }
    }

    // ========================================================================
    // 动态参数更新与磁盘落地 (update_parameters)
    // ========================================================================
    void save_length_factors(const std::string& path, const std::vector<std::vector<double>>& lists) {
        std::ofstream out(path);
        if (out) {
            for (const auto& list : lists) {
                for (size_t i = 0; i < list.size(); ++i) {
                    out << std::fixed << std::setprecision(6) << list[i];
                    if (i != list.size() - 1) out << ",";
                }
                out << "\n";
            }
        }
    }

    std::tuple<std::vector<double>, std::vector<std::vector<double>>> 
    update_parameters(size_t len_d0, size_t len_d2, std::vector<double> length_factors, const std::string& path) {
        std::vector<double> new_factors(length_factors.begin() + 1, length_factors.end());
        new_factors.push_back(static_cast<double>(len_d2) / static_cast<double>(len_d0 + 1));
        
        std::vector<std::vector<double>> list = {new_factors};
        save_length_factors(path, list);
        
        return {new_factors, list};
    }

    // ========================================================================
    // 宏观大数组细分控制 (update_parameters_big)
    // ========================================================================
    std::tuple<std::vector<std::vector<double>>, double>
    update_parameters_big(const std::vector<size_t>& actual_lengths2, const std::vector<size_t>& actual_lengths1, 
                        int n, std::vector<std::vector<double>> length_factors_list) {
        
        auto get_stats = [](const std::vector<size_t>& arr) {
            double sum = static_cast<double>(std::accumulate(arr.begin(), arr.end(), 0ULL));
            double max_val = static_cast<double>(*std::max_element(arr.begin(), arr.end()));
            double mean = sum / arr.size();
            return std::make_tuple(sum, max_val, mean);
        };

        auto [sum2, max2, mean2_val] = get_stats(actual_lengths2);
        auto [sum1, max1, mean1_val] = get_stats(actual_lengths1);

        double max_percent2 = max2 / sum2;
        double max_percent1 = max1 / sum1;
        double mean_percent2 = 1.0 / actual_lengths2.size(); 
        double mean_percent1 = 1.0 / actual_lengths1.size(); 

        double length_factor_multiplier = std::max(max_percent2 / mean_percent2, max_percent1 / mean_percent1);

        double ram_gb = std::round(get_system_memory_gb());
        double threshold_gb = ram_gb * 0.75;

        // 细分操作
        if (mean2_val * n > 20971520.0 * threshold_gb) {
            length_factors_list = split_length_factor_list(length_factors_list);
        }
        // 逆向细分（合并）
        if (mean2_val * n < 524288.0 * threshold_gb) {
            length_factors_list = reverse_split_length_factor_list(length_factors_list);
        }

        return {length_factors_list, length_factor_multiplier};
    }

    void save_length_factors_to_disk(const std::string& path, const std::vector<std::vector<double>>& matrix) {
        std::ofstream out(path, std::ios::trunc);
        if (!out) return;
        out << std::fixed << std::setprecision(6);
        for (const auto& row : matrix) {
            for (size_t i = 0; i < row.size(); ++i) {
                out << row[i] << (i == row.size() - 1 ? "" : ",");
            }
            out << "\n";
        }
    }

    } // end anonymous namespace

} // namespace BookGenerator


namespace {
    double solve_quadratic_and_predict(const std::vector<double>& y) {
        size_t n = y.size();
        if (n < 3) return y.empty() ? 3.0 : y.back();

        double s0 = static_cast<double>(n);
        double s1 = 0, s2 = 0, s3 = 0, s4 = 0;
        double sy = 0, sxy = 0, sx2y = 0;

        for (size_t i = 0; i < n; ++i) {
            double x = static_cast<double>(i);
            double x2 = x * x;
            double x3 = x2 * x;
            double x4 = x3 * x;

            s1 += x; s2 += x2; s3 += x3; s4 += x4;
            sy += y[i];
            sxy += x * y[i];
            sx2y += x2 * y[i];
        }

        auto det3x3 = [](double a, double b, double c, 
                        double d, double e, double f, 
                        double g, double h, double i) {
            return a*e*i + b*f*g + c*d*h - c*e*g - b*d*i - a*f*h;
        };

        double D = det3x3(s4, s3, s2, s3, s2, s1, s2, s1, s0);
        if (std::abs(D) < 1e-9) return y.back(); 

        double Da = det3x3(sx2y, s3, s2, sxy, s2, s1, sy, s1, s0);
        double Db = det3x3(s4, sx2y, s2, s3, sxy, s1, s2, sy, s0);
        double Dc = det3x3(s4, s3, sx2y, s3, s2, sxy, s2, s1, sy);

        double a = Da / D;
        double b = Db / D;
        double c = Dc / D;

        double next_x = static_cast<double>(n);
        return a * next_x * next_x + b * next_x + c;
    }

    // ========================================================================
    // 获取当前时间戳 (等价于 time.time())
    // ========================================================================
    double get_current_time() {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(now.time_since_epoch()).count();
    }

    // ========================================================================
    // 读取 Length Factors 配置文件 (等价于 np.loadtxt)
    // ========================================================================
    std::vector<std::vector<double>> load_length_factors(const std::string& path, double default_val) {
        std::vector<std::vector<double>> result;
        std::ifstream file(path);
        if (file) {
            std::string line;
            while (std::getline(file, line)) {
                std::vector<double> row;
                size_t pos = 0;
                while ((pos = line.find(',')) != std::string::npos) {
                    row.push_back(std::stod(line.substr(0, pos)));
                    line.erase(0, pos + 1);
                }
                if (!line.empty()) row.push_back(std::stod(line));
                if (!row.empty()) result.push_back(row);
            }
        }
        if (result.empty()) {
            result.push_back({default_val, default_val, default_val});
        }
        return result;
    }

    // ========================================================================
    // 初始化参数与内存阈值
    // ========================================================================
    struct InitParams {
        double length_factor;
        std::vector<double> length_factors;
        std::vector<std::vector<double>> length_factors_list;
        std::string lf_path;
        double length_factor_multiplier;
        size_t segment_size;
    };

    InitParams initialize_parameters_internal(int num_threads, const std::string& pathname, bool isfree) {
        InitParams p;
        fs::path p_path(pathname);
        p.lf_path = (p_path.parent_path() / "length_factors_list.txt").string();
        p.length_factor_multiplier = 2.0;

        // 尝试加载历史预测数据
        std::ifstream file(p.lf_path);
        if (file) {
            std::string line;
            while (std::getline(file, line)) {
                std::vector<double> row;
                std::stringstream ss(line);
                std::string val;
                while (std::getline(ss, val, ',')) row.push_back(std::stod(val));
                if (!row.empty()) p.length_factors_list.push_back(row);
            }
        }

        if (p.length_factors_list.empty()) {
            double default_lf = 3.2;
            p.length_factors = {default_lf, default_lf, default_lf};
            p.length_factors_list = {p.length_factors};
            p.length_factor = default_lf;
        } else {
            p.length_factors = BookGenerator::harmonic_mean_by_column(p.length_factors_list);
            p.length_factor = BookGenerator::predict_next_length_factor_quadratic(p.length_factors) * 1.2;
        }

        double ram_gb = std::round(BookGenerator::get_system_memory_gb());
        size_t base_unit = isfree ? 5120000 : 8192000;
        p.segment_size = base_unit * static_cast<size_t>(std::max(1.0, ram_gb - 5.0));

        return p;
    }

} // end anonymous namespace