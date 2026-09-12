#pragma once

#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace BC::detail {

// One persistent worker, one outstanding job, no unbounded queue. A job owns
// its buffers until wait() observes completion (including Direct IO's join).
class BCFamilyTempWorker {
public:
    explicit BCFamilyTempWorker(bool enabled) : enabled_(enabled) {
        if (enabled_) thread_ = std::thread([this] { loop(); });
    }
    BCFamilyTempWorker(const BCFamilyTempWorker &) = delete;
    BCFamilyTempWorker &operator=(const BCFamilyTempWorker &) = delete;
    ~BCFamilyTempWorker() {
        try { wait(); } catch (...) {}
        if (enabled_) {
            { std::lock_guard<std::mutex> lock(mutex_); stopping_ = true; }
            ready_.notify_one();
            thread_.join();
        }
    }
    bool enabled() const { return enabled_; }

    void submit(std::function<void()> job) {
        if (!enabled_) { job(); return; }
        std::lock_guard<std::mutex> lock(mutex_);
        if (error_) std::rethrow_exception(error_);
        if (pending_) throw std::logic_error("BC temp worker already has a job");
        job_ = std::move(job);
        pending_ = true;
        finished_ = false;
        ready_.notify_one();
    }
    void wait() {
        if (!enabled_) return;
        std::unique_lock<std::mutex> lock(mutex_);
        if (pending_) {
            done_.wait(lock, [this] { return finished_; });
            pending_ = false;
        }
        // An IO failure poisons the worker; never resume from partial writes.
        if (error_) std::rethrow_exception(error_);
    }

private:
    void loop() {
        for (;;) {
            std::function<void()> job;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                ready_.wait(lock, [this] { return stopping_ || static_cast<bool>(job_); });
                if (stopping_) return;
                job.swap(job_);
            }
            std::exception_ptr failure;
            try { job(); } catch (...) { failure = std::current_exception(); }
            // Destroy captures before exposing completion to their owners.
            job = {};
            {
                std::lock_guard<std::mutex> lock(mutex_);
                error_ = failure;
                finished_ = true;
            }
            done_.notify_one();
        }
    }
    bool enabled_ = false;
    std::mutex mutex_;
    std::condition_variable ready_, done_;
    std::function<void()> job_;
    std::exception_ptr error_;
    bool pending_ = false, finished_ = false, stopping_ = false;
    std::thread thread_;
};

} // namespace BC::detail
