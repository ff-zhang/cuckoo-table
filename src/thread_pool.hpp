//
// Created by Felix Zhang on 2025-08-26.
//

#pragma once

#include <cassert>
#include <queue>
#include <thread>
#include <vector>

using job_t = std::function<void()>;

class thread_pool {
public:
    thread_pool(const size_t num_threads) {
        for (size_t i = 0; i < num_threads; ++i) {
            threads_.emplace_back(std::thread(&thread_pool::worker, this, i));
            jobs_.emplace_back();
        }

        exits_ = std::vector<bool>(num_threads, false);

        mutexes_ = std::vector<std::mutex>(num_threads);
        conditions_ = std::vector<std::condition_variable>(num_threads);
    }

    thread_pool(const thread_pool&) = delete;
    thread_pool(thread_pool&&) = delete;
    thread_pool& operator=(const thread_pool&) = delete;
    thread_pool& operator=(thread_pool&&) = delete;

    void queue(const size_t index, const job_t& job) {
        assert(index < threads_.size());
        {
            std::unique_lock<std::mutex> lock(mutexes_[index]);
            jobs_[index].push(job);
        }
        conditions_[index].notify_one();
    }

    void kill(ssize_t index = -1) {
        if (index >= 0) {
            {
                std::unique_lock<std::mutex> lock(mutexes_[index]);
                exits_[index] = true;
            }
            conditions_[index].notify_all();
            threads_[index].join();

            return;
        }

        for (size_t i = 0; i < threads_.size(); ++i) {
            {
                std::unique_lock<std::mutex> lock(mutexes_[i]);
                exits_[i] = true;
            }
            conditions_[i].notify_all();
            threads_[i].join();
        }
    }

private:
    void worker(const size_t index) {
        while (true) {
            job_t job;
            {
                std::unique_lock<std::mutex> lock(mutexes_[index]);
                conditions_[index].wait(lock, [&] { return !jobs_[index].empty() || exits_[index]; });
                if (exits_[index]) {
                    return;
                }

                job = jobs_[index].front();
                jobs_[index].pop();
            }
            job();
        }
    }

    std::vector<bool> exits_;

    std::vector<std::thread> threads_;
    std::vector<std::queue<job_t>> jobs_;
    std::vector<std::mutex> mutexes_;
    std::vector<std::condition_variable> conditions_;
};
