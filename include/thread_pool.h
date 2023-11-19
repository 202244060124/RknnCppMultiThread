#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <cassert>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>

// namespace dpool {

class ThreadPool {
public:
    using MutexGuard = std::lock_guard<std::mutex>;
    using UniqueLock = std::unique_lock<std::mutex>;
    using Thread = std::thread;
    using ThreadID = std::thread::id;
    using Task = std::function<void()>;
    ThreadPool();

    explicit ThreadPool(size_t maxThreads);

    // disable the copy operations
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    ~ThreadPool();

    template<typename Func, typename... Ts>
    auto AddTaskToTaskQueue(Func&& func, Ts&&... params) -> std::future<typename std::result_of<Func(Ts...)>::type>
    {
        auto execute = std::bind(std::forward<Func>(func), std::forward<Ts>(params)...);

        using ReturnType = typename std::result_of<Func(Ts...)>::type;
        using PackagedTask = std::packaged_task<ReturnType()>;

        auto task = std::make_shared<PackagedTask>(std::move(execute));
        auto result = task->get_future();

        MutexGuard guard(m_mutex);
        assert(!m_quitFlag);

        m_tasksQueue.emplace([task]() { (*task)(); });
        if (m_idleThreads > 0) {
            m_conditionVar.notify_one();
        } else if (m_currentThreadNums < m_maxThreads) {
            Thread t(&ThreadPool::Run, this);
            assert(m_threadMaps.find(t.get_id()) == m_threadMaps.end());
            m_threadMaps[t.get_id()] = std::move(t);
            ++m_currentThreadNums;
        }

        return result;
    }

    size_t GetCurrentThreadNums(void) const;

private:
    void Run();

    void FinishedThreadsJoin();
    static constexpr size_t WAIT_SECONDS = 2;
    bool m_quitFlag;
    size_t m_currentThreadNums;
    size_t m_idleThreads;
    size_t m_maxThreads;

    mutable std::mutex m_mutex;
    std::condition_variable m_conditionVar;
    std::queue<Task> m_tasksQueue;
    std::queue<ThreadID> m_finishedThreadIdsQueue;
    std::unordered_map<ThreadID, Thread> m_threadMaps;
};
// } // namespace dpool

#endif /* THREAD_POOL_H */