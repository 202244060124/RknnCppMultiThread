#include "thread_pool.h"

constexpr size_t ThreadPool::WAIT_SECONDS;

ThreadPool::~ThreadPool()
{
    {
        MutexGuard guard(m_mutex);
        m_quitFlag = true;
    }
    m_conditionVar.notify_all();

    for (auto& elem : m_threadMaps) {
        assert(elem.second.joinable());
        elem.second.join();
    }
}
ThreadPool::ThreadPool() : ThreadPool(Thread::hardware_concurrency())
{
}

ThreadPool::ThreadPool(size_t maxThreads) : m_quitFlag(false), m_currentThreadNums(0), m_idleThreads(0), m_maxThreads(maxThreads)
{
}
size_t ThreadPool::GetCurrentThreadNums(void) const
{
    MutexGuard guard(m_mutex);
    return m_currentThreadNums;
}

void ThreadPool::Run()
{
    while (true) {
        Task task;
        {
            UniqueLock uniqueLock(m_mutex);
            ++m_idleThreads;
            auto hasTimedout = !m_conditionVar.wait_for(uniqueLock, std::chrono::seconds(WAIT_SECONDS),
                                                        [this]() { return m_quitFlag || !m_tasksQueue.empty(); });
            --m_idleThreads;
            if (m_tasksQueue.empty()) {
                if (m_quitFlag) {
                    --m_currentThreadNums;
                    return;
                }
                if (hasTimedout) {
                    --m_currentThreadNums;
                    FinishedThreadsJoin();
                    m_finishedThreadIdsQueue.emplace(std::this_thread::get_id());
                    return;
                }
            }
            task = std::move(m_tasksQueue.front());
            m_tasksQueue.pop();
        }
        task();
    }
}

void ThreadPool::FinishedThreadsJoin()
{
    while (!m_finishedThreadIdsQueue.empty()) {
        auto id = std::move(m_finishedThreadIdsQueue.front());
        m_finishedThreadIdsQueue.pop();
        auto iter = m_threadMaps.find(id);

        assert(iter != m_threadMaps.end());
        assert(iter->second.joinable());

        iter->second.join();
        m_threadMaps.erase(iter);
    }
}
