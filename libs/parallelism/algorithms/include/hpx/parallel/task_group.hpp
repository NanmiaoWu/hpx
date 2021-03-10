//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file task_group.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/async_combinators/when_all.hpp>
#include <hpx/async_local/dataflow.hpp>
#include <hpx/functional/bind.hpp>
#include <hpx/functional/bind_back.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/futures/traits/is_future.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/synchronization/spinlock.hpp>

#include <hpx/execution/executors/execution.hpp>
#include <hpx/executors/exception_list.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>

#include <memory>    // std::addressof

#include <exception>
#include <mutex>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx {
    namespace detail {
        /// \cond NOINTERNAL
        ///////////////////////////////////////////////////////////////////////
        inline void handle_task_group_exceptions(
            parallel::exception_list& errors)
        {
            try
            {
                std::rethrow_exception(std::current_exception());
            }
            catch (parallel::exception_list const& el)
            {
                for (std::exception_ptr const& e : el)
                    errors.add(e);
            }
            catch (...)
            {
                errors.add(std::current_exception());
            }
        }
        /// \endcond
    }    // namespace detail

    /// The class \a task_canceled_exception defines the type of objects thrown
    /// by task_group::run or task_group::wait if they detect
    /// that an exception is pending within the current parallel region.
    
    //class task_group_canceled_exception : public hpx::exception
    //{
    //public:
    //    task_group_canceled_exception() noexcept
    //      : hpx::exception(hpx::task_canceled_exception)
    //    {
    //    }
    //};

    /// The class task_group defines an interface for forking and
    /// joining parallel tasks. The \a define_task_group and
    /// \a define_task_group_restore_thread
    /// function templates create an object of type task_group and
    /// pass a reference to that object to a user-provided callable object.
    ///
    /// An object of class \a task_group cannot be constructed,
    /// destroyed, copied, or moved except by the implementation of the task
    /// region library. Taking the address of a task_group object via
    /// operator& or addressof is ill formed. The result of obtaining its
    /// address by any other means is unspecified.
    ///
    /// A \a task_group is active if it was created by the nearest
    /// enclosing task block, where "task block" refers to an invocation of
    /// define_task_group or define_task_group_restore_thread and "nearest
    /// enclosing" means the most
    /// recent invocation that has not yet completed. Code designated for
    /// execution in another thread by means other than the facilities in this
    /// section (e.g., using thread or async) are not enclosed in the task
    /// region and a task_group passed to (or captured by) such code
    /// is not active within that code. Performing any operation on a
    /// task_group that is not active results in undefined behavior.
    ///
    /// The \a task_group that is active before a specific call to the
    /// run member function is not active within the asynchronous function
    /// that invoked run. (The invoked function should not, therefore, capture
    /// the \a task_group from the surrounding block.)
    ///
    /// \code
    /// Example:
    ///     define_task_group([&](auto& tr) {
    ///         tr.run([&] {
    ///             tr.run([] { f(); });                // Error: tr is not active
    ///             define_task_group([&](auto& tr) {   // Nested task block
    ///                 tr.run(f);                      // OK: inner tr is active
    ///                 /// ...
    ///             });
    ///         });
    ///         /// ...
    ///     });
    /// \endcode
    ///
    /// \tparam ExPolicy The execution policy an instance of a \a task_group
    ///         was created with. This defaults to \a parallel_policy.
    ///
    template <typename ExPolicy = hpx::execution::parallel_policy>
    class task_group
    {
    private:
        /// \cond NOINTERNAL
        typedef hpx::lcos::local::spinlock mutex_type;

        explicit task_group(ExPolicy const& policy = ExPolicy())
          : id_(threads::get_self_id())
          , policy_(policy)
        {
        }

        void wait_for_completion(std::false_type)
        {
            when();
        }

        void wait_for_completion(std::true_type)
        {
            when().wait();
        }

        void wait_for_completion()
        {
            typedef typename parallel::util::detail::algorithm_result<ExPolicy>::type
                result_type;
            typedef hpx::traits::is_future<result_type> is_fut;
            wait_for_completion(is_fut());
        }

        ~task_group()
        {
            wait_for_completion();
        }

        task_group(task_group const&) = delete;
        task_group& operator=(task_group const&) = delete;

        task_group* operator&() const = delete;

        static void on_ready(std::vector<hpx::future<void>>&& results,
            parallel::exception_list&& errors)
        {
            for (hpx::future<void>& f : results)
            {
                if (f.has_exception())
                    errors.add(f.get_exception_ptr());
            }
            if (errors.size() != 0)
                throw std::forward<parallel::exception_list>(errors);
        }

        // return future representing the execution of all tasks
        typename parallel::util::detail::algorithm_result<ExPolicy>::type when(
            bool throw_on_error = false)
        {
            std::vector<hpx::future<void>> tasks;
            parallel::exception_list errors;

            {
                std::lock_guard<mutex_type> l(mtx_);
                std::swap(tasks_, tasks);
                std::swap(errors_, errors);
            }

            typedef parallel::util::detail::algorithm_result<ExPolicy> result;

            if (tasks.empty() && errors.size() == 0)
                return result::get();

            if (!throw_on_error)
                return result::get(hpx::when_all(tasks));

            return result::get(
                hpx::dataflow(hpx::util::one_shot(hpx::util::bind_back(
                                  &task_group::on_ready, std::move(errors))),
                    std::move(tasks)));
        }
        /// \endcond

    public:
        /// Refers to the type of the execution policy used to create the
        /// \a task_group.
        typedef ExPolicy execution_policy;

        /// Return the execution policy instance used to create this
        /// \a task_group
        execution_policy const& get_execution_policy() const
        {
            return policy_;
        }

        /// Causes the expression f() to be invoked asynchronously.
        /// The invocation of f is permitted to run on an unspecified thread
        /// in an unordered fashion relative to the sequence of operations
        /// following the call to run(f) (the continuation), or indeterminately
        /// sequenced within the same thread as the continuation.
        ///
        /// The call to \a run synchronizes with the invocation of f. The
        /// completion of f() synchronizes with the next invocation of wait on
        /// the same task_group or completion of the nearest enclosing
        /// task block (i.e., the \a define_task_group or
        /// \a define_task_group_restore_thread that created this task block).
        ///
        /// Requires: F shall be MoveConstructible. The expression, (void)f(),
        ///           shall be well-formed.
        ///
        /// Precondition: this shall be the active task_group.
        ///
        /// Postconditions: A call to run may return on a different thread than
        ///                 that on which it was called.
        ///
        /// \note The call to \a run is sequenced before the continuation as if
        ///       \a run returns on the same thread.
        ///       The invocation of the user-supplied callable object f may be
        ///       immediate or may be delayed until compute resources are
        ///       available. \a run might or might not return before invocation
        ///       of f completes.
        ///
        /// \throw This function may throw \a task_canceled_exception, as
        ///        described in Exception Handling.
        ///
        template <typename F, typename... Ts>
        void run(F&& f, Ts&&... ts)
        {

            hpx::future<void> result =
                parallel::execution::async_execute(policy_.executor(), std::forward<F>(f),
                    std::forward<Ts>(ts)...);

            std::lock_guard<mutex_type> l(mtx_);
            tasks_.push_back(std::move(result));
        }

        /// Causes the expression f() to be invoked asynchronously using the
        /// given executor.
        /// The invocation of f is permitted to run on an unspecified thread
        /// associated with the given executor and in an unordered fashion
        /// relative to the sequence of operations following the call to
        /// run(exec, f) (the continuation), or indeterminately sequenced
        /// within the same thread as the continuation.
        ///
        /// The call to \a run synchronizes with the invocation of f. The
        /// completion of f() synchronizes with the next invocation of wait on
        /// the same task_group or completion of the nearest enclosing
        /// task block (i.e., the \a define_task_group or
        /// \a define_task_group_restore_thread that created this task block).
        ///
        /// Requires: Executor shall be a type modeling the Executor concept.
        ///           F shall be MoveConstructible. The expression, (void)f(),
        ///           shall be well-formed.
        ///
        /// Precondition: this shall be the active task_group.
        ///
        /// Postconditions: A call to run may return on a different thread than
        ///                 that on which it was called.
        ///
        /// \note The call to \a run is sequenced before the continuation as if
        ///       \a run returns on the same thread.
        ///       The invocation of the user-supplied callable object f may be
        ///       immediate or may be delayed until compute resources are
        ///       available. \a run might or might not return before invocation
        ///       of f completes.
        ///
        /// \throw This function may throw \a task_canceled_exception, as
        ///        described in Exception Handling.
        ///
        template <typename Executor, typename F, typename... Ts>
        void run(Executor& exec, F&& f, Ts&&... ts)
        {

            hpx::future<void> result = parallel::execution::async_execute(
                exec, std::forward<F>(f), std::forward<Ts>(ts)...);

            std::lock_guard<mutex_type> l(mtx_);
            tasks_.push_back(std::move(result));
        }

        /// Blocks until the tasks spawned using this task_group have
        /// finished.
        ///
        /// Precondition: this shall be the active task_group.
        ///
        /// Postcondition: All tasks spawned by the nearest enclosing task
        ///                region have finished. A call to wait may return on
        ///                a different thread than that on which it was called.
        ///
        /// \note The call to \a wait is sequenced before the continuation as if
        ///       \a wait returns on the same thread.
        ///
        /// \throw This function may throw \a task_canceled_exception, as
        ///        described in Exception Handling.
        ///
        /// \code
        /// Example:
        ///     define_task_group([&](auto& tr) {
        ///         tr.run([&]{ process(a, w, x); }); // Process a[w] through a[x]
        ///         if (y < x) tr.wait();   // Wait if overlap between [w, x) and [y, z)
        ///         process(a, y, z);       // Process a[y] through a[z]
        ///     });
        /// \endcode
        ///
        void wait()
        {

            wait_for_completion();
        }

        /// Returns a reference to the execution policy used to construct this
        /// object.
        ///
        /// Precondition: this shall be the active task_group.
        ///
        ExPolicy& policy()
        {
            return policy_;
        }

        /// Returns a reference to the execution policy used to construct this
        /// object.
        ///
        /// Precondition: this shall be the active task_group.
        ///
        ExPolicy const& policy() const
        {
            return policy_;
        }

    private:
        mutable mutex_type mtx_;
        std::vector<hpx::future<void>> tasks_;
        parallel::exception_list errors_;
        threads::thread_id_type id_;
        ExPolicy policy_;
    };


}   // namespace hpx
/// \cond NOINTERNAL
namespace std {
    template <typename ExPolicy>
    hpx::task_group<ExPolicy>* addressof(
        hpx::task_group<ExPolicy>&) = delete;
}
/// \endcond
