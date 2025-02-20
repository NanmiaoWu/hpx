//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file plain_action.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/actions_base/basic_action.hpp>
#include <hpx/assert.hpp>
#include <hpx/components_base/component_type.hpp>
#include <hpx/components_base/traits/component_type_database.hpp>
#include <hpx/modules/format.hpp>
#include <hpx/naming_base/address.hpp>
#include <hpx/preprocessor/cat.hpp>
#include <hpx/preprocessor/expand.hpp>
#include <hpx/preprocessor/nargs.hpp>
#include <hpx/preprocessor/strip_parens.hpp>

#include <boost/utility/string_ref.hpp>

#include <cstdlib>
#include <stdexcept>
#include <string>
#if defined(__NVCC__) || defined(__CUDACC__)
#include <type_traits>
#endif
#include <utility>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace actions {

    /// \cond NOINTERNAL
    namespace detail {

        struct plain_function
        {
            // Only localities are valid targets for a plain action
            static bool is_target_valid(naming::id_type const& id)
            {
                return naming::is_locality(id);
            }
        };

        ///////////////////////////////////////////////////////////////////////
        inline std::string make_plain_action_name(boost::string_ref action_name)
        {
            return hpx::util::format("plain action({})", action_name);
        }
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    //  Specialized generic plain (free) action types allowing to hold a
    //  different number of arguments
    ///////////////////////////////////////////////////////////////////////////
    template <typename R, typename... Ps, R (*F)(Ps...), typename Derived>
    struct action<R (*)(Ps...), F, Derived>
      : public basic_action<detail::plain_function, R(Ps...),
            typename detail::action_type<action<R (*)(Ps...), F, Derived>,
                Derived>::type>
    {
    public:
        typedef
            typename detail::action_type<action, Derived>::type derived_type;

        static std::string get_action_name(
            naming::address::address_type /*lva*/)
        {
            return detail::make_plain_action_name(
                detail::get_action_name<derived_type>());
        }

        template <typename... Ts>
        static R invoke(naming::address::address_type /*lva*/,
            naming::address::component_type /*comptype*/, Ts&&... vs)
        {
            basic_action<detail::plain_function, R(Ps...),
                derived_type>::increment_invocation_count();
            return F(std::forward<Ts>(vs)...);
        }
    };

    template <typename R, typename... Ps, R (*F)(Ps...) noexcept,
        typename Derived>
    struct action<R (*)(Ps...) noexcept, F, Derived>
      : public basic_action<detail::plain_function, R(Ps...),
            typename detail::action_type<
                action<R (*)(Ps...) noexcept, F, Derived>, Derived>::type>
    {
    public:
        typedef
            typename detail::action_type<action, Derived>::type derived_type;

        static std::string get_action_name(
            naming::address::address_type /*lva*/)
        {
            return detail::make_plain_action_name(
                detail::get_action_name<derived_type>());
        }

        template <typename... Ts>
        static R invoke(naming::address::address_type /*lva*/,
            naming::address::component_type /* comptype */, Ts&&... vs)
        {
            basic_action<detail::plain_function, R(Ps...),
                derived_type>::increment_invocation_count();
            return F(std::forward<Ts>(vs)...);
        }
    };

    /// \endcond
}}    // namespace hpx::actions

namespace hpx { namespace traits {

    /// \cond NOINTERNAL
    template <>
    HPX_ALWAYS_EXPORT inline components::component_type component_type_database<
        hpx::actions::detail::plain_function>::get() noexcept
    {
        return hpx::components::component_plain_function;
    }

    template <>
    HPX_ALWAYS_EXPORT inline void
        component_type_database<hpx::actions::detail::plain_function>::set(
            components::component_type)
    {
        HPX_ASSERT(false);    // shouldn't be ever called
    }
    /// \endcond
}}    // namespace hpx::traits

/// \def HPX_DEFINE_PLAIN_ACTION(func, name)
/// \brief Defines a plain action type
///
/// \par Example:
///
/// \code
///       namespace app
///       {
///           void some_global_function(double d)
///           {
///               cout << d;
///           }
///
///           // This will define the action type 'app::some_global_action' which
///           // represents the function 'app::some_global_function'.
///           HPX_DEFINE_PLAIN_ACTION(some_global_function, some_global_action);
///       }
/// \endcode
///
/// \note Usually this macro will not be used in user code unless the intent is
/// to avoid defining the action_type in global namespace. Normally, the use of
/// the macro \a HPX_PLAIN_ACTION is recommended.
///
/// \note The macro \a HPX_DEFINE_PLAIN_ACTION can be used with 1 or 2
/// arguments. The second argument is optional. The default value for the
/// second argument (the typename of the defined action) is derived from the
/// name of the function (as passed as the first argument) by appending '_action'.
/// The second argument can be omitted only if the first argument with an
/// appended suffix '_action' resolves to a valid, unqualified C++ type name.
///
#define HPX_DEFINE_PLAIN_ACTION(...)                                           \
    HPX_DEFINE_PLAIN_ACTION_(__VA_ARGS__)                                      \
    /**/

/// \cond NOINTERNAL

#define HPX_DEFINE_PLAIN_DIRECT_ACTION(...)                                    \
    HPX_DEFINE_PLAIN_DIRECT_ACTION_(__VA_ARGS__)                               \
    /**/

#define HPX_DEFINE_PLAIN_ACTION_(...)                                          \
    HPX_PP_EXPAND(HPX_PP_CAT(                                                  \
        HPX_DEFINE_PLAIN_ACTION_, HPX_PP_NARGS(__VA_ARGS__))(__VA_ARGS__))     \
    /**/

#define HPX_DEFINE_PLAIN_DIRECT_ACTION_(...)                                   \
    HPX_PP_EXPAND(HPX_PP_CAT(HPX_DEFINE_PLAIN_DIRECT_ACTION_,                  \
        HPX_PP_NARGS(__VA_ARGS__))(__VA_ARGS__))                               \
    /**/

#define HPX_DEFINE_PLAIN_ACTION_1(func)                                        \
    HPX_DEFINE_PLAIN_ACTION_2(func, HPX_PP_CAT(func, _action))                 \
    /**/

#if defined(__NVCC__) || defined(__CUDACC__)
#define HPX_DEFINE_PLAIN_ACTION_2(func, name)                                  \
    struct name                                                                \
      : hpx::actions::make_action<                                             \
            typename std::add_pointer<                                         \
                typename std::remove_pointer<decltype(&func)>::type>::type,    \
            &func, name>::type                                                 \
    {                                                                          \
    } /**/
#else
#define HPX_DEFINE_PLAIN_ACTION_2(func, name)                                  \
    struct name                                                                \
      : hpx::actions::make_action<decltype(&func), &func, name>::type          \
    {                                                                          \
    } /**/
#endif

#define HPX_DEFINE_PLAIN_DIRECT_ACTION_1(func)                                 \
    HPX_DEFINE_PLAIN_DIRECT_ACTION_2(func, HPX_PP_CAT(func, _action))          \
    /**/

#define HPX_DEFINE_PLAIN_DIRECT_ACTION_2(func, name)                           \
    struct name                                                                \
      : hpx::actions::make_direct_action<decltype(&func), &func, name>::type   \
    {                                                                          \
    } /**/

/// \endcond

///////////////////////////////////////////////////////////////////////////////
/// \def HPX_DECLARE_PLAIN_ACTION(func, name)
/// \brief Declares a plain action type
///
#define HPX_DECLARE_PLAIN_ACTION(...)                                          \
    HPX_DECLARE_ACTION(__VA_ARGS__)                                            \
    /**/

/// \def HPX_PLAIN_ACTION(func, name)
///
/// \brief Defines a plain action type based on the given function
/// \a func and registers it with HPX.
///
/// The macro \a HPX_PLAIN_ACTION can be used to define a plain action (e.g. an
/// action encapsulating a global or free function) based on the given function
/// \a func. It defines the action type \a name representing the given function.
/// This macro additionally registers the newly define action type with HPX.
///
/// The parameter \p func is a global or free (non-member) function which
/// should be encapsulated into a plain action. The parameter \p name is the
/// name of the action type defined by this macro.
///
/// \par Example:
///
/// \code
///     namespace app
///     {
///         void some_global_function(double d)
///         {
///             cout << d;
///         }
///     }
///
///     // This will define the action type 'some_global_action' which represents
///     // the function 'app::some_global_function'.
///     HPX_PLAIN_ACTION(app::some_global_function, some_global_action);
/// \endcode
///
/// \note The macro \a HPX_PLAIN_ACTION has to be used at global namespace even
/// if the wrapped function is located in some other namespace. The newly
/// defined action type is placed into the global namespace as well.
///
/// \note The macro \a HPX_PLAIN_ACTION_ID can be used with 1, 2, or 3 arguments.
/// The second and third arguments are optional. The default value for the
/// second argument (the typename of the defined action) is derived from the
/// name of the function (as passed as the first argument) by appending '_action'.
/// The second argument can be omitted only if the first argument with an
/// appended suffix '_action' resolves to a valid, unqualified C++ type name.
/// The default value for the third argument is \a hpx::components::factory_check.
///
/// \note Only one of the forms of this macro \a HPX_PLAIN_ACTION or
///       \a HPX_PLAIN_ACTION_ID should be used for a particular action,
///       never both.
///
#define HPX_PLAIN_ACTION(...)                                                  \
    HPX_PLAIN_ACTION_(__VA_ARGS__)                                             \
/**/

/// \def HPX_PLAIN_ACTION_ID(func, actionname, actionid)
///
/// \brief Defines a plain action type based on the given function \a func and
///   registers it with HPX.
///
/// The macro \a HPX_PLAIN_ACTION_ID can be used to define a plain action (e.g. an
/// action encapsulating a global or free function) based on the given function
/// \a func. It defines the action type \a actionname representing the given function.
/// The parameter \a actionid
///
/// The parameter \a actionid specifies an unique integer value which will be
/// used to represent the action during serialization.
///
/// The parameter \p func is a global or free (non-member) function which
/// should be encapsulated into a plain action. The parameter \p name is the
/// name of the action type defined by this macro.
///
/// The second parameter has to be usable as a plain (non-qualified) C++
/// identifier, it should not contain special characters which cannot be part
/// of a C++ identifier, such as '<', '>', or ':'.
///
/// \par Example:
///
/// \code
///     namespace app
///     {
///         void some_global_function(double d)
///         {
///             cout << d;
///         }
///     }
///
///     // This will define the action type 'some_global_action' which represents
///     // the function 'app::some_global_function'.
///     HPX_PLAIN_ACTION_ID(app::some_global_function, some_global_action,
///       some_unique_id);
/// \endcode
///
/// \note The macro \a HPX_PLAIN_ACTION_ID has to be used at global namespace even
/// if the wrapped function is located in some other namespace. The newly
/// defined action type is placed into the global namespace as well.
///
/// \note Only one of the forms of this macro \a HPX_PLAIN_ACTION or
///       \a HPX_PLAIN_ACTION_ID should be used for a particular action,
///       never both.
///
#define HPX_PLAIN_ACTION_ID(func, name, id)                                    \
    HPX_DEFINE_PLAIN_ACTION(func, name);                                       \
    HPX_REGISTER_ACTION_DECLARATION(name, name);                               \
    HPX_REGISTER_ACTION_ID(name, name, id);                                    \
    /**/

/// \cond NOINTERNAL

#define HPX_PLAIN_DIRECT_ACTION(...)                                           \
    HPX_PLAIN_DIRECT_ACTION_(__VA_ARGS__)                                      \
/**/

/// \endcond

/// \cond NOINTERNAL

// macros for plain actions
#define HPX_PLAIN_ACTION_(...)                                                 \
    HPX_PP_EXPAND(                                                             \
        HPX_PP_CAT(HPX_PLAIN_ACTION_, HPX_PP_NARGS(__VA_ARGS__))(__VA_ARGS__)) \
/**/
#define HPX_PLAIN_ACTION_2(func, name)                                         \
    HPX_DEFINE_PLAIN_ACTION(func, name);                                       \
    HPX_REGISTER_ACTION_DECLARATION(name, name);                               \
    HPX_REGISTER_ACTION(name, name);                                           \
/**/
#define HPX_PLAIN_ACTION_1(func)                                               \
    HPX_PLAIN_ACTION_2(func, HPX_PP_CAT(func, _action));                       \
/**/

// same for direct actions
#define HPX_PLAIN_DIRECT_ACTION_(...)                                          \
    HPX_PP_EXPAND(HPX_PP_CAT(                                                  \
        HPX_PLAIN_DIRECT_ACTION_, HPX_PP_NARGS(__VA_ARGS__))(__VA_ARGS__))     \
/**/
#define HPX_PLAIN_DIRECT_ACTION_2(func, name)                                  \
    HPX_DEFINE_PLAIN_DIRECT_ACTION(func, name);                                \
    HPX_REGISTER_ACTION_DECLARATION(name, name);                               \
    HPX_REGISTER_ACTION(name, name);                                           \
/**/
#define HPX_PLAIN_DIRECT_ACTION_1(func)                                        \
    HPX_PLAIN_DIRECT_ACTION_2(func, HPX_PP_CAT(func, _action));                \
/**/
#define HPX_PLAIN_DIRECT_ACTION_ID(func, name, id)                             \
    HPX_DEFINE_PLAIN_DIRECT_ACTION(func, name);                                \
    HPX_REGISTER_ACTION_DECLARATION(name, name);                               \
    HPX_REGISTER_ACTION_ID(name, name, id);                                    \
    /**/

/// \endcond

#include <hpx/config/warnings_suffix.hpp>
