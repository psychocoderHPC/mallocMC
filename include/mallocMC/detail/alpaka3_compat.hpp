#pragma once

#include <alpaka/alpaka.hpp>

#include <bit>
#include <tuple>
#include <utility>
#include <type_traits>

namespace alpaka
{
    using AtomicAdd = onAcc::AtomicAdd;
    using AtomicSub = onAcc::AtomicSub;
    using AtomicMin = onAcc::AtomicMin;
    using AtomicMax = onAcc::AtomicMax;
    using AtomicExch = onAcc::AtomicExch;
    using AtomicAnd = onAcc::AtomicAnd;
    using AtomicOr = onAcc::AtomicOr;
    using AtomicXor = onAcc::AtomicXor;
    using AtomicCas = onAcc::AtomicCas;

    namespace warp = onAcc::warp;
    namespace memory_scope
    {
        using Block = onAcc::scope::Block;
        using Device = onAcc::scope::Device;
        using System = onAcc::scope::System;
    }
    namespace memory_order
    {
        inline constexpr auto seq_cst = onAcc::order::seq_cst;
    }

    struct Grid
    {
    };
    struct Block
    {
    };
    struct Thread
    {
    };
    struct Threads
    {
    };
    struct Elems
    {
    };
    struct TagCpuSerial
    {
    };

    template<uint32_t TDim>
    using DimInt = std::integral_constant<uint32_t, TDim>;

    template<typename TTag, typename TDim, typename TIdx>
    using TagToAcc = std::conditional_t<std::is_same_v<TTag, TagCpuSerial>, exec::CpuSerial, TTag>;

    template<typename TAcc>
    using AccToTag = std::conditional_t<std::is_same_v<TAcc, exec::CpuSerial>, TagCpuSerial, TAcc>;

    using EnabledAccTags = std::tuple<TagCpuSerial>;

    template<typename TAcc, typename T, typename TScope = onAcc::scope::Device>
    constexpr auto atomicAdd(TAcc const& acc, T* addr, T const& value, TScope const scope = TScope{}) -> T
    {
        return onAcc::atomicAdd(acc, addr, value, scope);
    }

    template<typename TAcc, typename T, typename TScope = onAcc::scope::Device>
    constexpr auto atomicSub(TAcc const& acc, T* addr, T const& value, TScope const scope = TScope{}) -> T
    {
        return onAcc::atomicSub(acc, addr, value, scope);
    }

    template<typename TAcc, typename T, typename TScope = onAcc::scope::Device>
    constexpr auto atomicMin(TAcc const& acc, T* addr, T const& value, TScope const scope = TScope{}) -> T
    {
        return onAcc::atomicMin(acc, addr, value, scope);
    }

    template<typename TAcc, typename T, typename TScope = onAcc::scope::Device>
    constexpr auto atomicMax(TAcc const& acc, T* addr, T const& value, TScope const scope = TScope{}) -> T
    {
        return onAcc::atomicMax(acc, addr, value, scope);
    }

    template<typename TAcc, typename T, typename TScope = onAcc::scope::Device>
    constexpr auto atomicExch(TAcc const& acc, T* addr, T const& value, TScope const scope = TScope{}) -> T
    {
        return onAcc::atomicExch(acc, addr, value, scope);
    }

    template<typename TAcc, typename T, typename TScope = onAcc::scope::Device>
    constexpr auto atomicAnd(TAcc const& acc, T* addr, T const& value, TScope const scope = TScope{}) -> T
    {
        return onAcc::atomicAnd(acc, addr, value, scope);
    }

    template<typename TAcc, typename T, typename TScope = onAcc::scope::Device>
    constexpr auto atomicOr(TAcc const& acc, T* addr, T const& value, TScope const scope = TScope{}) -> T
    {
        return onAcc::atomicOr(acc, addr, value, scope);
    }

    template<typename TAcc, typename T, typename TScope = onAcc::scope::Device>
    constexpr auto atomicXor(TAcc const& acc, T* addr, T const& value, TScope const scope = TScope{}) -> T
    {
        return onAcc::atomicXor(acc, addr, value, scope);
    }

    template<typename TAcc, typename T, typename TScope = onAcc::scope::Device>
    constexpr auto atomicCas(
        TAcc const& acc,
        T* addr,
        T const& compare,
        T const& value,
        TScope const scope = TScope{}) -> T
    {
        return onAcc::atomicCas(acc, addr, compare, value, scope);
    }

    template<typename TAtomicOp, typename TAcc, typename... TArgs>
    constexpr auto atomicOp(TAcc const& acc, TArgs&&... args)
    {
        return onAcc::atomicOp<TAtomicOp>(acc, ALPAKA_FORWARD(args)...);
    }

    template<typename TAcc>
    constexpr void syncBlockThreads(TAcc const& acc)
    {
        onAcc::syncBlockThreads(acc);
    }

    template<typename TAcc, typename TScope>
    constexpr void mem_fence(TAcc const& acc, TScope const scope)
    {
        onAcc::memFence(acc, scope, onAcc::order::seq_cst);
    }

    template<typename TAcc, typename T>
    constexpr auto popcount(TAcc const&, T const& value)
    {
        using Unsigned = std::make_unsigned_t<std::decay_t<T>>;
        return std::popcount(static_cast<Unsigned>(value));
    }

    template<typename TAcc, typename T>
    constexpr auto ffs(TAcc const&, T const& value)
    {
        using Unsigned = std::make_unsigned_t<std::decay_t<T>>;
        auto v = static_cast<Unsigned>(value);
        if(v == 0u)
            return 0;

        int32_t bit = 1;
        while((v & Unsigned{1u}) == 0u)
        {
            v >>= 1u;
            ++bit;
        }
        return bit;
    }

    template<typename TOrigin, typename TUnit, typename TAcc>
    constexpr auto getIdx(TAcc const& acc)
    {
        if constexpr(std::is_same_v<TOrigin, Grid> && std::is_same_v<TUnit, Threads>)
            return acc.getIdxWithin(onAcc::origin::grid, onAcc::unit::threads);
        else if constexpr(std::is_same_v<TOrigin, Block> && std::is_same_v<TUnit, Threads>)
            return acc.getIdxWithin(onAcc::origin::block, onAcc::unit::threads);
    }

    template<typename TOrigin, typename TUnit, typename TAcc>
    constexpr auto getWorkDiv(TAcc const& acc)
    {
        if constexpr(std::is_same_v<TOrigin, Grid> && std::is_same_v<TUnit, Threads>)
            return acc.getExtentsOf(onAcc::origin::grid, onAcc::unit::threads);
        else if constexpr(std::is_same_v<TOrigin, Block> && std::is_same_v<TUnit, Threads>)
            return acc.getExtentsOf(onAcc::origin::block, onAcc::unit::threads);
        else if constexpr(std::is_same_v<TOrigin, Thread> && std::is_same_v<TUnit, Elems>)
            return Vec{1u};
    }

    template<uint32_t TDim, typename TVec, typename TExtent>
    constexpr auto mapIdx(TVec const& idx, TExtent const& extent)
    {
        static_assert(TDim == 1u);
        return Vec{linearize(extent, idx)};
    }

    template<typename T, size_t TUniqueId, typename TAcc>
    constexpr decltype(auto) declareSharedVar(TAcc const& acc)
    {
        return onAcc::declareSharedVar<T, TUniqueId>(acc);
    }

    template<typename T, typename TIdx = std::size_t, typename TDevice, typename TExtent>
    inline auto allocBuf(TDevice const& device, TExtent const& extent)
    {
        return onHost::alloc<T>(device, extent);
    }

    template<typename TQueue, typename TDest, typename TSource>
    inline void memcpy(TQueue const& queue, TDest&& dest, TSource&& source)
    {
        onHost::memcpy(queue, ALPAKA_FORWARD(dest), ALPAKA_FORWARD(source));
    }

    template<typename TQueue, typename TView, typename TValue>
    inline void memset(TQueue const& queue, TView&& view, TValue const& value)
    {
        onHost::memset(queue, ALPAKA_FORWARD(view), value);
    }

    template<typename THandle>
    inline void wait(THandle&& handle)
    {
        onHost::wait(ALPAKA_FORWARD(handle));
    }

    template<typename TView>
    constexpr auto getPtrNative(TView&& view)
    {
        return onHost::data(ALPAKA_FORWARD(view));
    }

    namespace math
    {
        template<typename TAcc, typename T>
        constexpr auto min(TAcc const&, T const& a, T const& b)
        {
            return ::alpaka::math::min(a, b);
        }

        template<typename TAcc, typename T>
        constexpr auto max(TAcc const&, T const& a, T const& b)
        {
            return ::alpaka::math::max(a, b);
        }
    } // namespace math
} // namespace alpaka
