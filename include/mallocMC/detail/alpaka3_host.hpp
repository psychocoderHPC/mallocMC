#pragma once

#include <alpaka/alpaka.hpp>

#include <any>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace mallocMC::detail
{
    template<typename T>
    struct DeviceAllocation
    {
        std::any storage{};
        T* ptr = nullptr;

        template<typename TDevice>
        void allocate(TDevice const& device, std::size_t extent)
        {
            auto buffer = alpaka::onHost::alloc<T>(device, extent);
            ptr = alpaka::onHost::data(buffer);
            storage.emplace<decltype(buffer)>(std::move(buffer));
        }

        void reset()
        {
            storage.reset();
            ptr = nullptr;
        }
    };

    inline auto makeHostDevice()
    {
        auto selector = alpaka::onHost::makeDeviceSelector(alpaka::api::host, alpaka::deviceKind::cpu);
        return selector.makeDevice(0);
    }

    template<typename TExecutor>
    auto make1DThreadSpec(std::uint32_t numBlocks, std::uint32_t numThreads)
    {
        using Vec = alpaka::Vec<std::uint32_t, 1u>;
        if constexpr(
            std::is_same_v<TExecutor, alpaka::exec::CpuSerial>
#if !defined(ALPAKA_DISABLE_EXEC_CpuOmpBlocks)
            || std::is_same_v<TExecutor, alpaka::exec::CpuOmpBlocks>
#endif
#if !defined(ALPAKA_DISABLE_EXEC_CpuTbbBlocks)
            || std::is_same_v<TExecutor, alpaka::exec::CpuTbbBlocks>
#endif
        )
            return alpaka::onHost::ThreadSpec{Vec{numBlocks * numThreads}, Vec{1u}, TExecutor{}};
        return alpaka::onHost::ThreadSpec{Vec{numBlocks}, Vec{numThreads}, TExecutor{}};
    }
} // namespace mallocMC::detail
