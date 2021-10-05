/* Copyright 2019 Benjamin Worpitz, Erik Zenker, René Widera
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_ANY_BT_OMP5_ENABLED

#    if _OPENMP < 201307
#        error If ALPAKA_ACC_ANY_BT_OMP5_ENABLED is set, the compiler has to support OpenMP 4.0 or higher!
#    endif

#    include <alpaka/block/shared/st/Traits.hpp>
#    include <alpaka/block/shared/st/detail/BlockSharedMemStMemberImpl.hpp>

#    include <omp.h>

#    include <cstdint>
#    include <type_traits>

namespace alpaka
{
    //#############################################################################
    //! The OpenMP 5 block shared memory allocator.
    class BlockSharedMemStOmp5
        : public detail::BlockSharedMemStMemberImpl<4>
        , public concepts::Implements<ConceptBlockSharedSt, BlockSharedMemStOmp5>
    {
    public:
        using BlockSharedMemStMemberImpl<4>::BlockSharedMemStMemberImpl;
    };

    namespace traits
    {
        //#############################################################################
        template<typename T, std::size_t TuniqueId>
        struct DeclareSharedVar<T, TuniqueId, BlockSharedMemStOmp5>
        {
            //-----------------------------------------------------------------------------
            static auto declareVar(BlockSharedMemStOmp5 const& smem) -> T&
            {
                auto* data = blockSharedMemSt.template getVar<T>(TuniqueId);

                if(!data)
                {
#    pragma omp barrier
#    pragma onmp single
                    {
                        blockSharedMemSt.template alloc<T>(TuniqueId);
                    }
#    pragma omp barrier
                    // lookup for the data chunk allocated by the master thread
                    data = blockSharedMemSt.template getVar<T>(TuniqueId);
                }
                ALPAKA_ASSERT(data != nullptr);
                return std::ref(*data);
            }
        };
        //#############################################################################
        template<>
        struct FreeSharedVars<BlockSharedMemStOmp5>
        {
            //-----------------------------------------------------------------------------
            static auto freeVars(BlockSharedMemStOmp5 const& mem) -> void
            {
            }
        };
    } // namespace traits
} // namespace alpaka

#endif
