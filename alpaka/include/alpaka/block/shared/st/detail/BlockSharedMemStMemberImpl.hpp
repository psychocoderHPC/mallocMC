/* Copyright 2020 Jeffrey Kelling, Rene Widera
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/block/shared/st/Traits.hpp>
#include <alpaka/core/Assert.hpp>
#include <alpaka/core/Vectorize.hpp>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <type_traits>

namespace alpaka
{
    namespace detail
    {
        //#############################################################################
        //! Implementation of static block shared memory provider.
        //!
        //! externally allocated fixed-size memory, likely provided by
        //! BlockSharedMemDynMember.
        template<std::size_t TDataAlignBytes = core::vectorization::defaultAlignment>
        class BlockSharedMemStMemberImpl
        {
            struct MetaData
            {
                uint32_t id = std::numeric_limits<uint32_t>::max();
                uint32_t offset = 0;
            };

        public:
            //-----------------------------------------------------------------------------
#ifndef NDEBUG
            BlockSharedMemStMemberImpl(uint8_t* mem, std::size_t capacity)
                : m_mem(mem)
                , m_capacity(static_cast<std::uint32_t>(capacity))
            {
#    ifdef ALPAKA_DEBUG_OFFLOAD_ASSUME_HOST
                ALPAKA_ASSERT((m_mem == nullptr) == (m_capacity == 0u));
#    endif
                init();
            }
#else
            BlockSharedMemStMemberImpl(uint8_t* mem, std::size_t) : m_mem(mem)
            {
                init();
            }
#endif
            //-----------------------------------------------------------------------------
            BlockSharedMemStMemberImpl(BlockSharedMemStMemberImpl const&) = delete;
            //-----------------------------------------------------------------------------
            BlockSharedMemStMemberImpl(BlockSharedMemStMemberImpl&&) = delete;
            //-----------------------------------------------------------------------------
            auto operator=(BlockSharedMemStMemberImpl const&) -> BlockSharedMemStMemberImpl& = delete;
            //-----------------------------------------------------------------------------
            auto operator=(BlockSharedMemStMemberImpl&&) -> BlockSharedMemStMemberImpl& = delete;
            //-----------------------------------------------------------------------------
            /*virtual*/ ~BlockSharedMemStMemberImpl() = default;

            template<typename T>
            void alloc(uint32_t id) const
            {
                MetaData& metaDataPtr = getLatestVar<MetaData>();

                m_allocdBytes = allocPitch<T>(m_allocdBytes);
                m_allocdBytes += static_cast<std::uint32_t>(sizeof(T));


                // add meta data chunk to the end
                m_allocdBytes = allocPitch<MetaData>(m_allocdBytes);
                m_allocdBytes += static_cast<std::uint32_t>(sizeof(MetaData));
                MetaData& ptr = getLatestVar<MetaData>();
                ptr = MetaData{};

                metaDataPtr.id = id;
                metaDataPtr.offset = m_allocdBytes - sizeof(MetaData);


#if(defined ALPAKA_DEBUG_OFFLOAD_ASSUME_HOST) && (!defined NDEBUG)
                ALPAKA_ASSERT(m_allocdBytes <= m_capacity);
#endif
            }

#if BOOST_COMP_GNUC
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored                                                                                    \
        "-Wcast-align" // "cast from 'unsigned char*' to 'unsigned int*' increases required alignment of target type"
#endif
            template<typename T>
            T& getLatestVar() const
            {
                return *reinterpret_cast<T*>(&m_mem[m_allocdBytes - sizeof(T)]);
            }

            template<typename T>
            T* getVar(uint32_t id) const
            {
                MetaData* metaDataPtr = reinterpret_cast<MetaData*>(m_mem);
                uint32_t off = 0;

                while(metaDataPtr->offset != 0)
                {
                    if(metaDataPtr->id == id)
                        break;
                    off = metaDataPtr->offset;
                    metaDataPtr = reinterpret_cast<MetaData*>(m_mem + metaDataPtr->offset);
                }

                if(metaDataPtr->offset == 0)
                {
                    return nullptr;
                }

                uint32_t allocdBytes = allocPitch<T>(allocPitch<MetaData>(off) + sizeof(MetaData)) + sizeof(T);

                return reinterpret_cast<T*>(&m_mem[allocdBytes - sizeof(T)]);
            }

#if BOOST_COMP_GNUC
#    pragma GCC diagnostic pop
#endif

            void init() const
            {
                m_allocdBytes = allocPitch<MetaData>(0);
                m_allocdBytes += static_cast<std::uint32_t>(sizeof(MetaData));
                MetaData& ptr = getLatestVar<MetaData>();
                ptr = MetaData{};
            }

        private:
            mutable std::uint32_t m_allocdBytes = 0;
            mutable uint8_t* m_mem;
#ifndef NDEBUG
            const std::uint32_t m_capacity;
#endif

            template<typename T>
            std::uint32_t allocPitch(uint32_t allocedBytes) const
            {
                static_assert(
                    core::vectorization::defaultAlignment >= alignof(T),
                    "Unable to get block shared static memory for types with alignment higher than defaultAlignment!");
                constexpr std::uint32_t align = static_cast<std::uint32_t>(std::max(TDataAlignBytes, alignof(T)));
                return (allocedBytes / align + (allocedBytes % align > 0u)) * align;
            }
        };
    } // namespace detail
} // namespace alpaka
