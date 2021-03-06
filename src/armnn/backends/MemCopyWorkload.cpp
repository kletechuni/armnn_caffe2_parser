﻿//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include "MemCopyWorkload.hpp"
#include "backends/CpuTensorHandle.hpp"
#include "TypeUtils.hpp"

#include <cstring>
#include <boost/cast.hpp>

namespace armnn
{

namespace
{

template <typename SrcTensorHandleType, typename DstTensorHandleType>
void GatherTensorHandlePairs(const MemCopyQueueDescriptor& descriptor,
                             std::vector<std::pair<SrcTensorHandleType*, DstTensorHandleType*>>& tensorHandlePairs)
{
    const unsigned int numInputs = static_cast<unsigned int>(descriptor.m_Inputs.size());
    tensorHandlePairs.reserve(numInputs);

    for (unsigned int i = 0; i < numInputs; ++i)
    {
        SrcTensorHandleType* const srcTensorHandle = boost::polymorphic_downcast<SrcTensorHandleType*>(
            descriptor.m_Inputs[i]);
        DstTensorHandleType* const dstTensorHandle = boost::polymorphic_downcast<DstTensorHandleType*>(
            descriptor.m_Outputs[i]);

        tensorHandlePairs.emplace_back(srcTensorHandle, dstTensorHandle);
    }
}

} //namespace


CopyMemGenericWorkload::CopyMemGenericWorkload(const MemCopyQueueDescriptor& descriptor,
                                                         const WorkloadInfo& info)
    : BaseWorkload<MemCopyQueueDescriptor>(descriptor, info)
{
    GatherTensorHandlePairs(descriptor, m_TensorHandlePairs);
}

void CopyMemGenericWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "CopyMemGeneric_Execute");

    auto copyFunc = [](void* dst, const void* src, size_t size)
        {
            memcpy(dst, src, size);
        };

    for (const auto& pair : m_TensorHandlePairs)
    {
        CopyTensorContentsGeneric(pair.first, pair.second, copyFunc);
    }
}

} //namespace armnn
