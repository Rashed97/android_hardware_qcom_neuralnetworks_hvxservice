/*
 * Copyright (C) 2017 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define LOG_TAG "android.hardware.neuralnetworks@1.0-impl-hvx"

#include "OperationsUtils.h"
#include "HexagonUtils.h"
#include <algorithm>
#include <hidlmemory/mapping.h>
#include <numeric>
#include <vector>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace V1_0 {
namespace implementation {
namespace hexagon {

bool isHexagonAvailable() {
    int version = -1;
    hexagon::Controller::getInstance().version(&version);
    return version == 92;
}

hexagon_nn_padding_type getPadding(uint32_t pad) {
    switch (pad) {
        case ::android::nn::kPaddingSame:
            return NN_PAD_SAME;
        case ::android::nn::kPaddingValid:
            return NN_PAD_VALID;
        case ::android::nn::kPaddingUnknown:
        default:
            return NN_PAD_NA;
    };
}

hexagon_nn_padding_type getPadding(uint32_t filterWidth, uint32_t filterHeight,
                                   uint32_t paddingLeft, uint32_t paddingRight,
                                   uint32_t paddingTop, uint32_t paddingBottom) {
    return getPadding(::android::nn::getPaddingScheme(filterWidth, filterHeight, paddingLeft,
                                                      paddingRight, paddingTop, paddingBottom));
}

op_type getFloatActivationFunction(FusedActivationFunc act) {
    switch (act) {
        case FusedActivationFunc::RELU:
            return OP_Relu_f;
        case FusedActivationFunc::RELU1:
            return OP_ReluX_f;
        case FusedActivationFunc::RELU6:
            return OP_Clamp_f;
        case FusedActivationFunc::NONE:
            FALLTHROUGH_INTENDED;
        default:
            return OP_Nop;
    };
}

op_type getQuantizedActivationFunction(FusedActivationFunc act) {
    switch (act) {
        case FusedActivationFunc::RELU:
            return OP_QuantizedRelu_8;
        case FusedActivationFunc::RELU1:
            return OP_QuantizedReluX_8;
        case FusedActivationFunc::RELU6:
            return OP_QuantizedClamp_8;
        case FusedActivationFunc::NONE:
            FALLTHROUGH_INTENDED;
        default:
            return OP_Nop;
    };
}

uint32_t getSize(OperandType type) {
    static const uint32_t sizes[] = {
        4, // FLOAT32
        4, // INT32
        4, // UINT32
        4, // TENSOR_FLOAT32
        4, // TENSOR_INT32
        1, // TENSOR_SYMMETRICAL_QUANT8
    };
    if (static_cast<uint32_t>(type) >= sizeof(sizes) / sizeof(*sizes)) {
        LOG(ERROR) << "Error: type exceeds max enum value";
        return 0;
    }
    return sizes[static_cast<uint32_t>(type)];
}

std::vector<uint32_t> getAlignedDimensions(const std::vector<uint32_t>& dims, uint32_t N) {
    HEXAGON_SOFT_ASSERT_GE(N, dims.size(),
                           "Error: constant data dimensions " << dims.size() <<
                           " exceeds alignment of " << N);
    std::vector<uint32_t> dimensions(N - dims.size(), 1);
    dimensions.insert(dimensions.end(), dims.begin(), dims.end());
    return dimensions;
}

std::vector<RunTimePoolInfo> mapPools(const hidl_vec<hidl_memory>& pools) {
    std::vector<RunTimePoolInfo> poolInfos(pools.size());
    for (size_t i = 0; i < pools.size(); i++) {
        HEXAGON_SOFT_ASSERT(poolInfos[i].set(pools[i]), "Error setting pool " << i);
    }
    return poolInfos;
}


std::unordered_set<uint32_t> getPoolIndexes(const std::vector<RequestArgument>& inputsOutputs) {
    std::unordered_set<uint32_t> indexes;
    for (const RequestArgument& inputOutput : inputsOutputs) {
        indexes.insert(inputOutput.location.poolIndex);
    }
    return indexes;
}

namespace {
const uint8_t* getDataFromBlock(const hidl_vec<uint8_t>& block, uint32_t offset, uint32_t length) {
    HEXAGON_SOFT_ASSERT_LE(offset + length, block.size(),
                           "Error: trying to copy data from outside of block bounds");
    return block.data() + offset;
}

const uint8_t* getDataFromPool(const RunTimePoolInfo& pool, uint32_t offset, [[maybe_unused]] uint32_t length) {
    //HEXAGON_SOFT_ASSERT_LE(offset + length, pool->getSize(),
    //                       "Error: trying to copy data from outside of pool bounds");
    return pool.buffer + offset;
}
} // anonymous namespace

const uint8_t* getData(const Operand& operand, const hidl_vec<uint8_t>& block,
                       const std::vector<RunTimePoolInfo>& pools) {
    switch(operand.lifetime) {
        case OperandLifeTime::TEMPORARY_VARIABLE:
            return nullptr;
        case OperandLifeTime::MODEL_INPUT:
        case OperandLifeTime::MODEL_OUTPUT:
            HEXAGON_SOFT_ASSERT(false,
                   "Error: trying to retrieve data that is only known at runtime");
        case OperandLifeTime::CONSTANT_COPY:
            return getDataFromBlock(block, operand.location.offset, operand.location.length);
        case OperandLifeTime::CONSTANT_REFERENCE:
            return getDataFromPool(pools[operand.location.poolIndex], operand.location.offset,
                                   operand.location.length);
        default:
            HEXAGON_SOFT_ASSERT(false, "Error: unrecognized operand lifetime");
    }
}

bool operator==(const hexagon_nn_input& lhs, const hexagon_nn_input& rhs) {
    return lhs.src_id == rhs.src_id && lhs.output_idx == rhs.output_idx;
}

bool operator!=(const hexagon_nn_input& lhs, const hexagon_nn_input& rhs) {
    return !(lhs == rhs);
}

bool operator==(const hexagon_nn_output& lhs, const hexagon_nn_output& rhs) {
    return lhs.rank == rhs.rank && lhs.max_sizes[0] == rhs.max_sizes[0] &&
            lhs.max_sizes[1] == rhs.max_sizes[1] && lhs.max_sizes[2] == rhs.max_sizes[2] &&
            lhs.max_sizes[3] == rhs.max_sizes[3] && lhs.max_sizes[4] == rhs.max_sizes[4] &&
            lhs.max_sizes[5] == rhs.max_sizes[5] && lhs.max_sizes[6] == rhs.max_sizes[6] &&
            lhs.max_sizes[7] == rhs.max_sizes[7] && lhs.elementsize == rhs.elementsize &&
            lhs.zero_offset == rhs.zero_offset && lhs.stepsize == rhs.stepsize;
}

bool operator!=(const hexagon_nn_output& lhs, const hexagon_nn_output& rhs) {
    return !(lhs == rhs);
}

hexagon_nn_output make_hexagon_nn_output(const std::vector<uint32_t>& dims, uint32_t size) {
    std::vector<uint32_t> alignedDims = getAlignedDimensions(dims, 4);
    hexagon_nn_output output = {
        .rank = std::min(8u, static_cast<uint32_t>(alignedDims.size())),
        .max_sizes = {0, 0, 0, 0, 0, 0, 0, 0},
        .elementsize = size,
        .zero_offset = 0,
        .stepsize = 0.0f,
    };
    for (size_t i = 0; i < alignedDims.size() && i < 8; ++i) {
        output.max_sizes[i] = alignedDims[i];
    }
    return output;
}

namespace {

const char* kOps[] = {
    "OP_INPUT",
    "OP_OUTPUT",
    "OP_Nop",
    "OP_Const",
    "OP_Check",
    "OP_Close_f",
    "OP_Close_quint8",
    "OP_Close_q_quint8",
    "OP_Close_int32",
    "OP_Close_qint32",
    "OP_PPrint_8",
    "OP_PPrint_32",
    "OP_PPrint_f",
    "OP_PreFree",
    "OP_Flatten",
    "OP_QuantizedConv2d_8x8to32",
    "OP_QuantizedConv2d_8x8to32_ref",
    "OP_QuantizedMatMul_8x8to32",
    "OP_QuantizedMatMul_8x8to32_ref",
    "OP_QuantizeDownAndShrinkRange_32to8",
    "OP_QuantizeDownAndShrinkRange_32to8_ref",
    "OP_QuantizedRelu_8",
    "OP_QuantizedRelu_8_ref",
    "OP_QuantizedReluX_8",
    "OP_QuantizedReluX_8_ref",
    "OP_QuantizedMaxPool_8",
    "OP_QuantizedMaxPool_8_ref",
    "OP_QuantizedAvgPool_8",
    "OP_QuantizedAvgPool_8_ref",
    "OP_QuantizedConcat_8",
    "OP_QuantizedConcat_8_ref",
    "OP_QuantizedBiasAdd_8p8to32",
    "OP_QuantizedBiasAdd_8p8to32_ref",
    "OP_Min_f",
    "OP_Min_f_ref",
    "OP_Max_f",
    "OP_Max_f_ref",
    "OP_Quantize",
    "OP_Quantize_ref",
    "OP_Dequantize",
    "OP_Dequantize_ref",
    "OP_Supernode_8x8p8to8",
    "OP_Supernode_8x8p8to8_ref",
    "OP_QuantizedFlatten",
    "OP_Softmax_f",
    "OP_Conv2d_f",
    "OP_MatMul_f",
    "OP_Relu_f",
    "OP_ReluX_f",
    "OP_AvgPool_f",
    "OP_MaxPool_f",
    "OP_Concat_f",
    "OP_BiasAdd_f",
    "OP_LRN_f",
    "OP_Variable",
    "OP_Assign",
    "OP_Reshape",
    "OP_QuantizedReshape",
    "OP_Tanh_f",
    "OP_Sigmoid_f",
    "OP_Slice_8",
    "OP_Slice_f",
    "OP_QuantizedSlice_8",
    "OP_Add_f",
    "OP_Mul_f",
    "OP_Minimum_f",
    "OP_Maximum_f",
    "OP_Requantize_32to8",
    "OP_Requantize_32to8_ref",
    "OP_RequantizationRange_32",
    "OP_RequantizationRange_32_ref",
    "OP_Neg_f",
    "OP_Sub_f",
    "OP_AddN_f",
    "OP_Range_int32",
    "OP_Rank_int32",
    "OP_Transpose_int32",
    "OP_Transpose_f",
    "OP_InstanceNorm_f",
    "OP_QuantizedInstanceNorm_8",
    "OP_QuantizedInstanceNorm_8_ref",
    "OP_Sub_int32",
    "OP_Add_int32",
    "OP_Split_f",
    "OP_Dequantize_qint32_f",
    "OP_PRelu_f",
    "OP_QuantizedPRelu_8",
    "OP_QuantizedPRelu_8_ref",
    "OP_Sum_f",
    "OP_Prod_f",
    "OP_Mul_int32",
    "OP_LogicalAnd_int32",
    "OP_LogicalOr_int32",
    "OP_LogicalXor_int32",
    "OP_Shape_int32",
    "OP_Pack_int32",
    "OP_MirrorPad_f",
    "OP_ResizeNearestNeighbor_f",
    "OP_StridedSlice_int32",
    "OP_StridedSlice_f",
    "OP_ExpandDims_int32",
    "OP_ExpandDims_f",
    "OP_LogSoftmax_f",
    "OP_Split_int32",
    "OP_QuantizedSplit_8",
    "OP_Deconv_f",
    "OP_QuantizedDeconv_8x8to32",
    "OP_QuantizedDeconv_8x8to32_ref",
    "OP_QuantizedMul_8x8to32",
    "OP_QuantizedMul_8x8to32_ref",
    "OP_QuantizedAdd_8p8to32",
    "OP_QuantizedAdd_8p8to32_ref",
    "OP_QuantizedSigmoid_8",
    "OP_QuantizedSigmoid_8_ref",
    "OP_QuantizedTanh_8",
    "OP_QuantizedTanh_8_ref",
    "OP_QuantizedSoftmax_8",
    "OP_QuantizedSoftmax_8_ref",
    "OP_QuantizedLRN_8",
    "OP_QuantizedLRN_8_ref",
    "OP_Quantizedpad2d_frame_8p",
    "OP_Quantizedpad2d_frame_8p_ref",
    "OP_QuantizedSub_8p8to32",
    "OP_QuantizedSub_8p8to32_ref",
    "OP_QuantizedMaximum_8",
    "OP_QuantizedMaximum_8_ref",
    "OP_QuantizedMinimum_8",
    "OP_QuantizedMinimum_8_ref",
    "OP_Pad_f",
    "OP_SpaceToBatchND_f",
    "OP_BatchToSpaceND_f",
    "OP_QuantizedPad_8",
    "OP_ResizeBilinear_f",
    "OP_ConcatV2_f",
    "OP_ConcatV2_int32",
    "OP_Prod_int32",
    "OP_Slice_int32",
    "OP_QuantizedAdd_8p8to8",
    "OP_QuantizedResizeBilinear_8",
    "OP_Supernode_8x8p8to8_d32",
    "OP_Convert_to_d32",
    "OP_Convert_from_d32",
    "OP_QuantizedMaxPool_8_d32",
    "OP_QuantizedMaxPool_8_d32_ref",
    "OP_QuantizedConcat_8_d32",
    "OP_QuantizedConcat_8_d32_ref",
    "OP_QuantizedAvgPool_8_d32",
    "OP_QuantizedAvgPool_8_d32_ref",
    "OP_Sink",
    "OP_QuantizedPRelu_8_d32",
    "OP_QuantizedPRelu_8_d32_ref",
    "OP_AutoQuantize",
    "OP_AutoQuantize_ref",
    "OP_QuantizedDepthwiseConv2d_8x8to32",
    "OP_QuantizedDepthwiseConv2d_8x8to32_ref",
    "OP_DepthwiseConv2d_f",
    "OP_DepthwiseSupernode_8x8p8to8",
    "OP_DepthwiseSupernode_8x8p8to8_d32",
    "OP_QuantizedMul_8x8to8_d32",
    "OP_QuantizedMul_8x8to8_d32_ref",
    "OP_FullyConnected_u8",
    "OP_QuantizedAdd_8x8to8_d32",
    "OP_QuantizedAdd_8x8to8_d32_ref",
    "OP_QuantizedClamp_8",
    "OP_QuantizedClamp_8_ref",
    "OP_Clamp_f",
    "OP_QuantizeForTest_d32",
};

const char* kPadding[] = {
    "NN_PAD_NA",
    "NN_PAD_SAME",
    "NN_PAD_VALID",
    "NN_PAD_MIRROR_REFLECT",
    "NN_PAD_MIRROR_SYMMETRIC",
    "NN_PAD_SAME_CAFFE",
};

} // anonymous namespace

// printers
std::string toString(uint32_t val) {
    return std::to_string(val);
}

std::string toString(float val) {
    return std::to_string(val);
}

std::string toString(hexagon_nn_nn_id id) {
    return std::to_string(static_cast<int32_t>(id));
}

std::string toString(op_type op) {
    return static_cast<size_t>(op) < sizeof(kOps) / sizeof(char*) ?
            kOps[static_cast<size_t>(op)] : "<invalid op_type>";
}

std::string toString(hexagon_nn_padding_type padding) {
    return static_cast<size_t>(padding) < sizeof(kPadding) / sizeof(char*) ?
            kPadding[static_cast<size_t>(padding)] : "<invalid hexagon_nn_padding_type>";
}

std::string toString(const hexagon_nn_input& input) {
    return "hexagon_nn_input{.src_id: " + std::to_string(input.src_id) +
            ", .output_idx: " + std::to_string(input.output_idx) + "}";
}

std::string toString(const hexagon_nn_output& output) {
    return "hexagon_nn_output{.rank: " + std::to_string(output.rank) +
            ", .max_sizes: [" + std::to_string(output.max_sizes[0]) +
                ", " + std::to_string(output.max_sizes[1]) +
                ", " + std::to_string(output.max_sizes[2]) +
                ", " + std::to_string(output.max_sizes[3]) +
                ", " + std::to_string(output.max_sizes[4]) +
                ", " + std::to_string(output.max_sizes[5]) +
                ", " + std::to_string(output.max_sizes[6]) +
                ", " + std::to_string(output.max_sizes[7]) + "]" +
            ", .elementsize: " + std::to_string(output.elementsize) +
            ", .zero_offset: " + std::to_string(output.zero_offset) +
            ", .stepsize: " + std::to_string(output.stepsize) + "}";
}

std::string toString(const hexagon_nn_tensordef& tensordef) {
    return "hexagon_nn_tensordef{.batches: " + std::to_string(tensordef.batches) +
            ", .height: " + std::to_string(tensordef.height) +
            ", .width: " + std::to_string(tensordef.width) +
            ", .depth: " + std::to_string(tensordef.depth) +
            ", .data: " + std::to_string(reinterpret_cast<uintptr_t>(tensordef.data)) +
            ", .dataLen: " + std::to_string(tensordef.dataLen) +
            ", .data_valid_len: " + std::to_string(tensordef.data_valid_len) +
            ", .unused: " + std::to_string(tensordef.unused) + "}";
}

std::string toString(const hexagon_nn_perfinfo& perfinfo) {
    return "hexagon_nn_perfinfo{.node_id: " + std::to_string(perfinfo.node_id) +
            ", .executions: " + std::to_string(perfinfo.executions) +
            ", .counter_lo: " + std::to_string(perfinfo.counter_lo) +
            ", .counter_hi: " + std::to_string(perfinfo.counter_hi) + "}";
}

std::string toString(const ::android::nn::Shape& shape) {
    return "Shape{.type: " + toString(shape.type) +
            ", .dimensions: " + toString(shape.dimensions.data(), shape.dimensions.size()) +
            ", .scale: " + std::to_string(shape.scale) +
            ", .zeroPoint: " + std::to_string(shape.offset) + "}";
}

} // namespace hexagon
} // namespace implementation
} // namespace V1_0
} // namespace neuralnetworks
} // namespace hardware
} // namespace android
