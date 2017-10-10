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

#include "HexagonModel.h"
#include "HexagonOperations.h"
#include "OperationsUtils.h"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace V1_0 {
namespace implementation {
namespace hexagon {

using android::nn::Shape;

namespace {

bool addMul(const std::vector<uint32_t>& ins, const std::vector<uint32_t>& outs,
            HexagonModel* model, OperationType op) {
    HEXAGON_SOFT_ASSERT_EQ(3, ins.size(), "Need 3 inputs for " << toString(op));
    HEXAGON_SOFT_ASSERT_EQ(1, outs.size(), "Need 1 output for " << toString(op));

    // get output size
    const Shape in1Shape = model->getShape(ins[0]);
    const Shape in2Shape = model->getShape(ins[1]);
    Shape outShape       = model->getShape(outs[0]);
    HEXAGON_SOFT_ASSERT(addMulPrepare(in1Shape, in2Shape, &outShape), "Error getting shape");
    HEXAGON_SOFT_ASSERT(model->setShape(outs[0], outShape), "Error setting shape");

    return true;
}

bool add(const std::vector<uint32_t>& ins, const std::vector<uint32_t>& outs,
         HexagonModel* model) {
    return addMul(ins, outs, model, OperationType::ADD);
}

bool mul(const std::vector<uint32_t>& ins, const std::vector<uint32_t>& outs,
         HexagonModel* model) {
    return addMul(ins, outs, model, OperationType::MUL);
}

bool pool(const std::vector<uint32_t>& ins, const std::vector<uint32_t>& outs,
          HexagonModel* model, OperationType op) {
    HEXAGON_SOFT_ASSERT(ins.size() == 10 || ins.size() == 7,
                        "Need 7 or 10 inputs for " << toString(op));

    // get parameters
    const Shape inShape = model->getShape(ins[0]);

    // setup parameters
    int32_t padding_left;
    int32_t padding_right;
    int32_t padding_top;
    int32_t padding_bottom;
    int32_t stride_width;
    int32_t stride_height;
    int32_t filter_width;
    int32_t filter_height;

    // get parameters
    if (ins.size() == 10) {
        padding_left   = model->getScalar<int32_t>(ins[1]);
        padding_right  = model->getScalar<int32_t>(ins[2]);
        padding_top    = model->getScalar<int32_t>(ins[3]);
        padding_bottom = model->getScalar<int32_t>(ins[4]);
        stride_width   = model->getScalar<int32_t>(ins[5]);
        stride_height  = model->getScalar<int32_t>(ins[6]);
        filter_width   = model->getScalar<int32_t>(ins[7]);
        filter_height  = model->getScalar<int32_t>(ins[8]);

        HEXAGON_SOFT_ASSERT_NE(getPadding(filter_width, filter_height, padding_left,
                                          padding_right, padding_top, padding_bottom),
                               NN_PAD_NA, "Unknown padding");
    }
    else {
        const int32_t padding_implicit = model->getScalar<int32_t>(ins[1]);
        stride_width                   = model->getScalar<int32_t>(ins[2]);
        stride_height                  = model->getScalar<int32_t>(ins[3]);
        filter_width                   = model->getScalar<int32_t>(ins[4]);
        filter_height                  = model->getScalar<int32_t>(ins[5]);

        nn::calculateExplicitPadding(inShape.dimensions[2], stride_width, filter_width,
                                     padding_implicit, &padding_left, &padding_right);
        nn::calculateExplicitPadding(inShape.dimensions[1], stride_height, filter_height,
                                     padding_implicit, &padding_top, &padding_bottom);
    }

    // get output size
    Shape outShape = model->getShape(outs[0]);
    HEXAGON_SOFT_ASSERT(genericPoolingPrepare(inShape, padding_left, padding_right, padding_top,
                                              padding_bottom, stride_width, stride_height,
                                              filter_width, filter_height, &outShape),
                        "Error getting shape");
    HEXAGON_SOFT_ASSERT(model->setShape(outs[0], outShape), "Error setting shape");

    return true;
}

bool average_pool_2d(const std::vector<uint32_t>& ins, const std::vector<uint32_t>& outs,
                     HexagonModel* model) {
    return pool(ins, outs, model, OperationType::AVERAGE_POOL_2D);
}

bool l2_pool_2d(const std::vector<uint32_t>& ins, const std::vector<uint32_t>& outs,
                HexagonModel* model) {
    return pool(ins, outs, model, OperationType::L2_POOL_2D);
}

bool max_pool_2d(const std::vector<uint32_t>& ins, const std::vector<uint32_t>& outs,
                 HexagonModel* model) {
    return pool(ins, outs, model, OperationType::MAX_POOL_2D);
}

bool concatenation(const std::vector<uint32_t>& ins, const std::vector<uint32_t>& outs,
                   HexagonModel* model) {
    std::string name = toString(OperationType::CONCATENATION);
    HEXAGON_SOFT_ASSERT_LE(3, ins.size(), "Need at least 3 inputs for " << name);
    HEXAGON_SOFT_ASSERT_EQ(1, outs.size(), "Need 1 output for " << name);

    const size_t numInputTensors = ins.size() - 1;

    const int32_t axis = model->getScalar<int32_t>(ins[numInputTensors]);

    // get output size
    std::vector<Shape> inShapes(numInputTensors);
    for (size_t i = 0; i < numInputTensors; ++i) {
        inShapes[i] = model->getShape(ins[i]);
    }
    Shape outShape = model->getShape(outs[0]);
    HEXAGON_SOFT_ASSERT(concatenationPrepare(inShapes, axis, &outShape), "Error getting shape");
    HEXAGON_SOFT_ASSERT(model->setShape(outs[0], outShape), "Error setting shape");

    return true;
}

bool conv_2d(const std::vector<uint32_t>& ins, const std::vector<uint32_t>& outs,
             HexagonModel* model) {
    std::string name = toString(OperationType::CONV_2D);
    HEXAGON_SOFT_ASSERT(ins.size() == 10 || ins.size() == 7, "Need 7 or 10 inputs for " << name);
    HEXAGON_SOFT_ASSERT_EQ(1, outs.size(), "Need 1 output for " << name);

    // setup shapes
    const Shape inputShape  = model->getShape(ins[0]);
    const Shape filterShape = model->getShape(ins[1]);
    const Shape biasShape   = model->getShape(ins[2]);

    // setup parameters
    int32_t padding_left;
    int32_t padding_right;
    int32_t padding_top;
    int32_t padding_bottom;
    int32_t stride_width;
    int32_t stride_height;

    // get parameters
    if (ins.size() == 10) {
        padding_left   = model->getScalar<int32_t>(ins[3]);
        padding_right  = model->getScalar<int32_t>(ins[4]);
        padding_top    = model->getScalar<int32_t>(ins[5]);
        padding_bottom = model->getScalar<int32_t>(ins[6]);
        stride_width   = model->getScalar<int32_t>(ins[7]);
        stride_height  = model->getScalar<int32_t>(ins[8]);

        HEXAGON_SOFT_ASSERT_NE(getPadding(filterShape.dimensions[2], filterShape.dimensions[1],
                                          padding_left, padding_right,
                                          padding_top, padding_bottom),
                               NN_PAD_NA, "Unknown padding");
    }
    else {
        const int32_t padding_implicit = model->getScalar<int32_t>(ins[3]);
        stride_width                   = model->getScalar<int32_t>(ins[4]);
        stride_height                  = model->getScalar<int32_t>(ins[5]);

        nn::calculateExplicitPadding(inputShape.dimensions[2], stride_width,
                                     filterShape.dimensions[2], padding_implicit,
                                     &padding_left, &padding_right);
        nn::calculateExplicitPadding(inputShape.dimensions[1], stride_height,
                                     filterShape.dimensions[1], padding_implicit,
                                     &padding_top, &padding_bottom);
    }

    // get output size
    Shape outShape = model->getShape(outs[0]);
    HEXAGON_SOFT_ASSERT(convPrepare(inputShape, filterShape, biasShape, padding_left,
                                    padding_right, padding_top, padding_bottom, stride_width,
                                    stride_height, &outShape), "Error getting shape");
    HEXAGON_SOFT_ASSERT(model->setShape(outs[0], outShape), "Error setting shape");

    // enforce filter is a constant
    HEXAGON_SOFT_ASSERT(model->isConstant(ins[1]),
                        name << "requires filter to be constant data");

    return true;
}

bool depthwise_conv_2d(const std::vector<uint32_t>& ins, const std::vector<uint32_t>& outs,
                       HexagonModel* model) {
    std::string name = toString(OperationType::DEPTHWISE_CONV_2D);
    HEXAGON_SOFT_ASSERT(ins.size() == 8 || ins.size() == 11, "Need 8 or 11 inputs for " << name);
    HEXAGON_SOFT_ASSERT_EQ(1, outs.size(), "Need 1 output for " << name);

    // setup shapes
    const Shape inputShape  = model->getShape(ins[0]);
    const Shape filterShape = model->getShape(ins[1]);
    const Shape biasShape   = model->getShape(ins[2]);

    // setup parameters
    int32_t padding_left;
    int32_t padding_right;
    int32_t padding_top;
    int32_t padding_bottom;
    int32_t stride_width;
    int32_t stride_height;

    // get parameters
    if (ins.size() == 11) {
        padding_left   = model->getScalar<int32_t>(ins[3]);
        padding_right  = model->getScalar<int32_t>(ins[4]);
        padding_top    = model->getScalar<int32_t>(ins[5]);
        padding_bottom = model->getScalar<int32_t>(ins[6]);
        stride_width   = model->getScalar<int32_t>(ins[7]);
        stride_height  = model->getScalar<int32_t>(ins[8]);

        HEXAGON_SOFT_ASSERT_NE(getPadding(filterShape.dimensions[2], filterShape.dimensions[1],
                                          padding_left, padding_right,
                                          padding_top, padding_bottom),
                               NN_PAD_NA, "Unknown padding");
    }
    else {
        const int32_t padding_implicit = model->getScalar<int32_t>(ins[3]);
        stride_width                   = model->getScalar<int32_t>(ins[4]);
        stride_height                  = model->getScalar<int32_t>(ins[5]);

        nn::calculateExplicitPadding(inputShape.dimensions[2], stride_width,
                                     filterShape.dimensions[2], padding_implicit,
                                     &padding_left, &padding_right);
        nn::calculateExplicitPadding(inputShape.dimensions[1], stride_height,
                                     filterShape.dimensions[1], padding_implicit,
                                     &padding_top, &padding_bottom);
    }

    // get output size
    Shape outShape = model->getShape(outs[0]);
    HEXAGON_SOFT_ASSERT(depthwiseConvPrepare(inputShape, filterShape, biasShape, padding_left,
                                             padding_right, padding_top, padding_bottom,
                                             stride_width, stride_height, &outShape),
                        "Error getting shape");
    HEXAGON_SOFT_ASSERT(model->setShape(outs[0], outShape), "Error setting shape");

    // enforce filter is a constant
    HEXAGON_SOFT_ASSERT(model->isConstant(ins[1]),
                        name << " requires filter to be constant data");

    return true;
}

bool dequantize(const std::vector<uint32_t>& ins, const std::vector<uint32_t>& outs,
                HexagonModel* model) {
    std::string name = toString(OperationType::DEQUANTIZE);
    HEXAGON_SOFT_ASSERT_EQ(1, ins.size(), "Need 1 input for " << name);
    HEXAGON_SOFT_ASSERT_EQ(1, outs.size(), "Need 1 output for " << name);

    // get output size
    const Shape inputShape = model->getShape(ins[0]);
    Shape outShape         = model->getShape(outs[0]);

    HEXAGON_SOFT_ASSERT(dequantizePrepare(inputShape, &outShape), "Error getting shape");
    HEXAGON_SOFT_ASSERT(model->setShape(outs[0], outShape), "Error setting shape");

    return true;
}

bool fully_connected(const std::vector<uint32_t>& ins, const std::vector<uint32_t>& outs,
                     HexagonModel* model) {
    std::string name = toString(OperationType::FULLY_CONNECTED);
    HEXAGON_SOFT_ASSERT_EQ(4, ins.size(), "Need 4 inputs for " << name);
    HEXAGON_SOFT_ASSERT_EQ(1, outs.size(), "Need 1 output for " << name);

    // get output size
    const Shape inputShape   = model->getShape(ins[0]);
    const Shape weightsShape = model->getShape(ins[1]);
    const Shape biasShape    = model->getShape(ins[2]);
    Shape outShape           = model->getShape(outs[0]);
    HEXAGON_SOFT_ASSERT(fullyConnectedPrepare(inputShape, weightsShape, biasShape, &outShape),
                        "Error getting shape");
    HEXAGON_SOFT_ASSERT(model->setShape(outs[0], outShape), "Error setting shape");

    // enforce weight is a constant
    HEXAGON_SOFT_ASSERT(model->isConstant(ins[1]),
                        name << "requires weight to be constant data");

    return true;
}

bool local_response_normalization(const std::vector<uint32_t>& ins,
                                  const std::vector<uint32_t>& outs,
                                  HexagonModel* model) {
    std::string name = toString(OperationType::LOCAL_RESPONSE_NORMALIZATION);
    HEXAGON_SOFT_ASSERT_EQ(5, ins.size(), "Need 5 inputs for " << name);
    HEXAGON_SOFT_ASSERT_EQ(1, outs.size(), "Need 1 output for " << name);

    // get output size
    const Shape inShape = model->getShape(ins[0]);
    Shape outShape      = model->getShape(outs[0]);
    HEXAGON_SOFT_ASSERT(genericNormalizationPrepare(inShape, &outShape), "Error getting shape");
    HEXAGON_SOFT_ASSERT(model->setShape(outs[0], outShape), "Error setting shape");

    return true;
}

bool activation(const std::vector<uint32_t>& ins, const std::vector<uint32_t>& outs,
                HexagonModel* model, uint32_t numInputs, OperationType op) {
    HEXAGON_SOFT_ASSERT_EQ(numInputs, ins.size(), "Need " << numInputs << " input for "
                                                          << toString(op));
    HEXAGON_SOFT_ASSERT_EQ(1, outs.size(), "Need 1 output for " << toString(op));

    // get output size
    const Shape inShape = model->getShape(ins[0]);
    Shape outShape      = model->getShape(outs[0]);
    HEXAGON_SOFT_ASSERT(genericActivationPrepare(inShape, &outShape), "Error getting shape");
    HEXAGON_SOFT_ASSERT(model->setShape(outs[0], outShape), "Error setting shape");

    return true;
}

bool logistic(const std::vector<uint32_t>& ins, const std::vector<uint32_t>& outs,
              HexagonModel* model) {
    return activation(ins, outs, model, 1, OperationType::LOGISTIC);
}

bool relu(const std::vector<uint32_t>& ins, const std::vector<uint32_t>& outs,
          HexagonModel* model) {
    return activation(ins, outs, model, 1, OperationType::RELU);
}

bool relu1(const std::vector<uint32_t>& ins, const std::vector<uint32_t>& outs,
           HexagonModel* model) {
    return activation(ins, outs, model, 1, OperationType::RELU1);
}

bool relu6(const std::vector<uint32_t>& ins, const std::vector<uint32_t>& outs,
           HexagonModel* model) {
    return activation(ins, outs, model, 1, OperationType::RELU6);
}

bool softmax(const std::vector<uint32_t>& ins, const std::vector<uint32_t>& outs,
             HexagonModel* model) {
    return activation(ins, outs, model, 2, OperationType::SOFTMAX);
}

bool tanh(const std::vector<uint32_t>& ins, const std::vector<uint32_t>& outs,
          HexagonModel* model) {
    return activation(ins, outs, model, 1, OperationType::TANH);
}

bool reshape(const std::vector<uint32_t>& ins, const std::vector<uint32_t>& outs,
             HexagonModel* model) {
    std::string name = toString(OperationType::RESHAPE);
    HEXAGON_SOFT_ASSERT_EQ(2, ins.size(), "Need 2 inputs for " << name);
    HEXAGON_SOFT_ASSERT_EQ(1, outs.size(), "Need 1 output for " << name);

    // get output size
    const Shape inShape           = model->getShape(ins[0]);
    const Shape targetShape       = model->getShape(ins[1]);
    const int32_t* targetShapePtr = model->getPointer(ins[1]);
    int32_t targetShapeNumElem    = ::android::nn::getNumberOfElements(targetShape);
    Shape outShape                = model->getShape(outs[0]);
    HEXAGON_SOFT_ASSERT(targetShapePtr != nullptr, "pointer value is currently nullptr");

    HEXAGON_SOFT_ASSERT(reshapePrepare(inShape, targetShapePtr, targetShapeNumElem, &outShape),
                        "Error getting shape");
    HEXAGON_SOFT_ASSERT(model->setShape(outs[0], outShape), "Error setting shape");

    return true;
}

bool resize_bilinear(const std::vector<uint32_t>& ins, const std::vector<uint32_t>& outs,
                     HexagonModel* model) {
    std::string name = toString(OperationType::RESIZE_BILINEAR);
    HEXAGON_SOFT_ASSERT_EQ(3, ins.size(), "Need 3 inputs for " << name);
    HEXAGON_SOFT_ASSERT_EQ(1, outs.size(), "Need 1 output for " << name);

    // get parameters
    const int32_t width  = model->getScalar<int32_t>(ins[1]);
    const int32_t height = model->getScalar<int32_t>(ins[2]);

    // get output size
    const Shape inShape  = model->getShape(ins[0]);
    Shape outShape       = model->getShape(outs[0]);
    HEXAGON_SOFT_ASSERT(resizeBilinearPrepare(inShape, width, height, &outShape),
                        "Error getting shape");
    HEXAGON_SOFT_ASSERT(model->setShape(outs[0], outShape), "Error setting shape");

    return true;
}

}  // namespace

OperationCheckTable& getOperationCheckTable() {
    static OperationCheckTable table = {
        {OperationType::ADD,                          add                         },
        {OperationType::AVERAGE_POOL_2D,              average_pool_2d             },
        {OperationType::CONCATENATION,                concatenation               },
        {OperationType::CONV_2D,                      conv_2d                     },
        {OperationType::DEPTHWISE_CONV_2D,            depthwise_conv_2d           },
//      {OperationType::DEPTH_TO_SPACE,               depth_to_space              },
        {OperationType::DEQUANTIZE,                   dequantize                  },
//      {OperationType::EMBEDDING_LOOKUP,             embedding_lookup            },
//      {OperationType::FLOOR,                        floor                       },
        {OperationType::FULLY_CONNECTED,              fully_connected             },
//      {OperationType::HASHTABLE_LOOKUP,             hashtable_lookup            },
//      {OperationType::L2_NORMALIZATION,             l2_normalization            },
        {OperationType::L2_POOL_2D,                   l2_pool_2d                  },
        {OperationType::LOCAL_RESPONSE_NORMALIZATION, local_response_normalization},
        {OperationType::LOGISTIC,                     logistic                    },
//      {OperationType::LSH_PROJECTION,               lsh_projection              },
//      {OperationType::LSTM,                         lstm                        },
        {OperationType::MAX_POOL_2D,                  max_pool_2d                 },
        {OperationType::MUL,                          mul                         },
        {OperationType::RELU,                         relu                        },
        {OperationType::RELU1,                        relu1                       },
        {OperationType::RELU6,                        relu6                       },
        {OperationType::RESHAPE,                      reshape                     },
        {OperationType::RESIZE_BILINEAR,              resize_bilinear             },
//      {OperationType::RNN,                          rnn                         },
        {OperationType::SOFTMAX,                      softmax                     },
//      {OperationType::SPACE_TO_DEPTH,               space_to_depth              },
//      {OperationType::SVDF,                         svdf                        },
        {OperationType::TANH,                         tanh                        },
    };
    return table;
}

} // namespace hexagon
} // namespace implementation
} // namespace V1_0
} // namespace neuralnetworks
} // namespace hardware
} // namespace android
