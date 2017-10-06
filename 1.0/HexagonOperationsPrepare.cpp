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
namespace float32 {

bool add(const std::vector<uint32_t>& ins, const std::vector<uint32_t>& outs,
         HexagonModel* model) {
    HEXAGON_SOFT_ASSERT_EQ(3, ins.size(), "Need 3 inputs for float32::add");
    HEXAGON_SOFT_ASSERT_EQ(1, outs.size(), "Need 1 output for float32::add");

    // get parameters
    const hexagon_nn_input& in1 = model->getTensor(ins[0]);
    const hexagon_nn_input& in2 = model->getTensor(ins[1]);

    const op_type act           = model->getFloatActivation(ins[2]);

    // add node to graph
    return model->addFusedFloatOperation(OP_Add_f, NN_PAD_NA, {}, act, {in1, in2}, outs);
}

bool average_pool_2d(const std::vector<uint32_t>& ins, const std::vector<uint32_t>& outs,
                     HexagonModel* model) {
    HEXAGON_SOFT_ASSERT_EQ(10, ins.size(), "Need 10 inputs for float32::average_pool_2d");
    HEXAGON_SOFT_ASSERT_EQ(1, outs.size(), "Need 1 output for float32::average_pool_2d");

    // get parameters
    const hexagon_nn_input& input     = model->getTensor(ins[0]);

    const int32_t padding_left        = model->getScalar<int32_t>(ins[1]);
    const int32_t padding_right       = model->getScalar<int32_t>(ins[2]);
    const int32_t padding_top         = model->getScalar<int32_t>(ins[3]);
    const int32_t padding_bottom      = model->getScalar<int32_t>(ins[4]);
    const int32_t stride_width        = model->getScalar<int32_t>(ins[5]);
    const int32_t stride_height       = model->getScalar<int32_t>(ins[6]);
    const int32_t filter_width        = model->getScalar<int32_t>(ins[7]);
    const int32_t filter_height       = model->getScalar<int32_t>(ins[8]);
    const op_type act                 = model->getFloatActivation(ins[9]);

    const hexagon_nn_input window     = model->createShape(1, filter_height, filter_width, 1);
    const hexagon_nn_input stride     = model->createShape(1, stride_height, stride_width, 1);
    const hexagon_nn_padding_type pad = getPadding(filter_width, filter_height, padding_left,
                                                   padding_right, padding_top, padding_bottom);

    // add node to graph
    return model->addFloatOperationWithActivation(OP_AvgPool_f, pad, act,
                                                  {input, window, stride}, outs);
}

bool concatenation(const std::vector<uint32_t>& ins, const std::vector<uint32_t>& outs,
                   HexagonModel* model) {
    HEXAGON_SOFT_ASSERT_LE(3, ins.size(), "Need at least 3 inputs for float32::concatenation");
    HEXAGON_SOFT_ASSERT_EQ(1, outs.size(), "Need 1 output for float32::concatenation");

    const size_t numInputTensors = ins.size() - 1;

    // get parameters
    std::vector<hexagon_nn_input> inputs(numInputTensors + 1);
    for (size_t i = 0; i < numInputTensors; ++i) {
        inputs[i+1] = model->getTensor(ins[i]);
    }

    // axis being concatenated
    const int32_t axis = model->getScalar<int32_t>(ins[numInputTensors]);
    const int32_t dims = model->getShape(ins[0]).dimensions.size();
    inputs[0] = model->createScalar<int32_t>(axis + (4 - dims));

    // add node to graph
    return model->addBasicOperation(OP_Concat_f, NN_PAD_NA, inputs, outs);
}

bool conv_2d(const std::vector<uint32_t>& ins, const std::vector<uint32_t>& outs,
             HexagonModel* model) {
    HEXAGON_SOFT_ASSERT_EQ(10, ins.size(), "Need 10 inputs for float32::conv_2d");
    HEXAGON_SOFT_ASSERT_EQ(1, outs.size(), "Need 1 output for float32::conv_2d");

    // get parameters
    const hexagon_nn_input& input     = model->getTensor(ins[0]);
    const hexagon_nn_input  filter    = model->createConvFilterTensor(ins[1]);
    const hexagon_nn_input& bias      = model->getTensor(ins[2]);

    const int32_t padding_left        = model->getScalar<int32_t>(ins[3]);
    const int32_t padding_right       = model->getScalar<int32_t>(ins[4]);
    const int32_t padding_top         = model->getScalar<int32_t>(ins[5]);
    const int32_t padding_bottom      = model->getScalar<int32_t>(ins[6]);
    const int32_t stride_width        = model->getScalar<int32_t>(ins[7]);
    const int32_t stride_height       = model->getScalar<int32_t>(ins[8]);
    const op_type act                 = model->getFloatActivation(ins[9]);

    const hexagon_nn_input stride     = model->createShape(1, stride_height, stride_width, 1);
    const Shape filterShape           = model->getShape(ins[1]);
    const hexagon_nn_padding_type pad = getPadding(filterShape.dimensions[2],
                                                   filterShape.dimensions[1],
                                                   padding_left, padding_right,
                                                   padding_top, padding_bottom);

    // add node to graph
    return model->addFusedFloatOperation(OP_Conv2d_f, pad, bias, act,
                                         {input, filter, stride}, outs);
}

bool depthwise_conv_2d(const std::vector<uint32_t>& ins, const std::vector<uint32_t>& outs,
                       HexagonModel* model) {
    HEXAGON_SOFT_ASSERT_EQ(11, ins.size(), "Need 11 inputs for float32::depthwise_conv_2d");
    HEXAGON_SOFT_ASSERT_EQ(1, outs.size(), "Need 1 output for float32::depthwise_conv_2d");

    // get parameters
    const hexagon_nn_input& input      = model->getTensor(ins[0]);
    const hexagon_nn_input& bias       = model->getTensor(ins[2]);

    const int32_t padding_left         = model->getScalar<int32_t>(ins[3]);
    const int32_t padding_right        = model->getScalar<int32_t>(ins[4]);
    const int32_t padding_top          = model->getScalar<int32_t>(ins[5]);
    const int32_t padding_bottom       = model->getScalar<int32_t>(ins[6]);
    const int32_t stride_width         = model->getScalar<int32_t>(ins[7]);
    const int32_t stride_height        = model->getScalar<int32_t>(ins[8]);
    const int32_t depth_multiplier     = model->getScalar<int32_t>(ins[9]);
    const op_type act                  = model->getFloatActivation(ins[10]);

    const hexagon_nn_input filter      = model->createDepthwiseFilterTensor(ins[1],
                                                                            depth_multiplier);
    const hexagon_nn_input stride      = model->createShape(1, stride_height, stride_width, 1);
    const Shape filterShape            = model->getShape(ins[1]);
    const hexagon_nn_padding_type pad  = getPadding(filterShape.dimensions[2],
                                                   filterShape.dimensions[1],
                                                   padding_left, padding_right,
                                                   padding_top, padding_bottom);

    // add node to graph
    return model->addFusedFloatOperation(OP_DepthwiseConv2d_f, pad, bias, act,
                                         {input, filter, stride}, outs);
}

bool fully_connected(const std::vector<uint32_t>& ins, const std::vector<uint32_t>& outs,
                     HexagonModel* model) {
    HEXAGON_SOFT_ASSERT_EQ(4, ins.size(), "Need 4 inputs for float32::fully_connected");
    HEXAGON_SOFT_ASSERT_EQ(1, outs.size(), "Need 1 output for float32::fully_connected");

    // get parameters
    const hexagon_nn_input& input   = model->getTensor(ins[0]);
    const hexagon_nn_input& weights = model->createFullyConnectedWeightTensor(ins[1]);
    const hexagon_nn_input& bias    = model->getTensor(ins[2]);

    const op_type act               = model->getFloatActivation(ins[3]);

    // add node to graph
    return model->addFusedFloatOperation(OP_MatMul_f, NN_PAD_NA, bias, act,
                                         {input, weights}, outs);
}

bool l2_pool_2d(const std::vector<uint32_t>& ins, const std::vector<uint32_t>& outs,
                HexagonModel* model) {
    HEXAGON_SOFT_ASSERT_EQ(10, ins.size(), "Need 10 inputs for float32::l2_pool_2d");
    HEXAGON_SOFT_ASSERT_EQ(1, outs.size(), "Need 1 output for float32::l2_pool_2d");

    // get parameters
    const hexagon_nn_input& input     = model->getTensor(ins[0]);

    const int32_t padding_left        = model->getScalar<int32_t>(ins[1]);
    const int32_t padding_right       = model->getScalar<int32_t>(ins[2]);
    const int32_t padding_top         = model->getScalar<int32_t>(ins[3]);
    const int32_t padding_bottom      = model->getScalar<int32_t>(ins[4]);
    const int32_t stride_width        = model->getScalar<int32_t>(ins[5]);
    const int32_t stride_height       = model->getScalar<int32_t>(ins[6]);
    const int32_t filter_width        = model->getScalar<int32_t>(ins[7]);
    const int32_t filter_height       = model->getScalar<int32_t>(ins[8]);
    const op_type act                 = model->getFloatActivation(ins[9]);

    const hexagon_nn_input window     = model->createShape(1, filter_height, filter_width, 1);
    const hexagon_nn_input stride     = model->createShape(1, stride_height, stride_width, 1);
    const hexagon_nn_padding_type pad = getPadding(filter_width, filter_height, padding_left,
                                                   padding_right, padding_top, padding_bottom);
    // add node to graph
    return model->addFloatOperationWithActivation(OP_L2Pool_f, pad, act,
                                                  {input, window, stride}, outs);
}

bool local_response_normalization(const std::vector<uint32_t>& ins,
                                  const std::vector<uint32_t>& outs,
                                  HexagonModel* model) {
    HEXAGON_SOFT_ASSERT_EQ(5, ins.size(),
                           "Need 5 inputs for float32::local_response_normalization");
    HEXAGON_SOFT_ASSERT_EQ(1, outs.size(),
                           "Need 1 output for float32::local_response_normalization");

    // get parameters
    const hexagon_nn_input& input = model->getTensor(ins[0]);
    const hexagon_nn_input& bias  = model->getTensor(ins[2]);
    const hexagon_nn_input& alpha = model->getTensor(ins[3]);
    const hexagon_nn_input& beta  = model->getTensor(ins[4]);

    // create value that's [1, 1, 1, radius] with value of 1.0f
    const int32_t radius          = model->getScalar<int32_t>(ins[1]);
    const hexagon_nn_input window = model->createTensor<float>(1, 1, 1, radius * 2 + 1, {1.0f});

    // add node to graph
    return model->addBasicOperation(OP_LRN_f, NN_PAD_NA, {input, window, bias, alpha, beta}, outs);
}

bool logistic(const std::vector<uint32_t>& ins, const std::vector<uint32_t>& outs,
              HexagonModel* model) {
    HEXAGON_SOFT_ASSERT_EQ(1, ins.size(), "Need 1 input for float32::logistic");
    HEXAGON_SOFT_ASSERT_EQ(1, outs.size(), "Need 1 output for float32::logistic");

    // get parameters
    const hexagon_nn_input& input = model->getTensor(ins[0]);

    // add node to graph
    return model->addBasicOperation(OP_Sigmoid_f, NN_PAD_NA, {input}, outs);
}

bool max_pool_2d(const std::vector<uint32_t>& ins, const std::vector<uint32_t>& outs,
                 HexagonModel* model) {
    HEXAGON_SOFT_ASSERT_EQ(10, ins.size(), "Need 10 inputs for float32::max_pool_2d");
    HEXAGON_SOFT_ASSERT_EQ(1, outs.size(), "Need 1 output for float32::max_pool_2d");

    // get parameters
    const hexagon_nn_input& input     = model->getTensor(ins[0]);

    const int32_t padding_left        = model->getScalar<int32_t>(ins[1]);
    const int32_t padding_right       = model->getScalar<int32_t>(ins[2]);
    const int32_t padding_top         = model->getScalar<int32_t>(ins[3]);
    const int32_t padding_bottom      = model->getScalar<int32_t>(ins[4]);
    const int32_t stride_width        = model->getScalar<int32_t>(ins[5]);
    const int32_t stride_height       = model->getScalar<int32_t>(ins[6]);
    const int32_t filter_width        = model->getScalar<int32_t>(ins[7]);
    const int32_t filter_height       = model->getScalar<int32_t>(ins[8]);
    const op_type act                 = model->getFloatActivation(ins[9]);

    const hexagon_nn_input window     = model->createShape(1, filter_height, filter_width, 1);
    const hexagon_nn_input stride     = model->createShape(1, stride_height, stride_width, 1);
    const hexagon_nn_padding_type pad = getPadding(filter_width, filter_height, padding_left,
                                                   padding_right, padding_top, padding_bottom);
    // add node to graph
    return model->addFloatOperationWithActivation(OP_MaxPool_f, pad, act,
                                                  {input, window, stride}, outs);
}

bool mul(const std::vector<uint32_t>& ins, const std::vector<uint32_t>& outs,
         HexagonModel* model) {
    HEXAGON_SOFT_ASSERT_EQ(3, ins.size(), "Need 3 inputs for float32::mul");
    HEXAGON_SOFT_ASSERT_EQ(1, outs.size(), "Need 1 output for float32::mul");

    // get parameters
    const hexagon_nn_input& in1 = model->getTensor(ins[0]);
    const hexagon_nn_input& in2 = model->getTensor(ins[1]);

    const op_type act           = model->getFloatActivation(ins[2]);

    // add node to graph
    return model->addFusedFloatOperation(OP_Mul_f, NN_PAD_NA, {}, act,
                                         {in1, in2}, outs);
}

bool relu(const std::vector<uint32_t>& ins, const std::vector<uint32_t>& outs,
          HexagonModel* model) {
    HEXAGON_SOFT_ASSERT_EQ(1, ins.size(), "Need 1 input for float32::relu");
    HEXAGON_SOFT_ASSERT_EQ(1, outs.size(), "Need 1 output for float32::relu");

    // get parameters
    const hexagon_nn_input& input = model->getTensor(ins[0]);

    // add node to graph
    return model->addBasicOperation(OP_Relu_f, NN_PAD_NA, {input}, outs);
}

bool relu1(const std::vector<uint32_t>& ins, const std::vector<uint32_t>& outs,
           HexagonModel* model) {
    HEXAGON_SOFT_ASSERT_EQ(1, ins.size(), "Need 1 input for float32::relu1");
    HEXAGON_SOFT_ASSERT_EQ(1, outs.size(), "Need 1 output for float32::relu1");

    // get parameters
    const hexagon_nn_input& input = model->getTensor(ins[0]);
    const hexagon_nn_input  min   = model->createScalar(-1.0f);
    const hexagon_nn_input  max   = model->createScalar(1.0f);

    // add node to graph
    return model->addBasicOperation(OP_Clamp_f, NN_PAD_NA, {input, min, max}, outs);
}

bool relu6(const std::vector<uint32_t>& ins, const std::vector<uint32_t>& outs,
           HexagonModel* model) {
    HEXAGON_SOFT_ASSERT_EQ(1, ins.size(), "Need 1 input for float32::relu6");
    HEXAGON_SOFT_ASSERT_EQ(1, outs.size(), "Need 1 output for float32::relu6");

    // get parameters
    const hexagon_nn_input& input = model->getTensor(ins[0]);
    const hexagon_nn_input  max   = model->createScalar(6.0f);

    // add node to graph
    return model->addBasicOperation(OP_ReluX_f, NN_PAD_NA, {input, max}, outs);
}

bool reshape(const std::vector<uint32_t>& ins, const std::vector<uint32_t>& outs,
             HexagonModel* model) {
    HEXAGON_SOFT_ASSERT_EQ(2, ins.size(), "Need 2 inputs for float32::reshape");
    HEXAGON_SOFT_ASSERT_EQ(1, outs.size(), "Need 1 output for float32::reshape");

    // get parameters
    const hexagon_nn_input& input   = model->getTensor(ins[0]);
    const hexagon_nn_input& newdims = model->getTensor(ins[1]);

    // add node to graph
    return model->addBasicOperation(OP_Reshape, NN_PAD_NA, {input, newdims}, outs);
}

bool resize_bilinear(const std::vector<uint32_t>& ins, const std::vector<uint32_t>& outs,
                     HexagonModel* model) {
    HEXAGON_SOFT_ASSERT_EQ(3, ins.size(), "Need 3 inputs for float32::resize_bilinear");
    HEXAGON_SOFT_ASSERT_EQ(1, outs.size(), "Need 1 output for float32::resize_bilinear");

    // get parameters
    const hexagon_nn_input& input = model->getTensor(ins[0]);

    const int32_t width           = model->getScalar<int32_t>(ins[1]);
    const int32_t height          = model->getScalar<int32_t>(ins[2]);

    const hexagon_nn_input newdim = model->createValues<int32_t>({height, width});

    // add node to graph
    return model->addBasicOperation(OP_ResizeBilinear_f, NN_PAD_NA, {input, newdim}, outs);
}

bool softmax(const std::vector<uint32_t>& ins, const std::vector<uint32_t>& outs,
             HexagonModel* model) {
    HEXAGON_SOFT_ASSERT_EQ(2, ins.size(), "Need 2 inputs for float32::softmax");
    HEXAGON_SOFT_ASSERT_EQ(1, outs.size(), "Need 1 output for float32::softmax");

    // get parameters
    const hexagon_nn_input& input = model->getTensor(ins[0]);
    const hexagon_nn_input& beta  = model->getTensor(ins[1]);

    // add node to graph
    return model->addBasicOperation(OP_Softmax_f, NN_PAD_NA, {input, beta}, outs);
}

bool tanh(const std::vector<uint32_t>& ins, const std::vector<uint32_t>& outs,
          HexagonModel* model) {
    HEXAGON_SOFT_ASSERT_EQ(1, ins.size(), "Need 1 input for float32::tanh");
    HEXAGON_SOFT_ASSERT_EQ(1, outs.size(), "Need 1 output for float32::tanh");

    // get parameters
    const hexagon_nn_input& input = model->getTensor(ins[0]);

    // add node to graph
    return model->addBasicOperation(OP_Tanh_f, NN_PAD_NA, {input}, outs);
}

}  // float32 namespace

namespace quant8_asym {

bool add(const std::vector<uint32_t>& ins, const std::vector<uint32_t>& outs,
         HexagonModel* model) {
    HEXAGON_SOFT_ASSERT_EQ(3, ins.size(), "Need 3 inputs for quant8_asym::add");
    HEXAGON_SOFT_ASSERT_EQ(1, outs.size(), "Need 1 output for quant8_asym::add");

    // get parameters
    const hexagon_nn_input& in1     = model->getTensor(ins[0]);
    const hexagon_nn_input& in2     = model->getTensor(ins[1]);

    const op_type act               = model->getQuantizedActivation(ins[2]);

    const hexagon_nn_input& in1_min = model->getQuantizationMin(ins[0]);
    const hexagon_nn_input& in1_max = model->getQuantizationMax(ins[0]);
    const hexagon_nn_input& in2_min = model->getQuantizationMin(ins[1]);
    const hexagon_nn_input& in2_max = model->getQuantizationMax(ins[1]);

    // add node to graph
    return model->addFusedQuant8Operation(OP_QuantizedAdd_8p8to32, NN_PAD_NA, {}, act,
                                          {in1, in1_min, in1_max, in2, in2_min, in2_max}, outs);
}

bool average_pool_2d(const std::vector<uint32_t>& ins, const std::vector<uint32_t>& outs,
                     HexagonModel* model) {
    HEXAGON_SOFT_ASSERT_EQ(10, ins.size(), "Need 10 inputs for quant8_asym::average_pool_2d");
    HEXAGON_SOFT_ASSERT_EQ(1, outs.size(), "Need 1 output for quant8_asym::average_pool_2d");

    // get parameters
    const hexagon_nn_input& input     = model->getTensor(ins[0]);

    const int32_t padding_left        = model->getScalar<int32_t>(ins[1]);
    const int32_t padding_right       = model->getScalar<int32_t>(ins[2]);
    const int32_t padding_top         = model->getScalar<int32_t>(ins[3]);
    const int32_t padding_bottom      = model->getScalar<int32_t>(ins[4]);
    const int32_t stride_width        = model->getScalar<int32_t>(ins[5]);
    const int32_t stride_height       = model->getScalar<int32_t>(ins[6]);
    const int32_t filter_width        = model->getScalar<int32_t>(ins[7]);
    const int32_t filter_height       = model->getScalar<int32_t>(ins[8]);
    const op_type act                 = model->getQuantizedActivation(ins[9]);

    const hexagon_nn_input& in_min    = model->getQuantizationMin(ins[0]);
    const hexagon_nn_input& in_max    = model->getQuantizationMax(ins[0]);
    const hexagon_nn_input window     = model->createShape(1, filter_height, filter_width, 1);
    const hexagon_nn_input stride     = model->createShape(1, stride_height, stride_width, 1);
    const hexagon_nn_padding_type pad = getPadding(filter_width, filter_height, padding_left,
                                                   padding_right, padding_top, padding_bottom);

    // add node to graph
    return model->addQuant8OperationWithActivation(OP_QuantizedAvgPool_8, pad, act,
    //return model->addQuant8OperationWithActivation(OP_QuantizedAvgPool_8_ref, pad, act,
                                             {input, in_min, in_max, window, stride}, outs);
}

bool concatenation(const std::vector<uint32_t>& ins, const std::vector<uint32_t>& outs,
                   HexagonModel* model) {
    HEXAGON_SOFT_ASSERT_LE(3, ins.size(), "Need at least 3 inputs for quant8_asym::concatenation");
    HEXAGON_SOFT_ASSERT_EQ(1, outs.size(), "Need 1 output for quant8_asym::concatenation");

    const size_t numInputTensors = ins.size() - 1;

    // get parameters
    std::vector<hexagon_nn_input> inputs(numInputTensors * 3 + 1);
    for (size_t i = 0; i < numInputTensors; ++i) {
        inputs[i+1+numInputTensors*0] = model->getTensor(ins[i]);
        inputs[i+1+numInputTensors*1] = model->getQuantizationMin(ins[i]);
        inputs[i+1+numInputTensors*2] = model->getQuantizationMax(ins[i]);
    }

    // axis being concatenated
    const int32_t axis = model->getScalar<int32_t>(ins[numInputTensors]);
    const int32_t dims = model->getShape(ins[0]).dimensions.size();
    inputs[0] = model->createScalar<int32_t>(axis + (4 - dims));

    // add node to graph
    return model->addBasicOperation(OP_QuantizedConcat_8, NN_PAD_NA, inputs, outs);
}

bool conv_2d(const std::vector<uint32_t>& ins, const std::vector<uint32_t>& outs,
             HexagonModel* model) {
    HEXAGON_SOFT_ASSERT_EQ(10, ins.size(), "Need 10 inputs for quant8_asym::conv_2d");
    HEXAGON_SOFT_ASSERT_EQ(1, outs.size(), "Need 1 output for quant8_asym::conv_2d");

    // get parameters
    const hexagon_nn_input& input      = model->getTensor(ins[0]);
    const hexagon_nn_input  filter     = model->createConvFilterTensor(ins[1]);
    const hexagon_nn_input& bias       = model->getTensor(ins[2]);

    const int32_t padding_left         = model->getScalar<int32_t>(ins[3]);
    const int32_t padding_right        = model->getScalar<int32_t>(ins[4]);
    const int32_t padding_top          = model->getScalar<int32_t>(ins[5]);
    const int32_t padding_bottom       = model->getScalar<int32_t>(ins[6]);
    const int32_t stride_width         = model->getScalar<int32_t>(ins[7]);
    const int32_t stride_height        = model->getScalar<int32_t>(ins[8]);
    const op_type act                  = model->getQuantizedActivation(ins[9]);

    const hexagon_nn_input& input_min  = model->getQuantizationMin(ins[0]);
    const hexagon_nn_input& input_max  = model->getQuantizationMax(ins[0]);
    const hexagon_nn_input& filter_min = model->getQuantizationMin(ins[1]);
    const hexagon_nn_input& filter_max = model->getQuantizationMax(ins[1]);

    const hexagon_nn_input stride      = model->createShape(1, stride_height, stride_width, 1);
    const Shape filterShape            = model->getShape(ins[1]);
    const hexagon_nn_padding_type pad  = getPadding(filterShape.dimensions[2],
                                                   filterShape.dimensions[1],
                                                   padding_left, padding_right,
                                                   padding_top, padding_bottom);

    // add node to graph
    return model->addFusedQuant8Operation(OP_QuantizedConv2d_8x8to32, pad, bias, act,
                                          {input, filter, input_min, input_max,
                                            filter_min, filter_max, stride}, outs);
}

bool depthwise_conv_2d(const std::vector<uint32_t>& ins, const std::vector<uint32_t>& outs,
                       HexagonModel* model) {
    HEXAGON_SOFT_ASSERT_EQ(11, ins.size(), "Need 11 inputs for quant8_asym::depthwise_conv_2d");
    HEXAGON_SOFT_ASSERT_EQ(1, outs.size(), "Need 1 output for quant8_asym::depthwise_conv_2d");

    // get parameters
    const hexagon_nn_input& input       = model->getTensor(ins[0]);
    const hexagon_nn_input& bias        = model->getTensor(ins[2]);

    const int32_t padding_left          = model->getScalar<int32_t>(ins[3]);
    const int32_t padding_right         = model->getScalar<int32_t>(ins[4]);
    const int32_t padding_top           = model->getScalar<int32_t>(ins[5]);
    const int32_t padding_bottom        = model->getScalar<int32_t>(ins[6]);
    const int32_t stride_width          = model->getScalar<int32_t>(ins[7]);
    const int32_t stride_height         = model->getScalar<int32_t>(ins[8]);
    const int32_t depth_multiplier      = model->getScalar<int32_t>(ins[9]);
    const op_type act                   = model->getQuantizedActivation(ins[10]);

    const hexagon_nn_input& input_min  = model->getQuantizationMin(ins[0]);
    const hexagon_nn_input& input_max  = model->getQuantizationMax(ins[0]);
    const hexagon_nn_input& filter_min = model->getQuantizationMin(ins[0]);
    const hexagon_nn_input& filter_max = model->getQuantizationMax(ins[0]);
    const hexagon_nn_input filter      = model->createDepthwiseFilterTensor(ins[1],
                                                                            depth_multiplier);
    const hexagon_nn_input stride      = model->createShape(1, stride_height, stride_width, 1);
    const Shape filterShape            = model->getShape(ins[1]);
    const hexagon_nn_padding_type pad  = getPadding(filterShape.dimensions[2],
                                                    filterShape.dimensions[1],
                                                    padding_left, padding_right,
                                                    padding_top, padding_bottom);

    // add node to graph
    return model->addFusedQuant8Operation(OP_QuantizedDepthwiseConv2d_8x8to32, pad, bias, act,
                                          {input, filter, input_min, input_max, filter_min,
                                            filter_max, stride}, outs);
}

bool dequantize(const std::vector<uint32_t>& ins, const std::vector<uint32_t>& outs,
                HexagonModel* model) {
    HEXAGON_SOFT_ASSERT_EQ(1, ins.size(), "Need 1 input for quant8_asym::dequantize");
    HEXAGON_SOFT_ASSERT_EQ(1, outs.size(), "Need 1 output for quant8_asym::dequantize");

    // get parameters
    const hexagon_nn_input& input     = model->getTensor(ins[0]);

    const hexagon_nn_input& input_min = model->getQuantizationMin(ins[0]);
    const hexagon_nn_input& input_max = model->getQuantizationMax(ins[0]);

    // add node to graph
    return model->addBasicOperation(OP_Dequantize, NN_PAD_NA, {input, input_min, input_max}, outs);
}

bool fully_connected(const std::vector<uint32_t>& ins, const std::vector<uint32_t>& outs,
                     HexagonModel* model) {
    HEXAGON_SOFT_ASSERT_EQ(4, ins.size(), "Need 4 inputs for quant8::fully_connected");
    HEXAGON_SOFT_ASSERT_EQ(1, outs.size(), "Need 1 output for quant8::fully_connected");

    // get parameters
    const hexagon_nn_input& input       = model->getTensor(ins[0]);
    const hexagon_nn_input& weights     = model->getTensor(ins[1]);
    const hexagon_nn_input& bias        = model->getTensor(ins[2]);

    const op_type act                   = model->getQuantizedActivation(ins[3]);

    const hexagon_nn_input& input_min   = model->getQuantizationMin(ins[0]);
    const hexagon_nn_input& input_max   = model->getQuantizationMax(ins[0]);
    const hexagon_nn_input& weights_min = model->getQuantizationMin(ins[1]);
    const hexagon_nn_input& weights_max = model->getQuantizationMax(ins[1]);

    // add node to graph
    return model->addFusedQuant8Operation(OP_QuantizedMatMul_8x8to32, NN_PAD_NA, bias, act,
                                          {input, weights, input_min, input_max,
                                            weights_min, weights_max}, outs);
}

bool logistic(const std::vector<uint32_t>& ins, const std::vector<uint32_t>& outs,
              HexagonModel* model) {
    HEXAGON_SOFT_ASSERT_EQ(1, ins.size(), "Need 1 input for quant8_asym::logistic");
    HEXAGON_SOFT_ASSERT_EQ(1, outs.size(), "Need 1 output for quant8_asym::logistic");

    // get parameters
    const hexagon_nn_input& input = model->getTensor(ins[0]);

    const hexagon_nn_input& input_min = model->getQuantizationMin(ins[0]);

    // TFLite uses different max value
    const hexagon_nn_input  input_max = model->createQuantizationValue(ins[0], 256);

    // add node to graph
    //return model->addBasicOperation(OP_QuantizedSigmoid_8_ref, NN_PAD_NA,
    return model->addBasicOperation(OP_QuantizedSigmoid_8, NN_PAD_NA,
                                    {input, input_min, input_max}, outs);
}

bool max_pool_2d(const std::vector<uint32_t>& ins, const std::vector<uint32_t>& outs,
                 HexagonModel* model) {
    HEXAGON_SOFT_ASSERT_EQ(10, ins.size(), "Need 10 inputs for quant8_asym::max_pool_2d");
    HEXAGON_SOFT_ASSERT_EQ(1, outs.size(), "Need 1 output for quant8_asym::max_pool_2d");

    // get parameters
    const hexagon_nn_input& input     = model->getTensor(ins[0]);

    const int32_t padding_left        = model->getScalar<int32_t>(ins[1]);
    const int32_t padding_right       = model->getScalar<int32_t>(ins[2]);
    const int32_t padding_top         = model->getScalar<int32_t>(ins[3]);
    const int32_t padding_bottom      = model->getScalar<int32_t>(ins[4]);
    const int32_t stride_width        = model->getScalar<int32_t>(ins[5]);
    const int32_t stride_height       = model->getScalar<int32_t>(ins[6]);
    const int32_t filter_width        = model->getScalar<int32_t>(ins[7]);
    const int32_t filter_height       = model->getScalar<int32_t>(ins[8]);
    const op_type act                 = model->getQuantizedActivation(ins[9]);

    const hexagon_nn_input& input_min = model->getQuantizationMin(ins[0]);
    const hexagon_nn_input& input_max = model->getQuantizationMax(ins[0]);
    const hexagon_nn_input window     = model->createShape(1, filter_height, filter_width, 1);
    const hexagon_nn_input stride     = model->createShape(1, stride_height, stride_width, 1);
    const hexagon_nn_padding_type pad = getPadding(filter_width, filter_height, padding_left,
                                                   padding_right, padding_top, padding_bottom);

    // add node to graph
    return model->addQuant8OperationWithActivation(OP_QuantizedMaxPool_8, pad, act,
                                             {input, input_min, input_max, window, stride}, outs);
}

bool mul(const std::vector<uint32_t>& ins, const std::vector<uint32_t>& outs,
         HexagonModel* model) {
    HEXAGON_SOFT_ASSERT_EQ(3, ins.size(), "Need 3 inputs for quant8_asym::mul");
    HEXAGON_SOFT_ASSERT_EQ(1, outs.size(), "Need 1 output for quant8_asym::mul");

    // get parameters
    const hexagon_nn_input& in1     = model->getTensor(ins[0]);
    const hexagon_nn_input& in2     = model->getTensor(ins[1]);

    const op_type act               = model->getQuantizedActivation(ins[2]);

    const hexagon_nn_input& in1_min = model->getQuantizationMin(ins[0]);
    const hexagon_nn_input& in1_max = model->getQuantizationMax(ins[0]);
    const hexagon_nn_input& in2_min = model->getQuantizationMin(ins[1]);
    const hexagon_nn_input& in2_max = model->getQuantizationMax(ins[1]);

    // add node to graph
    return model->addFusedQuant8Operation(OP_QuantizedMul_8x8to32, NN_PAD_NA, {}, act,
                                          {in1, in1_min, in1_max, in2, in2_min, in2_max}, outs);
}

bool relu(const std::vector<uint32_t>& ins, const std::vector<uint32_t>& outs,
          HexagonModel* model) {
    HEXAGON_SOFT_ASSERT_EQ(1, ins.size(), "Need 1 input for quant8_asym::relu");
    HEXAGON_SOFT_ASSERT_EQ(1, outs.size(), "Need 1 output for quant8_asym::relu");

    // get parameters
    const hexagon_nn_input& input     = model->getTensor(ins[0]);

    const hexagon_nn_input& input_min = model->getQuantizationMin(ins[0]);
    const hexagon_nn_input& input_max = model->getQuantizationMax(ins[0]);

    // add node to graph
    return model->addBasicOperation(OP_QuantizedRelu_8, NN_PAD_NA,
                                    {input, input_min, input_max}, outs);
}

bool relu1(const std::vector<uint32_t>& ins, const std::vector<uint32_t>& outs,
           HexagonModel* model) {
    HEXAGON_SOFT_ASSERT_EQ(1, ins.size(), "Need 1 input for quant8_asym::relu1");
    HEXAGON_SOFT_ASSERT_EQ(1, outs.size(), "Need 1 output for quant8_asym::relu1");

    // get parameters
    const hexagon_nn_input& input = model->getTensor(ins[0]);
    const hexagon_nn_input  min   = model->createScalar(-1.0f);
    const hexagon_nn_input  max   = model->createScalar(1.0f);

    const hexagon_nn_input& input_min = model->getQuantizationMin(ins[0]);
    const hexagon_nn_input& input_max = model->getQuantizationMax(ins[0]);

    // add node to graph
    return model->addBasicOperation(OP_QuantizedClamp_8, NN_PAD_NA,
                                    {input, input_min, input_max, min, max}, outs);
}

bool relu6(const std::vector<uint32_t>& ins, const std::vector<uint32_t>& outs,
           HexagonModel* model) {
    HEXAGON_SOFT_ASSERT_EQ(1, ins.size(), "Need 1 input for quant8_asym::relu6");
    HEXAGON_SOFT_ASSERT_EQ(1, outs.size(), "Need 1 output for quant8_asym::relu6");

    // get parameters
    const hexagon_nn_input& input = model->getTensor(ins[0]);
    const hexagon_nn_input  max   = model->createScalar(6.0f);

    const hexagon_nn_input& input_min = model->getQuantizationMin(ins[0]);
    const hexagon_nn_input& input_max = model->getQuantizationMax(ins[0]);

    // add node to graph
    return model->addBasicOperation(OP_QuantizedReluX_8, NN_PAD_NA,
                                    {input, input_min, input_max, max}, outs);
}

bool reshape(const std::vector<uint32_t>& ins, const std::vector<uint32_t>& outs,
             HexagonModel* model) {
    HEXAGON_SOFT_ASSERT_EQ(2, ins.size(), "Need 2 inputs for quant8_asym::reshape");
    HEXAGON_SOFT_ASSERT_EQ(1, outs.size(), "Need 1 output for quant8_asym::reshape");

    // get parameters
    const hexagon_nn_input& input   = model->getTensor(ins[0]);
    const hexagon_nn_input& newdims = model->getTensor(ins[1]);

    const hexagon_nn_input& input_min = model->getQuantizationMin(ins[0]);
    const hexagon_nn_input& input_max = model->getQuantizationMax(ins[0]);

    // add node to graph
    return model->addBasicOperation(OP_QuantizedReshape, NN_PAD_NA,
                                    {input, newdims, input_min, input_max}, outs);
}

bool softmax(const std::vector<uint32_t>& ins, const std::vector<uint32_t>& outs,
             HexagonModel* model) {
    HEXAGON_SOFT_ASSERT_EQ(2, ins.size(), "Need 2 inputs for quant8_asym::softmax");
    HEXAGON_SOFT_ASSERT_EQ(1, outs.size(), "Need 1 output for quant8_asym::softmax");

    // get parameters
    const hexagon_nn_input& input = model->getTensor(ins[0]);
    const hexagon_nn_input& beta  = model->getTensor(ins[1]);

    const hexagon_nn_input& input_min = model->getQuantizationMin(ins[0]);
    const hexagon_nn_input& input_max = model->getQuantizationMax(ins[0]);

    // add node to graph
    //return model->addBasicOperation(OP_QuantizedSoftmax_8_ref, NN_PAD_NA,
    return model->addBasicOperation(OP_QuantizedSoftmax_8, NN_PAD_NA,
                                    {input, input_min, input_max, beta}, outs);
}

}  // quant8_asym namespace

}  // namespace

OperationPrepareTable& getOperationPrepareTable() {
    static OperationPrepareTable table = {
        // -------------------------- 32-BIT FLOAT ----------------------------
        {{OperationType::ADD, OperandType::TENSOR_FLOAT32},             float32::add             },
        {{OperationType::AVERAGE_POOL_2D, OperandType::TENSOR_FLOAT32}, float32::average_pool_2d },
        {{OperationType::CONCATENATION, OperandType::TENSOR_FLOAT32},   float32::concatenation   },
        {{OperationType::CONV_2D, OperandType::TENSOR_FLOAT32},         float32::conv_2d         },
        {{OperationType::DEPTHWISE_CONV_2D, OperandType::TENSOR_FLOAT32},
                                                                       float32::depthwise_conv_2d},
//      {{OperationType::DEPTH_TO_SPACE, OperandType::TENSOR_FLOAT32},  float32::depth_to_space  },
//      {{OperationType::EMBEDDING_LOOKUP, OperandType::TENSOR_FLOAT32},
//                                                                      float32::embedding_lookup},
//      {{OperationType::FLOOR, OperandType::TENSOR_FLOAT32},           float32::floor           },
        {{OperationType::FULLY_CONNECTED, OperandType::TENSOR_FLOAT32}, float32::fully_connected },
//      {{OperationType::HASHTABLE_LOOKUP, OperandType::TENSOR_FLOAT32},
//                                                                      float32::hashtable_lookup},
//      {{OperationType::L2_NORMALIZATION, OperandType::TENSOR_FLOAT32},
//                                                                      float32::l2_normalization},
        {{OperationType::L2_POOL_2D, OperandType::TENSOR_FLOAT32},      float32::l2_pool_2d      },
        {{OperationType::LOCAL_RESPONSE_NORMALIZATION, OperandType::TENSOR_FLOAT32},
                                                            float32::local_response_normalization},
        {{OperationType::LOGISTIC, OperandType::TENSOR_FLOAT32},        float32::logistic        },
//      {{OperationType::LSH_PROJECTION, OperandType::TENSOR_FLOAT32},  float32::lsh_projection  },
//      {{OperationType::LSTM, OperandType::TENSOR_FLOAT32},            float32::lstm            },
        {{OperationType::MAX_POOL_2D, OperandType::TENSOR_FLOAT32},     float32::max_pool_2d     },
        {{OperationType::MUL, OperandType::TENSOR_FLOAT32},             float32::mul             },
        {{OperationType::RELU, OperandType::TENSOR_FLOAT32},            float32::relu            },
        {{OperationType::RELU1, OperandType::TENSOR_FLOAT32},           float32::relu1           },
        {{OperationType::RELU6, OperandType::TENSOR_FLOAT32},           float32::relu6           },
        {{OperationType::RESHAPE, OperandType::TENSOR_FLOAT32},         float32::reshape         },
        {{OperationType::RESIZE_BILINEAR, OperandType::TENSOR_FLOAT32}, float32::resize_bilinear },
//      {{OperationType::RNN, OperandType::TENSOR_FLOAT32},             float32::rnn             },
        {{OperationType::SOFTMAX, OperandType::TENSOR_FLOAT32},         float32::softmax         },
//      {{OperationType::SPACE_TO_DEPTH, OperandType::TENSOR_FLOAT32},  float32::space_to_depth  },
//      {{OperationType::SVDF, OperandType::TENSOR_FLOAT32},            float32::svdf            },
        {{OperationType::TANH, OperandType::TENSOR_FLOAT32},            float32::tanh            },

        // -------------------- QUANTIZED 8-BIT ASYMMETRICAL ------------------
        {{OperationType::ADD, OperandType::TENSOR_QUANT8_ASYMM},     quant8_asym::add            },
        {{OperationType::AVERAGE_POOL_2D, OperandType::TENSOR_QUANT8_ASYMM},
                                                                     quant8_asym::average_pool_2d},
        {{OperationType::CONCATENATION, OperandType::TENSOR_QUANT8_ASYMM},
                                                                     quant8_asym::concatenation  },
        {{OperationType::CONV_2D, OperandType::TENSOR_QUANT8_ASYMM},    quant8_asym::conv_2d     },
        {{OperationType::DEPTHWISE_CONV_2D, OperandType::TENSOR_QUANT8_ASYMM},
                                                                   quant8_asym::depthwise_conv_2d},
//      {{OperationType::DEPTH_TO_SPACE, OperandType::TENSOR_QUANT8_ASYMM},
//                                                                   quant8_asym::depth_to_space },
        {{OperationType::DEQUANTIZE, OperandType::TENSOR_QUANT8_ASYMM}, quant8_asym::dequantize  },
//      {{OperationType::EMBEDDING_LOOKUP, OperandType::TENSOR_QUANT8_ASYMM},
//                                                                  quant8_asym::embedding_lookup},
        {{OperationType::FULLY_CONNECTED, OperandType::TENSOR_QUANT8_ASYMM},
                                                                     quant8_asym::fully_connected},
//      {{OperationType::HASHTABLE_LOOKUP, OperandType::TENSOR_QUANT8_ASYMM},
//                                                                  quant8_asym::hashtable_lookup},
        {{OperationType::LOGISTIC, OperandType::TENSOR_QUANT8_ASYMM}, quant8_asym::logistic      },
//      {{OperationType::LSH_PROJECTION, OperandType::TENSOR_QUANT8_ASYMM},
//                                                                   quant8_asym::lsh_projection },
        {{OperationType::MAX_POOL_2D, OperandType::TENSOR_QUANT8_ASYMM}, quant8_asym::max_pool_2d},
        {{OperationType::MUL, OperandType::TENSOR_QUANT8_ASYMM},     quant8_asym::mul            },
        {{OperationType::RELU, OperandType::TENSOR_QUANT8_ASYMM},    quant8_asym::relu           },
        {{OperationType::RELU1, OperandType::TENSOR_QUANT8_ASYMM},   quant8_asym::relu1          },
        {{OperationType::RELU6, OperandType::TENSOR_QUANT8_ASYMM},   quant8_asym::relu6          },
        {{OperationType::RESHAPE, OperandType::TENSOR_QUANT8_ASYMM}, quant8_asym::reshape        },
        {{OperationType::SOFTMAX, OperandType::TENSOR_QUANT8_ASYMM}, quant8_asym::softmax        },
//      {{OperationType::SPACE_TO_DEPTH, OperandType::TENSOR_QUANT8_ASYMM},
//                                                                   quant8_asym::space_to_depth },
    };
    return table;
}

} // namespace hexagon
} // namespace implementation
} // namespace V1_0
} // namespace neuralnetworks
} // namespace hardware
} // namespace android
