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
#include <numeric>
#include <unordered_set>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace V1_0 {
namespace implementation {
namespace hexagon {

static std::vector<OperandInfo> getOperandsInfo(const NeuralnetworksModel& model,
                                                const std::vector<RunTimePoolInfo>& pools) {
    std::vector<OperandInfo> info(model.operands.size());
    for (size_t i = 0; i < model.operands.size(); ++i) {
        const Operand& operand = model.operands[i];
        info[i] = {
            .type       = operand.type,
            .dimensions = operand.dimensions,
            .scale      = operand.scale,
            .zeroPoint  = operand.zeroPoint,
            .lifetime   = operand.lifetime,
            .buffer     = const_cast<uint8_t*>(getData(operand, model.operandValues, pools)),
            .length     = operand.location.length,
        };
    }
    return info;
}

Model::Model(const NeuralnetworksModel& model) : mNodeCount(0), mCompiled(false) {
    mGraphId = hexagon::Controller::getInstance().init();
    hexagon::Controller::getInstance().set_debug_level(mGraphId, 99);

    mPools = mapPools(model.pools);
    mOperands = getOperandsInfo(model, mPools);
    std::for_each(mPools.begin(), mPools.end(), [](RunTimePoolInfo& mem) { mem.update(); });

    mOperations = model.operations;
    mInputs     = model.inputIndexes;
    mOutputs    = model.outputIndexes;
}

Model::Model(Model&& other) {
    *this = std::move(other);
}

Model& Model::operator=(Model&& other) {
    mNodeCount      = other.mNodeCount;
    mGraphId        = other.mGraphId;
    mCompiled       = other.mCompiled;
    mOperands       = std::move(other.mOperands);
    mOperations     = std::move(other.mOperations);
    mInputs         = std::move(other.mInputs);
    mOutputs        = std::move(other.mOutputs);
    mPools          = std::move(other.mPools);
    other.mGraphId  = {};
    other.mCompiled = false;
    return *this;
}

Model::~Model() {
    if (mGraphId != hexagon_nn_nn_id{}) {
        hexagon::Controller::getInstance().teardown(mGraphId);
    }
}

std::string Model::getDebugLog() {
    char buffer[16*1024];
    int err = hexagon::Controller::getInstance().getlog(
            mGraphId, reinterpret_cast<uint8_t*>(buffer), sizeof(buffer));
    HEXAGON_SOFT_ASSERT_EQ(0, err, "failed getDebugLog");
    return buffer;
}

std::string Model::getLog() {
    char buffer[16*1024];
    int err = hexagon::Controller::getInstance().snpprint(
            mGraphId, reinterpret_cast<uint8_t*>(buffer), sizeof(buffer));
    HEXAGON_SOFT_ASSERT_EQ(0, err, "failed getLog");
    return buffer;
}

uint32_t Model::getNextNode() {
    return ++mNodeCount;
}

const int32_t* Model::getPointer(uint32_t operand) {
    return reinterpret_cast<const int32_t*>(mOperands[operand].buffer);
}

Shape Model::getShape(uint32_t operand) {
    return {
        .type       = mOperands[operand].type,
        .dimensions = mOperands[operand].dimensions,
        .scale      = mOperands[operand].scale,
        .offset     = mOperands[operand].zeroPoint,
    };
}

bool Model::setShape(uint32_t operand, const Shape& shape) {
    const hexagon_nn_output& output = mOperands[operand].hexagon_output;
    HEXAGON_SOFT_ASSERT_EQ(output, hexagon_nn_output{}, "Output has already been set");
    //mOperands[operand].type       = shape.type;
    mOperands[operand].dimensions = shape.dimensions;
    //mOperands[operand].scale      = shape.scale;
    //mOperands[operand].zeroPoint  = shape.offset;
    return true;
}

bool Model::isConstant(uint32_t operand) {
    OperandLifeTime lifetime = mOperands[operand].lifetime;
    return lifetime == OperandLifeTime::CONSTANT_COPY ||
            lifetime == OperandLifeTime::CONSTANT_REFERENCE;
}

hexagon_nn_input Model::createTensorInternal(uint32_t B, uint32_t H, uint32_t W, uint32_t D,
                                             const uint8_t* ptr, size_t size) {
    uint32_t node = getNextNode();
    bool success = hexagon::Controller::getInstance().append_const_node(
            mGraphId, node, B, H, W, D, ptr, size) == 0;
    HEXAGON_SOFT_ASSERT(success, "Failed to create tensor");
    return {.src_id = node, .output_idx = 0};
}

hexagon_nn_input Model::createShape(uint32_t B, uint32_t H, uint32_t W, uint32_t D) {
    uint32_t dump = 0;
    return createTensorInternal(B, H, W, D, reinterpret_cast<uint8_t*>(&dump), sizeof(dump));
}

hexagon_nn_input Model::addOperand(uint32_t operandIndex) {
    const OperandInfo& operand = mOperands[operandIndex];
    std::vector<uint32_t> dims = getAlignedDimensions(operand.dimensions, 4);
    HEXAGON_SOFT_ASSERT_NE(0ul, dims.size(), "Rank must be at most 4");
    hexagon_nn_input result = createTensorInternal(dims[0], dims[1], dims[2], dims[3],
                                                   operand.buffer, operand.length);
    HEXAGON_SOFT_ASSERT_NE(hexagon_nn_input{}, result, "Failed to add operand");
    return result;
}

const hexagon_nn_input& Model::getTensor(uint32_t operand) {
    hexagon_nn_input& tensor = mOperands[operand].hexagon_input;
    if (tensor == hexagon_nn_input{}) {
        tensor = addOperand(operand);
    }
    return tensor;
}

const hexagon_nn_input& Model::getQuantizationMin(uint32_t operand) {
    OperandInfo& operandInfo = mOperands[operand];
    if (operandInfo.hexagon_input_min == hexagon_nn_input{}) {
        float real_value = (std::numeric_limits<uint8_t>::min() - operandInfo.zeroPoint) * operandInfo.scale;
        operandInfo.hexagon_input_min = createValues<float>({real_value});
    }
    return operandInfo.hexagon_input_min;
}

const hexagon_nn_input& Model::getQuantizationMax(uint32_t operand) {
    OperandInfo& operandInfo = mOperands[operand];
    if (operandInfo.hexagon_input_max == hexagon_nn_input{}) {
        float real_value = (std::numeric_limits<uint8_t>::max() - operandInfo.zeroPoint) * operandInfo.scale;
        operandInfo.hexagon_input_max = createValues<float>({real_value});
    }
    return operandInfo.hexagon_input_max;
}

hexagon_nn_input Model::createQuantizationValue(uint32_t operand, uint32_t quant_value) {
    OperandInfo& operandInfo = mOperands[operand];
    float real_value = (quant_value - operandInfo.zeroPoint) * operandInfo.scale;
    return createValues<float>({real_value});
}

hexagon_nn_input Model::createConvFilterTensor(uint32_t operand) {
    OperandInfo& operandInfo = mOperands[operand];
    std::vector<uint32_t> dims = getAlignedDimensions(mOperands[operand].dimensions, 4);
    HEXAGON_SOFT_ASSERT_NE(0ul, dims.size(), "Need at most 4 dimensions");
    // NHWC --> HWCN
    if (getShape(operand).type == OperandType::TENSOR_FLOAT32) {
        std::vector<float> transposed = transpose<float>(dims[0], dims[1]*dims[2]*dims[3],
                                                         reinterpret_cast<const float*>(operandInfo.buffer));
        return createTensorInternal(dims[1], dims[2], dims[3], dims[0],
                                    reinterpret_cast<const uint8_t*>(transposed.data()), operandInfo.length);
    }
    else {
        std::vector<uint8_t> transposed = transpose<uint8_t>(dims[0], dims[1]*dims[2]*dims[3],
                                                             reinterpret_cast<const uint8_t*>(operandInfo.buffer));
        return createTensorInternal(dims[1], dims[2], dims[3], dims[0],
                                    reinterpret_cast<const uint8_t*>(transposed.data()), operandInfo.length);
    }
}

hexagon_nn_input Model::createDepthwiseFilterTensor(uint32_t operand, int32_t depth_multiplier) {
    OperandInfo& operandInfo = mOperands[operand];
    std::vector<uint32_t> dims = getAlignedDimensions(mOperands[operand].dimensions, 4);
    HEXAGON_SOFT_ASSERT_NE(0ul, dims.size(), "Need at most 4 dimensions");
    // NHWC --> HWCN
    return createTensorInternal(dims[1], dims[2], dims[3] / depth_multiplier, dims[0] * depth_multiplier,
                                operandInfo.buffer, operandInfo.length);
}

hexagon_nn_input Model::createFullyConnectedWeightTensor(uint32_t operand) {
    OperandInfo& operandInfo = mOperands[operand];
    std::vector<uint32_t> dims = getAlignedDimensions(mOperands[operand].dimensions, 4);
    HEXAGON_SOFT_ASSERT_NE(0ul, dims.size(), "Need at most 2 dimensions");
    // WC --> CW
    if (getShape(operand).type == OperandType::TENSOR_FLOAT32) {
        std::vector<float> transposed = transpose<float>(dims[0], dims[1],
                                                         reinterpret_cast<const float*>(operandInfo.buffer));
        return createTensorInternal(1, 1, dims[1], dims[0],
                                    reinterpret_cast<const uint8_t*>(transposed.data()), operandInfo.length);
    }
    else {
        std::vector<uint8_t> transposed = transpose<uint8_t>(dims[0], dims[1],
                                                             reinterpret_cast<const uint8_t*>(operandInfo.buffer));
        return createTensorInternal(1, 1, dims[1], dims[0],
                                    reinterpret_cast<const uint8_t*>(transposed.data()), operandInfo.length);
    }
}

op_type Model::getFloatActivation(uint32_t operand) {
    return getFloatActivationFunction(getScalar<FusedActivationFunc>(operand));
}

op_type Model::getQuantizedActivation(uint32_t operand) {
    return getQuantizedActivationFunction(getScalar<FusedActivationFunc>(operand));
}

static bool verifyOperationInputs(const std::vector<hexagon_nn_input>& inputs) {
    for (const hexagon_nn_input& input : inputs) {
        if (input == hexagon_nn_input{}) {
            return false;
        }
    }
    return true;
}

static bool verifyOperationOutputs(const std::vector<hexagon_nn_output>& outputs) {
    for (const hexagon_nn_output& output : outputs) {
        if (output == hexagon_nn_output{}) {
            return false;
        }
    }
    return true;
}

uint32_t Model::addOperationInternal(op_type op, hexagon_nn_padding_type pad,
                                     const std::vector<hexagon_nn_input>& inputs,
                                     const std::vector<hexagon_nn_output>& outputs) {
    HEXAGON_SOFT_ASSERT(verifyOperationInputs(inputs), "error adding operation: one or more inputs is invalid");
    HEXAGON_SOFT_ASSERT(verifyOperationOutputs(outputs), "error adding operation: one or more outputs is invalid");
    uint32_t node = getNextNode();
    return hexagon::Controller::getInstance().append_node(mGraphId, node, op, pad,
            inputs.data(), inputs.size(), outputs.data(), outputs.size()) == 0 ? node : 0;
}

std::vector<hexagon_nn_output> Model::getHexagonOutputs(const std::vector<uint32_t>& operands) {
    std::vector<hexagon_nn_output> outputs;
    for (uint32_t index : operands) {
        const OperandInfo& operand = mOperands[index];
        outputs.push_back(make_hexagon_nn_output(operand.dimensions, getSize(operand.type)));
        if (operand.type == OperandType::TENSOR_QUANT8_ASYMM) {
            outputs.push_back(make_hexagon_nn_output({1, 1, 1, 1}, sizeof(float)));
            outputs.push_back(make_hexagon_nn_output({1, 1, 1, 1}, sizeof(float)));
        }
    }
    return outputs;
}

bool Model::registerHexagonInputs(const std::vector<uint32_t>& operands, uint32_t node) {
    uint32_t idx = 0;
    for (uint32_t i = 0; i < static_cast<uint32_t>(operands.size()); ++i) {
        OperandInfo& operand = mOperands[operands[i]];
        HEXAGON_SOFT_ASSERT_EQ(operand.hexagon_input, hexagon_nn_input{},
                               "Error: operation output has already been registered");
        operand.hexagon_input = {.src_id = node, .output_idx = idx++};
        if (operand.type == OperandType::TENSOR_QUANT8_ASYMM) {
            operand.hexagon_input_min = {.src_id = node, .output_idx = idx++};
            operand.hexagon_input_max = {.src_id = node, .output_idx = idx++};
        }
    }
    return true;
}

bool Model::addBasicOperation(op_type op, hexagon_nn_padding_type pad,
                              const std::vector<hexagon_nn_input>& inputs,
                              const std::vector<uint32_t>& outputs) {
    std::vector<hexagon_nn_output> outs = getHexagonOutputs(outputs);
    uint32_t node = addOperationInternal(op, pad, inputs, outs);
    HEXAGON_SOFT_ASSERT_NE(0, node, "Error adding base operation");
    return registerHexagonInputs(outputs, node);
}

std::vector<hexagon_nn_input> Model::setupActivationArgs(op_type op) {
    switch (op) {
        case OP_Nop:
            return {};
        case OP_Relu_f:
            FALLTHROUGH_INTENDED;
        case OP_QuantizedRelu_8:
            return {};
        case OP_ReluX_f:
            FALLTHROUGH_INTENDED;
        case OP_QuantizedReluX_8:
            return {createValues<float>({6.0f})};
        case OP_Clamp_f:
            FALLTHROUGH_INTENDED;
        case OP_QuantizedClamp_8:
            return {createValues<float>({-1.0f}), createValues<float>({1.0f})};
        default:
            HEXAGON_SOFT_ASSERT(false, "Unknown activation symbol " << op);
    }
}

bool Model::addFloatOperationWithActivation(op_type op, hexagon_nn_padding_type pad, op_type activation,
                                            const std::vector<hexagon_nn_input>& inputs,
                                            const std::vector<uint32_t>& outputs) {
    std::vector<hexagon_nn_output> outs = getHexagonOutputs(outputs);
    std::vector<hexagon_nn_input> actArgs = setupActivationArgs(activation);

    uint32_t node = addOperationInternal(op, pad, inputs, outs);
    HEXAGON_SOFT_ASSERT_NE(0, node, "Error adding base operation");

    std::vector<hexagon_nn_input> buffer_in = {{.src_id = node, .output_idx = 0}};
    buffer_in.insert(buffer_in.end(), actArgs.begin(), actArgs.end());
    node = addOperationInternal(activation, NN_PAD_NA, buffer_in, outs);
    HEXAGON_SOFT_ASSERT_NE(0, node, "Error adding activation operation");

    return registerHexagonInputs(outputs, node);
}

bool Model::addQuant8OperationWithActivation(op_type op, hexagon_nn_padding_type pad, op_type activation,
                                             const std::vector<hexagon_nn_input>& inputs,
                                             const std::vector<uint32_t>& outputs) {
    std::vector<hexagon_nn_output> outs = getHexagonOutputs(outputs);
    std::vector<hexagon_nn_input> actArgs = setupActivationArgs(activation);

    uint32_t node = addOperationInternal(op, pad, inputs, outs);
    HEXAGON_SOFT_ASSERT_NE(0, node, "Error adding base operation");

    std::vector<hexagon_nn_input> buffer_in = {{.src_id = node, .output_idx = 0}, {.src_id = node, .output_idx = 1}, {.src_id = node, .output_idx = 2}};
    buffer_in.insert(buffer_in.end(), actArgs.begin(), actArgs.end());
    node = addOperationInternal(activation, NN_PAD_NA, buffer_in, outs);
    HEXAGON_SOFT_ASSERT_NE(0, node, "Error adding activation operation");

    return registerHexagonInputs(outputs, node);
}

bool Model::addFusedFloatOperation(op_type op,
                                   hexagon_nn_padding_type pad,
                                   const hexagon_nn_input& bias,
                                   op_type activation,
                                   const std::vector<hexagon_nn_input>& inputs,
                                   const std::vector<uint32_t>& outputs) {
    HEXAGON_SOFT_ASSERT_EQ(1, outputs.size(), "addFusedFloatOperation requires 1 output");
    std::vector<hexagon_nn_output> outs = getHexagonOutputs(outputs);
    std::vector<hexagon_nn_input> actArgs = setupActivationArgs(activation);
    uint32_t node;

    node = addOperationInternal(op, pad, inputs, outs);
    HEXAGON_SOFT_ASSERT_NE(0, node, "Error adding base operation");

    if (bias != hexagon_nn_input{}) {
        const hexagon_nn_input buffer1_in = {.src_id = node, .output_idx = 0};
        node = addOperationInternal(OP_BiasAdd_f, NN_PAD_NA, {buffer1_in, bias}, outs);
        HEXAGON_SOFT_ASSERT_NE(0, node, "Error adding bias operation");
    }

    std::vector<hexagon_nn_input> buffer2_in = {{.src_id = node, .output_idx = 0}};
    buffer2_in.insert(buffer2_in.end(), actArgs.begin(), actArgs.end());
    node = addOperationInternal(activation, NN_PAD_NA, buffer2_in, outs);
    HEXAGON_SOFT_ASSERT_NE(0, node, "Error adding activation operation");

    return registerHexagonInputs(outputs, node);
}

bool Model::addFusedQuant8Operation(op_type op,
                                    hexagon_nn_padding_type pad,
                                    const hexagon_nn_input& bias,
                                    op_type activation,
                                    const std::vector<hexagon_nn_input>& inputs,
                                    const std::vector<uint32_t>& outputs) {
    HEXAGON_SOFT_ASSERT_EQ(1, outputs.size(), "addFusedQuant8Operation requires 1 output");
    std::vector<hexagon_nn_input> actArgs = setupActivationArgs(activation);
    uint32_t node;

    hexagon_nn_output tensor_out8 = make_hexagon_nn_output(mOperands[outputs[0]].dimensions, sizeof(uint8_t));
    hexagon_nn_output tensor_out32 = make_hexagon_nn_output(mOperands[outputs[0]].dimensions, sizeof(int32_t));
    hexagon_nn_output scalar_out32 = make_hexagon_nn_output({1, 1, 1, 1}, sizeof(float));

    std::vector<hexagon_nn_output> out8 = {tensor_out8, scalar_out32, scalar_out32};
    std::vector<hexagon_nn_output> out32 = {tensor_out32, scalar_out32, scalar_out32};

    // base operation
    node = addOperationInternal(op, pad, inputs, out32);
    HEXAGON_SOFT_ASSERT_NE(0, node, "Error adding base operation");
    const hexagon_nn_input old_min = {.src_id = node, .output_idx = 1};
    const hexagon_nn_input old_max = {.src_id = node, .output_idx = 2};

    // add bias
    // TODO: prefer OP_BiasAdd_int32
    if (bias != hexagon_nn_input{}) {
        const hexagon_nn_input buffer1_in = {.src_id = node, .output_idx = 0};
        node = addOperationInternal(OP_Add_int32, NN_PAD_NA, {buffer1_in, bias}, {tensor_out32});
        HEXAGON_SOFT_ASSERT_NE(0, node, "Error adding bias operation");
    }

    // requantize
    const hexagon_nn_input& new_min = getQuantizationMin(outputs[0]);
    const hexagon_nn_input& new_max = getQuantizationMax(outputs[0]);
    const hexagon_nn_input buffer2_in = {.src_id = node, .output_idx = 0};
    node = addOperationInternal(OP_Requantize_32to8, NN_PAD_NA, {buffer2_in, old_min, old_max, new_min, new_max}, out8);
    HEXAGON_SOFT_ASSERT_NE(0, node, "Error adding requantize operation");

    // activation
    std::vector<hexagon_nn_input> buffer3 = {{.src_id = node, .output_idx = 0}, {.src_id = node, .output_idx = 1}, {.src_id = node, .output_idx = 2}};
    buffer3.insert(buffer3.end(), actArgs.begin(), actArgs.end());
    node = addOperationInternal(activation, NN_PAD_NA, buffer3, out8);
    HEXAGON_SOFT_ASSERT_NE(0, node, "Error adding activation operation");

    return registerHexagonInputs(outputs, node);
}

bool Model::verifyOperations() {
    std::vector<bool> supported = supportedOperations();
    return std::all_of(supported.begin(), supported.end(), [](bool valid) { return valid; });
}

bool Model::verifyOperands() {
    for (const OperandInfo& operand : mOperands) {
        for (uint32_t dim : operand.dimensions) {
            HEXAGON_SOFT_ASSERT_NE(0, dim, "At least one operand with unknown dimension");
        }
    }
    return true;
}

bool Model::addInputs() {
    // prepare OP_INPUT's outputs
    std::vector<hexagon_nn_output> outs;
    for (size_t i = 0; i < mInputs.size(); ++i) {
        OperandInfo& operand = mOperands[mInputs[i]];
        outs.push_back(make_hexagon_nn_output(operand.dimensions, getSize(operand.type)));
    }

    // add single input node for entire graph
    uint32_t node = addOperationInternal(OP_INPUT, NN_PAD_NA, {}, outs);
    HEXAGON_SOFT_ASSERT_NE(0, node, "Error adding input operation");

    // update operand information
    for (size_t i = 0; i < mInputs.size(); ++i) {
        OperandInfo& operand = mOperands[mInputs[i]];
        operand.hexagon_input = {.src_id = node, .output_idx = static_cast<uint32_t>(i)};
    }

    return true;
}

bool Model::addOperations() {
    for (const Operation& operation : mOperations) {
        OperationType operationType = operation.type;
        OperandType operandType = mOperands[operation.inputs[0]].type;
        OperationTuple opTuple = std::make_pair(operationType, operandType);
        HEXAGON_SOFT_ASSERT(getOperationPrepareTable().find(opTuple) != getOperationPrepareTable().end(),
                            "Operation not found");
        bool success = getOperationPrepareTable()[opTuple](operation.inputs, operation.outputs, this);
        HEXAGON_SOFT_ASSERT(success, "error adding operation");
    }
    return true;
}

bool Model::addOutputs() {
    // prepare OP_OUTPUT's inputs
    std::vector<hexagon_nn_input> ins(mOutputs.size());
    for (size_t i = 0; i < mOutputs.size(); ++i) {
        OperandInfo& operand = mOperands[mOutputs[i]];
        HEXAGON_SOFT_ASSERT_NE(operand.hexagon_input, hexagon_nn_input{},
                               "output operand has not been registered");
        ins[i] = operand.hexagon_input;
    }

    // add single output node for entire graph
    bool success = addBasicOperation(OP_OUTPUT, NN_PAD_NA, ins, {});
    HEXAGON_SOFT_ASSERT(success, "Error adding output operation");

    return true;
}

void Model::resetModel() {
    mCompiled = false;
    for (OperandInfo& operand : mOperands) {
        operand.hexagon_input = {};
        operand.hexagon_output = {};
    }
    if (mGraphId != hexagon_nn_nn_id{}) {
        hexagon::Controller::getInstance().teardown(mGraphId);
    }
    mGraphId = hexagon::Controller::getInstance().init();
    hexagon::Controller::getInstance().set_debug_level(mGraphId, 99);
}

std::vector<bool> Model::supportedOperations() {
    std::vector<bool> supported(mOperations.size());
    for (size_t i = 0; i < supported.size(); ++i) {
        const Operation& operation = mOperations[i];
        auto entry = getOperationCheckTable().find(operation.type);
        if (entry != getOperationCheckTable().end()) {
            supported[i] = entry->second(operation.inputs, operation.outputs, this);
        }
        else {
            supported[i] = false;
        }
    }
    return supported;
}

bool Model::compile() {
    if (!verifyOperations() || !verifyOperands()) {
        return false;
    }

    if (!addInputs() || !addOperations() || !addOutputs()) {
        resetModel();
        return false;
    }

    LOG(INFO) << "Graph constructed:" << getLog();
    LOG(INFO) << "Debug log:" << getDebugLog();

    int err = hexagon::Controller::getInstance().prepare(mGraphId);

    LOG(INFO) << "Graph constructed:" << getLog();
    LOG(INFO) << "Debug log:" << getDebugLog();

    return err == 0;
}

static hexagon_nn_tensordef convertToTensordef(const OperandInfo& operand) {
    std::vector<uint32_t> dimensions = getAlignedDimensions(operand.dimensions, 4);
    return {
        .batches        = dimensions[0],
        .height         = dimensions[1],
        .width          = dimensions[2],
        .depth          = dimensions[3],
        .data           = operand.buffer,
        .dataLen        = static_cast<int32_t>(operand.length),
        .data_valid_len = operand.length, // unused?
        .unused         = 0,
    };
}

static uint32_t getSize(const OperandInfo& operand) {
    return std::accumulate(operand.dimensions.begin(), operand.dimensions.end(),
                           getSize(operand.type), std::multiplies<>{});
}

static void updateOperand(const RequestArgument& inputOutput,
                          const std::vector<RunTimePoolInfo>& pools,
                          OperandInfo* operand) {
    const RunTimePoolInfo& pool = pools[inputOutput.location.poolIndex];
    uint32_t offset = inputOutput.location.offset;

    if (inputOutput.dimensions.size() > 0) {
        operand->dimensions = inputOutput.dimensions;
    }

    operand->buffer = pool.buffer + offset;
    operand->length = getSize(*operand);
}

bool Model::execute(const Request& request) {
    std::vector<RunTimePoolInfo> pools = mapPools(request.pools);

    LOG(INFO) << "REQUEST: " << toString(request);

    // prepare inputs
    std::vector<hexagon_nn_tensordef> inputs;
    for (size_t i = 0; i < request.inputs.size(); ++i) {
        OperandInfo& operandInfo = mOperands[mInputs[i]];
        updateOperand(request.inputs[i], pools, &operandInfo);
        inputs.push_back(convertToTensordef(operandInfo));
    }

    // prepare outputs
    std::vector<hexagon_nn_tensordef> outputs;
    for (size_t i = 0; i < request.outputs.size(); ++i) {
        OperandInfo& operandInfo = mOperands[mOutputs[i]];
        updateOperand(request.outputs[i], pools, &operandInfo);
        outputs.push_back(convertToTensordef(operandInfo));
    }

    // execute model
    int err = hexagon::Controller::getInstance().execute_new(mGraphId, inputs.data(),
                                                             inputs.size(), outputs.data(),
                                                             outputs.size());

    std::for_each(pools.begin(), pools.end(), [](RunTimePoolInfo& pool) { pool.update(); });

    LOG(INFO) << getDebugLog();
    LOG(INFO) << "EXECUTE WAS " << (err == 0 ? "SUCCESSFUL" : "UNSUCCESSFUL");

    return err == 0;
}

} // namespace hexagon
} // namespace implementation
} // namespace V1_0
} // namespace neuralnetworks
} // namespace hardware
} // namespace android
