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

#include "HexagonController.h"

#define LOAD_HEXAGON_FUNCTION(name) \
    mFn_##name = loadFunction<hexagon_nn_controller_##name##_fn>("hexagon_nn_controller_"#name);

#define PRINT_TYPE INFO

#define CONTROLLER_CHECK(function, ...)                         \
    int err = mFn_##function(__VA_ARGS__);                      \
    if (err != 0) {                                             \
        LOG(ERROR) << "Controller::" << #function << " failed"; \
        return err;                                             \
    }

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace V1_0 {
namespace implementation {
namespace hexagon {

Controller::Controller() {
    const char* filename = "libhexagon_nn_controller.so";

    mHandle = dlopen(filename, RTLD_LAZY | RTLD_LOCAL);
    if (mHandle == nullptr) {
        LOG(ERROR) << "FAILED TO LOAD LIBRARY libhexagon_nn_controller: " << dlerror();
    }

    LOAD_HEXAGON_FUNCTION(init)
    LOAD_HEXAGON_FUNCTION(getlog)
    LOAD_HEXAGON_FUNCTION(snpprint)
    LOAD_HEXAGON_FUNCTION(set_debug_level)
    LOAD_HEXAGON_FUNCTION(prepare)
    LOAD_HEXAGON_FUNCTION(append_node)
    LOAD_HEXAGON_FUNCTION(append_const_node)
    LOAD_HEXAGON_FUNCTION(execute_new)
    LOAD_HEXAGON_FUNCTION(execute)
    LOAD_HEXAGON_FUNCTION(teardown)
    LOAD_HEXAGON_FUNCTION(get_perfinfo)
    LOAD_HEXAGON_FUNCTION(reset_perfinfo)
    LOAD_HEXAGON_FUNCTION(version)
    LOAD_HEXAGON_FUNCTION(last_execution_cycles)
    LOAD_HEXAGON_FUNCTION(GetHexagonBinaryVersion)
    LOAD_HEXAGON_FUNCTION(PrintLog)
    LOAD_HEXAGON_FUNCTION(op_name_to_id)
    LOAD_HEXAGON_FUNCTION(op_id_to_name)
    LOAD_HEXAGON_FUNCTION(disable_dcvs)
    LOAD_HEXAGON_FUNCTION(set_powersave_level)
    LOAD_HEXAGON_FUNCTION(config)
}

Controller::~Controller() {
    if (mHandle != nullptr) {
        dlclose(mHandle);
    }
}

Controller& Controller::getInstance() {
    static Controller instance{};
    return instance;
}

hexagon_nn_nn_id Controller::init() {
    if (mFn_init == nullptr) {
        return hexagon_nn_nn_id{};
    }

    LOG(PRINT_TYPE) << "Controller::init()";

    hexagon_nn_nn_id id = mFn_init();

    LOG(PRINT_TYPE) << "Controller::init -- output::id: " << toString(id);

    return id;
}

int Controller::getlog(hexagon_nn_nn_id id, unsigned char *buf, uint32_t length) {
    if (mFn_getlog == nullptr) {
        return -1;
    }

    LOG(PRINT_TYPE) << "Controller::getlog(id: " << toString(id) << ", buf: "
                    << reinterpret_cast<void*>(buf) << ", length: " << length << ")";

    CONTROLLER_CHECK(getlog, id, buf, length);

    LOG(PRINT_TYPE) << "Controller::getlog -- output::buf: " << buf;

    return 0;
}

int Controller::snpprint(hexagon_nn_nn_id id, unsigned char *buf, uint32_t length) {
    if (mFn_snpprint == nullptr) {
        return -1;
    }

    LOG(PRINT_TYPE) << "Controller::snpprint(id: " << toString(id) << ", buf: "
                    << reinterpret_cast<void*>(buf) << ", length: " << length << ")";

    CONTROLLER_CHECK(snpprint, id, buf, length);

    //LOG(PRINT_TYPE) << "Controller::snpprint -- output::buf: " << buf;

    return 0;
}

int Controller::set_debug_level(hexagon_nn_nn_id id, int level) {
    if (mFn_set_debug_level == nullptr) {
        return -1;
    }

    LOG(PRINT_TYPE) << "Controller::set_debug_level(id: " << toString(id) << ", level: "
                    << level << ")";

    CONTROLLER_CHECK(set_debug_level, id, level);

    return 0;
}

int Controller::prepare(hexagon_nn_nn_id id) {
    if (mFn_prepare == nullptr) {
        return -1;
    }

    LOG(PRINT_TYPE) << "Controller::prepare(id: " << toString(id) << ")";

    CONTROLLER_CHECK(prepare, id);

    return 0;
}

int Controller::append_node(hexagon_nn_nn_id id,
                         uint32_t node_id,
                         op_type operation,
                         hexagon_nn_padding_type padding,
                         const hexagon_nn_input *inputs,
                         uint32_t num_inputs,
                         const hexagon_nn_output *outputs,
                         uint32_t num_outputs) {
    if (mFn_append_node == nullptr) {
        return -1;
    }

    LOG(PRINT_TYPE) << "Controller::append_node(id: " << toString(id) << ", node_id: " << node_id
                    << ", operation: " << toString(operation) << ", padding: " << toString(padding)
                    << ", inputs: " << inputs << ", num_inputs: " << num_inputs
                    << ", outputs: " << outputs << ", num_outputs: " << num_outputs << ")";

    LOG(PRINT_TYPE) << "Controller::append_node -- input::inputs: "
                    << toString(inputs, num_inputs);
    LOG(PRINT_TYPE) << "Controller::append_node -- input::outputs: "
                    << toString(outputs, num_outputs);

    CONTROLLER_CHECK(append_node, id, node_id, operation, padding, inputs, num_inputs,
                     outputs, num_outputs);

    return 0;
}

int Controller::append_const_node(hexagon_nn_nn_id id,
                               uint32_t node_id,
                               uint32_t batches,
                               uint32_t height,
                               uint32_t width,
                               uint32_t depth,
                               const uint8_t *data,
                               uint32_t data_len) {
    if (mFn_append_const_node == nullptr) {
        return -1;
    }

    LOG(PRINT_TYPE) << "Controller::append_const_node(id: " << toString(id) << ", node_id: "
                    << node_id << ", batches: " << batches << ", height: " << height << ", width: "
                    << width << ", depth: " << depth << ", data: "
                    << reinterpret_cast<const void*>(data) << ", data_len: " << data_len << ")";

    LOG(PRINT_TYPE) << "Controller::append_const_node -- input::data: "
                    << toString(reinterpret_cast<const float*>(data),
                                data_len / static_cast<uint32_t>(sizeof(float)));

    CONTROLLER_CHECK(append_const_node, id, node_id, batches, height, width, depth,
                     data, data_len);

    return 0;
}

int Controller::execute_new(hexagon_nn_nn_id id,
                         const hexagon_nn_tensordef *inputs,
                         uint32_t n_inputs,
                         hexagon_nn_tensordef *outputs,
                         uint32_t n_outputs) {
    if (mFn_execute_new == nullptr) {
        return -1;
    }

    LOG(PRINT_TYPE) << "Controller::execute_new(id: " << toString(id) << ", inputs: " << inputs
                    << ", n_inputs: " << n_inputs << ", outputs: " << outputs
                    << ", n_outputs: " << n_outputs << ")";

    LOG(PRINT_TYPE) << "Controller::execute_new -- input::inputs: "
                    << toString(inputs, n_inputs);
    for (uint32_t i = 0; i < n_inputs; ++i) {
        LOG(PRINT_TYPE) << "Controller::execute_new -- input::input[" << i << "]::data: "
                        << toString(reinterpret_cast<const float*>(inputs[i].data),
                                inputs[i].dataLen / static_cast<uint32_t>(sizeof(float)));
    }

    CONTROLLER_CHECK(execute_new, id, inputs, n_inputs, outputs, n_outputs);

    LOG(PRINT_TYPE) << "Controller::execute_new -- output::outputs: "
                    << toString(outputs, n_outputs);
    for (uint32_t i = 0; i < n_outputs; ++i) {
        LOG(PRINT_TYPE) << "Controller::execute_new -- input::output[" << i << "]::data: "
                        << toString(reinterpret_cast<const float*>(outputs[i].data),
                                outputs[i].dataLen / static_cast<uint32_t>(sizeof(float)));
    }


    return 0;
}

int Controller::execute(hexagon_nn_nn_id id,
                     uint32_t batches_in,
                     uint32_t height_in,
                     uint32_t width_in,
                     uint32_t depth_in,
                     const uint8_t *data_in,
                     uint32_t data_len_in,
                     uint32_t *batches_out,
                     uint32_t *height_out,
                     uint32_t *width_out,
                     uint32_t *depth_out,
                     uint8_t *data_out,
                     uint32_t data_out_max,
                     uint32_t *data_out_size) {
    if (mFn_execute == nullptr) {
        return -1;
    }

    LOG(PRINT_TYPE) << "Controller::execute(id: " << toString(id) << ", batches_in: " << batches_in
                    << ", height_in: " << height_in << ", width_in: " << width_in << ", depth_in: "
                    << depth_in << ", data_in: " << reinterpret_cast<const void*>(data_in)
                    << ", data_len_in: " << data_len_in << ", batches_out: " << batches_out
                    << ", height_out: " << height_out << ", width_out: " << width_out
                    << ", depth_out: " << depth_out << ", data_out: "
                    << reinterpret_cast<void*>(data_out) << " , data_out_max: " << data_out_max
                    << ", data_out_size: " << data_out_size << ")";

    LOG(PRINT_TYPE) << "Controller::execute -- input::data: "
                    << toString(reinterpret_cast<const float*>(data_in),
                            data_len_in / static_cast<uint32_t>(sizeof(float)));

    CONTROLLER_CHECK(execute, id, batches_in, height_in, width_in, depth_in, data_in, data_len_in,
                     batches_out, height_out, width_out, depth_out, data_out, data_out_max,
                     data_out_size);

    LOG(PRINT_TYPE) << "Controller::execute -- output::batches_out: " << *batches_out;
    LOG(PRINT_TYPE) << "Controller::execute -- output::height_out:  " << *height_out;
    LOG(PRINT_TYPE) << "Controller::execute -- output::width_out:   " << *width_out;
    LOG(PRINT_TYPE) << "Controller::execute -- output::depth_out:   " << *depth_out;
    LOG(PRINT_TYPE) << "Controller::execute -- output::data_out:    "
                    << toString(reinterpret_cast<const float*>(data_out),
                            *data_out_size / static_cast<uint32_t>(sizeof(float)));

    return 0;
}

int Controller::teardown(hexagon_nn_nn_id id) {
    if (mFn_teardown == nullptr) {
        return -1;
    }

    LOG(PRINT_TYPE) << "Controller::teardown(id: " << toString(id) << ")";

    CONTROLLER_CHECK(teardown, id);

    return 0;
}

int Controller::get_perfinfo(hexagon_nn_nn_id id,
                          hexagon_nn_perfinfo *info_out,
                          unsigned int info_out_len,
                          unsigned int *n_items_out) {
    if (mFn_get_perfinfo == nullptr) {
        return -1;
    }

    LOG(PRINT_TYPE) << "Controller::get_perfinfo(id: " << toString(id) << ", info_out: "
                    << info_out << ", info_out_len: " << info_out_len << ", n_items_out: "
                    << n_items_out << ")";

    CONTROLLER_CHECK(get_perfinfo, id, info_out, info_out_len, n_items_out);

    LOG(PRINT_TYPE) << "Controller::get_perfinfo -- output::info: "
                    << toString(info_out, *n_items_out);

    return 0;
}

int Controller::reset_perfinfo(hexagon_nn_nn_id id, uint32_t event) {
    if (mFn_reset_perfinfo == nullptr) {
        return -1;
    }

    LOG(PRINT_TYPE) << "Controller::reset_perfinfo(id: " << toString(id) << ", event: "
                    << event << ")";

    CONTROLLER_CHECK(reset_perfinfo, id, event);

    return 0;
}

int Controller::version(int *ver) {
    if (mFn_version == nullptr) {
        return -1;
    }

    LOG(PRINT_TYPE) << "Controller::version(ver: " << ver << ")";

    CONTROLLER_CHECK(version, ver);

    LOG(PRINT_TYPE) << "Controller::version -- output::ver: " << *ver;

    return 0;
}

int Controller::last_execution_cycles(hexagon_nn_nn_id id,
                                   unsigned int *cycles_lo,
                                   unsigned int *cycles_hi) {
    if (mFn_last_execution_cycles == nullptr) {
        return -1;
    }

    LOG(PRINT_TYPE) << "Controller::last_execution_cycles(id: " << toString(id) << ", cycles_lo: "
                    << cycles_lo << ", cycles_hi: " << cycles_hi << ")";

    CONTROLLER_CHECK(last_execution_cycles, id, cycles_lo, cycles_hi);

    LOG(PRINT_TYPE) << "Controller::last_execution_cycles -- output::cycles_lo: " << *cycles_lo;
    LOG(PRINT_TYPE) << "Controller::last_execution_cycles -- output::cycles_hi: " << *cycles_hi;

    return 0;
}

int Controller::GetHexagonBinaryVersion(int *ver) {
    if (mFn_GetHexagonBinaryVersion == nullptr) {
        return -1;
    }

    LOG(PRINT_TYPE) << "Controller::GetControllerBinaryVersion(ver: " << ver << ")";

    CONTROLLER_CHECK(GetHexagonBinaryVersion, ver);

    LOG(PRINT_TYPE) << "Controller::GetHexagonBinaryVersion -- output::ver: " << *ver;

    return 0;
}

int Controller::PrintLog(const uint8_t *data_in, unsigned int data_in_len) {
    if (mFn_PrintLog == nullptr) {
        return -1;
    }

    LOG(PRINT_TYPE) << "Controller::PrintLog(data_in: " << reinterpret_cast<const void*>(data_in)
                    << ", data_in_len: " << data_in_len << ")";

    CONTROLLER_CHECK(PrintLog, data_in, data_in_len);

    //LOG(PRINT_TYPE) << "Controller::PrintLog -- output::data: " << data_in;

    return 0;
}

int Controller::op_name_to_id(const char *name, unsigned int *id) {
    if (mFn_op_name_to_id == nullptr) {
        return -1;
    }

    LOG(PRINT_TYPE) << "Controller::op_name_to_id(name: " << reinterpret_cast<const void*>(name)
                    << ", id: " << id << ")";

    LOG(PRINT_TYPE) << "Controller::op_name_to_id -- input::name: " << name;

    CONTROLLER_CHECK(op_name_to_id, name, id);

    LOG(PRINT_TYPE) << "Controller::op_name_to_id -- output::id: " << *id;

    return 0;
}

int Controller::op_id_to_name(const unsigned int id, char *name, int name_len) {
    if (mFn_op_id_to_name == nullptr) {
        return -1;
    }

    LOG(PRINT_TYPE) << "Controller::op_id_to_name(id: " << id << ", name: "
                    << reinterpret_cast<void*>(name) << ", name_len: " << name_len << ")";

    CONTROLLER_CHECK(op_id_to_name, id, name, name_len);

    LOG(PRINT_TYPE) << "Controller::op_id_to_name -- output::name: "
                    << name;

    return 0;
}

int Controller::disable_dcvs() {
    if (mFn_disable_dcvs == nullptr) {
        return -1;
    }

    LOG(PRINT_TYPE) << "Controller::disable_dcvs()";

    CONTROLLER_CHECK(disable_dcvs);

    return 0;
}

int Controller::set_powersave_level(unsigned int level) {
    if (mFn_set_powersave_level == nullptr) {
        return -1;
    }

    LOG(PRINT_TYPE) << "Controller::set_powersave_level(level: " << level << ")";

    CONTROLLER_CHECK(set_powersave_level, level);

    return 0;
}

int Controller::config() {
    if (mFn_config == nullptr) {
        return -1;
    }

    LOG(PRINT_TYPE) << "Controller::config()";

    CONTROLLER_CHECK(config);

    return 0;
}

} // namespace hexagon
} // namespace implementation
} // namespace V1_0
} // namespace neuralnetworks
} // namespace hardware
} // namespace android
