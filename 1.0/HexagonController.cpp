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

#define CLOSE_HEXAGON_FUNCTION(name) mFn_##name = nullptr;

#define FOR_EACH_FUNCTION(MACRO)   \
    MACRO(init)                    \
    MACRO(getlog)                  \
    MACRO(snpprint)                \
    MACRO(set_debug_level)         \
    MACRO(prepare)                 \
    MACRO(append_node)             \
    MACRO(append_const_node)       \
    MACRO(execute_new)             \
    MACRO(execute)                 \
    MACRO(teardown)                \
    MACRO(get_perfinfo)            \
    MACRO(reset_perfinfo)          \
    MACRO(version)                 \
    MACRO(last_execution_cycles)   \
    MACRO(GetHexagonBinaryVersion) \
    MACRO(PrintLog)                \
    MACRO(op_name_to_id)           \
    MACRO(op_id_to_name)           \
    MACRO(disable_dcvs)            \
    MACRO(set_powersave_level)     \
    MACRO(config)

#define CONTROLLER_CHECK(function, ...)    \
    int err = mFn_##function(__VA_ARGS__); \
    if (err != 0) {                        \
        return err;                        \
    }

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace V1_0 {
namespace implementation {
namespace hexagon {

const char Controller::kFilename[] = "libhexagon_nn_controller.so";

Controller::Controller() {
    openNnlib();
}

Controller::~Controller() {
    closeNnlib();
}

bool Controller::openNnlib() {
    mHandle = dlopen(kFilename, RTLD_LAZY | RTLD_LOCAL);
    HEXAGON_SOFT_ASSERT_NE(mHandle, 0, "FAILED TO LOAD LIBRARY "/* << kFilename << ": " << dlerror()*/);
    FOR_EACH_FUNCTION(LOAD_HEXAGON_FUNCTION)
    return true;
}

bool Controller::closeNnlib() {
    FOR_EACH_FUNCTION(CLOSE_HEXAGON_FUNCTION)
    if (mHandle != nullptr) {
        int err = dlclose(mHandle);
        mHandle = nullptr;
        HEXAGON_SOFT_ASSERT_EQ(err, 0, "FAILED TO CLOSE LIBRARY " << kFilename);
    }
    return true;
}

bool Controller::resetNnlib() {
    return closeNnlib() && openNnlib();
}

Controller& Controller::getInstance() {
    static Controller instance{};
    return instance;
}

hexagon_nn_nn_id Controller::init() {
    if (mFn_init == nullptr) {
        return hexagon_nn_nn_id{};
    }

    hexagon_nn_nn_id id = mFn_init();

    return id;
}

int Controller::getlog(hexagon_nn_nn_id id, unsigned char *buf, uint32_t length) {
    if (mFn_getlog == nullptr) {
        return -1;
    }

    CONTROLLER_CHECK(getlog, id, buf, length);

    return 0;
}

int Controller::snpprint(hexagon_nn_nn_id id, unsigned char *buf, uint32_t length) {
    if (mFn_snpprint == nullptr) {
        return -1;
    }

    CONTROLLER_CHECK(snpprint, id, buf, length);

    return 0;
}

int Controller::set_debug_level(hexagon_nn_nn_id id, int level) {
    if (mFn_set_debug_level == nullptr) {
        return -1;
    }

    CONTROLLER_CHECK(set_debug_level, id, level);

    return 0;
}

int Controller::prepare(hexagon_nn_nn_id id) {
    if (mFn_prepare == nullptr) {
        return -1;
    }

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

    CONTROLLER_CHECK(execute_new, id, inputs, n_inputs, outputs, n_outputs);

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

    CONTROLLER_CHECK(execute, id, batches_in, height_in, width_in, depth_in, data_in, data_len_in,
                     batches_out, height_out, width_out, depth_out, data_out, data_out_max,
                     data_out_size);

    return 0;
}

int Controller::teardown(hexagon_nn_nn_id id) {
    if (mFn_teardown == nullptr) {
        return -1;
    }

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

    CONTROLLER_CHECK(get_perfinfo, id, info_out, info_out_len, n_items_out);

    return 0;
}

int Controller::reset_perfinfo(hexagon_nn_nn_id id, uint32_t event) {
    if (mFn_reset_perfinfo == nullptr) {
        return -1;
    }

    CONTROLLER_CHECK(reset_perfinfo, id, event);

    return 0;
}

int Controller::version(int *ver) {
    if (mFn_version == nullptr) {
        return -1;
    }

    CONTROLLER_CHECK(version, ver);

    return 0;
}

int Controller::last_execution_cycles(hexagon_nn_nn_id id,
                                   unsigned int *cycles_lo,
                                   unsigned int *cycles_hi) {
    if (mFn_last_execution_cycles == nullptr) {
        return -1;
    }

    CONTROLLER_CHECK(last_execution_cycles, id, cycles_lo, cycles_hi);

    return 0;
}

int Controller::GetHexagonBinaryVersion(int *ver) {
    if (mFn_GetHexagonBinaryVersion == nullptr) {
        return -1;
    }

    CONTROLLER_CHECK(GetHexagonBinaryVersion, ver);

    return 0;
}

int Controller::PrintLog(const uint8_t *data_in, unsigned int data_in_len) {
    if (mFn_PrintLog == nullptr) {
        return -1;
    }

    CONTROLLER_CHECK(PrintLog, data_in, data_in_len);

    return 0;
}

int Controller::op_name_to_id(const char *name, unsigned int *id) {
    if (mFn_op_name_to_id == nullptr) {
        return -1;
    }

    CONTROLLER_CHECK(op_name_to_id, name, id);

    return 0;
}

int Controller::op_id_to_name(const unsigned int id, char *name, int name_len) {
    if (mFn_op_id_to_name == nullptr) {
        return -1;
    }

    CONTROLLER_CHECK(op_id_to_name, id, name, name_len);

    return 0;
}

int Controller::disable_dcvs() {
    if (mFn_disable_dcvs == nullptr) {
        return -1;
    }

    CONTROLLER_CHECK(disable_dcvs);

    return 0;
}

int Controller::set_powersave_level(unsigned int level) {
    if (mFn_set_powersave_level == nullptr) {
        return -1;
    }

    CONTROLLER_CHECK(set_powersave_level, level);

    return 0;
}

int Controller::config() {
    if (mFn_config == nullptr) {
        return -1;
    }

    CONTROLLER_CHECK(config);

    return 0;
}

} // namespace hexagon
} // namespace implementation
} // namespace V1_0
} // namespace neuralnetworks
} // namespace hardware
} // namespace android
