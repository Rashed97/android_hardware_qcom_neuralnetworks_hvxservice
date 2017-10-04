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

#include "Device.h"
#include "HexagonModel.h"
#include "HexagonUtils.h"
#include "PreparedModel.h"
#include <android-base/logging.h>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace V1_0 {
namespace implementation {

Device::Device() : mCurrentStatus(DeviceStatus::AVAILABLE) {}

Device::~Device() {}

// Methods from IDevice follow.

Return<void> Device::getCapabilities(getCapabilities_cb _hidl_cb) {
    LOG(INFO) << "Device::getCapabilities";

    PerformanceInfo float32Performance = {
        .execTime   = 100.0f, // nanoseconds?
        .powerUsage = 1.0f,   // picoJoules
    };

    PerformanceInfo quantized8Performance = {
        .execTime   = 100.0f, // nanoseconds?
        .powerUsage = 1.0f,   // picoJoules
    };

    Capabilities capabilities = {
        .float32Performance    = float32Performance,
        .quantized8Performance = quantized8Performance,
    };

    _hidl_cb(ErrorStatus::NONE, capabilities);
    return Void();
}

Return<void> Device::getSupportedOperations(const Model& model,
                                            getSupportedOperations_cb _hidl_cb) {
    LOG(INFO) << "Device::getSupportedOperations";

    if (!nn::validateModel(model)) {
        std::vector<bool> supported;
        _hidl_cb(ErrorStatus::INVALID_ARGUMENT, supported);
        return Void();
    }

    hexagon::Model hexagonModel(model);
    std::vector<bool> supported = hexagonModel.supportedOperations();

    _hidl_cb(ErrorStatus::NONE, supported);
    return Void();
}

Return<ErrorStatus> Device::prepareModel(const Model& model,
                                         const sp<IPreparedModelCallback>& callback) {
    LOG(INFO) << "Device::prepareModel";
    if (callback.get() == nullptr) {
        LOG(ERROR) << "invalid callback passed to prepareModel";
        return ErrorStatus::INVALID_ARGUMENT;
    }
    if (!nn::validateModel(model)) {
        callback->notify(ErrorStatus::INVALID_ARGUMENT, nullptr);
        return ErrorStatus::INVALID_ARGUMENT;
    }

    hexagon::Model hexagonModel(model);

    // attempt to compile model; if this fails, it will be fully compiled later
    hexagonModel.compile();
    callback->notify(ErrorStatus::NONE, new PreparedModel(model, std::move(hexagonModel)));

    return ErrorStatus::NONE;
}

Return<DeviceStatus> Device::getStatus() {
    LOG(INFO) << "Device::getStatus";

    // TODO: remove dummy function
    // this is simply here to test connection to Hexagon
    int version = -1;
    hexagon::Controller::getInstance().version(&version);
    mCurrentStatus = version == 92 ? DeviceStatus::AVAILABLE : DeviceStatus::BUSY;

    LOG(INFO) << "current status: " << version << ", " << toString(mCurrentStatus);

    return mCurrentStatus;
}

}  // namespace implementation
}  // namespace V1_0
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
