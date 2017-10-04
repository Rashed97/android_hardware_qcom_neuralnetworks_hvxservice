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

#include "PreparedModel.h"
#include <android-base/logging.h>
#include <thread>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace V1_0 {
namespace implementation {

PreparedModel::PreparedModel(const Model& neuralNetworksModel, hexagon::Model&& model) :
        mNeuralNetworksModel(neuralNetworksModel), mHexagonModel(std::move(model)) {}

PreparedModel::~PreparedModel() {}

void PreparedModel::asyncExecute(const Request& request, const sp<IExecutionCallback>& callback) {
    ErrorStatus status = mHexagonModel.execute(request) == true ?
            ErrorStatus::NONE : ErrorStatus::GENERAL_FAILURE;
    callback->notify(status);
}

// Methods from IPreparedModel follow.
Return<ErrorStatus> PreparedModel::execute(const Request& request,
                                           const sp<IExecutionCallback>& callback) {
    LOG(INFO) << "PreparedModel::execute";
    if (callback.get() == nullptr) {
        LOG(ERROR) << "invalid callback passed to execute";
        return ErrorStatus::INVALID_ARGUMENT;
    }
    if (!nn::validateRequest(request, mNeuralNetworksModel)) {
        callback->notify(ErrorStatus::INVALID_ARGUMENT);
        return ErrorStatus::INVALID_ARGUMENT;
    }

    // This thread is intentionally detached because the sample driver service
    // is expected to live forever.
    std::thread([this, request, callback]{ return asyncExecute(request, callback); }).detach();
    return ErrorStatus::NONE;
}

}  // namespace implementation
}  // namespace V1_0
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
