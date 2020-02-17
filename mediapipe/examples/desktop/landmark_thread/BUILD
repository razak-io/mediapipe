# Copyright 2019 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//mediapipe/examples:__subpackages__"])

cc_binary(
    name = "multi_hand_tracking_gpu_out_thread",
    deps = [
        "//mediapipe/examples/desktop/landmark_thread:demo_run_graph_main_gpu_out_multi_thread",
        "//mediapipe/graphs/hand_tracking:multi_hand_mobile_calculators",
    ],
)

# Linux only
cc_binary(
    name = "hand_tracking_gpu_out_thread",
    deps = [
        "//mediapipe/examples/desktop/landmark_thread:demo_run_graph_main_gpu_out_thread",
        "//mediapipe/graphs/hand_tracking:mobile_calculators",
		"//mediapipe/gpu:gpu_buffer_to_image_frame_calculator",
		"//mediapipe/gpu:image_frame_to_gpu_buffer_calculator",	
    ],
)
