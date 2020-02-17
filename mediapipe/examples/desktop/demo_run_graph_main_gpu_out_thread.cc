// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// An example of sending OpenCV webcam frames into a MediaPipe graph.
// This example requires a linux computer and a GPU with EGL support drivers.
#include <cstdlib>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/commandlineflags.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include "mediapipe/gpu/gpu_shared_data_internal.h"
#include <iostream>
#include <thread>
#include <pthread.h>
#include <mutex>          // std::mutex


std::vector<std::vector<cv::Point2f>> landMarks[2];


//Take stream from /mediapipe/graphs/hand_tracking/hand_detection_desktop_live.pbtxt
// RendererSubgraph - LANDMARKS:hand_landmarks
#include "mediapipe/calculators/util/landmarks_to_render_data_calculator.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"

// input and output streams to be used/retrieved by calculators
constexpr char kInputStream[] = "input_video";
constexpr char kOutputStream[] = "output_video";
constexpr char kLandmarksStream[] = "hand_landmarks";
constexpr char kWindowName[] = "MediaPipe";

DEFINE_string(
    calculator_graph_config_file, "",
    "Name of file containing text format CalculatorGraphConfig proto.");
DEFINE_string(input_video_path, "",
              "Full path of video to load. "
              "If not provided, attempt to use a webcam.");
DEFINE_string(output_video_path, "",
              "Full path of where to save result (.mp4 only). "
              "If not provided, show result in a window.");

::mediapipe::Status RunMPPGraph(int index, int thr) {
  std::string calculator_graph_config_contents;
  MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
      FLAGS_calculator_graph_config_file, &calculator_graph_config_contents));
  LOG(INFO) << "Get calculator graph config contents: "
            << calculator_graph_config_contents;
  mediapipe::CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
          calculator_graph_config_contents);

  LOG(INFO) << "Initialize the calculator graph.";
  mediapipe::CalculatorGraph rgraph;
  MP_RETURN_IF_ERROR(rgraph.Initialize(config));

  mediapipe::CalculatorGraph lgraph;
  MP_RETURN_IF_ERROR(lgraph.Initialize(config));


  LOG(INFO) << "Initialize the GPU.";
  ASSIGN_OR_RETURN(auto rgpu_resources, mediapipe::GpuResources::Create());
  MP_RETURN_IF_ERROR(rgraph.SetGpuResources(std::move(rgpu_resources)));
  mediapipe::GlCalculatorHelper rgpu_helper;
  rgpu_helper.InitializeForTest(rgraph.GetGpuResources().get());

  ASSIGN_OR_RETURN(auto lgpu_resources, mediapipe::GpuResources::Create());
  MP_RETURN_IF_ERROR(lgraph.SetGpuResources(std::move(lgpu_resources)));
  mediapipe::GlCalculatorHelper lgpu_helper;
  lgpu_helper.InitializeForTest(lgraph.GetGpuResources().get());

  LOG(INFO) << "Initialize the camera or load the video.";
  cv::VideoCapture capture;
  const bool load_video = !FLAGS_input_video_path.empty();
  if (load_video) {
    capture.open(FLAGS_input_video_path);
  } else {
    capture.open(1);
  }
  RET_CHECK(capture.isOpened());

  cv::VideoWriter writer;
  std::ostringstream lstringStream, rstringStream;

  lstringStream << kWindowName << " left"; 
  rstringStream << kWindowName << " right";
  
  cv::String left  = lstringStream.str();
  cv::String right = rstringStream.str();

  const bool save_video = !FLAGS_output_video_path.empty();
  if (!save_video) {
    cv::namedWindow(left, /*flags=WINDOW_AUTOSIZE*/ 1);
	cv::namedWindow(right, /*flags=WINDOW_AUTOSIZE*/ 1);
#if (CV_MAJOR_VERSION >= 3) && (CV_MINOR_VERSION >= 2)
    capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    capture.set(cv::CAP_PROP_FPS, 30);
#endif
  }

  LOG(INFO) << "Start running the calculator graph.";
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller rpoller,
                   rgraph.AddOutputStreamPoller(kOutputStream));

  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller lpoller,
                   lgraph.AddOutputStreamPoller(kOutputStream));

  // hand landmarks stream
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller lpoller_landmark,
            rgraph.AddOutputStreamPoller(kLandmarksStream));

  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller rpoller_landmark,
            lgraph.AddOutputStreamPoller(kLandmarksStream));


  MP_RETURN_IF_ERROR(rgraph.StartRun({}));
  MP_RETURN_IF_ERROR(lgraph.StartRun({}));

  LOG(INFO) << "Start grabbing and processing frames.";
  bool grab_frames = true;
  while (grab_frames) {
    // Capture opencv camera or video frame.
    cv::Mat camera_frame_raw;
    capture >> camera_frame_raw;
    if (camera_frame_raw.empty()) break;  // End of video.
    cv::Mat camera_frame, rcamera_frame, lcamera_frame;
    cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);

	rcamera_frame = camera_frame(cv::Rect(0, 0, 640, 480));
	lcamera_frame = camera_frame(cv::Rect(640, 0, 640, 480));

    if (!load_video) {
      cv::flip(rcamera_frame, rcamera_frame, /*flipcode=HORIZONTAL*/ 1);
	  cv::flip(lcamera_frame, lcamera_frame, /*flipcode=HORIZONTAL*/ 1);
    }


	//std::cout << "Image Width: "  << camera_frame.cols  << std::endl;
	//std::cout << "Image Height: " << camera_frame.rows << std::endl;

    // Wrap Mat into an ImageFrame.
    auto rinput_frame = absl::make_unique<mediapipe::ImageFrame>(
        mediapipe::ImageFormat::SRGB, rcamera_frame.cols, rcamera_frame.rows,
        mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);
    cv::Mat rinput_frame_mat = mediapipe::formats::MatView(rinput_frame.get());
    rcamera_frame.copyTo(rinput_frame_mat);

    auto linput_frame = absl::make_unique<mediapipe::ImageFrame>(
        mediapipe::ImageFormat::SRGB, lcamera_frame.cols, lcamera_frame.rows,
        mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);
    cv::Mat linput_frame_mat = mediapipe::formats::MatView(linput_frame.get());
    lcamera_frame.copyTo(linput_frame_mat);


    // Prepare and add graph input packet.
    size_t frame_timestamp_us =
        (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;
    MP_RETURN_IF_ERROR(
        rgpu_helper.RunInGlContext([&rinput_frame, &frame_timestamp_us, &rgraph,
                                   &rgpu_helper]() -> ::mediapipe::Status {
          // Convert ImageFrame to GpuBuffer.
          auto rtexture = rgpu_helper.CreateSourceTexture(*rinput_frame.get());
          auto rgpu_frame = rtexture.GetFrame<mediapipe::GpuBuffer>();
          glFlush();
          rtexture.Release();
          // Send GPU image packet into the graph.
          MP_RETURN_IF_ERROR(rgraph.AddPacketToInputStream(
              kInputStream, mediapipe::Adopt(rgpu_frame.release())
                                .At(mediapipe::Timestamp(frame_timestamp_us))));
          return ::mediapipe::OkStatus();
        }));

    frame_timestamp_us =
        (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;
    MP_RETURN_IF_ERROR(
        lgpu_helper.RunInGlContext([&linput_frame, &frame_timestamp_us, &lgraph,
                                   &lgpu_helper]() -> ::mediapipe::Status {
          // Convert ImageFrame to GpuBuffer.
          auto ltexture = lgpu_helper.CreateSourceTexture(*linput_frame.get());
          auto lgpu_frame = ltexture.GetFrame<mediapipe::GpuBuffer>();
          glFlush();
          ltexture.Release();
          // Send GPU image packet into the graph.
          MP_RETURN_IF_ERROR(lgraph.AddPacketToInputStream(
              kInputStream, mediapipe::Adopt(lgpu_frame.release())
                                .At(mediapipe::Timestamp(frame_timestamp_us))));
          return ::mediapipe::OkStatus();
        }));


    // Get the graph result packet, or stop if that fails.
    mediapipe::Packet rpacket, lpacket;
    mediapipe::Packet rlandmark_packet, llandmark_packet;

    if (!rpoller.Next(&rpacket)) break;
	if (!rpoller_landmark.Next(&rlandmark_packet)) break;

    if (!lpoller.Next(&lpacket)) break;
	if (!lpoller_landmark.Next(&llandmark_packet)) break;



    std::unique_ptr<mediapipe::ImageFrame> routput_frame, loutput_frame;

	auto& routput_landmarks = rlandmark_packet.Get<mediapipe::NormalizedLandmarkList>();
	auto& loutput_landmarks = llandmark_packet.Get<mediapipe::NormalizedLandmarkList>();

    // Convert GpuBuffer to ImageFrame.
    MP_RETURN_IF_ERROR(rgpu_helper.RunInGlContext(
        [&rpacket, &routput_frame, &rgpu_helper]() -> ::mediapipe::Status {
          auto& rgpu_frame = rpacket.Get<mediapipe::GpuBuffer>();
          auto  rtexture = rgpu_helper.CreateSourceTexture(rgpu_frame);
          routput_frame = absl::make_unique<mediapipe::ImageFrame>(
              mediapipe::ImageFormatForGpuBufferFormat(rgpu_frame.format()),
              rgpu_frame.width(), rgpu_frame.height(),
              mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);
          rgpu_helper.BindFramebuffer(rtexture);
          const auto info =
              mediapipe::GlTextureInfoForGpuBufferFormat(rgpu_frame.format(), 0);
          glReadPixels(0, 0, rtexture.width(), rtexture.height(), info.gl_format,
                       info.gl_type, routput_frame->MutablePixelData());
          glFlush();
          rtexture.Release();
          return ::mediapipe::OkStatus();
        }));

    // Convert GpuBuffer to ImageFrame.
    MP_RETURN_IF_ERROR(lgpu_helper.RunInGlContext(
        [&lpacket, &loutput_frame, &lgpu_helper]() -> ::mediapipe::Status {
          auto& lgpu_frame = lpacket.Get<mediapipe::GpuBuffer>();
          auto  ltexture = lgpu_helper.CreateSourceTexture(lgpu_frame);
          loutput_frame = absl::make_unique<mediapipe::ImageFrame>(
              mediapipe::ImageFormatForGpuBufferFormat(lgpu_frame.format()),
              lgpu_frame.width(), lgpu_frame.height(),
              mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);
          lgpu_helper.BindFramebuffer(ltexture);
          const auto info =
              mediapipe::GlTextureInfoForGpuBufferFormat(lgpu_frame.format(), 0);
          glReadPixels(0, 0, ltexture.width(), ltexture.height(), info.gl_format,
                       info.gl_type, loutput_frame->MutablePixelData());
          glFlush();
          ltexture.Release();
          return ::mediapipe::OkStatus();
        }));


    // Convert back to opencv for display or saving.
    cv::Mat routput_frame_mat = mediapipe::formats::MatView(routput_frame.get());
    cv::cvtColor(routput_frame_mat, routput_frame_mat, cv::COLOR_RGB2BGR);

	cv::Mat loutput_frame_mat = mediapipe::formats::MatView(loutput_frame.get());
    cv::cvtColor(loutput_frame_mat, loutput_frame_mat, cv::COLOR_RGB2BGR);

    if (save_video) {
      if (!writer.isOpened()) {
        LOG(INFO) << "Prepare video writer.";
        writer.open(FLAGS_output_video_path,
                    mediapipe::fourcc('a', 'v', 'c', '1'),  // .mp4
                    capture.get(cv::CAP_PROP_FPS), routput_frame_mat.size());
        RET_CHECK(writer.isOpened());
      }
      writer.write(routput_frame_mat);
    } else {

      cv::imshow(right, routput_frame_mat);
      cv::imshow(left, loutput_frame_mat);

      // Press any key to exit.
      const int pressed_key = cv::waitKey(5);
      if (pressed_key >= 0 && pressed_key != 255) grab_frames = false;
	  for (int i = 0; i < loutput_landmarks.landmark_size(); ++i) {
			  const mediapipe::NormalizedLandmark& landmark = loutput_landmarks.landmark(i);
			  //LOG(INFO) << "Wrist " << j << "Point: " << i;
			  LOG(INFO) << "Left: " << i << ": "<<"x: " << landmark.x() * camera_frame.cols 
			  << " y: " << landmark.y() * camera_frame.rows << " z: " << landmark.z();
	   }

	   for (int i = 0; i < routput_landmarks.landmark_size(); ++i) {
			  const mediapipe::NormalizedLandmark& landmark = routput_landmarks.landmark(i);
			  LOG(INFO) << "Right: " << i << ": "<<"x: " << landmark.x() * camera_frame.cols 
			  << " y: " << landmark.y() * camera_frame.rows << " z: " << landmark.z();
	   }

    }
  }

  LOG(INFO) << "Shutting down.";
  if (writer.isOpened()) writer.release();
  MP_RETURN_IF_ERROR(rgraph.CloseInputStream(kInputStream));
  MP_RETURN_IF_ERROR(lgraph.CloseInputStream(kInputStream));

  ::mediapipe::Status r = rgraph.WaitUntilDone(); 
  ::mediapipe::Status l = lgraph.WaitUntilDone();

  bool status = r.ok() && l.ok(); 
  return lgraph.WaitUntilDone();
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::mediapipe::Status run_status = RunMPPGraph(640, 0);
  bool s = run_status.ok(); 
  if (!s) {
    LOG(ERROR) << "Failed to run the graph: " << run_status.message();
    return EXIT_FAILURE;
  } else {
    LOG(INFO) << "Success!";
  }
  return EXIT_SUCCESS;
}
