import 'dart:io';
import 'dart:isolate';

import 'package:camera/camera.dart';
import 'package:image/image.dart' as imageLib;
import 'package:object_detection/tflite/classifier.dart';
import 'package:object_detection/utils/image_utils.dart';
import 'package:stream_channel/isolate_channel.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

/// Manages separate Isolate instance for inference
class IsolateUtils {
  static const String DEBUG_NAME = "InferenceIsolate";

  IsolateChannel channel;

  Future<void> start(Function processResults) async {
    final receivePort = new ReceivePort();
    channel = new IsolateChannel.connectReceive(receivePort);
    channel.stream.listen((data) {
      processResults(data);
    });
    await Isolate.spawn(_entryPoint, receivePort.sendPort);
  }

  void process(IsolateData isolateData) {
    channel.sink.add(isolateData);
  }

  static void _entryPoint(SendPort sendPort) {
    IsolateChannel channel = new IsolateChannel.connectSend(sendPort);
    channel.stream.listen((data) {
      IsolateData isolateData = data;
      Classifier classifier = Classifier(
        interpreter: Interpreter.fromAddress(isolateData.interpreterAddress),
        labels: isolateData.labels,
      );
      imageLib.Image image =
          ImageUtils.convertCameraImage(isolateData.cameraImage);
      if (Platform.isAndroid) {
        image = imageLib.copyRotate(image, 90);
      }
      final results = classifier.predict(image);
      channel.sink.add(results);
    });
  }
}

/// Bundles data to pass between Isolate
class IsolateData {
  CameraImage cameraImage;
  int interpreterAddress;
  List<String> labels;

  IsolateData(
    this.cameraImage,
    this.interpreterAddress,
    this.labels,
  );
}
